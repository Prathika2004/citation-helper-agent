import os
import re
import json
import time
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import requests.exceptions

print("--- LOADING CITATION AGENT ---")
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
SCRAPEDO_API_KEY = os.getenv("SCRAPEDO_API_KEY")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")
OPENALEX_API_KEY = "prathikamuthu2004@gmail.com"

if not HF_TOKEN:
    print("❌ ERROR: Missing HF_TOKEN in .env file!")
    exit()

client = InferenceClient(token=HF_TOKEN)
API_CALL_DELAY = 1.5
API_QUOTA_EXCEEDED = False  # Flag to track API quota status

def query_ai(messages, json_mode=False):
    global API_QUOTA_EXCEEDED
    try:
        params = {
            "model": "openai/gpt-oss-120b",
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 350,
        }
        if json_mode:
            params["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content
        return content if content is not None else ""

    except requests.exceptions.HTTPError as e:
        if hasattr(e, "response") and e.response and e.response.status_code == 402:
            print("⚠️ API quota exceeded (402 Payment Required). Using fallback mode.")
            API_QUOTA_EXCEEDED = True
        else:
            print(f"HTTP error calling GPT API: {e}")
        return ""

    except Exception as e:
        print(f"Unexpected error calling GPT API: {e}")
        return ""

def clean_author_string(raw_authors):
    if isinstance(raw_authors, list):
        return ", ".join(raw_authors)
    elif isinstance(raw_authors, str):
        cleaned = re.sub(r"([A-Za-z])\s*,\s*([A-Za-z])", r"\1\2", raw_authors)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip(", ").strip()
    else:
        return raw_authors or ""

def extract_keywords(text):
    if API_QUOTA_EXCEEDED:
        # Fallback keyword extraction when API quota is exceeded
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:5]]
    
    system_msg = {
        "role": "system",
        "content": (
            "You are a JSON-only API. Respond only with a JSON object like: "
            '{"keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]}. '
            "Provide exactly 5 meaningful academic keywords."
        ),
    }
    user_msg = {"role": "user", "content": text}
    response = query_ai([system_msg, user_msg], json_mode=False)
    if not response:
        return []
    try:
        start = response.index("{")
        end = response.rindex("}") + 1
        data = json.loads(response[start:end])
        keywords = data.get("keywords", [])
        if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
            return keywords
        return []
    except Exception:
        return []

def identify_discipline(text):
    if API_QUOTA_EXCEEDED:
        # Simple fallback discipline identification
        science_keywords = ["experiment", "hypothesis", "data", "analysis", "research"]
        medicine_keywords = ["patient", "treatment", "diagnosis", "clinical", "health"]
        sociology_keywords = ["society", "community", "culture", "social", "behavior"]
        engineering_keywords = ["design", "system", "technology", "process", "development"]
        
        text_lower = text.lower()
        scores = {
            "science": sum(1 for kw in science_keywords if kw in text_lower),
            "medicine": sum(1 for kw in medicine_keywords if kw in text_lower),
            "sociology": sum(1 for kw in sociology_keywords if kw in text_lower),
            "engineering": sum(1 for kw in engineering_keywords if kw in text_lower)
        }
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "general"
    
    prompt = (
        "Analyze the following article text and identify the academic discipline: "
        "medicine, sociology, science, or engineering. "
        "Return exactly one discipline keyword."
    )
    messages = [{"role": "user", "content": prompt + f'\n\nText:\n"{text[:2000]}"'}]
    resp = query_ai(messages)
    if not resp:
        return "general"
    disc = resp.strip().lower()
    if disc in ["medicine", "sociology", "science", "engineering"]:
        return disc
    return "general"

def map_discipline_to_style(discipline):
    mapping = {
        "medicine": "APA",
        "sociology": "MLA",
        "science": "Chicago",
        "engineering": "IEEE",
        "general": "APA",
    }
    return mapping.get(discipline, "APA")

def build_google_books_query_from_text(text):
    sentences = text.split(".")
    query_snippet = ". ".join(sentences[:3]).strip()
    words = query_snippet.split()
    if len(words) > 50:
        query_snippet = " ".join(words[:50])
    return query_snippet

def ai_guess_book_metadata(text):
    if API_QUOTA_EXCEEDED:
        return {}
    
    prompt = (
        "You are an assistant that identifies books from text excerpts.\n"
        'If this text comes from a known book, return a JSON object with:\n'
        '{"title": "Book Title", "authors": ["Author1", "Author2"]}\n'
        "If unknown, return {}.\n"
        f"Text:\n{text[:1000]}"
    )
    messages = [{"role": "user", "content": prompt}]
    response = query_ai(messages)
    if not response:
        return {}

    try:
        start = response.index("{")
        end = response.rindex("}") + 1
        data = json.loads(response[start:end])
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def lookup_book_in_google_books(text):
    if not GOOGLE_BOOKS_API_KEY:
        print("GOOGLE_BOOKS_API_KEY missing; skipping Google Books lookup.")
        return None

    query = quote_plus(text)
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=1&key={GOOGLE_BOOKS_API_KEY}"

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            return None

        volume_info = items[0].get("volumeInfo", {})
        return {
            "title": volume_info.get("title", ""),
            "authors": volume_info.get("authors", []),
            "publisher": volume_info.get("publisher", ""),
            "published_date": volume_info.get("publishedDate", ""),
        }
    except Exception as e:
        print(f"Google Books API error: {e}")
        return None

def search_crossref(query, max_results=3):
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": max_results}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
        papers = []
        for item in items:
            authors = ", ".join(
                f"{a.get('given','')} {a.get('family','')}".strip()
                for a in item.get("author", [])
            )
            year = (
                item.get("published-print", item.get("published-online", {}))
                .get("date-parts", [[None]])[0][0]
                or ""
            )
            papers.append(
                {
                    "source": "Crossref",
                    "title": item.get("title", [""])[0],
                    "authors": authors,
                    "year": year,
                    "url": item.get("URL", ""),
                }
            )
        return papers
    except Exception as e:
        print(f"Crossref API error: {e}")
        return []

def scrape_google_scholar(query, limit=3):
    if not SCRAPEDO_API_KEY:
        print("SCRAPEDO_API_KEY missing; skipping Google Scholar scraping.")
        return []

    scholar_url = f"https://scholar.google.com/scholar?hl=en&q={quote_plus(query)}"
    encoded_url = quote_plus(scholar_url)
    api_url = (
        f"https://api.scrape.do/"
        f"?token={SCRAPEDO_API_KEY}"
        f"&url={encoded_url}"
        f"&render=true"
    )

    try:
        response = requests.get(api_url, timeout=60)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        paper_containers = soup.find_all("div", class_="gs_ri")
        results = []

        for paper in paper_containers[:limit]:
            title_elem = paper.find("h3", class_="gs_rt")
            authors_line = paper.find("div", class_="gs_a")

            title = title_elem.get_text(" ", strip=True) if title_elem else "No Title Found"
            title = re.sub(r"\[\w+\]\s*", "", title).strip()

            url = title_elem.find("a")["href"] if title_elem and title_elem.find("a") else None

            authors_text = authors_line.get_text().split("-")[0].strip() if authors_line else ""
            authors = [a.strip() for a in authors_text.split(",")]

            year_match = re.search(r"\b\d{4}\b", authors_line.get_text()) if authors_line else None
            year = year_match.group(0) if year_match else None

            results.append(
                {
                    "source": "Google Scholar (Scraped via scrape.do)",
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "url": url,
                }
            )
        return results

    except Exception as e:
        print(f"Google Scholar scraping failed: {e}")
        return []

def search_arxiv(query, max_results=3):
    url = f"http://export.arxiv.org/api/query?search_query=all:{quote_plus(query)}&start=0&max_results={max_results}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in root.findall("atom:entry", ns):
            title_elem = entry.find("atom:title", ns)
            published_elem = entry.find("atom:published", ns)
            id_elem = entry.find("atom:id", ns)
            title = title_elem.text.strip() if title_elem is not None else "No Title"
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
            year = published_elem.text[:4] if published_elem is not None else ""
            url = id_elem.text if id_elem is not None else ""
            papers.append(
                {
                    "source": "arXiv",
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "url": url,
                }
            )
        return papers
    except Exception as e:
        print(f"arXiv API error: {e}")
        return []

def search_pubmed(query, max_results=3):
    if not PUBMED_API_KEY:
        print("PUBMED_API_KEY missing; skipping PubMed lookup.")
        return []
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    # First, search for article IDs
    search_url = f"{base_url}esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "api_key": PUBMED_API_KEY
    }
    
    try:
        # Get article IDs
        search_response = requests.get(search_url, params=search_params, timeout=15)
        search_response.raise_for_status()
        search_data = search_response.json()
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        
        if not id_list:
            return []
        
        # Then fetch details for these articles
        fetch_url = f"{base_url}efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
            "api_key": PUBMED_API_KEY
        }
        
        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=15)
        fetch_response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(fetch_response.content)
        papers = []
        
        for article in root.findall(".//PubmedArticle"):
            medline_citation = article.find("MedlineCitation")
            article_data = medline_citation.find("Article")
            
            # Extract title
            title_elem = article_data.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "No Title"
            
            # Extract authors
            authors = []
            author_list = article_data.find(".//AuthorList")
            if author_list is not None:
                for author in author_list.findall("Author"):
                    last_name = author.find("LastName")
                    fore_name = author.find("ForeName")
                    if last_name is not None and fore_name is not None:
                        authors.append(f"{fore_name.text} {last_name.text}")
            
            # Extract publication date
            pub_date = article_data.find(".//PubDate")
            year = ""
            if pub_date is not None:
                year_elem = pub_date.find("Year")
                year = year_elem.text if year_elem is not None else ""
            
            # Extract URL
            pmid_elem = medline_citation.find("PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            
            papers.append({
                "source": "PubMed",
                "title": title,
                "authors": authors,
                "year": year,
                "url": url
            })
            
        return papers
    
    except Exception as e:
        print(f"PubMed API error: {e}")
        return []

def search_openalex(query, max_results=3):
    url = "https://api.openalex.org/works"
    params = {"search": query, "per-page": max_results}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        papers = []
        for item in data.get("results", []):
            authors = [author.get("author", {}).get("display_name", "") for author in item.get("authorships", [])]
            title = item.get("title", "No Title")
            year = item.get("publication_year", "")
            url = item.get("id", "").replace("https://openalex.org/", "https://doi.org/")
            papers.append(
                {
                    "source": "OpenAlex",
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "url": url,
                }
            )
        return papers
    except Exception as e:
        print(f"OpenAlex API error: {e}")
        return []

def aggregate_papers(query, max_per_source=3):
    papers = []
    papers.extend(search_crossref(query, max_results=max_per_source))
    papers.extend(scrape_google_scholar(query, limit=max_per_source))
    papers.extend(search_arxiv(query, max_results=max_per_source))
    papers.extend(search_pubmed(query, max_results=max_per_source))
    papers.extend(search_openalex(query, max_results=max_per_source))
    seen = set()
    dedup_papers = []
    for p in papers:
        key = p.get("url") or p.get("title").lower() or ""
        if key and key not in seen:
            seen.add(key)
            dedup_papers.append(p)
    return dedup_papers

def generate_citation_from_book_metadata(book_metadata, style):
    if API_QUOTA_EXCEEDED:
        # Simple fallback citation format
        title = book_metadata.get("title", "Unknown Title")
        authors = clean_author_string(book_metadata.get("authors", []))
        publisher = book_metadata.get("publisher", "Unknown Publisher")
        year = book_metadata.get("published_date", "n.d.")
        
        if style == "APA":
            return f"{authors} ({year}). {title}. {publisher}."
        elif style == "MLA":
            return f"{authors}. {title}. {publisher}, {year}."
        elif style == "Chicago":
            return f"{authors}. {year}. {title}. {publisher}."
        elif style == "IEEE":
            return f"[1] {authors}, {title}. {publisher}, {year}."
        else:
            return f"{authors}. {title}. {publisher}, {year}."
    
    title = book_metadata.get("title", "")
    authors = clean_author_string(book_metadata.get("authors", []))
    publisher = book_metadata.get("publisher", "")
    year = book_metadata.get("published_date", "")
    prompt = (
        f"Format the following book metadata as a {style} citation ONLY.\n"
        f"Return ONLY the citation, no extra text.\n\n"
        f"Title: {title}\nAuthors: {authors}\nPublisher: {publisher}\nYear: {year}"
    )
    
    response = query_ai([{"role": "user", "content": prompt}])
    if response is None:
        return "Citation could not be generated."
    citation = response.strip()
    return citation.split("\n")[0]

def generate_citation_from_text(article_text: str, style: str):
    if API_QUOTA_EXCEEDED:
        # Simple fallback citation
        return f"Citation unavailable due to API quota. Please try again later. ({style} style)"
    
    prompt = (
        f"You are a professional citation assistant.\n"
        f"Format the following article text into a {style} style bibliographic citation.\n"
        f"Return ONLY the formatted citation. No additional explanation or text.\n\n"
        f"Article Text:\n{article_text}\n"
    )
    messages = [
        {"role": "system", "content": "You are a precise citation assistant."},
        {"role": "user", "content": prompt},
    ]
    response = query_ai(messages)
    if response is None:
        return "Citation could not be generated."
    citation = response.strip()
    return citation.split("\n")[0]

def generate_citation(paper_metadata, style):
    if API_QUOTA_EXCEEDED:
        # Simple fallback citation
        authors = clean_author_string(paper_metadata.get("authors", []))
        title = paper_metadata.get("title", "Unknown Title")
        year = paper_metadata.get("year", "n.d.")
        url = paper_metadata.get("url", "")
        
        if style == "APA":
            return f"{authors} ({year}). {title}. Retrieved from {url}"
        elif style == "MLA":
            return f"{authors}. \"{title}\". {year}. {url}"
        elif style == "Chicago":
            return f"{authors}. {year}. \"{title}\". {url}"
        elif style == "IEEE":
            return f"[1] {authors}, \"{title}\", {year}. {url}"
        else:
            return f"{authors}. {title}. {year}. {url}"
    
    authors = clean_author_string(paper_metadata.get("authors", []))
    title = paper_metadata.get("title", "")
    year = paper_metadata.get("year", "")
    url = paper_metadata.get("url", "")
    prompt = (
        f"Format the following metadata as a {style} citation ONLY.\n"
        f"Return ONLY the formatted citation. No additional explanation or text.\n\n"
        f"Title: {title}\nAuthors: {authors}\nYear: {year}\nURL/DOI: {url}"
    )
    citation = query_ai([{"role": "user", "content": prompt}]).strip()
    time.sleep(API_CALL_DELAY)
    return citation.split("\n")[0] if citation else "Citation could not be generated."

def main():
    global API_QUOTA_EXCEEDED
    print("Paste your article text (finish input with an empty line):")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)

    article_text = " ".join(lines).strip()
    if not article_text:
        print("No article text provided. Exiting.")
        return

    # Step 1: AI guess book metadata from text
    book_guess = ai_guess_book_metadata(article_text)

    # Step 2: Identify discipline and map to citation style
    discipline = identify_discipline(article_text)
    style = map_discipline_to_style(discipline)

    print(f"\nIdentified academic discipline: {discipline.capitalize()}")
    print(f"Using citation style: {style}\n")

    if API_QUOTA_EXCEEDED:
        print("⚠️ Running in fallback mode due to API quota limitations.\n")

    # Step 3: Use AI guess if valid metadata
    if book_guess.get("title") and book_guess.get("authors"):
        citation = generate_citation_from_book_metadata(book_guess, style)
        print(f"Citation for your article (AI-identified book metadata) ({style}):\n{citation}\n")
    else:
        # Step 4: If AI guess unavailable, use Google Books API lookup with snippet
        query_snippet = build_google_books_query_from_text(article_text)
        book_meta = lookup_book_in_google_books(query_snippet)
        if book_meta:
            citation = generate_citation_from_book_metadata(book_meta, style)
            print(f"Citation for your article (Google Books metadata) ({style}):\n{citation}\n")
        else:
            # Step 5: Fallback to AI-generated citation from full text
            citation = generate_citation_from_text(article_text, style)
            print(f"Citation for your article (AI-generated from text) ({style}):\n{citation}\n")

if __name__ == "__main__":
    main()