# Citation Helper Agent

## Problem Statement
Formatting a citation correctly (APA, MLA, Chicago, IEEE) is tedious and easy to get wrong, and finding related academic sources for a given piece of text usually means manually searching several different databases (Crossref, arXiv, PubMed, OpenAlex, Google Scholar) one at a time.

## What This Project Solves
A FastAPI web app that takes a block of article/book text and:
- Identifies the academic discipline (medicine, sociology, science, engineering) and picks the citation style convention that fits it (APA/MLA/Chicago/IEEE).
- Identifies the source itself - first via an AI guess at book metadata, falling back to a Google Books lookup, falling back to generating a citation directly from the text.
- Searches five academic sources in parallel (Crossref, arXiv, PubMed, OpenAlex, and Google Scholar via a scraping API) for related papers on the same topic, deduplicates them, and formats each one as a properly styled citation.

## Approach
- **LLM reasoning** (`backend.py`): uses a Hugging Face-hosted model (`openai/gpt-oss-120b`) for discipline classification, keyword extraction, book-metadata guessing, and citation formatting.
- **Graceful degradation**: every AI-dependent function has a rule-based fallback (keyword frequency counting, keyword-matching discipline detection, template-based citation formatting) that kicks in automatically if the LLM API quota is exceeded, so the app degrades instead of failing outright.
- **Multi-source aggregation** (`backend.py`): queries Crossref, arXiv, PubMed, OpenAlex, and Google Scholar (scraped via scrape.do), then deduplicates by URL/title.
- **API layer** (`app.py`): a FastAPI server exposing `POST /generate-citation`, serving a static HTML/JS frontend.

## Tech Stack
FastAPI, Hugging Face Inference API, BeautifulSoup, Crossref/arXiv/PubMed/OpenAlex/Google Books APIs, scrape.do (Google Scholar scraping).

## How to Run
```bash
pip install -r requirements.txt
```
Create a `.env` file in the project root:
```
HF_TOKEN=your_huggingface_token_here
SCRAPEDO_API_KEY=your_scrapedo_key_here          # optional, enables Google Scholar
PUBMED_API_KEY=your_pubmed_key_here              # optional, enables PubMed
GOOGLE_BOOKS_API_KEY=your_google_books_key_here  # optional, enables book lookups
```
Then run the server:
```bash
uvicorn app:app --reload
```
Open `http://127.0.0.1:8000` in a browser.

## Limitations
- Only `HF_TOKEN` is required; the others are optional and each source is silently skipped if its key is missing.
- Google Scholar access depends on a third-party scraping service (scrape.do) rather than an official API, since Google Scholar has none.
