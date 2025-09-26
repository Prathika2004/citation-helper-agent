from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

from app import (
    ai_guess_book_metadata,
    identify_discipline,
    map_discipline_to_style,
    build_google_books_query_from_text,
    lookup_book_in_google_books,
    generate_citation_from_book_metadata,
    generate_citation_from_text,
    extract_keywords,
    aggregate_papers,
    generate_citation
)
app = FastAPI()

# ✅ Allow frontend (HTML/JS) to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

class TextInput(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serve the index.html file from /static folder"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/generate-citation")
async def generate_citation_api(data: TextInput):
    article_text = data.text

    # Identify discipline and style
    discipline = identify_discipline(article_text)
    style = map_discipline_to_style(discipline)

    # Try AI guess
    book_guess = ai_guess_book_metadata(article_text)
    if book_guess.get("title") and book_guess.get("authors"):
        citation = generate_citation_from_book_metadata(book_guess, style)
    else:
        query_snippet = build_google_books_query_from_text(article_text)
        book_meta = lookup_book_in_google_books(query_snippet)
        if book_meta:
            citation = generate_citation_from_book_metadata(book_meta, style)
        else:
            citation = generate_citation_from_text(article_text, style)

    # Related papers
    keywords = extract_keywords(article_text)
    papers = aggregate_papers(" ".join(keywords), max_per_source=2)

    related = []
    for p in papers:
        related.append({
            "title": p["title"],
            "authors": p["authors"],
            "year": p.get("year", ""),
            "url": p.get("url", ""),
            "citation": generate_citation(p, style)
        })

    return {
        "discipline": discipline,
        "style": style,
        "citation": citation,
        "related": related
    }
