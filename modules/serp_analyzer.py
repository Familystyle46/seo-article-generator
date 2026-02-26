"""Étape 2a — SERP Analyzer: scrape Google et extrait le contenu des concurrents."""

import requests
from bs4 import BeautifulSoup
import re
import time
import random

from .utils import HEADERS, scrape_page_text, random_delay


GOOGLE_SEARCH_URL = "https://www.google.fr/search"

# Domains to skip (low value or hard to scrape)
SKIP_DOMAINS = {
    "google.com", "google.fr", "youtube.com", "facebook.com",
    "twitter.com", "instagram.com", "linkedin.com", "amazon.com",
    "amazon.fr", "wikipedia.org", "reddit.com", "pinterest.com",
}


def _is_blocked(html: str) -> bool:
    """Detect if Google returned a CAPTCHA or block page."""
    return any(kw in html.lower() for kw in [
        "captcha", "unusual traffic", "avant d'accéder", "trafic inhabituel",
        "verify you are human", "not a robot",
    ])


def _extract_organic_results(soup: BeautifulSoup) -> list[dict]:
    """Parse Google SERP HTML and extract organic results."""
    results: list[dict] = []

    # Strategy: find all <div class="g"> which wrap individual results
    for g in soup.find_all("div", class_="g"):
        # Find the <a> with href (the actual link)
        a_tag = g.find("a", href=True)
        h3_tag = g.find("h3")

        if not a_tag or not h3_tag:
            continue

        href = a_tag["href"]
        if not href.startswith("http"):
            continue

        # Parse domain from href
        try:
            from urllib.parse import urlparse
            domain = urlparse(href).netloc.replace("www.", "")
        except Exception:
            domain = ""

        if domain in SKIP_DOMAINS:
            continue

        # Snippet text
        snippet_el = g.find("div", attrs={"data-sncf": True}) or g.find("div", class_="VwiC3b")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""

        results.append({
            "url": href,
            "title": h3_tag.get_text(strip=True),
            "snippet": snippet[:400],
            "content": "",
        })

    return results


def scrape_google_serp(keyword: str, num_results: int = 5) -> list[dict]:
    """
    Scrape Google search results for a keyword.
    Returns a list of dicts: url, title, snippet, content.
    Falls back gracefully if blocked.
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    params = {
        "q": keyword,
        "hl": "fr",
        "gl": "fr",
        "num": num_results + 5,  # request a few extra in case some are filtered
    }

    try:
        resp = session.get(GOOGLE_SEARCH_URL, params=params, timeout=20)

        if resp.status_code != 200 or _is_blocked(resp.text):
            print(f"[serp] Google blocked or returned {resp.status_code} for '{keyword}'")
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        results = _extract_organic_results(soup)

        if not results:
            # Fallback: broader search for any <a href> pointing to external sites
            for a in soup.find_all("a", href=True)[:30]:
                href = a["href"]
                if href.startswith("http") and "google" not in href:
                    results.append({
                        "url": href,
                        "title": a.get_text(strip=True)[:80],
                        "snippet": "",
                        "content": "",
                    })
                if len(results) >= num_results:
                    break

        # Scrape full content from top results
        for i, result in enumerate(results[:num_results]):
            random_delay(0.8, 2.0)
            content = scrape_page_text(result["url"], max_chars=3500)
            results[i]["content"] = content

        return results[:num_results]

    except Exception as e:
        print(f"[serp] Error scraping Google for '{keyword}': {e}")
        return []


def build_competitor_summary(results: list[dict]) -> str:
    """
    Combine competitor results into a readable summary for the brief builder.
    """
    if not results:
        return "Aucune donnée concurrente disponible (scraping indisponible ou bloqué)."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "—")
        url   = r.get("url", "")
        snip  = r.get("snippet", "")
        body  = r.get("content", "")

        block = f"### Concurrent {i}: {title}\nURL : {url}\n"
        if snip:
            block += f"Extrait SERP : {snip}\n"
        if body:
            block += f"Contenu page (extrait) :\n{body[:1200]}\n"
        lines.append(block)

    return "\n\n".join(lines)
