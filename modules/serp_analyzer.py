"""
Étape 2a — SERP Analyzer.
Stratégie en cascade :
  1. Scraping Google
  2. Scraping DuckDuckGo HTML (fallback si Google bloqué)
  3. Analyse Gemini IA (fallback si les 2 scraping échouent)
"""

import json
import re
import time
import random

import requests
from bs4 import BeautifulSoup

from .utils import HEADERS, scrape_page_text, random_delay, extract_json_from_text


GOOGLE_URL = "https://www.google.fr/search"
DDG_URL    = "https://html.duckduckgo.com/html/"

SKIP_DOMAINS = {
    "google.com", "google.fr", "youtube.com", "facebook.com",
    "twitter.com", "x.com", "instagram.com", "linkedin.com",
    "amazon.com", "amazon.fr", "wikipedia.org", "reddit.com",
    "pinterest.com", "duckduckgo.com", "bing.com",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_blocked(html: str) -> bool:
    return any(kw in html.lower() for kw in [
        "captcha", "unusual traffic", "avant d'accéder",
        "trafic inhabituel", "verify you are human", "not a robot",
    ])


def _domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _valid_url(href: str) -> bool:
    return href.startswith("http") and _domain(href) not in SKIP_DOMAINS


# ── Strategy 1 : Google ────────────────────────────────────────────────────────

def _parse_google(soup: BeautifulSoup) -> list[dict]:
    results: list[dict] = []

    # Tentative A — sélecteur classique div.g
    for g in soup.find_all("div", class_="g"):
        a   = g.find("a", href=True)
        h3  = g.find("h3")
        if not a or not h3:
            continue
        href = a["href"]
        if not _valid_url(href):
            continue
        snip_el = (g.find("div", attrs={"data-sncf": True})
                   or g.find("div", class_="VwiC3b")
                   or g.find("span", class_="aCOpRe"))
        snippet = snip_el.get_text(" ", strip=True) if snip_el else ""
        results.append({"url": href, "title": h3.get_text(strip=True),
                         "snippet": snippet[:400], "content": ""})

    if results:
        return results

    # Tentative B — data-hveid (nouveau layout Google)
    for div in soup.find_all("div", attrs={"data-hveid": True}):
        a  = div.find("a", href=True)
        h3 = div.find("h3")
        if not a or not h3:
            continue
        href = a["href"]
        if not _valid_url(href):
            continue
        results.append({"url": href, "title": h3.get_text(strip=True),
                         "snippet": "", "content": ""})

    if results:
        return results

    # Tentative C — tout h3 dans un lien externe
    for h3 in soup.find_all("h3"):
        a = h3.find_parent("a", href=True) or h3.find("a", href=True)
        if not a:
            continue
        href = a.get("href", "")
        if not _valid_url(href):
            continue
        results.append({"url": href, "title": h3.get_text(strip=True),
                         "snippet": "", "content": ""})

    return results


def scrape_google_serp(keyword: str, num_results: int = 5) -> list[dict]:
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        resp = session.get(
            GOOGLE_URL,
            params={"q": keyword, "hl": "fr", "gl": "fr", "num": num_results + 5},
            timeout=20,
        )
        if resp.status_code != 200 or _is_blocked(resp.text):
            print(f"[serp] Google bloqué (status={resp.status_code}) pour '{keyword}'")
            return []
        soup = BeautifulSoup(resp.text, "lxml")
        results = _parse_google(soup)
        for i, r in enumerate(results[:num_results]):
            random_delay(0.5, 1.5)
            results[i]["content"] = scrape_page_text(r["url"], max_chars=3500)
        return results[:num_results]
    except Exception as e:
        print(f"[serp] Google erreur pour '{keyword}': {e}")
        return []


# ── Strategy 2 : DuckDuckGo HTML ──────────────────────────────────────────────

def _scrape_duckduckgo(keyword: str, num_results: int = 5) -> list[dict]:
    ddg_headers = {**HEADERS, "Referer": "https://duckduckgo.com/"}
    try:
        resp = requests.post(
            DDG_URL,
            data={"q": keyword, "b": "", "kl": "fr-fr"},
            headers=ddg_headers,
            timeout=20,
        )
        if resp.status_code != 200:
            print(f"[serp] DuckDuckGo status={resp.status_code}")
            return []
        soup = BeautifulSoup(resp.text, "lxml")
        results: list[dict] = []
        for div in soup.find_all("div", class_="result"):
            a = div.find("a", class_="result__a")
            if not a:
                continue
            href = a.get("href", "")
            if not _valid_url(href):
                continue
            title   = a.get_text(strip=True)
            snip_el = (div.find("a", class_="result__snippet")
                       or div.find("div", class_="result__snippet"))
            snippet = snip_el.get_text(" ", strip=True) if snip_el else ""
            results.append({"url": href, "title": title,
                             "snippet": snippet[:400], "content": ""})
            if len(results) >= num_results:
                break
        for i, r in enumerate(results[:num_results]):
            random_delay(0.5, 1.5)
            results[i]["content"] = scrape_page_text(r["url"], max_chars=3500)
        return results[:num_results]
    except Exception as e:
        print(f"[serp] DuckDuckGo erreur pour '{keyword}': {e}")
        return []


# ── Strategy 3 : Gemini IA (fallback ultime) ──────────────────────────────────

def _gemini_competitor_analysis(keyword: str, num_results: int, gemini_model) -> str:
    """
    Demande à Gemini d'analyser les concurrents typiques pour ce mot-clé.
    Retourne une string formatée comme build_competitor_summary().
    """
    prompt = f"""Tu es un expert SEO. Analyse les {num_results} articles qui se classent \
typiquement en première page Google pour : "{keyword}"

Basé sur ta connaissance du sujet, génère une analyse concurrentielle réaliste et utile.
Réponds UNIQUEMENT avec un JSON valide :
{{
  "competitors": [
    {{
      "title": "Titre réaliste d'un article concurrent",
      "domain": "nom-de-domaine-typique.fr",
      "main_angles": ["Angle/sujet principal 1", "Angle 2", "Angle 3"],
      "content_structure": ["H2 typique 1", "H2 typique 2", "H2 typique 3", "H2 4"],
      "strengths": ["Point fort 1", "Point fort 2"],
      "weaknesses": ["Lacune exploitable 1", "Lacune 2"]
    }}
  ],
  "overall_gaps": ["Opportunité de contenu 1", "Opportunité 2", "Opportunité 3"]
}}"""

    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()
        json_str = extract_json_from_text(raw)
        if not json_str:
            return f"Analyse Gemini indisponible. Réponse : {raw[:200]}"

        data = json.loads(json_str)
        competitors = data.get("competitors", [])
        gaps = data.get("overall_gaps", [])

        lines = ["[⚡ Analyse générée par Gemini IA — scraping web bloqué]\n"]
        for i, c in enumerate(competitors, 1):
            lines.append(f"### Concurrent {i} (IA): {c.get('title', '')}")
            lines.append(f"Domaine : {c.get('domain', '')}")
            angles = c.get("main_angles", [])
            if angles:
                lines.append(f"Angles : {', '.join(angles)}")
            struct = c.get("content_structure", [])
            if struct:
                lines.append(f"Structure : {' → '.join(struct)}")
            weaknesses = c.get("weaknesses", [])
            if weaknesses:
                lines.append(f"Lacunes à exploiter : {', '.join(weaknesses)}")
            lines.append("")

        if gaps:
            lines.append("### Opportunités globales :")
            for g in gaps:
                lines.append(f"  - {g}")

        return "\n".join(lines)

    except Exception as e:
        print(f"[serp] Gemini fallback erreur : {e}")
        return f"Analyse Gemini échouée : {e}"


# ── Main entry point ──────────────────────────────────────────────────────────

def get_competitor_data(
    keyword: str,
    num_results: int = 5,
    gemini_model=None,
) -> tuple[list[dict], str]:
    """
    Cascade : Google → DuckDuckGo → Gemini IA.
    Retourne (results_list, source) où source = 'google' | 'duckduckgo' | string Gemini.
    Si Gemini utilisé : results_list est vide et source contient le résumé complet.
    """
    # 1. Google
    results = scrape_google_serp(keyword, num_results)
    if results:
        return results, "google"

    # 2. DuckDuckGo
    results = _scrape_duckduckgo(keyword, num_results)
    if results:
        return results, "duckduckgo"

    # 3. Gemini fallback
    if gemini_model:
        summary = _gemini_competitor_analysis(keyword, num_results, gemini_model)
        return [], summary

    return [], "Aucune donnée concurrente disponible (scraping bloqué)."


def build_competitor_summary(results: list[dict]) -> str:
    """Formate une liste de résultats scrapés en string pour le brief."""
    if not results:
        return "Aucune donnée concurrente disponible."
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        block = f"### Concurrent {i}: {r.get('title', '—')}\nURL : {r.get('url', '')}\n"
        if r.get("snippet"):
            block += f"Extrait SERP : {r['snippet']}\n"
        if r.get("content"):
            block += f"Contenu page (extrait) :\n{r['content'][:1200]}\n"
        lines.append(block)
    return "\n\n".join(lines)
