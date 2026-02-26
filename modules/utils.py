"""Shared utilities: sitemap parsing, web scraping, text cleaning, Gemini wrapper."""

import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import time
import random
import re
from urllib.parse import urljoin, urlparse


# ── Gemini wrapper ─────────────────────────────────────────────────────────────

# Ordre de préférence — utilisé comme fallback si le modèle choisi est indisponible
GEMINI_PREFERRED_ORDER = [
    "gemini-2.5-flash",        # Meilleur rapport qualité/vitesse — recommandé
    "gemini-2.5-pro",          # Qualité maximale
    "gemini-2.0-flash",        # Rapide et fiable
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


def list_available_gemini_models(client) -> list[str]:
    """
    Interroge l'API pour obtenir la liste des modèles réellement disponibles
    avec cette clé, dans l'ordre de préférence défini.
    """
    try:
        available = set()
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            # L'API retourne "models/gemini-2.0-flash" → on garde "gemini-2.0-flash"
            name = name.removeprefix("models/")
            if "gemini" in name:
                available.add(name)
        # Retourne dans l'ordre de préférence, en ne gardant que ceux disponibles
        ordered = [m for m in GEMINI_PREFERRED_ORDER if m in available]
        # Ajoute ceux non listés dans PREFERRED_ORDER mais retournés par l'API
        extras = sorted(available - set(GEMINI_PREFERRED_ORDER))
        return ordered + extras if ordered else list(available)
    except Exception as e:
        print(f"[GeminiWrapper] Impossible de lister les modèles : {e}")
        return GEMINI_PREFERRED_ORDER  # fallback statique


class GeminiWrapper:
    """
    Thin wrapper around google.genai client.
    Découvre dynamiquement les modèles disponibles et utilise le meilleur.
    """
    def __init__(self, client, model_name: str, available_models: list[str]):
        self._client           = client
        self._model_name       = model_name
        self._available_models = available_models  # modèles réels de cette clé

    def generate_content(self, prompt: str, _retries: int = 2):
        # Priorité : modèle demandé → modèles disponibles dans l'ordre de préférence
        candidates = [self._model_name] + [
            m for m in self._available_models if m != self._model_name
        ]
        last_error = None
        for model in candidates:
            # Retry loop pour les erreurs temporaires (503, overloaded)
            for attempt in range(_retries + 1):
                try:
                    response = self._client.models.generate_content(
                        model=model,
                        contents=prompt,
                    )
                    if model != self._model_name:
                        print(f"[GeminiWrapper] Fallback utilisé : {model} (demandé : {self._model_name})")
                    return response

                except Exception as e:
                    err_str = str(e).lower()

                    # Modèle indisponible (404/403) → passe au suivant sans retry
                    if any(k in err_str for k in ["404", "not_found", "not found", "403", "permission"]):
                        last_error = e
                        break  # sort du retry loop → passe au modèle suivant

                    # Quota épuisé (429) → message clair, on arrête tout
                    if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                        raise RuntimeError(
                            "⚠️ Quota Gemini épuisé (erreur 429).\n\n"
                            "Solution : active la facturation sur ton projet Google Cloud.\n"
                            "→ console.cloud.google.com/billing\n\n"
                            f"Détail : {str(e)[:200]}"
                        ) from e

                    # Surchargé / indisponible temporairement (503, overloaded) → retry avec délai
                    if any(k in err_str for k in ["503", "unavailable", "overloaded", "temporarily"]):
                        if attempt < _retries:
                            wait = 5 * (attempt + 1)  # 5s, 10s
                            print(f"[GeminiWrapper] {model} surchargé, attente {wait}s (tentative {attempt+1}/{_retries})…")
                            time.sleep(wait)
                            continue  # retry même modèle
                        else:
                            # Retries épuisés → passe au modèle suivant
                            print(f"[GeminiWrapper] {model} toujours indisponible après {_retries} retries → fallback")
                            last_error = e
                            break

                    # Autre erreur inconnue → remonte immédiatement
                    raise

        raise RuntimeError(
            f"Aucun modèle Gemini n'a pu répondre.\n"
            f"Modèles testés : {candidates}\n"
            f"Dernière erreur : {last_error}"
        )


def make_gemini(api_key: str, model_name: str) -> "GeminiWrapper":
    """
    Crée un GeminiWrapper en découvrant automatiquement les modèles disponibles.
    """
    from google import genai
    client = genai.Client(api_key=api_key)
    available = list_available_gemini_models(client)
    print(f"[GeminiWrapper] Modèles disponibles : {available}")
    return GeminiWrapper(client, model_name, available)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

SITEMAP_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "image": "http://www.google.com/schemas/sitemap-image/1.1",
    "news": "http://www.google.com/schemas/sitemap-news/0.9",
}


def fetch_sitemap(url: str, depth: int = 0, max_depth: int = 2) -> list[str]:
    """Fetch all page URLs from a sitemap or sitemap index (recursive)."""
    if depth > max_depth:
        return []

    urls: list[str] = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        content = resp.text.strip()

        # Try XML parsing
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            return []

        tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag

        if tag == "sitemapindex":
            # Recursively fetch child sitemaps (limit to first 8)
            for sitemap_el in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")[:8]:
                child_url = sitemap_el.text.strip()
                urls.extend(fetch_sitemap(child_url, depth + 1, max_depth))
        else:
            # Regular sitemap — collect all <loc> elements
            for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
                raw = loc.text.strip()
                urls.append(raw)

    except Exception as e:
        print(f"[utils] Sitemap fetch error ({url}): {e}")

    return list(dict.fromkeys(urls))  # deduplicate, preserve order


def scrape_page_text(url: str, max_chars: int = 5000) -> str:
    """Scrape and clean main text content from a web page."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove noise elements
        for tag in soup.find_all(["script", "style", "nav", "footer", "header",
                                   "aside", "form", "noscript", "iframe"]):
            tag.decompose()

        # Prefer main content containers
        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find(class_=re.compile(r"(content|article|post|entry)", re.I))
            or soup.body
        )

        text = main.get_text(separator=" ", strip=True) if main else ""
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]

    except Exception as e:
        print(f"[utils] Page scrape error ({url}): {e}")
        return ""


def random_delay(min_s: float = 1.5, max_s: float = 3.5) -> None:
    time.sleep(random.uniform(min_s, max_s))


def slug_from_keyword(keyword: str) -> str:
    """Convert a keyword to a URL-friendly slug."""
    slug = keyword.lower().strip()
    slug = re.sub(r"[àáâãäå]", "a", slug)
    slug = re.sub(r"[èéêë]", "e", slug)
    slug = re.sub(r"[ìíîï]", "i", slug)
    slug = re.sub(r"[òóôõö]", "o", slug)
    slug = re.sub(r"[ùúûü]", "u", slug)
    slug = re.sub(r"[ç]", "c", slug)
    slug = re.sub(r"[ñ]", "n", slug)
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "-", slug)
    return slug.strip("-")


def extract_json_from_text(text: str) -> str | None:
    """Extract the first JSON object or array from a text string."""
    # Try to find ```json ... ``` block first
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)

    # Fall back to raw JSON block
    raw = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if raw:
        return raw.group(1)

    return None
