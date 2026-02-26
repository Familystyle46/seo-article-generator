"""Étape 1 — Pépite Finder: analyse le CSV et choisit le meilleur mot-clé via Gemini."""

import io
import json
import pandas as pd

from .utils import extract_json_from_text


# ── CSV Parsing ────────────────────────────────────────────────────────────────

KEYWORD_ALIASES = ["keyword", "mot-clé", "mot clé", "mots-clés", "query", "term", "phrase", "requête"]
VOLUME_ALIASES  = ["volume", "search volume", "monthly searches", "avg. monthly searches", "searches"]
DIFF_ALIASES    = ["difficulty", "kd", "keyword difficulty", "concurrence", "competition", "score"]
CPC_ALIASES     = ["cpc", "cost per click", "coût par clic"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names based on known aliases."""
    rename_map: dict[str, str] = {}
    for col in df.columns:
        lower = col.lower().strip()
        if any(a in lower for a in KEYWORD_ALIASES) and "keyword" not in rename_map.values():
            rename_map[col] = "keyword"
        elif any(a in lower for a in VOLUME_ALIASES) and "volume" not in rename_map.values():
            rename_map[col] = "volume"
        elif any(a in lower for a in DIFF_ALIASES) and "difficulty" not in rename_map.values():
            rename_map[col] = "difficulty"
        elif any(a in lower for a in CPC_ALIASES) and "cpc" not in rename_map.values():
            rename_map[col] = "cpc"

    df = df.rename(columns=rename_map)

    # If still no 'keyword' column, use the first text-like column
    if "keyword" not in df.columns:
        for col in df.columns:
            if df[col].dtype == object:
                df = df.rename(columns={col: "keyword"})
                break

    return df


def parse_keyword_csv(file_content: bytes, encoding: str = "utf-8") -> pd.DataFrame:
    """Parse a keyword research CSV from any major tool (Ahrefs, SEMrush, etc.)."""
    # Try different encodings
    for enc in [encoding, "utf-8-sig", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(io.BytesIO(file_content), encoding=enc, on_bad_lines="skip")
            break
        except Exception:
            continue
    else:
        raise ValueError("Impossible de lire le fichier CSV. Vérifie l'encodage.")

    df = _normalize_columns(df)
    df = df.dropna(subset=["keyword"])
    df["keyword"] = df["keyword"].astype(str).str.strip()
    df = df[df["keyword"].str.len() > 2]  # remove garbage rows

    # Ensure numeric columns
    for col in ["volume", "difficulty", "cpc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce").fillna(0)

    return df.reset_index(drop=True)


# ── Keyword Selection ──────────────────────────────────────────────────────────

def find_best_keyword(
    df: pd.DataFrame,
    sitemap_urls: list[str],
    category_content: str = "",
    gemini_model=None,
) -> dict:
    """
    Ask Gemini to pick the single best keyword from the CSV,
    considering already-covered topics (sitemap) and category context.
    """
    # Serialize keyword data (max 100 rows to keep prompt manageable)
    records = []
    for _, row in df.head(100).iterrows():
        rec: dict = {"keyword": str(row.get("keyword", ""))}
        if "volume" in df.columns:
            rec["volume"] = int(row.get("volume", 0))
        if "difficulty" in df.columns:
            rec["difficulty"] = int(row.get("difficulty", 0))
        if "cpc" in df.columns:
            rec["cpc"] = float(row.get("cpc", 0))
        records.append(rec)

    # Covered topics inferred from URL slugs
    covered = [
        url.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ")
        for url in sitemap_urls[-60:]
    ]
    covered_text = "\n".join(f"- {t}" for t in covered[:60] if t)

    category_section = (
        f"\nCONTEXTE DE LA CATÉGORIE / PAGE CIBLE :\n{category_content[:2500]}\n"
        if category_content
        else ""
    )

    prompt = f"""Tu es un expert SEO spécialisé en stratégie de contenu.

LISTE DES MOTS-CLÉS DISPONIBLES (avec statistiques) :
{json.dumps(records, ensure_ascii=False, indent=2)}

SUJETS DÉJÀ COUVERTS PAR LE SITE (déduits des URLs du sitemap) :
{covered_text}
{category_section}
OBJECTIF : Sélectionne LE MEILLEUR mot-clé à cibler pour un nouvel article de blog.

CRITÈRES DE SÉLECTION (par ordre de priorité) :
1. Intention de recherche claire et exploitable (informational ou commercial)
2. Volume de recherche raisonnable (fort potentiel, sans être impossible à ranker)
3. PAS déjà couvert par une page existante du site
4. Pertinent par rapport au contexte de la catégorie (si fourni)
5. Potentiel de conversion ou de valeur pour le lecteur
6. Préférer les mots-clés de longue traîne qui ont une intention précise

Réponds UNIQUEMENT avec un objet JSON valide, aucun texte avant ni après :
{{
  "keyword": "le mot-clé choisi exactement comme dans le CSV",
  "volume": 0,
  "difficulty": 0,
  "reason": "Explication concise de ce choix (2-3 phrases)",
  "search_intent": "informational|commercial|transactional|navigational",
  "target_audience": "Description précise de l'audience cible",
  "semantic_variants": ["variante sémantique 1", "variante 2", "variante 3", "variante 4"],
  "angle_editorial": "L'angle original et différenciant à adopter pour cet article"
}}"""

    response = gemini_model.generate_content(prompt)
    raw = response.text.strip()

    json_str = extract_json_from_text(raw)
    if json_str:
        return json.loads(json_str)

    raise ValueError(f"Gemini n'a pas retourné un JSON valide pour la sélection du mot-clé.\nRéponse : {raw[:500]}")
