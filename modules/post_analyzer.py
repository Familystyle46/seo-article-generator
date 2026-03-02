"""
Étape 5 — Analyse post-rédaction via Gemini.
Score SEO 0-100, variantes titres H1, variantes meta, quick wins.
"""

import json
import re

from .utils import extract_json_from_text


# ── Score SEO ──────────────────────────────────────────────────────────────────

def score_article_seo(
    article_markdown: str,
    keyword: str,
    brief: dict,
    gemini_model,
) -> dict:
    """
    Analyse l'article et retourne un score SEO 0-100 avec recommandations.
    """
    word_count  = len(article_markdown.split())
    kw_lower    = keyword.lower()
    kw_count    = len(re.findall(re.escape(kw_lower), article_markdown.lower()))
    kw_density  = round((kw_count / max(word_count, 1)) * 100, 2)
    h2_count    = len(re.findall(r'^## ', article_markdown, re.MULTILINE))
    h3_count    = len(re.findall(r'^### ', article_markdown, re.MULTILINE))
    link_count  = len(re.findall(r'\[.+?\]\(https?://', article_markdown))
    has_faq     = bool(re.search(r'faq|questions fréquentes', article_markdown, re.IGNORECASE))
    has_figure  = bool(re.search(r'<figure', article_markdown, re.IGNORECASE))

    prompt = f"""Tu es un auditeur SEO expert. Analyse cet article et donne un score précis 0-100.

MOT-CLÉ PRINCIPAL : "{keyword}"

STATISTIQUES AUTO-CALCULÉES :
- Mots : {word_count}
- Densité mot-clé : {kw_density}% ({kw_count} occurrences)
- Structure : {h2_count} H2, {h3_count} H3
- Liens internes : {link_count}
- FAQ présente : {"Oui" if has_faq else "Non"}
- Balises médias : {"Oui" if has_figure else "Non"}

DÉBUT DE L'ARTICLE (2 500 premiers caractères) :
{article_markdown[:2500]}

Évalue sur 6 critères (0-100 chacun) et donne 3 actions concrètes prioritaires.

Réponds UNIQUEMENT avec un JSON valide :
{{
  "score_global": 78,
  "scores": {{
    "keyword_optimization": 85,
    "structure": 80,
    "content_depth": 75,
    "readability": 70,
    "eeat": 65,
    "internal_linking": 60
  }},
  "points_forts": ["Force 1 (soyez spécifique)", "Force 2"],
  "actions_prioritaires": [
    {{"action": "Action très concrète et spécifique", "impact": "fort", "effort": "faible"}},
    {{"action": "Action 2", "impact": "moyen", "effort": "moyen"}},
    {{"action": "Action 3", "impact": "faible", "effort": "faible"}}
  ],
  "verdict": "Synthèse en 1-2 phrases sur la qualité globale et le potentiel de classement"
}}"""

    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()
        json_str = extract_json_from_text(raw)
        if json_str:
            data = json.loads(json_str)
            data.update({
                "word_count": word_count,
                "kw_density": kw_density,
                "kw_count": kw_count,
                "h2_count": h2_count,
                "link_count": link_count,
            })
            return data
    except Exception as e:
        print(f"[post_analyzer] Score erreur : {e}")

    return {
        "score_global": 0,
        "scores": {},
        "points_forts": [],
        "actions_prioritaires": [],
        "verdict": "Analyse indisponible",
        "word_count": word_count,
        "kw_density": kw_density,
        "kw_count": kw_count,
        "h2_count": h2_count,
        "link_count": link_count,
    }


# ── Variantes titres H1 ────────────────────────────────────────────────────────

def generate_headline_variants(
    keyword: str,
    brief: dict,
    gemini_model,
) -> list[dict]:
    """
    Génère 8 variantes de titres H1 avec score CTR estimé.
    Formules : chiffres, question, guide, erreurs, comparaison, curiosité, urgence, résultat.
    """
    prompt = f"""Tu es un expert copywriting SEO spécialisé dans les titres à fort CTR.

MOT-CLÉ : "{keyword}"
H1 ACTUEL : "{brief.get('h1', keyword)}"
INTENTION : {brief.get('search_intent', 'informational')}
AUDIENCE : {brief.get('target_audience', '')}

Génère 8 variantes de titres H1 alternatives avec des formules différentes.
Chaque titre : contient le mot-clé, 50-65 caractères idéalement, style unique.

Formules à utiliser (une par titre) :
- Chiffres (ex: "7 façons de...")
- Question directe
- "Comment + résultat concret"
- "Guide complet / Tout savoir sur"
- "Erreurs à éviter"
- Comparaison / Versus
- Curiosité / Contre-intuition
- Urgence / Actualité

Réponds UNIQUEMENT avec un JSON valide :
{{
  "variants": [
    {{
      "title": "Titre H1 alternatif",
      "formula": "Chiffres",
      "ctr_boost": "+18%",
      "chars": 58,
      "note": "Les titres chiffrés ont 36% plus de clics en moyenne"
    }}
  ]
}}"""

    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()
        json_str = extract_json_from_text(raw)
        if json_str:
            return json.loads(json_str).get("variants", [])
    except Exception as e:
        print(f"[post_analyzer] Headlines erreur : {e}")
    return []


# ── Variantes meta description ─────────────────────────────────────────────────

def generate_meta_variants(
    keyword: str,
    brief: dict,
    gemini_model,
) -> list[dict]:
    """
    Génère 5 variantes de meta description (150-160 chars) avec angles différents.
    """
    prompt = f"""Tu es un expert SEO spécialisé dans les meta descriptions à fort CTR.

MOT-CLÉ : "{keyword}"
META ACTUELLE : "{brief.get('meta_description', '')}"

Génère 5 alternatives de meta descriptions. Chacune : 150-160 caractères exactement,
contient le mot-clé, angle unique et différent de la meta actuelle.

Angles à utiliser : Urgence, Bénéfice chiffré, Question, Preuve sociale, Promesse concrète.

Réponds UNIQUEMENT avec un JSON valide :
{{
  "variants": [
    {{
      "meta": "Meta description 150-160 caractères avec le mot-clé inclus naturellement...",
      "angle": "Urgence",
      "chars": 157
    }}
  ]
}}"""

    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()
        json_str = extract_json_from_text(raw)
        if json_str:
            return json.loads(json_str).get("variants", [])
    except Exception as e:
        print(f"[post_analyzer] Meta erreur : {e}")
    return []


# ── Quick Wins ─────────────────────────────────────────────────────────────────

def find_quick_wins(
    sitemap_urls: list[str],
    current_keyword: str,
    category_hint: str,
    gemini_model,
) -> list[dict]:
    """
    Identifie 5 opportunités de contenu non encore couvertes (quick wins SEO).
    Basé sur l'analyse des slugs du sitemap et de la niche.
    """
    slugs = []
    for url in sitemap_urls[:80]:
        parts = url.rstrip("/").split("/")
        if len(parts) >= 4:
            slug = parts[-1].replace("-", " ").replace("_", " ")
            if slug:
                slugs.append(slug)

    slugs_sample = "\n".join(f"- {s}" for s in slugs[:50]) if slugs else "(sitemap vide)"

    prompt = f"""Tu es un stratège de contenu SEO. Identifie les 5 meilleures opportunités
de quick wins — articles à fort potentiel SEO non encore couverts par ce site.

NICHE / SECTEUR : {category_hint or current_keyword}
MOT-CLÉ TRAITÉ CE JOUR : {current_keyword}

SUJETS DÉJÀ COUVERTS SUR LE SITE (slugs des URLs) :
{slugs_sample}

Identifie 5 opportunités qui :
1. Ne sont PAS encore couverts (d'après les slugs)
2. Sont complémentaires / proches du mot-clé actuel
3. Ont un fort potentiel de trafic organique
4. Peuvent se classer rapidement (faible concurrence probable)

Réponds UNIQUEMENT avec un JSON valide :
{{
  "quick_wins": [
    {{
      "keyword": "mot-clé exact à cibler",
      "type": "Informationnel / Guide / Comparaison / Définition",
      "potential": "fort / moyen / faible",
      "suggested_title": "Titre H1 suggéré pour l'article",
      "reason": "Pourquoi c'est une opportunité (1 phrase)"
    }}
  ]
}}"""

    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()
        json_str = extract_json_from_text(raw)
        if json_str:
            return json.loads(json_str).get("quick_wins", [])
    except Exception as e:
        print(f"[post_analyzer] Quick wins erreur : {e}")
    return []
