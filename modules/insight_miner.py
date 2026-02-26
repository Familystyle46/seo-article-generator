"""Étape 2b — Insight Miner: génère les questions et angles via Gemini."""

import json

from .utils import extract_json_from_text


def generate_insights(keyword: str, search_intent: str, gemini_model) -> dict:
    """
    Ask Gemini to generate People Also Ask questions, LSI keywords,
    user problems and editorial angles for the given keyword.
    """
    prompt = f"""Tu es un expert en intention de recherche SEO et en psychologie des internautes.

MOT-CLÉ CIBLE : "{keyword}"
INTENTION DE RECHERCHE : {search_intent}

Génère une analyse complète de ce que les internautes recherchent vraiment quand ils tapent ce mot-clé.

Réponds UNIQUEMENT avec un objet JSON valide, aucun texte avant ni après :
{{
  "people_also_ask": [
    "Question précise que les gens posent sur ce sujet ?",
    "Question 2 ?",
    "Question 3 ?",
    "Question 4 ?",
    "Question 5 ?"
  ],
  "related_subtopics": [
    "Sous-thème ou angle important 1",
    "Sous-thème 2",
    "Sous-thème 3",
    "Sous-thème 4",
    "Sous-thème 5"
  ],
  "user_problems": [
    "Problème concret que l'utilisateur cherche à résoudre",
    "Problème 2",
    "Problème 3"
  ],
  "editorial_angles": [
    "Angle éditorial original et différenciant pour se démarquer des concurrents",
    "Angle 2",
    "Angle 3"
  ],
  "lsi_keywords": [
    "mot-clé sémantiquement lié 1",
    "lsi 2",
    "lsi 3",
    "lsi 4",
    "lsi 5",
    "lsi 6",
    "lsi 7"
  ],
  "content_gaps": [
    "Ce que les concurrents ne couvrent probablement pas bien",
    "Gap 2"
  ]
}}"""

    response = gemini_model.generate_content(prompt)
    raw = response.text.strip()

    json_str = extract_json_from_text(raw)
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Minimal fallback
    return {
        "people_also_ask": [],
        "related_subtopics": [],
        "user_problems": [],
        "editorial_angles": [],
        "lsi_keywords": [],
        "content_gaps": [],
    }
