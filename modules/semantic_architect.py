"""
Étape 2c — Semantic Architect : construit le brief éditorial complet via Gemini.
Étape 2d — Maillage Architect : identifie les liens internes depuis le sitemap.
"""

import json

from .utils import extract_json_from_text


# ── Brief Builder ──────────────────────────────────────────────────────────────

def build_brief(
    keyword_data: dict,
    competitor_summary: str,
    insights: dict,
    target_words: int,
    gemini_model,
) -> dict:
    """
    Build a complete editorial brief with H1/H2/H3 structure,
    key points, media slots, and FAQ using Gemini.
    """
    keyword        = keyword_data.get("keyword", "")
    search_intent  = keyword_data.get("search_intent", "informational")
    target_audience = keyword_data.get("target_audience", "")
    angle          = keyword_data.get("angle_editorial", "")
    variants       = keyword_data.get("semantic_variants", [])
    lsi_keywords   = insights.get("lsi_keywords", [])
    paa            = insights.get("people_also_ask", [])
    subtopics      = insights.get("related_subtopics", [])
    user_problems  = insights.get("user_problems", [])
    content_gaps   = insights.get("content_gaps", [])

    prompt = f"""Tu es un Architecte SEO expert. Construis un brief éditorial détaillé et actionnable pour un article de blog optimisé SEO.

━━━━━━━━━━━━━━━━━━━━━━━━━━
MOT-CLÉ PRINCIPAL : {keyword}
INTENTION DE RECHERCHE : {search_intent}
AUDIENCE CIBLE : {target_audience}
ANGLE ÉDITORIAL : {angle}
LONGUEUR CIBLE : {target_words} mots
VARIANTES SÉMANTIQUES : {', '.join(variants)}
━━━━━━━━━━━━━━━━━━━━━━━━━━

ANALYSE CONCURRENTS (top résultats Google) :
{competitor_summary[:3000] if competitor_summary else "Non disponible"}

QUESTIONS DES INTERNAUTES (People Also Ask) :
{chr(10).join(f"- {q}" for q in paa)}

SOUS-THÈMES IMPORTANTS :
{chr(10).join(f"- {s}" for s in subtopics)}

PROBLÈMES UTILISATEURS À RÉSOUDRE :
{chr(10).join(f"- {p}" for p in user_problems)}

MOTS-CLÉS LSI À INTÉGRER NATURELLEMENT :
{', '.join(lsi_keywords)}

LACUNES DES CONCURRENTS (opportunités) :
{chr(10).join(f"- {g}" for g in content_gaps)}

━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCTIONS POUR LE BRIEF :
- H1 : doit contenir le mot-clé principal de façon naturelle et accrocheuse
- Meta description : 150-160 caractères, mot-clé inclus, incite au clic
- H2 : 4 à 6 sections principales, utilise des variantes sémantiques (pas le mot-clé exact répété à chaque fois)
- H3 : 2-3 sous-sections par H2 maximum, points précis et actionnables
- Prévois 2-3 emplacements pour images/médias avec description
- Section FAQ obligatoire avec 4-5 questions des internautes
- Intro : hook accrocheur + pyramide inversée (répondre d'abord, développer ensuite)
- L'article doit couvrir l'intention complète et se démarquer des concurrents
━━━━━━━━━━━━━━━━━━━━━━━━━━

Réponds UNIQUEMENT avec un objet JSON valide, aucun texte avant ni après :
{{
  "h1": "Le titre H1 complet et accrocheur (contient le mot-clé)",
  "meta_description": "Meta description de 150-160 caractères avec le mot-clé, incitant au clic",
  "intro_hook": "Première phrase d'accroche ultra-engageante pour l'intro (pose un problème ou une question)",
  "intro_key_points": ["Point essentiel à couvrir dans l'intro 1", "Point 2", "Point 3"],
  "sections": [
    {{
      "id": "s1",
      "h2": "Titre H2 original (variante sémantique)",
      "lsi_to_use": ["lsi keyword à placer dans cette section"],
      "key_points": ["Point clé 1 à développer", "Point clé 2", "Point clé 3"],
      "media_suggestion": "Description de l'image ou schéma idéal à insérer ici (ou null)",
      "subsections": [
        {{
          "id": "s1-1",
          "h3": "Titre H3 précis",
          "key_points": ["Point 1 du H3", "Point 2"]
        }},
        {{
          "id": "s1-2",
          "h3": "Titre H3 suivant",
          "key_points": ["Point 1", "Point 2"]
        }}
      ]
    }},
    {{
      "id": "s2",
      "h2": "Deuxième section H2",
      "lsi_to_use": ["autre lsi"],
      "key_points": ["Point clé 1", "Point clé 2"],
      "media_suggestion": null,
      "subsections": []
    }}
  ],
  "faq": [
    {{
      "question": "Question FAQ exacte (issue des People Also Ask) ?",
      "answer_hint": "Ce que la réponse doit absolument couvrir (2-3 éléments clés)"
    }},
    {{
      "question": "Deuxième question FAQ ?",
      "answer_hint": "Points à couvrir"
    }},
    {{
      "question": "Troisième question ?",
      "answer_hint": "Points à couvrir"
    }},
    {{
      "question": "Quatrième question ?",
      "answer_hint": "Points à couvrir"
    }}
  ],
  "conclusion_cta": "Idée pour la conclusion : ce qu'elle doit résumer + appel à l'action proposé"
}}"""

    response = gemini_model.generate_content(prompt)
    raw = response.text.strip()

    json_str = extract_json_from_text(raw)
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON invalide dans la réponse Gemini (brief): {e}\n---\n{json_str[:500]}")

    raise ValueError(f"Gemini n'a pas retourné de JSON pour le brief.\nRéponse brute : {raw[:500]}")


# ── Internal Linking ───────────────────────────────────────────────────────────

def find_internal_links(
    brief: dict,
    sitemap_urls: list[str],
    keyword: str,
    gemini_model,
) -> list[dict]:
    """
    Identify the most relevant internal links from the sitemap
    to naturally insert in the article.
    """
    if not sitemap_urls:
        return []

    # Filter to substantive URLs only (avoid category pages with <5 path segments)
    candidate_urls = [u for u in sitemap_urls if len(u.rstrip("/").split("/")) >= 4][:80]

    if not candidate_urls:
        candidate_urls = sitemap_urls[:60]

    # Build list of section titles for context
    sections_text = "\n".join(
        f"- H2: {s.get('h2', '')}" + "".join(f"\n  - H3: {ss.get('h3', '')}" for ss in s.get("subsections", []))
        for s in brief.get("sections", [])
    )

    prompt = f"""Tu es un expert en maillage interne SEO.

ARTICLE EN COURS DE RÉDACTION :
Mot-clé principal : {keyword}
H1 : {brief.get('h1', '')}

STRUCTURE DES SECTIONS :
{sections_text}

URLS DU SITE DISPONIBLES POUR MAILLAGE INTERNE :
{chr(10).join(candidate_urls[:70])}

OBJECTIF : Sélectionne 3 à 5 URLs du site à lier naturellement dans l'article.

RÈGLES :
- Choisir des pages complémentaires qui apportent une vraie valeur ajoutée au lecteur
- L'ancre de texte doit être naturelle, descriptive (jamais "cliquez ici" ou "en savoir plus")
- L'ancre doit contenir des mots-clés pertinents pour le SEO
- Préciser dans quelle section (id) placer chaque lien
- Chaque lien doit s'intégrer dans un contexte de phrase précis

Réponds UNIQUEMENT avec un objet JSON valide, aucun texte avant ni après :
{{
  "internal_links": [
    {{
      "url": "https://...",
      "anchor_text": "texte d'ancre naturel et descriptif",
      "target_section_id": "s1",
      "integration_hint": "Phrase ou contexte exact où placer ce lien (exemple : 'Après avoir expliqué X, mentionner que pour Y, voir notre guide sur...')"
    }},
    {{
      "url": "https://...",
      "anchor_text": "autre ancre",
      "target_section_id": "s2",
      "integration_hint": "Contexte d'intégration"
    }}
  ]
}}"""

    response = gemini_model.generate_content(prompt)
    raw = response.text.strip()

    json_str = extract_json_from_text(raw)
    if json_str:
        try:
            data = json.loads(json_str)
            return data.get("internal_links", [])
        except json.JSONDecodeError:
            pass

    return []
