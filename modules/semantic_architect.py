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
    Identify the most relevant internal links from the sitemap.
    Validates all returned URLs against the sitemap to prevent hallucinations.
    """
    if not sitemap_urls:
        return []

    # Build normalised lookup sets (with and without trailing slash)
    sitemap_set_exact    = set(sitemap_urls)
    sitemap_set_stripped = {u.rstrip("/"): u for u in sitemap_urls}

    # Keep substantive pages — exclude bare homepage (≤3 path parts)
    candidate_urls = [
        u for u in sitemap_urls
        if len(u.rstrip("/").split("/")) >= 4
    ][:100]

    if not candidate_urls:
        # Site has very short URLs — use all
        candidate_urls = sitemap_urls[:80]

    # Section titles for context
    sections_text = "\n".join(
        f"- [{s.get('id', '?')}] H2: {s.get('h2', '')}"
        + "".join(
            f"\n  - [{ss.get('id', '?')}] H3: {ss.get('h3', '')}"
            for ss in s.get("subsections", [])
        )
        for s in brief.get("sections", [])
    )

    prompt = f"""Tu es un expert en maillage interne SEO.

ARTICLE EN COURS DE RÉDACTION :
Mot-clé principal : {keyword}
H1 : {brief.get('h1', '')}

STRUCTURE DES SECTIONS (avec IDs) :
{sections_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LISTE COMPLÈTE DES URLs DISPONIBLES (copie exacte depuis le sitemap) :
{chr(10).join(candidate_urls)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RÈGLES ABSOLUES :
1. Tu dois choisir UNIQUEMENT des URLs figurant EXACTEMENT dans la liste ci-dessus
2. Copie l'URL mot pour mot, sans modification, sans invention
3. Sélectionne 3 à 5 URLs complémentaires (pas concurrentes) à l'article
4. L'ancre doit être naturelle, descriptive, avec des mots-clés SEO pertinents
5. Place chaque lien dans la section la plus cohérente (utilise l'ID de section)

Réponds UNIQUEMENT avec un JSON valide :
{{
  "internal_links": [
    {{
      "url": "URL EXACTE copiée depuis la liste ci-dessus",
      "anchor_text": "texte d'ancre naturel et descriptif (5-8 mots)",
      "target_section_id": "s1",
      "integration_hint": "Contexte précis où placer ce lien dans la section"
    }}
  ]
}}"""

    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()
    except Exception as e:
        print(f"[maillage] Erreur Gemini : {e}")
        return []

    json_str = extract_json_from_text(raw)
    if not json_str:
        print(f"[maillage] Pas de JSON dans la réponse Gemini")
        return []

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[maillage] JSON invalide : {e}")
        return []

    raw_links = data.get("internal_links", [])
    valid_links: list[dict] = []

    for link in raw_links:
        url = link.get("url", "").strip()
        if not url:
            continue

        # Validation 1 : URL exacte dans le sitemap
        if url in sitemap_set_exact:
            valid_links.append(link)
            continue

        # Validation 2 : même URL sans slash final
        url_stripped = url.rstrip("/")
        if url_stripped in sitemap_set_stripped:
            link["url"] = sitemap_set_stripped[url_stripped]  # version canonique
            valid_links.append(link)
            continue

        # Validation 3 : correspondance partielle (Gemini a peut-être tronqué)
        matched = next(
            (s for s in sitemap_urls if url_stripped in s or s.rstrip("/") in url_stripped),
            None,
        )
        if matched:
            link["url"] = matched
            valid_links.append(link)
            print(f"[maillage] URL approchée acceptée : {url} → {matched}")
            continue

        print(f"[maillage] URL hallucin\u00e9e ignorée : {url}")

    return valid_links
