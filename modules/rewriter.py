"""
Mode Réécriture — met à jour et améliore un article existant via Gemini + Claude.
Analyse d'abord les faiblesses, puis réécrit en conservant ce qui fonctionne.
"""

import re
import json
from datetime import datetime

import anthropic

from .utils import extract_json_from_text, scrape_page_text


REWRITE_SYSTEM_PROMPT = """\
Tu es un rédacteur SEO expert francophone. Tu dois améliorer et remettre à jour un article existant.

TES OBJECTIFS :
1. Conserver la structure et les sections qui fonctionnent bien
2. Mettre à jour toutes les informations potentiellement obsolètes
3. Approfondir les sections trop légères et ajouter les sujets manquants
4. Optimiser le mot-clé et ses variantes sémantiques (cible ~1,5% de densité)
5. Améliorer la lisibilité : phrases courtes, voix active, exemples concrets
6. Appliquer toutes les recommandations d'amélioration fournies

RÈGLES D'OR (identiques à la rédaction originale) :
- Pyramide inversée : réponse en premier, développement ensuite
- Titres H2/H3 créatifs et engageants (jamais génériques)
- 1 lien interne par paragraphe maximum, jamais groupés
- Balises <figure><img src="" alt="description SEO"></figure> aux emplacements médias
- Année actuelle mentionnée au moins une fois naturellement

FORMAT DE SORTIE OBLIGATOIRE :

===MARKDOWN===
---
title: "Titre de l'article"
date: AAAA-MM-JJ
description: "Meta description"
tags: ["tag1", "tag2"]
slug: "slug-article"
---

[article complet en Markdown]
===END MARKDOWN===

===HTML===
<article>
[article en HTML sémantique, sans DOCTYPE, sans <html>, sans CSS]
</article>
===END HTML===
"""


def analyze_article_for_rewrite(
    article_text: str,
    keyword: str,
    gemini_model,
) -> dict:
    """
    Analyse un article existant et identifie précisément les axes d'amélioration.
    Retourne un dict avec le plan de réécriture.
    """
    word_count = len(article_text.split())
    kw_count   = len(re.findall(re.escape(keyword.lower()), article_text.lower()))
    kw_density = round((kw_count / max(word_count, 1)) * 100, 2)

    prompt = f"""Tu es un auditeur SEO. Analyse cet article existant et produis un plan de réécriture.

MOT-CLÉ CIBLE : "{keyword}"
LONGUEUR ACTUELLE : {word_count} mots
DENSITÉ MOT-CLÉ : {kw_density}%

ARTICLE (3 000 premiers caractères) :
{article_text[:3000]}

Réponds UNIQUEMENT avec un JSON valide :
{{
  "score_actuel": 52,
  "score_potentiel": 85,
  "sections_a_conserver": [
    "Nom/titre de section qui fonctionne bien et pourquoi"
  ],
  "sections_a_ameliorer": [
    {{
      "section": "Nom de la section",
      "probleme": "Ce qui ne va pas",
      "solution": "Action concrète à appliquer"
    }}
  ],
  "sujets_manquants": [
    "Sujet important non traité 1",
    "Sujet 2"
  ],
  "infos_a_actualiser": [
    "Information potentiellement obsolète à vérifier/mettre à jour"
  ],
  "keywords_manquants": [
    "variante sémantique non utilisée",
    "LSI keyword absent"
  ],
  "priorite_rewrite": "Reformuler l'intro pour pyramide inversée, ajouter section sur X, densifier le mot-clé"
}}"""

    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()
        json_str = extract_json_from_text(raw)
        if json_str:
            data = json.loads(json_str)
            data["word_count"] = word_count
            data["kw_density"] = kw_density
            return data
    except Exception as e:
        print(f"[rewriter] Analyse erreur : {e}")

    return {
        "score_actuel": 0,
        "score_potentiel": 75,
        "sections_a_conserver": [],
        "sections_a_ameliorer": [],
        "sujets_manquants": [],
        "infos_a_actualiser": [],
        "keywords_manquants": [],
        "priorite_rewrite": "Analyse indisponible",
        "word_count": word_count,
        "kw_density": kw_density,
    }


def rewrite_article(
    article_text: str,
    keyword: str,
    analysis: dict,
    target_words: int,
    internal_links: list[dict],
    claude_client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-5-20250929",
) -> dict:
    """
    Réécrit un article existant via Claude en s'appuyant sur l'analyse Gemini.
    Retourne le même format que write_article().
    """
    current_year = datetime.now().year
    today        = datetime.now().strftime("%Y-%m-%d")

    # Build improvement instructions block
    keep_block = "\n".join(f"  ✓ {s}" for s in analysis.get("sections_a_conserver", []))
    improve_block = "\n".join(
        f"  • {i.get('section', '')}: {i.get('solution', '')}"
        for i in analysis.get("sections_a_ameliorer", [])
    )
    missing_block = "\n".join(f"  + {m}" for m in analysis.get("sujets_manquants", []))
    update_block  = "\n".join(f"  ↻ {u}" for u in analysis.get("infos_a_actualiser", []))
    kw_block      = ", ".join(analysis.get("keywords_manquants", []))

    links_block = ""
    if internal_links:
        lines = ["LIENS INTERNES À PLACER (1 par paragraphe max, jamais groupés) :"]
        for lk in internal_links:
            lines.append(
                f"  • Ancre : \"{lk.get('anchor_text', '')}\" → {lk.get('url', '')}"
            )
        links_block = "\n".join(lines)

    user_prompt = f"""Réécris et améliore cet article SEO pour le mot-clé : "{keyword}"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLAN DE RÉÉCRITURE (basé sur l'audit)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Score actuel : {analysis.get('score_actuel', '?')}/100
Score cible  : {analysis.get('score_potentiel', 80)}/100

CONSERVER (ces sections fonctionnent) :
{keep_block or "  (pas d'analyse disponible)"}

AMÉLIORER :
{improve_block or "  (pas d'analyse disponible)"}

SUJETS À AJOUTER :
{missing_block or "  (aucun identifié)"}

INFORMATIONS À ACTUALISER :
{update_block or "  (aucune)"}

VARIANTES SÉMANTIQUES À INTÉGRER : {kw_block or "—"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARTICLE ORIGINAL À RÉÉCRIRE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{article_text[:6000]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONSIGNES FINALES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Longueur cible : {target_words} mots minimum
• Date : {today} — Année : {current_year}
• Mot-clé "{keyword}" dans H1, premier paragraphe, au moins un H2
• Pyramide inversée : réponse immédiate, puis développement
• Tous les H2/H3 doivent être réécrits de façon créative
• Front matter YAML complet dans la version Markdown
{links_block}

FORMAT : ===MARKDOWN=== ... ===END MARKDOWN=== puis ===HTML=== ... ===END HTML==="""

    response = claude_client.messages.create(
        model=model,
        max_tokens=8000,
        system=REWRITE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = response.content[0].text

    md_match = re.search(r"===MARKDOWN===\s*(.*?)\s*===END MARKDOWN===", raw, re.DOTALL)
    markdown_content = md_match.group(1).strip() if md_match else raw.strip()

    html_match = re.search(r"===HTML===\s*(.*?)\s*===END HTML===", raw, re.DOTALL)
    if html_match:
        html_content = html_match.group(1).strip()
    else:
        try:
            import markdown as md_lib
            html_content = md_lib.markdown(markdown_content, extensions=["extra", "toc"])
        except ImportError:
            html_content = f"<article>{markdown_content}</article>"

    slug = re.sub(r"[^a-z0-9-]", "", keyword.lower().replace(" ", "-"))

    return {
        "markdown": markdown_content,
        "html":     html_content,
        "raw":      raw,
        "keyword":  keyword,
        "slug":     slug,
        "analysis": analysis,
        "generated_at": datetime.now().isoformat(),
        "mode": "rewrite",
    }


def load_article_from_url(url: str) -> str:
    """Scrape le texte d'un article existant depuis son URL."""
    text = scrape_page_text(url, max_chars=8000)
    if not text:
        raise ValueError(f"Impossible de récupérer le contenu de : {url}")
    return text
