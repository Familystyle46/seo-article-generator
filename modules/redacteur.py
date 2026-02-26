"""
Étape 4 — Rédacteur : rédige l'article complet en Markdown + HTML via Claude.
"""

import re
import json
from datetime import datetime

import anthropic

from .seo_calculator import budget_to_readable


CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

# ── Prompt building ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Tu es un rédacteur SEO expert francophone de niveau mondial. Tu rédiges des articles \
de blog qui classent en première page de Google ET qui captivent vraiment les lecteurs.

═══════════════════════════════════════════════
TES RÈGLES D'OR — ABSOLUES, NON NÉGOCIABLES
═══════════════════════════════════════════════

1. PYRAMIDE INVERSÉE
   Chaque section commence par l'information la plus utile.
   Le lecteur trouve sa réponse dans les 2 premières phrases.
   Ne jamais commencer par "Dans cette section, nous allons..."

2. TITRES CRÉATIFS
   Réécris CHAQUE titre H2/H3 avec une formulation originale et engageante.
   JAMAIS copier-coller un titre du brief.
   Utilise : chiffres, questions rhétoriques, formules de contraste, verbes d'action.

3. DISPERSION DES LIENS INTERNES
   Place chaque lien interne seul dans son paragraphe, de façon naturelle.
   JAMAIS deux liens dans le même paragraphe ou la même phrase.
   Le lien doit apporter de la valeur au lecteur, pas juste remplir une obligation.

4. BALISES MÉDIAS
   Insère des balises <figure> aux emplacements prévus dans le brief :
   <figure>
     <img src="" alt="[description SEO précise de l'image]" loading="lazy" width="800" height="500">
     <figcaption>[Légende descriptive et utile]</figcaption>
   </figure>
   Laisse src="" vide (l'utilisateur ajoutera les images après).

5. ANNÉE COURANTE
   Mentionne l'année actuelle naturellement dans le texte au moins une fois.

6. DENSITÉ MOT-CLÉ
   Cible ~1,5% de densité. Le mot-clé doit apparaître dans :
   - Le H1 (obligatoire)
   - Le premier paragraphe (obligatoire)
   - Au moins un H2
   - La meta description (déjà définie dans le brief)
   Utilise aussi les variantes et LSI pour éviter la sur-optimisation.

7. VOIX ET STYLE
   - Phrases courtes (15 mots max en moyenne)
   - Voix active, verbes forts
   - Exemples concrets et chiffres quand possible
   - Tutoie ou vouvoie de façon cohérente (choisis selon l'audience)
   - Évite absolument : "il est important de noter", "en conclusion", "n'oublions pas"

8. STRUCTURE HTML SÉMANTIQUE
   Utilise : <article>, <section>, <h1>, <h2>, <h3>, <p>, <ul>, <ol>, <li>,
   <strong>, <em>, <blockquote>, <table>, <figure>, <img>, <figcaption>
   FAQ : utilise le balisage Schema.org FAQPage (voir format ci-dessous)

9. FRONT MATTER MARKDOWN
   L'article Markdown DOIT commencer par un front matter YAML complet.

10. QUALITÉ E-E-A-T
    Montre l'expérience, l'expertise, l'autorité, la fiabilité.
    Cite des faits vérifiables, donne des conseils actionnables.

═══════════════════════════════════════════════
FORMAT DE SORTIE OBLIGATOIRE
═══════════════════════════════════════════════

Produis l'article dans les DEUX formats, séparés par des marqueurs exacts :

===MARKDOWN===
---
title: "Titre de l'article"
date: AAAA-MM-JJ
description: "Meta description"
tags: ["tag1", "tag2", "tag3"]
slug: "slug-de-larticle"
---

[article en Markdown complet]
===END MARKDOWN===

===HTML===
<article>
[article en HTML sémantique propre, sans DOCTYPE, sans <html>, sans CSS]
</article>
===END HTML===
"""


def _build_internal_links_block(internal_links: list[dict]) -> str:
    if not internal_links:
        return "Aucun lien interne identifié pour cet article."

    lines = ["LIENS INTERNES À PLACER (règle : un lien par paragraphe, jamais groupés) :"]
    for lk in internal_links:
        lines.append(
            f"  • Ancre : \"{lk.get('anchor_text', '')}\" → {lk.get('url', '')}\n"
            f"    Section cible : {lk.get('target_section_id', '?')}\n"
            f"    Comment intégrer : {lk.get('integration_hint', '')}"
        )
    return "\n".join(lines)


def _build_sections_block(brief: dict) -> str:
    lines: list[str] = []
    for s in brief.get("sections", []):
        lines.append(f"\n[H2] {s.get('h2', '')}")
        lines.append(f"  LSI à utiliser : {', '.join(s.get('lsi_to_use', []))}")
        lines.append(f"  Points à couvrir :")
        for pt in s.get("key_points", []):
            lines.append(f"    - {pt}")
        if s.get("media_suggestion"):
            lines.append(f"  📸 Média à insérer : {s['media_suggestion']}")
        for ss in s.get("subsections", []):
            lines.append(f"\n  [H3] {ss.get('h3', '')}")
            for pt in ss.get("key_points", []):
                lines.append(f"    - {pt}")
    return "\n".join(lines)


def _build_faq_block(brief: dict) -> str:
    faqs = brief.get("faq", [])
    if not faqs:
        return ""
    lines = ["\n[FAQ — utiliser le balisage Schema.org FAQPage]"]
    for item in faqs:
        lines.append(f"  Q: {item.get('question', '')}")
        lines.append(f"     Réponse doit couvrir : {item.get('answer_hint', '')}")
    return "\n".join(lines)


def build_user_prompt(
    keyword_data: dict,
    brief: dict,
    word_budget: dict,
    internal_links: list[dict],
    current_year: int,
) -> str:
    keyword        = keyword_data.get("keyword", "")
    total_words    = word_budget.get("total_calculated", 2000)
    keyword_count  = max(4, round(total_words * 0.015))
    slug           = re.sub(r"[^a-z0-9-]", "", keyword.lower().replace(" ", "-"))

    today = datetime.now().strftime("%Y-%m-%d")

    sections_block     = _build_sections_block(brief)
    faq_block          = _build_faq_block(brief)
    links_block        = _build_internal_links_block(internal_links)
    budget_block       = budget_to_readable(word_budget, brief)

    return f"""Rédige un article SEO complet en français sur : "{keyword}"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DONNÉES DU MOT-CLÉ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Intention de recherche : {keyword_data.get('search_intent', 'informational')}
Audience cible         : {keyword_data.get('target_audience', '')}
Angle éditorial        : {keyword_data.get('angle_editorial', '')}
Variantes sémantiques  : {', '.join(keyword_data.get('semantic_variants', []))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BRIEF ÉDITORIAL (structure à respecter)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
H1 : {brief.get('h1', keyword)}
Slug : {slug}
Date : {today}
Meta description : {brief.get('meta_description', '')}

Hook d'introduction : {brief.get('intro_hook', '')}
Points à couvrir dans l'intro : {', '.join(brief.get('intro_key_points', []))}

SECTIONS :{sections_block}
{faq_block}

Conclusion / CTA : {brief.get('conclusion_cta', '')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUDGET DE MOTS PAR SECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{budget_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{links_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAPPELS CRITIQUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Tous les titres H2/H3 doivent être RÉÉCRITS de façon créative (pas copiés du brief)
• Pyramide inversée dans chaque section
• Mot-clé "{keyword}" : environ {keyword_count} occurrences au total
• Année {current_year} mentionnée naturellement au moins une fois
• Balises <figure> avec src="" vide aux emplacements indiqués
• Liens internes : un par paragraphe maximum, jamais groupés
• Format de sortie obligatoire : ===MARKDOWN=== ... ===END MARKDOWN=== puis ===HTML=== ... ===END HTML==="""


# ── Main writer function ───────────────────────────────────────────────────────

def write_article(
    keyword_data: dict,
    brief: dict,
    word_budget: dict,
    internal_links: list[dict],
    claude_client: anthropic.Anthropic,
    model: str = CLAUDE_MODEL,
) -> dict:
    """
    Write the complete SEO article using Claude.
    Returns dict with: markdown, html, raw, keyword, slug.
    """
    current_year = datetime.now().year

    user_prompt = build_user_prompt(
        keyword_data=keyword_data,
        brief=brief,
        word_budget=word_budget,
        internal_links=internal_links,
        current_year=current_year,
    )

    response = claude_client.messages.create(
        model=model,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = response.content[0].text

    # ── Extract Markdown ───────────────────────────────────────────────────────
    md_match = re.search(r"===MARKDOWN===\s*(.*?)\s*===END MARKDOWN===", raw, re.DOTALL)
    markdown_content = md_match.group(1).strip() if md_match else raw.strip()

    # ── Extract HTML ───────────────────────────────────────────────────────────
    html_match = re.search(r"===HTML===\s*(.*?)\s*===END HTML===", raw, re.DOTALL)
    if html_match:
        html_content = html_match.group(1).strip()
    else:
        # Fallback: convert markdown to HTML
        try:
            import markdown as md_lib
            html_content = md_lib.markdown(markdown_content, extensions=["extra", "toc"])
        except ImportError:
            html_content = f"<p>{markdown_content}</p>"

    keyword  = keyword_data.get("keyword", "")
    slug     = re.sub(r"[^a-z0-9-]", "", keyword.lower().replace(" ", "-"))

    return {
        "markdown": markdown_content,
        "html": html_content,
        "raw": raw,
        "keyword": keyword,
        "slug": slug,
        "brief": brief,
        "word_budget": word_budget,
        "internal_links": internal_links,
        "generated_at": datetime.now().isoformat(),
    }
