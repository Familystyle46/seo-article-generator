"""Étape 3 — SEO Calculator: calcule le budget de mots par section via Gemini."""

import json

from .utils import extract_json_from_text


def calculate_word_budget(brief: dict, target_words: int, gemini_model) -> dict:
    """
    Ask Gemini to distribute the total word count across all sections,
    ensuring: no two sections have identical counts, all minimums respected,
    and the total matches target_words exactly (±30 words).
    """
    # Summarize structure for the prompt
    structure: list[dict] = []
    for s in brief.get("sections", []):
        entry = {
            "id": s.get("id"),
            "h2": s.get("h2", ""),
            "key_points_count": len(s.get("key_points", [])),
            "has_media": bool(s.get("media_suggestion")),
            "subsections": [
                {
                    "id": ss.get("id"),
                    "h3": ss.get("h3", ""),
                    "key_points_count": len(ss.get("key_points", [])),
                }
                for ss in s.get("subsections", [])
            ],
        }
        structure.append(entry)

    faq_count = len(brief.get("faq", []))

    prompt = f"""Tu es un expert en architecture d'articles SEO.

STRUCTURE DE L'ARTICLE :
{json.dumps(structure, ensure_ascii=False, indent=2)}

NOMBRE DE QUESTIONS FAQ : {faq_count}
LONGUEUR TOTALE CIBLE : {target_words} mots

RÈGLES DE CALCUL :
1. Introduction : 8-10% du total
2. Conclusion : 5-7% du total
3. Section FAQ : environ {faq_count * 60} à {faq_count * 90} mots (60-90 mots par question/réponse)
4. Le reste réparti entre les sections H2 et H3 selon leur importance (nombre de points, présence de média)
5. Chaque H2 reçoit un "chapeau" de 80-150 mots avant ses H3
6. AUCUNE deux sections NE DOIVENT avoir exactement le même nombre de mots (varier d'au moins 15 mots)
7. Minimum : 100 mots par H3, 80 mots par chapeau H2
8. Le total de TOUS les éléments doit faire {target_words} mots (±30 mots acceptable)

Réponds UNIQUEMENT avec un objet JSON valide, aucun texte avant ni après.
Utilise EXACTEMENT les mêmes ids que dans la structure fournie :
{{
  "intro": 220,
  "sections": {{
    "s1": {{
      "h2_intro": 100,
      "subsections": {{
        "s1-1": 250,
        "s1-2": 230
      }}
    }},
    "s2": {{
      "h2_intro": 120,
      "subsections": {{}}
    }}
  }},
  "faq": 320,
  "conclusion": 160,
  "total_calculated": {target_words}
}}"""

    response = gemini_model.generate_content(prompt)
    raw = response.text.strip()

    json_str = extract_json_from_text(raw)
    if json_str:
        try:
            budget = json.loads(json_str)
            # Patch total if missing
            if "total_calculated" not in budget:
                budget["total_calculated"] = target_words
            return budget
        except json.JSONDecodeError:
            pass

    # Fallback: equal distribution
    return _fallback_budget(brief, target_words)


def _fallback_budget(brief: dict, target_words: int) -> dict:
    """Simple equal-distribution fallback if Gemini fails."""
    sections = brief.get("sections", [])
    faq_items = len(brief.get("faq", []))

    intro_words      = int(target_words * 0.09)
    conclusion_words = int(target_words * 0.06)
    faq_words        = max(faq_items * 70, int(target_words * 0.10))
    body_words       = target_words - intro_words - conclusion_words - faq_words

    budget: dict = {
        "intro": intro_words,
        "sections": {},
        "faq": faq_words,
        "conclusion": conclusion_words,
        "total_calculated": target_words,
    }

    if not sections:
        return budget

    per_section = body_words // len(sections)
    for i, s in enumerate(sections):
        sid = s.get("id", f"s{i+1}")
        subs = s.get("subsections", [])
        h2_intro = 100
        remaining = per_section - h2_intro

        sub_budget: dict = {}
        if subs:
            per_sub = remaining // len(subs)
            for j, ss in enumerate(subs):
                ssid = ss.get("id", f"{sid}-{j+1}")
                sub_budget[ssid] = per_sub + (j * 5)  # slight variation
        else:
            h2_intro = per_section  # all words to the H2 block

        budget["sections"][sid] = {
            "h2_intro": h2_intro,
            "subsections": sub_budget,
        }

    return budget


def budget_to_readable(budget: dict, brief: dict) -> str:
    """Format the word budget as a readable string for the Claude prompt."""
    lines: list[str] = [f"Introduction : {budget.get('intro', 200)} mots"]

    section_budget = budget.get("sections", {})
    for s in brief.get("sections", []):
        sid = s.get("id", "")
        s_data = section_budget.get(sid, {})
        h2_intro = s_data.get("h2_intro", 100) if isinstance(s_data, dict) else 100
        lines.append(f"\n[H2] {s.get('h2', sid)} — chapeau : {h2_intro} mots")

        subs_budget = s_data.get("subsections", {}) if isinstance(s_data, dict) else {}
        for ss in s.get("subsections", []):
            ssid = ss.get("id", "")
            ss_words = subs_budget.get(ssid, 200)
            lines.append(f"  [H3] {ss.get('h3', ssid)} : {ss_words} mots")

    lines.append(f"\nFAQ : {budget.get('faq', 300)} mots")
    lines.append(f"Conclusion : {budget.get('conclusion', 150)} mots")
    lines.append(f"\nTOTAL : ~{budget.get('total_calculated', 0)} mots")
    return "\n".join(lines)
