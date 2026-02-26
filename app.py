"""
SEO Article Generator — Interface Streamlit
Pipeline : CSV → Pépite Finder → SERP + Insights + Brief → Budget mots → Rédaction Claude
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import anthropic
from dotenv import load_dotenv

# ── Modules ────────────────────────────────────────────────────────────────────
from modules.utils import fetch_sitemap, scrape_page_text, slug_from_keyword, make_gemini
from modules.pepite_finder import parse_keyword_csv, find_best_keyword
from modules.serp_analyzer import scrape_google_serp, build_competitor_summary
from modules.insight_miner import generate_insights
from modules.semantic_architect import build_brief, find_internal_links
from modules.seo_calculator import calculate_word_budget
from modules.redacteur import write_article

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv(override=True)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def _get_key(name: str) -> str:
    """Load API key: Streamlit secrets (cloud) → .env / environ (local)."""
    try:
        val = st.secrets.get(name, "")
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(name, "")


def _is_cloud() -> bool:
    """True when running on Streamlit Community Cloud (secrets are configured)."""
    try:
        return bool(st.secrets)
    except Exception:
        return False

st.set_page_config(
    page_title="SEO Article Generator",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main header */
.main-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.main-subtitle {
    color: #666;
    margin-bottom: 2rem;
    font-size: 0.95rem;
}
/* Step badges */
.step-badge {
    display: inline-block;
    background: #0066cc;
    color: white;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 8px;
}
/* Result panels */
.result-box {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
}
/* Keyword badge */
.kw-badge {
    display: inline-block;
    background: #e8f5e9;
    color: #2e7d32;
    border: 1px solid #a5d6a7;
    border-radius: 6px;
    padding: 4px 12px;
    font-weight: 600;
    font-size: 1.1rem;
    margin: 0.5rem 0;
}
/* Sidebar section headers */
[data-testid="stSidebar"] h3 {
    color: #333;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 1.5rem;
}
/* Generate button */
div[data-testid="stButton"] > button[kind="primary"] {
    width: 100%;
    padding: 0.75rem;
    font-size: 1.1rem;
    font-weight: 700;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _init_state() -> None:
    defaults = {
        "result": None,
        "keyword_data": None,
        "brief": None,
        "word_budget": None,
        "internal_links": [],
        "sitemap_urls": [],
        "competitor_summary": "",
        "insights": {},
        "history": [],
        "running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — .env persistence
# ══════════════════════════════════════════════════════════════════════════════

ENV_PATH = Path(__file__).parent / ".env"

def save_keys_to_env(anthropic: str, google: str) -> None:
    """Write API keys to .env file, preserving any other existing variables."""
    lines: list[str] = []
    existing: dict[str, str] = {}

    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k, _, v = stripped.partition("=")
                existing[k.strip()] = v.strip()
            else:
                lines.append(line)

    existing["ANTHROPIC_API_KEY"] = anthropic
    existing["GOOGLE_API_KEY"]    = google

    content = "\n".join(lines).rstrip() + "\n" if lines else ""
    content += "\n".join(f"{k}={v}" for k, v in existing.items()) + "\n"
    ENV_PATH.write_text(content.lstrip(), encoding="utf-8")
    # Reload into environment immediately
    os.environ["ANTHROPIC_API_KEY"] = anthropic
    os.environ["GOOGLE_API_KEY"]    = google


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — API Keys & Settings
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ✍️ SEO Generator")
    st.markdown("---")

    # ── Statut des clés (cloud: st.secrets / local: .env) ─────────────────────
    anthropic_key = _get_key("ANTHROPIC_API_KEY")
    google_key    = _get_key("GOOGLE_API_KEY")

    st.markdown("### 🔑 Clés API")
    st.markdown(
        f"{'✅' if anthropic_key else '❌'} Claude (Anthropic)  \n"
        f"{'✅' if google_key    else '❌'} Gemini (Google)",
    )

    # Bouton de test de connexion
    if google_key and st.button("🔍 Tester Gemini", use_container_width=True):
        with st.spinner("Connexion à l'API Gemini…"):
            try:
                from modules.utils import list_available_gemini_models
                from google import genai
                _client = genai.Client(api_key=google_key)
                _models = list_available_gemini_models(_client)
                if _models:
                    st.success(f"✅ Connecté — {len(_models)} modèle(s) disponible(s)")
                    for m in _models:
                        st.caption(f"  • {m}")
                else:
                    st.warning("⚠️ Connecté mais aucun modèle Gemini disponible")
            except Exception as e:
                st.error(f"❌ Erreur : {e}")

    # Expander pour modifier les clés
    if _is_cloud():
        # Sur Streamlit Cloud : les clés sont dans le dashboard, pas modifiables ici
        if not anthropic_key or not google_key:
            with st.expander("⚙️ Configurer les clés…"):
                st.info(
                    "**Streamlit Cloud détecté.**\n\n"
                    "Ajoute tes clés dans :\n"
                    "**App Settings → Secrets**\n\n"
                    "```toml\n"
                    "ANTHROPIC_API_KEY = \"sk-ant-...\"\n"
                    "GOOGLE_API_KEY = \"AIza...\"\n"
                    "```"
                )
    else:
        # En local : formulaire de saisie avec sauvegarde dans .env
        with st.expander("Modifier les clés…"):
            anthropic_key_input = st.text_input(
                "Anthropic (Claude)",
                value=anthropic_key,
                type="password",
                placeholder="sk-ant-...",
            )
            google_key_input = st.text_input(
                "Google (Gemini)",
                value=google_key,
                type="password",
                placeholder="AIza...",
            )
            if st.button("💾 Sauvegarder dans `.env`", use_container_width=True):
                save_keys_to_env(anthropic_key_input.strip(), google_key_input.strip())
                anthropic_key = anthropic_key_input.strip()
                google_key    = google_key_input.strip()
                st.success("Clés sauvegardées ✅ — rechargement…")
                st.rerun()

    st.markdown("### ⚙️ Paramètres article")
    target_words = st.slider(
        "Longueur cible (mots)",
        min_value=800,
        max_value=4000,
        value=2000,
        step=100,
        help="Longueur totale de l'article généré",
    )
    num_serp_results = st.slider(
        "Concurrents à analyser",
        min_value=0,
        max_value=7,
        value=3,
        help="0 = désactiver l'analyse SERP (plus rapide)",
    )
    gemini_model_name = st.selectbox(
        "Modèle Gemini",
        [
            "gemini-2.5-flash",          # ⭐ Recommandé — meilleur rapport qualité/vitesse
            "gemini-2.5-pro",            # 🏆 Qualité maximale (plus lent)
            "gemini-2.0-flash",          # ⚡ Rapide et fiable
            "gemini-2.0-flash-lite",     # ⚡ Ultra rapide
            "gemini-1.5-flash",          # Version précédente
            "gemini-1.5-pro",            # Version précédente, qualité
        ],
        index=0,
        help="gemini-2.5-flash est recommandé pour l'analyse SEO : rapide, précis et économique.",
    )

    claude_model_name = st.selectbox(
        "Modèle Claude",
        [
            "claude-sonnet-4-5-20250929",   # ⭐ Recommandé — meilleur pour le contenu
            "claude-opus-4-1-20250805",     # 🏆 Qualité maximale (plus lent)
            "claude-opus-4-20250514",       # 🏆 Opus 4 classique
            "claude-3-opus-20240229",       # Version précédente, très bon
            "claude-3-haiku-20240307",      # ⚡ Rapide et économique
        ],
        index=0,
        help="claude-sonnet-4-5 est recommandé : excellent équilibre qualité/vitesse pour la rédaction SEO.",
    )

    if anthropic_key and st.button("🔍 Tester Claude", use_container_width=True):
        with st.spinner("Test de la connexion Claude…"):
            try:
                _claude = anthropic.Anthropic(api_key=anthropic_key)
                # Test rapide avec le modèle sélectionné
                _resp = _claude.messages.create(
                    model=claude_model_name,
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Réponds juste: OK"}],
                )
                st.success(f"✅ Claude connecté — modèle : `{claude_model_name}`")
            except Exception as e:
                err = str(e)
                if "404" in err or "not_found" in err.lower():
                    st.error(f"❌ Modèle introuvable : `{claude_model_name}`\n\nEssaie un autre modèle dans le sélecteur.")
                elif "401" in err or "authentication" in err.lower():
                    st.error("❌ Clé Anthropic invalide. Vérifie ta clé API.")
                elif "403" in err:
                    st.error("❌ Accès refusé. Vérifie que ton compte a accès à ce modèle.")
                else:
                    st.error(f"❌ Erreur : {err}")

    st.markdown("---")
    st.markdown(
        "<small>Pipeline : CSV → Pépite → SERP → Brief → Budget → Rédaction</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_generate, tab_history, tab_help = st.tabs(["🚀 Générer", "📂 Historique", "❓ Guide"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Generate
# ─────────────────────────────────────────────────────────────────────────────

with tab_generate:
    st.markdown('<p class="main-title">Générateur d\'Articles SEO</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">CSV de mots-clés + sitemap → Article complet optimisé SEO en Markdown & HTML</p>', unsafe_allow_html=True)

    # ── Input section ──────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### 📄 Fichier CSV de mots-clés")
        csv_file = st.file_uploader(
            "Exporte de ton outil SEO (Ahrefs, SEMrush, Ubersuggest…)",
            type=["csv"],
            label_visibility="collapsed",
        )
        if csv_file:
            try:
                df_preview = parse_keyword_csv(csv_file.read())
                csv_file.seek(0)
                st.success(f"{len(df_preview)} mots-clés chargés")
                st.dataframe(df_preview.head(5), use_container_width=True, height=180)
            except Exception as e:
                st.error(f"Erreur lecture CSV : {e}")

    with col_right:
        st.markdown("#### 🗺️ Configuration du site")
        sitemap_url = st.text_input(
            "URL du Sitemap XML",
            placeholder="https://monsite.fr/sitemap.xml",
            help="Pour éviter les doublons et pour le maillage interne",
        )
        category_url = st.text_input(
            "URL de la catégorie cible (optionnel)",
            placeholder="https://monsite.fr/decoration/",
            help="Permet à Gemini de comprendre le contexte de ta boutique",
        )

    st.markdown("---")

    # ── Generate button ────────────────────────────────────────────────────────
    can_generate = bool(csv_file and sitemap_url and anthropic_key and google_key)
    if not can_generate:
        missing = []
        if not csv_file:        missing.append("fichier CSV")
        if not sitemap_url:     missing.append("URL sitemap")
        if not anthropic_key:   missing.append("clé Anthropic")
        if not google_key:      missing.append("clé Google")
        st.info(f"⚠️ Manquant : {', '.join(missing)}")

    generate_btn = st.button(
        "🚀 Générer l'article SEO",
        type="primary",
        disabled=not can_generate or st.session_state.running,
        use_container_width=True,
    )

    # ── Pipeline execution ─────────────────────────────────────────────────────
    if generate_btn:
        st.session_state.running = True
        st.session_state.result = None

        # Init clients
        try:
            gemini = make_gemini(google_key, gemini_model_name)
            claude_client = anthropic.Anthropic(api_key=anthropic_key)
        except Exception as e:
            st.error(f"Erreur initialisation API : {e}")
            st.session_state.running = False
            st.stop()

        pipeline_error = None
        t0 = time.time()

        # ── STEP 1 — Pépite Finder ─────────────────────────────────────────────
        with st.status("**Étape 1 — Pépite Finder** : sélection du meilleur mot-clé…", expanded=True) as s1:
            try:
                st.write("📥 Lecture du fichier CSV…")
                csv_file.seek(0)
                df = parse_keyword_csv(csv_file.read())

                st.write(f"🗺️ Récupération du sitemap ({sitemap_url})…")
                sitemap_urls = fetch_sitemap(sitemap_url)
                st.session_state.sitemap_urls = sitemap_urls
                st.write(f"   → {len(sitemap_urls)} URLs trouvées")

                category_content = ""
                if category_url.strip():
                    st.write(f"📖 Lecture de la page catégorie…")
                    category_content = scrape_page_text(category_url.strip(), max_chars=3000)

                st.write("🤖 Gemini analyse les mots-clés et choisit le meilleur…")
                keyword_data = find_best_keyword(df, sitemap_urls, category_content, gemini)
                st.session_state.keyword_data = keyword_data

                s1.update(label=f"✅ Étape 1 — Mot-clé choisi : **{keyword_data['keyword']}**", state="complete")

                # Display result
                kw = keyword_data["keyword"]
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Mot-clé", kw)
                col_b.metric("Volume", keyword_data.get("volume", "—"))
                col_c.metric("Intention", keyword_data.get("search_intent", "—"))
                with st.expander("Voir le détail du choix"):
                    st.markdown(f"**Raison :** {keyword_data.get('reason', '')}")
                    st.markdown(f"**Audience :** {keyword_data.get('target_audience', '')}")
                    st.markdown(f"**Angle :** {keyword_data.get('angle_editorial', '')}")
                    st.markdown(f"**Variantes :** {', '.join(keyword_data.get('semantic_variants', []))}")

            except Exception as e:
                s1.update(label=f"❌ Étape 1 échouée : {e}", state="error")
                pipeline_error = str(e)
                st.session_state.running = False

        if pipeline_error:
            st.error(f"Pipeline interrompu à l'étape 1 : {pipeline_error}")
            st.stop()

        # ── STEP 2 — Analysis + Brief ──────────────────────────────────────────
        with st.status("**Étape 2 — Analyse & Brief** : SERP + insights + structure…", expanded=True) as s2:
            try:
                keyword = st.session_state.keyword_data["keyword"]
                search_intent = st.session_state.keyword_data.get("search_intent", "informational")

                # 2a — SERP
                if num_serp_results > 0:
                    st.write(f"🔍 Scraping Google (top {num_serp_results} résultats)…")
                    serp_results = scrape_google_serp(keyword, num_results=num_serp_results)
                    competitor_summary = build_competitor_summary(serp_results)
                    n_found = len(serp_results)
                    st.write(f"   → {n_found} concurrents analysés")
                    if n_found == 0:
                        st.warning("Scraping Google bloqué ou sans résultat — le brief sera construit sans analyse concurrents.")
                else:
                    competitor_summary = ""
                    st.write("⏭️ Analyse SERP désactivée")

                st.session_state.competitor_summary = competitor_summary

                # 2b — Insights
                st.write("💡 Gemini génère les questions et insights (PAA, LSI, angles)…")
                insights = generate_insights(keyword, search_intent, gemini)
                st.session_state.insights = insights

                # 2c — Brief
                st.write("📋 Gemini construit le brief éditorial (H1/H2/H3 + FAQ)…")
                brief = build_brief(
                    keyword_data=st.session_state.keyword_data,
                    competitor_summary=competitor_summary,
                    insights=insights,
                    target_words=target_words,
                    gemini_model=gemini,
                )
                st.session_state.brief = brief

                # 2d — Internal links
                st.write("🔗 Identification du maillage interne…")
                internal_links = find_internal_links(
                    brief=brief,
                    sitemap_urls=st.session_state.sitemap_urls,
                    keyword=keyword,
                    gemini_model=gemini,
                )
                st.session_state.internal_links = internal_links
                st.write(f"   → {len(internal_links)} liens internes identifiés")

                s2.update(label=f"✅ Étape 2 — Brief complet ({len(brief.get('sections', []))} sections + FAQ)", state="complete")

                with st.expander("Voir le brief"):
                    st.markdown(f"**H1 :** {brief.get('h1', '')}")
                    st.markdown(f"**Meta :** {brief.get('meta_description', '')}")
                    for s in brief.get("sections", []):
                        st.markdown(f"**• {s.get('h2', '')}**")
                        for ss in s.get("subsections", []):
                            st.markdown(f"   &nbsp;&nbsp;&nbsp; ↳ {ss.get('h3', '')}")
                    if internal_links:
                        st.markdown("**Liens internes :**")
                        for lk in internal_links:
                            st.markdown(f"   • [{lk.get('anchor_text', '')}]({lk.get('url', '')})")

            except Exception as e:
                s2.update(label=f"❌ Étape 2 échouée : {e}", state="error")
                pipeline_error = str(e)
                st.session_state.running = False

        if pipeline_error:
            st.error(f"Pipeline interrompu à l'étape 2 : {pipeline_error}")
            st.stop()

        # ── STEP 3 — Word Budget ───────────────────────────────────────────────
        with st.status("**Étape 3 — Calculateur SEO** : budget de mots par section…", expanded=True) as s3:
            try:
                st.write("📊 Gemini calcule le budget de mots par section…")
                word_budget = calculate_word_budget(
                    brief=st.session_state.brief,
                    target_words=target_words,
                    gemini_model=gemini,
                )
                st.session_state.word_budget = word_budget
                s3.update(label=f"✅ Étape 3 — Budget calculé ({word_budget.get('total_calculated', target_words)} mots)", state="complete")

                with st.expander("Voir le budget de mots"):
                    st.json(word_budget)

            except Exception as e:
                s3.update(label=f"❌ Étape 3 échouée : {e}", state="error")
                pipeline_error = str(e)
                st.session_state.running = False

        if pipeline_error:
            st.error(f"Pipeline interrompu à l'étape 3 : {pipeline_error}")
            st.stop()

        # ── STEP 4 — Article Writing ───────────────────────────────────────────
        with st.status("**Étape 4 — Rédaction Claude** : génération de l'article…", expanded=True) as s4:
            try:
                st.write(f"✍️ Claude (`{claude_model_name}`) rédige l'article ({target_words} mots)…")
                st.write("   ⏳ Cette étape prend 30 à 90 secondes, patience…")

                result = write_article(
                    keyword_data=st.session_state.keyword_data,
                    brief=st.session_state.brief,
                    word_budget=st.session_state.word_budget,
                    internal_links=st.session_state.internal_links,
                    claude_client=claude_client,
                    model=claude_model_name,
                )
                st.session_state.result = result

                # Save to disk
                slug = result.get("slug", "article")
                ts   = datetime.now().strftime("%Y%m%d_%H%M")
                md_path   = OUTPUT_DIR / f"{ts}_{slug}.md"
                html_path = OUTPUT_DIR / f"{ts}_{slug}.html"
                md_path.write_text(result["markdown"], encoding="utf-8")
                html_path.write_text(result["html"], encoding="utf-8")

                # Add to history
                st.session_state.history.append({
                    "keyword": result["keyword"],
                    "slug": slug,
                    "generated_at": result["generated_at"],
                    "md_path": str(md_path),
                    "html_path": str(html_path),
                    "word_count_target": target_words,
                })

                elapsed = round(time.time() - t0, 1)
                s4.update(label=f"✅ Étape 4 — Article généré en {elapsed}s", state="complete")

            except Exception as e:
                s4.update(label=f"❌ Étape 4 échouée : {e}", state="error")
                pipeline_error = str(e)
                st.session_state.running = False

        if pipeline_error:
            st.error(f"Pipeline interrompu à l'étape 4 : {pipeline_error}")
        else:
            st.session_state.running = False

    # ── Results display ────────────────────────────────────────────────────────
    if st.session_state.result:
        result = st.session_state.result
        kw = result.get("keyword", "")

        st.markdown("---")
        st.markdown(f"## 📰 Article généré : <span class='kw-badge'>{kw}</span>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mot-clé", kw)
        col2.metric("Cible mots", st.session_state.word_budget.get("total_calculated", "—"))
        col3.metric("Sections", len(st.session_state.brief.get("sections", [])))
        col4.metric("Liens internes", len(st.session_state.internal_links))

        # Output tabs
        out_md, out_html, out_preview, out_brief = st.tabs(["📝 Markdown", "🌐 HTML", "👁️ Aperçu", "📋 Brief JSON"])

        with out_md:
            st.code(result["markdown"], language="markdown", line_numbers=True)
            st.download_button(
                "⬇️ Télécharger .md",
                data=result["markdown"],
                file_name=f"{result['slug']}.md",
                mime="text/markdown",
            )

        with out_html:
            st.code(result["html"], language="html", line_numbers=True)
            st.download_button(
                "⬇️ Télécharger .html",
                data=result["html"],
                file_name=f"{result['slug']}.html",
                mime="text/html",
            )

        with out_preview:
            st.markdown("*Aperçu rendu (Streamlit ne supporte pas tout le HTML natif — ouvrir le .html pour un rendu complet)*")
            st.markdown(result["markdown"], unsafe_allow_html=False)

        with out_brief:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown("**Brief éditorial**")
                st.json(st.session_state.brief)
            with col_b2:
                st.markdown("**Budget de mots**")
                st.json(st.session_state.word_budget)
                st.markdown("**Liens internes**")
                st.json(st.session_state.internal_links)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — History
# ─────────────────────────────────────────────────────────────────────────────

with tab_history:
    st.markdown("## 📂 Historique des articles générés")

    history = st.session_state.get("history", [])
    saved_files = sorted(OUTPUT_DIR.glob("*.md"), reverse=True)

    if saved_files:
        st.markdown(f"**{len(saved_files)} article(s) sauvegardé(s)** dans `output/`")
        for f in saved_files[:20]:
            name = f.stem
            html_f = f.with_suffix(".html")
            col_a, col_b, col_c = st.columns([3, 1, 1])
            col_a.markdown(f"`{name}`")
            col_b.download_button("⬇️ .md", data=f.read_text(encoding="utf-8"), file_name=f.name, mime="text/markdown", key=f"md_{name}")
            if html_f.exists():
                col_c.download_button("⬇️ .html", data=html_f.read_text(encoding="utf-8"), file_name=html_f.name, mime="text/html", key=f"html_{name}")
    else:
        st.info("Aucun article généré pour l'instant. Lance ta première génération !")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Help
# ─────────────────────────────────────────────────────────────────────────────

with tab_help:
    st.markdown("## ❓ Guide d'utilisation")

    st.markdown("""
### 🚀 Démarrage rapide

1. **Ajoute tes clés API** dans la barre latérale gauche
   - [Anthropic (Claude)](https://console.anthropic.com/) → `ANTHROPIC_API_KEY`
   - [Google Gemini](https://aistudio.google.com/app/apikey) → `GOOGLE_API_KEY`

2. **Prépare ton CSV** avec tes mots-clés (Ahrefs, SEMrush, Ubersuggest, etc.)
   - Colonne requise : Keyword / Mot-clé
   - Colonnes optionnelles : Volume, Difficulty/KD, CPC

3. **Entre l'URL de ton sitemap XML** pour que l'outil connaisse ton site

4. **Optionnel** : URL de la page catégorie pour contextualiser le choix du mot-clé

5. **Ajuste** la longueur cible et le nombre de concurrents à analyser

6. **Lance** la génération et attends 2-5 minutes

---

### 📋 Pipeline détaillé

| Étape | Module | IA | Durée |
|-------|--------|-----|-------|
| 1. Pépite Finder | `pepite_finder.py` | Gemini | ~10s |
| 2a. Analyse SERP | `serp_analyzer.py` | Google Scraping | ~20-40s |
| 2b. Insights | `insight_miner.py` | Gemini | ~5s |
| 2c. Brief | `semantic_architect.py` | Gemini | ~10s |
| 2d. Maillage | `semantic_architect.py` | Gemini | ~8s |
| 3. Budget mots | `seo_calculator.py` | Gemini | ~5s |
| 4. Rédaction | `redacteur.py` | Claude | ~40-80s |

---

### 📁 Format du CSV

Le parseur accepte automatiquement les exports de :
- **Ahrefs** : Keyword, Volume, KD, CPC, Traffic
- **SEMrush** : Keyword, Search Volume, Keyword Difficulty, CPC
- **Ubersuggest** : keyword, searchVolume, cpc, competition
- **Google Keyword Planner** : Keyword, Avg. monthly searches
- **Tout CSV générique** avec une colonne Keyword

---

### 🖼️ Images dans l'article

Les balises `<figure>` générées ont des `src=""` vides.
Après génération, tu dois :
1. Trouver ou créer une image pertinente pour chaque emplacement
2. Remplacer `src=""` par le chemin ou l'URL de ton image
3. L'attribut `alt` est déjà optimisé SEO

---

### ⚠️ Scraping Google

Le scraping direct Google peut parfois être bloqué (CAPTCHA).
- Si 0 concurrent est analysé, l'outil continue quand même avec Gemini seul
- Pour plus de fiabilité, réduis le nombre de concurrents à analyser
- Le SERP scraping est uniquement à usage personnel/interne

---

### 💡 Astuces SEO

- **Longueur** : 1500-2500 mots pour les articles informationnels, 800-1200 pour les commerciaux
- **Fréquence** : Génère 2-3 articles/semaine sur le même domaine pour construire l'autorité
- **Maillage** : Révise les liens internes pour t'assurer qu'ils pointent vers des pages existantes
- **Images** : Optimise les images (WebP, compression) avant upload — le SEO image compte
    """)
