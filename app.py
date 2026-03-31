import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
import plotly.graph_objects as go
from styles import apply_theme
from core.config import get_settings

from core.parser import extract_text_from_pdf, parse_sections, chunk_resume
from core.entities import build_entity_profile
from core.scorer import load_models, compute_all_signals
from core.store import build_collection
from core.gap_analysis import run_gap_analysis
from core.llm import analyse_with_llm, generate_rewrites, generate_interview_prep
from core.report import generate_report
from core.sample_data import SAMPLE_RESUME, SAMPLE_JD

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fit Scorer — Resume × JD",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
apply_theme()  # see styles.py for CSS definitions


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## ◈ Resume × JD Fit Scorer")
st.markdown(
    "<p style='color:#6b7280;font-size:0.9rem;'>5-signal analysis · section-level insights · LLM-powered recommendations</p>",
    unsafe_allow_html=True,
)

# ── Groq key check ────────────────────────────────────────────────────────────
get_settings()
settings = get_settings()

groq_key = settings.groq_api_key
if not groq_key:
    with st.expander("⚙️ Groq API key required for LLM analysis", expanded=True):
        key_input = st.text_input(
            "Paste your Groq API key (free at console.groq.com)",
            type="password",
            key="groq_input",
        )
        if key_input:
            os.environ["GROQ_API_KEY"] = key_input
            st.success("Key saved for this session.")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2, gap="large")
with col_l:
    st.markdown("**Resume** (PDF)")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
with col_r:
    st.markdown("**Job description**")
    jd_text = st.text_area(
        "JD", height=240,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed",
    )
 
btn_c1, btn_c2 = st.columns([3, 1], gap="small")
with btn_c1:
    analyse = st.button("◈ Analyse fit", type="primary", use_container_width='stretch')
with btn_c2:
    use_sample = st.button("Try sample ↗", use_container_width='stretch', help="Load a sample ML resume + JD")
 
if use_sample:
    st.session_state["sample_mode"] = True
    # Clear any previous results so fresh run happens
    for k in ["result", "gap_result", "llm_analysis", "rewrites", "interview_prep",
              "sections", "resume_text", "jd_text_used"]:
        st.session_state.pop(k, None)
    st.rerun()
 
sample_mode = st.session_state.get("sample_mode", False)
if sample_mode:
    st.markdown(
        "<div style='background:#0b1f14;border:1px solid #166534;border-radius:8px;"
        "padding:0.5rem 1rem;font-size:0.82rem;color:#4ade80;margin-bottom:0.5rem;'>"
        "◉ Demo mode — sample ML resume + Senior ML Engineer JD loaded.</div>",
        unsafe_allow_html=True,
    )
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
 
def score_colour(s):
    if s >= 75: return "#4ade80"
    if s >= 50: return "#fb923c"
    return "#f87171"
 
def verdict_class(v):
    v = v.lower()
    if "strong" in v: return "verdict-strong"
    if "good" in v:   return "verdict-good"
    if "moderate" in v: return "verdict-moderate"
    return "verdict-weak"
 
def status_pill(status):
    cls = {"strong":"pill-strong","partial":"pill-partial","weak":"pill-weak","absent":"pill-absent"}.get(status,"pill-partial")
    return f"<span class='{cls}'>{status.upper()}</span>"
 
def mini_metric(col, label, val, colour="#e2e2f0"):
    col.markdown(
        f"<div class='metric-mini'>"
        f"<div class='metric-mini-val' style='color:{colour};'>{val}</div>"
        f"<div class='metric-mini-label'>{label}</div></div>",
        unsafe_allow_html=True,
    )
 
def render_gauge(score):
    c = score_colour(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"font":{"size":38,"color":c,"family":"DM Mono, monospace"}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":0,"tickfont":{"color":"#6b7280","size":9}},
            "bar":{"color":c,"thickness":0.22},
            "bgcolor":"#13131d","borderwidth":0,
            "steps":[{"range":[0,100],"color":"#0d0d0f"}],
            "threshold":{"line":{"color":c,"width":3},"thickness":0.75,"value":score},
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0d0d0f", plot_bgcolor="#0d0d0f",
        font={"color":"#e2e2f0"},
        margin=dict(t=20,b=0,l=10,r=10), height=180,
    )
    st.plotly_chart(fig, use_container_width='stretch')
 
def render_radar(signal_scores):
    labels = {
        "section_semantic":"Semantic fit","cross_encoder":"Contextual match",
        "tool_f1":"Tool F1","domain_coverage":"Domain fit","experience_fit":"Experience fit",
    }
    cats = list(labels.values())
    vals = [signal_scores.get(k,0) for k in labels]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
        fillcolor="rgba(124,58,237,0.12)",
        line=dict(color="#7c3aed",width=2), marker=dict(size=5,color="#a78bfa"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#13131d",
            radialaxis=dict(visible=True,range=[0,100],tickfont=dict(color="#6b7280",size=9),gridcolor="#1e1e2e",linecolor="#1e1e2e"),
            angularaxis=dict(tickfont=dict(color="#c4c4d4",size=10),gridcolor="#1e1e2e",linecolor="#1e1e2e"),
        ),
        paper_bgcolor="#0d0d0f", showlegend=False,
        margin=dict(t=20,b=20,l=30,r=30), height=300,
    )
    st.plotly_chart(fig, use_container_width='stretch')
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# Analysis trigger — runs only on button press; stores everything in session state
# ═══════════════════════════════════════════════════════════════════════════════
 
if analyse or sample_mode:
    # Only re-run analysis if inputs changed or results not yet in state
    needs_run = "result" not in st.session_state
 
    if analyse:
        # Fresh run triggered by button
        needs_run = True
        st.session_state.pop("sample_mode", None)
        sample_mode = False
 
    if needs_run:
        if sample_mode:
            resume_text = SAMPLE_RESUME
            jd_text     = SAMPLE_JD
        else:
            if not uploaded_file:
                st.warning("Upload your resume PDF.")
                st.stop()
            if not jd_text.strip():
                st.warning("Paste a job description.")
                st.stop()
 
        progress = st.progress(0, text="Loading models...")
        bi_encoder, cross_encoder = load_models()
        progress.progress(12, text="Extracting resume text...")
 
        if not sample_mode:
            resume_text, extract_method = extract_text_from_pdf(uploaded_file)
            if not resume_text:
                st.error("Could not extract text from the PDF.")
                st.stop()
            if extract_method == "ocr":
                st.info("🔍 Scanned PDF — OCR used.")
 
        sections = parse_sections(resume_text)
        chunks   = chunk_resume(sections)
        progress.progress(28, text="Building entity profiles...")
 
        resume_profile = build_entity_profile(resume_text)
        jd_profile     = build_entity_profile(jd_text)
        progress.progress(40, text="Embedding chunks...")
 
        chunk_texts      = [c["text"] for c in chunks]
        chunk_embeddings = bi_encoder.encode(
            chunk_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        progress.progress(52, text="Building ChromaDB collection...")
        collection   = build_collection(chunks, chunk_embeddings)
        jd_embedding = bi_encoder.encode(
            [jd_text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )[0]
 
        progress.progress(62, text="Computing 5 signals...")
        result = compute_all_signals(
            chunks, resume_text, jd_text, jd_embedding,
            resume_profile, jd_profile, bi_encoder, cross_encoder,
            collection=collection,
        )
 
        progress.progress(72, text="Running gap analysis...")
        gap_result = run_gap_analysis(
            jd_text=jd_text, resume_text=resume_text,
            collection=collection, bi_encoder=bi_encoder,
            tool_overlap=result.tool_overlap,
        )
 
        progress.progress(82, text="Running LLM deep analysis...")
        llm_analysis = {}
        rewrites     = {}
        interview_prep = {}
        try:
            llm_analysis = analyse_with_llm(resume_text, jd_text)
        except Exception as e:
            st.warning(f"LLM analysis skipped: {e}")
 
        if llm_analysis:
            progress.progress(91, text="Generating resume rewrites...")
            try:
                rewrites = generate_rewrites(resume_text, jd_text, llm_analysis.get("priority_gaps", []))
            except Exception as e:
                st.warning(f"Rewrite suggestions skipped: {e}")
 
            progress.progress(96, text="Generating interview prep...")
            try:
                interview_prep = generate_interview_prep(resume_text, jd_text, llm_analysis.get("priority_gaps", []))
            except Exception as e:
                st.warning(f"Interview prep skipped: {e}")
 
        progress.progress(100, text="Done.")
        progress.empty()
 
        # ── Persist everything in session state ───────────────────────────────
        st.session_state["result"]         = result
        st.session_state["gap_result"]     = gap_result
        st.session_state["llm_analysis"]   = llm_analysis
        st.session_state["rewrites"]       = rewrites
        st.session_state["interview_prep"] = interview_prep
        st.session_state["sections"]       = sections
        st.session_state["resume_text"]    = resume_text
        st.session_state["jd_text_used"]   = jd_text
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# Results — only shown when state has results; survive any widget interaction
# ═══════════════════════════════════════════════════════════════════════════════
 
if "result" not in st.session_state:
    st.stop()
 
result         = st.session_state["result"]
gap_result     = st.session_state["gap_result"]
llm_analysis   = st.session_state["llm_analysis"]
rewrites       = st.session_state["rewrites"]
interview_prep = st.session_state["interview_prep"]
sections       = st.session_state["sections"]
resume_text    = st.session_state["resume_text"]
jd_text_used   = st.session_state["jd_text_used"]
 
verdict        = llm_analysis.get("overall_verdict", "")
verdict_reason = llm_analysis.get("verdict_reasoning", "")
v_cls          = verdict_class(verdict)
 
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
 
# ── Score banner ──────────────────────────────────────────────────────────────
banner_l, banner_r = st.columns([1, 2], gap="large")
with banner_l:
    render_gauge(result.final_score)
    if verdict:
        st.markdown(
            f"<div style='text-align:center;margin-top:0.4rem;'>"
            f"<span class='verdict-badge {v_cls}'>{verdict}</span></div>",
            unsafe_allow_html=True,
        )
 
with banner_r:
    if verdict_reason:
        st.markdown(
            f"<p style='color:#c4c4d4;font-size:0.92rem;line-height:1.8;"
            f"margin-top:1.2rem;'>{verdict_reason}</p>",
            unsafe_allow_html=True,
        )
    signal_labels = {
        "section_semantic":"Section semantic","cross_encoder":"Contextual match",
        "tool_f1":"Tool F1","domain_coverage":"Domain fit","experience_fit":"Experience fit",
    }
    sig_cols = st.columns(5)
    for col, (k, label) in zip(sig_cols, signal_labels.items()):
        val = result.signal_scores.get(k, 0)
        mini_metric(col, label, f"{val:.0f}", score_colour(val))
 
# Download button
if llm_analysis:
    report_md = generate_report(result, gap_result, llm_analysis, rewrites, interview_prep)
    st.download_button(
        "⬇ Download full report (.md)", data=report_md,
        file_name="fit_report.md", mime="text/markdown",
        use_container_width='stretch',
    )
 
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# Tabbed results
# ═══════════════════════════════════════════════════════════════════════════════
 
# Pre-initialise every widget key that lives inside a tab.
# Without this, the first interaction with any widget creates its session_state
# key mid-rerun, Streamlit treats it as a "new" widget, and resets the active
# tab back to index 0 (Overview). Initialising here before st.tabs() is called
# prevents that reset entirely.
st.session_state.setdefault("cov_filter", "All")
st.session_state.setdefault("req_sort", "JD emphasis (high → low)")
st.session_state.setdefault("interview_cat_filter", "All")
 
tab_overview, tab_coverage, tab_skills, tab_improve, tab_interview = st.tabs([
    "📊 Overview",
    "🔍 Coverage",
    "🛠 Skills & Tools",
    "✍ Improve",
    "🎯 Interview Prep",
])
 
 
# ── TAB 1: Overview ───────────────────────────────────────────────────────────
with tab_overview:
    t1_l, t1_r = st.columns([1, 1], gap="large")
 
    with t1_l:
        st.markdown("**Signal radar**")
        render_radar(result.signal_scores)
 
    with t1_r:
        st.markdown("**Section-wise semantic similarity**")
        SECTION_LABEL = {
            "experience":"Experience","work experience":"Experience",
            "professional experience":"Experience","employment":"Experience",
            "skills":"Skills","technical skills":"Skills","core competencies":"Skills",
            "projects":"Projects","education":"Education","academic background":"Education",
            "summary":"Summary","certifications":"Certifications","full_text":"Full text",
        }
        seen = set()
        for sec, score in sorted(result.section_semantic_breakdown.items(), key=lambda x: -x[1]):
            label = SECTION_LABEL.get(sec, sec.title())
            if label in seen:
                continue
            seen.add(label)
            c = score_colour(score)
            st.markdown(
                f"<div class='card' style='padding:0.9rem 1.2rem;margin-bottom:0.5rem;'>"
                f"<div style='display:flex;align-items:center;gap:1rem;'>"
                f"<span style='min-width:90px;font-size:0.85rem;color:#e2e2f0;'>{label}</span>"
                f"<div style='flex:1;background:#1e1e2e;border-radius:2px;height:5px;'>"
                f"<div style='width:{score}%;background:{c};height:5px;border-radius:2px;'></div></div>"
                f"<span style='font-family:\"DM Mono\",monospace;font-size:0.82rem;color:{c};min-width:32px;text-align:right;'>{score:.0f}</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
 
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
 
    # Experience alignment card
    exp = result.exp_alignment
    level_fit  = exp.get("level_fit","unknown").replace("_"," ").title()
    years_fit  = exp.get("years_fit","unknown").replace("_"," ")
    ea1, ea2, ea3, ea4 = st.columns(4)
    mini_metric(ea1, "Level fit", level_fit, score_colour(result.signal_scores.get("experience_fit",50)))
    mini_metric(ea2, "JD seniority", exp.get("jd_seniority","—").title())
    mini_metric(ea3, "JD years req.", exp.get("jd_years_required",0) or "—")
    mini_metric(ea4, "Years fit", years_fit.replace("gap ","gap "))
 
    # LLM domain + cultural signals
    domain_info = llm_analysis.get("domain_alignment","")
    cultural    = llm_analysis.get("cultural_signals","")
    if domain_info or cultural:
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        if domain_info:
            st.markdown(
                f"<div class='card'><div class='card-title'>Domain alignment</div>"
                f"<p style='color:#c4c4d4;font-size:0.87rem;margin:0;'>{domain_info}</p></div>",
                unsafe_allow_html=True,
            )
        if cultural:
            st.markdown(
                f"<div class='card'><div class='card-title'>Cultural signals</div>"
                f"<p style='color:#c4c4d4;font-size:0.87rem;margin:0;'>{cultural}</p></div>",
                unsafe_allow_html=True,
            )
 
 
# ── TAB 2: Coverage ───────────────────────────────────────────────────────────
with tab_coverage:
    reqs = gap_result.requirements
    total   = len(reqs)
    strong_n  = sum(1 for r in reqs if r.coverage_label=="strong")
    partial_n = sum(1 for r in reqs if r.coverage_label=="partial")
    weak_n    = sum(1 for r in reqs if r.coverage_label=="weak")
    absent_n  = sum(1 for r in reqs if r.coverage_label=="absent")
 
    cv1,cv2,cv3,cv4,cv5 = st.columns(5)
    mini_metric(cv1,"Total reqs", total)
    mini_metric(cv2,"Strong",  strong_n,  "#4ade80")
    mini_metric(cv3,"Partial", partial_n, "#fb923c")
    mini_metric(cv4,"Weak",    weak_n,    "#fbbf24")
    mini_metric(cv5,"Absent",  absent_n,  "#f87171")
 
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
 
    # Section heatmap
    st.markdown("**Section coverage heatmap**")
    hm = gap_result.section_heatmap
    if hm:
        secs_h   = [h.section.title()[:18] for h in hm]
        fig_hm = go.Figure()
        for name, attr, colour in [
            ("Strong",  "strong_count",  "#4ade80"),
            ("Partial", "partial_count", "#fb923c"),
            ("Weak",    "weak_count",    "#fbbf24"),
            ("Absent",  "absent_count",  "#f87171"),
        ]:
            fig_hm.add_trace(go.Bar(
                name=name, x=secs_h, y=[getattr(h, attr) for h in hm],
                marker_color=colour, opacity=0.85,
            ))
        fig_hm.update_layout(
            barmode="stack", paper_bgcolor="#0d0d0f", plot_bgcolor="#13131d",
            font=dict(color="#c4c4d4",size=11),
            legend=dict(orientation="h",y=-0.22,bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10,b=40,l=10,r=10), height=240,
            xaxis=dict(gridcolor="#1e1e2e",linecolor="#1e1e2e"),
            yaxis=dict(gridcolor="#1e1e2e",linecolor="#1e1e2e",title="Reqs"),
        )
        st.plotly_chart(fig_hm, use_container_width='stretch')
 
    st.markdown("**Per-requirement evidence**")
    # Filters — stored in session state so they don't wipe results
    fc1, fc2 = st.columns(2)
    with fc1:
        show_label = st.selectbox(
            "Filter by coverage",
            ["All","Absent","Weak","Partial","Strong"],
            key="cov_filter",
        )
    with fc2:
        sort_by = st.selectbox(
            "Sort by",
            ["JD emphasis (high → low)","Coverage score (low → high)"],
            key="req_sort",
        )
 
    filtered = reqs if show_label == "All" else [r for r in reqs if r.coverage_label == show_label.lower()]
    if sort_by == "Coverage score (low → high)":
        filtered = sorted(filtered, key=lambda r: r.coverage_score)
    else:
        filtered = sorted(filtered, key=lambda r: -r.jd_emphasis)
 
    LABEL_STYLE = {
        "strong":  ("#052e16","#4ade80","#166534","●"),
        "partial": ("#2d1a0a","#fb923c","#7c2d12","◐"),
        "weak":    ("#1a1400","#fbbf24","#713f12","○"),
        "absent":  ("#2d0a0a","#f87171","#7f1d1d","✗"),
    }
    def _req_card(req):
        bg, fg, border, icon = LABEL_STYLE.get(req.coverage_label, LABEL_STYLE["absent"])
        emp_w = round(req.jd_emphasis * 100)
        cov_w = round(req.coverage_score * 100)
        snippet = req.best_match_text[:160].replace("<","&lt;").replace(">","&gt;")
        if len(req.best_match_text) > 160:
            snippet += "…"
        st.markdown(
            f"<div style='background:#13131d;border:1px solid {border};"
            f"border-left:3px solid {fg};border-radius:0 10px 10px 0;"
            f"padding:0.9rem 1.1rem;margin-bottom:0.6rem;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:0.8rem;'>"
            f"<div style='flex:1;'>"
            f"<div style='color:#e2e2f0;font-size:0.86rem;font-weight:600;margin-bottom:0.4rem;'>{req.requirement[:130]}</div>"
            f"<div style='display:flex;gap:1.2rem;font-size:0.76rem;color:#6b7280;margin-bottom:0.4rem;'>"
            f"<span>Emphasis <b style='color:#a78bfa'>{emp_w}%</b></span>"
            f"<span>Coverage <b style='color:{fg}'>{cov_w}%</b></span>"
            f"<span>via <em>{req.best_match_section}</em></span></div>"
            f"<div style='background:#0d0d0f;border-radius:5px;padding:0.4rem 0.6rem;"
            f"font-family:DM Mono,monospace;font-size:0.73rem;color:#6b7280;"
            f"border:1px solid #1e1e2e;'>{snippet}</div>"
            f"</div>"
            f"<span style='background:{bg};color:{fg};border:1px solid {border};"
            f"border-radius:999px;padding:2px 9px;font-size:0.7rem;font-weight:600;"
            f"white-space:nowrap;font-family:DM Mono,monospace;'>{icon} {req.coverage_label.upper()}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
 
    FIRST_N = 10
    for req in filtered[:FIRST_N]:
        _req_card(req)
 
    if len(filtered) > FIRST_N:
        remaining = filtered[FIRST_N:]
        with st.expander(f"Show {len(remaining)} more requirements"):
            for req in remaining:
                _req_card(req)
 
 
# ── TAB 3: Skills & Tools ─────────────────────────────────────────────────────
with tab_skills:
    to = result.tool_overlap
 
    st.markdown("**Tool alignment**")
    tm1, tm2, tm3, tm4 = st.columns(4)
    mini_metric(tm1, "Matched",  len(to["matched"]), "#4ade80")
    mini_metric(tm2, "Missing",  len(to["missing"]), "#f87171")
    mini_metric(tm3, "Bonus",    len(to["bonus"]),   "#a5b4fc")
    mini_metric(tm4, "F1 score", f"{round(to['f1']*100)}%", score_colour(to['f1']*100))
 
    st.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)
 
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.markdown("<div class='card-title' style='color:#4ade80;margin-bottom:0.4rem;'>✓ Matched</div>", unsafe_allow_html=True)
        chips = " ".join(f"<span class='tool-chip tool-match'>{t}</span>" for t in to["matched"])
        st.markdown(chips or "<span style='color:#6b7280;font-size:0.82rem;'>None detected</span>", unsafe_allow_html=True)
    with tc2:
        st.markdown("<div class='card-title' style='color:#f87171;margin-bottom:0.4rem;'>✗ Missing from resume</div>", unsafe_allow_html=True)
        chips = " ".join(f"<span class='tool-chip tool-missing'>{t}</span>" for t in to["missing"])
        st.markdown(chips or "<span style='color:#6b7280;font-size:0.82rem;'>No gaps detected</span>", unsafe_allow_html=True)
    with tc3:
        st.markdown("<div class='card-title' style='color:#a5b4fc;margin-bottom:0.4rem;'>+ Bonus skills</div>", unsafe_allow_html=True)
        chips = " ".join(f"<span class='tool-chip tool-bonus'>{t}</span>" for t in to["bonus"][:14])
        st.markdown(chips or "<span style='color:#6b7280;font-size:0.82rem;'>—</span>", unsafe_allow_html=True)
 
    if gap_result.clusters:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("**Skill gap clusters** — missing tools grouped by category")
        PRIORITY_STYLE = {
            "critical":     ("#2d0a0a","#f87171","#7f1d1d"),
            "important":    ("#2d1a0a","#fb923c","#7c2d12"),
            "nice_to_have": ("#1e1b4b","#a5b4fc","#3730a3"),
        }
        gc1, gc2 = st.columns(2, gap="large")
        half = (len(gap_result.clusters) + 1) // 2
        for col, cluster_list in [(gc1, gap_result.clusters[:half]), (gc2, gap_result.clusters[half:])]:
            with col:
                for cl in cluster_list:
                    bg, fg, border = PRIORITY_STYLE.get(cl.priority, PRIORITY_STYLE["nice_to_have"])
                    miss = " ".join(f"<span class='tool-chip tool-missing'>{t}</span>" for t in cl.missing_tools)
                    alts = " ".join(f"<span class='tool-chip tool-match'>{t}</span>" for t in cl.has_alternatives)
                    alts_row = f"<div style='margin-top:0.4rem;font-size:0.74rem;color:#6b7280;'>You have: {alts}</div>" if alts else ""
                    st.markdown(
                        f"<div class='card' style='border-color:{border};margin-bottom:0.7rem;'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;'>"
                        f"<span style='font-weight:600;color:#e2e2f0;font-size:0.86rem;'>{cl.category}</span>"
                        f"<span style='background:{bg};color:{fg};border:1px solid {border};"
                        f"border-radius:999px;padding:1px 9px;font-size:0.68rem;"
                        f"font-family:DM Mono,monospace;'>{cl.priority.replace('_',' ')}</span>"
                        f"</div>{miss}{alts_row}</div>",
                        unsafe_allow_html=True,
                    )
 
    # Domain alignment
    dom = result.domain_overlap
    matched_d = dom.get("matched_domains",[])
    missing_d = dom.get("missing_domains",[])
    if matched_d or missing_d:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("**Domain alignment**")
        DOMAIN_LABEL = {
            "healthcare_ai":"Healthcare AI","finance_ai":"Finance AI",
            "nlp_llm":"NLP / LLM","computer_vision":"Computer Vision",
            "mlops_platform":"MLOps / Platform","data_engineering":"Data Engineering",
        }
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("<div class='card-title' style='color:#4ade80;'>Matched domains</div>", unsafe_allow_html=True)
            for d in matched_d:
                st.markdown(f"<span class='tool-chip tool-match'>{DOMAIN_LABEL.get(d,d)}</span>", unsafe_allow_html=True)
        with d2:
            st.markdown("<div class='card-title' style='color:#f87171;'>Missing domains</div>", unsafe_allow_html=True)
            for d in missing_d:
                st.markdown(f"<span class='tool-chip tool-missing'>{DOMAIN_LABEL.get(d,d)}</span>", unsafe_allow_html=True)
 
 
# ── TAB 4: Improve ───────────────────────────────────────────────────────────
with tab_improve:
    if not llm_analysis:
        st.info("Add a Groq API key to unlock LLM-powered improvement suggestions.")
        st.stop()
 
    # Hard requirements check
    hard_reqs = llm_analysis.get("hard_requirements_check", [])
    if hard_reqs:
        st.markdown("**Hard requirements check**")
        for req in hard_reqs:
            status = req.get("status","partial")
            icon   = {"met":"✅","partial":"⚠️","not_met":"❌"}.get(status,"⚠️")
            colour = {"met":"#4ade80","partial":"#fb923c","not_met":"#f87171"}.get(status,"#fb923c")
            st.markdown(
                f"<div class='card' style='padding:0.8rem 1.1rem;margin-bottom:0.5rem;'>"
                f"<div style='display:flex;gap:0.7rem;align-items:flex-start;'>"
                f"<span>{icon}</span>"
                f"<div><div style='color:#e2e2f0;font-size:0.85rem;font-weight:600;'>{req.get('requirement','')}</div>"
                f"<div style='color:#6b7280;font-size:0.8rem;margin-top:0.2rem;'>{req.get('evidence','')}</div>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
 
    # Strengths + Gaps side by side
    strengths = llm_analysis.get("top_strengths", [])
    gaps      = llm_analysis.get("priority_gaps", [])
    imp_l, imp_r = st.columns(2, gap="large")
 
    with imp_l:
        st.markdown("**Top strengths for this role**")
        for s in strengths:
            st.markdown(
                f"<div class='strength-block'>"
                f"<div style='font-weight:600;margin-bottom:0.3rem;'>{s.get('strength','')}</div>"
                f"<div style='font-size:0.8rem;opacity:0.85;'>{s.get('evidence','')} — <em>{s.get('relevance','')}</em></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
 
    with imp_r:
        st.markdown("**Priority gaps & fixes**")
        for g in gaps:
            sev   = g.get("severity","medium")
            sev_c = {"high":"#ef4444","medium":"#f59e0b","low":"#10b981"}.get(sev,"#f59e0b")
            st.markdown(
                f"<div class='gap-block gap-{sev}'>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:0.3rem;'>"
                f"<span style='font-weight:600;color:#e2e2f0;'>{g.get('gap','')}</span>"
                f"<span style='font-family:DM Mono,monospace;font-size:0.7rem;color:{sev_c};text-transform:uppercase;'>{sev}</span>"
                f"</div>"
                f"<div style='font-size:0.81rem;margin-bottom:0.3rem;'>{g.get('impact','')}</div>"
                f"<div style='font-size:0.81rem;color:#a78bfa;'>→ {g.get('fix','')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
 
    # Final recommendation
    rec = llm_analysis.get("final_recommendation","")
    if rec:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("**Final recommendation**")
        st.markdown(f"<div class='rec-block'>{rec}</div>", unsafe_allow_html=True)
 
    # Resume rewrite suggestions
    if rewrites:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("**Resume rewrite suggestions**")
 
        quick_wins = rewrites.get("quick_wins", [])
        if quick_wins:
            qw_items = "".join(
                f"<li style='color:#c4c4d4;font-size:0.85rem;margin-bottom:0.35rem;'>"
                f"<span style='color:#a78bfa;'>→</span> {qw}</li>"
                for qw in quick_wins
            )
            st.markdown(
                f"<div class='card' style='margin-bottom:1.2rem;'>"
                f"<div class='card-title'>Quick wins — under 5 minutes each</div>"
                f"<ul style='margin:0;padding-left:0.4rem;list-style:none;'>{qw_items}</ul></div>",
                unsafe_allow_html=True,
            )
 
        summary_rw = rewrites.get("summary_rewrite", {})
        if summary_rw:
            rw1, rw2 = st.columns(2, gap="large")
            with rw1:
                st.markdown(
                    f"<div class='card'><div class='card-title'>Current summary</div>"
                    f"<p style='color:#6b7280;font-size:0.84rem;line-height:1.7;margin:0;'>"
                    f"{summary_rw.get('original','—')}</p></div>",
                    unsafe_allow_html=True,
                )
            with rw2:
                st.markdown(
                    f"<div class='card' style='border-color:#3730a3;'>"
                    f"<div class='card-title' style='color:#a78bfa;'>Suggested summary</div>"
                    f"<p style='color:#e2e2f0;font-size:0.84rem;line-height:1.7;margin:0 0 0.5rem;'>"
                    f"{summary_rw.get('rewritten','—')}</p>"
                    f"<div style='font-size:0.77rem;color:#6b7280;'>{summary_rw.get('why','')}</div></div>",
                    unsafe_allow_html=True,
                )
 
        for rw in rewrites.get("rewrite_suggestions", []):
            sec_name = rw.get("section","Section")
            kws      = rw.get("keywords_added", [])
            kw_chips = " ".join(
                f"<span style='background:#1e1b4b;color:#a5b4fc;border:1px solid #3730a3;"
                f"border-radius:4px;padding:1px 6px;font-size:0.68rem;"
                f"font-family:DM Mono,monospace;'>{k}</span>"
                for k in kws
            )
            rwa, rwb = st.columns(2, gap="large")
            with rwa:
                st.markdown(
                    f"<div class='card'><div class='card-title'>{sec_name} — current</div>"
                    f"<p style='color:#6b7280;font-size:0.84rem;line-height:1.7;margin:0;'>"
                    f"{rw.get('original_snippet','—')}</p></div>",
                    unsafe_allow_html=True,
                )
            with rwb:
                kw_section = f"<div style='margin-top:0.4rem;'>{kw_chips}</div>" if kw_chips else ""
                st.markdown(
                    f"<div class='card' style='border-color:#3730a3;'>"
                    f"<div class='card-title' style='color:#a78bfa;'>{sec_name} — suggested</div>"
                    f"<p style='color:#e2e2f0;font-size:0.84rem;line-height:1.7;margin:0 0 0.4rem;'>"
                    f"{rw.get('rewritten','—')}</p>"
                    f"<div style='font-size:0.77rem;color:#6b7280;'>{rw.get('why','')}</div>"
                    f"{kw_section}</div>",
                    unsafe_allow_html=True,
                )
 
 
# ── TAB 5: Interview Prep ─────────────────────────────────────────────────────
with tab_interview:
    if not interview_prep:
        st.info("Add a Groq API key to unlock interview prep.")
        st.stop()
 
    CAT_STYLE = {
        "Technical":     ("#0c447c","#60a5fa"),
        "System Design": ("#3730a3","#a5b4fc"),
        "Behavioural":   ("#065f46","#34d399"),
        "Gap-related":   ("#7f1d1d","#f87171"),
    }
 
    questions = interview_prep.get("likely_questions", [])
    # Category filter — works correctly because results are in session state
    all_cats = sorted(set(q.get("category","Technical") for q in questions))
    cat_filter = st.selectbox("Filter by category", ["All"] + all_cats, key="interview_cat_filter")
    filtered_qs = questions if cat_filter == "All" else [q for q in questions if q.get("category") == cat_filter]
 
    for q in filtered_qs:
        cat    = q.get("category","Technical")
        bg, fg = CAT_STYLE.get(cat, ("#1e1e2e","#e2e2f0"))
        tps    = q.get("talking_points", [])
        tp_html = "".join(
            f"<li style='color:#bbf7d0;font-size:0.82rem;margin-bottom:0.25rem;'>{tp}</li>"
            for tp in tps
        )
        wo = q.get("watch_out","")
        wo_html = (
            f"<div style='margin-top:0.5rem;font-size:0.79rem;color:#fca5a5;'>⚠️ {wo}</div>"
        ) if wo else ""
        st.markdown(
            f"<div class='card' style='margin-bottom:0.8rem;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:0.8rem;margin-bottom:0.5rem;'>"
            f"<span style='font-weight:600;color:#e2e2f0;font-size:0.88rem;line-height:1.5;'>{q.get('question','')}</span>"
            f"<span style='background:{bg};color:{fg};border:1px solid {fg}44;"
            f"border-radius:999px;padding:2px 9px;font-size:0.68rem;white-space:nowrap;"
            f"font-family:DM Mono,monospace;'>{cat}</span>"
            f"</div>"
            f"<div style='font-size:0.79rem;color:#6b7280;margin-bottom:0.4rem;'>{q.get('why_asked','')}</div>"
            f"<ul style='margin:0;padding-left:1rem;'>{tp_html}</ul>"
            f"{wo_html}"
            f"</div>",
            unsafe_allow_html=True,
        )
 
    # System design
    sd = interview_prep.get("system_design_topic", {})
    if sd:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        sd_pts = "".join(
            f"<li style='color:#c4c4d4;font-size:0.84rem;margin-bottom:0.3rem;'>{pt}</li>"
            for pt in sd.get("suggested_approach", [])
        )
        st.markdown(
            f"<div class='card' style='border-color:#3730a3;'>"
            f"<div class='card-title' style='color:#a78bfa;'>System design topic</div>"
            f"<div style='font-weight:600;color:#e2e2f0;margin-bottom:0.5rem;'>{sd.get('topic','')}</div>"
            f"<ul style='margin:0;padding-left:1rem;'>{sd_pts}</ul></div>",
            unsafe_allow_html=True,
        )
 
    # Questions to ask
    q2ask = interview_prep.get("questions_to_ask_interviewer", [])
    if q2ask:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        items = "".join(
            f"<li style='color:#c4c4d4;font-size:0.84rem;margin-bottom:0.35rem;'>{q}</li>"
            for q in q2ask
        )
        st.markdown(
            f"<div class='card' style='border-color:#0f6e56;'>"
            f"<div class='card-title' style='color:#2dd4bf;'>Questions to ask the interviewer</div>"
            f"<ul style='margin:0;padding-left:1rem;'>{items}</ul></div>",
            unsafe_allow_html=True,
        )
 
# Debug
with st.expander("Debug — resume sections detected", expanded=False):
    for sec, sec_content in sections.items():
        st.markdown(f"**{sec.title()}** — {len(sec_content.split())} words")
        st.text(sec_content[:300] + ("..." if len(sec_content) > 300 else ""))
