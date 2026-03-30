import streamlit as st
import plotly.graph_objects as go
import os
from styles import apply_theme
from core.config import get_settings

from core.parser import extract_text_from_pdf, parse_sections, chunk_resume
from core.entities import build_entity_profile
from core.scorer import load_models, compute_all_signals
from core.store import build_collection
from core.gap_analysis import run_gap_analysis
from core.llm import analyse_with_llm

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fit Scorer — Resume × JD",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
apply_theme()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## ◈ Resume × JD Fit Scorer")
st.markdown(
    "<p style='color:#6b7280;font-size:0.9rem;'>5-signal analysis · section-level insights · LLM-powered recommendations</p>",
    unsafe_allow_html=True,
)

# ── Groq key check ────────────────────────────────────────────────────────────
groq_key = os.getenv("GROQ_API_KEY", "")
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
        "JD",
        height=260,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed",
    )

analyse = st.button("◈ Analyse fit", type="primary", use_container_width=True)


# ── Helper renderers ──────────────────────────────────────────────────────────

def verdict_class(v: str) -> str:
    v = v.lower()
    if "strong" in v: return "verdict-strong"
    if "good" in v:   return "verdict-good"
    if "moderate" in v: return "verdict-moderate"
    return "verdict-weak"


def status_pill(status: str) -> str:
    cls = {"strong": "pill-strong", "partial": "pill-partial", "weak": "pill-weak"}.get(status, "pill-partial")
    return f"<span class='{cls}'>{status.upper()}</span>"


def score_colour(score: float) -> str:
    if score >= 75: return "#4ade80"
    if score >= 50: return "#fb923c"
    return "#f87171"


def render_radar(signal_scores: dict) -> None:
    labels = {
        "section_semantic": "Semantic fit",
        "cross_encoder": "Contextual match",
        "tool_f1": "Tool alignment",
        "domain_coverage": "Domain fit",
        "experience_fit": "Experience fit",
    }
    cats = list(labels.values())
    vals = [signal_scores.get(k, 0) for k in labels]
    vals_closed = vals + [vals[0]]
    cats_closed = cats + [cats[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed, fill="toself",
        fillcolor="rgba(124,58,237,0.15)",
        line=dict(color="#7c3aed", width=2),
        marker=dict(size=6, color="#a78bfa"),
        name="Fit signals",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#13131d",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#6b7280", size=10), gridcolor="#1e1e2e", linecolor="#1e1e2e"),
            angularaxis=dict(tickfont=dict(color="#c4c4d4", size=11), gridcolor="#1e1e2e", linecolor="#1e1e2e"),
        ),
        paper_bgcolor="#0d0d0f",
        plot_bgcolor="#0d0d0f",
        showlegend=False,
        margin=dict(t=20, b=20, l=40, r=40),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_section_bars(section_breakdown: dict) -> None:
    LABEL_MAP = {
        "experience": "Experience",
        "work experience": "Experience",
        "professional experience": "Experience",
        "skills": "Skills",
        "technical skills": "Skills",
        "core competencies": "Skills",
        "projects": "Projects",
        "education": "Education",
        "academic background": "Education",
        "summary": "Summary",
        "certifications": "Certifications",
        "full_text": "Full text",
    }
    seen = set()
    deduped = {}
    for k, v in section_breakdown.items():
        label = LABEL_MAP.get(k, k.title())
        if label not in seen:
            deduped[label] = v
            seen.add(label)

    for label, score in sorted(deduped.items(), key=lambda x: -x[1]):
        colour = score_colour(score)
        st.markdown(f"""
        <div class='section-card'>
          <div class='section-card-title'>{label}</div>
          <div style='display:flex;align-items:center;gap:1rem;'>
            <div style='flex:1;background:#1e1e2e;border-radius:2px;height:4px;'>
              <div style='width:{score}%;background:{colour};height:4px;border-radius:2px;'></div>
            </div>
            <div style='font-family:"DM Mono",monospace;font-size:0.85rem;color:{colour};min-width:38px;text-align:right;'>{score:.0f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


def render_tool_chips(tool_overlap: dict) -> None:
    if tool_overlap["matched"]:
        st.markdown("**Matched tools**")
        chips = " ".join(f"<span class='tool-chip tool-match'>{t}</span>" for t in tool_overlap["matched"])
        st.markdown(chips, unsafe_allow_html=True)

    if tool_overlap["missing"]:
        st.markdown("**Missing from resume**")
        chips = " ".join(f"<span class='tool-chip tool-missing'>{t}</span>" for t in tool_overlap["missing"])
        st.markdown(chips, unsafe_allow_html=True)

    if tool_overlap["bonus"]:
        st.markdown("**Bonus skills (not required)**")
        chips = " ".join(f"<span class='tool-chip tool-bonus'>{t}</span>" for t in tool_overlap["bonus"][:12])
        st.markdown(chips, unsafe_allow_html=True)


def render_llm_analysis(analysis: dict) -> None:
    # Section analysis
    section_analysis = analysis.get("section_analysis", {})
    if section_analysis:
        st.markdown("#### Section-by-section breakdown")
        for sec_name, sec_data in section_analysis.items():
            score = sec_data.get("score", 0)
            status = sec_data.get("status", "partial")
            commentary = sec_data.get("commentary", "")
            highlights = sec_data.get("highlights", [])
            gaps = sec_data.get("gaps", [])
            colour = score_colour(score)
            pill = status_pill(status)
            highlights_html = "".join(f"<li style='color:#bbf7d0'>✓ {h}</li>" for h in highlights) if highlights else ""
            gaps_html = "".join(f"<li style='color:#fca5a5'>✗ {g}</li>" for g in gaps) if gaps else ""
            st.markdown(f"""
            <div class='section-card' style='margin-bottom:1rem;'>
              <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;'>
                <span style='font-weight:600;color:#e2e2f0;'>{sec_name.title()}</span>
                <span style='display:flex;align-items:center;gap:0.7rem;'>
                  {pill}
                  <span style='font-family:"DM Mono",monospace;font-size:0.9rem;color:{colour};'>{score}</span>
                </span>
              </div>
              <p style='color:#9ca3af;font-size:0.87rem;margin:0.5rem 0;'>{commentary}</p>
              <ul style='margin:0.5rem 0 0;padding-left:1.2rem;font-size:0.83rem;list-style:none;'>
                {highlights_html}{gaps_html}
              </ul>
            </div>
            """, unsafe_allow_html=True)

    # Hard requirements
    hard_reqs = analysis.get("hard_requirements_check", [])
    if hard_reqs:
        st.markdown("#### Hard requirements")
        for req in hard_reqs:
            status = req.get("status", "partial")
            icon = {"met": "✅", "partial": "⚠️", "not_met": "❌"}.get(status, "⚠️")
            colour = {"met": "#4ade80", "partial": "#fb923c", "not_met": "#f87171"}.get(status, "#fb923c")
            st.markdown(f"""
            <div class='section-card' style='padding:0.9rem 1.2rem;margin-bottom:0.6rem;'>
              <div style='display:flex;gap:0.7rem;align-items:flex-start;'>
                <span style='font-size:1rem;'>{icon}</span>
                <div>
                  <div style='color:#e2e2f0;font-size:0.87rem;font-weight:600;'>{req.get("requirement","")}</div>
                  <div style='color:#6b7280;font-size:0.82rem;margin-top:0.25rem;'>{req.get("evidence","")}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Strengths
    strengths = analysis.get("top_strengths", [])
    if strengths:
        st.markdown("#### Top strengths for this role")
        for s in strengths:
            st.markdown(f"""
            <div class='strength-block'>
              <div style='font-weight:600;margin-bottom:0.3rem;'>{s.get("strength","")}</div>
              <div style='font-size:0.82rem;opacity:0.8;'>{s.get("evidence","")} — <em>{s.get("relevance","")}</em></div>
            </div>
            """, unsafe_allow_html=True)

    # Priority gaps
    gaps = analysis.get("priority_gaps", [])
    if gaps:
        st.markdown("#### Priority gaps & fixes")
        for g in gaps:
            severity = g.get("severity", "medium")
            css_cls = f"gap-{severity}"
            sev_colour = {"high": "#ef4444", "medium": "#f59e0b", "low": "#10b981"}.get(severity, "#f59e0b")
            st.markdown(f"""
            <div class='insight-block {css_cls}'>
              <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;'>
                <span style='font-weight:600;color:#e2e2f0;'>{g.get("gap","")}</span>
                <span style='font-family:"DM Mono",monospace;font-size:0.72rem;color:{sev_colour};text-transform:uppercase;'>{severity}</span>
              </div>
              <div style='font-size:0.83rem;margin-bottom:0.5rem;'>{g.get("impact","")}</div>
              <div style='font-size:0.83rem;color:#a78bfa;'>→ {g.get("fix","")}</div>
            </div>
            """, unsafe_allow_html=True)

    # Final recommendation
    rec = analysis.get("final_recommendation", "")
    if rec:
        st.markdown("#### Recommendation")
        st.markdown(f"<div class='rec-block'>{rec}</div>", unsafe_allow_html=True)


def render_requirement_coverage(requirements: list) -> None:
    """Render per-requirement coverage table with evidence cards."""
    LABEL_COLOURS = {
        "strong":  ("#052e16", "#4ade80", "#166534"),
        "partial": ("#2d1a0a", "#fb923c", "#7c2d12"),
        "weak":    ("#1a1a0a", "#fbbf24", "#713f12"),
        "absent":  ("#2d0a0a", "#f87171", "#7f1d1d"),
    }
    LABEL_ICONS = {"strong": "●", "partial": "◐", "weak": "○", "absent": "✗"}

    for req in requirements:
        bg, fg, border = LABEL_COLOURS.get(req.coverage_label, LABEL_COLOURS["absent"])
        icon = LABEL_ICONS.get(req.coverage_label, "?")
        emphasis_bar = round(req.jd_emphasis * 100)
        coverage_bar = round(req.coverage_score * 100)

        evidence_snippet = req.best_match_text[:180].replace("<","&lt;").replace(">","&gt;")
        if len(req.best_match_text) > 180:
            evidence_snippet += "…"

        st.markdown(f"""
        <div style='background:#13131d;border:1px solid {border};border-left:3px solid {fg};
                    border-radius:0 10px 10px 0;padding:1rem 1.2rem;margin-bottom:0.7rem;'>
          <div style='display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;'>
            <div style='flex:1;'>
              <div style='color:#e2e2f0;font-size:0.87rem;font-weight:600;margin-bottom:0.5rem;'>
                {req.requirement[:140]}
              </div>
              <div style='display:flex;gap:1.5rem;font-size:0.78rem;color:#6b7280;margin-bottom:0.5rem;'>
                <span>JD emphasis
                  <span style='display:inline-block;width:60px;height:3px;background:#1e1e2e;
                               border-radius:2px;vertical-align:middle;margin:0 4px;'>
                    <span style='display:block;width:{emphasis_bar}%;height:3px;
                                 background:#7c3aed;border-radius:2px;'></span>
                  </span>{emphasis_bar}%
                </span>
                <span>Semantic coverage
                  <span style='display:inline-block;width:60px;height:3px;background:#1e1e2e;
                               border-radius:2px;vertical-align:middle;margin:0 4px;'>
                    <span style='display:block;width:{coverage_bar}%;height:3px;
                                 background:{fg};border-radius:2px;'></span>
                  </span>{coverage_bar}%
                </span>
                <span style='color:#4b5563;'>via <em>{req.best_match_section}</em></span>
              </div>
              <div style='background:#0d0d0f;border-radius:6px;padding:0.5rem 0.7rem;
                          font-family:"DM Mono",monospace;font-size:0.75rem;color:#6b7280;
                          border:1px solid #1e1e2e;'>
                {evidence_snippet}
              </div>
            </div>
            <div style='background:{bg};color:{fg};border:1px solid {border};border-radius:999px;
                        padding:3px 10px;font-size:0.72rem;font-weight:600;white-space:nowrap;
                        font-family:"DM Mono",monospace;'>
              {icon} {req.coverage_label.upper()}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


def render_gap_clusters(clusters: list) -> None:
    """Render missing tools grouped by category with priority labels."""
    PRIORITY_STYLES = {
        "critical":       ("#2d0a0a", "#f87171", "#7f1d1d"),
        "important":      ("#2d1a0a", "#fb923c", "#7c2d12"),
        "nice_to_have":   ("#1e1b4b", "#a5b4fc", "#3730a3"),
    }
    for cluster in clusters:
        bg, fg, border = PRIORITY_STYLES.get(cluster.priority, PRIORITY_STYLES["nice_to_have"])
        missing_chips = " ".join(
            f"<span style='background:#2d0a0a;color:#f87171;border:1px solid #7f1d1d;"
            f"border-radius:6px;padding:2px 9px;font-family:DM Mono,monospace;"
            f"font-size:0.72rem;margin:2px;display:inline-block;'>{t}</span>"
            for t in cluster.missing_tools
        )
        alt_chips = ""
        if cluster.has_alternatives:
            alt_chips = " ".join(
                f"<span style='background:#052e16;color:#4ade80;border:1px solid #166534;"
                f"border-radius:6px;padding:2px 9px;font-family:DM Mono,monospace;"
                f"font-size:0.72rem;margin:2px;display:inline-block;'>{t}</span>"
                for t in cluster.has_alternatives
            )

        st.markdown(f"""
        <div style='background:#13131d;border:1px solid {border};border-radius:10px;
                    padding:1rem 1.2rem;margin-bottom:0.8rem;'>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;'>
            <span style='font-weight:600;color:#e2e2f0;'>{cluster.category}</span>
            <span style='background:{bg};color:{fg};border:1px solid {border};border-radius:999px;
                         padding:2px 10px;font-size:0.7rem;font-weight:600;
                         font-family:"DM Mono",monospace;text-transform:uppercase;'>
              {cluster.priority.replace("_"," ")}
            </span>
          </div>
          <div style='margin-bottom:{"0.5rem" if alt_chips else "0"};'>{missing_chips}</div>
          {"<div style='font-size:0.75rem;color:#6b7280;margin-top:0.4rem;'>You have: " + alt_chips + "</div>" if alt_chips else ""}
        </div>
        """, unsafe_allow_html=True)


def render_section_heatmap(heatmap: list) -> None:
    """Render per-section coverage heatmap as stacked bars."""
    import plotly.graph_objects as go

    sections = [h.section.title()[:20] for h in heatmap]
    strong  = [h.strong_count  for h in heatmap]
    partial = [h.partial_count for h in heatmap]
    weak    = [h.weak_count    for h in heatmap]
    absent  = [h.absent_count  for h in heatmap]

    fig = go.Figure()
    for name, vals, colour in [
        ("Strong",  strong,  "#4ade80"),
        ("Partial", partial, "#fb923c"),
        ("Weak",    weak,    "#fbbf24"),
        ("Absent",  absent,  "#f87171"),
    ]:
        fig.add_trace(go.Bar(
            name=name, x=sections, y=vals,
            marker_color=colour, opacity=0.85,
        ))

    fig.update_layout(
        barmode="stack",
        paper_bgcolor="#0d0d0f",
        plot_bgcolor="#13131d",
        font=dict(color="#c4c4d4", size=12),
        legend=dict(orientation="h", y=-0.2, bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=40, l=10, r=10),
        height=280,
        xaxis=dict(gridcolor="#1e1e2e", linecolor="#1e1e2e"),
        yaxis=dict(gridcolor="#1e1e2e", linecolor="#1e1e2e", title="Requirements"),
    )
    st.plotly_chart(fig, use_container_width=True)



# ── Main analysis flow ────────────────────────────────────────────────────────

if analyse:
    if not uploaded_file:
        st.warning("Upload your resume PDF.")
        st.stop()
    if not jd_text.strip():
        st.warning("Paste a job description.")
        st.stop()

    progress = st.progress(0, text="Loading models...")

    with st.spinner(""):
        bi_encoder, cross_encoder = load_models()
    progress.progress(15, text="Extracting resume text...")

    resume_text, extract_method = extract_text_from_pdf(uploaded_file)
    if not resume_text:
        st.error("Could not extract text. Try a non-scanned PDF or improve scan quality.")
        st.stop()
    if extract_method == "ocr":
        st.info("🔍 Scanned PDF — OCR used. Accuracy depends on scan quality.")

    sections = parse_sections(resume_text)
    chunks = chunk_resume(sections)
    progress.progress(30, text="Building entity profiles...")

    resume_profile = build_entity_profile(resume_text)
    jd_profile = build_entity_profile(jd_text)
    progress.progress(40, text="Embedding resume chunks...")

    chunk_texts = [c["text"] for c in chunks]
    chunk_embeddings = bi_encoder.encode(
        chunk_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    progress.progress(50, text="Building ChromaDB collection...")

    collection = build_collection(chunks, chunk_embeddings)

    jd_embedding = bi_encoder.encode(
        [jd_text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )[0]
    progress.progress(60, text="Computing 5 signals...")

    result = compute_all_signals(
        chunks, resume_text, jd_text, jd_embedding,
        resume_profile, jd_profile, bi_encoder, cross_encoder,
        collection=collection,
    )
    progress.progress(70, text="Running gap analysis...")

    gap_result = run_gap_analysis(
        jd_text=jd_text,
        resume_text=resume_text,
        collection=collection,
        bi_encoder=bi_encoder,
        tool_overlap=result.tool_overlap,
    )
    progress.progress(82, text="Running LLM deep analysis...")

    llm_analysis = {}
    try:
        llm_analysis = analyse_with_llm(resume_text, jd_text)
    except Exception as e:
        st.warning(f"LLM analysis skipped: {e}")

    progress.progress(100, text="Done.")
    progress.empty()

    # ── Panel 1: Score hero ───────────────────────────────────────────────────
    verdict = llm_analysis.get("overall_verdict", "")
    verdict_reason = llm_analysis.get("verdict_reasoning", "")
    v_cls = verdict_class(verdict)
    score_colour_val = score_colour(result.final_score)

    st.markdown(f"""
    <div class='score-hero'>
      <div class='score-number'>{result.final_score}</div>
      <div class='score-label'>Overall fit score / 100</div>
      {'<div class="verdict-badge ' + v_cls + '">' + verdict + '</div>' if verdict else ''}
      {'<p style="color:#9ca3af;font-size:0.85rem;margin-top:0.8rem;max-width:600px;margin-left:auto;margin-right:auto;">' + verdict_reason + '</p>' if verdict_reason else ''}
    </div>
    """, unsafe_allow_html=True)

    # Mini signal metrics
    signal_labels = {
        "section_semantic": "Section semantic",
        "cross_encoder": "Contextual match",
        "tool_f1": "Tool F1",
        "domain_coverage": "Domain coverage",
        "experience_fit": "Experience fit",
    }
    cols = st.columns(5)
    for col, (k, label) in zip(cols, signal_labels.items()):
        val = result.signal_scores.get(k, 0)
        c = score_colour(val)
        col.markdown(f"""
        <div class='metric-mini'>
          <div class='metric-mini-val' style='color:{c};'>{val:.0f}</div>
          <div class='metric-mini-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Panel 2: Radar + Section breakdown ───────────────────────────────────
    st.markdown("### Signal breakdown")
    col_radar, col_sections = st.columns([1, 1], gap="large")

    with col_radar:
        render_radar(result.signal_scores)

        # Experience alignment
        exp = result.exp_alignment
        level_fit = exp.get("level_fit", "unknown").replace("_", " ").title()
        years_fit = exp.get("years_fit", "unknown").replace("_", " ")
        st.markdown(f"""
        <div class='section-card' style='margin-top:1rem;'>
          <div class='section-card-title'>Experience alignment</div>
          <div style='display:flex;flex-direction:column;gap:0.4rem;font-size:0.85rem;'>
            <div><span style='color:#6b7280;'>Level fit:</span> <span style='color:#e2e2f0;'>{level_fit}</span></div>
            <div><span style='color:#6b7280;'>JD requires:</span> <span style='color:#e2e2f0;'>{exp.get("jd_seniority","—").title()} · {exp.get("jd_years_required",0) or "—"} yrs</span></div>
            <div><span style='color:#6b7280;'>Resume shows:</span> <span style='color:#e2e2f0;'>{exp.get("resume_seniority","—").title()} · {exp.get("resume_years_inferred",0) or "—"} yrs</span></div>
            <div><span style='color:#6b7280;'>Years fit:</span> <span style='color:#e2e2f0;'>{years_fit}</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_sections:
        st.markdown("**Section-wise semantic similarity**")
        render_section_bars(result.section_semantic_breakdown)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Panel 3: Tool & domain analysis ──────────────────────────────────────
    st.markdown("### Tool & domain analysis")
    col_tools, col_domain = st.columns([3, 2], gap="large")

    with col_tools:
        render_tool_chips(result.tool_overlap)
        tool_stats = result.tool_overlap
        st.markdown(f"""
        <div style='display:flex;gap:1rem;margin-top:1rem;'>
          <div class='metric-mini' style='flex:1;'>
            <div class='metric-mini-val' style='color:#4ade80;'>{len(tool_stats["matched"])}</div>
            <div class='metric-mini-label'>Matched</div>
          </div>
          <div class='metric-mini' style='flex:1;'>
            <div class='metric-mini-val' style='color:#f87171;'>{len(tool_stats["missing"])}</div>
            <div class='metric-mini-label'>Missing</div>
          </div>
          <div class='metric-mini' style='flex:1;'>
            <div class='metric-mini-val' style='color:#a5b4fc;'>{len(tool_stats["bonus"])}</div>
            <div class='metric-mini-label'>Bonus</div>
          </div>
          <div class='metric-mini' style='flex:1;'>
            <div class='metric-mini-val' style='color:#e2e2f0;'>{round(tool_stats["f1"]*100)}</div>
            <div class='metric-mini-label'>F1 score</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_domain:
        dom_overlap = result.domain_overlap
        matched_domains = dom_overlap.get("matched_domains", [])
        missing_domains = dom_overlap.get("missing_domains", [])
        domain_display = {
            "healthcare_ai": "Healthcare AI",
            "finance_ai": "Finance AI",
            "nlp_llm": "NLP / LLM",
            "computer_vision": "Computer Vision",
            "mlops_platform": "MLOps / Platform",
            "data_engineering": "Data Engineering",
        }
        st.markdown("**Domain alignment**")
        for d in matched_domains:
            st.markdown(f"<span class='tool-chip tool-match'>{domain_display.get(d, d)}</span>", unsafe_allow_html=True)
        for d in missing_domains:
            st.markdown(f"<span class='tool-chip tool-missing'>{domain_display.get(d, d)}</span>", unsafe_allow_html=True)

        domain_info = llm_analysis.get("domain_alignment", "")
        cultural = llm_analysis.get("cultural_signals", "")
        if domain_info:
            st.markdown(f"<p style='color:#9ca3af;font-size:0.83rem;margin-top:1rem;'>{domain_info}</p>", unsafe_allow_html=True)
        if cultural:
            st.markdown(f"<p style='color:#9ca3af;font-size:0.83rem;'>{cultural}</p>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Panel 4: Requirement coverage ─────────────────────────────────────────
    st.markdown("### Requirement coverage")
    st.markdown(
        "<p style='color:#6b7280;font-size:0.85rem;'>Each JD requirement mapped to the best matching resume chunk via ChromaDB.</p>",
        unsafe_allow_html=True,
    )
    total_reqs = len(gap_result.requirements)
    strong_n  = sum(1 for r in gap_result.requirements if r.coverage_label == "strong")
    partial_n = sum(1 for r in gap_result.requirements if r.coverage_label == "partial")
    weak_n    = sum(1 for r in gap_result.requirements if r.coverage_label == "weak")
    absent_n  = sum(1 for r in gap_result.requirements if r.coverage_label == "absent")

    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    for col, label, val, colour in [
        (rc1, "Total reqs", total_reqs, "#e2e2f0"),
        (rc2, "Strong",     strong_n,   "#4ade80"),
        (rc3, "Partial",    partial_n,  "#fb923c"),
        (rc4, "Weak",       weak_n,     "#fbbf24"),
        (rc5, "Absent",     absent_n,   "#f87171"),
    ]:
        col.markdown(
            f"<div class='metric-mini'><div class='metric-mini-val' style='color:{colour};'>{val}</div>"
            f"<div class='metric-mini-label'>{label}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    col_heat, col_reqs = st.columns([1, 2], gap="large")
    with col_heat:
        st.markdown("**Section coverage heatmap**")
        render_section_heatmap(gap_result.section_heatmap)
    with col_reqs:
        fc1, fc2 = st.columns(2)
        with fc1:
            show_label = st.selectbox("Filter by coverage", ["All", "Absent", "Weak", "Partial", "Strong"], key="cov_filter")
        with fc2:
            sort_by = st.selectbox("Sort by", ["JD emphasis", "Coverage score"], key="req_sort")
        filtered = gap_result.requirements
        if show_label != "All":
            filtered = [r for r in filtered if r.coverage_label == show_label.lower()]
        if sort_by == "Coverage score":
            filtered = sorted(filtered, key=lambda r: r.coverage_score)
        else:
            filtered = sorted(filtered, key=lambda r: -r.jd_emphasis)
        render_requirement_coverage(filtered[:15])

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Panel 5: Gap clusters ──────────────────────────────────────────────────
    st.markdown("### Skill gap clusters")
    st.markdown(
        "<p style='color:#6b7280;font-size:0.85rem;'>Missing tools by category with priority and resume alternatives.</p>",
        unsafe_allow_html=True,
    )
    if gap_result.clusters:
        gc1, gc2 = st.columns(2, gap="large")
        half = (len(gap_result.clusters) + 1) // 2
        with gc1:
            render_gap_clusters(gap_result.clusters[:half])
        with gc2:
            render_gap_clusters(gap_result.clusters[half:])
    else:
        st.success("No significant tool gaps — your resume covers the JD tool requirements well.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Panel 6: LLM deep analysis ────────────────────────────────────────────
    if llm_analysis:
        st.markdown("### Deep analysis")
        render_llm_analysis(llm_analysis)
    else:
        st.info("Add a Groq API key to unlock LLM-powered deep analysis.")

    # Debug expander
    with st.expander("Debug — resume sections"):
        for sec, content in sections.items():
            st.markdown(f"**{sec.title()}** ({len(content.split())} words)")
            st.text(content[:300] + ("..." if len(content) > 300 else ""))