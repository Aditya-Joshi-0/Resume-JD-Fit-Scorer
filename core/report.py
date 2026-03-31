"""
Markdown report generator — produces a downloadable .md file
summarising the full analysis for a given resume + JD pair.
"""
from datetime import datetime


def _section_bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for markdown."""
    filled = round(score / 100 * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {score:.0f}/100"


def _label_icon(label: str) -> str:
    return {"strong": "✅", "partial": "⚠️", "weak": "🔸", "absent": "❌",
            "met": "✅", "not_met": "❌", "high": "🔴", "medium": "🟡", "low": "🟢"}.get(label, "•")


def generate_report(
    result,           # ScoringResult
    gap_result,       # GapAnalysisResult
    llm_analysis: dict,
    rewrites: dict,
    interview_prep: dict,
    jd_snippet: str = "",
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# Resume × JD Fit Report",
        f"Generated: {now}",
        "",
        "---",
        "",
    ]

    # ── Overall score ─────────────────────────────────────────────────────────
    verdict = llm_analysis.get("overall_verdict", "—")
    reasoning = llm_analysis.get("verdict_reasoning", "")
    lines += [
        f"## Overall Score: {result.final_score} / 100  —  {verdict}",
        "",
        f"> {reasoning}",
        "",
    ]

    # ── Signal breakdown ──────────────────────────────────────────────────────
    lines += ["## Signal Breakdown", ""]
    label_map = {
        "section_semantic": "Section semantic similarity",
        "cross_encoder":    "Cross-encoder contextual match",
        "tool_f1":          "Tool entity F1",
        "domain_coverage":  "Domain coverage",
        "experience_fit":   "Experience level fit",
    }
    for k, label in label_map.items():
        val = result.signal_scores.get(k, 0)
        lines.append(f"- **{label}**: {_section_bar(val)}")
    lines.append("")

    # ── Section semantic breakdown ────────────────────────────────────────────
    lines += ["## Section-wise Semantic Similarity", ""]
    for sec, score in sorted(result.section_semantic_breakdown.items(), key=lambda x: -x[1]):
        lines.append(f"- **{sec.title()}**: {_section_bar(score)}")
    lines.append("")

    # ── Tool analysis ─────────────────────────────────────────────────────────
    to = result.tool_overlap
    lines += [
        "## Tool Analysis",
        "",
        f"- **Matched ({len(to['matched'])})**: {', '.join(f'`{t}`' for t in to['matched']) or '—'}",
        f"- **Missing ({len(to['missing'])})**: {', '.join(f'`{t}`' for t in to['missing']) or '—'}",
        f"- **Bonus skills ({len(to['bonus'])})**: {', '.join(f'`{t}`' for t in to['bonus'][:10]) or '—'}",
        f"- **Tool F1 score**: {round(to['f1'] * 100)}%",
        "",
    ]

    # ── Experience alignment ──────────────────────────────────────────────────
    exp = result.exp_alignment
    lines += [
        "## Experience Alignment",
        "",
        f"- JD seniority: `{exp.get('jd_seniority','—')}`  |  Resume: `{exp.get('resume_seniority','—')}`",
        f"- Level fit: `{exp.get('level_fit','—').replace('_',' ')}`",
        f"- JD years required: `{exp.get('jd_years_required', '—')}`  |  Resume: `{exp.get('resume_years_inferred', '—')}`",
        f"- Years fit: `{exp.get('years_fit','—').replace('_',' ')}`",
        "",
    ]

    # ── Requirement coverage ──────────────────────────────────────────────────
    lines += ["## Requirement Coverage", ""]
    total = len(gap_result.requirements)
    strong  = sum(1 for r in gap_result.requirements if r.coverage_label == "strong")
    partial = sum(1 for r in gap_result.requirements if r.coverage_label == "partial")
    weak    = sum(1 for r in gap_result.requirements if r.coverage_label == "weak")
    absent  = sum(1 for r in gap_result.requirements if r.coverage_label == "absent")
    lines += [
        f"| Status  | Count | % |",
        f"|---------|-------|---|",
        f"| ✅ Strong  | {strong}  | {round(strong/total*100) if total else 0}% |",
        f"| ⚠️ Partial | {partial} | {round(partial/total*100) if total else 0}% |",
        f"| 🔸 Weak    | {weak}    | {round(weak/total*100) if total else 0}% |",
        f"| ❌ Absent  | {absent}  | {round(absent/total*100) if total else 0}% |",
        "",
    ]

    # Top absent/weak requirements
    critical = [r for r in gap_result.requirements if r.coverage_label in ("absent", "weak") and r.jd_emphasis > 0.4]
    if critical:
        lines += ["### Critical unmet requirements", ""]
        for r in critical[:6]:
            lines.append(f"- {_label_icon(r.coverage_label)} **{r.requirement[:100]}**")
            lines.append(f"  - Best match ({round(r.coverage_score*100)}%): *{r.best_match_text[:120]}...*")
        lines.append("")

    # ── Gap clusters ──────────────────────────────────────────────────────────
    if gap_result.clusters:
        lines += ["## Skill Gap Clusters", ""]
        for c in gap_result.clusters:
            icon = {"critical": "🔴", "important": "🟡", "nice_to_have": "🟢"}.get(c.priority, "•")
            lines.append(f"### {icon} {c.category} ({c.priority.replace('_',' ')})")
            lines.append(f"- Missing: {', '.join(f'`{t}`' for t in c.missing_tools)}")
            if c.has_alternatives:
                lines.append(f"- You have: {', '.join(f'`{t}`' for t in c.has_alternatives)}")
            lines.append("")

    # ── LLM deep analysis ─────────────────────────────────────────────────────
    if llm_analysis:
        lines += ["## Deep Analysis (LLM)", ""]

        strengths = llm_analysis.get("top_strengths", [])
        if strengths:
            lines += ["### Top Strengths", ""]
            for s in strengths:
                lines.append(f"**{s.get('strength','')}**")
                lines.append(f"> {s.get('evidence','')} — *{s.get('relevance','')}*")
                lines.append("")

        gaps = llm_analysis.get("priority_gaps", [])
        if gaps:
            lines += ["### Priority Gaps", ""]
            for g in gaps:
                icon = _label_icon(g.get("severity", "medium"))
                lines.append(f"{icon} **{g.get('gap','')}** ({g.get('severity','')})")
                lines.append(f"- Impact: {g.get('impact','')}")
                lines.append(f"- Fix: *{g.get('fix','')}*")
                lines.append("")

        reqs = llm_analysis.get("hard_requirements_check", [])
        if reqs:
            lines += ["### Hard Requirements Check", ""]
            for r in reqs:
                icon = _label_icon(r.get("status","partial"))
                lines.append(f"- {icon} **{r.get('requirement','')}** — {r.get('evidence','')}")
            lines.append("")

        rec = llm_analysis.get("final_recommendation", "")
        if rec:
            lines += ["### Final Recommendation", "", f"> {rec}", ""]

    # ── Resume rewrite suggestions ────────────────────────────────────────────
    if rewrites:
        lines += ["## Resume Rewrite Suggestions", ""]

        summary_rw = rewrites.get("summary_rewrite", {})
        if summary_rw:
            lines += [
                "### Summary rewrite",
                "",
                "**Original:**",
                f"> {summary_rw.get('original', '—')}",
                "",
                "**Suggested:**",
                f"> {summary_rw.get('rewritten', '—')}",
                "",
                f"*Why: {summary_rw.get('why', '')}*",
                "",
            ]

        for rw in rewrites.get("rewrite_suggestions", []):
            lines += [
                f"### {rw.get('section','Section')} — suggested rewrite",
                "",
                "**Original:**",
                f"> {rw.get('original_snippet','—')}",
                "",
                "**Rewritten:**",
                f"> {rw.get('rewritten','—')}",
                "",
                f"*Why: {rw.get('why','')}*",
            ]
            kws = rw.get("keywords_added", [])
            if kws:
                lines.append(f"*Keywords added: {', '.join(f'`{k}`' for k in kws)}*")
            lines.append("")

        quick_wins = rewrites.get("quick_wins", [])
        if quick_wins:
            lines += ["### Quick wins (< 5 min each)", ""]
            for qw in quick_wins:
                lines.append(f"- {qw}")
            lines.append("")

    # ── Interview prep ────────────────────────────────────────────────────────
    if interview_prep:
        lines += ["## Interview Prep", ""]

        for q in interview_prep.get("likely_questions", []):
            cat = q.get("category", "")
            lines += [
                f"### [{cat}] {q.get('question','')}",
                "",
                f"*Why asked: {q.get('why_asked','')}*",
                "",
                "**Talking points:**",
            ]
            for tp in q.get("talking_points", []):
                lines.append(f"- {tp}")
            wo = q.get("watch_out", "")
            if wo:
                lines.append(f"\n⚠️ Watch out: *{wo}*")
            lines.append("")

        sd = interview_prep.get("system_design_topic", {})
        if sd:
            lines += [
                f"### System Design: {sd.get('topic','')}",
                "",
            ]
            for pt in sd.get("suggested_approach", []):
                lines.append(f"- {pt}")
            lines.append("")

        q2ask = interview_prep.get("questions_to_ask_interviewer", [])
        if q2ask:
            lines += ["### Questions to ask the interviewer", ""]
            for q in q2ask:
                lines.append(f"- {q}")
            lines.append("")

    lines += ["---", f"*Generated by Fit Scorer · {now}*"]
    return "\n".join(lines)