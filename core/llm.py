"""
Groq API integration — three LLM chains:
  1. analyse_with_llm       — deep structured analysis (Phase 1/2)
  2. generate_rewrites      — section-level resume rewrite suggestions
  3. generate_interview_prep — likely interview Qs + talking points
"""
import os
import json
import re
import streamlit as st
from groq import Groq
from core.config import get_settings

settings = get_settings()
MODEL = settings.llm_model

@st.cache_resource(show_spinner=False)
def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to .streamlit/secrets.toml or as env var."
        )
    return Groq(api_key=api_key)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _clean_json(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    return text[start:end] if start != -1 and end > start else text


def _call(prompt: str, system: str, max_tokens: int = 3000, temp: float = 0.2) -> str:
    client = get_groq_client()
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=temp,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════════════════
# Chain 1 — Deep structured analysis
# ═══════════════════════════════════════════════════════════════════════════════

_ANALYSIS_SYSTEM = """You are a senior technical recruiter and career coach specialising in AI/ML roles.
Return ONLY valid JSON — no preamble, no markdown fences.
Be specific and evidence-based. Always reference concrete skills or experiences from the resume and JD."""

_ANALYSIS_PROMPT = """Analyse this resume against the JD and return a JSON object with exactly this structure:

{{
  "overall_verdict": "Strong match" | "Good match" | "Moderate match" | "Weak match",
  "verdict_reasoning": "<2 sentences referencing specific resume content and JD requirements>",
  "section_analysis": {{
    "skills":     {{"score":<0-100>,"status":"strong"|"partial"|"weak","commentary":"<specific>","highlights":["..."],"gaps":["..."]}},
    "experience": {{"score":<0-100>,"status":"strong"|"partial"|"weak","commentary":"<specific>","highlights":["..."],"gaps":["..."]}},
    "projects":   {{"score":<0-100>,"status":"strong"|"partial"|"weak","commentary":"<specific>","highlights":["..."],"gaps":["..."]}},
    "education":  {{"score":<0-100>,"status":"strong"|"partial"|"weak","commentary":"<specific>"}}
  }},
  "hard_requirements_check": [
    {{"requirement":"<from JD>","status":"met"|"partial"|"not_met","evidence":"<from resume>"}}
  ],
  "top_strengths": [
    {{"strength":"<specific>","evidence":"<concrete from resume>","relevance":"<why it matters for this role>"}}
  ],
  "priority_gaps": [
    {{"gap":"<specific>","severity":"high"|"medium"|"low","impact":"<candidacy impact>","fix":"<concrete action>"}}
  ],
  "experience_level_fit": "overqualified"|"strong"|"slight_overqualified"|"slight_underqualified"|"underqualified",
  "domain_alignment": "<1 sentence>",
  "cultural_signals": "<1 sentence>",
  "final_recommendation": "<3-4 sentences — what to emphasise, what to prepare for>"
}}

Return only the JSON.

RESUME:
{resume}

JOB DESCRIPTION:
{jd}
"""


def analyse_with_llm(resume_text: str, jd_text: str) -> dict:
    prompt = _ANALYSIS_PROMPT.format(
        resume=resume_text[:6000],
        jd=jd_text[:3000],
    )
    raw = _call(prompt, _ANALYSIS_SYSTEM, max_tokens=3000, temp=0.2)
    try:
        return json.loads(_clean_json(raw))
    except json.JSONDecodeError:
        return {
            "overall_verdict": "Analysis incomplete",
            "verdict_reasoning": raw[:400],
            "section_analysis": {}, "hard_requirements_check": [],
            "top_strengths": [], "priority_gaps": [],
            "experience_level_fit": "unknown", "domain_alignment": "",
            "cultural_signals": "", "final_recommendation": raw[:400],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Chain 2 — Section-level resume rewrite suggestions
# ═══════════════════════════════════════════════════════════════════════════════

_REWRITE_SYSTEM = """You are an expert technical resume writer specialising in AI/ML engineering roles.
Return ONLY valid JSON. Be specific — every suggestion must reference the actual resume content and JD.
Rewrites should be concrete, result-oriented, and use strong action verbs with metrics where possible."""

_REWRITE_PROMPT = """Given this resume and job description, generate targeted rewrite suggestions for each section.
Focus on the gaps identified below.

Return a JSON object:
{{
  "rewrite_suggestions": [
    {{
      "section": "Skills" | "Experience" | "Projects" | "Summary",
      "original_snippet": "<exact or paraphrased excerpt from the resume (2-3 lines max)>",
      "rewritten": "<improved version targeting the JD (2-4 lines)>",
      "why": "<1 sentence — what this change achieves>",
      "keywords_added": ["<keyword from JD now present in rewrite>"]
    }}
  ],
  "summary_rewrite": {{
    "original": "<current summary/objective from resume (first 3 lines)>",
    "rewritten": "<a punchy 3-sentence summary targeting this specific JD>",
    "why": "<what changed and why>"
  }},
  "quick_wins": [
    "<one-line actionable change that takes < 5 minutes — e.g. 'Add Kubernetes to skills section'>"
  ]
}}

Return only the JSON.

RESUME:
{resume}

JOB DESCRIPTION:
{jd}

KEY GAPS TO ADDRESS:
{gaps}
"""


def generate_rewrites(
    resume_text: str,
    jd_text: str,
    priority_gaps: list[dict],
) -> dict:
    gaps_text = "\n".join(
        f"- [{g.get('severity','').upper()}] {g.get('gap','')} — {g.get('fix','')}"
        for g in priority_gaps[:6]
    ) or "No specific gaps identified — focus on strengthening quantified achievements."

    prompt = _REWRITE_PROMPT.format(
        resume=resume_text[:5000],
        jd=jd_text[:2500],
        gaps=gaps_text,
    )
    raw = _call(prompt, _REWRITE_SYSTEM, max_tokens=2500, temp=0.3)
    try:
        return json.loads(_clean_json(raw))
    except json.JSONDecodeError:
        return {"rewrite_suggestions": [], "summary_rewrite": {}, "quick_wins": []}


# ═══════════════════════════════════════════════════════════════════════════════
# Chain 3 — Interview prep
# ═══════════════════════════════════════════════════════════════════════════════

_INTERVIEW_SYSTEM = """You are a senior technical interview coach for AI/ML engineering roles.
Return ONLY valid JSON. Be concrete — questions and talking points must be tailored to this specific resume and JD.
Do not generate generic questions. Every question should be answerable using the candidate's actual experience."""

_INTERVIEW_PROMPT = """Generate a targeted interview prep guide for this candidate applying to this role.

Return a JSON object:
{{
  "likely_questions": [
    {{
      "category": "Technical" | "System Design" | "Behavioural" | "Gap-related",
      "question": "<the actual interview question>",
      "why_asked": "<why this interviewer would ask this — links to resume gap or JD requirement>",
      "talking_points": [
        "<specific bullet — what from the resume to mention, how to frame it>"
      ],
      "watch_out": "<one thing to avoid saying or one common mistake for this question>"
    }}
  ],
  "system_design_topic": {{
    "topic": "<most likely system design question for this role>",
    "suggested_approach": "<3-4 bullet outline of how to structure the answer using the candidate's background>"
  }},
  "questions_to_ask_interviewer": [
    "<smart, specific question the candidate should ask that shows domain depth>"
  ]
}}

Return only the JSON.

RESUME:
{resume}

JOB DESCRIPTION:
{jd}

IDENTIFIED GAPS (prepare to address these):
{gaps}
"""


def generate_interview_prep(
    resume_text: str,
    jd_text: str,
    priority_gaps: list[dict],
) -> dict:
    gaps_text = "\n".join(
        f"- {g.get('gap','')}: {g.get('impact','')}"
        for g in priority_gaps[:5]
    ) or "No major gaps — focus on depth of LLM and MLOps experience."

    prompt = _INTERVIEW_PROMPT.format(
        resume=resume_text[:5000],
        jd=jd_text[:2500],
        gaps=gaps_text,
    )
    raw = _call(prompt, _INTERVIEW_SYSTEM, max_tokens=3000, temp=0.35)
    try:
        return json.loads(_clean_json(raw))
    except json.JSONDecodeError:
        return {"likely_questions": [], "system_design_topic": {}, "questions_to_ask_interviewer": []}