"""
Groq API integration for deep, structured resume-JD analysis.
Uses Llama 3.3-70B via Groq's free tier.
Set GROQ_API_KEY in Streamlit secrets or as env variable.
"""
import os
import json
import re
import streamlit as st
from groq import Groq
from core.config import Settings

settings = Settings.from_env()
MODEL = settings.llm_model


@st.cache_resource(show_spinner=False)
def get_groq_client() -> Groq:
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to .streamlit/secrets.toml or as an environment variable."
        )
    return Groq(api_key=api_key)


SYSTEM_PROMPT = """You are a senior technical recruiter and career coach specialising in AI/ML roles.
You analyse resumes against job descriptions and return ONLY valid JSON — no preamble, no markdown fences.

Your analysis must be specific, evidence-based, and actionable. Never give generic advice.
Always reference specific skills, tools, or experiences from the actual resume and JD provided."""


ANALYSIS_PROMPT = """Analyse this resume against the job description below and return a JSON object with exactly this structure:

{{
  "overall_verdict": "Strong match" | "Good match" | "Moderate match" | "Weak match",
  "verdict_reasoning": "<2 sentence explanation referencing specific resume content and JD requirements>",

  "section_analysis": {{
    "skills": {{
      "score": <0-100>,
      "status": "strong" | "partial" | "weak",
      "commentary": "<specific observation about the skills section>",
      "highlights": ["<specific skill or tool that matches well>", ...],
      "gaps": ["<specific missing skill or tool>", ...]
    }},
    "experience": {{
      "score": <0-100>,
      "status": "strong" | "partial" | "weak",
      "commentary": "<specific observation about work experience>",
      "highlights": ["<specific role, project, or achievement that maps well>", ...],
      "gaps": ["<specific experience gap>", ...]
    }},
    "projects": {{
      "score": <0-100>,
      "status": "strong" | "partial" | "weak",
      "commentary": "<specific observation about projects>",
      "highlights": ["<project that demonstrates required skills>", ...],
      "gaps": ["<type of project that would strengthen the application>", ...]
    }},
    "education": {{
      "score": <0-100>,
      "status": "strong" | "partial" | "weak",
      "commentary": "<assessment of educational background against requirements>"
    }}
  }},

  "hard_requirements_check": [
    {{
      "requirement": "<specific requirement from JD>",
      "status": "met" | "partial" | "not_met",
      "evidence": "<specific line or detail from resume that addresses this, or explains the gap>"
    }}
  ],

  "top_strengths": [
    {{
      "strength": "<specific strength>",
      "evidence": "<concrete example from resume>",
      "relevance": "<why this matters for this specific role>"
    }}
  ],

  "priority_gaps": [
    {{
      "gap": "<specific gap>",
      "severity": "high" | "medium" | "low",
      "impact": "<what this gap means for candidacy>",
      "fix": "<concrete, specific action to address this gap>"
    }}
  ],

  "experience_level_fit": "overqualified" | "strong" | "slight_overqualified" | "slight_underqualified" | "underqualified",
  "domain_alignment": "<1 sentence on how well the candidate's domain expertise aligns>",
  "cultural_signals": "<1 sentence on soft signals — communication style, scope of ownership, etc.>",
  "final_recommendation": "<3-4 sentence overall recommendation — what to emphasise in the application, what to prepare for in interviews>"
}}

Return only the JSON. No explanation outside it.

---
RESUME:
{resume}

---
JOB DESCRIPTION:
{jd}
"""


def _clean_json_response(text: str) -> str:
    """Strip markdown fences or stray text around the JSON blob."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Find the outermost { } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return text[start:end]
    return text


def analyse_with_llm(
    resume_text: str,
    jd_text: str,
    max_resume_chars: int = 6000,
    max_jd_chars: int = 3000,
) -> dict:
    """
    Run deep structured analysis via Groq/Llama 3.3.
    Returns parsed dict. Raises on API error.
    """
    client = get_groq_client()

    # Truncate to stay within context limits
    resume_truncated = resume_text[:max_resume_chars]
    jd_truncated = jd_text[:max_jd_chars]

    prompt = ANALYSIS_PROMPT.format(
        resume=resume_truncated,
        jd=jd_truncated,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,    # Low temp for consistent structured output
        max_tokens=3000,
    )

    raw = response.choices[0].message.content
    cleaned = _clean_json_response(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Graceful degradation — return a minimal structure with the raw text
        return {
            "overall_verdict": "Analysis incomplete",
            "verdict_reasoning": raw[:500],
            "section_analysis": {},
            "hard_requirements_check": [],
            "top_strengths": [],
            "priority_gaps": [],
            "experience_level_fit": "unknown",
            "domain_alignment": "",
            "cultural_signals": "",
            "final_recommendation": raw[:500],
        }
