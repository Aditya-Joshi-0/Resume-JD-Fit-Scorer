"""
5-signal scoring system:
  Signal 1 — Section-weighted semantic similarity (bi-encoder, per section)
  Signal 2 — Cross-encoder reranking (precise pair scoring)
  Signal 3 — Tool entity alignment (precision / recall / F1)
  Signal 4 — Domain alignment
  Signal 5 — Experience level fit

Final score = weighted blend. All signals in [0, 1] before scaling.
"""
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from dataclasses import dataclass

from core.entities import (
    EntityProfile,
    compute_tool_overlap,
    compute_domain_overlap,
    compute_experience_alignment,
)

# Section weights for Signal 1
SECTION_WEIGHTS = {
    "experience": 0.35,
    "work experience": 0.35,
    "professional experience": 0.35,
    "employment": 0.35,
    "skills": 0.30,
    "technical skills": 0.30,
    "core competencies": 0.30,
    "projects": 0.20,
    "certifications": 0.08,
    "education": 0.07,
    "academic background": 0.07,
    "summary": 0.05,
    "objective": 0.05,
    "full_text": 0.15,
}
DEFAULT_SECTION_WEIGHT = 0.10

SIGNAL_WEIGHTS = {
    "section_semantic": 0.30,
    "cross_encoder":    0.25,
    "tool_f1":          0.25,
    "domain_coverage":  0.12,
    "experience_fit":   0.08,
}

EXPERIENCE_FIT_SCORES = {
    "strong": 1.0,
    "slight_overqualified": 0.85,
    "slight_underqualified": 0.75,
    "overqualified": 0.60,
    "underqualified": 0.45,
    "unknown": 0.65,
}


@st.cache_resource(show_spinner=False)
def load_models():
    bi = SentenceTransformer("all-MiniLM-L6-v2")
    cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    return bi, cross


def signal_section_semantic(chunks, jd_embedding, bi_encoder):
    texts = [c["text"] for c in chunks]
    embeddings = bi_encoder.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    sims = embeddings @ jd_embedding

    section_scores = {}
    for chunk, sim in zip(chunks, sims):
        sec = chunk["section"]
        section_scores.setdefault(sec, []).append(float(sim))

    section_max = {sec: max(scores) for sec, scores in section_scores.items()}

    total_weight = 0.0
    weighted_sum = 0.0
    for sec, score in section_max.items():
        w = SECTION_WEIGHTS.get(sec, DEFAULT_SECTION_WEIGHT)
        weighted_sum += w * score
        total_weight += w

    blended = weighted_sum / total_weight if total_weight > 0 else 0.0
    return float(np.clip(blended, 0, 1)), section_max


def signal_cross_encoder(chunks, jd_text, bi_sims, cross_encoder, top_k=6):
    texts = [c["text"] for c in chunks]
    top_indices = np.argsort(bi_sims)[::-1][:top_k]
    top_texts = [texts[i] for i in top_indices]

    pairs = [(jd_text[:512], chunk[:512]) for chunk in top_texts]
    scores = cross_encoder.predict(pairs)

    sigmoid_scores = 1 / (1 + np.exp(-np.array(scores)))
    return float(np.max(sigmoid_scores))


@dataclass
class ScoringResult:
    final_score: float
    signal_scores: dict
    section_semantic_breakdown: dict
    tool_overlap: dict
    domain_overlap: dict
    exp_alignment: dict


def compute_all_signals(
    chunks,
    resume_text,
    jd_text,
    jd_embedding,
    resume_profile,
    jd_profile,
    bi_encoder,
    cross_encoder,
):
    s1, section_breakdown = signal_section_semantic(chunks, jd_embedding, bi_encoder)

    texts = [c["text"] for c in chunks]
    raw_embeds = bi_encoder.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    raw_sims = raw_embeds @ jd_embedding

    s2 = signal_cross_encoder(chunks, jd_text, raw_sims, cross_encoder)

    tool_overlap = compute_tool_overlap(resume_profile, jd_profile)
    domain_overlap = compute_domain_overlap(resume_profile, jd_profile)
    exp_alignment = compute_experience_alignment(resume_profile, jd_profile, resume_text)

    s3 = tool_overlap["f1"]
    s4 = domain_overlap["coverage"]
    s5 = EXPERIENCE_FIT_SCORES.get(exp_alignment["level_fit"], 0.65)

    if not jd_profile.tools:
        s3 = s1

    raw_signals = {
        "section_semantic": s1,
        "cross_encoder": s2,
        "tool_f1": s3,
        "domain_coverage": s4,
        "experience_fit": s5,
    }

    final = sum(SIGNAL_WEIGHTS[k] * v for k, v in raw_signals.items())
    final_score = round(float(np.clip(final, 0, 1)) * 100, 1)

    return ScoringResult(
        final_score=final_score,
        signal_scores={k: round(v * 100, 1) for k, v in raw_signals.items()},
        section_semantic_breakdown={k: round(v * 100, 1) for k, v in section_breakdown.items()},
        tool_overlap=tool_overlap,
        domain_overlap=domain_overlap,
        exp_alignment=exp_alignment,
    )
