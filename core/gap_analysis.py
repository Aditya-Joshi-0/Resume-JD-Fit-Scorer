"""
Phase 2 — Deep gap analysis.

Workflow:
1. Decompose JD into individual requirement sentences.
2. For each requirement, query ChromaDB for the best-matching resume chunk.
3. Score coverage: strong / partial / weak / absent.
4. Build section-gap heatmap: per section, which requirements are covered.
5. Cluster missing tools by category with priority labels.
6. Rank requirements by JD emphasis (frequency + early-mention bonus).
"""
import re
import numpy as np
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer

from core.store import query_top_k, query_all_sections
from core.entities import TOOLS_AND_FRAMEWORKS, DOMAIN_TERMS

# ── Coverage thresholds ───────────────────────────────────────────────────────

COVERAGE_THRESHOLDS = {
    "strong":  0.72,
    "partial": 0.52,
    "weak":    0.38,
    # below weak → "absent"
}

# ── Tool category map ─────────────────────────────────────────────────────────

TOOL_CATEGORIES = {
    "LLM / Generative AI": [
        "llm", "gpt", "gpt-4", "gpt-3", "claude", "gemini", "llama", "mistral",
        "falcon", "chatgpt", "openai", "anthropic", "cohere", "groq", "t5", "bart",
    ],
    "Fine-tuning": [
        "qlora", "lora", "peft", "rlhf", "dpo", "instruction tuning",
        "fine-tuning", "finetuning",
    ],
    "RAG / Vector DB": [
        "rag", "retrieval augmented", "langchain", "llamaindex", "haystack",
        "milvus", "chromadb", "pinecone", "weaviate", "qdrant", "faiss", "pgvector",
        "semantic kernel",
    ],
    "Model Serving": [
        "vllm", "triton", "torchserve", "bentoml", "ray serve", "seldon",
    ],
    "Web / API": [
        "fastapi", "flask", "django", "grpc", "celery", "rest api", "graphql",
        "microservices",
    ],
    "ML Frameworks": [
        "pytorch", "tensorflow", "keras", "jax", "flax", "scikit-learn", "sklearn",
        "xgboost", "lightgbm", "catboost", "hugging face", "transformers",
    ],
    "NLP": [
        "spacy", "nltk", "gensim", "bert", "roberta", "distilbert", "electra",
        "deberta", "sentence-transformers", "sbert",
    ],
    "Computer Vision / OCR": [
        "opencv", "paddleocr", "tesseract", "easyocr", "detectron2",
        "yolo", "yolov8", "clip", "dino", "sam",
    ],
    "MLOps": [
        "mlflow", "wandb", "weights and biases", "neptune", "comet",
        "airflow", "prefect", "dagster", "kubeflow", "dvc",
    ],
    "Infrastructure / Cloud": [
        "docker", "kubernetes", "k8s", "helm", "terraform",
        "aws", "gcp", "azure", "sagemaker", "vertex ai", "databricks",
    ],
    "Data Engineering": [
        "spark", "pyspark", "kafka", "flink", "dbt", "pandas", "numpy",
        "polars", "dask", "postgresql", "mysql", "mongodb", "redis",
        "elasticsearch", "bigquery", "redshift",
    ],
    "Languages": [
        "python", "java", "scala", "go", "rust", "c++", "r", "julia",
        "javascript", "typescript", "sql",
    ],
}

# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class RequirementCoverage:
    requirement:          str
    jd_emphasis:          float        # 0–1, based on position + frequency
    best_match_text:      str
    best_match_section:   str
    coverage_score:       float        # 0–1 similarity
    coverage_label:       str          # strong / partial / weak / absent

@dataclass
class GapCluster:
    category:            str
    missing_tools:       list[str]
    has_alternatives:    list[str]     # resume tools in same category
    priority:            str           # critical / important / nice_to_have

@dataclass
class SectionCoverage:
    section:             str
    requirements_count:  int
    strong_count:        int
    partial_count:       int
    weak_count:          int
    absent_count:        int
    coverage_pct:        float         # (strong + partial) / total

@dataclass
class GapAnalysisResult:
    requirements:        list[RequirementCoverage]
    clusters:            list[GapCluster]
    section_heatmap:     list[SectionCoverage]
    critical_gaps:       list[str]
    top_covered:         list[str]


# ── JD decomposition ──────────────────────────────────────────────────────────

def _split_jd_into_requirements(jd_text: str) -> list[str]:
    """
    Extract meaningful requirement phrases from JD text.
    Splits on bullet markers, newlines, and sentence boundaries.
    Filters out very short or boilerplate lines.
    """
    # Normalise bullet markers
    text = re.sub(r"[•·▪▸–—]\s*", "\n", jd_text)
    # Split on newlines and periods
    raw_lines = re.split(r"\n|(?<=[.?!])\s+", text)

    requirements = []
    for line in raw_lines:
        line = line.strip()
        # Keep lines that look like requirements: 8–200 chars, contain a verb or keyword
        if 8 < len(line) < 220:
            lower = line.lower()
            # Skip pure headers and boilerplate
            if any(skip in lower for skip in [
                "about us", "we are", "our company", "equal opportunity",
                "apply now", "send your", "job title", "location:", "salary",
            ]):
                continue
            requirements.append(line)

    return requirements


def _jd_emphasis_score(requirement: str, jd_text: str) -> float:
    """
    Score how much the JD emphasises a requirement.
    Factors: position bonus (early = more important) + frequency.
    Returns 0–1.
    """
    jd_lower = jd_text.lower()
    req_lower = requirement.lower()

    # Position: normalise to 0–1 (0 = top of JD)
    pos = jd_lower.find(req_lower[:30])
    position_score = 1.0 - (pos / max(len(jd_lower), 1)) if pos != -1 else 0.5

    # Frequency of key words from requirement in the full JD
    words = [w for w in req_lower.split() if len(w) > 4]
    freq_hits = sum(jd_lower.count(w) for w in words)
    freq_score = min(freq_hits / max(len(words) * 3, 1), 1.0)

    return round(0.5 * position_score + 0.5 * freq_score, 3)


# ── Coverage classification ───────────────────────────────────────────────────

def _coverage_label(score: float) -> str:
    if score >= COVERAGE_THRESHOLDS["strong"]:
        return "strong"
    if score >= COVERAGE_THRESHOLDS["partial"]:
        return "partial"
    if score >= COVERAGE_THRESHOLDS["weak"]:
        return "weak"
    return "absent"


# ── Tool clustering ───────────────────────────────────────────────────────────

def _cluster_missing_tools(
    missing_tools: set[str],
    resume_tools:  set[str],
) -> list[GapCluster]:
    """
    Group missing tools by category. For each category, also note which tools
    from the resume belong to the same category (alternatives/proximity).
    Assign priority based on category criticality.
    """
    PRIORITY_MAP = {
        "LLM / Generative AI":        "critical",
        "RAG / Vector DB":             "critical",
        "ML Frameworks":               "critical",
        "Languages":                   "critical",
        "Model Serving":               "important",
        "MLOps":                       "important",
        "Infrastructure / Cloud":      "important",
        "NLP":                         "important",
        "Fine-tuning":                 "important",
        "Data Engineering":            "nice_to_have",
        "Web / API":                   "nice_to_have",
        "Computer Vision / OCR":       "nice_to_have",
    }

    clusters = []
    for category, tools_in_cat in TOOL_CATEGORIES.items():
        missing_in_cat = [t for t in tools_in_cat if t in missing_tools]
        if not missing_in_cat:
            continue
        alternatives = [t for t in tools_in_cat if t in resume_tools]
        clusters.append(GapCluster(
            category=category,
            missing_tools=missing_in_cat,
            has_alternatives=alternatives,
            priority=PRIORITY_MAP.get(category, "nice_to_have"),
        ))

    # Sort: critical → important → nice_to_have, then by missing count desc
    priority_order = {"critical": 0, "important": 1, "nice_to_have": 2}
    clusters.sort(key=lambda c: (priority_order[c.priority], -len(c.missing_tools)))
    return clusters


# ── Section heatmap ───────────────────────────────────────────────────────────

def _build_section_heatmap(requirements: list[RequirementCoverage]) -> list[SectionCoverage]:
    """Aggregate coverage stats per section for all requirements."""
    section_map: dict[str, list[RequirementCoverage]] = {}
    for req in requirements:
        sec = req.best_match_section
        section_map.setdefault(sec, []).append(req)

    heatmap = []
    for sec, reqs in section_map.items():
        counts = {lbl: sum(1 for r in reqs if r.coverage_label == lbl)
                  for lbl in ("strong", "partial", "weak", "absent")}
        covered = counts["strong"] + counts["partial"]
        pct = round(covered / len(reqs) * 100, 1) if reqs else 0.0
        heatmap.append(SectionCoverage(
            section=sec,
            requirements_count=len(reqs),
            strong_count=counts["strong"],
            partial_count=counts["partial"],
            weak_count=counts["weak"],
            absent_count=counts["absent"],
            coverage_pct=pct,
        ))

    heatmap.sort(key=lambda s: -s.coverage_pct)
    return heatmap


# ── Master function ───────────────────────────────────────────────────────────

def run_gap_analysis(
    jd_text:          str,
    resume_text:      str,
    collection,                          # ChromaDB collection
    bi_encoder:       SentenceTransformer,
    tool_overlap:     dict,              # from entities.compute_tool_overlap
    max_requirements: int = 30,
) -> GapAnalysisResult:
    """
    Full gap analysis pipeline. Returns GapAnalysisResult.
    """
    # 1. Decompose JD
    raw_requirements = _split_jd_into_requirements(jd_text)
    # Deduplicate and limit
    seen = set()
    requirements = []
    for r in raw_requirements:
        key = r.lower()[:60]
        if key not in seen:
            seen.add(key)
            requirements.append(r)
        if len(requirements) >= max_requirements:
            break

    # 2. Embed all requirements in one batch
    req_embeddings = bi_encoder.encode(
        requirements,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # 3. Per-requirement ChromaDB lookup
    coverage_list: list[RequirementCoverage] = []
    for req_text, req_emb in zip(requirements, req_embeddings):
        hits = query_top_k(collection, req_emb, n_results=1)
        if not hits:
            continue
        best = hits[0]
        sim  = best["similarity"]
        coverage_list.append(RequirementCoverage(
            requirement=req_text,
            jd_emphasis=_jd_emphasis_score(req_text, jd_text),
            best_match_text=best["text"],
            best_match_section=best["section"],
            coverage_score=round(sim, 4),
            coverage_label=_coverage_label(sim),
        ))

    # Sort by emphasis desc so high-priority requirements surface first
    coverage_list.sort(key=lambda r: -r.jd_emphasis)

    # 4. Tool clustering
    missing_tools = set(tool_overlap.get("missing", []))
    resume_tools  = set(tool_overlap.get("matched", []) + tool_overlap.get("bonus", []))
    clusters = _cluster_missing_tools(missing_tools, resume_tools)

    # 5. Section heatmap
    heatmap = _build_section_heatmap(coverage_list)

    # 6. Summary lists
    critical_gaps = [
        r.requirement for r in coverage_list
        if r.coverage_label in ("absent", "weak") and r.jd_emphasis > 0.5
    ][:8]

    top_covered = [
        r.requirement for r in coverage_list
        if r.coverage_label == "strong"
    ][:6]

    return GapAnalysisResult(
        requirements=coverage_list,
        clusters=clusters,
        section_heatmap=heatmap,
        critical_gaps=critical_gaps,
        top_covered=top_covered,
    )