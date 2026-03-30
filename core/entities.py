"""
Structured entity extraction from resume and JD text.
Extracts: tools/frameworks, domains, seniority signals, education requirements,
          years-of-experience mentions.
"""
import re
from dataclasses import dataclass, field

# ── Entity dictionaries ────────────────────────────────────────────────────────

TOOLS_AND_FRAMEWORKS = [
    # LLMs / Generative AI
    "llm", "gpt", "gpt-4", "gpt-3", "claude", "gemini", "llama", "mistral",
    "falcon", "vicuna", "alpaca", "bloom", "t5", "bart", "chatgpt",
    "openai", "anthropic", "cohere", "together ai", "groq",
    # Fine-tuning / training
    "qlora", "lora", "peft", "rlhf", "dpo", "instruction tuning",
    "fine-tuning", "finetuning", "full fine-tuning",
    # RAG / vector
    "rag", "retrieval augmented", "langchain", "llamaindex", "haystack",
    "semantic kernel", "milvus", "chromadb", "pinecone", "weaviate",
    "qdrant", "faiss", "pgvector",
    # Deployment / serving
    "vllm", "triton", "torchserve", "bento ml", "bentoml", "ray serve",
    "fastapi", "flask", "django", "grpc", "celery",
    # ML frameworks
    "pytorch", "tensorflow", "keras", "jax", "flax", "mxnet",
    "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
    "hugging face", "transformers", "diffusers", "timm",
    # NLP specific
    "spacy", "nltk", "gensim", "stanza", "flair", "allennlp",
    "bert", "roberta", "distilbert", "xlnet", "electra", "deberta",
    "sentence-transformers", "sbert",
    # CV / OCR
    "opencv", "paddleocr", "tesseract", "easyocr", "detectron2",
    "yolo", "yolov8", "clip", "dino", "sam",
    # MLOps
    "mlflow", "wandb", "weights and biases", "neptune", "comet",
    "airflow", "prefect", "dagster", "kubeflow", "metaflow",
    "dvc", "bentoml", "seldon",
    # Infra / cloud
    "docker", "kubernetes", "k8s", "helm", "terraform",
    "aws", "gcp", "azure", "s3", "ec2", "lambda", "sagemaker",
    "vertex ai", "azure ml", "databricks", "snowflake",
    # Data
    "spark", "pyspark", "kafka", "flink", "dbt",
    "pandas", "numpy", "polars", "dask",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "bigquery", "redshift", "athena",
    # Languages
    "python", "java", "scala", "go", "rust", "c++", "r", "julia",
    "javascript", "typescript", "sql", "bash", "shell",
    # Other
    "git", "github", "gitlab", "ci/cd", "jenkins", "github actions",
    "rest api", "graphql", "grpc", "microservices",
    "minio", "supabase", "railway", "streamlit", "gradio",
]

DOMAIN_TERMS = {
    "healthcare_ai": [
        "healthcare", "clinical", "medical", "health insurance", "ehr", "emr",
        "hipaa", "fhir", "hl7", "radiology", "pathology", "genomics",
        "drug discovery", "clinical nlp", "icd", "cpt", "hedis",
        "prior authorization", "fraud waste abuse", "utilization management",
    ],
    "finance_ai": [
        "fintech", "trading", "quantitative", "risk modeling", "fraud detection",
        "credit scoring", "algorithmic trading", "portfolio", "regulatory",
        "anti money laundering", "aml", "kyc", "payments",
    ],
    "nlp_llm": [
        "natural language processing", "nlp", "text classification", "ner",
        "named entity recognition", "relation extraction", "summarization",
        "question answering", "sentiment analysis", "intent detection",
        "information extraction", "knowledge graph", "coreference",
    ],
    "computer_vision": [
        "computer vision", "image classification", "object detection",
        "segmentation", "ocr", "document ai", "video analysis",
        "face recognition", "pose estimation",
    ],
    "mlops_platform": [
        "mlops", "model deployment", "model serving", "model versioning",
        "data pipeline", "feature store", "a/b testing", "monitoring",
        "drift detection", "retraining", "scalability",
    ],
    "data_engineering": [
        "data engineering", "etl", "data pipeline", "data warehouse",
        "data lake", "data lakehouse", "streaming", "batch processing",
        "real-time", "data quality", "data governance",
    ],
}

SENIORITY_KEYWORDS = {
    "senior": ["senior", "sr.", "lead", "principal", "staff", "architect", "head of"],
    "mid": ["mid-level", "mid level", "software engineer ii", "engineer ii"],
    "junior": ["junior", "jr.", "associate", "entry level", "entry-level", "fresher"],
    "manager": ["manager", "director", "vp", "vice president", "c-level", "cto", "cdo"],
}

EDUCATION_PATTERNS = {
    "phd": [r"ph\.?d", r"doctorate", r"doctoral"],
    "masters": [r"m\.?tech", r"m\.?s\.?", r"master", r"msc", r"mba", r"m\.?e\.?"],
    "bachelors": [r"b\.?tech", r"b\.?e\.?", r"bachelor", r"bsc", r"b\.?s\.?", r"undergraduate"],
}

YEARS_PATTERN = re.compile(
    r"(\d+)\+?\s*(?:to\s*\d+\s*)?years?\s*(?:of\s*)?(?:experience|exp)",
    re.IGNORECASE,
)


# ── Dataclass for results ──────────────────────────────────────────────────────

@dataclass
class EntityProfile:
    tools: set[str] = field(default_factory=set)
    domains: dict[str, list[str]] = field(default_factory=dict)
    seniority: str = "unspecified"
    education_required: str = "unspecified"
    years_required: int = 0
    raw_years_mentions: list[int] = field(default_factory=list)


# ── Extraction functions ───────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    return text.lower()


def extract_tools(text: str) -> set[str]:
    norm = _normalise(text)
    found = set()
    for tool in TOOLS_AND_FRAMEWORKS:
        pattern = r"\b" + re.escape(tool) + r"\b"
        if re.search(pattern, norm):
            found.add(tool)
    return found


def extract_domains(text: str) -> dict[str, list[str]]:
    norm = _normalise(text)
    matched = {}
    for domain, terms in DOMAIN_TERMS.items():
        hits = [t for t in terms if re.search(r"\b" + re.escape(t) + r"\b", norm)]
        if hits:
            matched[domain] = hits
    return matched


def extract_seniority(text: str) -> str:
    norm = _normalise(text)
    for level, keywords in SENIORITY_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", norm):
                return level
    return "unspecified"


def extract_education(text: str) -> str:
    norm = _normalise(text)
    for level, patterns in EDUCATION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, norm):
                return level
    return "unspecified"


def extract_years_of_experience(text: str) -> tuple[int, list[int]]:
    """Return (max_years_mentioned, all_years_mentioned)."""
    matches = YEARS_PATTERN.findall(text)
    years = [int(m) for m in matches]
    return (max(years) if years else 0), years


def build_entity_profile(text: str) -> EntityProfile:
    years_max, years_all = extract_years_of_experience(text)
    return EntityProfile(
        tools=extract_tools(text),
        domains=extract_domains(text),
        seniority=extract_seniority(text),
        education_required=extract_education(text),
        years_required=years_max,
        raw_years_mentions=years_all,
    )


# ── Comparison helpers ─────────────────────────────────────────────────────────

def compute_tool_overlap(
    resume_profile: EntityProfile,
    jd_profile: EntityProfile,
) -> dict:
    matched = resume_profile.tools & jd_profile.tools
    missing = jd_profile.tools - resume_profile.tools
    bonus = resume_profile.tools - jd_profile.tools

    precision = len(matched) / len(resume_profile.tools) if resume_profile.tools else 0
    recall = len(matched) / len(jd_profile.tools) if jd_profile.tools else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    return {
        "matched": sorted(matched),
        "missing": sorted(missing),
        "bonus": sorted(bonus),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def compute_domain_overlap(
    resume_profile: EntityProfile,
    jd_profile: EntityProfile,
) -> dict:
    resume_domains = set(resume_profile.domains.keys())
    jd_domains = set(jd_profile.domains.keys())
    matched = resume_domains & jd_domains
    missing = jd_domains - resume_domains
    coverage = len(matched) / len(jd_domains) if jd_domains else 1.0
    return {
        "matched_domains": sorted(matched),
        "missing_domains": sorted(missing),
        "coverage": round(coverage, 3),
    }


SENIORITY_RANK = {
    "junior": 1, "mid": 2, "senior": 3, "manager": 4, "unspecified": 2
}

def compute_experience_alignment(
    resume_profile: EntityProfile,
    jd_profile: EntityProfile,
    resume_text: str,
) -> dict:
    jd_years = jd_profile.years_required
    resume_years_max, _ = extract_years_of_experience(resume_text)

    resume_rank = SENIORITY_RANK.get(resume_profile.seniority, 2)
    jd_rank = SENIORITY_RANK.get(jd_profile.seniority, 2)
    rank_diff = resume_rank - jd_rank

    if rank_diff == 0:
        level_fit = "strong"
    elif rank_diff == 1:
        level_fit = "slight_overqualified"
    elif rank_diff == -1:
        level_fit = "slight_underqualified"
    elif rank_diff >= 2:
        level_fit = "overqualified"
    else:
        level_fit = "underqualified"

    years_fit = "unknown"
    if jd_years > 0 and resume_years_max > 0:
        if resume_years_max >= jd_years:
            years_fit = "meets_requirement"
        else:
            years_fit = f"gap_{jd_years - resume_years_max}_years"

    return {
        "jd_seniority": jd_profile.seniority,
        "resume_seniority": resume_profile.seniority,
        "level_fit": level_fit,
        "jd_years_required": jd_years,
        "resume_years_inferred": resume_years_max,
        "years_fit": years_fit,
        "jd_education": jd_profile.education_required,
        "resume_education": resume_profile.education_required,
    }
