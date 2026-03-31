# Resume × JD Fit Scorer

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://your-deployed-url.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

## Overview

**Resume × JD Fit Scorer** is an AI-powered tool that provides deep semantic analysis of resume-to-job-description alignment. Unlike simple keyword matching, it uses a **5-signal hybrid approach** combining embeddings, cross-encoder reranking, tool/domain analysis, and experience level assessment to deliver actionable insights.

### Key Features

- **5-Signal Analysis Engine**
  - Section-weighted semantic similarity (30%)
  - Cross-encoder reranking (25%)
  - Tool entity alignment with precision/recall/F1 (25%)
  - Domain coverage matching (12%)
  - Experience level fit assessment (8%)

- **Intelligent Gap Analysis**
  - Decomposes JD requirements into individual components
  - Maps resume sections to unmet requirements
  - Categorizes missing skills by priority
  - Generates coverage heatmaps

- **LLM-Powered Recommendations** (with Groq API)
  - Resume rewrite suggestions based on JD specifics
  - Targeted interview preparation
  - Skill alignment insights

- **100% Free Stack**
  - Sentence-transformers for embeddings
  - ChromaDB for vector storage
  - Groq API for LLM analysis (free tier)
  - Streamlit for UI hosting

---

## Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| **UI Framework** | Streamlit | Interactive web interface |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) | Semantic encoding |
| **Cross-Encoding** | Sentence-transformers | Precision reranking |
| **Vector DB** | ChromaDB | Semantic search & storage |
| **Document Parsing** | pdfplumber | PDF extraction & chunking |
| **LLM** | Groq (llama-70b) | Analysis & recommendations |
| **Visualizations** | Plotly | Interactive charts |
| **Deployment** | Streamlit Community Cloud | Free hosting |

---

## How It Works

### Pipeline Overview

```
Resume (PDF) → Extract Text → Section Chunking → Embedding
                                                      ↓
Job Description → Preprocessing → Embedding → Vector Similarity
                                                      ↓
                                    ┌─────────────────┼─────────────────┐
                                    ↓                 ↓                 ↓
                        Semantic Similarity    Cross-Encoder       Entity Matching
                                    ↓                 ↓                 ↓
                                    └─────────────────┼─────────────────┘
                                                      ↓
                                        Score Aggregation → Fit Score (0-100)
                                                      ↓
                                    ┌─────────────────┼─────────────────┐
                                    ↓                 ↓                 ↓
                            Gap Analysis      Section Heatmap     Missing Skills
                                    ↓                 ↓                 ↓
                                    └─────────────────┼─────────────────┘
                                                      ↓
                            [Optional: LLM Recommendations]
```

### Signals Explained

1. **Section-Weighted Semantic Similarity** (30%)
   - Encodes resume sections (experience, skills, projects, etc.)
   - Compares against full JD using cosine similarity
   - Sections weighted by relevance (experience: 35%, skills: 30%, projects: 20%, etc.)

2. **Cross-Encoder Reranking** (25%)
   - Fine-tuned model for direct resume-JD pair scoring
   - More precise than bi-encoder alone
   - Captures nuanced contextual alignment

3. **Tool Entity Alignment** (25%)
   - Extracts 200+ tools/frameworks from both documents
   - Computes precision/recall/F1 metrics
   - Highlights specific technical gaps

4. **Domain Coverage** (12%)
   - Maps 50+ domain categories (healthcare AI, fintech, NLP, CV, etc.)
   - Assesses domain expertise match
   - Identifies adjacent transferable domains

5. **Experience Alignment** (8%)
   - Parses seniority signals from both documents
   - Evaluates over/underqualification
   - Contextualizes the fit narrative

---

## Installation & Running Locally

### Prerequisites
- Python 3.11+
- pip or conda

### Setup

```bash
# Clone repository
git clone https://github.com/Aditya-Joshi-0/Resume-JD-Fit-Scorer.git
cd resume-jd-fit-scorer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Usage

1. **Upload Resume** — Drag & drop a PDF or use file picker
2. **Paste Job Description** — Paste full JD text into text area
3. **View Analysis** — Multiple tabs with insights:
   - **Overview** — Main fit score & summary signals
   - **Coverage** — Section-by-section breakdown
   - **Requirements** — JD requirement mapping & missing skills
   - **Interview Prep** — LLM-generated interview talking points (requires Groq API key)
   - **Recommendations** — Resume rewrite suggestions

4. **LLM Analysis** (Optional)
   - Paste a free Groq API key (get one at [console.groq.com](https://console.groq.com))
   - Unlock LLM-powered suggestions

---

## Project Structure

```
Resume-JD-Fit-Scorer/
├── app.py                    # Streamlit UI & orchestration
├── styles.py                 # Custom CSS theming
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container configuration
│
├── core/
│   ├── config.py            # Environment & settings management
│   ├── parser.py            # PDF extraction & section parsing
│   ├── embedder.py          # Embedding generation wrapper
│   ├── entities.py          # Tool/domain entity extraction
│   ├── scorer.py            # 5-signal scoring engine
│   ├── store.py             # ChromaDB integration
│   ├── gap_analysis.py      # Requirement mapping & gap analysis
│   ├── llm.py               # Groq LLM integration
│   ├── report.py            # Visualization & report generation
│   └── sample_data.py       # Demo resume & JD for testing
│
└── README.md
```

---

## Deployment

### Streamlit Community Cloud (Recommended)

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo
4. Set Groq API key in Secrets (optional for LLM features)

### Docker Deployment

```bash
docker build -t resume-scorer .
docker run -p 8501:8501 resume-scorer
```

---

## API Requirements

### Free Tier Support

- **sentence-transformers** — Fully free, runs locally
- **ChromaDB** — Fully free, embedded
- **Groq API** — Free tier with generous rate limits (required for LLM features only)

---

## Contributing

Contributions welcome! Open an issue or PR for:
- Additional domain categories
- New scoring signals
- UI/UX improvements
- Bug fixes

---

## License

MIT License — See LICENSE for details

