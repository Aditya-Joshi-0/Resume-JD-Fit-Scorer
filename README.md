# Resume ↔ JD Fit Scorer

A semantic fit scoring tool that compares a resume against a job description using embeddings + skill overlap analysis.

## Stack (100% free)
| Layer | Tool |
|---|---|
| UI | Streamlit |
| PDF parsing | pdfplumber |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Scoring | Cosine similarity + keyword skill matching |
| Hosting | Streamlit Community Cloud |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
fit_scorer/
├── app.py              # Streamlit UI + orchestration
├── core/
│   ├── parser.py       # PDF extraction + section chunking
│   ├── embedder.py     # sentence-transformers wrapper
│   └── scorer.py       # Similarity + skill gap scoring
└── requirements.txt
```

## Phases
- [x] Phase 1 — Core pipeline (PDF → embed → score)
- [ ] Phase 2 — ChromaDB + richer skill gap analysis
- [ ] Phase 3 — Groq LLM suggestions + UI polish
- [ ] Phase 4 — Deploy + portfolio integration
