"""
Sample resume + JD for demo purposes.
Lets recruiters try the tool without uploading anything.
"""

SAMPLE_RESUME = """Aditya Sharma
Senior Data Scientist | NLP & LLM Specialist
Email: aditya@example.com | LinkedIn: linkedin.com/in/aditya | GitHub: github.com/aditya

SUMMARY
Senior Data Scientist with 3+ years of experience building production ML systems in the healthcare AI space. Deep expertise in LLMs, clinical NLP, VLMs, and MLOps. Proven track record deploying high-throughput inference pipelines and building end-to-end document intelligence systems at scale.

EXPERIENCE

Senior Data Scientist — Althea.ai (2022 – Aug 2025)
- Led development of a VLM-based document intelligence pipeline using Qwen 2.5 with vLLM and PagedAttention for 40% throughput improvement over baseline serving.
- Built a BERT-based NER and relation extraction system for clinical notes, achieving F1 > 0.91 on health insurance KIE tasks (prior authorization, EOB parsing).
- Designed a HEDIS compliance ML system combining structured EHR features with free-text clinical NLP, reducing false-negative rate by 28%.
- Architected a CHF readmission risk prediction pipeline using XGBoost + clinical NLP features deployed via FastAPI on AWS EC2.
- Built a custom MinIO-based model versioning platform replacing MLflow for internal artifact management, supporting 12 concurrent model experiments.
- Developed a fraud, waste, and abuse detection system for health insurance claims using ensemble models + rule engines.
- Deployed RAG pipeline using Milvus as vector store and PaddleOCR for document ingestion, reducing manual review time by 60%.

SKILLS
Languages: Python, SQL, Bash
Frameworks: PyTorch, HuggingFace Transformers, scikit-learn, XGBoost, FastAPI, Streamlit
LLM/VLM: vLLM, Qwen 2.5, LLaMA, BERT, RoBERTa, sentence-transformers
Vector/RAG: Milvus, FAISS, RAG pipelines, LangChain
MLOps: Docker, MLflow (prior), MinIO, GitHub Actions, AWS (EC2, S3)
NLP: Named Entity Recognition, Relation Extraction, Text Classification, Clinical NLP
OCR: PaddleOCR, Tesseract
Databases: PostgreSQL, MongoDB, Redis

PROJECTS
AI Skill Demand Tracker (2025–Present)
- Live job-posting scraper with LLM-based skill extraction, trend analytics dashboard
- Stack: FastAPI, Supabase (PostgreSQL), Railway, Streamlit

EDUCATION
M.Tech, Computer Science — MNNIT Prayagraj (2022) | GPA: 9.0 / 10
B.Tech, Computer Science — 2020
"""

SAMPLE_JD = """Senior ML Engineer — LLM Platform

About the Role
We are looking for a Senior ML Engineer to join our LLM Platform team. You will design, build, and maintain the infrastructure that powers our generative AI products used by millions of users. This is a hands-on engineering role requiring deep expertise in large language models, distributed systems, and production ML.

Responsibilities
- Design and implement scalable LLM inference infrastructure supporting low-latency, high-throughput serving
- Build and maintain RAG pipelines using vector databases (Pinecone, Weaviate, or similar)
- Fine-tune open-source LLMs (LLaMA, Mistral) using LoRA/QLoRA for domain-specific tasks
- Develop evaluation harnesses and automated benchmarking for LLM outputs
- Collaborate with product teams to deploy models via FastAPI microservices
- Own MLOps pipelines: experiment tracking, model versioning, CI/CD for ML systems
- Mentor junior engineers and conduct code reviews

Requirements
- 4+ years of experience in ML Engineering or Data Science
- Strong Python skills; production-grade code quality
- Hands-on experience with LLM serving frameworks (vLLM, TGI, or TorchServe)
- Experience with vector databases and RAG system design
- Familiarity with fine-tuning techniques: LoRA, QLoRA, PEFT
- MLOps experience: experiment tracking (MLflow, W&B), Docker, Kubernetes
- Experience with cloud platforms (AWS, GCP, or Azure)
- Bachelor's or Master's degree in Computer Science or related field

Nice to Have
- Experience with evaluation frameworks (LangSmith, RAGAS, custom harnesses)
- Contributions to open-source ML projects
- Experience with multi-modal models (VLMs)
- Knowledge of quantization, speculative decoding, or other inference optimization techniques

We offer competitive compensation, remote-first culture, and a small senior team working on genuinely hard problems.
"""