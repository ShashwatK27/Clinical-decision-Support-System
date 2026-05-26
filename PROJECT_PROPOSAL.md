# CDSS Final Year Project Proposal

## Project Title
Scalable Hybrid Clinical Decision Support System (CDSS)

## Overview
This project extends the existing CDSS to a scalable, demo-ready system suitable for a final-year project submission and demonstration. It will keep the repository's hybrid rule + retrieval architecture while adding features, scalability, evaluation, and deliverables (report, slides, demo).

## Objectives
- Produce a working, containerized CDSS that can be deployed as a demo.
- Improve extraction, normalization, and safety checks for prescriptions.
- Scale the vector search and embedding inference for higher throughput.
- Provide an evaluation suite and report documenting methods and results.

## MVP Features (must-have)
- CLI and REST API for prescription analysis (sync endpoint).
- Improved `preprocessing/parser.py` robustness and unit tests.
- Containerize app with a `Dockerfile` and `docker-compose` for local demo.
- Replace or add a scalable vector store option (Weaviate/Pinecone/FAISS remote) and abstraction layer in `vector_db/store.py`.
- Automated tests and CI pipeline (GitHub Actions) for builds and tests.

## Stretch Features (nice-to-have)
- Asynchronous background embedding worker (Celery/RQ) and caching for embeddings.
- Web UI improvements in `streamlit_app.py` and a simple demo web page.
- Role-based access / simple auth for demo users.
- Cloud deployment (Heroku/Azure App Service/GCP Cloud Run or AKS/EKS) with IaC snippets.

## Evaluation & Metrics
- Retrieval quality: Precision@K and recall on held-out synthetic/real cases.
- Parsing accuracy: token-level F1 for drug extraction on sample cases.
- Latency: time-to-first-byte for the API, and end-to-end response times under load.
- Scalability: requests-per-second before degradation (load test report).

## Deliverables
- Source code in the repository with clear README updates.
- `PROJECT_PROPOSAL.md`, final `REPORT.md` (or PDF) describing experiments and results.
- Presentation slides (10–12 slides) for the final demonstration.
- Dockerfiles and `docker-compose.yml` for local demo; optional cloud deployment scripts.
- Automated tests and CI workflow file (.github/workflows).

## Timeline (suggested, adjustable)
- Week 1: Finalize proposal, pick features, prepare dataset splits and evaluation plan.
- Week 2: Parser improvements, unit tests, and data validation scripts.
- Week 3: Build REST API, containerize app, and add embedding caching.
- Week 4: Integrate/replace vector store with scalable option; implement CI.
- Week 5: Load testing, evaluation, and performance tuning.
- Week 6: Prepare report, slides, and final polishing; run final demo.

## Next Steps (what I can do now)
- Draft a compact `REPORT.md` template and `slides/` starter PPTX.
- Add a `Dockerfile` and `docker-compose.yml` for the app.
- Scaffold a simple REST API endpoint and background worker skeleton.

Please tell me your deadline, preferred cloud provider (if any), and which stretch features you'd like prioritized.
