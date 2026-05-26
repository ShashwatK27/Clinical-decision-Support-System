# CDSS Architecture & Feature Roadmap

## Core goals from your plan
1. Read scanned prescriptions via OCR.
2. Create an impressive, modern web UI (startup-style interactive interface).
3. Add user authentication, profiles, and per-user history.
4. Provide a chatbot experience on the website.
5. Design a strong database schema for users, prescriptions, cases, and knowledge.
6. Expand disease knowledge so the system can predict likely conditions more precisely.

## Recommended architecture

### Backend
- Use Python backend (FastAPI or Flask) for:
  - Prescription analysis API
  - OCR pipeline for scanned prescription images
  - User auth, profile management, and history endpoints
  - Chatbot endpoint and conversational context management
  - Embedding generation and semantic search abstraction
  - Knowledge-base reasoning and disease scoring
- Keep the current `preprocessing/`, `mapping/`, `embeddings/`, and `vector_db/` logic, but refactor it into:
  - `api/` or `services/` layer
  - `ocr/` module for scanned-prescription ingestion
  - `user/` module for auth/profile/history
  - `knowledge/` module for disease knowledge and condition inference

### Frontend
- Build a modern web frontend with one of these:
  - React + Material UI / Tailwind
  - Next.js for SSR and polished landing pages
  - Or a custom Streamlit theme if you want faster MVP, but React gives a better "star AI" look.
- Core frontend pages:
  - Landing page with product-style branding
  - Dashboard for authenticated users
  - Prescription upload / text analysis page
  - History page with past analyses per user
  - Chatbot conversational UI
  - Knowledge Explorer / disease evidence page

### Storage
- Use a relational database for core app data:
  - PostgreSQL (recommended) or SQLite for prototype
- Optional NoSQL for logs or session history
- Keep vector embeddings in a scalable store:
  - local vector store for prototype
  - upgrade to Milvus / Weaviate / Pinecone later

## Proposed database schema

### `users`
- `id`
- `email`
- `password_hash`
- `full_name`
- `role` (`admin`, `clinician`, `student`)
- `created_at`
- `last_login`

### `profiles`
- `user_id`
- `organization`
- `specialty`
- `experience_years`
- `preferred_conditions`
- `notes`

### `prescriptions`
- `id`
- `user_id`
- `input_type` (`text`, `scan`, `image`)
- `raw_text`
- `ocr_text`
- `validated_text`
- `created_at`
- `analysis_result_json`

### `prescription_drugs`
- `id`
- `prescription_id`
- `drug_name`
- `corrected_name`
- `dose_value`
- `dose_unit`
- `frequency`

### `condition_predictions`
- `id`
- `prescription_id`
- `condition_label`
- `confidence`
- `source` (`rules`, `retrieval`, `hybrid`)
- `evidence`

### `disease_knowledge`
- `id`
- `condition_label`
- `description`
- `symptoms`
- `drug_associations`
- `severity`
- `metadata`

### `case_history`
- `id`
- `user_id`
- `prescription_id`
- `summary`
- `created_at`

### `chat_sessions`
- `id`
- `user_id`
- `session_id`
- `started_at`
- `last_message_at`
- `context_json`

### `chat_messages`
- `id`
- `chat_session_id`
- `sender` (`user`, `bot`)
- `message_text`
- `created_at`
- `metadata`

## OCR / scanned prescription pipeline
- Add an `ocr/` module using:
  - `pytesseract` for open-source OCR
  - `opencv-python` or `Pillow` for image preprocessing
- Steps:
  1. Upload image or PDF via UI
  2. Preprocess image (grayscale, denoise, threshold)
  3. Run OCR to text
  4. Parse text with `preprocessing/parser.py`
  5. Present parsed drugs and allow manual correction

## Chatbot integration
- Add a simple conversational agent with:
  - backend endpoint for messages
  - retrieval-augmented responses from disease knowledge and cases
  - optional use of an LLM API if accessible
- Example flows:
  - "What does this prescription treat?"
  - "What interaction risks should I know?"
  - "Explain the likely disease and evidence."

## Disease knowledge expansion
- Create a knowledge base of conditions and symptom-drug mappings.
- Add a `knowledge/` or `disease_knowledge.py` module that:
  - merges curated disease data with dataset-derived mappings
  - scores conditions using drug evidence and semantic case retrieval
  - returns explainable evidence statements
- Possible data sources:
  - local curated JSON or CSV
  - external medical ontologies if allowed

## Implementation phases
1. Prototype backend API and keep the existing analysis engine.
2. Add auth, user profiles, and persistent history.
3. Add OCR support and scanned-prescription ingestion.
4. Build the sleek frontend and chatbot UI.
5. Expand disease knowledge and evaluation.
6. Containerize, test, and deploy.

## Immediate next step
- I can scaffold the first backend API and database schema now,
- then add the OCR module and a polished frontend plan next.

---

> Note: The current repository has no auth, database, or OCR support yet, so these are new architectural additions rather than small fixes.
