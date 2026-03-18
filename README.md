# **Automated Academic Peer Review Assistant** 

An AI-powered system for automated academic peer review.  

This tool analyzes research papers to provide structured, reviewer-style feedback. It performs citation quality checks, novelty search, plagiarism detection, factual consistency analysis, and claim mapping, then synthesizes the results into a professional review report. Designed for researchers, educators, and institutions to accelerate the peer review process.  

---

## Key Features  

- **Automated Analysis Suite**  
  - **Novelty Search:** Retrieve and compare papers using FAISS + semantic embeddings.  
  - **Plagiarism Detection:** Detect exact and paraphrase overlaps via semantic similarity + string matching.  
  - **Factual Checks:** Validate numerical values and units for consistency and plausibility.  
  - **Claim Mapping:** Extract and match scientific claims against prior publications.  
  - **Citation Alert (via GROBID):** Parse references, check citation quality, and flag missing/incorrect citations.  

- **Enhanced Retrieval Features**  
  - **Deep Search Mode:** On demand, fetch up to *N* new papers (via ArXiv, Semantic Scholar, CrossRef), store locally, and rebuild FAISS index for fresh comparisons.  

- **LLM-Powered Review Synthesis**  
  - **Structured Review Generation:** Summarize findings into section-wise scores, strengths/weaknesses, claim novelty, and final recommendation.  

- **User-Friendly Interface**  
  - **Web App (Flask):** Upload a PDF, optionally enable deep search, and receive a detailed review report.  
  - **Exportable Results:** Outputs JSON artifacts (novelty, plagiarism, claim mapping, factual checks, citations) and a consolidated review file.  

---

## Technology Stack  

- **Core Libraries:** `PyPDF2`, `requests`, `argparse`, `json`, `re`  
- **NLP & Embeddings:** `sentence_transformers` (`all-MiniLM-L6-v2`), `faiss`, `scikit-learn`  
- **Citation Parsing & Alerts:** `grobid` (for PDF parsing + reference extraction)  
- **Claim & Factual Analysis:** `pint` (unit normalization), regex-based claim extraction  
- **Web & UI:** `Flask`, `Jinja2`  
- **LLM Integration (optional):** `google.generativeai` (Gemini), `groq`, Hugging Face Inference API  

---
## Website Overview

<img width="1835" height="1080" alt="Screenshot_20250925_193919" src="https://github.com/user-attachments/assets/74a1444b-03c6-4811-a6aa-138425cf8aa3" />
<img width="1840" height="971" alt="Screenshot_20250919_112236" src="https://github.com/user-attachments/assets/a1871891-43a0-485b-bd38-899c9480b883" />
<img width="1844" height="1080" alt="peerreview" src="https://github.com/user-attachments/assets/f5f61699-4b6a-4bed-b6e1-373f8df551c9" />

---

## Quick Start  

1. **Clone the repository and install dependencies:**  
   ```bash
   git clone https://github.com/BhaveshBhakta/Automated-Academic-Peer-Review-Assistant
   cd Automated-Academic-Peer-Review-Assistant
   pip install -r requirements.txt
    ````

2. **Set up environment variables (for LLM integration):**
   Create a `.env` file in the project root with:

   ```bash
   GEMINI_API_KEY=your_key_here
   GROQ_API_KEY=your_key_here
   HF_API_KEY=your_key_here
   ```

   *Note: These are optional, but required for LLM-based review synthesis.*

3. **Run PDF parsing (extract text + citations using GROBID):**

  ```bash
  docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.2
  ```
   (Keep grobid running in a separate terminal.)

   ```bash
   python utils/pdf_parse.py
   ```
4. **Build FAISS index for similarity search**

```bash
python utils/faiss_index.py \
    --pdf_dir data/pdfs \
    --index_path data/faiss_indexes/global_index.bin \
    --mapping_path data/faiss_indexes/global_mapping.json \
    --metadata_path data/metadata.json
```

5. **Run the application**

```bash
python app.py
```

6. **Run the react app**

Open:

```bash
npm run dev
```


---

## High-Level Architecture

```text
User (Browser, PDF Upload)
        ↓
     Flask App
        ↓
 ┌─────────────── Pipeline ────────────────┐
 │  1. PDF Parsing (GROBID: text + citations) │
 │  2. Citation Alert (check missing refs)    │
 │  3. Novelty Check (FAISS + Embeddings)     │
 │  4. Plagiarism Detection                   │
 │  5. Factual Consistency Check              │
 │  6. Claim Extraction & Mapping             │
 │  7. Review Synthesis (LLM/Heuristics)      │
 └───────────────────────────────────────────┘
        ↓
   Structured Review Report + JSON Outputs
```

**Deep Search Flow:**

* If the user enables **Deep Search**:

  1. Fetch up to *N* new papers (ArXiv, Semantic Scholar, CrossRef).
  2. Save PDFs + metadata locally.
  3. Rebuild FAISS index with new data.
  4. Run the pipeline again with updated knowledge base.

---

## Roadmap & Future Work

* **Scalability:** Containerize with Docker and add background workers for large-scale reviews.
* **Improved Claim Extraction:** Use advanced NLP/LLM models for precise claim detection.
* **Richer Novelty Detection:** Combine dense embeddings (FAISS) with sparse retrieval (BM25) for hybrid search.
