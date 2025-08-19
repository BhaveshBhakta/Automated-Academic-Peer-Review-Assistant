# Automated Academic Peer Review Assistant

A modular pipeline for automating key parts of the academic peer review process — including PDF parsing, reference extraction, novelty detection, plagiarism checking, factual accuracy, and **LLM-powered structured review synthesis**.

The system combines **traditional NLP tools** (e.g., GROBID for PDF parsing) with **state-of-the-art LLMs** (Gemini, Groq, HuggingFace fallback) to generate professional-style reviews.

---

## Features

* **PDF Parsing** with GROBID (extracts structure, references).
* **Novelty Detection** – compares manuscript claims with related works.
* **Claim Mapping** – aligns extracted claims with existing research.
* **Plagiarism Checking** – flags verbatim & paraphrased overlaps.
* **Factual Accuracy** – verifies reported numbers & methods.
* **Hierarchical Summarization** – merges multiple analysis reports into a coherent review.
* **LLM-Structured Synthesis (Layer 2)**

  * Outputs **Strengths / Weaknesses / Suggestions** per category.
  * Includes **Excerpts/Evidence** from plagiarism & factual reports.
  * Adds **Scoring system** (e.g., Novelty = 6/10).
* **Fallback LLM Clients** – Gemini → Groq → HuggingFace for reliability.
* **Batch Mode (multi-paper)** – run synthesis on a directory of papers.

---

## 🛠Installation

1. **Clone the repository**

```bash
git clone https://github.com/BhaveshBhakta/Automated-Academic-Peer-Review-Assistant.git
cd Automated-Academic-Peer-Review-Assistant
```

2. **Create a Python environment**

```bash
python3 -m venv langenv
source langenv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run GROBID (for PDF parsing & reference extraction)**

```bash
docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.2
```

(Keep this running in a separate terminal.)

---

## API Keys

Set your keys in a `.env` file at the repo root:

```env
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
HF_API_KEY=your_key_here
```

These are loaded automatically via `dotenv`.

---

## Usage & Test Runs

### 🔹 Step 1. Citation & Outdated Reference Check

Extracts citations and flags old references.

```bash
python grobid_citation_alerts.py tests/sample_paper.pdf 5
```

* **5** = maximum allowed reference age (years).
* Output → `data/results/citation_report.json`

---

### 🔹 Step 2. Novelty Detection (Similarity Check)

Compares the paper’s claims against related works.

```bash
python novelty_detector.py tests/sample_paper.pdf --output data/results/test_novelty.json
```

Output → `data/results/test_novelty.json`

---

### 🔹 Step 3. Claim Mapping

Aligns extracted claims with external datasets.

```bash
python claim_mapper.py tests/sample_paper.pdf --output data/results/test_claims.json
```

Output → `data/results/test_claims.json`

---

### 🔹 Step 4. Plagiarism Report

Checks for overlaps and paraphrasing.

```bash
python plagiarism_checker.py tests/sample_paper.pdf --output data/results/test_plagiarism.json
```

Output → `data/results/test_plagiarism.json`

---

### 🔹 Step 5. Factual Accuracy Check

Verifies numbers, citations, and methods consistency.

```bash
python factual_checker.py tests/sample_paper.pdf --output data/results/test_factual.json
```

Output → `data/results/test_factual.json`

---

### 🔹 Step 6. LLM Review Synthesis (Hierarchical Summarization)

Generates a professional review with **Strengths, Weaknesses, Suggestions, Scores, Excerpts**.

#### ✅ Single Paper

```bash
python utils/llm_review_synthesis.py \
  --novelty data/results/test_novelty.json \
  --claims data/results/test_claims.json \
  --plagiarism data/results/test_plagiarism.json \
  --factual data/results/test_factual.json \
  --output data/results/test_review.txt
```

Output → `data/results/test_review.txt`

#### Batch Mode (Multiple Papers)

```bash
python utils/llm_review_synthesis.py \
  --input_dir data/results/ \
  --output_dir data/reviews/
```

Output → `data/reviews/review_1.txt`, `review_2.txt`, etc.

---

## Roadmap

* [x] PDF parsing with GROBID
* [x] Novelty / Claim Mapping modules
* [x] Plagiarism & Factual consistency
* [x] LLM synthesis (Gemini + Groq + HuggingFace fallback)
* [x] Hierarchical summarization
* [x] Strengths / Weaknesses / Suggestions structure
* [x] Excerpts & scoring system
* [ ] Streamlit / Web UI for interactive demo
* [ ] Cloud deployment (Render / HuggingFace Spaces)

---

## License

MIT License
