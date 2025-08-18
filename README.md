# Automated Academic Peer Review Assistant

A modular pipeline for automating parts of the academic peer review process — including PDF parsing, reference extraction, novelty detection, and citation analysis.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/BhaveshBhakta/Automated-Academic-Peer-Review-Assistant.git
cd Automated-Academic-Peer-Review-Assistant
```

### 2. Create a Python environment

```bash
python3 -m venv testenv
source testenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run GROBID (Required for citation parsing)

Start the GROBID Docker container:

```bash
docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.2
```

Keep this running in a separate terminal while using citation-related features.

---

## Usage

### Example: Run citation parsing & outdated reference check

```bash
python grobid_citation_alerts.py "path/to/your.pdf" 5
```

* `5` = maximum allowed reference age (in years) before marking as outdated
* Output is saved to:

```
data/results/citation_report.json
```

---
