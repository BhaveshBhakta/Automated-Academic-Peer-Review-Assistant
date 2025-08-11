import os
import json
import argparse
import requests
import fitz 

GROBID_URL = "http://localhost:8070/api/processReferences"  # GROBID server URL

# ---------- Utility: Create folders ----------
def ensure_folders():
    os.makedirs("data/parsed_text", exist_ok=True)
    os.makedirs("data/references", exist_ok=True)

# ---------- Extract full text from PDF ----------
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        return text.strip()
    except Exception as e:
        print(f"[ERROR] Failed to extract text from {pdf_path}: {e}")
        return None

# ---------- Extract references using GROBID ----------
def extract_references_with_grobid(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            files = {"input": f}
            resp = requests.post(GROBID_URL, files=files, timeout=60)
        if resp.status_code == 200:
            return resp.text  # GROBID returns XML; can be parsed later
        else:
            print(f"[WARNING] GROBID failed for {pdf_path} — Status {resp.status_code}")
            return None
    except Exception as e:
        print(f"[ERROR] GROBID request failed for {pdf_path}: {e}")
        return None

# ---------- Main processing ----------
def process_pdfs():
    ensure_folders()
    pdf_dir = "data/pdfs"
    results = []

    for pdf_file in os.listdir(pdf_dir):
        if not pdf_file.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"[PROCESSING] {pdf_path}")

        # Extract full text
        full_text = extract_text_from_pdf(pdf_path)
        if full_text:
            text_path = os.path.join("data/parsed_text", pdf_file.replace(".pdf", ".txt"))
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"[Saved text] {text_path}")
        else:
            text_path = None

        # Extract references
        refs_xml = extract_references_with_grobid(pdf_path)
        if refs_xml:
            refs_path = os.path.join("data/references", pdf_file.replace(".pdf", "_refs.xml"))
            with open(refs_path, "w", encoding="utf-8") as f:
                f.write(refs_xml)
            print(f"[Saved references] {refs_path}")
        else:
            refs_path = None

        results.append({
            "pdf_path": pdf_path,
            "text_path": text_path,
            "refs_path": refs_path
        })

    # Save summary JSON
    with open("data/pdf_processing_summary.json", "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=4)
    print(f"[Saved summary] data/pdf_processing_summary.json")

# ---------- Entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2 — Parse PDFs & Extract References")
    parser.parse_args()
    process_pdfs()
