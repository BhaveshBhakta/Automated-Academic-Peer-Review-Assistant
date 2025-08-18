import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
PAPERS_JSON = "data/papers.json"
PARSED_TEXT_DIR = "data/parsed_text"
FAISS_INDEX_PATH = "data/faiss_index.bin"
MAPPING_PATH = "data/faiss_mapping.json"

# Load metadata
def load_papers():
    with open(PAPERS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

# Load parsed text for each paper
def load_text_for_paper(pdf_filename):
    txt_filename = pdf_filename.replace(".pdf", ".txt")
    txt_path = os.path.join(PARSED_TEXT_DIR, txt_filename)
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Build FAISS index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]  # vector dimension
    index = faiss.IndexFlatL2(dim)  # L2 distance (can switch to cosine)
    index.add(embeddings)
    return index

def main():
    print("[STEP 3] Building FAISS index for corpus...")

    papers = load_papers()

    # Load sentence transformer model
    print("[MODEL] Loading sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = []
    mapping = {}  # vector_id -> paper metadata

    for idx, paper in enumerate(papers):
        pdf_path = paper.get("pdf_path")
        if not pdf_path:
            texts.append("")
            mapping[idx] = {"title": paper.get("title"), "url": paper.get("url")}
            continue

        text = load_text_for_paper(os.path.basename(pdf_path))
        texts.append(text)
        mapping[idx] = {"title": paper.get("title"), "url": paper.get("url")}

    print(f"[EMBEDDING] Encoding {len(texts)} documents...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Build FAISS index
    print("[FAISS] Creating index...")
    index = build_faiss_index(embeddings)

    # Save FAISS index
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"[SAVED] FAISS index to {FAISS_INDEX_PATH}")

    # Save mapping
    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)
    print(f"[SAVED] Mapping file to {MAPPING_PATH}")

    print("[DONE] Step 3 completed.")

if __name__ == "__main__":
    main()
