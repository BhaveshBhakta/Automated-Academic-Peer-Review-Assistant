import os
import json
import faiss
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
import fitz  
from pdf_parse import extract_text_from_pdf 

# === Config ===
FAISS_INDEX_PATH = "data/faiss_index.bin"
FAISS_MAPPING_PATH = "data/faiss_mapping.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.40  # tweak for novelty sensitivity

# === Utils ===
def load_faiss_index_and_mapping():
    """Load FAISS index and mapping dictionary."""
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"No FAISS index found at {FAISS_INDEX_PATH}")
    if not os.path.exists(FAISS_MAPPING_PATH):
        raise FileNotFoundError(f"No mapping file found at {FAISS_MAPPING_PATH}")

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return index, mapping

# === Main novelty detection ===
def find_similar_papers(pdf_path, top_k=5):
    # Step 1: Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("[ERROR] No text extracted — cannot continue.")
        return

    # Step 2: Load FAISS index + mapping
    index, mapping = load_faiss_index_and_mapping()

    # Step 3: Load embedding model
    print(f"[MODEL] Loading {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Step 4: Embed query paper
    query_embedding = model.encode([text], convert_to_tensor=False, show_progress_bar=False)
    query_embedding = np.array(query_embedding, dtype="float32")

    # Step 5: Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    # Step 6: Show results & collect for JSON
    print("\n[RESULTS] Most similar papers:")
    results = []
    for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        paper_info = mapping.get(str(idx), {})
        title = paper_info.get("title", "Unknown title")
        url = paper_info.get("url", "No URL")
        similarity = 1 - score
        print(f"{rank}. {title}")
        print(f"   URL: {url}")
        print(f"   Similarity: {similarity:.4f}")
        if similarity >= SIMILARITY_THRESHOLD:
            print("   → This paper is NOT novel (very similar).")
        else:
            print("   → Likely novel (no strong match).")
        print("-" * 60)

        results.append({
        "index": int(idx), 
        "title": title,
        "url": url,
        "similarity": float(similarity),  
        "pdf_path": str(paper_info.get("pdf_path", ""))  
        })


    # Step 7: Save JSON for Step 5
    os.makedirs("data/results", exist_ok=True)
    json_path = f"data/results/{os.path.splitext(os.path.basename(pdf_path))[0]}_similar.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved similar papers JSON to: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4 — Novelty Detection using FAISS")
    parser.add_argument("pdf_path", help="Path to the PDF you want to check")
    parser.add_argument("--top_k", type=int, default=5, help="Number of similar papers to show")
    args = parser.parse_args()

    find_similar_papers(args.pdf_path, args.top_k)
