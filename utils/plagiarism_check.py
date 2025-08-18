import os
import sys
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=5):
    """Split into sentence-level n-grams (or small chunks)."""
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
    return [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

def detect_exact_overlap(paper_chunks, ref_chunks, threshold=0.8):
    """Detect copy-paste plagiarism via cosine similarity on TF-IDF."""
    vectorizer = TfidfVectorizer().fit(paper_chunks + ref_chunks)
    paper_vecs = vectorizer.transform(paper_chunks)
    ref_vecs = vectorizer.transform(ref_chunks)

    sim_matrix = cosine_similarity(paper_vecs, ref_vecs)
    flagged = []
    for i, sims in enumerate(sim_matrix):
        max_sim = np.max(sims)
        if max_sim >= threshold:
            flagged.append({
                "chunk": paper_chunks[i],
                "score": float(max_sim),
                "type": "exact_overlap"
            })
    return flagged

def detect_paraphrase_overlap(paper_chunks, ref_chunks, threshold=0.75):
    """Detect paraphrasing using sentence embeddings."""
    paper_embs = embedding_model.encode(paper_chunks, convert_to_tensor=True)
    ref_embs = embedding_model.encode(ref_chunks, convert_to_tensor=True)

    cos_scores = util.cos_sim(paper_embs, ref_embs)
    flagged = []
    for i in range(len(paper_chunks)):
        max_sim = float(cos_scores[i].max().item())
        if max_sim >= threshold:
            flagged.append({
                "chunk": paper_chunks[i],
                "score": max_sim,
                "type": "paraphrase_overlap"
            })
    return flagged

def plagiarism_check(paper_path, ref_corpus_paths, output_path):
    # Load input paper
    paper_text = load_text(paper_path)
    paper_chunks = chunk_text(paper_text)

    # Load reference corpus dynamically (can be your previous step integration!)
    ref_texts = []
    for path in ref_corpus_paths:
        ref_texts.append(load_text(path))
    ref_chunks = []
    for rt in ref_texts:
        ref_chunks.extend(chunk_text(rt))

    # Run detectors
    exact_issues = detect_exact_overlap(paper_chunks, ref_chunks)
    paraphrase_issues = detect_paraphrase_overlap(paper_chunks, ref_chunks)

    # Merge results
    results = {
        "paper": paper_path,
        "references": ref_corpus_paths,
        "exact_overlap": exact_issues,
        "paraphrase_overlap": paraphrase_issues,
        "summary": {
            "exact_overlap_count": len(exact_issues),
            "paraphrase_overlap_count": len(paraphrase_issues)
        }
    }

    # Save JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[PLAGIARISM CHECK] Found {len(exact_issues)} exact overlaps, {len(paraphrase_issues)} paraphrase overlaps")
    print(f"[PLAGIARISM CHECK] Saved JSON report → {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python plagiarism_check.py <paper_path> <ref_corpus_paths...> <output_path>")
        sys.exit(1)

    paper_path = sys.argv[1]
    ref_corpus_paths = sys.argv[2:-1]
    output_path = sys.argv[-1]

    plagiarism_check(paper_path, ref_corpus_paths, output_path)
