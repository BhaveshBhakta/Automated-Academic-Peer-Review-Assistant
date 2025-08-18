import os
import json
import re
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from utils.pdf_parse import extract_text_from_pdf as extract_text_from_pdf_fn
except Exception:
    extract_text_from_pdf_fn = None

# ---------------- CONFIG ----------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PARSED_TEXT_DIR = "data/parsed_text"
PAPERS_JSON = "data/papers.json"   
DEFAULT_CLAIM_SIM_THRESHOLD = 0.70  # claim similarity above this -> NOT novel
MIN_SENT_LEN = 30                   # ignore tiny sentences
CLAIM_KEYWORDS = [
    "we propose", "we present", "this paper", "our contribution",
    "we show", "we demonstrate", "we introduce", "in this work",
    "we report", "we observe", "we develop", "we design"
]
# ----------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def split_into_sentences(text):
    """
    Simple sentence splitter. Keeps sentences reasonably long.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Split on sentence boundaries
    sents = re.split(r'(?<=[\.\?\!])\s+', text)
    sents = [s.strip() for s in sents if len(s.strip()) >= MIN_SENT_LEN]
    return sents

def extract_claims_by_keywords(sentences):
    """
    Heuristic claim extractor: finds sentences containing claim keywords.
    """
    claims = []
    for s in sentences:
        low = s.lower()
        if any(kw in low for kw in CLAIM_KEYWORDS):
            claims.append(s)
    return claims

def extract_text_from_paper_meta(paper_meta, idx):
    """
    Try multiple ways to get a paper's text:
      1) If paper_meta has 'pdf_path' and file exists, use pdf parser (if available)
      2) If paper_meta has 'txt_path', read it
      3) fallback to data/parsed_text/paper_{idx+1}.txt
    Returns the text string or None.
    """
    # 1) pdf_path
    pdf_path = paper_meta.get("pdf_path") or paper_meta.get("pdf")
    if pdf_path and os.path.exists(pdf_path):
        if extract_text_from_pdf_fn:
            try:
                return extract_text_from_pdf_fn(pdf_path)
            except Exception:
                pass
    # 2) txt_path or parsed text in metadata
    txt_path = paper_meta.get("txt_path") or paper_meta.get("text_path")
    if txt_path and os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    # 3) fallback to data/parsed_text/paper_{idx+1}.txt
    fallback = os.path.join(PARSED_TEXT_DIR, f"paper_{idx+1}.txt")
    if os.path.exists(fallback):
        with open(fallback, "r", encoding="utf-8") as f:
            return f.read()
    return None

def gather_existing_claims(similar_list, papers_metadata):
    """
    For each entry in similar_list (from Step 4 JSON), get the paper text and extract claims.
    Returns a list of dicts: {'paper_index': idx, 'paper_title': title, 'claim': claim_text}
    """
    existing_claims = []
    for entry in similar_list:
        idx = int(entry.get("index"))
        # Guard: index may be out of range
        if idx < 0 or idx >= len(papers_metadata):
            continue
        meta = papers_metadata[idx]
        text = extract_text_from_paper_meta(meta, idx)
        if not text:
            continue
        sents = split_into_sentences(text)
        claims = extract_claims_by_keywords(sents)
        # If no explicit claims detected, optionally pick top informative sentences (fallback)
        if not claims:
            # choose top 3 longest sentences as possible claims
            candidates = sorted(sents, key=lambda x: len(x), reverse=True)[:3]
            claims = candidates
        for c in claims:
            existing_claims.append({
                "paper_index": idx,
                "paper_title": meta.get("title", ""),
                "claim": c
            })
    return existing_claims

def extract_new_claims_from_new_pdf(new_pdf_path):
    """
    Extract text from the uploaded new PDF and find candidate claims.
    """
    # 1) get full text
    text = None
    if extract_text_from_pdf_fn and os.path.exists(new_pdf_path):
        try:
            text = extract_text_from_pdf_fn(new_pdf_path)
        except Exception:
            text = None

    # 2) fallback: if user provided a parsed txt (rare) read that
    if not text:
        basename = os.path.splitext(os.path.basename(new_pdf_path))[0]
        alt = os.path.join(PARSED_TEXT_DIR, f"{basename}.txt")
        if os.path.exists(alt):
            with open(alt, "r", encoding="utf-8") as f:
                text = f.read()

    if not text:
        raise RuntimeError("Could not extract text for the new PDF (no parser available or file missing).")

    sents = split_into_sentences(text)
    new_claims = extract_claims_by_keywords(sents)
    # fallback: if no explicit claims, take top-5 longest sentences as candidate claims
    if not new_claims:
        new_claims = sorted(sents, key=lambda x: len(x), reverse=True)[:5]
    return new_claims, text

def embed_texts(model, texts):
    """
    Encode a list of texts and return numpy array embeddings.
    """
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return emb

def map_claims(new_claims, existing_claims, model, claim_threshold=DEFAULT_CLAIM_SIM_THRESHOLD):
    """
    Compare claims embeddings and return mapping results.
    """
    if not new_claims:
        return []

    if not existing_claims:
        # nothing to compare to -> everything is novel
        return [{"claim": c, "is_novel": True, "matched_claim": None, "matched_paper": None, "similarity": 0.0} for c in new_claims]

    new_emb = embed_texts(model, new_claims)
    existing_texts = [c["claim"] for c in existing_claims]
    existing_emb = embed_texts(model, existing_texts)

    sim_matrix = cosine_similarity(new_emb, existing_emb)  # shape (n_new, n_existing)
    mappings = []
    for i, c in enumerate(new_claims):
        best_j = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i][best_j])
        best_match = existing_claims[best_j]
        is_novel = best_score < claim_threshold
        mappings.append({
            "claim": c,
            "is_novel": bool(is_novel),
            "matched_claim": best_match["claim"],
            "matched_paper_title": best_match["paper_title"],
            "matched_paper_index": int(best_match["paper_index"]),
            "similarity": round(best_score, 4)
        })
    return mappings

# ---------------- Main CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Step 5 — Claim Extraction & Mapping")
    parser.add_argument("--new_pdf", required=True, help="Path to the new PDF (the one to analyze)")
    parser.add_argument("--similar_json", required=True, help="Path to the JSON produced by Step 4 (top similar papers)")
    parser.add_argument("--claim_threshold", type=float, default=DEFAULT_CLAIM_SIM_THRESHOLD, help="Similarity threshold for marking a claim as NOT novel")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="SentenceTransformer model name")
    args = parser.parse_args()

    # Load inputs
    similar_list = load_json(args.similar_json)
    papers_meta = load_json(PAPERS_JSON)

    print("[STEP5] Extracting new paper claims...")
    new_claims, _ = extract_new_claims_from_new_pdf(args.new_pdf)
    print(f"[STEP5] Found {len(new_claims)} candidate claims in the new paper.")

    print("[STEP5] Gathering claims from similar papers...")
    existing_claims = gather_existing_claims(similar_list, papers_meta)
    print(f"[STEP5] Collected {len(existing_claims)} claims from {len(similar_list)} similar papers.")

    # Load model
    print(f"[STEP5] Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    # Map claims
    print("[STEP5] Mapping claims (computing similarity)...")
    mappings = map_claims(new_claims, existing_claims, model, claim_threshold=args.claim_threshold)

    # Save results JSON
    os.makedirs("data/results", exist_ok=True)
    base = os.path.splitext(os.path.basename(args.new_pdf))[0]
    out_path = f"data/results/{base}_claim_mapping.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "new_pdf": args.new_pdf,
            "mappings": mappings,
            "num_new_claims": len(new_claims),
            "num_existing_claims": len(existing_claims)
        }, f, indent=2, ensure_ascii=False)
    print(f"[STEP5] Saved claim mapping to: {out_path}")

    # Print summary
    for m in mappings:
        print("\nClaim:", m["claim"])
        print("Similarity:", m["similarity"], "| Novel:", m["is_novel"])
        print("Matched paper:", m["matched_paper_title"], "| Matched claim:", m["matched_claim"])

if __name__ == "__main__":
    main()
