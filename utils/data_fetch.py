import os
import json
import argparse
import requests
import feedparser

# ---------- Utility: Create Folders ----------
def ensure_folders():
    os.makedirs("data/txt", exist_ok=True)
    os.makedirs("data/pdfs", exist_ok=True)

# ---------- Fetch from arXiv ----------
def fetch_arxiv(keyword, max_results=30):
    base_url = "http://export.arxiv.org/api/query?"
    query = f"search_query=all:{keyword}&start=0&max_results={max_results}"
    feed = feedparser.parse(requests.get(base_url + query).text)
    papers = []

    for entry in feed.entries:
        paper_id = entry.id.split("/abs/")[-1]
        pdf_link = f"http://arxiv.org/pdf/{paper_id}.pdf"
        papers.append({
            "title": entry.title.strip(),
            "abstract": entry.summary.strip(),
            "link": entry.link,
            "published": entry.published,
            "pdf_url": pdf_link
        })
    return papers

# ---------- Fetch from Semantic Scholar ----------
def fetch_semantic_scholar(keyword, max_results=20):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={keyword}&limit={max_results}&fields=title,abstract,url,openAccessPdf,publicationDate"
    resp = requests.get(url)
    data = resp.json()
    papers = []
    for paper in data.get("data", []):
        pdf_url = None
        if paper.get("openAccessPdf"):
            pdf_url = paper["openAccessPdf"].get("url")
        papers.append({
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "link": paper.get("url", ""),
            "published": paper.get("publicationDate", ""),
            "pdf_url": pdf_url
        })
    return papers

# ---------- Save TXT + PDF ----------
def save_papers(papers):
    all_metadata = []

    for idx, paper in enumerate(papers, start=1):
        # Save TXT
        txt_path = f"data/txt/paper_{idx}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Title: {paper['title']}\n")
            f.write(f"Published: {paper['published']}\n")
            f.write(f"Link: {paper['link']}\n\n")
            f.write(f"Abstract:\n{paper['abstract']}\n")
        print(f"[Saved TXT] {txt_path}")

        # Try downloading PDF if available
        pdf_path = None
        if paper.get("pdf_url"):
            try:
                pdf_resp = requests.get(paper["pdf_url"], timeout=15)
                if pdf_resp.status_code == 200 and pdf_resp.headers.get("Content-Type", "").lower().endswith("pdf"):
                    pdf_path = f"data/pdfs/paper_{idx}.pdf"
                    with open(pdf_path, "wb") as pdf_file:
                        pdf_file.write(pdf_resp.content)
                    print(f"[Saved PDF] {pdf_path}")
                else:
                    print(f"[No PDF Available] {paper['title']}")
            except Exception as e:
                print(f"[PDF Download Failed] {paper['title']} — {e}")

        # Store metadata
        paper_metadata = paper.copy()
        paper_metadata["txt_path"] = txt_path
        paper_metadata["pdf_path"] = pdf_path
        all_metadata.append(paper_metadata)

    # Save all metadata to JSON
    with open("data/papers.json", "w", encoding="utf-8") as jf:
        json.dump(all_metadata, jf, indent=4, ensure_ascii=False)
    print(f"[Saved JSON] data/papers.json")

# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch papers from arXiv + Semantic Scholar")
    parser.add_argument("--keyword", type=str, required=True, help="Search keyword")
    parser.add_argument("--max-arxiv", type=int, default=30, help="Max results from arXiv")
    parser.add_argument("--max-semantic", type=int, default=20, help="Max results from Semantic Scholar")
    args = parser.parse_args()

    ensure_folders()

    arxiv_papers = fetch_arxiv(args.keyword, args.max_arxiv)
    semantic_papers = fetch_semantic_scholar(args.keyword, args.max_semantic)

    all_papers = arxiv_papers + semantic_papers
    save_papers(all_papers)
