import argparse
import json
import os
from pathlib import Path
from llm_client import query_llm  #abstraction layer

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def summarize_section(label, data, include_excerpts=False):
    """
    Create structured input for the LLM.
    Optionally include excerpts for plagiarism/factual.
    """
    text = f"### {label} Analysis\n"

    if isinstance(data, dict):
        text += json.dumps(data, indent=2)
    elif isinstance(data, list):
        text += "\n".join([str(x) for x in data])
    else:
        text += str(data)

    if include_excerpts and isinstance(data, dict):
        # Pull top 2 excerpts if available
        passages = data.get("plagiarized_passages", [])[:2]
        if passages:
            text += "\n\nSample Excerpts:\n"
            for p in passages:
                text += f"- {p.get('text', '')[:300]}...\n"

    return text

def synthesize_review(novelty, claims, plagiarism, factual):
    """
    Calls LLM to generate a structured, scored review.
    """
    prompt = f"""
You are an academic peer reviewer. 
Synthesize the following analyses into a structured peer review.

Rules:
- Use sections: Novelty, Claims, Plagiarism, Factual Accuracy
- For each: list Strengths, Weaknesses, Suggestions
- Assign a 0-10 score for each section
- At the end, give an Overall Recommendation (Accept, Minor Revisions, Major Revisions, Reject)

Data:
{summarize_section("Novelty", novelty)}
{summarize_section("Claims", claims)}
{summarize_section("Plagiarism", plagiarism, include_excerpts=True)}
{summarize_section("Factual Accuracy", factual, include_excerpts=True)}
    """

    return query_llm(prompt)

def process_single_paper(novelty, claims, plagiarism, factual, output_path):
    novelty_data = load_json(novelty)
    claims_data = load_json(claims)
    plagiarism_data = load_json(plagiarism)
    factual_data = load_json(factual)

    review = synthesize_review(novelty_data, claims_data, plagiarism_data, factual_data)

    with open(output_path, "w") as f:
        f.write(review)
    print(f"✅ Review saved to {output_path}")

def batch_process(input_dir, output_dir):
    Path(output_dir).mkdir(exist_ok=True)
    papers = sorted(Path(input_dir).glob("paper_*"))
    for idx, paper_folder in enumerate(papers, start=1):
        novelty = paper_folder / "novelty.json"
        claims = paper_folder / "claims.json"
        plagiarism = paper_folder / "plagiarism.json"
        factual = paper_folder / "factual.json"

        output_path = Path(output_dir) / f"review_{idx}.txt"
        process_single_paper(novelty, claims, plagiarism, factual, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--novelty", type=str, help="Novelty JSON file")
    parser.add_argument("--claims", type=str, help="Claims JSON file")
    parser.add_argument("--plagiarism", type=str, help="Plagiarism JSON file")
    parser.add_argument("--factual", type=str, help="Factual JSON file")
    parser.add_argument("--output", type=str, help="Output review file")
    parser.add_argument("--batch_dir", type=str, help="Directory containing multiple papers", default=None)
    parser.add_argument("--output_dir", type=str, help="Where to save batch reviews", default="data/results/batch_reviews")

    args = parser.parse_args()

    if args.batch_dir:
        batch_process(args.batch_dir, args.output_dir)
    else:
        process_single_paper(args.novelty, args.claims, args.plagiarism, args.factual, args.output)
