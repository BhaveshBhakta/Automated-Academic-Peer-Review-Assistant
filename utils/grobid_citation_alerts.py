import requests
import xml.etree.ElementTree as ET
import argparse
import json
import os
from datetime import datetime

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

def extract_references_from_grobid(pdf_path):
    files = {'input': open(pdf_path, 'rb')}
    params = {
        "consolidateHeader": 1,
        "consolidateCitations": 1
    }
    print(f"📡 Sending {pdf_path} to GROBID at {GROBID_URL} ...")
    r = requests.post(GROBID_URL, files=files, data=params)
    if r.status_code != 200:
        raise Exception(f"GROBID request failed: {r.status_code} - {r.text}")

    # Parse TEI XML response
    root = ET.fromstring(r.text)
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    references = []
    for biblStruct in root.findall(".//tei:listBibl/tei:biblStruct", ns):
        year = None
        title = None

        # Try to get publication year
        date_el = biblStruct.find(".//tei:date", ns)
        if date_el is not None and "when" in date_el.attrib:
            year = date_el.attrib["when"][:4]

        # Try to get title
        title_el = biblStruct.find(".//tei:title", ns)
        if title_el is not None:
            title = title_el.text

        if title or year:
            references.append({
                "title": title,
                "year": year
            })

    return references

def check_outdated_references(references, year_threshold):
    current_year = datetime.now().year
    outdated = []
    for ref in references:
        if ref["year"] and ref["year"].isdigit():
            if current_year - int(ref["year"]) > year_threshold:
                outdated.append(ref)
    return outdated

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 6 — Citation parsing & outdated reference alerts")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("year_threshold", type=int, help="Max age (years) before a reference is considered outdated")
    args = parser.parse_args()

    refs = extract_references_from_grobid(args.pdf_path)
    outdated_refs = check_outdated_references(refs, args.year_threshold)

    result = {
        "total_references": len(refs),
        "outdated_references": outdated_refs
    }

    os.makedirs("data/results", exist_ok=True)
    output_path = "data/results/citation_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Report saved to {output_path} ({len(refs)} refs checked)")
    if outdated_refs:
        print(f"⚠️ Outdated references found: {len(outdated_refs)}")
    else:
        print("✅ No outdated references found")
