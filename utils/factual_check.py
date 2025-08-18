import os
import re
import json
import argparse
from collections import defaultdict, Counter
from statistics import mean, pstdev
from typing import List, Dict, Any, Optional
import numpy as np
import regex as re2  
from pint import UnitRegistry

try:
    from utils.pdf_parse import extract_text_from_pdf as extract_text_fn
except Exception:
    extract_text_fn = None

# -------------------- CONFIG --------------------
PARSED_TEXT_DIR = "data/parsed_text"
RESULTS_DIR = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# “Metric cues” to help bind numbers to semantic labels (no domain hardcoding)
METRIC_CUES = {
    "accuracy": ["accuracy", "top-1", "top1", "acc."],
    "f1": ["f1", "f-1", "f score", "f-score", "f1-score", "f measure", "f-measure"],
    "precision": ["precision"],
    "recall": ["recall", "sensitivity", "tpr"],
    "specificity": ["specificity", "tnr"],
    "auc": ["auc", "roc auc", "area under curve"],
    "latency": ["latency", "response time", "inference time"],
    "throughput": ["throughput", "qps", "rps", "samples/sec"],
    "memory": ["memory", "ram", "gpu memory", "vram"],
    "size": ["model size", "parameters", "params", "param"],
    "time": ["time", "duration", "runtime"],
    "freq": ["frequency", "clock", "hz"]
}
CONTEXT_WINDOW = 40  # chars around a number for metric cue matching

# % range sanity; other units get normalized with pint
PERCENT_MIN, PERCENT_MAX = 0.0, 100.0

# ------------------------------------------------

ureg = UnitRegistry()
Q_ = ureg.Quantity

# Common unit aliases we’ll map to pint-friendly tokens
UNIT_ALIASES = {
    # time
    "ms": "millisecond",
    "s": "second",
    "sec": "second",
    "secs": "second",
    # memory/storage
    "kb": "kilobyte",
    "mb": "megabyte",
    "gb": "gigabyte",
    "tb": "terabyte",
    "kib": "kibibyte",
    "mib": "mebibyte",
    "gib": "gibibyte",
    # length
    "mm": "millimeter",
    "cm": "centimeter",
    "m": "meter",
    "km": "kilometer",
    # frequency
    "hz": "hertz",
    "khz": "kilohertz",
    "mhz": "megahertz",
    "ghz": "gigahertz"
}

# Regex to extract numbers with optional units or %
NUM_UNIT_RE = re2.compile(
    r"""
    (?P<num>[+-]?\d{1,3}(?:[\d,]{0,3})*(?:\.\d+)?|\.\d+)      # number (with commas)
    \s*
    (?P<percent>%)
    |
    (?P<num2>[+-]?\d{1,3}(?:[\d,]{0,3})*(?:\.\d+)?|\.\d+)\s*
    (?P<unit>[a-zA-Zμµ]{1,6})?                                # short unit token (optional)
    """,
    re2.VERBOSE | re2.IGNORECASE
)

def normalize_unit_token(tok: str) -> Optional[str]:
    if tok is None:
        return None
    t = tok.strip().lower()
    return UNIT_ALIASES.get(t, t)

def clean_number_str(s: str) -> float:
    # remove thousand separators like 1,234.56
    return float(s.replace(",", ""))

def extract_numeric_mentions(text: str) -> List[Dict[str, Any]]:
    mentions = []
    for m in NUM_UNIT_RE.finditer(text):
        start, end = m.span()
        ctx_start = max(0, start - CONTEXT_WINDOW)
        ctx_end = min(len(text), end + CONTEXT_WINDOW)
        context = text[ctx_start:ctx_end].lower()

        if m.group("percent"):
            val = clean_number_str(m.group("num"))
            mentions.append({
                "raw": m.group(0),
                "value": val,
                "kind": "percent",
                "unit": "%",
                "value_si": val,      # keep as 0..100
                "si_unit": "%",
                "context": context,
                "span": [start, end],
            })
        else:
            num_str = m.group("num2")
            unit_tok = normalize_unit_token(m.group("unit"))
            val = clean_number_str(num_str)

            # Try to parse with pint if unit is present
            value_si = None
            si_unit = None
            kind = None

            if unit_tok:
                try:
                    q = Q_(val, unit_tok)
                    # Normalize to a canonical SI dimensionality
                    # We care about seconds, bytes, meters, hertz as examples
                    dim = q.dimensionality

                    if "[time]" in str(dim):
                        value_si = q.to("second").magnitude
                        si_unit = "second"
                        kind = "time"
                    elif "[length]" in str(dim):
                        value_si = q.to("meter").magnitude
                        si_unit = "meter"
                        kind = "length"
                    elif "byte" in unit_tok or "[information]" in str(dim):
                        # pint treats byte as information; standardize to bytes
                        try:
                            value_si = q.to("byte").magnitude
                            si_unit = "byte"
                        except Exception:
                            # if it's bit-based, convert to byte
                            value_si = q.to("byte").magnitude
                            si_unit = "byte"
                        kind = "memory"
                    elif "[frequency]" in str(dim):
                        value_si = q.to("hertz").magnitude
                        si_unit = "hertz"
                        kind = "frequency"
                    else:
                        # keep raw; unknown dimension
                        value_si = val
                        si_unit = unit_tok
                        kind = "other"
                except Exception:
                    # Unknown/not parseable unit → keep raw
                    value_si = val
                    si_unit = unit_tok
                    kind = "other"
            else:
                # no unit; keep as raw number
                value_si = val
                si_unit = None
                kind = "scalar"

            mentions.append({
                "raw": m.group(0),
                "value": val,
                "kind": kind,
                "unit": unit_tok,
                "value_si": float(value_si) if value_si is not None else None,
                "si_unit": si_unit,
                "context": context,
                "span": [start, end],
            })
    return mentions

def bind_metric_labels(mentions: List[Dict[str, Any]]) -> None:
    """
    Attach a 'metric' label to each mention using nearby cue words.
    """
    for m in mentions:
        ctx = m["context"]
        labels = []
        for metric, cues in METRIC_CUES.items():
            if any(cue in ctx for cue in cues):
                labels.append(metric)
        m["metric_labels"] = labels or []

def sanity_checks(mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issues = []
    for m in mentions:
        # Percent must be 0..100
        if m["kind"] == "percent":
            v = m["value_si"]
            if v < PERCENT_MIN or v > PERCENT_MAX:
                issues.append({
                    "type": "range_error_percent",
                    "message": f"Percentage {v}% out of 0–100 bounds",
                    "mention": m["raw"]
                })
        # Obvious nonsense: negative time/length/memory/frequency
        if m["kind"] in {"time", "length", "memory", "frequency"} and m["value_si"] is not None:
            if m["value_si"] < 0:
                issues.append({
                    "type": "negative_physical_quantity",
                    "message": f"Negative {m['kind']} value detected",
                    "mention": m["raw"]
                })
    return issues

def internal_consistency_checks(mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    If multiple values exist for the same metric (e.g., accuracy), flag large discrepancy.
    Uses simple heuristics by metric label + unit kind.
    """
    issues = []
    # Group by primary metric label if available, else by kind+si_unit
    buckets = defaultdict(list)
    for m in mentions:
        key = None
        if m["metric_labels"]:
            key = tuple(sorted(m["metric_labels"]))  # multi-label bucket
        else:
            key = (m["kind"], m.get("si_unit"))
        buckets[key].append(m)

    for key, items in buckets.items():
        if len(items) < 2:
            continue
        # For percents—check spread
        if any(x["kind"] == "percent" for x in items):
            vals = [x["value_si"] for x in items if x["kind"] == "percent"]
            if len(vals) >= 2:
                spread = abs(max(vals) - min(vals))
                if spread >= 10.0:  # >10 percentage points difference
                    issues.append({
                        "type": "inconsistent_percent",
                        "message": f"Conflicting percent values ({min(vals)}–{max(vals)}) for metric {key}",
                        "metric": str(key),
                        "values": vals
                    })
        # For times/memory/etc., check order-of-magnitude contradictions
        elif any(x["kind"] in {"time", "memory", "length", "frequency"} for x in items):
            vals = [x["value_si"] for x in items if x["value_si"] is not None]
            if len(vals) >= 2:
                ratio = max(vals) / max(1e-12, min(vals))
                if ratio >= 100:  # two orders of magnitude difference
                    issues.append({
                        "type": "inconsistent_physical_quantity",
                        "message": f"Values differ by ≥100× for {key}",
                        "metric": str(key),
                        "values": vals
                    })
    return issues

# ---------- Corpus statistics (no manual domain ranges) ----------

def build_corpus_stats(corpus_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Extract numeric mentions from all corpus texts and compute mean/std
    per (bucket_key) where bucket_key is a string like:
      - "percent::%"  (for percentages)
      - "time::second"
      - "memory::byte"
      - "length::meter"
      - "frequency::hertz"
      - "scalar::None"
    This gives empirical ranges to compare new paper values against.
    """
    agg = defaultdict(list)

    def bucket_key(kind: str, si_unit: Optional[str]) -> str:
        return f"{kind}::{si_unit}"

    # Walk all parsed text files
    if not os.path.isdir(corpus_dir):
        return {}

    for fname in os.listdir(corpus_dir):
        if not fname.lower().endswith(".txt"):
            continue
        try:
            with open(os.path.join(corpus_dir, fname), "r", encoding="utf-8") as f:
                text = f.read()
            mentions = extract_numeric_mentions(text)
            bind_metric_labels(mentions)
            for m in mentions:
                if m["value_si"] is None:
                    continue
                k = bucket_key(m["kind"], m.get("si_unit"))
                # Avoid absurd gigantic numbers in scalar bucket by clipping
                agg[k].append(float(m["value_si"]))
        except Exception:
            continue

    stats = {}
    for k, vals in agg.items():
        if len(vals) < 10:
            # too few examples → skip statistical check for this bucket
            continue
        mu = float(mean(vals))
        sigma = float(pstdev(vals)) if len(vals) > 1 else 0.0
        stats[k] = {
            "count": len(vals),
            "mean": mu,
            "std": sigma,
            "min": float(min(vals)),
            "max": float(max(vals)),
        }
    return stats

def zscore(val: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-12:
        return 0.0
    return (val - mu) / sigma

def statistical_plausibility_checks(mentions: List[Dict[str, Any]],
                                   stats: Dict[str, Dict[str, float]],
                                   z_thresh: float = 3.0) -> List[Dict[str, Any]]:
    """
    For each mention, if we have corpus stats for the same bucket, compute z-score.
    Flag if |z| >= z_thresh. This auto-adapts to your corpus—no manual ranges.
    """
    issues = []

    def bucket_key(kind: str, si_unit: Optional[str]) -> str:
        return f"{kind}::{si_unit}"

    for m in mentions:
        if m["value_si"] is None:
            continue
        k = bucket_key(m["kind"], m.get("si_unit"))
        if k in stats:
            mu = stats[k]["mean"]
            sd = stats[k]["std"]
            z = zscore(m["value_si"], mu, sd) if sd > 0 else 0.0
            if abs(z) >= z_thresh:
                issues.append({
                    "type": "statistical_outlier",
                    "message": f"Value {m['value_si']} {m.get('si_unit')} is a |z|≥{z_thresh:.1f} outlier vs corpus",
                    "bucket": k,
                    "zscore": float(z),
                    "mean": mu,
                    "std": sd,
                    "mention": m["raw"]
                })
    return issues

# -------------------- MAIN --------------------

def read_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        if extract_text_fn is None:
            raise RuntimeError("No PDF parser available (pdf_parse.extract_text_from_pdf not found).")
        return extract_text_fn(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

def main():
    parser = argparse.ArgumentParser(description="Step 7 — Factual Verification (two-layer checks)")
    parser.add_argument("path", help="Path to NEW paper (PDF or TXT)")
    parser.add_argument("--corpus_dir", default=PARSED_TEXT_DIR, help="Parsed text dir for learning empirical ranges")
    parser.add_argument("--z_thresh", type=float, default=3.0, help="Z-score threshold for outlier flagging")
    args = parser.parse_args()

    text = read_text(args.path)

    # 1) Extract numeric mentions + metric labels
    mentions = extract_numeric_mentions(text)
    bind_metric_labels(mentions)

    # 2) Deterministic sanity/internal consistency checks
    hard_issues = sanity_checks(mentions) + internal_consistency_checks(mentions)

    # 3) Build corpus stats & run statistical checks
    stats = build_corpus_stats(args.corpus_dir)
    stat_issues = statistical_plausibility_checks(mentions, stats, z_thresh=args.z_thresh)

    report = {
        "file": args.path,
        "num_mentions": len(mentions),
        "mentions": mentions[:2000],  # cap to keep JSON manageable
        "issues": {
            "hard_checks": hard_issues,
            "statistical_checks": stat_issues
        },
        "corpus_stats_available_for": sorted(stats.keys())
    }

    out_name = os.path.splitext(os.path.basename(args.path))[0] + "_factual_report.json"
    out_path = os.path.join(RESULTS_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Console summary
    print(f"\n[FACTUAL CHECK] Mentions extracted: {len(mentions)}")
    print(f"[FACTUAL CHECK] Hard-check issues: {len(hard_issues)}")
    print(f"[FACTUAL CHECK] Statistical issues: {len(stat_issues)}")
    print(f"[FACTUAL CHECK] Saved JSON report → {out_path}\n")

    if hard_issues:
        print("Top hard-check issues:")
        for i in hard_issues[:5]:
            print(" -", i["type"], ":", i["message"])

    if stat_issues:
        print("\nTop statistical issues:")
        for i in stat_issues[:5]:
            print(" -", i["type"], ":", i["message"])

if __name__ == "__main__":
    main()
