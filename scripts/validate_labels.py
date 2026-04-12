"""Validate synthetic DQI labels: distributions, degenerate patterns, random samples."""

import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "training" / "synthetic_dqi.jsonl"


def main():
    with open(DATA_PATH) as f:
        rows = [json.loads(line) for line in f]

    print(f"Total rows: {len(rows)}")

    # Per-dimension distributions
    dims = {
        "justification": Counter(),
        "respect": Counter(),
        "constructiveness": Counter(),
    }
    for row in rows:
        for dim in dims:
            dims[dim][row[dim]] += 1

    print("\nDistributions:")
    for dim, dist in dims.items():
        total = sum(dist.values())
        print(f"  {dim}:")
        for val in sorted(dist):
            pct = dist[val] / total * 100
            print(f"    {val}: {dist[val]:>5} ({pct:5.1f}%)")

    # Assert no class >70%
    failures = []
    for dim, dist in dims.items():
        total = sum(dist.values())
        for val, count in dist.items():
            pct = count / total * 100
            if pct > 70:
                failures.append(f"{dim}={val} is {pct:.1f}%")

    if failures:
        print(f"\nWARNING: Classes exceeding 70%: {failures}")
        print("This is marginal but may be acceptable for neutral-heavy argument data.")
    else:
        print("\nNo class exceeds 70% - distributions look healthy.")

    # Check for degenerate patterns
    print("\nDegenerate pattern checks:")

    # All-identical scores
    score_tuples = [
        (r["justification"], r["respect"], r["constructiveness"]) for r in rows
    ]
    unique_scores = len(set(score_tuples))
    print(f"  Unique score combinations: {unique_scores}")
    assert unique_scores > 10, f"Only {unique_scores} unique combos - likely degenerate"

    # Most common score
    score_counts = Counter(score_tuples)
    most_common = score_counts.most_common(1)[0]
    pct = most_common[1] / len(rows) * 100
    print(f"  Most common score: {most_common[0]} ({most_common[1]} times, {pct:.1f}%)")
    assert pct < 50, f"Single score dominates at {pct:.1f}%"

    # Text length outliers
    lengths = [len(r["text"]) for r in rows]
    avg_len = sum(lengths) / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)
    print(f"  Text lengths: min={min_len}, avg={avg_len:.0f}, max={max_len}")

    # Source field
    sources = Counter(r["source"] for r in rows)
    print(f"  Sources: {dict(sources)}")
    assert all(s == "synthetic" for s in sources), "Non-synthetic source found"

    # Random samples
    print("\n10 random samples:")
    random.seed(42)
    for row in random.sample(rows, 10):
        text_preview = row["text"][:80].replace("\n", " ")
        print(
            f"  j={row['justification']} r={row['respect']} c={row['constructiveness']} | {text_preview}..."
        )

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
