"""Merge unified + synthetic training data into final.jsonl with dedup and label merging."""

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "training"

UNIFIED = DATA_DIR / "unified.jsonl"
SYNTHETIC = DATA_DIR / "synthetic_dqi.jsonl"
FINAL = DATA_DIR / "final.jsonl"

DIMS = ["justification", "respect", "constructiveness"]


def count_labeled(rows, dim):
    """Count rows where dimension has a non-negative-one label."""
    return sum(1 for r in rows if r.get(dim, -1) != -1)


def main():
    with open(UNIFIED) as f:
        unified = [json.loads(line) for line in f]
    with open(SYNTHETIC) as f:
        synthetic = [json.loads(line) for line in f]

    print(f"Unified:   {len(unified)} rows")
    print(f"Synthetic: {len(synthetic)} rows")

    # Build index by text for dedup + label merge
    by_text = {}
    for row in unified:
        text = row["text"]
        by_text[text] = dict(row)

    merged_count = 0
    new_count = 0
    for row in synthetic:
        text = row["text"]
        if text in by_text:
            # Merge: synthetic labels fill in -1 gaps from unified
            existing = by_text[text]
            for dim in DIMS:
                if existing.get(dim, -1) == -1 and row.get(dim, -1) != -1:
                    existing[dim] = row[dim]
            merged_count += 1
        else:
            by_text[text] = dict(row)
            new_count += 1

    print(f"\nDedup: {merged_count} texts merged labels, {new_count} new texts added")

    final = list(by_text.values())
    print(f"Final: {len(final)} unique rows")

    # Per-dimension stats after merge
    print("\nPer-dimension labeled counts (after merge):")
    for dim in DIMS:
        count = count_labeled(final, dim)
        print(f"  {dim}: {count}")

    with open(FINAL, "w", encoding="utf-8") as f:
        for row in final:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved to {FINAL}")

    sources = Counter(r.get("source", "unknown") for r in final)
    print(f"Source distribution: {dict(sources)}")


if __name__ == "__main__":
    main()
