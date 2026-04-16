"""Step 3: Validate, balance to 10K, and generate report."""

import json
import random
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "training"
INPUT_FILE = DATA_DIR / "all_texts_labeled.jsonl"
OUTPUT_FILE = DATA_DIR / "discourse_quality_10k.jsonl"
REPORT_FILE = DATA_DIR / "dataset_report.md"

TARGET_PER_CELL = 2500
TARGET_TOTAL = 10000


def load_labeled() -> list[dict]:
    items = []
    seen_texts: set[str] = set()
    dupes = 0
    with open(INPUT_FILE) as f:
        for line in f:
            item = json.loads(line)
            key = item.get("text", "").strip()
            if key in seen_texts:
                dupes += 1
                continue
            seen_texts.add(key)
            items.append(item)
    if dupes:
        print(f"Dedup: removed {dupes} duplicate texts")
    return items


def cell_key(item: dict) -> tuple[int, int]:
    return (item["claim_risk"], item["argument_quality"])


def balance(items: list[dict]) -> list[dict]:
    buckets: dict[tuple[int, int], list[dict]] = {
        (0, 0): [],
        (0, 1): [],
        (1, 0): [],
        (1, 1): [],
    }
    for item in items:
        buckets[cell_key(item)].append(item)

    print("Raw cell distribution:")
    for k in sorted(buckets):
        print(f"  CR={k[0]}, AQ={k[1]}: {len(buckets[k])}")

    balanced = []
    for k in sorted(buckets):
        bucket = buckets[k]
        if len(bucket) >= TARGET_PER_CELL:
            balanced.extend(random.sample(bucket, TARGET_PER_CELL))
        else:
            print(
                f"  WARNING: Cell CR={k[0]},AQ={k[1]} has only {len(bucket)} (need {TARGET_PER_CELL})"
            )
            balanced.extend(bucket)

    random.shuffle(balanced)
    return balanced


def validate_sample(items: list[dict], n: int = 100) -> list[dict]:
    """Return a random sample for manual validation."""
    return random.sample(items, min(n, len(items)))


def word_count(text: str) -> int:
    return len(text.split())


def generate_report(items: list[dict], validation_sample: list[dict]) -> str:
    grid = Counter()
    sources = Counter()
    word_counts = []

    for item in items:
        grid[cell_key(item)] += 1
        sources[item.get("source", "unknown")] += 1
        word_counts.append(word_count(item["text"]))

    word_counts.sort()
    cr_1 = sum(1 for i in items if i["claim_risk"] == 1)
    aq_1 = sum(1 for i in items if i["argument_quality"] == 1)

    # Get examples from each cell
    cell_examples: dict[tuple[int, int], list[dict]] = {
        (0, 0): [],
        (0, 1): [],
        (1, 0): [],
        (1, 1): [],
    }
    shuffled = items.copy()
    random.shuffle(shuffled)
    for item in shuffled:
        k = cell_key(item)
        if len(cell_examples[k]) < 5:
            cell_examples[k].append(item)

    report = f"""# Dataset Report: discourse_quality_10k.jsonl

## Summary

- **Total samples**: {len(items)}
- **Generation date**: 2026-04-10

## Class Distribution

### Per head
| Head | Label 0 | Label 1 |
|------|---------|---------|
| claim_risk | {len(items) - cr_1} | {cr_1} |
| argument_quality | {len(items) - aq_1} | {aq_1} |

### 2x2 Grid
| | AQ=0 | AQ=1 |
|---|------|------|
| **CR=0** | {grid[(0, 0)]} | {grid[(0, 1)]} |
| **CR=1** | {grid[(1, 0)]} | {grid[(1, 1)]} |

## Text Length Stats (words)

| Stat | Value |
|------|-------|
| Min | {word_counts[0]} |
| Median | {word_counts[len(word_counts) // 2]} |
| Max | {word_counts[-1]} |
| Mean | {sum(word_counts) / len(word_counts):.1f} |

## Source Distribution

| Source | Count | % |
|--------|-------|---|
"""
    for src, count in sources.most_common():
        report += f"| {src} | {count} | {count / len(items) * 100:.1f}% |\n"

    report += "\n## Validation Sample (100 random posts)\n\n"
    report += "Manual inspection of 100 randomly sampled posts:\n\n"
    report += "| # | Text (truncated) | CR | AQ | Looks correct? |\n"
    report += "|---|------------------|----|----|----------------|\n"
    for i, item in enumerate(validation_sample[:100]):
        text_preview = item["text"][:80].replace("|", "\\|").replace("\n", " ")
        report += f"| {i + 1} | {text_preview} | {item['claim_risk']} | {item['argument_quality']} | |\n"

    for k in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        report += f"\n## Examples: CR={k[0]}, AQ={k[1]}\n\n"
        for j, ex in enumerate(cell_examples[k]):
            text_preview = ex["text"][:200].replace("\n", " ")
            report += f"{j + 1}. {text_preview}\n\n"

    return report


def main():
    print("Loading labeled data...")
    items = load_labeled()
    print(f"Loaded {len(items)} labeled texts")

    print("\nBalancing...")
    balanced = balance(items)
    print(f"Balanced dataset: {len(balanced)} samples")

    # Validation sample
    validation = validate_sample(balanced)

    # Write final dataset (without source field)
    with open(OUTPUT_FILE, "w") as f:
        for item in balanced:
            out = {
                "text": item["text"],
                "claim_risk": item["claim_risk"],
                "argument_quality": item["argument_quality"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Written to {OUTPUT_FILE}")

    # Generate report
    report = generate_report(balanced, validation)
    with open(REPORT_FILE, "w") as f:
        f.write(report)
    print(f"Report written to {REPORT_FILE}")


if __name__ == "__main__":
    random.seed(42)
    main()
