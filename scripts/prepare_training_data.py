"""
Prepare unified training data for deliberation quality model.

3 sources:
  - AQuA Europolis (910 samples, expert DQI labels)
  - AQuA SFU (1,043 samples, constructiveness + toxicity)
  - IBM Debater (30K samples, argument quality -> mapped to justification)

Output: data/training/unified.jsonl
Each line: {"text": str, "justification": 0-3, "respect": 0-3, "constructiveness": 0-2, "source": str}

Labels set to -1 when not available from that source (masked during training).
"""

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AQUA_DIR = Path("/tmp/aqua-data/data")


def load_europolis():
    """Load AQuA Europolis dataset. Gold-standard DQI labels."""
    rows = []
    path = AQUA_DIR / "europolis" / "europolis_whole.csv"
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row["cleaned_comment"].strip()
            if not text or len(text) < 20:
                continue

            just_raw = row.get("justification", "").strip()
            resp_raw = row.get("respect", "").strip()
            cgood_raw = row.get("cGood", "").strip()

            justification = int(just_raw) if just_raw else -1
            respect = round(float(resp_raw)) if resp_raw else -1
            constructiveness = int(cgood_raw) if cgood_raw else -1

            # Clamp respect to 0-3 range
            if respect >= 0:
                respect = min(respect, 3)

            rows.append(
                {
                    "text": text,
                    "justification": justification,
                    "respect": respect,
                    "constructiveness": constructiveness,
                    "source": "europolis",
                }
            )
    return rows


def load_sfu():
    """Load AQuA SFU corpus. Has constructiveness + toxicity (inverse respect)."""
    rows = []
    path = AQUA_DIR / "SFU" / "SFU_corpus.csv"
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row.get("comment_text", "").strip()
            if not text or len(text) < 20:
                continue

            # Constructiveness: yes=2, no=0
            is_constr = (
                row.get("expert_is_constructive", row.get("is_constructive", ""))
                .strip()
                .lower()
            )
            constructiveness = 2 if is_constr == "yes" else 0

            # Toxicity -> inverse respect: 1=not toxic (respect=2), 4=very toxic (respect=0)
            tox_raw = row.get(
                "expert_toxicity_level", row.get("toxicity_level", "")
            ).strip()
            # Handle multi-rater format like "1\n2"
            tox_val = tox_raw.split("\n")[0] if tox_raw else ""
            if tox_val and tox_val.isdigit():
                tox = int(tox_val)
                respect = max(0, 3 - tox)  # 1->2, 2->1, 3->0, 4->0
            else:
                respect = -1

            rows.append(
                {
                    "text": text,
                    "justification": -1,  # SFU doesn't have justification labels
                    "respect": respect,
                    "constructiveness": constructiveness,
                    "source": "sfu",
                }
            )
    return rows


def load_ibm_debater():
    """Load IBM Debater argument quality. Maps quality score to justification."""
    from datasets import load_dataset

    ds = load_dataset(
        "ibm-research/argument_quality_ranking_30k", "argument_quality_ranking"
    )
    rows = []

    for split in ["train", "validation", "test"]:
        for item in ds[split]:  # type: ignore[index]
            text = item["argument"].strip()  # type: ignore[index]
            if not text or len(text) < 20:
                continue

            # WA is 0-1 continuous quality score from 10 annotators
            # Map to justification scale: <0.4 -> 0, 0.4-0.6 -> 1, 0.6-0.8 -> 2, >0.8 -> 3
            wa = item["WA"]  # type: ignore[index]
            if wa < 0.4:
                justification = 0
            elif wa < 0.6:
                justification = 1
            elif wa < 0.8:
                justification = 2
            else:
                justification = 3

            rows.append(
                {
                    "text": text,
                    "justification": justification,
                    "respect": -1,  # IBM doesn't have respect labels
                    "constructiveness": -1,  # IBM doesn't have constructiveness labels
                    "source": "ibm",
                }
            )
    return rows


def main():
    print("Loading Europolis...")
    europolis = load_europolis()
    print(f"  {len(europolis)} samples")

    print("Loading SFU...")
    sfu = load_sfu()
    print(f"  {len(sfu)} samples")

    print("Loading IBM Debater...")
    ibm = load_ibm_debater()
    print(f"  {len(ibm)} samples")

    all_data = europolis + sfu + ibm
    print(f"\nTotal: {len(all_data)} samples")

    # Stats
    for dim in ["justification", "respect", "constructiveness"]:
        labeled = [r for r in all_data if r[dim] >= 0]
        dist = {}
        for r in labeled:
            v = r[dim]
            dist[v] = dist.get(v, 0) + 1
        print(f"  {dim}: {len(labeled)} labeled, dist={dict(sorted(dist.items()))}")

    out_path = OUT_DIR / "unified.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in all_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
