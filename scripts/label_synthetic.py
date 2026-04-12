"""
Generate synthetic DQI labels for ChangeMyView comments using Claude Haiku.

Downloads a CMV sample from HuggingFace, labels each comment with
justification/respect/constructiveness scores, saves as JSONL.

Cost: ~$3-5 for 5K comments via Haiku.

Usage:
    export ANTHROPIC_API_KEY=...
    python label_synthetic.py --count 5000
"""

import argparse
import json
import os
import time
from pathlib import Path

import anthropic

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELING_PROMPT = """Score this comment on 3 deliberation quality dimensions using the Discourse Quality Index.

Comment: "{text}"

Dimensions:
- justification: 0 (no justification, bare assertion) | 1 (inferior - vague, incomplete reason) | 2 (qualified - clear reason with some support) | 3 (sophisticated - links to evidence, data, or common good)
- respect: 0 (explicit disrespect, insult, ad hominem) | 1 (dismissive, ignoring opposing view) | 2 (neutral, implicit acknowledgment) | 3 (explicit respect for opposing view, charitable interpretation)
- constructiveness: 0 (purely positional, no room for compromise) | 1 (acknowledges complexity or nuance) | 2 (proposes compromise, alternative, or mediating position)

Return ONLY valid JSON on a single line: {{"justification": N, "respect": N, "constructiveness": N}}"""


def load_cmv_comments(count: int) -> list[str]:
    """Load ChangeMyView comments from HuggingFace."""
    from datasets import load_dataset

    print(f"Loading ChangeMyView data (target: {count} comments)...")
    # Use the webis CMV corpus
    try:
        ds = load_dataset("webis/tldr-17", split="train", streaming=True)
        comments = []
        for item in ds:
            text = item.get("content", item.get("body", ""))
            if text and 50 < len(text) < 2000:
                comments.append(text.strip())
            if len(comments) >= count:
                break
        return comments
    except Exception:
        pass

    # Fallback: load a generic argument dataset
    try:
        ds = load_dataset("Anthropic/persuasion", split="train", streaming=True)
        comments = []
        for item in ds:
            text = item.get("human_response", item.get("text", ""))
            if text and 50 < len(text) < 2000:
                comments.append(text.strip())
            if len(comments) >= count:
                break
        return comments
    except Exception:
        pass

    print("WARNING: Could not load CMV or argument dataset from HuggingFace.")
    print("Falling back to IBM Debater arguments (already downloaded).")
    ds = load_dataset(
        "ibm-research/argument_quality_ranking_30k", "argument_quality_ranking"
    )
    comments = [item["argument"] for item in ds["train"] if len(item["argument"]) > 50]  # type: ignore[index]
    return comments[:count]


def label_comment(client: anthropic.Anthropic, text: str) -> dict | None:
    """Label a single comment with DQI scores using Claude Haiku."""
    prompt = LABELING_PROMPT.format(text=text[:1500])
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()  # type: ignore[union-attr]
        result = json.loads(raw)
        # Validate
        for key in ["justification", "respect", "constructiveness"]:
            if key not in result or not isinstance(result[key], int):
                return None
        return result
    except Exception as e:
        print(f"  label error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--batch-delay", type=float, default=0.1)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return

    client = anthropic.Anthropic(api_key=api_key)
    comments = load_cmv_comments(args.count)
    print(f"Loaded {len(comments)} comments to label")

    out_path = OUT_DIR / "synthetic_dqi.jsonl"
    labeled = 0
    failed = 0

    # Resume from existing file
    existing = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                existing.add(json.loads(line)["text"][:100])
        print(f"Resuming: {len(existing)} already labeled")

    with open(out_path, "a", encoding="utf-8") as f:
        for i, text in enumerate(comments):
            if text[:100] in existing:
                continue

            scores = label_comment(client, text)
            if scores is None:
                failed += 1
                continue

            row = {
                "text": text,
                "justification": scores["justification"],
                "respect": scores["respect"],
                "constructiveness": scores["constructiveness"],
                "source": "synthetic",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            labeled += 1

            if labeled % 100 == 0:
                print(f"  {labeled}/{len(comments)} labeled ({failed} failed)")

            time.sleep(args.batch_delay)

    print(f"\nDone: {labeled} labeled, {failed} failed")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
