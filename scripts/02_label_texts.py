"""Step 2: Label all texts with OpenRouter using one consistent rubric."""

import asyncio
import json
import os
from collections import Counter
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "training"
INPUT_FILE = DATA_DIR / "all_texts_raw.jsonl"
OUTPUT_FILE = DATA_DIR / "all_texts_labeled.jsonl"

OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LABEL_MODEL = "google/gemini-2.0-flash-001"

LABELING_PROMPT = """Score this social media post on two dimensions. Apply the rubric strictly.

Post: "{text}"

claim_risk (does this post make strong claims without evidence or attribution?):
  0 = no: qualifies claims, cites sources, or doesn't make strong factual claims
  1 = yes: makes consequential claims without evidence, uses urgency/conspiracy/sweeping framing

argument_quality (does this post provide reasoning, evidence, or structured argument?):
  0 = no: pure assertion, reaction, or emotional expression without reasoning
  1 = yes: gives reasons, cites evidence, uses structured argument

Return JSON only: {{"claim_risk": N, "argument_quality": N}}"""


async def label_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    item: dict,
    idx: int,
    total: int,
) -> dict | None:
    async with sem:
        try:
            resp = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LABEL_MODEL,
                    "max_tokens": 64,
                    "temperature": 0.0,
                    "messages": [
                        {
                            "role": "user",
                            "content": LABELING_PROMPT.format(text=item["text"][:2000]),
                        }
                    ],
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()

            # Parse JSON response
            if not text.startswith("{"):
                start = text.index("{")
                text = text[start:]
            if not text.endswith("}"):
                end = text.rindex("}") + 1
                text = text[:end]
            labels = json.loads(text)

            cr = labels.get("claim_risk")
            aq = labels.get("argument_quality")
            if cr not in (0, 1) or aq not in (0, 1):
                return None

            if idx % 500 == 0:
                print(f"  Labeled {idx}/{total}")

            return {
                "text": item["text"],
                "claim_risk": cr,
                "argument_quality": aq,
                "source": item["source"],
            }
        except Exception as e:
            if idx % 500 == 0:
                print(f"  Error at {idx}: {e}")
            return None


async def main():
    items = []
    with open(INPUT_FILE) as f:
        for line in f:
            items.append(json.loads(line))

    print(f"Labeling {len(items)} texts with {LABEL_MODEL}...")

    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(50)
        tasks = [
            label_one(client, sem, item, i, len(items)) for i, item in enumerate(items)
        ]
        results = await asyncio.gather(*tasks)

    labeled = [r for r in results if r is not None]
    failed = len(results) - len(labeled)

    with open(OUTPUT_FILE, "w") as f:
        for item in labeled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nLabeled: {len(labeled)}, Failed: {failed}")
    print(f"Written to {OUTPUT_FILE}")

    grid = Counter()
    for item in labeled:
        grid[(item["claim_risk"], item["argument_quality"])] += 1
    print("\nRaw distribution:")
    for k in sorted(grid):
        print(f"  CR={k[0]}, AQ={k[1]}: {grid[k]}")


if __name__ == "__main__":
    asyncio.run(main())
