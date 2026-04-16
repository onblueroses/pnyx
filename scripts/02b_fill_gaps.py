"""Step 2b: Generate targeted posts for underfilled cells and label them."""

import asyncio
import json
import os
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "training"
LABELED_FILE = DATA_DIR / "all_texts_labeled.jsonl"

OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEN_MODEL = "google/gemini-2.5-flash"
LABEL_MODEL = "google/gemini-2.0-flash-001"

TOPICS = [
    "politics and elections",
    "public health and vaccines",
    "climate change and energy",
    "technology and AI",
    "immigration policy",
    "economic policy and inflation",
    "education reform",
    "criminal justice",
    "international conflicts",
    "food and nutrition claims",
    "cryptocurrency and finance",
    "housing and rent costs",
    "mental health",
    "environmental regulations",
    "labor and workers rights",
]

CELL_PROMPTS = {
    # CR=0, AQ=1: low claim risk + has reasoning
    (0, 1): """Generate exactly {count} realistic social media posts that:
- Provide clear reasoning, evidence, or structured arguments
- Qualify their claims, cite sources, or frame as opinion
- Use causal connectives (because, therefore, since)
- Reference evidence or data
- Do NOT make strong unsourced claims

Examples of what we want:
- "According to the WHO report, vaccination rates dropped 15% during 2024, which likely contributed to the measles resurgence we're seeing now."
- "I think remote work policies make sense for knowledge workers because studies show productivity stays the same while employee satisfaction increases significantly."
- "The housing crisis isn't just about supply - it's also driven by interest rates and investor speculation. The Fed data shows institutional purchases jumped 30% last year."

Topic area: {topic}
Vary length (short to long), style, and sophistication. Include realistic imperfections.
Return a JSON array of strings only: ["post 1", "post 2", ...]""",
    # CR=1, AQ=1: high claim risk + has reasoning
    (1, 1): """Generate exactly {count} realistic social media posts that:
- Make strong claims WITHOUT proper evidence or attribution
- BUT provide reasoning structure (because, therefore, since, examples)
- Are persuasive conspiracy theories, well-argued but unsourced claims, sophisticated misinformation
- Use logical connectives to build arguments, but the premises are unverified

Examples of what we want:
- "The government is clearly suppressing this information because every time someone posts about it, the content gets removed within hours. That pattern alone proves there's coordinated censorship."
- "Big pharma doesn't want you to know this, but natural immunity is stronger because your body develops a complete immune response, not just antibodies to one protein."
- "The economy is being deliberately crashed - look at how every policy decision in the last year has benefited large corporations while small businesses fail. This can't be coincidental."

Topic area: {topic}
Vary length, style, and sophistication. Include realistic imperfections.
Return a JSON array of strings only: ["post 1", "post 2", ...]""",
}


async def generate_batch(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    cell: tuple,
    topic: str,
    count: int,
) -> list[str]:
    async with sem:
        try:
            prompt = CELL_PROMPTS[cell].format(count=count, topic=topic)
            resp = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GEN_MODEL,
                    "max_tokens": 4096,
                    "temperature": 1.0,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            if text.startswith("["):
                posts = json.loads(text)
            elif "[" in text:
                posts = json.loads(text[text.index("[") : text.rindex("]") + 1])
            else:
                return []
            return [p for p in posts if isinstance(p, str) and len(p.strip()) > 10]
        except Exception as e:
            print(f"  Gen error: {e}")
            return []


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
    client: httpx.AsyncClient, sem: asyncio.Semaphore, text: str
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
                            "content": LABELING_PROMPT.format(text=text[:2000]),
                        }
                    ],
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            if not raw.startswith("{"):
                raw = raw[raw.index("{") :]
            if not raw.endswith("}"):
                raw = raw[: raw.rindex("}") + 1]
            labels = json.loads(raw)
            cr = labels.get("claim_risk")
            aq = labels.get("argument_quality")
            if cr not in (0, 1) or aq not in (0, 1):
                return None
            return {
                "text": text,
                "claim_risk": cr,
                "argument_quality": aq,
                "source": "generated_fill",
            }
        except Exception:
            return None


async def main():
    # Gaps to fill (overshoot by 50% since not all will land in target cell)
    gaps = {
        (0, 1): 1900,  # need ~1234 more, overshoot
        (1, 1): 600,  # need ~344 more, overshoot
    }

    async with httpx.AsyncClient() as client:
        gen_sem = asyncio.Semaphore(20)
        label_sem = asyncio.Semaphore(50)

        for cell, target in gaps.items():
            print(
                f"\nFilling cell CR={cell[0]}, AQ={cell[1]}: targeting {target} posts"
            )

            # Generate
            batches_needed = target // 50 + 1
            tasks = []
            for i in range(batches_needed):
                topic = TOPICS[i % len(TOPICS)]
                tasks.append(generate_batch(client, gen_sem, cell, topic, 50))
            results = await asyncio.gather(*tasks)
            texts = [t for batch in results for t in batch]
            print(f"  Generated {len(texts)} texts")

            # Label
            label_tasks = [label_one(client, label_sem, t) for t in texts]
            labeled = await asyncio.gather(*label_tasks)
            labeled = [r for r in labeled if r is not None]
            print(f"  Labeled {len(labeled)} texts")

            # Count how many landed in target cell
            on_target = [
                r for r in labeled if (r["claim_risk"], r["argument_quality"]) == cell
            ]
            print(f"  On target: {len(on_target)}")

            # Append new labeled items, skipping duplicates of existing texts
            existing_texts: set[str] = set()
            if LABELED_FILE.exists():
                with open(LABELED_FILE) as ef:
                    for eline in ef:
                        existing_texts.add(json.loads(eline).get("text", "").strip())
            new_count = 0
            with open(LABELED_FILE, "a") as f:
                for item in labeled:
                    if item.get("text", "").strip() in existing_texts:
                        continue
                    existing_texts.add(item["text"].strip())
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    new_count += 1
            print(
                f"  Appended {new_count} new items ({len(labeled) - new_count} duplicates skipped)"
            )

    # Print updated distribution
    from collections import Counter

    grid = Counter()
    with open(LABELED_FILE) as f:
        for line in f:
            d = json.loads(line)
            grid[(d["claim_risk"], d["argument_quality"])] += 1
    total = sum(grid.values())
    print(f"\nUpdated distribution (total {total}):")
    for k in sorted(grid):
        print(f"  CR={k[0]}, AQ={k[1]}: {grid[k]}")


if __name__ == "__main__":
    asyncio.run(main())
