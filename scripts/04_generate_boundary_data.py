"""Step 4: Generate targeted boundary examples for v3 retraining.

Targets 4 failure modes from v2 evaluation:
  A: Implicit/colloquial claims (CR=1) - model misses short, casual risky claims
  B: Factual-but-not-argumentative (AQ=0) - model sees reasoning in structured facts
  C: Compressed reasoning (AQ=1) - model misses short posts with actual reasoning
  D: AI slop (AQ=0) - well-structured AI text that sounds good but has no real argument
"""

import asyncio
import json
import os
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "training"
OUTPUT_FILE = DATA_DIR / "boundary_data_v3.jsonl"

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
    "housing and rent costs",
    "labor and workers rights",
    "food and nutrition claims",
    "cryptocurrency and finance",
    "mental health",
    "environmental regulations",
]

# Category A: Implicit/colloquial claims the model misses (CR=1, AQ=0)
# These are short, casual, or colloquial posts that contain risky claims
# but don't look like "typical" misinformation.
PROMPT_A = """Generate exactly {count} realistic social media posts that make implicit or colloquial claims without evidence. These should be SHORT (1-2 sentences), casual, and use everyday language - not formal misinformation style.

Key patterns to include:
- Rhetorical questions that imply a factual claim ("Yeah but what about X?")
- Casual predictions stated as fact ("It's gonna be epic / a disaster")
- Colloquial assertions ("Poles need to protect the border", "They're covering it up")
- Short dismissals that contain unsourced claims ("That's been debunked ages ago")
- Slang-heavy posts with embedded claims

These posts should NOT have reasoning or evidence - just bare assertions in casual language.

Claim Risk rubric (these are ALL CR=1):
  1 (high risk): Makes consequential claims without evidence or attribution.
  - Assertions of fact without sources
  - Conspiracy framing, urgency/virality pressure
  - Sweeping generalizations
  - Unsourced attribution

Topic area: {topic}
Vary length (but keep short), style, and tone. Include typos, slang, abbreviations.
Return a JSON array of strings only: ["post 1", "post 2", ...]"""

# Category B: Factual-but-not-argumentative (CR=0, AQ=0)
# Structured, informative text that the model wrongly scores as having reasoning.
PROMPT_B = """Generate exactly {count} realistic social media posts that are factual and well-structured but contain NO reasoning or argumentation. These should look like encyclopedia entries, news summaries, or factual observations.

Key patterns:
- "Sometimes X happens. This leads to Y." (describes sequence, doesn't argue)
- News-style summaries without opinion
- Factual descriptions that use connective words but don't actually argue
- Lists of facts without drawing conclusions
- Historical descriptions or process explanations

These posts should NOT use because/therefore to build an argument. They describe or report, they don't argue.

Argument Quality rubric (these are ALL AQ=0):
  0 (no reasoning): Pure assertion, reaction, or emotional expression.
  - Bare claims without reasoning
  - Factual descriptions without argumentation
  - Reporting without drawing conclusions

Claim Risk rubric (these are ALL CR=0):
  0 (low risk): Qualifies claims, cites sources, or doesn't make strong factual claims.

Topic area: {topic}
Vary length and style. Some should use words like "leads to", "results in", "causes" in a descriptive (not argumentative) way.
Return a JSON array of strings only: ["post 1", "post 2", ...]"""

# Category C: Compressed reasoning (CR=0, AQ=1)
# Short posts with actual reasoning that the model misses.
PROMPT_C = """Generate exactly {count} realistic social media posts that contain compressed but real reasoning in very short form. These are 1-2 sentences max but contain genuine argumentative structure.

Key patterns:
- "Because X, Y" in compressed form
- "X proves Y" or "X shows Y" with actual logic
- Short posts referencing principles or values as reasons
- Brief cause-effect arguments
- One-liner rebuttals with implicit reasoning structure

Examples:
- "Because we are trying to live in a fair and civilized society." (compressed value-based argument)
- "Collectivism opposes the Declaration of Independence, which states..." (principle-based reasoning)
- "If minimum wage tracked productivity, it'd be $24/hr - that gap alone proves the system is broken."

Argument Quality rubric (these are ALL AQ=1):
  1 (has reasoning): Provides reasons, evidence, or structured argument.
  - Causal connectives (because, therefore, since)
  - Evidence references
  - Structured reasoning, comparisons

Claim Risk rubric (these are ALL CR=0):
  0 (low risk): Qualifies claims, cites sources, or doesn't make strong factual claims.

Topic area: {topic}
Keep posts SHORT. Vary style. Include casual/informal language.
Return a JSON array of strings only: ["post 1", "post 2", ...]"""

# Category D: AI slop (CR=0, AQ=0)
# Well-formed AI-generated text that sounds good but has no real reasoning.
PROMPT_D = """Generate exactly {count} social media posts that sound like AI-generated responses - well-structured, polite, using sophisticated vocabulary, but containing NO actual reasoning or evidence. These should be the kind of response ChatGPT would give.

Key patterns:
- "It's important to consider multiple perspectives on this issue."
- "There are valid points on both sides of this debate."
- Hedge-heavy text that says nothing concrete
- Sophisticated vocabulary but empty of actual argument
- "Balanced" responses that don't actually take or defend a position
- Phrases like "nuanced", "multifaceted", "it's worth noting", "one could argue"

These posts should NOT contain real reasoning, evidence, or genuine argumentation - just the appearance of thoughtfulness.

Argument Quality rubric (these are ALL AQ=0):
  0 (no reasoning): Pure assertion, reaction, or emotional expression.
  - Bare claims without reasoning
  - Sophisticated-sounding but empty of argument

Claim Risk rubric (these are ALL CR=0):
  0 (low risk): Doesn't make strong factual claims.

Topic area: {topic}
Make them sound distinctly AI-generated. Vary length.
Return a JSON array of strings only: ["post 1", "post 2", ...]"""

CATEGORIES = {
    "A_implicit_claims": {"prompt": PROMPT_A, "cr": 1, "aq": 0, "target": 80},
    "B_factual_no_argument": {"prompt": PROMPT_B, "cr": 0, "aq": 0, "target": 80},
    "C_compressed_reasoning": {
        "prompt": PROMPT_C,
        "cr": 0,
        "aq": 1,
        "target": 80,
        "overshoot_mult": 5,
    },
    "D_ai_slop": {
        "prompt": PROMPT_D,
        "cr": 0,
        "aq": 0,
        "target": 80,
        "overshoot_mult": 2,
    },
}

# Labeling prompt uses the canonical rubric from DECISIONS.md
LABELING_PROMPT = """Score this social media post on two dimensions. Apply the rubric strictly.

Post: "{text}"

claim_risk (does this post make strong claims without evidence or attribution?):
  1 (high risk): Makes consequential claims without evidence or attribution.
    - Assertions of fact without sources
    - Conspiracy framing, urgency/virality pressure
    - Sweeping generalizations
    - Unsourced attribution
  0 (low risk): Qualifies claims, cites sources, or doesn't make strong factual claims.
    - Citations or references
    - Qualifiers (may, suggests, unclear)
    - Opinion framing without claiming factual authority
    - Questions, acknowledgment of uncertainty

argument_quality (does this post provide reasoning, evidence, or structured argument?):
  1 (has reasoning): Provides reasons, evidence, or structured argument.
    - Causal connectives (because, therefore, since)
    - Evidence references
    - Structured reasoning, comparisons
    - Acknowledgment of counterarguments
  0 (no reasoning): Pure assertion, reaction, or emotional expression.
    - Bare claims without reasoning
    - Emotional reactions, name-calling
    - Repetition without supporting argument

Return JSON only: {{"claim_risk": N, "argument_quality": N}}"""


async def generate_batch(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    prompt_template: str,
    topic: str,
    count: int,
) -> list[str]:
    async with sem:
        try:
            prompt = prompt_template.format(count=count, topic=topic)
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
            return {"claim_risk": cr, "argument_quality": aq}
        except Exception:
            return None


async def main():
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        print(f"Removed existing {OUTPUT_FILE.name}")

    stats = {"total": 0, "on_target": 0, "off_target": 0, "label_failed": 0}
    cat_stats = {}

    async with httpx.AsyncClient() as client:
        gen_sem = asyncio.Semaphore(15)
        label_sem = asyncio.Semaphore(40)

        for cat_name, cat in CATEGORIES.items():
            expected_cr, expected_aq = cat["cr"], cat["aq"]
            target = cat["target"]
            # Overshoot by 50% since some won't match expected labels
            mult = cat.get("overshoot_mult", 1.5)
            overshoot = int(target * mult)
            batches_needed = overshoot // 20 + 1

            print(f"\n{'=' * 60}")
            print(f"Category {cat_name}: CR={expected_cr}, AQ={expected_aq}")
            print(f"Target: {target}, generating ~{overshoot} candidates")

            # Generate
            tasks = []
            for i in range(batches_needed):
                topic = TOPICS[i % len(TOPICS)]
                tasks.append(generate_batch(client, gen_sem, cat["prompt"], topic, 20))
            results = await asyncio.gather(*tasks)
            texts = [t for batch in results for t in batch]
            print(f"  Generated: {len(texts)}")

            # Label each with the labeling model
            label_tasks = [label_one(client, label_sem, t) for t in texts]
            labels = await asyncio.gather(*label_tasks)

            on_target = 0
            off_target = 0
            failed = 0

            with open(OUTPUT_FILE, "a") as f:
                for text, label in zip(texts, labels):
                    if label is None:
                        failed += 1
                        continue

                    if (
                        label["claim_risk"] == expected_cr
                        and label["argument_quality"] == expected_aq
                    ):
                        # Label matches expectation - use expected labels
                        item = {
                            "text": text,
                            "claim_risk": expected_cr,
                            "argument_quality": expected_aq,
                            "source": f"boundary_{cat_name}",
                        }
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        on_target += 1
                    else:
                        off_target += 1

            cat_stats[cat_name] = {
                "generated": len(texts),
                "on_target": on_target,
                "off_target": off_target,
                "failed": failed,
            }
            stats["total"] += len(texts)
            stats["on_target"] += on_target
            stats["off_target"] += off_target
            stats["label_failed"] += failed

            print(
                f"  On target: {on_target}, Off target: {off_target}, Failed: {failed}"
            )

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for cat_name, s in cat_stats.items():
        hit_rate = s["on_target"] / max(s["generated"], 1) * 100
        print(f"  {cat_name}: {s['on_target']} examples ({hit_rate:.0f}% hit rate)")

    total_kept = stats["on_target"]
    print(f"\nTotal boundary examples: {total_kept}")
    print(f"Output: {OUTPUT_FILE}")

    # Print distribution
    from collections import Counter

    grid = Counter()
    with open(OUTPUT_FILE) as f:
        for line in f:
            d = json.loads(line)
            grid[(d["claim_risk"], d["argument_quality"])] += 1
    print("\nDistribution:")
    for k in sorted(grid):
        print(f"  CR={k[0]}, AQ={k[1]}: {grid[k]}")

    if total_kept < 300:
        print(
            f"\nWARNING: Only {total_kept} examples, target was 300+. May need to rerun."
        )


if __name__ == "__main__":
    asyncio.run(main())
