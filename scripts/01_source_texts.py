"""Step 1: Source ~12K texts from existing data + OpenRouter generation."""

import asyncio
import json
import os
import random
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "training"
OUT_FILE = DATA_DIR / "all_texts_raw.jsonl"

OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEN_MODEL = "google/gemini-2.5-flash"


def extract_sourced_texts() -> list[dict]:
    texts = []

    with open(DATA_DIR / "synthetic_dqi.jsonl") as f:
        for line in f:
            d = json.loads(line)
            texts.append({"text": d["text"], "source": "synthetic_dqi"})

    with open(DATA_DIR / "unified.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get("source") in ("europolis", "sfu"):
                texts.append({"text": d["text"], "source": d["source"]})

    print(f"Sourced texts: {len(texts)}")
    return texts


TOPICS = [
    "politics and elections",
    "public health and vaccines",
    "climate change and energy",
    "technology and AI",
    "social media and misinformation",
    "immigration policy",
    "economic policy and inflation",
    "education reform",
    "criminal justice",
    "international conflicts",
    "local community issues",
    "food and nutrition claims",
    "cryptocurrency and finance",
    "housing and rent costs",
    "mental health",
    "environmental regulations",
    "gun policy",
    "labor and workers rights",
    "space exploration",
    "privacy and surveillance",
]

STYLES = [
    "Twitter/X posts (1-2 sentences, casual, some with hashtags)",
    "Reddit comments (1-2 paragraphs, varied formality)",
    "Facebook comments (conversational, sometimes emotional)",
    "Forum posts (longer, more detailed)",
    "News comment sections (reactive, opinionated)",
    "YouTube comments (short, often low-effort)",
]

QUALITY_MIX = [
    "high claim risk, no reasoning (conspiracy theories, unsourced assertions, fear-mongering, sweeping generalizations)",
    "high claim risk, with reasoning (well-argued but unsourced claims, persuasive conspiracy theories, sophisticated misinformation)",
    "low claim risk, with reasoning (evidence-based posts, cited sources, structured arguments, academic-style)",
    "low claim risk, no reasoning (casual observations, questions, personal anecdotes, greetings, neutral statements)",
]


def build_generation_prompt(topic: str, style: str, quality: str, count: int) -> str:
    return f"""Generate exactly {count} realistic social media posts. Each post should feel authentic - like a real person wrote it, not an AI.

Topic area: {topic}
Style: {style}
Quality profile: {quality}

CRITICAL requirements:
- Vary vocabulary, sentence structure, sophistication, and length significantly between posts
- Include realistic imperfections: typos, slang, abbreviations, incomplete sentences
- Some posts should be very short (5-15 words), some medium (20-50 words), some longer (50-150 words)
- Mix first person, second person, and third person perspectives
- Include some posts that are responses to imagined previous posts ("replying to someone")
- Do NOT make every post sound polished or articulate
- Do NOT use the same sentence patterns repeatedly

Return a JSON array of strings, nothing else:
["post 1 text", "post 2 text", ...]"""


async def generate_batch(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    topic: str,
    style: str,
    quality: str,
    count: int,
) -> list[str]:
    async with sem:
        try:
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
                    "messages": [
                        {
                            "role": "user",
                            "content": build_generation_prompt(
                                topic, style, quality, count
                            ),
                        }
                    ],
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()

            # Extract JSON array
            if text.startswith("["):
                posts = json.loads(text)
            elif "[" in text:
                start = text.index("[")
                end = text.rindex("]") + 1
                posts = json.loads(text[start:end])
            else:
                return []
            return [p for p in posts if isinstance(p, str) and len(p.strip()) > 10]
        except Exception as e:
            print(f"  Error generating batch: {e}")
            return []


async def generate_posts(target: int = 6000) -> list[dict]:
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(20)

        batches = []
        posts_per_batch = 50
        for quality in QUALITY_MIX:
            for topic in TOPICS:
                style = random.choice(STYLES)
                batches.append((topic, style, quality, posts_per_batch))

        # 4 qualities * 20 topics = 80 batches * 50 = 4000 posts
        # Add extra batches to overshoot
        for _ in range(40):
            topic = random.choice(TOPICS)
            style = random.choice(STYLES)
            quality = random.choice(QUALITY_MIX)
            batches.append((topic, style, quality, posts_per_batch))

        # 120 batches * 50 = 6000 target posts
        print(f"Generating {len(batches)} batches of ~{posts_per_batch} posts each...")

        tasks = [generate_batch(client, sem, *b) for b in batches]
        results = await asyncio.gather(*tasks)

    all_posts = []
    for batch_posts in results:
        for text in batch_posts:
            all_posts.append({"text": text.strip(), "source": "generated"})

    print(f"Generated posts: {len(all_posts)}")
    return all_posts


async def main():
    sourced = extract_sourced_texts()
    generated = await generate_posts(target=6000)

    all_texts = sourced + generated
    random.shuffle(all_texts)

    with open(OUT_FILE, "w") as f:
        for item in all_texts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nTotal texts written to {OUT_FILE}: {len(all_texts)}")
    print(f"  Sourced: {len(sourced)}")
    print(f"  Generated: {len(generated)}")


if __name__ == "__main__":
    asyncio.run(main())
