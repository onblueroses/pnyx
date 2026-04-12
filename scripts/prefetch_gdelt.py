# pyright: reportReturnType=false
"""
Fetch GDELT data for demo topics.
Results cached to data/samples/.
Safe to re-run - skips existing files.
"""

import json
import time
from pathlib import Path
import requests

OUT_DIR = Path(__file__).parent.parent / "data" / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOPICS = [
    {
        "slug": "balkans-disinformation",
        "label": "Western Balkans: Russia/China influence + disinformation",
        "query": "Balkans disinformation propaganda",
        "start": "20250101",
        "end": "20260404",
    },
    {
        "slug": "hybrid-warfare-europe",
        "label": "Hybrid warfare / information operations in Europe",
        "query": "hybrid warfare information operations Europe",
        "start": "20250101",
        "end": "20260404",
    },
    {
        "slug": "ukraine-disinformation",
        "label": "Ukraine war narrative / disinformation",
        "query": "Ukraine disinformation narrative propaganda",
        "start": "20250101",
        "end": "20260404",
    },
]


BASE = "https://api.gdeltproject.org/api/v2/doc/doc"


def gdelt_get(params: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.get(BASE, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                wait = 2**attempt * 2
                print(f"  retry {attempt + 1}/{retries - 1} after {wait}s ({e})")
                time.sleep(wait)
            else:
                raise


def fetch_articles(query: str, start: str, end: str) -> list:
    data = gdelt_get(
        {
            "query": query,
            "mode": "ArtList",
            "maxrecords": 250,
            "format": "json",
            "startdatetime": f"{start}000000",
            "enddatetime": f"{end}235959",
            "sort": "DateAsc",
        }
    )
    return data.get("articles", [])


def fetch_timeline(query: str, start: str, end: str) -> list:
    data = gdelt_get(
        {
            "query": query,
            "mode": "TimelineVolRaw",
            "format": "json",
            "startdatetime": f"{start}000000",
            "enddatetime": f"{end}235959",
        }
    )
    return data.get("timeline", [{}])[0].get("data", [])


def fetch_source_countries(query: str, start: str, end: str) -> list:
    data = gdelt_get(
        {
            "query": query,
            "mode": "TimelineSourceCountry",
            "format": "json",
            "startdatetime": f"{start}000000",
            "enddatetime": f"{end}235959",
        }
    )
    return data.get("timeline", [])


def main():
    for topic in TOPICS:
        out_path = OUT_DIR / f"{topic['slug']}.json"
        if out_path.exists():
            print(f"[SKIP] {topic['slug']} (already cached)")
            continue

        print(f"\n[FETCH] {topic['label']}")

        print("  articles...")
        try:
            articles = fetch_articles(topic["query"], topic["start"], topic["end"])
        except Exception as e:
            print(f"  FAILED: {e}")
            articles = []
        time.sleep(1)

        print("  timeline...")
        try:
            timeline = fetch_timeline(topic["query"], topic["start"], topic["end"])
        except Exception as e:
            print(f"  FAILED: {e}")
            timeline = []
        time.sleep(1)

        print("  source countries...")
        try:
            by_country = fetch_source_countries(
                topic["query"], topic["start"], topic["end"]
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            by_country = []
        time.sleep(1)

        result = {
            "slug": topic["slug"],
            "label": topic["label"],
            "query": topic["query"],
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "article_count": len(articles),
            "articles": articles,
            "timeline": timeline,
            "by_country": by_country,
        }

        out_path.write_text(json.dumps(result, indent=2))
        print(f"  saved {len(articles)} articles → {out_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
