"""
Enrich cached GDELT articles with source credibility labels.

Reads each JSON file in data/samples/, looks up each article's domain
in data/source-credibility.json, and writes enriched versions to data/enriched/.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
ENRICHED_DIR = REPO_ROOT / "data" / "enriched"
CREDIBILITY_PATH = REPO_ROOT / "data" / "source-credibility.json"

UNKNOWN_ENTRY = {
    "type": "unknown",
    "factuality": "unknown",
    "bias": "unknown",
    "sources": [],
}


def normalize_domain(domain: str) -> str:
    """Strip www. prefix and lowercase."""
    domain = domain.strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def enrich_articles(articles: list, credibility_db: dict) -> tuple[list, dict]:
    """Add credibility labels to each article. Returns (enriched_articles, stats)."""
    stats = {"matched": 0, "unmatched": 0, "total": len(articles)}
    enriched = []

    for article in articles:
        domain = normalize_domain(article.get("domain", ""))
        cred = credibility_db.get(domain)

        enriched_article = dict(article)
        if cred:
            enriched_article["credibility"] = cred
            stats["matched"] += 1
        else:
            enriched_article["credibility"] = UNKNOWN_ENTRY
            stats["unmatched"] += 1

        enriched.append(enriched_article)

    return enriched, stats


def main():
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)

    with open(CREDIBILITY_PATH, encoding="utf-8") as f:
        credibility_db = json.load(f)
    log.info("Loaded credibility DB: %d domains", len(credibility_db))

    for sample_file in sorted(SAMPLES_DIR.glob("*.json")):
        with open(sample_file, encoding="utf-8") as f:
            data = json.load(f)

        articles = data.get("articles", [])
        enriched_articles, stats = enrich_articles(articles, credibility_db)

        output = dict(data)
        output["articles"] = enriched_articles
        output["credibility_stats"] = stats

        output_path = ENRICHED_DIR / sample_file.name
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        log.info(
            "%s: %d/%d matched, %d unmatched",
            sample_file.name,
            stats["matched"],
            stats["total"],
            stats["unmatched"],
        )


if __name__ == "__main__":
    main()
