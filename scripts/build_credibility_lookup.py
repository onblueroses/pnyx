# pyright: reportArgumentType=false
"""
Build a unified source credibility lookup from multiple datasets.

Sources:
  1. ISD State Media Profiles - domains mapped to state-affiliated media
  2. ramybaly/News-Media-Reliability - ~860 domains with factuality + bias from MBFC
  3. EUvsDisinfo-flagged outlets - curated from public EUvsDisinfo case database

Output: data/source-credibility.json
"""

import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "data" / "source-credibility.json"

# Known EUvsDisinfo-flagged outlets (curated from public database).
# These are outlets repeatedly cited in EUvsDisinfo disinformation cases.
EUVSDISINFO_OUTLETS = {
    "rt.com": {
        "country": "Russia",
        "notes": "Primary Russian state media, frequently flagged",
    },
    "sputniknews.com": {"country": "Russia", "notes": "Russian state-funded"},
    "tass.com": {"country": "Russia", "notes": "Russian state news agency"},
    "ria.ru": {"country": "Russia", "notes": "Russian state news agency"},
    "southfront.org": {"country": "Russia", "notes": "Pro-Kremlin military blog"},
    "news-front.info": {
        "country": "Russia",
        "notes": "Pro-Kremlin outlet, Crimea-based",
    },
    "strategic-culture.org": {
        "country": "Russia",
        "notes": "SVR-linked strategic analysis",
    },
    "globalresearch.ca": {
        "country": "Canada",
        "notes": "Conspiracy site, pro-Kremlin narratives",
    },
    "journal-neo.org": {
        "country": "Russia",
        "notes": "Russian Academy of Sciences front",
    },
    "katehon.com": {"country": "Russia", "notes": "Dugin-linked think tank"},
    "geopolitica.ru": {"country": "Russia", "notes": "Dugin geopolitics platform"},
    "oneworld.press": {
        "country": "Russia",
        "notes": "Pro-Kremlin English-language outlet",
    },
    "veteranstoday.com": {
        "country": "United States",
        "notes": "Conspiracy, amplifies Kremlin narratives",
    },
    "mintpressnews.com": {
        "country": "United States",
        "notes": "Anti-Western, amplifies state media",
    },
    "thegrayzone.com": {
        "country": "United States",
        "notes": "Contrarian, amplifies Kremlin framing",
    },
    "off-guardian.org": {
        "country": "United Kingdom",
        "notes": "Conspiracy, anti-mainstream",
    },
    "infowars.com": {
        "country": "United States",
        "notes": "Conspiracy, cross-amplifies disinfo",
    },
    "zerohedge.com": {
        "country": "United States",
        "notes": "Financial conspiracy, amplifies state media",
    },
    "mid.ru": {"country": "Russia", "notes": "Russian Ministry of Foreign Affairs"},
    "iz.ru": {"country": "Russia", "notes": "Russian state media, Izvestia"},
    "tvzvezda.ru": {"country": "Russia", "notes": "Russian MoD TV channel"},
    "baltnews.ee": {"country": "Russia", "notes": "Sputnik-linked Baltic outlet"},
    "baltnews.lt": {"country": "Russia", "notes": "Sputnik-linked Baltic outlet"},
    "baltnews.lv": {"country": "Russia", "notes": "Sputnik-linked Baltic outlet"},
    "pravda.ru": {"country": "Russia", "notes": "Russian state-aligned tabloid"},
    "kommersant.ru": {
        "country": "Russia",
        "notes": "Russian business daily, state-influenced",
    },
    "srbijadanas.com": {"country": "Serbia", "notes": "Pro-government Serbian outlet"},
    "b92.net": {
        "country": "Serbia",
        "notes": "Serbian outlet, occasionally amplifies narratives",
    },
    "sputnikportal.rs": {"country": "Serbia", "notes": "Sputnik Serbia"},
    "pink.rs": {"country": "Serbia", "notes": "Pro-government Serbian TV"},
}


def extract_domain(url: str) -> str:
    """Normalize a URL or domain to a bare domain."""
    if not url:
        return ""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def clone_or_use_cached(repo_url: str, name: str, tmp_dir: str) -> Path:
    """Clone a repo to tmp_dir/name, or reuse if already present."""
    target = Path(tmp_dir) / name
    if target.exists():
        log.info("Using cached clone: %s", target)
        return target
    log.info("Cloning %s ...", repo_url)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(target)],
        check=True,
        capture_output=True,
        text=True,
    )
    return target


def load_isd_state_media(csv_path: Path) -> dict:
    """Parse ISD State_Media_Matrix.csv into domain -> metadata."""
    entries = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_url = row.get("URL", "")
            domain = extract_domain(raw_url)
            if not domain:
                continue
            region = row.get("Region", "").strip()
            country = row.get("Country", "").strip()
            typology = row.get("Typology", "").strip()
            company = row.get("Media company", "").strip()
            entries[domain] = {
                "country": country,
                "region": region,
                "typology": typology,
                "company": company,
            }
    log.info("ISD: loaded %d domains", len(entries))
    return entries


def load_ramybaly(tsv_path: Path) -> dict:
    """Parse ramybaly acl2020 corpus.tsv into domain -> metadata."""
    entries = {}
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            domain = row.get("source_url_normalized", "").strip().lower()
            if not domain:
                continue
            if domain.startswith("www."):
                domain = domain[4:]
            factuality = row.get("fact", "").strip().lower()
            bias = row.get("bias", "").strip().lower()
            entries[domain] = {
                "factuality": factuality,
                "bias": bias,
            }
    log.info("ramybaly: loaded %d domains", len(entries))
    return entries


def merge_credibility(isd: dict, mbfc: dict, euvsdisinfo: dict) -> dict:
    """Merge all sources into a unified lookup keyed by domain."""
    merged = {}

    all_domains = set(isd.keys()) | set(mbfc.keys()) | set(euvsdisinfo.keys())
    for domain in sorted(all_domains):
        entry = {"sources": []}

        is_state = domain in isd
        is_euvsd = domain in euvsdisinfo
        mbfc_data = mbfc.get(domain)

        if is_state:
            isd_data = isd[domain]
            entry["type"] = "state-affiliated"
            entry["country"] = isd_data["country"]
            entry["sources"].append("isd")

        if is_euvsd:
            euvsd_data = euvsdisinfo[domain]
            if "type" not in entry:
                entry["type"] = "euvsdisinfo-flagged"
            entry.setdefault("country", euvsd_data["country"])
            entry["sources"].append("euvsdisinfo")

        if mbfc_data:
            entry.setdefault("type", "independent")
            entry["factuality"] = mbfc_data["factuality"]
            entry["bias"] = mbfc_data["bias"]
            entry["sources"].append("mbfc")
        else:
            if is_state or is_euvsd:
                entry.setdefault("factuality", "low")
                entry.setdefault("bias", "unknown")
            else:
                entry.setdefault("factuality", "unknown")
                entry.setdefault("bias", "unknown")

        if not is_state and not is_euvsd and mbfc_data:
            entry.setdefault("type", "independent")

        merged[domain] = entry

    log.info("Merged: %d total domains", len(merged))
    return merged


def main():
    tmp_dir = os.environ.get("CREDIBILITY_TMP", "/tmp")

    isd_path = Path(tmp_dir) / "isd-state-media"
    nmr_path = Path(tmp_dir) / "news-media-reliability"

    if not isd_path.exists():
        clone_or_use_cached(
            "https://github.com/Institute-for-Strategic-Dialogue/state-media-profiles.git",
            "isd-state-media",
            tmp_dir,
        )
    if not nmr_path.exists():
        clone_or_use_cached(
            "https://github.com/ramybaly/News-Media-Reliability.git",
            "news-media-reliability",
            tmp_dir,
        )

    isd_csv = isd_path / "State_Media_Matrix.csv"
    nmr_tsv = nmr_path / "data" / "acl2020" / "corpus.tsv"

    if not isd_csv.exists():
        log.error("ISD CSV not found at %s", isd_csv)
        sys.exit(1)
    if not nmr_tsv.exists():
        log.error("ramybaly TSV not found at %s", nmr_tsv)
        sys.exit(1)

    isd = load_isd_state_media(isd_csv)
    mbfc = load_ramybaly(nmr_tsv)

    merged = merge_credibility(isd, mbfc, EUVSDISINFO_OUTLETS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    log.info("Written to %s", OUTPUT_PATH)

    state_count = sum(1 for v in merged.values() if v.get("type") == "state-affiliated")
    euvsd_count = sum(
        1 for v in merged.values() if "euvsdisinfo" in v.get("sources", [])
    )
    mbfc_count = sum(1 for v in merged.values() if "mbfc" in v.get("sources", []))
    log.info(
        "Breakdown: %d state-affiliated, %d euvsdisinfo-flagged, %d with MBFC data",
        state_count,
        euvsd_count,
        mbfc_count,
    )


if __name__ == "__main__":
    main()
