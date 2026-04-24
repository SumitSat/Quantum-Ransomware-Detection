"""
Hybrid Analysis Bulk Downloader for Q-TERD
-------------------------------------------
Downloads behavioral JSON reports (Windows API call sequences) for
ransomware and benign samples from Hybrid Analysis.

Usage:
    python3 scripts/fetch_hybrid_analysis.py --api-key YOUR_KEY_HERE
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_URL = "https://www.hybrid-analysis.com/api/v2"
OUTPUT_DIR = Path("data/raw/hybrid_analysis")

RANSOMWARE_FAMILIES = [
    "WannaCry", "LockBit", "Ryuk", "REvil", "Maze",
    "Conti", "BlackCat", "Dharma", "GandCrab", "Sodinokibi"
]

BENIGN_TAGS = ["clean", "benign", "goodware"]

# Hybrid Analysis environment IDs:
# 100 = Windows 7 32-bit, 110 = Windows 7 64-bit, 120 = Windows 10 64-bit
WINDOWS_ENV_IDS = [100, 110, 120]

MAX_PER_FAMILY   = 300   # cap per ransomware family (~3,000 ransomware total)
MAX_BENIGN       = 1500  # benign samples for class balance
RATE_LIMIT_SLEEP = 0.5   # seconds between API calls (stay under 2000/min limit)

# ─── API Helpers ───────────────────────────────────────────────────────────────

def make_headers(api_key: str) -> dict:
    return {
        "api-key": api_key,
        "User-Agent": "Q-TERD Research / Hybrid Analysis Fetcher 1.0",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }


def search_samples(api_key: str, query: str, max_results: int = 300) -> list:
    """Search for samples matching query. Returns list of sample dicts."""
    headers = make_headers(api_key)
    samples = []
    page = 0

    while len(samples) < max_results:
        payload = {
            "query": query,
            "verdict": "malicious",
            "page": page,
            "page_size": min(20, max_results - len(samples))
        }
        try:
            r = requests.post(
                f"{BASE_URL}/search/terms",
                headers=headers,
                data=payload,
                timeout=30
            )
            if r.status_code == 200:
                data = r.json()
                results = data.get("result", [])
                if not results:
                    break
                samples.extend(results)
                page += 1
            elif r.status_code == 429:
                print("  [RATE LIMIT] Sleeping 60s...")
                time.sleep(60)
            else:
                print(f"  [WARN] Search returned {r.status_code}: {r.text[:200]}")
                break
        except Exception as e:
            print(f"  [ERROR] Search failed: {e}")
            break
        time.sleep(RATE_LIMIT_SLEEP)

    return samples[:max_results]


def download_report(api_key: str, job_id: str) -> dict | None:
    """Download behavioral summary JSON for a specific job ID."""
    headers = make_headers(api_key)
    try:
        r = requests.get(
            f"{BASE_URL}/report/{job_id}/summary",
            headers=headers,
            timeout=30
        )
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            print("  [RATE LIMIT] Sleeping 60s...")
            time.sleep(60)
            return None
        else:
            return None
    except Exception as e:
        print(f"  [ERROR] Download failed for {job_id}: {e}")
        return None


def extract_api_sequence(report: dict) -> list[str] | None:
    """
    Extract ordered Windows API call names from a Hybrid Analysis report.
    Returns list of API call name strings, or None if no behavioral data.
    """
    api_calls = []

    # Path 1: processes → api_calls list (most detailed)
    processes = report.get("processes", [])
    for proc in processes:
        for call in proc.get("api_calls", []):
            name = call.get("name") or call.get("api_name")
            if name:
                api_calls.append(name)

    # Path 2: Direct api_calls at root level (some report formats)
    if not api_calls:
        for call in report.get("api_calls", []):
            name = call.get("name") or call.get("api_name")
            if name:
                api_calls.append(name)

    # Path 3: behaviors/signatures (fallback, coarser)
    if not api_calls:
        behaviors = report.get("extracted_config", {})
        if not behaviors:
            behaviors = report.get("behavior", {})
        for sig in behaviors.get("signatures", []):
            name = sig.get("name")
            if name:
                api_calls.append(name)

    return api_calls if len(api_calls) >= 10 else None


def save_sample(output_path: Path, sha256: str, api_sequence: list[str], label: int, family: str):
    """Save extracted API sequence as a JSON file."""
    output_path.mkdir(parents=True, exist_ok=True)
    record = {
        "sha256": sha256,
        "label": label,        # 1 = ransomware, 0 = benign
        "family": family,
        "api_sequence": api_sequence
    }
    filepath = output_path / f"{sha256}.json"
    with open(filepath, "w") as f:
        json.dump(record, f)


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def download_ransomware(api_key: str):
    """Download behavioral reports for ransomware families."""
    ransomware_dir = OUTPUT_DIR / "ransomware"
    total_saved = 0

    for family in RANSOMWARE_FAMILIES:
        print(f"\n[RANSOMWARE] Fetching: {family}")
        samples = search_samples(api_key, query=family, max_results=MAX_PER_FAMILY)
        print(f"  Found {len(samples)} candidate samples")

        family_saved = 0
        for s in samples:
            sha256  = s.get("sha256", "")
            job_id  = s.get("job_id") or s.get("id") or sha256
            env_id  = s.get("environment_id", 0)

            if not sha256 or not job_id:
                continue

            # Prefer Windows environments
            if env_id not in WINDOWS_ENV_IDS and env_id != 0:
                continue

            # Skip if already downloaded
            if (ransomware_dir / f"{sha256}.json").exists():
                family_saved += 1
                continue

            report = download_report(api_key, job_id)
            if not report:
                time.sleep(RATE_LIMIT_SLEEP)
                continue

            api_seq = extract_api_sequence(report)
            if api_seq:
                save_sample(ransomware_dir, sha256, api_seq, label=1, family=family)
                family_saved += 1
                print(f"  ✓ {sha256[:16]}... ({len(api_seq)} API calls)")

            time.sleep(RATE_LIMIT_SLEEP)

        print(f"  [{family}] Saved: {family_saved} samples")
        total_saved += family_saved

    print(f"\n✅ Total ransomware samples saved: {total_saved}")
    return total_saved


def download_benign(api_key: str):
    """Download behavioral reports for clean/benign Windows applications."""
    benign_dir = OUTPUT_DIR / "benign"
    total_saved = 0

    print(f"\n[BENIGN] Fetching clean Windows application samples...")
    # Search for samples tagged as clean in Windows environment
    samples = search_samples(api_key, query="clean windows application", max_results=MAX_BENIGN)
    print(f"  Found {len(samples)} candidate samples")

    for s in samples:
        sha256  = s.get("sha256", "")
        job_id  = s.get("job_id") or s.get("id") or sha256
        verdict = s.get("verdict", "").lower()

        if not sha256 or not job_id:
            continue
        if verdict not in ("no specific threat", "whitelisted", "clean", ""):
            continue
        if (benign_dir / f"{sha256}.json").exists():
            total_saved += 1
            continue

        report = download_report(api_key, job_id)
        if not report:
            time.sleep(RATE_LIMIT_SLEEP)
            continue

        api_seq = extract_api_sequence(report)
        if api_seq:
            save_sample(benign_dir, sha256, api_seq, label=0, family="benign")
            total_saved += 1
            print(f"  ✓ {sha256[:16]}... ({len(api_seq)} API calls)")

        time.sleep(RATE_LIMIT_SLEEP)

    print(f"\n✅ Total benign samples saved: {total_saved}")
    return total_saved


def print_summary():
    """Print a summary of downloaded data."""
    ransomware_count = len(list((OUTPUT_DIR / "ransomware").glob("*.json"))) if (OUTPUT_DIR / "ransomware").exists() else 0
    benign_count     = len(list((OUTPUT_DIR / "benign").glob("*.json"))) if (OUTPUT_DIR / "benign").exists() else 0
    print("\n" + "="*50)
    print(f"  DOWNLOAD SUMMARY")
    print(f"  Ransomware samples : {ransomware_count}")
    print(f"  Benign samples     : {benign_count}")
    print(f"  Total              : {ransomware_count + benign_count}")
    print(f"  Output directory   : {OUTPUT_DIR.resolve()}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Analysis Bulk Downloader for Q-TERD")
    parser.add_argument("--api-key", required=True, help="Your Hybrid Analysis API key")
    parser.add_argument("--skip-benign", action="store_true", help="Skip benign sample download")
    args = parser.parse_args()

    print("="*50)
    print(" Q-TERD Hybrid Analysis Bulk Downloader")
    print("="*50)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    download_ransomware(args.api_key)

    if not args.skip_benign:
        download_benign(args.api_key)

    print_summary()
