"""
PharmaGuard — New-Data Detection Script
==========================================
Checks the FDA FAERS data portal for quarterly data releases that
are NOT yet listed in config.yaml.

How it works:
  1. Reads ``config.yaml`` to get the quarters we've already trained on.
  2. Generates all candidate quarters from 2012-Q4 (first FAERS ASCII
     release) through the current calendar quarter.
  3. Sends a lightweight HTTP HEAD request to the FDA download URL for
     each candidate quarter that is NOT in our config.
  4. If any new quarter exists on the FDA server but is missing from
     config, the script prints a summary and exits with code 1.
  5. If no new data is found, it exits with code 0.

Exit codes:
  0 — No new data available (all quarters already in config).
  1 — New data detected (at least one new quarter on the FDA server).
  2 — Script error (config not found, network issues, etc.).

Usage:
    python src/utils/check_new_data.py
"""

import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import yaml

# ── Project root (three levels up from this file) ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_trained_quarters(config_path: Path = None) -> list:
    """
    Read config.yaml and return the list of quarters already trained on.

    Parameters
    ----------
    config_path : Path, optional
        Path to config.yaml. Defaults to PROJECT_ROOT / config.yaml.

    Returns
    -------
    list[str]
        e.g. ['2026q1']
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}", file=sys.stderr)
        sys.exit(2)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    quarters = config.get("faers", {}).get("quarters", [])
    return [q.lower().strip() for q in quarters]


def generate_candidate_quarters() -> list:
    """
    Generate all candidate FAERS quarters from 2012-Q4 through the
    current calendar quarter.

    The FDA publishes data with a ~6 month lag, so we check up to the
    current quarter (the HEAD request will tell us if it actually exists).

    Returns
    -------
    list[str]
        e.g. ['2012q4', '2013q1', '2013q2', ..., '2026q2']
    """
    now = datetime.now()
    current_year = now.year
    current_quarter = (now.month - 1) // 3 + 1  # 1, 2, 3, or 4

    candidates = []
    for year in range(2012, current_year + 1):
        start_q = 4 if year == 2012 else 1
        end_q = current_quarter if year == current_year else 4
        for q in range(start_q, end_q + 1):
            candidates.append(f"{year}q{q}")

    return candidates


def check_quarter_exists(quarter: str, base_url: str, timeout: int = 30) -> bool:
    """
    Send an HTTP HEAD request to check if a FAERS quarter ZIP exists.

    Parameters
    ----------
    quarter : str
        Quarter string, e.g. '2026q2'.
    base_url : str
        FDA FAERS base download URL.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    bool
        True if the ZIP file exists (HTTP 200), False otherwise.
    """
    url = f"{base_url}/faers_ascii_{quarter}.zip"
    try:
        req = urllib.request.Request(url, method="HEAD")
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status == 200
    except Exception:
        return False


def main():
    """
    Main entry point. Check for new FDA FAERS data quarters.
    """
    print("=" * 60)
    print("PharmaGuard — New-Data Detection")
    print("=" * 60)

    # ── Load config ────────────────────────────────────────────
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_url = config.get("faers", {}).get(
        "base_url", "https://fis.fda.gov/content/Exports"
    )

    trained_quarters = load_trained_quarters(config_path)
    print(f"📋 Trained quarters in config.yaml: {trained_quarters}")

    # ── Generate candidate quarters ────────────────────────────
    all_candidates = generate_candidate_quarters()
    missing_quarters = [q for q in all_candidates if q not in trained_quarters]

    if not missing_quarters:
        print("\n✅ No new quarters to check — config covers all candidates.")
        sys.exit(0)

    print(f"🔍 Checking {len(missing_quarters)} candidate quarter(s) on FDA server...")

    # ── Check each missing quarter ─────────────────────────────
    new_quarters = []
    for quarter in missing_quarters:
        exists = check_quarter_exists(quarter, base_url)
        status = "✅ AVAILABLE" if exists else "—  not yet"
        print(f"   {quarter}: {status}")
        if exists:
            new_quarters.append(quarter)

    # ── Report results ─────────────────────────────────────────
    print()
    if new_quarters:
        print("=" * 60)
        print(f"🚨 NEW DATA DETECTED: {len(new_quarters)} new quarter(s) available!")
        print(f"   Quarters: {new_quarters}")
        print()
        print("Action required:")
        print("  1. Run: python run_retrain.py")
        print("  2. The script will automatically download, preprocess,")
        print("     retrain, version the model, and push to GitHub.")
        print("=" * 60)
        sys.exit(1)
    else:
        print("✅ No new data available on the FDA server.")
        sys.exit(0)


if __name__ == "__main__":
    main()
