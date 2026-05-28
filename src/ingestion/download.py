"""
PharmaGuard — FAERS Data Download Module
=========================================
Downloads quarterly FAERS ASCII ZIP files from the FDA website.
"""

import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def download_faers_zip(quarter: str, base_url: str, dest_dir: Path) -> Path:
    """
    Download a single FAERS quarterly ASCII ZIP file.

    Parameters
    ----------
    quarter : str
        Quarter identifier, e.g. '2026q1'.
    base_url : str
        FDA FAERS export base URL.
    dest_dir : Path
        Directory to save the downloaded ZIP.

    Returns
    -------
    Path
        Path to the downloaded ZIP file.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    filename = f"faers_ascii_{quarter}.zip"
    url = f"{base_url}/{filename}"
    dest_path = dest_dir / filename

    if dest_path.exists():
        logger.info("ZIP already exists, skipping download: %s", dest_path)
        return dest_path

    logger.info("Downloading FAERS data: %s", url)

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                if downloaded % (1024 * 1024) < 8192:  # log every ~1 MB
                    logger.info("  Progress: %.1f%% (%d / %d bytes)", pct, downloaded, total_size)

    logger.info("Download complete: %s (%.1f MB)", dest_path, dest_path.stat().st_size / 1e6)
    return dest_path


def download_all_quarters(config: dict) -> list:
    """
    Download all FAERS quarters specified in the config.

    Parameters
    ----------
    config : dict
        Parsed config.yaml dictionary.

    Returns
    -------
    list[Path]
        List of paths to downloaded ZIP files.
    """
    base_url = config["faers"]["base_url"]
    quarters = config["faers"]["quarters"]
    raw_dir = Path(config["paths"]["raw"])

    zip_paths = []
    for quarter in quarters:
        zip_path = download_faers_zip(quarter, base_url, raw_dir)
        zip_paths.append(zip_path)

    logger.info("All downloads complete. %d ZIP file(s) ready.", len(zip_paths))
    return zip_paths
