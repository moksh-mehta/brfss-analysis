"""Download a BRFSS annual LLCP SAS transport (.XPT) file from the CDC.

Raw data is never committed. Run this script to fetch it into data/raw/.

The exact URL for each survey year is verified against the CDC "BRFSS Annual
Survey Data" page and recorded in KNOWN_FILES, so that the analysis is tied to a
specific, checkable file. Pass --url to override.

Usage:
    python scripts/download_brfss.py --year 2024
    python scripts/download_brfss.py --url https://.../LLCP2024XPT.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# year -> URL. Entries are added only after checking the CDC site (Stage 1).
KNOWN_FILES: dict[int, str] = {
    # Verified 2026-10-08 on https://www.cdc.gov/brfss/annual_data/annual_2025.html
    2025: "https://www.cdc.gov/brfss/annual_data/2025/files/LLCP2025XPT.zip",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, help="Survey year with a verified URL in KNOWN_FILES")
    parser.add_argument("--url", help="Explicit download URL (overrides --year)")
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    url = args.url or KNOWN_FILES.get(args.year)
    if not url:
        parser.error(f"No verified URL for year {args.year}; pass --url. Known years: {sorted(KNOWN_FILES)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.out_dir / url.rsplit("/", 1)[-1]
    print(f"Downloading {url}", file=sys.stderr)
    download(url, archive)

    extracted = []
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                # Guard against path traversal in archive member names. CDC member
                # names can carry trailing whitespace (e.g. "LLCP2025.XPT "), so strip it.
                target = (args.out_dir / Path(name).name.strip()).resolve()
                if target.parent != args.out_dir.resolve():
                    raise RuntimeError(f"Refusing suspicious archive member: {name}")
                with z.open(name) as src, target.open("wb") as dst:
                    dst.write(src.read())
                extracted.append(target)

    manifest = {
        "url": url,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archive": {"file": archive.name, "sha256": sha256(archive)},
        "extracted": [{"file": p.name, "sha256": sha256(p)} for p in extracted],
    }
    (args.out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
