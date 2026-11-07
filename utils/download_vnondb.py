#!/usr/bin/env python3
"""
Download and verification script for the HANDS-VNOnDB dataset.
Fetches the preprocessed VNOnDB line dataset directly from the authors' repository
and extracts image-text pairs into data/HANDS-VNONDB/lines/.
"""

import argparse
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/trcphm/vnondb-vt-lines"


def download_with_progress(url: str) -> bytes:
    """Download file with live progress reporting."""
    print(f"Connecting to {url}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; HTR-ConvText-Downloader)"}
    )
    with urllib.request.urlopen(req) as resp:
        total_size = resp.headers.get("Content-Length")
        total_bytes = int(total_size) if total_size else None
        
        buffer = io.BytesIO()
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunks

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            buffer.write(chunk)
            downloaded += len(chunk)
            if total_bytes:
                percent = downloaded / total_bytes * 100
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_bytes / (1024 * 1024)
                sys.stdout.write(f"\rDownloading: {mb_down:.1f}/{mb_total:.1f} MB ({percent:.1f}%)")
            else:
                mb_down = downloaded / (1024 * 1024)
                sys.stdout.write(f"\rDownloading: {mb_down:.1f} MB")
            sys.stdout.flush()

    print("\nDownload complete.")
    buffer.seek(0)
    return buffer.getvalue()


def extract_dataset(zip_bytes: bytes, target_dir: Path):
    """Extract files from zip archive into target directory."""
    print(f"Extracting dataset to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    lines_dir = target_dir / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        members = z.namelist()
        total_members = len(members)
        print(f"Archive contains {total_members} files.")

        extracted = 0
        for member in members:
            # We want all line files inside target_dir/lines/
            if member.startswith("lines/"):
                rel_path = member[len("lines/"):]
                if not rel_path:
                    continue
                dest = lines_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                extracted += 1
            elif member.endswith(".ln"):
                dest = target_dir / member
                # Keep existing split file if already present, otherwise extract
                if not dest.exists():
                    with z.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                extracted += 1

    print(f"Extracted {extracted} files successfully.")


def verify_splits(base_dir: Path) -> bool:
    """Verify that all files in train.ln, val.ln, and test.ln exist with pair .png and .txt."""
    lines_dir = base_dir / "lines"
    if not lines_dir.is_dir():
        print(f"Error: Lines directory does not exist: {lines_dir}")
        return False

    split_names = ["train.ln", "val.ln", "test.ln"]
    all_ok = True
    total_samples = 0

    print("\n--- Verifying Dataset Splits ---")
    for sname in split_names:
        sfile = base_dir / sname
        if not sfile.exists():
            print(f"Split file missing: {sfile}")
            all_ok = False
            continue

        with open(sfile, "r", encoding="utf-8") as f:
            items = [line.strip() for line in f if line.strip()]

        total_samples += len(items)
        missing_png = 0
        missing_txt = 0

        for item in items:
            img_path = lines_dir / item
            txt_path = lines_dir / (Path(item).stem + ".txt")

            if not img_path.exists():
                missing_png += 1
            if not txt_path.exists():
                missing_txt += 1

        status = "PASSED" if (missing_png == 0 and missing_txt == 0) else "FAILED"
        print(
            f"Split {sname:8s}: {len(items):4d} items | Missing PNG: {missing_png:3d} | Missing TXT: {missing_txt:3d} | [{status}]"
        )
        if missing_png > 0 or missing_txt > 0:
            all_ok = False

    print(f"Total verified samples across splits: {total_samples}")
    if all_ok:
        # Show a random/sample transcription
        sample_img = "20140603_0003_BCCTC_tg_0_0.png"
        sample_txt = lines_dir / (Path(sample_img).stem + ".txt")
        if sample_txt.exists():
            try:
                sample_content = sample_txt.read_text(encoding="utf-8").strip()
                print(f"\nSample transcription check ({sample_img}):")
                print(f"  --> \"{sample_content}\"")
            except Exception as e:
                print(f"Sample read error: {e}")
        print("\nAll splits verified successfully! Dataset is ready for training and inference.")
    else:
        print("\nWarning: Some files are missing or could not be verified.")

    return all_ok


def main():
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    parser = argparse.ArgumentParser(description="Download and verify HANDS-VNOnDB dataset for HTR-ConvText")
    parser.add_argument(
        "--target-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "HANDS-VNONDB"),
        help="Target directory to extract dataset (default: data/HANDS-VNONDB)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing dataset without downloading"
    )
    args = parser.parse_args()

    target_path = Path(args.target_dir).resolve()

    if args.verify_only:
        success = verify_splits(target_path)
        sys.exit(0 if success else 1)

    print(f"Target directory: {target_path}")
    zip_bytes = download_with_progress(DATASET_URL)
    extract_dataset(zip_bytes, target_path)
    success = verify_splits(target_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
