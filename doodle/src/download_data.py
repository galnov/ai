"""
Data collection: downloading drawings from Google's Quick, Draw! dataset.

    python -m src.download_data                 # 10,000 drawings per category (about 330MB)
    python -m src.download_data --per-class 4000   # less data - faster download and training

Each category has a huge file in the dataset (over 100,000 drawings, 100MB and more). We don't download all of it:
we read only the header of the npy file and the first N rows, and then close the connection.
Each row = one 28x28 drawing = 784 bytes.
"""
import argparse
import ast
import ssl
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .config import CATEGORIES, IMG_SIZE, PER_CLASS, QUICKDRAW_URL, RAW_DIR, file_name

ROW = IMG_SIZE * IMG_SIZE


def ssl_context():
    """On a Mac, Python sometimes doesn't know the security certificates - certifi fixes that."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def read_exactly(resp, n):
    chunks, got = [], 0
    while got < n:
        chunk = resp.read(min(1 << 20, n - got))
        if not chunk:
            break
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def parse_npy_header(resp):
    """Reads the header of an npy file and returns (number of rows, row length)."""
    magic = read_exactly(resp, 8)
    if magic[:6] != b"\x93NUMPY":
        raise ValueError("The received file is not an npy file")
    major = magic[6]
    hlen_bytes = read_exactly(resp, 2 if major == 1 else 4)
    hlen = int.from_bytes(hlen_bytes, "little")
    header = ast.literal_eval(read_exactly(resp, hlen).decode("latin1"))
    if header.get("descr") not in ("|u1", "u1", "<u1") or header.get("fortran_order"):
        raise ValueError(f"Unexpected format: {header}")
    rows, cols = header["shape"]
    return rows, cols


def download_category(name, n, base_url=QUICKDRAW_URL, retries=3):
    out = RAW_DIR / f"{file_name(name)}.npy"
    if out.exists():
        arr = np.load(out, mmap_mode="r")
        if len(arr) >= n:
            return name, len(arr), "exists"
    url = base_url.format(urllib.parse.quote(name))
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60, context=ssl_context()) as resp:
                rows, cols = parse_npy_header(resp)
                if cols != ROW:
                    raise ValueError(f"Expected {ROW} columns, got {cols}")
                take = min(n, rows)
                data = read_exactly(resp, take * ROW)
            take = len(data) // ROW
            arr = np.frombuffer(data[:take * ROW], dtype=np.uint8).reshape(take, ROW)
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            np.save(out, arr)
            return name, take, "downloaded"
        except Exception as e:  # unstable network - try again
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{name}: {last_err}")


def main():
    ap = argparse.ArgumentParser(description="Download drawings from Quick, Draw!")
    ap.add_argument("--per-class", type=int, default=PER_CLASS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-url", default=QUICKDRAW_URL, help=argparse.SUPPRESS)
    args = ap.parse_args()

    names = [c[0] for c in CATEGORIES]
    mb = len(names) * args.per_class * ROW / 1e6
    print(f"Downloading {args.per_class:,} drawings for each of {len(names)} categories (about {mb:.0f}MB)...\n")
    failed = []
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(download_category, n, args.per_class, args.base_url): n for n in names}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                name, count, status = f.result()
                print(f"  [{i:2d}/{len(names)}] {name:14s} {count:7,} drawings ({status})")
            except Exception as e:
                failed.append((futures[f], str(e)))
                print(f"  [{i:2d}/{len(names)}] {futures[f]:14s} failed: {e}")
    print(f"\nFinished in {time.time() - t0:.0f} seconds. The files are in {RAW_DIR}")
    if failed:
        print(f"[!] Failed: {', '.join(n for n, _ in failed)}. Run the command again - what was already downloaded is kept.")
        if any("CERTIFICATE" in err for _, err in failed):
            print("    On a Mac: run 'Install Certificates.command' from the Python folder, or  pip install certifi")
    else:
        print("Next step:  python -m src.explore_data   then   python -m src.train")


if __name__ == "__main__":
    main()
