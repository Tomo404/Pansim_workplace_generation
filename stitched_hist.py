# stitched_hist.py
# ------------------------------
# Create "stitched" size histograms for a given TEÁOR from generated_companies.csv
# to inspect continuity across bin boundaries.
#
# Example:
#   python stitched_hist.py --csv generated_companies.csv --teaor 43 --out analysis_out
#   python stitched_hist.py --csv generated_companies.csv --top 10 --out analysis_out

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple, Optional

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def teaor_to_str(x) -> str:
    # handles "43", 43, 43.0
    s = str(x).strip()
    try:
        f = float(s.replace(",", "."))
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s

def make_bucket(size: int, fine_until: int, bucket_50_249: int, bucket_250_1000: int) -> str:
    if size <= fine_until:
        return str(size)

    if 50 <= size <= 249:
        lo = (size // bucket_50_249) * bucket_50_249
        if lo < 50: lo = 50
        hi = min(lo + bucket_50_249 - 1, 249)
        return f"{lo}-{hi}"

    # 250..1000
    lo = (size // bucket_250_1000) * bucket_250_1000
    if lo < 250: lo = 250
    hi = min(lo + bucket_250_1000 - 1, 1000)
    return f"{lo}-{hi}"

def bucket_sort_key(b: str) -> int:
    # "275-299" -> 275, "7" -> 7
    if "-" in b:
        return int(b.split("-")[0])
    return int(b)

def load_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        return None

def stitched_for_teaor(
    df,
    teaor: str,
    fine_until: int = 49,
    bucket_50_249: int = 10,
    bucket_250_1000: int = 50,
) -> List[Tuple[str, int, float]]:
    df["teaor"] = df["teaor"].apply(teaor_to_str)
    teaor = teaor_to_str(teaor)

    sub = df[df["teaor"] == teaor]
    if sub.empty:
        return []

    sizes = sub["company_size"].astype(int)
    # build bucket counts
    counts: Dict[str, int] = {}
    for s in sizes:
        b = make_bucket(int(s), fine_until, bucket_50_249, bucket_250_1000)
        counts[b] = counts.get(b, 0) + 1

    total = sum(counts.values())
    out = []
    for b in sorted(counts.keys(), key=bucket_sort_key):
        c = counts[b]
        out.append((b, c, (c / total) if total else 0.0))
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="generated_companies.csv")
    ap.add_argument("--out", default="analysis_out", help="output directory")
    ap.add_argument("--teaor", default=None, help="single TEÁOR code (e.g. 43)")
    ap.add_argument("--top", type=int, default=0, help="if >0, export stitched hist for top N TEÁOR by count")
    ap.add_argument("--fine-until", type=int, default=49, help="use 1-sized buckets up to this value")
    ap.add_argument("--bucket-50-249", type=int, default=10, help="bucket width for 50..249")
    ap.add_argument("--bucket-250-1000", type=int, default=50, help="bucket width for 250..1000")
    args = ap.parse_args()

    pd = load_pandas()
    if pd is None:
        raise RuntimeError("pandas is required. Please: pip install pandas openpyxl")

    ensure_outdir(args.out)

    df = pd.read_csv(args.csv)
    df["teaor"] = df["teaor"].apply(teaor_to_str)
    df["company_size"] = df["company_size"].astype(int)

    # Determine which TEÁORs to export
    teaors: List[str] = []
    if args.teaor:
        teaors = [teaor_to_str(args.teaor)]
    elif args.top and args.top > 0:
        top = (
            df.groupby("teaor")["company_size"]
            .size()
            .sort_values(ascending=False)
            .head(args.top)
            .index.tolist()
        )
        teaors = [str(t) for t in top]
    else:
        raise RuntimeError("Provide either --teaor <code> or --top <N>")

    for tea in teaors:
        rows = stitched_for_teaor(
            df,
            teaor=tea,
            fine_until=args.fine_until,
            bucket_50_249=args.bucket_50_249,
            bucket_250_1000=args.bucket_250_1000,
        )
        if not rows:
            print(f"TEÁOR {tea}: no rows")
            continue

        out_path = os.path.join(args.out, f"stitched_teaor_{tea}.csv")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("bucket,count,ratio\n")
            for b, c, r in rows:
                f.write(f"{b},{c},{r:.8f}\n")

        print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
