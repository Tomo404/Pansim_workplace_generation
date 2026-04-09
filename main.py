# main.py
from __future__ import annotations

from typing import Dict, List
from collections import defaultdict
import csv
import math

import pandas as pd

from data_loader import load_counts_from_tensor_files
from generator import CellKey, generate_companies_from_counts


# =========================
# CONFIG
# =========================
TENSOR_PATH = "ksh_outputs/tensor_ksh_dedup.npy"
DIMS_PATH = "ksh_outputs/tensor_ksh_dedup_dimensions.json"
CALIBRATION_XLSX = "ksh_outputs/calibration_table.xlsx"

ENABLE_COUNT_SCALING = False
TARGET_WORKERS = 3_900_000
BASE_WORKERS = 3_900_000

OUTPUT_CSV = "generated_companies.csv"
OUTPUT_SUMMARY_XLSX = "generation_summary_by_teaor.xlsx"

SEED = 12345
MAX_CELLS = None

DEFAULT_PROFILE = "decay_mild"
TEAOR_WHITELIST = None

BIN_ORDER = ["1-4", "5-9", "10-19", "20-49", "50-249", "250-1000"]
BIN_LOWER = {
    "1-4": 1,
    "5-9": 5,
    "10-19": 10,
    "20-49": 20,
    "50-249": 50,
    "250-1000": 250,
}
BIN_UPPER = {
    "1-4": 4,
    "5-9": 9,
    "10-19": 19,
    "20-49": 49,
    "50-249": 249,
    "250-1000": 1000,
}
CALIB_BIN_COLS = {
    "1-4": "company_count_1_4",
    "5-9": "company_count_5_9",
    "10-19": "company_count_10_19",
    "20-49": "company_count_20_49",
    "50-249": "company_count_50_249",
    "250-1000": "company_count_250_plus",
}


def normalize_teaor_code(value) -> str | None:
    if pd.isna(value):
        return None

    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = s.lstrip("0")
    return s if s else "0"


def flatten_generated(
    generated: Dict[CellKey, List[int]]
) -> List[dict]:
    rows: List[dict] = []

    for key, sizes in generated.items():
        for idx, size in enumerate(sizes):
            rows.append(
                {
                    "settlement": key.settlement,
                    "teaor": key.teaor,
                    "bin": key.bin_name,
                    "company_index_in_cell": idx,
                    "company_size": int(size),
                }
            )

    return rows


def scale_counts_proportionally(
    counts: Dict[CellKey, int],
    scale: float,
) -> Dict[CellKey, int]:
    if scale <= 0:
        raise ValueError("scale must be > 0")

    items = list(counts.items())

    scaled_floor = {}
    remainders = []

    for key, value in items:
        expected = value * scale
        base = int(math.floor(expected))
        scaled_floor[key] = base
        remainders.append((key, expected - base))

    target_total = int(round(sum(counts.values()) * scale))
    current_total = sum(scaled_floor.values())
    missing = target_total - current_total

    remainders.sort(key=lambda x: x[1], reverse=True)

    for i in range(missing):
        key = remainders[i][0]
        scaled_floor[key] += 1

    return scaled_floor


def write_csv(rows: List[dict], out_path: str) -> None:
    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_profiles_from_calibration_table(
    calibration_xlsx: str,
) -> tuple[Dict[str, str], Dict[tuple[str, str], str], Dict[tuple[str, str, str], int], pd.DataFrame]:
    df = pd.read_excel(calibration_xlsx)

    df["teaor_alag"] = df["teaor_alag"].map(normalize_teaor_code)
    df["target_avg_size_ksh"] = pd.to_numeric(df["target_avg_size_ksh"], errors="coerce")
    df["company_count_total"] = pd.to_numeric(df["company_count_total"], errors="coerce").fillna(0)

    for col in CALIB_BIN_COLS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df[df["teaor_alag"].notna()].copy()

    profile_by_teaor: Dict[str, str] = {}
    profile_by_teaor_bin: Dict[tuple[str, str], str] = {}
    window_by_boundary: Dict[tuple[str, str, str], int] = {}

    summary_rows = []

    SUPPORT_MIN = 0.01
    R_STRONG = 0.15
    R_ULTRA = 0.05

    for _, row in df.iterrows():
        teaor = row["teaor_alag"]
        company_total = int(row["company_count_total"])
        target_avg = row["target_avg_size_ksh"]

        if company_total <= 0 or pd.isna(target_avg):
            continue

        min_possible = 0
        max_possible = 0
        bin_counts = {}

        for b in BIN_ORDER:
            n = int(row[CALIB_BIN_COLS[b]])
            bin_counts[b] = n
            min_possible += n * BIN_LOWER[b]
            max_possible += n * BIN_UPPER[b]

        min_avg = min_possible / company_total if company_total > 0 else 0.0
        max_avg = max_possible / company_total if company_total > 0 else 0.0

        if max_avg <= min_avg:
            position = 0.5
        else:
            position = (target_avg - min_avg) / (max_avg - min_avg)
            position = max(0.0, min(1.0, position))

        # TEÁOR szintű alap profile választás
        if position < 0.15:
            base_profile = "decay_ultra"
        elif position < 0.30:
            base_profile = "decay_strong"
        elif position < 0.45:
            base_profile = "decay_mild"
        elif position < 0.60:
            base_profile = "uniform"
        elif position < 0.80:
            base_profile = "exp_mild"
        else:
            base_profile = "exp_strong"

        profile_by_teaor[teaor] = base_profile

        for b in BIN_ORDER:
            profile_by_teaor_bin[(teaor, b)] = base_profile

        # bin-határ simítás a calibration table binarányaiból
        for i in range(len(BIN_ORDER) - 1):
            b_lo = BIN_ORDER[i]
            b_hi = BIN_ORDER[i + 1]

            n_lo = bin_counts[b_lo]
            n_hi = bin_counts[b_hi]

            p_lo = (n_lo / company_total) if company_total > 0 else 0.0
            p_hi = (n_hi / company_total) if company_total > 0 else 0.0

            if p_hi < SUPPORT_MIN or p_lo <= 0:
                continue

            r = p_hi / p_lo

            if r < R_ULTRA:
                profile_by_teaor_bin[(teaor, b_lo)] = "decay_ultra"
                profile_by_teaor_bin[(teaor, b_hi)] = "exp_ultra"
                window_by_boundary[(teaor, b_lo, b_hi)] = 1

            elif r < R_STRONG:
                profile_by_teaor_bin[(teaor, b_lo)] = "decay_strong"
                profile_by_teaor_bin[(teaor, b_hi)] = "exp_strong"
                window_by_boundary[(teaor, b_lo, b_hi)] = 1

            else:
                # csak akkor írjuk felül, ha még nem extrémebb valami
                if profile_by_teaor_bin[(teaor, b_lo)] not in {"decay_ultra", "decay_strong"}:
                    profile_by_teaor_bin[(teaor, b_lo)] = "decay_mild"
                if profile_by_teaor_bin[(teaor, b_hi)] not in {"exp_ultra", "exp_strong"}:
                    profile_by_teaor_bin[(teaor, b_hi)] = "exp_mild"
                window_by_boundary[(teaor, b_lo, b_hi)] = 2

        summary_rows.append(
            {
                "teaor": teaor,
                "company_count_total": company_total,
                "target_avg_size_ksh": float(target_avg),
                "min_possible_avg": float(min_avg),
                "max_possible_avg": float(max_avg),
                "target_position_0_1": float(position),
                "chosen_base_profile": base_profile,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("teaor")
    return profile_by_teaor, profile_by_teaor_bin, window_by_boundary, summary_df


def build_generation_summary(rows: List[dict], profile_summary_df: pd.DataFrame) -> pd.DataFrame:
    if not rows:
        return profile_summary_df.copy()

    generated_df = pd.DataFrame(rows)

    gen_summary = (
        generated_df.groupby("teaor", as_index=False)
        .agg(
            generated_companies=("company_size", "size"),
            generated_workers=("company_size", "sum"),
            generated_avg_company_size=("company_size", "mean"),
        )
    )

    if profile_summary_df.empty:
        return gen_summary.sort_values("teaor")

    out = profile_summary_df.merge(gen_summary, on="teaor", how="left")
    out["generated_companies"] = out["generated_companies"].fillna(0).astype(int)
    out["generated_workers"] = out["generated_workers"].fillna(0).astype(int)
    out["generated_avg_company_size"] = pd.to_numeric(out["generated_avg_company_size"], errors="coerce")

    out["generated_minus_target_avg"] = (
        out["generated_avg_company_size"] - out["target_avg_size_ksh"]
    )

    return out.sort_values("teaor")


def main() -> None:
    counts = load_counts_from_tensor_files(
        tensor_npy_path=TENSOR_PATH,
        dimensions_json_path=DIMS_PATH,
        drop_zeros=True,
    )

    if ENABLE_COUNT_SCALING:
        scale = TARGET_WORKERS / BASE_WORKERS
        original_total = sum(counts.values())
        counts = scale_counts_proportionally(counts, scale)
        scaled_total = sum(counts.values())

        print(f"Scaling enabled: factor = {scale:.6f}")
        print(f"Original organization count total: {original_total:,}")
        print(f"Scaled organization count total:   {scaled_total:,}")

    profile_by_teaor, profile_by_teaor_bin, window_by_boundary, profile_summary_df = (
        load_profiles_from_calibration_table(CALIBRATION_XLSX)
    )

    if TEAOR_WHITELIST:
        counts = {k: v for k, v in counts.items() if k.teaor in TEAOR_WHITELIST}
        profile_by_teaor = {k: v for k, v in profile_by_teaor.items() if k in TEAOR_WHITELIST}
        profile_by_teaor_bin = {
            k: v for k, v in profile_by_teaor_bin.items() if k[0] in TEAOR_WHITELIST
        }
        window_by_boundary = {
            k: v for k, v in window_by_boundary.items() if k[0] in TEAOR_WHITELIST
        }
        profile_summary_df = profile_summary_df[profile_summary_df["teaor"].isin(TEAOR_WHITELIST)].copy()

    if MAX_CELLS is not None:
        counts_items = list(counts.items())[:MAX_CELLS]
        counts = dict(counts_items)
        print(f"DEBUG: limiting to {len(counts):,} cells")

    print(f"Loaded non-zero cells: {len(counts):,}")
    print(f"Loaded TEÁOR profiles from calibration table: {len(profile_by_teaor):,}")

    generated = generate_companies_from_counts(
        counts=counts,
        profile_by_teaor=profile_by_teaor,
        profile_by_teaor_bin=profile_by_teaor_bin,
        default_profile=DEFAULT_PROFILE,
        seed=SEED,
        window_by_boundary=window_by_boundary,
    )

    total_companies = sum(len(v) for v in generated.values())
    print(f"Generated companies: {total_companies:,}")

    total_workers = sum(sum(v) for v in generated.values())
    print(f"Generated workers: {total_workers:,}")

    rows = flatten_generated(generated)
    write_csv(rows, OUTPUT_CSV)
    print(f"Wrote: {OUTPUT_CSV}")

    summary_df = build_generation_summary(rows, profile_summary_df)
    summary_df.to_excel(OUTPUT_SUMMARY_XLSX, index=False, engine="openpyxl")
    print(f"Wrote: {OUTPUT_SUMMARY_XLSX}")


if __name__ == "__main__":
    main()