from __future__ import annotations

import pandas as pd
from pathlib import Path

from data_loader import load_counts_from_tensor_files

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent

GENERATED_CSV = BASE_DIR / "generated_companies_calibrated.csv"
TENSOR_PATH = BASE_DIR / "ksh_outputs" / "tensor_ksh_dedup.npy"
DIMS_PATH = BASE_DIR / "ksh_outputs" / "tensor_ksh_dedup_dimensions.json"
CALIBRATION_PATH = BASE_DIR / "ksh_outputs" / "calibration_table.xlsx"
IMPUTATION_SUMMARY_PATH = BASE_DIR / "imputation_summary_by_teaor_bin.xlsx"

OUTDIR = BASE_DIR / "validation_out"


# =========================
# LOADERS
# =========================
def load_generated(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["company_size"] = pd.to_numeric(df["company_size"], errors="coerce").fillna(0)
    return df


def load_target_workers(calibration_path: Path) -> dict[str, int]:
    df = pd.read_excel(calibration_path)

    df["teaor_alag"] = (
        df["teaor_alag"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.lstrip("0")
    )
    df["ksh_worker_total"] = pd.to_numeric(df["ksh_worker_total"], errors="coerce")

    df = df.dropna(subset=["teaor_alag", "ksh_worker_total"])

    return {
        row["teaor_alag"]: int(row["ksh_worker_total"])
        for _, row in df.iterrows()
    }

def load_imputation_targets(imputation_path: Path) -> pd.DataFrame:
    df = pd.read_excel(imputation_path)

    required = ["teaor", "bin", "target_company_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Az imputációs summary-ból hiányoznak oszlopok: {missing}. "
            f"Elérhető oszlopok: {list(df.columns)}"
        )

    df["teaor"] = df["teaor"].astype(str)
    df["bin"] = df["bin"].astype(str)
    df["target_company_count"] = pd.to_numeric(
        df["target_company_count"], errors="coerce"
    ).fillna(0)

    return df[["teaor", "bin", "target_company_count"]].copy()

# =========================
# VALIDATION
# =========================
def validate_workers(df: pd.DataFrame, targets: dict[str, int]) -> pd.DataFrame:
    df["teaor"] = df["teaor"].astype(str)

    agg = df.groupby("teaor")["company_size"].sum().reset_index()
    agg.rename(columns={"company_size": "generated_workers"}, inplace=True)

    target_df = pd.DataFrame([
        {"teaor": t, "target_workers": v}
        for t, v in targets.items()
    ])

    merged = agg.merge(target_df, on="teaor", how="outer")

    merged["generated_workers"] = merged["generated_workers"].fillna(0)
    merged["target_workers"] = merged["target_workers"].fillna(0)

    merged["difference"] = merged["generated_workers"] - merged["target_workers"]
    merged["ratio"] = merged["generated_workers"] / merged["target_workers"].replace(0, 1)

    return merged.sort_values("teaor")


def validate_company_counts(df: pd.DataFrame, counts: dict) -> pd.DataFrame:
    # generated counts
    gen_counts = (
        df.groupby(["settlement", "teaor", "bin"])
        .size()
        .reset_index(name="generated_count")
    )

    # tensor counts
    rows = []
    df["teaor"] = df["teaor"].astype(str)
    for key, val in counts.items():
        rows.append({
            "settlement": key.settlement,
            "teaor": str(key.teaor),
            "bin": key.bin_name,
            "input_count": val
        })
    tensor_df = pd.DataFrame(rows)

    merged = gen_counts.merge(
        tensor_df,
        on=["settlement", "teaor", "bin"],
        how="outer"
    )

    merged["generated_count"] = merged["generated_count"].fillna(0)
    merged["input_count"] = merged["input_count"].fillna(0)

    merged["difference"] = merged["generated_count"] - merged["input_count"]

    return merged


def validate_teaor_bin(df: pd.DataFrame, imputation_targets: pd.DataFrame) -> pd.DataFrame:
    df["teaor"] = df["teaor"].astype(str)

    generated = (
        df.groupby(["teaor", "bin"])
        .size()
        .reset_index(name="generated")
    )

    merged = generated.merge(
        imputation_targets.rename(columns={"target_company_count": "target"}),
        on=["teaor", "bin"],
        how="outer",
    )

    merged["generated"] = merged["generated"].fillna(0)
    merged["target"] = merged["target"].fillna(0)

    merged["difference"] = merged["generated"] - merged["target"]

    merged["ratio"] = merged.apply(
        lambda r: (r["generated"] / r["target"]) if r["target"] > 0 else None,
        axis=1,
    )

    return merged.sort_values(["teaor", "bin"])

def validate_settlement(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("settlement").agg(
        companies=("company_size", "count"),
        workers=("company_size", "sum")
    ).reset_index()

    return agg.sort_values("companies", ascending=False)


# =========================
# MAIN
# =========================
def main():
    OUTDIR.mkdir(exist_ok=True)

    print("Loading data...")
    df = load_generated(GENERATED_CSV)
    counts = load_counts_from_tensor_files(
        tensor_npy_path=str(TENSOR_PATH),
        dimensions_json_path=str(DIMS_PATH),
        drop_zeros=True,
    )
    targets = load_target_workers(CALIBRATION_PATH)
    imputation_targets = load_imputation_targets(IMPUTATION_SUMMARY_PATH)

    print("\n--- Worker validation ---")
    worker_df = validate_workers(df, targets)
    print(worker_df.head())

    print("\n--- Company count validation ---")
    comp_df = validate_company_counts(df, counts)
    print(comp_df.head())

    print("\n--- TEÁOR x bin validation (vs imputation targets) ---")
    tb_df = validate_teaor_bin(df, imputation_targets)
    print(tb_df.head())

    print("\n--- Settlement validation ---")
    st_df = validate_settlement(df)
    print(st_df.head())

    # SAVE
    worker_df.to_excel(OUTDIR / "workers_validation.xlsx", index=False)
    comp_df.to_excel(OUTDIR / "cell_validation.xlsx", index=False)
    tb_df.to_excel(OUTDIR / "teaor_bin_validation.xlsx", index=False)
    st_df.to_excel(OUTDIR / "settlement_summary.xlsx", index=False)

    print(f"\nSaved to: {OUTDIR}")


if __name__ == "__main__":
    main()