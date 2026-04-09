from __future__ import annotations

from pathlib import Path

import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parent

GENERATED_CSV = BASE_DIR / "generated_companies_calibrated.csv"
SETTLEMENT_WEIGHTS_XLSX = BASE_DIR / "ksh_outputs" / "settlement_weights.xlsx"

OUTPUT_SETTLEMENT_SUMMARY_XLSX = BASE_DIR / "settlement_distribution_summary.xlsx"
OUTPUT_TOP_COMPANIES_XLSX = BASE_DIR / "settlement_top_companies.xlsx"
OUTPUT_TOP_WORKERS_XLSX = BASE_DIR / "settlement_top_workers.xlsx"
OUTPUT_OUTLIERS_XLSX = BASE_DIR / "settlement_outliers.xlsx"

def normalize_bp_district(s: str) -> str:
    if not isinstance(s, str):
        return s

    s = s.lower().strip()

    if "budapest" in s:
        match = re.search(r"(\d{1,2})", s)
        if match:
            num = int(match.group(1))
            return f"budapest {num:02d} kerulet"

    return s


def normalize_settlement_name(s: str) -> str:
    if not isinstance(s, str):
        return s

    s = str(s).strip().lower()

    replacements = {
        "õ": "ő",
        "û": "ű",
        "ô": "ő",
        "ð": "ő",
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ö": "o",
        "ő": "o",
        "ú": "u",
        "ü": "u",
        "ű": "u",
    }

    for bad, good in replacements.items():
        s = s.replace(bad, good)

    s = " ".join(s.split())
    s = normalize_bp_district(s)
    return s

def load_generated_companies(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    required = ["settlement", "teaor", "bin", "company_index_in_cell", "company_size"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"A generated CSV-ből hiányoznak oszlopok: {missing}. "
            f"Elérhető: {list(df.columns)}"
        )

    df["settlement_original"] = df["settlement"].astype(str).str.strip()
    df["settlement"] = df["settlement_original"].map(normalize_settlement_name)
    df["company_size"] = pd.to_numeric(df["company_size"], errors="coerce").fillna(0).astype(int)
    return df


def load_settlement_weights(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)

    required = ["settlement", "county", "population", "population_share_country"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"A settlement_weights fájlból hiányoznak oszlopok: {missing}. "
            f"Elérhető: {list(df.columns)}"
        )

    df = df[["settlement", "county", "population", "population_share_country"]].copy()
    df["settlement_original"] = df["settlement"].astype(str).str.strip()
    df["settlement"] = df["settlement_original"].map(normalize_settlement_name)
    df["county"] = df["county"].astype(str).str.strip()
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0).astype(int)
    df["population_share_country"] = pd.to_numeric(
        df["population_share_country"], errors="coerce"
    ).fillna(0.0)

    df = (
        df.groupby("settlement", as_index=False)
        .agg(
            county=("county", "first"),
            population=("population", "sum"),
            population_share_country=("population_share_country", "sum"),
        )
    )

    return df

def build_settlement_summary(
    generated_df: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> pd.DataFrame:
    settlement_gen = (
        generated_df.groupby("settlement", as_index=False)
        .agg(
            generated_companies=("company_size", "size"),
            generated_workers=("company_size", "sum"),
            avg_company_size=("company_size", "mean"),
        )
    )

    df = settlement_gen.merge(weights_df, on="settlement", how="left")

    df["generated_companies"] = df["generated_companies"].fillna(0).astype(int)
    df["generated_workers"] = df["generated_workers"].fillna(0).astype(int)
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0).astype(int)
    df["population_share_country"] = pd.to_numeric(
        df["population_share_country"], errors="coerce"
    ).fillna(0.0)

    df["companies_per_1000_population"] = df.apply(
        lambda r: (r["generated_companies"] / r["population"] * 1000) if r["population"] > 0 else None,
        axis=1,
    )

    df["workers_per_1000_population"] = df.apply(
        lambda r: (r["generated_workers"] / r["population"] * 1000) if r["population"] > 0 else None,
        axis=1,
    )

    df["worker_to_company_ratio"] = df.apply(
        lambda r: (r["generated_workers"] / r["generated_companies"]) if r["generated_companies"] > 0 else None,
        axis=1,
    )

    # elvárt arányos cég- és worker-tömeg csak lakossági súly alapján
    total_companies = int(df["generated_companies"].sum())
    total_workers = int(df["generated_workers"].sum())

    df["expected_companies_by_population"] = (
        df["population_share_country"] * total_companies
    )
    df["expected_workers_by_population"] = (
        df["population_share_country"] * total_workers
    )

    df["company_deviation_vs_population"] = (
        df["generated_companies"] - df["expected_companies_by_population"]
    )
    df["worker_deviation_vs_population"] = (
        df["generated_workers"] - df["expected_workers_by_population"]
    )

    df["company_ratio_vs_population"] = df.apply(
        lambda r: (r["generated_companies"] / r["expected_companies_by_population"])
        if r["expected_companies_by_population"] > 0 else None,
        axis=1,
    )

    df["worker_ratio_vs_population"] = df.apply(
        lambda r: (r["generated_workers"] / r["expected_workers_by_population"])
        if r["expected_workers_by_population"] > 0 else None,
        axis=1,
    )

    df = df.sort_values("generated_workers", ascending=False)
    return df


def main() -> None:
    generated_df = load_generated_companies(GENERATED_CSV)
    weights_df = load_settlement_weights(SETTLEMENT_WEIGHTS_XLSX)

    summary_df = build_settlement_summary(generated_df, weights_df)

    top_companies_df = summary_df.sort_values("generated_companies", ascending=False).head(50).copy()
    top_workers_df = summary_df.sort_values("generated_workers", ascending=False).head(50).copy()

    # outlierek: legalább 1000 fős települések között
    filtered = summary_df[summary_df["population"] >= 1000].copy()

    outliers_df = filtered[
        [
            "settlement",
            "county",
            "population",
            "generated_companies",
            "generated_workers",
            "companies_per_1000_population",
            "workers_per_1000_population",
            "worker_to_company_ratio",
            "company_ratio_vs_population",
            "worker_ratio_vs_population",
        ]
    ].copy()

    outliers_df = outliers_df.sort_values(
        ["worker_ratio_vs_population", "generated_workers"],
        ascending=[False, False],
    )

    summary_df.to_excel(OUTPUT_SETTLEMENT_SUMMARY_XLSX, index=False, engine="openpyxl")
    top_companies_df.to_excel(OUTPUT_TOP_COMPANIES_XLSX, index=False, engine="openpyxl")
    top_workers_df.to_excel(OUTPUT_TOP_WORKERS_XLSX, index=False, engine="openpyxl")
    outliers_df.to_excel(OUTPUT_OUTLIERS_XLSX, index=False, engine="openpyxl")

    print("--- Settlement distribution validation ---")
    print(f"Generated company rows: {len(generated_df):,}")
    print(f"Settlement summary rows: {len(summary_df):,}")
    print(f"Total generated companies: {int(summary_df['generated_companies'].sum()):,}")
    print(f"Total generated workers: {int(summary_df['generated_workers'].sum()):,}")

    missing_population = summary_df["population"].isna().sum() + (summary_df["population"] == 0).sum()
    print(f"Settlements with missing/zero population: {missing_population:,}")

    print("\nTop 10 settlements by generated companies:")
    print(
        top_companies_df[
            ["settlement", "county", "population", "generated_companies", "generated_workers"]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 settlements by generated workers:")
    print(
        top_workers_df[
            ["settlement", "county", "population", "generated_companies", "generated_workers"]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(f"\nWrote: {OUTPUT_SETTLEMENT_SUMMARY_XLSX}")
    print(f"Wrote: {OUTPUT_TOP_COMPANIES_XLSX}")
    print(f"Wrote: {OUTPUT_TOP_WORKERS_XLSX}")
    print(f"Wrote: {OUTPUT_OUTLIERS_XLSX}")


if __name__ == "__main__":
    main()