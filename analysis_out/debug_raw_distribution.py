import pandas as pd

CSV_PATH = "generated_companies_calibrated.csv"

df = pd.read_csv("generated_companies_calibrated.csv")

teaor = "68"
sub = df[df["teaor"].astype(str) == teaor].copy()

raw = (
    sub.groupby("company_size")
    .size()
    .reset_index(name="count")
    .sort_values("company_size")
)

print(raw.to_string(index=False))