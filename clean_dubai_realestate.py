"""Dubai Real Estate Dataset - Data Cleaning & Preparation for Regression
Target variable: price (log-transformed)"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#1. Load data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "DATA", "realestatedata.csv"), on_bad_lines="skip")
original_shape = df.shape
original_columns = df.columns.tolist()
print(f"[OK] Loaded dataset: {original_shape[0]} rows x {original_shape[1]} columns")
print(f"     Columns: {original_columns}\n")

#2. Drop irrelevant columns FIRST (IDs, URLs, text descriptions)
# Must drop ID column before dedup so duplicates are detected correctly
cols_to_drop = []
if "Unnamed: 0" in df.columns:
    cols_to_drop.append("Unnamed: 0")
for c in df.columns:
    if any(tag in c.lower() for tag in ["url", "link", "description"]):
        cols_to_drop.append(c)
cols_to_drop = list(set(cols_to_drop))
df = df.drop(columns=cols_to_drop, errors="ignore")
print(f"[OK] Dropped irrelevant columns: {cols_to_drop}")
print(f"     Remaining columns: {df.columns.tolist()}\n")

#3. Remove duplicate rows (now that ID column is removed)
dups = df.duplicated().sum()
df = df.drop_duplicates()
print(f"[OK] Removed {dups} duplicate rows  ->  {len(df)} rows remain\n")

#4. Convert price to numeric (strip commas / currency symbols)
df["price"] = (
    df["price"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("AED", "", regex=False)
    .str.replace("$", "", regex=False)
    .str.strip()
)
df["price"] = pd.to_numeric(df["price"], errors="coerce")
print(f"[OK] Converted 'price' to numeric")

#5. Remove rows where price is zero or null
before = len(df)
df = df[(df["price"].notna()) & (df["price"] > 0)]
removed_price = before - len(df)
print(f"[OK] Removed {removed_price} rows with price = 0 or null  ->  {len(df)} rows remain")

#6. Convert 'no.beds' to numeric if possible
df["no.beds"] = (
    df["no.beds"]
    .astype(str)
    .str.strip()
    .str.replace("+", "", regex=False)
)
df["no.beds"] = pd.to_numeric(df["no.beds"], errors="coerce")
print(f"[OK] Converted 'no.beds' to numeric (coerced non-numeric -> NaN)")

#7. Handle missing values
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
numeric_impute = [c for c in numeric_cols if c != "price"]

for c in numeric_impute:
    nulls = df[c].isnull().sum()
    if nulls > 0:
        median_val = df[c].median()
        df[c] = df[c].fillna(median_val)
        print(f"     Filled {nulls} nulls in '{c}' with median = {median_val}")

for c in categorical_cols:
    nulls = df[c].isnull().sum()
    if nulls > 0:
        mode_val = df[c].mode()[0]
        df[c] = df[c].fillna(mode_val)
        print(f"     Filled {nulls} nulls in '{c}' with mode = '{mode_val}'")

print(f"[OK] Missing values handled  (remaining nulls: {df.isnull().sum().sum()})\n")

#8. Detect & remove extreme outliers in price using IQR (iterative)
total_outliers_removed = 0
iteration = 0
while True:
    iteration += 1
    Q1 = df["price"].quantile(0.25)
    Q3 = df["price"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(df)
    df = df[(df["price"] >= lower) & (df["price"] <= upper)]
    removed_this_round = before - len(df)
    total_outliers_removed += removed_this_round
    if iteration == 1:
        print(f"[OK] IQR outlier removal on 'price' (iterative):")
        print(f"     Initial Q1={Q1:,.0f}  Q3={Q3:,.0f}  IQR={IQR:,.0f}")
        print(f"     Initial Bounds: [{lower:,.0f}, {upper:,.0f}]")
    if removed_this_round == 0:
        break
    print(f"     Pass {iteration}: removed {removed_this_round} outliers  ->  {len(df)} rows remain")
outliers_removed = total_outliers_removed
print(f"     Total outliers removed: {total_outliers_removed} across {iteration} passes")
print(f"     Final price range: [{df['price'].min():,.0f}, {df['price'].max():,.0f}]")
print(f"     Rows remaining: {len(df)}\n")

#9. Remove any residual duplicates after outlier removal
dups2 = df.duplicated().sum()
if dups2 > 0:
    df = df.drop_duplicates()
    print(f"[OK] Removed {dups2} residual duplicate rows  ->  {len(df)} rows remain\n")

#10. Convert categorical variables to category dtype
categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
for c in categorical_cols:
    df[c] = df[c].astype(str).str.strip().astype("category")
print(f"[OK] Converted {len(categorical_cols)} columns to category dtype: {categorical_cols}")

#11. One-hot encode categorical variables 
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
print(f"[OK] Applied one-hot encoding  ->  {df.shape[1]} columns total\n")

#12. Final dedup after encoding (identical feature rows)
dups3 = df.duplicated().sum()
if dups3 > 0:
    df = df.drop_duplicates()
    print(f"[OK] Removed {dups3} post-encoding duplicate rows  ->  {len(df)} rows remain\n")

#13. Log-transform the price variable 
df["log_price"] = np.log1p(df["price"])
print(f"[OK] Created 'log_price' = log(1 + price)")
print(f"     Original price skewness : {df['price'].skew():.4f}")
print(f"     Log price skewness      : {df['log_price'].skew():.4f}\n")

#14. Final verification checks
print("--- SELF-VERIFICATION ---")
assert df.isnull().sum().sum() == 0, "FAIL: Nulls remain"
print("[CHECK] No missing values")
assert df.duplicated().sum() == 0, "FAIL: Duplicates remain"
print("[CHECK] No duplicate rows")
assert (df["price"] <= 0).sum() == 0, "FAIL: Non-positive prices"
print("[CHECK] All prices are positive")
Q1f = df["price"].quantile(0.25)
Q3f = df["price"].quantile(0.75)
IQRf = Q3f - Q1f
outliers_remain = ((df["price"] < Q1f - 1.5*IQRf) | (df["price"] > Q3f + 1.5*IQRf)).sum()
assert outliers_remain == 0, f"FAIL: {outliers_remain} IQR outliers remain"
print("[CHECK] No IQR outliers in price")
obj_cols = df.select_dtypes(include=["object","string","category"]).columns.tolist()
assert len(obj_cols) == 0, f"FAIL: Non-numeric columns: {obj_cols}"
print("[CHECK] All columns are numeric (ML-ready)")
print("--- ALL CHECKS PASSED ---\n")

#15. Summary
print("=" * 60)
print("                      SUMMARY")
print("=" * 60)
print(f"  Original shape        : {original_shape}")
print(f"  Final shape           : {df.shape}")
print(f"  Number of features    : {df.shape[1] - 2}  (excl. price & log_price)")
print(f"  Duplicates removed    : {dups + dups2 + dups3}")
print(f"  Columns removed       : {cols_to_drop}")
print(f"  Rows removed (price)  : {removed_price}")
print(f"  Outliers removed      : {outliers_removed}")
print(f"  Transformations       :")
print(f"    - Dropped irrelevant ID column")
print(f"    - Removed duplicate rows")
print(f"    - Converted 'price' to numeric (stripped symbols)")
print(f"    - Converted 'no.beds' to numeric")
print(f"    - Imputed numeric nulls with median")
print(f"    - Imputed categorical nulls with mode")
print(f"    - IQR-based outlier removal on 'price'")
print(f"    - One-hot encoding on categorical columns")
print(f"    - Log-transform: log_price = log(1 + price)")
print("=" * 60)

#16. Show first 5 rows
print("\nFirst 5 rows (key numeric columns):\n")
key_cols = ["price", "no.beds", "size.sq.meter", "log_price"]
print(df[key_cols].head(5).to_string())

#17. Save cleaned dataset
output_path = os.path.join(BASE_DIR, "DATA", "cleaned_dubai_realestate.csv")
df.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved to: {output_path}")
