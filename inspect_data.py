"""Initial Data Inspection — Dubai Real Estate Dataset
Before performing any cleaning or modelling, it is essential to gain
a preliminary understanding of the raw dataset. This script examines
the structure of 'realestatedata.csv', reporting the dimensionality
of the data, the data type of each feature, the extent of missing
values per column, and a representative sample value. This diagnostic
step informs the subsequent cleaning strategy in clean_dubai_realestate.py.
"""
import pandas as pd

# Load the raw dataset directly from the source file.
# The 'on_bad_lines="skip"' parameter ensures that any malformed rows
# (e.g., inconsistent delimiters or encoding issues) are silently
# discarded rather than causing the script to fail entirely.
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join('realestatedata.csv'), on_bad_lines='skip')

# Report the overall dimensionality of the dataset.
# This gives an immediate sense of the scale of the data we are
# working with and  how many observations and how many attributes.
print(f"Shape: {df.shape}")

# Iterate over each column to produce a summary.
# For every column we record three key diagnostics:
#   (1) dtype
#   (2) nulls
#   (3) sample
print(f"\nColumns ({len(df.columns)}):")
for c in df.columns:
    nulls = df[c].isnull().sum()
    dtype = df[c].dtype

    # Extract the first non-null value as a sample for human inspection.
    sample = str(df[c].dropna().iloc[0])[:60] if len(df[c].dropna()) > 0 else "ALL NULL"
    print(f"  [{c}] dtype={dtype}, nulls={nulls}, sample='{sample}'")
