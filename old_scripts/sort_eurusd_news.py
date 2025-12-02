import pandas as pd
import sys
from pathlib import Path

# ============================================
#   CONFIG
# ============================================

INPUT_FILE = "eurusd_news_filtered.csv"
OUTPUT_FILE = "eurusd_news_filtered_sorted.csv"
TIMESTAMP_COLUMN = "timestamp_utc"   # change this if your column name differs


# ============================================
#   SCRIPT
# ============================================

def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading: {input_path} ...")

    # Load with timestamp parsing
    try:
        df = pd.read_csv(
            input_path,
            parse_dates=[TIMESTAMP_COLUMN],
            dtype="string",  # prevents pandas from choking on weird GDELT strings
        )
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        sys.exit(1)

    print(f"Loaded {len(df):,} rows.")

    # Drop rows missing timestamp (rare but GDELT can be feral)
    missing = df[TIMESTAMP_COLUMN].isna().sum()
    if missing > 0:
        print(f"Dropping {missing:,} rows with missing timestamps.")
        df = df.dropna(subset=[TIMESTAMP_COLUMN])

    # Sort
    print("Sorting by timestamp...")
    df = df.sort_values(TIMESTAMP_COLUMN)

    # Save
    print(f"Writing sorted file to: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)

    print("Done.")
    print(f"Final row count: {len(df):,}")
    print(f"First timestamp: {df[TIMESTAMP_COLUMN].iloc[0]}")
    print(f"Last timestamp:  {df[TIMESTAMP_COLUMN].iloc[-1]}")


if __name__ == "__main__":
    main()