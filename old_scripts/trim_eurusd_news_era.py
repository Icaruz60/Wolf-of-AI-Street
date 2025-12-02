import pandas as pd

INPUT_PATH = "data/eurusd_merged.csv"
OUTPUT_PATH = "data/eurusd_merged_news_only.csv"

# First row with real news:
# 2020-01-01 17:00:00+00:00,1.1212,1.12166,1.12106,1.12143,9.0,...
START_TS = "2020-01-01 17:00:00+00:00"


def main():
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)

    if "timestamp_utc" not in df.columns:
        raise ValueError(f"'timestamp_utc' column not found. Columns: {list(df.columns)}")

    # Parse timestamps with timezone
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])

    total_rows = len(df)
    print(f"Total rows before trim: {total_rows}")

    start_ts = pd.Timestamp(START_TS)
    df_trim = df[df["timestamp_utc"] >= start_ts].copy().reset_index(drop=True)

    trimmed_rows = len(df_trim)
    print(f"Rows after trim (>= {START_TS}): {trimmed_rows}")
    if trimmed_rows == 0:
        raise ValueError("Trimmed dataset is empty. Check START_TS and file contents.")

    # Optional sanity: show first few rows
    print("First 3 rows after trim:")
    print(df_trim.head(3))

    print(f"Saving trimmed file to {OUTPUT_PATH} ...")
    df_trim.to_csv(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
