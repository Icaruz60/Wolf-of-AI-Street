import os
import glob

import pandas as pd


RAW_DIR = "data/raw_eurusd_m1"
OUT_PATH = "data/eurusd_hourly.csv"


def load_one_file(path: str) -> pd.DataFrame:
    """
    Load a single HistData Generic ASCII M1 file and return a DataFrame with:
      time, open, high, low, close, volume
    at 1-minute resolution.
    """
    print(f"Loading {path}...")

    # HistData Generic ASCII M1:
    # 20120201 000000;1.306600;1.306600;1.306560;1.306560;0
    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["datetime", "open", "high", "low", "close", "volume"],
    )

    # parse datetime string "YYYYMMDD HHMMSS"
    df["time"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M%S")

    # We don't need the original string column anymore
    df = df.drop(columns=["datetime"])

    # Reorder columns nicely
    df = df[["time", "open", "high", "low", "close", "volume"]]

    return df


def build_hourly():
    # find all CSVs
    pattern = os.path.join(RAW_DIR, "*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    dfs = []
    for path in files:
        df_m1 = load_one_file(path)
        dfs.append(df_m1)

    # concat all years
    df_m1_all = pd.concat(dfs, ignore_index=True)

    # sort just in case
    df_m1_all = df_m1_all.sort_values("time").reset_index(drop=True)

    # set index to datetime for resampling
    df_m1_all = df_m1_all.set_index("time")

    # resample to hourly OHLCV
    df_h1 = df_m1_all.resample("1H").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # drop hours with no trades (weekends etc.)
    df_h1 = df_h1.dropna(subset=["open", "high", "low", "close"])

    # move index back to a column named "time"
    df_h1 = df_h1.reset_index()
    # DROP VOLUME COLUMN (useless = always zero)
    df_h1 = df_h1.drop(columns=["volume"])

    print(df_h1.head())
    print("Rows (hours):", len(df_h1))

    # save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_h1.to_csv(OUT_PATH, index=False)

    print(f"Saved hourly data to {OUT_PATH}")


if __name__ == "__main__":
    build_hourly()