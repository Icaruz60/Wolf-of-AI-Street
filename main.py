import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class Config:
    csv_path: str = "data/eurusd_hourly.csv"
    time_column: str = "time"
    close_column: str = "close"
    volume_column: str = "volume"  # optional for now


cfg = Config()


def load_price_data(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(f"CSV file not found at {cfg.csv_path}")

    df = pd.read_csv(cfg.csv_path)

    print("=== Raw dataframe info ===")
    print(df.head())
    print("\nColumns:", list(df.columns))
    print("Num rows:", len(df))

    # check required columns
    for col in [cfg.time_column, cfg.close_column]:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

    # volume is optional, warn if missing
    if cfg.volume_column not in df.columns:
        print(f"[INFO] Volume column '{cfg.volume_column}' not found. "
              f"We'll handle that later (can fake it or drop it).")

    return df


def main():
    df = load_price_data(cfg)
    print("\n[OK] Data loaded successfully. "
          "Next step: build Dataset class to turn this into sequences + labels.")


if __name__ == "__main__":
    main()