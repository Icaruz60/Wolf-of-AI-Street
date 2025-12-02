import pandas as pd
import numpy as np
from training_scripts.main import cfg

def main():
    df = pd.read_csv(cfg.csv_path)
    df = df.sort_values(cfg.time_column).reset_index(drop=True)

    close = df[cfg.close_column].astype(float).values

    horizon = cfg.horizon
    future = np.roll(close, -horizon)
    raw_ret = (future - close) / close
    raw_ret[-horizon:] = 0.0

    # normalize to something like [-1,1]
    signal = raw_ret / cfg.max_return  # if max_return=0.01, that's ~[-1,1] for +/-1% moves

    rng = np.random.default_rng(42)
    noise = rng.normal(scale=0.2, size=len(signal))   # smaller noise

    news_score = signal + noise

    df[cfg.news_column] = news_score
    df.to_csv(cfg.csv_path, index=False)
    print(f"Added STRONG fake '{cfg.news_column}' to {cfg.csv_path}")

if __name__ == "__main__":
    main()
