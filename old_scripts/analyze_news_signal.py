import pandas as pd
import numpy as np

CSV_PATH = "data/eurusd_merged.csv"
TIME_COL = "time"
CLOSE_COL = "close"

NEWS_COLS = ["mean_score", "mean_tone", "n_articles"]
HORIZON = 4  # same as your model

def main():
    df = pd.read_csv(CSV_PATH).sort_values(TIME_COL).reset_index(drop=True)

    # basic checks
    print("Columns:", df.columns.tolist())
    print(df[NEWS_COLS].describe())

    close = df[CLOSE_COL].astype(float).values
    future = np.roll(close, -HORIZON)
    mask = np.arange(len(close)) < len(close) - HORIZON

    ret = (future - close) / close
    ret = ret[mask]

    news = df.loc[mask, NEWS_COLS].astype(float)

    # correlation of raw returns with news
    print("\nPearson corr(return, feature):")
    for c in NEWS_COLS:
        corr = np.corrcoef(ret, news[c].values)[0, 1]
        print(f"  {c}: {corr:.5f}")

    # sign-based
    sign_ret = np.sign(ret)

    # bucket by mean_score quantiles
    q_low = news["mean_score"].quantile(0.1)
    q_high = news["mean_score"].quantile(0.9)

    low_mask = news["mean_score"] <= q_low
    high_mask = news["mean_score"] >= q_high

    def bucket_stats(name, m):
        r = ret[m]
        s = sign_ret[m]
        print(f"\nBucket: {name}")
        print(f"  n={m.sum()}")
        print(f"  avg_return={r.mean():.6f}")
        print(f"  sign>0 fraction={(s > 0).mean():.3f}")
        print(f"  sign<0 fraction={(s < 0).mean():.3f}")

    bucket_stats("low mean_score (bottom 10%)", low_mask)
    bucket_stats("high mean_score (top 10%)", high_mask)

    # bucket by high article count
    art_q = news["n_articles"].quantile(0.9)
    art_mask = news["n_articles"] >= art_q
    bucket_stats("high n_articles (top 10%)", art_mask)

if __name__ == "__main__":
    main()
