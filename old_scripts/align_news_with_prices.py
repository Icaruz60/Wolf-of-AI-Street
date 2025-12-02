import pandas as pd
from pathlib import Path

# ==================================================
# Config
# ==================================================

PRICE_FILE = "data/eurusd_hourly.csv"
NEWS_FILE = "data/eurusd_news_filtered_sorted.csv"   # <-- root, not in /data
OUTPUT_FILE = "data/eurusd_merged.csv"          # put result in /data


# ==================================================
# Helper: detect columns
# ==================================================

def detect_timestamp(colnames, label):
    lower = [c.lower() for c in colnames]
    # ordered preference
    candidates = ["timestamp_utc", "datetime_utc", "timestamp", "datetime", "time"]
    for cand in candidates:
        if cand in lower:
            return colnames[lower.index(cand)]
    raise ValueError(f"Could not find timestamp column for {label}. Found columns: {colnames}")

def detect_score(colnames):
    lower = [c.lower() for c in colnames]
    # prefer eurusd_score, then generic *score*
    preferred = ["eurusd_score", "score"]
    for cand in preferred:
        if cand in lower:
            return colnames[lower.index(cand)]
    # fallback: any column containing "score"
    for i, c in enumerate(lower):
        if "score" in c:
            return colnames[i]
    raise ValueError(f"Could not find any score column. Found columns: {colnames}")

def detect_tone(colnames):
    lower = [c.lower() for c in colnames]
    # prefer tone_avg, then any tone*
    preferred = ["tone_avg", "avg_tone"]
    for cand in preferred:
        if cand in lower:
            return colnames[lower.index(cand)]
    for i, c in enumerate(lower):
        if c.startswith("tone"):
            return colnames[i]
    return None  # optional


# ==================================================
# Load price data
# ==================================================

print("Loading EURUSD price data (sample for column detection)...")
price_sample = pd.read_csv(PRICE_FILE, nrows=5)
price_ts_col = detect_timestamp(price_sample.columns.tolist(), "prices")
print(f"Using '{price_ts_col}' as price timestamp column.")

print("Reloading full price data with parsed dates...")
prices = pd.read_csv(PRICE_FILE, parse_dates=[price_ts_col])
prices = prices.sort_values(price_ts_col)
prices = prices.set_index(price_ts_col)
print(f"Loaded {len(prices):,} hourly candles.")


# ==================================================
# Load news
# ==================================================

print("Loading news data (sample for column detection)...")
news_sample = pd.read_csv(NEWS_FILE, nrows=5)

news_ts_col = detect_timestamp(news_sample.columns.tolist(), "news")
score_col = detect_score(news_sample.columns.tolist())
tone_col = detect_tone(news_sample.columns.tolist())

print(f"Using '{news_ts_col}' as news timestamp column.")
print(f"Using '{score_col}' as news score column.")
if tone_col:
    print(f"Using '{tone_col}' as news tone column.")
else:
    print("No tone column found; will skip tone features.")

print("Loading full news data...")
parse_cols = [news_ts_col]
news = pd.read_csv(NEWS_FILE, parse_dates=parse_cols)

# Round news timestamps down to hourly buckets
news["hour"] = news[news_ts_col].dt.floor("H")

# Filter news to price range
min_ts, max_ts = prices.index.min(), prices.index.max()
news = news[(news["hour"] >= min_ts) & (news["hour"] <= max_ts)]
print(f"Loaded {len(news):,} filtered news items inside price range.")


# ==================================================
# Aggregate news per hour
# ==================================================

print("Aggregating news per hour...")

aggregations = {
    score_col: "mean",
    news_ts_col: "count",  # number of articles
}

if tone_col:
    aggregations[tone_col] = "mean"

agg = news.groupby("hour").agg(aggregations)

# rename columns to clean, model-friendly names
rename_map = {
    score_col: "mean_score",
    news_ts_col: "n_articles",
}
if tone_col:
    rename_map[tone_col] = "mean_tone"

agg = agg.rename(columns=rename_map)
agg.index.name = "timestamp"

print(f"Aggregated into {len(agg):,} hourly rows with news features.")


# ==================================================
# Merge with price data
# ==================================================

print("Joining prices with news features...")
df = prices.join(agg, how="left")

# fill missing with zeros (no news that hour)
fill_cols = ["mean_score", "n_articles"]
if "mean_tone" in df.columns:
    fill_cols.append("mean_tone")

df[fill_cols] = df[fill_cols].fillna(0)

print(f"Final merged dataset size: {len(df):,} hours.")


# ==================================================
# Save
# ==================================================

Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE)
print(f"Saved → {OUTPUT_FILE}")
