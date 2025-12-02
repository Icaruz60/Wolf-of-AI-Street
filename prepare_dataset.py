#!/usr/bin/env python
"""
Unified EUR/USD news processing pipeline (full version, merged).

Steps:
1. Load + score + filter raw GDELT news  -> rawdata/eurusd_news_filtered.csv
2. Sort news by timestamp                -> data/eurusd_news_filtered_sorted.csv
3. Aggregate + merge with prices         -> data/eurusd_merged.csv
4. Trim merged dataset from START_TS     -> data/final_dataset.csv
"""

from __future__ import annotations
from pathlib import Path
from typing import Set, Dict
import pandas as pd

# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "rawdata"
DATA_DIR = ROOT / "data"

RAW_NEWS_CSV = RAW_DIR / "all_news_raw.csv"
FILTERED_NEWS_CSV = RAW_DIR / "eurusd_news_filtered.csv"
SORTED_NEWS_CSV = RAW_DIR / "eurusd_news_filtered_sorted.csv"
HOURLY_PRICE_CSV = RAW_DIR / "eurusd_hourly.csv"
MERGED_OUT_CSV = RAW_DIR / "eurusd_merged.csv"
FINAL_OUT_CSV = DATA_DIR / "final_dataset.csv"

TIMESTAMP_COL = "timestamp_utc"
MIN_SCORE_DEFAULT = 15

# This comes from your trim script
START_TS = "2020-01-01 17:00:00+00:00"

# ---------------------------------------------------------------------
# Heuristics for EURUSD relevance scoring
# ---------------------------------------------------------------------

FX_HEAVY_SOURCES: Set[str] = {
    "fxstreet.com",
    "dailyfx.com",
    "forexlive.com",
    "forexcrunch.com",
    "forextime.com",
    "forex.com",
    "thinkmarkets.com",
    "oanda.com",
    "ig.com",
    "capital.com",
    "seekingalpha.com",
    "investing.com",
    "bloomberg.com",
    "reuters.com",
}

THEME_WEIGHTS: Dict[str, int] = {
    "ECON_WORLDCURRENCIES_DOLLAR": 4,
    "ECON_WORLDCURRENCIES_EURO": 4,
    "ECON_WORLDCURRENCIES": 2,
    "ECON_CENTRAL_BANKS": 4,
    "EPU_CATS_MONETARY_POLICY": 4,
    "ECON_INTEREST_RATES": 3,
    "ECON_BONDS": 2,
    "ECON_INFLATION": 3,
    "ECON_GDP": 3,
    "ECON_UNEMPLOYMENT": 2,
    "ECON_RECESSION": 3,
    "ECON_TRADE_BALANCE": 3,
    "ECON_EXPORTS": 2,
    "ECON_IMPORTS": 2,
    "ECON_STOCK_MARKETS": 2,
    "ECON_COMMODITIES_OIL": 2,
    "FINANCE_STOCK_MARKET": 2,
    "FINANCE_SECURITIES": 1,
    "ECONOMY": 1,
    "FINANCE": 1,
}

ORG_WEIGHTS: Dict[str, int] = {
    "FEDERAL_RESERVE": 4,
    "FEDERAL_RESERVE_SYSTEM": 4,
    "EUROPEAN_CENTRAL_BANK": 4,
    "ECB": 4,
    "INTERNATIONAL_MONETARY_FUND": 2,
    "IMF": 2,
    "WORLD_BANK": 1,
    "BANK_OF_ENGLAND": 1,
}

LOCATION_WEIGHTS: Dict[str, int] = {
    "UNITED STATES": 3,
    "EUROPEAN UNION": 2,
    "EUROZONE": 2,
    "GERMANY": 2,
    "FRANCE": 2,
    "ITALY": 1,
    "SPAIN": 1,
    "NETHERLANDS": 1,
    "BELGIUM": 1,
}

def _safe_lower(s):
    if isinstance(s, str):
        return s.lower()
    return ""

def _parse_gdelt_tag_field(cell) -> Set[str]:
    out: Set[str] = set()
    if not isinstance(cell, str):
        return out
    for chunk in cell.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        tag = chunk.split(",")[0].strip()
        if tag:
            out.add(tag)
    return out


def eurusd_relevance_score(row: pd.Series) -> float:
    score = 0.0

    source = _safe_lower(row.get("source"))
    url = _safe_lower(row.get("url"))

    themes = _parse_gdelt_tag_field(row.get("themes"))
    orgs = _parse_gdelt_tag_field(row.get("organizations"))
    locs = _parse_gdelt_tag_field(row.get("locations"))

    for s in FX_HEAVY_SOURCES:
        if s in source or s in url:
            score += 3
            break

    for t in themes:
        score += THEME_WEIGHTS.get(t, 0)

    for o in orgs:
        score += ORG_WEIGHTS.get(o, 0)

    for loc in locs:
        upper_loc = loc.upper()
        for key, w in LOCATION_WEIGHTS.items():
            if key in upper_loc:
                score += w

    has_us = any("UNITED STATES" in l.upper() for l in locs) or any(
        t in themes for t in ["ECON_WORLDCURRENCIES_DOLLAR"]
    )
    has_eu = any(
        k in {t.upper() for t in themes}
        for k in ["ECON_WORLDCURRENCIES_EURO", "ECON_WORLDCURRENCIES"]
    ) or any("EUROPEAN UNION" in l.upper() or "GERMANY" in l.upper() for l in locs)

    if has_us and has_eu:
        score += 4

    return score

# ---------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------

def step1_filter_and_score(min_score: float = MIN_SCORE_DEFAULT) -> None:
    print(f"[1/4] Loading raw news from {RAW_NEWS_CSV}")
    df = pd.read_csv(RAW_NEWS_CSV)

    print(f"[1/4] Scoring {len(df):,} articles...")
    df["eurusd_score"] = df.apply(eurusd_relevance_score, axis=1)

    df = df[df["eurusd_score"] >= min_score].copy()
    df.sort_values("eurusd_score", ascending=False, inplace=True)

    FILTERED_NEWS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FILTERED_NEWS_CSV, index=False)
    print(f"[1/4] Saved filtered news → {FILTERED_NEWS_CSV}")


def step2_sort_filtered_news() -> None:
    print(f"[2/4] Loading filtered news → {FILTERED_NEWS_CSV}")
    df = pd.read_csv(FILTERED_NEWS_CSV)

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True, errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL])

    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    SORTED_NEWS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SORTED_NEWS_CSV, index=False)
    print(f"[2/4] Saved sorted news → {SORTED_NEWS_CSV}")


def step3_align_news_with_prices() -> None:
    print(f"[3/4] Loading sorted news → {SORTED_NEWS_CSV}")
    news = pd.read_csv(SORTED_NEWS_CSV)

    print(f"[3/4] Loading hourly prices → {HOURLY_PRICE_CSV}")
    prices = pd.read_csv(HOURLY_PRICE_CSV)

    candidate_cols = [
        TIMESTAMP_COL, "time", "Time", "timestamp", "Timestamp",
        "date", "Date"
    ]
    price_ts_col = None
    for c in candidate_cols:
        if c in prices.columns:
            price_ts_col = c
            break
    if price_ts_col is None:
        raise ValueError("No usable timestamp column found in hourly price CSV.")

    news[TIMESTAMP_COL] = pd.to_datetime(news[TIMESTAMP_COL], utc=True, errors="coerce")
    prices[price_ts_col] = pd.to_datetime(prices[price_ts_col], utc=True, errors="coerce")

    news = news.dropna(subset=[TIMESTAMP_COL])
    prices = prices.dropna(subset=[price_ts_col])

    if price_ts_col != TIMESTAMP_COL:
        prices = prices.rename(columns={price_ts_col: TIMESTAMP_COL})

    news[TIMESTAMP_COL] = news[TIMESTAMP_COL].dt.floor("H")

    agg = news.groupby(TIMESTAMP_COL).agg(
        news_count=("eurusd_score", "count"),
        news_score_sum=("eurusd_score", "sum"),
        news_score_mean=("eurusd_score", "mean"),
        news_score_max=("eurusd_score", "max"),
        news_tone_avg=("tone_avg", "mean"),
        news_tone_pos=("tone_pos", "mean"),
        news_tone_neg=("tone_neg", "mean"),
        news_tone_polarity=("tone_polarity", "mean"),
    )

    merged = prices.merge(agg.reset_index(), on=TIMESTAMP_COL, how="left")

    for col in [
        "news_count", "news_score_sum", "news_score_mean",
        "news_score_max", "news_tone_avg", "news_tone_pos",
        "news_tone_neg", "news_tone_polarity"
    ]:
        merged[col] = merged[col].fillna(0.0)

    MERGED_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_OUT_CSV, index=False)
    print(f"[3/4] Saved merged data → {MERGED_OUT_CSV}")


def step4_trim_merged_dataset() -> None:
    print(f"[4/4] Trimming merged dataset → start @ {START_TS}")
    df = pd.read_csv(MERGED_OUT_CSV)

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])

    start_ts = pd.Timestamp(START_TS)
    df_trim = df[df[TIMESTAMP_COL] >= start_ts].reset_index(drop=True)

    if df_trim.empty:
        raise ValueError("Trimmed dataset is empty. Check START_TS.")

    FINAL_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_trim.to_csv(FINAL_OUT_CSV, index=False)
    print(f"[4/4] Saved final dataset → {FINAL_OUT_CSV}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(min_score: float = MIN_SCORE_DEFAULT):
    step1_filter_and_score(min_score)
    step2_sort_filtered_news()
    step3_align_news_with_prices()
    step4_trim_merged_dataset()

if __name__ == "__main__":
    main()
