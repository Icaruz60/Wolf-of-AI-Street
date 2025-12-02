#!/usr/bin/env python
"""
Unified EUR/USD news processing pipeline.

Steps:
1. Load raw GDELT news from rawdata/all_news_raw.csv
2. Score + filter EURUSD-relevant news -> rawdata/eurusd_news_filtered.csv
3. Sort by timestamp -> data/eurusd_news_filtered_sorted.csv
4. Aggregate news per hour and merge with hourly prices
   -> data/eurusd_merged.csv

Assumptions:
- Raw news has at least:
    source, timestamp_utc, url, tone_avg, tone_pos, tone_neg,
    tone_polarity, themes, locations, persons, organizations
- Hourly prices CSV has a 'timestamp_utc' column compatible with pandas.to_datetime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set, Dict

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
MERGED_OUT_CSV = DATA_DIR / "eurusd_merged.csv"

TIMESTAMP_COL = "timestamp_utc"
MIN_SCORE_DEFAULT = 6  # change if you want stricter/looser filter

# ---------------------------------------------------------------------
# Heuristics for EURUSD relevance scoring
# ---------------------------------------------------------------------

# FX / macro-heavy sources; add your own if needed.
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

# Themes that clearly scream "macro / FX relevant".
THEME_WEIGHTS: Dict[str, int] = {
    # direct currency references
    "ECON_WORLDCURRENCIES_DOLLAR": 4,
    "ECON_WORLDCURRENCIES_EURO": 4,
    "ECON_WORLDCURRENCIES": 2,

    # monetary policy & central banks
    "ECON_CENTRAL_BANKS": 4,
    "EPU_CATS_MONETARY_POLICY": 4,
    "ECON_INTEREST_RATES": 3,
    "ECON_BONDS": 2,

    # inflation / growth / jobs
    "ECON_INFLATION": 3,
    "ECON_GDP": 3,
    "ECON_UNEMPLOYMENT": 2,
    "ECON_RECESSION": 3,

    # external balance / trade
    "ECON_TRADE_BALANCE": 3,
    "ECON_EXPORTS": 2,
    "ECON_IMPORTS": 2,

    # markets & risk sentiment
    "ECON_STOCK_MARKETS": 2,
    "ECON_COMMODITIES_OIL": 2,
    "FINANCE_STOCK_MARKET": 2,
    "FINANCE_SECURITIES": 1,

    # generic economic / financial
    "ECONOMY": 1,
    "FINANCE": 1,
}

# Organizations that obviously matter for EURUSD.
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

# Locations: we mostly care about US + major Eurozone states.
LOCATION_WEIGHTS: Dict[str, int] = {
    # GDELT location strings are like "3#Washington" etc.
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


def _safe_lower(s: str | float | int | None) -> str:
    if isinstance(s, str):
        return s.lower()
    return ""


def _parse_gdelt_tag_field(cell: str | float | int | None) -> Set[str]:
    """
    Parse GDELT-style 'THEMES', 'ORGANIZATIONS', 'LOCATIONS' columns.

    Example entry:
        "ECON_INFLATION,1;ECON_GDP,0.7;WB_2131_EMPLOYABILITY_SKILLS,0.6"

    We only care about the tag before the first comma.
    """
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
    """
    Heuristic score for "how relevant is this article to EUR/USD".

    Uses:
      - source
      - themes
      - locations
      - organizations
    """
    score = 0.0

    source = _safe_lower(row.get("source"))
    url = _safe_lower(row.get("url"))

    themes = _parse_gdelt_tag_field(row.get("themes"))
    orgs = _parse_gdelt_tag_field(row.get("organizations"))
    locs = _parse_gdelt_tag_field(row.get("locations"))

    # 1) Source-level weight.
    for s in FX_HEAVY_SOURCES:
        if s in source or s in url:
            score += 3
            break

    # 2) Theme-based weights.
    for t in themes:
        w = THEME_WEIGHTS.get(t, 0)
        if w:
            score += w

    # 3) Organization-based weights.
    for o in orgs:
        w = ORG_WEIGHTS.get(o, 0)
        if w:
            score += w

    # 4) Location-based weights (US + Eurozone states).
    for loc in locs:
        # GDELT loc tags often have "2#Texas" etc; we just search for key substring.
        upper_loc = loc.upper()
        for key, w in LOCATION_WEIGHTS.items():
            if key in upper_loc:
                score += w

    # 5) Bonus for "both sides of the pair show up".
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
    print(f"[1/3] Loading raw news from {RAW_NEWS_CSV}")
    df = pd.read_csv(RAW_NEWS_CSV)

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"'{TIMESTAMP_COL}' column not found in raw news CSV")

    print(f"[1/3] Scoring {len(df):,} articles for EUR/USD relevance...")
    df["eurusd_score"] = df.apply(eurusd_relevance_score, axis=1)

    print(f"[1/3] Filtering with min_score >= {min_score} ...")
    df = df[df["eurusd_score"] >= min_score].copy()
    df.sort_values("eurusd_score", ascending=False, inplace=True)

    print(f"[1/3] Remaining articles: {len(df):,}")
    FILTERED_NEWS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FILTERED_NEWS_CSV, index=False)
    print(f"[1/3] Saved filtered news to {FILTERED_NEWS_CSV}")


def step2_sort_filtered_news() -> None:
    print(f"[2/3] Loading filtered news from {FILTERED_NEWS_CSV}")
    df = pd.read_csv(FILTERED_NEWS_CSV)

    print("[2/3] Parsing timestamps and sorting...")
    df[TIMESTAMP_COL] = pd.to_datetime(
        df[TIMESTAMP_COL], utc=True, errors="coerce"
    )
    before = len(df)
    df = df.dropna(subset=[TIMESTAMP_COL]).copy()
    if len(df) < before:
        print(f"[2/3] Dropped {before - len(df)} rows with invalid timestamps.")

    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    SORTED_NEWS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SORTED_NEWS_CSV, index=False)
    print(f"[2/3] Saved time-sorted news to {SORTED_NEWS_CSV}")


def step3_align_news_with_prices() -> None:
    print(f"[3/3] Loading sorted news from {SORTED_NEWS_CSV}")
    news = pd.read_csv(SORTED_NEWS_CSV)

    print(f"[3/3] Loading hourly prices from {HOURLY_PRICE_CSV}")
    prices = pd.read_csv(HOURLY_PRICE_CSV)

    # --- figure out which column in prices is the timestamp ---
    candidate_cols = [TIMESTAMP_COL, "time", "Time", "timestamp", "Timestamp", "date", "Date"]
    price_ts_col = None
    for c in candidate_cols:
        if c in prices.columns:
            price_ts_col = c
            break

    if price_ts_col is None:
        raise ValueError(
            f"No usable timestamp column found in hourly prices CSV. "
            f"Looked for: {candidate_cols}. Got columns: {list(prices.columns)}"
        )

    # Normalize timestamps.
    news[TIMESTAMP_COL] = pd.to_datetime(news[TIMESTAMP_COL], utc=True, errors="coerce")
    prices[price_ts_col] = pd.to_datetime(prices[price_ts_col], utc=True, errors="coerce")

    news = news.dropna(subset=[TIMESTAMP_COL]).copy()
    prices = prices.dropna(subset=[price_ts_col]).copy()

    # Rename price timestamp column to match the constant so merge works cleanly.
    if price_ts_col != TIMESTAMP_COL:
        prices = prices.rename(columns={price_ts_col: TIMESTAMP_COL})

    # Floor news timestamps to the hour so they line up with hourly candles.
    news[TIMESTAMP_COL] = news[TIMESTAMP_COL].dt.floor("H")

    # Aggregate news per hour.
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

    print(f"[3/3] Aggregated news into {len(agg):,} hourly buckets.")

    # Merge with prices.
    merged = prices.merge(
        agg.reset_index(),
        on=TIMESTAMP_COL,
        how="left",
    )

    # Fill NaNs for hours with no news.
    for col in [
        "news_count",
        "news_score_sum",
        "news_score_mean",
        "news_score_max",
        "news_tone_avg",
        "news_tone_pos",
        "news_tone_neg",
        "news_tone_polarity",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    MERGED_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_OUT_CSV, index=False)
    print(f"[3/3] Saved merged hourly dataset to {MERGED_OUT_CSV}")


def main(min_score: float = MIN_SCORE_DEFAULT) -> None:
    step1_filter_and_score(min_score=min_score)
    step2_sort_filtered_news()
    step3_align_news_with_prices()


if __name__ == "__main__":
    # If you want CLI args, wire up argparse here; for now keep it simple.
    main()
