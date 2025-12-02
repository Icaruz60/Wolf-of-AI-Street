import csv
from pathlib import Path

import pandas as pd

INPUT_CSV = "all_news_raw.csv"
OUTPUT_CSV = "eurusd_news_filtered.csv"

#Threshhold
THRESHHOLD = 10

# Chunk size to keep memory sane on a 5.4GB CSV
CHUNK_SIZE = 100_000

# --- Heuristic keyword sets ---

# Domains that are by nature FX/markets heavy: a small bias in their favor
FX_HEAVY_SOURCES = {
    "fxstreet.com",
    "investing.com",
}

# Themes that are very tightly linked to currencies / FX / monetary policy
FX_STRONG_THEMES = [
    "ECON_CURRENCY_EXCHANGE_RATE",
    "EPU_CATS_MONETARY_POLICY",
]

# Themes that are macro but slightly broader; still relevant when paired with US + Eurozone
FX_MACRO_THEMES = [
    "EPU_ECONOMY",
    "EPU_ECONOMY_HISTORIC",
    "ECON_INFLATION",
    "ECON_TRADE_DISPUTE",
    "ECON_CENTRALBANK",
    "EPU_POLICY_CENTRAL_BANK",
]

# Central banks & institutions that clearly link to EUR or USD fundamentals
CB_ORG_KEYWORDS = [
    "European Central Bank",
    "ECB",
    "Federal Reserve",
    "Fed ",
    "Fed,",
    "FOMC",
    "Bundesbank",
    "Bank of England",
    "International Monetary Fund",
    "IMF",
]

# Crude EU country name list for locations field
EU_LOCATION_KEYWORDS = [
    "Eurozone",
    "European Union",
    "Germany",
    "France",
    "Italy",
    "Spain",
    "Netherlands",
    "Belgium",
    "Austria",
    "Finland",
    "Portugal",
    "Ireland",
    "Greece",
    "Slovakia",
    "Slovenia",
    "Estonia",
    "Latvia",
    "Lithuania",
    "Cyprus",
    "Luxembourg",
    "Malta",
]

# URL patterns that very strongly indicate pair-specific content
URL_STRONG_PATTERNS = [
    "eurusd",
    "eur-usd",
    "eur_usd",
    "euro-dollar",
    "euro_us_dollar",
]

# Weaker URL patterns (need to be combined with other signals)
URL_WEAK_PATTERNS = [
    "forex",
    "fx-",
    "/fx/",
    "currency",
    "currencies",
]

US_LOCATION_PATTERNS = [
    "United States",
    ",US#",
    "#US#",
]


def _contains_any(text: str, patterns) -> bool:
    return any(p in text for p in patterns)


def safe_str(val) -> str:
    """
    Convert a field to a safe string:
    - handles None, NaN, pd.NA without triggering the 'boolean value of NA' error
    """
    if val is None:
        return ""
    # pd.isna works on NaN and pd.NA without boolean ambiguity
    try:
        if pd.isna(val):
            return ""
    except TypeError:
        # Non-scalar weirdness; just fall back to str
        pass
    return str(val)


def eurusd_relevance_score(row) -> int:
    """
    Heuristic scorer:
    higher = more likely to be EUR/USD relevant.
    This is deliberately conservative.
    """
    source = safe_str(row.get("source"))
    url = safe_str(row.get("url")).lower()
    themes = safe_str(row.get("themes"))
    locations = safe_str(row.get("locations"))
    persons = safe_str(row.get("persons"))
    orgs = safe_str(row.get("organizations"))

    source_l = source.lower()

    score = 0

    # 1) URL hints: explicit pair or FX context
    if _contains_any(url, URL_STRONG_PATTERNS):
        score += 5
    if _contains_any(url, URL_WEAK_PATTERNS):
        score += 1

    # Small bump for FX-oriented sources
    if source_l in FX_HEAVY_SOURCES:
        score += 1

    # 2) Themes: explicit FX / currencies
    if _contains_any(themes, FX_STRONG_THEMES):
        score += 4

    # Any world-currency theme mentioning dollar or euro
    if "ECON_WORLDCURRENCIES_" in themes and ("DOLLAR" in themes or "EURO" in themes):
        score += 3

    if _contains_any(themes, FX_MACRO_THEMES):
        score += 2

    # 3) Central bank & institution mentions
    if _contains_any(orgs, CB_ORG_KEYWORDS) or _contains_any(persons, CB_ORG_KEYWORDS):
        score += 3

    # 4) Location structure: US + Euro-area hit
    has_us = _contains_any(locations, US_LOCATION_PATTERNS)
    has_eu = _contains_any(locations, EU_LOCATION_KEYWORDS)

    if has_us and has_eu:
        score += 3
    elif (has_us or has_eu) and (
        _contains_any(themes, FX_STRONG_THEMES)
        or _contains_any(themes, FX_MACRO_THEMES)
    ):
        # Macro story but only one side mentioned
        score += 1

    return score


def filter_chunk(chunk: pd.DataFrame, threshold: int = THRESHHOLD) -> pd.DataFrame:
    # Compute score row-wise; keep only strong hits.
    scores = chunk.apply(eurusd_relevance_score, axis=1)
    chunk = chunk.copy()
    chunk["eurusd_score"] = scores
    return chunk[chunk["eurusd_score"] >= threshold]


def main():
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    first = True
    for chunk in pd.read_csv(
        input_path,
        chunksize=CHUNK_SIZE,
        dtype={
            "source": "string",
            "timestamp_utc": "string",
            "url": "string",
            "tone_avg": "float64",
            "tone_pos": "float64",
            "tone_neg": "float64",
            "tone_polarity": "float64",
            "themes": "string",
            "locations": "string",
            "persons": "string",
            "organizations": "string",
        },
    ):
        filtered = filter_chunk(chunk, threshold=6)
        if filtered.empty:
            continue

        mode = "w" if first else "a"
        header = first
        filtered.to_csv(
            OUTPUT_CSV,
            mode=mode,
            index=False,
            header=header,
        )
        first = False


if __name__ == "__main__":
    main()
