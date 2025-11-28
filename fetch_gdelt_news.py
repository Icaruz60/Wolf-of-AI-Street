import os
import csv
from datetime import datetime, timedelta
from io import BytesIO
from urllib.parse import urlparse

import pandas as pd
import requests

# ===============================
# CONFIG
# ===============================

OUTPUT_CSV = "all_news_raw.csv"

# ✏️ Set your window here.
# For ~5 years of news, something like 2020-01-01 is totally fine.
START_DATE = datetime(2019, 1, 1)
END_DATE = datetime.utcnow()

GDELT_GKG_URL = "http://data.gdeltproject.org/gdeltv2/{date}.gkg.csv.zip"

TARGET_DOMAINS = [
    "reuters.com",
    "bloomberg.com",
    "investing.com",
    "fxstreet.com",
    "yahoo.com",
    "finance.yahoo.com",
    "marketwatch.com",
]


# ===============================
# HELPERS
# ===============================

def date_range(start: datetime, end: datetime, step_minutes: int = 15):
    """Yield every 15-minute slot between start and end."""
    d = start
    while d <= end:
        yield d
        d += timedelta(minutes=step_minutes)


def extract_domain(url: str) -> str:
    """Robust domain extraction from URL or bare hostname."""
    try:
        if not url:
            return ""
        if not (url.startswith("http://") or url.startswith("https://")):
            url = "http://" + url
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        return host.replace("www.", "")
    except Exception:
        return ""


def fetch_gkg_zip_bytes(url: str):
    """Download a GDELT GKG zip file into memory."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return BytesIO(r.content)
    except Exception:
        return None


# ===============================
# MAIN EXTRACTION
# ===============================

def fetch_news():
    total_slots = int((END_DATE - START_DATE).total_seconds() / 900) + 1
    print(f"Fetching GDELT GKG for ~{total_slots} time slots...")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["source", "headline", "timestamp_utc", "url", "tone"])

        for i, timestamp in enumerate(date_range(START_DATE, END_DATE), start=1):
            date_str = timestamp.strftime("%Y%m%d%H%M00")
            url = GDELT_GKG_URL.format(date=date_str)

            zip_bytes = fetch_gkg_zip_bytes(url)
            if zip_bytes is None:
                continue

            try:
                df = pd.read_csv(
                    zip_bytes,
                    compression="zip",
                    sep="\t",
                    header=None,
                    low_memory=False,
                )
            except Exception:
                continue

            # GKG 2.0 column mapping (from official codebook):
            # 1: DATE (YYYYMMDDHHMMSS)
            # 3: SourceCommonName (e.g. reuters.com)
            # 4: DocumentIdentifier (URL)
            # 15: V2Tone (tone stats)
            for _, row in df.iterrows():
                try:
                    raw_url = str(row[4])
                    source_name = str(row[3])

                    domain = extract_domain(raw_url)
                    if not domain:
                        domain = source_name.lower().replace("www.", "")

                    if domain not in TARGET_DOMAINS:
                        continue

                    raw_time = str(row[1])
                    ts = datetime.strptime(raw_time, "%Y%m%d%H%M%S")

                    tone_str = str(row[15])
                    tone_val = 0.0
                    if isinstance(tone_str, str) and tone_str:
                        try:
                            tone_val = float(tone_str.split(",")[0])
                        except Exception:
                            tone_val = 0.0

                    # GKG doesn't give a clean headline; we leave it empty for now.
                    headline = ""

                    writer.writerow([domain, headline, ts.isoformat(), raw_url, tone_val])
                except Exception:
                    continue

            if i % 200 == 0 or i == total_slots:
                pct = i / total_slots * 100
                print(f"{i}/{total_slots} time slots processed... ({pct:.2f}% done)")

    print(f"Done. Saved filtered news to {OUTPUT_CSV}")


if __name__ == "__main__":
    fetch_news()
