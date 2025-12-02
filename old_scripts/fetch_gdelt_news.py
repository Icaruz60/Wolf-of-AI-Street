import os
import csv
from datetime import datetime, timedelta
from io import BytesIO
from urllib.parse import urlparse
from time import time

import pandas as pd
import requests

# ===============================
# CONFIG
# ===============================

OUTPUT_CSV = "all_news_raw.csv"
CHECKPOINT_FILE = "gdelt_checkpoint.txt"

# Set your window here.
# 2020-01-01 is plenty for a strong model; push earlier if you want more.
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2024, 12, 31, 16, 0, 0)

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


def save_checkpoint(timestamp: datetime):
    """Save last processed timestamp."""
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(timestamp.isoformat())


def load_checkpoint():
    """Load last processed timestamp if exists."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    with open(CHECKPOINT_FILE, "r") as f:
        txt = f.read().strip()
        if not txt:
            return None
        try:
            return datetime.fromisoformat(txt)
        except Exception:
            return None


# ===============================
# MAIN EXTRACTION
# ===============================

def fetch_news():
    # Total number of 15-min slots over the whole period (for % and ETA)
    total_slots = int((END_DATE - START_DATE).total_seconds() / 900) + 1

    checkpoint = load_checkpoint()
    if checkpoint is not None and checkpoint > START_DATE:
        # resume from the next 15-minute slot after checkpoint
        resume_start = checkpoint + timedelta(minutes=15)
        print(f"Resuming from checkpoint: {resume_start}")
        start_timestamp = resume_start
        processed_before = int((checkpoint - START_DATE).total_seconds() / 900) + 1
        append_mode = True
    else:
        print("No valid checkpoint found, starting from BEGINNING.")
        start_timestamp = START_DATE
        processed_before = 0
        append_mode = False

    print(f"Fetching GDELT GKG from {start_timestamp} to {END_DATE} "
          f"(~{total_slots} total slots from {START_DATE})")

    # Open CSV: write header if starting fresh, else append
    file_mode = "a" if append_mode else "w"
    with open(OUTPUT_CSV, file_mode, newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        if not append_mode:
            writer.writerow([
                "source",
                "timestamp_utc",
                "url",
                "tone_avg",
                "tone_pos",
                "tone_neg",
                "tone_polarity",
                "themes",
                "locations",
                "persons",
                "organizations",
            ])

        start_ts = time()

        # Local loop counter (slots processed in THIS run)
        for local_i, timestamp in enumerate(date_range(start_timestamp, END_DATE), start=1):
            global_i = processed_before + local_i

            date_str = timestamp.strftime("%Y%m%d%H%M00")
            url = GDELT_GKG_URL.format(date=date_str)

            zip_bytes = fetch_gkg_zip_bytes(url)
            if zip_bytes is None:
                # save checkpoint anyway so we can skip this slot next run
                save_checkpoint(timestamp)
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
                save_checkpoint(timestamp)
                continue

            # GKG 2.0 schema (simplified):
            # 1: DATE (YYYYMMDDHHMMSS)
            # 3: SourceCommonName (e.g. reuters.com)
            # 4: DocumentIdentifier (URL)
            # 8: V2Themes
            # 10: V2Locations
            # 12: V2Persons
            # 14: V2Organizations
            # 15: V2Tone (csv: avg,pos,neg,polarity,activity,selfgroup)
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

                    # Tone
                    tone_str = str(row[15])
                    tone_avg = tone_pos = tone_neg = tone_polarity = 0.0
                    if isinstance(tone_str, str) and tone_str:
                        parts = tone_str.split(",")
                        try:
                            if len(parts) > 0:
                                tone_avg = float(parts[0])
                            if len(parts) > 1:
                                tone_pos = float(parts[1])
                            if len(parts) > 2:
                                tone_neg = float(parts[2])
                            if len(parts) > 3:
                                tone_polarity = float(parts[3])
                        except Exception:
                            tone_avg = tone_pos = tone_neg = tone_polarity = 0.0

                    themes_raw = str(row[8]) if not pd.isna(row[8]) else ""
                    locations_raw = str(row[10]) if not pd.isna(row[10]) else ""
                    persons_raw = str(row[12]) if not pd.isna(row[12]) else ""
                    orgs_raw = str(row[14]) if not pd.isna(row[14]) else ""

                    writer.writerow([
                        domain,
                        ts.isoformat(),
                        raw_url,
                        tone_avg,
                        tone_pos,
                        tone_neg,
                        tone_polarity,
                        themes_raw,
                        locations_raw,
                        persons_raw,
                        orgs_raw,
                    ])
                except Exception:
                    continue

            # Save checkpoint at the end of this slot
            save_checkpoint(timestamp)

            # Progress & ETA every 200 slots
            if global_i % 200 == 0 or global_i == total_slots:
                elapsed = time() - start_ts
                pct = global_i / total_slots
                if pct > 0:
                    est_total = elapsed / pct
                    remaining = est_total - elapsed
                else:
                    remaining = 0.0
                remaining_hours = remaining / 3600.0
                print(
                    f"{global_i}/{total_slots} time slots processed... "
                    f"({pct * 100:.2f}% done, ~{remaining_hours:.2f}h remaining)"
                )

    print(f"Done. Saved filtered news to {OUTPUT_CSV}")


if __name__ == "__main__":
    fetch_news()
