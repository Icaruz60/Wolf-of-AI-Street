import os
import csv
from datetime import datetime, timedelta
from io import BytesIO
from urllib.parse import urlparse
from time import time
import concurrent.futures
import threading

import pandas as pd
import requests

# ===============================
# CONFIG
# ===============================

OUTPUT_CSV = "all_news_raw.csv"
CHECKPOINT_FILE = "gdelt_checkpoint.txt"

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

# Number of parallel workers.
# 10–12 is a good upper bound without angering GDELT.
MAX_WORKERS = 12

# Thread-local storage for per-thread requests.Session
thread_local = threading.local()


# ===============================
# HELPERS
# ===============================

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


def get_session() -> requests.Session:
    """
    Get or create a requests.Session for the current thread.
    This lets us reuse TCP connections (keep-alive) per thread.
    """
    if not hasattr(thread_local, "session"):
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=2,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        thread_local.session = session
    return thread_local.session


def fetch_gkg_zip_bytes(url: str):
    """
    Download a GDELT GKG zip file into memory using a per-thread Session.

    Returns: (BytesIO_or_None, error_code_or_None)
      error_code ∈ {"timeout","http_status","network","other"} or None
    """
    try:
        session = get_session()
        r = session.get(url, timeout=10)
        if r.status_code != 200:
            return None, "http_status"
        return BytesIO(r.content), None
    except requests.exceptions.Timeout:
        return None, "timeout"
    except requests.exceptions.RequestException:
        # ConnectionError, etc.
        return None, "network"
    except Exception:
        return None, "other"


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


def index_to_timestamp(idx: int) -> datetime:
    """Convert a global slot index to a timestamp."""
    return START_DATE + timedelta(minutes=15 * idx)


def timestamp_to_index(ts: datetime) -> int:
    """Convert a timestamp to a global slot index (inverse of index_to_timestamp)."""
    delta = ts - START_DATE
    return int(delta.total_seconds() // 900)  # 900s = 15 minutes


# ===============================
# WORKER
# ===============================

def process_slot(slot_index: int, ts: datetime):
    """
    Worker: download + parse one GDELT 15-minute file.

    Returns: (slot_index, ts, rows, info)
      rows: list of CSV rows to append
      info: {"fetch_error": str|None, "parse_error": bool}
    """
    date_str = ts.strftime("%Y%m%d%H%M00")
    url = GDELT_GKG_URL.format(date=date_str)

    zip_bytes, fetch_err = fetch_gkg_zip_bytes(url)
    if zip_bytes is None:
        # Nothing fetched; treat as processed but with zero rows.
        return slot_index, ts, [], {"fetch_error": fetch_err, "parse_error": False}

    try:
        df = pd.read_csv(
            zip_bytes,
            compression="zip",
            sep="\t",
            header=None,
            low_memory=False,
        )
    except Exception:
        # File unreadable or corrupted.
        return slot_index, ts, [], {"fetch_error": fetch_err, "parse_error": True}

    rows = []

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
            art_ts = datetime.strptime(raw_time, "%Y%m%d%H%M%S")

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

            rows.append([
                domain,
                art_ts.isoformat(),
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
            # Skip bad line, keep going
            continue

    return slot_index, ts, rows, {"fetch_error": fetch_err, "parse_error": False}


# ===============================
# MAIN
# ===============================

def fetch_news_multithreaded():
    # Total number of 15-min slots over the whole period.
    total_slots = int((END_DATE - START_DATE).total_seconds() / 900) + 1

    checkpoint_ts = load_checkpoint()
    if checkpoint_ts is not None and checkpoint_ts > START_DATE:
        last_done_idx = timestamp_to_index(checkpoint_ts)
        processed_before = last_done_idx + 1
        print(f"Resuming from checkpoint: {checkpoint_ts} (global slot {last_done_idx})")
    else:
        last_done_idx = -1
        processed_before = 0
        print("No valid checkpoint found, starting from BEGINNING.")

    print(
        f"Fetching GDELT GKG from {START_DATE} to {END_DATE} "
        f"({total_slots} total slots, {processed_before} already processed)"
    )

    # Build list of remaining slots: (index, timestamp)
    slots_to_process = []
    for idx in range(processed_before, total_slots):
        ts = index_to_timestamp(idx)
        if ts > END_DATE:
            break
        slots_to_process.append((idx, ts))

    if not slots_to_process:
        print("Nothing to do. All slots already processed.")
        return

    append_mode = processed_before > 0
    file_mode = "a" if append_mode else "w"

    # Tracks which global slot indices have completed in THIS run
    completed_indices = set()
    last_checkpoint_idx = processed_before - 1

    # Error tracking
    fetch_error_counts = {
        "timeout": 0,
        "http_status": 0,
        "network": 0,
        "other": 0,
        "worker_exception": 0,
    }
    slots_with_fetch_error = 0
    slots_with_parse_error = 0

    start_ts = time()

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

        # Thread pool for parallel downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_slot = {
                executor.submit(process_slot, idx, ts): (idx, ts)
                for (idx, ts) in slots_to_process
            }

            total_futures = len(future_to_slot)

            for processed_count, future in enumerate(
                concurrent.futures.as_completed(future_to_slot), start=1
            ):
                idx, ts = future_to_slot[future]

                try:
                    slot_index, slot_ts, rows, info = future.result()
                except Exception:
                    # Worker crashed; treat as processed with no rows, log as worker_exception.
                    slot_index, slot_ts, rows, info = (
                        idx,
                        ts,
                        [],
                        {"fetch_error": "worker_exception", "parse_error": False},
                    )

                # Write rows to CSV (main thread only)
                if rows:
                    writer.writerows(rows)

                # Error accounting
                fe = info.get("fetch_error")
                if fe:
                    if fe not in fetch_error_counts:
                        fetch_error_counts[fe] = 0
                    fetch_error_counts[fe] += 1
                    slots_with_fetch_error += 1

                if info.get("parse_error"):
                    slots_with_parse_error += 1

                # Mark this slot as completed
                completed_indices.add(slot_index)

                # Advance checkpoint as far as we have a contiguous run
                while (last_checkpoint_idx + 1) in completed_indices:
                    last_checkpoint_idx += 1
                    cp_ts = index_to_timestamp(last_checkpoint_idx)
                    save_checkpoint(cp_ts)

                # Progress & ETA every 200 completed futures or at the end
                if processed_count % 200 == 0 or processed_count == total_futures:
                    elapsed = time() - start_ts

                    # How many slots are done in total (including previous runs)
                    total_done = processed_before + processed_count
                    pct = total_done / total_slots if total_slots > 0 else 0.0

                    if pct > 0:
                        est_total = elapsed / pct
                        remaining = est_total - elapsed
                    else:
                        remaining = 0.0

                    remaining_hours = remaining / 3600.0

                    timeouts = fetch_error_counts.get("timeout", 0)
                    http_errs = fetch_error_counts.get("http_status", 0)
                    net_errs = fetch_error_counts.get("network", 0)
                    other_errs = fetch_error_counts.get("other", 0)
                    worker_errs = fetch_error_counts.get("worker_exception", 0)

                    print(
                        f"{total_done}/{total_slots} time slots processed... "
                        f"({pct * 100:.2f}% done, ~{remaining_hours:.2f}h remaining) | "
                        f"errors so far: fetch_slots={slots_with_fetch_error} "
                        f"(timeouts={timeouts}, http={http_errs}, net={net_errs}, "
                        f"other={other_errs}, worker={worker_errs}), "
                        f"parse_slots={slots_with_parse_error}"
                    )

    print(f"Done. Saved filtered news to {OUTPUT_CSV}")


if __name__ == "__main__":
    fetch_news_multithreaded()
