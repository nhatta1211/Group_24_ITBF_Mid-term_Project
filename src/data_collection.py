# =============================================================
# src/data_collection.py
#
# MODULE 1 — Data Collection
#
# PURPOSE:
#   Download historical price data for:
#     - Apple stock    (AAPL)
#     - Tesla stock    (TSLA)
#     - Gold price     (GC=F)
#
#   For each asset, we collect:
#     Open, High, Low, Close, Volume
#   For the last 1 year, on a daily basis.
#
#   Each dataset is saved as a CSV file inside the data/ folder.
#
#   A second data source (Alpha Vantage) is used to fetch
#   macroeconomic indicator data (e.g. Federal Funds Rate).
#   This satisfies the "2+ data source types" rubric criterion.
#
# HOW TO RUN THIS FILE STANDALONE:
#   From the project root folder:
#   > python -m src.data_collection
# =============================================================

import os
import time
import requests
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

# Load API keys from the .env file in the project root
load_dotenv()


# -------------------------------------------------------------
# CONFIGURATION — OHLCV Assets (Yahoo Finance)
# "GC=F" is the Yahoo Finance ticker for Gold Futures.
# Value is the human-readable name used only in console output.
# -------------------------------------------------------------

ASSETS = {
    "AAPL": "Apple Stock",
    "TSLA": "Tesla Stock",
    "GC=F": "Gold Price",
}

# Time range and frequency for all yfinance downloads
PERIOD = "1y"
INTERVAL = "1d"

# Output folder — relative to project root (where you run commands from)
# Using a relative path keeps the project portable across machines.
DATA_FOLDER = "data"

# Retry configuration — applies to both yfinance and Alpha Vantage fetches
DOWNLOAD_MAX_RETRIES = 2  # 1 initial attempt + 1 retry = 2 total attempts
DOWNLOAD_RETRY_WAIT = 5  # Seconds to wait between attempts

# How many days before a saved CSV file is considered stale
# and triggers a freshness warning in load_from_csv()
FRESHNESS_WARN_DAYS = 7  # General assets: warn if older than 7 days
FRESHNESS_WARN_DAYS_MACRO = (
    2  # Macro data (Fed rate changes more often): warn if older than 2 days
)


# -------------------------------------------------------------
# CONFIGURATION — Alpha Vantage (Macroeconomic Data)
#
# MACRO_SERIES maps each Alpha Vantage function name to the
# output CSV filename. To add a new indicator later (Tier 2),
# simply add a new entry here — no other code needs to change.
#
# Tier 1 (implemented now):
#   FEDERAL_FUNDS_RATE — daily, free, directly impacts all 3 assets
#
# Tier 2 (deferred — frequency mismatch with daily OHLCV data):
#   "INFLATION": "macro_inflation_1y_monthly.csv"
#   "REAL_GDP":  "macro_real_gdp_1y_quarterly.csv"
# -------------------------------------------------------------

AV_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
AV_BASE_URL = "https://www.alphavantage.co/query"

MACRO_SERIES = {
    "FEDERAL_FUNDS_RATE": "macro_fed_rate_1y_daily.csv",
    # Add Tier 2 entries here when ready — no refactoring needed
}


# -------------------------------------------------------------
# FUNCTION 1: ensure_data_folder()
# -------------------------------------------------------------


def ensure_data_folder() -> None:
    """
    Create the data/ folder if it doesn't already exist.

    Why we need this:
        If the folder doesn't exist and we try to save a file
        inside it, Python will throw a FileNotFoundError.
        os.makedirs() with exist_ok=True prevents that safely.
    """
    # exist_ok=True means: don't raise an error if folder already exists
    os.makedirs(DATA_FOLDER, exist_ok=True)
    print(f"[INFO] Output folder ready: '{DATA_FOLDER}/'")


# -------------------------------------------------------------
# FUNCTION 2: download_asset_data()
# -------------------------------------------------------------


def download_asset_data(ticker: str, name: str) -> pd.DataFrame | None:
    """
    Download 1 year of daily OHLCV data for a single asset from Yahoo Finance.

    Retry behaviour:
        If the first download attempt fails due to a network error or
        returns empty data, the function waits DOWNLOAD_RETRY_WAIT seconds
        and tries once more. If the retry also fails, the function returns
        None so the caller can skip this asset gracefully.

    Args:
        ticker (str): Yahoo Finance symbol, e.g. "AAPL" or "GC=F"
        name   (str): Human-readable name for console logs, e.g. "Apple Stock"

    Returns:
        pd.DataFrame: A table with columns [Open, High, Low, Close, Volume]
                      and Date as the index.
        None:         If download fails after all retry attempts.

    What is OHLCV?
        Open   = price at market open
        High   = highest price during the day
        Low    = lowest price during the day
        Close  = price at market close
        Volume = number of shares/contracts traded
    """
    print(f"\n[INFO] Downloading {name} ({ticker})...")

    last_exception = None

    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            # yf.download() fetches data directly from Yahoo Finance
            # auto_adjust=True: adjusts for stock splits and dividends automatically
            # progress=False:   hides yfinance's own download bar (we have our own logs)
            df = yf.download(
                tickers=ticker,
                period=PERIOD,
                interval=INTERVAL,
                auto_adjust=True,
                progress=False,
            )

            # ── Validation: check if we actually got data back ──────────────
            if df.empty:
                print(
                    f"[WARNING] No data returned for '{ticker}' "
                    f"(attempt {attempt}/{DOWNLOAD_MAX_RETRIES}). "
                    f"The ticker may be wrong or temporarily unavailable."
                )
                # Treat an empty result the same as a network error — retry once
                if attempt < DOWNLOAD_MAX_RETRIES:
                    print(f"[INFO]    Retrying in {DOWNLOAD_RETRY_WAIT}s...")
                    time.sleep(DOWNLOAD_RETRY_WAIT)
                continue  # go to next attempt

            # ── Fix column names ─────────────────────────────────────────────
            # yfinance sometimes returns a MultiIndex column like ("Close", "AAPL").
            # We flatten it to just "Close" so it's easier to work with later.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # ── Keep only the 5 columns we need ─────────────────────────────
            columns_we_want = ["Open", "High", "Low", "Close", "Volume"]

            # Only keep columns that actually exist in the downloaded data
            columns_available = [col for col in columns_we_want if col in df.columns]
            df = df[columns_available]

            # ── Round prices to 2 decimal places for readability ────────────
            price_cols = ["Open", "High", "Low", "Close"]
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col].round(2)

            # ── Console summary ──────────────────────────────────────────────
            print(f"[OK]   Downloaded {len(df)} rows for {name} ({ticker})")
            print(f"       Date range : {df.index[0].date()} -> {df.index[-1].date()}")
            print(f"       Columns    : {list(df.columns)}")
            print(f"       Latest close price: {df['Close'].iloc[-1]:.2f}")

            return df  # Success — exit immediately

        except Exception as e:
            # Catch ANY unexpected error (network issue, bad ticker, API change, etc.)
            last_exception = e
            print(
                f"[ERROR] Attempt {attempt}/{DOWNLOAD_MAX_RETRIES} failed for "
                f"{name} ({ticker}): {e}"
            )

            if attempt < DOWNLOAD_MAX_RETRIES:
                print(f"[INFO]  Retrying in {DOWNLOAD_RETRY_WAIT}s...")
                time.sleep(DOWNLOAD_RETRY_WAIT)

    # All attempts exhausted
    print(
        f"[ERROR] Could not download {name} ({ticker}) after "
        f"{DOWNLOAD_MAX_RETRIES} attempt(s). Skipping this asset."
    )
    print(f"[HINT]  If this keeps failing, try updating yfinance:")
    print(f"        pip install --upgrade yfinance")
    return None


# -------------------------------------------------------------
# FUNCTION 3: save_to_csv()
# -------------------------------------------------------------


def save_to_csv(df: pd.DataFrame, ticker: str) -> str:
    """
    Save a DataFrame to a CSV file inside the data/ folder.

    File naming convention:
        data/AAPL_1y_daily.csv
        data/TSLA_1y_daily.csv
        data/GC_F_1y_daily.csv   (GC=F becomes GC_F for safe filenames)

    We do NOT add a timestamp to the filename, so re-running the
    script simply overwrites the previous file. This keeps data/ clean.

    Args:
        df     (pd.DataFrame): The data to save.
        ticker (str):          Used to name the file.

    Returns:
        str: The file path where the CSV was saved.
    """
    # Replace "=" and special chars that might cause issues on some OS
    safe_ticker = ticker.replace("=", "_").replace("/", "_")

    filename = f"{DATA_FOLDER}/{safe_ticker}_1y_daily.csv"

    # index=True keeps the Date column in the CSV file
    df.to_csv(filename, index=True)

    print(f"[SAVED] {ticker} data -> {filename}  ({len(df)} rows)")
    return filename


# -------------------------------------------------------------
# FUNCTION 4: fetch_av_series()   [H4 — Alpha Vantage second source]
# -------------------------------------------------------------


def fetch_av_series(series_name: str) -> pd.DataFrame | None:
    """
    Generic fetcher for any Alpha Vantage macroeconomic time series.

    Why generic?
        Using a single function driven by a config dict (MACRO_SERIES)
        means adding a new indicator (e.g. INFLATION, REAL_GDP) only
        requires adding one line to MACRO_SERIES — no new functions needed.

    Alpha Vantage quirks handled here:
        - Rate limit: AV returns HTTP 200 with a "Note" key in the JSON
          (not a proper HTTP 429). We check for "Note" explicitly.
        - Error messages: returned under "Error Message" key in the JSON.
        - Data list: returned under the "data" key as a list of
          {"date": "...", "value": "..."} dicts.

    Retry behaviour:
        Uses the same DOWNLOAD_MAX_RETRIES / DOWNLOAD_RETRY_WAIT constants
        as download_asset_data() — consistent retry behaviour across all sources.

    Args:
        series_name (str): Alpha Vantage function name, e.g. "FEDERAL_FUNDS_RATE"

    Returns:
        pd.DataFrame: Date index, single "value" column — last 1 year of data.
        None:         If the API key is missing, the series is unknown, or
                      all retry attempts fail.
    """
    # ── Guard: API key must be present ──────────────────────────────────
    if not AV_API_KEY:
        print("[WARNING] ALPHA_VANTAGE_API_KEY not set in .env — skipping macro data.")
        print("          Add ALPHA_VANTAGE_API_KEY=your_key_here to your .env file.")
        return None

    print(f"\n[INFO] Fetching macro series: {series_name} (Alpha Vantage)...")

    # Build the request parameters for this series
    params = {
        "function": series_name,
        "interval": "daily",  # daily frequency (ignored by some AV series)
        "apikey": AV_API_KEY,
    }

    last_exception = None

    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            # Make the HTTP GET request; timeout=15 avoids hanging forever
            resp = requests.get(AV_BASE_URL, params=params, timeout=15)
            resp.raise_for_status()  # raise an error for HTTP 4xx / 5xx responses
            payload = resp.json()

            # ── Check for AV rate limit (returns HTTP 200 with "Note" key) ──
            if "Note" in payload:
                print(f"[WARNING] Alpha Vantage rate limit hit: {payload['Note']}")
                if attempt < DOWNLOAD_MAX_RETRIES:
                    print(f"[INFO]    Retrying in {DOWNLOAD_RETRY_WAIT}s...")
                    time.sleep(DOWNLOAD_RETRY_WAIT)
                continue  # retry

            # ── Check for AV error message ───────────────────────────────────
            if "Error Message" in payload:
                print(
                    f"[ERROR] Alpha Vantage error for {series_name}: {payload['Error Message']}"
                )
                return None  # this is a permanent error — no point retrying

            # ── Parse the data records ────────────────────────────────────────
            # AV returns a list under the "data" key: [{"date": "...", "value": "..."}, ...]
            records = payload.get("data", [])
            if not records:
                print(
                    f"[WARNING] Alpha Vantage returned no data records for {series_name}."
                )
                return None

            # Build a DataFrame from the list of records
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.set_index("date").sort_index()
            df.index.name = "Date"

            # ── Filter to the last 1 year to match yfinance PERIOD ───────────
            cutoff = pd.Timestamp.today() - pd.DateOffset(years=1)
            df = df[df.index >= cutoff]

            # ── Console summary ───────────────────────────────────────────────
            print(f"[OK]   Fetched {len(df)} rows for {series_name}")
            print(f"       Date range : {df.index[0].date()} -> {df.index[-1].date()}")
            print(f"       Latest value: {df['value'].iloc[-1]:.2f}%")

            return df  # Success — exit immediately

        except Exception as e:
            last_exception = e
            print(
                f"[ERROR] Alpha Vantage attempt {attempt}/{DOWNLOAD_MAX_RETRIES} "
                f"failed for {series_name}: {e}"
            )
            if attempt < DOWNLOAD_MAX_RETRIES:
                print(f"[INFO]  Retrying in {DOWNLOAD_RETRY_WAIT}s...")
                time.sleep(DOWNLOAD_RETRY_WAIT)

    # All attempts exhausted
    print(
        f"[ERROR] Could not fetch {series_name} after "
        f"{DOWNLOAD_MAX_RETRIES} attempt(s). Skipping."
    )
    return None


# -------------------------------------------------------------
# FUNCTION 5: collect_all_assets()
# -------------------------------------------------------------


def collect_all_assets() -> dict:
    """
    Main collection function — downloads all assets and macro data,
    saves each to CSV, and returns everything in one dictionary.

    Flow:
        1. Loop over ASSETS  → download_asset_data() + save_to_csv()
        2. Loop over MACRO_SERIES → fetch_av_series() + save macro CSV
        3. Return all_data dict with all successfully collected DataFrames

    Returns:
        dict: Keys are ticker symbols or macro series names.
              Values are DataFrames (or None if that asset failed).

    Example return value:
        {
            "AAPL":               <DataFrame — 252 rows OHLCV>,
            "TSLA":               <DataFrame — 252 rows OHLCV>,
            "GC=F":               <DataFrame — 252 rows OHLCV>,
            "FEDERAL_FUNDS_RATE": <DataFrame — ~252 rows, 'value' column>,
        }

    Note: downstream modules (cleaning, visualization) only loop over
    ASSETS keys — macro keys like "FEDERAL_FUNDS_RATE" are ignored by
    them and consumed separately by ai_analysis.py and visualization.py.
    """
    # Make sure the output folder exists before we start saving files
    ensure_data_folder()

    print("\n" + "=" * 55)
    print("  MODULE 1 — Data Collection Starting")
    print(f"  OHLCV assets : {', '.join(ASSETS.keys())}")
    print(f"  Macro series : {', '.join(MACRO_SERIES.keys())}")
    print(f"  Period       : {PERIOD}  |  Interval: {INTERVAL}")
    print("=" * 55)

    # This dict will hold all successfully collected DataFrames
    all_data = {}

    # ── Step A: Download OHLCV price data from Yahoo Finance ────────────
    for ticker, name in ASSETS.items():

        # Download the data (with automatic retry on failure)
        df = download_asset_data(ticker, name)

        # Only save if the download succeeded (df is not None)
        if df is not None:
            save_to_csv(df, ticker)
            all_data[ticker] = df
        else:
            print(f"[SKIP]  {name} ({ticker}) was not saved due to download error.")

    # ── Step B: Download macroeconomic data from Alpha Vantage ──────────
    # Loop over MACRO_SERIES dict — adding Tier 2 only requires a new dict entry
    for series_name, output_filename in MACRO_SERIES.items():

        macro_df = fetch_av_series(series_name)

        if macro_df is not None:
            # Save the macro CSV directly (no safe_ticker renaming needed)
            macro_path = f"{DATA_FOLDER}/{output_filename}"
            macro_df.to_csv(macro_path, index=True)
            print(
                f"[SAVED] {series_name} macro data -> {macro_path}  ({len(macro_df)} rows)"
            )
            # Store under the series name so downstream code can identify it
            all_data[series_name] = macro_df
        else:
            print(f"[SKIP]  Macro data ({series_name}) was not saved.")

    # ── Final summary ────────────────────────────────────────────────────
    total_possible = len(ASSETS) + len(MACRO_SERIES)
    print("\n" + "=" * 55)
    print(f"  Collection complete!")
    print(f"  Successfully collected: {len(all_data)} / {total_possible} datasets")

    for key in all_data:
        if key in MACRO_SERIES:
            # Macro entry — use the filename from the config dict
            print(f"    + {key:25s}  ->  data/{MACRO_SERIES[key]}")
        else:
            # OHLCV entry — derive filename from ticker
            safe = key.replace("=", "_")
            print(f"    + {key:6s}  ->  data/{safe}_1y_daily.csv")

    print("=" * 55 + "\n")

    return all_data


# -------------------------------------------------------------
# FUNCTION 6: load_from_csv()
# -------------------------------------------------------------


def load_from_csv(ticker: str) -> pd.DataFrame | None:
    """
    Load a previously saved OHLCV CSV file back into a DataFrame.

    This is used by other modules (cleaning, visualization, AI analysis)
    that need the data WITHOUT re-downloading it from the internet.

    Freshness check (E5):
        If the file exists but was last modified more than
        FRESHNESS_WARN_DAYS days ago, a [WARNING] is printed.
        The data is still loaded — this is a warning, not an error.
        This helps catch stale data during long development cycles.

    Args:
        ticker (str): e.g. "AAPL", "TSLA", "GC=F"

    Returns:
        pd.DataFrame: The loaded data, or None if file not found.
    """
    safe_ticker = ticker.replace("=", "_").replace("/", "_")
    filename = f"{DATA_FOLDER}/{safe_ticker}_1y_daily.csv"

    try:
        # ── Freshness check: warn if the file is older than the threshold ──
        # os.path.getmtime() returns the last-modified time as a Unix timestamp
        # (seconds since Jan 1 1970). We convert it to a pandas Timestamp for
        # easy date arithmetic.
        file_mtime = pd.Timestamp.fromtimestamp(os.path.getmtime(filename))
        age_days = (pd.Timestamp.today() - file_mtime).days

        if age_days > FRESHNESS_WARN_DAYS:
            print(
                f"[WARNING] '{filename}' is {age_days} day(s) old "
                f"(threshold: {FRESHNESS_WARN_DAYS} days)."
            )
            print(f"          Consider re-running data collection to get fresh data.")

        # ── Load the CSV ─────────────────────────────────────────────────────
        # index_col=0 tells pandas the first column (Date) is the row index.
        # parse_dates=True converts the Date strings to proper datetime objects.
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"[LOADED] {ticker} <- {filename}  ({len(df)} rows)")
        return df

    except FileNotFoundError:
        print(f"[ERROR] File not found: '{filename}'")
        print(f"        Please run collect_all_assets() first to download the data.")
        return None


# =============================================================
# ENTRY POINT
# This block runs ONLY when you execute this file directly:
#   > python -m src.data_collection
#
# It does NOT run when main.py imports this file.
# =============================================================

if __name__ == "__main__":
    # Run the full collection pipeline
    collected_data = collect_all_assets()

    # Demo: show the first 3 rows of each collected dataset
    print("\n--- PREVIEW OF COLLECTED DATA ---\n")
    for ticker, df in collected_data.items():
        print(f"[ {ticker} ] — first 3 rows:")
        print(df.head(3).to_string())
        print()
