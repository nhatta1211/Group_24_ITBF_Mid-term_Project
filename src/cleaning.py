# =============================================================
# src/cleaning.py
#
# MODULE 2 — Data Cleaning & Feature Engineering
#
# PURPOSE:
#   Take raw CSV files from Module 1 (data/ folder),
#   clean them, and add useful financial indicators.
#
#   Steps performed:
#     1. Load CSV files from data/ folder
#     2. Handle missing values
#     3. Remove duplicate rows
#     4. Convert date column properly
#     5. Normalize numeric data types
#     6. Detect outliers
#     7. Add feature engineering columns:
#          - Daily Return
#          - 7-day Moving Average
#          - 30-day Moving Average
#          - Volatility (20-day rolling std)
#     8. Save cleaned data to output/ folder
#
# HOW TO RUN THIS FILE STANDALONE:
#   From the project root folder:
#   > python -m src.cleaning
# =============================================================

import os
import pandas as pd
import numpy as np


# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------

# Where to read raw data from (output of Module 1)
DATA_FOLDER   = "data"    # Where raw CSVs are stored (output of Module 1)

# Where to save cleaned data
OUTPUT_FOLDER = "output"  # Where cleaned CSVs will be saved

# The 3 assets we collected in Module 1
# Key   = ticker symbol
# Value = safe filename (GC=F becomes GC_F for filenames)
ASSETS = {
    "AAPL": "AAPL",
    "TSLA": "TSLA",
    "GC=F": "GC_F",
}

# Outlier detection threshold (Z-score method)
# Any value with Z-score above this is considered an outlier
# Z-score = how many standard deviations away from the mean
# 3.0 is the standard threshold used in finance
OUTLIER_ZSCORE_THRESHOLD = 3.0


# -------------------------------------------------------------
# FUNCTION 1: ensure_output_folder()
# -------------------------------------------------------------

def ensure_output_folder() -> None:
    """
    Create the output/ folder if it doesn't already exist.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"[INFO] Output folder ready: '{OUTPUT_FOLDER}/'")


# -------------------------------------------------------------
# FUNCTION 2: load_csv()
# -------------------------------------------------------------

def load_csv(ticker: str, safe_ticker: str) -> pd.DataFrame | None:
    """
    Load a raw CSV file (saved by Module 1) into a DataFrame.

    Args:
        ticker      (str): e.g. "AAPL", "GC=F"
        safe_ticker (str): safe filename version, e.g. "AAPL", "GC_F"

    Returns:
        pd.DataFrame: The loaded raw data.
        None:         If file is not found.
    """
    filename = f"{DATA_FOLDER}/{safe_ticker}_1y_daily.csv"

    try:
        # index_col=0    → first column (Date) becomes the row index
        # parse_dates=True → convert Date strings to proper datetime objects
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"[LOADED] {ticker} <- {filename}  ({len(df)} rows, {len(df.columns)} columns)")
        return df

    except FileNotFoundError:
        print(f"[ERROR] File not found: '{filename}'")
        print(f"        Please run Module 1 (data_collection.py) first.")
        return None


# -------------------------------------------------------------
# FUNCTION 2B: validate_raw_data()
# -------------------------------------------------------------

def validate_raw_data(df: pd.DataFrame, ticker: str) -> None:
    """
    Validate a freshly loaded raw DataFrame before any cleaning begins.

    Why validate early?
        If the raw data is fundamentally broken (empty, wrong columns,
        all-NaN), every downstream step will either crash with a confusing
        error or silently produce garbage output. Checking here — right
        after load — means we fail fast with a clear, descriptive message
        instead of a cryptic KeyError or ZeroDivisionError later.

    Checks performed:
        1. DataFrame is not empty (no rows at all).
        2. All required OHLCV columns are present.
        3. At least one required column has at least one non-NaN value
           (guards against a file that loaded but is entirely blank).

    Args:
        df     (pd.DataFrame): The raw DataFrame returned by load_csv().
        ticker (str):          Ticker symbol — used in error messages.

    Returns:
        None: If all checks pass, the function returns silently.

    Raises:
        ValueError: If any check fails. The message explains exactly
                    what is wrong so the user knows how to fix it.
    """
    # The five columns every raw OHLCV file must have
    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    # ── Check 1: DataFrame must not be empty ────────────────────────────
    if df.empty:
        raise ValueError(
            f"[{ticker}] Raw data is empty — the CSV file has no rows. "
            f"Re-run Module 1 (data_collection.py) to re-download the data."
        )

    # ── Check 2: All required columns must be present ───────────────────
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[{ticker}] Raw data is missing required columns: {missing_cols}. "
            f"Expected columns: {required_columns}. "
            f"Re-run Module 1 to re-download with the correct schema."
        )

    # ── Check 3: At least one required column must have real data ───────
    # all-NaN means the file loaded but every cell is blank — a corrupted
    # or empty download. We check all required columns together.
    all_nan_cols = [col for col in required_columns if df[col].isna().all()]
    if len(all_nan_cols) == len(required_columns):
        raise ValueError(
            f"[{ticker}] All required columns are entirely NaN — the file "
            f"appears to be corrupted or blank. "
            f"Re-run Module 1 to re-download a fresh copy."
        )

    # All checks passed — print confirmation
    print(f"[{ticker}] Raw data validated: {len(df)} rows, all required columns present.")


# -------------------------------------------------------------
# FUNCTION 3: handle_missing_values()
# -------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Detect and handle missing (NaN) values in the DataFrame.

    Strategy used:
        - For price columns (Open, High, Low, Close):
          Forward-fill → use the previous day's value.
          This is standard in finance because if a price is missing,
          the last known price is the best estimate.

        - For Volume:
          Fill with 0 — missing volume usually means no trading occurred.

        - Any remaining NaN: drop the entire row.

    Args:
        df     (pd.DataFrame): Raw DataFrame.
        ticker (str):          Used for log messages.

    Returns:
        pd.DataFrame: DataFrame with no missing values.
    """
    # Count missing values BEFORE fixing
    missing_before = df.isnull().sum().sum()  # Total NaN count across all columns

    if missing_before == 0:
        print(f"[{ticker}] No missing values found. Skipping.")
        return df

    print(f"[{ticker}] Missing values detected: {missing_before} total")

    # Show which columns have missing values
    per_column = df.isnull().sum()
    for col, count in per_column.items():
        if count > 0:
            print(f"         Column '{col}': {count} missing values")

    # ── Forward-fill price columns ───────────────────────────────────────
    # ffill() = forward fill: copy the value from the previous row
    price_cols = ["Open", "High", "Low", "Close"]
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # ── Fill Volume with 0 ───────────────────────────────────────────────
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    # ── Drop any remaining rows that still have NaN ──────────────────────
    df.dropna(inplace=True)

    # Count missing values AFTER fixing
    missing_after = df.isnull().sum().sum()
    print(f"[{ticker}] Missing values after cleaning: {missing_after}")

    return df


# -------------------------------------------------------------
# FUNCTION 4: remove_duplicates()
# -------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    A duplicate row = same date appearing more than once.
    We keep the first occurrence and drop the rest.

    Args:
        df     (pd.DataFrame): DataFrame to check.
        ticker (str):          Used for log messages.

    Returns:
        pd.DataFrame: DataFrame with no duplicate rows.
    """
    duplicates_count = df.index.duplicated().sum()  # Count duplicate dates

    if duplicates_count == 0:
        print(f"[{ticker}] No duplicate rows found. Skipping.")
        return df

    print(f"[{ticker}] Found {duplicates_count} duplicate rows — removing...")

    # keep="first" → keep the first occurrence, drop the rest
    df = df[~df.index.duplicated(keep="first")]

    print(f"[{ticker}] Rows after removing duplicates: {len(df)}")
    return df


# -------------------------------------------------------------
# FUNCTION 5: convert_date_column()
# -------------------------------------------------------------

def convert_date_column(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Ensure the Date index is properly formatted as datetime.

    Why this matters:
        If the date is stored as a plain string (e.g. "2024-01-15"),
        Python cannot do date math like "give me the last 30 days".
        Converting to datetime unlocks time-series operations.

    Also:
        - Sort the data chronologically (oldest date first)
        - Remove timezone information to avoid conflicts

    Args:
        df     (pd.DataFrame): DataFrame with a date-based index.
        ticker (str):          Used for log messages.

    Returns:
        pd.DataFrame: DataFrame with a clean DatetimeIndex.
    """
    # Convert index to datetime (in case it wasn't already)
    # No format string — pandas infers automatically.
    # yfinance saves dates as YYYY-MM-DD (ISO format), and parse_dates=True
    # in load_csv() already handles this. This call is a safety net for
    # any edge case where the index is still a plain string.
    df.index = pd.to_datetime(df.index, infer_datetime_format=True)

    # Remove timezone info if present (e.g. "2024-01-15 00:00:00+00:00" → "2024-01-15")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Sort chronologically — oldest date first
    df.sort_index(ascending=True, inplace=True)

    # Give the index a clear name
    df.index.name = "Date"

    print(f"[{ticker}] Date range: {df.index[0].date()} -> {df.index[-1].date()}")
    return df


# -------------------------------------------------------------
# FUNCTION 6: normalize_numeric_types()
# -------------------------------------------------------------

def normalize_numeric_types(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Ensure all price and volume columns have the correct numeric type.

    Why this matters:
        Sometimes pandas reads numbers as strings (e.g. "198.53" instead
        of 198.53). If that happens, math operations like addition or
        averaging will fail or give wrong results.

    Conversion rules:
        - Open, High, Low, Close → float64 (decimal numbers)
        - Volume                 → int64   (whole numbers, no decimals)

    Args:
        df     (pd.DataFrame): DataFrame to normalize.
        ticker (str):          Used for log messages.

    Returns:
        pd.DataFrame: DataFrame with correct numeric types.
    """
    # Define expected data types for each column
    type_map = {
        "Open":   "float64",
        "High":   "float64",
        "Low":    "float64",
        "Close":  "float64",
        "Volume": "int64",
    }

    for col, dtype in type_map.items():
        if col in df.columns:
            # errors="coerce" → if a value can't convert, replace with NaN
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype, errors="ignore")

    print(f"[{ticker}] Column types normalized:")
    for col in df.columns:
        print(f"         {col:10s}: {df[col].dtype}")

    return df


# -------------------------------------------------------------
# FUNCTION 7: detect_outliers()
# -------------------------------------------------------------

def detect_outliers(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Detect and flag outliers in price columns using the Z-score method.

    What is a Z-score?
        Z = (value - mean) / standard_deviation

        It tells you how many standard deviations a value is from
        the average. For example:
            Z = 0   → exactly average
            Z = 2   → 2 standard deviations above average
            Z = -3  → 3 standard deviations BELOW average (outlier!)

        In a normal distribution:
            95% of values fall within Z = ±2
            99.7% of values fall within Z = ±3

        So if |Z| > 3.0, the value is very unusual — likely an outlier.

    Strategy:
        We ADD an 'Is_Outlier' column (True/False) but do NOT remove
        the outlier rows. In finance, extreme moves are real events
        (earnings, crashes) — not errors. We flag them for awareness.

    Args:
        df     (pd.DataFrame): Cleaned DataFrame.
        ticker (str):          Used for log messages.

    Returns:
        pd.DataFrame: Same DataFrame with added 'Is_Outlier' column.
    """
    # We detect outliers on Daily_Return, not Close price.
    #
    # Why Daily_Return and not Close?
    #   Close price trends upward over time, so its mean and std are skewed
    #   by the long-term trend — a Z-score on raw price almost never triggers
    #   and misses genuine shock days. Daily_Return is stationary (it does not
    #   trend), so its mean and std are stable, making Z-score reliable.
    #   This also matches what ai_analysis.py reports: outlier days are
    #   described as "Z-score > 3.0 on daily return".
    #
    # Requires add_daily_return() to have run first — enforced in clean_asset().
    col = "Daily_Return"

    if col not in df.columns:
        print(f"[{ticker}] Daily_Return column missing — skipping outlier detection.")
        df["Is_Outlier"] = False
        return df

    # Drop NaN before computing stats (first row of Daily_Return is always NaN)
    returns_clean = df[col].dropna()

    # Step 1: Calculate mean and standard deviation of Daily Return
    mean = returns_clean.mean()
    std  = returns_clean.std()

    if std == 0:
        # Edge case: all returns identical (extremely unlikely but safe to handle)
        df["Is_Outlier"] = False
        return df

    # Step 2: Calculate Z-score for each row
    # np.abs() gives us the absolute value (we care about distance, not direction)
    z_scores = np.abs((df[col] - mean) / std)

    # Step 3: Flag rows where Z-score exceeds threshold
    # Rows where Daily_Return is NaN get NaN z-score → False after comparison
    df["Is_Outlier"] = z_scores > OUTLIER_ZSCORE_THRESHOLD

    outlier_count = int(df["Is_Outlier"].sum())
    print(f"[{ticker}] Outliers detected (|Z| > {OUTLIER_ZSCORE_THRESHOLD} on Daily_Return): {outlier_count} rows")

    if outlier_count > 0:
        print(f"         Outlier rows:")
        print(df[df["Is_Outlier"]][["Close", "Daily_Return"]].to_string())

    return df


# -------------------------------------------------------------
# FUNCTION 8: add_daily_return()
# -------------------------------------------------------------

def add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Daily Return column.

    Formula:
        Daily Return (%) = ((Close_today - Close_yesterday) / Close_yesterday) * 100

    Example:
        Yesterday Close = 100.00
        Today Close     = 102.50
        Daily Return    = ((102.50 - 100.00) / 100.00) * 100 = +2.50%

    What it tells you:
        How much the asset gained or lost (in %) compared to the day before.
        Positive = price went up. Negative = price went down.

    Note:
        The first row will be NaN because there is no "yesterday" for it.
        We keep it as NaN (not filled) because inventing a value would be wrong.
    """
    # pct_change() calculates (today - yesterday) / yesterday automatically
    df["Daily_Return"] = df["Close"].pct_change() * 100
    df["Daily_Return"] = df["Daily_Return"].round(4)  # Round to 4 decimal places
    return df


# -------------------------------------------------------------
# FUNCTION 9: add_moving_averages()
# -------------------------------------------------------------

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 7-day and 30-day Simple Moving Averages (SMA).

    Formula:
        SMA_7  = average of Close price over the last 7 days
        SMA_30 = average of Close price over the last 30 days

    Example (SMA_7):
        Day 7 Close prices: [100, 102, 98, 105, 103, 101, 104]
        SMA_7 = (100 + 102 + 98 + 105 + 103 + 101 + 104) / 7 = 101.86

    What it tells you:
        Moving averages smooth out daily noise and reveal the trend.
        - SMA_7  = short-term trend (reacts quickly to price changes)
        - SMA_30 = medium-term trend (slower, more stable)

        When SMA_7 crosses ABOVE SMA_30 → potential uptrend (bullish signal)
        When SMA_7 crosses BELOW SMA_30 → potential downtrend (bearish signal)

    Note:
        - First 6 rows of SMA_7  will be NaN (not enough history yet)
        - First 29 rows of SMA_30 will be NaN (not enough history yet)
    """
    # rolling(window=N) → look at last N rows
    # .mean()           → calculate average of those N rows
    df["SMA_7"]  = df["Close"].rolling(window=7).mean().round(2)
    df["SMA_30"] = df["Close"].rolling(window=30).mean().round(2)
    return df


# -------------------------------------------------------------
# FUNCTION 10: add_volatility()
# -------------------------------------------------------------

def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day Rolling Volatility.

    Formula:
        Volatility = Standard Deviation of Daily Returns over the last 20 days

    Example:
        Last 20 days of Daily Returns: [+1.2%, -0.8%, +2.1%, ...]
        Volatility = std([+1.2%, -0.8%, +2.1%, ...]) = e.g. 1.45%

    What it tells you:
        Volatility measures HOW MUCH the price swings day-to-day.

        Low volatility  (e.g. 0.5%) = price is stable, low risk
        High volatility (e.g. 3.0%) = price swings wildly, high risk

        In finance, higher volatility = higher risk AND higher potential reward.
        Gold typically has lower volatility than Tesla.

    Note:
        Requires 'Daily_Return' column — run add_daily_return() first.
        First 19 rows will be NaN (not enough history yet).
    """
    if "Daily_Return" not in df.columns:
        print("[WARNING] Daily_Return column missing. Run add_daily_return() first.")
        return df

    # rolling(20).std() = standard deviation of the last 20 daily returns
    df["Volatility_20d"] = df["Daily_Return"].rolling(window=20).std().round(4)
    return df


# -------------------------------------------------------------
# FUNCTION 10B: add_cumulative_return()
# -------------------------------------------------------------

def add_cumulative_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Cumulative Return column.

    Formula:
        Cumulative Return (%) = ((Close_today - Close_first) / Close_first) * 100

    Example:
        First Close price:  $150.00
        Today Close price:  $172.50
        Cumulative Return:  ((172.50 - 150.00) / 150.00) * 100 = +15.0%

    What it tells you:
        How much the asset has gained or lost in total (%) since the
        start of the tracked period (Day 1 of the 1-year dataset).

        Unlike Daily_Return (which resets each day), Cumulative_Return
        keeps accumulating — it answers "if I bought on Day 1, what
        is my total return today?"

    Why useful:
        - Direct comparison across assets: you can plot all three
          cumulative return lines on the same chart to see which
          asset performed best over the same period
        - Used in performance benchmarking against indices (e.g. S&P 500)
        - The total return figure shown in ai_analysis.py reports
          is derived from this same calculation

    Note:
        Requires 'Close' column. First row will be 0.0% by definition
        (day 1 compared to itself).
    """
    if "Close" not in df.columns:
        print("[WARNING] Close column missing — skipping Cumulative_Return.")
        return df

    # Get the first valid (non-NaN) close price as the baseline
    first_price = df["Close"].dropna().iloc[0]

    # Calculate cumulative return as % change from Day 1
    df["Cumulative_Return"] = ((df["Close"] - first_price) / first_price * 100).round(4)

    return df


# -------------------------------------------------------------
# FUNCTION 12: save_clean_data()
# -------------------------------------------------------------

def save_clean_data(df: pd.DataFrame, ticker: str, safe_ticker: str) -> str:
    """
    Save the cleaned and enriched DataFrame to a CSV in the output/ folder.

    File naming convention:
        output/AAPL_clean.csv
        output/TSLA_clean.csv
        output/GC_F_clean.csv

    Args:
        df          (pd.DataFrame): Cleaned DataFrame to save.
        ticker      (str):          e.g. "AAPL"
        safe_ticker (str):          e.g. "GC_F" for safe filenames

    Returns:
        str: File path of the saved CSV.
    """
    filename = f"{OUTPUT_FOLDER}/{safe_ticker}_clean.csv"

    # index=True → keep the Date column in the CSV
    df.to_csv(filename, index=True)

    print(f"[SAVED] {ticker} clean data -> {filename}  ({len(df)} rows, {len(df.columns)} columns)")
    return filename


# -------------------------------------------------------------
# FUNCTION 13: clean_asset() — master cleaning pipeline
# -------------------------------------------------------------

def clean_asset(ticker: str, safe_ticker: str) -> pd.DataFrame | None:
    """
    Run the full cleaning pipeline for ONE asset.

    This function calls all the cleaning and feature engineering
    functions in the correct order.

    Pipeline order:
        1.  Load CSV
        1B. Validate raw data      ← fail fast before any processing
        2.  Convert date column
        3.  Handle missing values
        4.  Remove duplicates
        5.  Normalize numeric types
        6.  Add Daily Return       ← moved before outlier detection
        7.  Detect outliers        ← now uses Daily_Return (correct)
        8.  Add Moving Averages (SMA_7, SMA_30)
        9.  Add Volatility
        10. Add Cumulative Return
        11. Save to output/

    Args:
        ticker      (str): e.g. "AAPL", "GC=F"
        safe_ticker (str): e.g. "AAPL", "GC_F"

    Returns:
        pd.DataFrame: The fully cleaned and enriched DataFrame.
        None:         If loading fails.
    """
    print(f"\n{'─' * 50}")
    print(f"  Cleaning: {ticker}")
    print(f"{'─' * 50}")

    # Step 1: Load raw CSV
    df = load_csv(ticker, safe_ticker)
    if df is None:
        return None

    # Step 1B: Validate raw data before any processing begins
    # Fail fast here rather than crashing mid-pipeline with a confusing error
    try:
        validate_raw_data(df, ticker)
    except ValueError as e:
        print(f"[ERROR] Validation failed for {ticker}: {e}")
        return None

    # Step 2: Fix date index
    df = convert_date_column(df, ticker)

    # Step 3: Handle missing values
    df = handle_missing_values(df, ticker)

    # Step 4: Remove duplicates
    df = remove_duplicates(df, ticker)

    # Step 5: Normalize data types
    df = normalize_numeric_types(df, ticker)

    # Step 6: Feature engineering — Daily Return
    # Must run BEFORE detect_outliers() since outlier detection now uses Daily_Return
    df = add_daily_return(df)

    # Step 7: Detect outliers (adds Is_Outlier column, based on Daily_Return Z-score)
    df = detect_outliers(df, ticker)

    # Step 8: Feature engineering — Moving Averages
    df = add_moving_averages(df)

    # Step 9: Feature engineering — Volatility
    df = add_volatility(df)

    # Step 10: Feature engineering — Cumulative Return
    df = add_cumulative_return(df)

    # Step 11: Save to output/
    save_clean_data(df, ticker, safe_ticker)

    # Print a quick summary of the final DataFrame
    print(f"\n[{ticker}] Final columns: {list(df.columns)}")
    print(f"[{ticker}] Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    return df


# -------------------------------------------------------------
# FUNCTION 14: clean_all_assets() — run all 3 assets
# -------------------------------------------------------------

def clean_all_assets() -> dict:
    """
    Run the full cleaning pipeline for ALL 3 assets.

    Returns:
        dict: {
            "AAPL": <cleaned DataFrame>,
            "TSLA": <cleaned DataFrame>,
            "GC=F": <cleaned DataFrame>,
        }
    """
    ensure_output_folder()

    print("\n" + "=" * 55)
    print("  MODULE 2 — Data Cleaning & Feature Engineering")
    print(f"  Assets: {', '.join(ASSETS.keys())}")
    print("=" * 55)

    cleaned_data = {}

    for ticker, safe_ticker in ASSETS.items():
        df = clean_asset(ticker, safe_ticker)
        if df is not None:
            cleaned_data[ticker] = df

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Cleaning complete!")
    print(f"  Successfully cleaned: {len(cleaned_data)} / {len(ASSETS)} assets")

    for ticker, safe_ticker in ASSETS.items():
        if ticker in cleaned_data:
            print(f"    + {ticker:6s} -> output/{safe_ticker}_clean.csv")

    print("=" * 55 + "\n")

    return cleaned_data


# -------------------------------------------------------------
# FUNCTION 15: load_clean_csv() — for use by Module 3 & 4
# -------------------------------------------------------------

def load_clean_csv(ticker: str, safe_ticker: str) -> pd.DataFrame | None:
    """
    Load a cleaned CSV file (saved by this module) into a DataFrame.

    This function is meant to be imported and used by:
        - Module 3 (visualization.py)
        - Module 4 (ai_analysis.py)

    So they don't need to re-run cleaning every time.

    Args:
        ticker      (str): e.g. "AAPL"
        safe_ticker (str): e.g. "GC_F"

    Returns:
        pd.DataFrame or None.
    """
    filename = f"{OUTPUT_FOLDER}/{safe_ticker}_clean.csv"

    try:
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"[LOADED] {ticker} clean data <- {filename}  ({len(df)} rows)")
        return df
    except FileNotFoundError:
        print(f"[ERROR] Clean file not found: '{filename}'")
        print(f"        Please run clean_all_assets() first.")
        return None


# =============================================================
# ENTRY POINT
# Run this file standalone to clean all assets:
#   > python -m src.cleaning
# =============================================================

if __name__ == "__main__":
    # Run the full cleaning pipeline
    cleaned = clean_all_assets()

    # Preview the result for each asset
    print("\n--- PREVIEW OF CLEANED DATA ---\n")
    for ticker, df in cleaned.items():
        print(f"[ {ticker} ] — last 5 rows:")
        # Show the last 5 rows (most recent dates)
        # These will have all feature columns filled
        cols_to_show = ["Close", "Daily_Return", "SMA_7", "SMA_30", "Volatility_20d", "Is_Outlier"]
        available    = [c for c in cols_to_show if c in df.columns]
        print(df[available].tail(5).to_string())
        print()
