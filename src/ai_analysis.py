"""
Module 4 — AI Analysis
======================
This module reads cleaned financial data (from output/ folder) and uses the
Groq API (free LLM: llama-3.3-70b-versatile) to generate 4 types of analysis:

  1. Trend Summary       — overall price direction and momentum
  2. Risk Commentary     — volatility and return distribution risk
  3. Asset Comparison    — side-by-side comparison of all 3 assets
  4. Unusual Events      — explanation of outlier days flagged in Module 2

All analysis text is saved to output/ as .txt files.

ANTI-HALLUCINATION STRATEGY:
  → Real numbers from the data are injected into every prompt
  → The AI is explicitly told NOT to invent figures it hasn't been given
  → Prompts use structured templates, not open-ended questions
  → Temperature is set low (0.3) for factual, consistent outputs

Run standalone:
    python -m src.ai_analysis

Requirements:
    pip install groq python-dotenv
    GROQ_API_KEY must be set in your .env file
"""

import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import skew
from groq import Groq, RateLimitError
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

# Key = Yahoo Finance ticker, Value = safe filename prefix (same as all other modules)
ASSETS = {
    "AAPL": "AAPL",
    "TSLA": "TSLA",
    "GC=F": "GC_F",
}

# Human-readable names for each ticker (used inside the AI prompts)
ASSET_NAMES = {
    "AAPL": "Apple Inc. (AAPL)",
    "TSLA": "Tesla Inc. (TSLA)",
    "GC=F": "Gold Futures (GC=F)",
}

# Folder paths — must match the rest of the project
OUTPUT_DIR = "output"   # Where cleaned CSVs are, and where .txt reports will be saved
DATA_DIR   = "data"     # Where raw CSVs and macro data from Module 1 are saved

# Groq API settings
# Model: llama-3.3-70b-versatile (free tier, agreed in project handoff)
GROQ_MODEL = "llama-3.3-70b-versatile"

# Temperature controls creativity vs. accuracy:
#   0.0 = fully deterministic (too robotic)
#   0.3 = low creativity, high factual accuracy (best for financial analysis)
#   1.0 = very creative (too risky for financial data — causes hallucination)
GROQ_TEMPERATURE = 0.3

# Max tokens per response (roughly 1 token ≈ 0.75 words)
GROQ_MAX_TOKENS = 1000

# Retry configuration for Groq rate-limit errors
GROQ_MAX_RETRIES = 3       # Maximum number of retry attempts before giving up
GROQ_RETRY_WAIT  = 60      # Seconds to wait between retries (Groq free tier resets ~60s)


# ──────────────────────────────────────────────
# STEP 0: RETRY HELPER
# ──────────────────────────────────────────────

def retry_with_backoff(fn, retries: int = GROQ_MAX_RETRIES, wait: int = GROQ_RETRY_WAIT):
    """
    Call fn() and retry up to `retries` times if a RateLimitError is raised.

    Why exponential backoff?
        Groq's free tier enforces requests-per-minute limits. When the limit
        is hit, the API raises RateLimitError. Waiting before retrying gives
        the quota time to reset. We double the wait on each attempt so that
        transient bursts resolve without hammering the endpoint.

    Args:
        fn      (callable): A zero-argument function that calls the Groq API.
        retries (int):      Maximum number of retry attempts (default: 3).
        wait    (int):      Initial wait time in seconds before first retry (default: 60).

    Returns:
        The return value of fn() on success.

    Raises:
        RateLimitError: If all retries are exhausted without success.
        Exception:      Any non-rate-limit exception is re-raised immediately
                        (we only retry on rate-limit errors, not network errors
                        or invalid-key errors, which won't be fixed by waiting).
    """
    current_wait = wait

    for attempt in range(1, retries + 1):
        try:
            return fn()

        except RateLimitError as e:
            if attempt == retries:
                # Final attempt exhausted — propagate the error
                print(f"[ERROR] Groq rate limit hit. All {retries} retries exhausted.")
                raise

            print(f"[WARNING] Groq rate limit hit (attempt {attempt}/{retries}). "
                  f"Waiting {current_wait}s before retry...")
            time.sleep(current_wait)
            current_wait *= 2   # Exponential backoff: 60s → 120s → 240s

        # Note: non-RateLimitError exceptions propagate immediately —
        # we don't retry on bad API keys, network errors, etc.


# ──────────────────────────────────────────────
# STEP 1: LOAD ENVIRONMENT & INITIALIZE GROQ CLIENT
# ──────────────────────────────────────────────

def load_groq_client():
    """
    Load the GROQ_API_KEY from the .env file and create a Groq API client.

    The .env file must exist in the project root and contain:
        GROQ_API_KEY=your_actual_key_here

    Returns:
        Groq: An authenticated Groq API client object
    """
    # Load variables from .env into the environment
    load_dotenv()

    # Read the API key from the environment
    api_key = os.getenv("GROQ_API_KEY")

    # Stop immediately if the key is missing — don't proceed without it
    if not api_key:
        print("[ERROR] GROQ_API_KEY not found in your .env file.")
        print("        → Create a .env file in your project root with:")
        print("          GROQ_API_KEY=your_actual_key_here")
        print("        → Get a free key at: https://console.groq.com")
        raise ValueError("Missing GROQ_API_KEY in .env file.")

    print("[OK] GROQ_API_KEY loaded successfully.")

    # Create and return the Groq client (this is what we use to call the AI)
    return Groq(api_key=api_key)


# ──────────────────────────────────────────────
# STEP 2: LOAD CLEANED DATA
# ──────────────────────────────────────────────

def load_clean_csv(ticker, safe_ticker):
    """
    Load a cleaned CSV file from the output/ folder.
    This is the same helper used in visualization.py — consistent across modules.

    Args:
        ticker (str): Yahoo Finance ticker, e.g. "AAPL"
        safe_ticker (str): OS-safe filename prefix, e.g. "GC_F"

    Returns:
        pd.DataFrame or None if the file is not found
    """
    filepath = os.path.join(OUTPUT_DIR, f"{safe_ticker}_clean.csv")

    if not os.path.exists(filepath):
        print(f"[ERROR] Cleaned CSV not found: {filepath}")
        print(f"        → Run Module 2 first: python -m src.cleaning")
        return None

    # Load CSV, parse Date as actual date objects (not plain text strings)
    df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
    print(f"[OK] Loaded cleaned data for {ticker} — {len(df)} rows")
    return df


# ──────────────────────────────────────────────
# STEP 3: EXTRACT KEY STATISTICS FROM DATA
# (These numbers are injected into every AI prompt)
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
# STEP 2B: LOAD MACRO CONTEXT (Fed Funds Rate)
# ──────────────────────────────────────────────

def load_macro_context() -> dict | None:
    """
    Load the Federal Funds Rate CSV saved by Module 1 (Alpha Vantage)
    and extract three values used to enrich the AI prompts:

        latest_rate   -- most recent Fed rate value (%)
        rate_3m_ago   -- rate value approximately 3 months ago (63 trading days)
        rate_change   -- difference: latest_rate minus rate_3m_ago
        direction     -- "rising", "falling", or "flat" (based on rate_change)

    Why inject macro context into prompts?
        The Federal Funds Rate directly influences stock valuations and
        commodity prices. A rising rate environment increases borrowing
        costs (bearish for growth stocks like AAPL/TSLA) and strengthens
        the dollar (generally bearish for Gold). Giving the AI this context
        makes the trend and risk analyses more financially grounded.

    Why read from data/ not output/?
        The Fed rate CSV does not need cleaning -- it is an official
        government series with no missing values or outliers. Reading it
        directly from data/ avoids unnecessary complexity.

    Returns:
        dict: {"latest_rate": float, "rate_3m_ago": float,
               "rate_change": float, "direction": str}
        None: If the file is missing or unreadable (macro context is
              optional -- prompts degrade gracefully without it).
    """
    filepath = os.path.join(DATA_DIR, "macro_fed_rate_1y_daily.csv")

    if not os.path.exists(filepath):
        print(f"[WARNING] Macro data file not found: {filepath}")
        print(f"          Fed rate context will not be included in AI prompts.")
        print(f"          Fix: run Module 1 with a valid ALPHA_VANTAGE_API_KEY.")
        return None

    try:
        df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
        df = df.sort_index()

        # Drop rows where value is NaN (sometimes AV returns sparse data)
        df = df.dropna(subset=["value"])

        if df.empty:
            print(f"[WARNING] Macro CSV is empty: {filepath}. Skipping macro context.")
            return None

        # Latest available Fed rate
        latest_rate = float(df["value"].iloc[-1])

        # Rate ~3 months ago: 63 trading days back (approx 3 calendar months)
        # If the dataset is shorter than 63 rows, use the earliest available row
        lookback = min(63, len(df) - 1)
        rate_3m_ago = float(df["value"].iloc[-(lookback + 1)])

        # Change over the 3-month window
        rate_change = round(latest_rate - rate_3m_ago, 4)

        # Classify direction with a small tolerance to avoid "rising/falling"
        # on negligible noise (±0.01% threshold)
        if rate_change > 0.01:
            direction = "rising"
        elif rate_change < -0.01:
            direction = "falling"
        else:
            direction = "flat"

        macro = {
            "latest_rate": round(latest_rate, 4),
            "rate_3m_ago": round(rate_3m_ago, 4),
            "rate_change": rate_change,
            "direction":   direction,
        }

        print(f"[OK] Macro context loaded: Fed rate = {latest_rate}% "
              f"({direction}, {rate_change:+.4f}% over past ~3 months)")
        return macro

    except Exception as e:
        print(f"[WARNING] Could not load macro context: {e}. Skipping.")
        return None


def extract_statistics(df, ticker):
    """
    Calculate summary statistics from a cleaned DataFrame.
    These numbers are embedded directly into the AI prompts to prevent hallucination.
    The AI is only allowed to comment on numbers it has been explicitly given.

    Stats extracted:
    - Price: start, end, min, max, % change over the period
    - Returns: mean, std dev, min (worst day), max (best day)
    - Volatility: average 20-day rolling volatility
    - Moving averages: latest SMA_7 and SMA_30 values
    - Outliers: count and dates of flagged outlier days

    Args:
        df (pd.DataFrame): Cleaned dataframe for one asset
        ticker (str): Ticker symbol (used for display)

    Returns:
        dict: Dictionary of key statistics as clean strings/numbers
    """
    # Remove rows where key columns have NaN (e.g., first few rows of SMA/Volatility)
    df_valid = df.dropna(subset=["Daily_Return"])

    # --- Price statistics ---
    start_price = df["Close"].iloc[0]          # First closing price in the dataset
    end_price = df["Close"].iloc[-1]            # Most recent closing price
    min_price = df["Close"].min()
    max_price = df["Close"].max()
    # Total % change from start to end
    total_return_pct = ((end_price - start_price) / start_price) * 100

    # --- Daily return statistics ---
    mean_return = df_valid["Daily_Return"].mean()
    std_return = df_valid["Daily_Return"].std()
    worst_day = df_valid["Daily_Return"].min()
    best_day = df_valid["Daily_Return"].max()

    # Find the actual date of the worst and best days
    worst_day_date = df_valid["Daily_Return"].idxmin().strftime("%Y-%m-%d")
    best_day_date = df_valid["Daily_Return"].idxmax().strftime("%Y-%m-%d")

    # --- Volatility ---
    if "Volatility_20d" in df.columns:
        avg_volatility = df["Volatility_20d"].dropna().mean()
    else:
        avg_volatility = None

    # --- Moving averages (latest values only) ---
    latest_sma7 = df["SMA_7"].dropna().iloc[-1] if "SMA_7" in df.columns else None
    latest_sma30 = df["SMA_30"].dropna().iloc[-1] if "SMA_30" in df.columns else None

    # --- Outlier days (flagged by Module 2 using Z-score > 3.0) ---
    outlier_count = 0
    outlier_info = "none detected"
    if "Is_Outlier" in df.columns:
        outliers = df[df["Is_Outlier"] == True]
        outlier_count = len(outliers)
        if outlier_count > 0:
            # Build a short text summary of each outlier: "date (return%)"
            outlier_lines = []
            for date, row in outliers.iterrows():
                ret = row.get("Daily_Return", float("nan"))
                if pd.notna(ret):
                    outlier_lines.append(f"{date.strftime('%Y-%m-%d')} (return: {ret:.2f}%)")
            outlier_info = "; ".join(outlier_lines) if outlier_lines else "dates unavailable"

    # --- Outlier percentage ---
    # What fraction of all trading days were statistically extreme?
    total_rows = len(df)
    outlier_pct = round((outlier_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

    # --- Sharpe Ratio (annualized, unadjusted) ---
    # Formula: (mean daily return / std daily return) * sqrt(252)
    # sqrt(252) scales from daily to annual (252 trading days per year)
    # No risk-free rate subtracted — acceptable simplification for a student project.
    # A Sharpe > 1.0 is generally considered good; < 0 means the asset lost money on average.
    if std_return and std_return != 0:
        sharpe_ratio = round((mean_return / std_return) * (252 ** 0.5), 4)
    else:
        sharpe_ratio = None  # Cannot compute if std is zero (flat price — extremely rare)

    # --- Max Drawdown ---
    # Measures the largest peak-to-trough decline in cumulative return.
    # Example: if cumulative return peaked at +30% then fell to +10%,
    # the drawdown at that point is (10 - 30) / (100 + 30) = -15.3%
    # We use the Close price series to compute this correctly.
    close_prices = df["Close"].dropna()
    if len(close_prices) > 1:
        # Running maximum up to each point in time (the "peak so far")
        rolling_peak = close_prices.cummax()
        # Drawdown at each point = (current price - peak) / peak * 100
        drawdown_series = (close_prices - rolling_peak) / rolling_peak * 100
        max_drawdown = round(drawdown_series.min(), 4)  # Most negative value = worst drawdown
    else:
        max_drawdown = None

    # --- Skewness of daily returns ---
    # Skewness measures the asymmetry of the return distribution.
    #   Negative skew (< 0): left tail is heavier — more frequent large losses than gains
    #   Positive skew (> 0): right tail is heavier — more frequent large gains than losses
    #   Near zero: roughly symmetric (close to a normal distribution)
    # scipy.stats.skew() handles NaN-free input, so we drop NaN first.
    daily_returns_clean = df_valid["Daily_Return"].dropna()
    if len(daily_returns_clean) > 2:
        # Need at least 3 data points for a meaningful skewness calculation
        skewness = round(float(skew(daily_returns_clean)), 4)
    else:
        skewness = None

    # Package everything into a dictionary
    stats = {
        "ticker": ticker,
        "name": ASSET_NAMES.get(ticker, ticker),
        "start_price": round(start_price, 2),
        "end_price": round(end_price, 2),
        "min_price": round(min_price, 2),
        "max_price": round(max_price, 2),
        "total_return_pct": round(total_return_pct, 2),
        "mean_return": round(mean_return, 4),
        "std_return": round(std_return, 4),
        "worst_day": round(worst_day, 2),
        "worst_day_date": worst_day_date,
        "best_day": round(best_day, 2),
        "best_day_date": best_day_date,
        "avg_volatility": round(avg_volatility, 4) if avg_volatility is not None else "N/A",
        "latest_sma7": round(latest_sma7, 2) if latest_sma7 is not None else "N/A",
        "latest_sma30": round(latest_sma30, 2) if latest_sma30 is not None else "N/A",
        "outlier_count": outlier_count,
        "outlier_info": outlier_info,
        "outlier_pct": outlier_pct,
        "sharpe_ratio": sharpe_ratio if sharpe_ratio is not None else "N/A",
        "max_drawdown": max_drawdown if max_drawdown is not None else "N/A",
        "skewness": skewness if skewness is not None else "N/A",
    }

    return stats


# ──────────────────────────────────────────────
# STEP 4: CALL GROQ API
# (Single reusable function for all prompt types)
# ──────────────────────────────────────────────

def call_groq(client, system_prompt, user_prompt):
    """
    Send a prompt to the Groq API and return the AI's response as a string.

    Uses a "system prompt" (sets AI role/rules) + "user prompt" (the actual question).
    This two-part structure is standard practice for LLM API calls.

    Anti-hallucination settings applied here:
    - Low temperature (0.3): keeps the AI factual and consistent
    - Explicit system instructions: AI is told to only use provided numbers
    - max_tokens capped: prevents overly long, wandering responses

    Retry behaviour:
    - RateLimitError is caught by retry_with_backoff() (up to 3 attempts, 60s wait).
    - All other exceptions are caught here and returned as an error string so
      the pipeline can continue with the remaining assets.

    Args:
        client (Groq): The authenticated Groq client from load_groq_client()
        system_prompt (str): Instructions that define the AI's role and constraints
        user_prompt (str): The actual analysis request with embedded data

    Returns:
        str: The AI's response text, or an error message if the call fails
    """
    def _api_call():
        """Inner zero-argument function passed to retry_with_backoff."""
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS,
            messages=[
                # System message: sets role and strict rules
                {"role": "system", "content": system_prompt},
                # User message: the actual data and question
                {"role": "user", "content": user_prompt},
            ]
        )

        # Extract the text content from the response
        content = response.choices[0].message.content.strip()

        # ── LLM response validation ────────────────────────────────────────
        # Guard against empty or suspiciously short responses (truncation,
        # refusal, or API glitch).
        if not content or len(content) < 50:
            print(f"[WARNING] LLM response is suspiciously short ({len(content)} chars). "
                  f"The response may be truncated or refused.")

        # Check if the response was cut off due to max_tokens being hit
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            print(f"[WARNING] LLM response was truncated — max_tokens ({GROQ_MAX_TOKENS}) "
                  f"was reached. Consider increasing GROQ_MAX_TOKENS for fuller reports.")

        return content

    try:
        # retry_with_backoff handles RateLimitError transparently.
        # Any non-rate-limit exception propagates to the outer except below.
        return retry_with_backoff(_api_call)

    except Exception as e:
        # Catch any remaining errors (network issue, invalid key, retries exhausted, etc.)
        print(f"[ERROR] Groq API call failed: {e}")
        return f"[ERROR] Analysis could not be generated. Reason: {e}"


# ──────────────────────────────────────────────
# STEP 5: SAVE ANALYSIS TO TEXT FILE
# ──────────────────────────────────────────────

def save_analysis(text, filename):
    """
    Save AI-generated analysis text to the output/ folder as a .txt file.

    Args:
        text (str): The full analysis text to save
        filename (str): Just the filename, e.g. "AAPL_analysis.txt"
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[SAVED] Analysis saved → {filepath}")


# ──────────────────────────────────────────────
# ANALYSIS 1: TREND SUMMARY (per asset)
# ──────────────────────────────────────────────

def analyze_trend(client, stats):
    """
    Analysis 1 — Trend Summary.

    Asks the AI to describe the overall price direction and momentum
    based on real price, SMA, and return data injected into the prompt.

    Prompt strategy:
    - Numbers are embedded directly (no vague descriptions)
    - AI is told to reference specific figures in its response
    - Structured output requested (clear sections)

    Args:
        client (Groq): Groq API client
        stats (dict): Statistics dictionary from extract_statistics()

    Returns:
        str: AI-generated trend summary
    """
    # ── SYSTEM PROMPT: defines the AI's role and strict rules ──
    system_prompt = """You are a professional financial analyst writing a report for a university project.
Your job is to analyze stock/commodity price data and explain the trend clearly.

STRICT RULES to prevent inaccurate analysis:
1. ONLY use the numbers provided in the user message. Do NOT invent or estimate any figures.
2. Do NOT reference any external events, news, or company information not provided.
3. Be factual, concise, and educational — suitable for a finance student.
4. Structure your response with these sections: Overview, Price Movement, Momentum Signal, Conclusion.
5. Keep total response under 300 words."""

    # ── USER PROMPT: embeds real statistics from the data ──
    user_prompt = f"""Analyze the price trend for {stats['name']} over the past year using the data below.

=== DATA PROVIDED ===
Asset: {stats['name']}
Period start price: ${stats['start_price']}
Period end price:   ${stats['end_price']}
Price range:        ${stats['min_price']} (low) to ${stats['max_price']} (high)
Total return:       {stats['total_return_pct']}% over the full period
Average daily return: {stats['mean_return']}%
Latest SMA-7 (short-term moving average):  ${stats['latest_sma7']}
Latest SMA-30 (medium-term moving average): ${stats['latest_sma30']}
=====================

Based ONLY on these numbers, write a Trend Summary with the following sections:
1. Overview — Is the overall trend bullish (upward), bearish (downward), or sideways?
2. Price Movement — Describe the price range and total return.
3. Momentum Signal — Compare SMA-7 vs SMA-30. If SMA-7 > SMA-30, momentum is currently bullish. If SMA-7 < SMA-30, it is bearish.
4. Conclusion — One sentence investment-neutral summary of the trend."""

    print(f"[INFO] Requesting Trend Summary for {stats['ticker']}...")
    return call_groq(client, system_prompt, user_prompt)


# ──────────────────────────────────────────────
# ANALYSIS 2: RISK COMMENTARY (per asset)
# ──────────────────────────────────────────────

def analyze_risk(client, stats):
    """
    Analysis 2 — Risk Commentary.

    Asks the AI to assess the risk profile of the asset based on
    volatility, return distribution, and worst/best day figures.

    Prompt strategy:
    - Volatility and return std dev are given explicitly
    - AI is prompted to interpret what these numbers mean in practice
    - Risk level is asked to be classified: Low / Moderate / High

    Args:
        client (Groq): Groq API client
        stats (dict): Statistics dictionary from extract_statistics()

    Returns:
        str: AI-generated risk commentary
    """
    system_prompt = """You are a professional risk analyst writing a report for a university finance assignment.
Your job is to assess the risk level of a financial asset based on provided data.

STRICT RULES:
1. ONLY use the numbers provided. Do NOT invent any statistics or reference external events.
2. Classify overall risk as: Low / Moderate / High — and justify it with the given numbers.
3. Be clear and educational — explain what each metric means for a finance student.
4. Structure: Risk Level, Volatility Analysis, Return Distribution, Worst/Best Day, Sharpe & Drawdown, Overall Assessment.
5. Keep total response under 350 words."""

    user_prompt = f"""Assess the risk profile of {stats['name']} using the data below.

=== DATA PROVIDED ===
Asset: {stats['name']}
Average 20-day rolling volatility:  {stats['avg_volatility']}% (std dev of daily returns)
Average daily return:                {stats['mean_return']}%
Std deviation of daily returns:      {stats['std_return']}%
Worst single-day return:             {stats['worst_day']}% on {stats['worst_day_date']}
Best single-day return:              {stats['best_day']}% on {stats['best_day_date']}
Number of statistical outlier days:  {stats['outlier_count']} (Z-score > 3.0 on daily return)
Outlier days as % of total period:   {stats['outlier_pct']}%
Sharpe Ratio (annualized):           {stats['sharpe_ratio']}
  (formula: mean daily return / std daily return × √252; no risk-free rate adjustment)
Max Drawdown:                        {stats['max_drawdown']}%
  (largest peak-to-trough decline in price over the period)
Skewness of daily returns:           {stats['skewness']}
  (negative = left-tail heavy = more frequent large losses; positive = right-tail heavy = more frequent large gains)
=====================

Based ONLY on these numbers, write a Risk Commentary with these sections:
1. Risk Level — Classify as Low / Moderate / High with one-sentence justification.
2. Volatility Analysis — What does the volatility figure tell us about daily price swings?
3. Return Distribution — What do the mean, std deviation, and skewness tell us about typical vs. extreme days? Is the distribution symmetric or skewed toward losses/gains?
4. Worst & Best Days — Comment on the magnitude of the worst and best single-day moves.
5. Sharpe Ratio & Max Drawdown — Interpret the Sharpe Ratio (is return adequate for the risk taken?) and the Max Drawdown (how severe was the worst losing streak?).
6. Overall Assessment — Is this asset suitable for risk-averse or risk-tolerant investors?"""

    print(f"[INFO] Requesting Risk Commentary for {stats['ticker']}...")
    return call_groq(client, system_prompt, user_prompt)


# ──────────────────────────────────────────────
# ANALYSIS 3: UNUSUAL EVENTS (per asset)
# ──────────────────────────────────────────────

def analyze_unusual_events(client, stats):
    """
    Analysis 3 — Unusual Events Detection.

    Asks the AI to explain what the outlier days (flagged by Module 2
    using Z-score > 3.0) might represent, and why extreme moves matter.

    Prompt strategy:
    - Exact outlier dates and return % are embedded in the prompt
    - AI is told these are statistically rare events (Z > 3.0 = top 0.3%)
    - AI must not invent causes — it should frame possibilities generically

    Args:
        client (Groq): Groq API client
        stats (dict): Statistics dictionary from extract_statistics()

    Returns:
        str: AI-generated unusual events analysis
    """
    system_prompt = """You are a financial analyst explaining unusual market events for a university report.
Your job is to explain what statistically extreme price movements mean for investors.

STRICT RULES:
1. ONLY use the dates and return percentages provided. Do NOT invent causes or reference specific news.
2. Explain WHY extreme moves matter in general (earnings, macro events, market shocks) without claiming any specific cause.
3. Be educational and factual. Suitable for a finance student with no prior experience.
4. Structure: What Are Outliers, Detected Events, Why They Matter, Investor Implication.
5. Keep total response under 300 words."""

    # Handle the case where no outliers were detected
    if stats['outlier_count'] == 0:
        outlier_section = "No statistical outliers (Z-score > 3.0) were detected for this asset during the period."
    else:
        outlier_section = (
            f"{stats['outlier_count']} outlier day(s) detected (Z-score > 3.0 on daily return):\n"
            f"  {stats['outlier_info']}"
        )

    user_prompt = f"""Analyze the unusual price events detected for {stats['name']} using the data below.

=== DATA PROVIDED ===
Asset: {stats['name']}
Outlier detection method: Z-score on daily return, threshold = 3.0
(A Z-score above 3.0 means the return was more than 3 standard deviations from the mean — statistically rare, occurring in less than 0.3% of normal trading days)

{outlier_section}

Overall worst single day: {stats['worst_day']}% on {stats['worst_day_date']}
Overall best single day:  {stats['best_day']}% on {stats['best_day_date']}
=====================

Based ONLY on this information, write an Unusual Events Analysis with these sections:
1. What Are Outliers — Brief explanation of Z-score method in simple terms.
2. Detected Events — List and briefly describe each outlier (or confirm none exist).
3. Why They Matter — What do extreme single-day moves signal to investors in general?
4. Investor Implication — How should investors think about these events in their decision-making?"""

    print(f"[INFO] Requesting Unusual Events Analysis for {stats['ticker']}...")
    return call_groq(client, system_prompt, user_prompt)


# ──────────────────────────────────────────────
# ANALYSIS 4: MACRO INDICATOR REPORT
# ──────────────────────────────────────────────

def analyze_macro(client, macro):
    """
    Analysis 4 — Macro Indicator Report (standalone).

    Generates a dedicated report analyzing the Federal Funds Rate
    and its implications for each of the three tracked assets.

    Why a dedicated file?
        Separating macro analysis from asset reports means:
        - Asset reports stay focused on their own price/return data
        - Adding Tier 2 indicators (CPI, GDP) later only requires
          updating this function and its prompt — no asset functions touched
        - The macro report is readable on its own without any asset context

    Prompt strategy:
        - All three assets are listed so the AI can discuss directional
          impact for each (growth stocks vs. commodity differently)
        - AI is told to discuss rate direction, not make predictions
        - Same anti-hallucination rules: only use provided numbers

    Args:
        client (Groq): Groq API client
        macro (dict): Output of load_macro_context() — must not be None

    Returns:
        str: AI-generated macro analysis text
    """
    system_prompt = """You are a macroeconomic analyst writing a report for a university finance project.
Your job is to explain what Federal Reserve interest rate data means for financial markets.

STRICT RULES:
1. ONLY use the numbers provided. Do NOT invent figures or reference specific news events.
2. Discuss directional implications — do NOT make price predictions or investment recommendations.
3. Be educational and clear — explain concepts suitable for a finance student.
4. Structure: Rate Overview, Recent Trend, Impact on Equities, Impact on Gold, Portfolio Implication.
5. Keep total response under 400 words."""

    user_prompt = f"""Analyze the Federal Funds Rate data below and explain its implications for the three tracked assets.

=== MACRO DATA PROVIDED ===
Indicator:                   Federal Funds Rate (US)
Current rate:                {macro['latest_rate']}%
Rate approximately 3 months ago: {macro['rate_3m_ago']}%
Change over past ~3 months:  {macro['rate_change']:+.4f}%
Rate direction:              {macro['direction']}

Assets affected:
  - Apple Inc. (AAPL)   — US large-cap growth stock
  - Tesla Inc. (TSLA)   — US large-cap high-growth stock
  - Gold Futures (GC=F) — Safe-haven commodity, USD-denominated
===========================

Based ONLY on this data, write a Macro Indicator Report with these sections:
1. Rate Overview — What is the current Fed Funds Rate and what does it represent?
2. Recent Trend — Describe the direction and magnitude of change over the past 3 months.
3. Impact on Equities — How does this rate environment generally affect growth stocks like AAPL and TSLA?
4. Impact on Gold — How does this rate environment generally affect Gold as a USD-denominated commodity?
5. Portfolio Implication — From a diversification perspective, how does the macro environment affect the risk-return balance across these three assets?"""

    print("[INFO] Requesting Macro Indicator report (Fed Funds Rate)...")
    return call_groq(client, system_prompt, user_prompt)


# ──────────────────────────────────────────────
# ANALYSIS 4: ASSET COMPARISON (all 3 assets)
# ──────────────────────────────────────────────

def analyze_comparison(client, all_stats):
    """
    Analysis 4 — Asset Comparison (shared report across all 3 assets).

    Asks the AI to compare AAPL, TSLA, and Gold side-by-side across
    return, risk, and trend metrics.

    Prompt strategy:
    - A structured comparison table of numbers is embedded in the prompt
    - AI is asked to rank assets across dimensions and give investment context
    - This is the only analysis that spans all 3 assets at once

    Args:
        client (Groq): Groq API client
        all_stats (dict): {ticker: stats_dict} for all 3 assets

    Returns:
        str: AI-generated comparison report
    """
    system_prompt = """You are a portfolio analyst writing a comparative asset report for a university finance project.
Your job is to compare multiple financial assets across return, risk, and trend metrics.

STRICT RULES:
1. ONLY use the numbers provided in the comparison table. Do NOT invent any additional data.
2. Rank assets clearly (1st, 2nd, 3rd) for each dimension you compare.
3. Explain tradeoffs — higher return often comes with higher risk.
4. Connect to basic portfolio theory: diversification, risk-return tradeoff.
5. Structure: Introduction, Return Comparison, Risk Comparison, Trend Comparison, Portfolio Recommendation.
6. Keep total response under 400 words."""

    # Build a comparison table from all available stats
    # Only include assets that were successfully loaded
    table_rows = []
    for ticker, stats in all_stats.items():
        if stats is None:
            continue
        table_rows.append(
            f"  {stats['name']:35s} | "
            f"Total Return: {stats['total_return_pct']:+7.2f}% | "
            f"Avg Daily Return: {stats['mean_return']:+6.4f}% | "
            f"Volatility: {stats['avg_volatility']}% | "
            f"Worst Day: {stats['worst_day']:+6.2f}% | "
            f"Best Day: {stats['best_day']:+6.2f}% | "
            f"SMA7: ${stats['latest_sma7']} | SMA30: ${stats['latest_sma30']}"
        )

    comparison_table = "\n".join(table_rows) if table_rows else "No data available."

    user_prompt = f"""Compare the following three assets using only the data provided below.

=== COMPARISON TABLE ===
{comparison_table}
========================

Based ONLY on this data, write an Asset Comparison Report with these sections:
1. Introduction — Briefly describe the three assets being compared (stock vs. commodity).
2. Return Comparison — Which asset had the best and worst total return? Rank them 1st to 3rd.
3. Risk Comparison — Which is most/least volatile? Rank them by risk level. Explain the tradeoff.
4. Trend Comparison — Based on SMA-7 vs SMA-30 relationship, what is each asset's current momentum?
5. Portfolio Recommendation — From a diversification perspective, which combination of these assets would reduce overall portfolio risk, and why? Reference the risk-return tradeoff."""

    print("[INFO] Requesting Asset Comparison report (all assets)...")
    return call_groq(client, system_prompt, user_prompt)


# ──────────────────────────────────────────────
# ASSEMBLE FULL REPORT (combines all 4 analyses)
# ──────────────────────────────────────────────

def build_full_report(ticker, stats, trend_text, risk_text, events_text, macro=None):
    """
    Combine the 3 per-asset analyses (trend, risk, unusual events) into
    one well-formatted report string, ready to be saved as a .txt file.

    Also includes:
    - Report metadata (timestamp, model, period) in the header
    - Section 4: Macro Context — a brief summary of the Fed rate environment
      drawn directly from the macro dict (Option B: no file I/O needed)
    - Enhanced disclaimer noting macro scope (Fed rate only; CPI/GDP excluded)

    Args:
        ticker (str): Ticker symbol
        stats (dict): Statistics dictionary (used for the header)
        trend_text (str): Output of analyze_trend()
        risk_text (str): Output of analyze_risk()
        events_text (str): Output of analyze_unusual_events()
        macro (dict or None): Output of load_macro_context(). If None,
                              Section 4 is replaced with a short notice.

    Returns:
        str: Full formatted report as a single string
    """
    separator = "=" * 60

    # --- Report metadata: captured at runtime so each report is timestamped ---
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Derive the data period from the stats (start/end price dates are not stored,
    # but we do have the full DataFrame period length — use a readable label instead)
    data_period = "Last 12 months (1-year daily data)"

    # --- Section 4: Macro Context block ---
    # Built directly from the macro dict — no file reading, no text parsing.
    # If macro is None (AV key missing or file not found), show a short notice.
    if macro is not None:
        # Determine a human-readable rate direction label
        direction_label = {
            "rising":  "Rising ↑",
            "falling": "Falling ↓",
            "flat":    "Flat →",
        }.get(macro["direction"], macro["direction"].capitalize())

        macro_section = f"""
{separator}
SECTION 4 — MACRO CONTEXT (Federal Funds Rate)
{separator}

Current Fed Funds Rate:        {macro['latest_rate']}%
Rate ~3 Months Ago:            {macro['rate_3m_ago']}%
Change Over Past ~3 Months:    {macro['rate_change']:+.4f}%
Rate Direction:                {direction_label}

Rate Sensitivity for {ticker}:"""

        # Add asset-specific rate sensitivity commentary
        # These are fixed financial relationships (see handoff rate sensitivity table)
        sensitivity_notes = {
            "AAPL": (
                "HIGH negative sensitivity. Rising rates compress P/E multiples by increasing\n"
                "  the discount rate applied to future earnings — bearish for large-cap growth stocks."
            ),
            "TSLA": (
                "VERY HIGH negative sensitivity. Tesla's valuation relies heavily on future growth\n"
                "  expectations, which are most vulnerable to discount rate increases."
            ),
            "GC=F": (
                "POSITIVE sensitivity. Gold tends to rise when real yields fall (rates drop or\n"
                "  inflation rises faster than rates), as it competes with yield-bearing assets."
            ),
        }
        note = sensitivity_notes.get(ticker, "See macro_analysis.txt for full context.")
        macro_section += f"\n  {note}\n\nFor full macro analysis, see: output/macro_analysis.txt"

    else:
        macro_section = f"""
{separator}
SECTION 4 — MACRO CONTEXT
{separator}

  Macro data unavailable — Alpha Vantage API key not configured or
  Module 1 was run without a valid ALPHA_VANTAGE_API_KEY.

  To enable: add ALPHA_VANTAGE_API_KEY to your .env file and re-run
  Module 1 (python -m src.data_collection)."""

    report = f"""
{separator}
AI FINANCIAL ANALYSIS REPORT
Asset:        {stats['name']}
Period:       {data_period}
Model:        {GROQ_MODEL} via Groq API
Generated at: {generated_at}
{separator}

KEY STATISTICS USED IN THIS REPORT:
  Start Price:              ${stats['start_price']}
  End Price:                ${stats['end_price']}
  Price Range:              ${stats['min_price']} — ${stats['max_price']}
  Total Return:             {stats['total_return_pct']}%
  Avg Daily Return:         {stats['mean_return']}%
  Daily Return StdDev:      {stats['std_return']}%
  Avg Volatility (20d):     {stats['avg_volatility']}%
  Latest SMA-7:             ${stats['latest_sma7']}
  Latest SMA-30:            ${stats['latest_sma30']}
  Sharpe Ratio (ann.):      {stats['sharpe_ratio']}
  Max Drawdown:             {stats['max_drawdown']}%
  Skewness:                 {stats['skewness']}
  Outlier Days:             {stats['outlier_count']} ({stats['outlier_pct']}% of period)

{separator}
SECTION 1 — TREND SUMMARY
{separator}

{trend_text}

{separator}
SECTION 2 — RISK COMMENTARY
{separator}

{risk_text}

{separator}
SECTION 3 — UNUSUAL EVENTS DETECTION
{separator}

{events_text}
{macro_section}

{separator}
DISCLAIMER
{separator}
This report was auto-generated by an AI model ({GROQ_MODEL} via Groq API)
for educational purposes only. It is NOT financial advice. All analysis
is grounded in the numerical data listed above. Do not make investment
decisions based on this report alone.

Macro scope: This report incorporates Federal Funds Rate data only.
Broader macroeconomic indicators (CPI, GDP, unemployment) are not
included in this analysis and may affect asset performance in ways
not captured here.
{separator}
"""
    return report


# ──────────────────────────────────────────────
# MAIN FUNCTION
# ──────────────────────────────────────────────

def analyze_all_assets():
    """
    Main entry point for Module 4.
    Loads all cleaned data, extracts statistics, calls Groq API for each
    analysis type, and saves all results to output/ as .txt files.
    """
    print("\n" + "=" * 55)
    print("  MODULE 4 — AI ANALYSIS (Groq / llama-3.3-70b)")
    print("=" * 55)

    # ── Step 1: Load Groq API client ──
    print("\n[INFO] Connecting to Groq API...")
    try:
        client = load_groq_client()
    except ValueError:
        # Error message was already printed inside load_groq_client()
        return

    # ── Step 2: Load cleaned data for all assets ──
    print("\n[INFO] Loading cleaned CSV files from output/ folder...")
    all_data = {}
    all_stats = {}

    for ticker, safe_ticker in ASSETS.items():
        df = load_clean_csv(ticker, safe_ticker)
        all_data[ticker] = df

        if df is not None:
            # Extract key statistics — these go into all prompts
            stats = extract_statistics(df, ticker)
            all_stats[ticker] = stats
            print(f"[INFO] Stats extracted for {ticker}: "
                  f"total return={stats['total_return_pct']}%, "
                  f"volatility={stats['avg_volatility']}%, "
                  f"outliers={stats['outlier_count']}")
        else:
            all_stats[ticker] = None

    print()

    # ── Step 2B: Load macro context (Fed rate from Alpha Vantage) ──
    # This is optional — if the file is missing, macro=None and prompts
    # run without the macro section. No step is skipped or aborted.
    print("[INFO] Loading macro context (Fed Funds Rate)...")
    macro = load_macro_context()

    # ── Step 3: Generate per-asset analyses (Analyses 1, 2, 3) ──
    for ticker, safe_ticker in ASSETS.items():
        stats = all_stats.get(ticker)

        if stats is None:
            print(f"[SKIP] {ticker} — no data available, skipping AI analysis.")
            continue

        print(f"\n--- Generating AI analysis for {ticker} ---")

        # Analysis 1: Trend Summary
        trend_text = analyze_trend(client, stats)

        # Analysis 2: Risk Commentary
        risk_text = analyze_risk(client, stats)

        # Analysis 3: Unusual Events
        events_text = analyze_unusual_events(client, stats)

        # Combine all 3 per-asset analyses into one formatted report
        full_report = build_full_report(ticker, stats, trend_text, risk_text, events_text, macro=macro)

        # Save to output/ folder
        save_analysis(full_report, f"{safe_ticker}_analysis.txt")

    # ── Step 4: Generate standalone macro indicator report ──
    # Only runs if macro data was successfully loaded.
    # Extensible: when Tier 2 indicators are added, update analyze_macro() only.
    if macro:
        print("\n--- Generating Macro Indicator report ---")
        macro_text = analyze_macro(client, macro)
        separator = "=" * 60
        macro_report = f"""
{separator}
AI MACRO INDICATOR REPORT
Indicators:   Federal Funds Rate (US)
Assets:       AAPL, TSLA, GC=F
Model:        {GROQ_MODEL} via Groq API
Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{separator}

{macro_text}

{separator}
DISCLAIMER
This report was auto-generated for educational purposes only.
It is NOT financial advice.
{separator}
"""
        save_analysis(macro_report, "macro_analysis.txt")
    else:
        print("[SKIP] Macro report skipped — no macro data available.")

    # ── Step 5: Generate the cross-asset comparison ──
    print("\n--- Generating Asset Comparison report ---")

    # Only pass assets that were successfully loaded
    valid_stats = {t: s for t, s in all_stats.items() if s is not None}

    if len(valid_stats) >= 2:
        comparison_text = analyze_comparison(client, valid_stats)

        # Build a nicely formatted comparison report
        separator = "=" * 60
        comparison_report = f"""
{separator}
AI ASSET COMPARISON REPORT
Assets:       {', '.join(valid_stats.keys())}
Model:        {GROQ_MODEL} via Groq API
Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{separator}

{comparison_text}

{separator}
DISCLAIMER
This report was auto-generated for educational purposes only.
It is NOT financial advice.
{separator}
"""
        save_analysis(comparison_report, "comparison_analysis.txt")
    else:
        print("[WARNING] Not enough assets loaded to generate a comparison. Skipping.")

    # ── Step 6: Final summary ──
    print("\n" + "=" * 55)
    print("  [OK] All AI analyses complete. Reports saved to output/")
    print("  Files generated:")
    for ticker, safe_ticker in ASSETS.items():
        if all_stats.get(ticker):
            print(f"    → output/{safe_ticker}_analysis.txt")
    if len(valid_stats) >= 2:
        print("    → output/comparison_analysis.txt")
    if macro:
        print("    → output/macro_analysis.txt")
    print("=" * 55 + "\n")


# ──────────────────────────────────────────────
# STANDALONE ENTRY POINT
# Runs when: python -m src.ai_analysis
# ──────────────────────────────────────────────

if __name__ == "__main__":
    analyze_all_assets()
