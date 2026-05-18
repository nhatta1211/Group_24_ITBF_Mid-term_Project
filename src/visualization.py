"""
Module 3 — Visualization
========================
This module reads cleaned financial data (from output/ folder) and creates
6 types of professional charts saved as PNG files in the charts/ folder.

Charts created per asset (AAPL, TSLA, GC=F):
  1. Price Trend Line Chart
  2. Volume Bar Chart
  3. Histogram of Daily Returns
  4. Moving Average Chart (SMA_7 and SMA_30)
  5. Volatility Chart (20-day rolling)
  6. Fed Rate Correlation Scatter + Regression Line

Shared charts (all assets combined):
  7. Correlation Heatmap of Daily Returns
  8. Fed Rate Overlay on Asset Price Trends
  9. Multi-Asset Volatility Comparison

Run standalone:
    python -m src.visualization
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ──────────────────────────────────────────────
# CONFIGURATION — edit here if you add new assets
# ──────────────────────────────────────────────

# Key = Yahoo Finance ticker, Value = safe filename prefix (same as Module 1 & 2)
ASSETS = {
    "AAPL": "AAPL",
    "TSLA": "TSLA",
    "GC=F": "GC_F",
}

# Folder paths
OUTPUT_DIR = "output"   # Where cleaned CSVs are stored (from Module 2)
CHARTS_DIR = "charts"   # Where we save PNG chart files
DATA_DIR   = "data"     # Where raw CSVs and macro data from Module 1 are saved

# Chart style — a clean, professional look
CHART_STYLE = "seaborn-v0_8-whitegrid"
FIGURE_DPI = 150          # Image resolution (dots per inch); 150 is good quality
TITLE_FONTSIZE = 13
LABEL_FONTSIZE = 11

# Colors for each asset — used consistently across charts
ASSET_COLORS = {
    "AAPL": "#1f77b4",   # Blue
    "TSLA": "#d62728",   # Red
    "GC=F": "#f5a623",   # Gold/orange (fitting for Gold!)
}


# ──────────────────────────────────────────────
# GLOBAL CHART STYLING (applies to ALL charts)
# ──────────────────────────────────────────────

# These settings are applied once at module load time.
# Every chart function below inherits them automatically —
# no need to repeat styling code inside each function.

plt.rcParams["figure.figsize"]      = (12, 6)   # Default canvas size for all charts
plt.rcParams["figure.dpi"]          = 100        # Screen-quality resolution (save_figure overrides to FIGURE_DPI)
plt.rcParams["font.size"]           = 10         # Base font size for tick labels and annotations
plt.rcParams["axes.spines.top"]     = False      # Remove top border (cleaner look)
plt.rcParams["axes.spines.right"]   = False      # Remove right border (cleaner look)


# ──────────────────────────────────────────────
# HELPER: LOAD CLEANED DATA
# ──────────────────────────────────────────────

def load_clean_csv(ticker, safe_ticker):
    """
    Load a cleaned CSV file for one asset from the output/ folder.
    This mirrors the load_clean_csv() function in cleaning.py.

    Args:
        ticker (str): Yahoo Finance ticker, e.g. "AAPL"
        safe_ticker (str): OS-safe filename prefix, e.g. "GC_F"

    Returns:
        pd.DataFrame or None if the file is not found
    """
    # Build the full path, e.g. "output/AAPL_clean.csv"
    filepath = os.path.join(OUTPUT_DIR, f"{safe_ticker}_clean.csv")

    if not os.path.exists(filepath):
        print(f"[ERROR] Cleaned CSV not found: {filepath}")
        print(f"        → Run Module 2 first: python -m src.cleaning")
        return None

    # Load CSV and parse the Date column as actual dates (not plain text)
    df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
    print(f"[OK] Loaded cleaned data for {ticker} — {len(df)} rows")
    return df


# ──────────────────────────────────────────────
# HELPER: SAVE A FIGURE
# ──────────────────────────────────────────────

def save_figure(fig, filename):
    """
    Save a matplotlib figure to the charts/ folder as a PNG file.

    Args:
        fig: The matplotlib Figure object to save
        filename (str): Just the filename, e.g. "AAPL_price_trend.png"
    """
    # Make sure the charts/ folder exists; create it if not
    os.makedirs(CHARTS_DIR, exist_ok=True)

    filepath = os.path.join(CHARTS_DIR, filename)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"[SAVED] Chart saved → {filepath}")
    plt.close(fig)  # Free memory after saving


def add_footnote(fig, source="Yahoo Finance"):
    """
    Add a small source footnote to the bottom-left of a figure.

    This is applied to every chart for academic credibility and to clearly
    identify where the data came from.

    Args:
        fig: The matplotlib Figure object to annotate
        source (str): Data source label, e.g. "Yahoo Finance" or
                      "Yahoo Finance / Alpha Vantage"
    """
    fig.text(
        0.01, 0.01,                      # Position: bottom-left corner
        f"Source: {source}",
        fontsize=8,
        color="grey",
        ha="left", va="bottom",
        transform=fig.transFigure        # Coordinates relative to the figure (not the axes)
    )


# ──────────────────────────────────────────────
# CHART 1: PRICE TREND LINE CHART
# ──────────────────────────────────────────────

def plot_price_trend(df, ticker, safe_ticker):
    """
    Chart 1 — Price Trend Line Chart.

    Shows how the closing price of the asset changed over 1 year.
    Outlier days (flagged by Module 2) are highlighted with red dots
    so you can spot unusual price movements.

    Why useful in finance:
    - The most fundamental chart in finance
    - Helps identify uptrends, downtrends, and turning points
    - Outlier markers highlight days of major news or market shocks

    Args:
        df (pd.DataFrame): Cleaned dataframe for one asset
        ticker (str): Ticker symbol, e.g. "AAPL"
        safe_ticker (str): Safe filename prefix, e.g. "GC_F"
    """
    print(f"[INFO] Drawing Price Trend chart for {ticker}...")

    color = ASSET_COLORS.get(ticker, "#333333")

    fig, ax = plt.subplots(figsize=(12, 5))
    plt.style.use(CHART_STYLE)

    # --- Draw the main price line ---
    ax.plot(df.index, df["Close"], color=color, linewidth=1.8,
            label="Close Price", zorder=2)

    # --- Highlight outlier days with red dots ---
    if "Is_Outlier" in df.columns:
        outliers = df[df["Is_Outlier"] == True]
        if not outliers.empty:
            ax.scatter(outliers.index, outliers["Close"],
                       color="red", s=50, zorder=3,
                       label=f"Outlier days ({len(outliers)})", alpha=0.8)

    # --- Labels and formatting ---
    ax.set_title(f"{ticker} — Closing Price Trend (1 Year)", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Price (USD)", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    # Format x-axis dates nicely (e.g., "Jan 2024")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()

    add_footnote(fig)
    save_figure(fig, f"{safe_ticker}_1_price_trend.png")


# ──────────────────────────────────────────────
# CHART 2: VOLUME BAR CHART
# ──────────────────────────────────────────────

def plot_volume(df, ticker, safe_ticker):
    """
    Chart 2 — Volume Bar Chart.

    Shows how many shares (or contracts) were traded each day over 1 year.
    Volume spikes usually coincide with major news events or earnings releases.

    Why useful in finance:
    - High volume + price move = strong signal (confirmed move)
    - Low volume + price move = weak signal (may reverse)
    - Volume is sometimes called "the fuel of the market"

    Args:
        df (pd.DataFrame): Cleaned dataframe for one asset
        ticker (str): Ticker symbol
        safe_ticker (str): Safe filename prefix
    """
    print(f"[INFO] Drawing Volume chart for {ticker}...")

    color = ASSET_COLORS.get(ticker, "#555555")

    fig, ax = plt.subplots(figsize=(12, 4))
    plt.style.use(CHART_STYLE)

    # --- Draw volume as vertical bars ---
    ax.bar(df.index, df["Volume"], color=color, alpha=0.6, width=1.5, label="Volume")

    # --- Draw a 30-day moving average of volume to show trend ---
    volume_ma = df["Volume"].rolling(30).mean()
    ax.plot(df.index, volume_ma, color="black", linewidth=1.5,
            linestyle="--", label="30-day Avg Volume")

    # --- Labels and formatting ---
    ax.set_title(f"{ticker} — Daily Trading Volume (1 Year)", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Volume (shares)", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    # Format large volume numbers with commas (e.g., 50,000,000)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()

    add_footnote(fig)
    save_figure(fig, f"{safe_ticker}_2_volume.png")


# ──────────────────────────────────────────────
# CHART 3: HISTOGRAM OF DAILY RETURNS
# ──────────────────────────────────────────────

def plot_returns_histogram(df, ticker, safe_ticker):
    """
    Chart 3 — Histogram of Daily Returns.

    Shows the distribution (spread) of daily percentage price changes.
    A normal (bell-shaped) distribution is the baseline assumption in finance.
    Fat tails (wider spread) mean the asset has more extreme moves than expected.

    Why useful in finance:
    - Reveals the risk profile of an asset
    - A wide histogram = high volatility = higher risk/reward
    - Skewness (lean left or right) shows if losses or gains dominate
    - Used in Value-at-Risk (VaR) and portfolio risk models

    Args:
        df (pd.DataFrame): Cleaned dataframe for one asset
        ticker (str): Ticker symbol
        safe_ticker (str): Safe filename prefix
    """
    print(f"[INFO] Drawing Daily Returns histogram for {ticker}...")

    color = ASSET_COLORS.get(ticker, "#444444")

    # Drop NaN values in Daily_Return (first row is always NaN)
    returns = df["Daily_Return"].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.style.use(CHART_STYLE)

    # --- Draw histogram bars ---
    ax.hist(returns, bins=50, color=color, alpha=0.7, edgecolor="white", label="Daily Return")

    # --- Draw a KDE (smooth curve) on top using seaborn ---
    # KDE = Kernel Density Estimate — a smoothed version of the histogram
    ax2 = ax.twinx()   # Second y-axis on the right for the KDE curve
    sns.kdeplot(returns, ax=ax2, color="black", linewidth=2, label="KDE (density)")
    ax2.set_ylabel("Density", fontsize=LABEL_FONTSIZE)
    ax2.tick_params(right=False, labelright=False)  # Hide right axis ticks

    # --- Mark the zero line (no gain, no loss) ---
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero Return")

    # --- Show mean return ---
    mean_return = returns.mean()
    ax.axvline(x=mean_return, color="green", linestyle=":", linewidth=1.5,
               label=f"Mean ({mean_return:.2f}%)")

    # --- Labels and formatting ---
    ax.set_title(f"{ticker} — Distribution of Daily Returns", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel("Daily Return (%)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Frequency (days)", fontsize=LABEL_FONTSIZE)

    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.4)

    add_footnote(fig)
    save_figure(fig, f"{safe_ticker}_3_returns_histogram.png")


# ──────────────────────────────────────────────
# CHART 4: MOVING AVERAGE CHART
# ──────────────────────────────────────────────

def plot_moving_averages(df, ticker, safe_ticker):
    """
    Chart 4 — Moving Average Chart (SMA_7 and SMA_30).

    Overlays the 7-day and 30-day simple moving averages on top of the close price.
    Moving averages "smooth out" short-term noise to reveal the underlying trend.

    Why useful in finance:
    - SMA_7 = short-term trend (reacts quickly to price changes)
    - SMA_30 = medium-term trend (slower, more stable)
    - GOLDEN CROSS: SMA_7 crosses above SMA_30 → bullish signal (buy)
    - DEATH CROSS:  SMA_7 crosses below SMA_30 → bearish signal (sell)
    - Used by technical analysts and algorithmic trading systems

    Args:
        df (pd.DataFrame): Cleaned dataframe for one asset
        ticker (str): Ticker symbol
        safe_ticker (str): Safe filename prefix
    """
    print(f"[INFO] Drawing Moving Average chart for {ticker}...")

    color = ASSET_COLORS.get(ticker, "#333333")

    fig, ax = plt.subplots(figsize=(12, 5))
    plt.style.use(CHART_STYLE)

    # --- Draw close price (faded) as background reference ---
    ax.plot(df.index, df["Close"], color=color, linewidth=1, alpha=0.35, label="Close Price")

    # --- Draw 7-day SMA ---
    ax.plot(df.index, df["SMA_7"], color="darkorange", linewidth=1.8,
            linestyle="-", label="SMA 7-day")

    # --- Draw 30-day SMA ---
    ax.plot(df.index, df["SMA_30"], color="navy", linewidth=2,
            linestyle="--", label="SMA 30-day")

    # --- Labels and formatting ---
    ax.set_title(f"{ticker} — Moving Averages (SMA 7 & 30 Days)", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Price (USD)", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()

    add_footnote(fig)
    save_figure(fig, f"{safe_ticker}_4_moving_averages.png")


# ──────────────────────────────────────────────
# CHART 5: VOLATILITY CHART
# ──────────────────────────────────────────────

def plot_volatility(df, ticker, safe_ticker):
    """
    Chart 5 — 20-Day Rolling Volatility Chart.

    Shows how volatile (risky) the asset has been over time.
    Volatility = the standard deviation of daily returns over the past 20 days.
    A higher value means prices were swinging more wildly.

    Why useful in finance:
    - Risk management: high volatility = more uncertainty = higher risk
    - Options pricing: the Black-Scholes model uses volatility as a key input
    - Compare volatility across assets to understand relative risk
    - Volatility clustering: high-vol periods tend to cluster together

    Args:
        df (pd.DataFrame): Cleaned dataframe for one asset
        ticker (str): Ticker symbol
        safe_ticker (str): Safe filename prefix
    """
    print(f"[INFO] Drawing Volatility chart for {ticker}...")

    color = ASSET_COLORS.get(ticker, "#333333")

    # Drop NaN rows (first ~20 days won't have Volatility_20d)
    vol_data = df["Volatility_20d"].dropna()

    fig, ax = plt.subplots(figsize=(12, 4))
    plt.style.use(CHART_STYLE)

    # --- Fill the area under the volatility curve for visual impact ---
    ax.fill_between(vol_data.index, vol_data, alpha=0.3, color=color)
    ax.plot(vol_data.index, vol_data, color=color, linewidth=1.8, label="Volatility (20d)")

    # --- Draw a horizontal line at the average volatility level ---
    avg_vol = vol_data.mean()
    ax.axhline(y=avg_vol, color="black", linestyle="--", linewidth=1.2,
               label=f"Avg Volatility ({avg_vol:.2f}%)")

    # --- Labels and formatting ---
    ax.set_title(f"{ticker} — 20-Day Rolling Volatility", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Volatility (Std Dev of Daily Return, %)", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()

    add_footnote(fig)
    save_figure(fig, f"{safe_ticker}_5_volatility.png")


# ──────────────────────────────────────────────
# CHART 6: CORRELATION HEATMAP (ALL ASSETS)
# ──────────────────────────────────────────────

def plot_correlation_heatmap(all_data):
    """
    Chart 6 — Correlation Heatmap (shared chart across all 3 assets).

    Shows how closely the daily returns of AAPL, TSLA, and Gold move together.
    Correlation ranges from -1 to +1:
      +1.0 = perfect positive correlation (move together)
       0.0 = no relationship
      -1.0 = perfect negative correlation (move opposite)

    Why useful in finance:
    - Portfolio diversification: low or negative correlation between assets
      reduces overall portfolio risk (MPT — Modern Portfolio Theory)
    - AAPL and TSLA are both tech stocks → might be positively correlated
    - Gold is often negatively correlated with stocks (safe-haven asset)
    - Investors use this to build diversified portfolios

    Args:
        all_data (dict): Dictionary of {ticker: DataFrame} for all assets
    """
    print("[INFO] Drawing Correlation Heatmap for all assets...")

    # --- Build a combined DataFrame with one "Daily_Return" column per asset ---
    returns_dict = {}
    for ticker, df in all_data.items():
        if df is not None and "Daily_Return" in df.columns:
            # Name each column by the ticker symbol
            returns_dict[ticker] = df["Daily_Return"]

    if len(returns_dict) < 2:
        print("[WARNING] Not enough assets loaded to draw a correlation heatmap. Skipping.")
        return

    # Combine all return series into a single DataFrame, aligned by date
    combined = pd.DataFrame(returns_dict)
    combined.dropna(inplace=True)  # Remove rows where any asset has NaN

    # --- Calculate the correlation matrix ---
    corr_matrix = combined.corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.style.use(CHART_STYLE)

    # --- Draw the heatmap using seaborn ---
    # annot=True → show the correlation number inside each cell
    # fmt=".2f"  → round to 2 decimal places
    # cmap       → color scheme: red = positive, blue = negative
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",         # Green = positive, Red = negative
        vmin=-1, vmax=1,       # Fix color scale from -1 to +1
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"}
    )

    ax.set_title("Correlation Heatmap — Daily Returns\n(AAPL, TSLA, Gold)",
                 fontsize=TITLE_FONTSIZE, fontweight="bold")

    # Rotate axis labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

    add_footnote(fig)
    save_figure(fig, "correlation_heatmap.png")



# ──────────────────────────────────────────────
# CHART 18: FED RATE OVERLAY ON PRICE TREND
# ──────────────────────────────────────────────

def plot_fed_rate_overlay(all_data):
    """
    Fed Rate Overlay — Asset Price Trends vs. Federal Funds Rate.

    Plots all three asset closing prices on the left y-axis and overlays
    the Federal Funds Rate on a shared right y-axis, so you can visually
    see how rate changes relate to price movements across the period.

    Why useful in finance:
    - The Fed rate is one of the most powerful macro drivers in markets
    - Rising rates increase borrowing costs → pressure on growth stocks
    - Rising rates strengthen the USD → headwind for Gold (USD-denominated)
    - Visual overlay makes the relationship immediately readable
    - Extensible: adding Tier 2 indicators (CPI, GDP) means adding more
      lines to the right axis — no structural change needed

    Design decision:
    - Dual y-axis (twinx): asset prices on left, Fed rate % on right
    - Fed rate drawn as a dashed black step line (rates change in steps,
      not continuously, so a step line is more accurate than a smooth line)
    - Each asset uses its consistent ASSET_COLORS color

    Args:
        all_data (dict): {ticker: DataFrame} for all loaded assets
    """
    print("[INFO] Drawing Fed Rate Overlay chart...")

    # Load the Fed rate CSV from data/ folder
    fed_path = os.path.join(DATA_DIR, "macro_fed_rate_1y_daily.csv")

    if not os.path.exists(fed_path):
        print(f"[WARNING] Fed rate data not found: {fed_path}")
        print(f"          Fed Rate Overlay chart will be skipped.")
        print(f"          Fix: run Module 1 with a valid ALPHA_VANTAGE_API_KEY.")
        return

    try:
        fed_df = pd.read_csv(fed_path, index_col="Date", parse_dates=True)
        fed_df = fed_df.sort_index().dropna(subset=["value"])
    except Exception as e:
        print(f"[WARNING] Could not load Fed rate data: {e}. Skipping Fed Rate Overlay chart.")
        return

    if fed_df.empty:
        print("[WARNING] Fed rate CSV is empty. Skipping Fed Rate Overlay chart.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 6))
    plt.style.use(CHART_STYLE)

    # --- Left axis: one price line per asset ---
    for ticker, df in all_data.items():
        if df is None or "Close" not in df.columns:
            continue
        color = ASSET_COLORS.get(ticker, "#333333")
        ax1.plot(df.index, df["Close"], color=color, linewidth=1.8,
                 label=ticker, zorder=2)

    ax1.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel("Asset Close Price (USD)", fontsize=LABEL_FONTSIZE)
    ax1.grid(True, alpha=0.3)

    # --- Right axis: Fed Funds Rate ---
    ax2 = ax1.twinx()
    ax2.step(fed_df.index, fed_df["value"], color="black", linewidth=2,
             linestyle="--", where="post", label="Fed Funds Rate (%)", zorder=3)
    ax2.set_ylabel("Federal Funds Rate (%)", fontsize=LABEL_FONTSIZE, color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # --- Combined legend from both axes ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left")

    # --- Title and formatting ---
    ax1.set_title(
        "Asset Price Trends vs. Federal Funds Rate (1 Year)",
        fontsize=TITLE_FONTSIZE, fontweight="bold"
    )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()

    add_footnote(fig, source="Yahoo Finance / Alpha Vantage")
    save_figure(fig, "fed_rate_overlay.png")


# ──────────────────────────────────────────────
# CHART 19: MULTI-ASSET VOLATILITY COMPARISON
# ──────────────────────────────────────────────

def plot_volatility_comparison(all_data):
    """
    Multi-Asset Volatility Comparison — 20-Day Rolling Volatility.

    Plots the 20-day rolling volatility of all three assets on the same
    chart so you can directly compare how their risk levels evolved over
    the same time period.

    Why useful in finance:
    - The per-asset volatility chart shows each asset in isolation
    - This chart shows them together — you can see if risk spikes were
      market-wide (all assets spiked together) or asset-specific
    - Helps identify which asset has been consistently riskier
    - Directly supports the Risk Comparison section of the AI analysis

    Design decision:
    - All three lines share the same y-axis (same unit: % std dev)
    - Each asset uses its consistent ASSET_COLORS color
    - A shared average line is NOT drawn here (each asset's average
      is already shown on its individual volatility chart)

    Args:
        all_data (dict): {ticker: DataFrame} for all loaded assets
    """
    print("[INFO] Drawing Multi-Asset Volatility Comparison chart...")

    # Check that at least 2 assets have Volatility_20d data
    valid = {
        ticker: df
        for ticker, df in all_data.items()
        if df is not None and "Volatility_20d" in df.columns
    }

    if len(valid) < 2:
        print("[WARNING] Not enough assets with Volatility_20d data. Skipping Volatility Comparison chart.")
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    plt.style.use(CHART_STYLE)

    for ticker, df in valid.items():
        color = ASSET_COLORS.get(ticker, "#333333")
        vol_data = df["Volatility_20d"].dropna()
        ax.plot(vol_data.index, vol_data, color=color, linewidth=1.8,
                label=ticker, zorder=2)

    # --- Labels and formatting ---
    ax.set_title(
        "20-Day Rolling Volatility Comparison — AAPL vs TSLA vs Gold",
        fontsize=TITLE_FONTSIZE, fontweight="bold"
    )
    ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Volatility (Std Dev of Daily Return, %)", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()

    add_footnote(fig)
    save_figure(fig, "volatility_comparison.png")


# ──────────────────────────────────────────────
# FED RATE CORRELATION SCATTER (per asset)
# ──────────────────────────────────────────────

def plot_fed_rate_correlation(all_data):
    """
    Fed Rate Correlation Scatter — Daily Return vs. Fed Rate Change per Asset.

    For each asset, plots a scatter chart where:
      - X-axis = daily change in the Federal Funds Rate
      - Y-axis = daily return of the asset on the same day
      - A linear regression line is overlaid to show the direction and
        strength of the relationship
      - The R² value is displayed on the chart

    Why useful in finance:
    - Directly visualizes rate sensitivity for each asset
    - A negative slope for AAPL/TSLA confirms the theory: rising rates
      compress growth stock multiples (bearish for equities)
    - A positive or flat slope for Gold is expected: Gold benefits when
      real yields fall (rates drop)
    - R² tells us how much of the asset's daily return is explained by
      the Fed rate change alone (usually low — markets have many drivers)

    Design decisions:
    - One file per asset (3 files total) for clarity — overlaying all
      3 on one chart would make the regression lines hard to read
    - Fed rate daily change is computed as diff() on the value column
    - Only dates present in BOTH the asset data and Fed rate data are used
      (inner join on date index)
    - scipy.stats.linregress provides slope, intercept, and R²
    - Regression line is drawn across the full X range (min to max)

    Args:
        all_data (dict): {ticker: DataFrame} for all loaded assets
    """
    # Import here to keep the dependency explicit and close to its usage
    from scipy.stats import linregress

    print("[INFO] Drawing Fed Rate Correlation Scatter charts...")

    # Load the Fed rate CSV
    fed_path = os.path.join(DATA_DIR, "macro_fed_rate_1y_daily.csv")

    if not os.path.exists(fed_path):
        print(f"[WARNING] Fed rate data not found: {fed_path}")
        print(f"          Fed Rate Correlation charts will be skipped.")
        print(f"          Fix: run Module 1 with a valid ALPHA_VANTAGE_API_KEY.")
        return

    try:
        fed_df = pd.read_csv(fed_path, index_col="Date", parse_dates=True)
        fed_df = fed_df.sort_index().dropna(subset=["value"])
    except Exception as e:
        print(f"[WARNING] Could not load Fed rate data: {e}. Skipping Fed Rate Correlation charts.")
        return

    if fed_df.empty:
        print("[WARNING] Fed rate CSV is empty. Skipping Fed Rate Correlation charts.")
        return

    # Compute daily change in Fed rate (today's rate minus yesterday's rate)
    # This is what we correlate against — not the rate level itself,
    # because assets react to CHANGES in rates, not the absolute level.
    fed_df["rate_change"] = fed_df["value"].diff()

    # Generate one chart per asset
    for ticker, safe_ticker in ASSETS.items():
        df = all_data.get(ticker)

        if df is None or "Daily_Return" not in df.columns:
            print(f"[SKIP] {ticker} — no data available, skipping Fed Rate Correlation chart.")
            continue

        # --- Align asset returns with Fed rate changes by date (inner join) ---
        # Only keep dates where both datasets have a value
        combined = pd.DataFrame({
            "asset_return":  df["Daily_Return"],
            "fed_rate_chg":  fed_df["rate_change"],
        }).dropna()  # Drop rows where either value is NaN (first rows, missing dates)

        if len(combined) < 10:
            # Not enough overlapping data points for a meaningful regression
            print(f"[WARNING] {ticker} — fewer than 10 overlapping data points with Fed rate. Skipping.")
            continue

        x = combined["fed_rate_chg"].values   # Fed rate daily change (X axis)
        y = combined["asset_return"].values   # Asset daily return (Y axis)

        # --- Linear regression using scipy ---
        # linregress returns: slope, intercept, r_value, p_value, std_err
        slope, intercept, r_value, p_value, _ = linregress(x, y)
        r_squared = r_value ** 2  # R² = how much variance in Y is explained by X

        color = ASSET_COLORS.get(ticker, "#333333")

        fig, ax = plt.subplots(figsize=(9, 6))

        # --- Scatter plot: one dot per trading day ---
        ax.scatter(x, y, color=color, alpha=0.45, s=25, label="Daily observations", zorder=2)

        # --- Regression line across the full X range ---
        x_line = [x.min(), x.max()]
        y_line = [slope * xi + intercept for xi in x_line]
        ax.plot(x_line, y_line, color="black", linewidth=2,
                linestyle="--", label=f"Regression line (slope={slope:.3f})", zorder=3)

        # --- Reference lines at zero for both axes ---
        ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        ax.axvline(0, color="grey", linewidth=0.8, linestyle=":")

        # --- R² annotation in the top-right corner of the plot ---
        ax.annotate(
            f"R² = {r_squared:.4f}",
            xy=(0.97, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.8)
        )

        # --- Labels and formatting ---
        ax.set_title(
            f"{ticker} — Daily Return vs. Fed Funds Rate Change",
            fontsize=TITLE_FONTSIZE, fontweight="bold"
        )
        ax.set_xlabel("Daily Change in Fed Funds Rate (%)", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(f"{ticker} Daily Return (%)", fontsize=LABEL_FONTSIZE)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        add_footnote(fig, source="Yahoo Finance / Alpha Vantage")
        save_figure(fig, f"{safe_ticker}_fed_rate_correlation.png")


# ──────────────────────────────────────────────
# MAIN FUNCTION — runs all charts for all assets
# ──────────────────────────────────────────────

def visualize_all_assets():
    """
    Main entry point for Module 3.
    Loads cleaned data for all assets and generates all charts.

    Returns:
        dict: {ticker: DataFrame} of all loaded cleaned data
    """
    print("\n" + "=" * 55)
    print("  MODULE 3 — VISUALIZATION")
    print("=" * 55)

    # Make sure the charts/ folder exists
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # Dictionary to store loaded DataFrames for all assets
    all_data = {}

    # ── Step 1: Load cleaned data for each asset ──
    print("\n[INFO] Loading cleaned CSV files from output/ folder...")
    for ticker, safe_ticker in ASSETS.items():
        df = load_clean_csv(ticker, safe_ticker)
        all_data[ticker] = df  # Will be None if file not found

    print()

    # ── Step 2: Generate per-asset charts ──
    for ticker, safe_ticker in ASSETS.items():
        df = all_data.get(ticker)

        # Skip this asset if data couldn't be loaded
        if df is None:
            print(f"[SKIP] {ticker} — no data available, skipping all charts for this asset.")
            continue

        print(f"\n--- Generating charts for {ticker} ---")

        # Chart 1: Price Trend
        plot_price_trend(df, ticker, safe_ticker)

        # Chart 2: Volume
        plot_volume(df, ticker, safe_ticker)

        # Chart 3: Daily Returns Histogram
        plot_returns_histogram(df, ticker, safe_ticker)

        # Chart 4: Moving Averages
        plot_moving_averages(df, ticker, safe_ticker)

        # Chart 5: Volatility
        plot_volatility(df, ticker, safe_ticker)

    # ── Step 3: Generate shared charts ──
    print("\n--- Generating shared charts ---")
    plot_correlation_heatmap(all_data)

    # Fed Rate Overlay — skipped gracefully if macro CSV missing
    plot_fed_rate_overlay(all_data)

    # Multi-asset Volatility Comparison
    plot_volatility_comparison(all_data)

    # Fed Rate Correlation Scatter — one chart per asset, skipped gracefully if macro CSV missing
    plot_fed_rate_correlation(all_data)

    # ── Step 4: Summary ──
    print("\n" + "=" * 55)
    print("  [OK] All charts saved to charts/ folder.")
    print("  Files generated:")

    # List all PNG files in charts/
    if os.path.exists(CHARTS_DIR):
        chart_files = sorted([f for f in os.listdir(CHARTS_DIR) if f.endswith(".png")])
        for f in chart_files:
            print(f"    → charts/{f}")

    print("=" * 55 + "\n")

    return all_data


# ──────────────────────────────────────────────
# STANDALONE ENTRY POINT
# Runs when: python -m src.visualization
# ──────────────────────────────────────────────

if __name__ == "__main__":
    visualize_all_assets()
