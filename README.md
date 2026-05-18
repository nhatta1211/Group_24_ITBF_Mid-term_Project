# AI-Powered Financial Data Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20LLaMA--3.3--70b-orange)
![Data](https://img.shields.io/badge/Data-Yahoo%20Finance%20%7C%20Alpha%20Vantage-lightblue)

An end-to-end automated pipeline that collects live financial data, cleans and engineers features, generates publication-quality charts, and produces grounded AI-written reports — all from a single command.

**Assets tracked:** AAPL · TSLA · GC=F (Gold Futures)
**Data sources:** Yahoo Finance (price data) · Alpha Vantage (Federal Funds Rate)
**LLM:** LLaMA-3.3-70b via Groq API

---

## Table of Contents

- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)

---

## Project Structure

```
ai-financial-agent/
│
├── src/
│   ├── data_collection.py     # Module 1 — downloads raw data
│   ├── cleaning.py            # Module 2 — cleans and engineers features
│   ├── visualization.py       # Module 3 — generates charts
│   ├── ai_analysis.py         # Module 4 — AI reports via Groq
│ 
├── main.py                    # Master pipeline runner
├── data/                      # Raw CSVs (auto-created by Module 1)
├── output/                    # Cleaned CSVs + AI reports (auto-created)
├── charts/                    # PNG charts (auto-created by Module 3)
│
├── .env                       # Your API keys — never commit this
├── .env.example               # Key template — safe to commit
├── requirements.txt           # Pinned dependencies
├── requirements-lock.txt      # Full pip freeze snapshot
├── README.md                  # This file
├── .gitignore
```

> `data/`, `output/`, and `charts/` are excluded from Git and created automatically on first run.

---

## Quickstart
```bash
# 1. Clone the repo

**Option A — Using Git (recommended):**
```bash
git clone https://github.com/nhatta1211/ITBF_Mid-term_Project.git
cd ITBF_Mid-term_Project
``

**Option B — Download ZIP (no Git required):**
1. Go to https://github.com/nhatta1211/ITBF_Mid-term_Project
2. Click the green **Code** button → **Download ZIP**
3. Extract the ZIP file
4. Open a terminal inside the extracted folder before continuing

**Option B — Download ZIP (no Git required):**
1. Go to https://github.com/nhatta1211/ITBF_Mid-term_Project
2. Click the green **Code** button → **Download ZIP**
3. Extract the ZIP file
4. Open a terminal inside the extracted folder before continuing

# 2. Create and activate a virtual environment
python -m venv my_env
my_env\Scripts\activate        # Windows
# source my_env/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
# This creates your .env file from the template.
# Open .env and fill in your actual API keys.
# (See Configuration section for where to get them)

# 5. Run the pipeline
python -m src.main
```
> **Note:** If Step 5 fails with `yfinance` returning 0 rows, update yfinance first:
> ```bash
> pip install --upgrade yfinance
> ```
---

## Installation

**Requirements:** Python 3.10+, internet connection.

```bash
pip install -r requirements.txt
```

For exact environment reproduction:

```bash
pip install -r requirements-lock.txt
```

**Important:** Always activate your virtual environment before running the project.
If you see `ModuleNotFoundError`, the virtual environment is likely not active.

```bash
my_env\Scripts\activate    # Windows
source my_env/bin/activate # macOS / Linux
```

**Installing scipy on Windows:** If `pip install scipy` fails with a compiler error:

```bash
pip install scipy --only-binary=:all:
```

---

## Configuration

Copy the template and fill in your keys:

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
```

```env
# .env
GROQ_API_KEY=your_groq_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```

| Service | URL | Cost | Notes |
|---|---|---|---|
| Groq | https://console.groq.com | Free | LLaMA-3.3-70b included |
| Alpha Vantage | https://www.alphavantage.co/support/#api-key | Free | 25 req/day limit |
| Yahoo Finance | No key needed | Free | Via `yfinance` library |

**If Alpha Vantage key is missing:** The pipeline still runs fully. Fed rate charts and macro context section are skipped gracefully with a `[WARNING]` message — no crash.

**Security:** Never commit `.env`. It is already in `.gitignore`. If you accidentally push a key, rotate it immediately at the provider's console.

---

## Usage

### Full pipeline

```bash
python -m src.main
```

Auto-skips data download if `data/` files already exist.

### Skip data download (development)

```bash
python -m src.main --skip-collection
```

Useful when iterating on cleaning, charts, or AI prompts. Saves ~30s per run.

### Watch mode (scheduled refresh)

```bash
python -m src.main --watch 3600    # re-run every hour
python -m src.main --watch 1800    # re-run every 30 minutes
```

Runs the full pipeline repeatedly on an interval. Data is always re-downloaded in watch mode. Press `Ctrl+C` to stop. Terminal must stay open.

### Run individual modules

```bash
python -m src.data_collection   # Step 1 only
python -m src.cleaning          # Step 2 only
python -m src.visualization     # Step 3 only
python -m src.ai_analysis       # Step 4 only
```

---

## Output Files

| File | Description |
|---|---|
| `output/*_clean.csv` | Cleaned data with engineered features |
| `output/*_analysis.txt` | AI report per asset (trend, risk, events, macro) |
| `output/macro_analysis.txt` | Fed rate analysis across all assets |
| `output/comparison_report.txt` | Cross-asset ranking and comparison |
| `output/pipeline_run_summary.txt` | Pass/fail summary of the last run |
| `charts/*.png` | All 9 chart types (6 per-asset + 3 shared) |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
Virtual environment is not active, or you are not in the project root folder.
```bash
my_env\Scripts\activate
python -m src.main
```

**Pipeline aborts — "Required files are missing"**
Running `--skip-collection` but `data/` is empty.
```bash
python -m src.main    # run without the flag first
```

**`yfinance` returns 0 rows**
Yahoo Finance may have changed their API. Update yfinance:
```bash
pip install --upgrade yfinance
```

**Groq `RateLimitError` during AI analysis**
The pipeline retries automatically. If it keeps failing, wait 1–2 minutes then re-run:
```bash
python -m src.ai_analysis
```

**Alpha Vantage daily limit exceeded**
Resets at midnight UTC. Not a critical error — pipeline still completes without Fed rate data.
