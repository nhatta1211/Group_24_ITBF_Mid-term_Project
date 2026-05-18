"""
main.py — Full Pipeline
=======================
This is the master script that runs all 4 modules of the
AI-Powered Financial Data Agent in sequence:

  Step 1 → Module 1: Data Collection   (downloads raw CSV files)
  Step 2 → Module 2: Data Cleaning     (cleans and adds features)
  Step 3 → Module 3: Visualization     (generates chart PNG files)
  Step 4 → Module 4: AI Analysis       (generates AI text reports)

Run the full pipeline with:
    python -m src.main

Skip Step 1 if CSV files already exist (saves ~30s during development):
    python -m src.main --skip-collection

Each step is independent. If one step fails, the pipeline stops
and tells you exactly what went wrong and how to fix it.
"""

import os                    # Used to check whether CSV files exist in data/
import time                  # Used to measure how long each step takes
import traceback             # Used to print detailed error messages
import argparse              # Used to parse command-line flags like --skip-collection
from datetime import datetime  # Used to timestamp pipeline run summaries

# ── Import the main function from each module ──
# These are the exact function names defined in each module file.
# If you rename a function there, update it here too.
from src.data_collection import collect_all_assets
from src.cleaning import clean_all_assets
from src.visualization import visualize_all_assets
from src.ai_analysis import analyze_all_assets


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

# Project name shown in the console header
PROJECT_NAME = "AI-Powered Financial Data Agent"

# Set to True to stop the whole pipeline if any one step fails.
# Set to False to try and continue even after a failed step.
STOP_ON_ERROR = True

# CSV files that must exist in data/ when --skip-collection is used.
# These are exactly the files that Module 1 produces.
# If any are missing, the pipeline aborts before wasting time on Steps 2-4.
EXPECTED_CSV_FILES = [
    "data/AAPL_1y_daily.csv",
    "data/TSLA_1y_daily.csv",
    "data/GC_F_1y_daily.csv",
]

# Path to the pipeline run summary file written at the end of every run.
# Saved to output/ so it lives alongside the other pipeline outputs.
SUMMARY_FILE = "output/pipeline_run_summary.txt"


# ──────────────────────────────────────────────
# HELPER: CHECK DATA FOLDER STATUS
# Runs every time the pipeline starts.
# ──────────────────────────────────────────────

def check_data_folder() -> bool:
    """
    Scan the data/ folder and report the status of every expected CSV file.

    Runs unconditionally at pipeline start. Returns True if all files are
    present and non-empty (pipeline will auto-skip Step 1), or False if
    any file is missing or empty (Step 1 must run).

    Three possible states per file:
        [FOUND]   -- file exists and has content. Safe to skip Step 1.
        [MISSING] -- file does not exist. Step 1 must run to download it.
        [EMPTY]   -- file exists but has 0 bytes. Likely a failed previous
                     run. Step 1 must run to fix it.

    Returns:
        bool: True if ALL files found and non-empty, False otherwise.
    """
    print("\n  Data folder status:")

    all_found = True

    for filepath in EXPECTED_CSV_FILES:
        if not os.path.isfile(filepath):
            print(f"    [MISSING] {filepath}")
            all_found = False
        elif os.path.getsize(filepath) == 0:
            print(f"    [EMPTY]   {filepath}  <- corrupted, Step 1 will re-download")
            all_found = False
        else:
            size_kb = os.path.getsize(filepath) / 1024
            print(f"    [FOUND]   {filepath}  ({size_kb:.1f} KB)")

    if all_found:
        print("    → [STATUS] All data files found. Step 1 will be skipped automatically.")
    else:
        print("    → [STATUS] One or more files missing or empty. Step 1 will run.")

    return all_found


# ──────────────────────────────────────────────
# HELPER: PRINT A SECTION BANNER
# Makes the console output easy to scan visually.
# ──────────────────────────────────────────────

def print_banner(step_number, total_steps, title):
    """
    Print a clearly visible section header to the console.

    Example output:
    ╔══════════════════════════════════════════════════════╗
    ║  STEP 2 / 4 — Data Cleaning & Processing            ║
    ╚══════════════════════════════════════════════════════╝

    Args:
        step_number (int): The current step number (e.g., 2)
        total_steps (int): Total number of steps (e.g., 4)
        title (str): The step title
    """
    width = 54
    label = f"  STEP {step_number} / {total_steps} — {title}"
    print("\n╔" + "═" * width + "╗")
    print(f"║{label:<{width}}║")
    print("╚" + "═" * width + "╝")


def print_step_result(success, step_title, elapsed_seconds):
    """
    Print a short summary after each step completes or fails.

    Args:
        success (bool): Whether the step finished without errors
        step_title (str): Name of the step
        elapsed_seconds (float): How many seconds the step took
    """
    status = "[OK]" if success else "[FAILED]"
    print(f"\n{status} {step_title} — completed in {elapsed_seconds:.1f}s")


# ──────────────────────────────────────────────
# HELPER: WRITE PIPELINE RUN SUMMARY TO FILE
# ──────────────────────────────────────────────

def write_run_summary(results_log, all_step_titles, total_elapsed, watch_run=None):
    """
    Write a pipeline run summary to output/pipeline_run_summary.txt.

    Called at the end of every pipeline run (including each --watch cycle).
    Overwrites the file each time — always shows the most recent run only.

    Why overwrite instead of append?
        The file is meant as a quick status check, not a full history log.
        For history, you would use a database or append-mode log file.
        Overwriting keeps the file small and always immediately readable.

    The file records:
        - Timestamp of this run
        - Whether --watch mode was active and which cycle number
        - Per-step result: PASSED / FAILED / SKIPPED with elapsed time
        - Total pipeline duration
        - Overall result: ALL PASSED or ISSUES DETECTED

    Args:
        results_log (dict):    {step_title: {"success": bool, "elapsed": float, "skipped": bool}}
        all_step_titles (list): All step titles in order (to detect aborted steps)
        total_elapsed (float): Total pipeline duration in seconds
        watch_run (int or None): If running in --watch mode, the cycle number (1, 2, 3...).
                                 None if this is a normal single run.
    """
    # Make sure the output/ folder exists before writing
    os.makedirs("output", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build the summary lines
    lines = []
    lines.append("=" * 56)
    lines.append("  PIPELINE RUN SUMMARY")
    lines.append("=" * 56)
    lines.append(f"  Timestamp:   {timestamp}")

    # If running in watch mode, show which cycle this is
    if watch_run is not None:
        lines.append(f"  Watch mode:  Yes (cycle #{watch_run})")
    else:
        lines.append(f"  Watch mode:  No (single run)")

    lines.append("")
    lines.append("  Step Results:")

    all_passed = True

    for step_title in all_step_titles:
        if step_title in results_log:
            log = results_log[step_title]
            if log.get("skipped"):
                lines.append(f"    —  {step_title:<34} SKIPPED  (--skip-collection)")
            else:
                status_word = "PASSED" if log["success"] else "FAILED"
                lines.append(f"    {'✓' if log['success'] else '✗'}  {step_title:<34} {status_word}  ({log['elapsed']:.1f}s)")
                if not log["success"]:
                    all_passed = False
        else:
            # Step never ran — pipeline was aborted before reaching it
            lines.append(f"    —  {step_title:<34} SKIPPED  (pipeline aborted)")
            all_passed = False

    lines.append("")
    lines.append(f"  Total time:  {total_elapsed:.1f}s")
    lines.append(f"  Result:      {'ALL PASSED' if all_passed else 'ISSUES DETECTED'}")
    lines.append("=" * 56)

    # Write to file (overwrite mode — 'w' not 'a')
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[SAVED] Pipeline run summary → {SUMMARY_FILE}")


# ──────────────────────────────────────────────
# STEP RUNNERS
# Each function runs one module and returns True/False.
# Wrapping each module in its own function makes error
# handling clean and beginner-friendly.
# ──────────────────────────────────────────────

def run_data_collection():
    """
    Step 1 — Run Module 1: Data Collection.

    Downloads 1 year of daily price data for AAPL, TSLA, and GC=F
    from Yahoo Finance, and macroeconomic data from Alpha Vantage.
    Saves all datasets as CSV files in data/.

    Returns:
        tuple: (success: bool, result: dict or None)
    """
    try:
        # Call Module 1's main function
        result = collect_all_assets()

        # Check that we actually got data back (not an empty dict)
        if not result:
            print("[WARNING] Data collection returned no results.")
            print("          → Check your internet connection and try again.")
            return False, None

        # Count how many assets were successfully collected
        successful = sum(1 for df in result.values() if df is not None)
        print(f"[INFO] {successful}/{len(result)} datasets collected successfully.")
        return True, result

    except Exception as e:
        # Something unexpected went wrong — print a helpful error message
        print(f"\n[ERROR] Data Collection failed: {e}")
        print("        → Common fixes:")
        print("          1. Check your internet connection")
        print("          2. Run: pip install --upgrade yfinance")
        print("          3. Check ALPHA_VANTAGE_API_KEY is set in your .env file")
        print("          4. Try running standalone: python -m src.data_collection")
        if STOP_ON_ERROR:
            traceback.print_exc()  # Show the full technical error for debugging
        return False, None


def run_cleaning():
    """
    Step 2 — Run Module 2: Data Cleaning & Processing.

    Reads raw CSVs from data/, cleans them (handles missing values,
    flags outliers), adds features (SMA, volatility), and saves
    cleaned CSVs to output/.

    Returns:
        tuple: (success: bool, result: dict or None)
    """
    try:
        result = clean_all_assets()

        if not result:
            print("[WARNING] Cleaning returned no results.")
            print("          → Make sure Step 1 ran successfully first.")
            return False, None

        successful = sum(1 for df in result.values() if df is not None)
        print(f"[INFO] {successful}/{len(result)} assets cleaned successfully.")
        return True, result

    except Exception as e:
        print(f"\n[ERROR] Data Cleaning failed: {e}")
        print("        → Common fixes:")
        print("          1. Make sure Step 1 (Data Collection) ran first")
        print("          2. Check that the data/ folder contains CSV files")
        print("          3. Try running standalone: python -m src.cleaning")
        if STOP_ON_ERROR:
            traceback.print_exc()
        return False, None


def run_visualization():
    """
    Step 3 — Run Module 3: Visualization.

    Reads cleaned CSVs from output/, generates chart types
    (price trend, volume, histogram, moving averages, volatility,
    correlation heatmap, and macro overlays), and saves PNG files
    to charts/.

    Returns:
        tuple: (success: bool, result: dict or None)
    """
    try:
        result = visualize_all_assets()

        # Visualization always runs — even partial results are useful
        print("[INFO] Chart generation complete.")
        return True, result

    except Exception as e:
        print(f"\n[ERROR] Visualization failed: {e}")
        print("        → Common fixes:")
        print("          1. Make sure Step 2 (Cleaning) ran first")
        print("          2. Check that the output/ folder contains *_clean.csv files")
        print("          3. Try running standalone: python -m src.visualization")
        if STOP_ON_ERROR:
            traceback.print_exc()
        return False, None


def run_ai_analysis():
    """
    Step 4 — Run Module 4: AI Analysis.

    Reads cleaned CSVs from output/, extracts key statistics,
    sends data-grounded prompts to the Groq API (llama-3.3-70b),
    and saves analysis reports as .txt files to output/.

    Returns:
        tuple: (success: bool, result: None)
            Note: analyze_all_assets() does not return data,
            it saves files directly to disk.
    """
    try:
        # analyze_all_assets() handles its own output (saves .txt files)
        analyze_all_assets()
        print("[INFO] AI analysis reports generated.")
        return True, None

    except Exception as e:
        print(f"\n[ERROR] AI Analysis failed: {e}")
        print("        → Common fixes:")
        print("          1. Check that GROQ_API_KEY is set in your .env file")
        print("          2. Verify your Groq API key at: https://console.groq.com")
        print("          3. Make sure Step 2 (Cleaning) ran first")
        print("          4. Try running standalone: python -m src.ai_analysis")
        if STOP_ON_ERROR:
            traceback.print_exc()
        return False, None


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def run_pipeline(skip_collection: bool = False, watch_run: int = None):
    """
    Master function — runs all 4 modules in sequence.

    Tracks the success/failure of each step and prints a final
    summary report so you know exactly what succeeded and what failed.
    Also writes a summary file to output/pipeline_run_summary.txt.

    Args:
        skip_collection (bool): If True, Step 1 (Data Collection) is skipped.
                                Use this when data/*.csv files already exist
                                and you want to save time during development.
        watch_run (int or None): If called from the --watch scheduler, this is
                                 the current cycle number (1, 2, 3...).
                                 None means a normal single run.
    """

    # ── Pipeline header ──
    print("\n" + "█" * 56)
    print(f"  {PROJECT_NAME}")
    if watch_run is not None:
        print(f"  Watch Mode — Cycle #{watch_run}")
    else:
        print(f"  Full Pipeline — 4 Steps")
    if skip_collection:
        print(f"  [INFO] --skip-collection flag detected.")
    print("█" * 56)

    # ── Always check what is already in data/ before any step runs ──────────
    all_files_present = check_data_folder()

    # ── Automatic skip / force logic ─────────────────────────────────────────
    # In watch mode: always force Step 1 to run regardless of what's in data/.
    # This ensures data is actually refreshed on every scheduled cycle —
    # a scheduler that never re-downloads data is pointless.
    #
    # In normal mode: auto-skip if all files present, abort if flag set but files missing.
    if watch_run is not None:
        # Force fresh download every watch cycle — override any skip flag
        skip_collection = False
        print("\n  [INFO] Watch mode active — Step 1 will always run to refresh data.")
    elif not skip_collection and all_files_present:
        skip_collection = True
        print("\n  [AUTO-SKIP] All files detected. Step 1 will be skipped automatically.")
        print("              Delete files in data/ and re-run to force a fresh download.")
    elif skip_collection and not all_files_present:
        print("\n[ERROR] --skip-collection was set but required files are missing.")
        print("        See [MISSING] / [EMPTY] entries above.")
        print("        Fix: run without --skip-collection to download the data first.")
        print("\n[PIPELINE ABORTED]\n")
        return

    # Record the pipeline start time so we can report total duration
    pipeline_start = time.time()

    # ── Define all pipeline steps ──
    steps = [
        (1, "Data Collection",            run_data_collection),
        (2, "Data Cleaning & Processing", run_cleaning),
        (3, "Visualization",              run_visualization),
        (4, "AI Analysis",                run_ai_analysis),
    ]

    total_steps = len(steps)
    all_step_titles = [title for _, title, _ in steps]

    # Track results for the final summary
    results_log = {}

    # ── Run each step in order ──
    for step_number, step_title, runner_fn in steps:

        # ── Handle --skip-collection ─────────────────────────────────────
        if skip_collection and step_number == 1:
            print_banner(step_number, total_steps, step_title)
            print("[INFO] Skipping data collection (--skip-collection flag is set).")
            print("       Using existing CSV files in data/.")
            results_log[step_title] = {"success": True, "elapsed": 0.0, "skipped": True}
            continue

        print_banner(step_number, total_steps, step_title)

        step_start = time.time()
        success, _ = runner_fn()
        elapsed = time.time() - step_start

        print_step_result(success, step_title, elapsed)

        results_log[step_title] = {"success": success, "elapsed": elapsed, "skipped": False}

        if not success and STOP_ON_ERROR:
            print("\n[PIPELINE STOPPED]")
            print(f"  Step '{step_title}' failed. Fix the error above and re-run.")
            print("  Tip: You can run each module individually to debug:")
            print("    python -m src.data_collection")
            print("    python -m src.cleaning")
            print("    python -m src.visualization")
            print("    python -m src.ai_analysis")
            break

    # ── Final Summary Report ──
    total_elapsed = time.time() - pipeline_start

    print("\n" + "═" * 56)
    print("  PIPELINE SUMMARY")
    print("═" * 56)

    all_passed = True
    for step_title, log in results_log.items():
        if log.get("skipped"):
            print(f"  —  {step_title:<34} SKIPPED  (--skip-collection)")
        else:
            status_icon = "✓" if log["success"] else "✗"
            status_word = "PASSED" if log["success"] else "FAILED"
            print(f"  {status_icon}  {step_title:<34} {status_word}  ({log['elapsed']:.1f}s)")
            if not log["success"]:
                all_passed = False

    # Steps that never ran due to early abort
    aborted_skips = [t for t in all_step_titles if t not in results_log]
    for skipped_title in aborted_skips:
        print(f"  —  {skipped_title:<34} SKIPPED  (pipeline aborted)")
        all_passed = False

    print("═" * 56)
    print(f"  Total time: {total_elapsed:.1f}s")

    if all_passed:
        print("\n  [OK] All steps completed successfully!")
        print("\n  Output files:")
        print("    data/          → raw CSV files (Module 1)")
        print("    output/        → cleaned CSVs + AI analysis .txt files")
        print("    charts/        → PNG chart images (Module 3)")
    else:
        print("\n  [WARNING] One or more steps did not complete.")
        print("  See the errors above for details.")

    print("═" * 56 + "\n")

    # ── Write run summary to file ──
    # Always called — even on partial/failed runs — so the file always
    # reflects the most recent pipeline attempt.
    write_run_summary(results_log, all_step_titles, total_elapsed, watch_run=watch_run)


# ──────────────────────────────────────────────
# CLI ARGUMENT PARSING
# Runs when: python -m src.main
# Runs when: python -m src.main --skip-collection
# ──────────────────────────────────────────────

def parse_args():
    """
    Parse command-line arguments passed to the script.

    Supported flags:
        --skip-collection     Skip Step 1, use existing CSV files in data/
        --watch <seconds>     Run the pipeline repeatedly every N seconds.
                              Press Ctrl+C to stop the scheduler.

    Returns:
        argparse.Namespace: Object with args.skip_collection and args.watch.
    """
    parser = argparse.ArgumentParser(
        description="AI-Powered Financial Data Agent — Full Pipeline",
        epilog=(
            "Examples:\n"
            "  python -m src.main                    # full pipeline (single run)\n"
            "  python -m src.main --skip-collection  # skip Step 1 (use existing CSVs)\n"
            "  python -m src.main --watch 3600       # run every hour (3600 seconds)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help=(
            "Skip Step 1 (Data Collection) and use existing CSV files in data/. "
            "Saves ~30s per run during development when data is already downloaded. "
            "Note: ignored in --watch mode (Step 1 always runs to refresh data)."
        ),
    )

    parser.add_argument(
        "--watch",
        type=int,
        default=None,
        metavar="SECONDS",
        help=(
            "Run the full pipeline repeatedly, waiting SECONDS between each run. "
            "Step 1 always runs in watch mode to ensure data is refreshed. "
            "Press Ctrl+C to stop. Example: --watch 3600 runs every hour."
        ),
    )

    return parser.parse_args()


# ──────────────────────────────────────────────
# STANDALONE ENTRY POINT
# Runs when: python -m src.main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.watch is not None:
        # ── Watch mode: run pipeline repeatedly every N seconds ──
        interval = args.watch

        print("\n" + "█" * 56)
        print(f"  {PROJECT_NAME}")
        print(f"  Watch Mode — interval: {interval}s ({interval / 60:.1f} min)")
        print(f"  Press Ctrl+C to stop.")
        print("█" * 56)

        cycle = 0

        try:
            while True:
                cycle += 1
                print(f"\n[INFO] Starting watch cycle #{cycle} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # In watch mode, skip_collection is always ignored —
                # run_pipeline() forces Step 1 when watch_run is set.
                run_pipeline(skip_collection=False, watch_run=cycle)

                print(f"\n[INFO] Cycle #{cycle} complete. Next run in {interval}s.")
                print(f"       Press Ctrl+C to stop.\n")
                time.sleep(interval)

        except KeyboardInterrupt:
            # User pressed Ctrl+C — exit cleanly without a stack trace
            print(f"\n\n[INFO] Watch mode stopped by user after {cycle} cycle(s).")
            print("[OK] Exiting.\n")

    else:
        # ── Normal single run ──
        run_pipeline(skip_collection=args.skip_collection)
