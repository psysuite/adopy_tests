"""
Real Data Analysis Pipeline - Convert PSA to GBF and create metrics Excel (wide + long).

UNIFIED PROCESSING: Uses same code path as synthetic data (UnifiedGBFProcessor).
Only differences: metadata columns (real has age/gender/etc vs synthetic has pse_true/jnd_true).
Metrics computed without B5 (no ground truth PSE/JND).

Run directly from PyCharm: Right-click → Run 'group_realdata_analysis'
Configure the parameters below before running.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.io.converter import convert_psa_to_gbf
from analysis.orchestration.unified_processor import UnifiedGBFProcessor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION - Modify these parameters
# ============================================================================

# Input: PSA format files (raw data from psysuite)
PSA_INPUT_DIR = "/data/CODE/python/adopy_tests/data/input/expdata"

# Output: GBF format files (converted data)
GBF_OUTPUT_DIR = "/data/CODE/python/adopy_tests/data/input/expdata/gbf"

# Excel output directory
EXCEL_OUTPUT_DIR = Path("/data/CODE/python/adopy_tests/R/indata")

# Project name (used in output filenames)
PROJECT_NAME = "BIS_fx_vs_ad_td"

# Pipeline control flags
RUN_CONVERSION = True        # Set False to skip PSA→GBF conversion
RUN_METRICS = True           # Set False to skip metrics calculation


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the real data analysis pipeline with unified processing."""

    print("="*70)
    print("REAL DATA ANALYSIS PIPELINE (UNIFIED PROCESSING)")
    print("="*70)
    print(f"\nProject: {PROJECT_NAME}")
    print()

    try:
        # Create directories
        os.makedirs(GBF_OUTPUT_DIR, exist_ok=True)
        os.makedirs(EXCEL_OUTPUT_DIR, exist_ok=True)

        # STEP 1: Convert PSA → GBF
        if RUN_CONVERSION:
            print("STEP 1: Converting PSA to GBF...")
            stats = convert_psa_to_gbf(PSA_INPUT_DIR, GBF_OUTPUT_DIR)
            print(f"✓ Converted: {stats['converted']}/{stats['total_files']} files\n")

        # STEP 2: Process GBF files with unified processor
        # (SAME CODE as synthetic - only data_type differs)
        if RUN_METRICS:
            print("STEP 2: Calculating metrics and generating Excel (unified processing)...")

            processor = UnifiedGBFProcessor(
                gbf_dir=Path(GBF_OUTPUT_DIR),
                model_name=None,  # Not used for real data (PosteriorExtractor always uses ABS1)
                data_type='real',  # Key difference: real data flag
                output_dir=EXCEL_OUTPUT_DIR,
                pse_grid=None,     # Not used for real data
                jnd_grid=None,     # Not used for real data
                offset=500,
                trial_blocks=[40, 60, 80, 100, 120, 140, 160, 180, 200],
                verbose=True,
            )

            # Process GBF files
            df_wide, df_long = processor.process()

            if df_wide.empty:
                print("✗ No data processed")
                sys.exit(1)

            # Save Excel files (using only PROJECT_NAME for real data)
            processor.save_excel(
                output_dir=EXCEL_OUTPUT_DIR,
                filename_prefix=f"{PROJECT_NAME}_rel2_logistic"
            )

            print(f"\n✓ Metrics and Excel generation complete")

        # Done
        print("\n" + "="*70)
        print("✓ REAL DATA ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nOutput files:")
        print(f"  - GBF files: {GBF_OUTPUT_DIR}/*.txt")
        print(f"  - Excel wide: {EXCEL_OUTPUT_DIR}/{PROJECT_NAME}_rel2_logistic_wide.xlsx")
        print(f"  - Excel long: {EXCEL_OUTPUT_DIR}/{PROJECT_NAME}_rel2_logistic_long.xlsx")
        print()
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
