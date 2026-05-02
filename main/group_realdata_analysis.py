"""
Run Analysis Script - Simple configurable pipeline for PSA conversion and progressive analysis.

Run directly from PyCharm: Right-click → Run 'run_analysis'
Configure the parameters below before running.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.io.converter import convert_psa_to_gbf
from analysis.orchestrator import analyze_batch


# ============================================================================
# CONFIGURATION - Modify these parameters
# ============================================================================

# Input: PSA format files (raw data from psysuite)
PSA_INPUT_DIR = "/data/CODE/python/adopy_tests/data/input/expdata"

# Output: GBF format files (converted data)
GBF_OUTPUT_DIR = "/data/CODE/python/adopy_tests/data/input/expdata/gbf"

# Output: Excel results
RESULTS_OUTPUT_DIR = "/data/CODE/python/adopy_tests/data/input/expdata"

# Project name (used in output filenames)
PROJECT_NAME = "BIS_fx_vs_ad_td_2model_rel"

# Fitting method: 'logistic', 'probit', or 'gaussfit'
FITTING_METHOD = "logistic"

# Bin size (only for gaussfit method)
BIN_SIZE = 50.0

# Pipeline control
RUN_CONVERSION = True   # Set False to skip conversion
RUN_ANALYSIS = True     # Set False to skip analysis


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the analysis pipeline."""
    
    print("="*70)
    print("POSTPROCESSING ANALYSIS PIPELINE")
    print("="*70)
    print(f"\nProject: {PROJECT_NAME}")
    print(f"Method:  {FITTING_METHOD}")
    print()
    
    try:
        # Create directories
        os.makedirs(GBF_OUTPUT_DIR, exist_ok=True)
        os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)
        
        # Step 1: Convert PSA → GBF
        if RUN_CONVERSION:
            print("STEP 1: Converting PSA to GBF...")
            stats = convert_psa_to_gbf(PSA_INPUT_DIR, GBF_OUTPUT_DIR)
            print(f"✓ Converted: {stats['converted']}/{stats['total_files']} files\n")
        
        # Step 2: Progressive Analysis
        if RUN_ANALYSIS:
            print("STEP 2: Progressive Analysis...")
            results = analyze_batch(
                input_dir=GBF_OUTPUT_DIR,
                output_dir=RESULTS_OUTPUT_DIR,
                project_name=PROJECT_NAME,
                method=FITTING_METHOD,
                bin_size=BIN_SIZE
            )
            print(f"✓ Analyzed: {len(results)} subjects\n")
        
        # Done
        print("="*70)
        print("✓ COMPLETED")
        print("="*70)
        print(f"\nOutput files:")
        print(f"  • results_{PROJECT_NAME}_{FITTING_METHOD}_prog_wide.xlsx")
        print(f"  • results_{PROJECT_NAME}_{FITTING_METHOD}_prog_long.xlsx")
        print()
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
