#!/usr/bin/env python3
"""
Generate simulated temporal bisection files on a PSE/JND grid.
Uses 1-model relative ADOpy approach.

PSE grid: [480, 490, 500, 510, 520]
JND grid: [20, 30, 40, 50, 60]
→ 25 files × 200 trials each
Output: 
  - GBF files: data/output/sim_grid/1model_rel/1model_rel_S_<PSE>_<JND>.txt
  - Plots: data/output/sim_grid/1model_rel/results/
  - Excel: data/output/sim_grid/1model_rel/results/results_1model_rel.xlsx
"""

import os
import sys
from pathlib import Path
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.simulation_engine import SimulationEngine
from utilities.plotting import plot_group_histograms, plot_group_model_histograms, plot_group_psychometric
from utilities.psychometric_analysis import consolidate_results, add_group_stats_to_excel

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME  = "1model_rel"
OUTPUT_DIR  = "../data/output/sim_grid/1model_rel"
RESULTS_DIR = "../data/output/sim_grid/1model_rel/results"
N_TRIALS    = 200
OFFSET      = 500

PSE_GRID = [480, 490, 500, 510, 520]
JND_GRID = [20, 30, 40, 50, 60]

USE_FIXED_TRIALS   = True
N_FIXED            = 10
FIXED_OFFSETS_REL  = [25, 75, 125, 175, 225]
SAVE_GBF_FILES = True

ADO_PARAMS = {"guess_rate": 0.5, "lapse_rate": 0.04, "noise_perc": 0.1}
BIS_PARAMS = {"min": 5, "max": 300, "offset": OFFSET, "ntrials": N_TRIALS, "is_absolute": False}

FIXED_TRIALS_CONFIG = {
    "use": USE_FIXED_TRIALS,
    "n_fixed": N_FIXED,
    "offsets": FIXED_OFFSETS_REL,
}

# ============================================================================


def main():
    engine = SimulationEngine(MODEL_NAME, offset=OFFSET)
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    grid = list(product(PSE_GRID, JND_GRID))
    print(f"Generating {len(grid)} files → {output_dir}")
    print(f"Results → {results_dir}")
    print()

    all_rows_list = []
    all_results = []

    for pse, jnd in grid:
        filename = f"{MODEL_NAME}_S_{pse}_{jnd}.txt"
        out_path = output_dir / filename

        rows, result_dict = engine.simulate_subject(
            pse=pse,
            jnd=jnd,
            ntrials=N_TRIALS,
            ado_params=ADO_PARAMS,
            bis_params=BIS_PARAMS,
            fixed_trials_config=FIXED_TRIALS_CONFIG,
        )

        # Convert rows to GBF format
        gbf_rows = [
            {
                'lat': row['lat'],
                'count': 1,
                'user_ans': row['user_ans'],
                'confl_magn': 0,
            }
            for row in rows
        ]

        if SAVE_GBF_FILES:
            engine.save_gbf_file(gbf_rows, str(out_path))
        print(f"  {filename}")
        
        # Collect for plots and Excel
        all_rows_list.append(rows)
        all_results.append(result_dict)

    print(f"\nDone. {len(grid)} files written to {output_dir}")
    
    # Generate group plots
    print(f"\nGenerating plots...")
    if all_rows_list:
        plot_group_histograms(all_rows_list, str(results_dir), MODEL_NAME, OFFSET)
        plot_group_psychometric(all_rows_list, str(results_dir), MODEL_NAME, OFFSET)
        print(f"Plots saved to {results_dir}")
    
    # Generate Excel report
    print(f"\nGenerating Excel report...")
    if all_results:
        try:
            excel_filepath = consolidate_results(all_results, str(results_dir), MODEL_NAME)
            add_group_stats_to_excel(excel_filepath)
            print(f"Results saved to: {excel_filepath}")
        except Exception as e:
            print(f"Warning: Could not generate Excel report: {e}")
    
    print(f"\nAll done!")


if __name__ == "__main__":
    main()
