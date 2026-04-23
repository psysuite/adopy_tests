#!/usr/bin/env python3
"""
Generate simulated temporal bisection files on a PSE/JND grid with random variation.
Uses 2-model relative ADOpy approach (pre + post).

PSE grid: [480, 500, 520]
JND grid: [20, 40, 60]
→ 9 groups × 20 subjects each = 180 files × 200 trials each
Each subject has PSE/JND ± 2.5 variation from group center

Output: 
  - GBF files: data/output/sim_gridrnd/2model_rel/
  - Plots: data/output/sim_gridrnd/2model_rel/results/
  - Excel: data/output/sim_gridrnd/2model_rel/results/results_2model_rel.xlsx
"""

import os
import sys
from pathlib import Path
from itertools import product

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.simulation_engine import SimulationEngine
from utilities.plotting import plot_group_histograms, plot_group_model_histograms, plot_group_psychometric
from utilities.psychometric_analysis import consolidate_results, add_group_stats_to_excel

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME  = "2model_rel"
OUTPUT_DIR  = "../data/output/sim_gridrnd/2model_rel"
RESULTS_DIR = "../data/output/sim_gridrnd/2model_rel/results"
N_SUBJECTS_PER_GROUP = 20
N_TRIALS    = 200
OFFSET      = 500
VARIATION   = 2.5
SAVE_GBF_FILES = True

PSE_GRID = [480, 500, 520]
JND_GRID = [20, 40, 60]

USE_FIXED_TRIALS   = True
N_FIXED            = 10
FIXED_OFFSETS_REL  = [225, 175, 125, 75, 25]

ADO_PARAMS = {"guess_rate": 0.5, "lapse_rate": 0.04, "noise_perc": 0.1}
BIS_PARAMS_HALF = {"min": 0, "max": 300, "offset": OFFSET, "ntrials": N_TRIALS // 2}

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
    total_subjects = len(grid) * N_SUBJECTS_PER_GROUP
    print(f"Generating {len(grid)} groups × {N_SUBJECTS_PER_GROUP} subjects = {total_subjects} files")
    print(f"Output directory: {output_dir}")
    print(f"Results directory: {results_dir}\n")

    all_rows_list = []
    all_results = []
    subj_counter = 0

    np.random.seed(42)  # For reproducibility

    for group_idx, (pse_center, jnd_center) in enumerate(grid, 1):
        print(f"Group {group_idx}/{len(grid)}: PSE_center={pse_center}, JND_center={jnd_center}")
        
        for subj_in_group in range(1, N_SUBJECTS_PER_GROUP + 1):
            subj_counter += 1
            
            # Apply random variation ±2.5
            pse = pse_center + np.random.uniform(-VARIATION, VARIATION)
            jnd = jnd_center + np.random.uniform(-VARIATION, VARIATION)
            
            # Ensure JND stays positive
            jnd = max(jnd, 1.0)

            try:
                rows, result_dict = engine.simulate_subject(
                    pse=pse,
                    jnd=jnd,
                    ntrials=N_TRIALS,
                    ado_params=ADO_PARAMS,
                    bis_params=BIS_PARAMS_HALF,
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
                    filename = f"{MODEL_NAME}_G{group_idx}_S{subj_in_group:02d}.txt"
                    out_path = output_dir / filename
                    engine.save_gbf_file(gbf_rows, str(out_path))

                # Collect for plots and Excel
                all_rows_list.append(rows)
                all_results.append(result_dict)

            except Exception as e:
                print(f"  Error subject {subj_counter}: {e}")

        print(f"  Group {group_idx} done: {N_SUBJECTS_PER_GROUP} subjects")

    print(f"\nDone. {len(all_results)} subjects simulated")
    
    # Generate group plots
    print(f"\nGenerating plots...")
    if all_rows_list:
        plot_group_histograms(all_rows_list, str(results_dir), MODEL_NAME, OFFSET)
        plot_group_model_histograms(all_rows_list, str(results_dir), MODEL_NAME, OFFSET)
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
