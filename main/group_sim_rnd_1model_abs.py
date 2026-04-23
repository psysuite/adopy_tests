#!/usr/bin/env python3
"""
Simulate temporal bisection experiment with 20 subjects, 60 trials each.
Uses 1-model absolute ADOpy approach with random PSE/JND values.

PSE: random uniform [485, 515]
JND: random uniform [20, 60]
Output:
  - GBF files: data/output/sim_rnd/1model_abs/
  - Plots: data/output/sim_rnd/1model_abs/results/
  - Excel: data/output/sim_rnd/1model_abs/results/results_1model_abs.xlsx
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.simulation_engine import SimulationEngine
from utilities.plotting import plot_group_histograms, plot_group_model_histograms, plot_group_psychometric
from utilities.psychometric_analysis import consolidate_results, add_group_stats_to_excel

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME  = "1model_abs"
OUTPUT_DIR  = "../data/output/sim_rnd/1model_abs"
RESULTS_DIR = "../data/output/sim_rnd/1model_abs/results"
N_SUBJECTS  = 20
N_TRIALS    = 200
OFFSET      = 500

USE_FIXED_TRIALS  = True
N_FIXED           = 10
FIXED_LATENCIES   = [275, 725, 325, 675, 375, 625, 425, 575, 475, 525]

ADO_PARAMS = {"guess_rate": 0.04, "lapse_rate": 0.04, "noise_perc": 0.05}
BIS_PARAMS = {"min": 200, "max": 800, "offset": OFFSET, "ntrials": N_TRIALS}

FIXED_TRIALS_CONFIG = {
    "use": USE_FIXED_TRIALS,
    "n_fixed": N_FIXED,
    "latencies": FIXED_LATENCIES,
}

# ============================================================================


def main():
    engine = SimulationEngine(MODEL_NAME, offset=OFFSET)
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Simulating {N_SUBJECTS} subjects with {N_TRIALS} trials each...")
    print(f"PSE range: [485, 515], JND range: [20, 60]")
    print(f"Output directory: {output_dir}\n")

    all_rows_list = []
    all_results = []

    # Generate random PSE/JND for each subject
    np.random.seed(42)  # For reproducibility

    for subj_id in range(1, N_SUBJECTS + 1):
        # Random PSE and JND
        pse = np.random.uniform(485, 515)
        jnd = np.random.uniform(20, 60)

        print(f"Subject {subj_id}: PSE={pse:.1f}, JND={jnd:.1f}")

        try:
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

            filename = f"{MODEL_NAME}_S{subj_id:03d}.txt"
            out_path = output_dir / filename
            engine.save_gbf_file(gbf_rows, str(out_path))

            # Collect for plots and Excel
            all_rows_list.append(rows)
            all_results.append(result_dict)

        except Exception as e:
            print(f"Error simulating subject {subj_id}: {e}")

    print(f"\nDone. {len(all_results)} subjects simulated")

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
