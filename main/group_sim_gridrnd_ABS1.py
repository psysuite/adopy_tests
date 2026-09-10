#!/usr/bin/env python3
"""
Generate simulated temporal bisection files on a PSE/JND grid with random variation and psychiometrics and stimuli distribution corresponding plots.
Uses ABS1 (1-model absolute) ADOpy approach.

PSE grid: [480, 500, 520]
JND grid: [20, 40, 60]
→ 9 groups × 20 subjects each = 180 files × 200 trials each
Each subject has PSE/JND ± 2.5 variation from group center, rounded to int.

GBF filename format: SXX_GZ_PSE_JND_ABS1.txt
  - XX: subject number within group (zero-padded)
  - Z:  group index
  - PSE/JND: jittered values rounded to int (used for simulation)

PHASES:
  Phase 1: DATA GENERATION (this script)
    - Phase 1a: Create simulation tasks
    - Phase 1b: Run subject simulations in parallel
    - Output: GBF files

  Phase 2: PLOTS CREATION (this script)
    - Output: plots files (xxx_yy_group_psychometric.png, xxx_yy_stimulus_distribution.png)

  Phase 3-4: CONSOLIDATION (separate script: main/generate_synthetic_data.py)
    - Phase 3: Calculate metrics (B1-B5 + posteriors Level B) and generate group plots
    - Phase 4: Generate grid plots and consolidated Excel files
    - Output: Plots + synthetic_data_wide.xlsx + synthetic_data_long.xlsx in R/indata/

MULTITHREADING:
  - Subject simulation runs in parallel (one thread per subject)
  - Progressive analysis runs in parallel after all simulations complete
  - Plots and Excel generation run sequentially after all analyses complete
"""

import os
import sys
from pathlib import Path
from itertools import product

import matplotlib

matplotlib.use('Agg')

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.config_synthetic import *
from analysis.core.simulation_engine import SimulationEngine
from analysis.plot.plotting import plot_group_histograms, plot_group_psychometric
from utilities.multithreading_utils import (
    SubjectSimulationTask,
    MultiThreadedSimulationRunner,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_NAME = "ABS1"
OUTPUT_DIR = str(Path(__file__).parent.parent / "data" / "output" / "sim_gridrnd" / MODEL_NAME)

FIXED_TRIALS_CONFIG = {
    "use": USE_FIXED_TRIALS,
    "n_fixed": N_FIXED,
    "offsets": FIXED_LATENCIES_ABS,
}


# ============================================================================

def main():
    """
    Generate synthetic data and groups' plot: Phase 1-2.

    Phase 1: Create & run simulation tasks in parallel
    Phase 2: Create groups' plots

    Output: GBF files and groups' plots saved locally

    Returns:
        List of (group_idx, pse_center, jnd_center, completed_tasks) tuples
    """
    engine = SimulationEngine(MODEL_NAME, offset=OFFSET)
    runner = MultiThreadedSimulationRunner(max_workers=MAX_WORKERS)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = list(product(PSE_GRID, JND_GRID))
    total_subjects = len(grid) * N_SUBJECTS_PER_GROUP
    print(f"Generating {len(grid)} groups × {N_SUBJECTS_PER_GROUP} subjects = {total_subjects} files")
    print(f"Output directory: {output_dir}\n")

    np.random.seed(42)

    all_group_results = []

    for group_idx, (pse_center, jnd_center) in enumerate(grid, 1):
        print(f"Group {group_idx}/{len(grid)}: PSE_center={pse_center}, JND_center={jnd_center}")

        group_dir = output_dir / f"group_{pse_center}_{jnd_center}"
        group_dir.mkdir(parents=True, exist_ok=True)
        group_results_dir = group_dir / "results"
        group_results_dir.mkdir(parents=True, exist_ok=True)

        # ====== PHASE 1a: Create simulation tasks ======
        print(f"  Phase 1a: Creating {N_SUBJECTS_PER_GROUP} simulation tasks...")
        tasks = []
        subj_counter = (group_idx - 1) * N_SUBJECTS_PER_GROUP

        for subj_in_group in range(1, N_SUBJECTS_PER_GROUP + 1):
            subj_counter += 1
            pse = int(round(pse_center + np.random.uniform(-VARIATION, VARIATION)))
            jnd = max(1, int(round(jnd_center + np.random.uniform(-VARIATION, VARIATION))))

            task = SubjectSimulationTask(
                subject_id=subj_counter,
                group_idx=group_idx,
                subj_in_group=subj_in_group,
                pse=float(pse),
                jnd=float(jnd),
                engine=engine,
                n_trials=N_TRIALS,
                ado_params=ADO_PARAMS_ABS,
                bis_params=BIS_PARAMS_ABS,
                fixed_trials_config=FIXED_TRIALS_CONFIG,
                group_dir=group_dir,
                model_name=MODEL_NAME,
                save_gbf=SAVE_GBF_FILES,
            )
            tasks.append(task)

        # ====== PHASE 1b: Run simulations in parallel ======
        print(f"  Phase 1b: Running {N_SUBJECTS_PER_GROUP} simulations in parallel...")
        completed_tasks = runner.run_subject_simulations(tasks, verbose=True)

        # ====== PHASE 2: Generate local plots (sequential) ======
        print(f"  Phase 2: Generating local plots...")
        group_label = f"PSE={pse_center}, JND={jnd_center}"
        group_rows_list = []

        for task in completed_tasks:
            if task.error:
                print(f"    Skipping subject {task.subject_id} due to error: {task.error}")
                continue
            group_rows_list.append(task.rows)

        if group_rows_list:
            plot_group_histograms(group_rows_list, str(group_results_dir), f"{MODEL_NAME}_G{group_idx}", OFFSET,
                                  group_label)
            plot_group_psychometric(group_rows_list, str(group_results_dir), f"{MODEL_NAME}_G{group_idx}", OFFSET,
                                    group_label)

        print(f"  Group {group_idx} done: {len(completed_tasks)} subjects processed\n")

        # Accumulate results for later consolidation
        all_group_results.append((group_idx, pse_center, jnd_center, completed_tasks))

    print(f"Done. {total_subjects} subjects across {len(grid)} groups")
    return all_group_results


if __name__ == "__main__":
    all_group_results = main()
    print(f"\n{'=' * 70}")
    print("Phase 1-2 (Data Generation + group plots) COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nNext step: Run generate_synthetic_data.py for Phases 3-4 (Consolidation)")
    print(f"  python main/generate_synthetic_data.py\n")
