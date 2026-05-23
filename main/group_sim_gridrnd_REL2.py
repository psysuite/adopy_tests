#!/usr/bin/env python3
"""
Generate simulated temporal bisection files on a PSE/JND grid with random variation.
Uses REL2 (2-model relative) ADOpy approach (pre + post).

PSE grid: [480, 500, 520]
JND grid: [20, 40, 60]
→ 9 groups × 20 subjects each = 180 files × 200 trials each
Each subject has PSE/JND ± 2.5 variation from group center, rounded to int.

GBF filename format: SXX_GZ_PSE_JND_REL2.txt
  - XX: subject number within group (zero-padded)
  - Z:  group index
  - PSE/JND: jittered values rounded to int (used for simulation)

Output:
  - GBF files: data/output/sim_gridrnd/REL2/
  - Plots:     data/output/sim_gridrnd/REL2/group_PSE_JND/results/
  - Excel:     data/output/sim_gridrnd/REL2/group_PSE_JND/results/
  - Grid/metric plots: data/output/sim_gridrnd/REL2/

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

from analysis.core.simulation_engine import SimulationEngine
from analysis.core.plotting import plot_group_histograms, plot_group_psychometric, create_grid_plots_from_groups
from analysis.core.psychometric_analysis import consolidate_results, add_group_stats_to_excel
from analysis.core.generate_analysis_plots import generate_all_analysis
from analysis.core.generate_analysis_data import generate_all_data
from analysis.core.plot_psychometric_curves import plot_psychometric_for_model
from analysis.core.plot_stimulus_distribution import plot_stimulus_for_model
from utilities.multithreading_utils import (
    SubjectSimulationTask,
    MultiThreadedSimulationRunner,
    backfill_skip_mode,
    parse_gbf_filename,
    load_gbf_files_for_group,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME              = "REL2"

MAX_WORKERS             = 4
USE_FIXED_TRIALS        = False
SAVE_GBF_FILES          = True
SKIP_PHASES_1_2         = False

OUTPUT_DIR              = str(Path(__file__).parent.parent / "data" / "output" / "sim_gridrnd" / MODEL_NAME)
N_SUBJECTS_PER_GROUP    = 20
N_TRIALS                = 200
OFFSET                  = 500

PSE_GRID                = [480, 500, 520]
JND_GRID                = [20, 40, 60]
VARIATION               = 2.5
N_FIXED                 = 10
FIXED_OFFSETS_REL       = [225, 175, 125, 75, 25]

FIXED_TRIALS_CONFIG = {
    "use": USE_FIXED_TRIALS,
    "n_fixed": N_FIXED,
    "offsets": FIXED_OFFSETS_REL,
}

ADO_PARAMS              = {"guess_rate": 0.5, "lapse_rate": 0.04, "noise_perc": 0.1}
BIS_PARAMS              = {"min": 1, "max": 300, "offset": OFFSET, "ntrials": N_TRIALS // 2}

# ============================================================================


def main():
    engine = SimulationEngine(MODEL_NAME, offset=OFFSET)
    runner = MultiThreadedSimulationRunner(max_workers=MAX_WORKERS)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = list(product(PSE_GRID, JND_GRID))
    total_subjects = len(grid) * N_SUBJECTS_PER_GROUP
    print(f"Generating {len(grid)} groups × {N_SUBJECTS_PER_GROUP} subjects = {total_subjects} files")
    print(f"Output directory: {output_dir}\n")

    np.random.seed(42)

    histogram_plots = {}
    psychometric_plots = {}

    for group_idx, (pse_center, jnd_center) in enumerate(grid, 1):
        print(f"Group {group_idx}/{len(grid)}: PSE_center={pse_center}, JND_center={jnd_center}")

        group_dir = output_dir / f"group_{pse_center}_{jnd_center}"
        group_dir.mkdir(parents=True, exist_ok=True)
        group_results_dir = group_dir / "results"
        group_results_dir.mkdir(parents=True, exist_ok=True)

        if not SKIP_PHASES_1_2:
            # ====== PHASE 1: Create simulation tasks ======
            print(f"  Phase 1: Creating {N_SUBJECTS_PER_GROUP} simulation tasks...")
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
                    ado_params=ADO_PARAMS,
                    bis_params=BIS_PARAMS,
                    fixed_trials_config=FIXED_TRIALS_CONFIG,
                    group_dir=group_dir,
                    model_name=MODEL_NAME,
                    save_gbf=SAVE_GBF_FILES,
                )
                tasks.append(task)

            # ====== PHASE 2: Run simulations in parallel ======
            print(f"  Phase 2: Running {N_SUBJECTS_PER_GROUP} simulations in parallel...")
            completed_tasks = runner.run_subject_simulations(tasks, verbose=True)
            analysis_tasks = []
        else:
            print(f"  Phases 1-2: SKIPPED (SKIP_PHASES_1_2=True)")
            print(f"  Loading GBF files from {group_dir}...")
            analysis_tasks = load_gbf_files_for_group(group_dir, MODEL_NAME)
            print(f"  Loaded {len(analysis_tasks)} subjects from GBF files")
            completed_tasks = []

        # ====== PHASE 3: Collect results and run progressive analyses ======
        print(f"  Phase 3: Running progressive analyses in parallel...")
        group_rows_list = []
        group_results = []
        analysis_tasks_to_run = []
        task_to_result_idx = {}

        if not SKIP_PHASES_1_2:
            for task in completed_tasks:
                if task.error:
                    print(f"    Skipping subject {task.subject_id} due to error: {task.error}")
                    continue
                group_rows_list.append(task.rows)
                group_results.append(task.result_dict)
                task_to_result_idx[task.result_dict['subj']] = len(group_results) - 1
                analysis_tasks_to_run.append((task.gbf_rows, task.result_dict['subj'], task.result_dict, task.rows, OFFSET))
        else:
            for gbf_rows, subj, result_dict, rows, offset in analysis_tasks:
                group_rows_list.append(rows)
                group_results.append(result_dict)
                task_to_result_idx[subj] = len(group_results) - 1
                analysis_tasks_to_run.append((gbf_rows, subj, result_dict, rows, offset))

        if analysis_tasks_to_run:
            analysis_results = runner.run_progressive_analyses(analysis_tasks_to_run, verbose=True, gamma=ADO_PARAMS["guess_rate"], lapse=ADO_PARAMS["lapse_rate"])

            for subj, (updated_dict, error_msg) in analysis_results.items():
                if error_msg:
                    print(f"    Warning: Progressive analysis failed for {subj}: {error_msg}")
                else:
                    result_idx = task_to_result_idx.get(subj)
                    if result_idx is not None:
                        group_results[result_idx] = updated_dict

            if SKIP_PHASES_1_2:
                backfill_skip_mode(group_results, analysis_tasks, N_TRIALS)

        # ====== PHASE 4: Generate plots and Excel (sequential) ======
        print(f"  Phase 4: Generating plots and Excel...")
        group_label = f"PSE={pse_center}, JND={jnd_center}"
        if group_rows_list:
            plot_group_histograms(group_rows_list, str(group_results_dir), f"{MODEL_NAME}_G{group_idx}", OFFSET, group_label)
            histogram_plots[group_idx] = str(group_results_dir / f"{MODEL_NAME}_G{group_idx}_group_histogram.png")

            plot_group_psychometric(group_rows_list, str(group_results_dir), f"{MODEL_NAME}_G{group_idx}", OFFSET, group_label)
            psychometric_plots[group_idx] = str(group_results_dir / f"{MODEL_NAME}_G{group_idx}_group_psychometric.png")

        if group_results:
            try:
                excel_filepath = consolidate_results(group_results, str(group_results_dir), f"{MODEL_NAME}_G{group_idx}")
                add_group_stats_to_excel(excel_filepath)
            except Exception as e:
                print(f"  Warning: Could not generate Excel for group {group_idx}: {e}")

        print(f"  Group {group_idx} done: {len(group_results)} subjects processed\n")

    print(f"Done. {total_subjects} subjects across {len(grid)} groups")

    # ====== PHASE 5: Generate analysis data (Excel columns) ======
    print(f"\nPhase 5: Generating analysis data (Excel columns)...")
    generate_all_data(
        model_name=MODEL_NAME,
        output_dir=output_dir,
        pse_grid=PSE_GRID,
        jnd_grid=JND_GRID,
        offset=OFFSET
    )

    # ====== PHASE 6: Create grid plots ======
    print(f"\nPhase 6: Creating grid plots...")
    if histogram_plots:
        create_grid_plots_from_groups(histogram_plots, str(output_dir), MODEL_NAME, PSE_GRID, JND_GRID, 'histogram')
    if psychometric_plots:
        create_grid_plots_from_groups(psychometric_plots, str(output_dir), MODEL_NAME, PSE_GRID, JND_GRID, 'psychometric')

    # ====== PHASE 7: Generate group plots (psychometric + stimulus distribution) ======
    print(f"\nPhase 7: Generating group plots...")
    plot_psychometric_for_model(MODEL_NAME)
    plot_stimulus_for_model(MODEL_NAME)

    # ====== PHASE 8: Generate analysis metric plots and export CSV ======
    print(f"\nPhase 8: Generating analysis metric plots and exporting CSV...")
    generate_all_analysis(
        model_name=MODEL_NAME,
        output_dir=output_dir,
        pse_grid=PSE_GRID,
        jnd_grid=JND_GRID,
        export_csv=True,
        csv_output_dir=Path(__file__).parent.parent / "data" / "output" / "stimulus_metrics_for_r",
        data_root=Path(__file__).parent.parent / "data" / "output" / "sim_gridrnd"
    )

    print(f"All done!")


if __name__ == "__main__":
    main()
