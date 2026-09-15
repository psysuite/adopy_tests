#!/usr/bin/env python3
"""
Generate simulated temporal bisection files on a PSE/JND grid with random variation.
Uses ABS1 (1-model absolute) ADOpy approach.

PSE grid: [480, 500, 520]
JND grid: [20, 40, 60]
→ 9 groups × 20 subjects each = 180 files × 200 trials each
Each subject has PSE/JND ± 2.5 variation from group center, rounded to int.

GBF filename format: SXX_GZ_PSE_JND_ABS1.txt
  - XX: subject number within group (zero-padded)
  - Z:  group index
  - PSE/JND: jittered values rounded to int (used for simulation)

PHASE A (this script):
  - Phase A1: Create simulation tasks
  - Phase A2: Run subject simulations in parallel
  - Output: GBF files

PHASE B-C (separate script: main/generate_synthetic_data.py):
  - Phase B: Calculate metrics (B1-B5 + posteriors Level B) → Excel
  - Phase C: Generate all plots (27 per-model + 3 consolidated)
  - Output: synthetic_data_wide.xlsx + synthetic_data_long.xlsx in R/indata/ + 30 PNG plots

MULTITHREADING:
  - Subject simulation runs in parallel (one thread per subject)
"""

import os
import sys
from pathlib import Path
from itertools import product

import matplotlib

matplotlib.use('Agg')

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.config import *
from analysis.core.simulation_engine import SimulationEngine
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

def main(overwrite: bool = True) -> list:
    """
    Generate synthetic data: Phase A (data generation only).

    Phase A1: Create & run simulation tasks in parallel
    (Phase C handles all plot generation)

    If overwrite=False and GBF files already exist, skip generation,
    just return all_group_results loaded from existing files.

    Output: GBF files saved locally

    Args:
        overwrite: If False, skip generation if files exist (default True)

    Returns:
        List of (group_idx, pse_center, jnd_center, completed_tasks) tuples
    """
    output_dir = Path(OUTPUT_DIR)
    grid = list(product(PSE_GRID, JND_GRID))
    total_subjects = len(grid) * N_SUBJECTS_PER_GROUP

    # Check if all GBF files already exist
    all_exist = True
    for group_idx, (pse_center, jnd_center) in enumerate(grid, 1):
        group_dir = output_dir / f"group_{pse_center}_{jnd_center}"
        expected_gbf_count = N_SUBJECTS_PER_GROUP
        existing_gbf = list(group_dir.glob("S*.txt")) if group_dir.exists() else []
        if len(existing_gbf) < expected_gbf_count:
            all_exist = False
            break

    # If overwrite=False and all files exist, load them without regenerating
    if not overwrite and all_exist:
        print(f"\n[SKIP GENERATION] overwrite=False and all {total_subjects} GBF files exist")
        print(f"  Loading existing GBF files from {output_dir}...\n")
        
        all_group_results = []
        for group_idx, (pse_center, jnd_center) in enumerate(grid, 1):
            group_dir = output_dir / f"group_{pse_center}_{jnd_center}"
            
            # Read GBF files to reconstruct tasks (minimal, just for return structure)
            gbf_files = sorted(group_dir.glob("S*.txt"))
            mock_tasks = []
            for gbf_file in gbf_files:
                # Create a minimal task-like object with just rows and subject_id
                class MockTask:
                    def __init__(self, subject_id, rows):
                        self.subject_id = subject_id
                        self.rows = rows
                        self.error = None
                
                # Read GBF to get subject metadata
                try:
                    with open(gbf_file) as f:
                        lines = f.readlines()
                    # Extract subject_id from filename (S##_G#_PSE_JND_MODEL.txt)
                    subj_id = gbf_file.stem.split('_')[0]  # S01, S02, etc.
                    # Create minimal rows list (header + 1 data row for metadata)
                    rows = [line.strip() for line in lines[:2]]  # Just header + first row
                    mock_tasks.append(MockTask(subj_id, rows))
                except Exception as e:
                    print(f"  ⚠ Error reading {gbf_file.name}: {e}")
            
            all_group_results.append((group_idx, pse_center, jnd_center, mock_tasks))
            print(f"  Group {group_idx}: loaded {len(mock_tasks)} GBF files")
        
        print(f"\n✓ Loaded {total_subjects} subjects across {len(grid)} groups (no plots regenerated)")
        return all_group_results

    # ====== NORMAL PATH: Generate from scratch ======
    print(f"\nGenerating {len(grid)} groups × {N_SUBJECTS_PER_GROUP} subjects = {total_subjects} files")
    print(f"Output directory: {output_dir}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    engine = SimulationEngine(MODEL_NAME, offset=OFFSET)
    runner = MultiThreadedSimulationRunner(max_workers=MAX_WORKERS)

    all_group_results = []

    for group_idx, (pse_center, jnd_center) in enumerate(grid, 1):
        print(f"Group {group_idx}/{len(grid)}: PSE_center={pse_center}, JND_center={jnd_center}")

        group_dir = output_dir / f"group_{pse_center}_{jnd_center}"
        group_dir.mkdir(parents=True, exist_ok=True)

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

        print(f"  Group {group_idx} done: {len(completed_tasks)} subjects processed\n")

        # Accumulate results for later consolidation
        all_group_results.append((group_idx, pse_center, jnd_center, completed_tasks))

    print(f"Done. {total_subjects} subjects across {len(grid)} groups")
    return all_group_results


if __name__ == "__main__":
    all_group_results = main()
    print(f"\n{'=' * 70}")
    print("Phase A (Data Generation) COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nNext step: Run generate_synthetic_data.py for Phases B-C (Metrics + Plots)")
    print(f"  python main/generate_synthetic_data.py\n")
