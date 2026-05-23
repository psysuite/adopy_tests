#!/usr/bin/env python3
"""
Generate psychometric curve plots for each group and assemble into 3x3 grid.

For each group (PSE x JND cell), produces:
  - Group-level psychometric curve fitted to aggregated data from all subjects
  - 3x3 grid of all groups per model

Output per model:
  data/output/sim_gridrnd/{model}/group_{pse}_{jnd}/results/
      {model}_G{idx}_group_psychometric.png     ← group plot

  data/output/sim_gridrnd/{model}/
      {model}_grid_psychometric.png             ← 3x3 grid

GBF filename format: S{subj_in_group:02d}_G{group_idx}_{pse_int}_{jnd_int}_{model_name}.txt
"""

import os
import sys
from pathlib import Path
from itertools import product

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.core.plotting import plot_group_psychometric, load_group_rows, fit_psychometric_curve, plot_generic_grid

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS   = ["ABS1", "REL1", "REL2"]
PSE_GRID = [480, 500, 520]
JND_GRID = [20, 40, 60]
OFFSET   = 500

# ============================================================================
# GRID PLOTTING
# ============================================================================

def plot_psychometric_into_axes(ax, data_entry, pse, jnd, offset=OFFSET):
    """Plot psychometric curve into provided axes (for grid assembly)."""
    rows = data_entry['rows']
    
    if not rows:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    
    stimuli = [row['lat'] for row in rows]
    responses = [row['user_ans'] for row in rows]
    
    result = fit_psychometric_curve(stimuli, responses, offset=offset)
    if result[0] is None:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        return
    
    mu, sigma, x_fit, y_fit, bins_valid, f = result
    jnd_est = sigma * 0.6745
    
    ax.plot(x_fit, y_fit, 'b-', linewidth=2, alpha=0.7, label=f'Fit: μ={mu:.1f}, JND={jnd_est:.1f}')
    ax.plot(bins_valid, f, 'ro', markersize=8, alpha=0.7, label='Data')
    ax.axvline(offset, color='g', linestyle='--', linewidth=2, alpha=0.7, label=f'True ({offset}ms)')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Stimulus latency (ms)', fontsize=10)
    ax.set_ylabel('P(response = 1)', fontsize=10)
    ax.set_title(f'PSE={pse}, JND={jnd}', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

# ============================================================================
# MAIN PER MODEL
# ============================================================================

def plot_psychometric_for_model(model_name: str) -> bool:
    script_dir = Path(__file__).parent.parent.parent
    output_dir = script_dir / "data" / "output" / "sim_gridrnd" / model_name

    if not output_dir.exists():
        print(f"  Output directory not found: {output_dir}")
        return False

    grid = list(product(PSE_GRID, JND_GRID))
    group_data = {}

    print(f"  Step 1: loading data and generating group plots...")

    for group_idx, (pse, jnd) in enumerate(grid, 1):
        group_dir = output_dir / f"group_{pse}_{jnd}"
        results_dir = group_dir / "results"

        if not results_dir.exists():
            print(f"    Skipping group {pse}_{jnd}: results directory not found")
            continue

        # Count subjects from Excel
        excel_files = list(results_dir.glob(
            f"{model_name}_G{group_idx}_results_summary.xlsx"))
        if not excel_files:
            print(f"    Skipping group {pse}_{jnd}: no Excel file found")
            continue

        try:
            df = pd.read_excel(excel_files[0])
            subj_mask = ~df['subj'].astype(str).str.startswith('GROUP_')
            n_subjects = subj_mask.sum()
        except Exception as e:
            print(f"    Error reading Excel for group {pse}_{jnd}: {e}")
            continue

        # Load all rows
        rows = load_group_rows(group_dir, model_name, group_idx, pse, jnd, n_subjects, OFFSET)

        if len(rows) == 0:
            print(f"    Warning: no data loaded for group {pse}_{jnd}")
            continue

        file_prefix = f"{model_name}_G{group_idx}"

        # Standalone group plot
        plot_group_psychometric(
            [rows],  # Wrap in list for compatibility with function signature
            str(results_dir),
            file_prefix,
            OFFSET,
            group_label=f"PSE={pse}, JND={jnd}"
        )

        group_data[(pse, jnd)] = {
            'rows': rows,
        }

    print(f"    ✓ {len(group_data)} group plots generated")

    if not group_data:
        print(f"  No data available for grid plot")
        return False

    # Step 2: 3x3 grid
    print(f"  Step 2: generating 3x3 grid...")
    plot_generic_grid(
        model_name=model_name,
        group_data=group_data,
        pse_grid=PSE_GRID,
        jnd_grid=JND_GRID,
        output_dir=output_dir,
        plot_func=plot_psychometric_into_axes,
        plot_func_kwargs={'offset': OFFSET},
        grid_filename_suffix='psychometric'
    )

    return True

