#!/usr/bin/env python3
"""
Generate stimulus distribution plots with envelope and stimulus_spread overlay.

For each group (PSE x JND cell), produces a group-level plot with:
  - Histogram of aggregated stimulus latencies (all subjects in group)
  - KDE envelope (smooth curve over histogram)
  - Two vertical dashed lines at stimulus_center ± stimulus_spread
  - Annotations showing stimulus_center, stimulus_spread, and bimodality_index

Output per model:
  data/output/sim_gridrnd/{model}/group_{pse}_{jnd}/results/
      {model}_G{idx}_stimulus_distribution.png     ← group plot

  data/output/sim_gridrnd/{model}/
      {model}_stimulus_distribution_grid.png       ← 3x3 grid

GBF filename format: S{subj_in_group:02d}_G{group_idx}_{pse_int}_{jnd_int}_{model_name}.txt

The envelope shows the empirical distribution of stimuli presented by the algorithm.
The dashed lines show where the algorithm is concentrating its exploration.
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
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.core.plotting import load_group_rows

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS   = ["ABS1", "REL1", "REL2"]
PSE_GRID = [480, 500, 520]
JND_GRID = [20, 40, 60]
OFFSET   = 500

# Fixed axis ranges for all plots (for consistent grid assembly)
X_MIN = 250
X_MAX = 750
Y_MIN = 0
Y_MAX = 700

# ============================================================================
# SINGLE GROUP PLOT
# ============================================================================

def plot_group_stimulus_distribution(
    model_name: str,
    pse: int, jnd: int,
    stimulus_center: float,
    stimulus_spread: float,
    all_latencies: np.ndarray,
    all_responses: np.ndarray,
    results_dir: Path,
    file_prefix: str,
    ax=None,
    dpi=300,
) -> Path | None:
    """
    Plot stimulus distribution for one group.

    x-axis: stimulus latency (ms)
    - Histogram: success (green) and failure (red) stimuli
    - KDE envelope: smooth curve over combined histogram
    - Dashed lines: stimulus_center ± stimulus_spread
    - Annotations: stimulus_spread and JND

    If ax is provided, draws into that axes (for grid assembly).
    Otherwise creates a standalone figure and saves it.
    """
    standalone = ax is None
    if standalone:
        # 5cm x 5cm at 300 DPI = 1.97" x 1.97", +33% width = 2.62", +25% more = 3.275" x 1.97"
        fig, ax = plt.subplots(figsize=(3.275, 1.97), dpi=dpi)

    if len(all_latencies) == 0:
        return None

    # Separate success and failure stimuli
    stimuli = all_latencies
    successes = all_responses
    success_stim = [s for s, succ in zip(stimuli, successes) if succ]
    failure_stim = [s for s, succ in zip(stimuli, successes) if not succ]

    # Create bins of 10ms (same as original plot_group_histograms)
    min_stim = min(stimuli)
    max_stim = max(stimuli)
    bins = np.arange(int(min_stim / 10) * 10, int(max_stim / 10) * 10 + 20, 10)

    # Plot histograms (exact same colors and style as original)
    ax.hist(success_stim, bins=bins, color='green', alpha=0.7, label='Success', edgecolor='black')
    ax.hist(failure_stim, bins=bins, color='red', alpha=0.7, label='Failure', edgecolor='black')

    # KDE envelope
    if len(all_latencies) > 5:
        try:
            kde = gaussian_kde(all_latencies, bw_method='scott')
            x_kde = np.linspace(min_stim - 20, max_stim + 20, 300)
            kde_vals = kde(x_kde)
            
            # Normalize KDE to match histogram density
            # Scale by bin width and total count to match histogram scale
            bin_width = bins[1] - bins[0]
            kde_vals_scaled = kde_vals * len(all_latencies) * bin_width
            
            ax.plot(x_kde, kde_vals_scaled, color='steelblue', linewidth=2.0,
                    alpha=0.7, zorder=5)
        except Exception:
            pass

    # Stimulus spread lines
    ax.axvline(stimulus_center, color='darkblue', linestyle='--', 
              linewidth=1.5, alpha=0.8, zorder=4)
    ax.axvline(stimulus_center - stimulus_spread, color='purple', 
              linestyle='--', linewidth=1.5, alpha=0.8, zorder=4)
    ax.axvline(stimulus_center + stimulus_spread, color='purple', 
              linestyle='--', linewidth=1.5, alpha=0.8, zorder=4)

    # Annotations
    ax.text(0.98, 0.98,
            f'Spread: {stimulus_spread:.1f} ms\n'
            f'JND: {jnd} ms',
            transform=ax.transAxes, fontsize=6,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, pad=0.3))

    ax.set_xlabel('')
    ax.set_ylabel('')
    if ax is not None:  # Only show title when part of grid
        ax.set_title(f'PSE={pse}, JND={jnd}', fontsize=5, fontweight='bold', pad=2)
    ax.legend(fontsize=4, loc='upper left', frameon=False)
    ax.grid(True, alpha=0.3)
    
    # Fixed axis ranges and ticks for consistent grid assembly
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks([300, 400, 500, 600, 700])
    ax.set_yticks([0, 200, 400, 600])
    ax.tick_params(labelsize=4)

    if standalone:
        plt.tight_layout()
        out_path = results_dir / f"{file_prefix}_stimulus_distribution.png"
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return out_path

    return None


# ============================================================================
# MAIN PER MODEL
# ============================================================================

def plot_stimulus_for_model(model_name: str) -> bool:
    script_dir = Path(__file__).parent.parent.parent
    output_dir = script_dir / "data" / "output" / "sim_gridrnd" / model_name

    if not output_dir.exists():
        print(f"  Output directory not found: {output_dir}")
        return False

    grid       = list(product(PSE_GRID, JND_GRID))
    group_data = {}

    print(f"  Step 1: loading data and generating group plots...")

    for group_idx, (pse, jnd) in enumerate(grid, 1):
        group_dir   = output_dir / f"group_{pse}_{jnd}"
        results_dir = group_dir / "results"

        if not results_dir.exists():
            print(f"    Skipping group {pse}_{jnd}: results directory not found")
            continue

        excel_files = list(results_dir.glob(
            f"{model_name}_G{group_idx}_results_summary.xlsx"))
        if not excel_files:
            print(f"    Skipping group {pse}_{jnd}: no Excel file found")
            continue

        try:
            df = pd.read_excel(excel_files[0])

            # Count real subject rows
            subj_mask  = ~df['subj'].astype(str).str.startswith('GROUP_')
            n_subjects = subj_mask.sum()

            # Get stimulus metrics from Excel (should be present from previous analysis)
            stimulus_center_col = None
            stimulus_spread_col = None
            
            # Find the columns for the last trial block (200 trials)
            for col in df.columns:
                if 'stimulus_center_200' in col:
                    stimulus_center_col = col
                elif 'stimulus_spread_200' in col:
                    stimulus_spread_col = col
            
            if stimulus_center_col is None or stimulus_spread_col is None:
                print(f"    Warning: stimulus metrics not found in Excel for group {pse}_{jnd}")
                continue

            # Group-mean values
            stimulus_center = float(df.loc[subj_mask, stimulus_center_col].mean())
            stimulus_spread = float(df.loc[subj_mask, stimulus_spread_col].mean())

            if not np.isfinite(stimulus_center) or not np.isfinite(stimulus_spread) or stimulus_spread <= 0:
                print(f"    Warning: invalid stimulus metrics for group {pse}_{jnd}, skipping")
                continue

        except Exception as e:
            print(f"    Error reading Excel for group {pse}_{jnd}: {e}")
            continue

        # Load all latencies from GBF files
        rows = load_group_rows(group_dir, model_name, group_idx, pse, jnd, n_subjects, OFFSET)
        
        if len(rows) == 0:
            print(f"    Warning: no latencies loaded for group {pse}_{jnd}")
            continue
        
        all_latencies = np.array([row['lat'] for row in rows], dtype=float)
        all_responses = np.array([row['res'] == 'true' for row in rows], dtype=bool)

        file_prefix = f"{model_name}_G{group_idx}"

        # Standalone group plot
        plot_group_stimulus_distribution(
            model_name=model_name,
            pse=pse, jnd=jnd,
            stimulus_center=stimulus_center,
            stimulus_spread=stimulus_spread,
            all_latencies=all_latencies,
            all_responses=all_responses,
            results_dir=results_dir,
            file_prefix=file_prefix,
        )

        group_data[(pse, jnd)] = {
            'model_name': model_name,
            'stimulus_center':  stimulus_center,
            'stimulus_spread':  stimulus_spread,
            'all_latencies':    all_latencies,
            'all_responses':    all_responses,
            'results_dir':      results_dir,
            'file_prefix':      file_prefix,
        }

    print(f"    ✓ {len(group_data)} group plots generated")

    if not group_data:
        print(f"  No data available for grid plot")
        return False

    # Step 2: 3x3 grid (using custom compact layout)
    print(f"  Step 2: generating 3x3 grid...")
    plot_stimulus_grid_compact(model_name, group_data, PSE_GRID, JND_GRID, output_dir)

    return True


def plot_stimulus_grid_compact(model_name: str, group_data: dict,
                               pse_grid: list, jnd_grid: list,
                               output_dir: Path) -> None:
    """
    Assemble a 3x3 grid of group stimulus distribution plots (rows = PSE, cols = JND).
    Compact layout with minimal spacing.
    """
    n_pse = len(pse_grid)
    n_jnd = len(jnd_grid)

    # 5cm x 5cm per subplot at 300 DPI = 1.97" x 1.97"
    dpi = 300
    subplot_width_inches = 2.2
    subplot_height_inches = 1.8  # Reduced height
    
    fig, axes = plt.subplots(n_pse, n_jnd,
                             figsize=(subplot_width_inches * n_jnd, subplot_height_inches * n_pse),
                             dpi=dpi)

    for pse_idx, pse in enumerate(pse_grid):
        for jnd_idx, jnd in enumerate(jnd_grid):
            ax  = axes[pse_idx][jnd_idx] if n_pse > 1 else axes[jnd_idx]
            key = (pse, jnd)

            if key not in group_data:
                ax.set_visible(False)
                continue

            entry = group_data[key]
            plot_group_stimulus_distribution(
                model_name=model_name,
                pse=pse, jnd=jnd,
                stimulus_center=entry['stimulus_center'],
                stimulus_spread=entry['stimulus_spread'],
                all_latencies=entry['all_latencies'],
                all_responses=entry['all_responses'],
                results_dir=entry['results_dir'],
                file_prefix=entry['file_prefix'],
                ax=ax,
                dpi=dpi,
            )

    # Very compact layout: minimal spacing
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.15, wspace=0.05)
    out_path = output_dir / f"{model_name}_stimulus_distribution_grid.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Grid saved: {out_path.name}")




