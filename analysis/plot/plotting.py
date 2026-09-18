"""
Consolidated plotting module for all plot generation.

Handles:
- Individual subject plots (histogram, psychometric)
- Group plots (aggregated histograms, psychometric curves)
- Grid assembly (3x3 grids of group plots)
"""

import os
import sys
import math
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from utilities.misc_generate_responses import get_jnd_from_sigma
from main.config import *

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utilities.gbf_helpers import load_gbf_with_success


# ============================================================================
# PSYCHOMETRIC CURVE FITTING
# ============================================================================

def fit_psychometric_curve(stimuli, responses, offset=500):
    """
    Fit logistic psychometric function to binned data.

    Uses logistic (not Gaussian) to match the simulation model in generate_response(),
    which uses np.random.logistic(0, sigma) with sigma = jnd / np.log(3).

    Returns:
        mu: threshold (PSE)
        sigma: scale parameter of logistic (jnd = sigma * np.log(3))
        x_fit: x values for fitted curve
        y_fit: y values for fitted curve
        bins_valid: valid bin centers
        f: proportion correct per bin
    """
    from scipy.special import expit

    if len(stimuli) < 3 or len(np.unique(responses)) < 2:
        return None, None, None, None, None, None

    # Bin the data
    bin_size = 10
    sr = sorted(list(zip(stimuli, responses)))

    if len(sr) == 0:
        return None, None, None, None, None, None

    bins = [i * bin_size for i in range(math.floor(sr[0][0] / bin_size),
                                        math.ceil(sr[-1][0] / bin_size) + 1)]

    x = np.array([s[0] for s in sr])
    i_binned = np.digitize(x, bins)
    x_binned = np.asarray(bins)[i_binned - 1]
    r = np.asarray([sr_item[1] for sr_item in sr])

    # Calculate proportion correct per bin
    f = []
    bins_valid = []
    for b in bins:
        mask = x_binned == b
        if np.sum(mask) > 0:
            f.append(np.sum(r[mask]) / np.sum(mask))
            bins_valid.append(b)

    if len(f) < 3:
        return None, None, None, None, None, None

    f = np.asarray(f)
    bins_valid = np.asarray(bins_valid)

    # Fit logistic: P(x) = expit((x - mu) / scale)
    # scale = jnd / np.log(3)
    logistic_fn = lambda x, mu, scale: expit((x - mu) / scale)
    try:
        stim_range = max(stimuli) - min(stimuli)
        p0 = [offset, stim_range / 10]
        bounds = ([offset - stim_range, 1], [offset + stim_range, stim_range])
        (mu, sigma), _ = curve_fit(logistic_fn, bins_valid, f, p0=p0, bounds=bounds, maxfev=10000)
    except Exception:
        return None, None, None, None, None, None

    # Generate fitted curve
    x_fit = np.linspace(min(stimuli) - 50, max(stimuli) + 50, 200)
    y_fit = logistic_fn(x_fit, mu, sigma)

    return mu, sigma, x_fit, y_fit, bins_valid, f

# ============================================================================
# GBF FILE HELPERS
# ============================================================================

def find_gbf_file(group_dir: Path, model_name: str, group_idx: int,
                  subj_in_group: int, pse: float, jnd: float) -> Path | None:
    """Find GBF file, trying exact PSE/JND then glob fallback."""
    pse_int = int(round(pse))
    jnd_int = int(round(jnd))
    exact = group_dir / f"S{subj_in_group:02d}_G{group_idx}_{pse_int}_{jnd_int}_{model_name}.txt"
    if exact.exists():
        return exact
    matches = list(group_dir.glob(f"S{subj_in_group:02d}_G{group_idx}_*_{model_name}.txt"))
    return matches[0] if matches else None


def load_group_rows(group_dir: Path, model_name: str, group_idx: int,
                    pse: float, jnd: float, n_subjects: int, offset: int = 500):
    """Load all rows for a group from GBF files."""
    all_rows = []
    
    for subj_in_group in range(1, n_subjects + 1):
        gbf_file = find_gbf_file(group_dir, model_name, group_idx,
                                 subj_in_group, pse, jnd)
        if gbf_file is None:
            continue
        try:
            rows = load_gbf_with_success(str(gbf_file), offset)
            all_rows.extend(rows)
        except Exception:
            continue
    
    return all_rows

def plot_stim_hist(rows, filepath, offset, subj):
    """Plot histogram of stimuli with success/failure coloring for a single subject."""
    stimuli = [row['lat'] for row in rows]
    successes = [row['res'] == 'true' for row in rows]

    # Create bins of 10ms
    min_stim = min(stimuli)
    max_stim = max(stimuli)
    bins = np.arange(int(min_stim / 10) * 10, int(max_stim / 10) * 10 + 20, 10)

    # Separate success and failure stimuli
    success_stim = [s for s, succ in zip(stimuli, successes) if succ]
    failure_stim = [s for s, succ in zip(stimuli, successes) if not succ]

    plt.figure(figsize=(12, 6))
    plt.hist(success_stim, bins=bins, color='green', alpha=0.7, label='Success', edgecolor='black')
    plt.hist(failure_stim, bins=bins, color='red', alpha=0.7, label='Failure', edgecolor='black')
    plt.axvline(offset, color='blue', linestyle='--', linewidth=2, label=f'Offset ({offset}ms)')
    plt.xlabel('Stimulus latency (ms)')
    plt.ylabel('Count')
    plt.title(f'Stimulus distribution for {subj}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_filename = filepath.replace('.txt', '_histogram.png')
    plt.savefig(plot_filename, bbox_inches='tight', dpi=100)
    plt.close()


def plot_stim_model_hist(rows, filepath, offset, subj):
    """Plot histogram of stimuli by model (pre vs post) for a single subject."""
    stimuli = [row['lat'] for row in rows]
    models = [row['model'] for row in rows]

    # Create bins of 10ms
    min_stim = min(stimuli)
    max_stim = max(stimuli)
    bins = np.arange(int(min_stim / 10) * 10, int(max_stim / 10) * 10 + 20, 10)

    # Plot histogram of stimuli by model (pre vs post)
    pre_stim = [s for s, m in zip(stimuli, models) if m == 'pre']
    post_stim = [s for s, m in zip(stimuli, models) if m == 'post']

    plt.figure(figsize=(12, 6))
    plt.hist(pre_stim, bins=bins, color='blue', alpha=0.7, label='exp_pre', edgecolor='black')
    plt.hist(post_stim, bins=bins, color='yellow', alpha=0.7, label='exp_post', edgecolor='black')
    plt.axvline(offset, color='red', linestyle='--', linewidth=2, label=f'Offset ({offset}ms)')
    plt.xlabel('Stimulus latency (ms)')
    plt.ylabel('Count')
    plt.title(f'Stimulus distribution by model for {subj}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_filename2 = filepath.replace('.txt', '_model_histogram.png')
    plt.savefig(plot_filename2, bbox_inches='tight', dpi=100)
    plt.close()

    print(f"Generated: {filepath}")
    return filepath


# ============================================================================
# GROUP PLOTS
# ============================================================================

def plot_group_histograms(all_rows_list, output_dir, file_prefix, offset, group_label: Optional[str] = None):
    """Create group histograms combining data from all subjects."""
    # Flatten all rows from all subjects
    all_rows = [row for rows in all_rows_list for row in rows]

    stimuli = [row['lat'] for row in all_rows]
    successes = [row['res'] == 'true' for row in all_rows]

    # Create bins of 10ms
    min_stim = min(stimuli)
    max_stim = max(stimuli)
    bins = np.arange(int(min_stim / 10) * 10, int(max_stim / 10) * 10 + 20, 10)

    # Separate success and failure stimuli
    success_stim = [s for s, succ in zip(stimuli, successes) if succ]
    failure_stim = [s for s, succ in zip(stimuli, successes) if not succ]

    # Plot 1: Success vs Failure
    plt.figure(figsize=(12, 6))
    plt.hist(success_stim, bins=bins, color='green', alpha=0.7, label='Success', edgecolor='black')
    plt.hist(failure_stim, bins=bins, color='red', alpha=0.7, label='Failure', edgecolor='black')
    plt.axvline(offset, color='blue', linestyle='--', linewidth=2, label=f'Offset ({offset}ms)')
    plt.xlabel('Stimulus latency (ms)')
    plt.ylabel('Count')
    title = group_label if group_label else 'Group stimulus distribution (all subjects)'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.xlim(200, 800)
    plt.ylim(0, 500)

    plot_filename = os.path.join(output_dir, f"{file_prefix}_group_histogram.png")
    plt.savefig(plot_filename, bbox_inches='tight', dpi=100)
    plt.close()


def plot_group_model_histograms(all_rows_list, output_dir, file_prefix, offset, group_label: Optional[str] = None):
    """Create group histograms by model (pre vs post) combining data from all subjects."""
    # Flatten all rows from all subjects
    all_rows = [row for rows in all_rows_list for row in rows]

    stimuli = [row['lat'] for row in all_rows]
    models = [row['model'] for row in all_rows]

    # Create bins of 10ms
    min_stim = min(stimuli)
    max_stim = max(stimuli)
    bins = np.arange(int(min_stim / 10) * 10, int(max_stim / 10) * 10 + 20, 10)

    # Plot 2: Pre vs Post model
    pre_stim = [s for s, m in zip(stimuli, models) if m == 'pre']
    post_stim = [s for s, m in zip(stimuli, models) if m == 'post']

    plt.figure(figsize=(12, 6))
    plt.hist(pre_stim, bins=bins, color='blue', alpha=0.7, label='exp_pre', edgecolor='black')
    plt.hist(post_stim, bins=bins, color='yellow', alpha=0.7, label='exp_post', edgecolor='black')
    plt.axvline(offset, color='red', linestyle='--', linewidth=2, label=f'Offset ({offset}ms)')
    plt.xlabel('Stimulus latency (ms)')
    plt.ylabel('Count')
    title = group_label if group_label else 'Group stimulus distribution by model (all subjects)'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_filename2 = os.path.join(output_dir, f"{file_prefix}_group_model_histogram.png")
    plt.savefig(plot_filename2, bbox_inches='tight', dpi=100)
    plt.close()

    print(f"Generated group histograms in {output_dir}")


def plot_group_psychometric(all_rows_list, output_dir, file_prefix, offset, group_label: Optional[str] = None):
    """Create group psychometric curve combining data from all subjects."""
    # Flatten all rows from all subjects
    all_rows = [row for rows in all_rows_list for row in rows]

    stimuli = [row['lat'] for row in all_rows]
    responses = [row['user_ans'] for row in all_rows]  # Use user_ans (0 or 1)

    if len(stimuli) == 0:
        print("No data to plot group psychometric")
        return

    # Fit psychometric curve
    result = fit_psychometric_curve(stimuli, responses, offset=offset)
    if result[0] is None:
        print(f"Could not fit psychometric curve for {file_prefix}")
        return

    mu, sigma, x_fit, y_fit, bins_valid, f = result
    jnd = get_jnd_from_sigma(sigma)  # Convert sigma to JND (logistic semi-IQR)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_fit, y_fit, 'b-', linewidth=2, alpha=0.7, label=f'Fit: μ={mu:.1f}, JND={jnd:.1f}')
    plt.plot(bins_valid, f, 'ro', markersize=8, alpha=0.7, label='Data')
    plt.axvline(offset, color='g', linestyle='--', linewidth=2, label=f'True threshold ({offset}ms)')
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Stimulus time (ms)')
    plt.ylabel('P(response = 2)')
    title = group_label if group_label else 'Group psychometric function (all subjects)'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.xlim(200, 800)
    plt.ylim(-0.05, 1.05)

    plot_filename = os.path.join(output_dir, f"{file_prefix}_group_psychometric.png")
    plt.savefig(plot_filename, bbox_inches='tight', dpi=100)
    plt.close()

    print(f"Generated group psychometric: {plot_filename}")


# ============================================================================
# GRID ASSEMBLY
# ============================================================================

def plot_generic_grid(model_name: str, group_data: dict, pse_grid: list, jnd_grid: list,
                      output_dir: Path, plot_func, plot_func_kwargs=None, grid_filename_suffix=''):
    """
    Generic grid assembly function.
    
    Args:
        model_name: Model identifier
        group_data: Dict mapping (pse, jnd) to data dict
        pse_grid: List of PSE values
        jnd_grid: List of JND values
        output_dir: Output directory
        plot_func: Function to call for each subplot (receives ax, data_entry, pse, jnd, **kwargs)
        plot_func_kwargs: Additional kwargs to pass to plot_func
        grid_filename_suffix: Suffix for output filename (e.g., 'psychometric', 'stimulus_distribution')
    """
    if plot_func_kwargs is None:
        plot_func_kwargs = {}
    
    n_pse = len(pse_grid)
    n_jnd = len(jnd_grid)

    fig, axes = plt.subplots(n_pse, n_jnd, figsize=(8 * n_jnd, 5.5 * n_pse))
    fig.suptitle(
        f'{model_name} — {grid_filename_suffix}\n'
        f'(rows = PSE, cols = JND)',
        fontsize=13, fontweight='bold', y=1.01
    )

    for pse_idx, pse in enumerate(pse_grid):
        for jnd_idx, jnd in enumerate(jnd_grid):
            ax = axes[pse_idx][jnd_idx] if n_pse > 1 else axes[jnd_idx]
            key = (pse, jnd)

            if key not in group_data:
                ax.set_visible(False)
                continue

            entry = group_data[key]
            plot_func(ax=ax, data_entry=entry, pse=pse, jnd=jnd, **plot_func_kwargs)

    plt.tight_layout()
    out_path = output_dir / f"{model_name}_grid_{grid_filename_suffix}.png"
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Grid saved: {out_path.name}")


def create_grid_of_plots(plot_files, output_path, title, n_rows: int = 3, n_cols: int = 3, labels: Optional[List[str]] = None):
    """
    Create a grid of subplots from individual plot files.
    
    Args:
        plot_files: List of file paths to individual plots (in grid order)
        output_path: Path where to save the combined grid plot
        title: Title for the combined plot
        n_rows: Number of rows in the grid
        n_cols: Number of columns in the grid
        labels: List of labels for each subplot (e.g., ["PSE_JND", ...])
    """
    from PIL import Image, ImageDraw, ImageFont
    
    if len(plot_files) != n_rows * n_cols:
        print(f"Warning: Expected {n_rows * n_cols} plots, got {len(plot_files)}")
        return
    
    # Load all images
    images = []
    for plot_file in plot_files:
        if os.path.exists(plot_file):
            images.append(Image.open(plot_file))
        else:
            print(f"Warning: Plot file not found: {plot_file}")
            return
    
    if len(images) != n_rows * n_cols:
        print(f"Error: Could not load all {n_rows * n_cols} images")
        return
    
    # Get image dimensions (assume all are the same)
    img_width, img_height = images[0].size
    
    # Add space for labels if provided
    label_height = 30 if labels else 0
    
    # Create grid
    grid_width = img_width * n_cols
    grid_height = (img_height + label_height) * n_rows
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Paste images into grid and add labels
    draw = ImageDraw.Draw(grid_image)
    
    # Try to use a default font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        x = col * img_width
        y = row * (img_height + label_height)
        
        # Paste image
        grid_image.paste(img, (x, y))
        
        # Add label if provided
        if labels and idx < len(labels):
            label_y = y + img_height + 5
            draw.text((x + 10, label_y), labels[idx], fill='black', font=font)
    
    # Save grid
    grid_image.save(output_path)
    print(f"Generated grid plot: {output_path}")


def create_grid_plots_from_groups(group_plot_data, output_dir, model_name, pse_grid, jnd_grid, plot_type='histogram'):
    """
    Create a grid of plots from group results.
    
    Args:
        group_plot_data: Dict mapping group_idx to plot file path
        output_dir: Output directory for the grid plot
        model_name: Model name for the output filename
        pse_grid: List of PSE values
        jnd_grid: List of JND values
        plot_type: Type of plot ('histogram', 'model_histogram', or 'psychometric')
    """
    n_pse = len(pse_grid)
    n_jnd = len(jnd_grid)
    
    # Organize plot files in grid order (PSE on rows, JND on columns)
    plot_files = []
    labels = []
    
    for pse_idx in range(n_pse):
        for jnd_idx in range(n_jnd):
            group_idx = pse_idx * n_jnd + jnd_idx + 1
            if group_idx in group_plot_data:
                plot_files.append(group_plot_data[group_idx])
                pse = pse_grid[pse_idx]
                jnd = jnd_grid[jnd_idx]
                labels.append(f"PSE={pse}, JND={jnd}")
            else:
                print(f"Warning: No plot data for group {group_idx}")
                return
    
    # Create grid
    output_path = os.path.join(output_dir, f"{model_name}_grid_{plot_type}.png")
    title = f"{model_name} - {plot_type} grid"
    create_grid_of_plots(plot_files, output_path, title, n_rows=n_pse, n_cols=n_jnd, labels=labels)


# ============================================================================
# PHASE 4: GRID PLOTS (3x3) FOR CONSOLIDATION
# ============================================================================

def create_phase_c_grid_psychometric_from_group_plots(model_name: str, 
                                                      pse_grid: list, jnd_grid: list,
                                                      model_output_dir: Path, trial_blocks: Optional[list] = None) -> bool:
    """
    Assemble 3x3 grid of GROUP psychometric plots.
    
    Each panel (i,j) corresponds to group PSE_GRID[i], JND_GRID[j].
    Reads pre-generated group_psychometric_*.png files and assembles them.
    
    Args:
        model_name: 'ABS1', 'REL1', 'REL2'
        pse_grid: [480, 500, 520]
        jnd_grid: [20, 40, 60]
        model_output_dir: Where group_psychometric plots are saved
        trial_blocks: unused (kept for API consistency)
    
    Returns:
        True if successful
    """
    try:
        from PIL import Image
        
        fig, axes = plt.subplots(len(pse_grid), len(jnd_grid), figsize=(15, 12))
        fig.suptitle(f'{model_name}: Group Psychometric Curves', fontsize=16, fontweight='bold')
        
        for i, pse in enumerate(pse_grid):
            for j, jnd in enumerate(jnd_grid):
                ax = axes[i, j]
                group_idx = i * len(jnd_grid) + j + 1
                
                # Try to load pre-generated group psychometric plot from temp/
                plot_path = model_output_dir / "temp" / f'{model_name}_G{group_idx}_group_psychometric.png'
                
                if plot_path.exists():
                    # Load image and display in axis
                    img = Image.open(plot_path)
                    ax.imshow(img)
                    ax.axis('off')
                else:
                    ax.text(0.5, 0.5, f'No data\nG{group_idx}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                
                ax.set_title(f'PSE={pse}, JND={jnd}', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        out_path = model_output_dir / f'{model_name}_grid_psychometric.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {out_path.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error in grid_psychometric: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_phase_c_grid_stimulus_distribution_from_group_plots(model_name: str, 
                                                                pse_grid: list, jnd_grid: list,
                                                                model_output_dir: Path, trial_blocks: Optional[list] = None) -> bool:
    """
    Assemble 3x3 grid of GROUP stimulus distribution histogram plots.
    
    Each panel (i,j) corresponds to group PSE_GRID[i], JND_GRID[j].
    Reads pre-generated group_histogram_*.png files and assembles them.
    
    Args:
        model_name: 'ABS1', 'REL1', 'REL2'
        pse_grid: [480, 500, 520]
        jnd_grid: [20, 40, 60]
        model_output_dir: Where group_histogram plots are saved
        trial_blocks: unused (kept for API consistency)
    
    Returns:
        True if successful
    """
    try:
        from PIL import Image
        
        fig, axes = plt.subplots(len(pse_grid), len(jnd_grid), figsize=(15, 12))
        fig.suptitle(f'{model_name}: Group Stimulus Distribution', fontsize=16, fontweight='bold')
        
        for i, pse in enumerate(pse_grid):
            for j, jnd in enumerate(jnd_grid):
                ax = axes[i, j]
                group_idx = i * len(jnd_grid) + j + 1
                
                # Try to load pre-generated group histogram plot from temp/
                plot_path = model_output_dir / "temp" / f'{model_name}_G{group_idx}_group_histogram.png'
                
                if plot_path.exists():
                    # Load image and display in axis
                    img = Image.open(plot_path)
                    ax.imshow(img)
                    ax.axis('off')
                else:
                    ax.text(0.5, 0.5, f'No data\nG{group_idx}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                
                ax.set_title(f'PSE={pse}, JND={jnd}', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        out_path = model_output_dir / f'{model_name}_stimulus_distribution_grid.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {out_path.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error in stimulus_distribution_grid: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_phase_c_asymmetry_modulo(df_model: 'pd.DataFrame', model_name: str, 
                                    pse_grid: list, jnd_grid: list, 
                                    output_dir: Path, trial_blocks: Optional[list] = None) -> bool:
    """
    Plot |asymmetry_index| vs trial blocks: 3 curves (one per JND) at middle PSE.
    Uses groups to avoid jitter issues.
    
    Groups (with middle PSE):
    - Curve 1 (JND≈20): G4 (PSE=500, JND≈20)
    - Curve 2 (JND≈40): G5 (PSE=500, JND≈40)
    - Curve 3 (JND≈60): G6 (PSE=500, JND≈60)
    
    Args:
        df_model: DataFrame for single model
        model_name: 'ABS1', 'REL1', or 'REL2'
        pse_grid: [480, 500, 520]
        jnd_grid: [20, 40, 60]
        output_dir: Where to save plot
        trial_blocks: [40, 60, 80, ..., 200]
    
    Returns:
        True if successful
    """
    if trial_blocks is None:
        trial_blocks = TRIAL_BLOCKS
    
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Groups at middle PSE (Logica 2: PSE row)
        # For PSE=500 (middle): G4, G5, G6
        group_sets = [
            ('G4', 20),
            ('G5', 40),
            ('G6', 60)
        ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        plotted = False
        for jnd_idx, (group_id, jnd_label) in enumerate(group_sets):
            # Filter data for this group
            group_data = df_model[df_model['group'] == group_id]
            
            if group_data.empty:
                continue
            
            # Extract asymmetry values for each trial block
            means_list = []
            stds_list = []
            
            for block in trial_blocks:
                col = f'asymmetry_{block}'
                if col in group_data.columns:
                    values = group_data[col].dropna()
                    if len(values) > 0:
                        # Take absolute value of asymmetry for each subject, then average
                        abs_values = np.abs(values)
                        means_list.append(abs_values.mean())
                        stds_list.append(abs_values.std())
            
            if means_list:
                color_idx = jnd_idx % len(colors)
                ax.plot(trial_blocks[:len(means_list)], means_list, marker='o', linewidth=2.5, markersize=8,
                       color=colors[color_idx], label=f'JND≈{int(jnd_label)}')
                ax.fill_between(trial_blocks[:len(means_list)], 
                               np.array(means_list) - np.array(stds_list),
                               np.array(means_list) + np.array(stds_list),
                               alpha=0.2, color=colors[color_idx])
                plotted = True
        
        if plotted:
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Zero')
            ax.set_xlabel('Trial Block', fontsize=12, fontweight='bold')
            ax.set_ylabel('|Asymmetry Index|', fontsize=12, fontweight='bold')
            ax.set_title(f'{model_name}: Evolution of |Asymmetry Index| (PSE≈500)',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(trial_blocks)
            ax.legend(fontsize=11)
        
        plt.tight_layout()
        out_path = Path(output_dir) / f'{model_name}_asymmetry_modulo.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {out_path.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error in asymmetry_modulo: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_phase_c_asymmetry_scatter_envelope(df_model: 'pd.DataFrame', model_name: str, 
                                             pse_grid: list, jnd_grid: list, 
                                             output_dir: Path, trial_blocks: Optional[list] = None) -> bool:
    """
    Plot scatter of asymmetry values with envelope curves (1 subplot per JND group).
    
    Grouping by JND (Logica 1):
    - Subplot 1: G1, G4, G7 (JND≈20)
    - Subplot 2: G2, G5, G8 (JND≈40)
    - Subplot 3: G3, G6, G9 (JND≈60)
    
    Args:
        df_model: DataFrame for single model
        model_name: 'ABS1', 'REL1', or 'REL2'
        pse_grid: [480, 500, 520]
        jnd_grid: [20, 40, 60]
        output_dir: Where to save plot
        trial_blocks: [40, 60, 80, ..., 200]
    
    Returns:
        True if successful
    """
    import pandas as pd
    
    if trial_blocks is None:
        trial_blocks = TRIAL_BLOCKS
    
    try:
        # Define 3 group sets by JND (Logica 1)
        group_sets = [
            (['G1', 'G4', 'G7'], 20),
            (['G2', 'G5', 'G8'], 40),
            (['G3', 'G6', 'G9'], 60)
        ]
        
        n_jnd = len(group_sets)
        fig, axes = plt.subplots(1, n_jnd, figsize=(6*n_jnd, 5))
        
        if n_jnd == 1:
            axes = [axes]
        
        for jnd_idx, (groups, jnd_label) in enumerate(group_sets):
            ax = axes[jnd_idx]
            
            all_asymmetry_values = {block: [] for block in trial_blocks}
            
            # Collect all asymmetry values for this JND group
            jnd_data = df_model[df_model['group'].isin(groups)]
            
            for block in trial_blocks:
                col = f'asymmetry_{block}'
                if col in jnd_data.columns:
                    all_asymmetry_values[block].extend(jnd_data[col].dropna().values)
            
            # Plot scatter
            for block in trial_blocks:
                values = all_asymmetry_values[block]
                if values:
                    x_jitter = np.random.normal(block, 1.5, len(values))
                    ax.scatter(x_jitter, values, alpha=0.4, s=30, color='gray')
            
            # Plot envelopes
            positive_envelope = []
            negative_envelope = []
            
            for block in trial_blocks:
                values = all_asymmetry_values[block]
                if values:
                    pos_vals = [v for v in values if v > 0]
                    neg_vals = [v for v in values if v < 0]
                    positive_envelope.append(np.max(pos_vals) if pos_vals else np.nan)
                    negative_envelope.append(np.min(neg_vals) if neg_vals else np.nan)
                else:
                    positive_envelope.append(np.nan)
                    negative_envelope.append(np.nan)
            
            ax.plot(trial_blocks, positive_envelope, 'g-', linewidth=2.5, marker='s', markersize=6, label='Max (>0)')
            ax.plot(trial_blocks, negative_envelope, 'r-', linewidth=2.5, marker='s', markersize=6, label='Min (<0)')
            ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            ax.set_xlabel('Trial Block', fontsize=11)
            ax.set_ylabel('Asymmetry Index', fontsize=11)
            ax.set_title(f'JND≈{int(jnd_label)}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        
        fig.suptitle(f'{model_name}: Asymmetry Index Distribution', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        out_path = Path(output_dir) / f'{model_name}_asymmetry_scatter_envelope.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {out_path.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error in asymmetry_scatter_envelope: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_phase_c_stimulus_center_evolution(df_model: 'pd.DataFrame', model_name: str, 
                                            output_dir: Path, trial_blocks: Optional[list] = None) -> bool:
    """
    Plot stimulus center evolution: 3 curves (one per PSE), each from specific groups.
    
    Grouping by PSE (Logica 2):
    - Curve 1 (PSE≈480): G1, G2, G3
    - Curve 2 (PSE≈500): G4, G5, G6
    - Curve 3 (PSE≈520): G7, G8, G9
    
    Args:
        df_model: DataFrame for single model
        model_name: 'ABS1', 'REL1', or 'REL2'
        output_dir: Where to save plot
        trial_blocks: [40, 60, 80, ..., 200]
    
    Returns:
        True if successful
    """
    if trial_blocks is None:
        trial_blocks = TRIAL_BLOCKS
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define 3 group sets by PSE
        group_sets = [
            (['G1', 'G2', 'G3'], 480),
            (['G4', 'G5', 'G6'], 500),
            (['G7', 'G8', 'G9'], 520)
        ]
        
        # Calculate global y-axis limits
        all_means = []
        all_stds = []
        for groups, pse_label in group_sets:
            group_data = df_model[df_model['group'].isin(groups)]
            for block in trial_blocks:
                col = f'stimulus_center_{block}'
                if col in group_data.columns:
                    vals = group_data[col].dropna()
                    if len(vals) > 0:
                        all_means.extend(vals)
                        all_stds.append(vals.std())
        
        if all_means:
            y_min = np.min(all_means) - np.max(all_stds) - 10
            y_max = np.max(all_means) + np.max(all_stds) + 10
        else:
            y_min, y_max = 400, 600
        
        # 3 curves: one per PSE group
        for groups, pse_label in group_sets:
            group_data = df_model[df_model['group'].isin(groups)]
            means_list = []
            stds_list = []
            
            for block in trial_blocks:
                col = f'stimulus_center_{block}'
                if col in group_data.columns:
                    vals = group_data[col].dropna()
                    if len(vals) > 0:
                        means_list.append(vals.mean())
                        stds_list.append(vals.std())
            
            if means_list:
                ax.plot(trial_blocks[:len(means_list)], means_list, marker='o', 
                       linewidth=2.5, label=f'PSE={int(pse_label)}')
                ax.fill_between(trial_blocks[:len(means_list)], 
                               np.array(means_list) - np.array(stds_list),
                               np.array(means_list) + np.array(stds_list),
                               alpha=0.2)
                ax.axhline(y=pse_label, color='red', linestyle='--', linewidth=1.5, alpha=0.3)
        
        ax.set_xlabel('Trial Block', fontsize=11, fontweight='bold')
        ax.set_ylabel('Stimulus Center (ms)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}: Stimulus Center Evolution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(trial_blocks)
        ax.legend(fontsize=10)
        ax.set_ylim([y_min, y_max])
        
        plt.tight_layout()
        out_path = Path(output_dir) / f'{model_name}_stimulus_center_evolution.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {out_path.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error in stimulus_center_evolution: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_phase_c_stimulus_spread_evolution(df_model: 'pd.DataFrame', model_name: str, 
                                            output_dir: Path, trial_blocks: Optional[list] = None) -> bool:
    """
    Plot stimulus spread evolution: 3 curves (one per JND), each from specific groups.
    
    Grouping by JND (Logica 1):
    - Curve 1 (JND≈20): G1, G4, G7
    - Curve 2 (JND≈40): G2, G5, G8
    - Curve 3 (JND≈60): G3, G6, G9
    
    Args:
        df_model: DataFrame for single model
        model_name: 'ABS1', 'REL1', or 'REL2'
        output_dir: Where to save plot
        trial_blocks: [40, 60, 80, ..., 200]
    
    Returns:
        True if successful
    """
    if trial_blocks is None:
        trial_blocks = TRIAL_BLOCKS
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define 3 group sets by JND
        group_sets = [
            (['G1', 'G4', 'G7'], 20),
            (['G2', 'G5', 'G8'], 40),
            (['G3', 'G6', 'G9'], 60)
        ]
        
        # Calculate global y-axis limits
        all_means = []
        all_stds = []
        for groups, jnd_label in group_sets:
            group_data = df_model[df_model['group'].isin(groups)]
            for block in trial_blocks:
                col = f'stimulus_spread_{block}'
                if col in group_data.columns:
                    vals = group_data[col].dropna()
                    if len(vals) > 0:
                        all_means.extend(vals)
                        all_stds.append(vals.std())
        
        if all_means:
            y_min = max(0, np.min(all_means) - np.max(all_stds) - 5)
            y_max = np.max(all_means) + np.max(all_stds) + 5
        else:
            y_min, y_max = 0, 100
        
        # 3 curves: one per JND group
        for groups, jnd_label in group_sets:
            group_data = df_model[df_model['group'].isin(groups)]
            means_list = []
            stds_list = []
            
            for block in trial_blocks:
                col = f'stimulus_spread_{block}'
                if col in group_data.columns:
                    vals = group_data[col].dropna()
                    if len(vals) > 0:
                        means_list.append(vals.mean())
                        stds_list.append(vals.std())
            
            if means_list:
                ax.plot(trial_blocks[:len(means_list)], means_list, marker='s', 
                       linewidth=2.5, label=f'JND={int(jnd_label)}')
                ax.fill_between(trial_blocks[:len(means_list)], 
                               np.array(means_list) - np.array(stds_list),
                               np.array(means_list) + np.array(stds_list),
                               alpha=0.2)
        
        ax.set_xlabel('Trial Block', fontsize=11, fontweight='bold')
        ax.set_ylabel('Stimulus Spread (ms)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}: Stimulus Spread Evolution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(trial_blocks)
        ax.legend(fontsize=10)
        ax.set_ylim([y_min, y_max])
        
        plt.tight_layout()
        out_path = Path(output_dir) / f'{model_name}_stimulus_spread_evolution.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {out_path.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error in stimulus_spread_evolution: {e}")
        import traceback
        traceback.print_exc()
        return False
        ax.grid(True, alpha=0.3)
        ax.set_xticks(trial_blocks)
        ax.legend(fontsize=10, loc='best', ncol=3)
        
        plt.tight_layout()
        out_path = Path(output_dir) / f'{model_name}_stimulus_spread_evolution.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {out_path.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error in stimulus_spread_evolution: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# PHASE C: POSTERIORS PLOTS PER MODELLO (from Excel data)
# ============================================================================

def create_phase_c_posteriors_b_pse_sd(df_model: 'pd.DataFrame', model_name: str, 
                                       pse_grid: list, jnd_grid: list, 
                                       output_dir: Path, trial_blocks: Optional[list] = None) -> bool:
    """
    Generate 3×3 grid plot: PSE posterior SD evolution (Level B metrics) - PER MODELLO.
    
    Args:
        df_model: DataFrame for single model (180 rows)
        model_name: 'ABS1', 'REL1', or 'REL2'
        pse_grid, jnd_grid: Grid parameters
        output_dir: Where to save
        trial_blocks: Trial block numbers
    
    Returns:
        True if successful
    """
    import pandas as pd
    from itertools import product
    
    if trial_blocks is None:
        trial_blocks = TRIAL_BLOCKS
    
    try:
        group_map = {}
        for group_idx, (pse_center, jnd_center) in enumerate(product(pse_grid, jnd_grid), 1):
            group_map[(pse_center, jnd_center)] = f'G{group_idx}'
        
        fig, axes = plt.subplots(len(pse_grid), len(jnd_grid), figsize=(15, 12))
        fig.suptitle(f'{model_name}: Posterior SD Evolution: PSE (Level B)', fontsize=16, fontweight='bold')
        
        for i, pse_center in enumerate(pse_grid):
            for j, jnd_center in enumerate(jnd_grid):
                ax = axes[i, j]
                group_id = group_map[(pse_center, jnd_center)]
                group_data = df_model[df_model['group'] == group_id]
                
                if group_data.empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'PSE={pse_center}, JND={jnd_center}')
                    continue
                
                pse_sd_cols = [f'posterior_sd_pse_{b}' for b in trial_blocks]
                pse_sd_data = group_data[pse_sd_cols]
                pse_sd_mean = pse_sd_data.mean()
                pse_sd_std = pse_sd_data.std()
                
                ax.errorbar(trial_blocks, pse_sd_mean, yerr=pse_sd_std, 
                           fmt='o-', linewidth=2, markersize=6, capsize=4)
                ax.set_title(f'PSE={pse_center}, JND={jnd_center} (n={len(group_data)})', fontsize=10)
                ax.set_xlabel('Trial Block')
                ax.set_ylabel('Posterior SD (PSE)')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        out_path = Path(output_dir) / f'posteriors_b_pse_sd_evolution_{model_name}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {out_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def create_phase_c_posteriors_b_jnd_sd(df_model: 'pd.DataFrame', model_name: str, 
                                       pse_grid: list, jnd_grid: list, 
                                       output_dir: Path, trial_blocks: Optional[list] = None) -> bool:
    """Generate 3×3 grid: JND posterior SD evolution (Level B) - per modello."""
    import pandas as pd
    from itertools import product
    
    if trial_blocks is None:
        trial_blocks = TRIAL_BLOCKS
    
    try:
        group_map = {}
        for group_idx, (pse_center, jnd_center) in enumerate(product(pse_grid, jnd_grid), 1):
            group_map[(pse_center, jnd_center)] = f'G{group_idx}'
        
        fig, axes = plt.subplots(len(pse_grid), len(jnd_grid), figsize=(15, 12))
        fig.suptitle(f'{model_name}: Posterior SD Evolution: JND (Level B)', fontsize=16, fontweight='bold')
        
        for i, pse_center in enumerate(pse_grid):
            for j, jnd_center in enumerate(jnd_grid):
                ax = axes[i, j]
                group_id = group_map[(pse_center, jnd_center)]
                group_data = df_model[df_model['group'] == group_id]
                
                if group_data.empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'PSE={pse_center}, JND={jnd_center}')
                    continue
                
                jnd_sd_cols = [f'posterior_sd_jnd_{b}' for b in trial_blocks]
                jnd_sd_data = group_data[jnd_sd_cols]
                jnd_sd_mean = jnd_sd_data.mean()
                jnd_sd_std = jnd_sd_data.std()
                
                ax.errorbar(trial_blocks, jnd_sd_mean, yerr=jnd_sd_std, 
                           fmt='s-', linewidth=2, markersize=6, capsize=4, color='orange')
                ax.set_title(f'PSE={pse_center}, JND={jnd_center} (n={len(group_data)})', fontsize=10)
                ax.set_xlabel('Trial Block')
                ax.set_ylabel('Posterior SD (JND)')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        out_path = Path(output_dir) / f'posteriors_b_jnd_sd_evolution_{model_name}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {out_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def create_phase_c_posteriors_a_heatmap(df_model: 'pd.DataFrame', model_name: str, 
                                        pse_grid: list, jnd_grid: list, 
                                        output_dir: Path) -> bool:
    """Generate 3×3 heatmap: PSE posterior SD at trial 200 (Level A) - per modello."""
    import pandas as pd
    from itertools import product
    
    try:
        group_map = {}
        for group_idx, (pse_center, jnd_center) in enumerate(product(pse_grid, jnd_grid), 1):
            group_map[(pse_center, jnd_center)] = f'G{group_idx}'
        
        heatmap_data = np.zeros((len(pse_grid), len(jnd_grid)))
        heatmap_count = np.zeros((len(pse_grid), len(jnd_grid)))
        
        for i, pse_center in enumerate(pse_grid):
            for j, jnd_center in enumerate(jnd_grid):
                group_id = group_map[(pse_center, jnd_center)]
                group_data = df_model[df_model['group'] == group_id]
                
                if not group_data.empty:
                    heatmap_data[i, j] = group_data['posterior_sd_pse_200'].mean()
                    heatmap_count[i, j] = len(group_data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
        
        ax.set_xticks(np.arange(len(jnd_grid)))
        ax.set_yticks(np.arange(len(pse_grid)))
        ax.set_xticklabels(jnd_grid)
        ax.set_yticklabels(pse_grid)
        ax.set_xlabel('JND', fontsize=12)
        ax.set_ylabel('PSE', fontsize=12)
        ax.set_title(f'{model_name}: Posterior SD (PSE) at Trial 200 (Level A)', fontsize=14, fontweight='bold')
        
        for i in range(len(pse_grid)):
            for j in range(len(jnd_grid)):
                value = heatmap_data[i, j]
                count = int(heatmap_count[i, j])
                ax.text(j, i, f'{value:.1f}\n(n={count})',
                       ha="center", va="center", color="white", fontsize=10)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Posterior SD (PSE)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        out_path = Path(output_dir) / f'posteriors_a_pse_sd_heatmap_{model_name}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {out_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


# ============================================================================
# PHASE C: COMPARISON PLOTS (consolidated - average 3 models)
# ============================================================================

def create_phase_c_comparison_posteriors_b_pse_sd(dfs_by_model: dict, 
                                                  pse_grid: list, jnd_grid: list, 
                                                  output_dir: Path, trial_blocks: Optional[list] = None) -> bool:
    """
    Generate 3×3 grid: PSE posterior SD evolution - WITH 3 CURVES (one per model).
    
    Each cell (i,j) shows 3 curves (ABS1, REL1, REL2) for that PSE/JND group.
    
    Args:
        dfs_by_model: Dict with keys 'ABS1', 'REL1', 'REL2' → DataFrames (180 rows each)
        pse_grid, jnd_grid: Grid parameters
        output_dir: Where to save
        trial_blocks: Trial block numbers
    
    Returns:
        True if successful
    """
    import pandas as pd
    from itertools import product
    
    if trial_blocks is None:
        trial_blocks = TRIAL_BLOCKS
    
    try:
        group_map = {}
        for group_idx, (pse_center, jnd_center) in enumerate(product(pse_grid, jnd_grid), 1):
            group_map[(pse_center, jnd_center)] = f'G{group_idx}'
        
        colors = {'ABS1': '#1f77b4', 'REL1': '#ff7f0e', 'REL2': '#2ca02c'}
        
        fig, axes = plt.subplots(len(pse_grid), len(jnd_grid), figsize=(15, 12))
        fig.suptitle('Posterior SD Evolution: PSE (Level B) - Comparison Across Models', 
                    fontsize=16, fontweight='bold')
        
        for i, pse_center in enumerate(pse_grid):
            for j, jnd_center in enumerate(jnd_grid):
                ax = axes[i, j]
                group_id = group_map[(pse_center, jnd_center)]
                
                # Plot 3 curves: one per model
                for model_name in ['ABS1', 'REL1', 'REL2']:
                    df_model = dfs_by_model.get(model_name)
                    if df_model is None:
                        continue
                    
                    group_data = df_model[df_model['group'] == group_id]
                    if group_data.empty:
                        continue
                    
                    pse_sd_cols = [f'posterior_sd_pse_{b}' for b in trial_blocks]
                    pse_sd_data = group_data[pse_sd_cols]
                    pse_sd_mean = pse_sd_data.mean()
                    pse_sd_std = pse_sd_data.std()
                    
                    ax.plot(trial_blocks, pse_sd_mean, 'o-', linewidth=2, markersize=6,
                           color=colors[model_name], label=model_name)
                    ax.fill_between(trial_blocks, 
                                   pse_sd_mean - pse_sd_std,
                                   pse_sd_mean + pse_sd_std,
                                   alpha=0.2, color=colors[model_name])
                
                ax.set_title(f'PSE={pse_center}, JND={jnd_center}', fontsize=10)
                ax.set_xlabel('Trial Block', fontsize=9)
                ax.set_ylabel('Posterior SD (PSE)', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(bottom=0)
                ax.legend(fontsize=8)
        
        plt.tight_layout()
        out_path = Path(output_dir) / 'posteriors_b_pse_sd_evolution.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {out_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_phase_c_comparison_posteriors_b_jnd_sd(dfs_by_model: dict, 
                                                  pse_grid: list, jnd_grid: list, 
                                                  output_dir: Path, trial_blocks: Optional[list] = None) -> bool:
    """Generate 3×3 grid: JND posterior SD evolution - WITH 3 CURVES (one per model)."""
    import pandas as pd
    from itertools import product
    
    if trial_blocks is None:
        trial_blocks = TRIAL_BLOCKS
    
    try:
        group_map = {}
        for group_idx, (pse_center, jnd_center) in enumerate(product(pse_grid, jnd_grid), 1):
            group_map[(pse_center, jnd_center)] = f'G{group_idx}'
        
        colors = {'ABS1': '#1f77b4', 'REL1': '#ff7f0e', 'REL2': '#2ca02c'}
        
        fig, axes = plt.subplots(len(pse_grid), len(jnd_grid), figsize=(15, 12))
        fig.suptitle('Posterior SD Evolution: JND (Level B) - Comparison Across Models', 
                    fontsize=16, fontweight='bold')
        
        for i, pse_center in enumerate(pse_grid):
            for j, jnd_center in enumerate(jnd_grid):
                ax = axes[i, j]
                group_id = group_map[(pse_center, jnd_center)]
                
                # Plot 3 curves: one per model
                for model_name in ['ABS1', 'REL1', 'REL2']:
                    df_model = dfs_by_model.get(model_name)
                    if df_model is None:
                        continue
                    
                    group_data = df_model[df_model['group'] == group_id]
                    if group_data.empty:
                        continue
                    
                    jnd_sd_cols = [f'posterior_sd_jnd_{b}' for b in trial_blocks]
                    jnd_sd_data = group_data[jnd_sd_cols]
                    jnd_sd_mean = jnd_sd_data.mean()
                    jnd_sd_std = jnd_sd_data.std()
                    
                    ax.plot(trial_blocks, jnd_sd_mean, 's-', linewidth=2, markersize=6,
                           color=colors[model_name], label=model_name)
                    ax.fill_between(trial_blocks, 
                                   jnd_sd_mean - jnd_sd_std,
                                   jnd_sd_mean + jnd_sd_std,
                                   alpha=0.2, color=colors[model_name])
                
                ax.set_title(f'PSE={pse_center}, JND={jnd_center}', fontsize=10)
                ax.set_xlabel('Trial Block', fontsize=9)
                ax.set_ylabel('Posterior SD (JND)', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(bottom=0)
                ax.legend(fontsize=8)
        
        plt.tight_layout()
        out_path = Path(output_dir) / 'posteriors_b_jnd_sd_evolution.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {out_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_phase_c_comparison_posteriors_a_heatmap(dfs_by_model: dict, 
                                                   pse_grid: list, jnd_grid: list, 
                                                   output_dir: Path) -> bool:
    """Generate 3×3 heatmap: PSE posterior SD at trial 200 - AVERAGED ACROSS 3 MODELS."""
    import pandas as pd
    from itertools import product
    
    try:
        group_map = {}
        for group_idx, (pse_center, jnd_center) in enumerate(product(pse_grid, jnd_grid), 1):
            group_map[(pse_center, jnd_center)] = f'G{group_idx}'
        
        heatmap_data = np.zeros((len(pse_grid), len(jnd_grid)))
        heatmap_count = np.zeros((len(pse_grid), len(jnd_grid)))
        
        for i, pse_center in enumerate(pse_grid):
            for j, jnd_center in enumerate(jnd_grid):
                group_id = group_map[(pse_center, jnd_center)]
                
                all_group_data = []
                for model_name in ['ABS1', 'REL1', 'REL2']:
                    df_model = dfs_by_model.get(model_name)
                    if df_model is not None:
                        group_data = df_model[df_model['group'] == group_id]
                        if not group_data.empty:
                            all_group_data.append(group_data)
                
                if all_group_data:
                    combined_data = pd.concat(all_group_data, ignore_index=True)
                    heatmap_data[i, j] = combined_data['posterior_sd_pse_200'].mean()
                    heatmap_count[i, j] = len(combined_data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
        
        ax.set_xticks(np.arange(len(jnd_grid)))
        ax.set_yticks(np.arange(len(pse_grid)))
        ax.set_xticklabels(jnd_grid)
        ax.set_yticklabels(pse_grid)
        ax.set_xlabel('JND', fontsize=12)
        ax.set_ylabel('PSE', fontsize=12)
        ax.set_title('Posterior SD (PSE) at Trial 200 (Level A) - Comparison Across Models', 
                    fontsize=14, fontweight='bold')
        
        for i in range(len(pse_grid)):
            for j in range(len(jnd_grid)):
                value = heatmap_data[i, j]
                count = int(heatmap_count[i, j])
                ax.text(j, i, f'{value:.1f}\n(n={count})',
                       ha="center", va="center", color="white", fontsize=10)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Posterior SD (PSE)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        out_path = Path(output_dir) / 'posteriors_a_pse_sd_heatmap.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {out_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


# ============================================================================
# PHASE C: COMPARISON GRID PSYCHOMETRIC (3 curves per cell, one per model)
# ============================================================================

def _fit_and_plot_psychometric_for_model(ax, stimuli, responses, model_name, colors):
    """
    Fit logistic psychometric curve and plot into ax for one model.
    Helper for create_phase_c_comparison_grid_psychometric.
    Uses fit_psychometric_curve (logistic) for consistency with the rest of Phase C.
    """
    result = fit_psychometric_curve(stimuli, responses, offset=500)
    if result[0] is None:
        return

    mu, sigma, x_fit, y_fit, bins_valid, f = result
    jnd = get_jnd_from_sigma(sigma)

    ax.plot(x_fit, y_fit, color=colors[model_name], linewidth=2,
            label=f'{model_name} (JND={jnd:.1f})')
    ax.plot(bins_valid, f, 'o', color=colors[model_name], markersize=3.5, alpha=0.6)


def create_phase_c_comparison_grid_psychometric(pse_grid: list, jnd_grid: list,
                                                output_dir: Path,
                                                n_subjects: int = None) -> bool:
    """
    3x3 grid of psychometric curves — one curve per model (ABS1, REL1, REL2) per cell.

    Each cell (i,j) = PSE_GRID[i] x JND_GRID[j].
    Reads GBF files directly (same as create_paper_plots.create_grid_psychometric).

    Output: {output_dir}/comparison_grid_psychometric.tif  (16cm x 16cm, 300 DPI, LZW TIFF)
    """
    from main.config import OUTPUT_BASE, N_SUBJECTS_PER_GROUP, OFFSET
    from itertools import product as iproduct

    colors = {"ABS1": "#1f77b4", "REL1": "#ff7f0e", "REL2": "#2ca02c"}
    models = ["ABS1", "REL1", "REL2"]

    project_root = Path(__file__).parent.parent.parent

    try:
        # 16cm x 16cm at 300 DPI
        fig, axes = plt.subplots(len(pse_grid), len(jnd_grid), figsize=(6.3, 6.3), dpi=300)
        fig.suptitle('Psychometric Functions by PSE/JND', fontsize=11, fontweight='bold', y=0.98)

        grid = list(iproduct(pse_grid, jnd_grid))

        for idx, (pse, jnd) in enumerate(grid):
            row = idx // len(jnd_grid)
            col = idx % len(jnd_grid)
            ax = axes[row, col]

            group_idx = row * len(jnd_grid) + col + 1

            for model_name in models:
                try:
                    group_dir = project_root / OUTPUT_BASE / model_name / f"group_{pse}_{jnd}"
                    if not group_dir.exists():
                        continue

                    n_subj = n_subjects if n_subjects is not None else N_SUBJECTS_PER_GROUP
                    rows = load_group_rows(group_dir, model_name, group_idx, pse, jnd, n_subj, OFFSET)
                    if not rows:
                        continue

                    stimuli = [r['lat'] for r in rows]
                    responses = [r['user_ans'] for r in rows]
                    _fit_and_plot_psychometric_for_model(ax, stimuli, responses, model_name, colors)

                except Exception as e:
                    print(f"    ⚠ {model_name} {pse}/{jnd}: {e}")

            ax.axvline(500, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(f'PSE={pse}, JND={jnd}', fontsize=9, fontweight='bold')
            ax.set_xlim(200, 800)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3, linewidth=0.4)
            ax.legend(fontsize=6.5, loc='upper left', framealpha=0.95)
            ax.tick_params(labelsize=6.5)

        plt.tight_layout()
        out_path = Path(output_dir) / 'comparison_grid_psychometric.tif'
        plt.savefig(out_path, dpi=300, bbox_inches='tight', format='tiff',
                    pil_kwargs={'compression': 'tiff_lzw'})
        plt.close()

        print(f"  ✓ {out_path.name}")
        return True

    except Exception as e:
        print(f"  ✗ Error in comparison_grid_psychometric: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# PHASE C: COMPARISON LATENCY KDE GRID (3 KDE curves per cell, one per model)
# ============================================================================

def create_phase_c_comparison_latency_kde_grid(pse_grid: list, jnd_grid: list,
                                               output_dir: Path,
                                               n_subjects: int = None) -> bool:
    """
    3x3 grid of latency KDE envelopes — one curve per model (ABS1, REL1, REL2) per cell.

    Each cell (i,j) = PSE_GRID[i] x JND_GRID[j].
    Reads GBF files directly via load_group_rows.

    Output: {output_dir}/comparison_latency_kde_grid.tif  (16cm x 12.7cm, 300 DPI, LZW TIFF)
    """
    from scipy.stats import gaussian_kde
    from main.config import OUTPUT_BASE, N_SUBJECTS_PER_GROUP, OFFSET
    from itertools import product as iproduct

    colors = {"ABS1": "#1f77b4", "REL1": "#ff7f0e", "REL2": "#2ca02c"}
    models = ["ABS1", "REL1", "REL2"]

    project_root = Path(__file__).parent.parent.parent

    try:
        fig, axes = plt.subplots(len(pse_grid), len(jnd_grid), figsize=(6.3, 5.0), dpi=300)
        fig.suptitle('Stimuli Latencies Distribution by PSE/JND', fontsize=11, fontweight='bold', y=0.98)

        handles = []
        labels = []

        grid = list(iproduct(pse_grid, jnd_grid))

        for idx, (pse, jnd) in enumerate(grid):
            row = idx // len(jnd_grid)
            col = idx % len(jnd_grid)
            ax = axes[row, col]

            group_idx = row * len(jnd_grid) + col + 1

            for model_name in models:
                try:
                    group_dir = project_root / OUTPUT_BASE / model_name / f"group_{pse}_{jnd}"
                    if not group_dir.exists():
                        continue

                    n_subj = n_subjects if n_subjects is not None else N_SUBJECTS_PER_GROUP
                    rows = load_group_rows(group_dir, model_name, group_idx, pse, jnd, n_subj, OFFSET)
                    if not rows:
                        continue

                    latencies = np.array([r['lat'] for r in rows])
                    if len(latencies) < 5:
                        continue

                    kde = gaussian_kde(latencies, bw_method='scott')
                    x_kde = np.linspace(latencies.min() - 20, latencies.max() + 20, 300)
                    kde_vals = kde(x_kde)
                    kde_vals_scaled = kde_vals * len(latencies) * 10  # bin_width=10

                    line, = ax.plot(x_kde, kde_vals_scaled, color=colors[model_name],
                                    linewidth=2, label=model_name, alpha=0.8)

                    if model_name not in labels:
                        handles.append(line)
                        labels.append(model_name)

                except Exception as e:
                    print(f"    ⚠ {model_name} {pse}/{jnd}: {e}")

            ax.set_title(f'PSE={pse}, JND={jnd}', fontsize=6, fontweight='bold')
            ax.axvline(500, color='gray', linestyle='--', linewidth=1, alpha=0.85)
            ax.set_xlim(200, 800)
            ax.set_ylim(0, 600)
            ax.set_xticks([300, 400, 500, 600, 700])
            ax.grid(True, alpha=0.3, linewidth=0.4)
            ax.tick_params(labelsize=6.5)

            if row < len(pse_grid) - 1:
                ax.set_xticklabels([])
            if col > 0:
                ax.set_yticklabels([])

        if handles:
            fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=8,
                       framealpha=0.95, bbox_to_anchor=(0.5, 0.95))

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = Path(output_dir) / 'comparison_latency_kde_grid.tif'
        plt.savefig(out_path, dpi=300, bbox_inches='tight', format='tiff',
                    pil_kwargs={'compression': 'tiff_lzw'})
        plt.close()

        print(f"  ✓ {out_path.name}")
        return True

    except Exception as e:
        print(f"  ✗ Error in comparison_latency_kde_grid: {e}")
        import traceback
        traceback.print_exc()
        return False
