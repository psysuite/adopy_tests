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

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utilities.gbf_helpers import load_gbf_with_success


# ============================================================================
# PSYCHOMETRIC CURVE FITTING
# ============================================================================

def fit_psychometric_curve(stimuli, responses, offset=500):
    """
    Fit cumulative Gaussian psychometric function to binned data.
    
    Returns:
        mu: threshold (PSE)
        sigma: slope (JND)
        x_fit: x values for fitted curve
        y_fit: y values for fitted curve
        bins_valid: valid bin centers
        f: proportion correct per bin
    """
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
    
    # Fit cumulative Gaussian
    try:
        stim_range = max(stimuli) - min(stimuli)
        p0 = [offset, stim_range / 10]
        bounds = ([offset - stim_range, 1], [offset + stim_range, stim_range])
        mu, sigma = curve_fit(norm.cdf, bins_valid, f, p0=p0, bounds=bounds, maxfev=10000)[0]
    except Exception:
        return None, None, None, None, None, None
    
    # Generate fitted curve
    x_fit = np.linspace(min(stimuli) - 50, max(stimuli) + 50, 200)
    y_fit = norm.cdf(x_fit, mu, sigma)
    
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

def plot_group_histograms(all_rows_list, output_dir, file_prefix, offset, group_label=None):
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

    plot_filename = os.path.join(output_dir, f"{file_prefix}_group_histogram.png")
    plt.savefig(plot_filename, bbox_inches='tight', dpi=100)
    plt.close()


def plot_group_model_histograms(all_rows_list, output_dir, file_prefix, offset, group_label=None):
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


def plot_group_psychometric(all_rows_list, output_dir, file_prefix, offset, group_label=None):
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
    jnd = sigma * 0.6745  # Convert sigma to JND

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


def create_grid_of_plots(plot_files, output_path, title, n_rows=3, n_cols=3, labels=None):
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
