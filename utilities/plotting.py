import os
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

# Plot histogram of stimuli with success/failure coloring
def plot_stim_hist(rows, filepath, offset, subj):

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
    responses = [row['user_ans'] for row in all_rows]  # Use user_ans (0 or 1), not res (success/failure)

    if len(stimuli) == 0:
        print("No data to plot group psychometric")
        return

    # Bin the data
    binSize = 10
    SR = sorted(list(zip(stimuli, responses)))
    bins = [i * binSize for i in range(math.floor(SR[0][0] / binSize), math.ceil(SR[-1][0] / binSize) + 1)]
    x = [s[0] for s in SR]
    i_binned = np.digitize(x, bins)
    x_binned = np.asarray(bins)[i_binned - 1]
    r = np.asarray([sr[1] for sr in SR])
    f = [sum(r[x_binned == b]) / len(r[x_binned == b]) if len(r[x_binned == b]) > 0 else math.nan for b in bins]
    goodx = [not math.isnan(y) for y in f]
    f = np.asarray(f)[goodx]
    bins = np.asarray(bins)[goodx]

    # Fit cumulative Gaussian
    try:
        stim_range = max(stimuli) - min(stimuli)
        p0 = [offset, stim_range / 10]
        bounds = ([offset - stim_range, 1], [offset + stim_range, stim_range])
        mu1, sigma1 = curve_fit(norm.cdf, bins, f, p0=p0, bounds=bounds, maxfev=10000)[0]
    except Exception as e:
        print(f"Group psychometric fit failed: {e}")
        mu1 = np.mean(stimuli)
        sigma1 = np.std(stimuli)

    # Plot
    plt.figure(figsize=(10, 6))
    x_fit = np.linspace(min(stimuli) - 50, max(stimuli) + 50, 200)
    plt.plot(x_fit, norm.cdf(x_fit, mu1, sigma1), 'b-', linewidth=2, alpha=0.7, label=f'Fit: μ={mu1:.1f}, σ={sigma1:.1f}')
    plt.plot(bins, f, 'ro', markersize=8, alpha=0.7, label='Data')
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
