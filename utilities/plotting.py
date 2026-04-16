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


def plot_group_histograms(all_rows_list, output_dir, file_prefix, offset):
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
    plt.title('Group stimulus distribution (all subjects)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_filename = os.path.join(output_dir, f"{file_prefix}_group_histogram.png")
    plt.savefig(plot_filename, bbox_inches='tight', dpi=100)
    plt.close()

def plot_group_model_histograms(all_rows_list, output_dir, file_prefix, offset):

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
    plt.title('Group stimulus distribution by model (all subjects)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_filename2 = os.path.join(output_dir, f"{file_prefix}_group_model_histogram.png")
    plt.savefig(plot_filename2, bbox_inches='tight', dpi=100)
    plt.close()

    print(f"Generated group histograms in {output_dir}")


def plot_group_psychometric(all_rows_list, output_dir, file_prefix, offset):
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
    plt.title('Group psychometric function (all subjects)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_filename = os.path.join(output_dir, f"{file_prefix}_group_psychometric.png")
    plt.savefig(plot_filename, bbox_inches='tight', dpi=100)
    plt.close()

    print(f"Generated group psychometric: {plot_filename}")
