#!/usr/bin/env python3
"""
Create publication-ready plots for paper.
"""

import os
import sys
import math
import glob
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from scipy.optimize import curve_fit
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.gbf_helpers import load_gbf_with_success
from analysis.core.data_loader import DataLoader
from analysis.io.metadata import extract_metadata

MODELS = ["ABS1", "REL1", "REL2"]
PSE_GRID = [480, 500, 520]
JND_GRID = [20, 40, 60]
OFFSET = 500
COLORS = {"ABS1": "#1f77b4", "REL1": "#ff7f0e", "REL2": "#2ca02c"}



def load_group_data(model_name, pse, jnd):
    """Load all GBF data for a group."""
    script_dir = Path(__file__).parent.parent
    group_dir = script_dir / "data" / "output" / "sim_gridrnd" / model_name / f"group_{pse}_{jnd}"
    
    gbf_files = sorted(list(group_dir.glob(f"S*_G*_*_{model_name}.txt")))
    
    all_rows = []
    for gbf_file in gbf_files:
        try:
            rows = load_gbf_with_success(str(gbf_file), OFFSET)
            all_rows.extend(rows)
        except Exception as e:
            print(f"Warning: {gbf_file.name}: {e}")
            continue
    
    return all_rows

def fit_and_plot_psychometric(ax, stimuli, responses, model_name):
    """Fit and plot psychometric curve for ONE model."""
    if len(stimuli) == 0:
        return
    
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
    
    try:
        stim_range = max(stimuli) - min(stimuli)
        p0 = [OFFSET, stim_range / 10]
        bounds = ([OFFSET - stim_range, 1], [OFFSET + stim_range, stim_range])
        mu, sigma = curve_fit(norm.cdf, bins, f, p0=p0, bounds=bounds, maxfev=10000)[0]
    except Exception as e:
        print(f"Fit failed for {model_name}: {e}")
        mu = np.mean(stimuli)
        sigma = np.std(stimuli)
    
    jnd = sigma * 0.6745
    x_fit = np.linspace(min(stimuli) - 50, max(stimuli) + 50, 200)
    ax.plot(x_fit, norm.cdf(x_fit, mu, sigma), color=COLORS[model_name], linewidth=2, 
            label=f'{model_name} (JND={jnd:.1f})')
    ax.plot(bins, f, 'o', color=COLORS[model_name], markersize=3.5, alpha=0.6)

def create_grid_psychometric():
    """Create 3x3 grid - publication ready (16cm x 16cm)."""
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "data" / "output" / "paper_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert cm to inches: 16 cm = 6.3 inches
    fig, axes = plt.subplots(3, 3, figsize=(6.3, 6.3), dpi=300)
    fig.suptitle('Psychometric Functions by PSE/JND Grid', fontsize=11, fontweight='bold', y=0.98)
    
    grid = list(product(PSE_GRID, JND_GRID))
    
    for idx, (pse, jnd) in enumerate(grid):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        for model_name in MODELS:
            try:
                rows = load_group_data(model_name, pse, jnd)
                if not rows:
                    continue
                
                stimuli = [row['lat'] for row in rows]
                responses = [row['user_ans'] for row in rows]
                fit_and_plot_psychometric(ax, stimuli, responses, model_name)
                
            except Exception as e:
                print(f"Error: {model_name} {pse}/{jnd}: {e}")
        
        ax.axvline(OFFSET, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'PSE={pse}, JND={jnd}', fontsize=9, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.legend(fontsize=6.5, loc='upper left', framealpha=0.95)
        ax.tick_params(labelsize=6.5)
    
    plt.tight_layout()
    output_path = output_dir / "grid_psychometric.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def create_latency_envelope_grid():
    """Create 3x3 grid of latency KDE envelopes by PSE/JND - publication ready (16cm x 16cm)."""
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "data" / "output" / "paper_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert cm to inches: 16 cm = 6.3 inches
    fig, axes = plt.subplots(3, 3, figsize=(6.3, 5.0), dpi=300)
    fig.suptitle('Latencies distribution by PSE/JND', fontsize=11, fontweight='bold', y=0.98)
    
    grid = list(product(PSE_GRID, JND_GRID))
    
    # Collect handles and labels for shared legend
    handles = []
    labels = []
    
    for idx, (pse, jnd) in enumerate(grid):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        for model_name in MODELS:
            try:
                rows = load_group_data(model_name, pse, jnd)
                if not rows:
                    continue
                
                latencies = np.array([row['lat'] for row in rows])
                
                if len(latencies) < 5:
                    continue
                
                # Compute KDE
                kde = gaussian_kde(latencies, bw_method='scott')
                x_kde = np.linspace(latencies.min() - 20, latencies.max() + 20, 300)
                kde_vals = kde(x_kde)
                
                # Normalize KDE to match histogram scale (bin_width=10ms)
                bin_width = 10
                kde_vals_scaled = kde_vals * len(latencies) * bin_width
                
                line, = ax.plot(x_kde, kde_vals_scaled, color=COLORS[model_name], linewidth=2, 
                                label=model_name, alpha=0.8)
                
                # Collect handles/labels only once per model
                if model_name not in labels:
                    handles.append(line)
                    labels.append(model_name)
                
            except Exception as e:
                print(f"Error: {model_name} {pse}/{jnd}: {e}")
        
        ax.set_title(f'PSE={pse}, JND={jnd}', fontsize=6, fontweight='bold')
        ax.axvline(500, color='gray', linestyle='--', linewidth=1, alpha=0.85)
        ax.set_xlim(250, 750)
        ax.set_ylim(0, 700)
        ax.set_xticks([300, 400, 500, 600, 700])
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.tick_params(labelsize=6.5)
        
        # X-axis labels only on bottom row (row 2)
        if row < 2:
            ax.set_xticklabels([])
        
        # Y-axis labels only on left column (col 0)
        if col > 0:
            ax.set_yticklabels([])
    
    # Add shared legend at top
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=8, 
               framealpha=0.95, bbox_to_anchor=(0.5, 0.95))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = output_dir / "grid_latency_envelope.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def create_latency_distribution_grid():
    """Create 2x2 grid of latency distributions by modality and algorithm with stacked success coloring."""
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data" / "input" / "expdata"
    output_dir = script_dir / "data" / "output" / "paper_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all real data files
    pattern = str(data_dir / "*.txt")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"WARNING: No data files found in {data_dir}")
        return None
    
    # Organize data by modality and algorithm (store both latencies and successes)
    data_by_condition = {
        ('Auditory', 'Adaptive'): {'latencies': [], 'successes': []},
        ('Auditory', 'Fixed'): {'latencies': [], 'successes': []},
        ('Visual', 'Adaptive'): {'latencies': [], 'successes': []},
        ('Visual', 'Fixed'): {'latencies': [], 'successes': []}
    }
    
    # Load all data
    for filepath in files:
        try:
            metadata = extract_metadata(filepath)
            if not metadata.valid:
                continue
            
            # Map modality codes to names
            modality_map = {'BISA': 'Auditory', 'BISV': 'Visual'}
            algorithm_map = {'AD': 'Adaptive', 'FX': 'Fixed'}
            
            modality = modality_map.get(metadata.modality, metadata.modality)
            algorithm = algorithm_map.get(metadata.algorithm, metadata.algorithm)
            
            # Load latencies and responses
            latencies, responses = DataLoader.load_and_expand(filepath)
            
            # Calculate success: if lat > 500, success=1 if user_ans==1; if lat < 500, success=1 if user_ans==0
            successes = np.where(latencies > 500, responses == 1, responses == 0).astype(int)
            
            key = (modality, algorithm)
            if key in data_by_condition:
                data_by_condition[key]['latencies'].extend(latencies.tolist())
                data_by_condition[key]['successes'].extend(successes.tolist())
        
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
            continue
    
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    fig.suptitle('Latency Distribution by Modality and Algorithm', fontsize=14, fontweight='bold')
    
    conditions = [
        ('Auditory', 'Adaptive'),
        ('Auditory', 'Fixed'),
        ('Visual', 'Adaptive'),
        ('Visual', 'Fixed')
    ]
    
    for idx, (modality, algorithm) in enumerate(conditions):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        latencies = np.array(data_by_condition[(modality, algorithm)]['latencies'])
        successes = np.array(data_by_condition[(modality, algorithm)]['successes'])
        
        if len(latencies) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{modality} - {algorithm}', fontsize=12, fontweight='bold')
            continue
        
        # Create bins
        bins = np.linspace(latencies.min() - 10, latencies.max() + 10, 31)
        
        # Count successes and failures in each bin
        hist_success, _ = np.histogram(latencies[successes == 1], bins=bins)
        hist_failure, _ = np.histogram(latencies[successes == 0], bins=bins)
        
        # Plot stacked histogram: red (failure) at bottom, green (success) on top
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]
        
        ax.bar(bin_centers, hist_failure, width=bin_width, color='#d62728', alpha=0.7, 
               edgecolor='black', linewidth=0.5, label='Failure')
        ax.bar(bin_centers, hist_success, width=bin_width, bottom=hist_failure, 
               color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=0.5, label='Success')
        
        # Add statistics
        sc = np.mean(latencies)  # Stimulus Center
        ss = np.std(latencies)   # Stimulus Spread
        
        stats_text = f'SC={sc:.1f}\nSS={ss:.1f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Latency (ms)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{modality} - {algorithm}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=9, loc='upper left')
    
    plt.tight_layout()
    output_path = output_dir / "latency_distribution_grid.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    return data_by_condition

def create_figure7_combined():
    """Create combined Figure 7: latency distribution (left 2x2) + entropy evolution (right 2x1)."""
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data" / "input" / "expdata"
    output_dir = script_dir / "data" / "output" / "paper_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all real data files for latency distribution
    pattern = str(data_dir / "*.txt")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"WARNING: No data files found in {data_dir}")
        return
    
    # Organize data by modality and algorithm
    data_by_condition = {
        ('Auditory', 'Adaptive'): {'latencies': [], 'successes': []},
        ('Auditory', 'Fixed'): {'latencies': [], 'successes': []},
        ('Visual', 'Adaptive'): {'latencies': [], 'successes': []},
        ('Visual', 'Fixed'): {'latencies': [], 'successes': []}
    }
    
    # Load all data
    for filepath in files:
        try:
            metadata = extract_metadata(filepath)
            if not metadata.valid:
                continue
            
            modality_map = {'BISA': 'Auditory', 'BISV': 'Visual'}
            algorithm_map = {'AD': 'Adaptive', 'FX': 'Fixed'}
            
            modality = modality_map.get(metadata.modality, metadata.modality)
            algorithm = algorithm_map.get(metadata.algorithm, metadata.algorithm)
            
            latencies, responses = DataLoader.load_and_expand(filepath)
            successes = np.where(latencies > 500, responses == 1, responses == 0).astype(int)
            
            key = (modality, algorithm)
            if key in data_by_condition:
                data_by_condition[key]['latencies'].extend(latencies.tolist())
                data_by_condition[key]['successes'].extend(successes.tolist())
        
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
            continue
    
    # Load real data from Excel (progressive format with n_trials)
    import pandas as pd
    excel_path = script_dir / "data" / "input" / "results_BIS_fx_vs_ad_td_2model_rel_logistic_prog_long.xlsx"
    
    df_real = None
    if excel_path.exists():
        try:
            df_real = pd.read_excel(excel_path)
            df_real.columns = df_real.columns.str.lower().str.strip()
            
            if 'modality' in df_real.columns:
                modality_map = {'BISA': 'Auditory', 'BISV': 'Visual', 'Auditory': 'Auditory', 'Visual': 'Visual'}
                df_real['modality'] = df_real['modality'].map(lambda x: modality_map.get(x, x))
            
            if 'algorithm' in df_real.columns:
                algorithm_map = {'AD': 'Adaptive', 'FX': 'Fixed', 'Adaptive': 'Adaptive', 'Fixed': 'Fixed'}
                df_real['algorithm'] = df_real['algorithm'].map(lambda x: algorithm_map.get(x, x))
        except Exception as e:
            print(f"Warning: Failed to load Excel file: {e}")
            df_real = None
    
    # Create figure with proper layout
    fig = plt.figure(figsize=(6.3, 3.94), dpi=300)  # 16cm x 10cm at 300 DPI
    
    # Manual positioning: left 2x2 grid tight, right 2x1 grid separated
    # Left side: columns 0-0.55 (2x2 latency distribution) - wider
    # Right side: columns 0.60-1 (2x1 entropy evolution) - narrower
    
    ax_left_tl = fig.add_axes([0.05, 0.54, 0.25, 0.38])  # Col 1 Top (Auditory Adaptive) - wider
    ax_left_tr = fig.add_axes([0.32, 0.54, 0.25, 0.38])  # Col 2 Top (Auditory Fixed) - wider
    ax_left_bl = fig.add_axes([0.05, 0.06, 0.25, 0.38])  # Col 1 Bottom (Visual Adaptive) - wider
    ax_left_br = fig.add_axes([0.32, 0.06, 0.25, 0.38])  # Col 2 Bottom (Visual Fixed) - wider
    
    ax_right_top = fig.add_axes([0.65, 0.54, 0.26, 0.38])  # Col 3 Top (Entropy Auditory) - narrower
    ax_right_bottom = fig.add_axes([0.65, 0.06, 0.26, 0.38])  # Col 3 Bottom (Entropy Visual) - narrower
    
    # LEFT SIDE: 2x2 latency distribution
    conditions = [
        ('Auditory', 'Adaptive'),
        ('Auditory', 'Fixed'),
        ('Visual', 'Adaptive'),
        ('Visual', 'Fixed')
    ]
    
    left_axes = [ax_left_tl, ax_left_tr, ax_left_bl, ax_left_br]
    
    for ax, (modality, algorithm) in zip(left_axes, conditions):
        latencies = np.array(data_by_condition[(modality, algorithm)]['latencies'])
        successes = np.array(data_by_condition[(modality, algorithm)]['successes'])
        
        if len(latencies) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{modality} - {algorithm}', fontsize=10, fontweight='bold')
            continue
        
        bins = np.linspace(latencies.min() - 10, latencies.max() + 10, 31)
        hist_success, _ = np.histogram(latencies[successes == 1], bins=bins)
        hist_failure, _ = np.histogram(latencies[successes == 0], bins=bins)
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]
        
        ax.bar(bin_centers, hist_failure, width=bin_width, color='#d62728', alpha=0.7, 
               edgecolor='black', linewidth=0.5, label='Failure')
        ax.bar(bin_centers, hist_success, width=bin_width, bottom=hist_failure, 
               color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=0.5, label='Success')
        
        sc = np.mean(latencies)
        ss = np.std(latencies)
        
        stats_text = f'SC={sc:.1f}\nSS={ss:.1f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                fontsize=5, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.2))
        
        ax.set_xlabel('Latency (ms)', fontsize=7)
        ax.set_ylabel('Frequency', fontsize=7)
        ax.set_title(f'{modality} - {algorithm}', fontsize=8, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=6)
        
        # Set identical X-axis limits and ticks for all latency plots
        ax.set_xlim(250, 750)
        ax.set_xticks([300, 400, 500, 600, 700])
        
        # Remove X-axis label and ticks from top row (first row)
        if modality == 'Auditory':
            ax.set_xlabel('')
            ax.set_xticklabels([])
        
        # Remove Y-axis label and ticks from column 2 (Fixed algorithm)
        if algorithm == 'Fixed':
            ax.set_ylabel('')
            ax.set_yticklabels([])
        
        if (modality, algorithm) == conditions[0]:
            ax.legend(fontsize=5, loc='upper left')
    
    # Add subtitle for left side (centered between col 1 and 2)
    fig.text(0.28, 0.98, 'Latency Distribution', fontsize=9, fontweight='bold', ha='center')
    
    # Add subtitle for right side (centered on col 3)
    fig.text(0.78, 0.98, 'Entropy Evolution', fontsize=9, fontweight='bold', ha='center')
    
    # RIGHT SIDE: 2x1 entropy evolution
    colors_algorithm = {'Adaptive': '#CC3333', 'Fixed': '#3333CC'}
    
    if df_real is not None:
        for ax, modality in zip([ax_right_top, ax_right_bottom], ['Auditory', 'Visual']):
            df_mod = df_real[df_real['modality'] == modality]
            
            if df_mod.empty:
                ax.text(0.5, 0.5, f'No data', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(f'{modality}', fontsize=10, fontweight='bold')
                continue
            
            # Plot individual subject lines
            for subj in df_mod['subj'].unique():
                df_subj = df_mod[df_mod['subj'] == subj]
                for algo in ['Adaptive', 'Fixed']:
                    df_algo = df_subj[df_subj['algorithm'] == algo].sort_values('n_trials')
                    if len(df_algo) > 0 and 'lat_entropy' in df_algo.columns:
                        color = colors_algorithm.get(algo, '#999999')
                        ax.plot(df_algo['n_trials'], df_algo['lat_entropy'], 
                               color=color, alpha=0.15, linewidth=0.7)
            
            # Plot group means
            for algo in ['Adaptive', 'Fixed']:
                df_algo = df_mod[df_mod['algorithm'] == algo]
                if len(df_algo) > 0 and 'lat_entropy' in df_algo.columns:
                    group_mean = df_algo.groupby('n_trials')['lat_entropy'].mean()
                    color = colors_algorithm.get(algo, '#999999')
                    ax.plot(group_mean.index, group_mean.values, 
                           color=color, linewidth=2, marker='o', markersize=5, label=algo)
            
            ax.set_xlabel('Trials', fontsize=7)
            ax.set_ylabel('Entropy', fontsize=7)
            ax.set_title(f'{modality}', fontsize=8, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)
            
            # Remove X-axis label and ticks from top row
            if modality == 'Auditory':
                ax.set_xlabel('')
                ax.set_xticklabels([])
            
            if modality == 'Auditory':
                ax.legend(fontsize=7, loc='best')
    
    output_path = output_dir / "Figure7_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def main():
    print("Creating publication-ready plots...\n")
    print("Generating grid psychometric plot...")
    create_grid_psychometric()
    
    print("\nGenerating latency envelope grid...")
    create_latency_envelope_grid()
    
    print("\nGenerating latency distribution grid...")
    create_latency_distribution_grid()
    
    print("\nGenerating combined Figure 7...")
    create_figure7_combined()
    
    # ==============================================================================
    # FIGURE 3: Grid Psychometric
    # ==============================================================================
    print("\nCopying Figure 3 (Grid Psychometric)...")
    script_dir = Path(__file__).parent.parent
    paper_plots_dir = script_dir / "data" / "output" / "paper_plots"
    r_output_dir = Path("/data/Dropbox/RDATA/R_bis_ad_fx/results_model_comparison/plots")
    r_output_dir.mkdir(parents=True, exist_ok=True)
    
    grid_psychometric_src = paper_plots_dir / "grid_psychometric.png"
    grid_psychometric_dst = r_output_dir / "Figure3.png"
    
    if grid_psychometric_src.exists():
        import shutil
        shutil.copy(str(grid_psychometric_src), str(grid_psychometric_dst))
        print(f"✓ Saved: {grid_psychometric_dst}")
    else:
        print(f"WARNING: {grid_psychometric_src} not found")

    # ==============================================================================
    # FIGURE 5: Latency Envelope Grid
    # ==============================================================================
    print("\nCopying Figure 5 (Latency Envelope Grid)...")
    
    grid_latency_envelope_src = paper_plots_dir / "grid_latency_envelope.png"
    grid_latency_envelope_dst = r_output_dir / "Figure5.png"
    
    if grid_latency_envelope_src.exists():
        import shutil
        shutil.copy(str(grid_latency_envelope_src), str(grid_latency_envelope_dst))
        print(f"✓ Saved: {grid_latency_envelope_dst}")
    else:
        print(f"WARNING: {grid_latency_envelope_src} not found")


    # ==============================================================================
    # FIGURE S1: Models Stimuli Distribution (REL1/REL2 top, ABS1 centered bottom)
    # ==============================================================================
    grid_abs1_path = script_dir / "data" / "output" / "sim_gridrnd" / "ABS1" / "ABS1_stimulus_distribution_grid.png"
    grid_rel1_path = script_dir / "data" / "output" / "sim_gridrnd" / "REL1" / "REL1_stimulus_distribution_grid.png"
    grid_rel2_path = script_dir / "data" / "output" / "sim_gridrnd" / "REL2" / "REL2_stimulus_distribution_grid.png"

    print("\nCreating Figure S1 (Models Stimuli Distribution - REL top, ABS1 centered)...")
    
    if grid_abs1_path.exists() and grid_rel1_path.exists() and grid_rel2_path.exists():
        img_abs1 = Image.open(grid_abs1_path)
        img_rel1 = Image.open(grid_rel1_path)
        img_rel2 = Image.open(grid_rel2_path)

        yspace = 150

        # Keep all images at same size (no scaling)
        # Arrange: REL1 and REL2 side by side on top, ABS1 centered below
        total_width = img_rel1.width + img_rel2.width
        total_height = img_rel1.height + img_abs1.height + 150  # Extra space for title
        
        # Create combined image at 300 DPI
        combined = Image.new('RGB', (total_width, total_height), color='white')
        
        # Paste REL1 and REL2 side by side at top
        combined.paste(img_rel1, (0, yspace))
        combined.paste(img_rel2, (img_rel1.width, yspace))
        
        # Paste ABS1 centered at bottom
        abs_x = (total_width - img_abs1.width) // 2
        combined.paste(img_abs1, (abs_x, yspace + img_rel1.height))
        
        # Add title and model subtitles
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(combined)
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 50)
            font_subtitle = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            font_title = ImageFont.load_default()
            font_subtitle = ImageFont.load_default()
        
        # Main title
        title = "Models Stimuli Distribution"
        bbox = draw.textbbox((0, 0), title, font=font_title)
        text_width = bbox[2] - bbox[0]
        x = (total_width - text_width) // 2
        draw.text((x, 40), title, fill='black', font=font_title)
        
        # Model subtitles - centered above each subplot
        # REL1 centered above its subplot
        rel1_text = "REL1"
        bbox = draw.textbbox((0, 0), rel1_text, font=font_subtitle)
        text_width = bbox[2] - bbox[0]
        x = (img_rel1.width - text_width) // 2
        draw.text((x, 100), rel1_text, fill='black', font=font_subtitle)
        
        # REL2 centered above its subplot
        rel2_text = "REL2"
        bbox = draw.textbbox((0, 0), rel2_text, font=font_subtitle)
        text_width = bbox[2] - bbox[0]
        x = img_rel1.width + (img_rel2.width - text_width) // 2
        draw.text((x, 100), rel2_text, fill='black', font=font_subtitle)
        
        # ABS1 centered above its subplot
        abs1_text = "ABS1"
        bbox = draw.textbbox((0, 0), abs1_text, font=font_subtitle)
        text_width = bbox[2] - bbox[0]
        x = abs_x + (img_abs1.width - text_width) // 2
        draw.text((x, 100 + img_rel1.height), abs1_text, fill='black', font=font_subtitle)
        
        figureS1_path = r_output_dir / "FigureS1.png"
        combined.save(figureS1_path, dpi=(300, 300))
        print(f"✓ Saved: {figureS1_path}")
    else:
        print("WARNING: Could not create Figure S1 - missing grid images")
    
    # ==============================================================================
    # FIGURE 7: Combined Latency Distribution and Entropy Evolution
    # ==============================================================================
    print("\nCopying Figure 7 (Combined)...")
    
    figure7_combined_src = paper_plots_dir / "Figure7_combined.png"
    figure7_combined_dst = r_output_dir / "Figure7.png"
    
    if figure7_combined_src.exists():
        import shutil
        shutil.copy(str(figure7_combined_src), str(figure7_combined_dst))
        print(f"✓ Saved: {figure7_combined_dst}")
    else:
        print(f"WARNING: {figure7_combined_src} not found")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
