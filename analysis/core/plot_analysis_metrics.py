#!/usr/bin/env python3
"""
Plot analysis metrics from group simulation results.

Generates analysis plots organized by PSE:
1. asymmetry_modulo.png - PSE=500 only, 3 curves (one per JND)
2. asymmetry_scatter_envelope.png - 3 plots (one per JND), each with 2 envelopes
3. stimulus_center_evolution.png - 3 plots (one per PSE), 3 curves each (one per JND)
4. stimulus_spread_evolution.png - 3 plots (one per PSE), 3 curves each (one per JND)
5. bimodality_index_evolution.png - 3 plots (one per PSE), 3 curves each (one per JND)

Saves plots to: data/output/sim_gridrnd/{model}/
"""

import os
import sys
from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TRIAL_BLOCKS = [40, 60, 80, 100, 120, 140, 160, 180, 200]


def load_stimulus_metrics(model_name, pse_grid, jnd_grid, output_dir: Path = None):
    """Load stimulus metrics from all groups, organized by (PSE, JND).
    
    Automatically detects which groups actually exist in the output directory.
    """
    data = {}
    actual_pse_set = set()
    actual_jnd_set = set()
    
    if output_dir is None:
        output_dir = Path(f"data/output/sim_gridrnd/{model_name}")
    else:
        output_dir = Path(output_dir)
    
    # First pass: discover which groups actually exist
    for pse in pse_grid:
        for jnd in jnd_grid:
            group_name = f"group_{pse}_{jnd}"
            group_dir = output_dir / group_name
            results_dir = group_dir / "results"
            
            excel_files = list(results_dir.glob("*_results_summary.xlsx"))
            if not excel_files:
                continue
            
            excel_path = excel_files[0]
            
            try:
                df = pd.read_excel(excel_path)
                
                # Extract stimulus metrics
                stimulus_center_cols = [f'stimulus_center_{n}' for n in TRIAL_BLOCKS]
                stimulus_spread_cols = [f'stimulus_spread_{n}' for n in TRIAL_BLOCKS]
                bimodality_cols = [f'bimodality_index_{n}' for n in TRIAL_BLOCKS]
                asymmetry_cols = [f'asymmetry_{n}' for n in TRIAL_BLOCKS]
                
                # Check if columns exist
                if not all(col in df.columns for col in stimulus_center_cols):
                    continue
                
                # Filter out last 2 rows (mean and stddev)
                df = df.iloc[:-2]
                
                data[(pse, jnd)] = {
                    'stimulus_center': df[stimulus_center_cols].values,
                    'stimulus_spread': df[stimulus_spread_cols].values,
                    'bimodality_index': df[bimodality_cols].values,
                    'asymmetry': df[asymmetry_cols].values,
                }
                
                actual_pse_set.add(pse)
                actual_jnd_set.add(jnd)
                
            except Exception as e:
                continue
    
    # Store actual grids in data for later use
    data['_actual_pse_grid'] = sorted(list(actual_pse_set))
    data['_actual_jnd_grid'] = sorted(list(actual_jnd_set))
    
    return data


def plot_asymmetry_modulo(model_name, data, pse_grid, jnd_grid, output_dir: Path):
    """Plot |asymmetry_index| vs trial blocks for PSE=500 only, 3 curves (one per JND)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Find PSE=500 in the grid, or use the middle PSE if 500 doesn't exist
    pse_target = 500 if 500 in pse_grid else pse_grid[len(pse_grid)//2]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    plotted = False
    for jnd_idx, jnd in enumerate(jnd_grid):
        key = (pse_target, jnd)
        if key not in data:
            continue
        
        asymmetry = data[key]['asymmetry']
        means = np.mean(np.abs(asymmetry), axis=0)
        stds = np.std(np.abs(asymmetry), axis=0)
        
        color_idx = jnd_idx % len(colors)
        ax.plot(TRIAL_BLOCKS, means, marker='o', linewidth=2.5, markersize=8, 
                color=colors[color_idx], label=f'JND={jnd}')
        ax.fill_between(TRIAL_BLOCKS, means - stds, means + stds, alpha=0.2, color=colors[color_idx])
        plotted = True
    
    if plotted:
        ax.set_xlabel('Trial Block', fontsize=12, fontweight='bold')
        ax.set_ylabel('|Asymmetry Index|', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}: Evolution of |Asymmetry Index| (PSE={pse_target})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(TRIAL_BLOCKS)
        ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'asymmetry_modulo.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_asymmetry_scatter_envelope(model_name, data, pse_grid, jnd_grid, output_dir: Path):
    """Plot scatter of asymmetry values with envelope curves - 1 plot per JND."""
    actual_pse_grid = data.pop('_actual_pse_grid', pse_grid)
    actual_jnd_grid = data.pop('_actual_jnd_grid', jnd_grid)
    
    n_jnd = len(actual_jnd_grid)
    fig, axes = plt.subplots(1, n_jnd, figsize=(6*n_jnd, 5))
    
    # Handle single subplot case
    if n_jnd == 1:
        axes = [axes]
    
    for jnd_idx, jnd in enumerate(actual_jnd_grid):
        ax = axes[jnd_idx]
        
        # Collect data for all PSE values with this JND
        all_asymmetry_values = {block: [] for block in TRIAL_BLOCKS}
        
        for pse in actual_pse_grid:
            key = (pse, jnd)
            if key not in data:
                continue
            
            asymmetry = data[key]['asymmetry']
            
            for block_idx, block in enumerate(TRIAL_BLOCKS):
                all_asymmetry_values[block].extend(asymmetry[:, block_idx])
        
        # Plot scatter
        for block in TRIAL_BLOCKS:
            values = all_asymmetry_values[block]
            if values:
                x_jitter = np.random.normal(block, 1.5, len(values))
                ax.scatter(x_jitter, values, alpha=0.4, s=30, color='gray')
        
        # Plot envelopes
        positive_envelope = []
        negative_envelope = []
        
        for block in TRIAL_BLOCKS:
            values = all_asymmetry_values[block]
            if values:
                pos_vals = [v for v in values if v > 0]
                neg_vals = [v for v in values if v < 0]
                positive_envelope.append(np.max(pos_vals) if pos_vals else np.nan)
                negative_envelope.append(np.min(neg_vals) if neg_vals else np.nan)
            else:
                positive_envelope.append(np.nan)
                negative_envelope.append(np.nan)
        
        ax.plot(TRIAL_BLOCKS, positive_envelope, 'g-', linewidth=2.5, marker='s', markersize=6, label='Max (>0)')
        ax.plot(TRIAL_BLOCKS, negative_envelope, 'r-', linewidth=2.5, marker='^', markersize=6, label='Min (<0)')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Trial Block', fontsize=11, fontweight='bold')
        ax.set_ylabel('Asymmetry Index', fontsize=11, fontweight='bold')
        ax.set_title(f'JND={jnd}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xticks(TRIAL_BLOCKS)
    
    fig.suptitle(f'{model_name}: Asymmetry Index Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'asymmetry_scatter_envelope.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_stimulus_center_evolution(model_name, data, pse_grid, jnd_grid, output_dir: Path):
    """Plot stimulus center evolution - 1 plot per PSE, N curves per JND."""
    actual_pse_grid = data.pop('_actual_pse_grid', pse_grid)
    actual_jnd_grid = data.pop('_actual_jnd_grid', jnd_grid)
    
    n_pse = len(actual_pse_grid)
    fig, axes = plt.subplots(1, n_pse, figsize=(6*n_pse, 5))
    
    # Handle single subplot case
    if n_pse == 1:
        axes = [axes]
    
    # Calculate global y-axis limits
    all_means = []
    all_stds = []
    for pse in actual_pse_grid:
        for jnd in actual_jnd_grid:
            if (pse, jnd) in data:
                stim_center = data[(pse, jnd)]['stimulus_center']
                means = np.mean(stim_center, axis=0)
                stds = np.std(stim_center, axis=0)
                all_means.extend(means)
                all_stds.extend(stds)
    
    y_min = np.min(all_means) - np.max(all_stds) - 10
    y_max = np.max(all_means) + np.max(all_stds) + 10
    
    for pse_idx, pse in enumerate(actual_pse_grid):
        ax = axes[pse_idx]
        
        for jnd in actual_jnd_grid:
            if (pse, jnd) not in data:
                continue
            
            stim_center = data[(pse, jnd)]['stimulus_center']
            means = np.mean(stim_center, axis=0)
            stds = np.std(stim_center, axis=0)
            
            ax.plot(TRIAL_BLOCKS, means, marker='o', linewidth=2.5, label=f'JND={jnd}')
            ax.fill_between(TRIAL_BLOCKS, means - stds, means + stds, alpha=0.2)
        
        # Add PSE reference line
        ax.axhline(y=pse, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'PSE={pse}')
        
        ax.set_xlabel('Trial Block', fontsize=11, fontweight='bold')
        ax.set_ylabel('Stimulus Center (ms)', fontsize=11, fontweight='bold')
        ax.set_title(f'PSE={pse}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xticks(TRIAL_BLOCKS)
        ax.set_ylim([y_min, y_max])
    
    fig.suptitle(f'{model_name}: Stimulus Center Evolution (should match PSE)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'stimulus_center_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_stimulus_spread_evolution(model_name, data, pse_grid, jnd_grid, output_dir: Path):
    """Plot stimulus spread evolution - 1 plot per PSE, N curves per JND."""
    actual_pse_grid = data.pop('_actual_pse_grid', pse_grid)
    actual_jnd_grid = data.pop('_actual_jnd_grid', jnd_grid)
    
    n_pse = len(actual_pse_grid)
    fig, axes = plt.subplots(1, n_pse, figsize=(6*n_pse, 5))
    
    # Handle single subplot case
    if n_pse == 1:
        axes = [axes]
    
    # Calculate global y-axis limits
    all_means = []
    all_stds = []
    for pse in actual_pse_grid:
        for jnd in actual_jnd_grid:
            if (pse, jnd) in data:
                stim_spread = data[(pse, jnd)]['stimulus_spread']
                means = np.mean(stim_spread, axis=0)
                stds = np.std(stim_spread, axis=0)
                all_means.extend(means)
                all_stds.extend(stds)
    
    y_min = max(0, np.min(all_means) - np.max(all_stds) - 5)
    y_max = np.max(all_means) + np.max(all_stds) + 5
    
    for pse_idx, pse in enumerate(actual_pse_grid):
        ax = axes[pse_idx]
        
        for jnd in actual_jnd_grid:
            if (pse, jnd) not in data:
                continue
            
            stim_spread = data[(pse, jnd)]['stimulus_spread']
            means = np.mean(stim_spread, axis=0)
            stds = np.std(stim_spread, axis=0)
            
            ax.plot(TRIAL_BLOCKS, means, marker='s', linewidth=2.5, label=f'JND={jnd}')
            ax.fill_between(TRIAL_BLOCKS, means - stds, means + stds, alpha=0.2)
        
        ax.set_xlabel('Trial Block', fontsize=11, fontweight='bold')
        ax.set_ylabel('Stimulus Spread (ms)', fontsize=11, fontweight='bold')
        ax.set_title(f'PSE={pse}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xticks(TRIAL_BLOCKS)
        ax.set_ylim([y_min, y_max])
    
    fig.suptitle(f'{model_name}: Stimulus Spread Evolution (should increase with JND)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'stimulus_spread_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_bimodality_index_evolution(model_name, data, pse_grid, jnd_grid, output_dir: Path):
    """Plot bimodality index evolution - 1 plot per PSE, N curves per JND."""
    actual_pse_grid = data.pop('_actual_pse_grid', pse_grid)
    actual_jnd_grid = data.pop('_actual_jnd_grid', jnd_grid)
    
    n_pse = len(actual_pse_grid)
    fig, axes = plt.subplots(1, n_pse, figsize=(6*n_pse, 5))
    
    # Handle single subplot case
    if n_pse == 1:
        axes = [axes]
    
    for pse_idx, pse in enumerate(actual_pse_grid):
        ax = axes[pse_idx]
        
        for jnd in actual_jnd_grid:
            if (pse, jnd) not in data:
                continue
            
            bimodality = data[(pse, jnd)]['bimodality_index']
            means = np.mean(bimodality, axis=0)
            stds = np.std(bimodality, axis=0)
            
            ax.plot(TRIAL_BLOCKS, means, marker='^', linewidth=2.5, label=f'JND={jnd}')
            ax.fill_between(TRIAL_BLOCKS, means - stds, means + stds, alpha=0.2)
        
        ax.set_xlabel('Trial Block', fontsize=11, fontweight='bold')
        ax.set_ylabel('Bimodality Index', fontsize=11, fontweight='bold')
        ax.set_title(f'PSE={pse}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xticks(TRIAL_BLOCKS)
        ax.set_ylim([0, 1])
    
    fig.suptitle(f'{model_name}: Bimodality Index Evolution (higher JND → more bimodal)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'bimodality_index_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_analysis_metrics(model_name: str, output_dir: Path, pse_grid, jnd_grid):
    """Generate all 5 analysis metric plots."""
    data = load_stimulus_metrics(model_name, pse_grid, jnd_grid, output_dir)
    
    if not data:
        print(f"No data found for {model_name}")
        return
    
    plot_asymmetry_modulo(model_name, data, pse_grid, jnd_grid, output_dir)
    plot_asymmetry_scatter_envelope(model_name, data, pse_grid, jnd_grid, output_dir)
    plot_stimulus_center_evolution(model_name, data, pse_grid, jnd_grid, output_dir)
    plot_stimulus_spread_evolution(model_name, data, pse_grid, jnd_grid, output_dir)
    plot_bimodality_index_evolution(model_name, data, pse_grid, jnd_grid, output_dir)
