#!/usr/bin/env python3
"""
Unified analysis plots and metrics export generator.

Consolidates:
- Analysis plots (asymmetry, stimulus metrics, entropy)
- CSV export for R analysis

Single entry point for post-simulation analysis.
Designed to be called from group_sim_gridrnd_*.py after GBF generation.

Usage:
    from analysis.core.generate_analysis_plots import generate_all_analysis
    
    generate_all_analysis(
        model_name="ABS1",
        output_dir=Path("data/output/sim_gridrnd/ABS1"),
        pse_grid=[480, 500, 520],
        jnd_grid=[20, 40, 60],
        export_csv=True,
        csv_output_dir=Path("data/output/stimulus_metrics_for_r")
    )
"""

import sys
from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

TRIAL_BLOCKS = [40, 60, 80, 100, 120, 140, 160, 180, 200]


class AnalysisPlotGenerator:
    """Generate analysis plots from simulation results."""
    
    def __init__(self, model_name: str, output_dir: Path, pse_grid: list, jnd_grid: list):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.pse_grid = pse_grid
        self.jnd_grid = jnd_grid
        self.data = {}
        self.actual_pse_grid = []
        self.actual_jnd_grid = []
    
    def load_stimulus_metrics(self):
        """Load stimulus metrics from all groups."""
        for pse in self.pse_grid:
            for jnd in self.jnd_grid:
                group_name = f"group_{pse}_{jnd}"
                group_dir = self.output_dir / group_name
                results_dir = group_dir / "results"
                
                excel_files = list(results_dir.glob("*_results_summary.xlsx"))
                if not excel_files:
                    continue
                
                try:
                    df = pd.read_excel(excel_files[0])
                    
                    # Check for required columns
                    stimulus_center_cols = [f'stimulus_center_{n}' for n in TRIAL_BLOCKS]
                    if not all(col in df.columns for col in stimulus_center_cols):
                        continue
                    
                    # Filter out summary rows (last 2 rows)
                    df = df.iloc[:-2]
                    
                    stimulus_spread_cols = [f'stimulus_spread_{n}' for n in TRIAL_BLOCKS]
                    asymmetry_cols = [f'asymmetry_{n}' for n in TRIAL_BLOCKS]
                    
                    self.data[(pse, jnd)] = {
                        'stimulus_center': df[stimulus_center_cols].values,
                        'stimulus_spread': df[stimulus_spread_cols].values,
                        'asymmetry': df[asymmetry_cols].values,
                    }
                    
                    self.actual_pse_grid.append(pse)
                    self.actual_jnd_grid.append(jnd)
                    
                except Exception as e:
                    logger.warning(f"Failed to load {group_name}: {e}")
                    continue
        
        self.actual_pse_grid = sorted(list(set(self.actual_pse_grid)))
        self.actual_jnd_grid = sorted(list(set(self.actual_jnd_grid)))
        
        return len(self.data) > 0
    
    def plot_asymmetry_modulo(self):
        """Plot |asymmetry_index| vs trial blocks for PSE=500 (or middle PSE)."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        pse_target = 500 if 500 in self.actual_pse_grid else self.actual_pse_grid[len(self.actual_pse_grid)//2]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        plotted = False
        for jnd_idx, jnd in enumerate(self.actual_jnd_grid):
            key = (pse_target, jnd)
            if key not in self.data:
                continue
            
            asymmetry = self.data[key]['asymmetry']
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
            ax.set_title(f'{self.model_name}: Evolution of |Asymmetry Index| (PSE={pse_target})', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(TRIAL_BLOCKS)
            ax.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'asymmetry_modulo.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_asymmetry_scatter_envelope(self):
        """Plot scatter of asymmetry values with envelope curves - 1 plot per JND."""
        n_jnd = len(self.actual_jnd_grid)
        fig, axes = plt.subplots(1, n_jnd, figsize=(6*n_jnd, 5))
        
        if n_jnd == 1:
            axes = [axes]
        
        for jnd_idx, jnd in enumerate(self.actual_jnd_grid):
            ax = axes[jnd_idx]
            
            all_asymmetry_values = {block: [] for block in TRIAL_BLOCKS}
            
            for pse in self.actual_pse_grid:
                key = (pse, jnd)
                if key not in self.data:
                    continue
                
                asymmetry = self.data[key]['asymmetry']
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
        
        fig.suptitle(f'{self.model_name}: Asymmetry Index Distribution', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'asymmetry_scatter_envelope.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_stimulus_center_evolution(self):
        """Plot stimulus center evolution - single plot with 3 curves (one per PSE), averaged over JND."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate global y-axis limits
        all_means = []
        all_stds = []
        for pse in self.actual_pse_grid:
            for jnd in self.actual_jnd_grid:
                if (pse, jnd) in self.data:
                    stim_center = self.data[(pse, jnd)]['stimulus_center']
                    means = np.mean(stim_center, axis=0)
                    stds = np.std(stim_center, axis=0)
                    all_means.extend(means)
                    all_stds.extend(stds)
        
        y_min = np.min(all_means) - np.max(all_stds) - 10
        y_max = np.max(all_means) + np.max(all_stds) + 10
        
        # Plot one curve per PSE, averaged over all JND values
        for pse in self.actual_pse_grid:
            pse_data = []
            for jnd in self.actual_jnd_grid:
                if (pse, jnd) in self.data:
                    stim_center = self.data[(pse, jnd)]['stimulus_center']
                    means = np.mean(stim_center, axis=0)
                    pse_data.append(means)
            
            if pse_data:
                # Average across all JND values
                avg_means = np.mean(pse_data, axis=0)
                stds = np.std(pse_data, axis=0)
                
                ax.plot(TRIAL_BLOCKS, avg_means, marker='o', linewidth=2.5, label=f'PSE={pse}')
                ax.fill_between(TRIAL_BLOCKS, avg_means - stds, avg_means + stds, alpha=0.2)
                ax.axhline(y=pse, color='red', linestyle='--', linewidth=1.5, alpha=0.3)
        
        ax.set_xlabel('Trial Block', fontsize=11, fontweight='bold')
        ax.set_ylabel('Stimulus Center (ms)', fontsize=11, fontweight='bold')
        ax.set_title(f'{self.model_name}: Stimulus Center Evolution (averaged over JND)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xticks(TRIAL_BLOCKS)
        ax.set_ylim([y_min, y_max])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stimulus_center_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_stimulus_spread_evolution(self):
        """Plot stimulus spread evolution - single plot with 3 curves (one per JND), averaged over PSE."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate global y-axis limits
        all_means = []
        all_stds = []
        for pse in self.actual_pse_grid:
            for jnd in self.actual_jnd_grid:
                if (pse, jnd) in self.data:
                    stim_spread = self.data[(pse, jnd)]['stimulus_spread']
                    means = np.mean(stim_spread, axis=0)
                    stds = np.std(stim_spread, axis=0)
                    all_means.extend(means)
                    all_stds.extend(stds)
        
        y_min = max(0, np.min(all_means) - np.max(all_stds) - 5)
        y_max = np.max(all_means) + np.max(all_stds) + 5
        
        # Plot one curve per JND, averaged over all PSE values
        for jnd in self.actual_jnd_grid:
            jnd_data = []
            for pse in self.actual_pse_grid:
                if (pse, jnd) in self.data:
                    stim_spread = self.data[(pse, jnd)]['stimulus_spread']
                    means = np.mean(stim_spread, axis=0)
                    jnd_data.append(means)
            
            if jnd_data:
                # Average across all PSE values
                avg_means = np.mean(jnd_data, axis=0)
                stds = np.std(jnd_data, axis=0)
                
                ax.plot(TRIAL_BLOCKS, avg_means, marker='s', linewidth=2.5, label=f'JND={jnd}')
                ax.fill_between(TRIAL_BLOCKS, avg_means - stds, avg_means + stds, alpha=0.2)
        
        ax.set_xlabel('Trial Block', fontsize=11, fontweight='bold')
        ax.set_ylabel('Stimulus Spread (ms)', fontsize=11, fontweight='bold')
        ax.set_title(f'{self.model_name}: Stimulus Spread Evolution (averaged over PSE)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xticks(TRIAL_BLOCKS)
        ax.set_ylim([y_min, y_max])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stimulus_spread_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_plots(self):
        """Generate all 4 analysis metric plots."""
        if not self.load_stimulus_metrics():
            logger.warning(f"No stimulus metrics found for {self.model_name}")
            return False
        
        self.plot_asymmetry_modulo()
        self.plot_asymmetry_scatter_envelope()
        self.plot_stimulus_center_evolution()
        self.plot_stimulus_spread_evolution()
        
        return True


class CSVExporter:
    """Export stimulus metrics to CSV for R analysis."""
    
    def __init__(self, pse_grid: list, jnd_grid: list):
        self.pse_grid = pse_grid
        self.jnd_grid = jnd_grid
    
    def export_model(self, model_name: str, data_root: Path) -> pd.DataFrame:
        """Export stimulus metrics for a single model."""
        rows_list = []
        
        for pse in self.pse_grid:
            for jnd in self.jnd_grid:
                group_name = f"group_{pse}_{jnd}"
                group_dir = data_root / model_name / group_name
                results_dir = group_dir / "results"
                
                excel_files = list(results_dir.glob("*_results_summary.xlsx"))
                if not excel_files:
                    continue
                
                try:
                    df = pd.read_excel(excel_files[0])
                    
                    # Filter out summary rows
                    if 'subj' in df.columns:
                        df = df[~df['subj'].astype(str).str.lower().isin(['mean', 'group_mean', 'group_std', 'stddev', 'std'])]
                    
                    # Check for required columns
                    stimulus_center_cols = [f'stimulus_center_{n}' for n in TRIAL_BLOCKS]
                    if not all(col in df.columns for col in stimulus_center_cols):
                        continue
                    
                    stimulus_spread_cols = [f'stimulus_spread_{n}' for n in TRIAL_BLOCKS]
                    asymmetry_cols = [f'asymmetry_{n}' for n in TRIAL_BLOCKS]
                    lat_entropy_cols = [f'lat_entropy_{n}' for n in TRIAL_BLOCKS]
                    pse_cols = [f'pse_{n}' for n in TRIAL_BLOCKS]
                    jnd_cols = [f'jnd_{n}' for n in TRIAL_BLOCKS]
                    
                    for idx, row in df.iterrows():
                        subject_id = row.get('subj') if 'subj' in df.columns else row.get('subject_id')
                        
                        if pd.isna(subject_id):
                            continue
                        
                        subject_id_str = str(subject_id).lower()
                        if subject_id_str in ['mean', 'group_mean', 'group_std', 'stddev', 'std', 'nan']:
                            continue
                        
                        pse_subj = row.get('pse', pse)
                        jnd_subj = row.get('jnd', jnd)
                        
                        # Extract group from subject_id
                        group = None
                        if isinstance(subject_id, str) and '_' in subject_id:
                            parts = subject_id.split('_')
                            if len(parts) >= 2:
                                group = parts[1]
                        
                        for n_trials in TRIAL_BLOCKS:
                            row_dict = {
                                'model': model_name,
                                'pse_true': pse_subj,
                                'jnd_true': jnd_subj,
                                'subject_id': subject_id,
                                'group': group,
                                'trial_block': n_trials,
                                'stimulus_center': row.get(f'stimulus_center_{n_trials}', np.nan),
                                'stimulus_spread': row.get(f'stimulus_spread_{n_trials}', np.nan),
                                'lat_entropy': row.get(f'lat_entropy_{n_trials}', np.nan),
                            }
                            
                            if f'pse_{n_trials}' in df.columns:
                                row_dict['pse_est'] = row.get(f'pse_{n_trials}', np.nan)
                            if f'jnd_{n_trials}' in df.columns:
                                row_dict['jnd_est'] = row.get(f'jnd_{n_trials}', np.nan)
                            if f'asymmetry_{n_trials}' in df.columns:
                                row_dict['asymmetry_index'] = row.get(f'asymmetry_{n_trials}', np.nan)
                            
                            rows_list.append(row_dict)
                
                except Exception as e:
                    logger.warning(f"Failed to process {group_name}: {e}")
                    continue
        
        return pd.DataFrame(rows_list)
    
    def export_all_models(self, models: list, data_root: Path, output_path: Path) -> bool:
        """Export all models to single CSV."""
        all_data = []
        
        for model_name in models:
            df = self.export_model(model_name, data_root)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            logger.error("No data exported from any model")
            return False
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(
            subset=['model', 'pse_true', 'jnd_true', 'subject_id', 'trial_block'],
            keep='first'
        )
        
        # Round numeric columns
        numeric_cols = ['stimulus_center', 'stimulus_spread', 'lat_entropy', 'pse_est', 'jnd_est']
        for col in numeric_cols:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].round(2)
        
        if 'asymmetry_index' in combined_df.columns:
            combined_df['asymmetry_index'] = combined_df['asymmetry_index'].round(3)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(combined_df)} rows to {output_path}")
        return True


def generate_all_analysis(
    model_name: str,
    output_dir: Path,
    pse_grid: list,
    jnd_grid: list,
    export_csv: bool = False,
    csv_output_dir: Path = None,
    data_root: Path = None
) -> bool:
    """
    Generate all analysis plots and optionally export CSV.
    
    Args:
        model_name: Model identifier (ABS1, REL1, REL2)
        output_dir: Directory containing group_* subdirectories
        pse_grid: List of PSE values
        jnd_grid: List of JND values
        export_csv: Whether to export CSV for R
        csv_output_dir: Directory for CSV output (if export_csv=True)
        data_root: Root directory for CSV export (if export_csv=True)
    
    Returns:
        True if successful, False otherwise
    """
    # Generate plots
    plotter = AnalysisPlotGenerator(model_name, output_dir, pse_grid, jnd_grid)
    if not plotter.generate_all_plots():
        return False
    
    logger.info(f"✓ Generated analysis plots for {model_name}")
    
    # Export CSV if requested
    if export_csv:
        if csv_output_dir is None or data_root is None:
            logger.warning("CSV export requested but csv_output_dir or data_root not provided")
            return True
        
        exporter = CSVExporter(pse_grid, jnd_grid)
        if exporter.export_all_models([model_name], data_root, csv_output_dir / "stimulus_metrics_all_models.csv"):
            logger.info(f"✓ Exported CSV for {model_name}")
    
    return True
