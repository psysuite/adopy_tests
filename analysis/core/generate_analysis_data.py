"""
Generate analysis data (Excel columns and CSV exports) from simulation results.

Consolidates:
- Progressive asymmetry calculations
- Progressive latency entropy calculations
- Progressive stimulus metrics calculations
- CSV export for R analysis

Designed to be called from:
- group_sim_gridrnd_*.py (during simulation)
- regenerate_simulation_data.py (post-processing)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

from analysis.core.psychometric_analysis import add_group_stats_to_excel, consolidate_results

logger = logging.getLogger(__name__)

TRIAL_BLOCKS = [40, 60, 80, 100, 120, 140, 160, 180, 200]


def add_progressive_asymmetry_to_excel(excel_path, group_dir, model_name, group_idx, offset=500):
    """Add progressive asymmetry columns to Excel file."""
    from analysis.core.psychometric_analysis import calculate_progressive_asymmetry
    from analysis.io.converter import read_gbf_file
    
    try:
        df = pd.read_excel(excel_path)
        
        # Process GBF files in order
        gbf_files = sorted(group_dir.glob("*.txt"))
        for idx, gbf_path in enumerate(gbf_files):
            if idx >= min(20, len(df)):
                break
            
            row = df.iloc[idx]
            if pd.isna(row.get('subj')):
                continue
            
            try:
                # Read GBF file
                gbf_rows = read_gbf_file(str(gbf_path))
                rows = [{'lat': r['lat'], 'user_ans': r['user_ans']} for r in gbf_rows]
                
                # Calculate progressive asymmetry
                prog_asymmetry = calculate_progressive_asymmetry(rows, offset)
                
                # Add to dataframe
                for n_trials, asym_idx in prog_asymmetry.items():
                    col_name = f'asymmetry_{n_trials}'
                    if col_name not in df.columns:
                        df[col_name] = np.nan
                    df.at[idx, col_name] = asym_idx
            except Exception:
                continue
        
        # Save Excel
        df.to_excel(excel_path, index=False)
        return True
        
    except Exception as e:
        logger.error(f"Error adding asymmetry to {excel_path}: {e}")
        return False


def add_progressive_lat_entropy_to_excel(excel_path, group_dir, model_name, group_idx, pse, jnd):
    """Add progressive lat_entropy columns to Excel file."""
    from analysis.core.psychometric_analysis import calculate_latency_statistics
    from analysis.io.converter import read_gbf_file
    
    try:
        df = pd.read_excel(excel_path)
        
        # Check if already exists
        existing_cols = [col for col in df.columns if col.startswith('lat_entropy_')]
        if existing_cols:
            return True
        
        # Process GBF files in order
        gbf_files = sorted(group_dir.glob("*.txt"))
        for idx, gbf_path in enumerate(gbf_files):
            if idx >= min(20, len(df)):
                break
            
            row = df.iloc[idx]
            if pd.isna(row.get('subj')):
                continue
            
            try:
                # Read GBF file
                gbf_rows = read_gbf_file(str(gbf_path))
                latencies = np.array([r['lat'] for r in gbf_rows], dtype=float)
                
                # Calculate progressive lat_entropy
                for block_size in TRIAL_BLOCKS:
                    if block_size > len(latencies):
                        break
                    
                    lat_block = latencies[:block_size]
                    stats = calculate_latency_statistics(lat_block)
                    lat_entropy = stats['lat_entropy']
                    
                    col_name = f'lat_entropy_{block_size}'
                    if col_name not in df.columns:
                        df[col_name] = np.nan
                    df.at[idx, col_name] = lat_entropy
            except Exception:
                continue
        
        # Save Excel
        df.to_excel(excel_path, index=False)
        return True
        
    except Exception as e:
        logger.error(f"Error adding lat_entropy to {excel_path}: {e}")
        return False


def add_progressive_stimulus_metrics_to_excel(excel_path, group_dir, model_name, group_idx):
    """Add progressive stimulus metrics columns to Excel file."""
    from analysis.core.psychometric_analysis import calculate_progressive_stimulus_metrics
    from analysis.io.converter import read_gbf_file
    
    try:
        df = pd.read_excel(excel_path)
        
        # Initialize columns if they don't exist
        for metric_name in ['stimulus_center', 'stimulus_spread', 'stimulus_min', 'stimulus_max']:
            for n_trials in TRIAL_BLOCKS:
                col_name = f'{metric_name}_{n_trials}'
                if col_name not in df.columns:
                    df[col_name] = np.nan
        
        # Process GBF files in order
        gbf_files = sorted(group_dir.glob("*.txt"))
        for idx, gbf_path in enumerate(gbf_files):
            if idx >= min(20, len(df)):
                break
            
            if idx >= len(df) - 2:  # Skip last 2 rows (mean/std)
                continue
            
            try:
                # Read GBF file
                gbf_rows = read_gbf_file(str(gbf_path))
                rows = [{'lat': r['lat'], 'user_ans': r['user_ans']} for r in gbf_rows]
                
                if not rows:
                    continue
                
                # Calculate metrics
                stim_metrics = calculate_progressive_stimulus_metrics(rows)
                
                # Add to dataframe
                for metric_name, metric_dict in stim_metrics.items():
                    for n_trials, value in metric_dict.items():
                        col_name = f'{metric_name}_{n_trials}'
                        df.at[idx, col_name] = value
            except Exception:
                continue
        
        # Save Excel
        df.to_excel(excel_path, index=False)
        return True
        
    except Exception as e:
        logger.error(f"Error adding stimulus metrics to {excel_path}: {e}")
        return False


def export_stimulus_metrics_to_csv(model_name, data_root, output_dir, pse_grid, jnd_grid):
    """Export stimulus metrics from Excel files to CSV for R analysis."""
    rows_list = []
    
    for pse in pse_grid:
        for jnd in jnd_grid:
            group_name = f"group_{pse}_{jnd}"
            group_dir = data_root / model_name / group_name
            results_dir = group_dir / "results"
            
            if not results_dir.exists():
                continue
            
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
                
                # Convert to long format
                for idx, row in df.iterrows():
                    subject_id = row.get('subj') if 'subj' in df.columns else row.get('subject_id')
                    
                    if pd.isna(subject_id):
                        continue
                    
                    subject_id_str = str(subject_id).lower()
                    if subject_id_str in ['mean', 'group_mean', 'group_std', 'stddev', 'std', 'nan']:
                        continue
                    
                    pse_subj = row.get('pse', pse)
                    jnd_subj = row.get('jnd', jnd)
                    
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
                        
                        # Always add pse_est and jnd_est (from progressive analysis columns)
                        pse_col = f'pse_{n_trials}'
                        jnd_col = f'jnd_{n_trials}'
                        
                        if pse_col in df.columns:
                            row_dict['pse_est'] = row.get(pse_col, np.nan)
                        else:
                            row_dict['pse_est'] = np.nan
                        
                        if jnd_col in df.columns:
                            row_dict['jnd_est'] = row.get(jnd_col, np.nan)
                        else:
                            row_dict['jnd_est'] = np.nan
                        
                        if f'asymmetry_{n_trials}' in df.columns:
                            row_dict['asymmetry_index'] = row.get(f'asymmetry_{n_trials}', np.nan)
                        
                        rows_list.append(row_dict)
                
            except Exception as e:
                logger.error(f"Error processing {group_name}: {e}")
                continue
    
    return pd.DataFrame(rows_list)


def add_final_estimates_to_excel(excel_path):
    """Add pse_est and jnd_est columns (final estimates from trial 200) to Excel."""
    try:
        df = pd.read_excel(excel_path)
        
        # Add pse_est and jnd_est from final trial block (200)
        if 'pse_200' in df.columns:
            df['pse_est'] = df['pse_200']
        else:
            df['pse_est'] = np.nan
        
        if 'jnd_200' in df.columns:
            df['jnd_est'] = df['jnd_200']
        else:
            df['jnd_est'] = np.nan
        
        # Save Excel
        df.to_excel(excel_path, index=False)
        return True
        
    except Exception as e:
        logger.error(f"Error adding final estimates to {excel_path}: {e}")
        return False


def regenerate_all_data_from_gbf(model_name, output_dir, pse_grid, jnd_grid, offset=500, ado_params=None, bis_params=None):
    """
    Regenerate all data from GBF files: reconstruct results, run analyses, create Excel, add columns.
    
    This is the main entry point for regenerate_simulation_data.py.
    
    Args:
        model_name: Model identifier (ABS1, REL1, REL2)
        output_dir: Directory containing group_* subdirectories
        pse_grid: List of PSE values
        jnd_grid: List of JND values
        offset: Offset for asymmetry calculations
        ado_params: ADO parameters for progressive analysis
        bis_params: BIS parameters for progressive analysis
    
    Returns:
        True if successful, False otherwise
    """
    from itertools import product
    from utilities.multithreading_utils import (
        MultiThreadedSimulationRunner,
        load_gbf_files_for_group,
        backfill_skip_mode,
    )
    
    if ado_params is None:
        ado_params = {"guess_rate": 0.5, "lapse_rate": 0.04}
    if bis_params is None:
        bis_params = {}
    
    runner = MultiThreadedSimulationRunner(max_workers=3)
    grid = list(product(pse_grid, jnd_grid))
    processed_count = 0
    
    for group_idx, (pse, jnd) in enumerate(grid, 1):
        group_dir = output_dir / f"group_{pse}_{jnd}"
        results_dir = group_dir / "results"
        
        if not group_dir.exists():
            logger.debug(f"  Skipping group {pse}_{jnd}: group directory not found")
            continue
        
        # Load GBF files
        analysis_tasks = load_gbf_files_for_group(group_dir, model_name, offset)
        
        if not analysis_tasks:
            logger.debug(f"  Skipping group {pse}_{jnd}: no GBF files found")
            continue
        
        logger.info(f"  Processing group {pse}_{jnd}: {len(analysis_tasks)} subjects")
        
        # Prepare data structures
        group_rows_list = []
        group_results = []
        analysis_tasks_to_run = []
        task_to_result_idx = {}
        
        # Reconstruct from GBF
        for gbf_rows, subj, result_dict, rows, offset_val in analysis_tasks:
            group_rows_list.append(rows)
            group_results.append(result_dict)
            task_to_result_idx[subj] = len(group_results) - 1
            analysis_tasks_to_run.append((gbf_rows, subj, result_dict, rows, offset_val))
        
        # Run progressive analyses
        if analysis_tasks_to_run:
            analysis_results = runner.run_progressive_analyses(
                analysis_tasks_to_run,
                verbose=False,
                gamma=ado_params.get("guess_rate", 0.5),
                lapse=ado_params.get("lapse_rate", 0.04)
            )
            
            for subj, (updated_dict, error_msg) in analysis_results.items():
                if error_msg:
                    logger.debug(f"    Warning: Progressive analysis failed for {subj}: {error_msg}")
                else:
                    result_idx = task_to_result_idx.get(subj)
                    if result_idx is not None:
                        group_results[result_idx] = updated_dict
            
            # Backfill for skip mode
            backfill_skip_mode(group_results, analysis_tasks, 200)
        
        # Create results directory
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Excel
        if group_results:
            try:
                excel_filepath = consolidate_results(group_results, str(results_dir), f"{model_name}_G{group_idx}")
                add_group_stats_to_excel(excel_filepath)
                
                # Add analysis columns (creates pse_40, pse_60, ..., pse_200, etc.)
                add_progressive_asymmetry_to_excel(excel_filepath, group_dir, model_name, group_idx, offset)
                add_progressive_lat_entropy_to_excel(excel_filepath, group_dir, model_name, group_idx, pse, jnd)
                add_progressive_stimulus_metrics_to_excel(excel_filepath, group_dir, model_name, group_idx)
                
                # Add final estimates (pse_est, jnd_est from trial 200) after progressive columns exist
                add_final_estimates_to_excel(excel_filepath)
                
                processed_count += 1
                logger.info(f"    ✓ Excel created with {len(group_results)} subjects")
            except Exception as e:
                logger.error(f"    Error creating Excel for group {pse}_{jnd}: {e}")
    
    if processed_count == 0:
        logger.warning(f"No groups processed for {model_name}. Check that GBF files exist in {output_dir}")
    
    return processed_count > 0


def generate_all_data(model_name, output_dir, pse_grid, jnd_grid, offset=500):
    """
    Generate all analysis data (Excel columns) for a model.
    
    Args:
        model_name: Model identifier (ABS1, REL1, REL2)
        output_dir: Directory containing group_* subdirectories
        pse_grid: List of PSE values
        jnd_grid: List of JND values
        offset: Offset for asymmetry calculations
    
    Returns:
        True if successful, False otherwise
    """
    from itertools import product
    
    grid = list(product(pse_grid, jnd_grid))
    processed_count = 0
    
    for group_idx, (pse, jnd) in enumerate(grid, 1):
        group_dir = output_dir / f"group_{pse}_{jnd}"
        results_dir = group_dir / "results"
        
        if not results_dir.exists():
            logger.debug(f"  Skipping group {pse}_{jnd}: results directory not found at {results_dir}")
            continue
        
        excel_files = list(results_dir.glob(f"{model_name}_G{group_idx}_results_summary.xlsx"))
        if not excel_files:
            logger.debug(f"  Skipping group {pse}_{jnd}: no Excel file found")
            continue
        
        excel_path = excel_files[0]
        
        logger.info(f"  Processing group {pse}_{jnd}...")
        
        # Add all metrics first (creates pse_40, pse_60, ..., pse_200, etc.)
        add_progressive_asymmetry_to_excel(excel_path, group_dir, model_name, group_idx, offset)
        add_progressive_lat_entropy_to_excel(excel_path, group_dir, model_name, group_idx, pse, jnd)
        add_progressive_stimulus_metrics_to_excel(excel_path, group_dir, model_name, group_idx)
        
        # Add final estimates (pse_est, jnd_est from trial 200) after progressive columns exist
        add_final_estimates_to_excel(excel_path)
        
        processed_count += 1
    
    if processed_count == 0:
        logger.warning(f"No groups processed for {model_name}. Check that Excel files exist in {output_dir}")
    
    return processed_count > 0
