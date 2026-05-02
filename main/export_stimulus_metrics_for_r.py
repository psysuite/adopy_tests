#!/usr/bin/env python3
"""
Export stimulus metrics and psychometric parameters to CSV files for R analysis.

Exports data in long format suitable for R analysis:
- One row per subject per trial block
- Columns: model, pse_true, jnd_true, subject_id, trial_block, 
           pse_est, jnd_est, stimulus_center, stimulus_spread, bimodality_index, asymmetry_index

This script handles:
- Absolute and relative paths
- Missing files with informative warnings
- Both stimulus metrics and psychometric parameters
- Proper data type conversions for R
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS = ["ABS1", "REL2", "ABS1"]  #(complete data with 20 subjects per group)
PSE_GRID = [480, 500, 520]
JND_GRID = [20, 40, 60]
TRIAL_BLOCKS = [40, 60, 80, 100, 120, 140, 160, 180, 200]


def get_data_root():
    """Get the root data directory (handles both absolute and relative paths)."""
    script_dir = Path(__file__).parent.parent  # adopy_tests directory
    data_dir = script_dir / "data" / "output" / "sim_gridrnd"
    
    if not data_dir.exists():
        logger.warning(f"Data directory not found at {data_dir}")
        logger.info("Trying alternative path...")
        # Try absolute path
        alt_data_dir = Path("/data/CODE/python/adopy_tests/data/output/sim_gridrnd")
        if alt_data_dir.exists():
            logger.info(f"Found data at {alt_data_dir}")
            return alt_data_dir
        else:
            raise FileNotFoundError(f"Could not find data directory. Tried:\n  {data_dir}\n  {alt_data_dir}")
    
    return data_dir


def export_stimulus_metrics(model_name, data_root):
    """Export stimulus metrics and psychometric parameters for a model to DataFrame."""
    rows_list = []
    groups_found = 0
    groups_processed = 0
    
    for pse in PSE_GRID:
        for jnd in JND_GRID:
            group_name = f"group_{pse}_{jnd}"
            group_dir = data_root / model_name / group_name
            results_dir = group_dir / "results"
            
            if not results_dir.exists():
                logger.debug(f"Results directory not found: {results_dir}")
                continue
            
            excel_files = list(results_dir.glob("*_results_summary.xlsx"))
            if not excel_files:
                logger.debug(f"No Excel files found in {results_dir}")
                continue
            
            groups_found += 1
            excel_path = excel_files[0]
            
            try:
                df = pd.read_excel(excel_path)
                
                # Filter out 'mean' and 'GROUP_' rows
                if 'subj' in df.columns:
                    df = df[~df['subj'].astype(str).str.lower().isin(['mean', 'group_mean', 'group_std', 'stddev', 'std'])]
                elif 'subject_id' in df.columns:
                    df = df[~df['subject_id'].astype(str).str.lower().isin(['mean', 'group_mean', 'group_std', 'stddev', 'std'])]
                
                # Check for required columns
                stimulus_center_cols = [f'stimulus_center_{n}' for n in TRIAL_BLOCKS]
                stimulus_spread_cols = [f'stimulus_spread_{n}' for n in TRIAL_BLOCKS]
                bimodality_cols = [f'bimodality_index_{n}' for n in TRIAL_BLOCKS]
                pse_cols = [f'pse_{n}' for n in TRIAL_BLOCKS]
                jnd_cols = [f'jnd_{n}' for n in TRIAL_BLOCKS]
                asymmetry_cols = [f'asymmetry_{n}' for n in TRIAL_BLOCKS]
                
                # Check if stimulus metrics exist
                if not all(col in df.columns for col in stimulus_center_cols):
                    logger.warning(f"Stimulus metrics not found in {group_name}")
                    continue
                
                # Convert to long format
                for idx, row in df.iterrows():
                    # Get subject ID
                    subject_id = row.get('subj') if 'subj' in df.columns else row.get('subject_id')
                    
                    # Skip if subject_id is NaN or invalid
                    if pd.isna(subject_id):
                        continue
                    
                    # Convert to string and skip summary rows
                    subject_id_str = str(subject_id).lower()
                    if subject_id_str in ['mean', 'group_mean', 'group_std', 'stddev', 'std', 'nan']:
                        continue
                    
                    # Extract for each trial block
                    for n_trials in TRIAL_BLOCKS:
                        try:
                            row_dict = {
                                'model': model_name,
                                'pse_true': pse,
                                'jnd_true': jnd,
                                'subject_id': subject_id,
                                'trial_block': n_trials,
                                'stimulus_center': row.get(f'stimulus_center_{n_trials}', np.nan),
                                'stimulus_spread': row.get(f'stimulus_spread_{n_trials}', np.nan),
                                'bimodality_index': row.get(f'bimodality_index_{n_trials}', np.nan),
                            }
                            
                            # Add psychometric parameters if available
                            if f'pse_{n_trials}' in df.columns:
                                row_dict['pse_est'] = row.get(f'pse_{n_trials}', np.nan)
                            if f'jnd_{n_trials}' in df.columns:
                                row_dict['jnd_est'] = row.get(f'jnd_{n_trials}', np.nan)
                            if f'asymmetry_{n_trials}' in df.columns:
                                row_dict['asymmetry_index'] = row.get(f'asymmetry_{n_trials}', np.nan)
                            
                            rows_list.append(row_dict)
                        except Exception as e:
                            logger.debug(f"Error processing row {idx} in {group_name}: {e}")
                            continue
                
                groups_processed += 1
                logger.info(f"  ✓ {group_name}: {len(df)} subjects")
                
            except Exception as e:
                logger.error(f"Error processing {excel_path}: {e}")
                continue
    
    logger.info(f"  Groups found: {groups_found}, processed: {groups_processed}")
    return pd.DataFrame(rows_list)

def main():
    """Main export function."""
    print("="*70)
    print("EXPORTING STIMULUS METRICS FOR R ANALYSIS")
    print("="*70)
    print()
    
    try:
        # Get data root directory
        data_root = get_data_root()
        logger.info(f"Using data directory: {data_root}\n")
        
        all_data = []
        
        for model_name in MODELS:
            print(f"Processing {model_name}...")
            df = export_stimulus_metrics(model_name, data_root)
            
            if df.empty:
                logger.warning(f"  No data found for {model_name}")
                continue
            
            all_data.append(df)
            print(f"  ✓ Exported {len(df)} rows for {model_name}\n")
        
        if not all_data:
            logger.error("No data exported from any model!")
            return 1
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates: keep only first occurrence of each (model, pse_true, jnd_true, subject_id, trial_block)
        combined_df = combined_df.drop_duplicates(
            subset=['model', 'pse_true', 'jnd_true', 'subject_id', 'trial_block'],
            keep='first'
        )
        
        # Determine output directory
        script_dir = Path(__file__).parent.parent
        output_dir = script_dir / "data" / "output" / "stimulus_metrics_for_r"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export combined data
        output_path = output_dir / "stimulus_metrics_all_models.csv"
        combined_df.to_csv(output_path, index=False)
        
        print("="*70)
        print("✓ EXPORT COMPLETE")
        print("="*70)
        print(f"\nOutput file: {output_path}")
        print(f"Total rows: {len(combined_df)}")
        print(f"Columns: {list(combined_df.columns)}")
        print(f"\nData summary:")
        print(f"  Models: {combined_df['model'].unique().tolist()}")
        print(f"  PSE values: {sorted(combined_df['pse_true'].unique().tolist())}")
        print(f"  JND values: {sorted(combined_df['jnd_true'].unique().tolist())}")
        print(f"  Trial blocks: {sorted(combined_df['trial_block'].unique().tolist())}")
        print(f"  Subjects per group: {combined_df.groupby(['model', 'pse_true', 'jnd_true'])['subject_id'].nunique().unique()}")
        print(f"  Total unique subjects: {combined_df['subject_id'].nunique()}")
        print()
        
        # Data quality check
        print("Data quality check:")
        print(f"  Missing values in stimulus_center: {combined_df['stimulus_center'].isna().sum()}")
        print(f"  Missing values in stimulus_spread: {combined_df['stimulus_spread'].isna().sum()}")
        print(f"  Missing values in bimodality_index: {combined_df['bimodality_index'].isna().sum()}")
        if 'pse_est' in combined_df.columns:
            print(f"  Missing values in pse_est: {combined_df['pse_est'].isna().sum()}")
        if 'jnd_est' in combined_df.columns:
            print(f"  Missing values in jnd_est: {combined_df['jnd_est'].isna().sum()}")
        
        # Check for duplicates
        duplicates = combined_df.duplicated(
            subset=['model', 'pse_true', 'jnd_true', 'subject_id', 'trial_block'],
            keep=False
        ).sum()
        print(f"  Duplicate rows (after removal): {duplicates}")
        print()
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
