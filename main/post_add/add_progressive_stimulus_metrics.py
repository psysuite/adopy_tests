#!/usr/bin/env python3
"""
Retroactively add progressive stimulus metrics to existing Excel files.
Useful for updating results after adding new analysis functions.

Adds columns:
- stimulus_center_40, stimulus_center_60, ..., stimulus_center_200
- stimulus_spread_40, stimulus_spread_60, ..., stimulus_spread_200
- stimulus_min_40, stimulus_min_60, ..., stimulus_min_200
- stimulus_max_40, stimulus_max_60, ..., stimulus_max_200
- bimodality_index_40, bimodality_index_60, ..., bimodality_index_200
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.core.psychometric_analysis import calculate_progressive_stimulus_metrics

MODELS = ["ABS1", "REL1", "REL2"]
PSE_GRID = [480, 500, 520]
JND_GRID = [20, 40, 60]
TRIAL_BLOCKS = [40, 60, 80, 100, 120, 140, 160, 180, 200]

def load_gbf_file(gbf_path):
    """Load GBF file and return list of trial dicts."""
    rows = []
    try:
        with open(gbf_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        lat = float(parts[0])
                        user_ans = int(parts[1])
                        rows.append({'lat': lat, 'user_ans': user_ans})
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"Warning: GBF file not found: {gbf_path}")
    
    return rows

def add_stimulus_metrics_to_excel(model_name):
    """Add progressive stimulus metrics to all Excel files for a model."""
    print(f"Processing {model_name}...")
    
    for pse in PSE_GRID:
        for jnd in JND_GRID:
            group_name = f"group_{pse}_{jnd}"
            group_dir = Path(f"../../data/output/sim_gridrnd/{model_name}/{group_name}")
            results_dir = group_dir / "results"
            
            if not results_dir.exists():
                print(f"  Warning: Results directory not found: {results_dir}")
                continue
            
            # Find Excel file
            excel_files = list(results_dir.glob("*_results_summary.xlsx"))
            if not excel_files:
                print(f"  Warning: No Excel file found for {group_name}")
                continue
            
            excel_path = excel_files[0]
            print(f"  Processing {group_name}...")
            
            try:
                df = pd.read_excel(excel_path)
                
                # Initialize columns if they don't exist
                for metric_name in ['stimulus_center', 'stimulus_spread', 'stimulus_min', 'stimulus_max', 'bimodality_index']:
                    for n_trials in TRIAL_BLOCKS:
                        col_name = f'{metric_name}_{n_trials}'
                        if col_name not in df.columns:
                            df[col_name] = np.nan
                
                # Extract group index from filename (e.g., G2 from REL1_G2_results_summary.xlsx)
                filename = excel_path.name
                group_idx = None
                for part in filename.split('_'):
                    if part.startswith('G') and part[1:].isdigit():
                        group_idx = int(part[1:])
                        break
                
                if group_idx is None:
                    print(f"    Warning: Could not extract group index from {filename}")
                    continue
                
                # Process each subject (skip last 2 rows which are mean and stddev)
                for idx, row in df.iterrows():
                    # Skip last 2 rows
                    if idx >= len(df) - 2:
                        continue
                    
                    # Get subject ID
                    subject_id = row.get('subj') if 'subj' in df.columns else row.get('subject_id')
                    
                    if subject_id is None:
                        continue
                    
                    # Subject number in group is idx + 1 (1-20)
                    subject_num = idx + 1
                    
                    # Construct GBF filename with zero-padding
                    gbf_filename = f"{model_name}_G{group_idx}_S{subject_num:02d}.txt"
                    gbf_path = group_dir / gbf_filename
                    
                    if not gbf_path.exists():
                        continue
                    
                    gbf_rows = load_gbf_file(str(gbf_path))
                    
                    if not gbf_rows:
                        continue
                    
                    # Calculate metrics
                    stim_metrics = calculate_progressive_stimulus_metrics(gbf_rows)
                    
                    # Add to dataframe
                    for metric_name, metric_dict in stim_metrics.items():
                        for n_trials, value in metric_dict.items():
                            col_name = f'{metric_name}_{n_trials}'
                            df.at[idx, col_name] = value
                
                # Save updated Excel
                df.to_excel(excel_path, index=False)
                print(f"    ✓ Updated {excel_path}")
                
            except Exception as e:
                print(f"  Error processing {excel_path}: {e}")

def main():
    print("Adding progressive stimulus metrics to Excel files...\n")
    
    for model_name in MODELS:
        add_stimulus_metrics_to_excel(model_name)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
