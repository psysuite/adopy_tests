#!/usr/bin/env python3
"""
Add progressive asymmetry calculations to existing Excel files.
Reads GBF files and adds asymmetry_40, asymmetry_60, ..., asymmetry_200 columns to Excel.
"""

import os
import sys
from pathlib import Path
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.core.psychometric_analysis import calculate_progressive_asymmetry
import pandas as pd

# Configuration
MODELS = {
    "ABS1": {"offset": 500},
    "REL1": {"offset": 500},
    "REL2": {"offset": 500},
}

PSE_GRID = [480, 500, 520]
JND_GRID = [20, 40, 60]

def add_asymmetry_to_model(model_name, offset):
    """Add progressive asymmetry to Excel files for a specific model."""
    output_dir = Path(f"../../data/output/sim_gridrnd/{model_name}")
    
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return False
    
    grid = list(product(PSE_GRID, JND_GRID))
    
    for group_idx, (pse, jnd) in enumerate(grid, 1):
        group_dir = output_dir / f"group_{pse}_{jnd}"
        group_results_dir = group_dir / "results"
        
        if not group_results_dir.exists():
            print(f"  Skipping group {pse}_{jnd}: results directory not found")
            continue
        
        # Find Excel file
        excel_files = list(group_results_dir.glob(f"{model_name}_G{group_idx}_results_summary.xlsx"))
        
        if not excel_files:
            print(f"  Skipping group {pse}_{jnd}: no Excel file found")
            continue
        
        excel_file = excel_files[0]
        
        print(f"  Processing group {pse}_{jnd}...")
        
        try:
            # Read Excel
            df = pd.read_excel(excel_file)
            
            # Process only first 20 rows (exclude mean/std rows at the end)
            for idx in range(min(20, len(df))):
                row = df.iloc[idx]
                
                # Skip if no subject ID
                if pd.isna(row['subj']):
                    continue
                
                # Subject number within group is idx + 1 (1-20)
                subj_num = idx + 1
                
                # Find GBF file
                gbf_file = group_dir / f"{model_name}_G{group_idx}_S{subj_num:02d}.txt"
                
                if not gbf_file.exists():
                    print(f"    Warning: GBF file not found: {gbf_file}")
                    continue
                
                # Read GBF file
                rows = []
                try:
                    with open(gbf_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 3:
                                lat = float(parts[0])
                                user_ans = int(parts[2])
                                rows.append({
                                    'lat': lat,
                                    'user_ans': user_ans,
                                })
                except Exception as e:
                    print(f"    Error reading {gbf_file}: {e}")
                    continue
                
                # Calculate progressive asymmetry
                prog_asymmetry = calculate_progressive_asymmetry(rows, offset)
                
                # Add to dataframe
                for n_trials, asym_idx in prog_asymmetry.items():
                    col_name = f'asymmetry_{n_trials}'
                    if col_name not in df.columns:
                        df[col_name] = None
                    df.at[idx, col_name] = asym_idx
            
            # Save Excel
            df.to_excel(excel_file, index=False)
            print(f"    ✓ Updated {excel_file}")
            
        except Exception as e:
            print(f"    Error processing {excel_file}: {e}")
            return False
    
    return True

def main():
    print("Adding progressive asymmetry to Excel files...\n")
    
    for model_name, config in MODELS.items():
        print(f"Processing {model_name}...")
        success = add_asymmetry_to_model(model_name, config["offset"])
        if success:
            print(f"  ✓ {model_name} updated\n")
        else:
            print(f"  ✗ Failed to update {model_name}\n")
    
    print("Done!")

if __name__ == "__main__":
    main()
