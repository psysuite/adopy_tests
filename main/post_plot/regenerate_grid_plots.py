#!/usr/bin/env python3
"""
Regenerate grid plots from existing group plots without re-running simulations.
Useful for updating grid plots after modifying individual group results.
"""

import os
import sys
from pathlib import Path
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utilities.plotting import create_grid_plots_from_groups

# Configuration
MODELS = {
    "ABS1": ["histogram", "psychometric"],
    "REL1": ["histogram", "psychometric"],
    "REL2": ["histogram", "model_histogram", "psychometric"],
}

PSE_GRID = [480, 500, 520]
JND_GRID = [20, 40, 60]

def regenerate_grids_for_model(model_name, plot_types):
    """Regenerate grid plots for a specific model."""
    output_dir = Path(f"../../data/output/sim_gridrnd/{model_name}")
    
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return False
    
    grid = list(product(PSE_GRID, JND_GRID))
    
    for plot_type in plot_types:
        print(f"  Regenerating {plot_type} grid...")
        
        # Collect plot files
        plot_files_dict = {}
        missing_plots = []
        
        for group_idx, (pse, jnd) in enumerate(grid, 1):
            group_dir = output_dir / f"group_{pse}_{jnd}"
            group_results_dir = group_dir / "results"
            
            # Construct expected plot filename with _group_ prefix
            if plot_type == "model_histogram":
                plot_filename = f"{model_name}_G{group_idx}_group_model_histogram.png"
            else:
                plot_filename = f"{model_name}_G{group_idx}_group_{plot_type}.png"
            
            plot_path = group_results_dir / plot_filename
            
            if plot_path.exists():
                plot_files_dict[group_idx] = str(plot_path)
            else:
                missing_plots.append(group_idx)
        
        # If some plots are missing, skip this plot type
        if missing_plots:
            print(f"    Warning: {len(missing_plots)} plots missing for {plot_type}, skipping grid creation")
            continue
        
        # Create grid
        if len(plot_files_dict) == len(grid):
            create_grid_plots_from_groups(
                plot_files_dict,
                str(output_dir),
                model_name,
                PSE_GRID,
                JND_GRID,
                plot_type
            )
        else:
            print(f"    Error: Not all plots found for {plot_type}")
            return False
    
    return True

def main():
    print("Regenerating grid plots...\n")
    
    for model_name, plot_types in MODELS.items():
        print(f"Processing {model_name}...")
        success = regenerate_grids_for_model(model_name, plot_types)
        if success:
            print(f"  ✓ {model_name} grid plots regenerated\n")
        else:
            print(f"  ✗ Failed to regenerate {model_name} grid plots\n")
    
    print("Done!")

if __name__ == "__main__":
    main()
