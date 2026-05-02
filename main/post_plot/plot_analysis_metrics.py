#!/usr/bin/env python3
"""
Wrapper script to regenerate analysis metric plots from group simulation results.

This script allows regenerating the 5 analysis plots without re-running simulations.
It imports the plotting functions from analysis/core/plot_analysis_metrics.py

Generates:
1. asymmetry_modulo.png - |asymmetry_index| evolution
2. asymmetry_scatter_envelope.png - asymmetry distribution with envelopes
3. stimulus_center_evolution.png - stimulus center (mean latency) evolution
4. stimulus_spread_evolution.png - stimulus spread (std latency) evolution
5. bimodality_index_evolution.png - bimodality index evolution

Saves plots to: data/output/sim_gridrnd/{model}/
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.core.plot_analysis_metrics import plot_analysis_metrics

# Configuration
MODELS = ['ABS1', 'REL1', 'REL2']
PSE_GRID = [480, 500, 520]
JND_GRID = [20, 40, 60]


def main():
    """Regenerate analysis metric plots for all models."""
    for model_name in MODELS:
        print(f"\nGenerating analysis plots for {model_name}...")
        
        output_dir = Path(f"../../data/output/sim_gridrnd/{model_name}")
        if not output_dir.exists():
            print(f"  Output directory not found: {output_dir}")
            continue
        
        # Generate plots
        plot_analysis_metrics(model_name, output_dir, PSE_GRID, JND_GRID)
        
        print(f"  ✓ All plots generated for {model_name}")
    
    print(f"\nDone!")


if __name__ == "__main__":
    main()
