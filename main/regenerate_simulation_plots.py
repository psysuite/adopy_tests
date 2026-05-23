#!/usr/bin/env python3
"""
Regenerate all plots and CSV export from existing simulation results.

Rigenerates:
- Group plots (histogram, psychometric, stimulus_distribution)
- Grid plots (3x3 grids)
- Analysis plots (asymmetry, stimulus metrics, entropy)
- CSV export for R

Usage:
    python regenerate_analysis_plots.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.core.generate_analysis_plots import generate_all_analysis
from analysis.core.plot_psychometric_curves import plot_psychometric_for_model
from analysis.core.plot_stimulus_distribution import plot_stimulus_for_model

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODELS = ['ABS1', 'REL1', 'REL2']
PSE_GRID = [480, 500, 520]
JND_GRID = [20, 40, 60]

DATA_ROOT = Path(__file__).parent.parent / 'data' / 'output' / 'sim_gridrnd'
CSV_OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'output' / 'stimulus_metrics_for_r'

EXPORT_CSV = True


def main():
    print("=" * 70)
    print("REGENERATING ALL PLOTS AND CSV")
    print("=" * 70)
    print()
    
    if not DATA_ROOT.exists():
        logger.error(f"Data root not found: {DATA_ROOT}")
        return 1
    
    logger.info(f"Data root: {DATA_ROOT}\n")
    
    success_count = 0
    for model_name in MODELS:
        model_dir = DATA_ROOT / model_name
        
        if not model_dir.exists():
            logger.warning(f"Model directory not found: {model_dir}")
            continue
        
        logger.info(f"Processing {model_name}...")
        
        # Phase 1: Group plots (psychometric + stimulus distribution)
        logger.info(f"  Phase 1: Generating group plots...")
        plot_psychometric_for_model(model_name)
        plot_stimulus_for_model(model_name)
        
        # Phase 2: Analysis plots + CSV
        logger.info(f"  Phase 2: Generating analysis plots and CSV...")
        if generate_all_analysis(
            model_name=model_name,
            output_dir=model_dir,
            pse_grid=PSE_GRID,
            jnd_grid=JND_GRID,
            export_csv=EXPORT_CSV,
            csv_output_dir=CSV_OUTPUT_DIR,
            data_root=DATA_ROOT
        ):
            success_count += 1
            logger.info(f"✓ {model_name} complete\n")
        else:
            logger.error(f"✗ {model_name} failed\n")
    
    print("=" * 70)
    if success_count == len(MODELS):
        print(f"✓ ALL MODELS COMPLETE ({success_count}/{len(MODELS)})")
    else:
        print(f"⚠ PARTIAL COMPLETION ({success_count}/{len(MODELS)})")
    print("=" * 70)
    
    return 0 if success_count == len(MODELS) else 1


if __name__ == "__main__":
    sys.exit(main())
