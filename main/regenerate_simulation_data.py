#!/usr/bin/env python3
"""
Regenerate all simulation data (Excel + CSV) from existing GBF files.

Regenerates:
- Excel files from GBF data
- Progressive asymmetry columns in Excel
- Progressive latency entropy columns in Excel
- Progressive stimulus metrics columns in Excel
- CSV export for R analysis

Usage:
    python regenerate_simulation_data.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.core.generate_analysis_data import regenerate_all_data_from_gbf, export_stimulus_metrics_to_csv

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODELS = ['ABS1', 'REL1', 'REL2']
PSE_GRID = [480, 500, 520]
JND_GRID = [20, 40, 60]
OFFSET = 500

DATA_ROOT = Path(__file__).parent.parent / 'data' / 'output' / 'sim_gridrnd'
CSV_OUTPUT_DIR = Path(__file__).parent.parent / 'R' / 'indata'


def main():
    print("=" * 70)
    print("REGENERATING SIMULATION DATA (EXCEL + CSV)")
    print("=" * 70)
    print()
    
    if not DATA_ROOT.exists():
        logger.error(f"Data root not found: {DATA_ROOT}")
        return 1
    
    logger.info(f"Data root: {DATA_ROOT}")
    logger.info(f"Models: {MODELS}\n")
    
    success_count = 0
    for model_name in MODELS:
        model_dir = DATA_ROOT / model_name
        
        if not model_dir.exists():
            logger.warning(f"Model directory not found: {model_dir}")
            continue
        
        logger.info(f"Processing {model_name}...")
        
        # Regenerate all data from GBF files
        if regenerate_all_data_from_gbf(
            model_name=model_name,
            output_dir=model_dir,
            pse_grid=PSE_GRID,
            jnd_grid=JND_GRID,
            offset=OFFSET
        ):
            success_count += 1
            logger.info(f"✓ {model_name} data regenerated\n")
        else:
            logger.warning(f"⚠ {model_name} had no groups to process\n")
    
    # Export CSV for all models
    logger.info("Exporting CSV for R analysis...")
    all_data = []
    
    for model_name in MODELS:
        model_dir = DATA_ROOT / model_name
        
        if not model_dir.exists():
            continue
        
        df = export_stimulus_metrics_to_csv(model_name, DATA_ROOT, CSV_OUTPUT_DIR, PSE_GRID, JND_GRID)
        
        if not df.empty:
            all_data.append(df)
            logger.info(f"  ✓ Exported {len(df)} rows for {model_name}")
        else:
            logger.info(f"  ⚠ No data exported for {model_name}")
    
    if all_data:
        # Combine all data
        combined_df = __import__('pandas').concat(all_data, ignore_index=True)
        
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
        
        # Create output directory
        CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Export CSV
        output_path = CSV_OUTPUT_DIR / "stimulus_metrics_all_models.csv"
        combined_df.to_csv(output_path, index=False)
        
        logger.info(f"✓ CSV exported to {output_path}")
    else:
        logger.warning("No CSV data to export")
    
    print("=" * 70)
    if success_count == len(MODELS):
        print(f"✓ ALL MODELS COMPLETE ({success_count}/{len(MODELS)})")
    else:
        print(f"⚠ PARTIAL COMPLETION ({success_count}/{len(MODELS)})")
    print("=" * 70)
    
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
