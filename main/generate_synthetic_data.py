#!/usr/bin/env python3
"""
Synthetic Data Generation - Full Pipeline Orchestrator

Phases 1-2: Generate all synthetic data (3 models × 9 groups each) + groups' plots
  - Runs group_sim_gridrnd_ABS1.py, group_sim_gridrnd_REL1.py, group_sim_gridrnd_REL2.py
  - Output: GBF files + group plots

Phase 3: Calculate metrics and consolidate (UNIFIED PROCESSING)
  - Input: GBF files from Phases 1-2
  - Uses: UnifiedGBFProcessor (same code for synthetic + real)
  - Calculate: B1-B5 metrics using MetricsCalculator
  - Output: Excel (wide + long) in R/indata/

Phase 4: Generate consolidated plots
  - Input: consolidated Excel files
  - Output: Grid plots (3×3) for posteriors Level B + posteriors Level A

Usage:
  python main/generate_synthetic_data.py
  
  Or run individual phases:
  python main/generate_synthetic_data.py --phase 1-2 ABS1
  python main/generate_synthetic_data.py --phase 1-2 REL1
  python main/generate_synthetic_data.py --phase 1-2 REL2
  python main/generate_synthetic_data.py --phase 3-4
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.config_synthetic import *
from analysis.orchestration.unified_processor import UnifiedGBFProcessor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS = ['ABS1', 'REL1', 'REL2']
OUTPUT_BASE = Path(__file__).parent.parent / OUTPUT_BASE
EXCEL_OUTPUT_DIR = Path(__file__).parent.parent / EXCEL_OUTPUT_DIR

# ============================================================================
# PHASE 1-2: GENERATION (Call external scripts)
# ============================================================================

def run_phase_1_2_for_model(model_name: str, verbose: bool = True) -> list:
    """
    Run Phase 1-2 (data generation + groups' plots) for a single model.
    
    Calls: group_sim_gridrnd_ABS1.py, group_sim_gridrnd_REL1.py, or group_sim_gridrnd_REL2.py
    
    Returns:
        List of (group_idx, pse_center, jnd_center, group_results) tuples
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"PHASE 1-2: DATA GENERATION + groups' plots - {model_name}")
        print(f"{'='*70}\n")
    
    # Import the module dynamically
    module_name = f"group_sim_gridrnd_{model_name}"
    try:
        # Add main directory to path temporarily
        sys.path.insert(0, str(Path(__file__).parent))
        mod = __import__(module_name)
        sys.path.pop(0)
    except ImportError as e:
        print(f"✗ Error: Could not import {module_name}: {e}")
        return []
    
    # Call main() to run Phase 1-2
    try:
        all_group_results = mod.main()
        if verbose:
            print(f"\n✓ {model_name} Phase 1-2 complete: {len(all_group_results)} groups processed")
        return all_group_results
    except Exception as e:
        print(f"✗ Error running Phase 1-2 for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_all_phases_1_2(verbose: bool = True) -> dict:
    """
    Run Phase 1-2 for all 3 models.
    
    Returns:
        {model_name: all_group_results, ...}
    """
    all_results = {}
    for model_name in MODELS:
        results = run_phase_1_2_for_model(model_name, verbose=verbose)
        if results:
            all_results[model_name] = results
    
    return all_results


# ============================================================================
# PHASE 3: CALCULATE METRICS + CONSOLIDATE (UNIFIED PROCESSING)
# ============================================================================

def phase_3_calculate_metrics_and_consolidate(verbose: bool = True) -> tuple:
    """
    Phase 3: Calculate metrics on all GBF files using UnifiedGBFProcessor.
    
    Uses the same unified code for both synthetic and real data.
    
    Returns:
        (df_wide, df_long) DataFrames
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"PHASE 3: METRICS CALCULATION + CONSOLIDATION (UNIFIED)")
        print(f"{'='*70}\n")
    
    all_dfs_wide = []
    all_dfs_long = []
    
    # Process each model with UnifiedGBFProcessor
    for model_name in MODELS:
        model_output_dir = OUTPUT_BASE / model_name
        
        if not model_output_dir.exists():
            if verbose:
                print(f"⚠ Skipping {model_name} (directory not found: {model_output_dir})")
            continue
        
        if verbose:
            print(f"Processing {model_name}...")
        
        try:
            processor = UnifiedGBFProcessor(
                gbf_dir=model_output_dir,
                model_name=model_name,
                data_type='synthetic',
                output_dir=EXCEL_OUTPUT_DIR,
                pse_grid=PSE_GRID,
                jnd_grid=JND_GRID,
                offset=OFFSET,
                trial_blocks=[40, 60, 80, 100, 120, 140, 160, 180, 200],
                verbose=verbose,
            )
            
            df_wide, df_long = processor.process()
            
            if not df_wide.empty:
                all_dfs_wide.append(df_wide)
                all_dfs_long.append(df_long)
                if verbose:
                    print(f"  ✓ {model_name}: {len(df_wide)} subjects\n")
            else:
                if verbose:
                    print(f"  ⚠ {model_name}: no data processed\n")
        
        except Exception as e:
            logger.error(f"Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all models
    if all_dfs_wide:
        df_wide = pd.concat(all_dfs_wide, ignore_index=True)
        df_long = pd.concat(all_dfs_long, ignore_index=True)
        
        if verbose:
            print(f"✓ Phase 3 complete: {len(df_wide)} total subjects")
    else:
        df_wide = pd.DataFrame()
        df_long = pd.DataFrame()
        if verbose:
            print(f"✗ Phase 3: No data processed")
    
    return df_wide, df_long


# ============================================================================
# PHASE 4: GENERATE PLOTS
# ============================================================================

def phase_4_generate_plots(df_wide: pd.DataFrame, df_long: pd.DataFrame, verbose: bool = True) -> bool:
    """
    Phase 4: Generate consolidated plots.
    
    - Grid plots (3×3) for posteriors Level B
    - Grid plots (3×3) for posteriors Level A (diagnostic)
    
    Returns:
        True if successful
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"PHASE 4: GRID PLOTS GENERATION")
        print(f"{'='*70}\n")
    
    try:
        # TODO: Implement plot generation
        # - posteriors_b_pse_sd_evolution.png (3×3 grid)
        # - posteriors_b_jnd_sd_evolution.png (3×3 grid)
        # - posteriors_a_pse_sd_evolution.png (3×3 grid per model)
        
        if verbose:
            print("⚠ Plot generation not yet implemented")
        
        return True
        
    except Exception as e:
        print(f"✗ Phase 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# SAVE EXCEL
# ============================================================================

def save_consolidated_excel(df_wide: pd.DataFrame, df_long: pd.DataFrame, verbose: bool = True) -> bool:
    """
    Save consolidated Excel files to R/indata/.
    
    Files:
    - synthetic_data_wide.xlsx
    - synthetic_data_long.xlsx
    
    Returns:
        True if successful
    """
    if verbose:
        print(f"\nSaving consolidated Excel files...")
    
    try:
        EXCEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save wide
        wide_path = EXCEL_OUTPUT_DIR / "synthetic_data_wide.xlsx"
        df_wide.to_excel(wide_path, index=False)
        if verbose:
            print(f"  ✓ {wide_path} ({len(df_wide)} rows)")
        
        # Save long
        long_path = EXCEL_OUTPUT_DIR / "synthetic_data_long.xlsx"
        df_long.to_excel(long_path, index=False)
        if verbose:
            print(f"  ✓ {long_path} ({len(df_long)} rows)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error saving Excel: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main():
    """
    Main orchestrator: Run all phases or specific phases.
    """
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Pipeline")
    parser.add_argument("--phase", default="1-2,3-4", help="Phases to run (e.g., '1-2,3-4' or '1-4 ABS1')")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"SYNTHETIC DATA GENERATION - FULL PIPELINE")
    print(f"{'='*70}")
    
    # Phase 1-2: Generation
    if "1-2" in args.phase:
        print(f"\nRunning Phases 1-2 (Generation)...")
        all_results = run_all_phases_1_2(verbose=True)
        
        if not all_results:
            print(f"✗ No results from Phase 1-2")
            return False
        
        print(f"\n✓ All models Phase 1-2 complete")
    
    # Phase 3-4: Consolidation + Plots
    if "3-4" in args.phase:
        # Phase 3 (now uses UnifiedGBFProcessor, no need to pass all_results)
        df_wide, df_long = phase_3_calculate_metrics_and_consolidate(verbose=True)
        
        if df_wide.empty:
            print(f"✗ Phase 3 produced no data")
            return False
        
        # Save Excel
        if not save_consolidated_excel(df_wide, df_long, verbose=True):
            return False
        
        # Phase 4
        if not phase_4_generate_plots(df_wide, df_long, verbose=True):
            print(f"⚠ Phase 4 had warnings but continuing...")
        
        print(f"\n✓ Phases 3-4 complete")
    else:
        print(f"⚠ Skipping Phases 3-4")
    
    # Done
    print(f"\n{'='*70}")
    print(f"✓ SYNTHETIC DATA GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput:")
    print(f"  - GBF files: {OUTPUT_BASE}/*/group_*/*.txt")
    print(f"  - Group plots: {OUTPUT_BASE}/*/group_*/results/*.png")
    print(f"  - Consolidated Excel: {EXCEL_OUTPUT_DIR}/*.xlsx")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
