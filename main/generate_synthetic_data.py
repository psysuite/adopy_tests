#!/usr/bin/env python3
"""
Synthetic Data Generation - 3-Phase Pipeline Orchestrator

PHASE A (1-2): Generate GBF files + group plots
  - Runs: group_sim_gridrnd_{MODEL}.py for each model
  - Input: None (simulation config)
  - Output: 180 GBF files per model + group plots
  - Location: /data/output/sim_gridrnd/{MODEL}/group_*/*.txt

PHASE B (3): Calculate metrics → Excel
  - Reads: GBF files from Phase A
  - Uses: UnifiedGBFProcessor + MetricsCalculator
  - Calculate: All B1-B5 metrics (pse_*, jnd_*, stimulus_*, asymmetry_*, posterior_sd_*, etc.)
  - Output: Excel (wide + long) → R/indata/
  - Features: Multithread with incremental append (crash-safe)

PHASE C: Generate all 30 plots from Excel
  - Reads: Consolidated Excel files from Phase B
  - Output: 30 plots total
    - 27 per-modello: 6 local + 3 posteriors per model × 3 models
      Location: /data/output/sim_gridrnd/{MODEL}/
    - 3 consolidated: Comparison plots (average 3 models)
      Location: /data/output/plots/

Execution flags (modify at top of file):
  overwrite_12: Skip Phase A if files already exist (default: False)
  overwrite_3:  Skip Phase B if Excel already exists (default: True)
  overwrite_c:  Regenerate Phase C plots (default: True)

Usage:
  python main/generate_synthetic_data.py              # Full pipeline: A→B→C
  
Standalone Phase C (read existing Excel):
  Just set overwrite_12=False, overwrite_3=False, overwrite_c=True
"""

import sys
import logging
from pathlib import Path
from itertools import product
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.config import *
from analysis.orchestration.unified_processor import UnifiedGBFProcessor
from analysis.io.report_generator import _validate_excel_path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# EXECUTION CONFIGURATION (modify these flags to control pipeline)
# ============================================================================
# Flag meanings:
#   overwrite_a = False  → Skip Phase A if GBF files already exist
#   overwrite_b = False  → Skip Phase B if Excel already exists
#   overwrite_c = False  → Skip Phase C (plots only)
#
# Common scenarios:
#   Full pipeline:     overwrite_a=True, overwrite_b=True, overwrite_c=True
#   Regenerate plots:  overwrite_a=False, overwrite_b=False, overwrite_c=True
#   Quick test:        overwrite_a=False, overwrite_b=False, overwrite_c=True
overwrite_a = False
overwrite_b = False
overwrite_c = True

verbose = True
# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_BASE = Path(__file__).parent.parent / OUTPUT_BASE
EXCEL_OUTPUT_DIR = Path(__file__).parent.parent / EXCEL_OUTPUT_DIR

# ============================================================================
# PHASE 1-2: GENERATION (Call external scripts)
# ============================================================================
def run_phase_a_for_model(model_name: str, verbose: bool = True, overwrite:bool=True) -> list:
    """
    Run Phase A (data generation) for a single model.
    
    Calls: group_sim_gridrnd_ABS1.py, group_sim_gridrnd_REL1.py, or group_sim_gridrnd_REL2.py
    
    Returns:
        List of (group_idx, pse_center, jnd_center, group_results) tuples
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"PHASE A: DATA GENERATION - {model_name}")
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
    
    # Call main() to run Phase A
    try:
        all_group_results = mod.main(overwrite_a)
        if verbose:
            print(f"\n✓ {model_name} Phase A complete: {len(all_group_results)} groups processed")
        return all_group_results
    except Exception as e:
        print(f"✗ Error running Phase A for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return []

# ============================================================================
# PHASE 3: CALCULATE METRICS + CONSOLIDATE (UNIFIED PROCESSING)
# ============================================================================
def run_phase_b_calculate_metrics(verbose: bool = True, overwrite: bool = True) -> tuple:
    """
    Phase B: Calculate metrics on all GBF files using UnifiedGBFProcessor.
    
    Logic:
    - If overwrite=True: Delete Excel files and recalculate ALL metrics from scratch
    - If overwrite=False: 
      - Load existing Excel files
      - Check which subject-group-model combinations are missing
      - Calculate ONLY missing ones and append to Excel
      - If all exist: skip calculation and load existing
    
    Args:
        verbose: Print progress
        overwrite: If True, recalculate all from scratch. If False, incremental update.
    
    Returns:
        (df_wide, df_long) DataFrames
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"PHASE B: METRICS CALCULATION (UNIFIED + MULTITHREAD)")
        print(f"{'='*70}\n")

    wide_path = EXCEL_OUTPUT_DIR / "synthetic_data_wide.xlsx"
    long_path = EXCEL_OUTPUT_DIR / "synthetic_data_long.xlsx"
    
    # ====== CASE 1: overwrite=True → Delete old Excel and recalculate all ======
    if overwrite:
        if verbose and wide_path.exists():
            print(f"[OVERWRITE=TRUE] Deleting existing Excel files...")
            print(f"  Recalculating ALL metrics from scratch...\n")
        
        # Delete old files
        if wide_path.exists():
            wide_path.unlink()
        if long_path.exists():
            long_path.unlink()
        
        # Process all models from scratch
        all_dfs_wide = []
        all_dfs_long = []
        
        for model_name in MODELS:
            model_output_dir = Path(OUTPUT_BASE) / model_name
            
            if not model_output_dir.exists():
                if verbose:
                    print(f"⚠ Skipping {model_name} (directory not found)")
                continue
            
            if verbose:
                print(f"Processing {model_name} (multithread, full calculation)...")
            
            try:
                processor = UnifiedGBFProcessor(
                    gbf_dir=model_output_dir,
                    model_name=model_name,
                    data_type='synthetic',
                    output_dir=EXCEL_OUTPUT_DIR,
                    pse_grid=PSE_GRID,
                    jnd_grid=JND_GRID,
                    offset=OFFSET,
                    trial_blocks=TRIAL_BLOCKS,
                    verbose=verbose,
                    use_multithread=True,
                )
                
                df_wide, df_long = processor.process()
                
                if not df_wide.empty:
                    all_dfs_wide.append(df_wide)
                    all_dfs_long.append(df_long)
                    if verbose:
                        print(f"  ✓ {model_name}: {len(df_wide)} subjects\n")
            
            except Exception as e:
                logger.error(f"Error processing {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Combine and save
        if all_dfs_wide:
            df_wide = pd.concat(all_dfs_wide, ignore_index=True)
            df_long = pd.concat(all_dfs_long, ignore_index=True)
            
            if verbose:
                print(f"✓ Phase B complete: {len(df_wide)} total subjects (recalculated)")
            
            if not save_consolidated_excel(df_wide, df_long, verbose=True):
                raise Exception(f"✗ Phase B: error saving Excel files")
            
            return df_wide, df_long
        else:
            raise Exception(f"✗ Phase B: No data processed")
    
    # ====== CASE 2: overwrite=False → Incremental update ======
    else:
        if verbose:
            print(f"[OVERWRITE=FALSE] Checking for incomplete data...\n")
        
        # Load existing Excel if present
        df_wide_existing = None
        df_long_existing = None
        processed_subjects = set()
        
        if wide_path.exists() and long_path.exists():
            try:
                df_wide_existing = pd.read_excel(wide_path)
                df_long_existing = pd.read_excel(long_path)
                
                # Track which (model, subject_id, group) combinations already exist
                for _, row in df_wide_existing.iterrows():
                    key = (row['model'], row['subject_id'], row['group'])
                    processed_subjects.add(key)
                
                if verbose:
                    print(f"Loaded existing Excel:")
                    print(f"  - {len(df_wide_existing)} rows in wide format")
                    print(f"  - {len(processed_subjects)} unique subject-group-model combinations\n")
            
            except Exception as e:
                print(f"⚠ Error loading existing Excel: {e}")
                print(f"  Will recalculate from scratch instead...\n")
                df_wide_existing = None
        
        # Expected: derive from config (not hardcoded)
        expected_combinations = len(MODELS) * len(PSE_GRID) * len(JND_GRID) * N_SUBJECTS_PER_GROUP
        
        if verbose:
            print(f"Expected combinations: {len(MODELS)} models × {len(PSE_GRID)} PSE × {len(JND_GRID)} JND × {N_SUBJECTS_PER_GROUP} subjects = {expected_combinations}\n")
        
        if df_wide_existing is not None and len(processed_subjects) == expected_combinations:
            if verbose:
                print(f"✓ All {expected_combinations} subject-group-model combinations exist (complete)")
                print(f"  Skipping Phase B calculation\n")
            
            return df_wide_existing, df_long_existing
        
        # ====== INCREMENTAL: Calculate only missing combinations ======
        if verbose:
            missing_count = expected_combinations - len(processed_subjects)
            print(f"⚠ {missing_count} combinations missing")
            print(f"  Will calculate only missing subjects...\n")
        
        all_dfs_wide = []
        all_dfs_long = []
        if df_wide_existing is not None:
            all_dfs_wide.append(df_wide_existing)
        
        # Process each model for missing subjects
        for model_name in MODELS:
            model_output_dir = Path(OUTPUT_BASE) / model_name
            
            if not model_output_dir.exists():
                continue
            
            if verbose:
                print(f"Processing {model_name} (incremental, only missing)...")
            
            try:
                processor = UnifiedGBFProcessor(
                    gbf_dir=model_output_dir,
                    model_name=model_name,
                    data_type='synthetic',
                    output_dir=EXCEL_OUTPUT_DIR,
                    pse_grid=PSE_GRID,
                    jnd_grid=JND_GRID,
                    offset=OFFSET,
                    trial_blocks=TRIAL_BLOCKS,
                    verbose=verbose,
                    use_multithread=True,
                )
                
                df_wide, df_long = processor.process()
                
                if not df_wide.empty:
                    # Filter to only missing subjects
                    df_new = df_wide[~df_wide.apply(
                        lambda row: (row['model'], row['subject_id'], row['group']) in processed_subjects,
                        axis=1
                    )]
                    
                    if len(df_new) > 0:
                        all_dfs_wide.append(df_new)
                        if verbose:
                            print(f"  ✓ {model_name}: {len(df_new)} missing subjects added\n")
                    else:
                        if verbose:
                            print(f"  ✓ {model_name}: all subjects already exist\n")
            
            except Exception as e:
                logger.error(f"Error processing {model_name}: {e}")
                continue
        
        # Combine and save
        if len(all_dfs_wide) > 1:  # More than just the existing data
            df_wide = pd.concat(all_dfs_wide, ignore_index=True)
            
            # IMPORTANT: Regenerate df_long from df_wide to avoid overwriting with empty DataFrame
            from analysis.io.report_generator import generate_long_format_from_dataframe
            df_long = generate_long_format_from_dataframe(df_wide, data_type='synthetic')
            
            if verbose:
                print(f"✓ Phase B complete: {len(df_wide)} total subjects (incremental update)")
                print(f"  Regenerated long format: {len(df_long)} rows\n")
            
            if not save_consolidated_excel(df_wide, df_long, verbose=True):
                raise Exception(f"✗ Phase B: error saving Excel files")
            
            return df_wide, df_long
        
        elif df_wide_existing is not None:
            return df_wide_existing, df_long_existing
        
        else:
            raise Exception(f"✗ Phase B: No data found or processed")

# ============================================================================
# PHASE C: GENERATE ALL PLOTS FROM EXCEL (30 plots total)
# ============================================================================
def phase_c_generate_all_plots(df_wide: 'pd.DataFrame|None' = None, 
                               excel_dir: Path|None = None, verbose: bool = True) -> bool:
    """
    Phase C: Generate ALL 86 PLOTS from Excel.
    
    **Per-Model Plots (27 per model × 3 models = 81 total):**
    - Group plots: 18 plots per model (2 per group: stimulus distribution histogram + psychometric curve × 9 groups)
    - Local analysis: 9 plots per model
      * Grid Psychometric (3×3)
      * Grid Stimulus Distribution (3×3)
      * Asymmetry Modulo
      * Asymmetry Scatter Envelope
      * Stimulus Center Evolution
      * Stimulus Spread Evolution
      * Posterior SD PSE (3×3 grid, Level B)
      * Posterior SD JND (3×3 grid, Level B)
      * Posteriors Heatmap (Level A)
    
    **Comparison Plots (5 consolidated plots across 3 models):**
    - Comparison Posteriors PSE (3×3 grid overlay)
    - Comparison Posteriors JND (3×3 grid overlay)
    - Comparison Posteriors Heatmap (Level A)
    - Comparison Grid Psychometric (model averages)
    - Comparison Latency KDE (kernel density estimates)
    
    **Grand Total: 81 (per-model) + 5 (comparison) = 86 plots**
    
    Output:
    - Per model plots: /data/output/sim_gridrnd/{MODEL}/
    - Comparison plots: /data/output/plots/
    
    Args:
        df_wide: DataFrame from Phase B (optional; loads from Excel if None)
        excel_dir: Path to directory containing Excel files
        verbose: Print progress
    
    Returns:
        True if successful
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"PHASE C: ALL PLOTS GENERATION (30 plots: 27 per-model + 3 consolidated)")
        print(f"{'='*70}\n")
    
    # Load data from Excel if not provided
    if df_wide is None:
        if excel_dir is None:
            print("✗ Either df_wide or excel_dir must be provided")
            return False
        
        excel_dir = Path(excel_dir)
        wide_file = excel_dir / 'synthetic_data_wide.xlsx'
        
        if not wide_file.exists():
            print(f"✗ Excel file not found: {wide_file}")
            return False
        
        if verbose:
            print(f"Loading data from {wide_file}...")
        
        try:
            df_wide = pd.read_excel(wide_file)
            if verbose:
                print(f"✓ Loaded {len(df_wide)} rows\n")
        except Exception as e:
            print(f"✗ Error loading Excel: {e}")
            return False
    
    # Import all Phase C plot functions
    from analysis.plot.plotting import (
        create_phase_c_grid_psychometric_from_group_plots,
        create_phase_c_grid_stimulus_distribution_from_group_plots,
        create_phase_c_asymmetry_modulo,
        create_phase_c_asymmetry_scatter_envelope,
        create_phase_c_stimulus_center_evolution,
        create_phase_c_stimulus_spread_evolution,
        create_phase_c_posteriors_b_pse_sd,
        create_phase_c_posteriors_b_jnd_sd,
        create_phase_c_posteriors_a_heatmap,
        create_phase_c_comparison_posteriors_b_pse_sd,
        create_phase_c_comparison_posteriors_b_jnd_sd,
        create_phase_c_comparison_posteriors_a_heatmap,
        create_phase_c_comparison_grid_psychometric,
        create_phase_c_comparison_latency_kde_grid,
    )
    from main.config import PSE_GRID, JND_GRID, MODELS, OUTPUT_BASE
    
    all_success = True
    plot_count = 0
    
    try:
        # Ensure output directories exist
        project_root = Path(__file__).parent.parent
        plots_dir = project_root / "data" / "output" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # ====== GENERATE PER-MODELLO PLOTS (81 plots: 27 per model × 3) ======
        print("GENERATING PER-MODEL PLOTS (27 plots per model × 3 = 81 total)\n")
        
        for model_name in MODELS:
            model_output_dir = project_root / OUTPUT_BASE / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Filter data for this model
            df_model = df_wide[df_wide['model'] == model_name]
            
            if df_model.empty:
                print(f"⚠ No data for model {model_name}, skipping...")
                continue
            
            if verbose:
                print(f"{model_name}: Processing {len(df_model)} subjects")
            
            # 2 grid plots (assembly of 3×3 = 9 group plots each)
            if verbose:
                print(f"  Grid plots (2):")
            
            if create_phase_c_grid_psychometric_from_group_plots(model_name, PSE_GRID, JND_GRID, model_output_dir):
                plot_count += 1
            else:
                all_success = False
            
            if create_phase_c_grid_stimulus_distribution_from_group_plots(model_name, PSE_GRID, JND_GRID, model_output_dir):
                plot_count += 1
            else:
                all_success = False
            
            # 4 local plots
            if verbose:
                print(f"  Local plots (4):")
            
            if create_phase_c_asymmetry_modulo(df_model, model_name, PSE_GRID, JND_GRID, model_output_dir):
                plot_count += 1
            else:
                all_success = False
            
            if create_phase_c_asymmetry_scatter_envelope(df_model, model_name, PSE_GRID, JND_GRID, model_output_dir):
                plot_count += 1
            else:
                all_success = False
            
            if create_phase_c_stimulus_center_evolution(df_model, model_name, model_output_dir):
                plot_count += 1
            else:
                all_success = False
            
            if create_phase_c_stimulus_spread_evolution(df_model, model_name, model_output_dir):
                plot_count += 1
            else:
                all_success = False
            
            # 9 group plots (2 per group: histogram + psychometric)
            if verbose:
                print(f"  Group plots (18 = 2 per group × 9 groups):")
            
            from analysis.plot.plotting import plot_group_histograms, plot_group_psychometric, load_group_rows
            
            # Create temp directory for group plots
            temp_dir = model_output_dir / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            for group_idx, (pse_center, jnd_center) in enumerate(product(PSE_GRID, JND_GRID), 1):
                group_dir = project_root / OUTPUT_BASE / model_name / f"group_{pse_center}_{jnd_center}"
                
                if not group_dir.exists():
                    if verbose:
                        print(f"    ⚠ Group {group_idx} not found: {group_dir}")
                    continue
                
                try:
                    # Load GBF files for this group
                    group_rows = load_group_rows(
                        group_dir,
                        model_name,
                        group_idx,
                        pse_center,
                        jnd_center,
                        N_SUBJECTS_PER_GROUP,
                        offset=OFFSET
                    )
                    
                    if group_rows:
                        # Wrap in list because plot functions expect list of lists
                        all_rows_list = [group_rows]
                        group_label = f"{model_name} G{group_idx}: PSE={pse_center}, JND={jnd_center}"
                        
                        # Histogram plot → save in temp/
                        plot_group_histograms(
                            all_rows_list,
                            str(temp_dir),
                            f"{model_name}_G{group_idx}",
                            OFFSET,
                            group_label=group_label
                        )
                        plot_count += 1
                        
                        # Psychometric plot → save in temp/
                        plot_group_psychometric(
                            all_rows_list,
                            str(temp_dir),
                            f"{model_name}_G{group_idx}",
                            OFFSET,
                            group_label=group_label
                        )
                        plot_count += 1
                
                except Exception as e:
                    print(f"    ✗ Error generating group plots for Group {group_idx}: {e}")
                    all_success = False
            
            # 3 posteriors plots
            if verbose:
                print(f"  Posteriors plots (3):")
            
            if create_phase_c_posteriors_b_pse_sd(df_model, model_name, PSE_GRID, JND_GRID, model_output_dir):
                plot_count += 1
            else:
                all_success = False
            
            if create_phase_c_posteriors_b_jnd_sd(df_model, model_name, PSE_GRID, JND_GRID, model_output_dir):
                plot_count += 1
            else:
                all_success = False
            
            if create_phase_c_posteriors_a_heatmap(df_model, model_name, PSE_GRID, JND_GRID, model_output_dir):
                plot_count += 1
            else:
                all_success = False
            
            print()
        
        # ====== GENERATE COMPARISON PLOTS (5 plots consolidated) ======
        print("GENERATING COMPARISON PLOTS (5 plots consolidated)\n")
        print(f"  Comparison plots:")
        
        # Prepare data by model for comparison functions
        dfs_by_model = {model_name: df_wide[df_wide['model'] == model_name] for model_name in MODELS}
        
        if create_phase_c_comparison_posteriors_b_pse_sd(dfs_by_model, PSE_GRID, JND_GRID, plots_dir):
            plot_count += 1
        else:
            all_success = False
        
        if create_phase_c_comparison_posteriors_b_jnd_sd(dfs_by_model, PSE_GRID, JND_GRID, plots_dir):
            plot_count += 1
        else:
            all_success = False
        
        if create_phase_c_comparison_posteriors_a_heatmap(dfs_by_model, PSE_GRID, JND_GRID, plots_dir):
            plot_count += 1
        else:
            all_success = False

        if create_phase_c_comparison_grid_psychometric(PSE_GRID, JND_GRID, plots_dir):
            plot_count += 1
        else:
            all_success = False

        if create_phase_c_comparison_latency_kde_grid(PSE_GRID, JND_GRID, plots_dir):
            plot_count += 1
        else:
            all_success = False
        
        print()
        
        if all_success and verbose:
            print(f"{'='*70}")
            print(f"✓ PHASE C COMPLETE: {plot_count} plots generated (81 per-model + 5 comparison)")
            print(f"{'='*70}")
            print(f"\nPlot locations:")
            print(f"  - Per-modello: {OUTPUT_BASE}/{{MODEL}}/")
            print(f"  - Consolidated: {plots_dir}/")
        
        return all_success
        
    except Exception as e:
        print(f"✗ Phase C failed: {e}")
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
        try:
            wide_path = _validate_excel_path(str(wide_path))
            df_wide.to_excel(wide_path, index=False)
            if verbose:
                print(f"  ✓ {wide_path} ({len(df_wide)} rows)")
        except ValueError as e:
            print(f"✗ Invalid Excel path for wide: {e}")
            return False
        
        # Save long
        long_path = EXCEL_OUTPUT_DIR / "synthetic_data_long.xlsx"
        try:
            long_path = _validate_excel_path(str(long_path))
            df_long.to_excel(long_path, index=False)
            if verbose:
                print(f"  ✓ {long_path} ({len(df_long)} rows)")
        except ValueError as e:
            print(f"✗ Invalid Excel path for long: {e}")
            return False
        
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
    Main orchestrator: Run phases based on flags.
    """
    print(f"\n{'='*70}")
    print(f"SYNTHETIC DATA GENERATION - flags: A:{overwrite_a}, B:{overwrite_b}, C:{overwrite_c}")
    print(f"{'='*70}")
    
    # Phase A: Generation
    print(f"\nRunning Phase A (GBF Generation)...")

    for model_name in MODELS:
        run_phase_a_for_model(model_name, verbose=verbose, overwrite=overwrite_a)
    print(f"\n✓ All models Phase A complete")
    
    # Phase B: Calculate metrics and consolidate
    df_wide, df_long = run_phase_b_calculate_metrics(verbose=True, overwrite=overwrite_b)

    # Phase C: Generate all plots (27 per-model + 3 consolidated)
    if overwrite_c:
        print(f"\nRunning Phase C (all plots from Excel)...")
        if not phase_c_generate_all_plots(df_wide=df_wide, excel_dir=EXCEL_OUTPUT_DIR, verbose=True):
            print(f"✗ Phase C failed")
            return False

    # Done
    print(f"\n{'='*70}")
    print(f"✓ SYNTHETIC DATA GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput:")
    print(f"  - GBF files: {OUTPUT_BASE}/*/group_*/*.txt")
    print(f"  - Consolidated Excel: {EXCEL_OUTPUT_DIR}/*.xlsx")
    print(f"  - All Phase C plots (30): {OUTPUT_BASE}/*/  +  data/output/plots/")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
