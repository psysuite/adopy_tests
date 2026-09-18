#!/usr/bin/env python3
"""
Patch to add progressive pct_correct columns to Excel files (synthetic and/or real).

This script:
1. Reads existing wide Excel
2. For each subject, reads GBF file and calculates pct_correct at each trial block
3. Adds columns: pct_correct_40, pct_correct_60, ..., pct_correct_200
4. Updates Excel in-place and regenerates long format

Structure for GBF files:
- SYNTHETIC: /data/CODE/python/adopy_tests/data/output/sim_gridrnd/{ABS1|REL1|REL2}/group_*/
- REAL: /data/CODE/python/adopy_tests/data (flat structure with *.txt files)

Usage (synthetic only, default):
    python3 patch_add_pct_correct.py

Usage (real only):
    python3 patch_add_pct_correct.py --data-type real

Usage (both):
    python3 patch_add_pct_correct.py --data-type both

Usage (custom paths):
    python3 patch_add_pct_correct.py \\
        --data-type synthetic \\
        --gbf-root /path/to/sim_gridrnd \\
        --excel-path /path/to/synthetic_data_wide.xlsx
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.config import TRIAL_BLOCKS, MAX_WORKERS
from analysis.io.converter import read_gbf_file
from analysis.io.report_generator import (
    generate_long_format_from_dataframe,
    save_wide_format_to_excel,
    save_long_format_to_excel,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def calculate_pct_correct_progressive(gbf_rows: list, trial_blocks: list = None, offset: float = 500) -> Dict[str, float]:
    """
    Calculate percentage of correct responses at each trial block.
    
    For psychophysical discrimination task:
    - Correct if: (lat > offset) == user_ans
    - pct_correct = (# correct) / (# trials) × 100
    
    Args:
        gbf_rows: List of GBF row dicts with 'lat' (latency) and 'user_ans' (response)
        trial_blocks: List of trial block sizes (default: TRIAL_BLOCKS)
        offset: Reference latency threshold (default: 500ms)
        
    Returns:
        Dict with keys like 'pct_correct_40', 'pct_correct_60', etc.
    """
    if trial_blocks is None:
        trial_blocks = TRIAL_BLOCKS
    
    result = {}
    
    for block_size in trial_blocks:
        if block_size > len(gbf_rows):
            continue
        
        block_rows = gbf_rows[:block_size]
        correct_count = 0
        
        for r in block_rows:
            lat = r.get('lat', 0)
            user_ans = r.get('user_ans', 0)
            # Correct if stimulus direction matches response
            correct_response = 1 if lat > offset else 0
            if user_ans == correct_response:
                correct_count += 1
        
        pct_correct = (correct_count / block_size) * 100.0
        result[f'pct_correct_{block_size}'] = float(pct_correct)
    
    return result


def discover_synthetic_gbf_files(gbf_root: Path) -> Dict[Tuple[str, str], Path]:
    """
    Discover synthetic GBF files: {ABS1|REL1|REL2}/group_*/
    
    Args:
        gbf_root: Root directory (e.g., /data/.../sim_gridrnd)
        
    Returns:
        Dict mapping (subject_id, model) -> gbf_path
    """
    gbf_lookup = {}
    
    for model_dir in ['ABS1', 'REL1', 'REL2']:
        model_path = gbf_root / model_dir
        if not model_path.exists():
            logger.warning(f"Model directory not found: {model_path}")
            continue
        
        for group_dir in sorted(model_path.glob('group_*_*')):
            if not group_dir.is_dir():
                continue
            
            for gbf_path in sorted(group_dir.glob('*.txt')):
                filename = gbf_path.stem
                try:
                    parts = filename.split('_')
                    subject_id = parts[0]  # S123
                    gbf_lookup[(subject_id, model_dir)] = gbf_path
                except Exception as e:
                    logger.debug(f"Failed to parse {filename}: {e}")
    
    return gbf_lookup


def discover_real_gbf_files(gbf_root: Path) -> Dict[str, Path]:
    """
    Discover real GBF files: flat directory with *.txt files
    
    Args:
        gbf_root: Root directory with GBF files
        
    Returns:
        Dict mapping subject_id -> gbf_path
    """
    gbf_lookup = {}
    
    for gbf_path in sorted(gbf_root.glob('*.txt')):
        filename = gbf_path.stem
        try:
            parts = filename.split('_')
            subject_id = parts[0]  # A01, A02, etc.
            gbf_lookup[subject_id] = gbf_path
        except Exception as e:
            logger.debug(f"Failed to parse {filename}: {e}")
    
    return gbf_lookup


def patch_synthetic(gbf_root: Path, excel_path: Path, verbose: bool = True) -> bool:
    """
    Add pct_correct columns to synthetic data Excel (multithreaded).
    
    Args:
        gbf_root: Root directory containing {ABS1|REL1|REL2}/group_*
        excel_path: Path to synthetic_data_wide.xlsx
        verbose: Print progress
        
    Returns:
        True if successful
    """
    if not excel_path.exists():
        logger.warning(f"Excel file not found: {excel_path}")
        return False
    
    try:
        df_wide = pd.read_excel(excel_path)
        logger.info(f"Loaded Excel: {len(df_wide)} rows")
    except Exception as e:
        logger.error(f"Failed to read Excel: {e}")
        return False
    
    # Check if columns already exist
    pct_cols = [c for c in df_wide.columns if c.startswith('pct_correct_')]
    if pct_cols:
        logger.info(f"Columns already exist: {pct_cols}")
        return True
    
    # Discover GBF files
    gbf_lookup = discover_synthetic_gbf_files(gbf_root)
    logger.info(f"Found {len(gbf_lookup)} synthetic GBF files in {gbf_root}")
    
    if not gbf_lookup:
        logger.error("No synthetic GBF files found")
        return False
    
    # Thread-safe storage for results
    results_lock = threading.Lock()
    results = {}  # idx -> pct_dict
    processed = [0]
    skipped = [0]
    
    def process_row(idx: int, row: pd.Series) -> None:
        """Process single row and store result."""
        subject_id = row.get('subject_id')
        model = row.get('model')
        
        if pd.isna(subject_id) or pd.isna(model):
            return
        
        key = (str(subject_id), str(model))
        if key not in gbf_lookup:
            with results_lock:
                skipped[0] += 1
            return
        
        gbf_path = gbf_lookup[key]
        
        try:
            gbf_rows = read_gbf_file(str(gbf_path))
            if not gbf_rows:
                logger.warning(f"Empty GBF: {gbf_path}")
                return
            
            pct_dict = calculate_pct_correct_progressive(gbf_rows)
            
            with results_lock:
                results[idx] = pct_dict
                processed[0] += 1
                if verbose and processed[0] % 30 == 0:
                    logger.info(f"  [{processed[0]}/{len(df_wide)}] Processed")
        
        except Exception as e:
            logger.error(f"Failed to process {gbf_path}: {e}")
    
    # Process rows in parallel
    logger.info(f"Using {MAX_WORKERS} worker threads...\n")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_row, idx, row): idx
            for idx, row in df_wide.iterrows()
        }
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    logger.info(f"Successfully processed {processed[0]}/{len(df_wide)} rows ({skipped[0]} skipped)")
    
    # Apply results to dataframe
    for idx, pct_dict in results.items():
        for col, val in pct_dict.items():
            df_wide.loc[idx, col] = val
    
    # Save updated Excel
    try:
        save_wide_format_to_excel(df_wide, str(excel_path))
        logger.info(f"✓ Updated wide Excel: {len(df_wide)} rows")
    except Exception as e:
        logger.error(f"Failed to save Excel: {e}")
        return False
    
    return True


def patch_real(gbf_root: Path, excel_path: Path, verbose: bool = True) -> bool:
    """
    Add pct_correct columns to real data Excel.
    
    Args:
        gbf_root: Root directory containing GBF files (flat structure)
        excel_path: Path to real_data_wide.xlsx
        verbose: Print progress
        
    Returns:
        True if successful
    """
    if not excel_path.exists():
        logger.warning(f"Excel file not found: {excel_path}")
        return False
    
    try:
        df_wide = pd.read_excel(excel_path)
        logger.info(f"Loaded Excel: {len(df_wide)} rows")
    except Exception as e:
        logger.error(f"Failed to read Excel: {e}")
        return False
    
    # Check if columns already exist
    pct_cols = [c for c in df_wide.columns if c.startswith('pct_correct_')]
    if pct_cols:
        logger.info(f"Columns already exist: {pct_cols}")
        return True
    
    # Discover GBF files
    gbf_lookup = discover_real_gbf_files(gbf_root)
    logger.info(f"Found {len(gbf_lookup)} real GBF files in {gbf_root}")
    
    if not gbf_lookup:
        logger.error("No real GBF files found")
        return False
    
    # Process each row
    processed = 0
    skipped = 0
    
    for idx, row in df_wide.iterrows():
        subject_id = row.get('subj')
        
        if pd.isna(subject_id):
            continue
        
        subject_id = str(subject_id)
        if subject_id not in gbf_lookup:
            skipped += 1
            if skipped <= 5:
                logger.debug(f"GBF not found for {subject_id}")
            continue
        
        gbf_path = gbf_lookup[subject_id]
        
        try:
            gbf_rows = read_gbf_file(str(gbf_path))
            if not gbf_rows:
                logger.warning(f"Empty GBF: {gbf_path}")
                continue
            
            pct_dict = calculate_pct_correct_progressive(gbf_rows)
            
            # Add columns to dataframe
            for col, val in pct_dict.items():
                df_wide.loc[idx, col] = val
            
            processed += 1
            if verbose and processed % 10 == 0:
                logger.info(f"  [{processed}/{len(df_wide)}] Processed {subject_id}")
        
        except Exception as e:
            logger.error(f"Failed to process {gbf_path}: {e}")
    
    logger.info(f"Successfully processed {processed}/{len(df_wide)} rows ({skipped} skipped)")
    
    # Save updated Excel
    try:
        save_wide_format_to_excel(df_wide, str(excel_path))
        logger.info(f"✓ Updated wide Excel: {len(df_wide)} rows")
    except Exception as e:
        logger.error(f"Failed to save Excel: {e}")
        return False
    
    return True


def regenerate_long_format(data_dir: Path, data_type: str) -> bool:
    """
    Regenerate long format Excel from updated wide format.
    
    Args:
        data_dir: Directory containing data Excel files
        data_type: 'synthetic' or 'real'
        
    Returns:
        True if successful
    """
    wide_path = data_dir / f'{data_type}_data_wide.xlsx'
    long_path = data_dir / f'{data_type}_data_long.xlsx'
    
    try:
        df_wide = pd.read_excel(wide_path)
        df_long = generate_long_format_from_dataframe(df_wide, data_type=data_type)
        save_long_format_to_excel(df_long, str(long_path))
        logger.info(f"✓ Regenerated long format: {len(df_long)} rows")
        return True
    except Exception as e:
        logger.error(f"Failed to regenerate long format: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Add pct_correct columns to Excel files"
    )
    parser.add_argument(
        '--data-type',
        choices=['synthetic', 'real', 'both'],
        default='synthetic',
        help='Which data type to patch (default: synthetic)'
    )
    parser.add_argument(
        '--gbf-root',
        type=Path,
        default=None,
        help='Root directory for synthetic GBF files (auto-detected if not provided)'
    )
    parser.add_argument(
        '--real-gbf-root',
        type=Path,
        default=None,
        help='Root directory for real GBF files (auto-detected if not provided)'
    )
    parser.add_argument(
        '--excel-path',
        type=Path,
        default=None,
        help='Path to synthetic_data_wide.xlsx (auto-detected if not provided)'
    )
    parser.add_argument(
        '--real-excel-path',
        type=Path,
        default=None,
        help='Path to real_data_wide.xlsx (auto-detected if not provided)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=None,
        help='Base data directory (auto-detected if not provided)'
    )
    parser.add_argument(
        '--no-regenerate-long',
        action='store_true',
        help='Skip regenerating long format'
    )
    
    args = parser.parse_args()
    
    # Auto-detect paths if not provided
    project_root = Path(__file__).parent.parent
    data_dir = args.data_dir or (project_root / 'data' / 'output')
    
    synthetic_gbf_root = args.gbf_root or (project_root / 'data' / 'output' / 'sim_gridrnd')
    synthetic_excel_path = args.excel_path or (project_root / 'R' / 'indata' / 'synthetic_data_wide.xlsx')
    
    real_gbf_root = args.real_gbf_root or (project_root / 'data')
    real_excel_path = args.real_excel_path or (data_dir / 'real_data_wide.xlsx')
    
    logger.info(f"\n{'='*70}")
    logger.info("PATCH: Adding pct_correct columns")
    logger.info(f"{'='*70}\n")
    
    success = True
    
    if args.data_type in ['synthetic', 'both']:
        logger.info("Patching SYNTHETIC data...")
        logger.info(f"  GBF root: {synthetic_gbf_root}")
        logger.info(f"  Excel: {synthetic_excel_path}\n")
        
        if synthetic_gbf_root.exists() and synthetic_excel_path.exists():
            if not patch_synthetic(synthetic_gbf_root, synthetic_excel_path):
                success = False
        else:
            logger.error(f"Synthetic data not found at expected paths")
            if not synthetic_gbf_root.exists():
                logger.error(f"  Missing GBF root: {synthetic_gbf_root}")
            if not synthetic_excel_path.exists():
                logger.error(f"  Missing Excel: {synthetic_excel_path}")
            success = False
    
    if args.data_type in ['real', 'both']:
        logger.info("\nPatching REAL data...")
        logger.info(f"  GBF root: {real_gbf_root}")
        logger.info(f"  Excel: {real_excel_path}\n")
        
        if real_gbf_root.exists() and real_excel_path.exists():
            if not patch_real(real_gbf_root, real_excel_path):
                success = False
        else:
            logger.warning(f"Real data not found at expected paths (optional)")
            if not real_gbf_root.exists():
                logger.warning(f"  Missing GBF root: {real_gbf_root}")
            if not real_excel_path.exists():
                logger.warning(f"  Missing Excel: {real_excel_path}")
    
    if not args.no_regenerate_long:
        logger.info("\nRegenerating long format...")
        if args.data_type in ['synthetic', 'both']:
            if synthetic_excel_path.exists():
                regenerate_long_format(data_dir, 'synthetic')
        if args.data_type in ['real', 'both']:
            if real_excel_path.exists():
                regenerate_long_format(data_dir, 'real')
    
    logger.info(f"\n{'='*70}")
    logger.info("✓ PATCH COMPLETE" if success else "✗ PATCH FAILED")
    logger.info(f"{'='*70}\n")
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
