"""
Temporary repair script: recalculate 6 convergence metrics using ground truth as reference.

This script reads the synthetic_data_wide.xlsx file, extracts pse_true/jnd_true,
and recalculates these 6 metrics using ground truth as reference:
  1. pse_stability_block
  2. jnd_stability_block
  3. pse_auc
  4. jnd_auc
  5. pse_error (final_pse - pse_true)
  6. jnd_error (final_jnd - jnd_true)

Then regenerates the long format from the updated wide format.

Usage:
    python main/repair_synthetic_metrics.py
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main.config import TRIAL_BLOCKS
from analysis.io.report_generator import generate_long_format_from_dataframe, _validate_excel_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Excel file path
EXCEL_PATH = PROJECT_ROOT / "R" / "indata" / "synthetic_data_wide.xlsx"

# Trial blocks for convergence analysis
BLOCKS = list(TRIAL_BLOCKS)  # [40, 60, 80, 100, 120, 140, 160, 180, 200]


def calculate_stability_point(values: list, ground_truth: float, threshold: float = 0.10) -> int:
    """
    Find first block where param is within threshold% of ground truth.
    
    Args:
        values: Parameter values at each block
        ground_truth: Reference value (pse_true or jnd_true)
        threshold: Fractional threshold (0.10 = 10%)
    
    Returns:
        Block number where stable, or 200 if never stable
    """
    if not values or abs(ground_truth) < 1e-10:
        return 200
    
    for i, block in enumerate(BLOCKS):
        if i >= len(values):
            break
        if np.isnan(values[i]):
            continue
        diff_pct = abs(values[i] - ground_truth) / abs(ground_truth) * 100
        if diff_pct < threshold * 100:
            return block
    
    return 200


def calculate_auc_metric(values: list, ground_truth: float, trial_interval: int = 20) -> float:
    """
    Calculate AUC: Sum of absolute errors across all trial blocks.
    
    Args:
        values: Parameter values at each block [40, 60, 80, ..., 200]
        ground_truth: Reference value (pse_true or jnd_true)
        trial_interval: Interval between trial blocks (default 20)
    
    Returns:
        Sum of absolute errors (in ms) across all blocks
        Formula: sum(|value[i] - ground_truth| * trial_interval)
        This represents total cumulative distance from ground truth
    """
    if not values or abs(ground_truth) < 1e-10:
        return np.nan
    
    cumulative_error = 0.0
    for val in values:
        if not np.isnan(val):
            error = abs(val - ground_truth)
            cumulative_error += error * trial_interval
    
    return float(cumulative_error) if cumulative_error > 0 else np.nan


def main():
    """Load Excel, recalculate 6 metrics, regenerate long format, overwrite both Excel files."""
    
    logger.info(f"Reading {EXCEL_PATH}...")
    if not EXCEL_PATH.exists():
        logger.error(f"File not found: {EXCEL_PATH}")
        return False
    
    try:
        df_wide = pd.read_excel(EXCEL_PATH)
    except Exception as e:
        logger.error(f"Failed to read Excel: {e}")
        return False
    
    logger.info(f"Loaded {len(df_wide)} rows")
    logger.info(f"Expected: 3 models × 9 groups × 20 subjects = 540 rows")
    
    # Initialize result columns
    df_wide['pse_stability_block'] = 200
    df_wide['jnd_stability_block'] = 200
    df_wide['pse_auc'] = np.nan
    df_wide['jnd_auc'] = np.nan
    df_wide['pse_error'] = np.nan
    df_wide['jnd_error'] = np.nan
    
    # Process each row
    for idx, row in df_wide.iterrows():
        pse_true = row['pse_true']
        jnd_true = row['jnd_true']
        
        # Skip if missing ground truth
        if pd.isna(pse_true) or pd.isna(jnd_true):
            logger.warning(f"Row {idx}: missing pse_true or jnd_true, skipping")
            continue
        
        # Extract PSE values at each block (in order: 40, 60, 80, ..., 200)
        pse_values = []
        for block in BLOCKS:
            col = f'pse_{block}'
            if col in df_wide.columns:
                val = row[col]
                pse_values.append(float(val) if not pd.isna(val) else np.nan)
        
        # Extract JND values at each block
        jnd_values = []
        for block in BLOCKS:
            col = f'jnd_{block}'
            if col in df_wide.columns:
                val = row[col]
                jnd_values.append(float(val) if not pd.isna(val) else np.nan)
        
        # Recalculate 6 metrics
        df_wide.loc[idx, 'pse_stability_block'] = calculate_stability_point(pse_values, pse_true)
        df_wide.loc[idx, 'jnd_stability_block'] = calculate_stability_point(jnd_values, jnd_true)
        df_wide.loc[idx, 'pse_auc'] = calculate_auc_metric(pse_values, pse_true)
        df_wide.loc[idx, 'jnd_auc'] = calculate_auc_metric(jnd_values, jnd_true)
        
        # Final values (at trial 200, last value in list)
        final_pse = pse_values[-1] if pse_values and not np.isnan(pse_values[-1]) else np.nan
        final_jnd = jnd_values[-1] if jnd_values and not np.isnan(jnd_values[-1]) else np.nan
        
        df_wide.loc[idx, 'pse_error'] = final_pse - pse_true if not np.isnan(final_pse) else np.nan
        df_wide.loc[idx, 'jnd_error'] = final_jnd - jnd_true if not np.isnan(final_jnd) else np.nan
        
        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx + 1}/{len(df_wide)} rows")
    
    # Overwrite wide format Excel
    logger.info(f"Writing wide format to {EXCEL_PATH}...")
    try:
        wide_path = _validate_excel_path(str(EXCEL_PATH))
        df_wide.to_excel(wide_path, index=False, engine='openpyxl')
        logger.info("✓ Wide format Excel updated successfully")
    except ValueError as e:
        logger.error(f"Invalid Excel path: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to write wide format Excel: {e}")
        return False
    
    # Regenerate long format
    logger.info("Regenerating long format...")
    try:
        df_long = generate_long_format_from_dataframe(df_wide, data_type='synthetic')
        logger.info(f"✓ Generated long format: {len(df_long)} rows")
    except Exception as e:
        logger.error(f"Failed to generate long format: {e}")
        return False
    
    # Save long format Excel
    excel_long_path = EXCEL_PATH.parent / "synthetic_data_long.xlsx"
    logger.info(f"Writing long format to {excel_long_path}...")
    try:
        long_path = _validate_excel_path(str(excel_long_path))
        df_long.to_excel(long_path, index=False, engine='openpyxl')
        logger.info(f"✓ Long format Excel saved successfully")
        return True
    except ValueError as e:
        logger.error(f"Invalid Excel path: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to write long format Excel: {e}")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
