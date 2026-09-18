"""
Report generator module for creating Excel outputs.

Unified functions for generating Excel reports in both wide and long formats
for BOTH synthetic and real data with the same code path.

For SYNTHETIC data:
  - Wide: One row per subject with all metrics
  - Long: Multiple rows per subject (one per trial block)
  - Metadata: model, pse_true, jnd_true, subject_id, group
  - Metrics: trial_block, pse_est, jnd_est, stimulus_center, ..., posterior_sd_pse_*, ...

For REAL data:
  - Wide: One row per subject with all metrics
  - Long: Multiple rows per subject (one per trial block)
  - Metadata: subject_id, age, gender, modality, algorithm, group
  - Metrics: n_trials, trial_block, pse, jnd, SC, SS, lat_entropy, ...
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


def _validate_excel_path(output_path: str) -> Path:
    """
    Validate Excel output path for security and format.
    
    Checks:
    - Must have .xlsx extension
    - Must not contain path traversal attempts
    - Parent directory must be within project
    
    Args:
        output_path: Path to validate
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: if path fails validation
    """
    path = Path(output_path)
    
    # Check extension
    if path.suffix.lower() != '.xlsx':
        raise ValueError(f"Excel path must end with .xlsx: {output_path}")
    
    # Resolve symlinks in parent (strict=False to handle non-existent paths)
    # This avoids crash if parent directory doesn't exist yet
    parent_resolved = path.parent.resolve(strict=False)
    resolved = parent_resolved / path.name
    
    project_root = Path(__file__).parent.parent.parent.resolve()
    
    try:
        resolved.relative_to(project_root)
    except ValueError:
        raise ValueError(f"Excel path must be within project scope: {output_path}")
    
    return path




def save_wide_format_to_excel(df_wide: pd.DataFrame, output_path: str) -> bool:
    """
    Save wide format DataFrame to Excel file.

    Args:
        df_wide: DataFrame with one row per subject
        output_path: Path for output Excel file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate path for security
        path = _validate_excel_path(output_path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        df_wide.to_excel(path, index=False, engine='openpyxl')
        logger.info(f"✓ Saved wide format: {output_path} ({len(df_wide)} rows)")
        return True
    except ValueError as e:
        logger.error(f"✗ Invalid path for wide format: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Error saving wide format to {output_path}: {e}")
        return False


def generate_long_format_from_dataframe(df_wide: pd.DataFrame, data_type: str = 'synthetic') -> pd.DataFrame:
    """
    Convert wide format DataFrame to long format (one row per subject per trial_block).

    Wide format: One row per subject with columns like pse_40, pse_60, ..., pse_200, jnd_40, etc.
    Long format: One row per subject per trial_block with metric columns (pse, jnd, SC, SS, etc.)

    Args:
        df_wide: DataFrame in wide format
        data_type: 'synthetic' or 'real' (determines metadata columns)

    Returns:
        DataFrame in long format with one row per subject per trial_block
    """
    if df_wide.empty:
        return pd.DataFrame()

    # Identify trial block columns (pse_40, pse_60, jnd_40, etc.)
    trial_block_columns = {}  # trial_block -> list of columns for that block

    for col in df_wide.columns:
        # Columns like 'pse_40', 'jnd_60', 'SC_40', 'lat_entropy_80', etc.
        if '_' in col:
            parts = col.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                trial_block = int(parts[1])
                if trial_block not in trial_block_columns:
                    trial_block_columns[trial_block] = []
                trial_block_columns[trial_block].append(col)

    if not trial_block_columns:
        logger.warning("No trial block columns found in wide format")
        return pd.DataFrame()

    # Identify metadata columns (everything that's not metric-related)
    metric_cols = set()
    for cols in trial_block_columns.values():
        metric_cols.update(cols)

    metadata_cols = [col for col in df_wide.columns if col not in metric_cols]

    # Build long format
    long_rows = []

    for _, wide_row in df_wide.iterrows():
        # Get metadata for this subject
        metadata = {col: wide_row[col] for col in metadata_cols}

        # Create one row per trial block
        for trial_block in sorted(trial_block_columns.keys()):
            long_row = dict(metadata)
            long_row['trial_block'] = trial_block

            # Add metric values for this trial block
            for col in trial_block_columns[trial_block]:
                # Strip trial_block suffix to get metric name
                metric_name = col.rsplit('_', 1)[0]
                long_row[metric_name] = wide_row[col]

            long_rows.append(long_row)

    df_long = pd.DataFrame(long_rows)

    # Reorder columns: metadata first, then trial_block, then metrics
    col_order = metadata_cols + ['trial_block'] + sorted([col for col in df_long.columns if col not in metadata_cols + ['trial_block']])
    df_long = df_long[[col for col in col_order if col in df_long.columns]]

    return df_long


def save_long_format_to_excel(df_long: pd.DataFrame, output_path: str) -> bool:
    """
    Save long format DataFrame to Excel file.

    Args:
        df_long: DataFrame in long format
        output_path: Path for output Excel file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate path for security
        path = _validate_excel_path(output_path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        df_long.to_excel(path, index=False, engine='openpyxl')
        logger.info(f"✓ Saved long format: {output_path} ({len(df_long)} rows)")
        return True
    except ValueError as e:
        logger.error(f"✗ Invalid path for long format: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Error saving long format to {output_path}: {e}")
        return False
