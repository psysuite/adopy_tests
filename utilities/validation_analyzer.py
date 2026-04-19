"""
ValidationAnalyzer - single-folder progressive psychometric analysis.

Reads .txt data files from one directory, fits logistic psychometric functions
progressively at standard block sizes, and computes stability points.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utilities.psychometric_helpers import calculate_stability_from_values, fit_logistic_psychometric


class ValidationAnalyzer:
    """
    Progressive psychometric analysis on a single folder of data files.

    Each file is expected to be tab-separated with at least columns:
      - lat       : stimulus latency (ms)
      - user_ans  : binary response (0/1)
    """

    DEFAULT_BLOCKS = [40, 60, 80, 100, 120, 140, 160, 180, 200]

    def __init__(
        self,
        data_dir: str,
        logger: Optional[logging.Logger] = None,
        file_pattern: str = "*.txt",
        blocks: Optional[List[int]] = None,
    ):
        """
        Args:
            data_dir: Directory containing data files
            logger: Optional logger (creates a silent one if not provided)
            file_pattern: Glob pattern to select files (default: *.txt)
            blocks: Block sizes for progressive analysis (default: [40,60,...,200])
        """
        self.data_dir = Path(data_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.file_pattern = file_pattern
        self.blocks = blocks or self.DEFAULT_BLOCKS

    def run_progressive_analysis(
        self,
        filepath: Path,
        blocks: Optional[List[int]] = None,
    ) -> Tuple[Dict[str, List[float]], List[int]]:
        """
        Fit logistic psychometric function progressively on a single file.

        Args:
            filepath: Path to tab-separated data file
            blocks: Override block sizes for this call

        Returns:
            Tuple of ({'PSE': [...], 'JND': [...]}, blocks_used)
        """
        blocks = blocks or self.blocks

        try:
            df = pd.read_csv(filepath, sep='\t')
        except Exception as e:
            self.logger.error(f"Failed to load {filepath.name}: {e}")
            return {'PSE': [], 'JND': []}, []

        pse_values, jnd_values, blocks_used = [], [], []

        for block_size in blocks:
            if block_size > len(df):
                self.logger.warning(f"{filepath.name}: block {block_size} > data length {len(df)}, stopping")
                break

            df_block = df.iloc[:block_size]
            pse, jnd = fit_logistic_psychometric(df_block['lat'].values, df_block['user_ans'].values)
            pse_values.append(pse)
            jnd_values.append(jnd)
            blocks_used.append(block_size)

        return {'PSE': pse_values, 'JND': jnd_values}, blocks_used

    def analyze_folder(
        self,
        stability_threshold: float = 0.10,
    ) -> pd.DataFrame:
        """
        Run progressive analysis on all files in data_dir.

        Returns:
            DataFrame with one row per file:
            filename, pse_final, jnd_final, pse_sp, jnd_sp,
            pse_1..pse_9, jnd_1..jnd_9
        """
        files = sorted(self.data_dir.glob(self.file_pattern))

        if not files:
            self.logger.warning(f"No files found in {self.data_dir} with pattern '{self.file_pattern}'")
            return pd.DataFrame()

        rows = []
        for filepath in files:
            params, blocks_used = self.run_progressive_analysis(filepath)

            if not params['PSE']:
                self.logger.warning(f"No results for {filepath.name}")
                continue

            row = {'filename': filepath.name}

            for i, (pse, jnd) in enumerate(zip(params['PSE'], params['JND'])):
                row[f'pse_{blocks_used[i]}'] = round(pse, 1)
                row[f'jnd_{blocks_used[i]}'] = round(jnd, 1)

            row['pse_final'] = round(params['PSE'][-1], 1)
            row['jnd_final'] = round(params['JND'][-1], 1)
            row['pse_sp'] = calculate_stability_from_values(params['PSE'], stability_threshold, blocks_used)
            row['jnd_sp'] = calculate_stability_from_values(params['JND'], stability_threshold, blocks_used)

            rows.append(row)

        return pd.DataFrame(rows)
