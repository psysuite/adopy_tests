"""
Progressive analyzer - unified progressive psychometric analysis.

Centralizes progressive analysis logic for both direct and binned data formats.
Single source of truth for progressive fitting across the codebase.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from analysis.core.data_loader import DataLoader
from analysis.core.psychometric_helpers import (
    fit_logistic_psychometric,
    fit_probit_psychometric,
    calculate_stability_from_values,
    calculate_latency_statistics,
)


logger = logging.getLogger(__name__)


@dataclass
class ProgressiveResult:
    """Results from progressive analysis for one subject.
    
    Attributes:
        trial_counts: List of trial counts analyzed [40, 60, 80, 100, 120, 140, 160, 180, 200]
        pse_values: PSE at each trial count (dict keyed by trial count)
        jnd_values: JND at each trial count (dict keyed by trial count)
        lat_mean: Mean latency at each count (dict keyed by trial count)
        lat_std: Std latency at each count (dict keyed by trial count)
        lat_range: Range latency at each count (dict keyed by trial count)
        lat_entropy: Entropy at each count (dict keyed by trial count)
        method: Fitting method used ('logistic', 'probit', 'gaussfit')
    """
    trial_counts: List[int] = field(default_factory=lambda: [40, 60, 80, 100, 120, 140, 160, 180, 200])
    pse_values: Dict[int, float] = field(default_factory=dict)
    jnd_values: Dict[int, float] = field(default_factory=dict)
    lat_mean: Dict[int, float] = field(default_factory=dict)
    lat_std: Dict[int, float] = field(default_factory=dict)
    lat_range: Dict[int, float] = field(default_factory=dict)
    lat_entropy: Dict[int, float] = field(default_factory=dict)
    method: str = ""


class ProgressiveAnalyzer:
    """
    Unified progressive psychometric analysis.

    Supports both direct format (lat, user_ans) and binned format (lat, count, resp).
    """

    DEFAULT_BLOCKS = [40, 60, 80, 100, 120, 140, 160, 180, 200]

    def __init__(
        self,
        blocks: Optional[List[int]] = None,
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Args:
            blocks: Block sizes for progressive analysis
            logger_instance: Optional logger instance
        """
        self.blocks = blocks or self.DEFAULT_BLOCKS
        self.logger = logger_instance or logging.getLogger(__name__)

    def run_progressive_analysis(
        self,
        filepath: str,
        method: str = 'logistic',
        blocks: Optional[List[int]] = None,
    ) -> ProgressiveResult:
        """
        Fit psychometric function progressively on a single file.

        Automatically detects data format (direct or binned) and expands as needed.
        Calculates latency statistics at each block size.

        Args:
            filepath: Path to data file
            method: Fitting method ('logistic' or 'probit')
            blocks: Override block sizes for this call

        Returns:
            ProgressiveResult containing PSE, JND, and latency statistics at each block size
        """
        blocks = blocks or self.blocks

        try:
            latencies, responses = DataLoader.load_and_expand(filepath)
        except Exception as e:
            self.logger.error(f"Failed to load {Path(filepath).name}: {e}")
            result = ProgressiveResult(method=method)
            for N in result.trial_counts:
                result.pse_values[N] = np.nan
                result.jnd_values[N] = np.nan
                result.lat_mean[N] = np.nan
                result.lat_std[N] = np.nan
                result.lat_range[N] = np.nan
                result.lat_entropy[N] = np.nan
            return result

        result = ProgressiveResult(method=method)
        
        # Initialize all values to NaN
        for N in result.trial_counts:
            result.pse_values[N] = np.nan
            result.jnd_values[N] = np.nan
            result.lat_mean[N] = np.nan
            result.lat_std[N] = np.nan
            result.lat_range[N] = np.nan
            result.lat_entropy[N] = np.nan

        total_trials = len(latencies)

        for block_size in blocks:
            if block_size > total_trials:
                self.logger.warning(f"{Path(filepath).name}: block {block_size} > data length {total_trials}, stopping")
                break

            # Extract first N trials in original order
            lat_block = latencies[:block_size]
            resp_block = responses[:block_size]

            # Calculate latency statistics
            stats = calculate_latency_statistics(lat_block)
            result.lat_mean[block_size] = stats['mean']
            result.lat_std[block_size] = stats['std']
            result.lat_range[block_size] = stats['range']
            result.lat_entropy[block_size] = stats['entropy']

            # Fit psychometric curve
            try:
                if method == 'logistic':
                    pse, jnd = fit_logistic_psychometric(lat_block, resp_block, fallback=True)
                elif method == 'probit':
                    pse, jnd = fit_probit_psychometric(lat_block, resp_block, fallback=True)
                else:
                    self.logger.error(f"Unknown method: {method}")
                    continue

                if np.isfinite(pse) and np.isfinite(jnd):
                    result.pse_values[block_size] = pse
                    result.jnd_values[block_size] = jnd
                else:
                    self.logger.warning(f"{Path(filepath).name}: N={block_size} returned NaN")
                    break

            except Exception as e:
                self.logger.error(f"{Path(filepath).name}: N={block_size} fitting failed - {e}")
                break

        return result

    def analyze_folder(
        self,
        data_dir: str,
        file_pattern: str = "*.txt",
        method: str = 'logistic',
        stability_threshold: float = 0.10,
    ) -> Dict[str, Dict]:
        """
        Run progressive analysis on all files in a directory.

        Args:
            data_dir: Directory containing data files
            file_pattern: Glob pattern for files
            method: Fitting method ('logistic' or 'probit')
            stability_threshold: Threshold for stability point calculation

        Returns:
            Dictionary with one entry per file containing ProgressiveResult data
        """
        data_path = Path(data_dir)
        files = sorted(data_path.glob(file_pattern))

        if not files:
            self.logger.warning(f"No files found in {data_path} with pattern '{file_pattern}'")
            return {}

        results = {}

        for filepath in files:
            result = self.run_progressive_analysis(str(filepath), method=method)

            # Get valid blocks (non-NaN PSE values)
            valid_blocks = [N for N in result.trial_counts if np.isfinite(result.pse_values[N])]
            
            if not valid_blocks:
                self.logger.warning(f"No results for {filepath.name}")
                continue

            # Extract lists for stability calculation
            pse_list = [result.pse_values[N] for N in valid_blocks]
            jnd_list = [result.jnd_values[N] for N in valid_blocks]

            pse_sp = calculate_stability_from_values(pse_list, stability_threshold, valid_blocks)
            jnd_sp = calculate_stability_from_values(jnd_list, stability_threshold, valid_blocks)

            results[filepath.name] = {
                'pse_values': result.pse_values,
                'jnd_values': result.jnd_values,
                'lat_mean': result.lat_mean,
                'lat_std': result.lat_std,
                'lat_range': result.lat_range,
                'lat_entropy': result.lat_entropy,
                'valid_blocks': valid_blocks,
                'pse_final': pse_list[-1] if pse_list else None,
                'jnd_final': jnd_list[-1] if jnd_list else None,
                'pse_sp': pse_sp,
                'jnd_sp': jnd_sp,
            }

        return results
