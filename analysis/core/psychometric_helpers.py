"""
Generic helpers for psychometric analysis.
Reusable across simulation, postprocessing, and validation contexts.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class ProgressiveAnalysisData:
    """Progressive psychometric analysis results for one subject."""
    subject_id: str
    blocks: List[int]
    pse_values: List[float]
    jnd_values: List[float]

    def get_final_pse(self) -> float:
        return self.pse_values[-1]

    def get_final_jnd(self) -> float:
        return self.jnd_values[-1]


@dataclass
class StabilityPoint:
    """Stability analysis results for one subject."""
    subject_id: str
    pse_stability_point: Optional[int]
    jnd_stability_point: Optional[int]
    pse_final: float
    jnd_final: float


@dataclass
class ValidationResult:
    """Validation comparison between original and simulated data for one subject."""
    subject_id: str
    pse_original: float
    pse_simulated: float
    pse_diff_pct: float
    jnd_original: float
    jnd_simulated: float
    jnd_diff_pct: float
    blocks: List[int] = field(default_factory=list)
    pse_original_evolution: List[float]  = field(default_factory=list)
    pse_simulated_evolution: List[float] = field(default_factory=list)
    jnd_original_evolution: List[float]  = field(default_factory=list)
    jnd_simulated_evolution: List[float] = field(default_factory=list)

    def is_pse_similar(self, threshold: float = 10.0) -> bool:
        return abs(self.pse_diff_pct) <= threshold

    def is_jnd_similar(self, threshold: float = 10.0) -> bool:
        return abs(self.jnd_diff_pct) <= threshold

# ============================================================================
# FUNCTIONS
# ============================================================================

def calculate_stability_from_values(
    values: list,
    threshold: float = 0.10,
    blocks: List[int] = None
) -> int:
    """
    Find the first block where a parameter is within threshold% of its final value.

    Args:
        values: Parameter values at each block [40, 60, 80, ...]
        threshold: Fractional threshold, e.g. 0.10 = 10%
        blocks: Block sizes corresponding to values (default: [40,60,...,200])

    Returns:
        Block number where parameter first stabilizes, or 200 if never stable.
    """
    if not values:
        return 200

    if blocks is None:
        blocks = [40, 60, 80, 100, 120, 140, 160, 180, 200][:len(values)]

    final_value = values[-1]

    if abs(final_value) < 1e-10:
        return 200

    for i, block in enumerate(blocks):
        diff_pct = abs(values[i] - final_value) / abs(final_value) * 100
        if diff_pct < threshold * 100:
            return block

    return 200

# ============================================================================
# region FITTERS
# ============================================================================

def fit_logistic_psychometric(
    latencies: np.ndarray,
    responses: np.ndarray,
    fallback: bool = True
) -> Tuple[float, float]:
    """
    Fit a logistic psychometric function via GLM (binomial/logit) and return (PSE, JND).

    Uses statsmodels GLM with binomial family and logit link:
      logit(P) = b0 + b1 * x
      PSE = -b0 / b1
      JND = 1.35 / |b1|

    Args:
        latencies: Stimulus values
        responses: Binary responses (0/1)
        fallback: If True, return (median, std) on failure instead of raising

    Returns:
        Tuple of (PSE, JND)
    """
    import warnings
    import statsmodels.api as sm
    from scipy.optimize import minimize

    try:
        X = sm.add_constant(latencies)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.GLM(responses, X, family=sm.families.Binomial())
            result = model.fit()
        b0, b1 = result.params
        if abs(b1) < 1e-10:
            raise ValueError("Slope too close to zero")
        return float(-b0 / b1), float(np.log(3) / abs(b1))
    except Exception:
        # Fallback: MLE via Nelder-Mead
        try:
            def neg_log_likelihood(params):
                b0, b1 = params
                p = np.clip(1.0 / (1.0 + np.exp(-(b0 + b1 * latencies))), 1e-10, 1 - 1e-10)
                return -np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p))
            res = minimize(neg_log_likelihood, [0.0, 0.01], method='Nelder-Mead')
            if res.success:
                b0, b1 = res.x
                if abs(b1) > 1e-10:
                    return float(-b0 / b1), float(np.log(3) / abs(b1))
        except Exception:
            pass
        if fallback:
            return float(np.median(latencies)), float(np.std(latencies))
        raise


def fit_probit_psychometric(
    latencies: np.ndarray,
    responses: np.ndarray,
    fallback: bool = True
) -> Tuple[float, float]:
    """
    Fit a probit psychometric function via GLM (binomial/probit) and return (PSE, JND).

    Uses statsmodels GLM with binomial family and probit link:
      probit(P) = b0 + b1 * x
      PSE = -b0 / b1
      JND = 0.6745 / |b1|

    Args:
        latencies: Stimulus values
        responses: Binary responses (0/1)
        fallback: If True, return (median, std) on failure

    Returns:
        Tuple of (PSE, JND)
    """
    import warnings
    import statsmodels.api as sm
    from scipy import stats as scipy_stats

    try:
        X = sm.add_constant(latencies)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.GLM(responses, X,
                           family=sm.families.Binomial(
                               link=sm.families.links.Probit()))
            result = model.fit()
        b0, b1 = result.params
        if abs(b1) < 1e-10:
            raise ValueError("Slope too close to zero")
        return float(-b0 / b1), float(0.6745 / abs(b1))
    except Exception:
        try:
            def neg_log_likelihood(params):
                b0, b1 = params
                p = np.clip(scipy_stats.norm.cdf(b0 + b1 * latencies), 1e-10, 1 - 1e-10)
                return -np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p))
            from scipy.optimize import minimize
            res = minimize(neg_log_likelihood, [0.0, 0.01], method='Nelder-Mead')
            if res.success:
                b0, b1 = res.x
                if abs(b1) > 1e-10:
                    return float(-b0 / b1), float(0.6745 / abs(b1))
        except Exception:
            pass
        if fallback:
            return float(np.median(latencies)), float(np.std(latencies))
        raise



def fit_gaussfit_psychometric(
    latencies: np.ndarray,
    responses: np.ndarray,
    guess_rate: float
) -> Tuple[float, float]:
    """
    Fit psychometric curve via grid search over cumulative Gaussian parameter space.

    p(response=1) = guess_rate + (1 - guess_rate) * Φ((x - pse) / jnd)

    Args:
        latencies: Individual trial stimulus values
        responses: Individual trial responses (0/1)
        guess_rate: Guess rate parameter

    Returns:
        Tuple of (PSE, JND), or (nan, nan) if fitting fails
    """
    from scipy import stats as scipy_stats

    if len(latencies) < 3:
        return np.nan, np.nan
    if len(np.unique(responses)) < 2:
        return np.nan, np.nan
    if len(np.unique(latencies)) < 3:
        return np.nan, np.nan

    lat_min, lat_max = np.min(latencies), np.max(latencies)
    lat_range = lat_max - lat_min
    if lat_range < 1e-10:
        return np.nan, np.nan

    pse_grid = np.linspace(lat_min, lat_max, 100)
    jnd_grid = np.linspace(1.0, max(lat_range / 2, 1.0), 100)

    best_pse, best_jnd = np.nan, np.nan
    best_ll = -np.inf

    for pse in pse_grid:
        for jnd in jnd_grid:
            p = np.clip(
                guess_rate + (1 - guess_rate) * scipy_stats.norm.cdf((latencies - pse) / jnd),
                1e-10, 1 - 1e-10
            )
            ll = np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p))
            if ll > best_ll:
                best_ll, best_pse, best_jnd = ll, pse, jnd

    if np.isfinite(best_ll):
        return float(best_pse), float(best_jnd)
    return np.nan, np.nan

# endregion


# ============================================================================
# LATENCY STATISTICS
# ============================================================================

def calculate_latency_statistics(latencies: np.ndarray) -> Dict[str, float]:
    """
    Calculate descriptive statistics for stimulus latencies.

    Renamed to match simulation metrics:
    - stimulus_center (SC): mean of stimulus latencies
    - stimulus_spread (SS): standard deviation of stimulus latencies
    - lat_entropy: Shannon entropy of stimulus latency distribution
    - lat_range: range of stimulus latencies

    Args:
        latencies: Array of stimulus presentation times (milliseconds)

    Returns:
        Dictionary with keys: stimulus_center, stimulus_spread, lat_range, lat_entropy (Shannon, 10 bins)
    """
    if len(latencies) == 0:
        return {'stimulus_center': np.nan, 'stimulus_spread': np.nan, 'lat_range': np.nan, 'lat_entropy': np.nan}

    stimulus_center = float(np.mean(latencies))
    stimulus_spread = float(np.std(latencies, ddof=0))
    lat_range = float(np.max(latencies) - np.min(latencies))

    if len(latencies) > 1 and lat_range > 0:
        counts, _ = np.histogram(latencies, bins=10)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        lat_entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        lat_entropy = 0.0

    return {'stimulus_center': stimulus_center, 'stimulus_spread': stimulus_spread, 'lat_range': lat_range, 'lat_entropy': lat_entropy}


# ============================================================================
# ASYMMETRY AND STIMULUS METRICS
# ============================================================================

def calculate_asymmetry_metrics(rows: List[Dict], offset: float = 500) -> Dict[str, Any]:
    """
    Calculate asymmetry metrics from trial rows.

    Computes how many stimuli were presented before vs after the offset,
    and derives an asymmetry index.

    Args:
        rows: List of trial dicts with 'lat' key (stimulus latency in ms)
        offset: Threshold latency (default 500ms)

    Returns:
        Dictionary with keys:
        - 'asymmetry_index': (n_after - n_before) / total, range [-1, 1]
        - 'n_before_offset': Number of stimuli < offset
        - 'n_after_offset': Number of stimuli >= offset
        - 'pct_before': Percentage of stimuli < offset
        - 'pct_after': Percentage of stimuli >= offset

    Notes:
        - asymmetry_index = 0: perfectly balanced (50-50)
        - asymmetry_index > 0: bias toward after (more stimuli after offset)
        - asymmetry_index < 0: bias toward before (more stimuli before offset)
        - asymmetry_index = ±1: all stimuli on one side
    """
    if not rows:
        return {
            'asymmetry_index': 0.0,
            'n_before_offset': 0,
            'n_after_offset': 0,
            'pct_before': 0.0,
            'pct_after': 0.0,
        }

    latencies = np.array([row['lat'] for row in rows])

    n_before = np.sum(latencies < offset)
    n_after = np.sum(latencies > offset)
    total = n_before + n_after  # Exclude trials exactly at offset

    if total == 0:
        return {
            'asymmetry_index': 0.0,
            'n_before_offset': 0,
            'n_after_offset': 0,
            'pct_before': 0.0,
            'pct_after': 0.0,
        }

    asymmetry_index = (n_after - n_before) / total

    return {
        'asymmetry_index': float(asymmetry_index),
        'n_before_offset': int(n_before),
        'n_after_offset': int(n_after),
        'pct_before': float(100 * n_before / total),
        'pct_after': float(100 * n_after / total),
    }


def calculate_progressive_asymmetry(rows: List[Dict], offset: float = 500, trial_counts: List[int]|None = None) -> Dict[int, float]:
    """
    Calculate asymmetry index at progressive trial counts.

    Args:
        rows: List of trial dicts with 'lat' key
        offset: Threshold latency (default 500ms)
        trial_counts: List of trial counts to calculate at (default: [40, 60, 80, 100, 120, 140, 160, 180, 200])

    Returns:
        Dictionary mapping trial_count to asymmetry_index
    """
    if trial_counts is None:
        trial_counts = [40, 60, 80, 100, 120, 140, 160, 180, 200]

    if not rows:
        return {n: 0.0 for n in trial_counts}

    result = {}

    for n_trials in trial_counts:
        if n_trials > len(rows):
            continue

        # Get first n_trials
        subset_rows = rows[:n_trials]
        asymmetry_data = calculate_asymmetry_metrics(subset_rows, offset)
        result[n_trials] = asymmetry_data['asymmetry_index']

    return result


def calculate_progressive_stimulus_metrics(rows: List[Dict], trial_counts: List[int]|None = None) -> Dict[str, Dict[int, float]]:
    """
    Calculate stimulus distribution metrics at progressive trial counts.

    Computes cumulative statistics of stimulus latencies to validate that
    the presented stimuli follow the intended PSE/JND parameters.

    Args:
        rows: List of trial dicts with 'lat' key (stimulus latency in ms)
        trial_counts: List of trial counts to calculate at (default: [40, 60, 80, 100, 120, 140, 160, 180, 200])

    Returns:
        Dictionary with keys:
        - 'stimulus_center': Dict mapping trial_count to mean latency
        - 'stimulus_spread': Dict mapping trial_count to std of latencies
        - 'stimulus_min': Dict mapping trial_count to min latency
        - 'stimulus_max': Dict mapping trial_count to max latency
        - 'bimodality_index': Dict mapping trial_count to bimodality measure

    Notes:
        - stimulus_center should correlate with PSE parameter
        - stimulus_spread should correlate with JND parameter
        - bimodality_index measures how much the distribution has two peaks
          (higher values = more bimodal, lower values = more unimodal)
    """
    if trial_counts is None:
        trial_counts = [40, 60, 80, 100, 120, 140, 160, 180, 200]

    if not rows:
        return {
            'stimulus_center': {n: 0.0 for n in trial_counts},
            'stimulus_spread': {n: 0.0 for n in trial_counts},
            'stimulus_min': {n: 0.0 for n in trial_counts},
            'stimulus_max': {n: 0.0 for n in trial_counts},
            'bimodality_index': {n: 0.0 for n in trial_counts},
        }

    result = {
        'stimulus_center': {},
        'stimulus_spread': {},
        'stimulus_min': {},
        'stimulus_max': {},
        'bimodality_index': {},
    }

    for n_trials in trial_counts:
        if n_trials > len(rows):
            continue

        # Get first n_trials
        subset_rows = rows[:n_trials]
        latencies = np.array([row['lat'] for row in subset_rows])

        # Basic statistics
        result['stimulus_center'][n_trials] = float(np.mean(latencies))
        result['stimulus_spread'][n_trials] = float(np.std(latencies))
        result['stimulus_min'][n_trials] = float(np.min(latencies))
        result['stimulus_max'][n_trials] = float(np.max(latencies))

        # Bimodality index: simple measure based on histogram peaks
        # Create histogram with 20 bins
        hist, bin_edges = np.histogram(latencies, bins=20)

        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append((i, hist[i]))

        # Bimodality measure: if we have 2+ peaks, compute ratio of top 2
        if len(peaks) >= 2:
            peaks.sort(key=lambda x: x[1], reverse=True)
            peak1, peak2 = peaks[0][1], peaks[1][1]
            # Ratio of second peak to first peak (0 to 1)
            bimodality = float(peak2 / peak1) if peak1 > 0 else 0.0
        else:
            bimodality = 0.0

        result['bimodality_index'][n_trials] = bimodality

    return result


# ============================================================================
# EXCEL CONSOLIDATION
# ============================================================================

import os
import pandas as pd
from pathlib import Path


def consolidate_results(
    results_list: list,
    output_dir: str,
    file_prefix: str
) -> str:
    """
    Consolidate analysis results from all subjects into an Excel report.

    Creates a consolidated Excel file containing results for all subjects with
    one row per subject and columns for all key metrics.

    Args:
        results_list: List of result dictionaries
        output_dir: Directory path where Excel file will be saved (str)
        file_prefix: Prefix for the Excel filename (str)

    Returns:
        Path to the created Excel file (str)

    Raises:
        TypeError: If results_list is not a list
        ValueError: If results_list is empty
        ValueError: If output_dir does not exist or is not a directory
        ValueError: If file_prefix is empty or whitespace-only
    """
    if not isinstance(results_list, list):
        raise TypeError(f"results_list must be a list, got {type(results_list)}")

    for i, item in enumerate(results_list):
        if not isinstance(item, dict):
            raise TypeError(f"results_list[{i}] must be a dict, got {type(item)}")

    if not isinstance(output_dir, str):
        raise ValueError(f"output_dir must be str, got {type(output_dir)}")

    output_path = Path(output_dir)
    if not output_path.exists():
        raise ValueError(f"output_dir does not exist: {output_dir}")

    if not output_path.is_dir():
        raise ValueError(f"output_dir is not a directory: {output_dir}")

    if not isinstance(file_prefix, str):
        raise ValueError(f"file_prefix must be str, got {type(file_prefix)}")

    if not file_prefix or file_prefix.strip() == "":
        raise ValueError("file_prefix cannot be empty or whitespace-only")

    if len(results_list) == 0:
        required_columns = ['subj', 'mu', 'sigma', 'jnd', 'pse', 'n_trials', 'accuracy',
                           'asymmetry_index', 'n_before_offset', 'n_after_offset',
                           'pct_before', 'pct_after']
        df = pd.DataFrame(columns=required_columns)
    else:
        df = pd.DataFrame(results_list)

        required_columns = {'subj', 'mu', 'sigma', 'jnd', 'pse', 'n_trials', 'accuracy'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Results missing required columns: {missing_columns}")

        # Drop unwanted columns
        for col in ['subject_id', 'status']:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Reorder columns: put core metrics first, then lat_entropy columns
        core_cols = ['subj', 'mu', 'sigma', 'jnd', 'pse', 'n_trials', 'accuracy',
                     'asymmetry_index', 'n_before_offset', 'n_after_offset',
                     'pct_before', 'pct_after']

        # Get all lat_entropy columns (lat_entropy_40, lat_entropy_60, etc.)
        lat_entropy_cols = sorted([col for col in df.columns if col.startswith('lat_entropy_')])

        # Get all other columns (progressive metrics, stimulus metrics, etc.)
        other_cols = [col for col in df.columns if col not in core_cols and col not in lat_entropy_cols]

        # Reorder: core + lat_entropy + others
        ordered_cols = [col for col in core_cols if col in df.columns] + lat_entropy_cols + other_cols
        df = df[ordered_cols]

    excel_filename = f"{file_prefix}_results_summary.xlsx"
    excel_filepath = os.path.join(output_dir, excel_filename)

    df.to_excel(excel_filepath, index=False, sheet_name='Results')

    if not os.path.exists(excel_filepath):
        raise FileNotFoundError(f"Excel file was not created at: {excel_filepath}")

    return excel_filepath


def add_group_stats_to_excel(excel_filepath: str, skip_cols: List[str]|None = None) -> None:
    """
    Append GROUP_mean and GROUP_std rows to an existing results Excel file.

    Args:
        excel_filepath: Full path to the Excel file produced by consolidate_results()
        skip_cols: Column names to exclude from statistics (default: subj, status)
    """
    if skip_cols is None:
        skip_cols = ['subj', 'status']

    df = pd.read_excel(excel_filepath)

    group_mean = {'subj': 'GROUP_mean'}
    group_std  = {'subj': 'GROUP_std'}

    for col in df.columns:
        if col in skip_cols:
            continue
        try:
            group_mean[col] = df[col].mean()
            group_std[col]  = df[col].std()
        except Exception:
            group_mean[col] = None
            group_std[col]  = None

    df_combined = pd.concat([df, pd.DataFrame([group_mean, group_std])], ignore_index=True)
    df_combined.to_excel(excel_filepath, index=False, sheet_name='Results')
