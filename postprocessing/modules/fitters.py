"""
Fitters module for psychometric curve fitting.

fit_logistic and fit_probit delegate to utilities.psychometric_helpers
(single source of truth for GLM fitting logic).
fit_gaussfit remains here as it is specific to the postprocessing pipeline.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import warnings

from utilities.psychometric_helpers import (
    fit_logistic_psychometric,
    fit_probit_psychometric,
    fit_gaussfit_psychometric,
)


@dataclass
class FitResult:
    """Result of psychometric curve fitting.
    
    Attributes:
        pse: Point of Subjective Equality (50% response point)
        jnd: Just Noticeable Difference (discrimination threshold)
        method: Fitting method used ('logistic', 'probit', 'gaussfit')
        success: Whether fitting succeeded
        error_message: Error message if fitting failed
    """
    pse: float
    jnd: float
    method: str
    success: bool
    error_message: str


def _expand_binned_data(latencies: np.ndarray, counts: np.ndarray, 
                        responses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand binned data to individual trials.
    
    Args:
        latencies: Stimulus values
        counts: Trial counts per stimulus
        responses: Response counts per stimulus (number of "1" responses)
        
    Returns:
        Tuple of (expanded_latencies, expanded_responses)
    """
    expanded_lat = []
    expanded_resp = []
    
    for lat, count, resp_count in zip(latencies, counts, responses):
        # Add 'resp_count' trials with response=1
        for _ in range(int(resp_count)):
            expanded_lat.append(lat)
            expanded_resp.append(1)
        # Add 'count - resp_count' trials with response=0
        for _ in range(int(count - resp_count)):
            expanded_lat.append(lat)
            expanded_resp.append(0)
    
    return np.array(expanded_lat), np.array(expanded_resp)


def _validate_data(latencies: np.ndarray, responses: np.ndarray, 
                   min_trials: int = 5) -> Tuple[bool, str]:
    """
    Validate data for fitting.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(latencies) < min_trials:
        return False, f"Insufficient trials: {len(latencies)} < {min_trials}"
    
    if len(np.unique(responses)) < 2:
        return False, "All responses are identical"
    
    if len(np.unique(latencies)) < 3:
        return False, "Too few unique stimulus levels"
    
    return True, ""


def fit_logistic(latencies: np.ndarray, counts: np.ndarray,
                 responses: np.ndarray) -> FitResult:
    """
    Fit logistic psychometric curve on binned data.
    Expands binned data to individual trials, then delegates to
    fit_logistic_psychometric() in utilities.psychometric_helpers.
    """
    exp_lat, exp_resp = _expand_binned_data(latencies, counts, responses)
    is_valid, error_msg = _validate_data(exp_lat, exp_resp, min_trials=5)
    if not is_valid:
        return FitResult(pse=np.nan, jnd=np.nan, method='logistic',
                         success=False, error_message=error_msg)
    try:
        pse, jnd = fit_logistic_psychometric(exp_lat, exp_resp, fallback=False)
        return FitResult(pse=pse, jnd=jnd, method='logistic', success=True, error_message="")
    except Exception as e:
        return FitResult(pse=np.nan, jnd=np.nan, method='logistic',
                         success=False, error_message=str(e))


def fit_probit(latencies: np.ndarray, counts: np.ndarray,
               responses: np.ndarray) -> FitResult:
    """
    Fit probit psychometric curve on binned data.
    Expands binned data to individual trials, then delegates to
    fit_probit_psychometric() in utilities.psychometric_helpers.
    """
    exp_lat, exp_resp = _expand_binned_data(latencies, counts, responses)
    is_valid, error_msg = _validate_data(exp_lat, exp_resp, min_trials=5)
    if not is_valid:
        return FitResult(pse=np.nan, jnd=np.nan, method='probit',
                         success=False, error_message=error_msg)
    try:
        pse, jnd = fit_probit_psychometric(exp_lat, exp_resp, fallback=False)
        return FitResult(pse=pse, jnd=jnd, method='probit', success=True, error_message="")
    except Exception as e:
        return FitResult(pse=np.nan, jnd=np.nan, method='probit',
                         success=False, error_message=str(e))


def fit_gaussfit(latencies: np.ndarray, counts: np.ndarray,
                 responses: np.ndarray, guess_rate: float,
                 bin_size: float = 10.0) -> FitResult:
    """
    Fit gaussfit psychometric curve on binned data.
    Bins, then unbins to individual trials, then delegates to
    fit_gaussfit_psychometric() in utilities.psychometric_helpers.
    """
    try:
        # Sort and bin
        sort_idx = np.argsort(latencies)
        s_lat = latencies[sort_idx]
        s_cnt = counts[sort_idx]
        s_resp = responses[sort_idx]

        lat_min, lat_max = np.min(s_lat), np.max(s_lat)
        n_bins = max(1, int(np.ceil((lat_max - lat_min) / bin_size)))

        bin_lats, bin_cnts, bin_resps = [], [], []
        for i in range(n_bins):
            b_start = lat_min + i * bin_size
            b_end = b_start + bin_size
            mask = (s_lat >= b_start) & (s_lat <= b_end if i == n_bins - 1 else s_lat < b_end)
            if np.any(mask):
                bin_lats.append(b_start + bin_size / 2)
                bin_cnts.append(np.sum(s_cnt[mask]))
                bin_resps.append(np.sum(s_resp[mask]))

        if not bin_lats:
            return FitResult(pse=np.nan, jnd=np.nan, method='gaussfit',
                             success=False, error_message="No data after binning")

        # Unbin to individual trials
        exp_lat, exp_resp = _expand_binned_data(
            np.array(bin_lats), np.array(bin_cnts), np.array(bin_resps)
        )

        if len(exp_lat) < 3 or len(np.unique(exp_lat)) < 3:
            return FitResult(pse=np.nan, jnd=np.nan, method='gaussfit',
                             success=False, error_message="Insufficient data after unbinning")

        pse, jnd = fit_gaussfit_psychometric(exp_lat, exp_resp, guess_rate)
        if np.isfinite(pse) and np.isfinite(jnd):
            return FitResult(pse=pse, jnd=abs(jnd), method='gaussfit',
                             success=True, error_message="")
        return FitResult(pse=np.nan, jnd=np.nan, method='gaussfit',
                         success=False, error_message="GaussFitm returned invalid parameters")

    except Exception as e:
        return FitResult(pse=np.nan, jnd=np.nan, method='gaussfit',
                         success=False, error_message=str(e))
