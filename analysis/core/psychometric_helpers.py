"""
Generic helpers for psychometric analysis.
Reusable across simulation, postprocessing, and validation contexts.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
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
        return float(-b0 / b1), float(1.35 / abs(b1))
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
                    return float(-b0 / b1), float(1.35 / abs(b1))
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
      JND = 0.675 / |b1|

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
        return float(-b0 / b1), float(0.675 / abs(b1))
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
                    return float(-b0 / b1), float(0.675 / abs(b1))
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