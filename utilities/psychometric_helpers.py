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
    pse_original_evolution: List[float] = field(default_factory=list)
    pse_simulated_evolution: List[float] = field(default_factory=list)
    jnd_original_evolution: List[float] = field(default_factory=list)
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


def compare_progressive_params(
    original: Dict[str, List[float]],
    simulated: Dict[str, List[float]],
    blocks: List[int] = None
) -> Dict:
    """
    Compare PSE and JND between original and simulated progressive results.

    Args:
        original: dict with keys 'PSE' and 'JND' (lists)
        simulated: dict with keys 'PSE' and 'JND' (lists)
        blocks: block sizes used

    Returns:
        Dict with pse_original, pse_simulated, pse_diff_pct,
              jnd_original, jnd_simulated, jnd_diff_pct,
              and evolution lists.
    """
    pse_orig = original['PSE'][-1] if original['PSE'] else 0
    pse_sim  = simulated['PSE'][-1] if simulated['PSE'] else 0
    jnd_orig = original['JND'][-1] if original['JND'] else 0
    jnd_sim  = simulated['JND'][-1] if simulated['JND'] else 0

    pse_diff = 100 * (pse_sim - pse_orig) / pse_orig if pse_orig != 0 else 0
    jnd_diff = 100 * (jnd_sim - jnd_orig) / jnd_orig if jnd_orig != 0 else 0

    return {
        'pse_original': pse_orig,
        'pse_simulated': pse_sim,
        'pse_diff_pct': pse_diff,
        'jnd_original': jnd_orig,
        'jnd_simulated': jnd_sim,
        'jnd_diff_pct': jnd_diff,
        'blocks': blocks,
        'pse_original_evolution': original['PSE'],
        'pse_simulated_evolution': simulated['PSE'],
        'jnd_original_evolution': original['JND'],
        'jnd_simulated_evolution': simulated['JND'],
    }
