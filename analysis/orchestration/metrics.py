"""
Unified metrics calculator for all analysis metrics from GBF data.


Handles both synthetic data (with ground_truth PSE/JND) and real data (without).

Metrics calculated (B1-B5):
  B1: Progressive psychometric (PSE/JND at trial blocks)
  B2: Latency statistics (center, spread, entropy, asymmetry)
  B3: Posterior SD convergence (Level B posteriors from GBF replay)
  B4: Auto-referential metrics (stability point, AUC)
  B5: Error metrics (synthetic only, vs ground truth)
"""

import logging
from typing import Dict, List, Optional, Any

import numpy as np

from main.config import TRIAL_BLOCKS
from analysis.core.extract_posterior_convergence import PosteriorExtractor
from analysis.core.psychometric_helpers import (
    fit_logistic_psychometric,
    calculate_stability_from_values,
    calculate_latency_statistics,
    calculate_progressive_asymmetry,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Unified calculator for ALL metrics from GBF trial data.
    
    Usage:
        # Synthetic data (with ground truth)
        calc = MetricsCalculator(
            model_type='ABS1',
            is_synthetic=True,
            ground_truth_pse=500.0,
            ground_truth_jnd=25.0
        )
        
        # Real data (no ground truth)
        calc = MetricsCalculator(
            model_type='REL2',
            is_synthetic=False
        )
        
        # Calculate all metrics from one GBF file
        gbf_rows = read_gbf_file('S01_G1_500_25_ABS1.txt')
        all_metrics = calc.calculate_all_metrics(gbf_rows)
    """

    def __init__(
        self,
        model_type: str,
        is_synthetic: bool,
        ground_truth_pse: Optional[float] = None,
        ground_truth_jnd: Optional[float] = None,
        offset: int = 500,
        trial_blocks: Optional[List[int]] = None,
    ):
        """
        Initialize metrics calculator.
        
        Args:
            model_type: 'ABS1', 'REL1', or 'REL2'
            is_synthetic: True for synthetic data, False for real data
            ground_truth_pse: Ground truth PSE (synthetic only)
            ground_truth_jnd: Ground truth JND (synthetic only)
            offset: Reference latency for asymmetry calculations
            trial_blocks: Trial block sizes for progressive analysis
        """
        self.model_type = model_type
        self.is_synthetic = is_synthetic
        self.ground_truth_pse = ground_truth_pse
        self.ground_truth_jnd = ground_truth_jnd
        self.offset = offset
        self.trial_blocks = trial_blocks or TRIAL_BLOCKS
        
        # Validation
        if is_synthetic and (ground_truth_pse is None or ground_truth_jnd is None):
            logger.warning(
                f"Synthetic data specified but ground truth values missing. "
                f"Error metrics (B5) will be skipped."
            )

    # ===== B1: PROGRESSIVE PSYCHOMETRIC (PSE/JND at trial blocks) =====

    def calculate_progressive_psychometric(self, gbf_rows: List[dict]) -> Dict[str, Any]:
        """
        Calculate PSE and JND at each trial block via logistic fitting.
        
        Args:
            gbf_rows: GBF data rows with 'lat' and 'user_ans' keys
            
        Returns:
            Dict with keys:
            - 'pse_values': List of PSE at each block
            - 'jnd_values': List of JND at each block
            - 'pse_40', 'pse_60', ..., 'pse_200': Individual PSE values
            - 'jnd_40', 'jnd_60', ..., 'jnd_200': Individual JND values
        """
        if not gbf_rows:
            logger.warning("Empty GBF rows for progressive psychometric")
            return self._empty_progressive_dict('pse', 'jnd')
        
        latencies = np.array([r['lat'] for r in gbf_rows], dtype=float)
        responses = np.array([r['user_ans'] for r in gbf_rows], dtype=int)
        
        result = {
            'pse_values': [],
            'jnd_values': [],
        }
        
        for block_size in self.trial_blocks:
            if block_size > len(latencies):
                break
            
            try:
                lat_block = latencies[:block_size]
                resp_block = responses[:block_size]
                pse, jnd = fit_logistic_psychometric(lat_block, resp_block, fallback=True)
                
                result['pse_values'].append(float(pse))
                result['jnd_values'].append(float(jnd))
                result[f'pse_{block_size}'] = float(pse)
                result[f'jnd_{block_size}'] = float(jnd)
            except Exception as e:
                logger.debug(f"Failed to fit at block {block_size}: {e}")
                result['pse_values'].append(np.nan)
                result['jnd_values'].append(np.nan)
                result[f'pse_{block_size}'] = np.nan
                result[f'jnd_{block_size}'] = np.nan
        
        return result

    # ===== B2: LATENCY STATISTICS (center, spread, entropy, asymmetry) =====
    def calculate_progressive_latency_stats(self, gbf_rows: List[dict]) -> Dict[str, Any]:
        """
        Calculate stimulus statistics at each trial block.
        
        Metrics:
        - stimulus_center: mean of latencies
        - stimulus_spread: std of latencies
        - lat_entropy: Shannon entropy of latency distribution
        - asymmetry_index: skew of stimulus distribution around offset
        
        Args:
            gbf_rows: GBF data rows
            
        Returns:
            Dict with keys like:
            - 'stimulus_center_40', 'stimulus_center_60', ..., 'stimulus_center_200'
            - 'stimulus_spread_40', ..., 'stimulus_spread_200'
            - 'lat_entropy_40', ..., 'lat_entropy_200'
            - 'asymmetry_40', ..., 'asymmetry_200'
        """
        if not gbf_rows:
            logger.warning("Empty GBF rows for latency stats")
            return self._empty_latency_dict()
        
        latencies = np.array([r['lat'] for r in gbf_rows], dtype=float)
        responses = np.array([r['user_ans'] for r in gbf_rows], dtype=int)
        rows_dict = [{'lat': r['lat'], 'user_ans': r['user_ans']} for r in gbf_rows]
        
        result = {}
        
        for block_size in self.trial_blocks:
            if block_size > len(latencies):
                break
            
            try:
                lat_block = latencies[:block_size]
                resp_block = responses[:block_size]
                rows_block = rows_dict[:block_size]
                
                # Latency statistics (center, spread, entropy)
                lat_stats = calculate_latency_statistics(lat_block)
                result[f'stimulus_center_{block_size}'] = float(lat_stats.get('stimulus_center', np.nan))
                result[f'stimulus_spread_{block_size}'] = float(lat_stats.get('stimulus_spread', np.nan))
                result[f'lat_entropy_{block_size}'] = float(lat_stats.get('lat_entropy', np.nan))
                
                # Asymmetry index
                asymmetry = calculate_progressive_asymmetry(rows_block, self.offset)
                if block_size in asymmetry:
                    result[f'asymmetry_{block_size}'] = float(asymmetry[block_size])
                else:
                    result[f'asymmetry_{block_size}'] = np.nan
                    
            except Exception as e:
                logger.debug(f"Failed to calculate latency stats at block {block_size}: {e}")
                result[f'stimulus_center_{block_size}'] = np.nan
                result[f'stimulus_spread_{block_size}'] = np.nan
                result[f'lat_entropy_{block_size}'] = np.nan
                result[f'asymmetry_{block_size}'] = np.nan
        
        return result

    # ===== B3: POSTERIOR SD CONVERGENCE (Level B) =====

    def calculate_posterior_sd_convergence(self, gbf_rows: List[dict]) -> Dict[str, Any]:
        """
        Calculate posterior SD (uncertainty) at each trial block via replay.
        
        Uses PosteriorExtractor to replay data through common ABS1 model
        and extract posterior SD at each trial block.
        
        Args:
            gbf_rows: GBF data rows
            
        Returns:
            Dict with keys like:
            - 'posterior_sd_pse_40', 'posterior_sd_pse_60', ..., 'posterior_sd_pse_200'
            - 'posterior_sd_jnd_40', ..., 'posterior_sd_jnd_200'
        """
        if not gbf_rows:
            logger.warning("Empty GBF rows for posterior SD")
            return self._empty_posterior_dict()
        
        try:
            # Extract latencies and responses
            latencies = [r['lat'] for r in gbf_rows]
            responses = [r['user_ans'] for r in gbf_rows]
            
            # Use PosteriorExtractor with ABS1 model (common model)
            extractor = PosteriorExtractor(model_type='ABS1', offset=self.offset)
            posterior_trajectory = extractor.extract_posterior_trajectory(latencies, responses)
            
            result = {}
            
            # Build result dict with posterior SD at each block
            trial_numbers = posterior_trajectory['trial_numbers']

            for block_size in self.trial_blocks:
                idx = np.argmax(trial_numbers >= block_size)
                if idx < len(posterior_trajectory['pse_sd']):
                    result[f'posterior_sd_pse_{block_size}'] = float(posterior_trajectory['pse_sd'][idx])
                    result[f'posterior_sd_jnd_{block_size}'] = float(posterior_trajectory['jnd_sd'][idx])

            return result
            
        except Exception as e:
            logger.warning(f"Failed to calculate posterior SD convergence: {e}")
            return self._empty_posterior_dict()

    # ===== B4: AUTO-REFERENTIAL METRICS =====

    def calculate_stability_points(self, progressive_dict: Dict) -> Dict[str, Any]:
        """
        Calculate stability points (trial block where parameter stabilizes).
        
        Stability = first block where param is within 10% of reference value.
        - For synthetic data: reference = ground_truth_pse/jnd
        - For real data: reference = final value (block 200)
        
        Args:
            progressive_dict: Result from calculate_progressive_psychometric()
            
        Returns:
            Dict with keys:
            - 'pse_stability_block': Block number for PSE stability (or NaN if no data)
            - 'jnd_stability_block': Block number for JND stability (or NaN if no data)
        """
        result = {}
        
        if 'pse_values' in progressive_dict and progressive_dict['pse_values']:
            pse_ref = self.ground_truth_pse if self.is_synthetic and self.ground_truth_pse is not None else None
            pse_stability = calculate_stability_from_values(
                progressive_dict['pse_values'],
                threshold=0.10,
                blocks=self.trial_blocks[:len(progressive_dict['pse_values'])],
                reference_value=pse_ref
            )
            # Guard: convert to int only if not None, otherwise use NaN
            result['pse_stability_block'] = int(pse_stability) if pse_stability is not None else np.nan
        else:
            result['pse_stability_block'] = np.nan
        
        if 'jnd_values' in progressive_dict and progressive_dict['jnd_values']:
            jnd_ref = self.ground_truth_jnd if self.is_synthetic and self.ground_truth_jnd is not None else None
            jnd_stability = calculate_stability_from_values(
                progressive_dict['jnd_values'],
                threshold=0.10,
                blocks=self.trial_blocks[:len(progressive_dict['jnd_values'])],
                reference_value=jnd_ref
            )
            # Guard: convert to int only if not None, otherwise use NaN
            result['jnd_stability_block'] = int(jnd_stability) if jnd_stability is not None else np.nan
        else:
            result['jnd_stability_block'] = np.nan
        
        return result

    def calculate_auc(self, progressive_dict: Dict, final_pse: float, final_jnd: float) -> Dict[str, float]:
        """
        Calculate Area Under Curve (cumulative fit quality).
        
        AUC measures how quickly estimate converges (lower is better).
        Reference value:
        - For synthetic data: ground_truth_pse/jnd
        - For real data: final_pse/jnd (block 200)
        
        Args:
            progressive_dict: Result from calculate_progressive_psychometric()
            final_pse: Final PSE estimate (at trial 200)
            final_jnd: Final JND estimate (at trial 200)
            
        Returns:
            Dict with keys:
            - 'pse_auc': AUC for PSE convergence
            - 'jnd_auc': AUC for JND convergence
        """
        result = {}
        
        pse_ref = self.ground_truth_pse if self.is_synthetic and self.ground_truth_pse is not None else final_pse
        jnd_ref = self.ground_truth_jnd if self.is_synthetic and self.ground_truth_jnd is not None else final_jnd
        
        if 'pse_values' in progressive_dict and pse_ref:
            pse_errors = []
            for pse_val in progressive_dict['pse_values']:
                # Skip NaN values (exclude missing fits from AUC calculation)
                if not np.isnan(pse_val):
                    error = abs(pse_val - pse_ref) / (abs(pse_ref) + 1e-10)
                    pse_errors.append(error)
            # AUC = mean of valid errors (or NaN if all NaN)
            result['pse_auc'] = float(np.mean(pse_errors)) if pse_errors else np.nan
        
        if 'jnd_values' in progressive_dict and jnd_ref:
            jnd_errors = []
            for jnd_val in progressive_dict['jnd_values']:
                # Skip NaN values (exclude missing fits from AUC calculation)
                if not np.isnan(jnd_val):
                    error = abs(jnd_val - jnd_ref) / (abs(jnd_ref) + 1e-10)
                    jnd_errors.append(error)
            # AUC = mean of valid errors (or NaN if all NaN)
            result['jnd_auc'] = float(np.mean(jnd_errors)) if jnd_errors else np.nan
        
        return result

    # ===== B5: ERROR METRICS (Synthetic ONLY) =====

    def calculate_error_metrics(self, final_pse: float, final_jnd: float) -> Dict[str, Any]:
        """
        Calculate error metrics vs ground truth (synthetic data only).
        
        Args:
            final_pse: Final PSE estimate (trial 200)
            final_jnd: Final JND estimate (trial 200)
            
        Returns:
            Dict with error metrics if synthetic, empty dict if real data:
            - 'pse_error': Absolute PSE error (ms)
            - 'jnd_error': Absolute JND error (ms)
            - 'pse_error_pct': Relative PSE error (%)
            - 'jnd_error_pct': Relative JND error (%)
        """
        result = {}
        
        if not self.is_synthetic:
            return result
        
        if self.ground_truth_pse is None or self.ground_truth_jnd is None:
            logger.warning("Synthetic data but ground truth not provided")
            return result
        
        if not np.isnan(final_pse):
            pse_error = final_pse - self.ground_truth_pse
            result['pse_error'] = float(pse_error)
            result['pse_error_pct'] = float(
                100.0 * pse_error / (abs(self.ground_truth_pse) + 1e-10)
            )
        
        if not np.isnan(final_jnd):
            jnd_error = final_jnd - self.ground_truth_jnd
            result['jnd_error'] = float(jnd_error)
            result['jnd_error_pct'] = float(
                100.0 * jnd_error / (abs(self.ground_truth_jnd) + 1e-10)
            )
        
        return result

    # ===== ORCHESTRATION =====

    def calculate_all_metrics(self, gbf_rows: List[dict]) -> Dict[str, Any]:
        """
        Master function: calculate ALL metrics from one GBF file.
        
        Orchestrates B1-B5 calculations and combines results into comprehensive
        behavioral metrics for model comparison.
        
        Args:
            gbf_rows: GBF data rows (from read_gbf_file())
                Each row contains: lat (latency in ms), res (success/failure), user_ans (response)
            
        Returns:
            Comprehensive dict with metrics organized by category:
            
            **B1 - Progressive Psychometric (PSE/JND at each trial block):**
            - pse_values: PSE estimates [block_40, block_60, ..., block_200]
            - jnd_values: JND estimates [block_40, block_60, ..., block_200]
            - slope_values: Slope estimates
            
            **B2 - Latency Statistics (stimulus properties at each trial block):**
            - stimulus_center_*: Center of stimulus distribution at each block
            - stimulus_spread_*: Spread (SD) of stimulus distribution
            - asymmetry_index_*: Asymmetry of stimulus presentation
            - lat_entropy_*: Shannon entropy of latency distribution
            
            **B3 - Posterior SD Convergence (posterior uncertainty at each block):**
            - posterior_sd_pse_*: Posterior SD of PSE estimate
            - posterior_sd_jnd_*: Posterior SD of JND estimate
            
            **B4 - Auto-referential (stability + convergence speed):**
            - pse_stability_block: First block where PSE SD < 10% of final
            - jnd_stability_block: First block where JND SD < 10% of final
            - pse_auc: Area Under error Curve (lower = faster convergence)
            - jnd_auc: Area Under error Curve for JND
            
            **B5 - Error Metrics (synthetic data only, vs ground truth):**
            - pse_error_pct: Final PSE error as % of true value
            - jnd_error_pct: Final JND error as % of true value
        """
        import time
        
        result = {}
        t_start = time.time()
        
        # B1: Progressive psychometric
        t_b1 = time.time()
        b1_metrics = self.calculate_progressive_psychometric(gbf_rows)
        t_b1 = time.time() - t_b1
        result.update(b1_metrics)
        
        # B2: Latency statistics
        t_b2 = time.time()
        b2_metrics = self.calculate_progressive_latency_stats(gbf_rows)
        t_b2 = time.time() - t_b2
        result.update(b2_metrics)
        
        # B3: Posterior SD convergence
        t_b3 = time.time()
        b3_metrics = self.calculate_posterior_sd_convergence(gbf_rows)
        t_b3 = time.time() - t_b3
        result.update(b3_metrics)
        
        # B4: Auto-referential (need final values from B1)
        t_b4 = time.time()
        if 'pse_values' in b1_metrics and b1_metrics['pse_values']:
            final_pse = b1_metrics['pse_values'][-1]
            final_jnd = b1_metrics['jnd_values'][-1] if 'jnd_values' in b1_metrics else np.nan
            
            b4_stability = self.calculate_stability_points(b1_metrics)
            result.update(b4_stability)
            
            b4_auc = self.calculate_auc(b1_metrics, final_pse, final_jnd)
            result.update(b4_auc)
            
            # B5: Error metrics (synthetic only)
            b5_metrics = self.calculate_error_metrics(final_pse, final_jnd)
            result.update(b5_metrics)
        t_b4 = time.time() - t_b4
        
        t_total = time.time() - t_start
        
        # Debug: print timing if any step > 0.1s
        if any([t_b1 > 0.1, t_b2 > 0.1, t_b3 > 0.1, t_b4 > 0.1]):
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Metrics timing: B1={t_b1:.3f}s, B2={t_b2:.3f}s, B3={t_b3:.3f}s, B4={t_b4:.3f}s | TOTAL={t_total:.3f}s")
        
        return result

    # ===== HELPERS =====

    def _empty_progressive_dict(self, *metric_names) -> Dict[str, Any]:
        """Create empty progressive dict structure."""
        result = {}
        for metric in metric_names:
            result[f'{metric}_values'] = []
            for block in self.trial_blocks:
                result[f'{metric}_{block}'] = np.nan
        return result

    def _empty_latency_dict(self) -> Dict[str, float]:
        """Create empty latency stats dict."""
        result = {}
        for metric in ['stimulus_center', 'stimulus_spread', 'lat_entropy', 'asymmetry']:
            for block in self.trial_blocks:
                result[f'{metric}_{block}'] = np.nan
        return result

    def _empty_posterior_dict(self) -> Dict[str, float]:
        """Create empty posterior SD dict."""
        result = {}
        for metric in ['posterior_sd_pse', 'posterior_sd_jnd']:
            for block in self.trial_blocks:
                result[f'{metric}_{block}'] = np.nan
        return result
