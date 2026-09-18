#!/usr/bin/env python3
"""
Extract posterior mean and SD from ADOpy ad every trial.

This replaces the autoreferential metrics (AUC, SP) with objective convergence
measures based on the posterior uncertainty. Works for both synthetic and real data.

The posterior SD directly measures "how uncertain are we about this parameter?"
and decreases monotonically as trials accumulate information.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from adopy.tasks.psi import Task2AFC, ModelLogistic, EnginePsi
from utilities.misc_generate_responses import get_sigma_from_jnd
from main.config import TRIAL_BLOCKS, OFFSET

# Constants for delta-method stability
SLOPE_MIN = 1e-6  # Minimum slope to prevent division by zero or negative JND SD


class PosteriorExtractor:
    """Extract posterior evolution from trial sequences."""

    def __init__(self, model_type: str = 'ABS1', offset: Optional[int] = None):
        """
        Initialize PosteriorExtractor for model type and offset.
        
        The extractor replays trial sequences to compute posterior mean and SD
        at each trial, enabling measurement of convergence speed and parameter stability.
        Uses ADOpy's EnginePsi to compute Bayesian posteriors.
        
        Args:
            model_type: One of 'ABS1' (absolute), 'REL1' (relative single), 'REL2' (relative dual).
                       Controls guess_rate, stimulus range, and parameter grids.
            offset: Reference latency (ms) used for relative discrimination (default: config.OFFSET=500).
                   For ABS1: not used.
                   For REL1/REL2: reference point for pre (<500) vs post (>500).
        """
        self.model_type = model_type
        self.offset = offset if offset is not None else OFFSET
        self._setup_adopy()

    def _setup_adopy(self):
        """Setup ADOpy engine with standard configuration matching bisection wrappers."""
        self.task = Task2AFC()
        self.model = ModelLogistic()

        # Use EXACT same grid as BISAbsADOpyWrapper and BISRelADOpyWrapper
        # for maximum precision (not speed)
        if self.model_type == 'ABS1':
            # ABS1: absolute latencies (200-800 ms)
            min_stim = 200
            max_stim = 800
            stimulus_range = np.linspace(min_stim, max_stim, 200)
            threshold_range = np.linspace(min_stim, max_stim, 200)
        else:  # REL1, REL2
            # REL1/REL2: relative latencies (0.1-300 ms magnitude)
            min_stim = 0.1
            max_stim = 300
            stimulus_range = np.linspace(min_stim, max_stim, 200)
            threshold_range = np.linspace(min_stim, max_stim, 200)

        self.designs = {'stimulus': stimulus_range}
        
        # Set guess_rate and lapse_rate based on model type
        if self.model_type == 'ABS1':
            guess_rate = 0.04
        else:  # REL1, REL2
            guess_rate = 0.5
        lapse_rate = 0.04  # Same for all models
        
        self.params = {
            'guess_rate': [guess_rate],
            'lapse_rate': [lapse_rate],
            'threshold': threshold_range,
            'slope': np.logspace(-2, 1, 200),  # 0.01 to 10 (same as wrappers)
        }

    def extract_posterior_trajectory(
        self,
        stimuli: List[float],
        responses: List[int],
    ) -> Dict[str, np.ndarray]:
        """
        Extract posterior evolution for a trial sequence.

        Args:
            stimuli: Stimulus values (one per trial)
            responses: Binary responses (0 or 1, one per trial)

        Returns:
            Dictionary with keys:
            - 'trial_numbers': [1, 2, 3, ..., n_trials]
            - 'pse_mean': posterior mean of threshold at each trial
            - 'pse_sd': posterior SD of threshold at each trial
            - 'jnd_mean': posterior mean of JND at each trial
            - 'jnd_sd': posterior SD of JND at each trial (via delta method)
            - 'slope_mean': posterior mean of slope (beta) at each trial
            - 'slope_sd': posterior SD of slope at each trial
        """
        if len(stimuli) != len(responses):
            raise ValueError("stimuli and responses must have same length")

        # Create engine
        engine = EnginePsi(self.model, self.designs, self.params)

        # Storage
        trial_numbers = []
        pse_means = []
        pse_sds = []
        slope_means = []
        slope_sds = []

        # Replay trial-by-trial
        for trial_idx, (stim, resp) in enumerate(zip(stimuli, responses)):
            # Update engine
            design = {'stimulus': float(stim)}
            engine.update(design, int(resp))

            # Extract posterior
            trial_numbers.append(trial_idx + 1)

            # PSE (threshold)
            pse_mean = float(engine.post_mean['threshold'])
            pse_sd = float(engine.post_sd['threshold'])
            pse_means.append(pse_mean)
            pse_sds.append(pse_sd)

            # Slope (beta)
            slope_mean = float(engine.post_mean['slope'])
            slope_sd = float(engine.post_sd['slope'])
            slope_means.append(slope_mean)
            slope_sds.append(slope_sd)

        # Convert slope to JND
        # JND = ln(3) / slope
        # Using delta method: d(JND)/d(slope) = -ln(3) / slope^2
        jnd_means = []
        jnd_sds = []
        for s, sd in zip(slope_means, slope_sds):
            # Guard slope against zero/negative to prevent division errors
            s_safe = max(s, SLOPE_MIN)
            sd_safe = max(sd, 1e-10)
            jnd_m = np.log(3) / s_safe
            jnd_s = np.log(3) / (s_safe ** 2) * sd_safe
            # Ensure JND values are physically valid (positive)
            jnd_means.append(max(jnd_m, SLOPE_MIN))
            jnd_sds.append(max(jnd_s, SLOPE_MIN))

        return {
            'trial_numbers': np.array(trial_numbers),
            'pse_mean': np.array(pse_means),
            'pse_sd': np.array(pse_sds),
            'jnd_mean': np.array(jnd_means),
            'jnd_sd': np.array(jnd_sds),
            'slope_mean': np.array(slope_means),
            'slope_sd': np.array(slope_sds),
        }

    def extract_posterior_trajectory_rel2(
        self,
        stimuli: List[float],
        responses: List[int],
        pre_post_labels: List[str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Extract posterior evolution for REL2 (dual-model approach).
        
        REL2 uses two separate estimators: one for pre (stimulus < 500ms)
        and one for post (stimulus > 500ms). This method extracts posteriors
        for each model independently.

        Args:
            stimuli: Absolute stimulus values (one per trial)
            responses: Binary responses (0 or 1, one per trial)
            pre_post_labels: List of 'pre' or 'post' labels (one per trial)

        Returns:
            Tuple of (posterior_pre, posterior_post), each with same structure as
            extract_posterior_trajectory() output.
            
        Raises:
            ValueError: if lengths don't match or model_type is not REL2
        """
        if self.model_type != 'REL2':
            raise ValueError(f"extract_posterior_trajectory_rel2() only works for REL2, not {self.model_type}")
        
        if len(stimuli) != len(responses) or len(stimuli) != len(pre_post_labels):
            raise ValueError("stimuli, responses, and pre_post_labels must have same length")

        # Separate trials by pre/post
        pre_mask = [label == 'pre' for label in pre_post_labels]
        post_mask = [label == 'post' for label in pre_post_labels]
        
        stimuli_pre = [s for s, m in zip(stimuli, pre_mask) if m]
        responses_pre = [r for r, m in zip(responses, pre_mask) if m]
        
        stimuli_post = [s for s, m in zip(stimuli, post_mask) if m]
        responses_post = [r for r, m in zip(responses, post_mask) if m]
        
        # Extract posteriors for each model
        posterior_pre = self.extract_posterior_trajectory(stimuli_pre, responses_pre)
        posterior_post = self.extract_posterior_trajectory(stimuli_post, responses_post)
        
        return posterior_pre, posterior_post


def calculate_convergence_metrics(
    posterior_data: Dict[str, np.ndarray],
    true_values: Optional[Dict[str, float]] = None,
    trial_blocks: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Calculate convergence metrics from posterior trajectory.

    Args:
        posterior_data: Output from extract_posterior_trajectory()
        true_values: Dict with 'pse' and 'jnd' ground truth values (for synthetic data)
        trial_blocks: Trial numbers at which to calculate metrics
                     (default: [40, 60, 80, 100, 120, 140, 160, 180, 200])

    Returns:
        Dictionary with convergence metrics:
        - 'final_pse_mean': PSE estimate at final trial
        - 'final_jnd_mean': JND estimate at final trial
        - 'final_pse_sd': PSE uncertainty at final trial
        - 'final_jnd_sd': JND uncertainty at final trial
        - 'pse_sd_progression': [SD at block 40, 60, ..., 200]
        - 'jnd_sd_progression': [SD at block 40, 60, ..., 200]
        - 'pse_error_progression': [error at block 40, 60, ..., 200] (only if true_values given)
        - 'jnd_error_progression': [error at block 40, 60, ..., 200] (only if true_values given)
        - 'pse_stability_block': First block where PSE SD < 10% of final SD
        - 'jnd_stability_block': First block where JND SD < 10% of final SD
    """
    if trial_blocks is None:
        trial_blocks = TRIAL_BLOCKS

    trial_numbers = posterior_data['trial_numbers']
    pse_sds = posterior_data['pse_sd']
    jnd_sds = posterior_data['jnd_sd']
    pse_means = posterior_data['pse_mean']
    jnd_means = posterior_data['jnd_mean']

    # Final values
    final_pse_mean = float(pse_means[-1])
    final_jnd_mean = float(jnd_means[-1])
    final_pse_sd = float(pse_sds[-1])
    final_jnd_sd = float(jnd_sds[-1])

    # Get values at trial blocks
    pse_sd_at_blocks = []
    jnd_sd_at_blocks = []
    pse_error_at_blocks = []
    jnd_error_at_blocks = []

    for block in trial_blocks:
        idx = np.argmax(trial_numbers >= block)
        if idx < len(trial_numbers):
            pse_sd_at_blocks.append(float(pse_sds[idx]))
            jnd_sd_at_blocks.append(float(jnd_sds[idx]))

            if true_values is not None:
                pse_error = abs(pse_means[idx] - true_values['pse'])
                jnd_error = abs(jnd_means[idx] - true_values['jnd'])
                pse_error_at_blocks.append(float(pse_error))
                jnd_error_at_blocks.append(float(jnd_error))

    # Stability points: first block where SD < 10% of final SD
    pse_stability_threshold = 0.1 * final_pse_sd
    jnd_stability_threshold = 0.1 * final_jnd_sd

    pse_stability_block = None
    jnd_stability_block = None

    for block, sd in zip(trial_blocks, pse_sd_at_blocks):
        if pse_stability_block is None and sd < pse_stability_threshold:
            pse_stability_block = block

    for block, sd in zip(trial_blocks, jnd_sd_at_blocks):
        if jnd_stability_block is None and sd < jnd_stability_threshold:
            jnd_stability_block = block

    result = {
        'final_pse_mean': final_pse_mean,
        'final_jnd_mean': final_jnd_mean,
        'final_pse_sd': final_pse_sd,
        'final_jnd_sd': final_jnd_sd,
        'pse_sd_progression': np.array(pse_sd_at_blocks),
        'jnd_sd_progression': np.array(jnd_sd_at_blocks),
        'pse_stability_block': pse_stability_block,
        'jnd_stability_block': jnd_stability_block,
    }

    if true_values is not None:
        result['pse_error_progression'] = np.array(pse_error_at_blocks)
        result['jnd_error_progression'] = np.array(jnd_error_at_blocks)

    return result


def process_synthetic_data(
    stimuli: List[float],
    responses: List[int],
    model_type: str,
    true_pse: float,
    true_jnd: float,
) -> pd.DataFrame:
    """
    Process synthetic data: extract posterior and calculate convergence.

    Returns DataFrame with one row per trial block:
    trial_block | pse_mean | pse_sd | jnd_mean | jnd_sd | pse_error | jnd_error
    """
    extractor = PosteriorExtractor(model_type=model_type)
    posterior = extractor.extract_posterior_trajectory(stimuli, responses)

    metrics = calculate_convergence_metrics(
        posterior,
        true_values={'pse': true_pse, 'jnd': true_jnd},
    )

    trial_blocks = TRIAL_BLOCKS

    df = pd.DataFrame({
        'trial_block': trial_blocks,
        'pse_mean': posterior['pse_mean'][[np.argmax(posterior['trial_numbers'] >= b) for b in trial_blocks]],
        'pse_sd': metrics['pse_sd_progression'],
        'jnd_mean': posterior['jnd_mean'][[np.argmax(posterior['trial_numbers'] >= b) for b in trial_blocks]],
        'jnd_sd': metrics['jnd_sd_progression'],
        'pse_error': metrics['pse_error_progression'],
        'jnd_error': metrics['jnd_error_progression'],
    })

    return df


def process_human_data(
    stimuli: List[float],
    responses: List[int],
    model_type: str,
) -> pd.DataFrame:
    """
    Process human data: extract posterior and calculate convergence.

    Since there's no ground truth, we can only report SD progression.

    Returns DataFrame with one row per trial block:
    trial_block | pse_mean | pse_sd | jnd_mean | jnd_sd
    """
    extractor = PosteriorExtractor(model_type=model_type)
    posterior = extractor.extract_posterior_trajectory(stimuli, responses)

    metrics = calculate_convergence_metrics(posterior)

    trial_blocks = TRIAL_BLOCKS

    df = pd.DataFrame({
        'trial_block': trial_blocks,
        'pse_mean': posterior['pse_mean'][[np.argmax(posterior['trial_numbers'] >= b) for b in trial_blocks]],
        'pse_sd': metrics['pse_sd_progression'],
        'jnd_mean': posterior['jnd_mean'][[np.argmax(posterior['trial_numbers'] >= b) for b in trial_blocks]],
        'jnd_sd': metrics['jnd_sd_progression'],
    })

    return df


if __name__ == '__main__':
    # Example: synthetic data
    print("=" * 80)
    print("EXAMPLE: Extracting posterior from synthetic data")
    print("=" * 80)

    # Simulate SOME data (just 50 trials for testing speed)
    from utilities.misc_generate_responses import generate_response

    pse_true = 500
    jnd_true = 40
    sigma = get_sigma_from_jnd(jnd_true)

    np.random.seed(42)
    n_trials = 50  # Much smaller for testing
    stimuli = np.random.uniform(200, 800, n_trials)
    responses = [generate_response(s, pse_true, sigma) for s in stimuli]

    print(f"\nSimulated {n_trials} trials...")
    print(f"True values: PSE={pse_true} ms, JND={jnd_true} ms")

    # Process
    print("\nExtracting posterior from ADOpy...")
    extractor = PosteriorExtractor(model_type='ABS1')
    posterior = extractor.extract_posterior_trajectory(stimuli, responses)

    print("✓ Posterior extracted!")

    # Metrics
    print("\nCalculating convergence metrics...")
    metrics = calculate_convergence_metrics(
        posterior,
        true_values={'pse': pse_true, 'jnd': jnd_true},
        trial_blocks=[10, 20, 30, 40, 50],
    )

    print(f"\nFinal estimates: PSE={metrics['final_pse_mean']:.1f}±{metrics['final_pse_sd']:.1f} ms")
    print(f"                 JND={metrics['final_jnd_mean']:.1f}±{metrics['final_jnd_sd']:.1f} ms")
    print(f"\nConvergence:")
    print(f"  PSE stability (SD < 10% final): Trial block {metrics['pse_stability_block']}")
    print(f"  JND stability (SD < 10% final): Trial block {metrics['jnd_stability_block']}")

    print(f"\nSD progression (PSE):")
    for block, sd in zip([10, 20, 30, 40, 50], metrics['pse_sd_progression']):
        print(f"  Block {block}: {sd:.2f} ms")

    print(f"\nError progression (PSE):")
    for block, err in zip([10, 20, 30, 40, 50], metrics['pse_error_progression']):
        print(f"  Block {block}: {err:.2f} ms")

    print("\n" + "=" * 80)
    print("✓ Done!")
