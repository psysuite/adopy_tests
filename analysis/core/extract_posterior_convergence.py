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
        metric: str = "max",
    ) -> Dict[str, np.ndarray]:
        """
        Extract posterior evolution for REL2 (dual-estimator approach).
        
        REL2 uses two separate estimators:
        - Pre estimator: processes pre-trials (stimulus < offset)
        - Post estimator: processes post-trials (stimulus > offset)
        
        Design: Every 20 trials contain 10 pre + 10 post (alternating).
        This method extracts posteriors aligned to global trial numbering [1-200],
        with pre and post posteriors combined conservatively.
        
        Alignment strategy:
        - For each global trial block [40, 60, 80, ..., 200]:
          - Extract sub-trials that are pre and sub-trials that are post
          - Combine using conservative metric (max or mean)
        
        Args:
            stimuli: Absolute stimulus values (one per trial) [1-200]
            responses: Binary responses (0 or 1, one per trial) [1-200]
            pre_post_labels: List of 'pre' or 'post' labels (one per trial) [1-200]
            metric: How to combine pre/post posteriors at each block:
                    'max' = max(pse_sd_pre, pse_sd_post) [conservative, default]
                    'mean' = mean(pse_sd_pre, pse_sd_post) [balanced]
        
        Returns:
            Dict with keys (same structure as extract_posterior_trajectory()):
            - 'trial_numbers': Global trial indices [1, 2, ..., 200]
            - 'pse_mean': PSE estimates at each trial
            - 'pse_sd': PSE posterior SDs (combined pre/post)
            - 'jnd_mean': JND estimates at each trial
            - 'jnd_sd': JND posterior SDs (combined pre/post)
            - 'slope_mean': Slope estimates at each trial
            - 'slope_sd': Slope posterior SDs (combined pre/post)
            
        Raises:
            ValueError: if lengths don't match, model_type is not REL2, or metric is invalid
        """
        if self.model_type != 'REL2':
            raise ValueError(f"extract_posterior_trajectory_rel2() only works for REL2, not {self.model_type}")
        
        if len(stimuli) != len(responses) or len(stimuli) != len(pre_post_labels):
            raise ValueError("stimuli, responses, and pre_post_labels must have same length")
        
        if metric not in ['max', 'mean']:
            raise ValueError(f"metric must be 'max' or 'mean', got {metric}")

        # Separate trials by pre/post (maintaining order for pre_post_labels indexing)
        pre_mask = np.array([label == 'pre' for label in pre_post_labels])
        post_mask = np.array([label == 'post' for label in pre_post_labels])
        
        stimuli = np.array(stimuli)
        responses = np.array(responses)
        
        stimuli_pre = stimuli[pre_mask].tolist()
        responses_pre = responses[pre_mask].tolist()
        
        stimuli_post = stimuli[post_mask].tolist()
        responses_post = responses[post_mask].tolist()
        
        # Extract posteriors for each estimator (separate sequences)
        posterior_pre = self.extract_posterior_trajectory(stimuli_pre, responses_pre)
        posterior_post = self.extract_posterior_trajectory(stimuli_post, responses_post)
        
        # Combine pre/post posteriors by reconstructing global trial alignment
        # For each global trial, determine if it's pre or post, and take the corresponding posterior
        trial_numbers_pre = posterior_pre['trial_numbers']  # [1, 2, ..., 100] for pre trials
        trial_numbers_post = posterior_post['trial_numbers']  # [1, 2, ..., 100] for post trials
        
        # Map global trial index to (is_pre, index_in_sequence)
        pre_idx = 0
        post_idx = 0
        
        combined_trial_numbers = []
        combined_pse_mean = []
        combined_pse_sd = []
        combined_jnd_mean = []
        combined_jnd_sd = []
        combined_slope_mean = []
        combined_slope_sd = []
        
        for trial_idx, (is_pre, is_post) in enumerate(zip(pre_mask, post_mask)):
            if is_pre:
                if pre_idx < len(trial_numbers_pre):
                    combined_trial_numbers.append(trial_idx + 1)  # Global trial number (1-indexed)
                    combined_pse_mean.append(posterior_pre['pse_mean'][pre_idx])
                    combined_pse_sd.append(posterior_pre['pse_sd'][pre_idx])
                    combined_jnd_mean.append(posterior_pre['jnd_mean'][pre_idx])
                    combined_jnd_sd.append(posterior_pre['jnd_sd'][pre_idx])
                    combined_slope_mean.append(posterior_pre['slope_mean'][pre_idx])
                    combined_slope_sd.append(posterior_pre['slope_sd'][pre_idx])
                    pre_idx += 1
            elif is_post:
                if post_idx < len(trial_numbers_post):
                    combined_trial_numbers.append(trial_idx + 1)  # Global trial number (1-indexed)
                    combined_pse_mean.append(posterior_post['pse_mean'][post_idx])
                    combined_pse_sd.append(posterior_post['pse_sd'][post_idx])
                    combined_jnd_mean.append(posterior_post['jnd_mean'][post_idx])
                    combined_jnd_sd.append(posterior_post['jnd_sd'][post_idx])
                    combined_slope_mean.append(posterior_post['slope_mean'][post_idx])
                    combined_slope_sd.append(posterior_post['slope_sd'][post_idx])
                    post_idx += 1
        
        # Convert to arrays
        combined_trial_numbers = np.array(combined_trial_numbers)
        combined_pse_mean = np.array(combined_pse_mean)
        combined_pse_sd_pre = np.array(combined_pse_sd)
        combined_jnd_mean = np.array(combined_jnd_mean)
        combined_jnd_sd_pre = np.array(combined_jnd_sd)
        combined_slope_mean = np.array(combined_slope_mean)
        combined_slope_sd_pre = np.array(combined_slope_sd)
        
        # Now extract at trial blocks and apply metric (max or mean of pre/post)
        # This requires re-aligning: for each block, find all pre and post trials up to that block
        pse_sd_final = np.zeros_like(combined_pse_sd_pre)
        jnd_sd_final = np.zeros_like(combined_jnd_sd_pre)
        slope_sd_final = np.zeros_like(combined_slope_sd_pre)
        
        # For each position in the combined array, determine if it's a "block boundary"
        # and apply metric across pre/post within that block
        for i, trial_num in enumerate(combined_trial_numbers):
            # Find corresponding pre and post posteriors at this trial
            pre_trials_up_to = np.sum(pre_mask[:trial_num])
            post_trials_up_to = np.sum(post_mask[:trial_num])
            
            if pre_trials_up_to > 0 and pre_trials_up_to <= len(posterior_pre['pse_sd']):
                pse_sd_pre_at_trial = posterior_pre['pse_sd'][pre_trials_up_to - 1]
                jnd_sd_pre_at_trial = posterior_pre['jnd_sd'][pre_trials_up_to - 1]
                slope_sd_pre_at_trial = posterior_pre['slope_sd'][pre_trials_up_to - 1]
            else:
                pse_sd_pre_at_trial = np.nan
                jnd_sd_pre_at_trial = np.nan
                slope_sd_pre_at_trial = np.nan
            
            if post_trials_up_to > 0 and post_trials_up_to <= len(posterior_post['pse_sd']):
                pse_sd_post_at_trial = posterior_post['pse_sd'][post_trials_up_to - 1]
                jnd_sd_post_at_trial = posterior_post['jnd_sd'][post_trials_up_to - 1]
                slope_sd_post_at_trial = posterior_post['slope_sd'][post_trials_up_to - 1]
            else:
                pse_sd_post_at_trial = np.nan
                jnd_sd_post_at_trial = np.nan
                slope_sd_post_at_trial = np.nan
            
            # Apply metric (max or mean)
            if metric == 'max':
                pse_sd_final[i] = np.nanmax([pse_sd_pre_at_trial, pse_sd_post_at_trial])
                jnd_sd_final[i] = np.nanmax([jnd_sd_pre_at_trial, jnd_sd_post_at_trial])
                slope_sd_final[i] = np.nanmax([slope_sd_pre_at_trial, slope_sd_post_at_trial])
            else:  # mean
                pse_sd_final[i] = np.nanmean([pse_sd_pre_at_trial, pse_sd_post_at_trial])
                jnd_sd_final[i] = np.nanmean([jnd_sd_pre_at_trial, jnd_sd_post_at_trial])
                slope_sd_final[i] = np.nanmean([slope_sd_pre_at_trial, slope_sd_post_at_trial])
        
        return {
            'trial_numbers': combined_trial_numbers,
            'pse_mean': combined_pse_mean,
            'pse_sd': pse_sd_final,
            'jnd_mean': combined_jnd_mean,
            'jnd_sd': jnd_sd_final,
            'slope_mean': combined_slope_mean,
            'slope_sd': slope_sd_final,
        }


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
