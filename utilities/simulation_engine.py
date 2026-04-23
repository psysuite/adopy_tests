"""
Simulation engine - unified simulation logic for temporal bisection experiments.

Centralizes subject simulation, file generation, and analysis across grid and random modes.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import logging

from utilities.misc_generate_responses import generate_response, get_sigma_from_jnd
from utilities.trial_sequence import create_trial_sequence_relative, create_trial_sequence_absolute
from bisection.BISAbsADOpyWrapper import BISAbsADOpyWrapper
from bisection.BISRelADOpyWrapper import BISRelADOpyWrapper

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Unified simulation engine for temporal bisection experiments."""

    def __init__(self, model_type: str, offset: int = 500):
        """
        Args:
            model_type: '1model_abs', '1model_rel', or '2model_rel'
            offset: Reference latency (default 500ms)
        """
        self.model_type = model_type
        self.offset = offset
        self._validate_model_type()

    def _validate_model_type(self):
        """Validate model type."""
        valid_models = ['1model_abs', '1model_rel', '2model_rel']
        if self.model_type not in valid_models:
            raise ValueError(f"Invalid model_type: {self.model_type}. Must be one of {valid_models}")

    def simulate_subject(
        self,
        pse: float,
        jnd: float,
        ntrials: int,
        ado_params: Dict,
        bis_params: Dict,
        fixed_trials_config: Dict = None,
    ) -> Tuple[List[Dict], Dict]:
        """
        Simulate experiment for a single subject.

        Args:
            pse: Point of subjective equality
            jnd: Just noticeable difference
            ntrials: Number of trials
            ado_params: ADOpy parameters (guess_rate, lapse_rate, noise_perc)
            bis_params: Bisection task parameters (min, max, offset, ntrials)
            fixed_trials_config: Config for fixed trials (use, n_fixed, offsets)

        Returns:
            Tuple of (rows, result_dict)
            - rows: List of trial dicts for plotting
            - result_dict: Statistics dict
        """
        sigma = get_sigma_from_jnd(jnd)

        if self.model_type == '1model_abs':
            return self._simulate_1model_abs(
                pse, sigma, jnd, ntrials, ado_params, bis_params, fixed_trials_config
            )
        elif self.model_type == '1model_rel':
            return self._simulate_1model_rel(
                pse, sigma, jnd, ntrials, ado_params, bis_params, fixed_trials_config
            )
        elif self.model_type == '2model_rel':
            return self._simulate_2model_rel(
                pse, sigma, jnd, ntrials, ado_params, bis_params, fixed_trials_config
            )
        raise Exception(f"simulate_subject: model {self.model_type} not recognized")

    def _simulate_1model_abs(
        self,
        pse: float,
        sigma: float,
        jnd: float,
        ntrials: int,
        ado_params: Dict,
        bis_params: Dict,
        fixed_trials_config: Dict,
    ) -> Tuple[List[Dict], Dict]:
        """Simulate 1-model absolute approach."""

        exp = BISAbsADOpyWrapper(adoparams=ado_params, taskparams=bis_params)

        fixed_config = fixed_trials_config or {'use': False}
        trial_sequence = create_trial_sequence_absolute(
            ntrials,
            fixed_config.get('latencies', []),
            fixed_config.get('n_fixed', 0),
            fixed_config.get('use', False),
        )

        rows = []
        for trial_id, (trial_info, trial_type) in enumerate(trial_sequence):
            stim_ms = trial_info if trial_type == 'fixed' else exp.get()
            user_ans = generate_response(stim_ms, pse, sigma)
            success = int((stim_ms > self.offset) == user_ans)
            exp.set(user_ans, stim_ms)

            rows.append({
                'lat': int(stim_ms),
                'res': str(success == 1).lower(),
                'user_ans': user_ans,
            })

        result_dict = {
            'subj': f'SIM_{pse:.0f}_{jnd:.0f}',
            'pse': pse,
            'jnd': jnd,
            'sigma': sigma,
            'mu': pse,
            'n_trials': ntrials,
            'accuracy': np.mean([r['user_ans'] for r in rows]),
            'status': 'completed',
            'model': '1model_abs',
        }
        return rows, result_dict

    def _simulate_1model_rel(
        self,
        pse: float,
        sigma: float,
        jnd: float,
        ntrials: int,
        ado_params: Dict,
        bis_params: Dict,
        fixed_trials_config: Dict,
    ) -> Tuple[List[Dict], Dict]:
        """Simulate 1-model relative approach."""

        exp = BISRelADOpyWrapper(ado_params, bis_params)

        fixed_config = fixed_trials_config or {'use': False}
        trial_sequence = create_trial_sequence_relative(
            ntrials,
            fixed_config.get('offsets', []),
            self.offset,
            fixed_config.get('n_fixed', 0),
            fixed_config.get('use', False),
        )

        rows = []
        for trial_id, (stim_info, pre_post, trial_type) in enumerate(trial_sequence):
            is_pre = (pre_post == 'pre')
            if trial_type == 'fixed':
                stim_ms = stim_info
            else:
                stim_q = exp.get(is_pre)
                stim_ms = self.offset - stim_q if is_pre else self.offset + stim_q

            user_ans = generate_response(stim_ms, pse, sigma)
            success = int((stim_ms > self.offset) == user_ans)
            exp.set(success, user_ans, abs(stim_ms - self.offset), stim_ms)

            rows.append({
                'lat': int(stim_ms),
                'res': str(success == 1).lower(),
                'user_ans': user_ans,
            })

        result_dict = {
            'subj': f'SIM_{pse:.0f}_{jnd:.0f}',
            'pse': pse,
            'jnd': jnd,
            'sigma': sigma,
            'mu': pse,
            'n_trials': ntrials,
            'accuracy': np.mean([r['user_ans'] for r in rows]),
            'status': 'completed',
            'model': '1model_rel',
        }
        return rows, result_dict

    def _simulate_2model_rel(
        self,
        pse: float,
        sigma: float,
        jnd: float,
        ntrials: int,
        ado_params: Dict,
        bis_params: Dict,
        fixed_trials_config: Dict,
    ) -> Tuple[List[Dict], Dict]:
        """Simulate 2-model relative approach."""

        exp_pre = BISRelADOpyWrapper(adoparams=ado_params, taskparams=bis_params)
        exp_post = BISRelADOpyWrapper(adoparams=ado_params, taskparams=bis_params)

        fixed_config = fixed_trials_config or {'use': False}
        trial_sequence = create_trial_sequence_relative(
            ntrials,
            fixed_config.get('offsets', []),
            self.offset,
            fixed_config.get('n_fixed', 0),
            fixed_config.get('use', False),
        )

        rows = []
        for trial_id, (stim_info, pre_post, trial_type) in enumerate(trial_sequence):
            is_pre = (pre_post == 'pre')
            exp = exp_pre if is_pre else exp_post

            if trial_type == 'fixed':
                stim_ms = stim_info
                magnitude = abs(stim_ms - self.offset)
            else:
                magnitude = exp.get(is_pre)
                stim_ms = self.offset - magnitude if is_pre else self.offset + magnitude

            user_ans = generate_response(stim_ms, pse, sigma)
            success = int((stim_ms > self.offset) == user_ans)
            exp.set(success, user_ans, magnitude, stim_ms)

            rows.append({
                'lat': int(stim_ms),
                'res': str(success == 1).lower(),
                'user_ans': user_ans,
                'model': 'pre' if is_pre else 'post',
            })

        result_dict = {
            'subj': f'SIM_{pse:.0f}_{jnd:.0f}',
            'pse': pse,
            'jnd': jnd,
            'sigma': sigma,
            'mu': pse,
            'n_trials': ntrials,
            'accuracy': np.mean([r['user_ans'] for r in rows]),
            'status': 'completed',
            'model': '2model_rel',
        }
        return rows, result_dict

    @staticmethod
    def save_gbf_file(rows: List[Dict], output_path: str) -> None:
        """
        Save trial rows to GBF format file (no header, tab-delimited).

        Args:
            rows: List of trial dicts with 'lat', 'count', 'user_ans', 'confl_magn'
            output_path: Path to output file
        """
        df = pd.DataFrame(rows)
        df.to_csv(output_path, sep='\t', index=False, header=False, float_format='%g')
