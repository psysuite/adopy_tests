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
            model_type: 'ABS1', 'REL1', or 'REL2'
            offset: Reference latency (default 500ms)
        """
        self.model_type = model_type
        self.offset = offset
        self._validate_model_type()

    def _validate_model_type(self):
        """Validate model type."""
        valid_models = ['ABS1', 'REL1', 'REL2']
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
        subject_id: int = None,
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
            subject_id: Optional subject identifier for result_dict

        Returns:
            Tuple of (rows, result_dict)
            - rows: List of trial dicts for plotting
            - result_dict: Statistics dict
        """
        sigma = get_sigma_from_jnd(jnd)

        if self.model_type == 'ABS1':
            return self._simulate_ABS1(
                pse, sigma, jnd, ntrials, ado_params, bis_params, fixed_trials_config, subject_id
            )
        elif self.model_type == 'REL1':
            return self._simulate_REL1(
                pse, sigma, jnd, ntrials, ado_params, bis_params, fixed_trials_config, subject_id
            )
        elif self.model_type == 'REL2':
            return self._simulate_REL2(
                pse, sigma, jnd, ntrials, ado_params, bis_params, fixed_trials_config, subject_id
            )
        raise Exception(f"simulate_subject: model {self.model_type} not recognized")

    def _simulate_ABS1(
        self,
        pse: float,
        sigma: float,
        jnd: float,
        ntrials: int,
        ado_params: Dict,
        bis_params: Dict,
        fixed_trials_config: Dict,
        subject_id: int = None,
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

        # Always use subject_id for naming (never fall back to PSE/JND to avoid collisions)
        if subject_id is None:
            raise ValueError("subject_id must be provided for simulation")
        
        subj_str = f'SIM_{subject_id}'

        result_dict = {
            'subj': subj_str,
            'pse': pse,
            'jnd': jnd,
            'sigma': sigma,
            'mu': pse,
            'n_trials': ntrials,
            'accuracy': np.mean([r['user_ans'] for r in rows]),
            'model': 'ABS1',
        }
        
        return rows, result_dict

    def _simulate_REL1(
        self,
        pse: float,
        sigma: float,
        jnd: float,
        ntrials: int,
        ado_params: Dict,
        bis_params: Dict,
        fixed_trials_config: Dict,
        subject_id: int = None,
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

        # Always use subject_id for naming (never fall back to PSE/JND to avoid collisions)
        if subject_id is None:
            raise ValueError("subject_id must be provided for simulation")
        
        subj_str = f'SIM_{subject_id}'

        result_dict = {
            'subj': subj_str,
            'pse': pse,
            'jnd': jnd,
            'sigma': sigma,
            'mu': pse,
            'n_trials': ntrials,
            'accuracy': np.mean([r['user_ans'] for r in rows]),
            'model': 'REL1',
        }
        
        return rows, result_dict

    def _simulate_REL2(
        self,
        pse: float,
        sigma: float,
        jnd: float,
        ntrials: int,
        ado_params: Dict,
        bis_params: Dict,
        fixed_trials_config: Dict,
        subject_id: int = None,
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

        # Always use subject_id for naming (never fall back to PSE/JND to avoid collisions)
        if subject_id is None:
            raise ValueError("subject_id must be provided for simulation")
        
        subj_str = f'SIM_{subject_id}'

        result_dict = {
            'subj': subj_str,
            'pse': pse,
            'jnd': jnd,
            'sigma': sigma,
            'mu': pse,
            'n_trials': ntrials,
            'accuracy': np.mean([r['user_ans'] for r in rows]),
            'model': 'REL2',
        }
        
        return rows, result_dict

