"""
Trial sequence generators for temporal bisection experiments.
Reusable across console, real-audio, and simulation scripts.
"""

import random
import numpy as np
from typing import List, Tuple


def create_trial_sequence_absolute(
    ntrials: int,
    fixed_latencies: List[float],
    n_fixed: int = 10,
    use_fixed_trials: bool = True,
) -> List[Tuple]:
    """
    Build trial sequence for 1-model absolute bisection.

    Returns list of (stim_info, trial_type) where:
      - trial_type == 'fixed'    -> stim_info is the latency (ms)
      - trial_type == 'adaptive' -> stim_info == 'adaptive'
    """
    if not use_fixed_trials:
        return [('adaptive', 'adaptive')] * ntrials

    n_adaptive = ntrials - 2 * n_fixed
    sequence = [(lat, 'fixed') for lat in fixed_latencies[:n_fixed]]

    remaining = [('adaptive', 'adaptive')] * n_adaptive
    remaining += [(lat, 'fixed') for lat in fixed_latencies[:n_fixed]]
    random.shuffle(remaining)

    return sequence + remaining


def create_trial_sequence_relative(
    ntrials: int,
    fixed_offsets: List[float],
    offset: float = 500.0,
    n_fixed: int = 10,
    use_fixed_trials: bool = True,
    block_dim: int = 10,
) -> List[Tuple]:
    """
    Build trial sequence for 1-model or 2-model relative bisection.

    Returns list of (stim_info, pre_post, trial_type) where:
      - trial_type == 'fixed'    -> stim_info is the absolute latency (ms)
      - trial_type == 'adaptive' -> stim_info == 'adaptive'
      - pre_post is 'pre' or 'post'
    """
    if use_fixed_trials:
        n_adaptive = ntrials - 2 * n_fixed

        # First n_fixed trials: alternating pre/post in order
        sequence = []
        for val in fixed_offsets:
            sequence.append((offset - val, 'pre', 'fixed'))
            sequence.append((offset + val, 'post', 'fixed'))

        n_adaptive_pre = n_adaptive // 2
        n_adaptive_post = n_adaptive - n_adaptive_pre

        remaining = (
            [('adaptive', 'pre', 'adaptive')] * n_adaptive_pre +
            [('adaptive', 'post', 'adaptive')] * n_adaptive_post
        )
        for val in fixed_offsets:
            remaining.append((offset - val, 'pre', 'fixed'))
            remaining.append((offset + val, 'post', 'fixed'))
        random.shuffle(remaining)

        return sequence + remaining

    else:
        # All adaptive with block randomization
        trial_order = []
        for _ in range(ntrials // block_dim):
            block = ['pre'] * (block_dim // 2) + ['post'] * (block_dim // 2)
            np.random.shuffle(block)
            trial_order.extend(block)

        remaining = ntrials % block_dim
        if remaining > 0:
            tail = ['pre'] * min(block_dim // 2, remaining) + ['post'] * max(0, remaining - block_dim // 2)
            np.random.shuffle(tail)
            trial_order.extend(tail)

        return [('adaptive', trial_order[i], 'adaptive') for i in range(ntrials)]
