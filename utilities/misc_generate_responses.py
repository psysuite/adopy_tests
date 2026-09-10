import numpy as np
import pandas as pd
from scipy.special import expit


def get_trial_params(trial_num, pse_dict, jnd_dict):
    """Get PSE and JND for current trial based on trial count milestones."""
    milestones = [40, 60, 80, 100, 120, 140, 160, 180, 200]

    # Find which milestone we're in
    for milestone in milestones:
        if trial_num < milestone:
            key = f"pse_{milestone}"
            pse = pse_dict.get(key)
            jnd = jnd_dict.get(f"jnd_{milestone}")
            sigma = get_sigma_from_jnd(jnd)
            return pse, sigma

    # Default to last milestone
    key = "pse_200"
    pse = pse_dict.get(key)
    jnd = jnd_dict.get("jnd_200")
    sigma = get_sigma_from_jnd(jnd)
    return pse, sigma


def get_jnd_from_sigma(sigma):
    """Convert logistic scale parameter to JND (semi-IQR).

    For a logistic distribution with scale parameter σ:
    JND = ln(3) * σ ≈ 1.0986 * σ

    This is the centralized function for JND <-> sigma conversion.
    Use this everywhere to ensure consistency.

    Args:
        sigma: Scale parameter of logistic distribution (float)

    Returns:
        JND (semi-IQR of logistic distribution) in same units as sigma
    """
    if sigma is None or pd.isna(sigma) or sigma == 0:
        return None
    return np.log(3) * sigma


def get_sigma_from_jnd(jnd):
    """Convert JND (semi-IQR of logistic distribution) to scale parameter.

    For a logistic distribution, semi-IQR = ln(3) * σ
    If JND = semi-IQR, then σ = JND / ln(3)

    This is the centralized function for JND <-> sigma conversion.
    Use this everywhere to ensure consistency.

    Args:
        jnd: JND value (semi-IQR of logistic distribution)

    Returns:
        Scale parameter σ of logistic distribution
    """
    if pd.isna(jnd) or jnd == 0:
        return None
    return jnd / np.log(3)


def generate_response(stim_ms, pse, sigma):
    """
    Generate a response using logistic psychophysical model.

    Internal perception = stimulus + random_noise
    where random_noise ~ Logistic(0, sigma)
    Response is 1 if perception > PSE, else 0.

    sigma here is the scale parameter of the logistic distribution.
    For logistic(0, scale), semi-IQR = ln(3) * scale
    """
    if pse is None or sigma is None or pd.isna(pse) or pd.isna(sigma):
        # Fallback: random response
        return np.random.randint(0, 2)

    # Internal perception: stimulus + logistic noise
    internal = stim_ms + np.random.logistic(0, sigma)
    response = int(internal > pse)
    return response


def generate_response_lapse_guess(stim_ms, pse_ms, jnd_ms, guess_rate=0.04, lapse_rate=0.04, rng=None):
    """
    Generate the raw bisection response.

    Coding:
        0 = S2 judged closer to S1
        1 = S2 judged closer to S3

    JND is the semi-IQR of the latent logistic distribution:
        JND = ln(3) / beta
        sigma = 1 / beta = JND / ln(3)
    """
    if rng is None:
        rng = np.random.default_rng()

    if jnd_ms <= 0:
        raise ValueError("JND must be > 0.")

    beta = np.log(3) / jnd_ms

    # Latent logistic choice probability
    p_latent = expit(beta * (stim_ms - pse_ms))

    # Observed response probability with lower/upper asymptotes
    p_response_1 = (guess_rate + (1.0 - guess_rate - lapse_rate) * p_latent)

    return int(rng.random() < p_response_1)


def write_subject_file(rows, filepath):
    # Write to file
    with open(filepath, 'w') as f:
        # Write header
        header = ['id', 'label', 'lat', 'confl', 'res', 'cor_ans', 'user_ans', 'elapsed', 'rep', 'confl_magn']
        f.write('\t'.join(header) + '\n')

        # Write rows
        for row in rows:
            values = [str(row[col]) for col in header]
            f.write('\t'.join(values) + '\n')
