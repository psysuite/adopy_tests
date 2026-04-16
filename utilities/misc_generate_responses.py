import numpy as np
import pandas as pd


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


def get_sigma_from_jnd(jnd):
    """Convert JND (75%-50% distance) to sigma (standard deviation)."""
    if pd.isna(jnd) or jnd == 0:
        return None
    return jnd / 0.6745


def generate_response(stim_ms, pse, sigma):
    """
    Generate a response using psychophysical model.

    Internal perception = stimulus + random_noise
    where random_noise ~ N(0, sigma)
    Response is 1 if perception > PSE, else 0.
    """
    if pse is None or sigma is None or pd.isna(pse) or pd.isna(sigma):
        # Fallback: random response
        return np.random.randint(0, 2)

    # Internal perception: stimulus + Gaussian noise
    # This is equivalent to N(stim_ms, sigma) but shows the additive nature
    internal = stim_ms + np.random.normal(0, sigma)
    response = int(internal > pse)
    return response


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
