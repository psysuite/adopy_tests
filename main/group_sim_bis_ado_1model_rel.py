#!/usr/bin/env python3
"""
Simulate temporal bisection experiment with 20 subjects, 60 trials each.
Uses 1-model relative ADOpy approach with random PSE/JND values.

PSE: random uniform [485, 515]
JND: random uniform [20, 60]
"""

import os
import sys
import numpy as np
import pandas as pd
import random

from utilities.misc_generate_responses import generate_response, get_sigma_from_jnd
from utilities.plotting import plot_group_histograms, plot_group_psychometric
from utilities.trial_sequence import create_trial_sequence_relative

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bisection import BISRelADOpyWrapper as qw
from utilities.psychometric_analysis import safe_analyze_subject, consolidate_results, print_summary_report, add_group_stats_to_excel
from utilities.logging_config import setup_logger


# Configuration
USE_FIXED_TRIALS = True  # Set to False to use only adaptive trials

# Fixed trial parameters
N_FIXED = 10  # Fixed trials (repeated twice: once at start, once mixed) - 5 pre + 5 post each time
FIXED_OFFSETS_REL = [25, 75, 125, 175, 225]  # Creates 10 latencies: 5 pre + 5 post


def simulate_subject(subj_id, pse, jnd, ntrials, output_dir, file_prefix, logger):
    """Simulate experiment for a single subject with fixed PSE/JND."""
    
    subj = f"S{subj_id:03d}"

    # Convert JND to sigma
    sigma = get_sigma_from_jnd(jnd)
    
    # Initialize ADOpy wrapper
    offset = 500

    ado_params = {"guess_rate": 0.5, "lapse_rate": 0.04, "noise_perc": 0.1}
    bis_params = {"min": 5, "max": 300, "offset": offset, "ntrials": ntrials, "is_absolute": False}
    exp = qw.BISRelADOpyWrapper(ado_params, bis_params)

    # Collect trial data for group plots
    rows = []
    
    # Create trial sequence
    trial_sequence = create_trial_sequence_relative(
        ntrials, FIXED_OFFSETS_REL, offset, N_FIXED, USE_FIXED_TRIALS
    )
    
    # Generate trials
    for trial_id, (stim_info, pre_post, trial_type) in enumerate(trial_sequence):
        is_pre = (pre_post == 'pre')
        
        if trial_type == 'fixed':
            stim_ms = stim_info
            stim_q = abs(stim_ms - offset)
        else:
            stim_q = exp.get(is_pre)
            stim_ms = offset - stim_q if is_pre else offset + stim_q
        
        # Generate response using psychophysical model
        user_ans = generate_response(stim_ms, pse, sigma)
        
        # Determine correctness
        success = int(user_ans == int(stim_ms > offset))
        
        # Build row for plotting
        row = {
            'lat': int(stim_ms),
            'res': str(success == 1).lower(),
            'user_ans': user_ans
        }
        rows.append(row)
        
        # Update ADOpy with success
        exp.set(success, user_ans, stim_q, stim_ms)
    
    # Create subject_row for analysis
    subject_row = pd.Series({
        'subj': subj,
        'pse': pse,
        'jnd': jnd
    })
    
    # Analyze results
    result_dict = safe_analyze_subject(exp, subject_row, output_dir, file_prefix, logger)
    
    # Get experiment statistics
    exp_stats = exp.print_statistics()
    result_dict.update(exp_stats)
    
    return rows, result_dict


def main():
    file_prefix = "SIM1REL"
    n_subjects = 20
    nTrials = 60

    # Output directory
    output_dir = "../data/output/sim"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logger
    log_file = os.path.join(output_dir, "sim_1model_rel.log")
    logger = setup_logger("sim_1model_rel", log_file=log_file)
    
    print(f"Simulating {n_subjects} subjects with {nTrials} trials each...")
    print(f"PSE range: [485, 515], JND range: [20, 60]")
    print(f"Output directory: {output_dir}\n")
    
    # Collect all results and rows for group plots
    all_results = []
    all_rows_list = []
    offset = 500
    
    # Generate random PSE/JND for each subject
    np.random.seed(42)  # For reproducibility
    
    for subj_id in range(1, n_subjects + 1):
        # Random PSE and JND
        pse = np.random.uniform(485, 515)
        jnd = np.random.uniform(20, 60)
        
        print(f"Subject {subj_id}: PSE={pse:.1f}, JND={jnd:.1f}")
        
        try:
            rows, result_dict = simulate_subject(subj_id, pse, jnd, nTrials, output_dir, file_prefix, logger)
            all_results.append(result_dict)
            all_rows_list.append(rows)
            
        except Exception as e:
            print(f"Error simulating subject {subj_id}: {e}")
    
    # Generate group histograms
    if all_rows_list:
        plot_group_histograms(all_rows_list, output_dir, file_prefix, offset)
        plot_group_psychometric(all_rows_list, output_dir, file_prefix, offset)
    
    # Consolidate results
    if all_results:
        try:
            excel_filepath = consolidate_results(all_results, output_dir, file_prefix)
            add_group_stats_to_excel(excel_filepath)
            print(f"\nResults saved to: {excel_filepath}")
        except Exception as e:
            print(f"Error consolidating results: {e}")
    
    # Print summary
    print_summary_report(all_results)
    
    print(f"\nDone! Simulated {n_subjects} subjects.")


if __name__ == "__main__":
    main()
