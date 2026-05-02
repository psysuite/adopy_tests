from bisection import BISRelADOpyWrapper as qw
from utilities.real_exp_accessories import run_bisection_trial
from utilities.trial_sequence import create_trial_sequence_relative
import numpy as np
import random

# Configuration
USE_FIXED_TRIALS = True  # Set to False to use only adaptive trials

# Audio parameters
SAMPLE_RATE = 44100
TONE_FREQUENCY = 1000  # Hz
TONE_DURATION = 100  # ms - duration of each tone
TOTAL_DURATION = 1000  # ms - total trial duration

# Experiment setup
offset = 500
nTrials = 56
nAdaptive = 36  # Adaptive trials (used only if USE_FIXED_TRIALS=True)
nFixed = 10  # Fixed trials (repeated twice: once at start, once mixed) - 5 pre + 5 post each time

# Fixed relative offsets (5 values, will be applied as ±offset)
FIXED_OFFSETS_REL = [25, 75, 125, 175, 225]  # Creates 10 latencies: 5 pre + 5 post

ado_params = {"guess_rate": 0.5, "lapse_rate": 0.04, "noise_perc": 0.1}
bis_params = {"min": 1, "max": 300, "offset": offset, "ntrials": nTrials}
exp = qw.BISRelADOpyWrapper(ado_params, bis_params)

max_slope = exp.params["slope"][len(exp.params["slope"]) - 1]
plot_file_name = (
    "../data/output/plots/subj_1model_rel_real_ntr" + str(nTrials) +
    "_guess" + str(ado_params["guess_rate"]) +
    "_lapse" + str(ado_params["lapse_rate"]) +
    "_slopemax" + str(max_slope) + ".png"
)

print("=" * 60)
print("BISECTION TEST - 1 MODEL RELATIVE (REAL AUDIO)")
print("=" * 60)
print(f"Test parameters: guess_rate={ado_params['guess_rate']}, lapse_rate={ado_params['lapse_rate']}")
print(f"Slope range: [{exp.params['slope'][0]}, {max_slope}]")
print(f"Threshold range: [{exp.params['threshold'][0]}, {exp.params['threshold'][len(exp.params['slope']) - 1]}]")
if USE_FIXED_TRIALS:
    print(f"Total trials: {nTrials} ({nFixed} fixed at start [5 pre + 5 post], {nAdaptive} adaptive, {nFixed} fixed mixed [5 pre + 5 post])")
else:
    print(f"Total trials: {nTrials} (all adaptive)")
print(f"Tone duration: {TONE_DURATION} ms")
print("=" * 60)

# Create trial sequence
trial_sequence = create_trial_sequence_relative(nTrials, FIXED_OFFSETS_REL, offset, nFixed, USE_FIXED_TRIALS)

for i, (stim_info, pre_post, trial_type) in enumerate(trial_sequence):
    is_pre = (pre_post == 'pre')
    
    if trial_type == 'fixed':
        stim_ms = stim_info
        stim_q = abs(stim_ms - offset)
        print(f"\n--- TRIAL {i + 1}/{nTrials} [FIXED-{pre_post.upper()}] ---")
    else:
        stim_q = exp.get(is_pre)
        stim_ms = offset - stim_q if is_pre else offset + stim_q
        print(f"\n--- TRIAL {i + 1}/{nTrials} [ADAPTIVE-{pre_post.upper()}] ---")
    
    print(f"Stimulus: {stim_ms:.1f} ms (magnitude: {stim_q:.1f}, {'pre' if is_pre else 'post'})")
    
    # Run bisection trial with audio
    user_ans = run_bisection_trial(stim_ms, TONE_DURATION, TOTAL_DURATION, SAMPLE_RATE, TONE_FREQUENCY)

    success = int(user_ans == int(stim_ms > offset))
    exp.set(success, user_ans, stim_q, stim_ms)
    
    result = "CORRECT" if success else "INCORRECT"
    print(f"Response recorded: {user_ans}, Result: {result}")

# Print statistics
print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)
exp.print_statistics()

# Fit and display results
mu, sigma = exp.gausFit(10)
print(f"\nFitted parameters: MU = {mu:.2f} ms, SIGMA = {sigma:.2f} ms")

# Plot psychometric function
exp.plot_psychometric(plot_file_name, 10)
print(f"Plot saved to: {plot_file_name}")
print("=" * 60)
