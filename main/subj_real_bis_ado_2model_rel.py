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
nTrials = 56
offset = 500
nAdaptive = 36  # Adaptive trials (used only if USE_FIXED_TRIALS=True)
nFixed = 10  # Fixed trials (repeated twice: once at start, once mixed) - 5 pre + 5 post each time

# Fixed relative offsets (5 values, will be applied as ±offset)
FIXED_OFFSETS_REL = [225, 175, 125, 75, 25]  # Creates 10 latencies: 5 pre + 5 post

# Initialize two models: pre and post
ado_params_pre = {"guess_rate": 0.5, "lapse_rate": 0.04, "noise_perc": 0.1}
bis_params_pre = {"min": 0, "max": 300, "offset": offset, "ntrials": int(nTrials / 2)}
exp_pre = qw.BISRelADOpyWrapper(adoparams=ado_params_pre, taskparams=bis_params_pre)

ado_params_post = {"guess_rate": 0.5, "lapse_rate": 0.04, "noise_perc": 0.1}
bis_params_post = {"min": 0, "max": 300, "offset": offset, "ntrials": int(nTrials / 2)}
exp_post = qw.BISRelADOpyWrapper(adoparams=ado_params_post, taskparams=bis_params_post)

# Create trial sequence
trial_sequence = create_trial_sequence_relative(nTrials, FIXED_OFFSETS_REL, offset, nFixed, USE_FIXED_TRIALS)

stimuli_ms = []
successes = []
models_used = []
user_responses = []

print("=" * 60)
print("BISECTION TEST - REAL AUDIO (2-MODEL RELATIVE)")
print("=" * 60)
if USE_FIXED_TRIALS:
    print(f"Total trials: {nTrials} ({nFixed} fixed at start [5 pre + 5 post], {nAdaptive} adaptive, {nFixed} fixed mixed [5 pre + 5 post])")
else:
    print(f"Total trials: {nTrials} (all adaptive)")
print(f"Tone duration: {TONE_DURATION} ms")
print(f"Total trial duration: {TOTAL_DURATION} ms")
print("=" * 60)

for i, (stim_info, pre_post, trial_type) in enumerate(trial_sequence):
    # Select model based on pre/post
    use_pre = (pre_post == 'pre')
    exp = exp_pre if use_pre else exp_post
    model_name = 'pre' if use_pre else 'post'
    
    if trial_type == 'fixed':
        onset_ms = stim_info
        magnitude = abs(onset_ms - offset)
        print(f"\n--- TRIAL {i + 1}/{nTrials} [FIXED-{pre_post.upper()}] ---")
    else:
        # Get magnitude from ADOpy
        magnitude = round(exp.get(), 1)
        
        # Convert magnitude to absolute stimulus time
        if use_pre:
            onset_ms = offset - magnitude
        else:
            onset_ms = offset + magnitude
        onset_ms = round(onset_ms, 1)
        print(f"\n--- TRIAL {i + 1}/{nTrials} [ADAPTIVE-{pre_post.upper()}] ---")
    
    print(f"Model: {model_name}, Onset: {onset_ms:.1f} ms (magnitude: {magnitude:.1f})")
    
    # Run bisection trial with audio
    user_ans = run_bisection_trial(onset_ms, TONE_DURATION, TOTAL_DURATION, SAMPLE_RATE, TONE_FREQUENCY)
    
    # Determine correctness
    # For relative bisection: if onset_ms > offset, it's "later" (answer should be 2)
    # if onset_ms < offset, it's "earlier" (answer should be 0)
    cor_ans = int(onset_ms > offset)
    success = cor_ans == user_ans

    stimuli_ms.append(onset_ms)
    successes.append(success)
    models_used.append(model_name)
    user_responses.append(user_ans)
    
    # Update ADOpy with success
    exp.set(success, user_ans, magnitude, onset_ms)

    result = "CORRECT" if success else "INCORRECT"
    print(f"Response: {user_ans}, Result: {result}")

# Combine all trial data into a single wrapper for unified fit
exp_combined = qw.BISRelADOpyWrapper(adoparams=ado_params_pre, taskparams=bis_params_pre)
exp_combined.stimuli_ms = stimuli_ms
exp_combined.responses = user_responses
exp_combined.offset = offset
exp_combined.range = exp_pre.range + exp_post.range

# Print statistics
print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)
exp_combined.print_statistics()

# Fit and display results
mu, sigma = exp_combined.gausFit(10)
print(f"\nFitted parameters: MU = {mu:.2f} ms, SIGMA = {sigma:.2f} ms")

# Plot psychometric function
plot_file_name = (
    "../data/output/plots/subj_2model_rel_real_ntr" + str(nTrials) +
    "_guess" + str(ado_params_pre["guess_rate"]) +
    "_lapse" + str(ado_params_pre["lapse_rate"]) + ".png"
)
exp_combined.plot_psychometric(plot_file_name, 10)
print(f"Plot saved to: {plot_file_name}")
print("=" * 60)
