from bisection import BISRelADOpyWrapper as qw
import numpy as np
import random

# Configuration
USE_FIXED_TRIALS = True  # Set to False to use only adaptive trials

nTrials = 56
offset = 500
nAdaptive = 36  # Adaptive trials (used only if USE_FIXED_TRIALS=True)
nFixed = 10  # Fixed trials (repeated twice: once at start, once mixed) - 5 pre + 5 post each time

# Fixed relative offsets (5 values, will be applied as ±offset)
FIXED_OFFSETS_REL = [25, 75, 125, 175, 225]  # Creates 10 latencies: 5 pre + 5 post

# Initialize two models: pre and post
ado_params_pre = {"guess_rate": 0.5, "lapse_rate": 0.04, "noise_perc": 0.1}
bis_params_pre = {"min": 0, "max": 300, "offset": offset, "ntrials": int(nTrials/2)}
exp_pre = qw.BISRelADOpyWrapper(adoparams=ado_params_pre, taskparams=bis_params_pre)

ado_params_post = {"guess_rate": 0.5, "lapse_rate": 0.04, "noise_perc": 0.1}
bis_params_post = {"min": 0, "max": 300, "offset": offset, "ntrials": int(nTrials/2)}
exp_post = qw.BISRelADOpyWrapper(adoparams=ado_params_post, taskparams=bis_params_post)

plot_file_name = "../data/output/plots/subj_2model_rel_ntr" + str(nTrials) + "_guess" + str(ado_params_pre["guess_rate"]) + "_lapse" + str(ado_params_pre["lapse_rate"]) + ".png"

print("="*60)
print("BISECTION TEST - 2 MODEL RELATIVE")
print("="*60)
print(f"Test parameters: guess_rate={ado_params_pre['guess_rate']}, lapse_rate={ado_params_pre['lapse_rate']}")
if USE_FIXED_TRIALS:
    print(f"Total trials: {nTrials} ({nFixed} fixed at start [5 pre + 5 post], {nAdaptive} adaptive, {nFixed} fixed mixed [5 pre + 5 post])")
else:
    print(f"Total trials: {nTrials} (all adaptive)")
print("="*60)

# Create trial sequence
if USE_FIXED_TRIALS:
    # First 10 trials: 5 pre + 5 post in order
    trial_sequence = []
    for offset_val in FIXED_OFFSETS_REL:
        trial_sequence.append((offset - offset_val, 'pre', 'fixed'))  # pre
        trial_sequence.append((offset + offset_val, 'post', 'fixed'))  # post
    
    # Remaining trials: adaptive + 10 fixed (5 pre + 5 post) mixed randomly
    # For adaptive trials, balance pre/post
    n_adaptive_pre = nAdaptive // 2
    n_adaptive_post = nAdaptive - n_adaptive_pre
    
    remaining_trials = []
    remaining_trials.extend([('adaptive', 'pre', 'adaptive')] * n_adaptive_pre)
    remaining_trials.extend([('adaptive', 'post', 'adaptive')] * n_adaptive_post)
    for offset_val in FIXED_OFFSETS_REL:
        remaining_trials.append((offset - offset_val, 'pre', 'fixed'))
        remaining_trials.append((offset + offset_val, 'post', 'fixed'))
    random.shuffle(remaining_trials)
    
    trial_sequence.extend(remaining_trials)
else:
    # All trials are adaptive with block randomization
    block_dim = 10
    trial_order = []
    for block in range(int(nTrials) // block_dim):
        block_trials = ['pre'] * int(block_dim/2) + ['post'] * int(block_dim/2)
        np.random.shuffle(block_trials)
        trial_order.extend(block_trials)
    
    # Handle remaining trials
    remaining = int(nTrials) % block_dim
    if remaining > 0:
        remaining_trials = ['pre'] * min(int(block_dim/2), remaining) + ['post'] * max(0, remaining - int(block_dim/2))
        np.random.shuffle(remaining_trials)
        trial_order.extend(remaining_trials)
    
    trial_sequence = [('adaptive', trial_order[i], 'adaptive') for i in range(nTrials)]

stimuli_ms = []
successes = []
models_used = []
user_responses = []

for i, (stim_info, pre_post, trial_type) in enumerate(trial_sequence):
    # Select model based on pre/post
    use_pre = (pre_post == 'pre')
    exp = exp_pre if use_pre else exp_post
    model_name = 'pre' if use_pre else 'post'
    
    if trial_type == 'fixed':
        stim_ms = stim_info
        magnitude = abs(stim_ms - offset)
        print(f"\n--- TRIAL {i + 1}/{nTrials} [FIXED-{pre_post.upper()}] ---")
    else:
        # Get magnitude from ADOpy
        magnitude = round(exp.get(), 1)
        
        # Convert magnitude to absolute stimulus time
        if use_pre:
            stim_ms = offset - magnitude
        else:
            stim_ms = offset + magnitude
        stim_ms = round(stim_ms, 1)
        print(f"\n--- TRIAL {i + 1}/{nTrials} [ADAPTIVE-{pre_post.upper()}] ---")
    
    print(f"Model: {model_name}, Stimulus: {stim_ms:.1f} ms (magnitude: {magnitude:.1f})")
    
    while True:
        print("Is the second tone closer to the first (1) or the third (2)? ", end="", flush=True)
        user_ans = input()
        if user_ans in ("1", "2"):
            user_ans = int(user_ans) - 1
            break
        else:
            print("Invalid input. Please enter 1 or 2.")

    # Determine correctness
    cor_ans = int(stim_ms > offset)
    success = cor_ans == user_ans

    stimuli_ms.append(stim_ms)
    successes.append(success)
    models_used.append(model_name)
    user_responses.append(user_ans)
    
    # Update ADOpy with success
    exp.set(success, user_ans)
    
    result = "CORRECT" if success else "INCORRECT"
    print(f"Response recorded: {user_ans}, Result: {result}")

# Combine all trial data into a single wrapper for unified fit
exp_combined = qw.BISRelADOpyWrapper(adoparams=ado_params_pre, taskparams=bis_params_pre)
exp_combined.stimuli_ms = stimuli_ms
exp_combined.responses = user_responses
exp_combined.offset = offset
exp_combined.range = exp_pre.range + exp_post.range

# Print statistics
print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)
exp_combined.print_statistics()

# Fit and display results
mu, sigma = exp_combined.gausFit(10)
print(f"\nFitted parameters: MU = {mu:.2f} ms, SIGMA = {sigma:.2f} ms")

# Plot psychometric function
exp_combined.plot_psychometric(plot_file_name, 10)
print(f"Plot saved to: {plot_file_name}")
print("="*60)
