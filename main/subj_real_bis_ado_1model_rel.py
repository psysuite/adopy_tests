from bisection import BISRelADOpyWrapper as qw
from utilities.real_exp_accessories import run_bisection_trial
import numpy as np

# Audio parameters
SAMPLE_RATE = 44100
TONE_FREQUENCY = 1000  # Hz
TONE_DURATION = 100  # ms - duration of each tone
TOTAL_DURATION = 1000  # ms - total trial duration

# Experiment setup
offset = 500
nTrials = 56

ado_params = {"guess_rate": 0.5, "lapse_rate": 0.04, "noise_perc": 0.1}
bis_params = {"min": 5, "max": 300, "offset": offset, "ntrials": nTrials, "is_absolute": False}
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
print(f"Total trials: {nTrials}")
print(f"Tone duration: {TONE_DURATION} ms")
print("=" * 60)

# Create trial order: every 8 trials, randomize 4 pre and 4 post
trial_order = []
for block in range(int(nTrials) // 8):
    block_trials = ['pre'] * 4 + ['post'] * 4
    np.random.shuffle(block_trials)
    trial_order.extend(block_trials)

# Handle remaining trials if nTrials is not divisible by 8
remaining = int(nTrials) % 8
if remaining > 0:
    remaining_trials = ['pre'] * min(4, remaining) + ['post'] * max(0, remaining - 4)
    np.random.shuffle(remaining_trials)
    trial_order.extend(remaining_trials)

for i in range(nTrials):
    # Select pre/post based on randomized trial order
    is_pre = (trial_order[i] == 'pre')
    
    stim_q = exp.get(is_pre)
    stim_ms = offset - stim_q if is_pre is True else offset + stim_q

    print(f"\n--- TRIAL {i + 1}/{nTrials} ---")
    print(f"Stimulus: {stim_ms:.1f} ms (magnitude: {stim_q:.1f}, {'pre' if is_pre else 'post'})")
    
    # Run bisection trial with audio
    user_ans = run_bisection_trial(stim_ms, TONE_DURATION, TOTAL_DURATION, SAMPLE_RATE, TONE_FREQUENCY)

    success = int(user_ans == int(stim_ms > offset))
    exp.set(success, user_ans, q_value=stim_q)
    
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
