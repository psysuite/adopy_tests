from bisection import BISAbsADOpyWrapper as qw
from utilities.real_exp_accessories import run_bisection_trial
from utilities.trial_sequence import create_trial_sequence_absolute
import random


# Configuration
USE_FIXED_TRIALS = True  # Set to False to use only adaptive trials
USE_REL_LOGIC = False  # Set to True to use pre/post block randomization like rel models

# Audio parameters
SAMPLE_RATE = 44100
TONE_FREQUENCY = 1000  # Hz
TONE_DURATION = 50  # ms - duration of each tone
TOTAL_DURATION = 1000  # ms - total trial duration

# Experiment setup
offset = 500
nTrials = 60  # Total trials
nAdaptive = 40  # Adaptive trials (used only if USE_FIXED_TRIALS=True and USE_REL_LOGIC=False)
nFixed = 10  # Fixed trials (repeated twice: once at start, once mixed)

# Fixed latencies to present at the beginning and repeat during the task
FIXED_LATENCIES = [275, 725, 325, 675, 375, 625, 575, 425, 575, 575, 525]

ado_params = {"guess_rate": 0.04, "lapse_rate": 0.04, "noise_perc": 0.05}
bis_params = {"min": 200, "max": 800, "offset": offset, "ntrials": nTrials}
exp = qw.BISAbsADOpyWrapper(adoparams=ado_params, taskparams=bis_params)

max_slope = exp.params["slope"][len(exp.params["slope"]) - 1]
plot_file_name = (
    "../data/output/plots/subj_1model_abs_real_ntr" + str(nTrials) +
    "_guess" + str(ado_params["guess_rate"]) +
    "_lapse" + str(ado_params["lapse_rate"]) +
    "_slopemax" + str(max_slope) + ".png"
)

print("=" * 60)
print("BISECTION TEST - REAL AUDIO")
print("=" * 60)
print(f"Test parameters: guess_rate={ado_params['guess_rate']}, lapse_rate={ado_params['lapse_rate']}")
print(f"Slope range: [{exp.params['slope'][0]}, {max_slope}]")
print(f"Threshold range: [{exp.params['threshold'][0]}, {exp.params['threshold'][len(exp.params['slope']) - 1]}]")
if USE_FIXED_TRIALS:
    print(f"Total trials: {nTrials} ({nFixed} fixed at start, {nAdaptive} adaptive, {nFixed} fixed mixed)")
else:
    print(f"Total trials: {nTrials} (all adaptive)")
print("=" * 60)

# Create trial sequence
trial_sequence = create_trial_sequence_absolute(nTrials, FIXED_LATENCIES, nFixed, USE_FIXED_TRIALS)

# Run trials
for i, (trial_info, trial_type) in enumerate(trial_sequence):
    if trial_type == 'fixed':
        stim_ms = trial_info
        print(f"\n--- TRIAL {i + 1}/{nTrials} [FIXED] ---")
    else:
        stim_ms = exp.get(exclude_zone=True)
        print(f"\n--- TRIAL {i + 1}/{nTrials} [ADAPTIVE] ---")
    
    user_ans = run_bisection_trial(stim_ms, TONE_DURATION, TOTAL_DURATION, SAMPLE_RATE, TONE_FREQUENCY)
    
    # Record response
    exp.set(user_ans, stim_ms)
    print(f"Response recorded: {user_ans}")

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
