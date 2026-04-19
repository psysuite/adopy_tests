from bisection import BISAbsADOpyWrapper as qw
from utilities.trial_sequence import create_trial_sequence_absolute
import random


# Configuration
USE_FIXED_TRIALS = True  # Set to False to use only adaptive trials
USE_REL_LOGIC = False  # Set to True to use pre/post block randomization like rel models

offset      = 500
nTrials     = 60  # Total trials
nAdaptive   = 40  # Adaptive trials (used only if USE_FIXED_TRIALS=True and USE_REL_LOGIC=False)
nFixed      = 10  # Fixed trials (repeated twice: once at start, once mixed) - 5 pre + 5 post each time

# Fixed latencies for absolute mode (10 values around offset)
FIXED_LATENCIES_ABS = [275, 725, 325, 675, 375, 625, 425, 575, 475, 525]

# Fixed relative offsets for rel-style mode (5 values, will be applied as ±offset)
FIXED_OFFSETS_REL = [25, 75, 125, 175, 225]  # Creates 10 latencies: 5 pre + 5 post

ado_params  = {"guess_rate":0.04, "lapse_rate":0.04, "noise_perc":0.1}
bis_params  = {"min":200, "max":800, "offset":offset, "ntrials":nTrials}
exp = qw.BISAbsADOpyWrapper(adoparams=ado_params, taskparams=bis_params)

max_slope = exp.params["slope"][len(exp.params["slope"])-1]
plot_file_name = "../data/output/plots/subj_1model_abs_ntr" + str(nTrials) + "_guess" + str(ado_params["guess_rate"]) + "_lapse" + str(ado_params["lapse_rate"]) + "_slopemax" + str(max_slope)+ ".png"

print("="*60)
print("BISECTION TEST - 1 MODEL ABSOLUTE")
print("="*60)
print(f"Test parameters: guess_rate={ado_params['guess_rate']}, lapse_rate={ado_params['lapse_rate']}")
print(f"Slope range: [{exp.params['slope'][0]}, {max_slope}]")
print(f"Threshold range: [{exp.params['threshold'][0]}, {exp.params['threshold'][len(exp.params['slope'])-1]}]")
if USE_FIXED_TRIALS and USE_REL_LOGIC:
    print(f"Total trials: {nTrials} ({nFixed} fixed at start [5 pre + 5 post], {nAdaptive} adaptive, {nFixed} fixed mixed [5 pre + 5 post])")
elif USE_FIXED_TRIALS:
    print(f"Total trials: {nTrials} ({nFixed} fixed at start, {nAdaptive} adaptive, {nFixed} fixed mixed)")
else:
    print(f"Total trials: {nTrials} (all adaptive)")
print("="*60)

# Create trial sequence
trial_sequence = create_trial_sequence_absolute(nTrials, FIXED_LATENCIES_ABS, nFixed, USE_FIXED_TRIALS)

# Run trials
for i, (trial_info, trial_type) in enumerate(trial_sequence):
    if trial_type in ('fixed', 'fixed_pre', 'fixed_post'):
        stim_ms = trial_info
        label = 'FIXED-PRE' if trial_type == 'fixed_pre' else 'FIXED-POST' if trial_type == 'fixed_post' else 'FIXED'
        print(f"\n--- TRIAL {i + 1}/{nTrials} [{label}] ---")
    else:
        stim_ms = exp.get(exclude_zone=True)
        print(f"\n--- TRIAL {i + 1}/{nTrials} [ADAPTIVE] ---")
    
    print(f"Stimulus: {stim_ms:.1f} ms")
    
    while True:
        print("Is the second tone closer to the first (1) or the third (2)? ", end="", flush=True)
        user_ans = input()
        if user_ans in ("1","2"):
            user_ans = int(user_ans) - 1
            break
        else:
            print("Invalid input. Please enter 1 or 2.")

    exp.set(user_ans, stim_ms)
    print(f"Response recorded: {user_ans}")

# Print statistics
print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)
exp.print_statistics()

# Fit and display results
mu, sigma = exp.gausFit(10)
print(f"\nFitted parameters: MU = {mu:.2f} ms, SIGMA = {sigma:.2f} ms")

# Plot psychometric function
exp.plot_psychometric(plot_file_name, 10)
print(f"Plot saved to: {plot_file_name}")
print("="*60)
