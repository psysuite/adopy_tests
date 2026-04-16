from bisection import BISAbsADOpyWrapper as qw
from utilities.real_exp_accessories import run_bisection_trial



# Audio parameters
SAMPLE_RATE = 44100
TONE_FREQUENCY = 1000  # Hz
TONE_DURATION = 50  # ms - duration of each tone
TOTAL_DURATION = 1000  # ms - total trial duration

# Experiment setup
offset = 500
nTrials = 50

ado_params = {"guess_rate": 0.04, "lapse_rate": 0.04, "noise_perc": 0.1}
bis_params = {"min": 150, "max": 850, "offset": offset, "ntrials": nTrials, "is_absolute": True}
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
print(f"Total trials: {nTrials}")
print("=" * 60)

# Run trials
for i in range(nTrials):
    onset_ms = exp.get(exclude_zone=False)
    
    print(f"\n--- TRIAL {i + 1}/{nTrials} ---")
    user_ans = run_bisection_trial(onset_ms, TONE_DURATION, TOTAL_DURATION, SAMPLE_RATE, TONE_FREQUENCY)
    
    # Record response
    exp.set(user_ans, onset_ms)
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
