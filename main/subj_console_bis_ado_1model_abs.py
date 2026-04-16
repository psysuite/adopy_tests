from bisection import BISAbsADOpyWrapper as qw


offset      = 500
nTrials     = 48

ado_params  = {"guess_rate":0.04, "lapse_rate":0.04, "noise_perc":0.1}
bis_params  = {"min":200, "max":800, "offset":offset, "ntrials":nTrials, "is_absolute":True}
exp = qw.BISAbsADOpyWrapper(adoparams=ado_params, taskparams=bis_params)

max_slope = exp.params["slope"][len(exp.params["slope"])-1]
plot_file_name = "../data/output/plots/subj_1model_abs_ntr" + str(nTrials) + "_guess" + str(ado_params["guess_rate"]) + "_lapse" + str(ado_params["lapse_rate"]) + "_slopemax" + str(max_slope)+ ".png"

print("="*60)
print("BISECTION TEST - 1 MODEL ABSOLUTE")
print("="*60)
print(f"Test parameters: guess_rate={ado_params['guess_rate']}, lapse_rate={ado_params['lapse_rate']}")
print(f"Slope range: [{exp.params['slope'][0]}, {max_slope}]")
print(f"Threshold range: [{exp.params['threshold'][0]}, {exp.params['threshold'][len(exp.params['slope'])-1]}]")
print(f"Total trials: {nTrials}")
print("="*60)

for i in range(nTrials):
    stim_ms = exp.get(exclude_zone=False)

    print(f"\n--- TRIAL {i + 1}/{nTrials} ---")
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
