from bisection import BISRelADOpyWrapper as qw
import numpy as np

nTrials = 56
offset = 500

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
print(f"Total trials: {nTrials}")
print("="*60)

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

stimuli_ms = []
successes = []
models_used = []
user_responses = []

for i in range(nTrials):
    # Select model based on randomized trial order
    use_pre = (trial_order[i] == 'pre')
    exp = exp_pre if use_pre else exp_post
    model_name = 'pre' if use_pre else 'post'
    
    # Get magnitude from ADOpy
    magnitude = round(exp.get(), 1)
    
    # Convert magnitude to absolute stimulus time
    if use_pre:
        stim_ms = offset - magnitude
    else:
        stim_ms = offset + magnitude
    stim_ms = round(stim_ms, 1)

    print(f"\n--- TRIAL {i + 1}/{nTrials} ---")
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
