"""
Shared configuration for synthetic data generation pipeline.

Parameters used by:
- generate_synthetic_data.py (Phase 1-4 orchestrator)
- group_sim_gridrnd_ABS1.py (Phase 1-2 data generation)
- group_sim_gridrnd_REL1.py (Phase 1-2 data generation)
- group_sim_gridrnd_REL2.py (Phase 1-2 data generation)
"""

# Grid parameters for PSE and JND
PSE_GRID = [480, 500, 520]
JND_GRID = [20, 40, 60]

# Psychophysical parameters
OFFSET = 500  # Reference latency in milliseconds

# Simulation parameters
N_SUBJECTS_PER_GROUP = 1  # Number of subjects per PSE/JND group
N_TRIALS = 200  # Number of trials per subject

# Output paths (relative to project root)
OUTPUT_BASE = "data/output/sim_gridrnd"
EXCEL_OUTPUT_DIR = "../R/indata"

MAX_WORKERS             = 10
SAVE_GBF_FILES          = True

USE_FIXED_TRIALS        = False

VARIATION               = 2.5
N_FIXED                 = 10
FIXED_LATENCIES_ABS     = [275, 725, 325, 675, 375, 625, 425, 575, 475, 525]
FIXED_LATENCIES_REL     = [225, 175, 125, 75, 25]


ADO_PARAMS_REL          = {"guess_rate": 0.5, "lapse_rate": 0.04, "noise_perc": 0.1}
BIS_PARAMS_REL          = {"min": 1, "max": 300, "offset": OFFSET, "ntrials": N_TRIALS}

ADO_PARAMS_ABS          = {"guess_rate": 0.04, "lapse_rate": 0.04, "noise_perc": 0.05}
BIS_PARAMS_ABS          = {"min": 200, "max": 800, "offset": OFFSET, "ntrials": N_TRIALS}
