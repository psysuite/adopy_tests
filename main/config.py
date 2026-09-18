"""
Shared configuration for synthetic data generation pipeline.

Parameters used by:
- generate_synthetic_data.py (Phase 1-4 orchestrator)
- group_sim_gridrnd_ABS1.py (Phase 1-2 data generation)
- group_sim_gridrnd_REL1.py (Phase 1-2 data generation)
- group_sim_gridrnd_REL2.py (Phase 1-2 data generation)
"""

MODELS = ['ABS1', 'REL1', 'REL2']
# MODELS = ['REL1']


# Psychophysical parameters
OFFSET = 500  # Reference latency in milliseconds

# Trial blocks for progressive analysis (metrics at each checkpoint)
TRIAL_BLOCKS = [40, 60, 80, 100, 120, 140, 160, 180, 200]

# Simulation parameters
PSE_GRID = [480, 500, 520]
JND_GRID = [20, 40, 60]

# Per-subject variation (jitter) applied within each group
# CRITICAL: This is how subjects are distinguished within a group!
# 
# Each group has a center (pse_center, jnd_center). For each subject in that group,
# we apply a random jitter ± VARIATION to create unique (pse_true, jnd_true) per subject.
# 
# Example:
#   - Group center: (pse_center=480, jnd_center=20)
#   - Subject S01 in group: pse_true = 480 + jitter1 ≈ 478.3, jnd_true = 20 + jitter1' ≈ 22.1
#   - Subject S02 in group: pse_true = 480 + jitter2 ≈ 481.9, jnd_true = 20 + jitter2' ≈ 18.7
#   - Subject S01 in DIFFERENT group G2 (pse_center=500, jnd_center=40):
#     pse_true ≈ 500.5, jnd_true ≈ 39.8  ← COMPLETELY DIFFERENT from S01 in G1!
#
# THEREFORE: Incremental processing key MUST include group: (subject_id, model, group)
# WITHOUT group in the key, S01+ABS1 in G1 would be confused with S01+ABS1 in G2!
VARIATION               = 2.5

N_SUBJECTS_PER_GROUP = 20  # Number of subjects per PSE/JND group
N_TRIALS = 200  # Number of trials per subject

# ============================================================================
# GROUP ORGANIZATION (for plotting and analysis)
# ============================================================================
#
# PSE_GRID = [480, 500, 520] (3 levels)
# JND_GRID = [20, 40, 60]     (3 levels)
# → 9 groups total (G1-G9)
#
# Group layout in PSE×JND grid:
#
#          JND=20  JND=40  JND=60
#  PSE=480  G1      G2      G3
#  PSE=500  G4      G5      G6
#  PSE=520  G7      G8      G9
#
# JND_GROUPS: Pool groups by JND level (collapse/ignore PSE dimension)
#   - JND=20: [G1, G4, G7]
#   - JND=40: [G2, G5, G8]
#   - JND=60: [G3, G6, G9]
JND_GROUPS = [
    (['G1', 'G4', 'G7'], 20),   # JND=20 (pools across all PSE)
    (['G2', 'G5', 'G8'], 40),   # JND=40 (pools across all PSE)
    (['G3', 'G6', 'G9'], 60),   # JND=60 (pools across all PSE)
]

# PSE_GROUPS: Pool groups by PSE level (collapse/ignore JND dimension)
#   - PSE=480: [G1, G2, G3]
#   - PSE=500: [G4, G5, G6]
#   - PSE=520: [G7, G8, G9]
PSE_GROUPS = [
    (['G1', 'G2', 'G3'], 480),  # PSE=480 (pools across all JND)
    (['G4', 'G5', 'G6'], 500),  # PSE=500 (pools across all JND)
    (['G7', 'G8', 'G9'], 520),  # PSE=520 (pools across all JND)
]

# JND_GROUPS_AT500: JND levels at middle PSE (PSE=500 only, vary JND)
#   - No pooling: each group is isolated
#   - Used for studying JND effect without PSE confound
JND_GROUPS_AT500 = [
    ('G4', 20),  # PSE=500, JND=20
    ('G5', 40),  # PSE=500, JND=40
    ('G6', 60),  # PSE=500, JND=60
]

OUTPUT_BASE = "data/output/sim_gridrnd"
EXCEL_OUTPUT_DIR = "R/indata"
MAX_WORKERS             = 9
SAVE_GBF_FILES          = True
USE_FIXED_TRIALS        = False
FIXED_LATENCIES_ABS     = [275, 725, 325, 675, 375, 625, 425, 575, 475, 525]
FIXED_LATENCIES_REL     = [225, 175, 125, 75, 25]


# ADOPY MODELS
ADO_PARAMS_REL          = {"guess_rate": 0.5, "lapse_rate": 0.04, "noise_perc": 0.1}
BIS_PARAMS_REL          = {"min": 1, "max": 300, "offset": OFFSET, "ntrials": N_TRIALS}

ADO_PARAMS_ABS          = {"guess_rate": 0.04, "lapse_rate": 0.04, "noise_perc": 0.05}
BIS_PARAMS_ABS          = {"min": 200, "max": 800, "offset": OFFSET, "ntrials": N_TRIALS}
