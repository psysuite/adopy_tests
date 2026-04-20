# Temporal Bisection Experiment - Implementation Guide

## Overview

This guide describes the implementation of temporal bisection experiments using three different ADOpy models. The codebase includes 9 main experiment files organized by model type and execution mode.

---

## The Three Models

### 1. 1-Model Absolute (1model_abs)

**Characteristics:**
- Uses a single ADOpy engine with absolute stimulus values
- Stimulus range: 200-800ms (centered around 500ms offset)
- Guess rate: 0.04 (low, since absolute comparison is easier)
- Lapse rate: 0.04
- Noise: 10%

**How it works:**
- ADOpy directly suggests stimulus times in milliseconds (e.g., 350ms, 650ms)
- User responds: "Is the second tone closer to the first (1) or third (2)?"
- Response is binary: 0 (closer to first) or 1 (closer to third)
- ADOpy updates based on the response and absolute stimulus value

**Best for:**
- Simple bisection tasks
- When stimuli can vary widely across the full range
- Experiments where absolute timing is important

---

### 2. 1-Model Relative (1model_rel)

**Characteristics:**
- Uses a single ADOpy engine with relative stimulus values (magnitudes)
- Magnitude range: 5-300ms (distance from offset)
- Guess rate: 0.5 (higher, since relative comparison is harder)
- Lapse rate: 0.04
- Noise: 10%
- Block randomization: 4 pre + 4 post every 8 trials

**How it works:**
- ADOpy suggests a magnitude (distance from offset)
- Randomly selects pre or post (in blocks of 8 trials)
- Converts magnitude to absolute stimulus:
  - Pre: `stimulus = offset - magnitude` (e.g., 500 - 50 = 450ms)
  - Post: `stimulus = offset + magnitude` (e.g., 500 + 50 = 550ms)
- User responds: 0 or 1
- ADOpy updates based on success (correct/incorrect) and magnitude

**Best for:**
- Balanced pre/post stimulus presentation
- When you want to focus on discrimination around the offset
- Experiments requiring equal sampling of both sides

---

### 3. 2-Model Relative (2model_rel)

**Characteristics:**
- Uses TWO separate ADOpy engines (one for pre, one for post)
- Magnitude range: 0-300ms for each model
- Guess rate: 0.5 for both
- Lapse rate: 0.04
- Noise: 10%
- Block randomization: 4 pre + 4 post every 8 trials

**How it works:**
- Maintains two independent ADOpy engines: `exp_pre` and `exp_post`
- Each trial randomly selects which model to use (block randomized)
- Selected model suggests a magnitude
- Converts to absolute stimulus (same as 1model_rel)
- User responds: 0 or 1
- Only the selected model updates based on success
- At the end, data from both models is combined for analysis

**Best for:**
- When pre and post stimuli might have different psychometric properties
- More sophisticated adaptive sampling
- Research comparing pre vs post discrimination

---

## File Organization

The 9 main experiment files are organized in a 3×3 matrix:

|                | Console Input | Real Audio | Group Simulation |
|----------------|---------------|------------|------------------|
| 1model_abs     | ✓             | ✓          | ✓                |
| 1model_rel     | ✓             | ✓          | ✓                |
| 2model_rel     | ✓             | ✓          | ✓                |

### File Naming Convention

- `subj_console_bis_ado_*` - Console-based experiments (keyboard input)
- `subj_real_bis_ado_*` - Real audio experiments (plays tones)
- `group_sim_bis_ado_*` - Group simulations (20 subjects, synthetic data)

---

## File Details

### Console Files (3 files)

**Location:** `main/subj_console_bis_ado_*.py`

**Purpose:** Interactive experiments with console input (no audio)

**Input:**
- User keyboard input (1 or 2) for each trial
- Number of trials (default: 60)

**Output:**
- Console display of trial information
- Statistics printed to console
- Individual psychometric plot: `{subject_id}_psychometric.png`
- Results saved to: `../data/output/console/`

**Execution:**
```bash
python main/subj_console_bis_ado_1model_abs.py
python main/subj_console_bis_ado_1model_rel.py
python main/subj_console_bis_ado_2model_rel.py
```

**Output format:**
```
============================================================
TEMPORAL BISECTION EXPERIMENT - 1MODEL_ABS
============================================================
TRIAL 1/60
First tone: 500ms | Second tone: XXXms | Third tone: 500ms
Is the second tone closer to the first (1) or the third (2)? 
```

---

### Real Audio Files (3 files)

**Location:** `main/subj_real_bis_ado_*.py`

**Purpose:** Experiments with actual audio playback

**Input:**
- User keyboard input (1 or 2) after hearing tones
- Number of trials (default: 60)
- Audio parameters:
  - Frequency: 1000 Hz
  - Amplitude: 1.0
  - Fade duration: 10ms
  - Sample rate: 44100 Hz

**Output:**
- Console display of trial information
- Audio playback of three tones
- Statistics printed to console
- Individual psychometric plot: `{subject_id}_psychometric.png`
- Results saved to: `../data/output/real/`

**Execution:**
```bash
python main/subj_real_bis_ado_1model_abs.py
python main/subj_real_bis_ado_1model_rel.py
python main/subj_real_bis_ado_2model_rel.py
```

**Audio implementation:**
- Uses `utilities/real_exp_accessories.py`
- Pre-generates tones with WAV caching
- Includes fade in/out to eliminate clicks
- 50ms padding at start/end of buffer

---

### Group Simulation Files (3 files)

**Location:** `main/group_sim_bis_ado_*.py`

**Purpose:** Simulate experiments for 20 subjects with synthetic responses

**Input:**
- Number of subjects: 20 (hardcoded)
- Number of trials per subject: 60
- Random PSE range: [485, 515] ms
- Random JND range: [20, 60] ms
- Random seed: 42 (for reproducibility)

**Output:**

1. **Excel file:** `{PREFIX}_results_summary.xlsx`
   - One row per subject with fitted parameters
   - Two summary rows: GROUP_mean, GROUP_std
   - Columns: subj, pse, jnd, fitted_pse, fitted_jnd, r_squared, etc.

2. **Group plots:**
   - `{PREFIX}_group_histogram.png` - Success/failure distribution (all 3 files)
   - `{PREFIX}_group_model_histogram.png` - Pre/post distribution (2model_rel only)
   - `{PREFIX}_group_psychometric.png` - Group psychometric curve (all 3 files)

3. **Individual plots:** (one per subject)
   - `{subject_id}_psychometric.png`

4. **Log file:** `sim_*_model_*.log`

**File prefixes:**
- 1model_abs: `SIM1ABS`
- 1model_rel: `SIM1REL`
- 2model_rel: `SIM2REL`

**Execution:**
```bash
python main/group_sim_bis_ado_1model_abs.py
python main/group_sim_bis_ado_1model_rel.py
python main/group_sim_bis_ado_2model_rel.py
```

**Output directory:** `../data/output/sim/`

**Response generation:**
- Uses psychophysical model: `generate_response(stimulus, pse, sigma)`
- Internal perception: `N(stimulus, sigma)`
- Response: 1 if perception > PSE, else 0
- Sigma derived from JND: `sigma = JND / 0.6745`

---

## Common Features Across All Files

### Uniform Output Structure

All 9 files share the same output format:

1. **Header:** 60 `=` characters with experiment title
2. **Trial display:** `TRIAL X/N` format
3. **Input prompt:** "Is the second tone closer to the first (1) or the third (2)?"
4. **Footer:** "EXPERIMENT COMPLETE"
5. **Statistics:** Via `print_statistics()` function
6. **Fitted parameters:** PSE and JND from Gaussian fit
7. **Psychometric plot:** Individual curve with data points

### Statistics Reported

- Total trials
- Offset (true threshold)
- Trials below/above offset (count and percentage)
- Stimulus range (min, max, mean, median, std)
- Response accuracy (if applicable)

### Psychometric Analysis

All files use:
- **Fitting method:** Cumulative Gaussian (`scipy.stats.norm.cdf`)
- **Bin size:** 10ms
- **Parameters extracted:** μ (PSE) and σ (related to JND)
- **Plot elements:**
  - Red dots: Binned data points
  - Blue line: Fitted curve
  - Green dashed line: True threshold (500ms)
  - Gray dotted line: 0.5 probability

---

## Dependencies

### Core Libraries
- `numpy` - Numerical operations
- `pandas` - Data handling
- `matplotlib` - Plotting
- `scipy` - Statistical functions and curve fitting
- `adopy` - Adaptive experimental design

### Custom Modules
- `bisection/BISAbsADOpyWrapper.py` - Absolute model wrapper
- `bisection/BISRelADOpyWrapper.py` - Relative model wrapper
- `bisection/psychometric_analysis.py` - Analysis functions
- `bisection/logging_config.py` - Logging setup
- `utilities/misc_generate_responses.py` - Response generation
- `utilities/plotting.py` - Plotting functions
- `utilities/real_exp_accessories.py` - Audio generation

---

## Workflow Comparison

### Console Workflow
1. Initialize ADOpy engine(s)
2. For each trial:
   - Get stimulus from ADOpy
   - Display trial info
   - Get user input
   - Update ADOpy
3. Analyze results
4. Generate plots
5. Save outputs

### Real Audio Workflow
1. Initialize ADOpy engine(s)
2. For each trial:
   - Get stimulus from ADOpy
   - Generate and play audio tones
   - Get user input
   - Update ADOpy
3. Analyze results
4. Generate plots
5. Save outputs

### Group Simulation Workflow
1. Initialize logger and output directory
2. For each subject (20 total):
   - Generate random PSE and JND
   - Initialize ADOpy engine(s)
   - For each trial (60 total):
     - Get stimulus from ADOpy
     - Generate synthetic response
     - Update ADOpy
     - Collect trial data
   - Analyze subject results
   - Generate individual plot
3. Generate group plots (histograms + psychometric)
4. Consolidate results to Excel
5. Add group statistics (mean, std)
6. Print summary report

---

## Key Differences Between Models

| Feature | 1model_abs | 1model_rel | 2model_rel |
|---------|------------|------------|------------|
| Engines | 1 | 1 | 2 (pre + post) |
| Stimulus space | Absolute (200-800ms) | Relative (5-300ms) | Relative (0-300ms each) |
| Guess rate | 0.04 | 0.5 | 0.5 |
| Pre/post balance | Natural | Block randomized | Block randomized |
| Update method | Response (0/1) | Success (correct/incorrect) | Success per model |

---

## Output File Summary

### Console & Real Audio
- Individual psychometric plots
- Console statistics
- Saved to `../data/output/console/` or `../data/output/real/`

### Group Simulation
- Excel: `{PREFIX}_results_summary.xlsx` (20 subjects + 2 summary rows)
- Plots:
  - `{PREFIX}_group_histogram.png` (all)
  - `{PREFIX}_group_model_histogram.png` (2model_rel only)
  - `{PREFIX}_group_psychometric.png` (all)
  - Individual: `S001_psychometric.png` ... `S020_psychometric.png`
- Log: `sim_*_model_*.log`
- Saved to `../data/output/sim/`

---

## Tips for Usage

1. **Start with console files** to understand the experiment flow
2. **Use real audio files** for actual data collection
3. **Use group simulation files** to:
   - Test analysis pipelines
   - Generate synthetic datasets
   - Validate experimental parameters
   - Compare model performance

4. **Choose the right model:**


---

*Last updated: 2026-04-16*


---

## 📚 Additional Documentation

For more detailed information, see the `docs/` folder:

- **[Quick Start Guide](docs/QUICK_START.txt)** - Visual quick reference
- **[Setup Complete](docs/SETUP_COMPLETE.md)** - Setup verification and next steps
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Complete project layout
- **[Build Artifacts](docs/BUILD_ARTIFACTS.md)** - Understanding the `build/` folder
- **[Test Summary](docs/TEST_SUMMARY.md)** - Test results and coverage report

## 🧪 Testing

```bash
# Run all tests
make test

# Run tests with coverage report
make coverage

# Clean build artifacts
make clean
```

Test coverage: **76%** (39 tests passing)

Coverage report: `build/htmlcov/index.html`
