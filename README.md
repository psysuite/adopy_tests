# Temporal Bisection Experiment - Implementation Guide

## Overview

This guide describes the implementation of temporal bisection experiments using three different ADOpy models. The codebase includes 6 main experiment files (console and real audio) plus a comprehensive group simulation pipeline with multithreading support and advanced psychometric analysis.

---

## The Three Models

### 1. ABS1 (Absolute Single-Engine Model)

**Characteristics:**
- Uses a single ADOpy engine with absolute stimulus values
- Stimulus range: 200-800ms (centered around 500ms offset)
- Guess rate: 0.04 (low, since absolute comparison is easier)
- Lapse rate: 0.04
- Noise: 5%

**How it works:**
- ADOpy directly suggests stimulus times in milliseconds (e.g., 350ms, 650ms)
- User responds: "Is the second tone closer to the first (1) or third (2)?"
- Response is binary: 0 (closer to first) or 1 (closer to third)
- ADOpy updates based on the response and absolute stimulus value

---

### 2. REL1 (Relative Single-Engine Model)

**Characteristics:**
- Uses a single ADOpy engine with relative stimulus values (magnitudes)
- Magnitude range: 5-300ms (distance from offset)
- Guess rate: 0.5 (higher, since relative comparison is harder)
- Lapse rate: 0.04
- Noise: 10%
- Block randomization: 5 pre + 5 post every 10 trials

**How it works:**
- ADOpy suggests a magnitude (distance from offset)
- Randomly selects pre or post (in blocks of 10 trials)
- Converts magnitude to absolute stimulus:
  - Pre: `stimulus = offset - magnitude` (e.g., 500 - 50 = 450ms)
  - Post: `stimulus = offset + magnitude` (e.g., 500 + 50 = 550ms)
- User responds: 0 or 1
- ADOpy updates based on success (correct/incorrect) and magnitude

---

### 3. REL2 (Relative Dual-Engine Model)

**Characteristics:**
- Uses TWO separate ADOpy engines (one for pre, one for post)
- Magnitude range: 0-300ms for each model
- Guess rate: 0.5 for both
- Lapse rate: 0.04
- Noise: 10%
- Block randomization: 5 pre + 5 post every 10 trials

**How it works:**
- Maintains two independent ADOpy engines: `exp_pre` and `exp_post`
- Each trial randomly selects which model to use (block randomized)
- Selected model suggests a magnitude
- Converts to absolute stimulus (same as REL1)
- User responds: 0 or 1
- Only the selected model updates based on success
- At the end, data from both models is combined for analysis

**This is the Best model:**
- assumes that pre and post stimuli might have different psychometric properties
- More sophisticated adaptive sampling
- Research comparing pre vs post discrimination

---

## File Organization

### Experiment Files (6 files)

The 6 main experiment files are organized in a 3×2 matrix:

|                | Console Input | Real Audio |
|----------------|---------------|------------|
| ABS1           | ✓             | ✓          |
| REL1           | ✓             | ✓          |
| REL2           | ✓             | ✓          |

**File Naming Convention:**
- `subj_console_bis_ado_*` - Console-based experiments (keyboard input)
- `subj_real_bis_ado_*` - Real audio experiments (plays tones)

---

## Experiment Files Details

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
python main/subj_console_bis_ado_ABS1.py
python main/subj_console_bis_ado_REL1.py
python main/subj_console_bis_ado_REL2.py
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
python main/subj_real_bis_ado_ABS1.py
python main/subj_real_bis_ado_REL1.py
python main/subj_real_bis_ado_REL2.py
```

---

## Group Simulation Pipeline

### Overview

The group simulation pipeline generates synthetic data for multiple subjects across a grid of PSE/JND parameters. It features:

- **Multithreading:** Parallel subject simulation and progressive analysis
- **Progressive Analysis:** Psychometric metrics calculated at trial intervals (40, 60, 80, ..., 200)
- **Advanced Metrics:** Asymmetry index, stimulus distribution analysis, bimodality detection
- **Grid Organization:** Variable grid size (default: 9 groups with PSE: 480/500/520 × JND: 20/40/60, configurable)
- **Batch Processing:** 20 subjects per group, 200 trials per subject

### Files (3 files)

**Location:** `main/group_sim_gridrnd_*.py`

**Purpose:** Simulate experiments on a PSE/JND grid with multithreading

**Input:**
- PSE grid: [480, 500, 520] (configurable)
- JND grid: [20, 40, 60] (configurable)
- Subjects per group: 20 (configurable)
- Trials per subject: 200 (configurable)
- Multithreading: MAX_WORKERS (default: 3-4 threads, configurable)

**Output (per model):**
1. **GBF files:** `SXX_GZ_PSE_JND_MODEL.txt` (N files, where N = grid_size × subjects_per_group)
   - Filename format embeds jittered PSE/JND values for skip mode loading
2. **Group plots:** Histograms, psychometric curves, model histograms (REL2 only)
3. **Grid plots:** Combined grids for all groups (size depends on PSE/JND grid dimensions)
4. **Excel:** `{model}_G{group}_results_summary.xlsx` with progressive metrics
5. **Analysis plots:** 5 metric evolution plots (asymmetry, stimulus center/spread, bimodality)

**Execution:**
```bash
python main/group_sim_gridrnd_ABS1.py
python main/group_sim_gridrnd_REL1.py
python main/group_sim_gridrnd_REL2.py
```

**Output directory:** `../data/output/sim_gridrnd/{model}/`

### Skip Mode

If `SKIP_PHASES_1_2=True` in the script:
- Phases 1-2 (task creation and simulation) are skipped
- GBF files are loaded from disk
- PSE/JND values are extracted from filename using `parse_gbf_filename()`
- If filename parsing fails (old format), falls back to progressive fit values with warning
- Phases 3-6 (analysis and plotting) proceed normally

This allows re-running analysis on existing simulations without re-generating GBF files.

### Architecture: 5-Phase Pipeline

**Phase 1: Task Creation**
- Create SubjectSimulationTask objects for all subjects in group
- Each task contains PSE/JND parameters with ±2.5 random variation

**Phase 2: Parallel Subject Simulation**
- Run all subject simulations in parallel (one thread per subject)
- Each thread runs 200 trials with ADOpy adaptive sampling
- Generates GBF files and trial data

**Phase 3: Parallel Progressive Analysis**
- Run progressive psychometric analysis in parallel
- Calculate PSE/JND at trial counts: 40, 60, 80, 100, 120, 140, 160, 180, 200
- Calculate asymmetry index at same trial counts
- Calculate stimulus distribution metrics (center, spread, bimodality)

**Phase 4: Sequential Plot & Excel Generation**
- Generate group plots (histograms, psychometric curves)
- Consolidate results to Excel with all metrics
- Add GROUP_mean and GROUP_std rows

**Phase 5: Grid Plot Creation**
- Combine group plots into grids (size depends on PSE/JND grid dimensions)
- One grid per plot type (histogram, psychometric, model_histogram)
- Labels show PSE/JND values for each subplot

**Phase 6: Analysis Metric Plots**
- Generate 5 analysis plots from all group Excel files:
  1. Asymmetry index evolution (modulo)
  2. Asymmetry scatter + envelope
  3. Stimulus center evolution
  4. Stimulus spread evolution
  5. Bimodality index evolution
- Plots aggregate data across all groups and subjects

### Multithreading Configuration

**MAX_WORKERS parameter:**
- Controls number of parallel threads for subject simulation and analysis
- Default: 3-4 (configurable per model)
- Recommendation: Set to number of CPU cores for optimal performance
- User manages manually (no automatic CPU detection)

---

## Advanced Metrics

### 1. Asymmetry Index

**Definition:** `(n_after - n_before) / total`

**Range:** [-1, 1]

**Interpretation:**
- `0`: Perfectly balanced (50-50 split)
- `> 0`: Bias toward stimuli after offset (right side)
- `< 0`: Bias toward stimuli before offset (left side)
- `±1`: All stimuli on one side

**Use case:** Detect if ADOpy is biased in stimulus selection (especially 1model_abs)

**Progressive calculation:** Computed at trial counts 40, 60, 80, ..., 200 to track convergence

---

### 2. Stimulus Distribution Metrics

**stimulus_center:** Mean of stimulus latencies
- Should correlate with PSE parameter
- Tracks how well the presented stimuli match intended PSE

**stimulus_spread:** Standard deviation of stimulus latencies
- Should correlate with JND parameter
- Tracks how well the presented stimuli match intended JND

**stimulus_min / stimulus_max:** Range of stimulus latencies
- Validates that stimuli stay within expected bounds

**bimodality_index:** Measure of distribution shape
- Range: [0, 1]
- `0`: Unimodal (single peak)
- `1`: Perfectly bimodal (two equal peaks)
- Detects if stimuli cluster around two distinct values

**Progressive calculation:** All metrics computed at trial counts 40, 60, 80, ..., 200

---

### 3. Psychometric Metrics (Progressive)

**PSE (Point of Subjective Equality):** Fitted mean of psychometric function
- Calculated at each trial count to track convergence
- Columns: `pse_40`, `pse_60`, ..., `pse_200`

**JND (Just Noticeable Difference):** Fitted standard deviation × 0.6745
- Calculated at each trial count to track convergence
- Columns: `jnd_40`, `jnd_60`, ..., `jnd_200`

---

## Excel Output Format

### Columns (per subject row)

**Metadata:**
- `subj`: Subject identifier
- `pse`: True PSE parameter
- `jnd`: True JND parameter
- `n_trials`: Total trials completed

**Psychometric metrics (progressive):**
- `pse_40`, `pse_60`, ..., `pse_200`: PSE at each trial count
- `jnd_40`, `jnd_60`, ..., `jnd_200`: JND at each trial count

**Asymmetry metrics (progressive):**
- `asymmetry_40`, `asymmetry_60`, ..., `asymmetry_200`: Asymmetry index at each trial count

**Stimulus distribution metrics (progressive):**
- `stimulus_center_40`, `stimulus_center_60`, ..., `stimulus_center_200`: Mean stimulus latency
- `stimulus_spread_40`, `stimulus_spread_60`, ..., `stimulus_spread_200`: Std of stimulus latency
- `bimodality_index_40`, `bimodality_index_60`, ..., `bimodality_index_200`: Bimodality measure

**Summary rows:**
- `GROUP_mean`: Mean of all metrics across subjects
- `GROUP_std`: Standard deviation of all metrics across subjects

---

## Data Analysis & Publication Workflow

### Overview

The project includes comprehensive data analysis pipelines for both simulated and real experimental data, with R-based statistical analysis and publication-ready figure generation.

### Python Analysis Scripts

**Simulation Data Processing:**
- `main/regenerate_simulation_data.py` - Regenerate all analysis from GBF files (skip mode)
- `main/regenerate_simulation_plots.py` - Regenerate plots from existing Excel files
- `main/progr_analyze_folder.py` - Progressive analysis on a folder of GBF files

**Real Data Analysis:**
- `main/group_realdata_analysis.py` - Analyze real experimental data with progressive metrics

**Publication Figures:**
- `main/create_paper_plots.py` - Generate publication-ready figures
  - `create_grid_psychometric(output_filename)` - 3×3 grid of psychometric functions
  - `create_latency_envelope_grid(output_filename)` - 3×3 grid of latency distributions
  - `create_latency_distribution_grid(input_dir, output_filename)` - 2×2 real data distributions
  - `create_figure7_combined(input_dir, excel_path, output_filename)` - Combined latency + entropy figure

### R Analysis Pipeline

**Simulation Analysis:**
- `R/sim_00_run_all.R` - Master script for all simulation analyses
- `R/sim_01_import_data.R` - Import CSV data from Python
- `R/sim_02_model_comparison.R` - Compare ABS1, REL1, REL2 performance
- `R/sim_03_stimulus_metrics_analysis.R` - Analyze stimulus center/spread correlations
- `R/sim_04_asymmetry_index_evolution.R` - Track asymmetry convergence
- `R/sim_05_lat_entropy_analysis.R` - Analyze latency entropy evolution

**Real Data Analysis:**
- `R/real_00_run_all.R` - Master script for all real data analyses
- `R/real_01_import_data.R` - Import real experimental data
- `R/real_02_descriptive_analysis.R` - Descriptive statistics and distributions
- `R/real_03_statistical_analysis.R` - Hypothesis testing and comparisons
- `R/real_04_convergence_analysis.R` - Convergence and stability analysis

**Publication Figures:**
- `R/00_create_paper_figures.R` - Generate all publication figures (Figures 4-10)
- `R/00_create_paper_tables.R` - Generate publication tables

**Utilities:**
- `R/effect_size_utils.R` - Effect size calculations
- `R/npar_posthoc.R` - Non-parametric post-hoc tests

### Data Flow

```
Simulations (Python)
    ↓
GBF files → regenerate_simulation_data.py → Excel files
    ↓
CSV export → R/sim_*.R → Statistical analysis
    ↓
R/00_create_paper_figures.R → Publication figures
```

### CSV Export Format

**File:** `R/indata/stimulus_metrics_all_models.csv`

**Columns (one row per trial block per subject):**
- `model`: Model name (ABS1, REL1, REL2)
- `pse_true`, `jnd_true`: True parameters
- `subject_id`: Subject identifier
- `group`: Group index
- `trial_block`: Trial count (40, 60, 80, ..., 200)
- `stimulus_center`: Mean stimulus latency
- `stimulus_spread`: Std of stimulus latency
- `lat_entropy`: Latency entropy
- `pse_est`: Estimated PSE at trial block
- `jnd_est`: Estimated JND at trial block
- `asymmetry_index`: Asymmetry metric

### Progressive Metrics (Rounded to 2 decimals)

All metrics are calculated cumulatively at trial blocks: 40, 60, 80, 100, 120, 140, 160, 180, 200

**Stimulus Distribution:**
- `stimulus_center`: Mean of stimulus latencies
- `stimulus_spread`: Standard deviation of stimulus latencies
- `stimulus_min`, `stimulus_max`: Range of stimulus latencies

**Psychometric:**
- `pse_est`: Fitted PSE from cumulative data
- `jnd_est`: Fitted JND from cumulative data

**Asymmetry:**
- `asymmetry_index`: (n_after - n_before) / total, range [-1, 1]

**Entropy:**
- `lat_entropy`: Shannon entropy of latency distribution


---

## Architecture Overview

### Core Modules

**Simulation Engine:**
- `analysis/core/simulation_engine.py` - Unified simulation logic for all three models

**Progressive Analysis:**
- `analysis/core/progressive_analyzer.py` - Progressive psychometric analysis
  - `run_progressive_analysis()` - Calculate PSE/JND at trial intervals
  - Returns: trial_counts, pse_values, jnd_values, lat_entropy

**Psychometric Analysis:**
- `analysis/core/psychometric_analysis.py` - Core analysis functions:
  - `calculate_progressive_asymmetry()` - Asymmetry at trial counts
  - `calculate_progressive_stimulus_metrics()` - Stimulus distribution metrics
  - `calculate_latency_statistics()` - Latency entropy and statistics
  - `analyze_subject_results()` - Complete subject analysis pipeline
  - `consolidate_results()` - Excel generation
  - `add_group_stats_to_excel()` - Add GROUP_mean/std rows

**Data Generation:**
- `analysis/core/generate_analysis_data.py` - Data pipeline functions:
  - `add_progressive_asymmetry_to_excel()` - Add asymmetry columns
  - `add_progressive_lat_entropy_to_excel()` - Add entropy columns
  - `add_progressive_stimulus_metrics_to_excel()` - Add stimulus metrics
  - `add_final_estimates_to_excel()` - Copy pse_200/jnd_200 to pse_est/jnd_est
  - `export_stimulus_metrics_to_csv()` - Export to CSV for R
  - `regenerate_all_data_from_gbf()` - Full pipeline from GBF files
  - `generate_all_data()` - Add metrics to existing Excel files

**Plotting:**
- `analysis/core/plotting.py` - Core plotting functions
- `analysis/core/plot_psychometric_curves.py` - Psychometric curve plotting
- `analysis/core/plot_stimulus_distribution.py` - Stimulus distribution plots
- `analysis/core/generate_analysis_plots.py` - Analysis metric plots

**Data Loading:**
- `analysis/core/data_loader.py` - Load and expand experimental data
- `analysis/io/metadata.py` - Extract metadata from filenames

**Multithreading:**
- `utilities/multithreading_utils.py` - Parallel execution:
  - `SubjectSimulationTask` - Task encapsulation
  - `MultiThreadedSimulationRunner` - Thread pool management
  - `run_subject_simulation()` - Single subject simulation
  - `run_progressive_analysis_task()` - Single subject analysis
  - `parse_gbf_filename()` - Extract PSE/JND from filename
  - `load_gbf_files_for_group()` - Load GBF files for skip mode
  - `backfill_skip_mode()` - Fill trial data from GBF files

---

## Common Features

### Uniform Output Structure

All experiment files share the same output format:

1. **Header:** 60 `=` characters with experiment title
2. **Trial display:** `TRIAL X/N` format
3. **Input prompt:** "Is the second tone closer to the first (1) or the third (2)?"
4. **Footer:** "EXPERIMENT COMPLETE"
5. **Statistics:** Via `print_statistics()` function
6. **Fitted parameters:** PSE and JND from psychometric fit
7. **Psychometric plot:** Individual curve with data points

### Statistics Reported

- Total trials
- Offset (true threshold)
- Trials below/above offset (count and percentage)
- Stimulus range (min, max, mean, median, std)
- Response accuracy (if applicable)
- Asymmetry metrics (n_before, n_after, asymmetry_index)

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

### Group Simulation Workflow (Multithreaded)
1. **Phase 1:** Create simulation tasks for all subjects in group
2. **Phase 2:** Run all subject simulations in parallel
3. **Phase 3:** Run progressive analyses in parallel
4. **Phase 4:** Generate plots and Excel sequentially
5. **Phase 5:** Create grid plots combining all 9 groups

---

## Key Differences Between Models

| Feature | ABS1 | REL1 | REL2 |
|---------|------|------|------|
| Engines | 1 | 1 | 2 (pre + post) |
| Stimulus space | Absolute (200-800ms) | Relative (5-300ms) | Relative (0-300ms each) |
| Guess rate | 0.04 | 0.5 | 0.5 |
| Pre/post balance | Natural (can be biased) | Block randomized (balanced) | Block randomized (balanced) |
| Update method | Response (0/1) | Success (correct/incorrect) | Success per model |
| Asymmetry bias | Can exhibit | Enforced balanced | Enforced balanced |

---

## Output Directory Structure

```
data/
├── output/
│   ├── console/
│   │   ├── {subject_id}_psychometric.png
│   │   └── ...
│   ├── real/
│   │   ├── {subject_id}_psychometric.png
│   │   └── ...
│   └── sim_gridrnd/
│       ├── ABS1/
│       │   ├── group_480_20/
│       │   │   ├── S*_G*_PSE_JND_ABS1.txt (20 GBF files)
│       │   │   └── results/
│       │   │       ├── ABS1_G1_group_histogram.png
│       │   │       ├── ABS1_G1_group_psychometric.png
│       │   │       └── ABS1_G1_results_summary.xlsx
│       │   ├── group_480_40/ ... (N groups total)
│       │   ├── ABS1_histogram_grid.png
│       │   ├── ABS1_psychometric_grid.png
│       │   ├── asymmetry_modulo.png
│       │   ├── asymmetry_scatter_envelope.png
│       │   ├── stimulus_center_evolution.png
│       │   ├── stimulus_spread_evolution.png
│       │   └── bimodality_index_evolution.png
│       ├── REL1/ (similar structure)
│       └── REL2/ (similar structure)
├── paper_plots/
│   ├── Figure3.png (grid psychometric)
│   ├── Figure5.png (latency envelope grid)
│   ├── Figure7.png (combined latency + entropy)
│   └── ...
└── input/
    ├── expdata/ (real experimental data files)
    └── results_BIS_fx_vs_ad_td_2model_rel_logistic_prog_long.xlsx

R/
├── indata/
│   └── stimulus_metrics_all_models.csv (exported from Python)
├── results_simulations/ (RDS files from R analysis)
├── results_real_logistic/ (RDS files from R analysis)
└── [R scripts]
```

---

## Dependencies

### Core Libraries
- `numpy >= 1.20.0` - Numerical operations
- `pandas >= 1.3.0` - Data handling
- `matplotlib >= 3.3.0` - Plotting
- `scipy >= 1.7.0` - Statistical functions and curve fitting
- `statsmodels >= 0.13.0` - GLM fitting
- `adopy >= 0.3.0` - Adaptive experimental design
- `openpyxl >= 3.0.0` - Excel file generation
- `Pillow >= 8.0.0` - Image processing for grid plots

### Custom Modules
- `utilities/` - Helper functions and utilities
- `analysis/` - Psychometric analysis pipeline
- `bisection/` - ADOpy wrappers

---

## Tips for Usage

1. **Start with console files** to understand the experiment flow
2. **Use real audio files** for actual data collection
3. **Use group simulations** to:
   - Generate systematic datasets for validation
   - Test analysis pipelines
   - Compare model performance across PSE/JND space
4. **Use asymmetry analysis** to:
   - Detect stimulus presentation bias (ABS1)
   - Validate that REL1/REL2 maintain balance
   - Track convergence of asymmetry across trials
5. **Use stimulus metrics** to:
   - Validate that presented stimuli match intended PSE/JND
   - Detect bimodal stimulus distributions
   - Analyze ADOpy sampling behavior

---
