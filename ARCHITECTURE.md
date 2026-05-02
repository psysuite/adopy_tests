# Architecture Overview

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TEMPORAL BISECTION SYSTEM                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXPERIMENT LAYER (main/)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐   │
│  │  Console Experiments │  │  Real Audio Expts    │  │  Group Simulations│  │
│  │  (3 files)           │  │  (3 files)           │  │  (3 files)       │   │
│  ├──────────────────────┤  ├──────────────────────┤  ├──────────────────┤   │
│  │ • subj_console_bis_  │  │ • subj_real_bis_     │  │ • group_sim_     │   │
│  │   ado_ABS1.py        │  │   ado_ABS1.py        │  │   gridrnd_ABS1.py│   │
│  │ • subj_console_bis_  │  │ • subj_real_bis_     │  │ • group_sim_     │   │
│  │   ado_REL1.py        │  │   ado_REL1.py        │  │   gridrnd_REL1.py│   │
│  │ • subj_console_bis_  │  │ • subj_real_bis_     │  │ • group_sim_     │   │
│  │   ado_REL2.py        │  │   ado_REL2.py        │  │   gridrnd_REL2.py│   │
│  │                      │  │                      │  │                  │   │
│  │ Input: Keyboard      │  │ Input: Audio + KB    │  │ Input: PSE/JND   │   │
│  │ Output: Plot + Stats │  │ Output: Plot + Stats │  │ Output: Plots +  │   │
│  │                      │  │                      │  │         Excel    │   │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────┘   │
│           │                         │                         │             │
│           └─────────────────────────┴─────────────────────────┘             │
│                                     │                                       │
└─────────────────────────────────────┼───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SIMULATION ENGINE (analysis/core/)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ SimulationEngine                                                       │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │ • simulate_subject()                                                   │ │
│  │   ├─ _simulate_ABS1()                                                  │ │
│  │   ├─ _simulate_REL1()                                                  │ │
│  │   └─ _simulate_REL2()                                                  │ │
│  │                                                                        │ │
│  │ • save_gbf_file()                                                      │ │
│  │                                                                        │ │
│  │ Dependencies:                                                          │ │
│  │ • BISAbsADOpyWrapper (bisection/)                                      │ │
│  │ • BISRelADOpyWrapper (bisection/)                                      │ │
│  │ • generate_response() (utilities/)                                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  MULTITHREADING LAYER (utilities/)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ MultiThreadedSimulationRunner                                          │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │ • run_subject_simulations()  [Phase 2 - Parallel]                      │ │
│  │ • run_progressive_analyses() [Phase 3 - Parallel]                      │ │
│  │                                                                        │ │
│  │ Uses ThreadPoolExecutor with MAX_WORKERS threads                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ SubjectSimulationTask                                                  │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │ • Encapsulates one subject's simulation parameters                     │ │
│  │ • Stores results: rows, result_dict, gbf_rows                          │ │
│  │ • Saves companion JSON with PSE/JND/sigma params alongside GBF file    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 PSYCHOMETRIC ANALYSIS (analysis/core/)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Metrics Calculation                                                    │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │ • calculate_asymmetry_metrics()                                        │ │
│  │   └─ asymmetry_index, n_before/after, pct_before/after                 │ │
│  │                                                                        │ │
│  │ • calculate_progressive_asymmetry()                                    │ │
│  │   └─ asymmetry at trial counts: 40, 60, 80, ..., 200                   │ │
│  │                                                                        │ │
│  │ • calculate_progressive_stimulus_metrics()                             │ │
│  │   ├─ stimulus_center (mean latency)                                    │ │
│  │   ├─ stimulus_spread (std latency)                                     │ │
│  │   ├─ stimulus_min/max                                                  │ │
│  │   └─ bimodality_index                                                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Subject Analysis                                                       │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │ • analyze_subject_results()                                            │ │
│  │   ├─ print_statistics()                                                │ │
│  │   ├─ gausFit()                                                         │ │
│  │   ├─ plot_psychometric()                                               │ │
│  │   └─ calculate asymmetry metrics                                       │ │
│  │                                                                        │ │
│  │ • safe_analyze_subject()                                               │ │
│  │   └─ Error handling wrapper                                            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Results Consolidation                                                  │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │ • consolidate_results()                                                │ │
│  │   └─ Create Excel with all metrics                                     │ │
│  │                                                                        │ │
│  │ • add_group_stats_to_excel()                                           │ │
│  │   └─ Add GROUP_mean and GROUP_std rows                                 │ │
│  │                                                                        │ │
│  │ • create_summary_report()                                              │ │
│  │ • print_summary_report()                                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PLOTTING LAYER (utilities/)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Group Plots (Phase 4)                                                  │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │ • plot_group_histograms()                                              │ │
│  │ • plot_group_psychometric()                                            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Grid Plots (Phase 5)                                                   │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │ • create_grid_plots_from_groups()                                      │ │
│  │   └─ Combine group plots into grid with PSE/JND labels                 │ │
│  │      (grid size depends on PSE/JND grid dimensions)                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Analysis Metric Plots (Phase 6)                                        │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │ • plot_analysis_metrics() [from analysis/core/]                        │ │
│  │   ├─ asymmetry_modulo.png (|asymmetry_index| evolution)                │ │
│  │   ├─ asymmetry_scatter_envelope.png (asymmetry distribution)           │ │
│  │   ├─ stimulus_center_evolution.png (mean stimulus latency)             │ │
│  │   ├─ stimulus_spread_evolution.png (std stimulus latency)              │ │
│  │   └─ bimodality_index_evolution.png (bimodality measure)               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UTILITY MODULES (utilities/)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  • misc_generate_responses.py                                               │
│    └─ generate_response(), get_sigma_from_jnd()                             │
│                                                                             │
│  • trial_sequence.py                                                        │
│    └─ create_trial_sequence_absolute(), create_trial_sequence_relative()    │
│                                                                             │
│  • data_loader.py                                                           │
│    └─ Load and normalize trial data                                         │
│                                                                             │
│  • logging_config.py                                                        │
│    └─ log_subject_error()                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADOpy WRAPPERS (bisection/)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  • BISAbsADOpyWrapper                                                       │
│    └─ Wraps ADOpy for absolute stimulus model                               │
│                                                                             │
│  • BISRelADOpyWrapper                                                       │
│    └─ Wraps ADOpy for relative stimulus model                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  data/output/                                                               │
│  ├── console/                                                               │
│  │   └── {subject_id}_psychometric.png                                      │
│  ├── real/                                                                  │
│  │   └── {subject_id}_psychometric.png                                      │
│  └── sim_gridrnd/                                                           │
│      ├── ABS1/                                                              │
│      │   ├── group_480_20/                                                  │
│      │   │   ├── S*_G*_*_*_ABS1.txt (20 GBF files)                          │
│      │   │   └── results/                                                   │
│      │   │       ├── ABS1_G1_group_histogram.png                            │
│      │   │       ├── ABS1_G1_group_psychometric.png                         │
│      │   │       └── ABS1_G1_results_summary.xlsx                           │
│      │   ├── group_480_40/ ... (N groups total, configurable)               │
│      │   ├── ABS1_histogram_grid.png                                        │
│      │   ├── ABS1_psychometric_grid.png                                     │
│      │   ├── asymmetry_modulo.png                                           │
│      │   ├── asymmetry_scatter_envelope.png                                 │
│      │   ├── stimulus_center_evolution.png                                  │
│      │   ├── stimulus_spread_evolution.png                                  │
│      │   └── bimodality_index_evolution.png                                 │
│      ├── REL1/ (similar structure)                                          │
│      └── REL2/ (similar structure)                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow: Group Simulation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GROUP SIMULATION WORKFLOW (6 PHASES)                     │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 1: Task Creation
┌──────────────────────────────────────────────────────────────────────────────┐
│ For each group (PSE, JND):                                                   │
│   For each subject (1-20):                                                   │
│     Create SubjectSimulationTask with:                                       │
│       • PSE ± 2.5 random variation                                           │
│       • JND ± 2.5 random variation                                           │
│       • 200 trials, ADOpy params, bisection params                           │
│                                                                              │
│ NOTE: If SKIP_PHASES_1_2=True, phases 1-2 are skipped and GBF files are     │
│       loaded from disk. PSE/JND are extracted from filename format:          │
│       SXX_GZ_PSE_JND_MODEL.txt. If parsing fails (old format), falls back    │
│       to progressive fit values with warning.                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
PHASE 2: Parallel Subject Simulation
┌──────────────────────────────────────────────────────────────────────────────┐
│ ThreadPoolExecutor (MAX_WORKERS threads):                                    │
│   For each task in parallel:                                                 │
│     • Run 200 trials with ADOpy adaptive sampling                            │
│     • Generate synthetic responses                                           │
│     • Save GBF file                                                          │
│     • Collect trial rows and result_dict                                     │
│                                                                              │
│ Output per subject:                                                          │
│   • rows: List of trial dicts with 'lat', 'res', 'user_ans'                  │
│   • result_dict: PSE, JND, sigma, mu, n_trials, accuracy                     │
│   • gbf_rows: Expanded trial data for analysis                               │
│   • GBF filename: SXX_GZ_PSE_JND_MODEL.txt (PSE/JND embedded in name)        │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
PHASE 3: Parallel Progressive Analysis
┌──────────────────────────────────────────────────────────────────────────────┐
│ ThreadPoolExecutor (MAX_WORKERS threads):                                    │
│   For each subject in parallel:                                              │
│     • calculate_progressive_asymmetry()                                      │
│       └─ asymmetry_40, asymmetry_60, ..., asymmetry_200                      │
│                                                                              │
│     • calculate_progressive_stimulus_metrics()                               │
│       ├─ stimulus_center_40, ..., stimulus_center_200                        │
│       ├─ stimulus_spread_40, ..., stimulus_spread_200                        │
│       ├─ stimulus_min_40, ..., stimulus_min_200                              │
│       ├─ stimulus_max_40, ..., stimulus_max_200                              │
│       └─ bimodality_index_40, ..., bimodality_index_200                      │
│                                                                              │
│ Output per subject:                                                          │
│   • Updated result_dict with all progressive metrics                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
PHASE 4: Sequential Plot & Excel Generation
┌──────────────────────────────────────────────────────────────────────────────┐
│ For each group (sequential):                                                 │
│   • plot_group_histograms()                                                  │
│   • plot_group_psychometric()                                                │
│   • consolidate_results() → Excel file                                       │
│   • add_group_stats_to_excel() → Add GROUP_mean/std rows                     │
│                                                                              │
│ Output per group:                                                            │
│   • Group plots (PNG)                                                        │
│   • Excel with all metrics (one row per subject + GROUP_mean/std)            │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
PHASE 5: Grid Plot Creation
┌──────────────────────────────────────────────────────────────────────────────┐
│ For each plot type (histogram, psychometric):                                │
│   • create_grid_plots_from_groups()                                          │
│     └─ Combine group plots into grid with PSE/JND labels                     │
│        (grid size depends on PSE/JND grid dimensions)                        │
│                                                                              │
│ Output:                                                                      │
│   • {model}_histogram_grid.png                                               │
│   • {model}_psychometric_grid.png                                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
PHASE 6: Analysis Metric Plots
┌──────────────────────────────────────────────────────────────────────────────┐
│ Generate 5 analysis plots from all group Excel files:                        │
│   • plot_analysis_metrics()                                                  │
│     ├─ asymmetry_modulo.png (|asymmetry_index| evolution)                    │
│     ├─ asymmetry_scatter_envelope.png (asymmetry distribution)               │
│     ├─ stimulus_center_evolution.png (mean stimulus latency)                 │
│     ├─ stimulus_spread_evolution.png (std stimulus latency)                  │
│     └─ bimodality_index_evolution.png (bimodality measure)                   │
│                                                                              │
│ Output:                                                                      │
│   • 5 PNG files aggregating metrics across all groups and subjects           │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow: Real Data Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REAL DATA ANALYSIS WORKFLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

ENTRY POINT: group_realdata_analysis.py
┌──────────────────────────────────────────────────────────────────────────────┐
│ Configuration:                                                               │
│   • PSA_INPUT_DIR: Raw data from psysuite (PSA format)                       │
│   • GBF_OUTPUT_DIR: Converted data (GBF format)                              │
│   • RESULTS_OUTPUT_DIR: Excel results                                        │
│   • PROJECT_NAME: Used in output filenames                                   │
│   • FITTING_METHOD: 'logistic', 'probit', or 'gaussfit'                      │
│   • BIN_SIZE: For gaussfit method                                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STEP 1: PSA → GBF Conversion
┌──────────────────────────────────────────────────────────────────────────────┐
│ convert_psa_to_gbf(psa_input_dir, gbf_output_dir)                            │
│   ├─ Read PSA files from input directory                                     │
│   ├─ Parse PSA format (stimulus latency, response, user answer)              │
│   ├─ Convert to GBF format (tab-separated: lat, res, user_ans)               │
│   └─ Save GBF files to output directory                                      │
│                                                                              │
│ Output: GBF files ready for analysis                                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STEP 2: Progressive Analysis
┌──────────────────────────────────────────────────────────────────────────────┐
│ analyze_batch(input_dir, output_dir, project_name, method, bin_size)         │
│   │                                                                          │
│   ├─ For each GBF file:                                                      │
│   │   ├─ Read trial data                                                     │
│   │   ├─ Call analyze_subject_results()                                      │
│   │   │   ├─ print_statistics()                                              │
│   │   │   ├─ gausFit() or logistic_fit() or probit_fit()                     │
│   │   │   ├─ plot_psychometric()                                             │
│   │   │   └─ calculate_asymmetry_metrics()                                   │
│   │   │       └─ asymmetry_index, n_before/after, pct_before/after           │
│   │   │                                                                      │
│   │   └─ Store results in results_list                                       │
│   │                                                                          │
│   └─ Consolidate all results:                                                │
│       ├─ consolidate_results() → Create Excel file                           │
│       └─ add_group_stats_to_excel() → Add GROUP_mean/std rows                │
│                                                                              │
│ Output:                                                                      │
│   • results_{PROJECT_NAME}_{METHOD}_prog_wide.xlsx (wide format)             │
│   • results_{PROJECT_NAME}_{METHOD}_prog_long.xlsx (long format)             │
└──────────────────────────────────────────────────────────────────────────────┘
```

```
main/
├── group_sim_gridrnd_ABS1.py
├── group_sim_gridrnd_REL1.py
├── group_sim_gridrnd_REL2.py
│   ├── SimulationEngine (analysis/core/)
│   ├── MultiThreadedSimulationRunner (utilities/)
│   ├── SubjectSimulationTask (utilities/)
│   ├── parse_gbf_filename (utilities/multithreading_utils.py)
│   ├── load_gbf_files_for_group (utilities/multithreading_utils.py)
│   ├── backfill_skip_mode (utilities/multithreading_utils.py)
│   ├── plot_group_* (utilities/)
│   ├── create_grid_plots_from_groups (utilities/)
│   ├── plot_analysis_metrics (analysis/core/)
│   ├── consolidate_results (analysis/core/)
│   ├── add_group_stats_to_excel (analysis/core/)
│   └── run_progressive_analyses (utilities/)
│
├── group_realdata_analysis.py
│   ├── convert_psa_to_gbf (analysis/io/)
│   └── analyze_batch (analysis/orchestrator/)
│
├── subj_console_bis_ado_ABS1.py
├── subj_console_bis_ado_REL1.py
├── subj_console_bis_ado_REL2.py
│   ├── BISAbsADOpyWrapper (bisection/)
│   ├── BISRelADOpyWrapper (bisection/)
│   ├── analyze_subject_results (analysis/core/)
│   └── plot_psychometric (utilities/)
│
├── subj_real_bis_ado_ABS1.py
├── subj_real_bis_ado_REL1.py
├── subj_real_bis_ado_REL2.py
    ├── BISAbsADOpyWrapper (bisection/)
    ├── BISRelADOpyWrapper (bisection/)
    ├── analyze_subject_results (analysis/core/)
    └── plot_psychometric (utilities/)

analysis/core/
├── simulation_engine.py
│   ├── SimulationEngine class
│   ├── BISAbsADOpyWrapper (bisection/)
│   ├── BISRelADOpyWrapper (bisection/)
│   └── generate_response (utilities/)analyze_subject_results
│
└── psychometric_analysis.py
    ├── calculate_asymmetry_metrics (core metric)
    ├── calculate_progressive_asymmetry (uses calculate_asymmetry_metrics)
    ├── calculate_progressive_stimulus_metrics (core metric)
    ├── analyze_subject_results (uses calculate_asymmetry_metrics)
    ├── consolidate_results
    ├── add_group_stats_to_excel
    └── safe_analyze_subject

utilities/
├── multithreading_utils.py
│   ├── SubjectSimulationTask
│   ├── MultiThreadedSimulationRunner
│   ├── parse_gbf_filename()
│   ├── load_gbf_files_for_group()
│   └── backfill_skip_mode()
│
├── plotting.py
│   ├── plot_group_histograms
│   ├── plot_group_psychometric
│   └── create_grid_plots_from_groups
│
├── misc_generate_responses.py
├── trial_sequence.py
├── data_loader.py
└── logging_config.py

bisection/
├── BISAbsADOpyWrapper
└── BISRelADOpyWrapper
```

## Key Design Patterns

### 1. **Separation of Concerns**
- **Simulation**: `SimulationEngine` handles trial generation
- **Analysis**: `psychometric_analysis` handles metrics and reporting
- **Plotting**: `plotting` handles visualization
- **Multithreading**: `multithreading_utils` handles parallelization

### 2. **Centralized Metrics**
- All metric calculations in `psychometric_analysis.py`
- `calculate_asymmetry_metrics()` used by:
  - `analyze_subject_results()` (during real subject analysis - console/real audio experiments)
  - `calculate_progressive_asymmetry()` (during progressive analysis in Phase 3 of group simulations)

### 3. **Progressive Analysis**
- Metrics calculated at trial counts: 40, 60, 80, 100, 120, 140, 160, 180, 200
- Enables tracking of convergence and stability
- Runs in parallel (Phase 3) for efficiency

### 4. **6-Phase Pipeline**
- Phase 1: Sequential task creation
- Phase 2: Parallel subject simulation
- Phase 3: Parallel progressive analysis
- Phase 4: Sequential plotting and Excel
- Phase 5: Sequential grid plot creation
- Phase 6: Sequential analysis metric plots

### 5. **Error Handling**
- `safe_analyze_subject()` wraps analysis with try-catch
- Failures don't block other subjects
- Errors logged with subject ID and operation

