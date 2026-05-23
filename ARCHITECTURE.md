# Architecture Overview

## Current System Architecture (Snapshot)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TEMPORAL BISECTION SYSTEM - CURRENT STATE                │
└─────────────────────────────────────────────────────────────────────────────┘

ENTRY POINTS (main/)
├── group_sim_gridrnd_ABS1.py, REL1.py, REL2.py
│   └── 8-Phase Pipeline: Simulation → Analysis → Plots → Excel → Data Generation
├── regenerate_analysis_plots.py
│   └── Regenerate plots from existing Excel files
├── regenerate_simulation_data.py
│   └── Regenerate Excel + CSV from existing GBF files
├── group_realdata_analysis.py
│   └── Real data analysis pipeline
├── progr_analyze_folder.py
│   └── Progressive analysis on folder of GBF files
├── create_paper_plots.py
│   └── Assemble publication-ready figures
└── subj_console_bis_ado_*.py, subj_real_bis_ado_*.py
    └── Individual subject experiments

CORE SIMULATION & ANALYSIS (analysis/core/)
├── simulation_engine.py
│   ├── SimulationEngine class
│   ├── simulate_subject() → (rows, result_dict, gbf_rows)
│   └── Dependencies: BISAbsADOpyWrapper, BISRelADOpyWrapper
├── generate_analysis_data.py
│   ├── regenerate_all_data_from_gbf() - Main entry point for data regeneration
│   ├── generate_all_data() - Add analysis columns to existing Excel
│   ├── add_progressive_asymmetry_to_excel()
│   ├── add_progressive_lat_entropy_to_excel()
│   ├── add_progressive_stimulus_metrics_to_excel()
│   └── export_stimulus_metrics_to_csv()
├── generate_analysis_plots.py
│   ├── AnalysisPlotGenerator class
│   ├── plot_asymmetry_modulo()
│   ├── plot_asymmetry_scatter_envelope()
│   ├── plot_stimulus_center_evolution() - 1 plot, 3 curves (PSE), avg over JND
│   ├── plot_stimulus_spread_evolution() - 1 plot, 3 curves (JND), avg over PSE
│   └── generate_all_analysis() - Main entry point
├── psychometric_analysis.py
│   ├── analyze_subject_results()
│   ├── calculate_asymmetry_metrics()
│   ├── calculate_progressive_asymmetry()
│   ├── calculate_progressive_stimulus_metrics()
│   ├── calculate_latency_statistics()
│   ├── consolidate_results() → Excel file
│   ├── add_group_stats_to_excel()
│   └── safe_analyze_subject()
├── progressive_analyzer.py
│   ├── ProgressiveAnalyzer class
│   ├── run_progressive_analysis()
│   └── analyze_folder()
├── psychometric_helpers.py
│   ├── fit_logistic_psychometric()
│   ├── fit_probit_psychometric()
│   └── calculate_stability_from_values()
├── data_loader.py
│   ├── DataLoader class
│   └── load_and_expand() - Auto-detects format (direct/binned)
└── plotting.py
    ├── plot_group_histograms()
    ├── plot_group_psychometric()
    ├── load_group_rows()
    ├── find_gbf_file()
    └── plot_generic_grid()

PLOT GENERATION (analysis/core/)
├── plot_psychometric_curves.py
│   ├── plot_psychometric_for_model() - Generate group + grid plots
│   └── plot_psychometric_into_axes()
└── plot_stimulus_distribution.py
    ├── plot_stimulus_for_model() - Generate group + grid plots
    ├── plot_group_stimulus_distribution()
    └── plot_stimulus_grid_compact()

I/O & CONVERSION (analysis/io/)
├── converter.py
│   ├── convert_psa_to_gbf()
│   ├── save_gbf_file()
│   └── read_gbf_file()
├── parser.py
│   ├── TrialData dataclass
│   └── parse_data_file()
├── metadata.py
│   ├── SubjectMetadata dataclass
│   └── extract_metadata()
├── fitters.py
│   ├── FitResult dataclass
│   ├── fit_logistic()
│   ├── fit_probit()
│   └── expand_binned_data()
└── report_generator.py
    ├── generate_wide_format()
    └── generate_long_format()

ORCHESTRATION (analysis/orchestrator.py)
├── ProgressiveResultWithMetadata class
├── analyze_subject_progressive()
└── analyze_batch()

MULTITHREADING (utilities/multithreading_utils.py)
├── SubjectSimulationTask class
├── MultiThreadedSimulationRunner class
├── load_gbf_files_for_group()
├── parse_gbf_filename()
├── backfill_skip_mode()
└── run_progressive_analysis_task()

PLOTTING UTILITIES (utilities/plotting.py)
├── plot_group_histograms()
├── plot_group_psychometric()
├── create_grid_plots_from_groups()
└── Dependencies: matplotlib, PIL

UTILITIES (utilities/)
├── misc_generate_responses.py
│   ├── generate_response()
│   └── get_sigma_from_jnd()
├── trial_sequence.py
│   ├── create_trial_sequence_absolute()
│   └── create_trial_sequence_relative()
├── gbf_helpers.py
│   ├── calculate_trial_success()
│   └── load_gbf_with_success()
├── logging_config.py
│   ├── setup_logger()
│   └── log_subject_error()
└── real_exp_accessories.py
    ├── generate_tone()
    └── run_bisection_trial()

ADOpy WRAPPERS (bisection/)
├── BISADOpyWrapper.py (base class)
│   ├── print_statistics()
│   ├── gausFit()
│   ├── plot_psychometric()
│   └── get()
├── BISAbsADOpyWrapper.py
│   └── Absolute stimulus model with exclusion window
└── BISRelADOpyWrapper.py
    └── Relative stimulus model (1-model and 2-model variants)
```

## 8-Phase Group Simulation Pipeline

```
PHASE 1: Task Creation (Sequential)
├── For each group (PSE, JND):
│   └── For each subject (1-20):
│       └── Create SubjectSimulationTask with PSE±2.5, JND±2.5, 200 trials
└── Output: List of tasks

PHASE 2: Parallel Subject Simulation (ThreadPoolExecutor)
├── For each task in parallel:
│   ├── Run 200 trials with ADOpy adaptive sampling
│   ├── Generate synthetic responses
│   ├── Save GBF file (SXX_GZ_PSE_JND_MODEL.txt)
│   └── Collect rows, result_dict, gbf_rows
└── Output: GBF files + result dicts

PHASE 3: Parallel Progressive Analysis (ThreadPoolExecutor)
├── For each subject in parallel:
│   ├── calculate_progressive_asymmetry() → asymmetry_40..200
│   └── calculate_progressive_stimulus_metrics() → stimulus_center/spread_40..200
└── Output: Updated result_dict with all metrics

PHASE 4: Sequential Plot & Excel Generation
├── For each group:
│   ├── plot_group_histograms()
│   ├── plot_group_psychometric()
│   ├── consolidate_results() → Excel
│   └── add_group_stats_to_excel() → Add GROUP_mean/std rows
└── Output: Group plots + Excel files

PHASE 5: Generate Analysis Data (Excel Columns)
├── regenerate_all_data_from_gbf() OR generate_all_data()
├── add_progressive_asymmetry_to_excel()
├── add_progressive_lat_entropy_to_excel()
├── add_progressive_stimulus_metrics_to_excel()
└── Output: Excel files with all analysis columns

PHASE 6: Create Grid Plots
├── create_grid_plots_from_groups()
│   ├── {model}_histogram_grid.png
│   └── {model}_psychometric_grid.png
└── Output: Grid plots

PHASE 7: Generate Group Plots (Psychometric + Stimulus Distribution)
├── plot_psychometric_for_model() → {model}_grid_psychometric.png
├── plot_stimulus_for_model() → {model}_stimulus_distribution_grid.png
└── Output: Group-level plots

PHASE 8: Generate Analysis Metric Plots & Export CSV
├── generate_all_analysis()
│   ├── asymmetry_modulo.png
│   ├── asymmetry_scatter_envelope.png
│   ├── stimulus_center_evolution.png (1 plot, 3 PSE curves, avg JND)
│   ├── stimulus_spread_evolution.png (1 plot, 3 JND curves, avg PSE)
│   └── export_stimulus_metrics_to_csv()
└── Output: Analysis plots + CSV for R
```

## Data Regeneration Scripts

```
regenerate_analysis_plots.py
├── Calls: plot_psychometric_for_model(), plot_stimulus_for_model()
├── Calls: generate_all_analysis()
└── Purpose: Regenerate plots from existing Excel files

regenerate_simulation_data.py
├── Calls: regenerate_all_data_from_gbf()
├── Calls: export_stimulus_metrics_to_csv()
└── Purpose: Regenerate Excel + CSV from existing GBF files
```

## Data Formats

### Input
- **PSA Format** (Real data): Tab-delimited with columns: id, label, lat, confl, res, cor_ans, elapsed, rep, user_ans
- **GBF Format** (Simulated/Converted): Tab-delimited, no header: lat, count, user_ans, [confl_magn]

### Processing
- **Direct Format**: lat, user_ans columns
- **Binned Format**: lat, count, resp columns (auto-expanded by DataLoader)

### Output
- **GBF Files**: SXX_GZ_PSE_JND_MODEL.txt (PSE/JND embedded in filename)
- **Excel Reports**: Wide format (1 row/subject) + GROUP_mean/std rows
- **PNG Plots**: Psychometric, histograms, grids, analysis metrics
- **CSV**: stimulus_metrics_all_models.csv for R analysis

## Key Metrics

### Psychometric
- PSE, JND, sigma, mu, accuracy

### Asymmetry
- asymmetry_index [-1, 1], n_before/after, pct_before/after

### Stimulus Distribution
- stimulus_center (mean), stimulus_spread (std), stimulus_min/max
- lat_entropy (Shannon entropy, 10 bins)

### Progressive (at trial counts: 40, 60, 80, 100, 120, 140, 160, 180, 200)
- All above metrics at each trial count
- Enables convergence and stability analysis
