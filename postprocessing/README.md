# Postprocessing Package

Python postprocessing pipeline for temporal bisection psychometric data analysis.

## Features

- **Three Fitting Methods**: Logistic, Probit, and GaussFitm grid search
- **Progressive Analysis**: Analyze data at multiple trial counts (40, 60, 80, 100, 120, 140, 160, 180, 200)
- **Format Conversion**: Convert PSA (raw psysuite) to GBF format
- **Excel Reports**: Wide and long format outputs

## Usage

The easiest way to run the pipeline is `postprocessing/run_analysis.py`:

1. Configure the parameters at the top of the file:
   ```python
   PSA_INPUT_DIR  = "/path/to/psa/files"
   GBF_OUTPUT_DIR = "/path/to/gbf/output"
   RESULTS_OUTPUT_DIR = "/path/to/results"
   PROJECT_NAME   = "my_project"
   FITTING_METHOD = "logistic"   # 'logistic', 'probit', or 'gaussfit'
   RUN_CONVERSION = True         # set False if GBF files already exist
   RUN_ANALYSIS   = True
   ```
2. Run it directly from PyCharm or terminal.

## File Formats

### Input (PSA — raw psysuite output)

Tab-delimited with header row containing at least `lat` and `user_ans` columns.

### Input (GBF — converted format)

Tab-delimited, no header. Each line: `latency count response [confl_magn]`

Optional prefix lines:
- `@ 0.0` — guess rate parameter
- `# comment` or `* comment` — skipped

### Filename Convention

```
{ID}_{age}_{gender}_{modality}_{algorithm}_{group}.txt
```

Example: `A01_26_f_BISA_AD_TD.txt`

## Python API

### Progressive Analysis

```python
from postprocessing.modules.progressive_analyzer import analyze_batch

results = analyze_batch(
    input_dir='data/gbf',
    output_dir='results',
    project_name='my_study',
    method='logistic',   # 'logistic', 'probit', 'gaussfit'
    bin_size=10.0        # only used for gaussfit
)
```

### Format Conversion

```python
from postprocessing.modules.converter import convert_psa_to_gbf

stats = convert_psa_to_gbf(input_dir='data/psa', output_dir='data/gbf')
print(f"Converted: {stats['converted']}/{stats['total_files']}")
```

### Psychometric Fitting (binned data)

`fitters.py` provides wrappers for binned data that delegate to the core fitting
functions in `utilities/psychometric_helpers.py`:

```python
from postprocessing.modules.fitters import fit_logistic, fit_probit, fit_gaussfit
import numpy as np

latencies = np.array([300, 400, 500, 600, 700])
counts    = np.ones(5)
responses = np.array([0, 0, 1, 1, 1])

result = fit_logistic(latencies, counts, responses)
print(f"PSE: {result.pse:.1f}, JND: {result.jnd:.1f}")
```

### Psychometric Fitting (individual trials)

For individual trial data, use the core functions directly:

```python
from utilities.psychometric_helpers import (
    fit_logistic_psychometric,
    fit_probit_psychometric,
    fit_gaussfit_psychometric,
)

pse, jnd = fit_logistic_psychometric(latencies, responses)
```

## Architecture

Fitting logic lives in a single place — `utilities/psychometric_helpers.py`:

```
utilities/psychometric_helpers.py
  ├── fit_logistic_psychometric()    ← GLM binomial/logit
  ├── fit_probit_psychometric()      ← GLM binomial/probit
  ├── fit_gaussfit_psychometric()    ← grid search, cumulative Gaussian
  ├── calculate_latency_statistics()
  └── calculate_stability_from_values()

postprocessing/modules/fitters.py   ← wrappers for binned data → FitResult
postprocessing/modules/converter.py ← PSA → GBF conversion
postprocessing/modules/parser.py    ← GBF file reader
postprocessing/modules/metadata.py  ← filename parser
postprocessing/modules/progressive_analyzer.py ← orchestrator + Excel output
postprocessing/modules/report_generator.py     ← wide/long Excel generation
```

## Fitting Methods

### Logistic
GLM with binomial family and logit link. `PSE = -b0/b1`, `JND = 1.35/|b1|`.

### Probit
GLM with binomial family and probit link. `PSE = -b0/b1`, `JND = 0.675/|b1|`.

### GaussFitm
Grid search over PSE/JND space using cumulative Gaussian:
`p = guess_rate + (1 - guess_rate) * Φ((x - PSE) / JND)`.
Slower but robust to initialization issues.

## Output Format

### Wide Format
One row per subject: metadata + `pse_40..pse_200`, `jnd_40..jnd_200`, `lat_mean/std/range/entropy_40..200`.

### Long Format
Nine rows per subject (one per trial count): metadata + `n_trials`, `pse`, `jnd`, `lat_mean`, `lat_std`, `lat_range`, `lat_entropy`.

## Dependencies

```
numpy >= 1.20.0
pandas >= 1.3.0
scipy >= 1.7.0
statsmodels >= 0.13.0
openpyxl >= 3.0.0
```
