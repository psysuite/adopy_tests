# Patch: Add Progressive Response Accuracy (pct_correct)

## Overview

This patch adds progressive response accuracy columns to existing Excel files:
- `pct_correct_40`, `pct_correct_60`, ..., `pct_correct_200`

These columns track the percentage of correct responses at each trial block, showing how subject performance converges during the adaptive procedure.

## Changes

### 1. Python Code (analysis/orchestration/metrics.py)

**New method:** `MetricsCalculator.calculate_progressive_pct_correct(gbf_rows)`
- Calculates % correct at each trial block (40, 60, 80, ..., 200)
- Integrated into `calculate_all_metrics()` as step B2b
- Will automatically generate columns when full metrics are recalculated

**Semantic:**
- `pct_correct = (# correct responses / # trials) × 100` at each block
- Values: 0-100 (percentage)
- Interpretation: Higher = better subject learning

### 2. Immediate Patch (Multithreaded)

**File:** `patch_add_pct_correct.py`

Adds `pct_correct_*` columns to `synthetic_data_wide.xlsx` without recalculating all metrics.

**Usage (synthetic only, default):**

```bash
python3 main/patch_add_pct_correct.py
```

**Usage (with custom paths):**

```bash
python3 main/patch_add_pct_correct.py \
    --gbf-root /data/CODE/python/adopy_tests/data/output/sim_gridrnd \
    --excel-path /data/CODE/python/adopy_tests/data/output/synthetic_data_wide.xlsx
```

**What it does:**
1. Discovers all synthetic GBF files: `sim_gridrnd/{ABS1|REL1|REL2}/group_*/`
2. For each subject in Excel, reads corresponding GBF file
3. Calculates `pct_correct_*` from GBF response data (multithreaded, ~8 workers)
4. Updates Excel with new columns
5. Regenerates `synthetic_data_long.xlsx` with new columns included

**Performance:**
- ~1-2 min for all 540 synthetic subjects (180 per model × 3 models, multithreaded)
- ~3x speedup with 8 threads vs sequential

**Output:**
- Updated `synthetic_data_wide.xlsx`
- Regenerated `synthetic_data_long.xlsx`

### 3. R Code Updates

**Files modified:**
- `../../R/sim_01_import_data.R`: Added documentation about new columns
- `../../R/real_01_import_data.R`: Added documentation + automatic conversion of `pct_correct_*` to numeric

Now `pct_correct_*` columns will be automatically cleaned and available for analysis.

## Timeline

### Immediate (now)
- Run patch: `python3 main/patch_add_pct_correct.py`
- Verify columns appear in Excel
- Update R analysis to use `pct_correct_*` as needed

### Future (when posteriors are recalculated)
- `MetricsCalculator.calculate_all_metrics()` will automatically generate `pct_correct_*`
- Excel files will be regenerated with final versions
- Patch script can be deleted (already integrated into metrics.py)

## Column Schema

### Wide Format
```
synthetic_data_wide.xlsx:
  [metadata] model, pse_true, jnd_true, subject_id, group, n_trials
  [B1] pse_40, pse_60, ..., pse_200, jnd_40, jnd_60, ..., jnd_200
  [B2] stimulus_center_40, ..., stimulus_spread_200, lat_entropy_*, asymmetry_*
  [B2b] pct_correct_40, pct_correct_60, ..., pct_correct_200  <- NEW
  [B3] posterior_sd_pse_*, posterior_sd_jnd_*
  [B4] pse_stability_block, jnd_stability_block, pse_auc, jnd_auc
  [B5] pse_error_pct, jnd_error_pct
```

### Long Format
```
synthetic_data_long.xlsx:
  [metadata] model, pse_true, jnd_true, subject_id, group
  trial_block (40, 60, 80, ..., 200)
  [metrics] pse, jnd, stimulus_center, stimulus_spread, lat_entropy, asymmetry
  pct_correct  <- NEW (unpivoted from wide format)
  posterior_sd_pse, posterior_sd_jnd, slope_mean, slope_sd
```

## Verification

After running patch, verify columns are present:

```bash
# Python
python3 << 'EOF'
import pandas as pd
df = pd.read_excel('data/output/synthetic_data_wide.xlsx')
pct_cols = [c for c in df.columns if c.startswith('pct_correct_')]
print(f"Found columns: {pct_cols}")
print(f"Sample values: {df[pct_cols].iloc[0]}")
EOF
```

## Notes

- **Multithreaded:** Uses `MAX_WORKERS` (default 8) from config
- **Reversible:** Patch reads GBF files fresh, so no data loss
- **Safe:** Creates backup columns, doesn't modify existing data
- **Incremental:** Can be run multiple times (skips if columns exist)

## Future Work

When posteriors are recalculated (separate task), all metrics including `pct_correct_*` will be regenerated from Python code, and this patch script will become unnecessary (but can be kept for reference).
