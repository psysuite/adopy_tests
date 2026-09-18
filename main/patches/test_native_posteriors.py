#!/usr/bin/env python3
"""
Test script: compare posterior SD convergence across ABS1/REL1/REL2
using each model's NATIVE estimator (same model used in data generation).

Metric: posterior_sd_jnd for all three models.
REL2: max(pre, post) per trial block (conservative).

PARALLELIZED with ThreadPoolExecutor for speed.

Usage:
    cd /data/CODE/python/adopy_tests
    source venv/bin/activate
    python3 main/patches/test_native_posteriors.py 2>&1 | tee test_native_posteriors.log
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent  # main/patches -> main -> adopy_tests
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime

from analysis.core.extract_posterior_convergence import PosteriorExtractor
from analysis.io.converter import read_gbf_file
from main.config import TRIAL_BLOCKS, OFFSET

# ============================================================================
# CONFIG
# ============================================================================
DATA_ROOT  = PROJECT_ROOT / "data" / "output" / "sim_gridrnd"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "test_native_posteriors"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ['ABS1', 'REL1', 'REL2']
MAX_WORKERS = 20  # Parallel threads
MAX_SUBJECTS_PER_GROUP = 1  # None = all 20

# Thread-safe progress counter
progress_lock = threading.Lock()
progress_counter = {'count': 0}


# ============================================================================
# HELPERS
# ============================================================================

def get_pre_post_labels(stimuli: list, offset: int = OFFSET) -> list:
    return ['pre' if s < offset else 'post' for s in stimuli]


def jnd_sd_at_blocks_from_trajectory(jnd_sds: np.ndarray,
                                      trial_numbers: np.ndarray) -> list:
    """Extract jnd_sd values at each TRIAL_BLOCKS checkpoint."""
    values = []
    for block in TRIAL_BLOCKS:
        idx = int(np.argmax(trial_numbers >= block))
        if idx < len(trial_numbers):
            values.append(float(jnd_sds[idx]))
        else:
            values.append(np.nan)
    return values


def extract_jnd_sd_for_file(gbf_path: Path, model_type: str, jnd_true: float) -> dict | None:
    """
    Read one GBF file, extract posterior trajectory with native model.
    Returns jnd_sd at each trial block + stability_point + auc.

    REL2: uses max(pre, post) per block (conservative).
    
    stability_block: First block where |posterior_jnd_sd - jnd_true| / jnd_true < 10%
    (i.e., posterior converges within 10% of ground truth JND)
    """
    try:
        rows = read_gbf_file(str(gbf_path))
        if not rows:
            return None

        latencies = [float(r['lat']) for r in rows]
        responses  = [int(r['user_ans']) for r in rows]

        extractor = PosteriorExtractor(model_type=model_type, offset=OFFSET)

        if model_type == 'REL2':
            pre_post = get_pre_post_labels(latencies)
            posterior = extractor.extract_posterior_trajectory_rel2(
                latencies, responses, pre_post, metric='max'
            )
            trial_numbers = posterior['trial_numbers']
            jnd_sds       = posterior['jnd_sd']
        else:
            traj = extractor.extract_posterior_trajectory(latencies, responses)
            trial_numbers = traj['trial_numbers']
            jnd_sds       = traj['jnd_sd']

        # Values at TRIAL_BLOCKS
        jnd_sd_blocks = jnd_sd_at_blocks_from_trajectory(jnd_sds, trial_numbers)
        final_jnd_sd = float(jnd_sds[-1]) if len(jnd_sds) > 0 else np.nan

        # Stability point: first block where posterior JND within 10% of ground truth
        stability_block = None
        if jnd_true > 0:
            for block, sd in zip(TRIAL_BLOCKS, jnd_sd_blocks):
                if not np.isnan(sd):
                    error_pct = abs(sd - jnd_true) / jnd_true * 100
                    if error_pct < 10:  # Within 10% of ground truth
                        stability_block = block
                        break

        # AUC
        valid = [sd / (final_jnd_sd + 1e-10) for sd in jnd_sd_blocks if not np.isnan(sd)]
        auc = float(np.mean(valid)) if valid else np.nan

        return {
            'jnd_sd_blocks': jnd_sd_blocks,
            'stability_block': stability_block,
            'auc': auc,
            'final_jnd_sd': final_jnd_sd,
        }
    except Exception as e:
        print(f"  ✗ Error processing {gbf_path.name}: {e}")
        return None


def process_subject(gbf_path: Path, model_type: str, pse_center: float, jnd_center: float) -> dict | None:
    """Process one subject and print progress."""
    start_time = datetime.now()
    
    # Extract JND_true from filename
    # Format: S17_G2_481_42_REL1.txt -> PSE_true=481, JND_true=42
    filename_parts = gbf_path.stem.split('_')
    try:
        jnd_true = float(filename_parts[3])
    except (IndexError, ValueError):
        jnd_true = jnd_center  # Fallback to center if parsing fails
    
    result = extract_jnd_sd_for_file(gbf_path, model_type, jnd_true)
    
    if result is None:
        return None

    row = {
        'model': model_type,
        'pse_center': pse_center,
        'jnd_center': jnd_center,
        'jnd_true': jnd_true,
        'subject': gbf_path.stem,
        'stability_block': result['stability_block'],
        'auc': result['auc'],
        'final_jnd_sd': result['final_jnd_sd'],
    }
    for i, block in enumerate(TRIAL_BLOCKS):
        row[f'jnd_sd_{block}'] = result['jnd_sd_blocks'][i]

    # Update global counter with thread-safe print
    elapsed = (datetime.now() - start_time).total_seconds()
    with progress_lock:
        progress_counter['count'] += 1
        print(f"  [{progress_counter['count']}] {model_type} {gbf_path.stem}: "
              f"stab={result['stability_block']}, auc={result['auc']:.3f} "
              f"[{elapsed:.1f}s] ✓")

    return row


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print(f"NATIVE POSTERIOR TEST — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Parallelized with {MAX_WORKERS} threads")
    print(f"REL2: max(pre, post) per block (conservative)")
    print("=" * 70)

    all_rows = []

    for model in MODELS:
        model_dir = DATA_ROOT / model
        if not model_dir.exists():
            print(f"  ✗ {model}: directory not found")
            continue

        print(f"\n[{model}] Collecting files...")
        group_dirs = sorted(model_dir.glob("group_*_*"))
        
        # Build task list
        tasks = []
        for group_dir in group_dirs:
            parts = group_dir.name.split('_')
            pse_center = float(parts[1])
            jnd_center = float(parts[2])

            gbf_files = sorted(group_dir.glob("*.txt"))
            if MAX_SUBJECTS_PER_GROUP is not None:
                gbf_files = gbf_files[:MAX_SUBJECTS_PER_GROUP]

            for gbf_path in gbf_files:
                tasks.append((gbf_path, model, pse_center, jnd_center))

        print(f"  {len(tasks)} subjects to process")

        # Process in parallel
        print(f"  Starting parallel processing...")
        n_success = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_subject, gbf, mdl, pse, jnd): (gbf, mdl)
                for gbf, mdl, pse, jnd in tasks
            }

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_rows.append(result)
                    n_success += 1

        print(f"  {n_success}/{len(tasks)} subjects processed successfully")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("\n✗ No data. Check DATA_ROOT.")
        return

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY (mean ± SD)")
    print("=" * 70)
    print(f"\n{'Model':<8} {'Stability block':<22} {'AUC (norm.)':<20} {'Final JND SD':<18} N")
    print("-" * 75)

    for model in MODELS:
        sub = df[df['model'] == model]
        if len(sub) == 0:
            continue
        stab = sub['stability_block'].dropna()
        auc  = sub['auc'].dropna()
        fsd  = sub['final_jnd_sd'].dropna()
        print(
            f"{model:<8} "
            f"{stab.mean():.1f} ± {stab.std():.1f}{'':>6}"
            f"{auc.mean():.3f} ± {auc.std():.3f}{'':>4}"
            f"{fsd.mean():.2f} ± {fsd.std():.2f}{'':>4}"
            f"{len(sub)}"
        )

    # ========================================================================
    # PLOT 1: JND posterior SD evolution
    # ========================================================================
    colors = {'ABS1': '#1f77b4', 'REL1': '#ff7f0e', 'REL2': '#2ca02c'}

    fig, ax = plt.subplots(figsize=(9, 6))
    for model in MODELS:
        sub = df[df['model'] == model]
        if len(sub) == 0:
            continue
        means, ses = [], []
        for block in TRIAL_BLOCKS:
            vals = sub[f'jnd_sd_{block}'].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            ses.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        means, ses = np.array(means), np.array(ses)
        ax.plot(TRIAL_BLOCKS, means, marker='o', color=colors[model],
                label=model, linewidth=2)
        ax.fill_between(TRIAL_BLOCKS, means - ses, means + ses,
                        alpha=0.2, color=colors[model])

    ax.set_xlabel('Trial Block')
    ax.set_ylabel('posterior_sd_jnd')
    ax.set_title('JND Posterior SD Evolution\n(native estimator per model; REL2 = max(pre,post))')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = OUTPUT_DIR / 'native_jnd_sd_evolution.png'
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Plot: {p1}")

    # ========================================================================
    # PLOT 2: bar chart
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, col, title, ylabel in [
        (axes[0], 'stability_block', 'JND Stability Point\n(first block < 10% final SD)', 'Trial Block'),
        (axes[1], 'auc',             'JND Convergence AUC\n(lower = faster)', 'Normalized AUC'),
    ]:
        vals  = [df[df['model'] == m][col].dropna().values for m in MODELS]
        means = [v.mean() if len(v) else np.nan for v in vals]
        sems  = [v.std() / np.sqrt(len(v)) if len(v) > 1 else 0 for v in vals]

        bars = ax.bar(MODELS, means, yerr=sems, capsize=6,
                      color=[colors[m] for m in MODELS], alpha=0.85)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, mean in zip(bars, means):
            if not np.isnan(mean):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max([s for s in sems if not np.isnan(s)], default=1) * 0.15,
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    p2 = OUTPUT_DIR / 'native_jnd_summary_bars.png'
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot: {p2}")

    # ========================================================================
    # PLOT 3: by JND group
    # ========================================================================
    jnd_groups = sorted(df['jnd_center'].unique())
    fig, axes = plt.subplots(1, len(jnd_groups), figsize=(5 * len(jnd_groups), 5), sharey=True)
    if len(jnd_groups) == 1:
        axes = [axes]

    for ax, jnd_g in zip(axes, jnd_groups):
        sub_g = df[df['jnd_center'] == jnd_g]
        for model in MODELS:
            sub = sub_g[sub_g['model'] == model]
            if len(sub) == 0:
                continue
            means, ses = [], []
            for block in TRIAL_BLOCKS:
                vals = sub[f'jnd_sd_{block}'].dropna()
                means.append(vals.mean() if len(vals) else np.nan)
                ses.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            means, ses = np.array(means), np.array(ses)
            ax.plot(TRIAL_BLOCKS, means, marker='o', color=colors[model],
                    label=model, linewidth=2)
            ax.fill_between(TRIAL_BLOCKS, means - ses, means + ses,
                            alpha=0.2, color=colors[model])
        ax.set_title(f'JND_true = {jnd_g}ms')
        ax.set_xlabel('Trial Block')
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel('posterior_sd_jnd')
        ax.legend(fontsize=8)

    fig.suptitle('JND Posterior SD by True JND Group', fontsize=13)
    plt.tight_layout()
    p3 = OUTPUT_DIR / 'native_jnd_sd_by_jnd_group.png'
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot: {p3}")

    # Save CSV
    csv_path = OUTPUT_DIR / 'native_posterior_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV:  {csv_path}")

    print(f"\n{'='*70}")
    print(f"✓ COMPLETED — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✓ Total subjects processed: {len(df)}")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
