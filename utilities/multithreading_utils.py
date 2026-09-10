"""
Utilities for multithreaded simulation and analysis.
Handles thread pool management and synchronization for subject simulation.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

from analysis.io.converter import save_gbf_file, read_gbf_file


class SubjectSimulationTask:
    """Encapsulates a single subject simulation task."""

    def __init__(
        self,
        subject_id: int,
        group_idx: int,
        subj_in_group: int,
        pse: float,
        jnd: float,
        engine: Any,
        n_trials: int,
        ado_params: Dict,
        bis_params: Dict,
        fixed_trials_config: Dict,
        group_dir: Path,
        model_name: str,
        save_gbf: bool = True,
    ):
        self.subject_id = subject_id
        self.group_idx = group_idx
        self.subj_in_group = subj_in_group
        self.pse = pse
        self.jnd = jnd
        self.engine = engine
        self.n_trials = n_trials
        self.ado_params = ado_params
        self.bis_params = bis_params
        self.fixed_trials_config = fixed_trials_config
        self.group_dir = group_dir
        self.model_name = model_name
        self.save_gbf = save_gbf

        self.rows = None
        self.result_dict = None
        self.gbf_rows = None
        self.error = None

def run_subject_simulation(task: SubjectSimulationTask) -> SubjectSimulationTask:
    """
    Run a single subject simulation in a thread.

    Args:
        task: SubjectSimulationTask with all parameters

    Returns:
        Updated SubjectSimulationTask with results or error
    """
    try:
        # Simulate subject
        rows, result_dict = task.engine.simulate_subject(
            pse=task.pse,
            jnd=task.jnd,
            ntrials=task.n_trials,
            ado_params=task.ado_params,
            bis_params=task.bis_params,
            fixed_trials_config=task.fixed_trials_config,
            subject_id=task.subject_id,
        )

        # Convert to GBF format
        gbf_rows = [
            {
                'lat': row['lat'],
                'count': 1,
                'user_ans': row['user_ans'],
                'confl_magn': 0,
            }
            for row in rows
        ]

        # Save GBF file if requested
        if task.save_gbf:
            pse_int = int(round(task.pse))
            jnd_int = int(round(task.jnd))
            filename = f"S{task.subj_in_group:02d}_G{task.group_idx}_{pse_int}_{jnd_int}_{task.model_name}.txt"
            out_path = task.group_dir / filename
            save_gbf_file(gbf_rows, str(out_path))

        # Update subj to match the GBF filename (without extension)
        pse_int = int(round(task.pse))
        jnd_int = int(round(task.jnd))
        subj_name = f"S{task.subj_in_group:02d}_G{task.group_idx}_{pse_int}_{jnd_int}_{task.model_name}"
        result_dict['subj'] = subj_name
        # Store the integer pse/jnd used for simulation
        result_dict['pse'] = float(pse_int)
        result_dict['jnd'] = float(jnd_int)

        task.rows = rows
        task.result_dict = result_dict
        task.gbf_rows = gbf_rows

    except Exception as e:
        task.error = e

    return task



def parse_gbf_filename(stem: str) -> Tuple[float, float] | Tuple[None, None]:
    """
    Parse GBF filename to extract PSE and JND.
    Expected format: SXX_GZ_PSE_JND_MODEL
    Returns (pse, jnd) as floats, or (None, None) if parsing fails.
    """
    parts = stem.split('_')
    # Format: S02_G1_482_21_REL1 → parts = ['S02', 'G1', '482', '21', 'REL1']
    if len(parts) >= 5:
        try:
            return float(parts[2]), float(parts[3])
        except ValueError:
            pass
    return None, None


def load_gbf_files_for_group(group_dir: Path, model_name: str, offset: float = 500) -> list:
    """
    Load GBF files from a group directory.
    Extracts PSE/JND from filename (new format: SXX_GZ_PSE_JND_MODEL.txt).
    Returns list of (gbf_rows, subj, result_dict, rows, offset).
    """
    analysis_tasks = []
    old_format_count = 0

    for gbf_path in sorted(group_dir.glob("*.txt")):
        filename = gbf_path.stem
        pse_val, jnd_val = parse_gbf_filename(filename)

        if pse_val is None:
            old_format_count += 1

        try:
            gbf_rows = read_gbf_file(str(gbf_path))

            sigma_val = (jnd_val / np.log(3)) if jnd_val is not None else None
            result_dict = {
                'subj': filename,
                'model': model_name,
                'pse': pse_val,
                'jnd': jnd_val,
                'mu': pse_val,
                'sigma': sigma_val,
            }

            rows = [
                {
                    'lat': row['lat'],
                    'user_ans': row['user_ans'],
                    'res': 'true' if (row['user_ans'] == 1 and row['lat'] > offset) or
                                     (row['user_ans'] == 0 and row['lat'] < offset) else 'false'
                }
                for row in gbf_rows
            ]

            analysis_tasks.append((gbf_rows, filename, result_dict, rows, offset))
        except Exception as e:
            print(f"    Warning: Could not read GBF file {gbf_path}: {e}")

    if old_format_count > 0:
        print(f"    WARNING: {old_format_count} file(s) have old filename format — "
              f"PSE/JND will be estimated from progressive fit (less accurate).")

    return analysis_tasks


def backfill_skip_mode(group_results: list, analysis_tasks: list, n_trials_default: int = 200) -> None:
    """
    Fill n_trials/accuracy for subjects loaded from GBF files (skip mode).
    PSE/JND are read from the result_dict (already set by filename parsing).
    Falls back to progressive fit values (pse_N/jnd_N) if PSE/JND are missing,
    and prints a warning that results will be less accurate.

    Args:
        group_results: List of result dicts (modified in place)
        analysis_tasks: List of (gbf_rows, subj, result_dict, rows, offset) tuples
        n_trials_default: Fallback n_trials if GBF data not found
    """
    gbf_lookup = {subj: (gbf_rows, rows) for gbf_rows, subj, _, rows, _ in analysis_tasks}

    for result_dict in group_results:
        subj = result_dict.get('subj')
        gbf_data = gbf_lookup.get(subj)

        if gbf_data:
            gbf_rows_data, rows_data = gbf_data
            n_trials = len(gbf_rows_data)
            correct = sum(1 for r in rows_data if r.get('res') == 'true')
            accuracy = (correct / n_trials * 100) if n_trials > 0 else 0.0
        else:
            n_trials = n_trials_default
            accuracy = 0.0

        result_dict['n_trials'] = n_trials
        result_dict['accuracy'] = accuracy

        # If PSE/JND missing (old filename format), fall back to progressive fit
        if result_dict.get('pse') is None:
            pse_keys = sorted(
                [k for k in result_dict if k.startswith('pse_') and k[4:].isdigit()],
                key=lambda k: int(k[4:])
            )
            jnd_keys = sorted(
                [k for k in result_dict if k.startswith('jnd_') and k[4:].isdigit()],
                key=lambda k: int(k[4:])
            )
            final_pse = result_dict.get(pse_keys[-1]) if pse_keys else None
            final_jnd = result_dict.get(jnd_keys[-1]) if jnd_keys else None
            result_dict['pse'] = final_pse
            result_dict['jnd'] = final_jnd
            result_dict['mu'] = final_pse
            result_dict['sigma'] = (final_jnd / np.log(3)) if final_jnd is not None else None
            if final_pse is None:
                print(f"    WARNING: Could not determine PSE/JND for {subj} — "
                      f"results will be inaccurate.")


class MultiThreadedSimulationRunner:
    """Manages multithreaded simulation of subjects."""

    def __init__(self, max_workers: int|None = None):
        """
        Initialize the runner.

        Args:
            max_workers: Maximum number of worker threads. If None, uses CPU count.
        """
        self.max_workers = max_workers

    def run_subject_simulations(
        self,
        tasks: List[SubjectSimulationTask],
        verbose: bool = True,
    ) -> List[SubjectSimulationTask]:
        """
        Run multiple subject simulations in parallel.

        Args:
            tasks: List of SubjectSimulationTask objects
            verbose: Whether to print progress

        Returns:
            List of completed SubjectSimulationTask objects
        """
        completed_tasks = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(run_subject_simulation, task): task
                for task in tasks
            }

            completed = 0
            for future in as_completed(futures):
                completed += 1
                task = future.result()
                completed_tasks.append(task)

                if verbose:
                    status = "✓" if task.error is None else "✗"
                    print(f"  [{completed}/{len(tasks)}] Subject {task.subject_id}: {status}")
                    if task.error:
                        print(f"    Error: {task.error}")

        return completed_tasks
