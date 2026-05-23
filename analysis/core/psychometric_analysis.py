"""
Psychometric analysis functions for temporal bisection task data.

Provides functions to analyze subject responses, fit psychometric functions,
generate visualizations, and consolidate results into reports.

Includes asymmetry analysis to detect stimulus presentation bias.
"""

import os
import io
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import logging

from utilities.logging_config import log_subject_error
from scipy.stats import norm as _scipy_norm

def analyze_subject_results(
    exp,
    subject_row: pd.Series,
    output_dir: str,
    file_prefix: str,
    rows: List[Dict] = None,
    offset: float = 500
) -> Dict[str, Any]:
    """
    Analyze psychometric results for a single subject.
    
    Executes all analysis steps for a subject after trial generation:
    - Computes statistics via exp.print_statistics()
    - Fits psychometric function via exp.gausFit(10)
    - Generates psychometric plot via exp.plot_psychometric()
    - Derives metrics (JND, accuracy, asymmetry)
    - Returns structured result dictionary
    
    Args:
        exp: ADOpy wrapper instance with completed trials (BISAbsADOpyWrapper)
        subject_row: pandas Series with subject metadata containing:
            - 'subj': Subject identifier (str)
            - 'pse': Point of Subjective Equality (float)
        output_dir: Directory path for output files (str)
        file_prefix: Prefix for generated files (str)
        rows: List of trial dicts with 'lat' key for asymmetry calculation (optional)
        offset: Offset latency for asymmetry calculation (default 500ms)
        
    Returns:
        Dictionary containing analysis results with keys:
        - 'subj': Subject identifier
        - 'mu': Fitted mean from gausFit
        - 'sigma': Fitted standard deviation from gausFit
        - 'jnd': Just Noticeable Difference (sigma * 0.6745)
        - 'pse': Point of Subjective Equality
        - 'n_trials': Number of trials
        - 'accuracy': Accuracy percentage (0-100)
        - 'asymmetry_index': Stimulus presentation asymmetry [-1, 1]
        - 'n_before_offset': Number of stimuli before offset
        - 'n_after_offset': Number of stimuli after offset
        - 'pct_before': Percentage of stimuli before offset
        - 'pct_after': Percentage of stimuli after offset
        - 'status': 'success' or 'error'
        
    Raises:
        TypeError: If exp is not a valid ADOpy wrapper instance
        TypeError: If subject_row is not a pandas Series
        ValueError: If subject_row missing required fields
        ValueError: If output_dir is not a valid directory
        ValueError: If file_prefix is empty or invalid
    """
    # Validate exp parameter
    if exp is None:
        raise TypeError("exp cannot be None")
    
    if not hasattr(exp, 'print_statistics') or not callable(exp.print_statistics):
        raise TypeError("exp must be a valid ADOpy wrapper instance with print_statistics method")
    
    if not hasattr(exp, 'gausFit') or not callable(exp.gausFit):
        raise TypeError("exp must be a valid ADOpy wrapper instance with gausFit method")
    
    if not hasattr(exp, 'plot_psychometric') or not callable(exp.plot_psychometric):
        raise TypeError("exp must be a valid ADOpy wrapper instance with plot_psychometric method")
    
    # Validate subject_row parameter
    if not isinstance(subject_row, pd.Series):
        raise TypeError(f"subject_row must be a pandas Series, got {type(subject_row)}")
    
    required_fields = {'subj', 'pse'}
    missing_fields = required_fields - set(subject_row.index)
    if missing_fields:
        raise ValueError(f"subject_row missing required fields: {missing_fields}")
    
    # Validate field types
    if not isinstance(subject_row['subj'], str):
        raise ValueError(f"subject_row['subj'] must be str, got {type(subject_row['subj'])}")

    if not isinstance(subject_row['pse'], (int, float)):
        raise ValueError(f"subject_row['pse'] must be numeric, got {type(subject_row['pse'])}")
    
    # Validate output_dir parameter
    if not isinstance(output_dir, str):
        raise ValueError(f"output_dir must be str, got {type(output_dir)}")
    
    output_path = Path(output_dir)
    if not output_path.exists():
        raise ValueError(f"output_dir does not exist: {output_dir}")
    
    if not output_path.is_dir():
        raise ValueError(f"output_dir is not a directory: {output_dir}")
    
    # Validate file_prefix parameter
    if not isinstance(file_prefix, str):
        raise ValueError(f"file_prefix must be str, got {type(file_prefix)}")
    
    if not file_prefix or file_prefix.strip() == "":
        raise ValueError("file_prefix cannot be empty or whitespace-only")
    
    # All validations passed - proceed with statistics computation
    
    # Step 1: Call exp.print_statistics() and capture output
    old_stdout = sys.stdout
    sys.stdout = statistics_buffer = io.StringIO()
    
    try:
        exp.print_statistics()
    finally:
        sys.stdout = old_stdout
    
    statistics_output = statistics_buffer.getvalue()
    
    # Step 2: Extract trial count from statistics output
    n_trials = None
    for line in statistics_output.split('\n'):
        if 'Total trials:' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                try:
                    n_trials = int(parts[1].strip())
                except (ValueError, IndexError):
                    n_trials = None
            break
    
    if n_trials is None:
        if hasattr(exp, 'stimuli_ms'):
            n_trials = len(exp.stimuli_ms)
        else:
            n_trials = 0
    
    # Step 3: Fit psychometric function
    mu, sigma = exp.gausFit(10)
    
    if sigma <= 0:
        raise ValueError(f"Fitted sigma must be positive, got {sigma}")
    
    # Step 4: Extract subject metadata
    subj = subject_row['subj']
    pse = float(subject_row['pse'])
    
    # Step 5: Generate psychometric plot
    plot_filename = generate_plot_filename(file_prefix, subj)
    plot_filepath = os.path.join(output_dir, plot_filename)
    
    exp.plot_psychometric(plot_filepath, 10)
    
    if not os.path.exists(plot_filepath):
        raise FileNotFoundError(f"Plot file was not created at: {plot_filepath}")
    
    # Step 6: Calculate derived metrics
    jnd = sigma * 0.6745
    accuracy = np.mean(exp.successes) * 100 if hasattr(exp, 'successes') else 0.0
    
    # Step 7: Calculate asymmetry metrics
    asymmetry_data = calculate_asymmetry_metrics(rows, offset) if rows else {
        'asymmetry_index': None,
        'n_before_offset': None,
        'n_after_offset': None,
        'pct_before': None,
        'pct_after': None,
    }
    
    # Step 8: Build and return result dictionary
    result = {
        'subj': subj,
        'mu': mu,
        'sigma': sigma,
        'jnd': jnd,
        'pse': pse,
        'n_trials': n_trials,
        'accuracy': accuracy,
        'asymmetry_index': asymmetry_data['asymmetry_index'],
        'n_before_offset': asymmetry_data['n_before_offset'],
        'n_after_offset': asymmetry_data['n_after_offset'],
        'pct_before': asymmetry_data['pct_before'],
        'pct_after': asymmetry_data['pct_after'],
        'status': 'success'
    }
    
    return result



def generate_plot_filename(
    file_prefix: str,
    subj: str
) -> str:
    """
    Generate psychometric plot filename with correct naming convention.
    
    Creates a filename following the pattern:
    {file_prefix}_{subj}_psychometric_plot.png
    
    Args:
        file_prefix: Prefix for the filename (str)
        subj: Subject identifier (str)
        
    Returns:
        Filename string following the naming convention (str)
        
    Raises:
        TypeError: If any parameter has incorrect type
        ValueError: If any parameter is empty or invalid
    """
    if not isinstance(file_prefix, str):
        raise TypeError(f"file_prefix must be str, got {type(file_prefix)}")
    
    if not file_prefix or file_prefix.strip() == "":
        raise ValueError("file_prefix cannot be empty or whitespace-only")
    
    if not isinstance(subj, str):
        raise TypeError(f"subj must be str, got {type(subj)}")
    
    if not subj or subj.strip() == "":
        raise ValueError("subj cannot be empty or whitespace-only")
    
    filename = f"{file_prefix}_{subj}_psychometric_plot.png"
    
    return filename



def safe_analyze_subject(
    exp,
    subject_row: pd.Series,
    output_dir: str,
    file_prefix: str,
    logger: logging.Logger,
    rows: List[Dict] = None,
    offset: float = 500
) -> Dict[str, Any]:
    """
    Safely analyze psychometric results for a single subject with error handling.
    
    Wraps analyze_subject_results() with try-catch error handling to ensure
    that failures for individual subjects do not prevent processing of other subjects.
    
    Args:
        exp: ADOpy wrapper instance with completed trials (BISAbsADOpyWrapper)
        subject_row: pandas Series with subject metadata containing:
            - 'subj': Subject identifier (str)
            - 'pse': Point of Subjective Equality (float)
        output_dir: Directory path for output files (str)
        file_prefix: Prefix for generated files (str)
        logger: Logger instance for error logging (logging.Logger)
        rows: List of trial dicts with 'lat' key for asymmetry calculation (optional)
        offset: Offset latency for asymmetry calculation (default 500ms)
        
    Returns:
        Dictionary containing analysis results with keys:
        - 'subj': Subject identifier
        - 'mu': Fitted mean from gausFit (or None on error)
        - 'sigma': Fitted standard deviation from gausFit (or None on error)
        - 'jnd': Just Noticeable Difference (or None on error)
        - 'pse': Point of Subjective Equality (or None on error)
        - 'n_trials': Number of trials (or None on error)
        - 'accuracy': Accuracy percentage (or None on error)
        - 'asymmetry_index': Asymmetry index (or None on error)
        - 'status': 'success' or 'error'
        
    Notes:
        - On error, returns result dict with status='error' and other fields set to None
        - Logs errors with format: [subject_id] [operation] error_message
        - Continues processing subsequent subjects even if this subject fails
        - Captures all exception types to ensure robustness
    """
    try:
        result = analyze_subject_results(exp, subject_row, output_dir, file_prefix, rows, offset)
        return result
        
    except Exception as e:
        subject_id = subject_row.get('subj', 'unknown') if isinstance(subject_row, pd.Series) else 'unknown'
        
        error_str = str(e)
        if 'print_statistics' in error_str or 'statistics' in error_str.lower():
            operation = 'print_statistics'
        elif 'gausFit' in error_str or 'fit' in error_str.lower():
            operation = 'gausFit'
        elif 'plot_psychometric' in error_str or 'plot' in error_str.lower():
            operation = 'plot_psychometric'
        else:
            operation = 'analyze_subject_results'
        
        log_subject_error(logger, subject_id, operation, error_str)
        
        error_result = {
            'subj': subject_row.get('subj', 'unknown') if isinstance(subject_row, pd.Series) else 'unknown',
            'mu': None,
            'sigma': None,
            'jnd': None,
            'pse': subject_row.get('pse', None) if isinstance(subject_row, pd.Series) else None,
            'n_trials': None,
            'accuracy': None,
            'asymmetry_index': None,
            'n_before_offset': None,
            'n_after_offset': None,
            'pct_before': None,
            'pct_after': None,
            'status': 'error'
        }
        
        return error_result


def consolidate_results(
    results_list: list,
    output_dir: str,
    file_prefix: str
) -> str:
    """
    Consolidate analysis results from all subjects into an Excel report.

    Creates a consolidated Excel file containing results for all subjects with
    one row per subject and columns for all key metrics (mu, sigma, jnd, pse,
    accuracy, asymmetry, etc.).

    Args:
        results_list: List of result dictionaries from analyze_subject_results(),
            where each dictionary contains keys:
            - 'subj': Subject identifier (str)
            - 'mu': Fitted mean (float)
            - 'sigma': Fitted standard deviation (float)
            - 'jnd': Just Noticeable Difference (float)
            - 'pse': Point of Subjective Equality (float)
            - 'n_trials': Number of trials (int)
            - 'accuracy': Accuracy percentage (float)
            - 'asymmetry_index': Asymmetry index (float)
            - 'n_before_offset': Count before offset (int)
            - 'n_after_offset': Count after offset (int)
            - 'pct_before': Percentage before offset (float)
            - 'pct_after': Percentage after offset (float)
            - 'status': 'success' or 'error' (str)
        output_dir: Directory path where Excel file will be saved (str)
        file_prefix: Prefix for the Excel filename (str)

    Returns:
        Path to the created Excel file (str)

    Raises:
        TypeError: If results_list is not a list
        TypeError: If output_dir is not a string
        TypeError: If file_prefix is not a string
        ValueError: If results_list is empty
        ValueError: If output_dir does not exist or is not a directory
        ValueError: If file_prefix is empty or whitespace-only
    """
    if not isinstance(results_list, list):
        raise TypeError(f"results_list must be a list, got {type(results_list)}")

    for i, item in enumerate(results_list):
        if not isinstance(item, dict):
            raise TypeError(f"results_list[{i}] must be a dict, got {type(item)}")

    if not isinstance(output_dir, str):
        raise ValueError(f"output_dir must be str, got {type(output_dir)}")

    output_path = Path(output_dir)
    if not output_path.exists():
        raise ValueError(f"output_dir does not exist: {output_dir}")

    if not output_path.is_dir():
        raise ValueError(f"output_dir is not a directory: {output_dir}")

    if not isinstance(file_prefix, str):
        raise ValueError(f"file_prefix must be str, got {type(file_prefix)}")

    if not file_prefix or file_prefix.strip() == "":
        raise ValueError("file_prefix cannot be empty or whitespace-only")

    if len(results_list) == 0:
        required_columns = ['subj', 'mu', 'sigma', 'jnd', 'pse', 'n_trials', 'accuracy',
                           'asymmetry_index', 'n_before_offset', 'n_after_offset',
                           'pct_before', 'pct_after']
        df = pd.DataFrame(columns=required_columns)
    else:
        df = pd.DataFrame(results_list)

        required_columns = {'subj', 'mu', 'sigma', 'jnd', 'pse', 'n_trials', 'accuracy'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Results missing required columns: {missing_columns}")

        # Drop unwanted columns
        for col in ['subject_id', 'status']:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Reorder columns: put core metrics first, then lat_entropy columns
        core_cols = ['subj', 'mu', 'sigma', 'jnd', 'pse', 'n_trials', 'accuracy',
                     'asymmetry_index', 'n_before_offset', 'n_after_offset',
                     'pct_before', 'pct_after']
        
        # Get all lat_entropy columns (lat_entropy_40, lat_entropy_60, etc.)
        lat_entropy_cols = sorted([col for col in df.columns if col.startswith('lat_entropy_')])
        
        # Get all other columns (progressive metrics, stimulus metrics, etc.)
        other_cols = [col for col in df.columns if col not in core_cols and col not in lat_entropy_cols]
        
        # Reorder: core + lat_entropy + others
        ordered_cols = [col for col in core_cols if col in df.columns] + lat_entropy_cols + other_cols
        df = df[ordered_cols]

    excel_filename = f"{file_prefix}_results_summary.xlsx"
    excel_filepath = os.path.join(output_dir, excel_filename)

    df.to_excel(excel_filepath, index=False, sheet_name='Results')

    if not os.path.exists(excel_filepath):
        raise FileNotFoundError(f"Excel file was not created at: {excel_filepath}")

    return excel_filepath



def create_summary_report(results_list: list) -> Dict[str, Any]:
    """
    Create a summary report of analysis results.
    
    Counts successful and failed analyses from the results list and calculates
    the overall success rate percentage.
    
    Args:
        results_list: List of result dictionaries from analyze_subject_results(),
            where each dictionary contains a 'status' key with value 'success' or 'error'
            
    Returns:
        Dictionary containing summary statistics with keys:
        - 'total_subjects': Total number of subjects processed (int)
        - 'successful_analyses': Number of successful analyses (int)
        - 'failed_analyses': Number of failed analyses (int)
        - 'success_rate': Success rate as percentage (float, 0-100)
    """
    if not isinstance(results_list, list):
        raise TypeError(f"results_list must be a list, got {type(results_list)}")
    
    if len(results_list) == 0:
        raise ValueError("results_list cannot be empty")
    
    for i, item in enumerate(results_list):
        if not isinstance(item, dict):
            raise TypeError(f"results_list[{i}] must be a dict, got {type(item)}")
        
        if 'status' not in item:
            raise ValueError(f"results_list[{i}] missing required 'status' key")
    
    successful_count = sum(1 for result in results_list if result.get('status') == 'success')
    failed_count = sum(1 for result in results_list if result.get('status') == 'error')
    total_count = len(results_list)
    
    success_rate = (successful_count / total_count) * 100 if total_count > 0 else 0.0
    
    summary = {
        'total_subjects': total_count,
        'successful_analyses': successful_count,
        'failed_analyses': failed_count,
        'success_rate': success_rate
    }
    
    return summary


def print_summary_report(results_list: list) -> None:
    """
    Print a formatted summary report to console.
    
    Displays a formatted summary of analysis results with clear section headers
    showing total subjects processed, successful analyses, failed analyses, and
    the overall success rate percentage.
    
    Args:
        results_list: List of result dictionaries from analyze_subject_results(),
            where each dictionary contains a 'status' key with value 'success' or 'error'
    """
    summary = create_summary_report(results_list)
    
    total_subjects = summary['total_subjects']
    successful_analyses = summary['successful_analyses']
    failed_analyses = summary['failed_analyses']
    success_rate = summary['success_rate']
    
    print("========================================")
    print("ANALYSIS SUMMARY")
    print("========================================")
    print(f"Total subjects processed: {total_subjects}")
    print(f"Successful analyses: {successful_analyses}")
    print(f"Failed analyses: {failed_analyses}")
    print(f"Success rate: {success_rate:.2f}%")
    print("========================================")


def add_group_stats_to_excel(excel_filepath: str, skip_cols: List[str] = None) -> None:
    """
    Append GROUP_mean and GROUP_std rows to an existing results Excel file.

    Args:
        excel_filepath: Full path to the Excel file produced by consolidate_results()
        skip_cols: Column names to exclude from statistics (default: subj, status)
    """
    if skip_cols is None:
        skip_cols = ['subj', 'status']

    df = pd.read_excel(excel_filepath)

    group_mean = {'subj': 'GROUP_mean'}
    group_std  = {'subj': 'GROUP_std'}

    for col in df.columns:
        if col in skip_cols:
            continue
        try:
            group_mean[col] = df[col].mean()
            group_std[col]  = df[col].std()
        except Exception:
            group_mean[col] = None
            group_std[col]  = None

    df_combined = pd.concat([df, pd.DataFrame([group_mean, group_std])], ignore_index=True)
    df_combined.to_excel(excel_filepath, index=False, sheet_name='Results')


def calculate_asymmetry_metrics(rows: List[Dict], offset: float = 500) -> Dict[str, Any]:
    """
    Calculate asymmetry metrics from trial rows.

    Computes how many stimuli were presented before vs after the offset,
    and derives an asymmetry index.

    Args:
        rows: List of trial dicts with 'lat' key (stimulus latency in ms)
        offset: Threshold latency (default 500ms)

    Returns:
        Dictionary with keys:
        - 'asymmetry_index': (n_after - n_before) / total, range [-1, 1]
        - 'n_before_offset': Number of stimuli < offset
        - 'n_after_offset': Number of stimuli >= offset
        - 'pct_before': Percentage of stimuli < offset
        - 'pct_after': Percentage of stimuli >= offset

    Notes:
        - asymmetry_index = 0: perfectly balanced (50-50)
        - asymmetry_index > 0: bias toward after (more stimuli after offset)
        - asymmetry_index < 0: bias toward before (more stimuli before offset)
        - asymmetry_index = ±1: all stimuli on one side
    """
    if not rows:
        return {
            'asymmetry_index': 0.0,
            'n_before_offset': 0,
            'n_after_offset': 0,
            'pct_before': 0.0,
            'pct_after': 0.0,
        }

    latencies = np.array([row['lat'] for row in rows])

    n_before = np.sum(latencies < offset)
    n_after = np.sum(latencies > offset)
    total = n_before + n_after  # Exclude trials exactly at offset

    if total == 0:
        return {
            'asymmetry_index': 0.0,
            'n_before_offset': 0,
            'n_after_offset': 0,
            'pct_before': 0.0,
            'pct_after': 0.0,
        }

    asymmetry_index = (n_after - n_before) / total

    return {
        'asymmetry_index': float(asymmetry_index),
        'n_before_offset': int(n_before),
        'n_after_offset': int(n_after),
        'pct_before': float(100 * n_before / total),
        'pct_after': float(100 * n_after / total),
    }


def calculate_progressive_asymmetry(rows: List[Dict], offset: float = 500, trial_counts: List[int] = None) -> Dict[int, float]:
    """
    Calculate asymmetry index at progressive trial counts.
    
    Args:
        rows: List of trial dicts with 'lat' key
        offset: Threshold latency (default 500ms)
        trial_counts: List of trial counts to calculate at (default: [40, 60, 80, 100, 120, 140, 160, 180, 200])
        
    Returns:
        Dictionary mapping trial_count to asymmetry_index
    """
    if trial_counts is None:
        trial_counts = [40, 60, 80, 100, 120, 140, 160, 180, 200]
    
    if not rows:
        return {n: 0.0 for n in trial_counts}
    
    result = {}
    
    for n_trials in trial_counts:
        if n_trials > len(rows):
            continue
        
        # Get first n_trials
        subset_rows = rows[:n_trials]
        asymmetry_data = calculate_asymmetry_metrics(subset_rows, offset)
        result[n_trials] = asymmetry_data['asymmetry_index']
    
    return result


def calculate_progressive_stimulus_metrics(rows: List[Dict], trial_counts: List[int] = None) -> Dict[str, Dict[int, float]]:
    """
    Calculate stimulus distribution metrics at progressive trial counts.
    
    Computes cumulative statistics of stimulus latencies to validate that
    the presented stimuli follow the intended PSE/JND parameters.
    
    Args:
        rows: List of trial dicts with 'lat' key (stimulus latency in ms)
        trial_counts: List of trial counts to calculate at (default: [40, 60, 80, 100, 120, 140, 160, 180, 200])
        
    Returns:
        Dictionary with keys:
        - 'stimulus_center': Dict mapping trial_count to mean latency
        - 'stimulus_spread': Dict mapping trial_count to std of latencies
        - 'stimulus_min': Dict mapping trial_count to min latency
        - 'stimulus_max': Dict mapping trial_count to max latency
        - 'bimodality_index': Dict mapping trial_count to bimodality measure
        
    Notes:
        - stimulus_center should correlate with PSE parameter
        - stimulus_spread should correlate with JND parameter
        - bimodality_index measures how much the distribution has two peaks
          (higher values = more bimodal, lower values = more unimodal)
    """
    if trial_counts is None:
        trial_counts = [40, 60, 80, 100, 120, 140, 160, 180, 200]
    
    if not rows:
        return {
            'stimulus_center': {n: 0.0 for n in trial_counts},
            'stimulus_spread': {n: 0.0 for n in trial_counts},
            'stimulus_min': {n: 0.0 for n in trial_counts},
            'stimulus_max': {n: 0.0 for n in trial_counts},
            'bimodality_index': {n: 0.0 for n in trial_counts},
        }
    
    result = {
        'stimulus_center': {},
        'stimulus_spread': {},
        'stimulus_min': {},
        'stimulus_max': {},
        'bimodality_index': {},
    }
    
    for n_trials in trial_counts:
        if n_trials > len(rows):
            continue
        
        # Get first n_trials
        subset_rows = rows[:n_trials]
        latencies = np.array([row['lat'] for row in subset_rows])
        
        # Basic statistics
        result['stimulus_center'][n_trials] = float(np.mean(latencies))
        result['stimulus_spread'][n_trials] = float(np.std(latencies))
        result['stimulus_min'][n_trials] = float(np.min(latencies))
        result['stimulus_max'][n_trials] = float(np.max(latencies))
        
        # Bimodality index: simple measure based on histogram peaks
        # Create histogram with 20 bins
        hist, bin_edges = np.histogram(latencies, bins=20)
        
        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append((i, hist[i]))
        
        # Bimodality measure: if we have 2+ peaks, compute ratio of top 2
        if len(peaks) >= 2:
            peaks.sort(key=lambda x: x[1], reverse=True)
            peak1, peak2 = peaks[0][1], peaks[1][1]
            # Ratio of second peak to first peak (0 to 1)
            bimodality = float(peak2 / peak1) if peak1 > 0 else 0.0
        else:
            bimodality = 0.0
        
        result['bimodality_index'][n_trials] = bimodality
    
    return result


def calculate_latency_statistics(latencies: np.ndarray) -> Dict[str, float]:
    """
    Calculate descriptive statistics for stimulus latencies.
    
    Renamed to match simulation metrics:
    - stimulus_center (SC): mean of stimulus latencies
    - stimulus_spread (SS): standard deviation of stimulus latencies
    - lat_entropy: Shannon entropy of stimulus latency distribution
    - lat_range: range of stimulus latencies

    Args:
        latencies: Array of stimulus presentation times (milliseconds)

    Returns:
        Dictionary with keys: stimulus_center, stimulus_spread, lat_range, lat_entropy (Shannon, 10 bins)
    """
    if len(latencies) == 0:
        return {'stimulus_center': np.nan, 'stimulus_spread': np.nan, 'lat_range': np.nan, 'lat_entropy': np.nan}

    stimulus_center = float(np.mean(latencies))
    stimulus_spread = float(np.std(latencies, ddof=0))
    lat_range = float(np.max(latencies) - np.min(latencies))

    if len(latencies) > 1 and lat_range > 0:
        counts, _ = np.histogram(latencies, bins=10)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        lat_entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        lat_entropy = 0.0

    return {'stimulus_center': stimulus_center, 'stimulus_spread': stimulus_spread, 'lat_range': lat_range, 'lat_entropy': lat_entropy}

