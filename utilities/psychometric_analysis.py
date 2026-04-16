"""
Psychometric analysis functions for temporal bisection task data.

Provides functions to analyze subject responses, fit psychometric functions,
generate visualizations, and consolidate results into reports.
"""

import os
import io
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import logging

from utilities.logging_config import log_subject_error


def analyze_subject_results(
    exp,
    subject_row: pd.Series,
    output_dir: str,
    file_prefix: str
) -> Dict[str, Any]:
    """
    Analyze psychometric results for a single subject.
    
    Executes all analysis steps for a subject after trial generation:
    - Computes statistics via exp.print_statistics()
    - Fits psychometric function via exp.gausFit(10)
    - Generates psychometric plot via exp.plot_psychometric()
    - Derives metrics (JND, accuracy)
    - Returns structured result dictionary
    
    Args:
        exp: ADOpy wrapper instance with completed trials (BISAbsADOpyWrapper)
        subject_row: pandas Series with subject metadata containing:
            - 'subj': Subject identifier (str)
            - 'pse': Point of Subjective Equality (float)
        output_dir: Directory path for output files (str)
        file_prefix: Prefix for generated files (str)
        
    Returns:
        Dictionary containing analysis results with keys:
        - 'subj': Subject identifier
        - 'mu': Fitted mean from gausFit
        - 'sigma': Fitted standard deviation from gausFit
        - 'jnd': Just Noticeable Difference (sigma * 0.6745)
        - 'pse': Point of Subjective Equality
        - 'n_trials': Number of trials
        - 'accuracy': Accuracy percentage (0-100)
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
    # Redirect stdout to capture the statistics output
    old_stdout = sys.stdout
    sys.stdout = statistics_buffer = io.StringIO()
    
    try:
        exp.print_statistics()
    finally:
        sys.stdout = old_stdout
    
    statistics_output = statistics_buffer.getvalue()
    
    # Step 2: Extract trial count from statistics output
    # The statistics output contains "Total trials: N" line
    n_trials = None
    for line in statistics_output.split('\n'):
        if 'Total trials:' in line:
            # Extract the number from the line
            parts = line.split(':')
            if len(parts) >= 2:
                try:
                    n_trials = int(parts[1].strip())
                except (ValueError, IndexError):
                    n_trials = None
            break
    
    # If we couldn't extract from output, try to get from exp object
    if n_trials is None:
        if hasattr(exp, 'stimuli_ms'):
            n_trials = len(exp.stimuli_ms)
        else:
            n_trials = 0
    
    # Step 3: Fit psychometric function
    mu, sigma = exp.gausFit(10)
    
    # Validate that sigma is positive
    if sigma <= 0:
        raise ValueError(f"Fitted sigma must be positive, got {sigma}")
    
    # Step 4: Extract subject metadata
    subj = subject_row['subj']
    pse = float(subject_row['pse'])
    
    # Step 5: Generate psychometric plot
    plot_filename = generate_plot_filename(file_prefix, subj)
    plot_filepath = os.path.join(output_dir, plot_filename)
    
    # Call exp.plot_psychometric() with full path and binning parameter
    exp.plot_psychometric(plot_filepath, 10)
    
    # Verify that the plot file was created
    if not os.path.exists(plot_filepath):
        raise FileNotFoundError(f"Plot file was not created at: {plot_filepath}")
    
    # Step 6: Calculate derived metrics
    jnd = sigma * 0.6745
    accuracy = np.mean(exp.successes) * 100 if hasattr(exp, 'successes') else 0.0
    
    # Step 7: Build and return result dictionary
    result = {
        'subj': subj,
        'mu': mu,
        'sigma': sigma,
        'jnd': jnd,
        'pse': pse,
        'n_trials': n_trials,
        'accuracy': accuracy,
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
        
    Examples:
        >>> generate_plot_filename("M1ABS", "subj001", 25, "M", "BISA")
        'M1ABS_subj001_25_M_BISA_psychometric_plot.png'
        
        >>> generate_plot_filename("TEST", "s002", 30, "F", "BISV")
        'TEST_s002_30_F_BISV_psychometric_plot.png'
    """
    # Validate file_prefix
    if not isinstance(file_prefix, str):
        raise TypeError(f"file_prefix must be str, got {type(file_prefix)}")
    
    if not file_prefix or file_prefix.strip() == "":
        raise ValueError("file_prefix cannot be empty or whitespace-only")
    
    # Validate subj
    if not isinstance(subj, str):
        raise TypeError(f"subj must be str, got {type(subj)}")
    
    if not subj or subj.strip() == "":
        raise ValueError("subj cannot be empty or whitespace-only")
    
    # Generate filename following the pattern
    filename = f"{file_prefix}_{subj}_psychometric_plot.png"
    
    return filename



def safe_analyze_subject(
    exp,
    subject_row: pd.Series,
    output_dir: str,
    file_prefix: str,
    logger: logging.Logger
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
        
    Returns:
        Dictionary containing analysis results with keys:
        - 'subj': Subject identifier
        - 'mu': Fitted mean from gausFit (or None on error)
        - 'sigma': Fitted standard deviation from gausFit (or None on error)
        - 'jnd': Just Noticeable Difference (or None on error)
        - 'pse': Point of Subjective Equality (or None on error)
        - 'n_trials': Number of trials (or None on error)
        - 'accuracy': Accuracy percentage (or None on error)
        - 'status': 'success' or 'error'
        
    Notes:
        - On error, returns result dict with status='error' and other fields set to None
        - Logs errors with format: [subject_id] [operation] error_message
        - Continues processing subsequent subjects even if this subject fails
        - Captures all exception types to ensure robustness
    """
    try:
        # Attempt to analyze the subject
        result = analyze_subject_results(exp, subject_row, output_dir, file_prefix)
        return result
        
    except Exception as e:
        # Extract subject ID for error logging
        subject_id = subject_row.get('subj', 'unknown') if isinstance(subject_row, pd.Series) else 'unknown'
        
        # Determine which operation failed based on exception type and message
        error_str = str(e)
        if 'print_statistics' in error_str or 'statistics' in error_str.lower():
            operation = 'print_statistics'
        elif 'gausFit' in error_str or 'fit' in error_str.lower():
            operation = 'gausFit'
        elif 'plot_psychometric' in error_str or 'plot' in error_str.lower():
            operation = 'plot_psychometric'
        else:
            operation = 'analyze_subject_results'
        
        # Log the error with subject and operation context
        log_subject_error(logger, subject_id, operation, error_str)
        
        # Return error result dictionary
        error_result = {
            'subj': subject_row.get('subj', 'unknown') if isinstance(subject_row, pd.Series) else 'unknown',
            'mu': None,
            'sigma': None,
            'jnd': None,
            'pse': subject_row.get('pse', None) if isinstance(subject_row, pd.Series) else None,
            'n_trials': None,
            'accuracy': None,
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
    accuracy, etc.).

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

    Notes:
        - Excel filename follows pattern: {file_prefix}_results_summary.xlsx
        - DataFrame includes all columns from result dictionaries
        - File is saved to the specified output directory
        - Returns the full path to the created Excel file

    Examples:
        >>> results = [
        ...     {'subj': 'subj001',
        ...      'mu': 100.5, 'sigma': 15.2, 'jnd': 10.25, 'pse': 100.0,
        ...      'n_trials': 200, 'accuracy': 85.5, 'status': 'success'},
        ...     {'subj': 'subj002',
        ...      'mu': 105.0, 'sigma': 14.8, 'jnd': 9.98, 'pse': 105.0,
        ...      'n_trials': 200, 'accuracy': 87.0, 'status': 'success'}
        ... ]
        >>> path = consolidate_results(results, '/output', 'M1ABS')
        >>> path
        '/output/M1ABS_results_summary.xlsx'
    """
    # Validate results_list parameter
    if not isinstance(results_list, list):
        raise TypeError(f"results_list must be a list, got {type(results_list)}")

    # Validate each item in results_list is a dictionary (skip if empty)
    for i, item in enumerate(results_list):
        if not isinstance(item, dict):
            raise TypeError(f"results_list[{i}] must be a dict, got {type(item)}")

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

    # Handle empty results list - create empty DataFrame with required columns
    if len(results_list) == 0:
        required_columns = ['subj', 'mu', 'sigma',
                           'jnd', 'pse', 'n_trials', 'accuracy', 'status']
        df = pd.DataFrame(columns=required_columns)
    else:
        # Create DataFrame from results list
        df = pd.DataFrame(results_list)

        # Ensure all required columns are present
        required_columns = {'subj', 'mu', 'sigma',
                           'jnd', 'pse', 'n_trials', 'accuracy', 'status'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Results missing required columns: {missing_columns}")

    # Generate Excel filename following the naming convention
    excel_filename = f"{file_prefix}_results_summary.xlsx"
    excel_filepath = os.path.join(output_dir, excel_filename)

    # Save DataFrame to Excel file
    df.to_excel(excel_filepath, index=False, sheet_name='Results')

    # Verify that the Excel file was created
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
        
    Raises:
        TypeError: If results_list is not a list
        ValueError: If results_list is empty
        ValueError: If any result item is not a dictionary or missing 'status' key
        
    Notes:
        - Success rate is calculated as: (successful_analyses / total_subjects) * 100
        - Success rate is 0.0 if all analyses failed
        - Success rate is 100.0 if all analyses succeeded
        - Floating-point precision may result in values like 99.99999999
        
    Examples:
        >>> results = [
        ...     {'subj': 'subj001', 'status': 'success'},
        ...     {'subj': 'subj002', 'status': 'success'},
        ...     {'subj': 'subj003', 'status': 'error'}
        ... ]
        >>> summary = create_summary_report(results)
        >>> summary
        {'total_subjects': 3, 'successful_analyses': 2, 'failed_analyses': 1, 'success_rate': 66.66666666666666}
    """
    # Validate results_list parameter
    if not isinstance(results_list, list):
        raise TypeError(f"results_list must be a list, got {type(results_list)}")
    
    if len(results_list) == 0:
        raise ValueError("results_list cannot be empty")
    
    # Validate each item in results_list
    for i, item in enumerate(results_list):
        if not isinstance(item, dict):
            raise TypeError(f"results_list[{i}] must be a dict, got {type(item)}")
        
        if 'status' not in item:
            raise ValueError(f"results_list[{i}] missing required 'status' key")
    
    # Count successful and failed analyses
    successful_count = sum(1 for result in results_list if result.get('status') == 'success')
    failed_count = sum(1 for result in results_list if result.get('status') == 'error')
    total_count = len(results_list)
    
    # Calculate success rate percentage
    success_rate = (successful_count / total_count) * 100 if total_count > 0 else 0.0
    
    # Build and return summary dictionary
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
            
    Raises:
        TypeError: If results_list is not a list
        ValueError: If results_list is empty
        ValueError: If any result item is not a dictionary or missing 'status' key
        
    Notes:
        - Prints to stdout with consistent formatting
        - Uses section headers with separator lines for clarity
        - Success rate displayed as percentage with 2 decimal places
        - Output format is human-readable and suitable for console display
        
    Examples:
        >>> results = [
        ...     {'subj': 'subj001', 'status': 'success'},
        ...     {'subj': 'subj002', 'status': 'success'},
        ...     {'subj': 'subj003', 'status': 'error'}
        ... ]
        >>> print_summary_report(results)
        ========================================
        ANALYSIS SUMMARY
        ========================================
        Total subjects processed: 3
        Successful analyses: 2
        Failed analyses: 1
        Success rate: 66.67%
        ========================================
    """
    # Get summary statistics
    summary = create_summary_report(results_list)
    
    # Extract values from summary
    total_subjects = summary['total_subjects']
    successful_analyses = summary['successful_analyses']
    failed_analyses = summary['failed_analyses']
    success_rate = summary['success_rate']
    
    # Print formatted summary report
    print("========================================")
    print("ANALYSIS SUMMARY")
    print("========================================")
    print(f"Total subjects processed: {total_subjects}")
    print(f"Successful analyses: {successful_analyses}")
    print(f"Failed analyses: {failed_analyses}")
    print(f"Success rate: {success_rate:.2f}%")
    print("========================================")
