"""
Progressive analyzer module for temporal bisection data analysis.

Orchestrates GBF file parsing, progressive psychometric fitting (via ValidationAnalyzer),
latency statistics, and Excel report generation.
Fitting logic lives in utilities.psychometric_helpers (single source of truth).
"""

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
import logging

from .parser import parse_data_file
from .metadata import extract_metadata, SubjectMetadata
from .fitters import fit_gaussfit
from utilities.psychometric_helpers import (
    fit_logistic_psychometric,
    fit_probit_psychometric,
    calculate_latency_statistics,
    calculate_stability_from_values,
)


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ProgressiveResult:
    """Results from progressive analysis for one subject.
    
    Attributes:
        metadata: Subject metadata
        trial_counts: List of trial counts analyzed [40, 60, 80, 100, 120, 140, 160, 180, 200]
        pse_values: PSE at each trial count (dict keyed by trial count)
        jnd_values: JND at each trial count (dict keyed by trial count)
        lat_mean: Mean latency at each count (dict keyed by trial count)
        lat_std: Std latency at each count (dict keyed by trial count)
        lat_range: Range latency at each count (dict keyed by trial count)
        lat_entropy: Entropy at each count (dict keyed by trial count)
        method: Fitting method used ('logistic', 'probit', 'gaussfit')
    """
    metadata: SubjectMetadata
    trial_counts: List[int] = field(default_factory=lambda: [40, 60, 80, 100, 120, 140, 160, 180, 200])
    pse_values: Dict[int, float] = field(default_factory=dict)
    jnd_values: Dict[int, float] = field(default_factory=dict)
    lat_mean: Dict[int, float] = field(default_factory=dict)
    lat_std: Dict[int, float] = field(default_factory=dict)
    lat_range: Dict[int, float] = field(default_factory=dict)
    lat_entropy: Dict[int, float] = field(default_factory=dict)
    method: str = ""


def analyze_subject_progressive(filepath: str, method: str = 'logistic', 
                                bin_size: float = 10.0) -> ProgressiveResult:
    """
    Perform progressive psychometric analysis on a single subject's data.
    
    Args:
        filepath: Path to subject's data file
        method: Fitting method ('logistic', 'probit', 'gaussfit')
        bin_size: Bin size for gaussfit method
        
    Returns:
        ProgressiveResult containing all progressive analysis results
        
    Algorithm:
        1. Parse data file
        2. Extract metadata from filename
        3. Initialize result structure with NaN values
        4. For each N in [40, 60, 80, 100, 120, 140, 160, 180, 200]:
           a. Check if total trials >= N
           b. Extract first N trials in original order
           c. Fit psychometric curve using specified method
           d. Calculate latency statistics from first N trials
           e. Store PSE, JND, and statistics
           f. Log progress message
        5. Return complete ProgressiveResult
        
    Error handling:
        - Skip trial count if insufficient data
        - Set NaN for failed fits but continue
        - Log all errors with context
    """
    # Parse data file
    trial_data = parse_data_file(filepath)
    
    if not trial_data.valid:
        logger.error(f"Failed to parse file {filepath}: {trial_data.error_message}")
        # Return empty result with invalid metadata
        return ProgressiveResult(
            metadata=SubjectMetadata(
                subject_id="", age=0, gender="", modality="", 
                algorithm="", group="", valid=False, filename=filepath
            ),
            method=method
        )
    
    # Extract metadata from filename
    metadata = extract_metadata(filepath)
    
    if not metadata.valid:
        logger.warning(f"Failed to extract metadata from filename: {filepath}")
    
    # Initialize result structure
    result = ProgressiveResult(metadata=metadata, method=method)
    
    # Initialize all values to NaN
    for N in result.trial_counts:
        result.pse_values[N] = np.nan
        result.jnd_values[N] = np.nan
        result.lat_mean[N] = np.nan
        result.lat_std[N] = np.nan
        result.lat_range[N] = np.nan
        result.lat_entropy[N] = np.nan
    
    # Expand binned data to individual trials for progressive analysis
    expanded_lat = []
    expanded_resp = []
    for lat, count, resp in zip(trial_data.latencies, trial_data.counts, trial_data.responses):
        for _ in range(int(count)):
            expanded_lat.append(lat)
            expanded_resp.append(resp)
    
    expanded_lat = np.array(expanded_lat)
    expanded_resp = np.array(expanded_resp)
    total_trials = len(expanded_lat)
    
    logger.info(f"Analyzing {metadata.subject_id}: {total_trials} trials, method={method}")
    
    # Perform progressive analysis
    for N in result.trial_counts:
        if total_trials < N:
            logger.info(f"  N={N}: Skipping (insufficient trials: {total_trials} < {N})")
            continue
        
        # Extract first N trials in original order
        first_n_lat = expanded_lat[:N]
        first_n_resp = expanded_resp[:N]
        
        # Calculate latency statistics
        stats = calculate_latency_statistics(first_n_lat)
        result.lat_mean[N] = stats['mean']
        result.lat_std[N] = stats['std']
        result.lat_range[N] = stats['range']
        result.lat_entropy[N] = stats['entropy']
        
        # Fit psychometric curve directly on individual trials
        try:
            if method == 'logistic':
                pse, jnd = fit_logistic_psychometric(first_n_lat, first_n_resp, fallback=True)
                fit_success = True
            elif method == 'probit':
                pse, jnd = fit_probit_psychometric(first_n_lat, first_n_resp, fallback=True)
                fit_success = True
            elif method == 'gaussfit':
                # gaussfit still needs binned format
                unique_lats = np.unique(first_n_lat)
                binned_lats, binned_counts, binned_resps = [], [], []
                for lat in unique_lats:
                    mask = first_n_lat == lat
                    binned_lats.append(lat)
                    binned_counts.append(np.sum(mask))
                    binned_resps.append(np.sum(first_n_resp[mask]))
                fit_result = fit_gaussfit(np.array(binned_lats), np.array(binned_counts),
                                          np.array(binned_resps), trial_data.guess_rate, bin_size)
                fit_success = fit_result.success
                pse, jnd = fit_result.pse, fit_result.jnd
            else:
                logger.error(f"Unknown method: {method}")
                continue

            if fit_success and np.isfinite(pse) and np.isfinite(jnd):
                result.pse_values[N] = pse
                result.jnd_values[N] = jnd
                logger.info(f"  N={N}: PSE={pse:.1f}, JND={jnd:.1f}, "
                            f"lat_mean={stats['mean']:.1f}, lat_std={stats['std']:.1f}, "
                            f"entropy={stats['entropy']:.2f}")
            else:
                logger.warning(f"  N={N}: Fitting failed or returned NaN")
                result.pse_values[N] = np.nan
                result.jnd_values[N] = np.nan

        except Exception as e:
            logger.error(f"  N={N}: Exception during fitting - {str(e)}")
            result.pse_values[N] = np.nan
            result.jnd_values[N] = np.nan
    
    return result


def analyze_batch(input_dir: str, output_dir: str, project_name: str,
                  method: str = 'logistic', bin_size: float = 10.0) -> List[ProgressiveResult]:
    """
    Analyze all subjects in input directory and generate Excel reports.
    
    Args:
        input_dir: Directory containing subject data files
        output_dir: Directory for Excel output files
        project_name: Project name for output filenames
        method: Fitting method ('logistic', 'probit', 'gaussfit')
        bin_size: Bin size for gaussfit method
        
    Returns:
        List of ProgressiveResult objects for successfully analyzed subjects
        
    Behavior:
        1. Find all .txt files in input_dir
        2. Print header with method and file count
        3. For each file:
           a. Print progress message (file N of M)
           b. Call analyze_subject_progressive() with error handling
           c. Add to results list or skipped list
        4. Maintain list of skipped/failed filenames
        5. Print summary: total files, successfully processed, skipped count
        6. Print list of skipped filenames if any
        7. Generate Excel reports (wide and long formats)
        8. Return list of ProgressiveResult objects
    """
    import os
    import glob
    from .report_generator import generate_wide_format, generate_long_format
    
    # Find all .txt files
    pattern = os.path.join(input_dir, "*.txt")
    files = sorted(glob.glob(pattern))
    
    if not files:
        logger.warning(f"No .txt files found in {input_dir}")
        print(f"No .txt files found in {input_dir}")
        return []
    
    # Print header
    print("=" * 60)
    print("PROGRESSIVE PSYCHOMETRIC ANALYSIS")
    print("=" * 60)
    print(f"Method: {method}")
    print(f"Found {len(files)} files to process")
    print()
    
    # Process files
    results = []
    skipped_files = []
    
    for i, filepath in enumerate(files, 1):
        filename = os.path.basename(filepath)
        print(f"Processing file {i}/{len(files)}: {filename}")
        
        try:
            result = analyze_subject_progressive(filepath, method, bin_size)
            
            if result.metadata.valid:
                results.append(result)
            else:
                skipped_files.append(filename)
                logger.warning(f"Skipping {filename}: Invalid metadata")
                
        except Exception as e:
            skipped_files.append(filename)
            logger.error(f"Error processing {filename}: {str(e)}")
            print(f"  Error: {str(e)}")
    
    # Print summary
    print()
    print("=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Method: {method}")
    print(f"Total files: {len(files)}")
    print(f"Successfully processed: {len(results)}")
    print(f"Long format rows: {len(results) * 9} (9 per subject)")
    print(f"Skipped/Failed: {len(skipped_files)}")
    
    if skipped_files:
        print()
        print("Skipped files:")
        for filename in skipped_files:
            print(f"  - {filename}")
    
    # Generate Excel reports if we have results
    if results:
        print()
        print("=" * 60)
        print("WRITING RESULTS TO EXCEL")
        print("=" * 60)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filenames
        wide_filename = f"results_{project_name}_{method}_prog_wide.xlsx"
        long_filename = f"results_{project_name}_{method}_prog_long.xlsx"
        
        wide_path = os.path.join(output_dir, wide_filename)
        long_path = os.path.join(output_dir, long_filename)
        
        # Generate reports
        generate_wide_format(results, wide_path)
        generate_long_format(results, long_path)
        
        print()
        print("=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print("The data includes:")
        print(f"  - PSE and JND at each progressive point (using {method} method)")
        print("  - Latency statistics (mean, std, range, entropy)")
        print("  - Ready for statistical analysis")
        print("=" * 60)
    else:
        print()
        print("=" * 60)
        print("NO RESULTS TO WRITE")
        print("All files were skipped or failed processing.")
        print("=" * 60)
    
    return results
