"""
Analysis orchestrator - unified pipeline for analyzing temporal bisection data.

Orchestrates progressive psychometric fitting, metadata extraction, and Excel report generation.
Works with both simulated and real data (PSA/GBF formats).
"""

import glob
import logging
import os
from typing import List

import numpy as np

from analysis.core.progressive_analyzer import ProgressiveAnalyzer, ProgressiveResult
from analysis.io.converter import convert_psa_to_gbf
from analysis.io.metadata import extract_metadata, SubjectMetadata
from analysis.io.report_generator import generate_wide_format, generate_long_format
from analysis.core.psychometric_helpers import calculate_latency_statistics
from analysis.io.fitters import fit_gaussfit

logger = logging.getLogger(__name__)


class ProgressiveResultWithMetadata(ProgressiveResult):
    """Extended ProgressiveResult with subject metadata."""
    
    def __init__(self, *args, metadata: SubjectMetadata = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata or SubjectMetadata(
            subject_id="", age=0, gender="", modality="", 
            algorithm="", group="", valid=False, filename=""
        )


def analyze_subject_progressive(filepath: str, method: str = 'logistic', 
                                bin_size: float = 10.0) -> ProgressiveResultWithMetadata:
    """
    Perform progressive psychometric analysis on a single subject's data.
    
    Wraps core analyzer with metadata extraction and gaussfit support.
    
    Args:
        filepath: Path to subject's data file
        method: Fitting method ('logistic', 'probit', 'gaussfit')
        bin_size: Bin size for gaussfit method
        
    Returns:
        ProgressiveResultWithMetadata containing analysis results and metadata
    """
    # Extract metadata from filename
    metadata = extract_metadata(filepath)
    
    if not metadata.valid:
        logger.warning(f"Failed to extract metadata from filename: {filepath}")
    
    # Run progressive analysis using core analyzer
    analyzer = ProgressiveAnalyzer()
    
    if method in ('logistic', 'probit'):
        result = analyzer.run_progressive_analysis(filepath, method=method)
        result_with_meta = ProgressiveResultWithMetadata(
            trial_counts=result.trial_counts,
            pse_values=result.pse_values,
            jnd_values=result.jnd_values,
            lat_mean=result.lat_mean,
            lat_std=result.lat_std,
            lat_range=result.lat_range,
            lat_entropy=result.lat_entropy,
            method=method,
            metadata=metadata
        )
    elif method == 'gaussfit':
        from analysis.core.data_loader import DataLoader
        try:
            latencies, responses = DataLoader.load_and_expand(filepath)
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return ProgressiveResultWithMetadata(method=method, metadata=metadata)
        
        result_with_meta = ProgressiveResultWithMetadata(method=method, metadata=metadata)
        
        for N in result_with_meta.trial_counts:
            result_with_meta.pse_values[N] = np.nan
            result_with_meta.jnd_values[N] = np.nan
            result_with_meta.lat_mean[N] = np.nan
            result_with_meta.lat_std[N] = np.nan
            result_with_meta.lat_range[N] = np.nan
            result_with_meta.lat_entropy[N] = np.nan
        
        total_trials = len(latencies)
        logger.info(f"Analyzing {metadata.subject_id}: {total_trials} trials, method={method}")
        
        for N in result_with_meta.trial_counts:
            if total_trials < N:
                logger.info(f"  N={N}: Skipping (insufficient trials: {total_trials} < {N})")
                continue
            
            first_n_lat = latencies[:N]
            first_n_resp = responses[:N]
            
            stats = calculate_latency_statistics(first_n_lat)
            result_with_meta.lat_mean[N] = stats['mean']
            result_with_meta.lat_std[N] = stats['std']
            result_with_meta.lat_range[N] = stats['range']
            result_with_meta.lat_entropy[N] = stats['entropy']
            
            try:
                unique_lats = np.unique(first_n_lat)
                binned_lats, binned_counts, binned_resps = [], [], []
                for lat in unique_lats:
                    mask = first_n_lat == lat
                    binned_lats.append(lat)
                    binned_counts.append(np.sum(mask))
                    binned_resps.append(np.sum(first_n_resp[mask]))
                
                fit_result = fit_gaussfit(np.array(binned_lats), np.array(binned_counts),
                                          np.array(binned_resps), 0.5, bin_size)
                
                if fit_result.success and np.isfinite(fit_result.pse) and np.isfinite(fit_result.jnd):
                    result_with_meta.pse_values[N] = fit_result.pse
                    result_with_meta.jnd_values[N] = fit_result.jnd
                    logger.info(f"  N={N}: PSE={fit_result.pse:.1f}, JND={fit_result.jnd:.1f}, "
                                f"lat_mean={stats['mean']:.1f}, lat_std={stats['std']:.1f}, "
                                f"entropy={stats['entropy']:.2f}")
                else:
                    logger.warning(f"  N={N}: Fitting failed or returned NaN")
            except Exception as e:
                logger.error(f"  N={N}: Exception during fitting - {str(e)}")
    else:
        logger.error(f"Unknown method: {method}")
        result_with_meta = ProgressiveResultWithMetadata(method=method, metadata=metadata)
    
    return result_with_meta

# called by group_real_data_analysis.py
def analyze_batch(input_dir: str, output_dir: str, project_name: str,
                  method: str = 'logistic', bin_size: float = 10.0) -> List[ProgressiveResultWithMetadata]:
    """
    Analyze all subjects in input directory and generate Excel reports.
    
    Args:
        input_dir: Directory containing subject data files
        output_dir: Directory for Excel output files
        project_name: Project name for output filenames
        method: Fitting method ('logistic', 'probit', 'gaussfit')
        bin_size: Bin size for gaussfit method
        
    Returns:
        List of ProgressiveResultWithMetadata objects for successfully analyzed subjects
    """
    pattern = os.path.join(input_dir, "*.txt")
    files = sorted(glob.glob(pattern))
    
    if not files:
        logger.warning(f"No .txt files found in {input_dir}")
        print(f"No .txt files found in {input_dir}")
        return []
    
    print("=" * 60)
    print("PROGRESSIVE PSYCHOMETRIC ANALYSIS")
    print("=" * 60)
    print(f"Method: {method}")
    print(f"Found {len(files)} files to process")
    print()
    
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
    
    if results:
        print()
        print("=" * 60)
        print("WRITING RESULTS TO EXCEL")
        print("=" * 60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        wide_filename = f"results_{project_name}_{method}_prog_wide.xlsx"
        long_filename = f"results_{project_name}_{method}_prog_long.xlsx"
        
        wide_path = os.path.join(output_dir, wide_filename)
        long_path = os.path.join(output_dir, long_filename)
        
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
