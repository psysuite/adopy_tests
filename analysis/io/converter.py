"""
Converter module for PSA to GBF format conversion.

This module provides functionality to convert psysuite BIS results (PSA format)
to Gaussian bootstrap fit format (GBF format).
"""

from typing import Dict, Any, List
import pandas as pd
import os
import glob
import logging


logger = logging.getLogger(__name__)


def convert_psa_to_gbf(input_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Convert all PSA format files in input_dir to GBF format in output_dir.
    
    Args:
        input_dir: Directory containing PSA format .txt files
        output_dir: Directory for GBF format output files
        
    Returns:
        Dictionary with conversion statistics:
            - total_files: int
            - converted: int
            - failed: List[str]
            
    Behavior:
        - Reads PSA files as tab-delimited tables
        - Removes columns: id, label, conflict, response, correct_answer, elapsed, repetition
        - Retains: latency, count, user_answer
        - Adds column of ones after latency
        - Writes tab-delimited output without headers
        - Continues on error, logging failures
    """
    # Find all .txt files
    pattern = os.path.join(input_dir, "*.txt")
    files = sorted(glob.glob(pattern))
    
    if not files:
        logger.warning(f"No .txt files found in {input_dir}")
        return {
            'total_files': 0,
            'converted': 0,
            'failed': []
        }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process files
    converted_count = 0
    failed_files = []
    
    print(f"Converting {len(files)} PSA files to GBF format...")
    
    for i, filepath in enumerate(files, 1):
        filename = os.path.basename(filepath)
        
        try:
            # Read PSA file as tab-delimited table
            df = pd.read_csv(filepath, sep='\t')
            
            # Check if required columns exist
            required_cols = ['lat', 'user_ans']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"File {filename} missing required columns")
                failed_files.append(filename)
                continue
            
            # Remove unwanted columns (keep: lat, user_ans, confl_magn)
            # MATLAB removes: id, label, confl, res, cor_ans, elapsed, rep
            cols_to_remove = ['id', 'label', 'confl', 'res', 'cor_ans', 'elapsed', 'rep']
            for col in cols_to_remove:
                if col in df.columns:
                    df = df.drop(columns=[col])
            
            # Keep only: lat, user_ans, confl_magn (if present)
            cols_to_keep = ['lat', 'user_ans']
            if 'confl_magn' in df.columns:
                cols_to_keep.append('confl_magn')
            
            df = df[cols_to_keep]
            
            # Add count column (all ones) after lat
            df.insert(1, 'count', 1)
            
            # Final columns: lat, count, user_ans, confl_magn (if present)
            # Convert confl_magn to int if present (to match MATLAB format)
            if 'confl_magn' in df.columns:
                df['confl_magn'] = df['confl_magn'].astype(int)
            
            # Write to output directory with same filename
            output_path = os.path.join(output_dir, filename)
            # Use float_format to prevent .0 suffix on integers
            df.to_csv(output_path, sep='\t', index=False, header=False, float_format='%g')
            
            converted_count += 1
            
            if i % 10 == 0 or i == len(files):
                print(f"  Converted {i}/{len(files)} files...")
                
        except Exception as e:
            logger.error(f"Error converting {filename}: {str(e)}")
            failed_files.append(filename)
    
    print(f"Conversion complete: {converted_count}/{len(files)} files converted")
    
    if failed_files:
        print(f"Failed files: {len(failed_files)}")
        for filename in failed_files:
            print(f"  - {filename}")
    
    return {
        'total_files': len(files),
        'converted': converted_count,
        'failed': failed_files
    }


def save_gbf_file(rows: List[Dict], output_path: str) -> None:
    """
    Save trial rows to GBF format file (no header, tab-delimited).

    Args:
        rows: List of trial dicts with 'lat', 'count', 'user_ans', 'confl_magn'
        output_path: Path to output file
    """
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False, header=False, float_format='%g')


def read_gbf_file(filepath: str) -> List[Dict]:
    """
    Read GBF format file (tab-delimited, no header).
    
    Args:
        filepath: Path to GBF file
        
    Returns:
        List of trial dicts with 'lat', 'count', 'user_ans', 'confl_magn'
    """
    df = pd.read_csv(filepath, sep='\t', header=None)
    
    # GBF format: lat, count, user_ans, confl_magn (optional)
    column_names = ['lat', 'count', 'user_ans']
    if len(df.columns) > 3:
        column_names.append('confl_magn')
    
    df.columns = column_names[:len(df.columns)]
    
    return df.to_dict('records')
