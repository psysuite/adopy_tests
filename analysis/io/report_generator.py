"""
Report generator module for creating Excel outputs.

This module provides functionality to generate Excel reports in both
wide format (one row per subject) and long format (9 rows per subject).
"""

from typing import List
import pandas as pd
import os

from analysis.core.progressive_analyzer import ProgressiveResult


def generate_wide_format(results: List[ProgressiveResult], 
                         output_path: str) -> None:
    """
    Generate wide format Excel report (one row per subject).
    
    Args:
        results: List of ProgressiveResult objects
        output_path: Path for output Excel file
        
    Output columns:
        - Metadata: subj, age, gender, modality, algorithm, group
        - Parameters: pse_40, pse_60, ..., pse_200
        - Parameters: jnd_40, jnd_60, ..., jnd_200
        - Statistics: SC_40, ..., SC_200 (stimulus center)
        - Statistics: SS_40, ..., SS_200 (stimulus spread)
        - Statistics: lat_range_40, ..., lat_range_200
        - Statistics: lat_entropy_40, ..., lat_entropy_200
        
    Implementation:
        - Use pandas DataFrame for data organization
        - Write to Excel using openpyxl engine
        - Handle NaN values appropriately
    """
    if not results:
        print("No results to write (wide format)")
        return
    
    # Build data rows
    rows = []
    
    for result in results:
        row = {
            'subj': result.metadata.subject_id,
            'age': result.metadata.age,
            'gender': result.metadata.gender,
            'modality': result.metadata.modality,
            'algorithm': result.metadata.algorithm,
            'group': result.metadata.group,
        }
        
        # Add PSE values for each trial count
        for N in result.trial_counts:
            row[f'pse_{N}'] = result.pse_values.get(N, float('nan'))
        
        # Add JND values for each trial count
        for N in result.trial_counts:
            row[f'jnd_{N}'] = result.jnd_values.get(N, float('nan'))
        
        # Add latency statistics for each trial count
        for N in result.trial_counts:
            row[f'SC_{N}'] = result.lat_mean.get(N, float('nan'))
        
        for N in result.trial_counts:
            row[f'SS_{N}'] = result.lat_std.get(N, float('nan'))
        
        for N in result.trial_counts:
            row[f'lat_range_{N}'] = result.lat_range.get(N, float('nan'))
        
        for N in result.trial_counts:
            row[f'lat_entropy_{N}'] = result.lat_entropy.get(N, float('nan'))
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to Excel
    df.to_excel(output_path, index=False, engine='openpyxl')
    
    print(f"Wide format: {output_path}")


def generate_long_format(results: List[ProgressiveResult], 
                         output_path: str) -> None:
    """
    Generate long format Excel report (9 rows per subject).
    
    Args:
        results: List of ProgressiveResult objects
        output_path: Path for output Excel file
        
    Output columns:
        - Metadata: subj, age, gender, modality, algorithm, group
        - Trial count: n_trials
        - Parameters: pse, jnd
        - Statistics: SC (stimulus center), SS (stimulus spread), lat_range, lat_entropy
        
    Implementation:
        - Create one row per (subject, trial_count) combination
        - Use pandas DataFrame with flat structure
        - Write to Excel using openpyxl engine
    """
    if not results:
        print("No results to write (long format)")
        return
    
    # Build data rows
    rows = []
    
    for result in results:
        for N in result.trial_counts:
            row = {
                'subj': result.metadata.subject_id,
                'age': result.metadata.age,
                'gender': result.metadata.gender,
                'modality': result.metadata.modality,
                'algorithm': result.metadata.algorithm,
                'group': result.metadata.group,
                'n_trials': N,
                'pse': result.pse_values.get(N, float('nan')),
                'jnd': result.jnd_values.get(N, float('nan')),
                'SC': result.lat_mean.get(N, float('nan')),
                'SS': result.lat_std.get(N, float('nan')),
                'lat_range': result.lat_range.get(N, float('nan')),
                'lat_entropy': result.lat_entropy.get(N, float('nan')),
            }
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to Excel
    df.to_excel(output_path, index=False, engine='openpyxl')
    
    print(f"Long format: {output_path}")
