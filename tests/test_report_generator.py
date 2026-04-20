"""Tests for report generator module."""

import pytest
import pandas as pd
from pathlib import Path
from postprocessing.modules.report_generator import generate_wide_format, generate_long_format
from postprocessing.modules.progressive_analyzer import ProgressiveResult
from postprocessing.modules.metadata import SubjectMetadata


def create_sample_results(n_subjects=3):
    """Helper to create sample ProgressiveResult objects."""
    results = []
    trial_counts = [40, 60, 80, 100, 120, 140, 160, 180, 200]
    
    for i in range(n_subjects):
        metadata = SubjectMetadata(
            subject_id=f"TEST{i:03d}",
            age=25 + i,
            gender="M" if i % 2 == 0 else "F",
            modality="BISA",
            algorithm="AD",
            group="control",
            valid=True,
            filename=f"TEST{i:03d}_25_M_BISA_AD_control.txt"
        )
        
        result = ProgressiveResult(
            metadata=metadata,
            trial_counts=trial_counts,
            pse_values={n: 600.0 + i * 10 + n * 0.1 for n in trial_counts},
            jnd_values={n: 50.0 + i * 5 + n * 0.05 for n in trial_counts},
            lat_mean={n: 550.0 + i * 10 + n * 0.2 for n in trial_counts},
            lat_std={n: 100.0 + i * 5 + n * 0.1 for n in trial_counts},
            lat_range={n: 400.0 + i * 10 + n * 0.3 for n in trial_counts},
            lat_entropy={n: 2.5 + i * 0.1 + n * 0.001 for n in trial_counts},
            method="logistic"
        )
        results.append(result)
    
    return results


def test_generate_wide_format_basic(temp_dir):
    """Test basic wide format generation."""
    results = create_sample_results(3)
    output_file = temp_dir / "test_wide.xlsx"
    
    generate_wide_format(results, str(output_file))
    
    assert output_file.exists()
    
    # Read back and verify
    df = pd.read_excel(output_file)
    assert len(df) == 3  # One row per subject
    assert 'subj' in df.columns
    assert 'age' in df.columns
    assert 'pse_40' in df.columns
    assert 'pse_200' in df.columns
    assert 'jnd_40' in df.columns
    assert 'lat_mean_40' in df.columns


def test_generate_long_format_basic(temp_dir):
    """Test basic long format generation."""
    results = create_sample_results(3)
    output_file = temp_dir / "test_long.xlsx"
    
    generate_long_format(results, str(output_file))
    
    assert output_file.exists()
    
    # Read back and verify
    df = pd.read_excel(output_file)
    assert len(df) == 27  # 9 rows per subject * 3 subjects
    assert 'subj' in df.columns
    assert 'n_trials' in df.columns
    assert 'pse' in df.columns
    assert 'jnd' in df.columns
    assert 'lat_mean' in df.columns


def test_generate_wide_format_empty_results(temp_dir):
    """Test wide format with empty results list."""
    output_file = temp_dir / "test_wide_empty.xlsx"
    
    generate_wide_format([], str(output_file))
    
    # Empty results don't create a file
    assert not output_file.exists()


def test_generate_long_format_empty_results(temp_dir):
    """Test long format with empty results list."""
    output_file = temp_dir / "test_long_empty.xlsx"
    
    generate_long_format([], str(output_file))
    
    # Empty results don't create a file
    assert not output_file.exists()


def test_generate_wide_format_single_subject(temp_dir):
    """Test wide format with single subject."""
    results = create_sample_results(1)
    output_file = temp_dir / "test_wide_single.xlsx"
    
    generate_wide_format(results, str(output_file))
    
    assert output_file.exists()
    
    df = pd.read_excel(output_file)
    assert len(df) == 1


def test_generate_long_format_single_subject(temp_dir):
    """Test long format with single subject."""
    results = create_sample_results(1)
    output_file = temp_dir / "test_long_single.xlsx"
    
    generate_long_format(results, str(output_file))
    
    assert output_file.exists()
    
    df = pd.read_excel(output_file)
    assert len(df) == 9  # 9 progressive points


def test_wide_format_column_order(temp_dir):
    """Test that wide format has correct column order."""
    results = create_sample_results(1)
    output_file = temp_dir / "test_wide_order.xlsx"
    
    generate_wide_format(results, str(output_file))
    
    df = pd.read_excel(output_file)
    columns = df.columns.tolist()
    
    # Metadata columns should come first
    assert columns[0] == 'subj'
    assert 'age' in columns[:6]
    assert 'gender' in columns[:6]
    assert 'modality' in columns[:6]
    assert 'algorithm' in columns[:6]
    assert 'group' in columns[:6]


def test_long_format_trial_counts(temp_dir):
    """Test that long format has correct trial counts."""
    results = create_sample_results(1)
    output_file = temp_dir / "test_long_counts.xlsx"
    
    generate_long_format(results, str(output_file))
    
    df = pd.read_excel(output_file)
    trial_counts = df['n_trials'].unique()
    
    expected_counts = [40, 60, 80, 100, 120, 140, 160, 180, 200]
    assert len(trial_counts) == 9
    assert all(tc in expected_counts for tc in trial_counts)
