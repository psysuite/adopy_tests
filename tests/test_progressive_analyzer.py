"""Tests for progressive analyzer module."""

import pytest
import numpy as np
from postprocessing.modules.progressive_analyzer import (
    analyze_subject_progressive,
    analyze_batch,
    ProgressiveResult
)


def test_analyze_subject_progressive_basic(temp_dir):
    """Test basic progressive analysis."""
    # Create a test file with enough trials
    test_file = temp_dir / "TEST001_25_M_BISA_AD_control.txt"
    
    # Generate 200 trials
    lines = ["@ 0.0\n"]
    for i in range(200):
        latency = 400 + (i % 5) * 100  # Cycle through 400, 500, 600, 700, 800
        response = 1 if latency > 600 else 0
        lines.append(f"{latency} 1 {response}\n")
    
    test_file.write_text("".join(lines))
    
    # Analyze
    result = analyze_subject_progressive(str(test_file), method="logistic")
    
    assert isinstance(result, ProgressiveResult)
    assert result.metadata.subject_id == "TEST001"
    assert result.method == "logistic"
    assert len(result.trial_counts) == 9
    
    # Check that we have results for all trial counts
    for n in result.trial_counts:
        assert n in result.pse_values
        assert n in result.jnd_values
        assert n in result.lat_mean
        assert n in result.lat_std
        assert n in result.lat_range
        assert n in result.lat_entropy


def test_analyze_subject_progressive_insufficient_trials(temp_dir):
    """Test progressive analysis with insufficient trials."""
    test_file = temp_dir / "TEST001_25_M_BISA_AD_control.txt"
    
    # Generate only 50 trials
    lines = ["@ 0.0\n"]
    for i in range(50):
        latency = 400 + (i % 5) * 100
        response = 1 if latency > 600 else 0
        lines.append(f"{latency} 1 {response}\n")
    
    test_file.write_text("".join(lines))
    
    # Analyze
    result = analyze_subject_progressive(str(test_file), method="logistic")
    
    assert isinstance(result, ProgressiveResult)
    
    # Should have results for 40 trials
    assert not np.isnan(result.pse_values[40])
    
    # Should NOT have results for 200 trials
    assert np.isnan(result.pse_values[200])


def test_analyze_batch_basic(temp_dir):
    """Test batch analysis."""
    # Create input and output directories
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    # Create multiple test files
    for i in range(3):
        test_file = input_dir / f"TEST{i:03d}_25_M_BISA_AD_control.txt"
        
        lines = ["@ 0.0\n"]
        for j in range(200):
            latency = 400 + (j % 5) * 100
            response = 1 if latency > 600 else 0
            lines.append(f"{latency} 1 {response}\n")
        
        test_file.write_text("".join(lines))
    
    # Analyze batch
    results = analyze_batch(str(input_dir), str(output_dir), "test_project", method="logistic")
    
    assert len(results) == 3
    assert all(isinstance(r, ProgressiveResult) for r in results)
    assert all(r.method == "logistic" for r in results)


def test_analyze_batch_with_failures(temp_dir):
    """Test batch analysis with some failing files."""
    # Create input and output directories
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    # Create one good file
    good_file = input_dir / "TEST001_25_M_BISA_AD_control.txt"
    lines = ["@ 0.0\n"]
    for i in range(200):
        latency = 400 + (i % 5) * 100
        response = 1 if latency > 600 else 0
        lines.append(f"{latency} 1 {response}\n")
    good_file.write_text("".join(lines))
    
    # Create one malformed file
    bad_file = input_dir / "malformed.txt"
    bad_file.write_text("This is not valid data")
    
    # Analyze batch
    results = analyze_batch(str(input_dir), str(output_dir), "test_project", method="logistic")
    
    # Should have at least the good file
    assert len(results) >= 1


def test_analyze_batch_empty_directory(temp_dir):
    """Test batch analysis with empty directory."""
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    results = analyze_batch(str(input_dir), str(output_dir), "test_project", method="logistic")
    assert len(results) == 0


def test_progressive_result_dataclass(sample_metadata):
    """Test ProgressiveResult dataclass."""
    trial_counts = [40, 60, 80]
    
    result = ProgressiveResult(
        metadata=sample_metadata,
        trial_counts=trial_counts,
        pse_values={40: 600.0, 60: 605.0, 80: 610.0},
        jnd_values={40: 50.0, 60: 52.0, 80: 54.0},
        lat_mean={40: 550.0, 60: 555.0, 80: 560.0},
        lat_std={40: 100.0, 60: 102.0, 80: 104.0},
        lat_range={40: 400.0, 60: 405.0, 80: 410.0},
        lat_entropy={40: 2.5, 60: 2.6, 80: 2.7},
        method="logistic"
    )
    
    assert result.metadata.subject_id == "TEST001"
    assert result.method == "logistic"
    assert len(result.trial_counts) == 3
    assert result.pse_values[40] == 600.0


def test_analyze_subject_progressive_invalid_file(temp_dir):
    """Test progressive analysis returns invalid result for unparseable file."""
    bad_file = temp_dir / "bad.txt"
    bad_file.write_text("# only comments, no data\n")

    result = analyze_subject_progressive(str(bad_file), method="logistic")

    assert isinstance(result, ProgressiveResult)
    assert not result.metadata.valid


def test_analyze_subject_progressive_unknown_method(temp_dir):
    """Test progressive analysis with unknown method logs error and skips fitting."""
    test_file = temp_dir / "TEST001_25_M_BISA_AD_control.txt"
    lines = ["@ 0.0\n"]
    for i in range(200):
        latency = 400 + (i % 5) * 100
        response = 1 if latency > 600 else 0
        lines.append(f"{latency} 1 {response}\n")
    test_file.write_text("".join(lines))

    result = analyze_subject_progressive(str(test_file), method="unknown_method")

    assert isinstance(result, ProgressiveResult)
    # All PSE values should be NaN since method is unknown
    assert all(np.isnan(v) for v in result.pse_values.values())


def test_analyze_batch_all_files_fail(temp_dir):
    """Test batch analysis when all files have invalid metadata → no results written."""
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Files with invalid filename format (too few parts → invalid metadata)
    for i in range(2):
        f = input_dir / f"badname{i}.txt"
        lines = ["@ 0.0\n"]
        for j in range(200):
            latency = 400 + (j % 5) * 100
            response = 1 if latency > 600 else 0
            lines.append(f"{latency} 1 {response}\n")
        f.write_text("".join(lines))

    results = analyze_batch(str(input_dir), str(output_dir), "test_project", method="logistic")

    assert len(results) == 0


def test_analyze_subject_progressive_gaussfit(temp_dir):
    """Test progressive analysis with gaussfit method."""
    test_file = temp_dir / "TEST001_25_M_BISA_AD_control.txt"
    lines = ["@ 0.0\n"]
    for i in range(200):
        latency = 400 + (i % 5) * 100
        response = 1 if latency > 600 else 0
        lines.append(f"{latency} 1 {response}\n")
    test_file.write_text("".join(lines))

    result = analyze_subject_progressive(str(test_file), method="gaussfit", bin_size=50.0)

    assert isinstance(result, ProgressiveResult)
    assert result.method == "gaussfit"
