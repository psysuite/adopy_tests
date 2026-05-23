"""Basic tests for core modules to verify they work independently."""

import pytest
import numpy as np
import tempfile
import os
from postprocessing.modules.parser import parse_data_file, TrialData
from postprocessing.modules.metadata import extract_metadata, SubjectMetadata
from analysis.core.psychometric_analysis import calculate_latency_statistics


def test_parser_imports():
    """Test that parser module imports correctly."""
    assert TrialData is not None
    assert parse_data_file is not None


def test_parser_basic():
    """Test basic parser functionality."""
    # Create a temporary file with test data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("@ 0.5\n")
        f.write("# This is a comment\n")
        f.write("500 1 1\n")
        f.write("400 1 0\n")
        f.write("600 1 1\n")
        temp_path = f.name
    
    try:
        result = parse_data_file(temp_path)
        assert result.valid
        assert len(result.latencies) == 3
        assert result.guess_rate == 0.5
        assert len(result.latencies) == len(result.counts) == len(result.responses)
    finally:
        os.unlink(temp_path)


def test_metadata_imports():
    """Test that metadata module imports correctly."""
    assert SubjectMetadata is not None
    assert extract_metadata is not None


def test_metadata_basic():
    """Test basic metadata extraction."""
    filename = "A01_26_f_BISA_AD_TD.txt"
    result = extract_metadata(filename)
    assert result.valid
    assert result.subject_id == "A01"
    assert result.age == 26
    assert result.gender == "f"
    assert result.modality == "BISA"
    assert result.algorithm == "AD"
    assert result.group == "TD"


def test_statistics_imports():
    """Test that statistics module imports correctly."""
    assert calculate_latency_statistics is not None


def test_statistics_basic():
    """Test basic statistics calculation."""
    latencies = np.array([400, 450, 500, 550, 600])
    stats = calculate_latency_statistics(latencies)
    assert 'mean' in stats
    assert 'std' in stats
    assert 'range' in stats
    assert 'entropy' in stats
    assert stats['mean'] == 500.0
    assert stats['range'] == 200.0
    assert not np.isnan(stats['entropy'])


def test_parser_two_column_format():
    """Test parser with 2-column format (latency response, no count)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("500 1\n")
        f.write("400 0\n")
        f.write("600 1\n")
        temp_path = f.name
    try:
        result = parse_data_file(temp_path)
        assert result.valid
        assert len(result.latencies) == 3
        assert all(c == 1 for c in result.counts)
    finally:
        os.unlink(temp_path)


def test_parser_malformed_guess_rate():
    """Test parser ignores malformed @ guess rate line."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("@ not_a_number\n")
        f.write("500 1 1\n")
        temp_path = f.name
    try:
        result = parse_data_file(temp_path)
        assert result.valid
        assert result.guess_rate == 0.0  # default, malformed line ignored
    finally:
        os.unlink(temp_path)


def test_parser_malformed_data_lines():
    """Test parser skips malformed data lines and continues."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("500 1 1\n")
        f.write("not_a_number abc\n")   # malformed → skipped
        f.write("600 1 0\n")
        temp_path = f.name
    try:
        result = parse_data_file(temp_path)
        assert result.valid
        assert len(result.latencies) == 2
    finally:
        os.unlink(temp_path)


def test_parser_file_not_found():
    """Test parser returns invalid TrialData for missing file."""
    result = parse_data_file("/nonexistent/path/file.txt")
    assert not result.valid
    assert "File not found" in result.error_message


def test_parser_empty_file():
    """Test parser returns invalid TrialData for file with no data lines."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("# only comments\n")
        temp_path = f.name
    try:
        result = parse_data_file(temp_path)
        assert not result.valid
        assert "No valid data lines" in result.error_message
    finally:
        os.unlink(temp_path)


def test_metadata_too_few_parts():
    """Test metadata returns invalid for filename with fewer than 6 parts."""
    result = extract_metadata("A01_26_f_BISA.txt")
    assert not result.valid


def test_metadata_non_numeric_age():
    """Test metadata returns invalid when age field is not numeric."""
    result = extract_metadata("A01_abc_f_BISA_AD_TD.txt")
    assert not result.valid
    assert result.subject_id == "A01"
