"""Test that shared fixtures work correctly."""

import pytest
from postprocessing.modules.parser import TrialData
from postprocessing.modules.metadata import SubjectMetadata
from postprocessing.modules.progressive_analyzer import ProgressiveResult


def test_sample_trial_data(sample_trial_data):
    """Test sample_trial_data fixture."""
    assert isinstance(sample_trial_data, TrialData)
    assert sample_trial_data.valid is True
    assert len(sample_trial_data.latencies) == 5


def test_sample_metadata(sample_metadata):
    """Test sample_metadata fixture."""
    assert isinstance(sample_metadata, SubjectMetadata)
    assert sample_metadata.valid is True
    assert sample_metadata.subject_id == "TEST001"


def test_sample_progressive_result(sample_progressive_result):
    """Test sample_progressive_result fixture."""
    assert isinstance(sample_progressive_result, ProgressiveResult)
    assert sample_progressive_result.method == "logistic"
    assert len(sample_progressive_result.trial_counts) == 9


def test_temp_dir(temp_dir):
    """Test temp_dir fixture."""
    assert temp_dir.exists()
    assert temp_dir.is_dir()


def test_create_gbf_file(create_gbf_file):
    """Test create_gbf_file factory fixture."""
    filepath = create_gbf_file("test_gbf.txt")
    assert filepath.exists()
    content = filepath.read_text()
    assert "@ 0.0" in content


def test_create_psa_file(create_psa_file):
    """Test create_psa_file factory fixture."""
    filepath = create_psa_file("test_psa.txt")
    assert filepath.exists()
    content = filepath.read_text()
    assert "lat" in content  # Changed from "latency" to "lat"
