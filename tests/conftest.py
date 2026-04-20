"""Shared pytest fixtures for postprocessing tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
from postprocessing.modules.parser import TrialData
from postprocessing.modules.metadata import SubjectMetadata
from postprocessing.modules.progressive_analyzer import ProgressiveResult


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_trial_data():
    """Create sample TrialData for testing."""
    return TrialData(
        latencies=[400, 500, 600, 700, 800],
        counts=[5, 5, 5, 5, 5],
        responses=[0, 2, 3, 4, 5],
        guess_rate=0.0,
        valid=True,
        error_message=""
    )


@pytest.fixture
def sample_metadata():
    """Create sample SubjectMetadata for testing."""
    return SubjectMetadata(
        subject_id="TEST001",
        age=25,
        gender="M",
        modality="BISA",
        algorithm="AD",
        group="control",
        valid=True,
        filename="TEST001_25_M_BISA_AD_control.txt"
    )


@pytest.fixture
def sample_progressive_result(sample_metadata):
    """Create sample ProgressiveResult for testing."""
    trial_counts = [40, 60, 80, 100, 120, 140, 160, 180, 200]
    
    result = ProgressiveResult(
        metadata=sample_metadata,
        trial_counts=trial_counts,
        pse_values={n: 600.0 + n * 0.1 for n in trial_counts},
        jnd_values={n: 50.0 + n * 0.05 for n in trial_counts},
        lat_mean={n: 550.0 + n * 0.2 for n in trial_counts},
        lat_std={n: 100.0 + n * 0.1 for n in trial_counts},
        lat_range={n: 400.0 + n * 0.3 for n in trial_counts},
        lat_entropy={n: 2.5 + n * 0.001 for n in trial_counts},
        method="logistic"
    )
    
    return result


@pytest.fixture
def gbf_file_content():
    """Sample GBF format file content."""
    return """@ 0.0
# Comment line
* Another comment
400 5 0
500 5 2
600 5 3
700 5 4
800 5 5
"""


@pytest.fixture
def psa_file_content():
    """Sample PSA format file content."""
    return """id\tlabel\tlat\tconfl\tres\tcor_ans\telapsed\trep\tuser_ans
1\tA\t400\t0\t0\t0\t1234\t1\t0
2\tB\t500\t0\t1\t1\t1235\t1\t1
3\tC\t600\t0\t1\t1\t1236\t1\t1
4\tD\t700\t0\t1\t1\t1237\t1\t1
5\tE\t800\t0\t1\t1\t1238\t1\t1
"""


@pytest.fixture
def create_gbf_file(temp_dir, gbf_file_content):
    """Factory fixture to create GBF test files."""
    def _create_file(filename="test.txt", content=None):
        filepath = temp_dir / filename
        filepath.write_text(content or gbf_file_content)
        return filepath
    return _create_file


@pytest.fixture
def create_psa_file(temp_dir, psa_file_content):
    """Factory fixture to create PSA test files."""
    def _create_file(filename="test.txt", content=None):
        filepath = temp_dir / filename
        filepath.write_text(content or psa_file_content)
        return filepath
    return _create_file
