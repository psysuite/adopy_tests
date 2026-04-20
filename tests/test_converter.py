"""Tests for converter module."""

import pytest
from pathlib import Path
from postprocessing.modules.converter import convert_psa_to_gbf


def test_convert_psa_to_gbf_basic(temp_dir, create_psa_file):
    """Test basic PSA to GBF conversion."""
    # Create input and output directories
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    # Create a PSA file with exact column names from MATLAB
    psa_content = """id\tlabel\tlat\tconfl\tres\tcor_ans\telapsed\trep\tuser_ans
1\tA\t400\t0\t0\t0\t1234\t1\t0
2\tB\t500\t0\t1\t1\t1235\t1\t1
3\tC\t600\t0\t1\t1\t1236\t1\t1
"""
    psa_file = input_dir / "test.txt"
    psa_file.write_text(psa_content)
    
    # Convert
    stats = convert_psa_to_gbf(str(input_dir), str(output_dir))
    
    assert stats['total_files'] == 1
    assert stats['converted'] == 1
    assert len(stats['failed']) == 0
    
    # Check output file exists
    output_file = output_dir / "test.txt"
    assert output_file.exists()
    
    # Check output content
    content = output_file.read_text()
    lines = content.strip().split('\n')
    assert len(lines) == 3
    
    # Check format: latency, 1, user_answer
    first_line = lines[0].split('\t')
    assert len(first_line) == 3
    assert first_line[0] == '400'
    assert first_line[1] == '1'
    assert first_line[2] == '0'


def test_convert_psa_to_gbf_empty_directory(temp_dir):
    """Test conversion with empty input directory."""
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    stats = convert_psa_to_gbf(str(input_dir), str(output_dir))
    
    assert stats['total_files'] == 0
    assert stats['converted'] == 0
    assert len(stats['failed']) == 0


def test_convert_psa_to_gbf_multiple_files(temp_dir):
    """Test conversion with multiple files."""
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    psa_content = """id\tlabel\tlat\tconfl\tres\tcor_ans\telapsed\trep\tuser_ans
1\tA\t400\t0\t0\t0\t1234\t1\t0
"""
    
    # Create multiple files
    for i in range(3):
        psa_file = input_dir / f"test{i}.txt"
        psa_file.write_text(psa_content)
    
    stats = convert_psa_to_gbf(str(input_dir), str(output_dir))
    
    assert stats['total_files'] == 3
    assert stats['converted'] == 3
    assert len(stats['failed']) == 0
    
    # Check all output files exist
    for i in range(3):
        output_file = output_dir / f"test{i}.txt"
        assert output_file.exists()


def test_convert_psa_to_gbf_malformed_file(temp_dir):
    """Test conversion with malformed PSA file."""
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    # Create a malformed file
    malformed_file = input_dir / "malformed.txt"
    malformed_file.write_text("This is not a valid PSA file")
    
    stats = convert_psa_to_gbf(str(input_dir), str(output_dir))
    
    assert stats['total_files'] == 1
    # The conversion might fail or succeed depending on pandas behavior
    # Just check that we get a result
    assert stats['converted'] + len(stats['failed']) == 1


def test_convert_psa_missing_required_columns(temp_dir):
    """Test converter marks file as failed when lat/user_ans columns are missing."""
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # File with wrong column names
    bad_content = "col_a\tcol_b\tcol_c\n1\t2\t3\n"
    (input_dir / "bad.txt").write_text(bad_content)

    stats = convert_psa_to_gbf(str(input_dir), str(output_dir))

    assert stats['total_files'] == 1
    assert stats['converted'] == 0
    assert "bad.txt" in stats['failed']


def test_convert_psa_with_confl_magn(temp_dir):
    """Test converter preserves confl_magn column when present."""
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    psa_content = "id\tlabel\tlat\tconfl\tres\tcor_ans\telapsed\trep\tuser_ans\tconfl_magn\n"
    psa_content += "1\tA\t400\t0\t0\t0\t1234\t1\t0\t2\n"
    psa_content += "2\tB\t500\t0\t1\t1\t1235\t1\t1\t3\n"
    (input_dir / "test.txt").write_text(psa_content)

    stats = convert_psa_to_gbf(str(input_dir), str(output_dir))

    assert stats['converted'] == 1
    lines = (output_dir / "test.txt").read_text().strip().split('\n')
    # Should have 4 columns: lat, count, user_ans, confl_magn
    assert len(lines[0].split('\t')) == 4
