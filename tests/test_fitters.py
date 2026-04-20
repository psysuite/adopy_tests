"""Tests for fitters module."""

import pytest
import numpy as np
from postprocessing.modules.fitters import fit_logistic, fit_probit, fit_gaussfit, FitResult


def test_fit_logistic_basic():
    """Test basic logistic fitting."""
    # Create synthetic data with more realistic psychometric curve
    # Use more data points with gradual transition
    latencies = np.array([300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900])
    counts = np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30])
    responses = np.array([1, 2, 4, 7, 10, 14, 18, 22, 25, 27, 28, 29, 30])
    
    result = fit_logistic(latencies, counts, responses)
    
    assert isinstance(result, FitResult)
    assert result.method == "logistic"
    assert result.success
    assert 200 < result.pse < 800  # PSE should be within data range
    assert result.jnd > 0


def test_fit_probit_basic():
    """Test basic probit fitting."""
    # Create synthetic data with more realistic psychometric curve
    latencies = np.array([300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900])
    counts = np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30])
    responses = np.array([1, 2, 4, 7, 10, 14, 18, 22, 25, 27, 28, 29, 30])
    
    result = fit_probit(latencies, counts, responses)
    
    assert isinstance(result, FitResult)
    assert result.method == "probit"
    assert result.success
    assert 200 < result.pse < 800  # PSE should be within data range
    assert result.jnd > 0


def test_fit_gaussfit_basic():
    """Test basic gaussfit fitting."""
    # Create synthetic data
    latencies = np.array([400, 500, 600, 700, 800])
    counts = np.array([10, 10, 10, 10, 10])
    responses = np.array([1, 3, 5, 7, 9])
    
    result = fit_gaussfit(latencies, counts, responses, bin_size=50, guess_rate=0.0)
    
    assert isinstance(result, FitResult)
    assert result.method == "gaussfit"
    assert result.success
    assert result.jnd > 0


def test_fit_logistic_insufficient_trials():
    """Test logistic with insufficient trials."""
    latencies = np.array([400, 500])
    counts = np.array([1, 1])
    responses = np.array([0, 1])
    
    result = fit_logistic(latencies, counts, responses)
    
    assert not result.success
    assert np.isnan(result.pse)
    assert np.isnan(result.jnd)


def test_fit_logistic_identical_responses():
    """Test logistic with all identical responses."""
    latencies = np.array([400, 500, 600, 700, 800])
    counts = np.array([5, 5, 5, 5, 5])
    responses = np.array([5, 5, 5, 5, 5])  # All correct
    
    result = fit_logistic(latencies, counts, responses)
    
    assert not result.success
    assert np.isnan(result.pse)
    assert np.isnan(result.jnd)


def test_fit_logistic_few_stimuli():
    """Test logistic with too few unique stimuli."""
    latencies = np.array([400, 400])
    counts = np.array([5, 5])
    responses = np.array([2, 3])
    
    result = fit_logistic(latencies, counts, responses)
    
    assert not result.success
    assert np.isnan(result.pse)
    assert np.isnan(result.jnd)


def test_fit_probit_insufficient_trials():
    """Test probit with insufficient trials."""
    latencies = np.array([400, 500])
    counts = np.array([1, 1])
    responses = np.array([0, 1])
    
    result = fit_probit(latencies, counts, responses)
    
    assert not result.success
    assert np.isnan(result.pse)
    assert np.isnan(result.jnd)


def test_fit_gaussfit_insufficient_trials():
    """Test gaussfit with insufficient trials after unbinning."""
    latencies = np.array([400, 500])
    counts = np.array([1, 1])
    responses = np.array([0, 1])
    
    result = fit_gaussfit(latencies, counts, responses, bin_size=50, guess_rate=0.0)
    
    assert not result.success
    assert np.isnan(result.pse)
    assert np.isnan(result.jnd)


def test_fit_result_dataclass():
    """Test FitResult dataclass."""
    result = FitResult(
        pse=600.0,
        jnd=50.0,
        method="logistic",
        success=True,
        error_message=""
    )
    
    assert result.pse == 600.0
    assert result.jnd == 50.0
    assert result.method == "logistic"
    assert result.success


def test_fit_logistic_glm_fallback():
    """Test logistic fallback when GLM fails (all-identical latencies after expansion)."""
    # Two unique latencies only → _validate_data fails with 'Too few unique stimulus levels'
    latencies = np.array([400, 400, 500, 500])
    counts    = np.array([3, 3, 3, 3])
    responses = np.array([1, 2, 1, 2])

    result = fit_logistic(latencies, counts, responses)
    assert not result.success
    assert np.isnan(result.pse)


def test_fit_probit_glm_fallback():
    """Test probit fallback when GLM fails (too few unique stimuli)."""
    latencies = np.array([400, 400, 500, 500])
    counts    = np.array([3, 3, 3, 3])
    responses = np.array([1, 2, 1, 2])

    result = fit_probit(latencies, counts, responses)
    assert not result.success
    assert np.isnan(result.pse)


def test_fit_gaussfit_insufficient_after_binning():
    """Test gaussfit returns failure when unbinned data has < 3 unique stimuli."""
    # All latencies fall in the same bin → only 1 unique stimulus after binning
    latencies = np.array([500, 500, 500])
    counts    = np.array([1, 1, 1])
    responses = np.array([0, 1, 0])

    result = fit_gaussfit(latencies, counts, responses, guess_rate=0.0, bin_size=200.0)
    assert not result.success
    assert np.isnan(result.pse)


def test_fit_gaussfit_invalid_parameters():
    """Test gaussfit returns failure when grid search yields no finite result."""
    # Constant responses → gaussfit_psychometric returns nan
    latencies = np.array([400, 500, 600, 700, 800])
    counts    = np.array([5, 5, 5, 5, 5])
    responses = np.array([0, 0, 0, 0, 0])  # all zeros → identical responses

    result = fit_gaussfit(latencies, counts, responses, guess_rate=0.0, bin_size=50.0)
    assert not result.success
    assert np.isnan(result.pse)
