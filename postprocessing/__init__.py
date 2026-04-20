"""
MATLAB to Python Postprocessing Package

This package converts MATLAB postprocessing scripts for temporal bisection
psychometric data analysis into Python. It provides modules for:
- Parsing trial data files
- Extracting metadata from filenames
- Converting PSA format to GBF format
- Fitting psychometric curves (logistic, probit, gaussfit)
- Progressive analysis at multiple trial counts
- Generating Excel reports in wide and long formats
"""

__version__ = "1.0.0"
__author__ = "Temporal Bisection Research Team"

from .modules.parser import TrialData, parse_data_file
from .modules.metadata import SubjectMetadata, extract_metadata
from utilities.psychometric_helpers import calculate_latency_statistics
from .modules.converter import convert_psa_to_gbf
from .modules.progressive_analyzer import analyze_batch, analyze_subject_progressive, ProgressiveResult
from .modules.fitters import fit_logistic, fit_probit, fit_gaussfit, FitResult

__all__ = [
    "TrialData",
    "parse_data_file",
    "SubjectMetadata",
    "extract_metadata",
    "calculate_latency_statistics",
    "convert_psa_to_gbf",
    "analyze_batch",
    "analyze_subject_progressive",
    "ProgressiveResult",
    "fit_logistic",
    "fit_probit",
    "fit_gaussfit",
    "FitResult",
]
