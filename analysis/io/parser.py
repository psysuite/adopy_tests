"""
Parser module for reading trial data files.

This module provides functionality to parse trial data files in GBF or PSA format,
extracting latencies, counts, responses, and guess rate parameters.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class TrialData:
    """Represents parsed trial data from a file.
    
    Attributes:
        latencies: Stimulus presentation times (milliseconds)
        counts: Trial counts (1 for individual trials, >1 for binned data)
        responses: Binary responses (0 or 1)
        guess_rate: Guess rate parameter (default 0.0)
        valid: Whether parsing succeeded
        error_message: Error message if parsing failed
    """
    latencies: np.ndarray
    counts: np.ndarray
    responses: np.ndarray
    guess_rate: float
    valid: bool
    error_message: str


def parse_data_file(filepath: str) -> TrialData:
    """
    Parse a trial data file in GBF or PSA format.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        TrialData object containing parsed data
        
    Behavior:
        - Lines starting with '@' contain guess rate parameter
        - Lines starting with '#' or '*' are comments (skipped)
        - Data lines with 3+ values: [latency, count, response]
        - Data lines with 2 values: [latency, response], count=1
        - Returns empty data with valid=False on error
        
    Examples:
        >>> data = parse_data_file("subject_data.txt")
        >>> if data.valid:
        ...     print(f"Parsed {len(data.latencies)} trials")
        ...     print(f"Guess rate: {data.guess_rate}")
    """
    latencies_list = []
    counts_list = []
    responses_list = []
    guess_rate = 0.0
    
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Extract guess rate from @ lines
                if line.startswith('@'):
                    try:
                        guess_rate = float(line[1:].strip())
                    except ValueError:
                        pass  # Ignore malformed guess rate lines
                    continue
                
                # Skip comment lines
                if line.startswith('#') or line.startswith('*'):
                    continue
                
                # Try to parse data line
                try:
                    parts = line.split()
                    if len(parts) >= 3:
                        # 3+ values: [latency, count, response]
                        latency = float(parts[0])
                        count = int(parts[1])
                        response = int(parts[2])
                        latencies_list.append(latency)
                        counts_list.append(count)
                        responses_list.append(response)
                    elif len(parts) == 2:
                        # 2 values: [latency, response], count=1
                        latency = float(parts[0])
                        response = int(parts[1])
                        latencies_list.append(latency)
                        counts_list.append(1)
                        responses_list.append(response)
                    # Silently skip lines with < 2 values
                except (ValueError, IndexError):
                    # Silently skip malformed data lines
                    pass
        
        # Check if we got any valid data
        if not latencies_list:
            return TrialData(
                latencies=np.array([]),
                counts=np.array([]),
                responses=np.array([]),
                guess_rate=0.0,
                valid=False,
                error_message="No valid data lines found in file"
            )
        
        # Convert to numpy arrays
        return TrialData(
            latencies=np.array(latencies_list),
            counts=np.array(counts_list),
            responses=np.array(responses_list),
            guess_rate=guess_rate,
            valid=True,
            error_message=""
        )
        
    except FileNotFoundError:
        return TrialData(
            latencies=np.array([]),
            counts=np.array([]),
            responses=np.array([]),
            guess_rate=0.0,
            valid=False,
            error_message=f"File not found: {filepath}"
        )
    except Exception as e:
        return TrialData(
            latencies=np.array([]),
            counts=np.array([]),
            responses=np.array([]),
            guess_rate=0.0,
            valid=False,
            error_message=f"Error reading file: {str(e)}"
        )
