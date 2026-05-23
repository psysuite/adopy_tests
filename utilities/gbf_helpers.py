"""
Helper functions for working with GBF files and trial data.
"""

from typing import List, Dict


def calculate_trial_success(lat: float, user_ans: int, offset: float = 500) -> bool:
    """
    Calculate if a trial was successful based on bisection task logic.
    
    Success rule:
    - If lat > offset and user_ans == 1 (responded "longer"): success
    - If lat < offset and user_ans == 0 (responded "shorter"): success
    - Otherwise: failure
    
    Args:
        lat: Stimulus latency in ms
        user_ans: User response (0 = "shorter", 1 = "longer")
        offset: Reference latency (default 500ms)
        
    Returns:
        True if trial was successful, False otherwise
    """
    if lat > offset and user_ans == 1:
        return True
    elif lat < offset and user_ans == 0:
        return True
    else:
        return False


def load_gbf_with_success(gbf_path: str, offset: float = 500) -> List[Dict]:
    """
    Load GBF file and add success/failure field.
    
    Args:
        gbf_path: Path to GBF file
        offset: Reference latency for success calculation (default 500ms)
        
    Returns:
        List of trial dicts with lat, user_ans, and res fields
        
    Raises:
        FileNotFoundError: If GBF file not found
        ValueError: If GBF file format is invalid
    """
    rows = []
    
    try:
        with open(gbf_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    lat = float(parts[0])
                    user_ans = int(parts[2])
                    success = calculate_trial_success(lat, user_ans, offset)
                    
                    rows.append({
                        'lat': lat,
                        'user_ans': user_ans,
                        'res': 'true' if success else 'false',
                    })
    except FileNotFoundError:
        raise FileNotFoundError(f"GBF file not found: {gbf_path}")
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid GBF file format in {gbf_path}: {e}")
    
    return rows
