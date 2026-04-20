"""
Metadata module for extracting subject information from filenames.

This module provides functionality to parse standardized filenames and extract
subject metadata including ID, age, gender, modality, algorithm, and group.
"""

from dataclasses import dataclass
import os


@dataclass
class SubjectMetadata:
    """Subject metadata extracted from filename.
    
    Attributes:
        subject_id: Subject identifier string
        age: Subject age in years
        gender: Gender code (e.g., 'm', 'f')
        modality: Experimental modality (e.g., 'BISA', 'BISV')
        algorithm: Algorithm type (e.g., 'AD', 'FX')
        group: Group identifier (e.g., 'TD', 'ASD')
        valid: Whether metadata extraction succeeded
        filename: Original filename
    """
    subject_id: str
    age: int
    gender: str
    modality: str
    algorithm: str
    group: str
    valid: bool
    filename: str


def extract_metadata(filename: str) -> SubjectMetadata:
    """
    Extract metadata from filename pattern: {ID}_{age}_{gender}_{modality}_{algorithm}_{group}.txt
    
    Args:
        filename: Name of the data file (with or without path)
        
    Returns:
        SubjectMetadata object with extracted fields
        
    Behavior:
        - Splits filename by underscores
        - Validates 6 fields present
        - Converts age to integer
        - Returns valid=False if parsing fails
        
    Examples:
        >>> metadata = extract_metadata("A01_26_f_BISA_AD_TD.txt")
        >>> if metadata.valid:
        ...     print(f"Subject: {metadata.subject_id}, Age: {metadata.age}")
        Subject: A01, Age: 26
    """
    # Extract just the filename without path
    base_filename = os.path.basename(filename)
    
    # Remove .txt extension if present
    if base_filename.endswith('.txt'):
        base_filename = base_filename[:-4]
    
    # Split by underscores
    parts = base_filename.split('_')
    
    # Validate we have exactly 6 parts
    if len(parts) < 6:
        return SubjectMetadata(
            subject_id="",
            age=0,
            gender="",
            modality="",
            algorithm="",
            group="",
            valid=False,
            filename=filename
        )
    
    # Extract fields
    subject_id = parts[0]
    age_str = parts[1]
    gender = parts[2]
    modality = parts[3]
    algorithm = parts[4]
    group = parts[5]
    
    # Try to convert age to integer
    try:
        age = int(age_str)
    except ValueError:
        return SubjectMetadata(
            subject_id=subject_id,
            age=0,
            gender=gender,
            modality=modality,
            algorithm=algorithm,
            group=group,
            valid=False,
            filename=filename
        )
    
    return SubjectMetadata(
        subject_id=subject_id,
        age=age,
        gender=gender,
        modality=modality,
        algorithm=algorithm,
        group=group,
        valid=True,
        filename=filename
    )
