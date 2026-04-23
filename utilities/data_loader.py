"""
Data loader - normalizes different data formats to unified trial arrays.

Supports:
  - Format 1: Tab-separated with header (lat, user_ans columns + others)
  - Format 2: Binned format with header (lat, count, resp columns)
  - Format 3: Binned format without header (lat, count, resp, confl_magn)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


class DataLoader:
    """Loads and normalizes data from different formats."""

    @staticmethod
    def load_and_expand(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data file and return expanded trial arrays.

        Detects format automatically:
        - If has 'lat' and 'user_ans' columns: direct format with header
        - If has 'lat', 'count', 'resp' columns: binned format with header
        - If 4 numeric columns: binned format without header (lat, count, resp, confl_magn)

        Args:
            filepath: Path to data file

        Returns:
            Tuple of (latencies, responses) as numpy arrays
            Each trial is a separate element (expanded from binned format if needed)

        Raises:
            ValueError: If file format is invalid
            FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Try to read without header first (for headerless files)
        try:
            df_no_header = pd.read_csv(filepath, sep='\t', header=None)
        except Exception as e:
            raise ValueError(f"Failed to read file {filepath}: {e}")

        if df_no_header.empty:
            raise ValueError(f"File is empty: {filepath}")

        # Try to read with header
        try:
            df_with_header = pd.read_csv(filepath, sep='\t')
            cols_lower = [c.lower() for c in df_with_header.columns]

            # Check for direct format (has 'lat' and 'user_ans')
            if 'lat' in cols_lower and 'user_ans' in cols_lower:
                return DataLoader._load_direct_format(df_with_header, cols_lower)
            
            # Check for binned format with header (has 'lat', 'count', 'resp')
            elif 'lat' in cols_lower and 'count' in cols_lower and 'user_ans' in cols_lower:
                return DataLoader._load_binned_format(df_with_header, cols_lower)
        except:
            pass

        # If no header detected, try binned format without header
        # Expected: lat, count, resp, confl_magn (4 columns)
        if len(df_no_header.columns) == 4:
            return DataLoader._load_binned_format_no_header(df_no_header)
        
        raise ValueError(
            f"Invalid format: expected ('lat' + 'user_ans') or ('lat' + 'count' + 'user_ans'), "
            f"got {df_no_header.columns.tolist() if len(df_no_header.columns) > 0 else 'unknown'}"
        )

    @staticmethod
    def _load_direct_format(df: pd.DataFrame, cols_lower: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load direct format: lat, user_ans (+ other columns)

        Args:
            df: DataFrame with columns including 'lat' and 'user_ans'
            cols_lower: Lowercase column names

        Returns:
            Tuple of (latencies, responses)
        """
        # Find actual column names (case-insensitive)
        lat_col = df.columns[cols_lower.index('lat')]
        resp_col = df.columns[cols_lower.index('user_ans')]

        latencies = df[lat_col].values.astype(float)
        responses = df[resp_col].values.astype(int)

        return latencies, responses

    @staticmethod
    def _load_binned_format(df: pd.DataFrame, cols_lower: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load binned format with header: lat, count, resp

        Expands each row by 'count' times.

        Args:
            df: DataFrame with columns including 'lat', 'count', 'user_ans'
            cols_lower: Lowercase column names

        Returns:
            Tuple of (latencies, responses) - expanded
        """
        # Find actual column names (case-insensitive)
        lat_col = df.columns[cols_lower.index('lat')]
        count_col = df.columns[cols_lower.index('count')]
        resp_col = df.columns[cols_lower.index('user_ans')]

        expanded_lat = []
        expanded_resp = []

        for _, row in df.iterrows():
            lat = float(row[lat_col])
            count = int(row[count_col])
            resp = int(row[resp_col])

            for _ in range(count):
                expanded_lat.append(lat)
                expanded_resp.append(resp)

        return np.array(expanded_lat), np.array(expanded_resp)

    @staticmethod
    def _load_binned_format_no_header(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load binned format without header: lat, count, resp, confl_magn

        Expands each row by 'count' times.

        Args:
            df: DataFrame with 4 columns (no header)

        Returns:
            Tuple of (latencies, responses) - expanded
        """
        expanded_lat = []
        expanded_resp = []

        for _, row in df.iterrows():
            lat = float(row[0])      # Column 0: latency
            count = int(row[1])      # Column 1: count
            resp = int(row[2])       # Column 2: response
            # Column 3: confl_magn (ignored)

            for _ in range(count):
                expanded_lat.append(lat)
                expanded_resp.append(resp)

        return np.array(expanded_lat), np.array(expanded_resp)
