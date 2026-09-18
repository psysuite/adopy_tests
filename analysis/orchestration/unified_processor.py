"""
Unified GBF Processing - Calculate metrics and generate Excel reports (wide + long format).

Works for BOTH synthetic and real data with same code path.
Only difference: metadata columns (synthetic: pse_true, jnd_true, group vs real: subj_id, age, gender, etc.)

Usage:
    processor = UnifiedGBFProcessor(
        gbf_dir='/path/to/gbf',
        model_name='ABS1',
        data_type='synthetic',  # or 'real'
        output_dir='/path/to/output',
        pse_grid=[480, 500, 520],  # Only for synthetic
        jnd_grid=[20, 40, 60],      # Only for synthetic
        offset=500
    )
    
    df_wide, df_long = processor.process()
    processor.save_excel(output_dir, filename_prefix='results')
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np

from main.config import TRIAL_BLOCKS
from analysis.orchestration.metrics import MetricsCalculator
from analysis.io.converter import read_gbf_file
from analysis.io.report_generator import (
    save_wide_format_to_excel,
    generate_long_format_from_dataframe,
    save_long_format_to_excel,
    _validate_excel_path,
)

logger = logging.getLogger(__name__)


class UnifiedGBFProcessor:
    """
    Unified processor for GBF files (synthetic or real data).
    
    Reads GBF files, calculates metrics, generates wide and long format DataFrames.
    """
    
    def __init__(
        self,
        gbf_dir: Path,
        model_name: Optional[str] = None,
        data_type: str = 'synthetic',  # 'synthetic' or 'real'
        output_dir: Optional[Path] = None,
        pse_grid: Optional[List[float]] = None,
        jnd_grid: Optional[List[float]] = None,
        offset: float = 500,
        trial_blocks: Optional[List[int]] = None,
        verbose: bool = True,
        use_multithread: bool = True,
    ):
        """
        Initialize processor.
        
        Args:
            gbf_dir: Directory containing GBF files
            model_name: 'ABS1', 'REL1', or 'REL2' (synthetic only; None for real data)
                        Note: For real data, model_name is not used in metrics calculation
                        (PosteriorExtractor always uses ABS1 as common model)
            data_type: 'synthetic' or 'real'
            output_dir: Where to save Excel files
            pse_grid: PSE grid (synthetic only)
            jnd_grid: JND grid (synthetic only)
            offset: Reference latency in ms
            trial_blocks: Trial blocks for metrics (default: [40, 60, 80, 100, 120, 140, 160, 180, 200])
            verbose: Print progress
            use_multithread: Use multithreading for processing (default True)
        """
        self.gbf_dir = Path(gbf_dir)
        self.model_name = model_name
        self.data_type = data_type
        self.output_dir = Path(output_dir) if output_dir else self.gbf_dir
        self.pse_grid = pse_grid or []
        self.jnd_grid = jnd_grid or []
        self.offset = offset
        self.trial_blocks = trial_blocks or TRIAL_BLOCKS
        self.verbose = verbose
        self.use_multithread = use_multithread
        
        # Derive data_type-specific filenames (avoid real/synthetic collision)
        self.wide_filename = f'{data_type}_data_wide.xlsx'
        self.long_filename = f'{data_type}_data_long.xlsx'
        
        # Validate
        if data_type not in ['synthetic', 'real']:
            raise ValueError(f"data_type must be 'synthetic' or 'real', got {data_type}")
        
        if data_type == 'synthetic':
            if not model_name:
                raise ValueError("model_name required for synthetic data")
            if not pse_grid or not jnd_grid:
                raise ValueError("pse_grid and jnd_grid required for synthetic data")
        
        # Results
        self.df_wide = None
        self.df_long = None
        self._all_rows_wide = []
        self._all_rows_long = []
        self._processed_subjects = set()  # Track which subject_id already processed
        
        # Multithread lock for Excel append
        import threading
        self._append_lock = threading.Lock()
    
    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process all GBF files in directory.
        
        If use_multithread=True:
        - Load existing wide Excel (if exists, overwrite=False)
        - Skip already-processed subjects
        - Process remaining files in parallel
        - Append incrementally to Excel
        
        Returns:
            (df_wide, df_long)
        """
        import time
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"UNIFIED GBF PROCESSING - {self.data_type.upper()}")
            print(f"{'='*70}\n")
            print(f"GBF directory: {self.gbf_dir}")
            print(f"Model: {self.model_name}")
            print(f"Data type: {self.data_type}")
            print(f"Multithread: {self.use_multithread}\n")
        
        # Load existing wide Excel if available
        self._load_existing_wide()
        
        # Discover GBF structure
        t0 = time.time()
        gbf_files = self._discover_gbf_files()
        t_discover = time.time() - t0
        
        if not gbf_files:
            logger.warning(f"No GBF files found in {self.gbf_dir}")
            return self.df_wide if self.df_wide is not None else pd.DataFrame(), \
                   self.df_long if self.df_long is not None else pd.DataFrame()
        
        # Filter out already-processed subjects
        if self.data_type == 'synthetic':
            gbf_files_to_process = [
                (path, meta) for path, meta in gbf_files 
                if (meta.get('subject_id'), self.model_name) not in self._processed_subjects
            ]
        else:  # real
            gbf_files_to_process = [
                (path, meta) for path, meta in gbf_files 
                if meta.get('subj') not in self._processed_subjects
            ]
        
        if self.verbose:
            print(f"Found {len(gbf_files)} total GBF files (discovery took {t_discover:.2f}s)")
            if len(gbf_files_to_process) < len(gbf_files):
                print(f"  {len(gbf_files) - len(gbf_files_to_process)} already processed, skipping")
            print(f"  Processing {len(gbf_files_to_process)} files\n")
        
        if not gbf_files_to_process:
            if self.verbose:
                print("✓ All files already processed\n")
            # Generate long format from existing wide if needed
            if self.df_wide is not None and not self.df_wide.empty:
                self.df_long = generate_long_format_from_dataframe(self.df_wide, self.data_type)
            return self.df_wide, self.df_long
        
        # Process files (multithread or sequential)
        t_start = time.time()
        if self.use_multithread:
            self._process_multithread(gbf_files_to_process)
        else:
            self._process_sequential(gbf_files_to_process)
        
        t_process = time.time() - t_start
        
        if self.verbose:
            print(f"  Total processing time: {t_process:.2f}s\n")
        
        # Create/Update DataFrames
        if self.df_wide is None or self.df_wide.empty:
            self.df_wide = pd.DataFrame(self._all_rows_wide)
        else:
            # Append new rows to existing df_wide
            if self._all_rows_wide:
                df_new = pd.DataFrame(self._all_rows_wide)
                self.df_wide = pd.concat([self.df_wide, df_new], ignore_index=True)
        
        # Generate long format at the end (fast)
        t_long_start = time.time()
        self.df_long = generate_long_format_from_dataframe(self.df_wide, self.data_type)
        t_long = time.time() - t_long_start
        
        if self.verbose:
            print(f"✓ Processing complete ({t_long:.2f}s for long format):")
            print(f"  Wide format: {len(self.df_wide)} rows")
            print(f"  Long format: {len(self.df_long)} rows\n")
        
        return self.df_wide, self.df_long
    
    def _load_existing_wide(self) -> None:
        """
        Load existing wide Excel file and track processed subjects.
        
        For synthetic data: track (subject_id, model) pairs
        For real data: track subj only (one subject = one row)
        
        Only called if use_multithread=True (for incremental processing).
        """
        wide_path = self.output_dir / self.wide_filename  # Use data_type-specific filename
        if not wide_path.exists():
            return
        
        try:
            self.df_wide = pd.read_excel(wide_path)
            
            # Extract processed subjects - INCLUDING model for synthetic data
            if self.data_type == 'synthetic':
                if 'subject_id' in self.df_wide.columns and 'model' in self.df_wide.columns:
                    # Track (subject_id, model) pairs - skip only if BOTH match
                    self._processed_subjects = set(
                        zip(self.df_wide['subject_id'], self.df_wide['model'])
                    )
            else:  # real
                if 'subj' in self.df_wide.columns:
                    self._processed_subjects = set(self.df_wide['subj'].unique())
            
            if self.verbose:
                print(f"[INCREMENTAL] Loaded existing wide Excel: {len(self.df_wide)} rows")
                print(f"  Already processed {len(self._processed_subjects)} (subject, model) pairs\n")
        
        except Exception as e:
            logger.warning(f"Could not load existing wide Excel: {e}")
            self.df_wide = None
            self._processed_subjects = set()
    
    def _process_sequential(self, gbf_files: List[Tuple[Path, Dict[str, Any]]]) -> None:
        """
        Process GBF files sequentially.
        
        Args:
            gbf_files: List of (gbf_path, metadata) tuples
        """
        for idx, (gbf_path, metadata) in enumerate(gbf_files, 1):
            self._process_gbf_file(gbf_path, metadata)
            
            if self.verbose and idx % max(1, len(gbf_files) // 5) == 0:
                print(f"  [{idx}/{len(gbf_files)}] {gbf_path.name}")
    
    def _process_multithread(self, gbf_files: List[Tuple[Path, Dict[str, Any]]]) -> None:
        """
        Process GBF files in parallel using ThreadPoolExecutor.
        
        Each thread appends its result to Excel with lock protection.
        
        Args:
            gbf_files: List of (gbf_path, metadata) tuples
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from main.config import MAX_WORKERS
        
        if self.verbose:
            print(f"Using {MAX_WORKERS} worker threads...\n")
        
        processed_count = [0]  # Use list to allow modification in nested function
        
        def worker(gbf_path: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
            """Process single file and return wide row."""
            try:
                from analysis.io.converter import read_gbf_file
                
                # Read GBF file
                gbf_rows = read_gbf_file(str(gbf_path))
                if not gbf_rows:
                    logger.warning(f"Empty GBF file: {gbf_path}")
                    return None
                
                # Extract latencies and responses
                latencies = [float(row['lat']) for row in gbf_rows]
                responses = [int(row['user_ans']) for row in gbf_rows]
                
                # Get ground truth for synthetic
                ground_truth_pse = metadata.get('pse_true') if self.data_type == 'synthetic' else None
                ground_truth_jnd = metadata.get('jnd_true') if self.data_type == 'synthetic' else None
                
                # Calculate metrics
                model_type = self.model_name if self.data_type == 'synthetic' else 'ABS1'
                calc = MetricsCalculator(
                    model_type=model_type,
                    is_synthetic=(self.data_type == 'synthetic'),
                    ground_truth_pse=ground_truth_pse,
                    ground_truth_jnd=ground_truth_jnd,
                    offset=self.offset,
                    trial_blocks=self.trial_blocks,
                )
                metrics = calc.calculate_all_metrics(gbf_rows=gbf_rows)
                
                # Create wide format row
                wide_row = self._create_wide_row(metadata, latencies, responses, metrics)
                return wide_row
            
            except Exception as e:
                logger.error(f"Error processing {gbf_path}: {e}")
                return None
        
        # Submit all tasks
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(worker, gbf_path, metadata): (gbf_path, metadata)
                for gbf_path, metadata in gbf_files
            }
            
            # Process completed tasks
            for future in as_completed(futures):
                gbf_path, metadata = futures[future]
                
                try:
                    wide_row = future.result()
                    if wide_row:
                        # Append to Excel with lock
                        self._append_row_to_excel(wide_row)
                        processed_count[0] += 1
                        
                        if self.verbose and processed_count[0] % max(1, len(gbf_files) // 5) == 0:
                            print(f"  [{processed_count[0]}/{len(gbf_files)}] {gbf_path.name}")
                
                except Exception as e:
                    logger.error(f"Error retrieving result for {gbf_path}: {e}")
    
    def _append_row_to_excel(self, wide_row: Dict[str, Any]) -> None:
        """
        Append a single row to the wide Excel file with lock protection.
        
        Args:
            wide_row: Row dict to append
        """
        with self._append_lock:
            # Load or initialize wide DataFrame (use data_type-specific filename)
            wide_path = self.output_dir / self.wide_filename
            
            # Validate path for security
            try:
                wide_path = _validate_excel_path(str(wide_path))
            except ValueError as e:
                logger.error(f"Invalid Excel path: {e}")
                return
            
            if wide_path.exists():
                try:
                    df = pd.read_excel(wide_path)
                except Exception as e:
                    logger.warning(f"Could not read existing Excel, creating new: {e}")
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()
            
            # Append new row
            df = pd.concat([df, pd.DataFrame([wide_row])], ignore_index=True)
            
            # Write back
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                df.to_excel(wide_path, index=False)
            except Exception as e:
                logger.error(f"Error appending to Excel: {e}")
            
            # Also append to in-memory list
            self._all_rows_wide.append(wide_row)
    
    def _discover_gbf_files(self) -> List[Tuple[Path, Dict[str, Any]]]:
        """
        Discover GBF files based on data_type structure.
        
        Returns:
            List of (gbf_path, metadata_dict) tuples
        """
        gbf_files = []
        
        if self.data_type == 'synthetic':
            # Structure: gbf_dir/group_{pse}_{jnd}/S{subj}_G{group}_{pse}_{jnd}_{model}.txt
            for group_dir in sorted(self.gbf_dir.glob('group_*_*')):
                if not group_dir.is_dir():
                    continue
                
                # Extract PSE, JND from dirname
                try:
                    parts = group_dir.name.split('_')
                    pse_center = float(parts[1])
                    jnd_center = float(parts[2])
                except (IndexError, ValueError):
                    continue
                
                # Find GBF files in this group
                for gbf_path in sorted(group_dir.glob('*.txt')):
                    # Parse filename: S{subj}_G{group}_{pse}_{jnd}_{model}.txt
                    filename = gbf_path.stem
                    try:
                        parts = filename.split('_')
                        subj_id = parts[0]  # S123
                        group_idx = parts[1]  # G1
                        pse_true = float(parts[2])
                        jnd_true = float(parts[3])
                        
                        metadata = {
                            'subject_id': subj_id,  # Key for incremental processing
                            'group_idx': group_idx,
                            'pse_true': pse_true,
                            'jnd_true': jnd_true,
                            'pse_center': pse_center,
                            'jnd_center': jnd_center,
                        }
                        gbf_files.append((gbf_path, metadata))
                    except (IndexError, ValueError):
                        logger.warning(f"Could not parse GBF filename: {filename}")
                        continue
        
        else:  # real data
            # Structure: flat directory with GBF files
            # Filename: {subj}_{age}_{gender}_{modality}_{algorithm}_{group}.txt
            # Example: A01_26_f_BISA_AD_TD.txt
            for gbf_path in sorted(self.gbf_dir.glob('*.txt')):
                filename = gbf_path.stem
                try:
                    parts = filename.split('_')
                    if len(parts) >= 5:
                        subj_id = parts[0]      # A01
                        age = int(parts[1])     # 26
                        gender = parts[2]       # f
                        modality = parts[3]     # BISA
                        algorithm = parts[4]    # AD
                        group = parts[5] if len(parts) > 5 else 'TD'  # TD
                        
                        metadata = {
                            'subj': subj_id,  # Key for incremental processing
                            'age': age,
                            'gender': gender,
                            'modality': modality,
                            'algorithm': algorithm,
                            'group': group,
                        }
                        gbf_files.append((gbf_path, metadata))
                    else:
                        logger.warning(f"Could not parse real data GBF filename: {filename}")
                except (IndexError, ValueError) as e:
                    logger.warning(f"Could not parse real data GBF filename {filename}: {e}")
                    continue
        
        return gbf_files
    
    def _process_gbf_file(self, gbf_path: Path, metadata: Dict[str, Any]) -> None:
        """
        Process single GBF file: read, calculate metrics, create row.
        
        Args:
            gbf_path: Path to GBF file
            metadata: Metadata dict from filename/structure
        """
        import time
        
        try:
            t_total = time.time()
            
            # Read GBF file
            t_read = time.time()
            gbf_rows = read_gbf_file(str(gbf_path))
            t_read = time.time() - t_read
            
            if not gbf_rows:
                logger.warning(f"Empty GBF file: {gbf_path}")
                return
            
            # Extract latencies and responses
            t_extract = time.time()
            latencies = [float(row['lat']) for row in gbf_rows]
            responses = [int(row['user_ans']) for row in gbf_rows]
            t_extract = time.time() - t_extract
            
            # Get ground truth for synthetic
            ground_truth_pse = metadata.get('pse_true') if self.data_type == 'synthetic' else None
            ground_truth_jnd = metadata.get('jnd_true') if self.data_type == 'synthetic' else None
            
            # For synthetic: use model_name from config
            # For real: use 'ABS1' as default (not used in actual metrics, just placeholder)
            model_type = self.model_name if self.data_type == 'synthetic' else 'ABS1'
            
            # Calculate metrics
            t_calc = time.time()
            calc = MetricsCalculator(
                model_type=model_type,
                is_synthetic=(self.data_type == 'synthetic'),
                ground_truth_pse=ground_truth_pse,
                ground_truth_jnd=ground_truth_jnd,
                offset=self.offset,
                trial_blocks=self.trial_blocks,
            )
            t_calc_init = time.time() - t_calc
            
            t_metrics = time.time()
            metrics = calc.calculate_all_metrics(gbf_rows=gbf_rows)
            t_metrics = time.time() - t_metrics
            
            # Create wide format row
            t_row = time.time()
            wide_row = self._create_wide_row(metadata, latencies, responses, metrics)
            self._all_rows_wide.append(wide_row)
            t_row = time.time() - t_row
            
            t_total = time.time() - t_total
            
            if self.verbose and t_total > 0.5:  # Print only if takes > 0.5s
                print(f"    {gbf_path.name}:")
                print(f"      read={t_read:.3f}s, extract={t_extract:.3f}s, calc_init={t_calc_init:.3f}s")
                print(f"      metrics={t_metrics:.3f}s, row={t_row:.3f}s | TOTAL={t_total:.3f}s")
        
        except Exception as e:
            logger.error(f"Error processing {gbf_path}: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_wide_row(
        self,
        metadata: Dict[str, Any],
        latencies: List[float],
        responses: List[int],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create a wide format row with metadata + metrics.
        
        Args:
            metadata: From filename/structure
            latencies: Trial latencies
            responses: Trial responses
            metrics: Calculated metrics
            
        Returns:
            Dict with all columns for wide format
        """
        row = {}
        
        # Add metadata columns (different for synthetic vs real)
        if self.data_type == 'synthetic':
            row['model'] = self.model_name
            row['pse_true'] = metadata.get('pse_true')
            row['jnd_true'] = metadata.get('jnd_true')
            row['subject_id'] = metadata.get('subject_id')
            row['group'] = metadata.get('group_idx')
        else:  # real
            row['subj'] = metadata.get('subj')
            row['age'] = metadata.get('age')
            row['gender'] = metadata.get('gender')
            row['modality'] = metadata.get('modality')
            row['algorithm'] = metadata.get('algorithm')
            row['group'] = metadata.get('group')
        
        # Add trial info
        row['n_trials'] = len(latencies)
        
        # Add all metrics EXCEPT lists (pse_values, jnd_values)
        for key, value in metrics.items():
            # Skip list values like pse_values, jnd_values
            if not isinstance(value, list):
                row[key] = value
        
        return row
    
    def save_excel(self, output_dir: Optional[Path] = None, filename_prefix: str = 'results') -> Tuple[Path, Path]:
        """
        Save wide and long format DataFrames to Excel.
        
        Args:
            output_dir: Directory for output (default: self.output_dir)
            filename_prefix: Prefix for filenames
            
        Returns:
            (wide_path, long_path)
        """
        if self.df_wide is None or self.df_long is None:
            logger.error("No data to save. Call process() first.")
            raise RuntimeError("Call process() before save_excel()")
        
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save wide format
        wide_path = output_dir / f"{filename_prefix}_wide.xlsx"
        save_wide_format_to_excel(self.df_wide, str(wide_path))
        
        if self.verbose:
            print(f"✓ Saved wide format: {wide_path}")
        
        # Save long format
        long_path = output_dir / f"{filename_prefix}_long.xlsx"
        save_long_format_to_excel(self.df_long, str(long_path))
        
        if self.verbose:
            print(f"✓ Saved long format: {long_path}")
        
        return wide_path, long_path

