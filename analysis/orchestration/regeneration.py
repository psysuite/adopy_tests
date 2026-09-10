"""
Consolidated regeneration pipeline for simulation analysis (Phases 5-8).

Eliminates duplication across:
- group_sim_gridrnd_*.py (Phases 5-8 inline)
- regenerate_simulation_data.py (Phase 5 recreation)
- regenerate_simulation_plots.py (Phases 6-8 recreation)

Provides high-level orchestration functions callable from any entry point.
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from analysis.core.generate_analysis_data import (
    generate_all_data,
    regenerate_all_data_from_gbf,
    export_stimulus_metrics_to_csv,
)
from analysis.plot.generate_analysis_plots import generate_all_analysis
from analysis.plot.plot_psychometric_curves import plot_psychometric_for_model
from analysis.plot.plot_stimulus_distribution import plot_stimulus_for_model
from analysis.plot.plotting import create_grid_plots_from_groups

logger = logging.getLogger(__name__)


class RegenerationPipeline:
    """
    Orchestrate regeneration of analysis data and plots from existing simulation results.
    
    Consolidates Phases 5-8 of group_sim_gridrnd_* pipeline into reusable class.
    Callable from:
    - group_sim_gridrnd_* (after simulation completes)
    - regenerate_simulation_data.py (standalone data regeneration)
    - regenerate_simulation_plots.py (standalone plot regeneration)
    
    Usage:
        # After group simulation, regenerate all phases
        pipeline = RegenerationPipeline(
            model_name='ABS1',
            output_dir=Path('data/output/sim_gridrnd/ABS1'),
            pse_grid=[480, 500, 520],
            jnd_grid=[20, 40, 60]
        )
        pipeline.regenerate_all()
        
        # Or regenerate just data
        pipeline.regenerate_analysis_data()
        
        # Or regenerate just plots
        pipeline.regenerate_all_plots()
    """

    def __init__(
        self,
        model_name: str,
        output_dir: Path,
        pse_grid: Optional[List[int]] = None,
        jnd_grid: Optional[List[int]] = None,
        offset: int = 500,
        csv_output_dir: Optional[Path] = None,
        is_synthetic: bool = True,
    ):
        """
        Initialize regeneration pipeline.
        
        Args:
            model_name: 'ABS1', 'REL1', 'REL2', etc.
            output_dir: Base output directory (data/output/sim_gridrnd/{MODEL} for synthetic,
                       data/input/expdata/gbf for real)
            pse_grid: List of PSE values (None for real data)
            jnd_grid: List of JND values (None for real data)
            offset: Reference latency (default 500ms)
            csv_output_dir: Optional directory for CSV export (default: R/indata)
            is_synthetic: True for synthetic data, False for real data
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.pse_grid = pse_grid or []
        self.jnd_grid = jnd_grid or []
        self.offset = offset
        self.is_synthetic = is_synthetic
        
        if csv_output_dir is None:
            csv_output_dir = self.output_dir.parent.parent / 'R' / 'indata'
        self.csv_output_dir = Path(csv_output_dir)

    def phase_4_create_excel_from_gbf(self, verbose: bool = True) -> bool:
        """
        Phase 4: Create initial Excel results file from GBF files (real data only).
        
        For real data, after PSA→GBF conversion, we need to create an initial Excel
        with subject rows before adding metric columns in Phase 5.
        
        For synthetic data, Excel is created during simulation (skipped).
        
        Args:
            verbose: Print progress messages
            
        Returns:
            True if successful, False otherwise
        """
        if self.is_synthetic:
            if verbose:
                logger.info(f"Phase 4: Skipped (synthetic data)")
            return True
        
        if verbose:
            logger.info(f"Phase 4: Creating Excel from GBF files ({self.model_name})...")
        
        try:
            from analysis.io.converter import read_gbf_file
            from analysis.core.psychometric_helpers import consolidate_results, add_group_stats_to_excel
            
            # For real data, GBF files are directly in output_dir
            gbf_files = sorted(self.output_dir.glob("*.txt"))
            
            if not gbf_files:
                if verbose:
                    logger.warning(f"No GBF files found in {self.output_dir}")
                return False
            
            if verbose:
                logger.info(f"  Found {len(gbf_files)} GBF files")
            
            # Reconstruct result dicts from GBF files
            group_results = []
            for gbf_path in gbf_files:
                try:
                    gbf_rows = read_gbf_file(str(gbf_path))
                    
                    # Reconstruct result dict from GBF
                    result_dict = {
                        'subj': gbf_path.stem,
                        'n_trials': len(gbf_rows),
                        'responses': [r['user_ans'] for r in gbf_rows],
                        'latencies': [r['lat'] for r in gbf_rows],
                    }
                    group_results.append(result_dict)
                    
                except Exception as e:
                    logger.debug(f"  Warning: Could not read {gbf_path}: {e}")
                    continue
            
            if not group_results:
                if verbose:
                    logger.warning(f"No valid GBF files processed")
                return False
            
            # Create Excel
            excel_filepath = consolidate_results(
                group_results,
                str(self.output_dir),
                f"{self.model_name}_results"
            )
            
            # Add group statistics
            add_group_stats_to_excel(excel_filepath)
            
            if verbose:
                logger.info(f"✓ Phase 4 complete: Excel created with {len(group_results)} subjects")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Phase 4 failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def phase_5_generate_analysis_data(self, verbose: bool = True) -> bool:
        """
        Phase 5: Regenerate analysis data (Excel columns) from existing GBF files.
        
        Uses MetricsCalculator to compute all metrics (B1-B5) and populate Excel.
        
        Args:
            verbose: Print progress messages
            
        Returns:
            True if successful, False otherwise
        """
        if verbose:
            logger.info(f"Phase 5: Generating analysis data for {self.model_name}...")
        
        try:
            generate_all_data(
                model_name=self.model_name,
                output_dir=str(self.output_dir),
                pse_grid=self.pse_grid,
                jnd_grid=self.jnd_grid,
                offset=self.offset,
                is_synthetic=self.is_synthetic
            )
            if verbose:
                logger.info(f"✓ Phase 5 complete: Analysis data generated")
            return True
        except Exception as e:
            logger.error(f"✗ Phase 5 failed: {e}")
            return False

    def phase_6_7_generate_plots(self, verbose: bool = True) -> bool:
        """
        Phases 6-7: Generate group-level plots (psychometric + stimulus distribution).
        
        For synthetic data: creates 3x3 grid of plots per model.
        For real data: skipped (no grid structure without PSE/JND groups).
        
        Args:
            verbose: Print progress messages
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_synthetic:
            if verbose:
                logger.info(f"Phases 6-7: Skipped (real data)")
            return True
        
        if verbose:
            logger.info(f"Phases 6-7: Generating group plots...")
        
        try:
            plot_psychometric_for_model(self.model_name)
            plot_stimulus_for_model(self.model_name)
            
            if verbose:
                logger.info(f"✓ Phases 6-7 complete: Group plots generated")
            return True
        except Exception as e:
            logger.error(f"✗ Phases 6-7 failed: {e}")
            return False

    def phase_8_analysis_plots_and_csv(self, verbose: bool = True) -> bool:
        """
        Phase 8: Generate analysis metric plots and export CSV.
        
        For synthetic data: creates analysis plots (grids) and CSV.
        For real data: skipped (no grid structure).
        
        Args:
            verbose: Print progress messages
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_synthetic:
            if verbose:
                logger.info(f"Phase 8: Skipped (real data)")
            return True
        
        if verbose:
            logger.info(f"Phase 8: Generating analysis plots and CSV export...")
        
        try:
            generate_all_analysis(
                model_name=self.model_name,
                output_dir=str(self.output_dir),
                pse_grid=self.pse_grid,
                jnd_grid=self.jnd_grid,
                export_csv=True,
                csv_output_dir=str(self.csv_output_dir),
                data_root=str(self.output_dir.parent)
            )
            
            if verbose:
                logger.info(f"✓ Phase 8 complete: Analysis plots and CSV exported")
            return True
        except Exception as e:
            logger.error(f"✗ Phase 8 failed: {e}")
            return False

    def regenerate_analysis_data(self, verbose: bool = True) -> bool:
        """
        Phases 4-5: Create/regenerate analysis data (Excel).
        
        For real data: Phase 4 (create Excel from GBF) + Phase 5 (add metrics)
        For synthetic data: Phase 5 only (add metrics to existing Excel)
        
        Args:
            verbose: Print progress messages
            
        Returns:
            True if successful, False otherwise
        """
        # Phase 4: Create Excel from GBF (real data only)
        if not self.phase_4_create_excel_from_gbf(verbose=verbose):
            if self.is_synthetic:
                return False  # Synthetic must have Excel already
            else:
                logger.warning("Phase 4 warning: Excel creation had issues, continuing...")
        
        # Phase 5: Add metric columns
        return self.phase_5_generate_analysis_data(verbose=verbose)

    def regenerate_all_plots(self, verbose: bool = True) -> bool:
        """
        Phases 6-8 combined (all plot-related regeneration).
        
        Used by regenerate_simulation_plots.py.
        
        Args:
            verbose: Print progress messages
            
        Returns:
            True if all phases successful, False otherwise
        """
        if verbose:
            logger.info(f"Regenerating all plots for {self.model_name}...")
        
        success = True
        success = self.phase_6_7_generate_plots(verbose=verbose) and success
        success = self.phase_8_analysis_plots_and_csv(verbose=verbose) and success
        
        if verbose:
            if success:
                logger.info(f"✓ All plots regenerated for {self.model_name}")
            else:
                logger.error(f"✗ Plot regeneration had errors for {self.model_name}")
        
        return success

    def regenerate_all(self, verbose: bool = True) -> bool:
        """
        All Phases 4-8: Full regeneration of analysis data and plots.
        
        For real data: Phases 4-8 (create Excel + add metrics + plots)
        For synthetic data: Phases 5-8 (add metrics to existing Excel + plots)
        
        Used by:
        - group_sim_gridrnd_*.py after simulation completes
        - group_realdata_analysis.py after PSA→GBF conversion
        
        Args:
            verbose: Print progress messages
            
        Returns:
            True if all phases successful, False otherwise
        """
        if verbose:
            logger.info(f"Full regeneration (Phases 4-8) for {self.model_name}...")
        
        success = True
        
        # Phase 4: Create Excel from GBF (real data only)
        success = self.phase_4_create_excel_from_gbf(verbose=verbose) and success
        
        # Phase 5: Add metric columns
        success = self.phase_5_generate_analysis_data(verbose=verbose) and success
        
        # Phases 6-8: Plot regeneration
        success = self.phase_6_7_generate_plots(verbose=verbose) and success
        success = self.phase_8_analysis_plots_and_csv(verbose=verbose) and success
        
        if verbose:
            if success:
                logger.info(f"✓ Full regeneration complete for {self.model_name}")
            else:
                logger.error(f"✗ Full regeneration had errors for {self.model_name}")
        
        return success


