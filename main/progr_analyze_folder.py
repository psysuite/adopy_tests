#!/usr/bin/env python3
"""
read all the .txt in input_folder, run progressive analysis and for each file, prints:
 the JND of each block and the stability point
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.progressive_analyzer import ProgressiveAnalyzer

# ============================================================================
INPUT_FOLDER = "/data/CODE/python/adopy_tests/data/input/expdata"
INPUT_FOLDER = "/data/CODE/python/adopy_tests/data/output/sim_grid/2model_rel"
STABILITY_THRESHOLD = 0.10
# ============================================================================

BLOCKS = [40, 60, 80, 100, 120, 140, 160, 180, 200]


def main():
    logging.basicConfig(level=logging.WARNING)

    analyzer = ProgressiveAnalyzer(blocks=BLOCKS)

    input_dir = Path(INPUT_FOLDER)
    txt_files = sorted(input_dir.glob("*.txt"))

    if not txt_files:
        print(f"No .txt file was found in {input_dir}")
        return

    print(f"Found {len(txt_files)} files in {input_dir}")
    print()

    block_cols = "".join([f"{b:>8}" for b in BLOCKS])
    print(f"{'File':<50} {block_cols} {'JND_SP':>8}")
    print("-" * (50 + 8 * len(BLOCKS) + 8))

    for filepath in txt_files:
        result = analyzer.run_progressive_analysis(str(filepath))

        # Get valid blocks
        valid_blocks = [N for N in result.trial_counts if result.jnd_values[N] > 0]
        
        if not valid_blocks:
            print(f"{filepath.name:<50} {'ERROR':>8}")
            continue

        jnd_cols = "".join([f"{result.jnd_values[N]:>8.1f}" for N in BLOCKS])
        
        # Calculate stability point
        jnd_list = [result.jnd_values[N] for N in valid_blocks]
        from utilities.psychometric_helpers import calculate_stability_from_values
        jnd_sp = calculate_stability_from_values(jnd_list, STABILITY_THRESHOLD, valid_blocks)

        print(f"{filepath.name:<50} {jnd_cols} {jnd_sp:>8}")


if __name__ == "__main__":
    main()
