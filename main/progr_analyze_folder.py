#!/usr/bin/env python3
"""
read all the .txt in input_folder, run progressive analysis and for each file, prints:
 the JND of each block and the stability point
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.validation_analyzer import ValidationAnalyzer
from utilities.psychometric_helpers import calculate_stability_from_values

# ============================================================================
INPUT_FOLDER = "/data/CODE/python/adopy_tests/data/input"
STABILITY_THRESHOLD = 0.10
# ============================================================================

BLOCKS = [40, 60, 80, 100, 120, 140, 160, 180, 200]


def main():
    logging.basicConfig(level=logging.WARNING)

    analyzer = ValidationAnalyzer(
        data_dir=INPUT_FOLDER,
        blocks=BLOCKS,
    )

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
        params, blocks_used = analyzer.run_progressive_analysis(filepath)

        if not params['JND']:
            print(f"{filepath.name:<50} {'ERROR':>8}")
            continue

        jnd_sp = calculate_stability_from_values(params['JND'], STABILITY_THRESHOLD, blocks_used)
        jnd_cols = "".join([f"{v:>8.1f}" for v in params['JND']])

        print(f"{filepath.name:<50} {jnd_cols} {jnd_sp:>8}")


if __name__ == "__main__":
    main()
