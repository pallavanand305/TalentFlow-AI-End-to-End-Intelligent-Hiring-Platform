#!/usr/bin/env python3
"""
Script to run baseline model training

This is a convenience script that runs the baseline model training
with proper environment setup.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.training.train_baseline_model import main

if __name__ == "__main__":
    asyncio.run(main())