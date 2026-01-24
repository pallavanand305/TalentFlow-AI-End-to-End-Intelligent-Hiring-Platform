#!/usr/bin/env python3
"""
Convenience script for training semantic similarity model

This script provides an easy way to train the semantic model with common configurations.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.training.train_semantic_model import main

if __name__ == "__main__":
    asyncio.run(main())