#!/usr/bin/env python3
"""
Simple test for pure Python differential privacy random module.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Direct import to avoid sfl module dependencies
try:
    from sfl.security.privacy.random import (
        uniform_real,
        bernoulli_neg_exp,
        secure_normal_real,
        normal_discrete,
        secure_laplace_real,
    )

    print("✅ Successfully imported pure Python DP random module")

    # Test basic functionality
    print("Testing functions...")

    # Test uniform_real
    x = uniform_real(0, 1, size=(2, 3))
    print(f"uniform_real shape: {x.shape}, dtype: {x.dtype}")

    # Test bernoulli_neg_exp
    x = bernoulli_neg_exp(0.5, size=(2, 3))
    print(f"bernoulli_neg_exp shape: {x.shape}, dtype: {x.dtype}")

    # Test secure_normal_real
    x = secure_normal_real(0, 1, size=(2, 3))
    print(f"secure_normal_real shape: {x.shape}, dtype: {x.dtype}")

    # Test normal_discrete
    x = normal_discrete(0, 1, size=(2, 3))
    print(f"normal_discrete shape: {x.shape}, dtype: {x.dtype}")

    # Test secure_laplace_real
    x = secure_laplace_real(0, 1, size=(2, 3))
    print(f"secure_laplace_real shape: {x.shape}, dtype: {x.dtype}")

    print("✅ All basic tests passed!")

except ImportError as e:
    print(f"❌ Import error: {e}")

    # Let's try to debug the import
    print("Debugging import...")

    # Try importing just the distributions file
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "distributions", "sfl/differential_privacy/random/distributions.py"
        )
        distributions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(distributions)

        print("✅ Successfully loaded distributions module directly")

        # Test functions
        x = distributions.uniform_real(0, 1, size=(2, 3))
        print(f"uniform_real works: shape {x.shape}")

    except Exception as e2:
        print(f"❌ Direct import failed: {e2}")
