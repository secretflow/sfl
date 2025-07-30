#!/usr/bin/env python3
"""
Standalone test for pure Python differential privacy random module.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from sfl.security.privacy.random import (
    uniform_real,
    bernoulli_neg_exp,
    secure_normal_real,
    normal_discrete,
    secure_laplace_real,
)


def test_all_functions():
    """Test all functions with basic functionality."""

    print("Testing pure Python differential privacy random module...")

    # Test uniform_real
    print("1. Testing uniform_real...")
    x = uniform_real(0, 1, size=(3, 4))
    print(f"   Shape: {x.shape}")
    print(f"   Min: {x.min():.4f}, Max: {x.max():.4f}")
    assert x.shape == (3, 4)
    assert x.min() >= 0 and x.max() < 1

    # Test bernoulli_neg_exp
    print("2. Testing bernoulli_neg_exp...")
    x = bernoulli_neg_exp(0.5, size=(3, 4))
    print(f"   Shape: {x.shape}")
    print(f"   Unique values: {set(x.flatten())}")
    assert x.shape == (3, 4)
    assert set(x.flatten()).issubset({0, 1})

    # Test secure_normal_real
    print("3. Testing secure_normal_real...")
    x = secure_normal_real(0, 1, size=(3, 4))
    print(f"   Shape: {x.shape}")
    print(f"   Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    assert x.shape == (3, 4)

    # Test normal_discrete
    print("4. Testing normal_discrete...")
    x = normal_discrete(0, 1, size=(3, 4))
    print(f"   Shape: {x.shape}")
    print(f"   Unique values: {set(x.flatten())}")
    print(f"   All integers: {all(v == int(v) for v in x.flatten())}")
    assert x.shape == (3, 4)
    assert all(v == int(v) for v in x.flatten())

    # Test secure_laplace_real
    print("5. Testing secure_laplace_real...")
    x = secure_laplace_real(0, 1, size=(3, 4))
    print(f"   Shape: {x.shape}")
    print(f"   Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    assert x.shape == (3, 4)

    # Test scalar outputs
    print("6. Testing scalar outputs...")
    x = uniform_real(0, 1)
    print(f"   uniform_real scalar: {x}")
    assert x.shape == ()

    x = bernoulli_neg_exp(0.5)
    print(f"   bernoulli_neg_exp scalar: {x}")
    assert x.shape == ()

    x = secure_normal_real(0, 1)
    print(f"   secure_normal_real scalar: {x}")
    assert x.shape == ()

    x = normal_discrete(0, 1)
    print(f"   normal_discrete scalar: {x}")
    assert x.shape == ()

    x = secure_laplace_real(0, 1)
    print(f"   secure_laplace_real scalar: {x}")
    assert x.shape == ()

    # Test parameter validation
    print("7. Testing parameter validation...")
    try:
        uniform_real(1, 0)
        assert False, "Should raise ValueError"
    except ValueError:
        print("   ✓ uniform_real validation works")

    try:
        bernoulli_neg_exp(-1)
        assert False, "Should raise ValueError"
    except ValueError:
        print("   ✓ bernoulli_neg_exp validation works")

    try:
        secure_normal_real(0, -1)
        assert False, "Should raise ValueError"
    except ValueError:
        print("   ✓ secure_normal_real validation works")

    try:
        normal_discrete(0, -1)
        assert False, "Should raise ValueError"
    except ValueError:
        print("   ✓ normal_discrete validation works")

    try:
        secure_laplace_real(0, -1)
        assert False, "Should raise ValueError"
    except ValueError:
        print("   ✓ secure_laplace_real validation works")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_all_functions()
