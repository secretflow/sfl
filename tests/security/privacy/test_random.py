"""
Tests for pure Python implementation of secure random distributions.
"""

import unittest
import numpy as np
from sfl.security.privacy.random import (
    uniform_real,
    bernoulli_neg_exp,
    secure_normal_real,
    normal_discrete,
    secure_laplace_real,
)


class TestSecureRandom(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducible tests

    def test_uniform_real_basic(self):
        """Test basic functionality of uniform_real."""
        x = uniform_real(0, 1, size=(3, 4))
        self.assertEqual(x.shape, (3, 4))
        self.assertTrue(np.all(x >= 0))
        self.assertTrue(np.all(x < 1))
        self.assertEqual(x.dtype, np.float64)

    def test_uniform_real_scalar(self):
        """Test uniform_real with scalar output."""
        x = uniform_real(0, 1)
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.shape, ())
        self.assertTrue(0 <= x.item() < 1)

    def test_uniform_real_range(self):
        """Test uniform_real with custom range."""
        low, high = -5, 10
        x = uniform_real(low, high, size=1000)
        self.assertTrue(np.all(x >= low))
        self.assertTrue(np.all(x < high))

    def test_bernoulli_neg_exp_basic(self):
        """Test basic functionality of bernoulli_neg_exp."""
        x = bernoulli_neg_exp(0.5, size=(3, 4))
        self.assertEqual(x.shape, (3, 4))
        self.assertTrue(np.all(np.logical_or(x == 0, x == 1)))
        self.assertEqual(x.dtype, np.int32)

    def test_bernoulli_neg_exp_scalar(self):
        """Test bernoulli_neg_exp with scalar output."""
        x = bernoulli_neg_exp(0.5)
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.shape, ())
        self.assertIn(x.item(), [0, 1])

    def test_bernoulli_neg_exp_parameter(self):
        """Test bernoulli_neg_exp with different parameters."""
        # With p=0, should always return 1 (exp(0) = 1)
        x = bernoulli_neg_exp(0.0, size=100)
        self.assertTrue(np.all(x == 1))

        # With large p, should mostly return 0
        x = bernoulli_neg_exp(10.0, size=100)
        self.assertTrue(np.sum(x) < 50)  # Most should be 0

    def test_secure_normal_real_basic(self):
        """Test basic functionality of secure_normal_real."""
        x = secure_normal_real(0, 1, size=(3, 4))
        self.assertEqual(x.shape, (3, 4))
        self.assertEqual(x.dtype, np.float64)

    def test_secure_normal_real_statistics(self):
        """Test statistical properties of secure_normal_real."""
        mean, stdv = 5.0, 2.0
        x = secure_normal_real(mean, stdv, size=10000)

        # Check that values are reasonable (within 5 std dev)
        self.assertTrue(np.all(np.abs(x - mean) < 5 * stdv))

        # Check mean is approximately correct
        sample_mean = np.mean(x)
        self.assertAlmostEqual(sample_mean, mean, delta=0.5)

        # Check std is approximately correct
        sample_std = np.std(x)
        self.assertAlmostEqual(sample_std, stdv, delta=0.5)

    def test_normal_discrete_basic(self):
        """Test basic functionality of normal_discrete."""
        x = normal_discrete(0, 1, size=(3, 4))
        self.assertEqual(x.shape, (3, 4))
        self.assertTrue(np.all(x == x.astype(int)))
        self.assertEqual(x.dtype, np.int32)

    def test_normal_discrete_scalar(self):
        """Test normal_discrete with scalar output."""
        x = normal_discrete(0, 1)
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.shape, ())
        self.assertEqual(x.dtype, np.int32)

    def test_normal_discrete_zero_std(self):
        """Test normal_discrete with zero standard deviation."""
        x = normal_discrete(5, 0, size=100)
        self.assertTrue(np.all(x == 5))

    def test_secure_laplace_real_basic(self):
        """Test basic functionality of secure_laplace_real."""
        x = secure_laplace_real(0, 1, size=(3, 4))
        self.assertEqual(x.shape, (3, 4))
        self.assertEqual(x.dtype, np.float64)

    def test_secure_laplace_real_statistics(self):
        """Test statistical properties of secure_laplace_real."""
        mean, stdv = 0.0, 1.0
        x = secure_laplace_real(mean, stdv, size=10000)

        # Check that values are reasonable
        self.assertTrue(np.all(np.abs(x) < 10))  # Most values within 10 units

        # Check mean is approximately correct
        sample_mean = np.mean(x)
        self.assertAlmostEqual(sample_mean, mean, delta=0.5)

        # For Laplace, variance = 2 * b^2, so std = sqrt(2) * b
        # Here b = stdv, so theoretical std = sqrt(2) * stdv
        expected_std = math.sqrt(2) * stdv
        sample_std = np.std(x)
        self.assertAlmostEqual(sample_std, expected_std, delta=0.5)

    def test_parameter_validation(self):
        """Test parameter validation."""
        with self.assertRaises(ValueError):
            uniform_real(1, 0)  # low > high

        with self.assertRaises(ValueError):
            bernoulli_neg_exp(-1)  # negative gamma

        with self.assertRaises(ValueError):
            secure_normal_real(0, -1)  # negative stdv

        with self.assertRaises(ValueError):
            normal_discrete(0, -1)  # negative stdv

        with self.assertRaises(ValueError):
            secure_laplace_real(0, -1)  # negative stdv

    def test_shape_validation(self):
        """Test shape parameter validation."""
        # These should work
        uniform_real(0, 1, size=5)
        uniform_real(0, 1, size=(2, 3))
        uniform_real(0, 1, size=[2, 3])

        # This should raise error
        with self.assertRaises(ValueError):
            uniform_real(0, 1, size="invalid")

    def test_reproducibility_separate_calls(self):
        """Test that separate calls produce different results."""
        x1 = uniform_real(0, 1, size=100)
        x2 = uniform_real(0, 1, size=100)

        # Should not be identical
        self.assertFalse(np.array_equal(x1, x2))

    def test_edge_cases(self):
        """Test edge cases."""
        # Test zero range
        x = uniform_real(5, 5, size=10)
        self.assertTrue(np.all(x == 5))

        # Test zero std normal
        x = secure_normal_real(3, 0, size=10)
        self.assertTrue(np.all(x == 3))

        # Test zero std laplace
        x = secure_laplace_real(7, 0, size=10)
        self.assertTrue(np.all(x == 7))


if __name__ == "__main__":
    unittest.main()
