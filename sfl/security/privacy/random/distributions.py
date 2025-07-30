# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import secrets
from typing import Optional, Union, Sequence
import math


class _SecureRandomGenerator:
    """Secure random number generator using secrets module."""

    def __init__(self):
        self._state = secrets.randbits(64)

    def rand_uint64(self) -> int:
        """Generate a cryptographically secure 64-bit unsigned integer."""
        self._state = (self._state * 6364136223846793005 + 1) & ((1 << 64) - 1)
        self._state ^= secrets.randbits(64)
        return self._state

    def rand_float(self) -> float:
        """Generate a cryptographically secure float in [0, 1)."""
        return self.rand_uint64() / (1 << 64)


# Global secure random generator
_rng = _SecureRandomGenerator()


def _validate_shape(
    shape: Optional[Union[int, Sequence[int]]],
) -> Optional[Sequence[int]]:
    """Validate and normalize shape parameter."""
    if shape is None:
        return None

    if isinstance(shape, int):
        return (shape,)

    if isinstance(shape, (list, tuple)):
        return tuple(int(s) for s in shape)

    raise ValueError("shape must be int, tuple/list of ints, or None")


def _generate_array(shape: Optional[Sequence[int]], generator_func) -> np.ndarray:
    """Generate array of random values with given shape."""
    if shape is None:
        return np.array(generator_func())

    total_elements = 1
    for s in shape:
        total_elements *= s

    values = [generator_func() for _ in range(total_elements)]
    return np.array(values).reshape(shape)


class UniformReal:
    """Uniform real distribution in [low, high)."""

    def __init__(self, low: float, high: float):
        if low > high:
            raise ValueError("low must be <= high")
        self.low = low
        self.high = high

    def __call__(self) -> float:
        """Generate a uniform random float in [low, high)."""
        return self.low + (self.high - self.low) * _rng.rand_float()


class BernoulliNegExp:
    """Bernoulli distribution with parameter exp(-gamma)."""

    def __init__(self, gamma: float):
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = gamma

    def __call__(self) -> int:
        """Sample from Bernoulli(exp(-gamma))."""
        gamma = self.gamma

        # Handle gamma > 1 using geometric distribution trick
        while gamma > 1:
            gamma -= 1
            if BernoulliNegExp(1)() == 0:
                return 0

        # Use exponential distribution via inverse transform sampling
        counter = 1
        while _rng.rand_float() <= gamma / counter:
            counter += 1

        return counter % 2


class SecureNormalReal:
    """Secure normal distribution using Box-Muller transform."""

    def __init__(self, mean: float, stdv: float):
        if stdv < 0:
            raise ValueError("stdv must be non-negative")
        self.mean = mean
        self.stdv = stdv
        self._cached_value = None
        self._has_cached = False

    def __call__(self) -> float:
        """Generate a secure normal random variable."""
        if self._has_cached:
            self._has_cached = False
            return self._cached_value

        # Box-Muller transform
        u1 = _rng.rand_float()
        u2 = _rng.rand_float()

        # Ensure u1 and u2 are not zero
        u1 = max(u1, 1e-15)
        u2 = max(u2, 1e-15)

        r = math.sqrt(-2.0 * math.log(1.0 - u2))
        theta = 2.0 * math.pi * u1

        z1 = r * math.sin(theta)
        z2 = r * math.cos(theta)

        # Cache one value for efficiency
        self._cached_value = z2 * self.stdv + self.mean
        self._has_cached = True

        return z1 * self.stdv + self.mean


class NormalDiscrete:
    """Discrete normal distribution for differential privacy."""

    def __init__(self, mean: float, stdv: float):
        if stdv < 0:
            raise ValueError("stdv must be non-negative")
        self.mean = mean
        self.stdv = stdv

    def __call__(self) -> int:
        """Sample from discrete normal distribution."""
        if self.stdv == 0:
            return int(round(self.mean))

        # Use discrete Gaussian sampling via rejection sampling
        # Based on the algorithm from [CK20]
        tau = 1.0 / (1.0 + math.floor(self.stdv))
        sigma2 = self.stdv**2

        while True:
            # Sample from discrete Laplace
            geom_x = 0
            while BernoulliNegExp(tau)():
                geom_x += 1

            # Random sign
            bern_b = _rng.rand_float() < 0.5
            lap_y = (1 - 2 * bern_b) * geom_x

            # Rejection sampling
            prob_ratio = math.exp(-((abs(lap_y) - tau * sigma2) ** 2) / (2 * sigma2))
            if _rng.rand_float() <= prob_ratio:
                return int(round(self.mean + lap_y))


class SecureLaplaceReal:
    """Secure Laplace distribution using four uniform variates."""

    def __init__(self, mean: float, stdv: float):
        if stdv < 0:
            raise ValueError("stdv must be non-negative")
        self.mean = mean
        self.stdv = stdv

    def __call__(self) -> float:
        """Generate a secure Laplace random variable."""
        # Use four uniform variates for security
        u1 = _rng.rand_float()
        u2 = _rng.rand_float()
        u3 = _rng.rand_float()
        u4 = _rng.rand_float()

        # Ensure values are not zero
        u1 = max(u1, 1e-15)
        u2 = max(u2, 1e-15)
        u3 = max(u3, 1e-15)
        u4 = max(u4, 1e-15)

        # Generate Laplace via exponential distribution
        e1 = -math.log(1.0 - u1)
        e2 = -math.log(1.0 - u2)
        e3 = -math.log(1.0 - u3)
        e4 = -math.log(1.0 - u4)

        # Combine to get Laplace distribution
        laplace = e1 * math.cos(math.pi * u2) + e3 * math.cos(math.pi * u4)

        return laplace * self.stdv + self.mean


def uniform_real(
    low: float = 0.0,
    high: float = 1.0,
    size: Optional[Union[int, Sequence[int]]] = None,
) -> np.ndarray:
    """
    Generate uniform random real numbers in [low, high).

    Args:
        low: Lower bound of the uniform distribution
        high: Upper bound of the uniform distribution
        size: Output shape. If None, returns a scalar

    Returns:
        Array of uniform random real numbers
    """
    shape = _validate_shape(size)
    dist = UniformReal(low, high)
    return _generate_array(shape, dist)


def bernoulli_neg_exp(
    p: float = 0.5, size: Optional[Union[int, Sequence[int]]] = None
) -> np.ndarray:
    """
    Generate Bernoulli random variables with parameter exp(-p).

    Args:
        p: Parameter for Bernoulli(exp(-p)) distribution
        size: Output shape. If None, returns a scalar

    Returns:
        Array of Bernoulli random variables (0 or 1)
    """
    shape = _validate_shape(size)
    dist = BernoulliNegExp(p)
    return _generate_array(shape, dist)


def secure_normal_real(
    mean: float = 0.0,
    stdv: float = 1.0,
    size: Optional[Union[int, Sequence[int]]] = None,
) -> np.ndarray:
    """
    Generate secure normal random real numbers.

    Uses Box-Muller transform with secure random generation to prevent
    floating point attacks.

    Args:
        mean: Mean of the normal distribution
        stdv: Standard deviation of the normal distribution
        size: Output shape. If None, returns a scalar

    Returns:
        Array of secure normal random real numbers
    """
    shape = _validate_shape(size)
    dist = SecureNormalReal(mean, stdv)
    return _generate_array(shape, dist)


def normal_discrete(
    mean: float = 0.0,
    stdv: float = 1.0,
    size: Optional[Union[int, Sequence[int]]] = None,
) -> np.ndarray:
    """
    Generate discrete normal random variables.

    Samples from the discrete Gaussian distribution for differential privacy.

    Args:
        mean: Mean of the discrete normal distribution
        stdv: Standard deviation of the discrete normal distribution
        size: Output shape. If None, returns a scalar

    Returns:
        Array of discrete normal random integers
    """
    shape = _validate_shape(size)
    dist = NormalDiscrete(mean, stdv)
    return _generate_array(shape, dist)


def secure_laplace_real(
    mean: float = 0.0,
    stdv: float = 1.0,
    size: Optional[Union[int, Sequence[int]]] = None,
) -> np.ndarray:
    """
    Generate secure Laplace random real numbers.

    Uses four uniform variates to prevent floating point attacks.

    Args:
        mean: Mean of the Laplace distribution
        stdv: Scale parameter of the Laplace distribution
        size: Output shape. If None, returns a scalar

    Returns:
        Array of secure Laplace random real numbers
    """
    shape = _validate_shape(size)
    dist = SecureLaplaceReal(mean, stdv)
    return _generate_array(shape, dist)
