# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import secrets
import numpy as np
from typing import Union, Optional, List, Tuple
import sys


class UniformReal:
    """
    Samples a uniform distribution in the range [from, to) of type T.
    
    Args:
        from_: Lower bound of the uniform distribution
        to: Upper bound of the uniform distribution
    
    Returns:
        A sample of real numbers from a uniform distribution
    """
    
    def __init__(self, from_: float, to: float):
        if from_ > to:
            raise ValueError("from must be less than or equal to to")
        if to - from_ > sys.float_info.max:
            raise ValueError("Range too large")
        self.from_ = from_
        self.to_ = to
    
    def __call__(self) -> float:
        """Generate a random sample from uniform distribution."""
        # Use secrets for cryptographically secure random numbers
        # Use 53 bits for maximum precision in IEEE 754 double precision
        random_bits = secrets.randbits(53)
        divisor = 2**53
        x = random_bits / divisor
        return x * (self.to_ - self.from_) + self.from_


class BernoulliNegExp:
    """
    Sample from Bernoulli(exp(-gamma)).
    
    Args:
        gamma: Parameter to sample from Bernoulli(exp(-gamma)), must be non-negative
    
    Returns:
        A sample from the Bernoulli(exp(-gamma)) distribution
    
    Reference:
        [CK20] Canonne, Kamath, Steinke, "The Discrete Gaussian for Differential Privacy"
    """
    
    def __init__(self, gamma: float):
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = gamma
    
    def __call__(self) -> int:
        """Generate a sample from Bernoulli(exp(-gamma)) distribution."""
        gamma = self.gamma
        
        while gamma > 1:
            gamma -= 1
            bernoulli = BernoulliNegExp(1.0)
            if bernoulli() == 0:
                return 0
        
        uniform = UniformReal(0.0, 1.0)
        counter = 1
        while uniform() <= gamma / counter:
            counter += 1
        
        return counter % 2


class SecureNormalReal:
    """
    Samples a normal distribution using the Box-Muller method.
    
    Samples from the Gaussian distribution are generated using two samples from
    normal distribution using the Box-Muller method as detailed in [HB21b],
    to prevent against reconstruction attacks due to limited floating point
    precision.
    
    Args:
        mean: Mean of the normal distribution
        stdv: Standard deviation of the normal distribution
    
    Returns:
        A sample of real numbers from the normal distribution
    
    Reference:
        [HB21] Holohan, Naoise, and Stefano Braghin. "Secure Random Sampling in
        Differential Privacy." arXiv preprint arXiv:2107.10138 (2021).
    """
    
    def __init__(self, mean: float, stdv: float):
        if stdv < 0:
            raise ValueError("stdv must be non-negative")
        self.mean = mean
        self.stdv = stdv
    
    def transform(self, val: float, mean: float, std: float) -> float:
        """Transform standard normal to specified mean and std."""
        return val * std + mean
    
    def pi(self) -> float:
        """Return the value of pi."""
        return math.pi
    
    def __call__(self) -> float:
        """Generate a sample from normal distribution using Box-Muller method."""
        uniform = UniformReal(0.0, 1.0)
        
        u1 = uniform()
        u2 = uniform()
        
        r = math.sqrt(-2.0 * math.log(1.0 - u2))
        theta = 2.0 * self.pi() * u1
        
        n1 = self.transform(r * math.sin(theta), self.mean, self.stdv)
        n2 = self.transform(r * math.cos(theta), self.mean, self.stdv)
        
        return (n1 + n2) / math.sqrt(2.0)


class NormalDiscrete:
    """
    The Discrete Gaussian mechanism in differential privacy.
    Re-purposed for approximate (epsilon,delta)-differential privacy.
    
    Args:
        mean: Mean of the discrete normal distribution
        stdv: Standard deviation of the discrete normal distribution
    
    Returns:
        A sample of integers from the discrete normal distribution
    
    Reference:
        [CK20] Canonne, Kamath, Steinke, "The Discrete Gaussian for Differential Privacy"
    """
    
    def __init__(self, mean: int, stdv: float):
        if stdv < 0:
            raise ValueError("stdv must be non-negative")
        self.mean = mean
        self.stdv = stdv
    
    def __call__(self) -> int:
        """Generate a sample from discrete normal distribution."""
        if self.stdv == 0:
            return self.mean
        
        tau = 1 / (1 + math.floor(self.stdv))
        sigma2 = self.stdv ** 2
        
        bernoulli1 = BernoulliNegExp(tau)
        
        while True:
            geom_x = 0
            while bernoulli1():
                geom_x += 1
            
            # Use secrets for fair coin flip (binomial distribution with p=0.5)
            bern_b = secrets.randbits(1)
            if bern_b and (not geom_x):
                continue
            
            lap_y = (1 - 2 * bern_b) * geom_x
            
            exponent = pow((abs(lap_y) - tau * sigma2), 2) / (2 * sigma2)
            bernoulli2 = BernoulliNegExp(exponent)
            
            if bernoulli2():
                return int(self.mean + lap_y)


class SecureLaplaceReal:
    """
    The classical Laplace mechanism in differential privacy.
    
    Samples are generated using 4 uniform variates to prevent against 
    reconstruction attacks due to limited floating point precision.
    
    Args:
        mean: Mean of the Laplace distribution
        stdv: Scale parameter of the Laplace distribution
    
    Returns:
        A sample of real numbers from the Laplace distribution
    
    Reference:
        [HB21] Holohan, Naoise, and Stefano Braghin. "Secure Random Sampling in
        Differential Privacy." arXiv preprint arXiv:2107.10138 (2021).
    """
    
    def __init__(self, mean: float, stdv: float):
        if stdv < 0:
            raise ValueError("stdv must be non-negative")
        self.mean = mean
        self.stdv = stdv
    
    def transform(self, val: float, mean: float, std: float) -> float:
        """Transform standard Laplace to specified mean and std."""
        return val * std + mean
    
    def pi(self) -> float:
        """Return the value of pi."""
        return math.pi
    
    def __call__(self) -> float:
        """Generate a sample from Laplace distribution."""
        uniform = UniformReal(0.0, 1.0)
        
        u1 = uniform()
        u2 = uniform()
        u3 = uniform()
        u4 = uniform()
        
        la = math.log(1.0 - u1) * math.cos(self.pi() * u2) + \
             math.log(1.0 - u3) * math.cos(self.pi() * u4)
        
        return self.transform(la, self.mean, self.stdv)


class SecureBinomial:
    """
    Secure binomial distribution using cryptographically secure random numbers.
    
    Args:
        n: Number of trials
        p: Probability of success
    
    Returns:
        A sample from the binomial distribution B(n, p)
    """
    
    def __init__(self, n: int, p: float):
        if n < 0:
            raise ValueError("n must be non-negative")
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")
        self.n = n
        self.p = p
    
    def __call__(self) -> int:
        """Generate a sample from binomial distribution."""
        if self.n == 0 or self.p == 0:
            return 0
        if self.p == 1:
            return self.n
        
        # Use cryptographically secure random bits
        successes = 0
        for _ in range(self.n):
            if secrets.randbits(53) / (2**53) < self.p:
                successes += 1
        return successes


def _generate_array(dist, size: Union[int, Tuple[int, ...], List[int]]) -> np.ndarray:
    """Generate numpy array with given size."""
    if isinstance(size, int):
        size = (size,)
    elif isinstance(size, list):
        size = tuple(size)
    
    total_elements = int(np.prod(size))
    samples = [dist() for _ in range(total_elements)]
    return np.array(samples).reshape(size)


# Enhanced convenience functions with size parameter
def uniform_real(from_: float, to: float, size: Optional[Union[int, Tuple[int, ...], List[int]]] = None) -> Union[float, np.ndarray]:
    """Generate random floats from uniform distribution [from, to).
    
    Args:
        from_: Lower bound of the uniform distribution
        to: Upper bound of the uniform distribution
        size: Output shape. If None, returns a single float.
    
    Returns:
        float or numpy.ndarray: Random samples from uniform distribution
    """
    dist = UniformReal(from_, to)
    if size is None:
        return dist()
    return _generate_array(dist, size)


def normal_real(mean: float, stdv: float, size: Optional[Union[int, Tuple[int, ...], List[int]]] = None) -> Union[float, np.ndarray]:
    """Generate random floats from normal distribution.
    
    Args:
        mean: Mean of the normal distribution
        stdv: Standard deviation of the normal distribution
        size: Output shape. If None, returns a single float.
    
    Returns:
        float or numpy.ndarray: Random samples from normal distribution
    """
    dist = SecureNormalReal(mean, stdv)
    if size is None:
        return dist()
    return _generate_array(dist, size)


def discrete_normal(mean: int, stdv: float, size: Optional[Union[int, Tuple[int, ...], List[int]]] = None) -> Union[int, np.ndarray]:
    """Generate random integers from discrete normal distribution.
    
    Args:
        mean: Mean of the discrete normal distribution
        stdv: Standard deviation of the discrete normal distribution
        size: Output shape. If None, returns a single int.
    
    Returns:
        int or numpy.ndarray: Random samples from discrete normal distribution
    """
    dist = NormalDiscrete(mean, stdv)
    if size is None:
        return dist()
    return _generate_array(dist, size)


def laplace_real(mean: float, stdv: float, size: Optional[Union[int, Tuple[int, ...], List[int]]] = None) -> Union[float, np.ndarray]:
    """Generate random floats from Laplace distribution.
    
    Args:
        mean: Mean of the Laplace distribution
        stdv: Scale parameter of the Laplace distribution
        size: Output shape. If None, returns a single float.
    
    Returns:
        float or numpy.ndarray: Random samples from Laplace distribution
    """
    dist = SecureLaplaceReal(mean, stdv)
    if size is None:
        return dist()
    return _generate_array(dist, size)


def bernoulli_neg_exp(gamma: float, size: Optional[Union[int, Tuple[int, ...], List[int]]] = None) -> Union[int, np.ndarray]:
    """Generate samples from Bernoulli(exp(-gamma)) distribution.
    
    Args:
        gamma: Parameter to sample from Bernoulli(exp(-gamma))
        size: Output shape. If None, returns a single int.
    
    Returns:
        int or numpy.ndarray: Random samples from Bernoulli distribution
    """
    dist = BernoulliNegExp(gamma)
    if size is None:
        return dist()
    return _generate_array(dist, size)


def binomial(n: int, p: float, size: Optional[Union[int, Tuple[int, ...], List[int]]] = None) -> Union[int, np.ndarray]:
    """Generate samples from secure binomial distribution.
    
    Args:
        n: Number of trials
        p: Probability of success
        size: Output shape. If None, returns a single int.
    
    Returns:
        int or numpy.ndarray: Random samples from binomial distribution
    """
    dist = SecureBinomial(n, p)
    if size is None:
        return dist()
    return _generate_array(dist, size)
