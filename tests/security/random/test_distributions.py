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
import pytest
import statistics

from sfl.security.random import (
    UniformReal,
    BernoulliNegExp,
    SecureNormalReal,
    NormalDiscrete,
    SecureLaplaceReal,
    SecureBinomial,
    uniform_real,
    normal_real,
    discrete_normal,
    laplace_real,
    bernoulli_neg_exp,
    binomial,
)


class TestUniformReal:
    def test_init_validation(self):
        with pytest.raises(ValueError):
            UniformReal(1.0, 0.0)
    
    def test_range(self):
        uniform = UniformReal(0.0, 1.0)
        for _ in range(100):
            val = uniform()
            assert 0.0 <= val < 1.0
    
    def test_convenience_function(self):
        val = uniform_real(0.0, 1.0)
        assert 0.0 <= val < 1.0


class TestBernoulliNegExp:
    def test_init_validation(self):
        with pytest.raises(ValueError):
            BernoulliNegExp(-1.0)
    
    def test_output_range(self):
        bernoulli = BernoulliNegExp(0.5)
        for _ in range(100):
            val = bernoulli()
            assert val in [0, 1]
    
    def test_convenience_function(self):
        val = bernoulli_neg_exp(0.5)
        assert val in [0, 1]


class TestSecureNormalReal:
    def test_init_validation(self):
        with pytest.raises(ValueError):
            SecureNormalReal(0.0, -1.0)
    
    def test_output_type(self):
        normal = SecureNormalReal(0.0, 1.0)
        val = normal()
        assert isinstance(val, float)
    
    def test_convenience_function(self):
        val = normal_real(0.0, 1.0)
        assert isinstance(val, float)
    
    def test_statistics(self):
        # Test that samples are reasonable
        normal = SecureNormalReal(0.0, 1.0)
        samples = [normal() for _ in range(1000)]
        
        # Check that mean is close to 0
        mean = statistics.mean(samples)
        assert abs(mean) < 0.5
        
        # Check that std is close to 1
        std = statistics.stdev(samples)
        assert 0.5 < std < 2.0


class TestNormalDiscrete:
    def test_init_validation(self):
        with pytest.raises(ValueError):
            NormalDiscrete(0, -1.0)
    
    def test_zero_stdv(self):
        discrete = NormalDiscrete(5, 0.0)
        assert discrete() == 5
    
    def test_output_type(self):
        discrete = NormalDiscrete(0, 1.0)
        val = discrete()
        assert isinstance(val, int)
    
    def test_convenience_function(self):
        val = discrete_normal(0, 1.0)
        assert isinstance(val, int)
    
    def test_statistics(self):
        # Test that samples are reasonable
        discrete = NormalDiscrete(0, 2.0)
        samples = [discrete() for _ in range(100)]
        
        # Check that samples are integers
        for s in samples:
            assert isinstance(s, int)
        
        # Check that mean is close to 0
        mean = statistics.mean(samples)
        assert abs(mean) < 2.0


class TestSecureLaplaceReal:
    def test_init_validation(self):
        with pytest.raises(ValueError):
            SecureLaplaceReal(0.0, -1.0)
    
    def test_output_type(self):
        laplace = SecureLaplaceReal(0.0, 1.0)
        val = laplace()
        assert isinstance(val, float)
    
    def test_convenience_function(self):
        val = laplace_real(0.0, 1.0)
        assert isinstance(val, float)


class TestSecureBinomial:
    def test_init_validation(self):
        with pytest.raises(ValueError):
            SecureBinomial(-1, 0.5)
        with pytest.raises(ValueError):
            SecureBinomial(10, -0.1)
        with pytest.raises(ValueError):
            SecureBinomial(10, 1.5)
    
    def test_boundary_cases(self):
        assert SecureBinomial(0, 0.5)() == 0
        assert SecureBinomial(10, 0.0)() == 0
        assert SecureBinomial(10, 1.0)() == 10
    
    def test_output_range(self):
        binomial = SecureBinomial(10, 0.5)
        for _ in range(100):
            val = binomial()
            assert 0 <= val <= 10
    
    def test_convenience_function(self):
        val = binomial(10, 0.5)
        assert 0 <= val <= 10
    
    def test_statistics(self):
        # Test that mean is close to n*p
        binomial = SecureBinomial(100, 0.3)
        samples = [binomial() for _ in range(1000)]
        
        mean = statistics.mean(samples)
        expected = 100 * 0.3
        assert abs(mean - expected) < 5.0


class TestIntegration:
    def test_all_distributions_work(self):
        """Test that all distributions can be instantiated and called."""
        distributions = [
            UniformReal(0.0, 1.0),
            BernoulliNegExp(0.5),
            SecureNormalReal(0.0, 1.0),
            NormalDiscrete(0, 1.0),
            SecureLaplaceReal(0.0, 1.0),
            SecureBinomial(10, 0.5),
        ]
        
        for dist in distributions:
            val = dist()
            assert val is not None
    
    def test_all_convenience_functions(self):
        """Test that all convenience functions work."""
        functions = [
            lambda: uniform_real(0.0, 1.0),
            lambda: normal_real(0.0, 1.0),
            lambda: discrete_normal(0, 1.0),
            lambda: laplace_real(0.0, 1.0),
            lambda: bernoulli_neg_exp(0.5),
            lambda: binomial(10, 0.5),
        ]
        
        for func in functions:
            val = func()
            assert val is not None
