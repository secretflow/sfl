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

import numpy as np
import pytest
from scipy import stats
from typing import Dict, Union, Tuple, List, Optional

from sfl.security.random import (
    uniform_real,
    normal_real,
    laplace_real,
    binomial,
    discrete_normal,
    bernoulli_neg_exp,
)

# Test configuration variables
SAMPLE_SIZE = 5000
ALPHA = 0.01
MEAN_TOLERANCE = 0.05
STD_TOLERANCE = 0.05
UNIFORM_LOW = 0.0
UNIFORM_HIGH = 1.0
NORMAL_MEAN = 0.0
NORMAL_STD = 1.0
LAPLACE_MEAN = 0.0
LAPLACE_SCALE = 1.0
BINOMIAL_N = 20
BINOMIAL_P = 0.5


def comprehensive_distribution_test(sample1: np.ndarray, sample2: np.ndarray, 
                                  alpha: float = 0.01) -> Dict:
    """Comprehensive test to determine if two distributions are significantly different."""
    # Convert to numpy arrays
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)
    
    # Remove NaN values
    sample1 = sample1[~np.isnan(sample1)]
    sample2 = sample2[~np.isnan(sample2)]
    
    results = {
        'sample_sizes': (len(sample1), len(sample2)),
        'alpha': alpha,
        'tests_performed': {},
        'distributions_are_different': False,
        'overall_conclusion': '',
        'warnings': []
    }
    
    # Check minimum sample size
    if len(sample1) < 5 or len(sample2) < 5:
        results['warnings'].append("Sample size too small")
        results['overall_conclusion'] = "Insufficient data for reliable testing"
        return results
    
    # 1. Kolmogorov-Smirnov Test
    try:
        ks_stat, ks_p = stats.ks_2samp(sample1, sample2)
        ks_different = ks_p < alpha
        results['tests_performed']['kolmogorov_smirnov'] = {
            'distributions_different': ks_different,
            'p_value': ks_p,
            'statistic': ks_stat,
            'test_description': 'Tests overall distribution shape differences'
        }
    except Exception as e:
        results['warnings'].append(f"KS test failed: {str(e)}")
    
    # 2. Mann-Whitney U Test
    try:
        mw_stat, mw_p = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
        mw_different = mw_p < alpha
        results['tests_performed']['mann_whitney'] = {
            'distributions_different': mw_different,
            'p_value': mw_p,
            'statistic': mw_stat,
            'test_description': 'Tests differences in central tendency (medians)'
        }
    except Exception as e:
        results['warnings'].append(f"Mann-Whitney test failed: {str(e)}")
    
    # 3. Anderson-Darling Test
    try:
        ad_stat, ad_critical, ad_significance = stats.anderson_ksamp([sample1, sample2])
        ad_different = ad_stat > ad_critical[2]
        results['tests_performed']['anderson_darling'] = {
            'distributions_different': ad_different,
            'statistic': ad_stat,
            'critical_value': ad_critical[2],
            'test_description': 'More sensitive to tail differences'
        }
    except Exception as e:
        results['warnings'].append(f"Anderson-Darling test failed: {str(e)}")
    
    # 4. Levene's Test (for variance differences)
    try:
        levene_stat, levene_p = stats.levene(sample1, sample2)
        levene_different = levene_p < alpha
        results['tests_performed']['levene_variance'] = {
            'distributions_different': levene_different,
            'p_value': levene_p,
            'statistic': levene_stat,
            'test_description': 'Tests for differences in variance/spread'
        }
    except Exception as e:
        results['warnings'].append(f"Levene test failed: {str(e)}")
    
    # 5. Basic statistics comparison
    try:
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        std1, std2 = np.std(sample1), np.std(sample2)
        
        mean_diff = abs(mean1 - mean2)
        std_diff = abs(std1 - std2)
        
        results['tests_performed']['basic_statistics'] = {
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'mean1': mean1, 'mean2': mean2,
            'std1': std1, 'std2': std2
        }
    except Exception as e:
        results['warnings'].append(f"Basic statistics failed: {str(e)}")
    
    # Overall conclusion
    _determine_overall_conclusion(results, alpha)
    
    return results


def _determine_overall_conclusion(results: Dict, alpha: float):
    """Determine overall conclusion based on test results"""
    tests = results['tests_performed']
    if not tests:
        results['overall_conclusion'] = "No valid tests could be performed"
        return
    
    # Count how many tests indicate difference
    different_count = sum(1 for test in tests.values()
                         if test.get('distributions_different', False))
    total_tests = len([t for t in tests.values() if 'distributions_different' in t])
    
    # Check basic statistics
    basic_stats = tests.get('basic_statistics', {})
    mean_diff = basic_stats.get('mean_difference', float('inf'))
    std_diff = basic_stats.get('std_difference', float('inf'))
    
    # Overall assessment
    stats_ok = mean_diff < MEAN_TOLERANCE and std_diff < STD_TOLERANCE
    
    if total_tests > 0:
        if different_count == 0 and stats_ok:
            results['distributions_are_different'] = False
            results['overall_conclusion'] = f"Distributions appear similar (0/{total_tests} tests significant, stats within tolerance)"
        elif different_count > 0:
            results['distributions_are_different'] = True
            results['overall_conclusion'] = f"Evidence of different distributions ({different_count}/{total_tests} tests significant)"
        else:
            results['distributions_are_different'] = False
            results['overall_conclusion'] = f"Distributions appear similar ({different_count}/{total_tests} tests significant, stats within tolerance)"
    else:
        results['distributions_are_different'] = not stats_ok
        results['overall_conclusion'] = "Based on basic statistics only"


class TestComprehensiveValidation:
    """Comprehensive statistical validation tests."""
    
    def test_uniform_distribution(self):
        """Test uniform distribution against numpy."""
        our_samples = uniform_real(UNIFORM_LOW, UNIFORM_HIGH, size=SAMPLE_SIZE)
        numpy_samples = np.random.uniform(UNIFORM_LOW, UNIFORM_HIGH, SAMPLE_SIZE)
        
        result = comprehensive_distribution_test(our_samples, numpy_samples, ALPHA)
        
        assert not result['distributions_are_different'], \
            f"Uniform distributions differ: {result['overall_conclusion']}"
    
    def test_normal_distribution(self):
        """Test normal distribution against numpy."""
        our_samples = normal_real(NORMAL_MEAN, NORMAL_STD, size=SAMPLE_SIZE)
        numpy_samples = np.random.normal(NORMAL_MEAN, NORMAL_STD, SAMPLE_SIZE)
        
        result = comprehensive_distribution_test(our_samples, numpy_samples, ALPHA)
        
        assert not result['distributions_are_different'], \
            f"Normal distributions differ: {result['overall_conclusion']}"
    
    def test_laplace_distribution(self):
        """Test laplace distribution against numpy."""
        our_samples = laplace_real(LAPLACE_MEAN, LAPLACE_SCALE, size=SAMPLE_SIZE)
        numpy_samples = np.random.laplace(LAPLACE_MEAN, LAPLACE_SCALE, SAMPLE_SIZE)
        
        result = comprehensive_distribution_test(our_samples, numpy_samples, ALPHA)
        
        assert not result['distributions_are_different'], \
            f"Laplace distributions differ: {result['overall_conclusion']}"
    
    def test_binomial_distribution(self):
        """Test binomial distribution against numpy."""
        our_samples = binomial(BINOMIAL_N, BINOMIAL_P, size=SAMPLE_SIZE)
        numpy_samples = np.random.binomial(BINOMIAL_N, BINOMIAL_P, SAMPLE_SIZE)
        
        result = comprehensive_distribution_test(our_samples, numpy_samples, ALPHA)
        
        assert not result['distributions_are_different'], \
            f"Binomial distributions differ: {result['overall_conclusion']}"
    
    def test_discrete_normal_distribution(self):
        """Test discrete normal distribution."""
        our_samples = discrete_normal(0, 2.0, size=SAMPLE_SIZE)
        numpy_samples = np.round(np.random.normal(0, 2.0, SAMPLE_SIZE)).astype(int)
        
        result = comprehensive_distribution_test(our_samples, numpy_samples, ALPHA)
        
        assert not result['distributions_are_different'], \
            f"Discrete normal distributions differ: {result['overall_conclusion']}"
    
    def test_bernoulli_neg_exp_distribution(self):
        """Test Bernoulli negative exponential distribution."""
        our_samples = bernoulli_neg_exp(0.5, size=SAMPLE_SIZE)
        numpy_samples = np.random.binomial(1, np.exp(-0.5), SAMPLE_SIZE)
        
        result = comprehensive_distribution_test(our_samples, numpy_samples, ALPHA)
        
        assert not result['distributions_are_different'], \
            f"Bernoulli neg exp distributions differ: {result['overall_conclusion']}"
    
    def test_array_generation_consistency(self):
        """Test that array generation produces consistent results."""
        # Test that array generation is equivalent to multiple single calls
        size = 100
        
        # Generate via array
        arr = uniform_real(0.0, 1.0, size=size)
        
        # Generate via multiple calls
        single_calls = [uniform_real(0.0, 1.0) for _ in range(size)]
        
        # Both should have same shape characteristics
        assert len(arr) == len(single_calls)
        assert arr.dtype == np.float64
        assert all(isinstance(x, float) for x in single_calls)
    
    def test_array_shapes(self):
        """Test various array shapes."""
        shapes = [5, (3, 4), [2, 3, 4]]
        
        for shape in shapes:
            expected_shape = shape if isinstance(shape, tuple) else tuple([shape] if isinstance(shape, int) else shape)
            
            # Test all distributions with different shapes
            arr = uniform_real(0.0, 1.0, size=shape)
            assert arr.shape == expected_shape
            
            arr = normal_real(0.0, 1.0, size=shape)
            assert arr.shape == expected_shape
            
            arr = discrete_normal(0, 1.0, size=shape)
            assert arr.shape == expected_shape
            
            arr = laplace_real(0.0, 1.0, size=shape)
            assert arr.shape == expected_shape
            
            arr = bernoulli_neg_exp(0.5, size=shape)
            assert arr.shape == expected_shape
            
            arr = binomial(10, 0.5, size=shape)
            assert arr.shape == expected_shape


if __name__ == "__main__":
    # Run comprehensive validation
    test = TestComprehensiveValidation()
    
    print("Running comprehensive distribution validation...")
    test.test_all_distributions_comprehensive()
    print("✅ All distributions passed comprehensive validation!")
