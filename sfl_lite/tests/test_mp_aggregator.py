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

import jax.numpy as jnp
import mplang.v1 as mp
import pytest

from sfl_lite.security.aggregation.mp_aggregator import MPAggregator

from .test_utils import fetch_from_label_party


class TestMPAggregator:
    @pytest.fixture(scope="function")
    def aggregator(self):
        """Fixture providing an MPAggregator instance."""
        return MPAggregator()

    def test_sum(self, simulator, aggregator):
        """Test sum aggregation with scalar values."""

        @mp.function
        def create_test_data():
            x = mp.device("P0")(lambda: 0)()
            y = mp.device("P1")(lambda: 1)()
            return [x, y]

        @mp.function
        def test_sum_func(agg):
            data = create_test_data()
            result = agg.sum(data)
            result = mp.put("P0", result)
            return result

        # Perform sum and get results
        result = mp.evaluate(simulator, test_sum_func, aggregator)
        fetched_result = fetch_from_label_party(simulator, result)

        assert jnp.array_equal(fetched_result, jnp.array(1))

    def test_average(self, simulator, aggregator):
        """Test average aggregation with scalar values."""

        @mp.function
        def create_test_data():
            a = mp.device("P0")(lambda: 2)()
            b = mp.device("P1")(lambda: 2)()
            return [a, b]

        @mp.function
        def test_average_func(agg):
            data = create_test_data()
            result = agg.average(data)
            result = mp.put("P0", result)
            return result

        # Perform average and get results
        result = mp.evaluate(simulator, test_average_func, aggregator)
        fetched_result = fetch_from_label_party(simulator, result)

        assert fetched_result is not None
        assert jnp.array_equal(fetched_result, jnp.array(2.0))
        assert fetched_result.dtype == jnp.float64

    def test_sum_with_array_input(self, simulator, aggregator):
        """Test sum aggregation with array values."""

        @mp.function
        def create_test_data():
            x = mp.device("P0")(lambda: jnp.array([0, 1]))()
            y = mp.device("P1")(lambda: jnp.array([1, 2]))()
            return [x, y]

        @mp.function
        def test_sum_func(agg):
            data = create_test_data()
            result = agg.sum(data)
            result = mp.put("P0", result)
            return result

        # Perform sum and get results
        result = mp.evaluate(simulator, test_sum_func, aggregator)
        fetched_result = fetch_from_label_party(simulator, result)

        expected = jnp.array([1, 3])
        assert fetched_result is not None
        assert jnp.array_equal(fetched_result, expected)
        assert fetched_result.dtype == jnp.int64

    def test_average_with_array_input(self, simulator, aggregator):
        """Test average aggregation with array values."""

        @mp.function
        def create_test_data():
            a = mp.device("P0")(lambda: jnp.array([2, 4]))()
            b = mp.device("P1")(lambda: jnp.array([2, 4]))()
            return [a, b]

        @mp.function
        def test_average_func(agg):
            data = create_test_data()
            result = agg.average(data)
            result = mp.put("P0", result)
            return result

        # Perform average and get results
        result = mp.evaluate(simulator, test_average_func, aggregator)
        fetched_result = fetch_from_label_party(simulator, result)

        expected = jnp.array([2.0, 4.0])
        assert fetched_result is not None
        assert jnp.array_equal(fetched_result, expected)
        assert fetched_result.dtype == jnp.float64
