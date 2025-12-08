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


class TestMPAggregator:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.sim3 = mp.Simulator.simple(3)
        self.agg = MPAggregator()

    def test_sum(self):
        @mp.function
        def create_test_data():
            x = mp.device("P0")(lambda: 0)()
            y = mp.device("P1")(lambda: 1)()
            return [x, y]

        @mp.function
        def test_sum_func():
            data = create_test_data()
            return self.agg.sum(data)

        # Perform sum and get results
        result = mp.evaluate(self.sim3, test_sum_func)
        revealed_result = mp.evaluate(self.sim3, lambda: mp.reveal(result))
        fetched = mp.fetch(self.sim3, revealed_result)

        assert len(fetched) == 3  # 3 parties
        assert all(arr is not None for arr in fetched)
        assert all(jnp.array_equal(arr, jnp.array(1)) for arr in fetched)
        assert fetched[0].dtype == jnp.int64

    def test_average(self):
        @mp.function
        def create_test_data():
            a = mp.device("P0")(lambda: 2)()
            b = mp.device("P1")(lambda: 2)()
            return [a, b]

        @mp.function
        def test_average_func():
            data = create_test_data()
            return self.agg.average(data)

        # Perform average and get results
        result = mp.evaluate(self.sim3, test_average_func)
        revealed_result = mp.evaluate(self.sim3, lambda: mp.reveal(result))
        fetched = mp.fetch(self.sim3, revealed_result)

        assert len(fetched) == 3  # 3 parties
        assert all(arr is not None for arr in fetched)
        assert all(jnp.array_equal(arr, jnp.array(2.0)) for arr in fetched)
        assert fetched[0].dtype == jnp.float64

    def test_sum_with_array_input(self):
        @mp.function
        def create_test_data():
            x = mp.device("P0")(lambda: jnp.array([0, 1]))()
            y = mp.device("P1")(lambda: jnp.array([1, 2]))()
            return [x, y]

        @mp.function
        def test_sum_func():
            data = create_test_data()
            return self.agg.sum(data)

        # Perform sum and get results
        result = mp.evaluate(self.sim3, test_sum_func)
        revealed_result = mp.evaluate(self.sim3, lambda: mp.reveal(result))
        fetched = mp.fetch(self.sim3, revealed_result)

        expected = jnp.array([1, 3])
        assert len(fetched) == 3  # 3 parties
        assert all(arr is not None for arr in fetched)
        assert all(jnp.array_equal(arr, expected) for arr in fetched)
        assert fetched[0].dtype == jnp.int64

    def test_average_with_array_input(self):
        @mp.function
        def create_test_data():
            a = mp.device("P0")(lambda: jnp.array([2, 4]))()
            b = mp.device("P1")(lambda: jnp.array([2, 4]))()
            return [a, b]

        @mp.function
        def test_average_func():
            data = create_test_data()
            return self.agg.average(data)

        # Perform average and get results
        result = mp.evaluate(self.sim3, test_average_func)
        revealed_result = mp.evaluate(self.sim3, lambda: mp.reveal(result))
        fetched = mp.fetch(self.sim3, revealed_result)

        expected = jnp.array([2.0, 4.0])
        assert len(fetched) == 3  # 3 parties
        assert all(arr is not None for arr in fetched)
        assert all(jnp.array_equal(arr, expected) for arr in fetched)
        assert fetched[0].dtype == jnp.float64
