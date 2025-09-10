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
import mplang
import mplang.simp as simp
import pytest

from sfl_lite.security.aggregation.mp_aggregator import MPAggregator


class TestMPAggregator:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.sim3 = mplang.Simulator.simple(3)
        mplang.set_ctx(self.sim3)
        self.agg = MPAggregator()

    def test_sum(self):
        # Create test data
        x = simp.runAt(0, lambda: 0)()
        y = simp.runAt(1, lambda: 1)()

        # Perform sum
        z = self.agg.sum({0: x, 1: y})

        # Verify results
        revealed_z = simp.reveal(z)
        fetched = mplang.fetch(None, revealed_z)
        assert len(fetched) == 3  # 3 parties
        assert all(arr is not None for arr in fetched)
        assert all(jnp.array_equal(arr, jnp.array(1)) for arr in fetched)
        assert fetched[0].dtype == jnp.int64

    def test_average(self):
        # Create test data
        a = simp.runAt(0, lambda: 2)()
        b = simp.runAt(1, lambda: 2)()

        # Perform average
        avg = self.agg.average({0: a, 1: b})

        # Verify results
        revealed_avg = simp.reveal(avg)
        fetched = mplang.fetch(None, revealed_avg)
        assert len(fetched) == 3  # 3 parties
        assert all(arr is not None for arr in fetched)
        assert all(jnp.array_equal(arr, jnp.array(2.)) for arr in fetched)
        assert fetched[0].dtype == jnp.float64

    def test_sum_with_array_input(self):
        # Create test data as jnp arrays
        x = simp.runAt(0, lambda: jnp.array([0, 1]))()
        y = simp.runAt(1, lambda: jnp.array([1, 2]))()

        # Perform sum
        z = self.agg.sum({0: x, 1: y})

        # Verify results
        revealed_z = simp.reveal(z)
        fetched = mplang.fetch(None, revealed_z)
        expected = jnp.array([1, 3])
        print(fetched, expected)
        assert len(fetched) == 3  # 3 parties
        assert all(arr is not None for arr in fetched)
        assert all(jnp.array_equal(arr, expected) for arr in fetched)
        assert fetched[0].dtype == jnp.int64

    def test_average_with_array_input(self):
        # Create test data as jnp arrays
        a = simp.runAt(0, lambda: jnp.array([2, 4]))()
        b = simp.runAt(1, lambda: jnp.array([2, 4]))()

        # Perform average
        avg = self.agg.average({0: a, 1: b})

        # Verify results
        revealed_avg = simp.reveal(avg)
        fetched = mplang.fetch(None, revealed_avg)
        expected = jnp.array([2., 4.])
        assert len(fetched) == 3  # 3 parties
        assert all(arr is not None for arr in fetched)
        assert all(jnp.array_equal(arr, expected) for arr in fetched)
        assert fetched[0].dtype == jnp.float64
