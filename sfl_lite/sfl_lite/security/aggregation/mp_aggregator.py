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


from functools import partial, reduce
from typing import Dict

import jax.numpy as jnp
import mplang
from mplang import simp
from mplang.core import MPObject

from sfl_lite.security.aggregation.aggregator import Aggregator


class MPAggregator(Aggregator):
    """Aggregator based on MPObject.

    The computation will be performed according to MPLang setups.

    """

    def __init__(self):
        pass

    @mplang.function
    def sum(self, data: Dict[int, MPObject]) -> MPObject:
        """Sum of array elements over a given axis.

        Args:
            data: dict of party: MPObjects.

        Returns:
            a device object holds the sum.
        """
        assert data, "Data to aggregate should not be None or empty!"
        sealed_data = [simp.sealFrom(value, party) for party, value in data.items()]

        def _sum(value):
            return reduce(jnp.add, value)

        return simp.srun(
            _sum,
        )(sealed_data)

    @mplang.function
    def average(self, data: Dict[int, MPObject]) -> MPObject:
        """Compute the average of array elements over a given axis.

        Args:
            data: dict of party: MPObjects.

        Returns:
            a device object holds the average.
        """
        assert data, "Data to aggregate should not be None or empty!"
        sealed_data = [simp.sealFrom(value, party) for party, value in data.items()]

        def _average(value):
            return reduce(jnp.add, value) / len(value)

        return simp.srun(_average)(sealed_data)
