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


from functools import reduce
from typing import List

import jax.numpy as jnp
import mplang.v1 as mp
from mplang.v1.core import MPObject

from sfl_lite.security.aggregation.aggregator import Aggregator


class MPAggregator(Aggregator):
    """Aggregator based on MPObject.

    The computation will be performed according to MPLang setups.

    """

    def __init__(self, secure_device: str = "SP0"):
        self.secure_device = secure_device

    @mp.function
    def sum(self, data: List[MPObject]) -> MPObject:
        """Sum of array elements over a given axis.

        Args:
            data: list of MPObjects.

        Returns:
            a device object holds the sum.
        """
        assert data, "Data to aggregate should not be None or empty!"

        # Put all data items to secure device
        secure_data = [mp.put(self.secure_device, item) for item in data]

        # Secure computation on the designated device
        def _sum(*values):
            return reduce(jnp.add, values)

        result = mp.device(self.secure_device)(_sum)(*secure_data)
        return result

    @mp.function
    def average(self, data: List[MPObject]) -> MPObject:
        """Compute the average of array elements over a given axis.

        Args:
            data: list of MPObjects.

        Returns:
            a device object holds the average.
        """
        assert data, "Data to aggregate should not be None or empty!"

        # Put all data items to secure device
        secure_data = [mp.put(self.secure_device, item) for item in data]

        # Secure computation on the designated device
        def _average(*values):
            return reduce(jnp.add, values) / len(values)

        result = mp.device(self.secure_device)(_average)(*secure_data)
        return result
