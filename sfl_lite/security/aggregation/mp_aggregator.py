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
from typing import Dict

import jax.numpy as jnp
import mplang
import mplang.smpc as smpc
from mplang.core.base import MPObject

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
        sealed_data = [smpc.sealFrom(data, party) for party, data in data.items()]

        def _sum(data):
            return reduce(jnp.add, data)

        return smpc.srun(
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
        sealed_data = [smpc.sealFrom(data, party) for party, data in data.items()]

        def _average(data):
            return reduce(jnp.add, data) / len(data)

        return smpc.srun(_average)(sealed_data)


if __name__ == "__main__":
    # example usage
    import random
    from functools import partial

    import mplang
    import mplang.simp as simp

    sim3 = mplang.Simulator(3)
    mplang.set_ctx(sim3)
    x = simp.runAt(0, partial(random.randint, 0, 10))()
    # make a random number at P1
    y = simp.runAt(1, partial(random.randint, 0, 10))()
    agg = MPAggregator()
    z = agg.sum({0: x, 1: y})
    # Print the results.
    print("x:", x)
    print("fetch(x):", mplang.fetch(None, x))
    print("y:", y)
    print("fetch(y):", mplang.fetch(None, y))
    print("z:", z)
    print("fetch(z):", mplang.fetch(None, smpc.reveal(z)))

    # Example usage of average
    a = simp.runAt(0, partial(random.randint, 0, 10))()
    b = simp.runAt(1, partial(random.randint, 0, 10))()
    print("a:", a)
    print("fetch(a):", mplang.fetch(None, a))
    print("b:", b)
    print("fetch(b):", mplang.fetch(None, b))
    avg = agg.average({0: a, 1: b})
    print("avg:", avg)
    print("fetch(avg):", mplang.fetch(None, smpc.reveal(avg)))
