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

"""Pytest configuration and shared fixtures for sfl_lite tests."""

import mplang.v1 as mp
import pytest

from .test_utils import create_test_cluster_spec


@pytest.fixture(scope="function")
def cluster_spec():
    """
    Fixture that provides a standard 3-node cluster specification for testing.

    Returns:
        ClusterSpec: A cluster specification with 3 nodes (P0, P1, P2) and
                     one secure computation device (SP0)
    """
    return create_test_cluster_spec()


@pytest.fixture(scope="function")
def simulator(cluster_spec):
    """
    Fixture that provides a simulator initialized with the test cluster.

    Args:
        cluster_spec: The cluster specification fixture

    Returns:
        mp.Simulator: An initialized simulator
    """
    return mp.Simulator(cluster_spec)
