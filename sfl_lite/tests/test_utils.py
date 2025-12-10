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

"""Utility functions for sfl_lite tests."""

import jax.numpy as jnp
import jax.random as random
import mplang.v1 as mp


def create_test_cluster_spec(base_port=61920):
    """
    Create a standard 3-node cluster specification for testing.

    Args:
        base_port: Base port number for node endpoints

    Returns:
        ClusterSpec: A cluster specification with 3 nodes (P0, P1, P2) and
                     one secure computation device (SP0)
    """
    return mp.ClusterSpec.from_dict(
        {
            "nodes": [
                {"name": f"node_{i}", "endpoint": f"127.0.0.1:{base_port + i}"}
                for i in range(3)
            ],
            "devices": {
                "SP0": {
                    "kind": "SPU",
                    "members": [f"node_{i}" for i in range(3)],
                    "config": {"protocol": "SEMI2K", "field": "FM128"},
                },
                "P0": {"kind": "PPU", "members": ["node_0"]},  # alice
                "P1": {"kind": "PPU", "members": ["node_1"]},  # bob
                "P2": {"kind": "PPU", "members": ["node_2"]},  # charlie
            },
        }
    )


def fetch_from_label_party(sim, mp_object, label_party_idx=0):
    """
    Utility function to fetch cleartext value from the label party.

    mp.fetch returns one value per party in the cluster. Since computations
    on labels happen on the label device (typically P0), we extract that party's result.

    Args:
        sim: The MP simulator
        mp_object: The secure MPObject to fetch
        label_party_idx: Index of the label party (default 0 for P0)

    Returns:
        The cleartext value from the label party
    """
    raw_result = mp.fetch(sim, mp_object)

    # Handle multi-party return
    if isinstance(raw_result, (list, tuple)) and len(raw_result) > 0:
        return raw_result[label_party_idx]
    else:
        return raw_result


@mp.function
def create_random_federated_data(
    seed=42, n_samples=30, n_features_p0=2, n_features_p1=2
):
    """
    Create random federated learning dataset with features split across parties.

    Args:
        seed: Random seed for reproducibility
        n_samples: Number of samples
        n_features_p0: Number of features for party P0
        n_features_p1: Number of features for party P1

    Returns:
        Tuple of (X_p0, X_p1, y) as MPObjects
    """
    # Initialize random keys
    key = random.PRNGKey(seed)
    key_p0, key_p1, key_y = random.split(key, 3)

    # P0 (Alice): generate random features
    X_p0 = mp.device("P0")(
        lambda: random.normal(key_p0, (n_samples, n_features_p0), dtype=jnp.float32)
    )()

    # P1 (Bob): generate random features
    X_p1 = mp.device("P1")(
        lambda: random.normal(key_p1, (n_samples, n_features_p1), dtype=jnp.float32)
    )()

    # Generate random labels on P0
    y = mp.device("P0")(lambda: random.normal(key_y, (n_samples,), dtype=jnp.float32))()

    return X_p0, X_p1, y


@mp.function
def create_linear_federated_data(
    seed=42, n_samples=50, with_noise=False, noise_scale=2.0
):
    """
    Create federated data with known linear relationship for testing.

    Creates data where y = 2*x1[0] + 3*x1[1] + 1*x2[0] + 0.5*x2[1] + 1.5 [+ noise]

    Args:
        seed: Random seed for reproducibility
        n_samples: Number of samples
        with_noise: Whether to add noise to the labels
        noise_scale: Scale of noise to add (if with_noise=True)

    Returns:
        Tuple of (X_p0, X_p1, y) as MPObjects
    """
    key = random.PRNGKey(seed)
    key1, key2, key3 = random.split(key, 3)

    # Alice features: x1 with 2 features
    X1 = mp.device("P0")(
        lambda: random.uniform(
            key1, (n_samples, 2), minval=0, maxval=10, dtype=jnp.float32
        )
    )()

    # Bob features: x2 with 2 features
    X2 = mp.device("P1")(
        lambda: random.uniform(
            key2, (n_samples, 2), minval=0, maxval=10, dtype=jnp.float32
        )
    )()

    # Create linear relationship using matrix operations
    alice_weights = jnp.array([2.0, 3.0], dtype=jnp.float32)
    bob_weights = jnp.array([1.0, 0.5], dtype=jnp.float32)

    # Compute alice contribution
    alice_contrib = mp.device("P0")(lambda x, w: jnp.dot(x, w))(X1, alice_weights)

    # Compute bob contribution
    bob_contrib = mp.device("P1")(lambda x, w: jnp.dot(x, w))(X2, bob_weights)

    # Combine on P0 for final labels
    if with_noise:
        noise = mp.device("P0")(
            lambda: random.normal(key3, (n_samples,), dtype=jnp.float32) * noise_scale
        )()
        y = mp.device("P0")(
            lambda a_contrib, b_contrib, n: a_contrib + b_contrib + n + 1.5
        )(alice_contrib, bob_contrib, noise)
    else:
        y = mp.device("P0")(lambda a_contrib, b_contrib: a_contrib + b_contrib + 1.5)(
            alice_contrib, bob_contrib
        )

    return X1, X2, y


@mp.function
def create_single_party_data(seed=42, n_samples=30, n_features=3):
    """
    Create federated data where only one party (P0) has features.

    Useful for testing edge cases where federated learning works with a single contributor.

    Args:
        seed: Random seed for reproducibility
        n_samples: Number of samples
        n_features: Number of features for P0

    Returns:
        Tuple of (X_p0, y) as MPObjects
    """
    key = random.PRNGKey(seed)
    key1, key2 = random.split(key, 2)

    # Only P0 (Alice) has features
    X_p0 = mp.device("P0")(
        lambda: random.normal(key1, (n_samples, n_features), dtype=jnp.float32)
    )()

    # Generate labels on P0
    y = mp.device("P0")(lambda: random.normal(key2, (n_samples,), dtype=jnp.float32))()

    return X_p0, y
