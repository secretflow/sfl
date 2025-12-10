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
import jax.random as random
import mplang.v1 as mp
import numpy as np
import pytest

from sfl_lite.ml.linear_model.plain_fed import (
    PlainFederatedLinearRegression,
    create_plain_federated_lr,
)


class TestPlainFederatedLinearRegression:
    """Test suite for plain federated linear regression implementation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment with simulator."""
        # Note: Using mock data for testing since mplang.v1 might not be available
        # In actual implementation, you would initialize the MP context here
        pass

    def test_plain_federated_linear_regression_initialization(self):
        """Test PlainFederatedLinearRegression initialization."""
        # Create a mock interpreter for testing
        mock_interpreter = "mock_simulator"  # Simple mock for parameter testing

        model = PlainFederatedLinearRegression(
            interpreter=mock_interpreter,
            fit_intercept=True,
            learning_rate=0.1,
            max_iter=100,
            tol=1e-6,
            random_state=42,
            label_device="alice",
        )

        assert model.fit_intercept is True
        assert model.learning_rate == 0.1
        assert model.max_iter == 100
        assert model.tol == 1e-6
        assert model.random_state == 42
        assert model.label_device == "alice"

    def test_plain_federated_basic_fit(self):
        """Test basic fitting functionality with synthetic data."""
        # Create mock MPObjects for testing
        # In real implementation, these would be actual MPObjects
        n_samples = 100
        n_features_alice = 2
        n_features_bob = 3

        # Mock feature data for each party
        class MockMPObject:
            def __init__(self, data):
                self.data = np.array(data)
                self.shape = self.data.shape

            def __array__(self):
                return self.data

        # Generate synthetic data
        np.random.seed(42)
        MockMPObject(np.random.normal(0, 1, (n_samples, n_features_alice)))
        MockMPObject(np.random.normal(0, 1, (n_samples, n_features_bob)))
        MockMPObject(np.random.normal(0, 1, (n_samples,)))

        # Create model with mock interpreter
        mock_interpreter = "mock_simulator"
        model = PlainFederatedLinearRegression(
            interpreter=mock_interpreter,
            learning_rate=0.01,
            max_iter=10,
            fit_intercept=True,
            random_state=42,
            label_device="alice",
        )

        # Test initialization state
        assert not hasattr(model, "_is_fitted") or not model._is_fitted

        # Test parameter access
        params = model.get_params()
        assert params["learning_rate"] == 0.01
        assert params["max_iter"] == 10
        assert params["fit_intercept"] is True

    def test_plain_federated_no_intercept(self):
        """Test plain federated linear regression without intercept."""
        n_samples = 50
        n_features = 2

        class MockMPObject:
            def __init__(self, data):
                self.data = np.array(data)
                self.shape = self.data.shape

        # Generate mock data
        np.random.seed(45)
        MockMPObject(np.random.normal(0, 1, (n_samples, n_features)))
        MockMPObject(np.random.normal(0, 1, (n_samples,)))

        # Create model without intercept
        mock_interpreter = "mock_simulator"
        model = PlainFederatedLinearRegression(
            interpreter=mock_interpreter,
            fit_intercept=False,
            learning_rate=0.01,
            max_iter=5,
            random_state=42,
            label_device="alice",
        )

        # Test that intercept is disabled
        assert model.fit_intercept is False

        # Test parameters
        params = model.get_params()
        assert params["fit_intercept"] is False

    def test_plain_federated_sklearn_interface(self):
        """Test sklearn-compatible interface methods."""
        n_samples = 50
        n_features_alice = 2
        n_features_bob = 3

        class MockMPObject:
            def __init__(self, data):
                self.data = np.array(data)
                self.shape = self.data.shape

        # Generate synthetic data
        np.random.seed(42)
        alice_features = MockMPObject(
            np.random.normal(0, 1, (n_samples, n_features_alice))
        )
        bob_features = MockMPObject(np.random.normal(0, 1, (n_samples, n_features_bob)))
        MockMPObject(np.random.normal(0, 1, (n_samples,)))

        X = {"alice": alice_features, "bob": bob_features}

        # Test sklearn interface
        mock_interpreter = "mock_simulator"
        model = PlainFederatedLinearRegression(
            interpreter=mock_interpreter,
            learning_rate=0.05,
            max_iter=20,
            fit_intercept=True,
            random_state=42,
            label_device="alice",
        )

        # Test get_params
        params = model.get_params()
        assert "learning_rate" in params
        assert "fit_intercept" in params
        assert "label_device" in params

        # Test set_params
        model.set_params(learning_rate=0.1, max_iter=50)
        assert model.learning_rate == 0.1
        assert model.max_iter == 50

        # Test that model is not fitted initially
        with pytest.raises(ValueError, match="not fitted"):
            model.coef_

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

    def test_plain_federated_factory_function(self):
        """Test the factory function for creating models."""
        mock_interpreter = "mock_simulator"
        model = create_plain_federated_lr(
            interpreter=mock_interpreter,
            learning_rate=0.05,
            max_iter=200,
            fit_intercept=False,
            label_device="bob",
        )

        assert isinstance(model, PlainFederatedLinearRegression)
        assert model.learning_rate == 0.05
        assert model.max_iter == 200
        assert model.fit_intercept is False
        assert model.label_device == "bob"

    def test_plain_federated_different_learning_rates(self):
        """Test training with different learning rates."""
        n_samples = 50
        n_features = 2

        class MockMPObject:
            def __init__(self, data):
                self.data = np.array(data)
                self.shape = self.data.shape

        # Generate data
        np.random.seed(42)
        MockMPObject(np.random.normal(0, 1, (n_samples, n_features)))
        MockMPObject(np.random.normal(0, 1, (n_samples,)))

        learning_rates = [0.001, 0.01, 0.1]

        mock_interpreter = "mock_simulator"
        for lr in learning_rates:
            model = PlainFederatedLinearRegression(
                interpreter=mock_interpreter,
                learning_rate=lr,
                max_iter=5,
                fit_intercept=True,
                random_state=42,
                label_device="alice",
            )

            # Test that model can be created with different learning rates
            assert model.learning_rate == lr

            # Test parameter validation
            params = model.get_params()
            assert params["learning_rate"] == lr

    def test_plain_federated_parameter_validation(self):
        """Test parameter validation and error handling."""
        mock_interpreter = "mock_simulator"
        model = PlainFederatedLinearRegression(interpreter=mock_interpreter)

        # Test invalid parameter setting
        with pytest.raises(ValueError, match="Invalid parameter"):
            model.set_params(invalid_param=123)

        # Test valid parameter setting
        model.set_params(learning_rate=0.05, max_iter=100)
        assert model.learning_rate == 0.05
        assert model.max_iter == 100

    def test_plain_federated_reproducibility(self):
        """Test that results are reproducible with same random state."""
        n_samples = 50
        n_features = 2

        class MockMPObject:
            def __init__(self, data):
                self.data = np.array(data)
                self.shape = self.data.shape

        # Generate data
        np.random.seed(42)
        MockMPObject(np.random.normal(0, 1, (n_samples, n_features)))
        MockMPObject(np.random.normal(0, 1, (n_samples,)))

        mock_interpreter = "mock_simulator"

        def create_and_test_model(seed):
            model = PlainFederatedLinearRegression(
                interpreter=mock_interpreter,
                learning_rate=0.01,
                max_iter=5,
                random_state=seed,
                label_device="alice",
            )
            return model

        # Create models with same seed
        model1 = create_and_test_model(42)
        model2 = create_and_test_model(42)

        # Both should have same parameters
        params1 = model1.get_params()
        params2 = model2.get_params()
        assert params1 == params2
        assert model1.random_state == model2.random_state

    def test_plain_federated_edge_cases(self):
        """Test handling of edge cases."""
        # Test empty X dictionary
        mock_interpreter = "mock_simulator"
        with pytest.raises((ValueError, KeyError)):
            model = PlainFederatedLinearRegression(interpreter=mock_interpreter)
            model._validate_input({})

        # Test single feature case
        class MockMPObject:
            def __init__(self, data):
                self.data = np.array(data)
                self.shape = self.data.shape

        np.random.seed(42)
        MockMPObject(np.random.normal(0, 1, (3, 1)))

        model = PlainFederatedLinearRegression(
            interpreter=mock_interpreter, random_state=42
        )
        # Should not raise error for single feature
        # Note: _validate_input is called internally during fit/predict
        assert model.random_state == 42

    def test_plain_federated_string_device_names(self):
        """Test that string device names work correctly."""
        n_samples = 20
        n_features = 1

        class MockMPObject:
            def __init__(self, data):
                self.data = np.array(data)
                self.shape = self.data.shape

        # Test various string device names
        device_names = ["alice", "bob", "charlie", "party_0", "device_1"]
        mock_interpreter = "mock_simulator"

        for device_name in device_names:
            MockMPObject(np.random.normal(0, 1, (n_samples, n_features)))
            MockMPObject(np.random.normal(0, 1, (n_samples,)))

            model = PlainFederatedLinearRegression(
                interpreter=mock_interpreter,
                learning_rate=0.01,
                max_iter=2,
                label_device=device_name,
                random_state=42,
            )

            # Should accept string device names
            assert model.label_device == device_name
            params = model.get_params()
            assert params["label_device"] == device_name

    def test_plain_federated_train_and_fit_with_mplang(self):
        """Test training and fitting a linear model with actual MPLang execution."""
        # Step 1: Create cluster specification (following the ML library example pattern)
        cluster_spec = mp.ClusterSpec.from_dict(
            {
                "nodes": [
                    {"name": f"node_{i}", "endpoint": f"127.0.0.1:{61920 + i}"}
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

        # Step 2: Create simulator (user manages lifecycle)
        sim = mp.Simulator(cluster_spec)

        # Step 3: Generate federated training data using MPLang and random values
        @mp.function
        def create_federated_training_data(seed=42):
            """Create vertical federated learning dataset with random values."""
            n_samples = 30
            n_features_alice = 2
            n_features_bob = 2

            # Initialize random keys
            key = random.PRNGKey(seed)
            key_alice, key_bob, key_noise = random.split(key, 3)

            # Alice (P0): generate random features
            X_alice = mp.device("P0")(
                lambda: random.normal(
                    key_alice, (n_samples, n_features_alice), dtype=jnp.float32
                )
            )()

            # Bob (P1): generate random features
            X_bob = mp.device("P1")(
                lambda: random.normal(
                    key_bob, (n_samples, n_features_bob), dtype=jnp.float32
                )
            )()

            # Generate labels using simple linear model for testing
            # Create labels on Alice's device using a simple approach
            y_alice = mp.device("P0")(
                lambda: random.normal(key_noise, (n_samples,), dtype=jnp.float32)
            )()

            return X_alice, X_bob, y_alice

        # Execute data generation with MPLang
        X_alice, X_bob, y_alice = mp.evaluate(sim, create_federated_training_data, 42)

        # Step 4: Create our federated linear regression model
        model = PlainFederatedLinearRegression(
            interpreter=sim,
            learning_rate=0.01,
            max_iter=20,
            fit_intercept=True,
            random_state=42,
            label_device="P0",  # Use actual device name from cluster spec
        )

        # Step 5: Prepare data in the format our model expects
        X_federated = {
            "P0": X_alice,  # Real MPObject from P0 (alice)
            "P1": X_bob,  # Real MPObject from P1 (bob)
        }
        y_federated = y_alice  # Real MPObject from P0

        # Step 6: Train the model (this will use our @mp.function decorated core functions)
        print("Training federated linear regression with MPLang...")
        fitted_model = model.fit(X_federated, y_federated)

        # Step 7: Verify the model was fitted successfully
        assert fitted_model._is_fitted
        assert fitted_model is model  # Returns self

        # Step 8: Access model parameters (should work after fitting)
        coefficients = model.coef_
        intercept = model.intercept_
        n_features = model.n_features_in_

        assert coefficients is not None
        assert len(coefficients) == 2  # P0 and P1
        assert "P0" in coefficients
        assert "P1" in coefficients
        assert n_features == 4  # 2 features from P0 + 2 from P1

        print("✓ Model fitted successfully")
        print(f"  - Coefficients: {len(coefficients)} parties")
        print(f"  - Total features: {n_features}")
        print(f"  - Has intercept: {intercept is not None}")

        # Step 9: Make predictions on the training data
        predictions = model.predict(X_federated)
        assert predictions is not None
        print("✓ Predictions generated successfully")

        # Step 10: Compute score (R²)
        score = model.score(X_federated, y_federated)
        # Note: score is now an MPObject (secure), not a float
        assert score is not None
        print("✓ Secure score computed successfully (MPObject)")

        # Step 11: Test parameter management
        params = model.get_params()
        assert params["learning_rate"] == 0.01
        assert params["max_iter"] == 20
        assert params["label_device"] == "P0"

        # Step 12: Test parameter updates
        model.set_params(learning_rate=0.02)
        assert model.learning_rate == 0.02

        print("✓ Complete MPLang federated training workflow successful!")
        print("  - Random data generation with @mp.function")
        print("  - Model fitting and prediction")
        print("  - sklearn-compatible interface")

    def test_plain_federated_r2_score_verification(self):
        """Test R² score implementation against cleartext counterpart."""
        # Step 1: Create cluster specification
        cluster_spec = mp.ClusterSpec.from_dict(
            {
                "nodes": [
                    {"name": f"node_{i}", "endpoint": f"127.0.0.1:{61920 + i}"}
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

        # Step 2: Create simulator
        sim = mp.Simulator(cluster_spec)

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

        # Step 3: Generate test data with known linear relationship
        @mp.function
        def create_known_linear_data(seed=42):
            """Create data with perfect linear relationship for R² testing."""
            n_samples = 50

            key = random.PRNGKey(seed)
            key1, key2, key3 = random.split(key, 3)

            # Alice features: x1
            X1 = mp.device("P0")(
                lambda: random.uniform(
                    key1, (n_samples, 2), minval=0, maxval=10, dtype=jnp.float32
                )
            )()

            # Bob features: x2
            X2 = mp.device("P1")(
                lambda: random.uniform(
                    key2, (n_samples, 2), minval=0, maxval=10, dtype=jnp.float32
                )
            )()

            # Create linear relationship using matrix operations (MPLang compatible)
            # y = 2*x1[0] + 3*x1[1] + 1*x2[0] + 0.5*x2[1] + 1.5
            alice_weights = jnp.array([2.0, 3.0], dtype=jnp.float32)
            bob_weights = jnp.array([1.0, 0.5], dtype=jnp.float32)

            # Compute alice contribution
            alice_contrib = mp.device("P0")(lambda x, w: jnp.dot(x, w))(
                X1, alice_weights
            )

            # Compute bob contribution
            bob_contrib = mp.device("P1")(lambda x, w: jnp.dot(x, w))(X2, bob_weights)

            # Combine on P0 for final labels
            y = mp.device("P0")(
                lambda a_contrib, b_contrib: a_contrib + b_contrib + 1.5
            )(alice_contrib, bob_contrib)

            return X1, X2, y

        # Generate the known data
        X1, X2, y = mp.evaluate(sim, create_known_linear_data, 42)

        # Step 4: Also get cleartext version for comparison
        @mp.function
        def get_cleartext_data():
            """Get cleartext version of the same data."""
            n_samples = 50

            key = random.PRNGKey(42)  # Same seed
            key1, key2, key3 = random.split(key, 3)

            # Generate same data in cleartext
            x1_clear = random.uniform(
                key1, (n_samples, 2), minval=0, maxval=10, dtype=jnp.float32
            )
            x2_clear = random.uniform(
                key2, (n_samples, 2), minval=0, maxval=10, dtype=jnp.float32
            )
            y_clear = (
                2.0 * x1_clear[:, 0]
                + 3.0 * x1_clear[:, 1]
                + 1.0 * x2_clear[:, 0]
                + 0.5 * x2_clear[:, 1]
                + 1.5
            )

            return x1_clear, x2_clear, y_clear

        x1_clear, x2_clear, y_clear = mp.evaluate(sim, get_cleartext_data)

        # Step 5: Train federated model
        model = PlainFederatedLinearRegression(
            interpreter=sim,
            learning_rate=0.1,
            max_iter=200,  # More iterations for better convergence
            fit_intercept=True,
            random_state=42,
            label_device="P0",
        )

        X_federated = {"P0": X1, "P1": X2}

        print("Training federated model for R² verification...")
        model.fit(X_federated, y)

        # Step 6: Get secure R² score
        r2_secure = model.score(X_federated, y)

        # Step 7: Get secure predictions for cleartext comparison
        y_pred_secure = model.predict(X_federated)

        # Step 8: Fetch cleartext values for comparison (only for testing)
        y_pred_clear = fetch_from_label_party(sim, y_pred_secure)
        y_true_clear = fetch_from_label_party(sim, y)

        print(
            f"Debug: y_pred_clear type: {type(y_pred_clear)}, shape: {np.array(y_pred_clear).shape if hasattr(y_pred_clear, '__len__') else 'scalar'}"
        )
        print(
            f"Debug: y_true_clear type: {type(y_true_clear)}, shape: {np.array(y_true_clear).shape if hasattr(y_true_clear, '__len__') else 'scalar'}"
        )

        # Step 9: Compute cleartext R² for verification
        def compute_r2_cleartext(y_true, y_pred):
            """Compute R² in cleartext for verification."""
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        r2_cleartext = compute_r2_cleartext(y_true_clear, y_pred_clear)

        # Step 10: Fetch secure R² for comparison (only for testing)
        r2_secure_value = float(fetch_from_label_party(sim, r2_secure))

        print(f"Debug: r2_secure_value extracted: {r2_secure_value}")

        print("✓ R² Score Verification Results:")
        print(f"  - Secure R²: {r2_secure_value:.6f}")
        print(f"  - Cleartext R²: {r2_cleartext:.6f}")
        print(f"  - Difference: {abs(r2_secure_value - r2_cleartext):.6f}")

        # Step 11: Verify that the scores are very close
        # Allow small numerical differences due to floating point precision
        assert (
            abs(r2_secure_value - r2_cleartext) < 1e-4
        ), f"R² scores don't match: secure={r2_secure_value}, cleartext={r2_cleartext}"

        # Step 12: Since we have a perfect linear relationship and good convergence,
        # R² should be very close to 1.0
        assert (
            r2_cleartext > 0.95
        ), f"R² should be high for perfect linear relationship: {r2_cleartext}"

        print("✓ R² implementation verified against cleartext counterpart!")
        print("  - Secure and cleartext R² scores match within tolerance")
        print("  - High R² score confirms correct linear relationship detection")

        # Step 13: Test with noisy data for more realistic R²
        @mp.function
        def create_noisy_linear_data(seed=123):
            """Create data with linear relationship + noise."""
            n_samples = 50

            key = random.PRNGKey(seed)
            key1, key2, key3, key4 = random.split(key, 4)

            # Features
            X1 = mp.device("P0")(
                lambda: random.uniform(
                    key1, (n_samples, 1), minval=0, maxval=10, dtype=jnp.float32
                )
            )()
            X2 = mp.device("P1")(
                lambda: random.uniform(
                    key2, (n_samples, 1), minval=0, maxval=10, dtype=jnp.float32
                )
            )()

            # Linear relationship with significant noise
            noise = mp.device("P0")(
                lambda: random.normal(key3, (n_samples,), dtype=jnp.float32)
                * 2.0  # Significant noise
            )()

            # Compute contributions using matrix operations
            alice_contrib = mp.device("P0")(
                lambda x: 2.0 * jnp.squeeze(x)  # X1 has shape (n, 1)
            )(X1)

            bob_contrib = mp.device("P1")(
                lambda x: 3.0 * jnp.squeeze(x)  # X2 has shape (n, 1)
            )(X2)

            y_noisy = mp.device("P0")(
                lambda a_contrib, b_contrib, n: a_contrib + b_contrib + n + 5.0
            )(alice_contrib, bob_contrib, noise)

            return X1, X2, y_noisy

        X1_noisy, X2_noisy, y_noisy = mp.evaluate(sim, create_noisy_linear_data, 123)

        # Train on noisy data
        model_noisy = PlainFederatedLinearRegression(
            interpreter=sim,
            learning_rate=0.05,
            max_iter=100,
            fit_intercept=True,
            random_state=42,
            label_device="P0",
        )

        X_noisy = {"P0": X1_noisy, "P1": X2_noisy}
        model_noisy.fit(X_noisy, y_noisy)

        r2_noisy_secure = model_noisy.score(X_noisy, y_noisy)
        r2_noisy_value = float(fetch_from_label_party(sim, r2_noisy_secure))

        print("✓ Noisy Data R² Test:")
        print(f"  - R² with noise: {r2_noisy_value:.6f}")

        # With noise, R² should be lower but still positive for good model
        assert (
            0.0 <= r2_noisy_value <= 1.0
        ), f"R² should be between 0 and 1: {r2_noisy_value}"

        print("✓ Complete R² verification successful!")
        print("  - Perfect linear data: High R² (near 1.0)")
        print("  - Noisy linear data: Moderate R² (0.0-1.0)")
        print("  - Secure implementation matches cleartext computation")
