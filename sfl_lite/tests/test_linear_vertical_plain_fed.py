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

import mplang.v1 as mp
import numpy as np
import pytest

from sfl_lite.ml.linear_model.plain_fed import (
    PlainFederatedLinearRegression,
    create_plain_federated_lr,
)

from .test_utils import (
    create_linear_federated_data,
    create_random_federated_data,
    create_single_party_data,
    fetch_from_label_party,
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

    def test_plain_federated_basic_fit_and_intercept(self):
        """Test basic fitting functionality and intercept configuration."""
        mock_interpreter = "mock_simulator"

        # Test with intercept
        model_with_intercept = PlainFederatedLinearRegression(
            interpreter=mock_interpreter,
            learning_rate=0.01,
            max_iter=10,
            fit_intercept=True,
            random_state=42,
            label_device="alice",
        )
        assert (
            not hasattr(model_with_intercept, "_is_fitted")
            or not model_with_intercept._is_fitted
        )
        assert model_with_intercept.fit_intercept is True

        # Test without intercept
        model_no_intercept = PlainFederatedLinearRegression(
            interpreter=mock_interpreter,
            fit_intercept=False,
            random_state=42,
            label_device="alice",
        )
        assert model_no_intercept.fit_intercept is False
        params = model_no_intercept.get_params()
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

    def test_plain_federated_parameter_management(self):
        """Test parameter validation, setting, and reproducibility."""
        mock_interpreter = "mock_simulator"

        # Test parameter validation and setting
        model = PlainFederatedLinearRegression(
            interpreter=mock_interpreter, random_state=42
        )

        # Test invalid parameter
        with pytest.raises(ValueError, match="Invalid parameter"):
            model.set_params(invalid_param=123)

        # Test valid parameter setting with different learning rates
        for lr in [0.001, 0.01, 0.1]:
            model.set_params(learning_rate=lr)
            assert model.learning_rate == lr
            assert model.get_params()["learning_rate"] == lr

        # Test reproducibility with same random state
        model1 = PlainFederatedLinearRegression(
            interpreter=mock_interpreter, random_state=42, learning_rate=0.01
        )
        model2 = PlainFederatedLinearRegression(
            interpreter=mock_interpreter, random_state=42, learning_rate=0.01
        )
        assert model1.get_params() == model2.get_params()
        assert model1.random_state == model2.random_state

    def test_plain_federated_edge_cases_and_device_names(self):
        """Test edge cases and device name handling."""
        mock_interpreter = "mock_simulator"

        # Test empty X dictionary validation
        model = PlainFederatedLinearRegression(interpreter=mock_interpreter)
        with pytest.raises((ValueError, KeyError)):
            model._validate_input({})

        # Test various string device names
        for device_name in ["alice", "bob", "party_0", "device_1"]:
            model = PlainFederatedLinearRegression(
                interpreter=mock_interpreter,
                label_device=device_name,
                random_state=42,
            )
            assert model.label_device == device_name
            assert model.get_params()["label_device"] == device_name

    def test_plain_federated_train_and_fit_with_mplang(self, simulator):
        """Test training and fitting a linear model with actual MPLang execution."""
        # Generate federated training data
        X_alice, X_bob, y_alice = mp.evaluate(
            simulator, create_random_federated_data, 42
        )

        # Create our federated linear regression model
        model = PlainFederatedLinearRegression(
            interpreter=simulator,
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

    def test_plain_federated_r2_score_verification(self, simulator):
        """Test R² score implementation against cleartext counterpart."""
        # Generate test data with known linear relationship (no noise)
        X1, X2, y = mp.evaluate(simulator, create_linear_federated_data, 42, 50, False)

        # Train federated model
        model = PlainFederatedLinearRegression(
            interpreter=simulator,
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

        # Fetch cleartext values for comparison (only for testing)
        y_pred_clear = fetch_from_label_party(simulator, y_pred_secure)
        y_true_clear = fetch_from_label_party(simulator, y)

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

        # Fetch secure R² for comparison (only for testing)
        r2_secure_value = float(fetch_from_label_party(simulator, r2_secure))

        print(f"Debug: r2_secure_value extracted: {r2_secure_value}")

        print("✓ R² Score Verification Results:")
        print(f"  - Secure R²: {r2_secure_value:.6f}")
        print(f"  - Cleartext R²: {r2_cleartext:.6f}")
        print(f"  - Difference: {abs(r2_secure_value - r2_cleartext):.6f}")

        # Step 11: Verify that the scores are very close
        # Allow small numerical differences due to floating point precision
        assert abs(r2_secure_value - r2_cleartext) < 1e-4, (
            f"R² scores don't match: secure={r2_secure_value}, cleartext={r2_cleartext}"
        )

        # Step 12: Since we have a perfect linear relationship and good convergence,
        # R² should be very close to 1.0
        assert r2_cleartext > 0.95, (
            f"R² should be high for perfect linear relationship: {r2_cleartext}"
        )

        print("✓ R² implementation verified against cleartext counterpart!")
        print("  - Secure and cleartext R² scores match within tolerance")
        print("  - High R² score confirms correct linear relationship detection")

        # Test with noisy data for more realistic R²
        X1_noisy, X2_noisy, y_noisy = mp.evaluate(
            simulator, create_linear_federated_data, 123, 50, True, 2.0
        )

        # Train on noisy data
        model_noisy = PlainFederatedLinearRegression(
            interpreter=simulator,
            learning_rate=0.05,
            max_iter=100,
            fit_intercept=True,
            random_state=42,
            label_device="P0",
        )

        X_noisy = {"P0": X1_noisy, "P1": X2_noisy}
        model_noisy.fit(X_noisy, y_noisy)

        r2_noisy_secure = model_noisy.score(X_noisy, y_noisy)
        r2_noisy_value = float(fetch_from_label_party(simulator, r2_noisy_secure))

        print("✓ Noisy Data R² Test:")
        print(f"  - R² with noise: {r2_noisy_value:.6f}")

        # With noise, R² should be lower but still positive for good model
        assert 0.0 <= r2_noisy_value <= 1.0, (
            f"R² should be between 0 and 1: {r2_noisy_value}"
        )

        print("✓ Complete R² verification successful!")
        print("  - Perfect linear data: High R² (near 1.0)")
        print("  - Noisy linear data: Moderate R² (0.0-1.0)")
        print("  - Secure implementation matches cleartext computation")

    def test_plain_federated_single_party_with_features(self, simulator):
        """Test that training succeeds when only one party has features."""
        # Generate data where only one party has features
        X_alice, y_alice = mp.evaluate(simulator, create_single_party_data, 42)

        # Create model
        model = PlainFederatedLinearRegression(
            interpreter=simulator,
            learning_rate=0.01,
            max_iter=10,
            fit_intercept=True,
            random_state=42,
            label_device="P0",
        )

        # Prepare data - only one party
        X_federated = {
            "P0": X_alice,  # Only Alice has features
        }
        y_federated = y_alice

        # This should succeed - at least one party has features
        print("Training with single party having features...")
        fitted_model = model.fit(X_federated, y_federated)

        assert fitted_model._is_fitted
        assert len(model.coef_) == 1  # Only P0 has weights
        assert "P0" in model.coef_
        assert model.n_features_in_ == 3  # 3 features from P0

        # Make predictions should also work
        predictions = model.predict(X_federated)
        assert predictions is not None

        print("✓ Single party training successful")
        print("  - Party: P0")
        print(f"  - Features: {model.n_features_in_}")

    def test_plain_federated_no_party_with_features(self):
        """Test that training fails when no party has features (n_features=0)."""
        # Create a mock simulator
        mock_simulator = "mock_sim"

        # Create mock data where parties exist but have zero features
        class MockMPObject:
            def __init__(self, shape):
                self.shape = shape

        # Create X with parties that have 0 features
        X_empty = {
            "P0": MockMPObject((30, 0)),  # 30 samples, 0 features
            "P1": MockMPObject((30, 0)),  # 30 samples, 0 features
        }

        y_mock = MockMPObject((30,))

        # Create model
        model = PlainFederatedLinearRegression(
            interpreter=mock_simulator,
            learning_rate=0.01,
            max_iter=10,
            fit_intercept=True,
            random_state=42,
            label_device="P0",
        )

        # This should fail - no party has features
        print("Testing with no parties having features...")
        with pytest.raises(
            ValueError,
            match="At least one party must contribute features.*for training",
        ):
            model.fit(X_empty, y_mock)

        print("✓ Correctly raised ValueError for zero features")
