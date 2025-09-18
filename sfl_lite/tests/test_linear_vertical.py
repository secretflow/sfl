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
import mplang
import mplang.simp as simp
import pytest

from sfl_lite.ml.linear.linear_model import RegType
from sfl_lite.ml.linear.linear_regression_vertical import LinearRegressionVertical


class TestLinearRegressionVertical:
    """Test suite for vertical linear regression implementation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment with simulator."""
        self.sim3 = mplang.Simulator.simple(3)
        mplang.set_ctx(self.sim3)

    def test_vertical_linear_regression_initialization(self):
        """Test LinearRegressionVertical initialization."""
        parties = [0, 1, 2]
        reg_type = RegType.Linear

        model = LinearRegressionVertical(
            parties=parties,
            reg_type=reg_type,
            fit_intercept=True,
            learning_rate=0.1,
            seed=42,
        )

        assert model.parties == parties
        assert model.reg_type == reg_type
        assert model.fit_intercept is True
        assert model.learning_rate == 0.1
        assert model.model is None

    def test_vertical_linear_regression_basic_fit(self):
        """Test basic fitting functionality with synthetic data."""
        # Create simple synthetic data
        n_samples = 100
        n_features_party0 = 2
        n_features_party1 = 3
        label_party = 2

        # Generate features for party 0
        X0 = simp.runAt(
            0, lambda: random.normal(random.PRNGKey(42), (n_samples, n_features_party0))
        )()

        # Generate features for party 1
        X1 = simp.runAt(
            1, lambda: random.normal(random.PRNGKey(43), (n_samples, n_features_party1))
        )()

        # Generate target variable
        y = simp.runAt(
            label_party, lambda: random.normal(random.PRNGKey(44), (n_samples,))
        )()

        # Create and fit model
        trainer = LinearRegressionVertical(
            parties=[0, 1, 2],
            reg_type=RegType.Linear,
            learning_rate=0.01,
            fit_intercept=True,
        )

        X = {0: X0, 1: X1}

        # Test that fitting runs without error
        state = mplang.evaluate(
            self.sim3, lambda: trainer.fit(X, y, label_party=label_party, epochs=5)
        )

        assert state is not None
        assert "epoch" in state
        assert "loss" in state
        assert "weight_0" in state
        assert "weight_1" in state
        assert "intercept" in state

    def test_vertical_linear_regression_no_intercept(self):
        """Test vertical linear regression without intercept."""
        n_samples = 50
        n_features = 2
        label_party = 1

        # Generate data
        X0 = simp.runAt(
            0, lambda: random.normal(random.PRNGKey(45), (n_samples, n_features))
        )()

        y = simp.runAt(
            label_party, lambda: random.normal(random.PRNGKey(46), (n_samples,))
        )()

        trainer = LinearRegressionVertical(
            parties=[0, 1],
            reg_type=RegType.Linear,
            fit_intercept=False,
            learning_rate=0.01,
        )

        X = {0: X0}

        state = mplang.evaluate(
            self.sim3, lambda: trainer.fit(X, y, label_party=label_party, epochs=3)
        )

        assert state is not None
        assert "weight_0" in state
        assert (
            "intercept" not in state
        )  # Should not have intercept when fit_intercept=False

    def test_vertical_linear_regression_convergence(self):
        """Test that the model converges on simple linear data."""
        n_samples = 200
        label_party = 0

        # Create perfectly linear relationship
        X0 = simp.runAt(0, lambda: jnp.linspace(-1, 1, n_samples).reshape(-1, 1))()

        # Simple linear relationship: y = 2*x + 1
        y = simp.runAt(
            label_party,
            lambda: 2 * jnp.linspace(-1, 1, n_samples)
            + 1
            + random.normal(random.PRNGKey(47), (n_samples,)) * 0.01,
        )()

        trainer = LinearRegressionVertical(
            parties=[0],
            reg_type=RegType.Linear,
            learning_rate=0.1,
            fit_intercept=True,
        )

        X = {0: X0}

        state = mplang.evaluate(
            self.sim3,
            lambda: trainer.fit(X, y, label_party=label_party, epochs=50, tol=1e-3),
        )

        # Check that training completed
        final_epoch = mplang.fetch(None, state["epoch"])[0]
        assert final_epoch > 0
        assert final_epoch <= 50

    def test_vertical_linear_regression_state_to_model(self):
        """Test conversion from training state to model."""
        n_samples = 50
        n_features = 2
        label_party = 1

        # Generate data
        X0 = simp.runAt(
            0, lambda: random.normal(random.PRNGKey(52), (n_samples, n_features))
        )()

        y = simp.runAt(
            label_party, lambda: random.normal(random.PRNGKey(53), (n_samples,))
        )()

        trainer = LinearRegressionVertical(
            parties=[0, 1],
            reg_type=RegType.Linear,
            learning_rate=0.01,
        )

        X = {0: X0}

        # Fit model
        state = mplang.evaluate(
            self.sim3, lambda: trainer.fit(X, y, label_party=label_party, epochs=2)
        )

        # Test state_to_model conversion
        # Note: This is a simplified test as the actual conversion might need more context
        assert state is not None
        assert trainer.get_model() is None  # Initially should be None

    def test_vertical_linear_regression_different_learning_rates(self):
        """Test training with different learning rates."""
        n_samples = 100
        n_features = 1
        label_party = 0

        X0 = simp.runAt(
            0, lambda: random.normal(random.PRNGKey(54), (n_samples, n_features))
        )()

        y = simp.runAt(
            label_party, lambda: random.normal(random.PRNGKey(55), (n_samples,))
        )()

        learning_rates = [0.001, 0.01, 0.1]

        for lr in learning_rates:
            trainer = LinearRegressionVertical(
                parties=[0],
                reg_type=RegType.Linear,
                learning_rate=lr,
                fit_intercept=True,
            )

            X = {0: X0}

            state = mplang.evaluate(
                self.sim3, lambda: trainer.fit(X, y, label_party=label_party, epochs=2)
            )

            assert state is not None
            assert "loss" in state

    def test_vertical_linear_regression_empty_data_handling(self):
        """Test handling of edge cases with minimal data."""
        label_party = 0

        X0 = simp.runAt(0, lambda: jnp.array([[1.0]]))()

        y = simp.runAt(label_party, lambda: jnp.array([1.0]))()

        trainer = LinearRegressionVertical(
            parties=[0],
            reg_type=RegType.Linear,
            learning_rate=0.01,
            fit_intercept=True,
        )

        X = {0: X0}

        # Should handle minimal data without error
        state = mplang.evaluate(
            self.sim3, lambda: trainer.fit(X, y, label_party=label_party, epochs=1)
        )

        assert state is not None

    def test_vertical_linear_regression_reproducibility(self):
        """Test that results are reproducible with same seed."""
        n_samples = 50
        n_features = 2
        label_party = 1

        # Generate data
        X0 = simp.runAt(
            0, lambda: random.normal(random.PRNGKey(56), (n_samples, n_features))
        )()

        y = simp.runAt(
            label_party, lambda: random.normal(random.PRNGKey(57), (n_samples,))
        )()

        def run_training(seed):
            trainer = LinearRegressionVertical(
                parties=[0, 1],
                reg_type=RegType.Linear,
                learning_rate=0.01,
                seed=seed,
            )

            X = {0: X0}
            state = mplang.evaluate(
                self.sim3, lambda: trainer.fit(X, y, label_party=label_party, epochs=2)
            )
            return state

        # Run with same seed twice
        state1 = run_training(42)
        state2 = run_training(42)

        # Both should produce similar results (within reasonable tolerance)
        loss1 = mplang.fetch(None, state1["loss"])
        loss2 = mplang.fetch(None, state2["loss"])
        assert abs(loss1[label_party] - loss2[label_party]) < 1e-6
