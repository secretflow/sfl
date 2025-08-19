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

from sfl_lite.ml.linear.linear_model import LinearModel, RegType, linear_model_predict


class TestLinearModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.sim3 = mplang.Simulator(3)
        mplang.set_ctx(self.sim3)

    def test_linear_model_creation(self):
        """Test basic LinearModel creation."""
        # Create test weights
        weight0 = simp.runAt(0, lambda: jnp.array([1.0, 2.0]))()
        weight1 = simp.runAt(1, lambda: jnp.array([0.5, 1.5]))()
        intercept = simp.runAt(0, lambda: 0.5)()

        # Create model
        model = LinearModel(
            weights={0: weight0, 1: weight1},
            reg_type=RegType.Linear,
            intercept_party=0,
            intercept=intercept
        )

        # Verify model attributes
        assert model.reg_type == RegType.Linear
        assert model.intercept_party == 0
        assert 0 in model.weights
        assert 1 in model.weights

    def test_linear_model_predict_basic(self):
        """Test basic linear model prediction."""
        # Create test data
        weight0 = simp.runAt(0, lambda: jnp.array([1.0, 2.0]))()
        weight1 = simp.runAt(1, lambda: jnp.array([0.5, 1.5]))()
        intercept = simp.runAt(0, lambda: 0.5)()

        # Create model
        model = LinearModel(
            weights={0: weight0, 1: weight1},
            reg_type=RegType.Linear,
            intercept_party=0,
            intercept=intercept
        )

        # Create input data
        X0 = simp.runAt(0, lambda: jnp.array([[1.0, 1.0]]))()
        X1 = simp.runAt(1, lambda: jnp.array([[2.0, 1.0]]))()

        # Perform prediction
        X = {0: X0, 1: X1}
        y_pred = linear_model_predict(model, X)

        # Debug: print intermediate values
        print("DEBUG: y_pred =", y_pred)
        print("DEBUG: y_pred type =", type(y_pred))

        # Get results
        fetched = mplang.fetch(None, y_pred)
        print("DEBUG: fetched =", fetched)

        # Basic verification
        assert fetched is not None
        assert len(fetched) == 3  # 3 parties

        # Check if any party has valid result
        valid_results = [arr for arr in fetched if arr is not None]
        assert len(valid_results) > 0

        # Expected calculation: (1*1 + 1*2) + (2*0.5 + 1*1.5) + 0.5 = 3 + 2.5 + 0.5 = 6.0
        expected = jnp.array([6.0])
        for arr in valid_results:
            if arr is not None:
                print(f"DEBUG: arr = {arr}, expected = {expected}")
                assert jnp.allclose(arr, expected)

    def test_linear_model_predict_single_feature(self):
        """Test linear model prediction with single feature."""
        # Create test data
        weight0 = simp.runAt(0, lambda: jnp.array([2.0]))()
        weight1 = simp.runAt(1, lambda: jnp.array([3.0]))()
        intercept = simp.runAt(0, lambda: 1.0)()

        # Create model
        model = LinearModel(
            weights={0: weight0, 1: weight1},
            reg_type=RegType.Linear,
            intercept_party=0,
            intercept=intercept
        )

        # Create input data
        X0 = simp.runAt(0, lambda: jnp.array([[1.0]]))()
        X1 = simp.runAt(1, lambda: jnp.array([[2.0]]))()

        # Perform prediction
        X = {0: X0, 1: X1}
        y_pred = linear_model_predict(model, X)

        # Debug: print intermediate values
        print("DEBUG: Single feature test")
        print("DEBUG: y_pred =", y_pred)

        # Get results
        fetched = mplang.fetch(None, y_pred)
        print("DEBUG: fetched =", fetched)

        # Basic verification
        assert fetched is not None
        valid_results = [arr for arr in fetched if arr is not None]
        assert len(valid_results) > 0

        # Expected calculation: (1*2) + (2*3) + 1 = 2 + 6 + 1 = 9.0
        expected = jnp.array([9.0])
        for arr in valid_results:
            if arr is not None:
                print(f"DEBUG: arr = {arr}, expected = {expected}")
                assert jnp.allclose(arr, expected)

    def test_linear_model_predict_multiple_samples(self):
        """Test linear model prediction with multiple samples."""
        # Create test data
        weight0 = simp.runAt(0, lambda: jnp.array([1.0, 2.0]))()
        weight1 = simp.runAt(1, lambda: jnp.array([0.5, 1.5]))()
        intercept = simp.runAt(0, lambda: 0.0)()

        # Create model
        model = LinearModel(
            weights={0: weight0, 1: weight1},
            reg_type=RegType.Linear,
            intercept_party=0,
            intercept=intercept
        )

        # Create input data with 2 samples
        X0 = simp.runAt(0, lambda: jnp.array([[1.0, 2.0], [3.0, 4.0]]))()
        X1 = simp.runAt(1, lambda: jnp.array([[0.5, 1.5], [2.5, 3.5]]))()

        # Perform prediction
        X = {0: X0, 1: X1}
        y_pred = linear_model_predict(model, X)

        # Debug: print intermediate values
        print("DEBUG: Multiple samples test")
        print("DEBUG: y_pred =", y_pred)

        # Get results
        fetched = mplang.fetch(None, y_pred)
        print("DEBUG: fetched =", fetched)

        # Basic verification
        assert fetched is not None
        valid_results = [arr for arr in fetched if arr is not None]
        assert len(valid_results) > 0

        # Expected calculations:
        # Sample 1: (1*1 + 2*2) + (0.5*0.5 + 1.5*1.5) = 5 + 2.5 = 7.5
        # Sample 2: (3*1 + 4*2) + (2.5*0.5 + 3.5*1.5) = 11 + 6.5 = 17.5
        expected = jnp.array([7.5, 17.5])
        for arr in valid_results:
            if arr is not None:
                print(f"DEBUG: arr = {arr}, expected = {expected}")
                assert jnp.allclose(arr, expected)

    def test_linear_model_missing_intercept_party(self):
        """Test error handling when intercept_party is None."""
        # Create test data
        weight0 = simp.runAt(0, lambda: jnp.array([1.0]))()
        weight1 = simp.runAt(1, lambda: jnp.array([2.0]))()

        # Create model without intercept_party
        model = LinearModel(
            weights={0: weight0, 1: weight1},
            reg_type=RegType.Linear,
            intercept_party=None
        )

        # Create input data
        X0 = simp.runAt(0, lambda: jnp.array([[1.0]]))()
        X1 = simp.runAt(1, lambda: jnp.array([[1.0]]))()

        # Should raise ValueError
        X = {0: X0, 1: X1}
        with pytest.raises(ValueError, match="intercept_party is None"):
            linear_model_predict(model, X)

    def test_logistic_model_creation(self):
        """Test Logistic regression model creation."""
        # Create test weights
        weight0 = simp.runAt(0, lambda: jnp.array([1.0, -1.0]))()
        weight1 = simp.runAt(1, lambda: jnp.array([0.5, -0.5]))()
        intercept = simp.runAt(0, lambda: 0.0)()

        # Create logistic model
        model = LinearModel(
            weights={0: weight0, 1: weight1},
            reg_type=RegType.Logistic,
            intercept_party=0,
            intercept=intercept
        )

        # Verify model attributes
        assert model.reg_type == RegType.Logistic
