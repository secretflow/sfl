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

from typing import Any, Dict, Optional

import jax.numpy as jnp
import jax.random as random
import mplang.v1 as mp
from mplang.v1.core import MPObject

from sfl_lite.ml.linear_model.linear_regression.lib.functional import (
    loss_and_grad,
    r2_score,
)
from sfl_lite.ml.linear_model.linear_regression.lib.model import LinearRegressionModel
from sfl_lite.ml.linear_model.linear_regression.lib.template import (
    LinearRegressionTemplate,
)


# Pure computational functions for heavy lifting
@mp.function
def initialize_weights(
    feature_shapes: Dict[str, int],
    label_device: str,
    fit_intercept: bool,
    random_seed: int = 42,
) -> Dict[str, MPObject]:
    """
    Pure function to initialize model weights across devices.

    Args:
        feature_shapes: Dict mapping device names to number of features
        label_device: Device that holds labels and intercept
        fit_intercept: Whether to initialize intercept
        random_seed: Random seed for initialization

    Returns:
        Dict containing initialized weights for each device
    """
    key = random.PRNGKey(random_seed)
    weights = {}

    for device_name, n_features in feature_shapes.items():
        key, subkey = random.split(key)
        # Initialize weights with small random values
        device_weights = mp.device(device_name)(
            lambda k, n: random.normal(k, (n,)) * 0.01
        )(subkey, n_features)
        weights[device_name] = device_weights

    # Initialize intercept on label device if needed
    if fit_intercept:
        key, subkey = random.split(key)
        weights["intercept"] = mp.device(label_device)(
            lambda k: random.normal(k, ()) * 0.01
        )(subkey)

    return weights


@mp.function
def compute_predictions(
    X: Dict[str, MPObject],
    weights: Dict[str, MPObject],
    label_device: str,
    fit_intercept: bool,
) -> MPObject:
    """
    Pure function to compute predictions from features and weights.

    Args:
        X: Feature data on each device
        weights: Model weights on each device
        label_device: Device to aggregate predictions
        fit_intercept: Whether model has intercept term

    Returns:
        Aggregated predictions on label device
    """
    # Compute local predictions on each device
    local_predictions = {}
    for device_name, features in X.items():
        if device_name in weights:
            local_pred = mp.device(device_name)(lambda x, w: jnp.dot(x, w))(
                features, weights[device_name]
            )
            local_predictions[device_name] = local_pred

    # Aggregate predictions on label device - parallel transfer and sum
    # Transfer all predictions to label device in parallel
    preds_on_label_device = [
        mp.device(label_device)(lambda x: x)(pred)
        for pred in local_predictions.values()
    ]
    # Sum all predictions in a single operation
    prediction_sum = mp.device(label_device)(lambda p: jnp.sum(jnp.stack(p), axis=0))(
        preds_on_label_device
    )

    # Add intercept if present
    if fit_intercept and "intercept" in weights:
        prediction_sum = mp.device(label_device)(
            lambda pred, intercept: pred + intercept
        )(prediction_sum, weights["intercept"])

    return prediction_sum


@mp.function
def update_weights_step(
    X: Dict[str, MPObject],
    weights: Dict[str, MPObject],
    gradient: MPObject,
    learning_rate: float,
    label_device: str,
    fit_intercept: bool,
) -> Dict[str, MPObject]:
    """
    Pure function to update weights using gradient descent.

    Args:
        X: Feature data on each device
        weights: Current model weights
        gradient: Computed gradient on label device
        learning_rate: Learning rate for update
        label_device: Device holding labels and gradient
        fit_intercept: Whether to update intercept

    Returns:
        Updated weights dict
    """
    updated_weights = {}

    # Update weights for each device
    for device_name, features in X.items():
        if device_name in weights:
            # Transfer gradient to device
            device_gradient = mp.device(device_name)(lambda g: g)(gradient)

            # Compute local weight gradient: X^T @ gradient
            weight_gradient = mp.device(device_name)(
                lambda x, g: jnp.dot(x.T, g) / len(g)
            )(features, device_gradient)

            # Update weights: w = w - lr * grad
            updated_weights[device_name] = mp.device(device_name)(
                lambda w, g, lr: w - lr * g
            )(weights[device_name], weight_gradient, learning_rate)

    # Update intercept if present
    if fit_intercept and "intercept" in weights:
        intercept_gradient = mp.device(label_device)(lambda g: jnp.mean(g))(gradient)

        updated_weights["intercept"] = mp.device(label_device)(
            lambda b, g, lr: b - lr * g
        )(weights["intercept"], intercept_gradient, learning_rate)

    return updated_weights


class PlainFederatedLinearRegression(LinearRegressionTemplate):
    """
    Plain federated linear regression with sklearn-compatible interface.

    This implementation uses device-based computation and follows the new template
    structure. It separates pure computational functions for better modularity.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.

    learning_rate : float, default=0.01
        Learning rate for gradient descent optimization.

    max_iter : int, default=1000
        Maximum number of iterations for training.

    tol : float, default=1e-6
        Tolerance for stopping criterion.

    random_state : int, optional
        Random seed for weight initialization.

    label_device : str, default="party_0"
        Device name that holds the labels.

    interpreter : required
        MPLang interpreter context (e.g., simulator created by mp.Simulator).
        Used to execute @mp.function decorated functions. This is required for
        all federated computations.
    """

    def __init__(
        self,
        interpreter,
        fit_intercept: bool = True,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
        label_device: str = "party_0",
    ):
        if interpreter is None:
            raise ValueError(
                "interpreter is required and must be provided (e.g., mp.Simulator)"
            )

        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state if random_state is not None else 42
        self.label_device = label_device
        self.interpreter = interpreter

        # Model state
        self._model = None
        self._is_fitted = False
        self._n_features_in = None

    def fit(
        self,
        X: Dict[str, MPObject],
        y: MPObject,
        sample_weight: Optional[MPObject] = None,
        **kwargs,
    ) -> "PlainFederatedLinearRegression":
        """
        Fit the federated linear regression model.

        Args:
            X: Dict mapping device names to their feature data as MPObjects
            y: Target values as MPObject on label device
            sample_weight: Optional sample weights (not implemented)
            **kwargs: Additional arguments

        Returns:
            self: Fitted model instance
        """
        if sample_weight is not None:
            raise NotImplementedError("sample_weight is not yet supported")

        # Validate input data
        self._validate_input(X)

        # Initialize model
        feature_shapes = {}
        for device_name, features in X.items():
            # Use robust feature shape extraction (same logic as _validate_input)
            n_features = (
                features.shape[1]
                if hasattr(features, "shape") and len(features.shape) > 1
                else 1
            )
            feature_shapes[device_name] = n_features

        # Validate that at least one party contributes features
        if not feature_shapes or all(n == 0 for n in feature_shapes.values()):
            raise ValueError(
                "At least one party must contribute features (n_features >= 1) for training"
            )

        # Initialize weights using pure function with interpreter
        weights = mp.evaluate(
            self.interpreter,
            initialize_weights,
            feature_shapes,
            self.label_device,
            self.fit_intercept,
            self.random_state,
        )

        # Create initial model
        model_weights = {k: v for k, v in weights.items() if k != "intercept"}
        intercept = weights.get("intercept") if self.fit_intercept else None

        self._model = LinearRegressionModel(
            weights=model_weights, label_party=self.label_device, intercept=intercept
        )  # Create secure training function that uses mp.while_loop

        @mp.function
        def secure_training_loop():
            # Initialize training state as MPObjects
            import jax.numpy as jnp

            # Create control variables as MPObjects on label device
            epoch = mp.device(self.label_device)(lambda: jnp.array(0))()
            tol_val = mp.device(self.label_device)(lambda: jnp.array(self.tol))()
            max_epochs = mp.device(self.label_device)(
                lambda: jnp.array(self.max_iter)
            )()
            initial_loss = mp.device(self.label_device)(
                lambda: jnp.array(float("inf"))
            )()

            state = {
                "epoch": epoch,
                "loss": initial_loss,
                "tol_val": tol_val,
                "max_epochs": max_epochs,
                "weights": weights,
            }

            def cond(state):
                """Condition function for while loop."""
                current_epoch = state["epoch"]
                current_loss = state["loss"]
                max_epochs = state["max_epochs"]
                tol_val = state["tol_val"]

                # Seal values to make them available for secure computation
                sealed_epoch = mp.seal(current_epoch)
                sealed_loss = mp.seal(current_loss)
                sealed_max = mp.seal(max_epochs)
                sealed_tol = mp.seal(tol_val)

                # Perform condition check in secure computation
                def check_continue_condition(epoch, loss, max_ep, tol):
                    not_max_epochs = epoch < max_ep
                    above_tol = loss > tol
                    return jnp.logical_and(not_max_epochs, above_tol)

                # Run secure computation and reveal result so all parties can see it
                condition = mp.srun_jax(
                    check_continue_condition,
                    sealed_epoch,
                    sealed_loss,
                    sealed_max,
                    sealed_tol,
                )
                return mp.reveal(condition)

            def body(state):
                """Body function for while loop - perform one training step."""
                current_weights = state["weights"]

                # Compute predictions
                y_pred = compute_predictions(
                    X, current_weights, self.label_device, self.fit_intercept
                )

                # Compute loss and gradients
                loss, gradient = loss_and_grad(y_pred, y, self.label_device)

                # Update weights
                updated_weights = update_weights_step(
                    X,
                    current_weights,
                    gradient,
                    self.learning_rate,
                    self.label_device,
                    self.fit_intercept,
                )

                # Create new state - update epoch using mp.run_jax
                new_epoch = mp.run_jax(lambda e: e + 1, state["epoch"])

                new_state = {
                    "epoch": new_epoch,
                    "loss": loss,
                    "tol_val": state["tol_val"],  # Keep existing values
                    "max_epochs": state["max_epochs"],
                    "weights": updated_weights,
                }

                return new_state

            # Run secure training loop
            final_state = mp.while_loop(cond, body, state)
            return final_state["weights"]

        # Execute secure training using interpreter
        final_weights = mp.evaluate(self.interpreter, secure_training_loop)

        # Update model state (no cleartext access - just storing MPObjects)
        model_weights = {k: v for k, v in final_weights.items() if k != "intercept"}
        intercept = final_weights.get("intercept") if self.fit_intercept else None

        self._model = LinearRegressionModel(
            weights=model_weights, label_party=self.label_device, intercept=intercept
        )
        self._is_fitted = True

        return self

    def predict(self, X: Dict[str, MPObject], **kwargs) -> MPObject:
        """
        Make predictions using the fitted model.

        Args:
            X: Dict mapping device names to their feature data as MPObjects
            **kwargs: Additional arguments

        Returns:
            y_pred: Predicted values as MPObject
        """
        if not self._is_fitted:
            raise ValueError("This model is not fitted yet.")

        # Validate input (but don't update _n_features_in since we're already fitted)
        if not isinstance(X, dict) or len(X) == 0:
            raise ValueError(
                "X must be a non-empty dictionary mapping party names to MPObjects"
            )

        # Prepare weights dict for prediction
        weights = dict(self._model.weights)
        if self.fit_intercept and self._model.intercept is not None:
            weights["intercept"] = self._model.intercept

        # Use interpreter for predictions
        return mp.evaluate(
            self.interpreter,
            compute_predictions,
            X,
            weights,
            self.label_device,
            self.fit_intercept,
        )

    def score(
        self,
        X: Dict[str, MPObject],
        y: MPObject,
        sample_weight: Optional[MPObject] = None,
        **kwargs,
    ) -> MPObject:
        """
        Return the coefficient of determination R^2 of the prediction.

        Args:
            X: Feature data for scoring
            y: True target values
            sample_weight: Optional sample weights (not implemented)
            **kwargs: Additional arguments

        Returns:
            score: R^2 score as MPObject (1.0 is perfect prediction, 0.0 means no predictive power)
        """
        if sample_weight is not None:
            raise NotImplementedError("sample_weight is not yet supported")

        y_pred = self.predict(X)
        # Use interpreter for R² computation - return the secure MPObject
        r2 = mp.evaluate(self.interpreter, r2_score, y_pred, y)

        # Return the secure R² as MPObject (no cleartext fetching)
        # Note: In a real federated setting, this would stay secure
        # For sklearn compatibility, some applications might expect a float,
        # but we maintain security by returning MPObject
        return r2

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "fit_intercept": self.fit_intercept,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "label_device": self.label_device,
            "interpreter": self.interpreter,
        }

    def set_params(self, **params) -> "PlainFederatedLinearRegression":
        """Set parameters for this estimator."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter {param}")
        return self

    @property
    def coef_(self) -> Dict[str, MPObject]:
        """Coefficients of the linear model distributed across devices."""
        if not self._is_fitted:
            raise ValueError("This model is not fitted yet.")
        return self._model.weights

    @property
    def intercept_(self) -> Optional[MPObject]:
        """Intercept (bias) term of the linear model."""
        if not self._is_fitted:
            raise ValueError("This model is not fitted yet.")
        return self._model.intercept

    @property
    def n_features_in_(self) -> int:
        """Number of features seen during fit."""
        if not self._is_fitted:
            raise ValueError("This model is not fitted yet.")
        return self._n_features_in

    def _validate_input(self, X: Dict[str, MPObject]):
        """
        Validate input data format and consistency.

        Args:
            X: Feature data from all parties

        Raises:
            ValueError: If input format is invalid
        """
        if not isinstance(X, dict):
            raise ValueError("X must be a dictionary mapping party names to MPObjects")

        if len(X) == 0:
            raise ValueError("X cannot be empty")

        # Validate that all values are MPObject-like (have shape attribute)
        for party_name, features in X.items():
            if not hasattr(features, "shape"):
                raise ValueError(
                    f"Features for party '{party_name}' must have a shape attribute"
                )

        # Store number of features during validation if not fitted
        if not self._is_fitted:
            total_features = sum(
                data.shape[1] if hasattr(data, "shape") and len(data.shape) > 1 else 1
                for data in X.values()
            )
            # Store for later use in n_features_in_ property
            self._n_features_in = total_features


# Factory function for convenient model creation
def create_plain_federated_lr(
    interpreter,
    fit_intercept: bool = True,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
    random_state: Optional[int] = None,
    label_device: str = "party_0",
) -> PlainFederatedLinearRegression:
    """
    Factory function to create a plain federated linear regression model.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    learning_rate : float, default=0.01
        Learning rate for optimization.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-6
        Tolerance for convergence.
    random_state : int, optional
        Random seed for reproducibility.
    label_device : str, default="party_0"
        Device name that holds the labels.
    interpreter : required
        MPLang interpreter context (e.g., simulator created by mp.Simulator).
        This is required for executing all federated computations.

    Returns
    -------
    model : PlainFederatedLinearRegression
        Configured federated linear regression model.

    Examples
    --------
    >>> sim = mp.Simulator(cluster_spec)
    >>> model = create_plain_federated_lr(sim, learning_rate=0.05)
    >>> model.fit(X_federated, y, label_device="alice")
    """
    return PlainFederatedLinearRegression(
        interpreter=interpreter,
        fit_intercept=fit_intercept,
        learning_rate=learning_rate,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        label_device=label_device,
    )
