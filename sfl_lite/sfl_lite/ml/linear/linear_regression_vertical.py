# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional, Tuple, Type

import jax.numpy as jnp
import jax.random as random
import mplang
import mplang.simp as simp
from mplang.core import MPObject

from sfl_lite.ml.linear.linear_model import (
    LinearModel,
    AbstractLinearModel,
    RegType,
    loss_and_grad,
)


class LinearRegressionVertical:
    """
    Vertical Linear Regression using MPLang for secure multi-party computation.

    This class is designed to accept any AbstractLinearModel implementation,
    making it flexible to use different federated learning strategies.
    """

    def __init__(
        self,
        linear_model_class: Type[AbstractLinearModel],
        reg_type: RegType,
        fit_intercept: bool = True,
        learning_rate: float = 0.1,
    ):
        """
        Initialize vertical linear regression configuration.

        Args:
            linear_model_class: Class that implements AbstractLinearModel interface
            reg_type: Type of regression (linear or logistic)
            fit_intercept: Whether to fit an intercept term, default give to label holder
            learning_rate: Learning rate for gradient descent
        """
        self.linear_model_class = linear_model_class
        self.reg_type = reg_type
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate

    @staticmethod
    @mplang.function
    def _initialize_model(
        X: Dict[int, MPObject],
        label_party: int,
        key: random.PRNGKey,
        reg_type: RegType,
        fit_intercept: bool,
    ) -> Tuple[LinearModel, random.PRNGKey]:
        """Initialize model for all parties."""
        weights = {}
        intercept = None
        current_key = key

        for party_id, X_party in X.items():
            current_key, subkey = random.split(current_key)
            feature_num = X_party.shape[1]  # Infer feature number from actual data
            weight = simp.runAt(
                party_id,
                lambda: random.uniform(
                    subkey, shape=(feature_num,), minval=-0.1, maxval=0.1
                ),
            )()
            weights[party_id] = weight

        if fit_intercept:
            current_key, subkey = random.split(current_key)
            intercept = simp.runAt(
                label_party,
                lambda: random.uniform(subkey, shape=(), minval=-0.1, maxval=0.1),
            )()
        model = LinearModel(
            weights=weights,
            reg_type=reg_type,
            label_party=label_party,
            intercept=intercept if fit_intercept else None,
        )
        return model, current_key

    @mplang.function
    def fit(
        self,
        X: Dict[int, MPObject],
        y: MPObject,
        label_party: int,
        world_size: int,
        key: Optional[random.PRNGKey] = None,
        epochs: int = 100,
        tol: float = 1e-4,
    ) -> Tuple[Dict, random.PRNGKey]:
        """
        Fit the vertical linear regression model using the provided linear model class.

        Args:
            X: Dictionary mapping party identifiers to their feature matrices
            y: Target values (held by one party)
            label_party: Party ID that holds the labels
            world_size: Total number of parties in the simulation
            key: PRNG key for random number generation
            epochs: Number of training epochs
            tol: Tolerance for stopping criteria

        Returns:
            Tuple of (final_state, updated_key)
        """
        # Use provided key or generate from default
        if key is None:
            key = random.PRNGKey(42)

        # Initialize model parameters
        initial_model, updated_key = self._initialize_model(
            X, label_party, key, self.reg_type, self.fit_intercept
        )

        # Create linear model instance
        linear_model = self.linear_model_class()

        # Create training state
        epoch = simp.constant(0)
        tol = simp.constant(tol)
        max_epochs = simp.constant(epochs)
        initial_loss = simp.runAt(label_party, lambda: jnp.array(float("inf")))()

        state = {
            "epoch": epoch,
            "loss": initial_loss,
        }

        # Add weights for each party
        for party_id, weight in initial_model.weights.items():
            state[f"weight_{party_id}"] = weight

        # Add intercept if present
        if initial_model.intercept is not None:
            state["intercept"] = initial_model.intercept

        # Broadcast gradient to all parties
        world_mask = mplang.Mask.all(world_size)

        def cond(state):
            """Condition function for while loop."""
            current_epoch = state["epoch"]
            current_loss = state["loss"]

            # Check if we've reached max epochs
            not_max_epochs = simp.run(lambda e, max_e: e < max_e)(
                current_epoch, max_epochs
            )

            # Check if loss is above tolerance
            above_tol = simp.runAt(
                label_party, lambda loss, threshold: loss > threshold
            )(current_loss, tol)
            above_tol = simp.bcast_m(world_mask, label_party, above_tol)

            return simp.run(lambda a, b: jnp.logical_and(a, b))(
                not_max_epochs, above_tol
            )

        def body(state):
            """Body function for while loop - perform one epoch of training."""
            # Extract current parameters from state
            current_weights = {}
            for party_id in X.keys():
                current_weights[party_id] = state[f"weight_{party_id}"]
            current_intercept = state.get("intercept")

            # Create model for current iteration
            current_model = LinearModel(
                weights=current_weights,
                reg_type=self.reg_type,
                label_party=label_party,
                intercept=current_intercept,
            )

            # Use the provided linear model for prediction
            y_pred = linear_model.predict(current_model, X)

            # Compute loss gradients
            loss, g = loss_and_grad(y_pred, y, label_party)

            # Use the provided linear model for weight update
            updated_weights, updated_intercept = linear_model.weight_update(
                current_model, X, g, self.learning_rate, world_size
            )

            # Create new state with updated parameters
            new_state = {
                "epoch": simp.run(lambda e: e + 1)(state["epoch"]),
                "loss": loss,
            }

            # Update weights for each party
            for party_id, weight in updated_weights.items():
                new_state[f"weight_{party_id}"] = weight

            # Update intercept if present
            if updated_intercept is not None:
                new_state["intercept"] = updated_intercept

            return new_state

        # Run training loop
        final_state = simp.while_loop(cond, body, state)

        return final_state, updated_key

    @staticmethod
    def state_to_model(state: Dict, label_party: int, reg_type: RegType) -> LinearModel:
        """
        Convert training state to LinearModel instance.
        """
        final_weights = {}
        for key in state:
            if key.startswith("weight_"):
                party_id = int(key.split("_")[1])
                final_weights[party_id] = state[key]
        final_intercept = state.get("intercept")

        return LinearModel(
            weights=final_weights,
            reg_type=reg_type,
            label_party=label_party,
            intercept=final_intercept,
        )


# Example usage
if __name__ == "__main__":
    # Create a simulator with 3 parties
    sim = mplang.Simulator.simple(3)
    mplang.set_ctx(sim)

    # Generate synthetic data
    n_samples = 100
    n_features_party0 = 2
    n_features_party1 = 3
    label_party = 0

    # Features for party 0
    X0 = simp.runAt(
        0, lambda: random.normal(random.PRNGKey(42), (n_samples, n_features_party0))
    )()

    # Features for party 1
    X1 = simp.runAt(
        1, lambda: random.normal(random.PRNGKey(43), (n_samples, n_features_party1))
    )()

    # Target variable (held by party 0)
    y = simp.runAt(
        label_party, lambda: random.normal(random.PRNGKey(44), (n_samples,))
    )()

    # 使用新的抽象接口
    from sfl_lite.ml.linear.plain_fed_linear_model import PlainFedLinearModel

    trainer = LinearRegressionVertical(
        linear_model_class=PlainFedLinearModel,
        reg_type=RegType.Linear,
        learning_rate=0.01,
        fit_intercept=True,
    )

    X = {0: X0, 1: X1}
    state, updated_key = mplang.evaluate(
        sim, lambda: trainer.fit(X, y, label_party=label_party, world_size=3, epochs=1)
    )

    model = LinearRegressionVertical.state_to_model(
        state, label_party=label_party, reg_type=RegType.Linear
    )
    print(model)
