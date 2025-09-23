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

from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import jax.random as random
import mplang
import mplang.simp as simp
from mplang.core import MPObject

from sfl_lite.ml.linear.linear_model import (
    LinearModel,
    RegType,
    grad_compute,
    linear_model_predict,
    mse_loss,
    sync_and_update_weights,
)


class LinearRegressionVertical:
    """
    Vertical Linear Regression using MPLang for secure multi-party computation.
    This implementation provides a modern, secure approach to vertical federated
    learning using natural language interfaces and homomorphic encryption.
    """

    def __init__(
        self,
        reg_type: RegType,
        fit_intercept: bool = True,
        learning_rate: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize vertical linear regression.

        Args:
            reg_type: Type of regression (linear or logistic)
            fit_intercept: Whether to fit an intercept term, default give to label holder
            learning_rate: Learning rate for gradient descent
            seed: Random seed for reproducibility
        """
        self.reg_type = reg_type
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.key = random.PRNGKey(seed)
        self.model = None

    @mplang.function
    def _initialize_model(
        self, X: Dict[int, MPObject], intercept_party: int, key: random.PRNGKey
    ) -> Tuple[LinearModel, random.PRNGKey]:
        """Initialize model for all parties.

        Args:
            X: Dictionary mapping party identifiers to their feature matrices
            intercept_party: Party ID that holds the intercept
            key: PRNG key for random number generation

        Returns:
            Tuple of (model, updated_key) where updated_key is the new PRNG key
        """
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

        if self.fit_intercept:
            current_key, subkey = random.split(current_key)
            intercept = simp.runAt(
                intercept_party,
                lambda: random.uniform(subkey, shape=(), minval=-0.1, maxval=0.1),
            )()
        model = LinearModel(
            weights=weights,
            reg_type=self.reg_type,
            intercept_party=intercept_party,
            intercept=intercept if self.fit_intercept else None,
        )
        return model, current_key

    @mplang.function
    def fit(
        self,
        X: Dict[int, MPObject],
        y: MPObject,
        label_party: int,
        epochs: int = 100,
        tol: float = 1e-4,
        # previously I did try to infer from current parties,
        # but when the network has n parties and the input data X and y only have <= n - 1 parties,
        # deadlock bug will occur when broadcast in conditioning
        # So for now, we use all parties to broadcast gradient,
        # remove this parameter later
        # this paramter is optional for now in order to keep the API consistent
        # (many cases don't need to specify this)
        world_size: Optional[int] = None,
    ):
        """
        Fit the vertical linear regression model.

        The party holding the label (`y`) computes predictions and gradients.
        The gradients are then used by each worker to update their model weights on their respective device.

        Args:
            X: Dictionary mapping party identifiers to their feature matrices
            y: Target values (held by one party)
            label_party: Party ID that holds the labels
            epochs: Number of training epochs
            tol: Tolerance for stopping criteria
            world_size: Total number of parties in the simulation (required for broadcasting)
        """

        # Initialize model parameters for all parties with actual data shape
        initial_model, updated_key = self._initialize_model(X, label_party, self.key)
        # Note: In the functional paradigm, we would return the updated key,
        # but since fit() is not expected to be pure, we update self.key here
        self.key = updated_key

        # Create training state as a dictionary of MPObjects for each party
        epoch = simp.constant(0)
        tol = simp.constant(tol)
        max_epochs = simp.constant(epochs)
        initial_loss = simp.runAt(label_party, lambda: jnp.array(float("inf")))()

        # Create state structure that preserves party associations
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
        if world_size is None:
            world_size = max(max(X.keys()), label_party) + 1
        world_mask = mplang.Mask.all(world_size)

        def cond(state):
            """Condition function for while loop - check convergence criteria."""
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
            # Continue if not max epochs AND loss above tolerance
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
                intercept_party=label_party,
                intercept=current_intercept,
            )

            # Compute predictions from all parties
            y_pred = linear_model_predict(current_model, X)

            # Compute loss
            loss = simp.runAt(label_party, mse_loss)(y_pred, y)

            # Compute gradients
            g = grad_compute(y_pred, y, label_party)

            # Update weights and intercept
            updated_weights, updated_intercept = sync_and_update_weights(
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

        # Run training loop with simp.while_loop
        final_state = simp.while_loop(cond, body, state)

        return final_state

    def state_to_model(self, state: Dict, label_party: int):
        # Extract final parameters and create model
        final_weights = {}
        for key in state:
            if key.startswith("weight_"):
                party_id = int(key.split("_")[1])
                final_weights[party_id] = state[key]
        final_intercept = state.get("intercept")

        self.model = LinearModel(
            weights=final_weights,
            reg_type=self.reg_type,
            intercept_party=label_party,
            intercept=final_intercept,
        )
        return self.get_model()

    def get_model(self):
        return self.model


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

    # Target variable (held by party 2)
    y = simp.runAt(
        label_party, lambda: random.normal(random.PRNGKey(44), (n_samples,))
    )()

    # Create model
    trainer = LinearRegressionVertical(
        reg_type=RegType.Linear,
        learning_rate=0.01,
        fit_intercept=True,
    )

    # Fit model
    X = {0: X0, 1: X1}
    state = mplang.evaluate(
        sim, lambda: trainer.fit(X, y, label_party=label_party, world_size=3, epochs=1)
    )
    model = trainer.state_to_model(state, label_party=label_party)
    print(model)
    print(model.weights[0].mptype)
    print(model.weights[1].mptype)
    print(model.intercept.mptype)
    w0 = mplang.fetch(sim, simp.runAt(0, lambda x: x)(model.weights[0]))
    w1 = mplang.fetch(sim, simp.runAt(1, lambda x: x)(model.weights[1]))
    b = mplang.fetch(sim, simp.runAt(0, lambda x: x)(model.intercept))
    print("Learned party0_weight (first 5):", w0[:5])
    print("Learned party1_weight (first 5):", w1[:5])
    print("Learned bias:", b)
