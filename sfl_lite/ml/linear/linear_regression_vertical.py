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

from typing import Dict, List, Optional

import jax.numpy as jnp
import jax.random as random
import mplang
import mplang.simp as simp
from mplang.core.base import MPObject

from sfl_lite.ml.linear.linear_model import LinearModel, RegType, linear_model_predict
from sfl_lite.security.aggregation import MPAggregator


class LinearRegressionVertical:
    """
    Vertical Linear Regression using MPLang for secure multi-party computation.
    This implementation provides a modern, secure approach to vertical federated
    learning using natural language interfaces and homomorphic encryption.
    """

    def __init__(self, parties: List[int], reg_type: RegType, fit_intercept: bool = True, learning_rate: float = 0.1, seed: int = 42):
        """
        Initialize vertical linear regression.

        Args:
            parties: List of party identifiers
            reg_type: Type of regression (linear or logistic)
            fit_intercept: Whether to fit an intercept term, default give to label holder
            learning_rate: Learning rate for gradient descent
            seed: Random seed for reproducibility
        """
        self.parties = parties
        self.reg_type = reg_type
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.key = random.PRNGKey(seed)
        self.model = None

    def _initialize_model(self, X: Dict[int, MPObject], intercept_party: int) -> LinearModel:
        """Initialize model for all parties."""
        weights = {}
        intercept = None
        for party_id, X_party in X.items():
            self.key, subkey = random.split(self.key)
            feature_num = X_party.shape[1]  # Infer feature number from actual data
            weight = simp.runAt(party_id, lambda: random.uniform(subkey, shape=(feature_num,), minval=-0.1, maxval=0.1))()

            if party_id == intercept_party and self.fit_intercept:
                intercept = simp.runAt(intercept_party, lambda: random.uniform(subkey, shape=(), minval=-0.1, maxval=0.1))()

            weights[party_id] = weight

        model = LinearModel(
            weights=weights,
            reg_type=self.reg_type,
            intercept_party=intercept_party ,
            intercept=intercept if self.fit_intercept else None,
        )
        return model

    def fit(
        self,
        X: Dict[int, MPObject],
        y: MPObject,
        label_party: int,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        tol: float = 1e-4
    ):
        """
        Fit the vertical linear regression model.

        Args:
            X: Dictionary mapping party identifiers to their feature matrices
            y: Target values (held by one party)
            epochs: Number of training epochs
            batch_size: Batch size for training (None for full batch)
            tol: Tolerance for stopping criteria
        """

        # Initialize model parameters for all parties with actual data shape
        self.model = self._initialize_model(X, label_party)

        # Import required modules for secure computation
        import mplang.smpc as smpc

        def not_converged(loss_value):
            """Check if loss is still above tolerance."""
            loss_value = jnp.asarray(loss_value)
            return loss_value > tol

        def cond(state):
            """Condition function for while loop - check if not converged."""
            current_epoch, current_loss = state
            
            # Seal the loss value for secure comparison
            sealed_loss = smpc.seal(current_loss)
            # Check if loss is above tolerance
            not_conv = smpc.srun(not_converged)(sealed_loss)
            # Reveal the result for all parties
            return smpc.reveal(not_conv)

        def body(state):
            """Body function for while loop - perform one epoch of training."""
            current_epoch, current_loss = state
            
            # Compute predictions from all parties
            y_pred = linear_model_predict(
                self.model,
                X
            )

            # Compute loss
            new_loss = simp.runAt(label_party, lambda y_true, y_pred: jnp.mean((y_true - y_pred) ** 2))(y, y_pred)

            # Compute residual
            residual = simp.runAt(label_party, lambda y_true, y_pred: y_true - y_pred)(y, y_pred)

            # Update parameters for each party (simplified gradient descent)
            for party_id in self.parties:
                if party_id in X:
                    # Simplified gradient computation
                    gradient = simp.runAt(
                        party_id,
                        lambda X_feat, res: jnp.mean(X_feat * res, axis=0) * self.learning_rate
                    )(X[party_id], residual)

                    # Update weights
                    new_weight = simp.runAt(
                        party_id,
                        lambda w, g: w + g
                    )(self.model.weights[party_id], gradient)

                    self.model.weights[party_id] = new_weight

            # Increment epoch counter
            new_epoch = simp.runAt(label_party, lambda e: e + 1)(current_epoch)
            
            # Print progress every 10 epochs
            _ = simp.runAt(label_party, lambda epoch_num, loss_val: 
                          jnp.where(epoch_num % 10 == 0, 
                                   print(f"Epoch {epoch_num}, Loss: {loss_val}"), 
                                   None))(new_epoch, new_loss)
            
            return (new_epoch, new_loss)

        # Initial loss computation
        initial_y_pred = linear_model_predict(self.model, X)
        initial_epoch = simp.runAt(label_party, lambda: jnp.array(0))()
        initial_loss = simp.runAt(label_party, lambda y_true, y_pred: jnp.mean((y_true - y_pred) ** 2))(y, initial_y_pred)

        # Run training loop with while_loop, but limit by max epochs
        max_epochs = epochs
        
        def limited_cond(state):
            """Modified condition that also checks epoch limit."""
            current_epoch, current_loss = state
            epoch_check = simp.runAt(label_party, lambda e: e < max_epochs)(current_epoch)
            conv_check = cond(state)
            return epoch_check & conv_check

        # Run training loop with while_loop
        final_epoch, final_loss = simp.while_loop(
            limited_cond, 
            body, 
            (initial_epoch, initial_loss)
        )
        
        # Print final convergence message
        _ = simp.runAt(label_party, lambda: print(f"Training completed at epoch {final_epoch}, final loss: {final_loss}"))()

    def get_model(self):
        return self.model


# Example usage
if __name__ == "__main__":
    # Create a simulator with 3 parties
    sim = mplang.Simulator(3)
    mplang.set_ctx(sim)

    # Generate synthetic data
    n_samples = 100
    n_features_party0 = 2
    n_features_party1 = 3

    # Features for party 0
    X0 = simp.runAt(0, lambda: random.normal(random.PRNGKey(42), (n_samples, n_features_party0)))()

    # Features for party 1
    X1 = simp.runAt(1, lambda: random.normal(random.PRNGKey(43), (n_samples, n_features_party1)))()

    # Target variable (held by party 2)
    y = simp.runAt(2, lambda: random.normal(random.PRNGKey(44), (n_samples,)))()

    # Create model
    trainer = LinearRegressionVertical(
        parties=[0, 1, 2],
        reg_type=RegType.Linear,
        learning_rate=0.01,
        fit_intercept=True,
    )

    # Fit model
    X = {0: X0, 1: X1}
    trainer.fit(X, y, label_party=2, epochs=10)

    # Get trained model
    model = trainer.get_model()
    
    # Make predictions
    predictions = linear_model_predict(model, X)
    print("Predictions:", predictions)
