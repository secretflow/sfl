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
from mplang.core import MPObject

from sfl_lite.ml.linear.linear_model import (
    grad_compute,
    linear_model_predict,
    LinearModel,
    mse_loss,
    RegType,
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
        parties: List[int],
        reg_type: RegType,
        fit_intercept: bool = True,
        learning_rate: float = 0.1,
        seed: int = 42,
    ):
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

    @mplang.function
    def _initialize_model(
        self, X: Dict[int, MPObject], intercept_party: int
    ) -> LinearModel:
        """Initialize model for all parties."""
        weights = {}
        intercept = None
        for party_id, X_party in X.items():
            self.key, subkey = random.split(self.key)
            feature_num = X_party.shape[1]  # Infer feature number from actual data
            weight = simp.runAt(
                party_id,
                lambda: random.uniform(
                    subkey, shape=(feature_num,), minval=-0.1, maxval=0.1
                ),
            )()
            weights[party_id] = weight

        if self.fit_intercept:
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
        return model

    @mplang.function
    def fit(
        self,
        X: Dict[int, MPObject],
        y: MPObject,
        label_party: int,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        tol: float = 1e-4,
    ):
        """
        Fit the vertical linear regression model.

        y device compute the residual r (in y device),
        r and each worker's x will compute loss (in secure device),
        and each worker will update their own weight (in each worker's devce).

        Args:
            X: Dictionary mapping party identifiers to their feature matrices
            y: Target values (held by one party)
            epochs: Number of training epochs
            batch_size: Batch size for training (None for full batch)
            tol: Tolerance for stopping criteria
        """

        # Initialize model parameters for all parties with actual data shape
        initial_model = self._initialize_model(X, label_party)
        
        # Create training state as a dictionary of MPObjects for each party
        epoch = simp.constant(0)
        tol = simp.run(lambda t: jnp.array(t))(tol)
        max_epochs = simp.constant(epochs)
        initial_loss = simp.run(lambda: jnp.array(float('inf')))()
        
        # Create state structure that preserves party associations
        state = {
            'epoch': epoch,
            'loss': initial_loss,
            'X': X
        }
        
        # Add weights for each party
        for party_id, weight in initial_model.weights.items():
            state[f'weight_{party_id}'] = weight
            r = simp.runAt(party_id, lambda w,x : x @ w)(weight, X[party_id])
            print(f'weight_{party_id}:', weight)
            print(f'r_{party_id}:', r)
        
        print(state)
        print(initial_model)
        # Add intercept if present
        if initial_model.intercept is not None:
            state['intercept'] = initial_model.intercept

        def cond(state):
            """Condition function for while loop - check convergence criteria."""
            current_epoch = state['epoch']
            current_loss = state['loss']
            
            # Check if we've reached max epochs
            not_max_epochs = simp.run(lambda e, max_e: e < max_e)(current_epoch, max_epochs)
            
            # Check if loss is above tolerance
            above_tol = simp.run(lambda l, t: l > t)(current_loss, tol)
            
            # Continue if not max epochs AND loss above tolerance
            return simp.run(lambda a, b: a & b)(not_max_epochs, above_tol)

        def body(state):
            """Body function for while loop - perform one epoch of training."""
            # Extract current parameters from state
            current_weights = {}
            X = state['X']
            for party_id in X.keys():
                current_weights[party_id] = state[f'weight_{party_id}']
                r = simp.runAt(party_id, lambda w,x : x @ w)(state[f'weight_{party_id}'], X[party_id])
                print(f'body weight_{party_id}:', state[f'weight_{party_id}'])
                print(f'body r_{party_id}:', r)
            
            current_intercept = state.get('intercept')
            
            # Create model for current iteration
            current_model = LinearModel(
                weights=current_weights,
                reg_type=self.reg_type,
                intercept_party=label_party,
                intercept=current_intercept,
            )
            
            # Compute predictions from all parties
            y_pred_party = {
                party_id: simp.runAt(party_id, lambda x, w: x @ w)(x, state[f'weight_{party_id}'])
                for party_id, x in X.items()
            }
            y_pred = linear_model_predict(current_model, X)

            # Compute loss
            loss = simp.runAt(label_party, mse_loss)(y_pred, y)

            # Compute gradients
            g = grad_compute(y_pred, y, label_party)

            # Update weights and intercept
            # updated_weights, updated_intercept = sync_and_update_weights(
            #     current_model, X, g, self.learning_rate
            # )
            updated_weights, updated_intercept = current_weights,current_intercept 

            # Create new state with updated parameters
            new_state = {
                'epoch': simp.run(lambda e: e + 1)(state['epoch']),
                'loss': loss,
            }
            
            # Update weights for each party
            for party_id, weight in updated_weights.items():
                new_state[f'weight_{party_id}'] = weight
                
            # Update intercept if present
            if updated_intercept is not None:
                new_state['intercept'] = updated_intercept
            
            return new_state

        # Run training loop with simp.while_loop
        final_state = simp.while_loop(cond, body, state)
        
        # Extract final parameters and create model
        final_weights = {}
        for party_id in X.keys():
            final_weights[party_id] = final_state[f'weight_{party_id}']
            
        final_intercept = final_state.get('intercept')
        
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
    
    @mplang.function
    def train():
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
        y = simp.runAt(label_party, lambda: random.normal(random.PRNGKey(44), (n_samples,)))()

        # Create model
        trainer = LinearRegressionVertical(
            parties=[0, 1, 2],
            reg_type=RegType.Linear,
            learning_rate=0.01,
            fit_intercept=True,
        )

        # Fit model
        X = {0: X0,1: X1}
        model =  trainer.fit(X, y, label_party=label_party, epochs=0)

        # Make predictions
        predictions = linear_model_predict(model, X)
        print("Predictions:", predictions)
        return model

    
    result = mplang.evaluate(sim, train)
    print(f"Simulation completed. Final sum: {mplang.fetch(sim, result)}")

    compiled = mplang.compile(sim, train)
    print("compiled:", compiled.compiler_ir())