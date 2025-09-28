# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional, Tuple

import jax.numpy as jnp
import mplang
import mplang.simp as simp
from mplang.core import MPObject

from sfl_lite.ml.linear.linear_model import (
    LinearModel,
    AbstractLinearModel,
    Aggregator,
    MPAggregator,
)


class PlainFedLinearModel(AbstractLinearModel):
    """
    Plain federated linear model implementation using the abstract template.

    This class implements the AbstractLinearModel interface for plain federated learning
    without additional security measures. Not secure for production use.
    """

    def predict(
        self,
        model: LinearModel,
        X: Dict[int, MPObject],
        agg: Optional[Aggregator] = None,
    ) -> MPObject:
        """
        Make predictions using the linear model.

        Args:
            model: LinearModel instance containing weights and configuration
            X: Dict mapping party id to party sample data
            agg: Optional aggregator for combining results

        Returns:
            y_pred: Predicted values as MPObject
        """
        if model.label_party is None:
            raise ValueError("label_party is None, it should be int")
        if agg is None:
            agg = MPAggregator()

        y_pred_party = {
            party_id: simp.runAt(party_id, lambda x, w: x @ w)(
                x, model.weights[party_id]
            )
            for party_id, x in X.items()
        }
        y_pred_no_intercept = simp.revealTo(agg.sum(y_pred_party), model.label_party)

        if model.intercept is not None:
            return simp.runAt(model.label_party, lambda x, b: x + b)(
                y_pred_no_intercept, model.intercept
            )
        return y_pred_no_intercept

    def weight_update(
        self,
        model: LinearModel,
        X: Dict[int, MPObject],
        gradient: MPObject,
        learning_rate: float,
        world_size: int,
    ) -> Tuple[Dict[int, MPObject], Optional[MPObject]]:
        """
        Update model weights and intercept based on computed gradient.

        Args:
            model: LinearModel instance to update
            X: Dict mapping party id to party sample data
            gradient: Computed gradient for weight update
            learning_rate: Learning rate for the update step
            world_size: Total number of parties in the simulation

        Returns:
            Tuple of (updated_weights, updated_intercept)
        """
        # Create world mask based on the provided world_size
        world_mask = mplang.Mask.all(world_size)
        broadcasted_gradient = simp.bcast_m(world_mask, model.label_party, gradient)

        updated_weights = {}
        for party_id, weight in model.weights.items():
            updated_weight = simp.runAt(
                party_id, lambda w, x, g: w - learning_rate * (x.T @ g)
            )(weight, X[party_id], broadcasted_gradient)
            updated_weights[party_id] = updated_weight

        # Update intercept if present
        updated_intercept = None
        if model.intercept is not None and model.label_party is not None:
            updated_intercept = simp.runAt(
                model.label_party, lambda b, g: b - learning_rate * jnp.mean(g)
            )(model.intercept, broadcasted_gradient)

        return updated_weights, updated_intercept
