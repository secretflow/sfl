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


import dataclasses
from enum import Enum, unique
from typing import Dict, Optional

import jax.numpy as jnp
import mplang
import mplang.simp as simp
from jax import grad
from mplang.core import MPObject

from sfl_lite.security.aggregation import Aggregator, MPAggregator


@unique
class RegType(Enum):
    Linear = "linear"
    Logistic = "logistic"


@dataclasses.dataclass
class LinearModel:
    """
    Unified linear regression or logistic regression model.

    Attributes:

        weights : Dict[int, MPObject] Maps from party id to weight.
        reg_type : RegType
        intercept_party : int Party id of intercept.
        intercept : MPObject
    """

    weights: Dict[int, MPObject]
    reg_type: RegType
    intercept_party: Optional[int] = None
    intercept: Optional[MPObject] = None


@mplang.function
def linear_model_predict(
    model: LinearModel, X: Dict[int, MPObject], agg: Optional[Aggregator] = None
) -> MPObject:
    """
    Predict with linear model.

    Args:
        model : LinearModel
        x : Dict[int, MPObject] Maps from party id to party sample data
    Returns:
        y_pred : MPObject
    """
    if model.intercept_party is None:
        raise ValueError("intercept_party is None, it should be int")
    if agg is None:
        agg = MPAggregator()
    y_pred_party = {
        party_id: simp.runAt(party_id, lambda x, w: x @ w)(x, model.weights[party_id])
        for party_id, x in X.items()
    }
    y_pred_no_intercept = simp.revealTo(agg.sum(y_pred_party), model.intercept_party)
    if model.intercept is not None:
        return simp.runAt(model.intercept_party, lambda x, b: x + b)(
            y_pred_no_intercept, model.intercept
        )
    return y_pred_no_intercept


def mse_loss(y_pred, y):
    """
    Vectorized Mean Squared Error loss
    y_pred: predicted values
    y: true target values
    Returns average squared error
    """
    return jnp.mean((y_pred - y) ** 2)


@mplang.function
def grad_compute(y_pred: MPObject, y: MPObject, label_party: int) -> MPObject:
    gradient = simp.runAt(label_party, grad(mse_loss))(y_pred, y)
    return gradient


@mplang.function
def sync_and_update_weights(
    model: LinearModel, X: Dict[int, MPObject], gradient: MPObject, learning_rate: float
):
    """
    Broadcast the gradient to all parties and update their weights and intercept.

    Args:
        model: LinearModel
        gradient: MPObject (computed gradient)
        learning_rate: float
    """
    # Get all unique party IDs involved in the computation
    all_parties = set(model.weights.keys())
    if model.intercept_party is not None:
        all_parties.add(model.intercept_party)
    
    # Create world mask based on actual parties involved
    world_mask = mplang.Mask.all(max(all_parties))
    broadcasted_gradient = simp.bcast_m(world_mask, model.intercept_party, gradient)

    updated_weights = {}
    for party_id, weight in model.weights.items():
        updated_weight = simp.runAt(
            party_id, lambda w, x, g: w - learning_rate * x.T @ g / x.shape[0]
        )(weight, X[party_id], broadcasted_gradient)
        updated_weights[party_id] = updated_weight

    # Update intercept if present
    updated_intercept = None
    if model.intercept is not None and model.intercept_party is not None:
        updated_intercept = simp.runAt(
            model.intercept_party, lambda b, g: b - learning_rate * jnp.mean(g)
        )(model.intercept, broadcasted_gradient)
    return updated_weights, updated_intercept
