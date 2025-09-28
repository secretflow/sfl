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

import abc
import dataclasses
from enum import Enum, unique
from typing import Dict, Optional, Tuple

from jax import grad, value_and_grad
import jax.numpy as jnp
import mplang
import mplang.simp as simp
from mplang.core import MPObject
from mplang.frontend import phe

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
        label_party : int Party id of intercept.
        intercept : MPObject
    """

    weights: Dict[int, MPObject]
    reg_type: RegType
    label_party: Optional[int] = None
    intercept: Optional[MPObject] = None


class AbstractLinearModel(abc.ABC):
    """
    Abstract base class for linear models with predict and weight update methods.
    """

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass


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
def loss_and_grad(
    y_pred: MPObject, y: MPObject, label_party: int
) -> Tuple[MPObject, MPObject]:
    loss, gradient = simp.runAt(
        label_party, lambda y_pred, y: value_and_grad(mse_loss)(y_pred, y)
    )(y_pred, y)
    return loss, gradient
