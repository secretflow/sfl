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


from typing import Tuple

import jax
import jax.numpy as jnp
import mplang.v1 as mp
from mplang.v1.core import MPObject


@mp.function
def mse_loss(y_pred: MPObject, y: MPObject) -> MPObject:
    """
    Compute Mean Squared Error loss.

    Args:
        y_pred: Predicted values
        y: True target values

    Returns:
        Average squared error as MPObject
    """
    return mp.run_jax(lambda pred, y_true: jnp.mean((pred - y_true) ** 2), y_pred, y)


@mp.function
def loss_and_grad(
    y_pred: MPObject, y: MPObject, device: str
) -> Tuple[MPObject, MPObject]:
    """
    Compute MSE loss and its gradient using JAX automatic differentiation.

    Args:
        y_pred: Predicted values
        y: True target values
        device: Device to perform computation on

    Returns:
        Tuple of (loss, gradient)
    """

    # Define a pure JAX function for loss computation
    def mse_loss_jax(pred, y_true):
        return jnp.mean((pred - y_true) ** 2)

    # Use value_and_grad to compute both loss and gradient automatically
    loss, gradient = mp.device(device)(jax.value_and_grad(mse_loss_jax))(y_pred, y)

    return loss, gradient


@mp.function
def r2_score(y_pred: MPObject, y: MPObject) -> MPObject:
    """
    Coefficient of determination (R²) score.

    R² = 1 - (SS_res / SS_tot)
    where:
    - SS_res = Σ(y_true - y_pred)²  (residual sum of squares)
    - SS_tot = Σ(y_true - y_mean)²  (total sum of squares)

    Args:
        y_pred: Predicted values
        y: True target values

    Returns:
        R² score (1.0 is perfect prediction, 0.0 means no predictive power)
    """

    def compute_r2(pred, y_true):
        # Residual sum of squares
        ss_res = jnp.sum((y_true - pred) ** 2)

        # Total sum of squares
        y_mean = jnp.mean(y_true)
        ss_tot = jnp.sum((y_true - y_mean) ** 2)

        # R² = 1 - (SS_res / SS_tot)
        # Handle edge case where SS_tot is 0 (all y values are the same)
        r2 = jnp.where(ss_tot == 0, 0.0, 1.0 - (ss_res / ss_tot))

        return r2

    return mp.run_jax(compute_r2, y_pred, y)
