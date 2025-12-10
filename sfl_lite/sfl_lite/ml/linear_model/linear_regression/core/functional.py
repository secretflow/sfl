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

import jax.numpy as jnp
import mplang.v1 as mp
from jax import grad, value_and_grad
from mplang.v1.core import MPObject


@mp.function
def mse_loss(y_pred: MPObject, y: MPObject) -> MPObject:
    """
    Vectorized Mean Squared Error loss
    y_pred: predicted values
    y: true target values
    Returns average squared error
    """
    return mp.run_jax(lambda pred, true: jnp.mean((pred - true) ** 2), y_pred, y)


@mp.function
def grad_compute(y_pred: MPObject, y: MPObject, device: str) -> MPObject:
    gradient = mp.device(device)(grad(mse_loss))(y_pred, y)
    return gradient


@mp.function
def loss_and_grad(
    y_pred: MPObject, y: MPObject, device: str
) -> Tuple[MPObject, MPObject]:
    # Define pure JAX functions for loss and gradient computation
    def mse_loss_jax(pred, true):
        return jnp.mean((pred - true) ** 2)

    def mse_grad_jax(pred, true):
        return 2 * (pred - true) / len(pred)

    # Compute loss and gradient on the specified device
    loss = mp.device(device)(mse_loss_jax)(y_pred, y)
    gradient = mp.device(device)(mse_grad_jax)(y_pred, y)

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

    def compute_r2(pred, true):
        # Residual sum of squares
        ss_res = jnp.sum((true - pred) ** 2)

        # Total sum of squares
        y_mean = jnp.mean(true)
        ss_tot = jnp.sum((true - y_mean) ** 2)

        # R² = 1 - (SS_res / SS_tot)
        # Handle edge case where SS_tot is 0 (all y values are the same)
        r2 = jnp.where(ss_tot == 0, 0.0, 1.0 - (ss_res / ss_tot))

        return r2

    return mp.run_jax(compute_r2, y_pred, y)
