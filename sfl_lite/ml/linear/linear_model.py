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
from typing import Dict

import jax.numpy as jnp
import mplang
import mplang.mpi as mpi
import mplang.simp as simp
import mplang.smpc as smpc
from jax import grad
from mplang.core.base import MPObject
from mplang.core.mask import Mask

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
    intercept_party: int = None
    intercept: MPObject = None


@mplang.function
def linear_model_predict(
    model: LinearModel, X: Dict[int, MPObject], agg: Aggregator = None
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
    y_pred_no_intercept = smpc.revealTo(agg.sum(y_pred_party), model.intercept_party)
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
    # Broadcast gradient to all parties
    world_mask = Mask.all(max(list(model.weights.keys()) + [model.intercept_party]) + 1)
    broadcasted_gradient = mpi.bcast_m(world_mask, model.intercept_party, gradient)

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


if __name__ == "__main__":
    # 示例：创建一个简单的线性模型并进行预测
    import random

    # 初始化模拟器(3个参与方)
    sim3 = mplang.Simulator(3)
    mplang.set_ctx(sim3)

    # 设置特征维度
    n_features = 2

    # 在不同参与方生成权重向量(jax数组)
    party0_weight = simp.runAt(
        0, lambda: jnp.array([random.uniform(0.5, 1.5) for _ in range(n_features)])
    )()  # P0权重向量
    party1_weight = simp.runAt(
        1, lambda: jnp.array([random.uniform(0.5, 1.5) for _ in range(n_features)])
    )()  # P1权重向量
    intercept = simp.runAt(0, lambda: random.uniform(0.1, 1.0))()  # P0生成截距

    # 创建模型
    model = LinearModel(
        weights={0: party0_weight, 1: party1_weight},
        reg_type=RegType.Linear,
        intercept_party=0,
        intercept=intercept,
    )

    # 在不同参与方生成输入数据(jax数组)
    n_samples = 1
    party0_X = simp.runAt(
        0,
        lambda: jnp.array(
            [
                [random.uniform(0.5, 2.0) for _ in range(n_features)]
                for _ in range(n_samples)
            ]
        ),
    )()  # P0输入矩阵
    party1_X = simp.runAt(
        1,
        lambda: jnp.array(
            [
                [random.uniform(0.5, 2.0) for _ in range(n_features)]
                for _ in range(n_samples)
            ]
        ),
    )()  # P1输入矩阵

    # 生成标签（假设在party0）
    y_true = simp.runAt(
        0, lambda: jnp.array([random.uniform(1.0, 3.0) for _ in range(n_samples)])
    )()

    # 进行预测
    X = {0: party0_X, 1: party1_X}
    y_pred = linear_model_predict(model, X)

    # 打印结果
    print("party0_weight:", party0_weight)
    print("fetch(party0_weight):", mplang.fetch(None, party0_weight))
    print("party1_weight:", party1_weight)
    print("fetch(party1_weight):", mplang.fetch(None, party1_weight))
    print("intercept:", intercept)
    print("fetch(intercept):", mplang.fetch(None, intercept))
    print("party0_X:", party0_X)
    print("fetch(party0_X):", mplang.fetch(None, party0_X))
    print("party1_X:", party1_X)
    print("fetch(party1_X):", mplang.fetch(None, party1_X))
    print("y_true:", y_true)
    print("fetch(y_true):", mplang.fetch(None, y_true))
    print("y_pred:", y_pred)
    print("fetch(y_pred):", mplang.fetch(None, y_pred))

    # 线性模型梯度下降示例
    learning_rate = 0.1
    n_steps = 5
    for step in range(n_steps):
        # 预测
        y_pred = linear_model_predict(model, X)
        # 计算梯度
        gradient = grad_compute(y_pred, y_true, label_party=0)
        print("gradient:", mplang.fetch(None, gradient))
        # 更新权重和截距
        updated_weights, updated_intercept = sync_and_update_weights(
            model, X, gradient, learning_rate
        )
        print(f"\nStep {step+1}:")
        print("party0_weight:", mplang.fetch(None, updated_weights[0]))
        print("party1_weight:", mplang.fetch(None, updated_weights[1]))
        print("intercept:", mplang.fetch(None, updated_intercept))
        model.weights = updated_weights
        model.intercept = updated_intercept
        y_pred_new = linear_model_predict(model, X)
        print("y_pred:", mplang.fetch(None, y_pred_new))
