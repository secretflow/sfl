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

"""
Plain neural network models for split learning, inspired by MPLang patterns.

This module provides clean, modular neural network architectures using Flax NNX,
similar to MPLang examples. These are standard NNs without any split learning concepts.

Key features:
1. Clean separation from split learning logic
2. Standardized model creation interface
3. Configuration-driven model building
4. Compatible with JAX/Flax NNX patterns
"""

from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

# =============================================================================
# Plain Neural Network Models (Using Flax NNX)
# =============================================================================


class DNN(nnx.Module):
    """DNN-like MLP model."""

    def __init__(self, input_dim: int = 784, num_classes: int = 10, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(input_dim, 128, rngs=rngs)
        self.bn = nnx.BatchNorm(128, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, rngs=rngs)
        self.linear2 = nnx.Linear(128, num_classes, rngs=rngs)

    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nnx.relu(self.dropout(self.bn(self.linear1(x))))
        return self.linear2(x)


class MiniONN(nnx.Module):
    """MiniONN-like CNN model."""

    def __init__(self, num_classes: int = 10, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 16, (5, 5), rngs=rngs)
        self.conv2 = nnx.Conv(16, 16, (5, 5), rngs=rngs)
        self.dense1 = nnx.Linear(784, 100, rngs=rngs)
        self.dense2 = nnx.Linear(100, num_classes, rngs=rngs)

    def __call__(self, x):
        x = nnx.max_pool(nnx.relu(self.conv1(x)), (2, 2), strides=(2, 2))
        x = nnx.max_pool(nnx.relu(self.conv2(x)), (2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nnx.relu(self.dense1(x))
        return self.dense2(x)


class LeNet(nnx.Module):
    """LeNet-like CNN model."""

    def __init__(self, num_classes: int = 10, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(3, 20, (5, 5), rngs=rngs)
        self.conv2 = nnx.Conv(20, 50, (5, 5), rngs=rngs)
        self.dense1 = nnx.Linear(3200, 500, rngs=rngs)
        self.dense2 = nnx.Linear(500, num_classes, rngs=rngs)

    def __call__(self, x):
        x = nnx.max_pool(nnx.relu(self.conv1(x)), (2, 2), strides=(2, 2))
        x = nnx.max_pool(nnx.relu(self.conv2(x)), (2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nnx.relu(self.dense1(x))
        return self.dense2(x)


class Chameleon(nnx.Module):
    """Chameleon-like CNN model."""

    def __init__(self, num_classes: int = 10, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 5, (5, 5), strides=(2, 2), rngs=rngs)
        self.dense1 = nnx.Linear(980, 100, rngs=rngs)
        self.dense2 = nnx.Linear(100, num_classes, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nnx.relu(self.dense1(x))
        return self.dense2(x)


class AlexNet(nnx.Module):
    """AlexNet-like CNN model adapted for CIFAR-10."""

    def __init__(self, num_classes: int = 10, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            3, 96, (11, 11), strides=(4, 4), padding=((9, 9), (9, 9)), rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(96, rngs=rngs)

        self.conv2 = nnx.Conv(96, 256, (5, 5), padding=((1, 1), (1, 1)), rngs=rngs)
        self.bn2 = nnx.BatchNorm(256, rngs=rngs)

        self.conv3 = nnx.Conv(256, 384, (3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
        self.conv4 = nnx.Conv(384, 384, (3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
        self.conv5 = nnx.Conv(384, 256, (3, 3), padding=((1, 1), (1, 1)), rngs=rngs)

        self.dense1 = nnx.Linear(6400, 256, rngs=rngs)
        self.dense2 = nnx.Linear(256, 256, rngs=rngs)
        self.dense3 = nnx.Linear(256, num_classes, rngs=rngs)

    def __call__(self, x):
        x = nnx.avg_pool(nnx.relu(self.conv1(x)), (3, 3), strides=(2, 2))
        x = self.bn1(x)

        x = nnx.avg_pool(nnx.relu(self.conv2(x)), (2, 2), strides=(1, 1))
        x = self.bn2(x)

        x = nnx.relu(self.conv3(x))
        x = nnx.relu(self.conv4(x))
        x = nnx.relu(self.conv5(x))

        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nnx.relu(self.dense1(x))
        x = nnx.relu(self.dense2(x))
        return self.dense3(x)


class VGG16(nnx.Module):
    """VGG16-like CNN model adapted for CIFAR-10."""

    def __init__(self, num_classes: int = 10, *, rngs: nnx.Rngs):
        # Block 1
        self.conv1_1 = nnx.Conv(3, 64, (3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
        self.conv1_2 = nnx.Conv(64, 64, (3, 3), padding=((1, 1), (1, 1)), rngs=rngs)

        # Block 2
        self.conv2_1 = nnx.Conv(64, 128, (3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
        self.conv2_2 = nnx.Conv(128, 128, (3, 3), padding=((1, 1), (1, 1)), rngs=rngs)

        # Block 3
        self.conv3_1 = nnx.Conv(128, 256, (3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
        self.conv3_2 = nnx.Conv(256, 256, (3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
        self.conv3_3 = nnx.Conv(256, 256, (3, 3), padding=((1, 1), (1, 1)), rngs=rngs)

        # Block 4
        self.conv4_1 = nnx.Conv(256, 512, (3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
        self.conv4_2 = nnx.Conv(512, 512, (3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
        self.conv4_3 = nnx.Conv(512, 512, (3, 3), padding=((1, 1), (1, 1)), rngs=rngs)

        # Block 5
        self.conv5_1 = nnx.Conv(512, 512, (3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
        self.conv5_2 = nnx.Conv(512, 512, (3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
        self.conv5_3 = nnx.Conv(512, 512, (3, 3), padding=((1, 1), (1, 1)), rngs=rngs)

        # Classifier
        self.dense1 = nnx.Linear(2048, 256, rngs=rngs)
        self.dense2 = nnx.Linear(256, 256, rngs=rngs)
        self.dense3 = nnx.Linear(256, num_classes, rngs=rngs)

    def __call__(self, x):
        # Block 1
        x = nnx.relu(self.conv1_1(x))
        x = nnx.relu(self.conv1_2(x))
        x = nnx.avg_pool(x, (2, 2), strides=(2, 2))

        # Block 2
        x = nnx.relu(self.conv2_1(x))
        x = nnx.relu(self.conv2_2(x))
        x = nnx.avg_pool(x, (2, 2), strides=(2, 2))

        # Block 3
        x = nnx.relu(self.conv3_1(x))
        x = nnx.relu(self.conv3_2(x))
        x = nnx.relu(self.conv3_3(x))
        x = nnx.avg_pool(x, (2, 2), strides=(2, 2))

        # Block 4
        x = nnx.relu(self.conv4_1(x))
        x = nnx.relu(self.conv4_2(x))
        x = nnx.relu(self.conv4_3(x))
        x = nnx.avg_pool(x, (2, 2), strides=(2, 2))

        # Block 5
        x = nnx.relu(self.conv5_1(x))
        x = nnx.relu(self.conv5_2(x))
        x = nnx.relu(self.conv5_3(x))
        x = nnx.avg_pool(x, (2, 2), strides=(2, 2))

        # Classifier
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nnx.relu(self.dense1(x))
        x = nnx.relu(self.dense2(x))
        return self.dense3(x)


class CustomCNN(nnx.Module):
    """Custom CNN model."""

    def __init__(self, num_classes: int = 10, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(3, 2, (1, 1), rngs=rngs)
        self.dense1 = nnx.Linear(450, num_classes, rngs=rngs)

    def __call__(self, x):
        x = nnx.avg_pool(self.conv1(x), (2, 2), strides=(1, 1))
        x = x.reshape((x.shape[0], -1))  # Flatten
        return self.dense1(x)


class LogisticRegression(nnx.Module):
    """Logistic regression model."""

    def __init__(self, input_dim: int = 784, num_classes: int = 10, *, rngs: nnx.Rngs):
        self.dense = nnx.Linear(input_dim, num_classes, rngs=rngs)

    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten
        return self.dense(x)


class MLP2Layer(nnx.Module):
    """2-layer MLP."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        num_classes: int = 10,
        *,
        rngs: nnx.Rngs,
    ):
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.bn = nnx.BatchNorm(hidden_dim, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, num_classes, rngs=rngs)

    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nnx.relu(self.dropout(self.bn(self.linear1(x))))
        return self.linear2(x)


class MLP3Layer(nnx.Module):
    """3-layer MLP."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Tuple[int, int] = (256, 128),
        num_classes: int = 10,
        *,
        rngs: nnx.Rngs,
    ):
        self.linear1 = nnx.Linear(input_dim, hidden_dims[0], rngs=rngs)
        self.bn1 = nnx.BatchNorm(hidden_dims[0], rngs=rngs)
        self.dropout1 = nnx.Dropout(0.1, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dims[0], hidden_dims[1], rngs=rngs)
        self.bn2 = nnx.BatchNorm(hidden_dims[1], rngs=rngs)
        self.dropout2 = nnx.Dropout(0.1, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dims[1], num_classes, rngs=rngs)

    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nnx.relu(self.dropout1(self.bn1(self.linear1(x))))
        x = nnx.relu(self.dropout2(self.bn2(self.linear2(x))))
        return self.linear3(x)


# =============================================================================
# Model Factory (Inspired by MPLang patterns)
# =============================================================================


class ModelFactory:
    """Factory for creating standardized neural network models."""

    @staticmethod
    def create_model(model_name: str, *, rngs: nnx.Rngs, **kwargs) -> nnx.Module:
        """
        Create a model by name.

        Args:
            model_name: Name of the model to create
            rngs: Random number generator state
            **kwargs: Model-specific arguments

        Returns:
            Initialized model
        """
        model_map = {
            "dnn": DNN,
            "minionn": MiniONN,
            "lenet": LeNet,
            "chameleon": Chameleon,
            "alexnet": AlexNet,
            "vgg16": VGG16,
            "custom": CustomCNN,
            "logistic": LogisticRegression,
            "mlp2": MLP2Layer,
            "mlp3": MLP3Layer,
        }

        if model_name not in model_map:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(model_map.keys())}"
            )

        return model_map[model_name](rngs=rngs, **kwargs)

    @staticmethod
    def list_models() -> List[str]:
        """List all available models."""
        return [
            "dnn",
            "minionn",
            "lenet",
            "chameleon",
            "alexnet",
            "vgg16",
            "custom",
            "logistic",
            "mlp2",
            "mlp3",
        ]
