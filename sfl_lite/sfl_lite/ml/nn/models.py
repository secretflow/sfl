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

This module provides clean, modular neural network architectures using Flax,
similar to MPLang examples. These are standard NNs without any split learning concepts.

Key features:
1. Clean separation from split learning logic
2. Standardized model creation interface
3. Configuration-driven model building
4. Compatible with JAX/Flax patterns
"""

from typing import Dict, Any, Optional, Tuple, List
import jax
import jax.numpy as jnp
from flax import linen as flax_nn


# =============================================================================
# Plain Neural Network Models (Using Flax)
# =============================================================================

class DNN(flax_nn.Module):
    """DNN-like MLP model."""
    
    input_dim: int = 784
    num_classes: int = 10
    
    def setup(self):
        self.dense1 = flax_nn.Dense(128)
        self.dense2 = flax_nn.Dense(128)
        self.dense3 = flax_nn.Dense(self.num_classes)
    
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = self.dense1(x)
        x = flax_nn.relu(x)
        x = self.dense2(x)
        x = flax_nn.relu(x)
        x = self.dense3(x)
        return x


class MiniONN(flax_nn.Module):
    """MiniONN-like CNN model."""
    
    num_classes: int = 10
    
    def setup(self):
        self.conv1 = flax_nn.Conv(16, (5, 5))
        self.conv2 = flax_nn.Conv(16, (5, 5))
        self.dense1 = flax_nn.Dense(100)
        self.dense2 = flax_nn.Dense(self.num_classes)
    
    def __call__(self, x):
        x = self.conv1(x)
        x = flax_nn.max_pool(x, (2, 2), strides=(2, 2))
        x = flax_nn.relu(x)
        
        x = self.conv2(x)
        x = flax_nn.max_pool(x, (2, 2), strides=(2, 2))
        x = flax_nn.relu(x)
        
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = self.dense1(x)
        x = flax_nn.relu(x)
        x = self.dense2(x)
        return x


class LeNet(flax_nn.Module):
    """LeNet-like CNN model."""
    
    num_classes: int = 10
    
    def setup(self):
        self.conv1 = flax_nn.Conv(20, (5, 5))
        self.conv2 = flax_nn.Conv(50, (5, 5))
        self.dense1 = flax_nn.Dense(500)
        self.dense2 = flax_nn.Dense(self.num_classes)
    
    def __call__(self, x):
        x = self.conv1(x)
        x = flax_nn.max_pool(x, (2, 2), strides=(2, 2))
        x = flax_nn.relu(x)
        
        x = self.conv2(x)
        x = flax_nn.max_pool(x, (2, 2), strides=(2, 2))
        x = flax_nn.relu(x)
        
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = self.dense1(x)
        x = flax_nn.relu(x)
        x = self.dense2(x)
        return x


class Chameleon(flax_nn.Module):
    """Chameleon-like CNN model."""
    
    num_classes: int = 10
    
    def setup(self):
        self.conv1 = flax_nn.Conv(5, (5, 5), strides=(2, 2))
        self.dense1 = flax_nn.Dense(100)
        self.dense2 = flax_nn.Dense(self.num_classes)
    
    def __call__(self, x):
        x = self.conv1(x)
        x = flax_nn.relu(x)
        
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = self.dense1(x)
        x = flax_nn.relu(x)
        x = self.dense2(x)
        return x


class AlexNet(flax_nn.Module):
    """AlexNet-like CNN model adapted for CIFAR-10."""
    
    num_classes: int = 10
    
    def setup(self):
        self.conv1 = flax_nn.Conv(96, (11, 11), strides=(4, 4), padding=((9, 9), (9, 9)))
        self.bn1 = flax_nn.BatchNorm(use_running_average=True)
        
        self.conv2 = flax_nn.Conv(256, (5, 5), padding=((1, 1), (1, 1)))
        self.bn2 = flax_nn.BatchNorm(use_running_average=True)
        
        self.conv3 = flax_nn.Conv(384, (3, 3), padding=((1, 1), (1, 1)))
        self.conv4 = flax_nn.Conv(384, (3, 3), padding=((1, 1), (1, 1)))
        self.conv5 = flax_nn.Conv(256, (3, 3), padding=((1, 1), (1, 1)))
        
        self.dense1 = flax_nn.Dense(256)
        self.dense2 = flax_nn.Dense(256)
        self.dense3 = flax_nn.Dense(self.num_classes)
    
    def __call__(self, x):
        x = self.conv1(x)
        x = flax_nn.relu(x)
        x = flax_nn.avg_pool(x, (3, 3), strides=(2, 2))
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = flax_nn.relu(x)
        x = flax_nn.avg_pool(x, (2, 2), strides=(1, 1))
        x = self.bn2(x)
        
        x = self.conv3(x)
        x = flax_nn.relu(x)
        x = self.conv4(x)
        x = flax_nn.relu(x)
        x = self.conv5(x)
        x = flax_nn.relu(x)
        
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = self.dense1(x)
        x = flax_nn.relu(x)
        x = self.dense2(x)
        x = flax_nn.relu(x)
        x = self.dense3(x)
        return x


class VGG16(flax_nn.Module):
    """VGG16-like CNN model adapted for CIFAR-10."""
    
    num_classes: int = 10
    
    def setup(self):
        # Block 1
        self.conv1_1 = flax_nn.Conv(64, (3, 3), padding=((1, 1), (1, 1)))
        self.conv1_2 = flax_nn.Conv(64, (3, 3), padding=((1, 1), (1, 1)))
        
        # Block 2
        self.conv2_1 = flax_nn.Conv(128, (3, 3), padding=((1, 1), (1, 1)))
        self.conv2_2 = flax_nn.Conv(128, (3, 3), padding=((1, 1), (1, 1)))
        
        # Block 3
        self.conv3_1 = flax_nn.Conv(256, (3, 3), padding=((1, 1), (1, 1)))
        self.conv3_2 = flax_nn.Conv(256, (3, 3), padding=((1, 1), (1, 1)))
        self.conv3_3 = flax_nn.Conv(256, (3, 3), padding=((1, 1), (1, 1)))
        
        # Block 4
        self.conv4_1 = flax_nn.Conv(512, (3, 3), padding=((1, 1), (1, 1)))
        self.conv4_2 = flax_nn.Conv(512, (3, 3), padding=((1, 1), (1, 1)))
        self.conv4_3 = flax_nn.Conv(512, (3, 3), padding=((1, 1), (1, 1)))
        
        # Block 5
        self.conv5_1 = flax_nn.Conv(512, (3, 3), padding=((1, 1), (1, 1)))
        self.conv5_2 = flax_nn.Conv(512, (3, 3), padding=((1, 1), (1, 1)))
        self.conv5_3 = flax_nn.Conv(512, (3, 3), padding=((1, 1), (1, 1)))
        
        # Classifier
        self.dense1 = flax_nn.Dense(256)
        self.dense2 = flax_nn.Dense(256)
        self.dense3 = flax_nn.Dense(self.num_classes)
    
    def __call__(self, x):
        # Block 1
        x = self.conv1_1(x)
        x = flax_nn.relu(x)
        x = self.conv1_2(x)
        x = flax_nn.relu(x)
        x = flax_nn.avg_pool(x, (2, 2), strides=(2, 2))
        
        # Block 2
        x = self.conv2_1(x)
        x = flax_nn.relu(x)
        x = self.conv2_2(x)
        x = flax_nn.relu(x)
        x = flax_nn.avg_pool(x, (2, 2), strides=(2, 2))
        
        # Block 3
        x = self.conv3_1(x)
        x = flax_nn.relu(x)
        x = self.conv3_2(x)
        x = flax_nn.relu(x)
        x = self.conv3_3(x)
        x = flax_nn.relu(x)
        x = flax_nn.avg_pool(x, (2, 2), strides=(2, 2))
        
        # Block 4
        x = self.conv4_1(x)
        x = flax_nn.relu(x)
        x = self.conv4_2(x)
        x = flax_nn.relu(x)
        x = self.conv4_3(x)
        x = flax_nn.relu(x)
        x = flax_nn.avg_pool(x, (2, 2), strides=(2, 2))
        
        # Block 5
        x = self.conv5_1(x)
        x = flax_nn.relu(x)
        x = self.conv5_2(x)
        x = flax_nn.relu(x)
        x = self.conv5_3(x)
        x = flax_nn.relu(x)
        x = flax_nn.avg_pool(x, (2, 2), strides=(2, 2))
        
        # Classifier
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = self.dense1(x)
        x = flax_nn.relu(x)
        x = self.dense2(x)
        x = flax_nn.relu(x)
        x = self.dense3(x)
        return x


class CustomCNN(flax_nn.Module):
    """Custom CNN model."""
    
    num_classes: int = 10
    
    def setup(self):
        self.conv1 = flax_nn.Conv(2, (1, 1))
        self.dense1 = flax_nn.Dense(self.num_classes)
    
    def __call__(self, x):
        x = self.conv1(x)
        x = flax_nn.avg_pool(x, (2, 2), strides=(1, 1))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = self.dense1(x)
        return x


class LogisticRegression(flax_nn.Module):
    """Logistic regression model."""
    
    input_dim: int = 784
    num_classes: int = 10
    
    def setup(self):
        self.dense = flax_nn.Dense(self.num_classes)
    
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = self.dense(x)
        return x


class MLP2Layer(flax_nn.Module):
    """2-layer MLP."""
    
    input_dim: int = 784
    hidden_dim: int = 128
    num_classes: int = 10
    
    def setup(self):
        self.dense1 = flax_nn.Dense(self.hidden_dim)
        self.dense2 = flax_nn.Dense(self.num_classes)
    
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = self.dense1(x)
        x = flax_nn.relu(x)
        x = self.dense2(x)
        return x


class MLP3Layer(flax_nn.Module):
    """3-layer MLP."""
    
    input_dim: int = 784
    hidden_dims: Tuple[int, int] = (256, 128)
    num_classes: int = 10
    
    def setup(self):
        self.dense1 = flax_nn.Dense(self.hidden_dims[0])
        self.dense2 = flax_nn.Dense(self.hidden_dims[1])
        self.dense3 = flax_nn.Dense(self.num_classes)
    
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = self.dense1(x)
        x = flax_nn.relu(x)
        x = self.dense2(x)
        x = flax_nn.relu(x)
        x = self.dense3(x)
        return x


# =============================================================================
# Model Factory (Inspired by MPLang patterns)
# =============================================================================

class ModelFactory:
    """Factory for creating standardized neural network models."""
    
    @staticmethod
    def create_model(model_name: str, **kwargs) -> flax_nn.Module:
        """
        Create a model by name.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Model-specific arguments
        
        Returns:
            Initialized model
        """
        model_map = {
            'dnn': DNN,
            'minionn': MiniONN,
            'lenet': LeNet,
            'chameleon': Chameleon,
            'alexnet': AlexNet,
            'vgg16': VGG16,
            'custom': CustomCNN,
            'logistic': LogisticRegression,
            'mlp2': MLP2Layer,
            'mlp3': MLP3Layer,
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")
        
        return model_map[model_name](**kwargs)
    
    @staticmethod
    def list_models() -> List[str]:
        """List all available models."""
        return ['dnn', 'minionn', 'lenet', 'chameleon', 'alexnet', 'vgg16', 'custom', 'logistic', 'mlp2', 'mlp3']
