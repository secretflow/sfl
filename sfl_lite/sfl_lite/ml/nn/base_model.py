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
Simple neural network models for split learning.

This module provides basic neural network architectures that can be used
as base models (local feature processing) or fuse models (aggregation and prediction)
in split learning scenarios. These are standard NNs without any split learning logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import jax.numpy as jnp
import jax.random as random
from flax import linen as flax_nn
import flax


class BaseNN(flax_nn.Module):
    """
    A simple fully-connected neural network for base models.
    
    This processes local features and outputs embeddings.
    No split learning logic - just a standard NN.
    """
    
    hidden_dims: Tuple[int, ...] = (128, 64)
    output_dim: int = 32
    activation: str = 'relu'
    
    def setup(self):
        """Setup layers."""
        # Hidden layers
        self.hidden_layers = [
            flax_nn.Dense(hidden_dim, name=f'hidden_{i}')
            for i, hidden_dim in enumerate(self.hidden_dims)
        ]
        
        # Output layer
        self.output_layer = flax_nn.Dense(self.output_dim, name='output')
    
    def __call__(self, x):
        """Forward pass."""
        h = x
        
        # Apply hidden layers with activation
        for layer in self.hidden_layers:
            h = layer(h)
            if self.activation == 'relu':
                h = flax_nn.relu(h)
            elif self.activation == 'tanh':
                h = flax_nn.tanh(h)
            elif self.activation == 'sigmoid':
                h = flax_nn.sigmoid(h)
        
        # Output layer (no activation - embeddings can be any real values)
        return self.output_layer(h)


class FuseNN(flax_nn.Module):
    """
    A simple fully-connected neural network for fuse models.
    
    This takes concatenated embeddings from all parties and produces final predictions.
    No split learning logic - just a standard NN.
    """
    
    hidden_dims: Tuple[int, ...] = (256, 128)
    output_dim: int = 1  # Default for binary classification
    activation: str = 'relu'
    final_activation: Optional[str] = None  # 'sigmoid' for binary classification, 'softmax' for multi-class
    
    def setup(self):
        """Setup layers."""
        # Hidden layers
        self.hidden_layers = [
            flax_nn.Dense(hidden_dim, name=f'hidden_{i}')
            for i, hidden_dim in enumerate(self.hidden_dims)
        ]
        
        # Output layer
        self.output_layer = flax_nn.Dense(self.output_dim, name='output')
    
    def __call__(self, x):
        """Forward pass."""
        h = x
        
        # Apply hidden layers with activation
        for layer in self.hidden_layers:
            h = layer(h)
            if self.activation == 'relu':
                h = flax_nn.relu(h)
            elif self.activation == 'tanh':
                h = flax_nn.tanh(h)
            elif self.activation == 'sigmoid':
                h = flax_nn.sigmoid(h)
        
        # Output layer
        h = self.output_layer(h)
        
        # Apply final activation if specified
        if self.final_activation == 'sigmoid':
            h = flax_nn.sigmoid(h)
        elif self.final_activation == 'softmax':
            h = flax_nn.softmax(h)
        elif self.final_activation == 'tanh':
            h = flax_nn.tanh(h)
        
        return h


class CNNBase(flax_nn.Module):
    """
    Convolutional neural network for base models (e.g., image data).
    
    Processes local features and outputs embeddings.
    """
    
    conv_dims: Tuple[int, ...] = (32, 64)
    kernel_sizes: Tuple[int, ...] = (3, 3)
    dense_dims: Tuple[int, ...] = (128,)
    output_dim: int = 32
    activation: str = 'relu'
    
    def setup(self):
        """Setup layers."""
        # Convolutional layers
        self.conv_layers = [
            flax_nn.Conv(conv_dim, (kernel_size, kernel_size), name=f'conv_{i}')
            for i, (conv_dim, kernel_size) in enumerate(zip(self.conv_dims, self.kernel_sizes))
        ]
        
        # Dense layers
        self.dense_layers = [
            flax_nn.Dense(dense_dim, name=f'dense_{i}')
            for i, dense_dim in enumerate(self.dense_dims)
        ]
        
        # Output layer
        self.output_layer = flax_nn.Dense(self.output_dim, name='output')
    
    def __call__(self, x):
        """Forward pass."""
        h = x
        
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            h = conv_layer(h)
            if self.activation == 'relu':
                h = flax_nn.relu(h)
            elif self.activation == 'tanh':
                h = flax_nn.tanh(h)
            h = flax_nn.max_pool(h, (2, 2), strides=(2, 2))
        
        # Flatten for dense layers
        h = h.reshape((h.shape[0], -1))
        
        # Apply dense layers
        for dense_layer in self.dense_layers:
            h = dense_layer(h)
            if self.activation == 'relu':
                h = flax_nn.relu(h)
            elif self.activation == 'tanh':
                h = flax_nn.tanh(h)
        
        # Output layer
        return self.output_layer(h)


class RNNBase(flax_nn.Module):
    """
    Recurrent neural network for base models (e.g., sequence data).
    
    Processes local features and outputs embeddings.
    """
    
    hidden_size: int = 128
    num_layers: int = 2
    output_dim: int = 32
    rnn_type: str = 'lstm'  # 'lstm', 'gru', or 'rnn'
    dropout_rate: float = 0.1
    
    def setup(self):
        """Setup layers."""
        # Use a simple dense layer approach for sequence processing
        # This is a simplified RNN-like structure using dense layers
        self.sequence_layers = [
            flax_nn.Dense(self.hidden_size, name=f'seq_{i}')
            for i in range(self.num_layers)
        ]
        
        self.output_layer = flax_nn.Dense(self.output_dim, name='output')
    
    def __call__(self, x):
        """Forward pass."""
        # x shape: (batch_size, sequence_length, features)
        
        # Process each timestep
        h = x
        
        # Apply sequence processing layers
        for layer in self.sequence_layers:
            # Process each timestep
            h = layer(h)
            h = flax_nn.tanh(h)  # Simple activation
        
        # Take last timestep
        h = h[:, -1, :]
        
        # Output layer
        return self.output_layer(h)


# Utility functions for model creation
def create_base_model(input_dim: int, 
                     hidden_dims: Tuple[int, ...] = (128, 64), 
                     output_dim: int = 32,
                     model_type: str = 'dense') -> flax_nn.Module:
    """
    Create a base model based on the specified type.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: Hidden layer dimensions
        output_dim: Output embedding dimension
        model_type: Type of model ('dense', 'cnn', 'rnn')
    
    Returns:
        Initialized base model
    """
    if model_type == 'dense':
        return BaseNN(
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
    elif model_type == 'cnn':
        return CNNBase(
            dense_dims=hidden_dims,
            output_dim=output_dim
        )
    elif model_type == 'rnn':
        return RNNBase(
            hidden_size=hidden_dims[0] if hidden_dims else 128,
            output_dim=output_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_fuse_model(input_dims: Dict[int, int],
                     hidden_dims: Tuple[int, ...] = (256, 128),
                     output_dim: int = 1,
                     final_activation: Optional[str] = None) -> flax_nn.Module:
    """
    Create a fuse model.
    
    Args:
        input_dims: Dict mapping party_id to embedding dimension
        hidden_dims: Hidden layer dimensions
        output_dim: Final output dimension
        final_activation: Final activation function
    
    Returns:
        Initialized fuse model
    """
    total_input_dim = sum(input_dims.values())
    
    return FuseNN(
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        final_activation=final_activation
    )
