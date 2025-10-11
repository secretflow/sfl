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
Tests for the basic neural network models in base_model.py
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from sfl_lite.ml.nn.base_model import (
    BaseNN, FuseNN, CNNBase, RNNBase,
    create_base_model, create_fuse_model
)


class TestBaseNN:
    """Test the BaseNN model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = BaseNN(
            hidden_dims=(64, 32),
            output_dim=16,
            activation='relu'
        )
        
        # Check attributes
        assert model.hidden_dims == (64, 32)
        assert model.output_dim == 16
        assert model.activation == 'relu'
    
    def test_forward_pass(self):
        """Test forward pass with random data."""
        key = jax.random.PRNGKey(42)
        model = BaseNN(hidden_dims=(64, 32), output_dim=16)
        
        # Initialize parameters
        batch_size = 32
        input_dim = 100
        x = jax.random.normal(key, (batch_size, input_dim))
        
        params = model.init(key, x)
        
        # Forward pass
        output = model.apply(params, x)
        
        # Check output shape
        assert output.shape == (batch_size, 16)
        
        # Check output is finite
        assert jnp.all(jnp.isfinite(output))
    
    def test_different_activations(self):
        """Test different activation functions."""
        key = jax.random.PRNGKey(42)
        batch_size = 16
        input_dim = 50
        x = jax.random.normal(key, (batch_size, input_dim))
        
        for activation in ['relu', 'tanh', 'sigmoid']:
            model = BaseNN(
                hidden_dims=(32, 16),
                output_dim=8,
                activation=activation
            )
            params = model.init(key, x)
            output = model.apply(params, x)
            
            assert output.shape == (batch_size, 8)
            assert jnp.all(jnp.isfinite(output))


class TestFuseNN:
    """Test the FuseNN model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = FuseNN(
            hidden_dims=(128, 64),
            output_dim=10,
            activation='relu',
            final_activation='softmax'
        )
        
        assert model.hidden_dims == (128, 64)
        assert model.output_dim == 10
        assert model.final_activation == 'softmax'
    
    def test_forward_pass(self):
        """Test forward pass with concatenated embeddings."""
        key = jax.random.PRNGKey(42)
        model = FuseNN(hidden_dims=(64, 32), output_dim=5)
        
        # Simulate concatenated embeddings from multiple parties
        batch_size = 16
        total_embedding_dim = 128  # Sum of all party embeddings
        x = jax.random.normal(key, (batch_size, total_embedding_dim))
        
        params = model.init(key, x)
        output = model.apply(params, x)
        
        assert output.shape == (batch_size, 5)
        assert jnp.all(jnp.isfinite(output))
    
    def test_classification_outputs(self):
        """Test different output types for classification."""
        key = jax.random.PRNGKey(42)
        batch_size = 8
        input_dim = 64
        x = jax.random.normal(key, (batch_size, input_dim))
        
        # Binary classification
        binary_model = FuseNN(output_dim=1, final_activation='sigmoid')
        binary_params = binary_model.init(key, x)
        binary_output = binary_model.apply(binary_params, x)
        assert binary_output.shape == (batch_size, 1)
        assert jnp.all(binary_output >= 0)  # Sigmoid outputs [0,1]
        assert jnp.all(binary_output <= 1)
        
        # Multi-class classification
        multi_model = FuseNN(output_dim=3, final_activation='softmax')
        multi_params = multi_model.init(key, x)
        multi_output = multi_model.apply(multi_params, x)
        assert multi_output.shape == (batch_size, 3)
        
        # Check softmax normalization
        row_sums = jnp.sum(multi_output, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)


class TestCNNBase:
    """Test the CNNBase model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = CNNBase(
            conv_dims=(32, 64),
            kernel_sizes=(3, 3),
            dense_dims=(128,),
            output_dim=32
        )
        
        assert model.conv_dims == (32, 64)
        assert model.kernel_sizes == (3, 3)
        assert model.output_dim == 32
    
    def test_forward_pass(self):
        """Test forward pass with image data."""
        key = jax.random.PRNGKey(42)
        model = CNNBase(
            conv_dims=(16, 32),
            kernel_sizes=(3, 3),
            dense_dims=(64,),
            output_dim=16
        )
        
        # Simulate image data: (batch, height, width, channels)
        batch_size = 8
        height, width, channels = 28, 28, 3
        x = jax.random.normal(key, (batch_size, height, width, channels))
        
        params = model.init(key, x)
        output = model.apply(params, x)
        
        assert output.shape == (batch_size, 16)
        assert jnp.all(jnp.isfinite(output))
    
    def test_different_image_sizes(self):
        """Test with different image sizes."""
        key = jax.random.PRNGKey(42)
        model = CNNBase(
            conv_dims=(8, 16),
            kernel_sizes=(5, 3),
            dense_dims=(32,),
            output_dim=8
        )
        
        # Test different image sizes
        image_sizes = [(32, 32, 1), (64, 64, 3), (16, 16, 1)]
        
        for height, width, channels in image_sizes:
            x = jax.random.normal(key, (4, height, width, channels))
            params = model.init(key, x)
            output = model.apply(params, x)
            
            assert output.shape == (4, 8)
            assert jnp.all(jnp.isfinite(output))


class TestRNNBase:
    """Test the RNNBase model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = RNNBase(
            hidden_size=128,
            num_layers=2,
            output_dim=32,
            rnn_type='lstm'
        )
        
        assert model.hidden_size == 128
        assert model.num_layers == 2
        assert model.rnn_type == 'lstm'
    
    def test_forward_pass(self):
        """Test forward pass with sequence data."""
        key = jax.random.PRNGKey(42)
        model = RNNBase(
            hidden_size=64,
            num_layers=1,
            output_dim=16,
            rnn_type='lstm'
        )
        
        # Simulate sequence data: (batch, sequence_length, features)
        batch_size = 8
        sequence_length = 10
        features = 20
        x = jax.random.normal(key, (batch_size, sequence_length, features))
        
        params = model.init(key, x)
        output = model.apply(params, x)
        
        assert output.shape == (batch_size, 16)
        assert jnp.all(jnp.isfinite(output))
    
    def test_different_rnn_types(self):
        """Test different RNN types."""
        key = jax.random.PRNGKey(42)
        batch_size = 4
        sequence_length = 5
        features = 10
        x = jax.random.normal(key, (batch_size, sequence_length, features))
        
        for rnn_type in ['lstm', 'gru', 'rnn']:
            model = RNNBase(
                hidden_size=32,
                num_layers=1,
                output_dim=8,
                rnn_type=rnn_type
            )
            params = model.init(key, x)
            output = model.apply(params, x)
            
            assert output.shape == (batch_size, 8)
            assert jnp.all(jnp.isfinite(output))


class TestUtilityFunctions:
    """Test utility functions for model creation."""
    
    def test_create_base_model(self):
        """Test create_base_model utility."""
        # Test dense model
        dense_model = create_base_model(
            input_dim=100,
            hidden_dims=(64, 32),
            output_dim=16,
            model_type='dense'
        )
        assert isinstance(dense_model, BaseNN)
        
        # Test CNN model
        cnn_model = create_base_model(
            input_dim=100,  # Not used for CNN but required by signature
            hidden_dims=(64, 32),
            output_dim=16,
            model_type='cnn'
        )
        assert isinstance(cnn_model, CNNBase)
        
        # Test RNN model
        rnn_model = create_base_model(
            input_dim=100,  # Not used for RNN but required by signature
            hidden_dims=(64, 32),
            output_dim=16,
            model_type='rnn'
        )
        assert isinstance(rnn_model, RNNBase)
    
    def test_create_fuse_model(self):
        """Test create_fuse_model utility."""
        input_dims = {0: 32, 1: 16, 2: 8}  # 3 parties with different embedding dims
        
        fuse_model = create_fuse_model(
            input_dims=input_dims,
            hidden_dims=(128, 64),
            output_dim=5,
            final_activation='softmax'
        )
        
        assert isinstance(fuse_model, FuseNN)
        assert fuse_model.output_dim == 5
        assert fuse_model.final_activation == 'softmax'
    
    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_base_model(
                input_dim=100,
                model_type='invalid_type'
            )


class TestModelIntegration:
    """Test integration between base and fuse models."""
    
    def test_end_to_end_flow(self):
        """Test a complete flow from base to fuse model."""
        key = jax.random.PRNGKey(42)
        
        # Create models
        base_model = BaseNN(hidden_dims=(64, 32), output_dim=16)
        fuse_model = FuseNN(hidden_dims=(128, 64), output_dim=3, final_activation='softmax')
        
        # Sample data
        batch_size = 8
        input_dim = 50
        x = jax.random.normal(key, (batch_size, input_dim))
        
        # Initialize parameters
        base_params = base_model.init(key, x)
        
        # Base model forward pass
        embeddings = base_model.apply(base_params, x)
        assert embeddings.shape == (batch_size, 16)
        
        # Fuse model forward pass (simulating concatenated embeddings)
        # In real split learning, this would be concatenated from multiple parties
        fuse_params = fuse_model.init(key, embeddings)
        predictions = fuse_model.apply(fuse_params, embeddings)
        
        assert predictions.shape == (batch_size, 3)
        assert jnp.all(jnp.isfinite(predictions))
        
        # Check softmax normalization
        row_sums = jnp.sum(predictions, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)


if __name__ == "__main__":
    # Run basic tests
    print("Running basic NN tests...")
    
    # Test BaseNN
    key = jax.random.PRNGKey(42)
    base_model = BaseNN(hidden_dims=(64, 32), output_dim=16)
    x = jax.random.normal(key, (4, 100))
    params = base_model.init(key, x)
    output = base_model.apply(params, x)
    print(f"BaseNN output shape: {output.shape}")
    
    # Test FuseNN
    fuse_model = FuseNN(hidden_dims=(128, 64), output_dim=3, final_activation='softmax')
    fuse_params = fuse_model.init(key, output)
    predictions = fuse_model.apply(fuse_params, output)
    print(f"FuseNN predictions shape: {predictions.shape}")
    print(f"Prediction sums (should be ~1.0): {jnp.sum(predictions, axis=1)}")
    
    # Test CNN
    cnn_model = CNNBase(
        conv_dims=(16, 32),
        kernel_sizes=(3, 3),
        dense_dims=(64,),
        output_dim=10
    )
    img_data = jax.random.normal(key, (2, 28, 28, 3))
    cnn_params = cnn_model.init(key, img_data)
    cnn_output = cnn_model.apply(cnn_params, img_data)
    print(f"CNN output shape: {cnn_output.shape}")
    
    # Test RNN
    rnn_model = RNNBase(hidden_size=32, output_dim=8, rnn_type='lstm')
    seq_data = jax.random.normal(key, (2, 10, 20))
    rnn_params = rnn_model.init(key, seq_data)
    rnn_output = rnn_model.apply(rnn_params, seq_data)
    print(f"RNN output shape: {rnn_output.shape}")
    
    print("All tests completed successfully!")
