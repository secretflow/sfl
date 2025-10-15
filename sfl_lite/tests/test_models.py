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
Unit tests for neural network models in models.py
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from sfl_lite.ml.nn.models import (
    DNN, MiniONN, LeNet, Chameleon, AlexNet, VGG16,
    CustomCNN, LogisticRegression, MLP2Layer, MLP3Layer,
    ModelFactory
)


class TestDNN:
    """Test the DNN (formerly SecureML) model."""
    
    def test_initialization_default(self):
        """Test DNN model with default parameters."""
        rngs = nnx.Rngs(42)
        model = DNN(rngs=rngs)
        assert model.linear1.in_features == 784
        assert model.linear2.out_features == 10
    
    def test_initialization_custom(self):
        """Test DNN model with custom parameters."""
        rngs = nnx.Rngs(42)
        model = DNN(input_dim=100, num_classes=5, rngs=rngs)
        assert model.linear1.in_features == 100
        assert model.linear2.out_features == 5
    
    def test_forward_pass(self):
        """Test forward pass with random data."""
        rngs = nnx.Rngs(42)
        model = DNN(input_dim=100, num_classes=5, rngs=rngs)
        
        # Initialize parameters
        batch_size = 32
        x = jax.random.normal(rngs(), (batch_size, 100))
        
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 5)
        assert jnp.all(jnp.isfinite(output))
    
    def test_flatten_behavior(self):
        """Test that the model correctly flattens input."""
        rngs = nnx.Rngs(42)
        model = DNN(input_dim=28*28, num_classes=10, rngs=rngs)
        
        # Test with 2D image-like input
        batch_size = 8
        x = jax.random.normal(rngs(), (batch_size, 28, 28))
        
        output = model(x)
        
        assert output.shape == (batch_size, 10)


class TestMiniONN:
    """Test the MiniONN CNN model."""
    
    def test_initialization_default(self):
        """Test MiniONN model with default parameters."""
        rngs = nnx.Rngs(42)
        model = MiniONN(rngs=rngs)
        assert model.dense2.out_features == 10
    
    def test_initialization_custom(self):
        """Test MiniONN model with custom parameters."""
        rngs = nnx.Rngs(42)
        model = MiniONN(num_classes=5, rngs=rngs)
        assert model.dense2.out_features == 5
    
    def test_forward_pass(self):
        """Test forward pass with image data."""
        rngs = nnx.Rngs(42)
        model = MiniONN(num_classes=5, rngs=rngs)
        
        # Simulate image data: (batch, height, width, channels)
        batch_size = 16
        x = jax.random.normal(rngs(), (batch_size, 28, 28, 1))
        
        output = model(x)
        
        assert output.shape == (batch_size, 5)
        assert jnp.all(jnp.isfinite(output))


class TestLeNet:
    """Test the LeNet CNN model."""
    
    def test_initialization_default(self):
        """Test LeNet model with default parameters."""
        rngs = nnx.Rngs(42)
        model = LeNet(rngs=rngs)
        assert model.dense2.out_features == 10
    
    def test_forward_pass(self):
        """Test forward pass with image data."""
        rngs = nnx.Rngs(42)
        model = LeNet(num_classes=10, rngs=rngs)
        
        batch_size = 8
        x = jax.random.normal(rngs(), (batch_size, 32, 32, 3))
        
        output = model(x)
        
        assert output.shape == (batch_size, 10)
        assert jnp.all(jnp.isfinite(output))


class TestChameleon:
    """Test the Chameleon CNN model."""
    
    def test_initialization_default(self):
        """Test Chameleon model with default parameters."""
        rngs = nnx.Rngs(42)
        model = Chameleon(rngs=rngs)
        assert model.dense2.out_features == 10
    
    def test_forward_pass(self):
        """Test forward pass with image data."""
        rngs = nnx.Rngs(42)
        model = Chameleon(num_classes=5, rngs=rngs)
        
        batch_size = 4
        x = jax.random.normal(rngs(), (batch_size, 28, 28, 1))
        
        output = model(x)
        
        assert output.shape == (batch_size, 5)
        assert jnp.all(jnp.isfinite(output))


class TestAlexNet:
    """Test the AlexNet CNN model."""
    
    def test_initialization_default(self):
        """Test AlexNet model with default parameters."""
        rngs = nnx.Rngs(42)
        model = AlexNet(rngs=rngs)
        assert model.dense3.out_features == 10
    
    def test_forward_pass(self):
        """Test forward pass with image data."""
        rngs = nnx.Rngs(42)
        model = AlexNet(num_classes=10, rngs=rngs)
        
        # AlexNet expects larger images - use 64x64 to avoid size issues
        batch_size = 2
        x = jax.random.normal(rngs(), (batch_size, 64, 64, 3))
        
        output = model(x)
        
        assert output.shape == (batch_size, 10)
        assert jnp.all(jnp.isfinite(output))


class TestVGG16:
    """Test the VGG16 CNN model."""
    
    def test_initialization_default(self):
        """Test VGG16 model with default parameters."""
        rngs = nnx.Rngs(42)
        model = VGG16(rngs=rngs)
        assert model.dense3.out_features == 10
    
    def test_forward_pass(self):
        """Test forward pass with image data."""
        rngs = nnx.Rngs(42)
        model = VGG16(num_classes=10, rngs=rngs)
        
        # VGG16 expects reasonably sized images - use 64x64 to avoid size issues
        batch_size = 1  # VGG16 is large, use small batch
        x = jax.random.normal(rngs(), (batch_size, 64, 64, 3))
        
        output = model(x)
        
        assert output.shape == (batch_size, 10)
        assert jnp.all(jnp.isfinite(output))


class TestCustomCNN:
    """Test the CustomCNN model."""
    
    def test_initialization_default(self):
        """Test CustomCNN model with default parameters."""
        rngs = nnx.Rngs(42)
        model = CustomCNN(rngs=rngs)
        assert model.dense1.out_features == 10
    
    def test_forward_pass(self):
        """Test forward pass with image data."""
        rngs = nnx.Rngs(42)
        model = CustomCNN(num_classes=3, rngs=rngs)
        
        batch_size = 8
        x = jax.random.normal(rngs(), (batch_size, 16, 16, 3))
        
        output = model(x)
        
        assert output.shape == (batch_size, 3)
        assert jnp.all(jnp.isfinite(output))


class TestLogisticRegression:
    """Test the LogisticRegression model."""
    
    def test_initialization_default(self):
        """Test LogisticRegression model with default parameters."""
        rngs = nnx.Rngs(42)
        model = LogisticRegression(rngs=rngs)
        assert model.dense.in_features == 784
        assert model.dense.out_features == 10
    
    def test_forward_pass(self):
        """Test forward pass with data."""
        rngs = nnx.Rngs(42)
        model = LogisticRegression(input_dim=50, num_classes=3, rngs=rngs)
        
        batch_size = 16
        x = jax.random.normal(rngs(), (batch_size, 50))
        
        output = model(x)
        
        assert output.shape == (batch_size, 3)
        assert jnp.all(jnp.isfinite(output))


class TestMLP2Layer:
    """Test the MLP2Layer model."""
    
    def test_initialization_default(self):
        """Test MLP2Layer model with default parameters."""
        rngs = nnx.Rngs(42)
        model = MLP2Layer(rngs=rngs)
        assert model.linear1.in_features == 784
        assert model.linear1.out_features == 128
        assert model.linear2.out_features == 10
    
    def test_forward_pass(self):
        """Test forward pass with data."""
        rngs = nnx.Rngs(42)
        model = MLP2Layer(input_dim=100, hidden_dim=64, num_classes=5, rngs=rngs)
        
        batch_size = 8
        x = jax.random.normal(rngs(), (batch_size, 100))
        
        output = model(x)
        
        assert output.shape == (batch_size, 5)
        assert jnp.all(jnp.isfinite(output))


class TestMLP3Layer:
    """Test the MLP3Layer model."""
    
    def test_initialization_default(self):
        """Test MLP3Layer model with default parameters."""
        rngs = nnx.Rngs(42)
        model = MLP3Layer(rngs=rngs)
        assert model.linear1.in_features == 784
        assert model.linear1.out_features == 256
        assert model.linear2.out_features == 128
        assert model.linear3.out_features == 10
    
    def test_forward_pass(self):
        """Test forward pass with data."""
        rngs = nnx.Rngs(42)
        model = MLP3Layer(input_dim=100, hidden_dims=(64, 32), num_classes=3, rngs=rngs)
        
        batch_size = 4
        x = jax.random.normal(rngs(), (batch_size, 100))
        
        output = model(x)
        
        assert output.shape == (batch_size, 3)
        assert jnp.all(jnp.isfinite(output))


class TestModelFactory:
    """Test the ModelFactory class."""
    
    def test_create_dnn(self):
        """Test creating DNN model via factory."""
        rngs = nnx.Rngs(42)
        model = ModelFactory.create_model('dnn', input_dim=100, num_classes=5, rngs=rngs)
        assert isinstance(model, DNN)
        assert model.linear1.in_features == 100
        assert model.linear2.out_features == 5
    
    def test_create_minionn(self):
        """Test creating MiniONN model via factory."""
        rngs = nnx.Rngs(42)
        model = ModelFactory.create_model('minionn', num_classes=5, rngs=rngs)
        assert isinstance(model, MiniONN)
        assert model.dense2.out_features == 5
    
    def test_create_lenet(self):
        """Test creating LeNet model via factory."""
        rngs = nnx.Rngs(42)
        model = ModelFactory.create_model('lenet', num_classes=10, rngs=rngs)
        assert isinstance(model, LeNet)
        assert model.dense2.out_features == 10
    
    def test_create_chameleon(self):
        """Test creating Chameleon model via factory."""
        rngs = nnx.Rngs(42)
        model = ModelFactory.create_model('chameleon', num_classes=5, rngs=rngs)
        assert isinstance(model, Chameleon)
        assert model.dense2.out_features == 5
    
    def test_create_alexnet(self):
        """Test creating AlexNet model via factory."""
        rngs = nnx.Rngs(42)
        model = ModelFactory.create_model('alexnet', num_classes=10, rngs=rngs)
        assert isinstance(model, AlexNet)
        assert model.dense3.out_features == 10
    
    def test_create_vgg16(self):
        """Test creating VGG16 model via factory."""
        rngs = nnx.Rngs(42)
        model = ModelFactory.create_model('vgg16', num_classes=10, rngs=rngs)
        assert isinstance(model, VGG16)
        assert model.dense3.out_features == 10
    
    def test_create_custom_cnn(self):
        """Test creating CustomCNN model via factory."""
        rngs = nnx.Rngs(42)
        model = ModelFactory.create_model('custom', num_classes=3, rngs=rngs)
        assert isinstance(model, CustomCNN)
        assert model.dense1.out_features == 3
    
    def test_create_logistic(self):
        """Test creating LogisticRegression model via factory."""
        rngs = nnx.Rngs(42)
        model = ModelFactory.create_model('logistic', input_dim=100, num_classes=3, rngs=rngs)
        assert isinstance(model, LogisticRegression)
        assert model.dense.in_features == 100
        assert model.dense.out_features == 3
    
    def test_create_mlp2(self):
        """Test creating MLP2Layer model via factory."""
        rngs = nnx.Rngs(42)
        model = ModelFactory.create_model('mlp2', input_dim=100, hidden_dim=64, num_classes=3, rngs=rngs)
        assert isinstance(model, MLP2Layer)
        assert model.linear1.in_features == 100
        assert model.linear1.out_features == 64
        assert model.linear2.out_features == 3
    
    def test_create_mlp3(self):
        """Test creating MLP3Layer model via factory."""
        rngs = nnx.Rngs(42)
        model = ModelFactory.create_model('mlp3', input_dim=100, hidden_dims=(64, 32), num_classes=3, rngs=rngs)
        assert isinstance(model, MLP3Layer)
        assert model.linear1.in_features == 100
        assert model.linear1.out_features == 64
        assert model.linear2.out_features == 32
        assert model.linear3.out_features == 3
    
    def test_invalid_model_name(self):
        """Test error handling for invalid model name."""
        rngs = nnx.Rngs(42)
        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.create_model('invalid_model', rngs=rngs)
    
    def test_list_models(self):
        """Test listing all available models."""
        models = ModelFactory.list_models()
        expected_models = ['dnn', 'minionn', 'lenet', 'chameleon', 'alexnet', 'vgg16', 
                          'custom', 'logistic', 'mlp2', 'mlp3']
        assert set(models) == set(expected_models)
        assert len(models) == 10


class TestModelIntegration:
    """Integration tests for all models."""
    
    def test_all_models_via_factory(self):
        """Test that all models can be created and used via factory."""
        rngs = nnx.Rngs(42)
        models = ModelFactory.list_models()
        
        for model_name in models:
            # Create model with appropriate parameters
            if model_name in ['dnn', 'logistic', 'mlp2', 'mlp3']:
                model = ModelFactory.create_model(model_name, input_dim=50, num_classes=3, rngs=rngs)
                # Test with 2D data
                x = jax.random.normal(rngs(), (2, 50))
            elif model_name == 'vgg16':
                # VGG16 requires larger input due to multiple pooling layers
                model = ModelFactory.create_model(model_name, num_classes=3, rngs=rngs)
                x = jax.random.normal(rngs(), (1, 64, 64, 3))  # Larger input for VGG16
            elif model_name in ['alexnet']:
                # AlexNet also benefits from larger input
                model = ModelFactory.create_model(model_name, num_classes=3, rngs=rngs)
                x = jax.random.normal(rngs(), (1, 64, 64, 3))  # Larger input for AlexNet
            elif model_name in ['minionn', 'chameleon']:
                # MiniONN and Chameleon expect 28x28 input with 1 channel
                model = ModelFactory.create_model(model_name, num_classes=3, rngs=rngs)
                x = jax.random.normal(rngs(), (1, 28, 28, 1))
            elif model_name == 'lenet':
                # LeNet expects 32x32 input with 3 channels
                model = ModelFactory.create_model(model_name, num_classes=3, rngs=rngs)
                x = jax.random.normal(rngs(), (1, 32, 32, 3))
            elif model_name == 'custom':
                # CustomCNN expects 16x16 input with 3 channels
                model = ModelFactory.create_model(model_name, num_classes=3, rngs=rngs)
                x = jax.random.normal(rngs(), (1, 16, 16, 3))
            else:
                # Other CNN models
                model = ModelFactory.create_model(model_name, num_classes=3, rngs=rngs)
                x = jax.random.normal(rngs(), (1, 32, 32, 1))  # Default for others
            
            output = model(x)
            
            assert output.shape[-1] == 3  # Check output classes
            assert jnp.all(jnp.isfinite(output))
    
    def test_model_consistency(self):
        """Test that models produce consistent outputs."""
        rngs = nnx.Rngs(42)
        model = ModelFactory.create_model('dnn', input_dim=10, num_classes=2, rngs=rngs)
        
        x = jax.random.normal(rngs(), (1, 10))
        
        # Run multiple times with same input
        output1 = model(x)
        output2 = model(x)
        
        # Should be identical
        np.testing.assert_array_equal(output1, output2)


class TestBackwardPass:
    """Test backward pass and gradient computation for all models."""
    
    def test_dnn_backward_pass(self):
        """Test gradient computation for DNN model."""
        rngs = nnx.Rngs(42)
        model = DNN(input_dim=10, num_classes=3, rngs=rngs)
        
        # Create sample data
        batch_size = 4
        x = jax.random.normal(rngs(), (batch_size, 10))
        y = jax.random.randint(rngs(), (batch_size,), 0, 3)
        
        # Define loss function
        def loss_fn(model, x, y):
            logits = model(x)
            log_probs = jax.nn.log_softmax(logits)
            return -jnp.mean(jnp.sum(jax.nn.one_hot(y, 3) * log_probs, axis=1))
        
        # Compute gradients
        loss_value, grads = nnx.value_and_grad(loss_fn)(model, x, y)
        
        # Check loss is finite
        assert jnp.isfinite(loss_value)
        
        # Check gradients are finite and not zero
        def check_grad_tree(tree):
            leaves, _ = jax.tree_util.tree_flatten(tree)
            return all(jnp.all(jnp.isfinite(leaf)) and jnp.any(leaf != 0) for leaf in leaves)
        
        assert check_grad_tree(grads)
    
    def test_mlp_models_backward_pass(self):
        """Test gradient computation for MLP models."""
        rngs = nnx.Rngs(42)
        
        models_to_test = [
            ('logistic', LogisticRegression, {'input_dim': 20, 'num_classes': 3}),
            ('mlp2', MLP2Layer, {'input_dim': 20, 'hidden_dim': 10, 'num_classes': 3}),
            ('mlp3', MLP3Layer, {'input_dim': 20, 'hidden_dims': (16, 8), 'num_classes': 3}),
        ]
        
        for model_name, model_class, kwargs in models_to_test:
            model = model_class(rngs=rngs, **kwargs)
            
            batch_size = 4
            x = jax.random.normal(rngs(), (batch_size, 20))
            y = jax.random.randint(rngs(), (batch_size,), 0, 3)
            
            def loss_fn(model, x, y):
                logits = model(x)
                log_probs = jax.nn.log_softmax(logits)
                return -jnp.mean(log_probs[jnp.arange(y.shape[0]), y])
            
            loss_value, grads = nnx.value_and_grad(loss_fn)(model, x, y)
            
            assert jnp.isfinite(loss_value), f"Loss not finite for {model_name}"
            
            def check_grad_tree(tree):
                leaves, _ = jax.tree_util.tree_flatten(tree)
                return all(jnp.all(jnp.isfinite(leaf)) and jnp.any(leaf != 0) for leaf in leaves)
            
            assert check_grad_tree(grads), f"Gradients not valid for {model_name}"
    
    def test_regression_backward_pass(self):
        """Test backward pass for regression tasks."""
        rngs = nnx.Rngs(42)
        model = DNN(input_dim=10, num_classes=1, rngs=rngs)  # Single output for regression
        
        batch_size = 4
        x = jax.random.normal(rngs(), (batch_size, 10))
        y = jax.random.normal(rngs(), (batch_size, 1))  # Continuous targets
        
        def loss_fn(model, x, y):
            predictions = model(x)
            return jnp.mean((predictions - y) ** 2)  # MSE loss
        
        loss_value, grads = nnx.value_and_grad(loss_fn)(model, x, y)
        
        assert jnp.isfinite(loss_value)
        
        def check_grad_tree(tree):
            leaves, _ = jax.tree_util.tree_flatten(tree)
            return all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
        
        assert check_grad_tree(grads)


if __name__ == "__main__":
    # Run basic tests
    print("Running basic model tests...")
    
    rngs = nnx.Rngs(42)
    
    # Test DNN forward pass
    dnn_model = DNN(input_dim=100, num_classes=5, rngs=rngs)
    x = jax.random.normal(rngs(), (4, 100))
    output = dnn_model(x)
    print(f"DNN output shape: {output.shape}")
    
    # Test DNN backward pass
    y = jax.random.randint(rngs(), (4,), 0, 5)
    def loss_fn(model, x, y):
        logits = model(x)
        log_probs = jax.nn.log_softmax(logits)
        return -jnp.mean(log_probs[jnp.arange(y.shape[0]), y])
    
    loss_value, grads = nnx.value_and_grad(loss_fn)(dnn_model, x, y)
    print(f"DNN loss: {loss_value:.4f}")
    print(f"DNN gradients computed successfully")
    
    # Test MiniONN
    minionn_model = MiniONN(num_classes=5, rngs=rngs)
    img_data = jax.random.normal(rngs(), (2, 28, 28, 1))
    minionn_output = minionn_model(img_data)
    print(f"MiniONN output shape: {minionn_output.shape}")
    
    # Test factory
    factory_model = ModelFactory.create_model('dnn', input_dim=50, num_classes=3, rngs=rngs)
    factory_x = jax.random.normal(rngs(), (1, 50))
    factory_output = factory_model(factory_x)
    print(f"Factory DNN output shape: {factory_output.shape}")
    
    # Test backward pass via factory
    factory_y = jax.random.randint(rngs(), (1,), 0, 3)
    def factory_loss_fn(model, x, y):
        logits = model(x)
        log_probs = jax.nn.log_softmax(logits)
        return -jnp.mean(log_probs[jnp.arange(y.shape[0]), y])
    
    factory_loss, factory_grads = nnx.value_and_grad(factory_loss_fn)(factory_model, factory_x, factory_y)
    print(f"Factory DNN loss: {factory_loss:.4f}")
    print(f"Factory DNN gradients computed successfully")
    
    print("All model tests completed successfully!")
