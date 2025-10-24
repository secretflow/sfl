# SFL Lite Project Guide

## Project Basics

- **Tech Stack**: Python 3.10 + JAX + Flax NNX + MPLang
- **Runtime**: Python 3.10 with UV package manager support
- **Core Framework**: Federated learning framework based on JAX/Flax with MPLang for distributed computing
- **License**: Apache-2.0

## Common Commands

### Environment Setup
```bash
# Create and activate environment
uv venv --python 3.10
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -e .[dev] --upgrade
```

### Development Commands
```bash
# Run tests
pytest

# Code formatting
ruff check sfl_lite/
ruff format sfl_lite/

# Type checking
mypy sfl_lite/

# Install development dependencies
uv pip install -e .[dev] --upgrade
```

### Verify Installation
```bash
python -c "import sfl_lite; print('SFL Lite installed successfully!')"
```

## Code Standards

### File Organization
- **Package Structure**: `sfl_lite/` main package containing `ml/`, `security/` submodules
- **Test Files**: `tests/` directory with `test_*.py` naming format
- **Module Imports**: Use combination of relative and absolute imports

### Naming Conventions
- **Class Names**: PascalCase (e.g., `LinearModel`, `DNN`, `SLModel`)
- **Function Names**: snake_case (e.g., `grad_compute`, `mse_loss`)
- **Variable Names**: snake_case (e.g., `label_party`, `base_models`)
- **Constants**: UPPER_SNAKE_CASE
- **Private Methods**: Prefix with underscore `_`

### Code Style
- **Line Length**: 88 characters (Black configuration)
- **Import Order**: Standard library -> Third-party -> Local imports
- **Type Annotations**: Required, support JAX types
- **Docstrings**: Google-style docstring format

## Core Architecture

### Main Modules
1. **ml/linear/**: Linear model implementations
   - `linear_model.py`: Core linear/logistic regression models
   - `plain_fed_linear_model.py`: Federated linear models
   - `linear_regression_vertical.py`: Vertical linear regression

2. **ml/nn/**: Neural network models
   - `models.py`: Basic neural network architectures (DNN, etc.)
   - `sl/`: Split learning related models
   - `sl_model.py`: Split learning model core

3. **security/aggregation/**: Secure aggregation
   - `aggregator.py`: Aggregation base classes
   - `mp_aggregator.py`: MPLang aggregation implementation

### Key Design Patterns
- **Data Classes**: Use `@dataclass` for model structure definitions
- **Enum Classes**: Use `@unique` and `Enum` for constants
- **MPLang Integration**: Use `@mpd.function` decorators for distributed computing
- **JAX Functional**: Pure function design supporting `jax.jit` compilation

## Testing Strategy

### Testing Framework
- **Unit Tests**: Use pytest
- **Test Path**: `tests/` directory
- **Test Files**: `test_*.py` format
- **Test Classes**: `Test*` naming

### Testing Patterns
- **Simulator Tests**: Use `mplang.Simulator.simple(n)` for multi-party simulation
- **Data Generation**: Use JAX random number generation for test data
- **Assertions**: Use `jnp.allclose` for numerical comparisons

### Test Commands
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_linear_model.py

# Verbose output
pytest -v --tb=short
```

## Development Notes

### Dependency Management
- **MPLang**: Currently uses Git dependency, may be published to PyPI in future
- **JAX/Flax**: Strict version requirements (flax>=0.10.4)
- **Python Version**: Only supports Python 3.10 (<3.11)

### Performance Optimization
- **JIT Compilation**: Use `@jax.jit` decorators to optimize computation
- **Vectorization**: Prioritize JAX vector operations
- **Memory Management**: Pay attention to JAX memory model

### Security Considerations
- **Federated Learning**: Data never leaves local devices
- **Secure Aggregation**: Built-in differential privacy support
- **Audit Trail**: Operations are traceable

### MPLang Integration
- **Device Decorators**: Use `@mpd.function` to mark distributed functions
- **Context Management**: Use `mplang.set_ctx(sim)` to set simulator
- **Data Retrieval**: Use `mplang.fetch()` to get results

## Debugging Tips

### Common Debug Patterns
```python
# Print debug information
print("DEBUG:", variable)

# Get MPLang object values
fetched = mplang.fetch(None, mp_object)

# Check model state
graph_def, state = nnx.split(model)
```

### Error Handling
- **Input Validation**: Check input data format and dimensions
- **Exception Handling**: Use try-except for distributed computing exceptions
- **Logging**: Record key operations and error messages

## Project Documentation

### Core Documentation
- **README.md**: Project introduction and quick start
- **mplang_error_guide.md**: MPLang error handling guide
- **Code Documentation**: Detailed docstrings in each module

### External Resources
- **Homepage**: https://github.com/secretflow/sfl
- **Documentation**: https://sfl.readthedocs.io
- **Dependencies**: MPLang (https://github.com/secretflow/mplang)

## Commit Guidelines

### Code Review Checklist
- [ ] Run `ruff check sfl_lite/` to check code style
- [ ] Run `pytest` to ensure tests pass
- [ ] Run `mypy sfl_lite/` for type checking
- [ ] Update relevant documentation

### Pre-commit Requirements
```bash
# Code formatting
ruff format sfl_lite/

# Run tests
pytest

# Check types
mypy sfl_lite/
```