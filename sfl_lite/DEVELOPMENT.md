# Development Guide for SFL-Lite

This guide provides instructions for developing, testing, and maintaining the SFL-Lite project.

## Table of Contents

- [Getting Started](#getting-started)
- [Code Formatting and Linting](#code-formatting-and-linting)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Contributing Guidelines](#contributing-guidelines)

## Getting Started

### Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

### Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies**:
   ```bash
   cd sfl_lite
   uv sync --extra dev
   ```

3. **Install the package in editable mode**:
   ```bash
   uv pip install -e .
   ```

## Code Formatting and Linting

We use **Ruff** for both code formatting and linting. Ruff is an extremely fast Python linter and formatter that's a drop-in replacement for Black and other tools. A convenient shell script is provided to handle both.

### Quick Start

#### Format code (recommended before committing):
```bash
./format.sh
```

This will:
- Format code with Ruff
- Check for linting issues with Ruff

#### Auto-fix linting issues:
```bash
./format.sh --fix
```

This will:
- Format code with Ruff
- Auto-fix Ruff linting issues (including unsafe fixes)

#### Check without modifying files:
```bash
./format.sh --check
```

This will:
- Check Ruff formatting
- Check Ruff linting
- Exit with error if issues found (useful for CI)

### Manual Commands

If you prefer to run tools manually:

#### Ruff (Formatter & Linter)
```bash
# Format code
uv run ruff format .

# Check formatting without modifying
uv run ruff format --check .

# Check for linting issues
uv run ruff check .

# Auto-fix linting issues
uv run ruff check . --fix

# Auto-fix including unsafe fixes
uv run ruff check . --fix --unsafe-fixes
```

## Running Tests

### Run all tests:
```bash
uv run pytest tests/
```

### Run specific test file:
```bash
uv run pytest tests/test_linear_vertical_plain_fed.py
```

### Run specific test:
```bash
uv run pytest tests/test_linear_vertical_plain_fed.py::TestPlainFederatedLinearRegression::test_plain_federated_train_and_fit_with_mplang
```

### Run with verbose output:
```bash
uv run pytest tests/ -v
```

### Run with coverage:
```bash
uv run pytest tests/ --cov=sfl_lite/ --cov-report=html
```

Then open `htmlcov/index.html` in your browser to view the coverage report.

## Project Structure

```
sfl_lite/
├── sfl_lite/                    # Main package
│   ├── __init__.py
│   ├── ml/                      # Machine learning modules
│   │   └── linear_model/        # Linear models
│   │       ├── plain_fed.py     # Plain federated linear regression
│   │       └── linear_regression/
│   │           └── lib/         # Core implementations
│   │               ├── functional.py
│   │               ├── model.py
│   │               └── template.py
│   └── security/                # Security modules
│       └── aggregation/         # Aggregation protocols
│           └── mp_aggregator.py
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures
│   ├── test_utils.py            # Test utility functions
│   ├── test_linear_vertical_plain_fed.py
│   └── test_mp_aggregator.py
│
├── format.sh                    # Code formatting script
├── pyproject.toml               # Project configuration
└── DEVELOPMENT.md               # This file
```

### Test Structure

- **`conftest.py`**: Contains pytest fixtures that are automatically available to all tests
  - `cluster_spec`: Provides a 3-node cluster specification
  - `simulator`: Provides an initialized MPLang simulator

- **`test_utils.py`**: Contains utility functions for tests
  - `create_test_cluster_spec()`: Create cluster configuration
  - `fetch_from_label_party()`: Fetch cleartext values from label party
  - `create_random_federated_data()`: Generate random test data
  - `create_linear_federated_data()`: Generate data with known linear relationship
  - `create_single_party_data()`: Generate single-party test data

## Contributing Guidelines

### Pre-commit Hooks (Optional but Recommended)

We provide pre-commit hooks to automatically check code quality before each commit:

1. **Install pre-commit**:
   ```bash
   uv pip install pre-commit
   ```

2. **Install the git hooks**:
   ```bash
   cd sfl_lite
   pre-commit install
   ```

3. **Now pre-commit will run automatically on git commit**. You can also run it manually:
   ```bash
   pre-commit run --all-files
   ```

The hooks will:
- Format code with Ruff
- Check and fix linting issues with Ruff
- Remove trailing whitespace
- Fix end-of-file issues
- Check YAML syntax
- Prevent large files from being committed

### Before Committing

1. **Format your code**:
   ```bash
   ./format.sh
   ```

2. **Run tests**:
   ```bash
   uv run pytest tests/
   ```

3. **Fix any issues**:
   ```bash
   ./format.sh --fix
   ```

### Writing Tests

#### Using Fixtures (Recommended)

Fixtures provide clean dependency injection:

```python
def test_example(simulator, aggregator):
    """Test with fixtures via dependency injection."""
    # simulator and aggregator are automatically provided
    result = mp.evaluate(simulator, some_function, aggregator)
    assert result is not None
```

#### Using Utility Functions

Import utility functions from `test_utils.py`:

```python
from .test_utils import (
    create_random_federated_data,
    fetch_from_label_party,
)

def test_example(simulator):
    """Test using utility functions."""
    X, y = mp.evaluate(simulator, create_random_federated_data, 42)
    result = fetch_from_label_party(simulator, X)
    assert result is not None
```

### Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for public functions and classes
- Keep functions focused and small
- Write descriptive test names

### Best Practices

1. **Import Style**:
   - Use relative imports within the test package: `from .test_utils import ...`
   - Use absolute imports for the main package: `from sfl_lite.ml... import ...`

2. **Test Organization**:
   - Group related tests in classes
   - Use descriptive test names that explain what's being tested
   - Add docstrings to test functions

3. **Fixtures**:
   - Use fixtures for setup/teardown logic
   - Define fixtures in `conftest.py` for shared use
   - Use `scope="function"` for most fixtures (default)

4. **Assertions**:
   - Use specific assertions (`assert x == y` instead of `assert x`)
   - Add descriptive error messages: `assert x == y, f"Expected {y}, got {x}"`

## Continuous Integration

The CI pipeline (`.github/workflows/ci.yml`) runs on every push and pull request:

### Format and Lint Job
1. **Ruff Formatting Check**: Ensures all code is properly formatted
2. **Ruff Linting**: Checks code style and potential issues

### Test Job
3. **Unit Tests**: Runs all tests with pytest
4. **Coverage Report**: Generates test coverage and uploads to Codecov

The CI workflow uses:
- **Python 3.11** (matching project requirements)
- **uv** for fast dependency management
- **GitHub Actions cache** for faster builds

Ensure your code passes locally before pushing:

```bash
# Quick check before push
./format.sh --check && uv run pytest tests/
```

You can also test the exact CI commands locally:

```bash
# Format check (as in CI)
cd sfl_lite
uv run ruff format --check .

# Lint check (as in CI)
uv run ruff check .

# Run tests (as in CI)
uv run pytest tests/ -v --tb=short

# Run tests with coverage (as in CI)
uv run pytest tests/ --cov=sfl_lite/ --cov-report=xml --cov-report=term
```

## Troubleshooting

### Import Errors in Tests

If you see `ModuleNotFoundError` in tests:

1. Make sure the package is installed in editable mode:
   ```bash
   uv pip install -e .
   ```

2. Check that `__init__.py` files exist in all package directories

3. Use relative imports in test files: `from .test_utils import ...`

### Formatting or Linting Issues

If you encounter formatting or linting issues:

1. Run the formatter first: `uv run ruff format .`
2. Then fix linting issues: `uv run ruff check . --fix`
3. For stubborn issues, use: `./format.sh --fix`

### Test Failures

If tests fail unexpectedly:

1. Check if you're using the correct Python version (3.11)
2. Ensure all dependencies are installed: `uv sync --extra dev`
3. Clear pytest cache: `rm -rf .pytest_cache __pycache__`

## Additional Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pytest Documentation](https://docs.pytest.org/)
- [uv Documentation](https://github.com/astral-sh/uv)

## Questions?

For questions or issues, please open an issue on the GitHub repository.
