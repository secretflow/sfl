# SFL Lite

Next-generation federated learning framework that democratizes distributed AI through natural language interfaces, delivering enterprise-grade performance with consumer-grade simplicity.

## Vision
Build the next-generation federated learning framework that democratizes distributed AI through natural language interfaces, delivering enterprise-grade performance with consumer-grade simplicity.

## Core Design Principles

### 1. Fast Deployment
**"From zero to federated in minutes, not hours"**
- **One-command setup**: Single script deployment across any infrastructure
- **Minimal Dependency**: Minimal Required Dependencies
- **Smart defaults**: Intelligent configuration that works out-of-the-box

### 2. High Performance
**"Production-scale performance without the complexity"**
- **XLA-accelerated**: Leveraging cutting-edge compilation for maximum throughput
- **Efficient communication**: Optimized protocols with automatic compression and batching
- **Intelligent scheduling**: Smart client selection and resource management
- **Hardware optimization**: Native support for GPUs, TPUs, and distributed systems

### 3. AI Friendly
**"Natural language → Federated learning workflows"**
- **Conversational interface**: `"Train a ResNet on 100clients with differential privacy"`
- **Agent Friendly**: Agent should be able to use the framework to satisfy the user's request with ease.

### 4. Observable Security
**"Trust, but verify"**
- **Transparent security**: Detailed logs and metrics for auditability
- **Secure by design**: Built-in security features like differential privacy
- **Audit trail**: Comprehensive tracking of all operations

## Quick Start

### 1. Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and setup the project
```bash
# Clone the repository
git clone https://github.com/secretflow/sfl.git
cd sfl/sfl_lite

# Sync dependencies (creates virtual environment automatically)
uv sync --extra dev
```

This will:
- Create a virtual environment at `.venv/` (Python 3.11)
- Install all dependencies including dev dependencies
- Install the package in editable mode

### 3. Verify installation
```bash
# Test the installation
uv run python -c "import sfl_lite; print('SFL Lite installed successfully!')"
```

## Development

### Quick Commands

```bash
# Format and lint code (run before committing)
./format.sh

# Auto-fix linting issues
./format.sh --fix

# Check formatting without modifying files
./format.sh --check

# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_linear_vertical_plain_fed.py -v
```

For detailed development guidelines, see [DEVELOPMENT.md](DEVELOPMENT.md).
