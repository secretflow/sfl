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

### 1. Create a new uv environment
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a new environment with Python 3.10
uv venv sfl-lite-env --python 3.10
source sfl-lite-env/bin/activate  # On Windows: sfl-lite-env\Scripts\activate
```

### 2. Sync all required packages
```bash
# Clone the repository
git clone https://github.com/secretflow/sfl.git
cd sfl/sfl_lite

# Sync dependencies
uv sync --active --extra dev
```

### 3. Verify installation
```bash
# Test the installation
python -c "import sfl_lite; print('SFL Lite installed successfully!')"

# Run a simple example
sfl-lite --help
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
ruff check sfl_lite/
```
