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

## Installation

```bash
# Install from source
cd sfl_lite
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import sfl_lite as sfl

# Create a federated learning setup with natural language
config = sfl.configure("Train a ResNet on 100 clients with differential privacy")

# Start training
sfl.train(config)
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black sfl_lite/
ruff check sfl_lite/
```

