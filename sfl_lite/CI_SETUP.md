# CI Configuration for SFL-Lite

## Overview

SFL-Lite uses a **dual CI system**:
- **GitHub Actions** (`.github/workflows/sfl-lite-ci.yml`) - Formatting and linting checks with **Ruff**
- **CircleCI** (`.circleci/continue-config.yml`) - Unit tests execution

This document describes both systems and how they work together.

---

## GitHub Actions - Code Quality Checks

### Workflow File

**Location**: `.github/workflows/sfl-lite-ci.yml` (at repository root)

**Note**: GitHub Actions workflows must be at the repository root, not in subdirectories. The workflow uses `working-directory: sfl_lite` to target the sfl_lite package.

**Important**: The repository also has a `black.yml` workflow that uses Black for the entire SFL repository. To prevent conflicts:
- **Black is configured to exclude `sfl_lite/`** (in root `pyproject.toml`)
- The `sfl-lite-ci.yml` workflow uses **Ruff** for sfl_lite code
- Both workflows can run simultaneously without conflicts

### Trigger Events

The workflow runs on:
- **Push** to `main` branch (only when `sfl_lite/**` files change)
- **Pull requests** to `main` branch (only when `sfl_lite/**` files change)

**Path filtering** ensures the workflow only runs when sfl_lite code is modified.

### Jobs

#### 1. Format and Lint Job

**Purpose**: Ensure code quality and consistency

**Steps**:
1. Checkout code
2. Install uv (with caching)
3. Set up Python 3.11
4. Install dependencies
5. Check Ruff formatting (`ruff format --check`)
6. Run Ruff linter (`ruff check`)

**Fast Failure**: If formatting or linting fails, the job fails immediately

#### 2. Test Job

**Purpose**: Verify code functionality and coverage

**Steps**:
1. Checkout code
2. Install uv (with caching)
3. Set up Python 3.11
4. Install dependencies (including dev dependencies)
5. Run unit tests with pytest
6. Generate coverage report
7. Upload coverage to Codecov

**Matrix Strategy**: Currently runs on Python 3.11 (can be expanded for multi-version testing)

### Key Features

- **Fast**: Uses `uv` for dependency management (10-100x faster than pip)
- **Cached**: Dependencies are cached to speed up subsequent runs
- **Comprehensive**: Checks formatting, linting, tests, and coverage
- **Modern**: Uses latest GitHub Actions (v4/v5)

## Pre-commit Hooks (`.pre-commit-config.yaml`)

### Purpose

Automatically check code quality before each commit, preventing CI failures.

### Hooks

#### Ruff Hooks
1. **ruff-format**: Formats code automatically
2. **ruff**: Lints and auto-fixes issues

#### Pre-commit Standard Hooks
3. **trailing-whitespace**: Removes trailing whitespace
4. **end-of-file-fixer**: Ensures files end with a newline
5. **check-yaml**: Validates YAML syntax
6. **check-added-large-files**: Prevents large files from being committed
7. **check-merge-conflict**: Detects merge conflict markers
8. **mixed-line-ending**: Normalizes line endings

### Installation

```bash
# Install pre-commit
uv pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Comparison: Before vs After

### Before (with Black)

```yaml
# Old approach (if it existed)
- name: Check Black formatting
  run: black --check --diff sfl_lite/ tests/

- name: Run Ruff linter
  run: ruff check .
```

### After (with Ruff only)

```yaml
# New unified approach
- name: Check Ruff formatting
  run: ruff format --check .

- name: Run Ruff linter
  run: ruff check .
```

**Benefits**:
- ✅ Single tool for both jobs
- ✅ Faster execution (Rust-based)
- ✅ Consistent configuration
- ✅ Easier maintenance

## Local Testing

### Test CI Steps Locally

```bash
cd sfl_lite

# 1. Format check (Job 1, Step 5)
uv run ruff format --check .

# 2. Lint check (Job 1, Step 6)
uv run ruff check .

# 3. Run tests (Job 2, Step 5)
uv run pytest tests/ -v --tb=short

# 4. Coverage (Job 2, Step 6)
uv run pytest tests/ --cov=sfl_lite/ --cov-report=xml --cov-report=term
```

### Quick Pre-push Check

```bash
# One-liner to verify everything
./format.sh --check && uv run pytest tests/
```

---

## CircleCI - Unit Tests

### Configuration Files

**Location**: `.circleci/continue-config.yml` (at repository root)

CircleCI handles unit test execution for the entire SFL repository, including sfl_lite.

### How It Works

1. **Path filtering** (`.circleci/config.yml`): Detects changes to `sfl_lite/.*` files
2. **Test execution** (`.circleci/continue-config.yml`): Runs the `linux_sfl_lite_test` job
3. **Dependencies**: Uses `uv sync` with Python 3.11
4. **Test command**: `pytest --cov=sfl_lite/ --cov-report=xml tests/`

### Key Features

- **Caching**: uv cache and virtual environment cached for faster runs
- **Python 3.11**: Matches project requirements
- **Coverage reporting**: Generates XML coverage reports
- **Resource optimization**: Uses appropriate instance sizes

**You don't need to modify CircleCI config** - it automatically runs tests when sfl_lite files change.

---

## Black Exclusion Configuration

Since the main SFL repository uses Black for formatting, but sfl_lite uses Ruff, we need to ensure Black ignores the `sfl_lite/` directory.

### Configuration Location

**File**: `/pyproject.toml` (at repository root)

```toml
[tool.black]
line-length = 88
target-version = ['py310']
include = '\.pyi?$'
extend-exclude = '''
/(
    .*_pb2\.py
  | sfl_lite
)/
'''
```

### How It Works

1. **Black's `extend-exclude`**: Tells Black to skip `sfl_lite/` directory
2. **Repository-wide Black check** (`black.yml`): Runs but ignores sfl_lite
3. **SFL-Lite Ruff check** (`sfl-lite-ci.yml`): Runs only for sfl_lite

This ensures:
- ✅ No formatting conflicts between Black and Ruff
- ✅ Both CI workflows can run without interference
- ✅ Each package uses its preferred formatter

### Verification

Test that Black ignores sfl_lite:

```bash
cd /path/to/sfl

# Check what Black would format (should not include sfl_lite/)
black --check --verbose sfl/ secretflow_fl/ | grep -v sfl_lite

# Or explicitly test the exclude
black --check --diff . 2>&1 | grep "sfl_lite" || echo "✅ sfl_lite is excluded"
```

---

## CI Configuration Files

### GitHub Actions
- **File**: `.github/workflows/sfl-lite-ci.yml`
- **Language**: GitHub Actions YAML
- **Purpose**: Code quality checks (formatting & linting)
- **Runs on**: ubuntu-latest
- **Python**: 3.11

### CircleCI
- **File**: `.circleci/continue-config.yml`
- **Language**: CircleCI YAML
- **Purpose**: Unit tests execution
- **Runs on**: Linux executor (Docker)
- **Python**: 3.11

### Pre-commit Config
- **File**: `sfl_lite/.pre-commit-config.yaml`
- **Language**: Pre-commit YAML
- **Purpose**: Local git hooks
- **Ruff version**: v0.13.1

## Integration with Development Workflow

### Developer Flow

1. **Write code** → Make changes
2. **Pre-commit** → Auto-format and lint on commit (if installed)
3. **Manual check** → Run `./format.sh --check` before push
4. **Push** → Both CI systems validate code
5. **PR** → All CI checks must pass before merge

### Dual CI Flow

```
Push/PR to main (sfl_lite/** changes)
    │
    ├─→ GitHub Actions
    │   ├─→ Ruff Format Check → ✅ or ❌
    │   └─→ Ruff Lint Check   → ✅ or ❌
    │
    └─→ CircleCI
        ├─→ Install Dependencies
        ├─→ Run Unit Tests      → ✅ or ❌
        └─→ Generate Coverage   → Report
```

**Both must pass** for the PR to be mergeable.

## Benefits of This Setup

### For Developers
- **Fast feedback**: Pre-commit catches issues before CI
- **Consistent formatting**: No debates about style
- **Easy to use**: `./format.sh` handles everything
- **Clear errors**: Ruff provides helpful messages

### For CI
- **Fast builds**: uv + caching = seconds, not minutes
- **Reliable**: Deterministic formatting and linting
- **Modern**: Latest tooling and best practices
- **Maintainable**: Single tool (Ruff) instead of multiple

### For the Project
- **High quality**: Automated checks ensure consistency
- **Low overhead**: Minimal CI time and cost
- **Good DX**: Developer experience is smooth
- **Future-proof**: Ruff is actively maintained

## Troubleshooting

### CI Fails: Formatting Check

**Error**: `Would reformat: file.py`

**Solution**:
```bash
./format.sh  # Format locally and commit
```

### CI Fails: Linting Check

**Error**: `Ruff found issues: E501, F401, etc.`

**Solution**:
```bash
./format.sh --fix  # Auto-fix and commit
```

### CI Fails: Tests

**Error**: Test failures or errors

**Solution**:
```bash
uv run pytest tests/ -v  # Run locally to debug
```

### Pre-commit Hook Fails

**Error**: Hook failures prevent commit

**Solution**:
```bash
# Bypass temporarily (not recommended)
git commit --no-verify

# Or fix the issues
pre-commit run --all-files
```

## GitHub Actions Setup

### Required Secrets (for Codecov)

If you want coverage reporting, add to repository secrets:
- `CODECOV_TOKEN`: Your Codecov token (optional, public repos work without)

### Branch Protection Rules (Recommended)

For `main` branch:
- ✅ Require status checks to pass
  - ✅ Format and Lint
  - ✅ Test
- ✅ Require branches to be up to date
- ✅ Require linear history (optional)

## Future Enhancements

Potential improvements:
1. **Multi-Python testing**: Test on Python 3.11, 3.12, etc.
2. **OS matrix**: Test on Ubuntu, macOS, Windows
3. **Performance tests**: Add benchmark tests
4. **Security scanning**: Add dependency scanning
5. **Release automation**: Auto-publish on tags

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Ruff GitHub Actions](https://github.com/astral-sh/ruff-pre-commit)
- [uv GitHub Action](https://github.com/astral-sh/setup-uv)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Codecov GitHub Action](https://github.com/codecov/codecov-action)

---

**Created**: January 2025  
**Maintainer**: SFL-Lite Team  
**Status**: Active
