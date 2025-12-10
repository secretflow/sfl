#!/usr/bin/env bash
# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== SFL-Lite Code Formatter ===${NC}"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed${NC}"
    echo "Please install uv: https://github.com/astral-sh/uv"
    exit 1
fi

# Parse command line arguments
FIX=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)
            FIX=true
            shift
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./format.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fix     Auto-fix issues (includes unsafe fixes)"
            echo "  --check   Only check, don't modify files"
            echo "  --help    Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./format.sh           # Format code and check linting with Ruff"
            echo "  ./format.sh --fix     # Format code and auto-fix Ruff issues"
            echo "  ./format.sh --check   # Only check formatting, don't modify"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$CHECK_ONLY" = true ]; then
    echo -e "${YELLOW}→ Checking code formatting and linting (read-only mode)...${NC}"
    echo ""
    
    # Check Ruff formatting
    echo -e "${YELLOW}[1/2] Checking Ruff formatting...${NC}"
    if uv run ruff format --check .; then
        echo -e "${GREEN}✓ Ruff formatting check passed${NC}"
    else
        echo -e "${RED}✗ Ruff formatting check failed${NC}"
        echo -e "${YELLOW}Run './format.sh' to auto-format${NC}"
        exit 1
    fi
    echo ""
    
    # Check Ruff linting
    echo -e "${YELLOW}[2/2] Checking Ruff linting...${NC}"
    if uv run ruff check .; then
        echo -e "${GREEN}✓ Ruff linting check passed${NC}"
    else
        echo -e "${RED}✗ Ruff linting check failed${NC}"
        echo -e "${YELLOW}Run './format.sh --fix' to auto-fix${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}✓ All checks passed!${NC}"
    
elif [ "$FIX" = true ]; then
    echo -e "${YELLOW}→ Formatting and fixing code (with auto-fix)...${NC}"
    echo ""
    
    # Format with Ruff
    echo -e "${YELLOW}[1/2] Running Ruff formatter...${NC}"
    uv run ruff format .
    echo -e "${GREEN}✓ Ruff formatting complete${NC}"
    echo ""
    
    # Fix with Ruff (including unsafe fixes)
    echo -e "${YELLOW}[2/2] Running Ruff linter with auto-fix...${NC}"
    uv run ruff check . --fix --unsafe-fixes
    echo -e "${GREEN}✓ Ruff linting and fixes complete${NC}"
    
    echo ""
    echo -e "${GREEN}✓ All formatting and fixes applied!${NC}"
    
else
    echo -e "${YELLOW}→ Formatting code...${NC}"
    echo ""
    
    # Format with Ruff
    echo -e "${YELLOW}[1/2] Running Ruff formatter...${NC}"
    uv run ruff format .
    echo -e "${GREEN}✓ Ruff formatting complete${NC}"
    echo ""
    
    # Check with Ruff (no auto-fix)
    echo -e "${YELLOW}[2/2] Running Ruff linter...${NC}"
    if uv run ruff check .; then
        echo -e "${GREEN}✓ Ruff linting passed${NC}"
    else
        echo -e "${YELLOW}⚠ Ruff found issues${NC}"
        echo -e "${YELLOW}Run './format.sh --fix' to auto-fix${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}✓ Formatting complete!${NC}"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
