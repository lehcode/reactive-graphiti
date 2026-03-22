#!/bin/bash
# Graphiti LiteLLM Edition Quality Benchmark Script
# Performs static analysis and fails if quality thresholds are exceeded

set -uo pipefail

# Colors for report
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Quality Thresholds
MAX_RUFF_ERRORS=0
MAX_TYPECHECK_ERRORS=0
MAX_SHELLCHECK_WARNINGS=5

# Directories to scan
PYTHON_DIRS="graphiti_core server mcp_server"
SHELL_SCRIPTS=$(find scripts -name "*.sh" 2>/dev/null || true)

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}       GRAPHITI CODE QUALITY BENCHMARK          ${NC}"
echo -e "${BLUE}==================================================${NC}"

# 1. ShellCheck
SC_ERRORS=0
SC_WARNINGS=0
if [ -n "$SHELL_SCRIPTS" ]; then
    echo -e "\n${BLUE}[1/3] Running ShellCheck...${NC}"
    SC_OUTPUT=$(shellcheck -f json $SHELL_SCRIPTS)
    SC_ERRORS=$(echo "$SC_OUTPUT" | jq '[.[] | select(.level == "error")] | length')
    SC_WARNINGS=$(echo "$SC_OUTPUT" | jq '[.[] | select(.level == "warning")] | length')

    if [ "$SC_ERRORS" -gt 0 ]; then
        echo -e "${RED}✘ Failed: $SC_ERRORS ShellCheck errors found.${NC}"
        shellcheck "$SHELL_SCRIPTS"
    elif [ "$SC_WARNINGS" -gt "$MAX_SHELLCHECK_WARNINGS" ]; then
        echo -e "${RED}✘ Failed: $SC_WARNINGS ShellCheck warnings (Threshold: $MAX_SHELLCHECK_WARNINGS).${NC}"
        shellcheck "$SHELL_SCRIPTS"
    else
        echo -e "${GREEN}✔ Passed: $SC_ERRORS errors, $SC_WARNINGS warnings.${NC}"
    fi
else
    echo -e "\n${BLUE}[1/3] Skipping ShellCheck (no scripts found).${NC}"
fi

# 2. Ruff (Linting & Formatting)
echo -e "\n${BLUE}[2/3] Running Ruff Check...${NC}"
RUFF_OUTPUT=$(ruff check $PYTHON_DIRS 2>&1 || true)
# Extract number from "Found X errors."
RUFF_ERRORS=$(echo "$RUFF_OUTPUT" | grep -o "Found [0-9]* error" | head -n 1 | awk '{print $2}' || echo 0)
if [ -z "$RUFF_ERRORS" ]; then RUFF_ERRORS=0; fi

if [ "$RUFF_ERRORS" -gt "$MAX_RUFF_ERRORS" ]; then
    echo -e "${RED}✘ Failed: $RUFF_ERRORS Ruff linting errors found.${NC}"
    echo "$RUFF_OUTPUT"
else
    echo -e "${GREEN}✔ Passed: No Ruff linting errors.${NC}"
fi

# 3. Pyright (Type Checking)
echo -e "\n${BLUE}[3/3] Running Pyright...${NC}"
PYRIGHT_OUTPUT=$(pyright $PYTHON_DIRS 2>&1 || true)
PYRIGHT_ERRORS=$(echo "$PYRIGHT_OUTPUT" | grep -c "error:" || echo 0)

if [ "$PYRIGHT_ERRORS" -gt "$MAX_TYPECHECK_ERRORS" ]; then
    echo -e "${RED}✘ Failed: $PYRIGHT_ERRORS type-checking errors found.${NC}"
    echo "$PYRIGHT_OUTPUT"
else
    echo -e "${GREEN}✔ Passed: No type-checking errors.${NC}"
fi

# Summary Report
echo -e "\n${BLUE}==================================================${NC}"
echo -e "                SUMMARY REPORT                    "
echo -e "${BLUE}==================================================${NC}"
printf "%-25s | %-10s | %-10s\n" "Metric" "Found" "Threshold"
echo "--------------------------------------------------"
printf "%-25s | %-10s | %-10s\n" "ShellCheck Warnings" "$SC_WARNINGS" "$MAX_SHELLCHECK_WARNINGS"
printf "%-25s | %-10s | %-10s\n" "Ruff Errors" "$RUFF_ERRORS" "$MAX_RUFF_ERRORS"
printf "%-25s | %-10s | %-10s\n" "Type-Check Errors" "$PYRIGHT_ERRORS" "$MAX_TYPECHECK_ERRORS"
echo "--------------------------------------------------"

# Final Verdict
if [ "$SC_ERRORS" -gt 0 ] || \
   [ "$SC_WARNINGS" -gt "$MAX_SHELLCHECK_WARNINGS" ] || \
   [ "$RUFF_ERRORS" -gt "$MAX_RUFF_ERRORS" ] || \
   [ "$PYRIGHT_ERRORS" -gt "$MAX_TYPECHECK_ERRORS" ]; then
    echo -e "${RED}RESULT: QUALITY BENCHMARK FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}RESULT: QUALITY BENCHMARK PASSED${NC}"
    exit 0
fi
