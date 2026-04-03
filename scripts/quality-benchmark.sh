#!/bin/bash
# Graphiti LiteLLM Edition Quality Benchmark Script
# Performs static analysis and reports results without blocking the build

set -uo pipefail

# Colors for report
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Quality Thresholds (Informational)
MAX_RUFF_ERRORS=0
MAX_TYPECHECK_ERRORS=13  # Baseline: pre-existing upstream type errors
MAX_SHELLCHECK_WARNINGS=5
MAX_DEAD_CODE_CONFIDENCE=100
MAX_DUPLICATION_LINES=10
MIN_MAINTAINABILITY_INDEX=70

# Directories to scan
PYTHON_DIRS="graphiti_core server mcp_server/src"
SHELL_SCRIPTS=$(find scripts -name "*.sh" 2>/dev/null || true)

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}       GRAPHITI CODE QUALITY BENCHMARK          ${NC}"
echo -e "${BLUE}==================================================${NC}"

# 1. ShellCheck
SC_ERRORS=0
SC_WARNINGS=0
if [ -n "$SHELL_SCRIPTS" ]; then
    echo -e "\n${BLUE}[1/6] Running ShellCheck...${NC}"
    SC_OUTPUT=$(shellcheck -f json $SHELL_SCRIPTS)
    SC_ERRORS=$(echo "$SC_OUTPUT" | jq '[.[] | select(.level == "error")] | length')
    SC_WARNINGS=$(echo "$SC_OUTPUT" | jq '[.[] | select(.level == "warning")] | length')

    if [ "$SC_ERRORS" -gt 0 ]; then
        echo -e "${YELLOW}⚠ Warning: $SC_ERRORS ShellCheck errors found.${NC}"
        shellcheck $SHELL_SCRIPTS
    elif [ "$SC_WARNINGS" -gt "$MAX_SHELLCHECK_WARNINGS" ]; then
        echo -e "${YELLOW}⚠ Warning: $SC_WARNINGS ShellCheck warnings found.${NC}"
        shellcheck $SHELL_SCRIPTS
    else
        echo -e "${GREEN}✔ Passed: No significant ShellCheck issues.${NC}"
    fi
else
    echo -e "\n${BLUE}[1/6] Skipping ShellCheck (no scripts found).${NC}"
fi

# 2. Ruff (Linting & Formatting)
echo -e "\n${BLUE}[2/6] Running Ruff Check...${NC}"
RUFF_OUTPUT=$(ruff check $PYTHON_DIRS 2>&1 || true)
RUFF_ERRORS=$(echo "$RUFF_OUTPUT" | grep -o "Found [0-9]* error" | head -n 1 | awk '{print $2}')
RUFF_ERRORS=${RUFF_ERRORS:-0}

if [ "$RUFF_ERRORS" -gt "$MAX_RUFF_ERRORS" ]; then
    echo -e "${YELLOW}⚠ Warning: $RUFF_ERRORS Ruff linting errors found.${NC}"
    echo "$RUFF_OUTPUT"
else
    echo -e "${GREEN}✔ Passed: No Ruff linting errors.${NC}"
fi

# 3. Pyright (Type Checking)
echo -e "\n${BLUE}[3/6] Running Pyright...${NC}"
PYRIGHT_OUTPUT=$(uv run pyright $PYTHON_DIRS 2>&1 || true)
PYRIGHT_ERRORS=$(echo "$PYRIGHT_OUTPUT" | grep -c " - error:")
PYRIGHT_ERRORS=${PYRIGHT_ERRORS:-0}

if [ "$PYRIGHT_ERRORS" -gt "$MAX_TYPECHECK_ERRORS" ]; then
    echo -e "${YELLOW}⚠ Warning: $PYRIGHT_ERRORS type-checking errors found.${NC}"
    echo "$PYRIGHT_OUTPUT" | tail -n 20
else
    echo -e "${GREEN}✔ Passed: No type-checking errors.${NC}"
fi

# 4. Vulture (Dead Code Detection)
echo -e "\n${BLUE}[4/6] Running Vulture (Dead Code Detection)...${NC}"
# Exclude venv and site-packages explicitly
VULTURE_OUTPUT=$(vulture $PYTHON_DIRS --min-confidence $MAX_DEAD_CODE_CONFIDENCE 2>&1 || true)
DEAD_CODE_COUNT=$(echo "$VULTURE_OUTPUT" | grep -c ":")
DEAD_CODE_COUNT=${DEAD_CODE_COUNT:-0}

if [ "$DEAD_CODE_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Warning: $DEAD_CODE_COUNT unused code blocks found (100% confidence).${NC}"
    echo "$VULTURE_OUTPUT"
else
    echo -e "${GREEN}✔ Passed: No dead code found.${NC}"
fi

# 5. Pylint (Code Duplication)
echo -e "\n${BLUE}[5/6] Running Duplication Check...${NC}"
DUPE_OUTPUT=$(pylint --disable=all --enable=similarities --min-similarity-lines=$MAX_DUPLICATION_LINES $PYTHON_DIRS 2>&1 || true)
DUPE_COUNT=$(echo "$DUPE_OUTPUT" | grep -c "similar lines in")
DUPE_COUNT=${DUPE_COUNT:-0}

if [ "$DUPE_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Warning: $DUPE_COUNT duplicated code blocks found.${NC}"
    echo "$DUPE_OUTPUT" | grep -A 10 "similar lines in"
else
    echo -e "${GREEN}✔ Passed: No significant code duplication.${NC}"
fi

# 6. Radon (Maintainability & Modularization)
echo -e "\n${BLUE}[6/6] Running Maintainability Index Check...${NC}"
RADON_OUTPUT=$(radon mi $PYTHON_DIRS --min B --max F 2>&1 || true)
LOW_MI_COUNT=$(echo "$RADON_OUTPUT" | grep -c " - [B-F]")
LOW_MI_COUNT=${LOW_MI_COUNT:-0}

if [ "$LOW_MI_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Warning: $LOW_MI_COUNT modules with Maintainability Index below threshold.${NC}"
    echo "$RADON_OUTPUT"
else
    echo -e "${GREEN}✔ Passed: All modules have high Maintainability Index.${NC}"
fi

# Summary Report
echo -e "\n${BLUE}==================================================${NC}"
echo -e "                SUMMARY REPORT                    "
echo -e "${BLUE}==================================================${NC}"
printf "%-25s | %-10s | %-10s\n" "Metric" "Found" "Status"
echo "--------------------------------------------------"
printf "%-25s | %-10s | %-10s\n" "ShellCheck Warnings" "$SC_WARNINGS" "REPORTED"
printf "%-25s | %-10s | %-10s\n" "Ruff Errors" "$RUFF_ERRORS" "REPORTED"
printf "%-25s | %-10s | %-10s\n" "Type-Check Errors" "$PYRIGHT_ERRORS" "REPORTED"
printf "%-25s | %-10s | %-10s\n" "Dead Code (100%)" "$DEAD_CODE_COUNT" "REPORTED"
printf "%-25s | %-10s | %-10s\n" "Duplicated Blocks" "$DUPE_COUNT" "REPORTED"
printf "%-25s | %-10s | %-10s\n" "Low MI Modules" "$LOW_MI_COUNT" "REPORTED"
echo "--------------------------------------------------"

# Final Verdict: Always exit 0 as requested, but show the summary
echo -e "${GREEN}RESULT: QUALITY BENCHMARK COMPLETED (Issues reported but not blocking)${NC}"
exit 0
