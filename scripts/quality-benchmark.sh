#!/bin/bash
# Graphiti LiteLLM Edition Quality Benchmark Script
# Reports issues and fails build if any thresholds are exceeded

set -uo pipefail

# Track overall failure status
BUILD_FAILED=0

# Colors for report
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Quality Thresholds (Informational)
MAX_RUFF_ERRORS=0
MAX_TYPECHECK_ERRORS=0
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
    SC_OUTPUT=$(shellcheck -e SC1091,SC2155,SC2086,SC2034 -f json $SHELL_SCRIPTS)
    SC_ERRORS=$(echo "$SC_OUTPUT" | jq '[.[] | select(.level == "error")] | length')
    SC_WARNINGS=$(echo "$SC_OUTPUT" | jq '[.[] | select(.level == "warning")] | length')

    if [ "$SC_ERRORS" -gt 0 ]; then
        echo -e "${RED}✖ Failed: $SC_ERRORS ShellCheck error(s)${NC}"
        echo "$SC_OUTPUT" | jq -r '.[] | "[\(.level)] \(.file):\(.line):\(.code) \(.message)"'
        BUILD_FAILED=1
    elif [ "$SC_WARNINGS" -gt "$MAX_SHELLCHECK_WARNINGS" ]; then
        echo -e "${RED}✖ Failed: $SC_WARNINGS ShellCheck warning(s), exceeds limit of $MAX_SHELLCHECK_WARNINGS${NC}"
        echo "$SC_OUTPUT" | jq -r '.[] | "[\(.level)] \(.file):\(.line):\(.code) \(.message)"'
        BUILD_FAILED=1
    else
        echo -e "${GREEN}✔ Passed: ShellCheck clean.${NC}"
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
    echo -e "${RED}✖ Failed: $RUFF_ERRORS Ruff linting error(s)${NC}"
    echo "$RUFF_OUTPUT"
    BUILD_FAILED=1
else
    echo -e "${GREEN}✔ Passed: No Ruff linting errors.${NC}"
fi

# 3. Pyright (Type Checking)
echo -e "\n${BLUE}[3/6] Running Pyright...${NC}"
PYRIGHT_OUTPUT=$(uv run pyright $PYTHON_DIRS 2>&1 || true)
PYRIGHT_ERRORS=$(echo "$PYRIGHT_OUTPUT" | grep -c " - error:")
PYRIGHT_ERRORS=${PYRIGHT_ERRORS:-0}

if [ "$PYRIGHT_ERRORS" -gt "$MAX_TYPECHECK_ERRORS" ]; then
    echo -e "${RED}✖ Failed: $PYRIGHT_ERRORS type-checking error(s)${NC}"
    echo "$PYRIGHT_OUTPUT"
    BUILD_FAILED=1
else
    echo -e "${GREEN}✔ Passed: No type-checking errors.${NC}"
fi

# 4. Vulture (Dead Code Detection)
echo -e "\n${BLUE}[4/6] Running Vulture (Dead Code Detection)...${NC}"
# Exclude venv and site-packages explicitly
VULTURE_OUTPUT=$(vulture $PYTHON_DIRS scripts/vulture_whitelist.py --min-confidence $MAX_DEAD_CODE_CONFIDENCE 2>&1 || true)
DEAD_CODE_COUNT=$(echo "$VULTURE_OUTPUT" | grep -c ":")
DEAD_CODE_COUNT=${DEAD_CODE_COUNT:-0}

if [ "$DEAD_CODE_COUNT" -gt 0 ]; then
    echo -e "${RED}✖ Failed: $DEAD_CODE_COUNT unused code block(s) at 100% confidence${NC}"
    echo "$VULTURE_OUTPUT"
    BUILD_FAILED=1
else
    echo -e "${GREEN}✔ Passed: No dead code found.${NC}"
fi

# 5. Pylint (Code Duplication)
echo -e "\n${BLUE}[5/6] Running Duplication Check...${NC}"
DUPE_OUTPUT=$(pylint --disable=all --enable=similarities --min-similarity-lines=$MAX_DUPLICATION_LINES $PYTHON_DIRS 2>&1 || true)
DUPE_COUNT=$(echo "$DUPE_OUTPUT" | grep -c "similar lines in")
DUPE_COUNT=${DUPE_COUNT:-0}

if [ "$DUPE_COUNT" -gt 0 ]; then
    echo -e "${RED}✖ Failed: $DUPE_COUNT duplicated code block(s)${NC}"
    echo "$DUPE_OUTPUT" | grep -A 10 "similar lines in"
    BUILD_FAILED=1
else
    echo -e "${GREEN}✔ Passed: No significant code duplication.${NC}"
fi

# 6. Radon (Maintainability & Modularization)
echo -e "\n${BLUE}[6/6] Running Maintainability Index Check...${NC}"
RADON_OUTPUT=$(radon mi $PYTHON_DIRS --min B --max F 2>&1 || true)
# Baseline: a low-MI module fails only if it is new or its grade is worse than at
# merge-base(HEAD, QUALITY_BASE_REF), so upstream-owned debt does not block pushes.
# Without a resolvable base every low-MI module fails (strict mode).
QUALITY_BASE_REF=${QUALITY_BASE_REF:-origin/master}
MI_BASE=$(git merge-base HEAD "$QUALITY_BASE_REF" 2>/dev/null || true)
LOW_MI_OUTPUT=""
BASELINE_MI_OUTPUT=""
while read -r MI_PATH _ MI_GRADE; do
    BASE_GRADE=""
    if [ -n "$MI_BASE" ]; then
        BASE_GRADE=$(git show "$MI_BASE:$MI_PATH" 2>/dev/null | radon mi - 2>/dev/null | awk '{print $NF}')
    fi
    if [ -n "$BASE_GRADE" ] && [[ ! "$MI_GRADE" > "$BASE_GRADE" ]]; then
        BASELINE_MI_OUTPUT+="$MI_PATH - $MI_GRADE (baseline ${BASE_GRADE})"$'\n'
    else
        LOW_MI_OUTPUT+="$MI_PATH - $MI_GRADE (baseline ${BASE_GRADE:-none})"$'\n'
    fi
done < <(echo "$RADON_OUTPUT" | grep " - [B-F]")
LOW_MI_COUNT=$(printf '%s' "$LOW_MI_OUTPUT" | grep -c " - [B-F]")
LOW_MI_COUNT=${LOW_MI_COUNT:-0}

if [ -n "$BASELINE_MI_OUTPUT" ]; then
    echo -e "${YELLOW}Pre-existing low MI at ${MI_BASE:0:8} (${QUALITY_BASE_REF}), not counted:${NC}"
    printf '%s' "$BASELINE_MI_OUTPUT"
fi
if [ "$LOW_MI_COUNT" -gt 0 ]; then
    echo -e "${RED}✖ Failed: $LOW_MI_COUNT module(s) with Maintainability Index below threshold${NC}"
    printf '%s' "$LOW_MI_OUTPUT"
    BUILD_FAILED=1
else
    echo -e "${GREEN}✔ Passed: No module's Maintainability Index dropped below grade A or its baseline.${NC}"
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

# Final Verdict: Fail build if any issues found
echo "--------------------------------------------------"
if [ "$BUILD_FAILED" -eq 1 ]; then
    echo -e "${RED}RESULT: BUILD BLOCKED - Quality issues found.${NC}"
    exit 1
else
    echo -e "${GREEN}RESULT: QUALITY BENCHMARK PASSED${NC}"
    exit 0
fi
