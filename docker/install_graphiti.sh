#!/bin/sh
set -e

# Install graphiti-core with optional FalkorDB support
# Uses build arg INSTALL_FALKORDB to control extras

if [ -n "$GRAPHITI_VERSION" ]; then
    if [ "$INSTALL_FALKORDB" = "true" ]; then
        uv pip install --system --upgrade "graphiti-core[falkordb]==$GRAPHITI_VERSION"
    else
        uv pip install --system --upgrade "graphiti-core==$GRAPHITI_VERSION"
    fi
else
    if [ "$INSTALL_FALKORDB" = "true" ]; then
        uv pip install --system --upgrade "graphiti-core[falkordb]"
    else
        uv pip install --system --upgrade graphiti-core
    fi
fi
