# ruff: noqa
"""Vulture whitelist for scripts/quality-benchmark.sh.

Each name below is required by a signature the code does not control; referencing
it here marks it as used. Do not list names that are genuinely dead.
"""

# __aexit__(self, exc_type, exc, tb) protocol args (driver.py, falkordb/kuzu/neptune drivers)
exc_type
exc
tb

# Abstract interface params kept for implementations (graph_operations.py)
transaction

# Upstream no-op stub KuzuDriver.delete_all_indexes(database_) in the deprecated Kuzu driver (kuzu_driver.py)
database_

# Pydantic hooks: model_post_init(__context), settings_customise_sources(file_secret_settings)
__context
file_secret_settings

# Reserved MCP tool param absorbing stray client args; must stay (CLAUDE.md fork context)
codemode
