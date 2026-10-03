# AGENTS.md

## Repo structure

This is a **Python monorepo** with three independently versioned packages, all under Python 3.10+:

| Directory | Package | Entry point |
|-----------|---------|-------------|
| `graphiti_core/` | `graphiti-core` library | `graphiti_core/graphiti.py` — `Graphiti` class |
| `server/` | REST API | `server/graph_service/main.py` — FastAPI app |
| `mcp_server/` | MCP server | `mcp_server/main.py` |

Each subpackage has its own `pyproject.toml`, `Makefile`, and test suite. The root `conftest.py` imports fixtures from `tests/helpers_test.py` and excludes `mcp_server/*` from collection.

## Commands (exact)

```bash
# Root library — install, format, lint, test
make install          # uv sync --extra dev
make format           # ruff check --select I --fix && ruff format
make lint             # ruff check + pyright ./graphiti_core
make test             # DISABLE_FALKORDB=1 DISABLE_KUZU=1 DISABLE_NEPTUNE=1 uv run pytest -m "not integration"
make check            # format → lint → test
```

```bash
# Server
cd server/ && make test   # uv run pytest (integration e2e; skips if OPENAI_API_KEY unset)
```

```bash
# MCP server
cd mcp_server/ && uv run pytest  # testpaths = tests/
```

Run a single test: `uv run pytest tests/test_file.py::test_method_name -v`

## Testing gotchas

- **`make test` does NOT set `DISABLE_NEO4J`** — the `graph_driver` fixture in `helpers_test.py` will try to connect to `bolt://localhost:7687`. If no Neo4j is running, those tests hang on connection retry. Run with `NEO4J_PASSWORD=test make test` or disable explicitly:

  ```bash
  DISABLE_NEO4J=1 make test
  ```

- The fully-green no-DB unit gate (CI command):

  ```bash
  DISABLE_NEO4J=1 DISABLE_FALKORDB=1 DISABLE_KUZU=1 DISABLE_NEPTUNE=1 uv run pytest tests/ \
    --ignore=tests/test_graphiti_int.py \
    --ignore=tests/test_graphiti_mock.py \
    --ignore=tests/test_node_int.py \
    --ignore=tests/test_edge_int.py \
    --ignore=tests/test_entity_exclusion_int.py \
    --ignore=tests/driver/ \
    --ignore=tests/evals/
  ```

- **`tests/test_add_triplet.py`** has pre-existing failures (mock embedder doesn't stub `create_batch` → `zip(strict=True)` raises). CI never runs this file — ignore it.
- Integration tests are marked with `@pytest.mark.integration` and use `asyncio_mode = auto`.
- `server/` and `mcp_server/` e2e tests self-skip (exit 5) when `OPENAI_API_KEY` is unset — expected.

## DB drivers

`helpers_test.py` selects active drivers from these env vars:
- `DISABLE_NEO4J`, `DISABLE_FALKORDB`, `ENABLE_KUZU`/`DISABLE_KUZU`, `DISABLE_NEPTUNE`

Kuzu is deprecated (upstream unmaintained); excluded by default. Enable with `ENABLE_KUZU=1`.

**FalkorDB concurrency bug:** The async driver drops connections ("Connection closed by server") when Graphiti issues concurrent queries on one connection (e.g., inside `Graphiti.search`). **Prefer Neo4j** for local hybrid-search development.

## Services

Server startup **requires** `OPENAI_API_KEY` (any value) or the Pydantic `Settings` model fails.

```bash
# REST API (port 8000)
OPENAI_API_KEY=x DB_BACKEND=falkordb FALKORDB_HOST=localhost FALKORDB_PORT=6379 \
  uv run uvicorn server.graph_service.main:app --reload --port 8000

# MCP server (port 8001)
OPENAI_API_KEY=x FALKORDB_URI=redis://localhost:6379 \
  uv run python mcp_server/main.py --transport http --host 0.0.0.0 --port 8001 --database-provider falkordb
```

Health at `/healthcheck`, Swagger at `/docs`. Both need a real `OPENAI_API_KEY` for LLM extraction + embeddings (startup works without one only when the key is present but unused).

Set `GRAPHITI_TELEMETRY_ENABLED=false` when running anything to suppress PostHog.

## Coding conventions

- **Ruff** for lint/format. 100-char line limit, single quotes, `I` for import sorting (`E501` ignored).
- **Pyright** type checking, `basic` mode in root, `standard` in `server/`. Exclude: `graphiti_core/driver/neptune_driver.py`.
- Use `typing_extensions.TypedDict` instead of `typing.TypedDict` (Pydantic requirement on <3.12).
- Commit messages: imperative present-tense, optionally `(#issue)`.
