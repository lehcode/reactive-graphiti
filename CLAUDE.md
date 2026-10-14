# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Fork context

- `reactive-graphiti` is a fork of getzep/graphiti. Remotes: `origin` (GitLab, primary), `github-fork` (lehcode), `upstream` (getzep). Upstream is merged periodically — keep diffs against upstream files small; prefer additive changes.
- Fork-specific: `graphiti_core/llm_client/litellm_client.py` (`LiteLLMClient`, formerly `OpenAIGenericClient`) is used by `server/` and the MCP server for Ollama/vLLM/LiteLLM endpoints. `mcp_server/src/services/factories.py` has a try/except import shim falling back to upstream `OpenAIGenericClient` when an upstream `graphiti-core` wheel is installed — keep it.
- MCP tools accept a reserved `codemode: bool = False` param (absorbs stray client args); don't remove it.
- Root `Dockerfile` builds the FastAPI `server/` image and installs `graphiti-core` **from PyPI**, not local source. MCP images live in `mcp_server/docker/`.
- `scripts/quality-benchmark.sh` — manual quality gate (shellcheck, ruff, pyright, dead code, duplication) over `graphiti_core server mcp_server/src`; fails on any ruff/type error. Not wired into CI.

## Architecture

### Ingestion (`graphiti_core/graphiti.py`)

`add_episode` pipeline: fetch previous episodes as context → `extract_nodes` → `resolve_extracted_nodes` (deterministic exact/fuzzy dedup in `utils/maintenance/dedup_helpers.py`, LLM fallback) → `_extract_and_resolve_edges` (`extract_edges` + `resolve_extracted_edges`: dedup + temporal invalidation of contradicted edges) → `extract_attributes_from_nodes` (summaries/attributes) → `_process_episode_data` (persist) → optional `update_community`. Step logic lives in `utils/maintenance/{node,edge}_operations.py`.

`add_episode_bulk` takes `RawEpisode`s and uses `utils/bulk_utils.py` (in-memory bulk dedup). `use_combined_extraction` (single LLM call for nodes+edges, `utils/maintenance/combined_extraction.py`) exists only as a `bulk_utils` flag, default off, not wired anywhere.

### LLM calls

Prompts: `prompt_library.<module>.<version>(context)` (`prompts/lib.py`); each prompt module declares a `Versions` TypedDict. Structured output via Pydantic `response_model`. Custom entity types = `dict[str, type[BaseModel]]`; custom edge types = `edge_types` + `edge_type_map` keyed by `(source_label, target_label)`.

### Driver layer (`graphiti_core/driver/`)

Three layers (redesign in progress, `spec/driver-operations-redesign.md`, Phase 1 non-breaking):

1. Namespaces — `graphiti.nodes.*` / `graphiti.edges.*` (`namespaces/`): intended public API; handles embedding then delegates to ops.
2. Operations ABCs — `driver/operations/*_ops.py`, implemented per backend in `driver/<backend>/operations/`, exposed as `*_ops` properties on `GraphDriver`.
3. `GraphDriver` (`driver.py`) — connection/session/query execution.

Legacy paths still active: `EntityNode.save()`-style methods in `nodes.py`/`edges.py` (internals like `add_episode` still use them), and `graph_operations_interface`/`search_interface` override hooks. Provider query strings also live in `models/{nodes,edges}/*_db_queries.py`, `graph_queries.py`, `search/search_utils.py`.

**Adding a query:** add to the ABC, implement in every backend's `operations/` (Neo4j, FalkorDB, Neptune; Kuzu deprecated), add a namespace wrapper if public. See CONTRIBUTING.md "Adding a Graph Driver".

`group_id` is the partition key; on FalkorDB it maps to a separate graph — `@handle_multiple_group_ids` (`decorators.py`) fans out per group via `driver.clone()`.

### Search (`graphiti_core/search/`)

`Graphiti.search()` → edges only, preset `EDGE_HYBRID_SEARCH_RRF` (or `NODE_DISTANCE` with a center node). `Graphiti.search_()` → `SearchResults` (nodes/edges/episodes/communities) driven by a `SearchConfig`; presets in `search_config_recipes.py`. Rerankers in `cross_encoder/`.

Gotcha: `search()` sets `.limit` on the shared module-level recipe object — copy a recipe (`model_copy()`) before mutating it.

### Services

- `mcp_server/`: `main.py` → `src/graphiti_mcp_server.py` (FastMCP tools). `src/config/schema.py` loads YAML from `CONFIG_PATH` (default `config/config.yaml`) with `${VAR:default}` expansion; env overrides use `__` nesting. `src/services/factories.py` builds LLM/embedder/DB clients per provider. `src/services/queue_service.py` runs one asyncio queue + worker per `group_id` so episodes in a group are processed sequentially.
- `server/`: FastAPI; `graph_service/zep_graphiti.py` subclasses `Graphiti` and builds the client per request (`X-LLM-Tier` header selects LLM tier).

### Optional integrations

Third-party deps must be optional extras (also listed in `dev`), imported via the `TYPE_CHECKING` + `try/except ImportError` pattern; never import them in `__init__.py` (CONTRIBUTING.md).
