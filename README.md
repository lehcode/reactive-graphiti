# reactive-graphiti

> Temporal context graphs for AI agents — extended with LiteLLM, streaming MCP, and reactive processing.

reactive-graphiti is a fork of [graphiti-core](https://github.com/getzep/graphiti) that adds first-class support for any OpenAI-compatible LLM provider, streaming MCP transport modes, and queue-based reactive episode processing. It is fully compatible with the upstream graphiti API while adding the enhancements described below.

## Key Enhancements

### LiteLLM Client

`LiteLLMClient` (`graphiti_core/llm_client/litellm_client.py`) replaces the upstream `OpenAIGenericClient` with a provider-agnostic client that works with any OpenAI-compatible endpoint:

- **Ollama**, **LM Studio**, **vLLM**, **LiteLLM proxy**, or any `v1/chat/completions`-compatible API
- Default `max_tokens=16384` (16K) — better suited to local models than the upstream 8K default
- Auto-detects providers that return the JSON Schema definition instead of data and transparently falls back from `json_schema` → `json_object` structured-output mode
- Extracts the first valid JSON object from responses that append trailing text (common with some local models)

### Streaming MCP Transport

The MCP server (`mcp_server/`) now supports three transport modes:

| Transport | Use case |
|-----------|----------|
| `sse` | Default; Server-Sent Events for web clients |
| `http` | Streamable HTTP for modern clients (e.g. Cursor) |
| `stdio` | Standard I/O for traditional MCP tooling |

### Reactive Episode Processing

- Queue-based async worker handles episode ingestion with back-pressure
- `SEMAPHORE_LIMIT` environment variable controls concurrency (default: 10)
- Upsert-safe graph writes — stateless HTTP mode prevents "Session not found" errors after server restarts
- Episode name and effective group ID included in all processing logs

### Multi-Provider Factory

The MCP server selects LLM and embedder providers at runtime via a config file (`mcp_server/src/services/factories.py`):

- **LLM providers**: OpenAI, Anthropic, Gemini, Groq, Azure OpenAI, LiteLLM
- **Embedding providers**: OpenAI, Voyage, Sentence Transformers, Gemini

### Modular Search

Search utilities are split into focused modules under `graphiti_core/search/search_utils/`:

- `discovery.py` — entity discovery
- `fulltext.py` — BM25 keyword search
- `similarity.py` — semantic similarity
- `traversal.py` — graph traversal
- `rerankers.py` — cross-encoder reranking

---

## What is a Context Graph?

A **context graph** is a temporal graph of entities, relationships, and facts — like *"Kendra loves Adidas shoes (as of
March 2026)."* Unlike traditional knowledge graphs, each fact in a context graph has a validity window: when it became
true, and when (if ever) it was superseded. Entities evolve over time with updated summaries. Everything traces back to
**episodes** — the raw data that produced it.

What makes Graphiti unique is its ability to autonomously build context graphs from unstructured and structured data,
handling changing relationships while preserving full temporal history.

A context graph contains:

| Component | What it stores |
|-----------|----------------|
| **Entities** (nodes) | People, products, policies, concepts — with summaries that evolve over time |
| **Facts / Relationships** (edges) | Triplets (Entity → Relationship → Entity) with temporal validity windows |
| **Episodes** (provenance) | Raw data as ingested — the ground truth stream. Every derived fact traces back here |
| **Custom Types** (ontology) | Developer-defined entity and edge types via Pydantic models |

## Why reactive-graphiti?

Traditional RAG approaches often rely on batch processing and static data summarization, making them inefficient for
frequently changing data. Graphiti addresses these challenges by providing:

- **Temporal Fact Management:** Facts have validity windows. When information changes, old facts are
  invalidated — not deleted. Query what's true now, or what was true at any point in time.
- **Episodes & Provenance:** Every entity and relationship traces back to the episodes (raw data) that produced it.
  Full lineage from derived fact to source.
- **Prescribed & Learned Ontology:** Define entity and edge types upfront via Pydantic models (prescribed), or let
  structure emerge from your data (learned). Start simple, evolve as patterns appear.
- **Incremental Graph Construction:** New data integrates immediately without batch recomputation. The graph evolves
  in real-time as episodes are ingested.
- **Hybrid Retrieval:** Combines semantic embeddings, keyword (BM25), and graph traversal for low-latency,
  high-precision queries without reliance on LLM summarization.
- **Scalability:** Efficiently manages large datasets with parallel processing, pluggable graph backends, suitable
  for enterprise workloads.

## Graphiti vs. GraphRAG

| Aspect | GraphRAG | Graphiti |
|--------|----------|----------|
| **Primary Use** | Static document summarization | Dynamic, evolving context for agents |
| **Data Handling** | Batch-oriented processing | Continuous, incremental updates |
| **Knowledge Structure** | Entity clusters & community summaries | Temporal context graph — entities, facts with validity windows, episodes, communities |
| **Retrieval Method** | Sequential LLM summarization | Hybrid semantic, keyword, and graph-based search |
| **Adaptability** | Low | High |
| **Temporal Handling** | Basic timestamp tracking | Explicit bi-temporal tracking with automatic fact invalidation |
| **Contradiction Handling** | LLM-driven summarization judgments | Automatic fact invalidation with temporal history preserved |
| **Query Latency** | Seconds to tens of seconds | Typically sub-second latency |
| **Custom Entity Types** | No | Yes, customizable via Pydantic models |
| **Scalability** | Moderate | High, optimized for large datasets |

## Installation

Requirements:

- Python 3.10 or higher
- Neo4j 5.26 / FalkorDB 1.1.2 / Kuzu 0.11.2 / Amazon Neptune Database Cluster or Neptune Analytics Graph

```bash
pip install graphiti-core
```

or

```bash
uv add graphiti-core
```

> [!TIP]
> Use `LiteLLMClient` for local models (Ollama, LM Studio) or any OpenAI-compatible provider. It is the recommended
> client for self-hosted deployments. See [Using with Ollama](#using-graphiti-with-ollama-local-llm) below.

### Installing with FalkorDB Support

```bash
pip install graphiti-core[falkordb]

# or with uv
uv add graphiti-core[falkordb]
```

### Installing with Kuzu Support

```bash
pip install graphiti-core[kuzu]

# or with uv
uv add graphiti-core[kuzu]
```

### Installing with Amazon Neptune Support

```bash
pip install graphiti-core[neptune]

# or with uv
uv add graphiti-core[neptune]
```

### Optional LLM provider extras

```bash
# Anthropic
pip install graphiti-core[anthropic]

# Groq
pip install graphiti-core[groq]

# Google Gemini
pip install graphiti-core[google-genai]

# Multiple providers
pip install graphiti-core[anthropic,groq,google-genai]

# FalkorDB + providers
pip install graphiti-core[falkordb,anthropic,google-genai]
```

## Concurrency and Rate Limits

Graphiti's ingestion pipelines support high concurrency. The `SEMAPHORE_LIMIT` environment variable controls the number
of concurrent LLM operations (default: `10`). Lower this value if you hit provider 429 rate limit errors; raise it if
your provider supports higher throughput.

## Quick Start

For a complete working example, see the [Quickstart Example](examples/quickstart/README.md). It demonstrates:

1. Connecting to Neo4j, Amazon Neptune, FalkorDB, or Kuzu
2. Initializing Graphiti indices and constraints
3. Adding episodes (text and structured JSON)
4. Searching for relationships using hybrid search
5. Reranking results using graph distance
6. Searching for nodes using predefined search recipes

### Running with Docker Compose

- **Neo4j:**

  ```bash
  docker compose up
  ```

- **FalkorDB:**

  ```bash
  docker compose --profile falkordb up
  ```

## Using Graphiti with Ollama (Local LLM)

Use `LiteLLMClient` for Ollama and other OpenAI-compatible providers. It is optimized for local models with a higher
default max token limit (16K vs 8K) and full support for structured outputs including automatic fallback for providers
that do not support `json_schema` mode.

Install the models:

```bash
ollama pull deepseek-r1:7b # LLM
ollama pull nomic-embed-text # embeddings
```

```python
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.litellm_client import LiteLLMClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

llm_config = LLMConfig(
    api_key='ollama',
    model='deepseek-r1:7b',
    small_model='deepseek-r1:7b',
    base_url='http://localhost:11434/v1',
)

llm_client = LiteLLMClient(config=llm_config)

graphiti = Graphiti(
    'bolt://localhost:7687',
    'neo4j',
    'password',
    llm_client=llm_client,
    embedder=OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key='ollama',
            embedding_model='nomic-embed-text',
            embedding_dim=768,
            base_url='http://localhost:11434/v1',
        )
    ),
    cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config),
)
```

Ensure Ollama is running (`ollama serve`) before initializing.

## Using Graphiti with Azure OpenAI

```python
from openai import AsyncOpenAI
from graphiti_core import Graphiti
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient

azure_client = AsyncOpenAI(
    base_url='https://your-resource-name.openai.azure.com/openai/v1/',
    api_key='your-api-key',
)

llm_client = AzureOpenAILLMClient(
    azure_client=azure_client,
    config=LLMConfig(model='gpt-5-mini', small_model='gpt-5-mini')
)
embedder_client = AzureOpenAIEmbedderClient(
    azure_client=azure_client,
    model='text-embedding-3-small'
)

graphiti = Graphiti(
    'bolt://localhost:7687',
    'neo4j',
    'password',
    llm_client=llm_client,
    embedder=embedder_client,
)
```

Use Azure's v1 API endpoint format: `https://your-resource-name.openai.azure.com/openai/v1/`. See
`examples/azure-openai/` for a complete example.

## Using Graphiti with Google Gemini

```bash
uv add "graphiti-core[google-genai]"
```

```python
from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

api_key = '<your-google-api-key>'

graphiti = Graphiti(
    'bolt://localhost:7687',
    'neo4j',
    'password',
    llm_client=GeminiClient(config=LLMConfig(api_key=api_key, model='gemini-2.0-flash')),
    embedder=GeminiEmbedder(
        config=GeminiEmbedderConfig(api_key=api_key, embedding_model='embedding-001')
    ),
    cross_encoder=GeminiRerankerClient(
        config=LLMConfig(api_key=api_key, model='gemini-2.5-flash-lite')
    ),
)
```

## MCP Server

The `mcp_server/` directory contains a Model Context Protocol server for Graphiti. It supports three transport modes —
SSE (default), streamable HTTP, and stdio — making it compatible with Claude Desktop, Cursor, VS Code, and standard
MCP tooling.

Features:

- Episode management (add, retrieve, delete)
- Entity and relationship management
- Semantic and hybrid search
- Group management
- Graph maintenance operations
- Queue-based async processing with configurable concurrency

See the [MCP server README](mcp_server/README.md) for setup and configuration.

## REST Service

The `server/` directory contains a FastAPI service for the Graphiti API. See the
[server README](server/README.md) for details.

## Database Configuration

Database names are configured in the driver constructors.

### Neo4j

```python
from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver

driver = Neo4jDriver(
    uri='bolt://localhost:7687',
    user='neo4j',
    password='password',
    database='my_custom_database'
)

graphiti = Graphiti(graph_driver=driver)
```

### FalkorDB

```python
from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver

driver = FalkorDriver(
    host='localhost',
    port=6379,
    database='my_custom_graph'
)

graphiti = Graphiti(graph_driver=driver)
```

### Kuzu

```python
from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver

driver = KuzuDriver(db='/tmp/graphiti.kuzu')
graphiti = Graphiti(graph_driver=driver)
```

### Amazon Neptune

```python
from graphiti_core import Graphiti
from graphiti_core.driver.neptune_driver import NeptuneDriver

driver = NeptuneDriver(
    host='<NEPTUNE_ENDPOINT>',
    aoss_host='<AMAZON_OPENSEARCH_SERVERLESS_HOST>',
    port=8182,
    aoss_port=443,
)

graphiti = Graphiti(graph_driver=driver)
```

Contributing a new graph backend? See [Adding a graph driver](CONTRIBUTING.md#adding-a-graph-driver).

## Contributing

Contributions are welcome — code, documentation, bug reports, and questions. For code contribution guidelines see
[CONTRIBUTING.md](CONTRIBUTING.md). Open an issue or PR on this repository.

## Upstream

This project is a fork of [graphiti](https://github.com/getzep/graphiti) by Zep Software, Inc., licensed under the
[Apache 2.0 License](LICENSE).
