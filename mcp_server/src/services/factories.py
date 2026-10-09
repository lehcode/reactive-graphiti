"""Factory classes for creating LLM, Embedder, and Database clients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder import EmbedderClient, OpenAIEmbedder
from graphiti_core.llm_client import LLMClient, OpenAIClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig

# Shim for upstream graphiti-core: OpenAIGenericClient → LiteLLMClient
try:
    from graphiti_core.llm_client.litellm_client import LiteLLMClient
except ImportError:
    from graphiti_core.llm_client.openai_generic_client import (
        OpenAIGenericClient as LiteLLMClient,  # noqa: F401
    )

if TYPE_CHECKING:
    from graphiti_core.cross_encoder import CrossEncoderClient

from config.schema import (
    DatabaseConfig,
    EmbedderConfig,
    LLMConfig,
)

# Try to import FalkorDriver if available
try:
    from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: F401

    HAS_FALKOR = True
except ImportError:
    HAS_FALKOR = False

# Kuzu support removed - FalkorDB is now the default

# Try to import additional providers if available
try:
    from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient

    HAS_AZURE_EMBEDDER = True
except ImportError:
    HAS_AZURE_EMBEDDER = False

try:
    from graphiti_core.embedder.gemini import GeminiEmbedder

    HAS_GEMINI_EMBEDDER = True
except ImportError:
    HAS_GEMINI_EMBEDDER = False

try:
    from graphiti_core.embedder.voyage import VoyageAIEmbedder

    HAS_VOYAGE_EMBEDDER = True
except ImportError:
    HAS_VOYAGE_EMBEDDER = False

try:
    from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient

    HAS_AZURE_LLM = True
except ImportError:
    HAS_AZURE_LLM = False

try:
    from graphiti_core.llm_client.anthropic_client import AnthropicClient

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from graphiti_core.llm_client.gemini_client import GeminiClient

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from graphiti_core.llm_client.groq_client import GroqClient

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    from graphiti_core.llm_client.litellm_client import LiteLLMClient

    HAS_OPENAI_GENERIC = True
except ImportError:
    from graphiti_core.llm_client.openai_generic_client import (
        OpenAIGenericClient as LiteLLMClient,  # noqa: F401
    )

    HAS_OPENAI_GENERIC = True


def resolve_max_input_tokens(
    api_url: str,
    api_key: str,
    model: str,
    configured: int | None,
    *,
    client: Any | None = None,
) -> int | None:
    """Resolve max_input_tokens from LiteLLM /model/info endpoint.

    Returns the LiteLLM-detected value if available, otherwise ``configured``,
    or ``None`` when neither is available.  Never raises.

    Precedence:
        * LiteLLM value (from GET ``{base}/model/info``) wins.
        * If ``configured`` is set and differs, logs a WARNING with both values.
        * Lookup failure (network, non-200, missing model/field) → ``configured``
          with WARNING; neither → ``None`` with WARNING.
        * Logs resolved value at INFO.
        * Never raises. Never logs ``api_key``.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Strip trailing /v1 and trailing / from api_url
    base = api_url.rstrip('/')
    if base.endswith('/v1'):
        base = base[:-3]

    url = f'{base}/model/info'

    lite_value: int | None = None
    try:
        headers = {'Authorization': f'Bearer {api_key}'}
        # Use the injectable client (sync for compatibility), or a short-lived one
        if client is None:
            import httpx

            owned_client = httpx.Client()
            try:
                response = owned_client.get(url, headers=headers, timeout=5.0)
            finally:
                owned_client.close()
        else:
            response = client.get(url, headers=headers, timeout=5.0)
        if response.status_code == 200:
            payload = response.json()
            # LiteLLM proxy returns {"data": [{"model_name": ..., "model_info": {...}}]}
            entries = payload.get('data', []) if isinstance(payload, dict) else []
            for entry in entries:
                if isinstance(entry, dict) and entry.get('model_name') == model:
                    model_info = entry.get('model_info') or {}
                    raw = model_info.get('max_input_tokens')
                    if isinstance(raw, int) and raw > 0:
                        lite_value = raw
                        break
        elif response.status_code >= 400:
            logger.warning(
                f'/model/info returned {response.status_code}, using configured max_input_tokens'
            )
    except Exception as e:
        logger.warning(
            f'/model/info lookup failed ({type(e).__name__}), using configured max_input_tokens'
        )

    # Determine precedence
    if lite_value is not None and configured is not None and lite_value != configured:
        logger.warning(f'max_input_tokens mismatch: LiteLLM={lite_value}, configured={configured}')

    resolved = lite_value if lite_value is not None else configured

    if resolved is None:
        logger.warning('max_input_tokens not resolved; budgeting disabled')
    else:
        logger.info(f'max_input_tokens resolved to {resolved}')

    return resolved


def _core_llm_config_accepts_max_input_tokens(config_cls: Any) -> bool:
    """Whether the installed graphiti-core LLMConfig takes ``max_input_tokens``.

    Upstream PyPI graphiti-core (used by Dockerfile.standalone) lacks it; passing it there
    would raise TypeError at startup.
    """
    import inspect

    try:
        return 'max_input_tokens' in inspect.signature(config_cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def _validate_api_key(provider_name: str, api_key: str | None, logger) -> str:
    """Validate API key is present.

    Args:
        provider_name: Name of the provider (e.g., 'OpenAI', 'Anthropic')
        api_key: The API key to validate
        logger: Logger instance for output

    Returns:
        The validated API key

    Raises:
        ValueError: If API key is None or empty
    """
    if not api_key:
        raise ValueError(
            f'{provider_name} API key is not configured. Please set the appropriate environment variable.'
        )

    logger.info(f'Creating {provider_name} client')

    return api_key


class LLMClientFactory:
    """Factory for creating LLM clients based on configuration."""

    @staticmethod
    def create(config: LLMConfig) -> LLMClient:
        """Create an LLM client based on the configured provider."""
        import logging

        logger = logging.getLogger(__name__)

        provider = config.provider.lower()

        match provider:
            case 'openai':
                if not config.providers.openai:
                    raise ValueError('OpenAI provider configuration not found')

                api_key = config.providers.openai.api_key
                _validate_api_key('OpenAI', api_key, logger)

                from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig

                # Use the same model for both main and small model slots
                small_model = config.model

                llm_kwargs = {
                    'api_key': api_key,
                    'model': config.model,
                    'small_model': small_model,
                    'temperature': config.temperature if config.temperature is not None else 0.0,
                    'max_tokens': config.max_tokens,
                }
                # Resolve max_input_tokens from /model/info; pass only if supported
                lite_val = resolve_max_input_tokens(
                    config.providers.openai.api_url,
                    api_key,
                    config.model,
                    config.max_input_tokens,
                )
                if lite_val is not None and _core_llm_config_accepts_max_input_tokens(
                    CoreLLMConfig
                ):
                    llm_kwargs['max_input_tokens'] = lite_val

                llm_config = CoreLLMConfig(**llm_kwargs)

                if config.structured_output_mode is not None or config.model.startswith(
                    ('ollama', 'vllm', 'llama', 'phi')
                ):
                    # Use LiteLLMClient for Ollama and other OpenAI-compatible providers
                    # This uses the standard Chat Completions API instead of Responses API
                    return LiteLLMClient(
                        config=llm_config,
                        max_tokens=config.max_tokens,
                        structured_output_mode=config.structured_output_mode,
                    )

                # Check if this is a reasoning model (o1, o3, gpt-5 family)
                reasoning_prefixes = ('o1', 'o3', 'gpt-5')
                is_reasoning_model = config.model.startswith(reasoning_prefixes)

                if is_reasoning_model:
                    return OpenAIClient(config=llm_config, reasoning='minimal', verbosity='low')

                return OpenAIClient(config=llm_config)

            case 'azure_openai':
                if not HAS_AZURE_LLM:
                    raise ValueError(
                        'Azure OpenAI LLM client not available in current graphiti-core version'
                    )
                if not config.providers.azure_openai:
                    raise ValueError('Azure OpenAI provider configuration not found')
                azure_config = config.providers.azure_openai

                if not azure_config.api_url:
                    raise ValueError('Azure OpenAI API URL is required')

                # Currently using API key authentication
                # TODO: Add Azure AD authentication support for v1 API compatibility
                api_key = azure_config.api_key
                _validate_api_key('Azure OpenAI', api_key, logger)

                # Azure OpenAI should use the standard AsyncOpenAI client with v1 compatibility endpoint
                # See: https://github.com/getzep/graphiti README Azure OpenAI section
                from openai import AsyncOpenAI

                # Ensure the base_url ends with /openai/v1/ for Azure v1 compatibility
                base_url = azure_config.api_url
                if not base_url.endswith('/'):
                    base_url += '/'
                if not base_url.endswith('openai/v1/'):
                    base_url += 'openai/v1/'

                azure_client = AsyncOpenAI(
                    base_url=base_url,
                    api_key=api_key,
                )

                # Then create the LLMConfig
                from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig

                llm_config = CoreLLMConfig(
                    api_key=api_key,
                    base_url=base_url,
                    model=config.model,
                    temperature=config.temperature if config.temperature is not None else 0.0,
                    max_tokens=config.max_tokens,
                )

                return AzureOpenAILLMClient(
                    azure_client=azure_client,
                    config=llm_config,
                    max_tokens=config.max_tokens,
                )

            case 'anthropic':
                if not HAS_ANTHROPIC:
                    raise ValueError(
                        'Anthropic client not available in current graphiti-core version'
                    )
                if not config.providers.anthropic:
                    raise ValueError('Anthropic provider configuration not found')

                api_key = config.providers.anthropic.api_key
                _validate_api_key('Anthropic', api_key, logger)

                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    model=config.model,
                    temperature=config.temperature if config.temperature is not None else 0.0,
                    max_tokens=config.max_tokens,
                )
                return AnthropicClient(config=llm_config)

            case 'gemini':
                if not HAS_GEMINI:
                    raise ValueError('Gemini client not available in current graphiti-core version')
                if not config.providers.gemini:
                    raise ValueError('Gemini provider configuration not found')

                api_key = config.providers.gemini.api_key
                _validate_api_key('Gemini', api_key, logger)

                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    model=config.model,
                    temperature=config.temperature if config.temperature is not None else 0.0,
                    max_tokens=config.max_tokens,
                )
                return GeminiClient(config=llm_config)

            case 'groq':
                if not HAS_GROQ:
                    raise ValueError('Groq client not available in current graphiti-core version')
                if not config.providers.groq:
                    raise ValueError('Groq provider configuration not found')

                api_key = config.providers.groq.api_key
                _validate_api_key('Groq', api_key, logger)

                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    base_url=config.providers.groq.api_url,
                    model=config.model,
                    temperature=config.temperature if config.temperature is not None else 0.0,
                    max_tokens=config.max_tokens,
                )
                return GroqClient(config=llm_config)

            case 'openai_generic':
                # OpenAI Generic client - uses /chat/completions with response_format
                # instead of /responses endpoint. Compatible with LiteLLM, Ollama, vLLM, etc.
                if not HAS_OPENAI_GENERIC:
                    raise ValueError(
                        'OpenAI Generic client not available in current graphiti-core version'
                    )
                if not config.providers.openai:
                    raise ValueError('OpenAI provider configuration not found')

                api_key = config.providers.openai.api_key
                base_url = config.providers.openai.api_url
                _validate_api_key('OpenAI Generic', api_key, logger)

                from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig

                llm_kwargs = {
                    'api_key': api_key,
                    'base_url': base_url,
                    'model': config.model,
                    'small_model': 'gpt-4.1-mini',
                    'temperature': config.temperature if config.temperature is not None else 0.0,
                    'max_tokens': config.max_tokens,
                }
                lite_val = resolve_max_input_tokens(
                    base_url,
                    api_key,
                    config.model,
                    config.max_input_tokens,
                )
                if lite_val is not None and _core_llm_config_accepts_max_input_tokens(
                    CoreLLMConfig
                ):
                    llm_kwargs['max_input_tokens'] = lite_val

                llm_config = CoreLLMConfig(**llm_kwargs)
                return LiteLLMClient(config=llm_config)

            case _:
                raise ValueError(f'Unsupported LLM provider: {provider}')


class EmbedderFactory:
    """Factory for creating Embedder clients based on configuration."""

    @staticmethod
    def create(config: EmbedderConfig) -> EmbedderClient:
        """Create an Embedder client based on the configured provider."""
        import logging

        logger = logging.getLogger(__name__)

        provider = config.provider.lower()

        match provider:
            case 'openai':
                if not config.providers.openai:
                    raise ValueError('OpenAI provider configuration not found')

                api_key = config.providers.openai.api_key
                _validate_api_key('OpenAI Embedder', api_key, logger)

                from graphiti_core.embedder.openai import OpenAIEmbedderConfig

                embedder_config = OpenAIEmbedderConfig(
                    api_key=api_key,
                    embedding_model=config.model,
                    base_url=config.providers.openai.api_url,  # Support custom endpoints like Ollama
                    embedding_dim=config.dimensions,  # Support custom embedding dimensions
                )
                return OpenAIEmbedder(config=embedder_config)

            case 'azure_openai':
                if not HAS_AZURE_EMBEDDER:
                    raise ValueError(
                        'Azure OpenAI embedder not available in current graphiti-core version'
                    )
                if not config.providers.azure_openai:
                    raise ValueError('Azure OpenAI provider configuration not found')
                azure_config = config.providers.azure_openai

                if not azure_config.api_url:
                    raise ValueError('Azure OpenAI API URL is required')

                # Currently using API key authentication
                # TODO: Add Azure AD authentication support for v1 API compatibility
                api_key = azure_config.api_key
                _validate_api_key('Azure OpenAI Embedder', api_key, logger)

                # Azure OpenAI should use the standard AsyncOpenAI client with v1 compatibility endpoint
                # See: https://github.com/getzep/graphiti README Azure OpenAI section
                from openai import AsyncOpenAI

                # Ensure the base_url ends with /openai/v1/ for Azure v1 compatibility
                base_url = azure_config.api_url
                if not base_url.endswith('/'):
                    base_url += '/'
                if not base_url.endswith('openai/v1/'):
                    base_url += 'openai/v1/'

                azure_client = AsyncOpenAI(
                    base_url=base_url,
                    api_key=api_key,
                )

                return AzureOpenAIEmbedderClient(
                    azure_client=azure_client,
                    model=config.model or 'text-embedding-3-small',
                )

            case 'gemini':
                if not HAS_GEMINI_EMBEDDER:
                    raise ValueError(
                        'Gemini embedder not available in current graphiti-core version'
                    )
                if not config.providers.gemini:
                    raise ValueError('Gemini provider configuration not found')

                api_key = config.providers.gemini.api_key
                _validate_api_key('Gemini Embedder', api_key, logger)

                from graphiti_core.embedder.gemini import GeminiEmbedderConfig

                gemini_config = GeminiEmbedderConfig(
                    api_key=api_key,
                    embedding_model=config.model or 'models/text-embedding-004',
                    embedding_dim=config.dimensions or 768,
                )
                return GeminiEmbedder(config=gemini_config)

            case 'voyage':
                if not HAS_VOYAGE_EMBEDDER:
                    raise ValueError(
                        'Voyage embedder not available in current graphiti-core version'
                    )
                if not config.providers.voyage:
                    raise ValueError('Voyage provider configuration not found')

                api_key = config.providers.voyage.api_key
                _validate_api_key('Voyage Embedder', api_key, logger)

                from graphiti_core.embedder.voyage import VoyageAIEmbedderConfig

                voyage_config = VoyageAIEmbedderConfig(
                    api_key=api_key,
                    embedding_model=config.model or 'voyage-3',
                    embedding_dim=config.dimensions or 1024,
                )
                return VoyageAIEmbedder(config=voyage_config)

            case 'openai_generic':
                # OpenAI-compatible embedder (LiteLLM, Ollama, vLLM, etc.)
                # Uses the same OpenAIEmbedder with custom base_url
                if not config.providers.openai:
                    raise ValueError(
                        'OpenAI provider configuration not found for openai_generic embedder'
                    )

                api_key = config.providers.openai.api_key
                _validate_api_key('OpenAI Generic Embedder', api_key, logger)

                from graphiti_core.embedder.openai import OpenAIEmbedderConfig

                embedder_config = OpenAIEmbedderConfig(
                    api_key=api_key,
                    embedding_model=config.model,
                    base_url=config.providers.openai.api_url,
                    embedding_dim=config.dimensions,
                )
                return OpenAIEmbedder(config=embedder_config)

            case _:
                raise ValueError(f'Unsupported Embedder provider: {provider}')


class CrossEncoderFactory:
    """Factory for creating cross-encoder (reranker) clients based on configuration.

    Graphiti defaults the cross_encoder to OpenAIRerankerClient, which needs an OpenAI API key.
    To keep the server usable on non-OpenAI setups, pick a reranker from the LLM provider, then
    the embedder provider, and fall back to the local BGE reranker.
    """

    @staticmethod
    def create(llm_config: LLMConfig, embedder_config: EmbedderConfig) -> CrossEncoderClient:
        """Create a cross-encoder client based on the configured providers."""
        import logging

        logger = logging.getLogger(__name__)

        # Try the LLM provider first, then the embedder, before falling back to a local model.
        for source, config in (('LLM', llm_config), ('embedder', embedder_config)):
            reranker = CrossEncoderFactory._reranker_for_provider(source, config, logger)
            if reranker is not None:
                return reranker

        # No provider reranker available (e.g. Anthropic LLM + Voyage embedder), so use the
        # local BGE cross-encoder, which needs no API key.
        logger.warning(
            'No provider reranker available, using local BGERerankerClient '
            '(downloads BAAI/bge-reranker-v2-m3, ~2.3 GB, on first run)'
        )
        try:
            from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient
        except ImportError as e:
            raise ValueError(
                'No provider reranker is available for this configuration, and the local '
                'BGE fallback requires the optional sentence-transformers dependency. '
                "Install the MCP server's 'providers' extra (uv sync --extra providers), "
                "install graphiti-core's 'sentence-transformers' extra "
                "(pip install 'graphiti-core[sentence-transformers]'), or configure the "
                'LLM or embedder provider as OpenAI or Gemini with a valid API key.'
            ) from e

        return BGERerankerClient()

    @staticmethod
    def _reranker_for_provider(
        source: str, config: LLMConfig | EmbedderConfig, logger
    ) -> CrossEncoderClient | None:
        """Return a reranker for this provider, or None if it has no native one."""
        provider = config.provider.lower()

        match provider:
            case 'openai':
                if not config.providers.openai:
                    return None
                from graphiti_core.cross_encoder.openai_reranker_client import (
                    OpenAIRerankerClient,
                )

                logger.info(f'Using OpenAIRerankerClient from {source} provider')
                return OpenAIRerankerClient(
                    config=GraphitiLLMConfig(
                        api_key=config.providers.openai.api_key,
                        base_url=config.providers.openai.api_url,
                    )
                )

            case 'azure_openai':
                azure_config = config.providers.azure_openai
                if not azure_config or not azure_config.api_url:
                    return None
                from openai import AsyncOpenAI

                base_url = azure_config.api_url
                if not base_url.endswith('/'):
                    base_url += '/'
                if not base_url.endswith('openai/v1/'):
                    base_url += 'openai/v1/'
                from graphiti_core.cross_encoder.openai_reranker_client import (
                    OpenAIRerankerClient,
                )

                logger.info(f'Using OpenAIRerankerClient (Azure) from {source} provider')
                return OpenAIRerankerClient(
                    client=AsyncOpenAI(base_url=base_url, api_key=azure_config.api_key)
                )

            case 'gemini':
                if not config.providers.gemini:
                    return None
                from graphiti_core.cross_encoder.gemini_reranker_client import (
                    GeminiRerankerClient,
                )

                logger.info(f'Using GeminiRerankerClient from {source} provider')
                return GeminiRerankerClient(
                    config=GraphitiLLMConfig(api_key=config.providers.gemini.api_key)
                )

            case _:
                # anthropic, groq, voyage etc. have no native reranker in graphiti-core.
                return None


class DatabaseDriverFactory:
    """Factory for creating Database drivers based on configuration.

    Note: This returns configuration dictionaries that can be passed to Graphiti(),
    not driver instances directly, as the drivers require complex initialization.
    """

    @staticmethod
    def create_config(config: DatabaseConfig) -> dict:
        """Create database configuration dictionary based on the configured provider."""
        provider = config.provider.lower()

        match provider:
            case 'neo4j':
                # Use Neo4j config if provided, otherwise use defaults
                if config.providers.neo4j:
                    neo4j_config = config.providers.neo4j
                else:
                    # Create default Neo4j configuration
                    from config.schema import Neo4jProviderConfig

                    neo4j_config = Neo4jProviderConfig()

                # Check for environment variable overrides (for CI/CD compatibility)
                import os

                uri = os.environ.get('NEO4J_URI', neo4j_config.uri)
                username = os.environ.get('NEO4J_USER', neo4j_config.username)
                password = os.environ.get('NEO4J_PASSWORD', neo4j_config.password)
                database = os.environ.get('NEO4J_DATABASE', neo4j_config.database)

                return {
                    'uri': uri,
                    'user': username,
                    'password': password,
                    'database': database,
                    # Note: use_parallel_runtime would need to be passed
                    # to the driver after initialization if supported
                }

            case 'falkordb':
                if not HAS_FALKOR:
                    raise ValueError(
                        'FalkorDB driver not available in current graphiti-core version'
                    )

                # Use FalkorDB config if provided, otherwise use defaults
                if config.providers.falkordb:
                    falkor_config = config.providers.falkordb
                else:
                    # Create default FalkorDB configuration
                    from config.schema import FalkorDBProviderConfig

                    falkor_config = FalkorDBProviderConfig()

                # Check for environment variable overrides (for CI/CD compatibility)
                import os
                from urllib.parse import urlparse

                uri = os.environ.get('FALKORDB_URI', falkor_config.uri)
                username = os.environ.get('FALKORDB_USERNAME')
                password = os.environ.get('FALKORDB_PASSWORD', falkor_config.password)
                database = os.environ.get('FALKORDB_DATABASE', falkor_config.database)

                # Parse the URI to extract host and port
                parsed = urlparse(uri)
                host = parsed.hostname or 'localhost'
                port = parsed.port or 6379
                username = username or parsed.username or falkor_config.username

                return {
                    'driver': 'falkordb',
                    'host': host,
                    'port': port,
                    'username': username,
                    'password': password,
                    'database': database,
                }

            case _:
                raise ValueError(f'Unsupported Database provider: {provider}')
