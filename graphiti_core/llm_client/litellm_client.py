"""
Copyright 2024, Zep Software, Inc.
Copyright 2025-2026, Anton Repin <robot@pimeleon.org>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import re
import typing
from typing import Any, Literal

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient, get_extraction_language_instruction
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import EmptyResponseError, RateLimitError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gpt-4.1-mini'

StructuredOutputMode = Literal['json_schema', 'json_object']


class LiteLLMClient(LLMClient):
    """
    LiteLLMClient is a client class for interacting with various LLM models via LiteLLM.

    This class extends the LLMClient and provides methods to initialize the client,
    get an embedder, and generate responses from the language model.

    This client targets any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI,
    vLLM, llama.cpp, Ollama, DeepSeek, Together, etc.). It defaults to native
    ``json_schema`` structured output (constrained decoding) and can fall back to
    ``json_object`` for the minority of providers that do not support ``json_schema``.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.
        structured_output_mode (StructuredOutputMode): How structured output is requested.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = 16384,
        structured_output_mode: StructuredOutputMode = 'json_schema',
    ):
        """
        Initialize the LiteLLMClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 16384 (16K) for better compatibility with local models.
            structured_output_mode (StructuredOutputMode): Whether to request structured
                output via native ``json_schema`` (the default, uses constrained decoding)
                or to fall back to ``json_object``. Set to ``'json_object'`` for providers
                that do not support the ``json_schema`` response format (e.g. DeepSeek); in
                that mode the schema is injected into the prompt instead of being enforced
                by the API.

        """
        # removed caching to simplify the `generate_response` override
        if cache:
            raise NotImplementedError('Caching is not implemented for OpenAI')

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        # Override max_tokens to support higher limits for local models
        self.max_tokens = max_tokens
        self.structured_output_mode: StructuredOutputMode = structured_output_mode

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    def _build_response_format(self, response_model: type[BaseModel] | None) -> dict[str, Any]:
        """Build the ``response_format`` payload for the chat completion request.

        Uses native ``json_schema`` when a response model is provided and the client is in
        ``json_schema`` mode; otherwise falls back to ``json_object``. In ``json_object``
        mode the schema is not enforced by the API — ``generate_response`` injects it into
        the prompt instead.
        """
        if response_model is None or self.structured_output_mode == 'json_object':
            return {'type': 'json_object'}

        # Native json_schema. We intentionally omit "strict": true — strict mode requires
        # the schema to meet OpenAI's strict subset (additionalProperties: false, every
        # field required), which raw model_json_schema() routinely violates (that's why the
        # dedicated OpenAIClient uses responses.parse() instead). So adherence is best-effort
        # on OpenAI-proper; constrained-decoding servers (vLLM, llama.cpp) still enforce it.
        return {
            'type': 'json_schema',
            'json_schema': {
                'name': getattr(response_model, '__name__', 'structured_response'),
                'schema': response_model.model_json_schema(),
            },
        }

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Strip a wrapping markdown code fence from a JSON payload.

        OpenAI-compatible models served via Ollama/llama.cpp etc. frequently wrap their
        output in a ```json … ``` fence even when a json_schema/json_object response_format
        is requested, which breaks a bare ``json.loads``. No-op when there is no fence.
        """
        stripped = text.strip()
        if stripped.startswith('```'):
            stripped = re.sub(r'^```[a-zA-Z0-9_-]*[ \t]*\r?\n?', '', stripped)
            stripped = re.sub(r'\r?\n?```[ \t]*$', '', stripped)
        return stripped.strip()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract the first valid JSON object from text that may contain trailing content.

        Some OpenAI-compatible providers return a JSON object followed by explanatory
        prose, which breaks a bare ``json.loads`` with an "Extra data" error. This finds
        and returns the first complete top-level object by brace matching. Fenced payloads
        should be passed through ``_strip_code_fences`` first.

        Args:
            text: Raw response text that may contain JSON with trailing content

        Returns:
            Parsed JSON as a dictionary

        Raises:
            json.JSONDecodeError: If no valid JSON object can be extracted
        """
        text = text.strip()

        # Try standard parsing first (fast path)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Only handle "Extra data" errors - other errors should propagate
            if 'Extra data' not in str(e):
                raise

        # Find the first complete JSON object by matching braces
        if not text.startswith('{'):
            raise json.JSONDecodeError('No JSON object found', text, 0)

        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    # Found complete JSON object
                    json_str = text[: i + 1]
                    return json.loads(json_str)

        raise json.JSONDecodeError('Incomplete JSON object', text, len(text))

    @staticmethod
    def _is_schema_returned_as_data(response: dict[str, Any]) -> bool:
        """Detect if the model returned the JSON Schema definition instead of data.

        Some providers (e.g. LiteLLM with Gemini) receiving a ``json_schema`` response
        format echo the schema definition itself back rather than data conforming to it.

        Args:
            response: The parsed JSON response from the LLM

        Returns:
            True if the response appears to be a JSON Schema definition
        """
        # JSON Schema keywords that are never present in real extracted data
        schema_keywords = {'$defs', '$schema', 'definitions', 'properties'}
        if any(key in response for key in schema_keywords):
            return True

        # A bare top-level "type": "object" is another JSON Schema tell
        return response.get('type') == 'object'

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})
        try:
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                response_format=self._build_response_format(response_model),  # type: ignore[arg-type]
            )
            result = response.choices[0].message.content or ''
            # An empty body (refusal, length finish_reason, or a flaky endpoint) would make
            # json.loads raise a cryptic JSONDecodeError; surface a clear error instead.
            if not result:
                raise EmptyResponseError('LLM returned an empty response')
            # Many OpenAI-compatible/local models wrap JSON in a ```json fence even under a
            # structured response_format; strip it before parsing. Some also append prose
            # after the object, so extract the first complete object rather than bare-loading.
            return self._extract_json(self._strip_code_fences(result))
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
    ) -> dict[str, typing.Any]:
        self._apply_attribute_extraction_preamble(messages, attribute_extraction)
        if max_tokens is None:
            max_tokens = self.max_tokens

        # In json_object fallback mode the API does not enforce the schema, so embed it in
        # the prompt to guide the model. In json_schema mode the schema is enforced via
        # response_format, so no prompt injection is needed.
        if response_model is not None and self.structured_output_mode == 'json_object':
            serialized_model = json.dumps(response_model.model_json_schema())
            messages[
                -1
            ].content += (
                f'\n\nRespond with a JSON object in the following format:\n\n{serialized_model}'
            )

        # Add multilingual extraction instructions
        messages[0].content += get_extraction_language_instruction(group_id)

        # Wrap entire operation in tracing span
        with self.tracer.start_span('llm.generate') as span:
            attributes = {
                'llm.provider': 'litellm',
                'model.size': model_size.value,
                'max_tokens': max_tokens,
                'structured_output.mode': self.structured_output_mode,
            }
            if prompt_name:
                attributes['prompt.name'] = prompt_name
            span.add_attributes(attributes)

            # Track whether we've already flipped to json_object mode within this call so a
            # provider that returns the schema even in fallback mode fails loudly instead of
            # looping.
            schema_fallback_attempted = False
            while True:
                try:
                    # Delegate to the base tenacity wrapper so transient JSONDecodeError /
                    # RateLimitError get backoff-retried (4 attempts) — most relevant in the
                    # json_object fallback path for less-reliable providers. This is the clean
                    # retry mechanism (same pattern as Gliner2Client); the old hand-rolled
                    # re-prompt loop is intentionally not reinstated.
                    response = await self._generate_response_with_retry(
                        messages, response_model, max_tokens=max_tokens, model_size=model_size
                    )
                except Exception as e:
                    span.set_status('error', str(e))
                    span.record_exception(e)
                    raise

                # Some providers (e.g. LiteLLM with Gemini) echo the JSON Schema definition
                # back instead of data when json_schema is requested. Detect that, flip to
                # json_object mode (schema injected into the prompt), and retry once.
                if (
                    response_model is not None
                    and self.structured_output_mode == 'json_schema'
                    and self._is_schema_returned_as_data(response)
                ):
                    if schema_fallback_attempted:
                        raise ValueError(
                            'Provider returned schema definition even in json_object fallback mode'
                        )
                    logger.warning(
                        'Provider returned schema definition instead of data. '
                        'Switching to json_object mode with embedded schema.'
                    )
                    self.structured_output_mode = 'json_object'
                    schema_fallback_attempted = True
                    span.add_attributes({'structured_output.fallback_triggered': True})
                    serialized_model = json.dumps(response_model.model_json_schema())
                    messages[-1].content += (
                        '\n\nRespond with a JSON object in the following format:'
                        f'\n\n{serialized_model}'
                    )
                    continue

                return response
