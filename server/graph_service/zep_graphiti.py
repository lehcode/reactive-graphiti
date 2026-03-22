import logging
import os
from typing import Annotated

from fastapi import Depends, Header, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient  # type: ignore
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore

from graph_service.config import Settings, ZepEnvDep
from graph_service.dto import FactResult

logger = logging.getLogger(__name__)


class ZepGraphiti(Graphiti):
    def __init__(self, uri: str, user: str, password: str, llm_client: LLMClient | None = None):
        super().__init__(uri, user, password, llm_client)

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(
            name=name,
            uuid=uuid,
            group_id=group_id,
            summary=summary,
        )
        await new_node.generate_name_embedding(self.embedder)
        await new_node.save(self.driver)
        return new_node

    async def get_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            return edge
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_group(self, group_id: str):
        try:
            edges = await EntityEdge.get_by_group_ids(self.driver, [group_id])
        except GroupsEdgesNotFoundError:
            logger.warning(f'No edges found for group {group_id}')
            edges = []

        nodes = await EntityNode.get_by_group_ids(self.driver, [group_id])

        episodes = await EpisodicNode.get_by_group_ids(self.driver, [group_id])

        for edge in edges:
            await edge.delete(self.driver)

        for node in nodes:
            await node.delete(self.driver)

        for episode in episodes:
            await episode.delete(self.driver)

    async def delete_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            await edge.delete(self.driver)
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node(self, uuid: str):
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            await episode.delete(self.driver)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e


def create_configured_client(settings: Settings, llm_tier: str | None = None) -> ZepGraphiti:
    # Priority: Env Var > Pydantic Settings > Default
    model_name = os.getenv('GRAPHITI_EXTRACTOR_MODEL') or settings.model_name or 'tier1-graphiti-kg'
    embedding_model = (
        os.getenv('GRAPHITI_EMBEDDING_MODEL') or settings.embedding_model_name or 'tier1-embeddings'
    )

    # Handle Tier switching from UI (X-LLM-Tier header)
    if llm_tier == 'tier2':
        # Use specific aliases defined in the LiteLLM proxy config
        model_name = 'tier2-gemini-3-flash'
        embedding_model = 'tier2-gemini-embeddings'
        logger.info(f'UI requested Tier 2: Swapping models to {model_name} / {embedding_model}')

    # 1. Initialize custom embedder with correct model, base_url
    embedder_config = OpenAIEmbedderConfig(
        embedding_model=embedding_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        embedding_dim=1024,  # Standard for mxbai-embed-large
    )
    embedder = OpenAIEmbedder(config=embedder_config)

    # 2. Configure LLM properly
    llm_config = GraphitiLLMConfig(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=model_name,
        temperature=0,
    )
    llm_client = OpenAIGenericClient(config=llm_config)

    # 3. Create the Graphiti client with our custom LLM
    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        llm_client=llm_client,
    )

    # 4. Swap the default embedder
    client.embedder = embedder

    logger.info(f'Configured ZepGraphiti with LLM: {llm_config.model} at {llm_config.base_url}')
    return client


async def get_graphiti(settings: ZepEnvDep, x_llm_tier: str | None = Header(None)):
    client = create_configured_client(settings, x_llm_tier)
    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: ZepEnvDep):
    client = create_configured_client(settings)
    await client.build_indices_and_constraints()
    await client.close()


def get_fact_result_from_edge(edge: EntityEdge):
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
