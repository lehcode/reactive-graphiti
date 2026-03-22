import asyncio
import logging
import os
import traceback
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from functools import partial

from fastapi import APIRouter, FastAPI, status
from graphiti_core.nodes import EpisodeType  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore

from graph_service.dto import AddEntityNodeRequest, AddMessagesRequest, Result
from graph_service.zep_graphiti import ZepGraphitiDep


class AsyncWorker:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.task = None
        self.settings = None
        self.MAX_RETRIES = 5
        self.RETRY_DELAY_BASE = 10

    async def worker(self):
        logger.info('AsyncWorker: Worker loop STARTED.')
        while True:
            try:
                job, retry_count = await self.queue.get()
                logger.info(f'AsyncWorker: DEQUEUED job. Remaining in queue: {self.queue.qsize()}')

                if self.settings is not None:
                    await job(self.settings)
                else:
                    logger.warning('AsyncWorker: Settings not initialized yet. Waiting...')
                    await asyncio.sleep(5)
                    await self.queue.put((job, retry_count))
            except asyncio.CancelledError:
                break
            except Exception as e:
                error_type = type(e).__name__
                if retry_count < self.MAX_RETRIES:
                    delay = self.RETRY_DELAY_BASE * (2**retry_count)
                    logger.error(
                        f'AsyncWorker: Job failed ({error_type}). Retry {retry_count + 1}/{self.MAX_RETRIES} in {delay}s...'
                    )
                    await asyncio.sleep(delay)
                    await self.queue.put((job, retry_count + 1))
                else:
                    logger.critical(
                        f'AsyncWorker: Job EXHAUSTED max retries. DISCARDING job. Error: {e}'
                    )
                    logger.debug(traceback.format_exc())
            finally:
                self.queue.task_done()

    async def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self.worker())
            logger.info('AsyncWorker: Safe background worker started.')

    async def stop(self):
        if self.task:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
        while not self.queue.empty():
            self.queue.get_nowait()


async_worker = AsyncWorker()

logger = logging.getLogger('GraphitiIngest')


@asynccontextmanager
async def lifespan(_: FastAPI):
    from graph_service.config import get_settings

    async_worker.settings = get_settings()
    await async_worker.start()
    yield
    await async_worker.stop()


router = APIRouter(lifespan=lifespan)


async def add_messages_task(request: AddMessagesRequest, settings: any):
    from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.prompts.models import Message as LLMMessage

    from ..zep_graphiti import create_configured_client

    g = create_configured_client(settings, request.llm_tier)
    try:
        for message in request.messages:
            try:
                content_to_ingest = message.content
                name_to_ingest = message.name or 'Unnamed Signal'
                source_desc = message.source_description or 'API Ingest'

                if request.simulate_slack:
                    logger.info(
                        f'Worker: GENERATING SLACK MESSAGE from prompt: {message.content[:50]}...'
                    )

                    slack_prompt = f"""
                    Convert the following prompt into a realistic Slack message from an Execue.io GTM Ops environment.
                    The message should look like it came from a professional Slack channel (e.g. #gtm-signals, #sales-alerts).
                    Include:
                    1. A realistic user handle (e.g. @sarah_ops, @mike_revops).
                    2. A timestamp or relative time (e.g. [10:42 AM]).
                    3. The channel name.
                    4. Professional but casual Slack formatting (emojis, bolding).

                    PROMPT: {message.content}

                    Respond with a JSON object:
                    {{
                        "slack_message": "the formatted slack message",
                        "summary": "a short technical name for this signal"
                    }}
                    """

                    try:
                        chat_model = os.getenv('AGENT_CHAT_MODEL') or 'tier1-qwen-ultra'
                        if request.llm_tier == 'tier2':
                            chat_model = 'tier2-gemini-3-flash'

                        chat_config = GraphitiLLMConfig(
                            api_key=settings.openai_api_key,
                            base_url=settings.openai_base_url,
                            model=chat_model,
                            temperature=0.7,
                        )
                        chat_client = OpenAIGenericClient(config=chat_config)

                        resp = await chat_client.generate_response(
                            messages=[LLMMessage(role='user', content=slack_prompt)]
                        )
                        content_to_ingest = resp.get('slack_message', message.content)
                        name_to_ingest = f'Slack: {resp.get("summary", "Signal")}'
                        source_desc = 'Slack (Generated)'
                        logger.info(
                            f'Worker: Slack generation SUCCESS using {chat_model}. New name: {name_to_ingest}'
                        )
                    except Exception as gen_err:
                        logger.error(f'Worker: Slack generation failed, falling back: {gen_err}')

                logger.info(
                    f"Worker: Processing signal '{name_to_ingest}' for group {request.group_id}"
                )

                reference_time = (
                    message.timestamp.isoformat()
                    if message.timestamp
                    else datetime.now(timezone.utc).isoformat()
                )

                await g.add_episode(
                    name=name_to_ingest,
                    episode_body=content_to_ingest,
                    source=EpisodeType.message,
                    source_description=source_desc,
                    group_id=request.group_id,
                    reference_time=reference_time,
                )
            except Exception as e:
                logger.error(f'Worker: Error processing message in batch: {e}')
                raise  # Re-raise to trigger AsyncWorker retry
    finally:
        await g.close()


@router.post('/messages', status_code=status.HTTP_202_ACCEPTED)
async def add_messages(
    request: AddMessagesRequest,
    graphiti: ZepGraphitiDep,
):
    await async_worker.queue.put((partial(add_messages_task, request), 0))
    return Result(message='Messages added to processing queue', success=True)


@router.post('/entity-node', status_code=status.HTTP_201_CREATED)
async def add_entity_node(
    request: AddEntityNodeRequest,
    graphiti: ZepGraphitiDep,
):
    node = await graphiti.save_entity_node(
        uuid=request.uuid,
        group_id=request.group_id,
        name=request.name,
        summary=request.summary,
    )
    return node


@router.delete('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def delete_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_entity_edge(uuid)
    return Result(message='Entity Edge deleted', success=True)


@router.delete('/group/{group_id}', status_code=status.HTTP_200_OK)
async def delete_group(group_id: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_group(group_id)
    return Result(message='Group deleted', success=True)


@router.delete('/episode/{uuid}', status_code=status.HTTP_200_OK)
async def delete_episode(uuid: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_episodic_node(uuid)
    return Result(message='Episode deleted', success=True)


@router.post('/clear', status_code=status.HTTP_200_OK)
async def clear(
    graphiti: ZepGraphitiDep,
):
    await clear_data(graphiti.driver)
    await graphiti.build_indices_and_constraints()
    return Result(message='Graph cleared', success=True)
