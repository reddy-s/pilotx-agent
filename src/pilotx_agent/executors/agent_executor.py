import logging
import os
import uuid

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    TaskState,
    Message,
    Role,
    TextPart,
    TaskStatusUpdateEvent,
    TaskStatus,
    DataPart,
    Part
)
from a2a.utils import new_task, new_agent_text_message
from google.adk.agents import SequentialAgent, ParallelAgent, LoopAgent
from google.protobuf import struct_pb2 as _struct_pb2
from litellm import ContextWindowExceededError

from ..agents.abstract import AbstractAgent, AbstractSequentialAgent
from ..auth import PilotXBackend
from ..utils import ExceededContextLength, UnauthorisedRequest

logger = logging.getLogger(__name__)


class PilotXAgentExecutor(AgentExecutor):
    """Generic Agent Executor that can work with any agent implementing AbstractAgent."""

    def __init__(
        self,
        agent: (
            AbstractAgent
            | SequentialAgent
            | ParallelAgent
            | LoopAgent
            | AbstractSequentialAgent
        ),
        streaming: bool = True,
    ):
        # Use the runner instead of the agent directly
        self.agent = agent
        self.streaming = streaming

    async def execute(
        self, context: RequestContext, event_queue: EventQueue, streaming: bool = None
    ) -> None:
        query = context.get_user_input()
        task = context.current_task or new_task(context.message)

        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            auth_success, user_info = await PilotXBackend.authenticate(context)

            if auth_success:
                user_id, user_name = user_info["uid"], user_info["name"]
            else:
                if os.getenv("DEV_MODE", "").lower() == "true":
                    auth_success, user_id, user_name = True, "test_user", "Test User"
                else:
                    raise UnauthorisedRequest(user_info["context"])

            if auth_success and user_id:
                async for event in self.agent.runner.stream(
                    prompt=query, user_id=user_id, session_id=task.context_id
                ):
                    if event["content"]:
                        if event["lastResponse"]:
                            if event["type"] == "text":
                                message = new_agent_text_message(
                                    text=event["content"],
                                    context_id=task.context_id,
                                    task_id=task.id,
                                )
                            elif event["type"] == "json":
                                message = Message(
                                    role=Role.agent,
                                    parts=[Part(root=DataPart(data=event["content"]))],
                                    message_id=str(uuid.uuid4()),
                                    task_id=task.id,
                                    context_id=task.context_id,
                                )
                            metadata = {
                                "type": event["type"],
                                "finished": False,
                                "lastResponse": event["lastResponse"],
                                "agent": event["agent"],
                            }
                            await updater.update_status(
                                TaskState.working, message=message, metadata=metadata
                            )
                        else:
                            # This is a partial/streaming response
                            metadata = _struct_pb2.Struct()
                            metadata.update(
                                {
                                    "type": event["type"],
                                    "lastResponse": event["lastResponse"],
                                    "finished": False,
                                    "agent": event["agent"],
                                    "function_name": event["function_name"],
                                }
                            )
                            if event["type"] == "json":
                                parts = [DataPart(data=event["content"])]
                            else:
                                parts = [TextPart(text=event["content"])]

                            await event_queue.enqueue_event(
                                TaskStatusUpdateEvent(
                                    status=TaskStatus(
                                        state=TaskState.working,
                                        message=Message(
                                            role=Role.agent,
                                            parts=parts,
                                            message_id=str(uuid.uuid4()),
                                            task_id=task.id,
                                            context_id=task.context_id,
                                            metadata=metadata,
                                        ),
                                    ),
                                    final=False,
                                    context_id=task.context_id,
                                    task_id=task.id,
                                )
                            )

                message = new_agent_text_message(
                    text="done",
                    context_id=task.context_id,
                    task_id=task.id,
                )
                state = await self.agent.runner.get_current_session_state(
                    app_name=self.agent.runner.runner.app_name, user_id=user_id, session_id=task.context_id
                )
                state = {
                    **state,
                    "type": "status",
                    "lastResponse": True,
                    "finished": True,
                    "agent": "Orchestrator",
                }
                await updater.update_status(
                    TaskState.completed, message=message, metadata=state
                )

        except UnauthorisedRequest as uar:
            await updater.update_status(
                TaskState.auth_required,
                new_agent_text_message(
                    text=uar.message, context_id=task.context_id, task_id=task.id
                ),
            )
        except ContextWindowExceededError as cwee:
            await updater.update_status(
                TaskState.rejected,
                new_agent_text_message(
                    text=cwee.message, context_id=task.context_id, task_id=task.id
                ),
            )
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(
                    text=error_message, context_id=task.context_id, task_id=task.id
                ),
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.error("Cancel not supported")
        raise Exception("cancel not supported")
