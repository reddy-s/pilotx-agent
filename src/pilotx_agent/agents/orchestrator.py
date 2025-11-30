import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse

from .abstract import AbstractAgent
from .entities import AgentConfig, AgentType, SessionType
from ..config import ServiceConfig
from .workflow import InsightsWorkflowAgent

logger = logging.getLogger(__name__)


class Orchestrator(AbstractAgent):
    def __init__(self, session_type: SessionType = SessionType.InMemory) -> None:
        service_config = ServiceConfig.get_or_create_instance()
        base_agent_config = service_config.agents.get(AgentType.Orchestrator.value)

        super().__init__(
            agent_type=AgentType.Orchestrator,
            config=AgentConfig(**base_agent_config),
            global_instruction=service_config.globalInstruction,
            session_type=session_type,
            sub_agents=[InsightsWorkflowAgent(session_type=session_type).agent],
        )

    def _after_model_callback(self, callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[
        LlmResponse]:
        if not hasattr(callback_context.state, "turn"):
            callback_context.state["turn"] = 0
        callback_context.state["turn"] += 1
