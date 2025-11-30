import logging

from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse

from .abstract import AbstractAgent
from .entities import AgentConfig, AgentType, SessionType
from .utils import TokenUsage
from ..config import ServiceConfig
from .entities.completions import FAQProposerResponse

logger = logging.getLogger(__name__)


class FAQProposer(AbstractAgent):
    def __init__(self, session_type: SessionType = SessionType.InMemory) -> None:
        service_config = ServiceConfig.get_or_create_instance()
        base_agent_config = service_config.agents.get(AgentType.FAQProposer.value)

        super().__init__(
            agent_type=AgentType.FAQProposer,
            config=AgentConfig(**base_agent_config),
            global_instruction=service_config.globalInstruction,
            session_type=session_type,
            output_schema=FAQProposerResponse,
        )
