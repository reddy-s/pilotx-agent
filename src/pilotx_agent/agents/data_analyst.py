import logging
import json

from typing import Dict, Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools import BaseTool, ToolContext

from toolbox_core import ToolboxSyncClient

from .abstract import AbstractAgent
from .entities import (
    AgentConfig,
    AgentType,
    SessionType,
    ToolResponse,
    DataAnalystResponse,
)
from .plugins import JailbreakDetector
from .utils import TokenUsage, MlflowTracedSyncTool
from ..config import ServiceConfig

logger = logging.getLogger(__name__)


class DataAnalyst(AbstractAgent):
    def __init__(self, session_type: SessionType = SessionType.InMemory) -> None:
        service_config = ServiceConfig.get_or_create_instance()
        base_agent_config = service_config.agents.get(AgentType.DataAnalyst.value)
        toolbox_config = service_config.toolbox

        toolbox = ToolboxSyncClient(toolbox_config.get("uri"))
        toolbox_tools = toolbox.load_toolset(toolbox_config.get("toolsetId"))

        traced_tools = [MlflowTracedSyncTool(t) for t in toolbox_tools]

        super().__init__(
            agent_type=AgentType.DataAnalyst,
            config=AgentConfig(**base_agent_config),
            global_instruction=service_config.globalInstruction,
            tools=[*traced_tools],
            session_type=session_type,
            plugins=[JailbreakDetector()],
            output_schema=DataAnalystResponse,
        )

    def _after_tool_callback(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        tool_context: ToolContext,
        tool_response: dict,
    ) -> Optional[dict]:
        logger.info(
            f"DataAnalyst: after_tool_callback invoked with tool_response: {tool_response}"
        )
        return tool_response
