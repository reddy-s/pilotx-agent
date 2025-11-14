import logging

from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse

from .abstract import AbstractAgent
from .entities import AgentConfig, AgentType, SessionType
from .utils import TokenUsage
from ..config import ServiceConfig

logger = logging.getLogger(__name__)


class BusinessAnalyst(AbstractAgent):
    def __init__(self, session_type: SessionType = SessionType.InMemory) -> None:
        service_config = ServiceConfig.get_or_create_instance()
        base_agent_config = service_config.agents.get(AgentType.BusinessAnalyst.value)

        super().__init__(
            agent_type=AgentType.BusinessAnalyst,
            config=AgentConfig(**base_agent_config),
            global_instruction=service_config.globalInstruction,
            session_type=session_type,
        )

    def _after_model_callback(
        self, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        if llm_response.usage_metadata and not llm_response.partial:
            input_tokens = llm_response.usage_metadata.prompt_token_count
            total_tokens = llm_response.usage_metadata.total_token_count
            turn_output_tokens = llm_response.usage_metadata.candidates_token_count
            turn_cost, input_cost, _ = TokenUsage.compute_token_cost(
                input_tokens, turn_output_tokens
            )
            used = TokenUsage.get_used_context_length(total_tokens)

            callback_context.state["turnUsage"] = {
                "turnCost": turn_cost,
                "inputCost": input_cost,
                "turnInputTokens": input_tokens,
                "totalTokens": total_tokens,
                "contextUsed": used,
            }

            usage_state = callback_context.state.get("app:convUsage", None)
            if not usage_state:
                usage_state = {
                    "totalCost": 0.0,
                    "totalInputCost": 0.0,
                    "totalTokens": 0,
                    "totalInputTokens": 0,
                    "contextUsed": 0,
                }

            usage_state["totalCost"] += callback_context.state["turnUsage"]["turnCost"]
            usage_state["totalInputCost"] += callback_context.state["turnUsage"][
                "inputCost"
            ]
            usage_state["totalTokens"] = callback_context.state["turnUsage"][
                "totalTokens"
            ]
            usage_state["totalInputTokens"] = callback_context.state["turnUsage"][
                "turnInputTokens"
            ]
            usage_state["contextUsed"] = callback_context.state["turnUsage"][
                "contextUsed"
            ]
            callback_context.state["app:convUsage"] = usage_state
