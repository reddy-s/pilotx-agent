from .abstract import AbstractSequentialAgent
from .data_analyst import DataAnalyst
from .business_analyst import BusinessAnalyst
from .entities.config import SessionType
from .faq_proposer import FAQProposer


class InsightsWorkflowAgent(AbstractSequentialAgent):
    def __init__(self, session_type: SessionType = SessionType.InMemory) -> None:
        super().__init__(
            name="InsightsWorkflow",
            description="Agent to manage the workflow of translating a user's question into actionable business insights.",
            sub_agents=[
                DataAnalyst(session_type=session_type).agent,
                BusinessAnalyst(session_type=session_type).agent,
                FAQProposer(session_type=session_type).agent,
            ],
            session_type=session_type,
        )
