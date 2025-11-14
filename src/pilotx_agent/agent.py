from .agents import WorkflowAgent
from .agents.entities import SessionType


insights_workflow = WorkflowAgent(session_type=SessionType.InMemory)
root_agent = insights_workflow.agent
