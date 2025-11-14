from .agents import Orchestrator
from .agents.entities import SessionType


orchestrator = Orchestrator(session_type=SessionType.InMemory)
root_agent = orchestrator.agent
