from .agents import DataAnalyst
from .agents.entities import SessionType


data_analyst = DataAnalyst(session_type=SessionType.InMemory)
root_agent = data_analyst.agent
