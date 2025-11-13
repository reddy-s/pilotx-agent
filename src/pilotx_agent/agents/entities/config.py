from enum import Enum
from typing import Optional

from pydantic import BaseModel


class AgentType(Enum):
    DataAnalyst = "DataAnalyst"


class AgentConfig(BaseModel):
    name: str
    description: str
    instruction: str
    modelName: str = "openai/gpt-4o"
    outputKey: Optional[str] = None
    baseUrl: Optional[str] = None


class ContentRoles(Enum):
    User = "user"
    System = "system"
    Assistant = "assistant"


class SessionType(Enum):
    InMemory = "in-memory"
    Database = "Database"
    VertexAI = "VertexAI"
    Firestore = "Firestore"


class ResponseTypes(str, Enum):
    STRUCTURED_RESPONSE = "StructuredResponse"
    ERROR_RESPONSE = "ErrorResponse"
