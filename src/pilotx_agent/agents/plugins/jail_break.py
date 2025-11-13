import logging

from typing import Optional

from google.adk.agents import InvocationContext
from google.adk.plugins import BasePlugin
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JailbreakDetector(BasePlugin):
    def __init__(self) -> None:
        super().__init__(name="JailbreakDetector")

    async def on_user_message_callback(
        self, *, invocation_context: InvocationContext, user_message: types.Content
    ) -> Optional[types.Content]:
        logger.info(
            f"JailbreakDetector: on_user_message_callback invoked with user_message: {user_message}"
        )
