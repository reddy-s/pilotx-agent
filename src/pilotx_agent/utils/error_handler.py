import logging
import asyncio

from functools import wraps
from tenacity import RetryError
from typing import Any, cast, Dict, Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=Callable[..., dict[str, Any]])


def handle_tool_error(
    tool_name: str, error: Exception | str, detail: str | None = None
) -> dict[str, Any]:

    if isinstance(error, Exception):
        error_name = type(error).__name__
        error_detail = str(error)
    else:
        error_name = error
        error_detail = detail or "Unknown error"

    logger.error(f"Error in {tool_name} tool: {error_name} - {error_detail}")

    return {
        "status": "error",
        "tool": tool_name,
        "error_name": error_name,
        "error_message": error_detail,
    }


def handle_tool_error_with_message(
    error_name: str, error_detail: str, tool_name: str
) -> Dict[str, Any]:
    """
    Handles tool errors by logging them and returning a standardized error response.
    """
    logger.error(f"Error in {tool_name} tool: {error_name} - {error_detail}")

    return {
        "status": "error",
        "tool": tool_name,
        "error_name": error_name,
        "error_message": error_detail,
    }


def wrap_tool_with_retry_handling(tool_name: str):
    """
    Decorator that wraps a retried tool function to catch RetryError and return
    structured ADK-style error dicts instead of crashing the agent.

    Usage:
    ------
    @wrap_tool_with_retry_handling("get_fields")
    def get_fields(tool_context): ...
    """

    def decorator(func: T) -> T:
        def _handle_exception(e: Exception) -> dict[str, Any]:
            if isinstance(e, RetryError):
                last_exc = e.last_attempt.exception()
                logger.error(
                    f"[tool: {tool_name}] Retry failed after all attempts: {last_exc}"
                )
                return handle_tool_error(tool_name, last_exc)
            else:
                return handle_tool_error(tool_name, e)

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def wrapper(*args, **kwargs) -> dict[str, Any]:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    return _handle_exception(e)

            return cast(T, wrapper)
        else:

            @wraps(func)
            def wrapper(*args, **kwargs) -> dict[str, Any]:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return _handle_exception(e)

            return wrapper

    return decorator
