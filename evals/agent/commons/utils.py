import asyncio
import logging
import litellm
import mlflow

from typing import Union
from pilotx_agent.agents.abstract import AbstractAgent, AbstractSequentialAgent


@mlflow.trace
def run_agent(
    prompt: str,
    instance: Union[AbstractAgent, AbstractSequentialAgent],
    user_id: str,
    session_id: str,
    timeout: int = 1200,
) -> str:
    async def get_response():
        try:
            res = await asyncio.wait_for(
                instance.runner.invoke(
                    prompt=prompt,
                    user_id=user_id,
                    session_id=session_id,
                ),
                timeout=timeout,
            )
            return res
        except asyncio.TimeoutError:
            logging.warning(
                f"Agent execution timed out for prompt: {prompt[:100]}..."
            )
            return "Error: Agent execution timed out."
        except asyncio.CancelledError:
            logging.warning(
                f"Agent execution cancelled for prompt: {prompt[:100]}..."
            )
            return "Error: Agent execution cancelled."
        except litellm.exceptions.RateLimitError:
            logging.warning(
                f"Agent execution hit rate limit for prompt: {prompt[:100]}..."
            )
            return "Error: Agent execution hit rate limit."
        except Exception as e:
            logging.error(f"Agent execution failed: {e}")
            return f"Error: Agent execution failed - {str(e)}"

    try:
        # Run the async function in a fresh event loop
        response = asyncio.run(get_response())
        return response
    except Exception as e:
        logging.error(f"Evaluation function failed: {e}")
        return f"Evaluation Error: {str(e)}"