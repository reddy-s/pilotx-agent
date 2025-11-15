import os
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_fixed
from google.adk.tools import FunctionTool

from ...utils import EnvironmentVariableNotFound

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", None)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    reraise=True,
)
def search_web_for_latest_information(query: str, max_results: int = 3) -> dict:
    """
    Executes a web search for the latest information using the Tavily API. The search
    includes the provided query, and optionally limits the number of results to the
    specified maximum. Returns a dictionary containing a response with the answer and
    first result.

    This function retries the API call up to three times in case of failures, waiting
    two seconds between each attempt, and re-raises any exceptions that occur.

    :param query: The search query string to look for information.
    :type query: str
    :param max_results: The maximum number of results to retrieve. Defaults to 1.
    :type max_results: int
    :return: A dictionary containing the response with the primary answer and
        optionally the first result from the search.
    :rtype: dict
    """
    if TAVILY_API_KEY is None:
        raise EnvironmentVariableNotFound("TAVILY_API_KEY")
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    response = tavily_client.search(
        query=query,
        max_results=max_results,
        include_answer=True,
    )
    results = response.get("results", [{}])
    if len(results) == 0:
        return {
            "answer": response.get("answer", None),
        }
    return {"answer": response.get("answer", None), **results[0]}


func_search_web_for_latest_information = FunctionTool(
    func=search_web_for_latest_information
)
