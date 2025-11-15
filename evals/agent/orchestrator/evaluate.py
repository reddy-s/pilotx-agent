from typing import List, Union

from mlflow.genai import Scorer
from mlflow.genai.scorers import Guidelines

from pilotx_agent.agents import Orchestrator

from ..commons import AbstractEvaluationRunner


class OrchestratorEvaluation(AbstractEvaluationRunner):
    def __init__(self):
        super().__init__(
            instance=Orchestrator(),
            experiment="pilotx_orchestrator_evaluation"
        )

    def get_scorers(self) -> List[Union[Guidelines, Scorer]]:
        scorers = [
            Guidelines(
                name="english",
                guidelines=["The response must be in English"],
            ),
            Guidelines(
                name="clarify",
                guidelines=["The response must be clear, coherent, and concise"],
            ),
        ]
        return scorers

    def get_dataset(self) -> list[dict]:
        data = [
            {
                "inputs": {"prompt": "What is the total head count for the Transportation Segment?"},
                "expectations": {
                    "expected_facts": [
                        "Orchestrator hands it over to the Insights Generation Workflow",
                        "execute_sql_tool is used with sql query",
                        "Response returned by the tool helps answer the question",
                        "Agent responds back with insights"
                    ]
                }
            }
        ]
        return data
