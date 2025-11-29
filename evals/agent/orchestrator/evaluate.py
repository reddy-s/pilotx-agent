from typing import List, Union

from mlflow.genai import Scorer
from mlflow.genai.scorers import Guidelines, Correctness

from pilotx_agent.agents import Orchestrator

from ..commons import AbstractEvaluationRunner, UsesCorrectTools


class OrchestratorEvaluation(AbstractEvaluationRunner):
    def __init__(self):
        super().__init__(
            instance=Orchestrator(),
            experiment="pilotx_orchestrator_evaluation"
        )

    def get_scorers(self) -> List[Union[Guidelines, Scorer]]:
        scorers = [
            Correctness(),
            Guidelines(
                name="clarify",
                guidelines=["The response must be clear, coherent, and concise"],
            ),
            UsesCorrectTools(),
        ]
        return scorers

    def get_dataset(self) -> list[dict]:
        data = [
            {
                "inputs": {
                    "prompt": "What is the total head count for the Transportation Segment?"
                },
                "expectations": {
                    "expected_facts": [
                        "Transportation segment is 1,056 distinct personnel",
                    ]
                },
            }
        ]
        return data
