import re

from mlflow.genai.scorers.base import Scorer


class TurnCounter(Scorer):
    def __init__(self):
        super().__init__(
            name="turn_counting",
            description="Counts number of questions the agent asks in a single turn.",
        )

    def __call__(self, **kwargs):
        response = kwargs.get("output", "")
        question_count = len(re.findall(r"\?", response))
        return question_count
