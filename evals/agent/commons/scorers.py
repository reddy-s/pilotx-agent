import re
from typing import Any

from mlflow.entities import Feedback, SpanType, Trace
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


class UsesCorrectTools(Scorer):
    def __init__(self):
        super().__init__(
            name="uses_correct_tools",
            description="Evaluate if agent used tools appropriately",
        )

    def __call__(self, *, inputs: Any = None, outputs: Any = None, expectations: dict[str, Any] | None = None,
                 trace: Trace | None = None) -> int | float | bool | str | Feedback | list[Feedback]:
        """Evaluate if agent used tools appropriately"""
        expected_tools = expectations["tool_calls"]

        tool_spans = trace.search_spans(span_type=SpanType.TOOL)
        tool_names = list(set([s.name for s in tool_spans]))

        score = "yes" if tool_names == expected_tools else "no"
        rationale = (
            "The agent used the correct tools."
            if tool_names == expected_tools
            else f"The agent used the incorrect tools: {tool_names}"
        )
        # Return a Feedback object with the score and rationale
        return Feedback(value=score, rationale=rationale)
