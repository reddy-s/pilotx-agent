import mlflow
from mlflow.entities import SpanType


class MlflowTracedSyncTool:
    """Wraps a ToolboxSyncTool (or any callable) and adds an MLflow TOOL span."""

    def __init__(self, inner):
        # Use object.__setattr__ to avoid recursion with __setattr__ / __getattr__
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        # Delegate everything we don't override to the inner tool
        return getattr(self._inner, name)

    @property
    def name(self) -> str:
        return self._inner._name

    def __call__(self, *args, **kwargs):
        # Try to get a nice name; fall back to class name
        tool_name = self.name

        with mlflow.start_span(
            name=tool_name,
            span_type=SpanType.TOOL,
        ) as span:
            # Optional: add some metadata
            span.set_attributes(
                {
                    "toolbox.tool_name": tool_name,
                    "toolbox.type": type(self._inner).__name__,
                }
            )

            result = self._inner(*args, **kwargs)

            # Capture outputs if you want them visible in the trace
            span.set_outputs({"result": result})

            return result