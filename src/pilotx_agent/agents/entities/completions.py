from typing import Literal, Optional, List, Dict, Union
from pydantic import BaseModel, Field


class VisualizationResponse(BaseModel):
    visualizationType: Literal[
        "bar", "pie", "kpi", "line", "table", "heatmap", "scatter", "text"
    ] = Field(
        ..., description="Visualization type. Use 'text' for plain textual output."
    )
    title: str = Field(..., description="Human-readable title for the visualization.")
    subtitle: Optional[str] = Field(
        default=None, description="Optional subtitle or note under the title."
    )

    dimensions: List[str] = Field(
        default_factory=list,
        description="Names of categorical/temporal columns (must exist in `data`).",
    )
    measures: List[str] = Field(
        default_factory=list,
        description="Names of numeric columns (must exist in `data`).",
    )
    data: Dict[str, List[Union[str, int, float]]] = Field(
        default_factory=dict,
        description="Columnar dataset: {column: [values...]}. All arrays equal length. Max 15 rows.",
    )

    # Used when visualizationType='text'
    text: Optional[str] = Field(
        default=None,
        description="Plain text/markdown to display when visualizationType='text'.",
    )


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


class ErrorResponse(BaseModel):
    type: Literal["ErrorResponse"] = "ErrorResponse"
    error_description: str


class ToolResponse(BaseModel):
    type: Literal["VisualizationResponse", "ErrorResponse"] = "VisualizationResponse"
    context: VisualizationResponse | ErrorResponse
    usage: Optional[TokenUsage]
    data: Optional[list[dict] | dict]

    class Config:
        arbitrary_types_allowed = True
