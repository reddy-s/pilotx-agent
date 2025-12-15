from typing import Literal, Optional, List, Dict, Union
from pydantic import BaseModel, Field


class VisualizationResponse(BaseModel):
    visualizationType: Literal[
        "bar", "pie", "kpi", "line", "text"
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
    explanation: str = Field(
        default=None,
        description="Textual explanation or insights about the visualization.",
    )


class DataAnalystResponse(BaseModel):
    data: List[VisualizationResponse] = Field(
        ..., description="List of visualizations."
    )


class FAQ(BaseModel):
    question: str = Field(..., description="Next Question")
    netInformationGainScore: float = Field(
        ...,
        description="Estimated Net Information Gain Score if the user asks the question. RANGE: 0.0 - 5.0",
    )


class FAQProposerResponse(BaseModel):
    faqs: List[FAQ] = Field(..., description="List of FAQs of the json structure { \"question\": \"question\", \"netInformationGainScore\": 0.0 }")


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
