from pydantic import (
    BaseModel,
    Field,
)


class ChatRequest(
    BaseModel,
):
    """
    Request model for
    the chat endpoint.
    """

    query: str = Field(
        ...,
        description=(
            "User's question."
        ),
        examples=[
            "What is the refund policy?"
        ],
    )


class ChatResponse(
    BaseModel,
):
    """
    Response model for
    the chat endpoint.
    """

    answer: str = Field(
        ...,
        description=(
            "Generated answer."
        ),
    )

    sources: list[str] = Field(
        default_factory=list,
        description=(
            "Documents used "
            "to generate the answer."
        ),
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score "
            "between 0 and 1."
        ),
    )