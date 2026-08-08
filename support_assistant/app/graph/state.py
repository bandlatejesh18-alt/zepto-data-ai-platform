from typing import TypedDict


class GraphState(
    TypedDict,
):
    """
    Shared state for
    the LangGraph workflow.
    """

    query: str

    intent: str

    answer: str

    sources: list[str]

    confidence: float