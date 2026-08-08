from fastapi import (
    APIRouter,
)

from app.api.models import (
    ChatRequest,
    ChatResponse,
)

from app.graph.graph import (
    support_graph,
)

from app.graph.state import (
    GraphState,
)


router = APIRouter()


@router.post(
    "/ask",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):
    """
    Answer user questions
    using the LangGraph
    workflow.
    """

    state: GraphState = {
        "query": request.query,
        "intent": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0,
    }

    result = support_graph.invoke(
        state,
    )

    return ChatResponse(
        answer=result[
            "answer"
        ],
        sources=result[
            "sources"
        ],
        confidence=result[
            "confidence"
        ],
    )