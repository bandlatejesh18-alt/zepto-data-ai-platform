import os

from app.prompts import (
    build_prompt,
)

from app.retrieval import (
    retrieve_documents,
    format_context,
)

from app.graph.state import (
    GraphState,
)

from app.config import MOCK_LLM


def classify_intent(
    state: GraphState,
):
    """
    Classify the user query as either
    a policy question or a general question.
    """

    query = state[
        "query"
    ].lower()

    keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours",
    ]

    if any(
        keyword in query
        for keyword in keywords
    ):
        state[
            "intent"
        ] = "policy_question"

    else:
        state[
            "intent"
        ] = "general_question"

    return state


def retrieve_and_answer(
    state: GraphState,
):
    """
    Retrieve relevant documents
    and generate an answer for
    policy questions.
    """

    results = retrieve_documents(
        state["query"],
        top_k=3,
    )

    documents = results.get(
        "documents",
        [],
    )


    # Handle empty retrieval results

    if (
        not documents
        or not documents[0]
    ):

        state[
            "answer"
        ] = (
            "No relevant documents "
            "were found."
        )

        state[
            "sources"
        ] = []

        state[
            "confidence"
        ] = 0.0

        return state


    # Format retrieved context

    context = format_context(
        results,
    )


    # Get the most similar chunk

    top_chunk = documents[
        0
    ][
        0
    ]


    # MOCK_LLM is the required
    # graded baseline.

    if MOCK_LLM == "1":

        answer = (
            "Based on the retrieved "
            "context: "
            f"{top_chunk[:200]}"
        )


    else:

        # Optional real LLM path

        prompt = build_prompt(
            state["query"],
            context,
        )

        # Real LLM integration
        # will be added later.

        answer = (
            "Real LLM Response"
        )


    state[
        "answer"
    ] = answer


    # Extract document IDs
    # from ChromaDB metadata

    metadatas = results.get(
        "metadatas",
        [],
    )


    if metadatas:

        state[
            "sources"
        ] = [
            metadata[
                "document_id"
            ]
            for metadata
            in metadatas[0]
        ]

    else:

        state[
            "sources"
        ] = []


    # Simple retrieval-based
    # confidence score.

    distances = results.get(
        "distances",
        [],
    )


    if (
        distances
        and distances[0]
    ):

        top_distance = (
            distances[0][0]
        )

        state[
            "confidence"
        ] = round(
            max(
                0.0,
                1 - top_distance,
            ),
            2,
        )

    else:

        state[
            "confidence"
        ] = 0.0


    return state


def direct_answer(
    state: GraphState,
):
    """
    Generate a response for
    general questions.
    """
    
    if MOCK_LLM == "1":
        
        answer = (
            "I can only answer "
            "questions about "
            "Zepto policies "
            "right now."
        )

    else:

        prompt = build_prompt(
            state["query"],
            "",
        )

        # Real LLM integration
        # will be added later.

        answer = "Real LLM Response"

    state[
        "answer"
    ] = answer

    state[
        "sources"
    ] = []

    state[
        "confidence"
    ] = 1.0

    return state