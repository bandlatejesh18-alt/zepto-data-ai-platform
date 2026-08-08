from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.graph.nodes import (
    classify_intent,
    direct_answer,
    retrieve_and_answer,
)

from app.graph.state import (
    GraphState,
)


graph = StateGraph(
    GraphState,
)


graph.add_node(
    "classify_intent",
    classify_intent,
)

graph.add_node(
    "retrieve_and_answer",
    retrieve_and_answer,
)

graph.add_node(
    "direct_answer",
    direct_answer,
)


def route_intent(
    state: GraphState,
):
    """
    Route based on
    classified intent.
    """

    return state[
        "intent"
    ]


graph.add_edge(
    START,
    "classify_intent",
)

graph.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "policy_question":
            "retrieve_and_answer",

        "general_question":
            "direct_answer",
    },
)

graph.add_edge(
    "retrieve_and_answer",
    END,
)

graph.add_edge(
    "direct_answer",
    END,
)


support_graph = graph.compile()