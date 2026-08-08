from fastapi import (
    FastAPI,
)

from app.api.routes import (
    router,
)


app = FastAPI(
    title="Support Assistant API",
    description=(
        "AI-powered customer support "
        "assistant using LangGraph "
        "and Retrieval-Augmented Generation."
    ),
    version="1.0.0",
)


app.include_router(
    router,
)


@app.get("/")
def root():
    """
    Health check endpoint.
    """

    return {
        "message": (
            "Support Assistant API is running."
        )
    }