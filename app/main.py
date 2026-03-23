"""FastAPI application -- language feedback endpoint."""

from dotenv import load_dotenv
load_dotenv()  

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.feedback import get_feedback
from app.models import FeedbackRequest, FeedbackResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")

app = FastAPI(
    title="Language Feedback API",
    description="Analyses learner-written sentences and returns structured correction feedback.",
    version="1.0.0",
)

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logging.getLogger(__name__).exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    try:
        return await get_feedback(request)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        try:
            from anthropic import APIError as AnthropicError
            if isinstance(exc, AnthropicError):
                raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ImportError:
            pass
        try:
            from openai import APIError as OpenAIError
            if isinstance(exc, OpenAIError):
                raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ImportError:
            pass
        raise
