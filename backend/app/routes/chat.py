
from typing import Optional, Literal, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.chat_service import generate_chat_response


router = APIRouter(
    prefix="/chat",
    tags=["AI Chat"]
)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class MultimodalContext(BaseModel):
    facial_analysis: Optional[dict[str, Any]] = None
    voice_analysis: Optional[dict[str, Any]] = None
    transcript: Optional[str] = None
    linguistic_analysis: Optional[dict[str, Any]] = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)

    history: list[HistoryMessage] = Field(
        default_factory=list
    )

    # Retained for backward compatibility
    detected_emotion: Optional[str] = None

    # New multimodal context
    analysis_context: Optional[MultimodalContext] = None


@router.post("/message")
async def chat_message(request: ChatRequest):

    try:
        history = [
            item.model_dump()
            for item in request.history[-12:]
        ]

        analysis_context = (
            request.analysis_context.model_dump(exclude_none=True)
            if request.analysis_context
            else None
        )

        reply = await generate_chat_response(
            message=request.message,
            history=history,
            detected_emotion=request.detected_emotion,
            analysis_context=analysis_context
        )

        return {
            "success": True,
            "reply": reply
        }

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        )

    except Exception as error:
        print("Gemini API error:", error)

        raise HTTPException(
            status_code=502,
            detail="Unable to generate an AI response."
        )
