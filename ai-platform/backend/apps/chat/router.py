"""
API router for conversational AI chat service
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import structlog

from backend.core.database import get_db
from backend.apps.chat.chat_service import chat_service

logger = structlog.get_logger()

router = APIRouter()


class ChatStartRequest(BaseModel):
    """Request model for starting a chat session."""
    context_domain: str = "general"


class ChatMessageRequest(BaseModel):
    """Request model for sending a chat message."""
    message: str
    context_override: Optional[str] = None


@router.post("/session/start")
async def start_chat_session(
    request: ChatStartRequest,
    user_id: int,  # Would come from authentication middleware
    db = Depends(get_db)
):
    """
    Start a new conversational AI chat session.

    This endpoint initiates a context-aware conversation
    that can help with various healthcare topics.
    """
    try:
        session = await chat_service.start_chat_session(user_id, request.context_domain)

        if "error" in session:
            raise HTTPException(
                status_code=500,
                detail=session["error"]
            )

        return session

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error starting chat session", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to start chat session"
        )


@router.post("/session/{session_id}/message")
async def send_chat_message(
    session_id: int,
    request: ChatMessageRequest,
    user_id: int,  # Would come from authentication middleware
    db = Depends(get_db)
):
    """
    Send a message in an ongoing chat session.

    This endpoint processes user messages and returns
    contextual AI responses with follow-up suggestions.
    """
    try:
        response = await chat_service.process_chat_message(
            user_id,
            request.message,
            request.context_override
        )

        if "error" in response:
            raise HTTPException(
                status_code=500,
                detail=response["error"]
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing chat message", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to process chat message"
        )


@router.post("/session/{session_id}/context/switch")
async def switch_chat_context(
    session_id: int,
    new_context: str,
    user_id: int,  # Would come from authentication middleware
    db = Depends(get_db)
):
    """
    Switch the context domain of an ongoing chat session.

    This endpoint allows users to change the focus of
    their conversation to different healthcare topics.
    """
    try:
        switch_result = await chat_service.switch_context(user_id, new_context)

        if "error" in switch_result:
            raise HTTPException(
                status_code=400,
                detail=switch_result["error"]
            )

        return switch_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error switching chat context", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to switch chat context"
        )


@router.post("/session/end")
async def end_chat_session(
    user_id: int,  # Would come from authentication middleware
    db = Depends(get_db)
):
    """
    End the current chat session.

    This endpoint concludes the conversation and
    provides a summary of the discussion.
    """
    try:
        summary = await chat_service.end_chat_session(user_id)

        if "error" in summary:
            raise HTTPException(
                status_code=400,
                detail=summary["error"]
            )

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error ending chat session", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to end chat session"
        )


@router.get("/history/{user_id}")
async def get_chat_history(
    user_id: int,
    limit: int = 50,
    db = Depends(get_db)
):
    """
    Get chat history for a user.

    This endpoint returns conversation history
    with insights and topic summaries.
    """
    try:
        history = await chat_service.get_chat_history(user_id, limit)

        if "error" in history:
            raise HTTPException(
                status_code=500,
                detail=history["error"]
            )

        return history

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting chat history", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get chat history"
        )


@router.get("/contexts")
async def get_available_contexts():
    """
    Get available conversation context domains.

    This endpoint returns all available context domains
    that users can switch to during conversations.
    """
    try:
        contexts = [
            {
                "id": "general",
                "name": "General Healthcare",
                "description": "General questions about health and wellness",
                "icon": "💬"
            },
            {
                "id": "guidelines",
                "name": "Clinical Guidelines",
                "description": "Evidence-based treatment guidelines and protocols",
                "icon": "📋"
            },
            {
                "id": "trials",
                "name": "Clinical Trials",
                "description": "Finding and understanding clinical research studies",
                "icon": "🔬"
            },
            {
                "id": "medication",
                "name": "Medication Management",
                "description": "Information about medications and drug interactions",
                "icon": "💊"
            },
            {
                "id": "symptoms",
                "name": "Symptom Checker",
                "description": "Understanding symptoms and when to seek care",
                "icon": "🔍"
            },
            {
                "id": "mental_health",
                "name": "Mental Health Support",
                "description": "Emotional support and mental wellness resources",
                "icon": "🧠"
            }
        ]

        return {
            "contexts": contexts,
            "total_contexts": len(contexts)
        }

    except Exception as e:
        logger.error("Error getting available contexts", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get available contexts"
        )
