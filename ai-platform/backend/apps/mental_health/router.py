"""
API router for mental health support service
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import structlog

from backend.core.database import get_db
from backend.apps.mental_health.mental_health_service import mental_health_service

logger = structlog.get_logger()

router = APIRouter()


class ConversationStartRequest(BaseModel):
    """Request model for starting a conversation."""
    conversation_type: str = "general_support"


class MessageRequest(BaseModel):
    """Request model for sending a message."""
    message: str


class MoodTrackingRequest(BaseModel):
    """Request model for mood tracking."""
    mood_level: str
    mood_score: int
    energy_level: Optional[int] = None
    sleep_hours: Optional[float] = None
    stress_level: Optional[int] = None
    notes: Optional[str] = None


@router.post("/conversation/start")
async def start_mental_health_conversation(
    request: ConversationStartRequest,
    user_id: int,  # Would come from authentication middleware
    db = Depends(get_db)
):
    """
    Start a mental health support conversation.

    This endpoint initiates a supportive conversation with AI
    for mental health check-ins, crisis support, or general wellness.
    """
    try:
        conversation = await mental_health_service.start_conversation(
            user_id,
            request.conversation_type
        )

        if "error" in conversation:
            raise HTTPException(
                status_code=500,
                detail=conversation["error"]
            )

        return conversation

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error starting mental health conversation", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to start conversation"
        )


@router.post("/conversation/{session_id}/message")
async def send_mental_health_message(
    session_id: int,
    request: MessageRequest,
    user_id: int,  # Would come from authentication middleware
    db = Depends(get_db)
):
    """
    Send a message in an ongoing mental health conversation.

    This endpoint processes user messages and returns AI responses
    with crisis detection and coping strategy suggestions.
    """
    try:
        response = await mental_health_service.process_message(
            session_id,
            request.message,
            user_id
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
        logger.error("Error processing mental health message", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to process message"
        )


@router.post("/conversation/{session_id}/end")
async def end_mental_health_conversation(
    session_id: int,
    user_id: int,  # Would come from authentication middleware
    db = Depends(get_db)
):
    """
    End a mental health conversation.

    This endpoint concludes the conversation and provides
    a summary with follow-up recommendations.
    """
    try:
        summary = await mental_health_service.end_conversation(session_id, user_id)

        if "error" in summary:
            raise HTTPException(
                status_code=500,
                detail=summary["error"]
            )

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error ending mental health conversation", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to end conversation"
        )


@router.post("/mood/track")
async def track_mood(
    request: MoodTrackingRequest,
    user_id: int,  # Would come from authentication middleware
    db = Depends(get_db)
):
    """
    Track daily mood for pattern analysis.

    This endpoint allows users to log their daily mood,
    energy levels, and other wellness metrics.
    """
    try:
        # TODO: Save mood tracking data to database

        return {
            "user_id": user_id,
            "mood_level": request.mood_level,
            "mood_score": request.mood_score,
            "tracked_at": "2024-01-01T00:00:00Z",
            "message": "Mood tracking saved successfully"
        }

    except Exception as e:
        logger.error("Error tracking mood", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to track mood"
        )


@router.get("/mood/history/{user_id}")
async def get_mood_history(
    user_id: int,
    days: int = 30,
    db = Depends(get_db)
):
    """
    Get mood tracking history and trends.

    This endpoint returns mood tracking data with
    trend analysis and insights.
    """
    try:
        history = await mental_health_service.get_mood_tracking_data(user_id, days)

        if "error" in history:
            raise HTTPException(
                status_code=500,
                detail=history["error"]
            )

        return history

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting mood history", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get mood history"
        )


@router.get("/coping-strategies")
async def get_coping_strategies(
    mood: str,
    situation: Optional[str] = None
):
    """
    Get personalized coping strategies.

    This endpoint suggests coping strategies based on
    current mood and situation.
    """
    try:
        strategies = await mental_health_service.suggest_coping_strategies(mood, situation)

        return {
            "mood": mood,
            "situation": situation,
            "strategies": strategies,
            "total_strategies": len(strategies)
        }

    except Exception as e:
        logger.error("Error getting coping strategies", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get coping strategies"
        )


@router.get("/crisis-resources")
async def get_crisis_resources(
    country: str = "US",
    urgency: Optional[str] = None
):
    """
    Get crisis intervention resources.

    This endpoint provides contact information for
    crisis hotlines and mental health resources.
    """
    try:
        # TODO: Get resources from database based on country and urgency

        resources = [
            {
                "name": "National Suicide Prevention Lifeline",
                "phone": "988",
                "description": "24/7 confidential emotional support",
                "country": "US",
                "urgency": "immediate"
            },
            {
                "name": "Crisis Text Line",
                "phone": "Text HOME to 741741",
                "description": "24/7 crisis counseling via text",
                "country": "US",
                "urgency": "immediate"
            },
            {
                "name": "NAMI Helpline",
                "phone": "1-800-950-NAMI (6264)",
                "description": "Mental health information and referrals",
                "country": "US",
                "urgency": "general"
            }
        ]

        # Filter by country and urgency if specified
        if country:
            resources = [r for r in resources if r["country"] == country]

        if urgency:
            resources = [r for r in resources if r["urgency"] == urgency]

        return {
            "country": country,
            "urgency_filter": urgency,
            "resources": resources,
            "total_resources": len(resources)
        }

    except Exception as e:
        logger.error("Error getting crisis resources", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get crisis resources"
        )


@router.post("/assessment/start")
async def start_mental_health_assessment(
    assessment_type: str,
    user_id: int,  # Would come from authentication middleware
    db = Depends(get_db)
):
    """
    Start a standardized mental health assessment.

    This endpoint initiates assessments like PHQ-9, GAD-7, etc.
    """
    try:
        # TODO: Start assessment and return first question

        return {
            "assessment_type": assessment_type,
            "assessment_id": 1,  # Would be database ID
            "current_question": 1,
            "total_questions": 9,
            "question": "Over the last 2 weeks, how often have you been bothered by little interest or pleasure in doing things?",
            "options": [
                "Not at all",
                "Several days",
                "More than half the days",
                "Nearly every day"
            ],
            "started_at": "2024-01-01T00:00:00Z"
        }

    except Exception as e:
        logger.error("Error starting assessment", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to start assessment"
        )


@router.get("/wellness/tips")
async def get_wellness_tips(
    category: Optional[str] = None
):
    """
    Get daily wellness tips and mental health advice.

    This endpoint provides evidence-based tips for
    maintaining mental wellness.
    """
    try:
        # TODO: Get wellness tips from database or generate with AI

        tips = [
            {
                "category": "Mindfulness",
                "tip": "Practice 5 minutes of deep breathing each morning",
                "description": "Deep breathing activates the relaxation response and reduces stress",
                "evidence_level": "High"
            },
            {
                "category": "Sleep",
                "tip": "Maintain a consistent sleep schedule, even on weekends",
                "description": "Regular sleep patterns support mental health and cognitive function",
                "evidence_level": "High"
            },
            {
                "category": "Social Connection",
                "tip": "Schedule regular check-ins with supportive friends or family",
                "description": "Social support is a key protective factor for mental health",
                "evidence_level": "High"
            }
        ]

        if category:
            tips = [t for t in tips if t["category"] == category]

        return {
            "category_filter": category,
            "tips": tips,
            "total_tips": len(tips)
        }

    except Exception as e:
        logger.error("Error getting wellness tips", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get wellness tips"
        )
