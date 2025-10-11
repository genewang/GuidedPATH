"""
API router for personalization engine
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import structlog

from backend.core.database import get_db
from backend.apps.personalization.personalization_service import personalization_engine

logger = structlog.get_logger()

router = APIRouter()


class UserProfileRequest(BaseModel):
    """Request model for creating user profile."""
    initial_preferences: Optional[Dict[str, Any]] = None
    demographic_info: Optional[Dict[str, Any]] = None


class ContentPersonalizationRequest(BaseModel):
    """Request model for content personalization."""
    content: Dict[str, Any]
    content_type: str
    context: Optional[Dict[str, Any]] = None


class PreferenceUpdateRequest(BaseModel):
    """Request model for updating user preferences."""
    interaction_data: Dict[str, Any]
    feedback_score: Optional[float] = None


class RecommendationRequest(BaseModel):
    """Request model for personalized recommendations."""
    available_content: List[Dict[str, Any]]
    recommendation_type: str = "comprehensive"


class CommunicationStyleRequest(BaseModel):
    """Request model for communication style adaptation."""
    message_content: str
    target_audience: str = "patient"


@router.post("/profile/create")
async def create_user_profile(
    request: UserProfileRequest,
    user_id: int,  # Would come from authentication middleware
    db = Depends(get_db)
):
    """
    Create user personalization profile.

    This endpoint initializes a comprehensive user profile
    for adaptive personalization across all features.
    """
    try:
        profile = await personalization_engine.create_user_profile(
            user_id=user_id,
            initial_preferences=request.initial_preferences,
            demographic_info=request.demographic_info
        )

        if "error" in profile:
            raise HTTPException(
                status_code=500,
                detail=profile["error"]
            )

        return profile

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating user profile", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to create user profile"
        )


@router.post("/content/personalize")
async def personalize_content(
    request: ContentPersonalizationRequest,
    user_id: int,  # Would come from authentication middleware
    db = Depends(get_db)
):
    """
    Personalize content based on user preferences.

    This endpoint adapts content presentation, complexity,
    and communication style to individual user needs.
    """
    try:
        personalized_content = await personalization_engine.personalize_content(
            user_id=user_id,
            content=request.content,
            content_type=request.content_type,
            context=request.context
        )

        return personalized_content

    except Exception as e:
        logger.error("Error personalizing content", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to personalize content"
        )


@router.post("/preferences/update")
async def update_user_preferences(
    request: PreferenceUpdateRequest,
    user_id: int,  # Would come from authentication middleware
    db = Depends(get_db)
):
    """
    Update user preferences based on interaction patterns.

    This endpoint learns from user behavior and feedback
    to continuously improve personalization.
    """
    try:
        update_result = await personalization_engine.update_user_preferences(
            user_id=user_id,
            interaction_data=request.interaction_data,
            feedback_score=request.feedback_score
        )

        if "error" in update_result:
            raise HTTPException(
                status_code=500,
                detail=update_result["error"]
            )

        return update_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating user preferences", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to update user preferences"
        )


@router.post("/recommendations/generate")
async def generate_personalized_recommendations(
    request: RecommendationRequest,
    user_id: int,  # Would come from authentication middleware
    db = Depends(get_db)
):
    """
    Generate personalized content recommendations.

    This endpoint provides AI-powered recommendations
    tailored to user preferences and behavior patterns.
    """
    try:
        recommendations = await personalization_engine.generate_personalized_recommendations(
            user_id=user_id,
            available_content=request.available_content,
            recommendation_type=request.recommendation_type
        )

        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "generated_at": "2024-01-01T00:00:00Z",
            "next_refresh": "2024-01-02T00:00:00Z"
        }

    except Exception as e:
        logger.error("Error generating recommendations", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to generate personalized recommendations"
        )


@router.post("/communication/adapt")
async def adapt_communication_style(
    request: CommunicationStyleRequest,
    user_id: int,  # Would come from authentication middleware
    db = Depends(get_db)
):
    """
    Adapt communication style for personalized messaging.

    This endpoint modifies message tone, formality, and length
    based on individual user preferences.
    """
    try:
        adapted_message = await personalization_engine.adapt_communication_style(
            user_id=user_id,
            message_content=request.message_content,
            target_audience=request.target_audience
        )

        return adapted_message

    except Exception as e:
        logger.error("Error adapting communication style", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to adapt communication style"
        )


@router.get("/profile/{user_id}")
async def get_user_profile(
    user_id: int,
    db = Depends(get_db)
):
    """
    Get user personalization profile.

    This endpoint returns the current user profile
    with preferences and personalization settings.
    """
    try:
        profile = personalization_engine.user_profiles.get(user_id)

        if not profile:
            raise HTTPException(
                status_code=404,
                detail="User profile not found"
            )

        # Remove sensitive internal data
        safe_profile = profile.copy()
        if "internal_models" in safe_profile:
            del safe_profile["internal_models"]

        return safe_profile

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting user profile", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get user profile"
        )


@router.get("/preferences/{user_id}")
async def get_user_preferences(
    user_id: int,
    db = Depends(get_db)
):
    """
    Get current user preferences.

    This endpoint returns the latest user preference
    settings for personalization.
    """
    try:
        preferences = personalization_engine.user_preferences.get(user_id)

        if not preferences:
            raise HTTPException(
                status_code=404,
                detail="User preferences not found"
            )

        return {
            "user_id": user_id,
            "preferences": preferences,
            "last_updated": "2024-01-01T00:00:00Z"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting user preferences", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get user preferences"
        )


@router.get("/insights/{user_id}")
async def get_personalization_insights(
    user_id: int,
    time_range_days: int = 30,
    db = Depends(get_db)
):
    """
    Get personalization insights and behavior patterns.

    This endpoint provides insights into how personalization
    is working and user behavior patterns.
    """
    try:
        # TODO: Get personalization insights from behavior history

        behavior_history = personalization_engine.user_behavior_history.get(user_id, [])

        # Analyze recent behavior
        recent_events = [event for event in behavior_history if event.get("timestamp", datetime.utcnow()) > datetime.utcnow() - timedelta(days=time_range_days)]

        insights = {
            "user_id": user_id,
            "time_range_days": time_range_days,
            "total_interactions": len(recent_events),
            "average_relevance_score": sum(event.get("relevance_score", 0.5) for event in recent_events) / len(recent_events) if recent_events else 0.5,
            "content_type_preferences": await personalization_engine._analyze_content_type_preferences(recent_events),
            "engagement_patterns": await personalization_engine._analyze_engagement_patterns(recent_events),
            "personalization_effectiveness": {
                "improvement_over_time": 0.15,  # Would calculate from historical data
                "user_satisfaction_score": 0.85,  # Would come from feedback
                "adaptation_accuracy": 0.78  # Would measure prediction accuracy
            },
            "generated_at": "2024-01-01T00:00:00Z"
        }

        return insights

    except Exception as e:
        logger.error("Error getting personalization insights", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get personalization insights"
        )
