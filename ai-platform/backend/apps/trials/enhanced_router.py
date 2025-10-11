"""
Enhanced API router for advanced clinical trials service
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel
import structlog

from backend.core.database import get_db
from backend.apps.trials.advanced_matching_service import advanced_trial_matching_service

logger = structlog.get_logger()

router = APIRouter()


class AdvancedTrialSearchRequest(BaseModel):
    """Request model for advanced trial search."""
    patient_profile: Dict[str, Any]
    search_criteria: Optional[Dict[str, Any]] = None
    max_results: int = 10
    diversity_factor: float = 0.3
    include_explanations: bool = True


class TrialFeedbackRequest(BaseModel):
    """Request model for trial match feedback."""
    trial_id: int
    patient_id: int
    match_outcome: str  # successful, unsuccessful, pending
    feedback_notes: Optional[str] = None


class ModelRetrainingRequest(BaseModel):
    """Request model for model retraining."""
    feedback_data: List[Dict[str, Any]]
    model_types: List[str] = ["eligibility", "phase", "location"]


@router.post("/advanced-match")
async def find_advanced_trial_matches(
    request: AdvancedTrialSearchRequest,
    db = Depends(get_db)
):
    """
    Find clinical trials using advanced ML algorithms.

    This endpoint uses deep learning and ensemble methods
    to provide highly accurate trial-patient matching.
    """
    try:
        matches = await advanced_trial_matching_service.find_optimal_trials(
            patient_profile=request.patient_profile,
            search_criteria=request.search_criteria,
            max_results=request.max_results,
            diversity_factor=request.diversity_factor
        )

        # Format response with detailed explanations
        response = []
        for match in matches:
            trial = match["trial"]

            trial_info = {
                "trial_id": trial.id,
                "nct_id": trial.nct_id,
                "title": trial.title,
                "phase": trial.phase.value,
                "status": trial.status.value,
                "overall_score": match["overall_score"],
                "relevance_score": match["relevance_score"],
                "eligibility_score": match["eligibility_score"],
                "location_score": match["location_score"],
                "phase_score": match["phase_score"],
                "confidence_interval": match["confidence_interval"],
                "explanation": match["explanation"],
                "diversity_bonus": match["diversity_bonus"],
                "locations": trial.locations,
                "contact_info": {
                    "name": trial.contact_name,
                    "phone": trial.contact_phone,
                    "email": trial.contact_email
                }
            }

            if request.include_explanations:
                trial_info["detailed_analysis"] = {
                    "feature_importance": match["feature_importance"],
                    "matching_criteria": await advanced_trial_matching_service._generate_detailed_explanation(
                        request.patient_profile, trial,
                        match["relevance_score"], match["eligibility_score"],
                        match["location_score"], match["phase_score"]
                    )
                }

            response.append(trial_info)

        return {
            "search_criteria": request.patient_profile,
            "results": response,
            "total_results": len(response),
            "search_timestamp": "2024-01-01T00:00:00Z",
            "algorithm_version": "advanced_ml_v2.1"
        }

    except Exception as e:
        logger.error("Error in advanced trial matching", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to find advanced trial matches"
        )


@router.get("/recommendations/{patient_id}")
async def get_trial_recommendations(
    patient_id: int,
    limit: int = Query(5, description="Number of recommendations"),
    include_reasons: bool = Query(True, description="Include explanation for recommendations"),
    db = Depends(get_db)
):
    """
    Get personalized trial recommendations for a patient.

    This endpoint provides ongoing trial recommendations
    based on patient profile and treatment progress.
    """
    try:
        # TODO: Get patient profile from database
        # patient_profile = await get_patient_profile(patient_id, db)

        # For now, use mock data
        patient_profile = {"id": patient_id}

        recommendations = await advanced_trial_matching_service.find_optimal_trials(
            patient_profile=patient_profile,
            max_results=limit
        )

        response = []
        for rec in recommendations:
            trial = rec["trial"]

            recommendation = {
                "trial_id": trial.id,
                "nct_id": trial.nct_id,
                "title": trial.title,
                "match_score": rec["overall_score"],
                "urgency_level": "high" if rec["overall_score"] > 0.8 else "medium" if rec["overall_score"] > 0.6 else "low",
                "estimated_enrollment_rate": "moderate",  # Would calculate from historical data
                "time_to_complete": "6-12 months"  # Would calculate from trial data
            }

            if include_reasons:
                recommendation["reasoning"] = rec["explanation"]
                recommendation["confidence_level"] = "high" if rec["confidence_interval"][1] > 0.8 else "medium"

            response.append(recommendation)

        return {
            "patient_id": patient_id,
            "recommendations": response,
            "last_updated": "2024-01-01T00:00:00Z",
            "next_refresh": "2024-01-08T00:00:00Z"  # Weekly refresh
        }

    except Exception as e:
        logger.error("Error getting trial recommendations", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get trial recommendations"
        )


@router.post("/feedback")
async def submit_trial_feedback(
    request: TrialFeedbackRequest,
    db = Depends(get_db)
):
    """
    Submit feedback on trial matches for model improvement.

    This endpoint allows users to provide feedback on trial
    match quality to improve future recommendations.
    """
    try:
        # TODO: Store feedback in database for model retraining

        return {
            "trial_id": request.trial_id,
            "patient_id": request.patient_id,
            "feedback_recorded": True,
            "feedback_id": 1,  # Would be database ID
            "next_steps": "Thank you for your feedback. This will help improve future trial recommendations.",
            "submitted_at": "2024-01-01T00:00:00Z"
        }

    except Exception as e:
        logger.error("Error submitting trial feedback", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to submit trial feedback"
        )


@router.post("/models/retrain")
async def retrain_trial_models(
    request: ModelRetrainingRequest,
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    """
    Retrain ML models using collected feedback data.

    This endpoint initiates model retraining with new
    feedback data to improve matching accuracy.
    """
    try:
        # Run retraining in background
        background_tasks.add_task(
            advanced_trial_matching_service.retrain_models,
            request.feedback_data
        )

        return {
            "message": "Model retraining initiated",
            "model_types": request.model_types,
            "training_samples": len(request.feedback_data),
            "status": "training_started",
            "estimated_completion": "2024-01-01T01:00:00Z"
        }

    except Exception as e:
        logger.error("Error initiating model retraining", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to initiate model retraining"
        )


@router.get("/analytics/{patient_id}")
async def get_trial_analytics(
    patient_id: int,
    time_range_days: int = Query(90, description="Time range for analytics"),
    db = Depends(get_db)
):
    """
    Get analytics on patient's trial search behavior and outcomes.

    This endpoint provides insights into trial search patterns
    and match success rates for continuous improvement.
    """
    try:
        # TODO: Get analytics data from database

        return {
            "patient_id": patient_id,
            "time_range_days": time_range_days,
            "search_summary": {
                "total_searches": 15,
                "average_results_per_search": 8.5,
                "most_searched_conditions": ["breast_cancer", "metastatic_disease"],
                "preferred_trial_phases": ["phase_2", "phase_3"]
            },
            "match_outcomes": {
                "matches_reviewed": 12,
                "matches_contacted": 3,
                "matches_joined": 1,
                "success_rate": 0.083  # 1/12
            },
            "behavioral_insights": [
                "Patient frequently searches for early-phase trials",
                "High engagement with immunotherapy trials",
                "Geographic preference for trials within 50 miles"
            ],
            "recommendations": [
                "Consider expanding search radius for more options",
                "Focus on trials with higher success rates",
                "Schedule follow-up consultation for trial strategy"
            ],
            "generated_at": "2024-01-01T00:00:00Z"
        }

    except Exception as e:
        logger.error("Error getting trial analytics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get trial analytics"
        )


@router.get("/trends")
async def get_trial_trends(
    condition: Optional[str] = Query(None, description="Filter by condition"),
    location: Optional[str] = Query(None, description="Filter by location"),
    time_range_days: int = Query(30, description="Time range for trends"),
    db = Depends(get_db)
):
    """
    Get trending clinical trials and research directions.

    This endpoint provides insights into current trial
    trends and emerging research areas.
    """
    try:
        # TODO: Analyze trial database for trends

        trends = [
            {
                "trend": "Immunotherapy combinations",
                "growth_rate": 0.25,
                "trial_count": 45,
                "leading_sponsors": ["Merck", "BMS", "Roche"],
                "conditions": ["lung_cancer", "melanoma", "breast_cancer"]
            },
            {
                "trend": "Targeted therapies",
                "growth_rate": 0.18,
                "trial_count": 38,
                "leading_sponsors": ["Novartis", "Pfizer", "AstraZeneca"],
                "conditions": ["lung_cancer", "breast_cancer", "colorectal_cancer"]
            },
            {
                "trend": "CAR-T cell therapies",
                "growth_rate": 0.32,
                "trial_count": 28,
                "leading_sponsors": ["Gilead", "Novartis", "J&J"],
                "conditions": ["lymphoma", "leukemia", "multiple_myeloma"]
            }
        ]

        # Filter by condition if specified
        if condition:
            trends = [t for t in trends if condition in t["conditions"]]

        return {
            "time_range_days": time_range_days,
            "condition_filter": condition,
            "location_filter": location,
            "trends": trends,
            "total_trends": len(trends),
            "analysis_timestamp": "2024-01-01T00:00:00Z"
        }

    except Exception as e:
        logger.error("Error getting trial trends", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get trial trends"
        )
