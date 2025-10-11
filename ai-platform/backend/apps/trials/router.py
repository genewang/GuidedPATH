"""
API router for clinical trials service
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
import structlog

from backend.core.database import get_db
from backend.apps.trials.matching_service import trial_matching_service

logger = structlog.get_logger()

router = APIRouter()


class TrialSearchRequest(BaseModel):
    """Request model for trial search."""
    search_query: Optional[str] = None
    conditions: Optional[List[str]] = None
    max_results: int = 10
    minimum_score: float = 0.3


class TrialMatchResponse(BaseModel):
    """Response model for trial matches."""
    trial_id: int
    nct_id: str
    title: str
    phase: str
    status: str
    overall_score: float
    condition_similarity: float
    eligibility_score: float
    location_score: float
    phase_score: float
    explanation: str
    unmet_criteria: List[str]
    locations: List[Dict[str, Any]]
    contact_info: Dict[str, str]


@router.post("/match", response_model=List[TrialMatchResponse])
async def find_matching_trials(
    request: TrialSearchRequest,
    user_id: int,  # This would come from authentication middleware
    db = Depends(get_db)
):
    """
    Find clinical trials that match a patient's profile.

    This endpoint uses AI to match patients with appropriate clinical trials
    based on their medical history, current condition, and eligibility criteria.
    """
    try:
        # TODO: Get patient record from user_id
        # patient_record = await get_patient_record(user_id, db)

        # For now, use mock patient data
        patient_record = None

        if not patient_record:
            raise HTTPException(
                status_code=404,
                detail="Patient record not found"
            )

        # Find matching trials
        matches = await trial_matching_service.find_matching_trials(
            patient_record=patient_record,
            search_query=request.search_query,
            max_results=request.max_results,
            minimum_score=request.minimum_score
        )

        # Format response
        response = []
        for match in matches:
            trial = match["trial"]
            response.append(TrialMatchResponse(
                trial_id=trial.id,
                nct_id=trial.nct_id,
                title=trial.title,
                phase=trial.phase.value,
                status=trial.status.value,
                overall_score=match["overall_score"],
                condition_similarity=match["condition_similarity"],
                eligibility_score=match["eligibility_score"],
                location_score=match["location_score"],
                phase_score=match["phase_score"],
                explanation=match["explanation"],
                unmet_criteria=match["unmet_criteria"],
                locations=trial.locations,
                contact_info={
                    "name": trial.contact_name or "Not available",
                    "phone": trial.contact_phone or "Not available",
                    "email": trial.contact_email or "Not available"
                }
            ))

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error finding matching trials", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to find matching trials"
        )


@router.get("/search")
async def search_trials(
    q: str = Query(..., description="Search query"),
    condition: Optional[str] = Query(None, description="Filter by condition"),
    phase: Optional[str] = Query(None, description="Filter by phase"),
    status: str = Query("RECRUITING", description="Trial status filter"),
    location: Optional[str] = Query(None, description="Location filter"),
    max_results: int = Query(20, description="Maximum results"),
    db = Depends(get_db)
):
    """
    Search clinical trials with filters.

    This endpoint provides a search interface for finding clinical trials
    with various filtering options.
    """
    try:
        # TODO: Implement trial search with filters
        # For now, return mock results

        return {
            "query": q,
            "filters": {
                "condition": condition,
                "phase": phase,
                "status": status,
                "location": location
            },
            "results": [
                {
                    "id": 1,
                    "nct_id": "NCT12345678",
                    "title": "Sample Clinical Trial",
                    "phase": "PHASE_2",
                    "status": "RECRUITING",
                    "conditions": ["Cancer"],
                    "locations": ["New York, NY"],
                    "brief_summary": "This is a sample trial description..."
                }
            ],
            "total": 1
        }

    except Exception as e:
        logger.error("Error searching trials", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to search trials"
        )


@router.get("/{trial_id}")
async def get_trial_details(
    trial_id: int,
    db = Depends(get_db)
):
    """
    Get detailed information about a specific clinical trial.
    """
    try:
        # TODO: Get trial from database
        # trial = await get_trial_by_id(trial_id, db)

        # For now, return mock data
        return {
            "id": trial_id,
            "nct_id": "NCT12345678",
            "title": "Sample Clinical Trial",
            "phase": "PHASE_2",
            "status": "RECRUITING",
            "conditions": ["Cancer"],
            "interventions": ["Drug Therapy"],
            "eligibility_criteria": "Sample eligibility criteria...",
            "locations": [
                {
                    "facility": "Sample Hospital",
                    "city": "New York",
                    "state": "NY",
                    "country": "USA",
                    "contact": "trial@hospital.com"
                }
            ],
            "contact_info": {
                "name": "Dr. Smith",
                "phone": "555-0123",
                "email": "trial@hospital.com"
            },
            "study_start_date": "2024-01-01",
            "primary_completion_date": "2025-01-01",
            "enrollment_target": 100
        }

    except Exception as e:
        logger.error("Error getting trial details", error=str(e), trial_id=trial_id)
        raise HTTPException(
            status_code=404,
            detail="Trial not found"
        )


@router.post("/{trial_id}/interest")
async def express_trial_interest(
    trial_id: int,
    user_id: int,  # This would come from authentication middleware
    db = Depends(get_db)
):
    """
    Express interest in a clinical trial.

    This endpoint allows patients to indicate interest in a trial,
    which can trigger notifications to trial coordinators.
    """
    try:
        # TODO: Record user interest in database
        # TODO: Send notification to trial coordinators

        return {
            "message": "Interest recorded successfully",
            "trial_id": trial_id,
            "next_steps": "You will be contacted by the trial coordinator within 2-3 business days"
        }

    except Exception as e:
        logger.error("Error recording trial interest", error=str(e), trial_id=trial_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to record trial interest"
        )
