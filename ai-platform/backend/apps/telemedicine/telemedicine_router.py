"""
API router for telemedicine integration service
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import structlog

from backend.core.database import get_db
from backend.apps.telemedicine.telemedicine_service import telemedicine_service

logger = structlog.get_logger()

router = APIRouter()


class ConsultationRequest(BaseModel):
    """Request model for telemedicine consultation."""
    specialty: str
    urgency: str = "routine"
    symptoms: List[str]
    preferred_time: Optional[str] = None
    insurance_info: Optional[Dict[str, Any]] = None


class AppointmentSearchRequest(BaseModel):
    """Request model for searching appointments."""
    specialty: str
    urgency: str = "routine"
    max_results: int = 10
    preferred_platform: Optional[str] = None


class ConsultationOutcomeRequest(BaseModel):
    """Request model for consultation outcome tracking."""
    consultation_id: str
    outcome_type: str
    patient_satisfaction: int
    clinical_effectiveness: str
    follow_up_needed: bool


@router.post("/consultation/schedule")
async def schedule_telemedicine_consultation(
    request: ConsultationRequest,
    patient_id: int,  # Would come from authentication middleware
    preferred_platform: Optional[str] = None,
    db = Depends(get_db)
):
    """
    Schedule telemedicine consultation.

    This endpoint schedules virtual consultations across
    multiple telemedicine platforms with intelligent routing.
    """
    try:
        consultation_request = {
            "specialty": request.specialty,
            "urgency": request.urgency,
            "symptoms": request.symptoms,
            "preferred_time": request.preferred_time,
            "insurance_info": request.insurance_info
        }

        result = await telemedicine_service.schedule_cross_platform_consultation(
            patient_id=patient_id,
            consultation_request=consultation_request,
            platform_preferences=[preferred_platform] if preferred_platform else None
        )

        if "error" in result:
            raise HTTPException(
                status_code=400,
                detail=result["error"]
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error scheduling telemedicine consultation", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to schedule telemedicine consultation"
        )


@router.get("/appointments/available")
async def get_available_appointments(
    request: AppointmentSearchRequest,
    db = Depends(get_db)
):
    """
    Get available telemedicine appointments.

    This endpoint searches across all integrated platforms
    for available appointment slots matching criteria.
    """
    try:
        appointments = await telemedicine_service.platform_integration.get_available_appointments(
            specialty=request.specialty,
            urgency=request.urgency,
            max_results=request.max_results,
            preferred_platform=request.preferred_platform
        )

        return {
            "search_criteria": {
                "specialty": request.specialty,
                "urgency": request.urgency,
                "preferred_platform": request.preferred_platform
            },
            "available_appointments": appointments,
            "total_results": len(appointments),
            "search_timestamp": "2024-01-01T00:00:00Z"
        }

    except Exception as e:
        logger.error("Error getting available appointments", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get available appointments"
        )


@router.get("/options/{patient_id}")
async def get_telemedicine_options(
    patient_id: int,
    medical_need: str,
    db = Depends(get_db)
):
    """
    Get comprehensive telemedicine options for patient needs.

    This endpoint provides platform recommendations and
    availability across all integrated services.
    """
    try:
        # Parse medical need (in production, use NLP to extract from text)
        medical_need_parsed = {
            "specialty": medical_need.split()[0] if medical_need else "general",
            "urgency": "routine"
        }

        options = await telemedicine_service.find_telemedicine_options(
            patient_id=patient_id,
            medical_need=medical_need_parsed
        )

        if "error" in options:
            raise HTTPException(
                status_code=500,
                detail=options["error"]
            )

        return options

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting telemedicine options", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get telemedicine options"
        )


@router.post("/consultation/{consultation_id}/integrate")
async def integrate_consultation_results(
    consultation_id: str,
    results: Dict[str, Any],
    platform: str,
    db = Depends(get_db)
):
    """
    Integrate telemedicine consultation results.

    This endpoint processes consultation outcomes and
    integrates them into the patient's healthcare record.
    """
    try:
        integration_result = await telemedicine_service.platform_integration.integrate_consultation_results(
            consultation_id=consultation_id,
            platform=platform,
            results=results
        )

        if "error" in integration_result:
            raise HTTPException(
                status_code=500,
                detail=integration_result["error"]
            )

        return integration_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error integrating consultation results", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to integrate consultation results"
        )


@router.post("/consultation/track-outcome")
async def track_consultation_outcome(
    request: ConsultationOutcomeRequest,
    db = Depends(get_db)
):
    """
    Track telemedicine consultation outcomes.

    This endpoint records consultation outcomes for
    analytics and platform performance tracking.
    """
    try:
        outcome_data = {
            "outcome_type": request.outcome_type,
            "patient_satisfaction": request.patient_satisfaction,
            "clinical_effectiveness": request.clinical_effectiveness,
            "follow_up_needed": request.follow_up_needed
        }

        tracking_result = await telemedicine_service.track_consultation_outcomes(
            consultation_id=request.consultation_id,
            outcome_data=outcome_data
        )

        if "error" in tracking_result:
            raise HTTPException(
                status_code=500,
                detail=tracking_result["error"]
            )

        return tracking_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error tracking consultation outcome", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to track consultation outcome"
        )


@router.get("/platforms/available")
async def get_available_platforms():
    """
    Get available telemedicine platforms.

    This endpoint returns all integrated telemedicine
    platforms with their capabilities and features.
    """
    try:
        platforms = [
            {
                "id": "teladoc",
                "name": "Teladoc",
                "description": "Comprehensive telemedicine platform",
                "supported_services": ["urgent_care", "therapy", "dermatology"],
                "features": ["video_consultations", "prescription_management", "follow_up_care"],
                "average_rating": 4.2,
                "response_time": "15-30 minutes",
                "coverage_areas": ["US", "Canada", "UK"],
                "insurance_integration": True,
                "cost_range": "$50-$100"
            },
            {
                "id": "amwell",
                "name": "Amwell",
                "description": "Leading telehealth platform",
                "supported_services": ["primary_care", "mental_health", "specialty"],
                "features": ["mobile_app", "emergency_consultations", "care_coordination"],
                "average_rating": 4.3,
                "response_time": "10-20 minutes",
                "coverage_areas": ["US"],
                "insurance_integration": True,
                "cost_range": "$40-$90"
            },
            {
                "id": "doctor_on_demand",
                "name": "Doctor on Demand",
                "description": "On-demand virtual healthcare",
                "supported_services": ["urgent_care", "preventive_care", "chronic_care"],
                "features": ["same_day_appointments", "prescription_delivery", "health_records"],
                "average_rating": 4.1,
                "response_time": "5-15 minutes",
                "coverage_areas": ["US"],
                "insurance_integration": True,
                "cost_range": "$75-$150"
            }
        ]

        return {
            "platforms": platforms,
            "total_platforms": len(platforms),
            "last_updated": "2024-01-01T00:00:00Z"
        }

    except Exception as e:
        logger.error("Error getting available platforms", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get available platforms"
        )


@router.get("/analytics/platform-performance")
async def get_platform_performance_analytics(
    time_range_days: int = 30,
    db = Depends(get_db)
):
    """
    Get telemedicine platform performance analytics.

    This endpoint provides insights into platform
    performance, user satisfaction, and service quality.
    """
    try:
        # TODO: Get platform performance data from database

        return {
            "time_range_days": time_range_days,
            "platform_performance": [
                {
                    "platform": "Teladoc",
                    "total_consultations": 1250,
                    "average_satisfaction": 4.2,
                    "average_wait_time": 18,  # minutes
                    "success_rate": 0.94,
                    "specialty_breakdown": {
                        "urgent_care": 0.4,
                        "therapy": 0.3,
                        "dermatology": 0.3
                    }
                },
                {
                    "platform": "Amwell",
                    "total_consultations": 980,
                    "average_satisfaction": 4.3,
                    "average_wait_time": 12,
                    "success_rate": 0.96,
                    "specialty_breakdown": {
                        "primary_care": 0.5,
                        "mental_health": 0.3,
                        "specialty": 0.2
                    }
                }
            ],
            "overall_metrics": {
                "total_consultations": 2230,
                "average_satisfaction": 4.25,
                "average_wait_time": 15,
                "success_rate": 0.95
            },
            "trends": [
                "Increasing demand for mental health consultations",
                "Improving wait times across platforms",
                "High satisfaction with urgent care services"
            ],
            "generated_at": "2024-01-01T00:00:00Z"
        }

    except Exception as e:
        logger.error("Error getting platform performance analytics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get platform performance analytics"
        )
