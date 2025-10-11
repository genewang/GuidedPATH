"""
API router for symptom checker and triage service
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import structlog

from backend.core.database import get_db
from backend.apps.symptoms.symptom_service import symptom_checker_service

logger = structlog.get_logger()

router = APIRouter()


class SymptomAnalysisRequest(BaseModel):
    """Request model for symptom analysis."""
    symptoms: List[str]
    patient_context: Optional[Dict[str, Any]] = None


class SymptomReportRequest(BaseModel):
    """Request model for reporting symptoms."""
    primary_symptom: str
    symptom_description: Optional[str] = None
    severity: str  # mild, moderate, severe, critical
    associated_symptoms: Optional[List[str]] = None
    duration: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class TriageResponse(BaseModel):
    """Response model for triage recommendations."""
    urgency_level: str
    recommended_action: str
    time_frame: Optional[str]
    reasoning: Optional[str]
    self_care_advice: List[str]
    when_to_seek_care: List[str]
    emergency_indicators: List[str]
    confidence_score: float
    requires_clinician_review: bool


@router.post("/analyze", response_model=TriageResponse)
async def analyze_symptoms(
    request: SymptomAnalysisRequest,
    db = Depends(get_db)
):
    """
    Analyze symptoms and provide triage recommendations.

    This endpoint uses AI to analyze symptom descriptions and provide
    evidence-based triage recommendations with urgency levels.
    """
    try:
        analysis = await symptom_checker_service.analyze_symptoms(
            request.symptoms,
            request.patient_context
        )

        if "error" in analysis:
            raise HTTPException(
                status_code=500,
                detail=analysis["error"]
            )

        urgency_assessment = analysis["urgency_assessment"]
        triage_recommendations = analysis["triage_recommendations"]

        return TriageResponse(
            urgency_level=urgency_assessment["urgency"],
            recommended_action=triage_recommendations["recommended_action"],
            time_frame=triage_recommendations["time_frame"],
            reasoning=urgency_assessment["reasoning"],
            self_care_advice=triage_recommendations["self_care_advice"],
            when_to_seek_care=triage_recommendations["when_to_seek_care"],
            emergency_indicators=triage_recommendations["emergency_indicators"],
            confidence_score=analysis["confidence_score"],
            requires_clinician_review=analysis["requires_immediate_attention"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error analyzing symptoms", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze symptoms"
        )


@router.post("/report")
async def report_symptoms(
    request: SymptomReportRequest,
    user_id: int,  # Would come from authentication middleware
    db = Depends(get_db)
):
    """
    Report symptoms for tracking and analysis.

    This endpoint allows patients to report symptoms for
    ongoing tracking and pattern analysis.
    """
    try:
        # TODO: Create symptom report in database
        # TODO: Generate triage result
        # TODO: Store for pattern analysis

        return {
            "message": "Symptoms reported successfully",
            "report_id": 1,  # Would be actual database ID
            "triage_urgency": "routine",
            "next_steps": "Monitor symptoms and follow triage recommendations",
            "reported_at": "2024-01-01T00:00:00Z"
        }

    except Exception as e:
        logger.error("Error reporting symptoms", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to report symptoms"
        )


@router.get("/patient/{patient_id}/history")
async def get_symptom_history(
    patient_id: int,
    limit: int = 20,
    db = Depends(get_db)
):
    """
    Get symptom history for a patient.

    This endpoint returns historical symptom reports
    for pattern analysis and tracking.
    """
    try:
        # TODO: Get symptom history from database

        return {
            "patient_id": patient_id,
            "symptom_history": [
                {
                    "id": 1,
                    "primary_symptom": "Headache",
                    "severity": "moderate",
                    "reported_at": "2024-01-01T00:00:00Z",
                    "triage_urgency": "routine",
                    "resolved": True
                }
            ],
            "total_reports": 1,
            "patterns_identified": []
        }

    except Exception as e:
        logger.error("Error getting symptom history", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get symptom history"
        )


@router.get("/patient/{patient_id}/patterns")
async def get_symptom_patterns(
    patient_id: int,
    db = Depends(get_db)
):
    """
    Get identified symptom patterns for a patient.

    This endpoint analyzes symptom history to identify
    patterns and trends that may be clinically significant.
    """
    try:
        # TODO: Get symptom reports and analyze patterns

        patterns = await symptom_checker_service.track_symptom_patterns(
            patient_id,
            []  # Would be actual symptom reports
        )

        return {
            "patient_id": patient_id,
            "patterns": patterns,
            "analyzed_at": "2024-01-01T00:00:00Z",
            "recommendations": [
                "Continue monitoring symptoms",
                "Discuss patterns with healthcare provider if concerning"
            ]
        }

    except Exception as e:
        logger.error("Error getting symptom patterns", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze symptom patterns"
        )


@router.post("/triage/{report_id}")
async def generate_triage_report(
    report_id: int,
    db = Depends(get_db)
):
    """
    Generate a comprehensive triage report for a symptom report.

    This endpoint creates a detailed triage analysis including
    urgency assessment, recommendations, and clinical reasoning.
    """
    try:
        # TODO: Get symptom report from database
        # symptom_report = await get_symptom_report(report_id, db)

        # For now, use mock data
        symptom_report = None

        if not symptom_report:
            raise HTTPException(
                status_code=404,
                detail="Symptom report not found"
            )

        triage_result = await symptom_checker_service.generate_triage_report(symptom_report)

        return {
            "report_id": report_id,
            "urgency_level": triage_result.urgency_level.value,
            "recommended_action": triage_result.recommended_action,
            "reasoning": triage_result.reasoning,
            "self_care_advice": triage_result.self_care_advice,
            "when_to_seek_care": triage_result.when_to_seek_care,
            "emergency_indicators": triage_result.emergency_indicators,
            "confidence_score": triage_result.confidence_score,
            "requires_clinician_review": triage_result.requires_clinician_review,
            "generated_at": triage_result.generated_at.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error generating triage report", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to generate triage report"
        )


@router.get("/conditions/search")
async def search_medical_conditions(
    query: str,
    symptoms: Optional[List[str]] = None,
    db = Depends(get_db)
):
    """
    Search for medical conditions based on symptoms.

    This endpoint helps identify potential conditions
    that may be associated with reported symptoms.
    """
    try:
        # TODO: Implement condition search based on symptoms

        return {
            "query": query,
            "symptoms": symptoms or [],
            "potential_conditions": [
                {
                    "condition": "Sample Condition",
                    "description": "Brief description of the condition",
                    "likelihood": "medium",
                    "associated_symptoms": ["symptom1", "symptom2"],
                    "recommended_actions": ["See a doctor", "Monitor symptoms"]
                }
            ]
        }

    except Exception as e:
        logger.error("Error searching medical conditions", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to search medical conditions"
        )
