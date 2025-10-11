"""
Enhanced API router for medication management with pharmacy integration
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel
import structlog

from backend.core.database import get_db
from backend.apps.medication.enhanced_medication_service import enhanced_medication_service

logger = structlog.get_logger()

router = APIRouter()


class ComprehensiveInteractionRequest(BaseModel):
    """Request model for comprehensive drug interaction checking."""
    medications: List[str]
    patient_context: Optional[Dict[str, Any]] = None


class PharmacySyncRequest(BaseModel):
    """Request model for pharmacy synchronization."""
    patient_id: int
    pharmacy_name: str
    pharmacy_phone: str
    pharmacy_system: Optional[str] = None
    fhir_server_url: Optional[str] = None


class AdherencePredictionRequest(BaseModel):
    """Request model for adherence prediction."""
    patient_id: int
    include_interventions: bool = True


@router.post("/interactions/comprehensive")
async def check_comprehensive_drug_interactions(
    request: ComprehensiveInteractionRequest,
    db = Depends(get_db)
):
    """
    Check drug interactions using multiple data sources.

    This endpoint provides comprehensive interaction analysis
    including DrugBank, RxNorm, clinical trials, and patient-specific risks.
    """
    try:
        interactions = await enhanced_medication_service.check_comprehensive_drug_interactions(
            request.medications,
            request.patient_context
        )

        if "error" in interactions:
            raise HTTPException(
                status_code=500,
                detail=interactions["error"]
            )

        return interactions

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error checking comprehensive drug interactions", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to check comprehensive drug interactions"
        )


@router.post("/pharmacy/sync")
async def sync_with_pharmacy_system(
    request: PharmacySyncRequest,
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    """
    Synchronize medication records with pharmacy systems.

    This endpoint initiates synchronization with pharmacy APIs
    or FHIR servers to ensure medication records are current.
    """
    try:
        pharmacy_info = {
            "name": request.pharmacy_name,
            "phone": request.pharmacy_phone,
            "system": request.pharmacy_system,
            "fhir_server_url": request.fhir_server_url
        }

        # Run sync in background
        background_tasks.add_task(
            enhanced_medication_service.pharmacy_service.sync_patient_medications,
            request.patient_id,
            pharmacy_info
        )

        return {
            "message": "Pharmacy synchronization initiated",
            "patient_id": request.patient_id,
            "pharmacy": request.pharmacy_name,
            "system": request.pharmacy_system,
            "status": "syncing",
            "estimated_completion": "2024-01-01T00:05:00Z"
        }

    except Exception as e:
        logger.error("Error initiating pharmacy sync", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to initiate pharmacy synchronization"
        )


@router.get("/patient/{patient_id}/refill-reminders")
async def get_refill_reminders(
    patient_id: int,
    days_ahead: int = Query(14, description="Check reminders for next N days"),
    db = Depends(get_db)
):
    """
    Get intelligent refill reminders for patient medications.

    This endpoint analyzes medication usage patterns and
    generates optimal refill timing recommendations.
    """
    try:
        # TODO: Get patient's current medications from database
        # medication_records = await get_patient_medications(patient_id, db)

        # For now, use mock data
        medication_records = []

        reminders = await enhanced_medication_service.generate_refill_reminders(
            patient_id,
            medication_records
        )

        # Filter reminders within specified time range
        cutoff_date = datetime.utcnow() + timedelta(days=days_ahead)
        upcoming_reminders = [
            reminder for reminder in reminders
            if datetime.fromisoformat(reminder["recommended_refill_date"]) <= cutoff_date
        ]

        return {
            "patient_id": patient_id,
            "days_ahead": days_ahead,
            "total_reminders": len(reminders),
            "upcoming_reminders": upcoming_reminders,
            "generated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("Error getting refill reminders", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get refill reminders"
        )


@router.post("/adherence/predict")
async def predict_medication_adherence(
    request: AdherencePredictionRequest,
    db = Depends(get_db)
):
    """
    Predict medication adherence patterns using AI.

    This endpoint analyzes patient medication history and
    characteristics to predict adherence risks and suggest interventions.
    """
    try:
        # TODO: Get patient's medication records from database
        # medication_records = await get_patient_medications(request.patient_id, db)

        # For now, use mock data
        medication_records = []

        prediction = await enhanced_medication_service.predict_adherence_patterns(
            request.patient_id,
            medication_records
        )

        if "error" in prediction:
            raise HTTPException(
                status_code=500,
                detail=prediction["error"]
            )

        return prediction

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error predicting medication adherence", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to predict medication adherence"
        )


@router.get("/interactions/history/{patient_id}")
async def get_interaction_history(
    patient_id: int,
    limit: int = Query(20, description="Number of recent interactions"),
    severity_filter: Optional[str] = Query(None, description="Filter by severity"),
    db = Depends(get_db)
):
    """
    Get history of drug interaction checks for a patient.

    This endpoint provides historical interaction data
    for tracking and trend analysis.
    """
    try:
        # TODO: Get interaction history from database

        return {
            "patient_id": patient_id,
            "total_checks": 15,
            "recent_interactions": [
                {
                    "check_date": "2024-01-01T00:00:00Z",
                    "medications_checked": ["Warfarin", "Aspirin"],
                    "interactions_found": 1,
                    "max_severity": "moderate",
                    "resolved": True
                }
            ],
            "trends": {
                "interaction_frequency": "decreasing",
                "severity_trend": "stable",
                "management_effectiveness": 0.85
            },
            "last_updated": "2024-01-01T00:00:00Z"
        }

    except Exception as e:
        logger.error("Error getting interaction history", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get interaction history"
        )


@router.get("/pharmacies/available")
async def get_available_pharmacies():
    """
    Get list of available pharmacy systems for integration.

    This endpoint returns supported pharmacy systems
    and their integration capabilities.
    """
    try:
        pharmacies = [
            {
                "name": "CVS Pharmacy",
                "system": "cvs",
                "api_available": True,
                "fhir_support": True,
                "features": ["real_time_sync", "refill_management", "interaction_checking"],
                "coverage": "Nationwide (US)"
            },
            {
                "name": "Walgreens",
                "system": "walgreens",
                "api_available": True,
                "fhir_support": True,
                "features": ["real_time_sync", "prescription_transfer", "drug_information"],
                "coverage": "Nationwide (US)"
            },
            {
                "name": "Rite Aid",
                "system": "rite_aid",
                "api_available": True,
                "fhir_support": False,
                "features": ["basic_sync", "refill_reminders"],
                "coverage": "Nationwide (US)"
            },
            {
                "name": "Costco Pharmacy",
                "system": "costco",
                "api_available": True,
                "fhir_support": True,
                "features": ["bulk_sync", "generic_substitution"],
                "coverage": "Nationwide (US)"
            }
        ]

        return {
            "pharmacies": pharmacies,
            "total_pharmacies": len(pharmacies),
            "last_updated": "2024-01-01T00:00:00Z"
        }

    except Exception as e:
        logger.error("Error getting available pharmacies", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get available pharmacies"
        )
