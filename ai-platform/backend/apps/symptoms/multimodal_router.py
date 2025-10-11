"""
Enhanced API router for multimodal symptom checker
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from pydantic import BaseModel
import structlog

from backend.core.database import get_db
from backend.apps.symptoms.multimodal_service import multimodal_symptom_service

logger = structlog.get_logger()

router = APIRouter()


class MultimodalAnalysisRequest(BaseModel):
    """Request model for multimodal symptom analysis."""
    text_input: Optional[str] = None
    patient_context: Optional[Dict[str, Any]] = None


class VoiceSymptomRequest(BaseModel):
    """Request model for voice symptom report."""
    patient_id: int
    additional_context: Optional[str] = None


class ImageSymptomRequest(BaseModel):
    """Request model for image symptom report."""
    description: str
    patient_id: int


@router.post("/analyze/multimodal")
async def analyze_multimodal_symptoms(
    text_input: Optional[str] = Form(None),
    voice_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None),
    patient_context: Optional[str] = Form(None),
    db = Depends(get_db)
):
    """
    Analyze symptoms using multiple input modalities.

    This endpoint accepts text descriptions, voice recordings,
    and images for comprehensive symptom analysis.
    """
    try:
        # Parse inputs
        voice_data = None
        image_data = None

        if voice_file and voice_file.content_type.startswith("audio/"):
            voice_data = await voice_file.read()

        if image_file and image_file.content_type.startswith("image/"):
            image_data = await image_file.read()

        # Parse patient context
        context = None
        if patient_context:
            try:
                context = json.loads(patient_context)
            except:
                context = {"notes": patient_context}

        # Run multimodal analysis
        analysis = await multimodal_symptom_service.analyze_multimodal_symptoms(
            text_input=text_input,
            voice_input=voice_data,
            image_input=image_data,
            patient_context=context
        )

        if "error" in analysis:
            raise HTTPException(
                status_code=500,
                detail=analysis["error"]
            )

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in multimodal symptom analysis", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze multimodal symptoms"
        )


@router.post("/analyze/voice")
async def analyze_voice_symptoms(
    voice_file: UploadFile = File(...),
    request: VoiceSymptomRequest = Depends(),
    db = Depends(get_db)
):
    """
    Process voice-recorded symptom report.

    This endpoint transcribes and analyzes voice recordings
    of symptom descriptions for enhanced assessment.
    """
    try:
        if not voice_file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an audio file"
            )

        voice_data = await voice_file.read()

        analysis = await multimodal_symptom_service.process_voice_symptom_report(
            voice_data=voice_data,
            patient_id=request.patient_id,
            additional_context=request.additional_context
        )

        if "error" in analysis:
            raise HTTPException(
                status_code=500,
                detail=analysis["error"]
            )

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error analyzing voice symptoms", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze voice symptoms"
        )


@router.post("/analyze/image")
async def analyze_image_symptoms(
    image_file: UploadFile = File(...),
    request: ImageSymptomRequest = Depends(),
    db = Depends(get_db)
):
    """
    Process image-based symptom report.

    This endpoint analyzes images of rashes, wounds, or other
    visible symptoms for enhanced assessment.
    """
    try:
        if not image_file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image file"
            )

        image_data = await image_file.read()

        analysis = await multimodal_symptom_service.process_image_symptom_report(
            image_data=image_data,
            description=request.description,
            patient_id=request.patient_id
        )

        if "error" in analysis:
            raise HTTPException(
                status_code=500,
                detail=analysis["error"]
            )

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error analyzing image symptoms", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze image symptoms"
        )


@router.get("/capabilities")
async def get_multimodal_capabilities():
    """
    Get available multimodal analysis capabilities.

    This endpoint returns information about supported
    input modalities and analysis features.
    """
    try:
        capabilities = {
            "text_analysis": {
                "available": True,
                "features": ["symptom_extraction", "severity_assessment", "condition_identification"],
                "supported_languages": ["en"],  # Would include more languages
                "confidence_threshold": 0.7
            },
            "voice_analysis": {
                "available": settings.ENABLE_VOICE_FEATURES,
                "features": ["transcription", "emotion_analysis", "speech_feature_extraction"],
                "supported_formats": ["wav", "mp3", "m4a", "ogg"],
                "max_duration_minutes": 10,
                "confidence_threshold": 0.6
            },
            "image_analysis": {
                "available": settings.ENABLE_VIDEO_FEATURES,
                "features": ["skin_lesion_classification", "wound_assessment", "visual_symptom_analysis"],
                "supported_formats": ["jpg", "jpeg", "png", "bmp"],
                "max_file_size_mb": 10,
                "supported_conditions": ["skin_lesions", "wounds", "rashes"],
                "confidence_threshold": 0.8
            },
            "multimodal_fusion": {
                "available": True,
                "features": ["cross_modal_validation", "enhanced_confidence_scoring"],
                "supported_combinations": ["text+voice", "text+image", "voice+image", "all_three"],
                "fusion_algorithms": ["weighted_average", "attention_based", "rule_based"]
            },
            "vital_signs_extraction": {
                "available": False,  # Would require facial recognition models
                "features": ["heart_rate_estimation", "respiratory_rate", "stress_indicators"],
                "modalities": ["voice", "image"]
            }
        }

        return {
            "capabilities": capabilities,
            "last_updated": "2024-01-01T00:00:00Z"
        }

    except Exception as e:
        logger.error("Error getting multimodal capabilities", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get multimodal capabilities"
        )


@router.get("/patient/{patient_id}/multimodal-history")
async def get_multimodal_analysis_history(
    patient_id: int,
    limit: int = 20,
    modality_filter: Optional[str] = None,
    db = Depends(get_db)
):
    """
    Get history of multimodal symptom analyses for a patient.

    This endpoint provides historical multimodal analyses
    for tracking symptom patterns across different input types.
    """
    try:
        # TODO: Get multimodal analysis history from database

        return {
            "patient_id": patient_id,
            "total_analyses": 15,
            "modality_breakdown": {
                "text_only": 8,
                "voice_only": 3,
                "image_only": 2,
                "multimodal": 2
            },
            "recent_analyses": [
                {
                    "analysis_id": 1,
                    "timestamp": "2024-01-01T00:00:00Z",
                    "modalities_used": ["text", "voice"],
                    "urgency_level": "routine",
                    "confidence_score": 0.85,
                    "symptoms_analyzed": ["headache", "fatigue"]
                }
            ],
            "trends": {
                "modality_usage_trend": "increasing_multimodal",
                "average_confidence": 0.78,
                "common_symptom_patterns": ["headache", "fatigue", "nausea"]
            },
            "last_updated": "2024-01-01T00:00:00Z"
        }

    except Exception as e:
        logger.error("Error getting multimodal history", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get multimodal analysis history"
        )
