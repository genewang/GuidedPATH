"""
API router for wearable device integration
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import structlog

from backend.core.database import get_db
from backend.apps.wearables.wearable_service import wearable_service

logger = structlog.get_logger()

router = APIRouter()


class DeviceConnectionRequest(BaseModel):
    """Request model for wearable device connection."""
    device_type: str
    device_id: str
    auth_tokens: Dict[str, str]


class DataSyncRequest(BaseModel):
    """Request model for wearable data synchronization."""
    device_type: str
    sync_period_hours: int = 24


class DashboardRequest(BaseModel):
    """Request model for wearable dashboard."""
    time_range_days: int = 7


@router.post("/device/connect")
async def connect_wearable_device(
    request: DeviceConnectionRequest,
    patient_id: int,  # Would come from authentication middleware
    db = Depends(get_db)
):
    """
    Connect wearable device for patient.

    This endpoint establishes connection with wearable devices
    and sets up data synchronization.
    """
    try:
        result = await wearable_service.device_manager.connect_wearable_device(
            patient_id=patient_id,
            device_type=request.device_type,
            device_id=request.device_id,
            auth_tokens=request.auth_tokens
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
        logger.error("Error connecting wearable device", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to connect wearable device"
        )


@router.post("/patient/{patient_id}/devices/connect")
async def connect_multiple_devices(
    patient_id: int,
    device_connections: List[DeviceConnectionRequest],
    db = Depends(get_db)
):
    """
    Connect multiple wearable devices for patient.

    This endpoint allows connecting multiple devices
    in a single operation.
    """
    try:
        # Convert requests to device connection format
        connections = [
            {
                "device_type": conn.device_type,
                "device_id": conn.device_id,
                "auth_tokens": conn.auth_tokens
            }
            for conn in device_connections
        ]

        result = await wearable_service.connect_patient_devices(patient_id, connections)

        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error connecting multiple devices", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to connect multiple devices"
        )


@router.post("/data/sync")
async def sync_wearable_data(
    request: DataSyncRequest,
    patient_id: int,  # Would come from authentication middleware
    db = Depends(get_db)
):
    """
    Sync wearable device data for patient.

    This endpoint retrieves and processes data from
    connected wearable devices.
    """
    try:
        result = await wearable_service.device_manager.sync_wearable_data(
            patient_id=patient_id,
            device_type=request.device_type,
            sync_period_hours=request.sync_period_hours
        )

        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error syncing wearable data", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to sync wearable data"
        )


@router.get("/dashboard/{patient_id}")
async def get_wearable_dashboard(
    patient_id: int,
    time_range_days: int = 7,
    db = Depends(get_db)
):
    """
    Get comprehensive wearable data dashboard.

    This endpoint provides a complete overview of patient
    health data from all connected wearable devices.
    """
    try:
        dashboard = await wearable_service.get_patient_wearable_dashboard(
            patient_id=patient_id,
            time_range_days=time_range_days
        )

        if "error" in dashboard:
            raise HTTPException(
                status_code=500,
                detail=dashboard["error"]
            )

        return dashboard

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting wearable dashboard", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get wearable dashboard"
        )


@router.get("/insights/{patient_id}")
async def get_wearable_insights(
    patient_id: int,
    time_range_days: int = 7,
    db = Depends(get_db)
):
    """
    Get health insights from wearable data.

    This endpoint provides AI-generated insights and
    trends from wearable device data.
    """
    try:
        insights = await wearable_service.device_manager.get_wearable_insights(
            patient_id=patient_id,
            time_range_days=time_range_days
        )

        if "error" in insights:
            raise HTTPException(
                status_code=500,
                detail=insights["error"]
            )

        return insights

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting wearable insights", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get wearable insights"
        )


@router.get("/devices/available")
async def get_available_devices():
    """
    Get available wearable devices for integration.

    This endpoint returns all supported wearable devices
    with their capabilities and features.
    """
    try:
        devices = [
            {
                "id": "fitbit",
                "name": "Fitbit",
                "category": "fitness_tracker",
                "supported_data_types": ["heart_rate", "steps", "sleep", "activity", "oxygen_saturation"],
                "api_authentication": "oauth2",
                "features": ["real_time_sync", "mobile_app_integration", "social_features"],
                "battery_life_days": 7,
                "water_resistance": "50m",
                "price_range": "$100-$400",
                "compatibility": ["iOS", "Android"]
            },
            {
                "id": "apple_health",
                "name": "Apple Health",
                "category": "health_platform",
                "supported_data_types": ["heart_rate", "steps", "sleep", "workouts", "vitals"],
                "api_authentication": "api_key",
                "features": ["comprehensive_health_data", "medical_records", "research_kit"],
                "battery_life_days": 1,  # Phone-dependent
                "water_resistance": "IP68",
                "price_range": "Included with iPhone",
                "compatibility": ["iOS"]
            },
            {
                "id": "garmin",
                "name": "Garmin",
                "category": "sports_watch",
                "supported_data_types": ["heart_rate", "steps", "sleep", "stress", "body_battery"],
                "api_authentication": "oauth2",
                "features": ["advanced_metrics", "gps_tracking", "performance_analytics"],
                "battery_life_days": 14,
                "water_resistance": "100m",
                "price_range": "$200-$1000",
                "compatibility": ["iOS", "Android"]
            },
            {
                "id": "oura",
                "name": "Oura Ring",
                "category": "smart_ring",
                "supported_data_types": ["sleep", "heart_rate", "temperature", "activity"],
                "api_authentication": "api_key",
                "features": ["sleep_staging", "readiness_score", "discreet_design"],
                "battery_life_days": 7,
                "water_resistance": "100m",
                "price_range": "$300-$400",
                "compatibility": ["iOS", "Android"]
            }
        ]

        return {
            "devices": devices,
            "total_devices": len(devices),
            "categories": ["fitness_tracker", "health_platform", "sports_watch", "smart_ring"],
            "last_updated": "2024-01-01T00:00:00Z"
        }

    except Exception as e:
        logger.error("Error getting available devices", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get available devices"
        )


@router.get("/patient/{patient_id}/alerts")
async def get_wearable_alerts(
    patient_id: int,
    active_only: bool = True,
    db = Depends(get_db)
):
    """
    Get wearable device alerts for patient.

    This endpoint returns health alerts generated from
    wearable device data analysis.
    """
    try:
        # TODO: Get alerts from database

        alerts = []

        if active_only:
            # Filter for active alerts only
            alerts = [alert for alert in alerts if alert.get("status") == "active"]

        return {
            "patient_id": patient_id,
            "total_alerts": len(alerts),
            "active_alerts": len([a for a in alerts if a.get("status") == "active"]),
            "alerts": alerts,
            "last_updated": "2024-01-01T00:00:00Z"
        }

    except Exception as e:
        logger.error("Error getting wearable alerts", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get wearable alerts"
        )


@router.get("/analytics/data-quality")
async def get_data_quality_analytics(
    patient_id: Optional[int] = None,
    time_range_days: int = 30,
    db = Depends(get_db)
):
    """
    Get data quality analytics for wearable integrations.

    This endpoint provides insights into data completeness,
    accuracy, and reliability across devices.
    """
    try:
        # TODO: Calculate data quality metrics from database

        return {
            "time_range_days": time_range_days,
            "patient_filter": patient_id,
            "data_quality_metrics": {
                "overall_completeness": 0.87,
                "data_accuracy": 0.92,
                "sync_reliability": 0.89,
                "device_connectivity": 0.94
            },
            "device_breakdown": [
                {
                    "device_type": "fitbit",
                    "completeness": 0.91,
                    "accuracy": 0.95,
                    "sync_success_rate": 0.96,
                    "data_points_per_day": 1440  # Every minute
                },
                {
                    "device_type": "apple_health",
                    "completeness": 0.85,
                    "accuracy": 0.90,
                    "sync_success_rate": 0.92,
                    "data_points_per_day": 720   # Every 2 minutes
                }
            ],
            "quality_trends": {
                "completeness_trend": "improving",
                "accuracy_trend": "stable",
                "reliability_trend": "improving"
            },
            "recommendations": [
                "Ensure consistent device charging for better data completeness",
                "Update device firmware for improved accuracy",
                "Consider device replacement if sync issues persist"
            ],
            "generated_at": "2024-01-01T00:00:00Z"
        }

    except Exception as e:
        logger.error("Error getting data quality analytics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get data quality analytics"
        )
