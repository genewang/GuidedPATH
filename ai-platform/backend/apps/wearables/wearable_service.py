"""
Wearable device integration and health monitoring service
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json
import base64

import aiohttp
import structlog

from backend.core.config import settings

logger = structlog.get_logger()


class WearableDeviceManager:
    """
    Manager for wearable device integrations and data synchronization.
    """

    def __init__(self):
        """Initialize wearable device manager."""
        self.device_configs = {
            "fitbit": {
                "api_base": "https://api.fitbit.com/1/user/-",
                "client_id": settings.FITBIT_CLIENT_ID,
                "client_secret": settings.FITBIT_CLIENT_SECRET,
                "supported_data_types": ["heart_rate", "steps", "sleep", "activity", "oxygen_saturation"],
                "authentication": "oauth2"
            },
            "apple_health": {
                "api_base": "https://api.apple.com/health",
                "api_key": settings.APPLE_HEALTH_API_KEY,
                "supported_data_types": ["heart_rate", "steps", "sleep", "workouts", "vitals"],
                "authentication": "api_key"
            },
            "garmin": {
                "api_base": "https://apis.garmin.com/wellness-api",
                "api_key": settings.GARMIN_API_KEY,
                "supported_data_types": ["heart_rate", "steps", "sleep", "stress", "body_battery"],
                "authentication": "oauth2"
            },
            "oura": {
                "api_base": "https://api.ouraring.com/v1",
                "api_key": settings.OURA_API_KEY,
                "supported_data_types": ["sleep", "heart_rate", "temperature", "activity"],
                "authentication": "api_key"
            },
            "whoop": {
                "api_base": "https://api.whoop.com/v1",
                "api_key": settings.WHOOP_API_KEY,
                "supported_data_types": ["heart_rate", "sleep", "strain", "recovery"],
                "authentication": "oauth2"
            }
        }

        # Data processing configurations
        self.data_processing_rules = {
            "heart_rate": {
                "normal_range": (60, 100),
                "alert_thresholds": {
                    "low": 50,
                    "high": 120,
                    "critical_high": 150
                }
            },
            "steps": {
                "daily_goal": 10000,
                "sedentary_threshold": 5000
            },
            "sleep": {
                "recommended_hours": (7, 9),
                "deep_sleep_target": 0.25  # 25% of total sleep
            }
        }

    async def connect_wearable_device(
        self,
        patient_id: int,
        device_type: str,
        device_id: str,
        auth_tokens: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Connect and authenticate wearable device for patient.

        Args:
            patient_id: Patient identifier
            device_type: Type of wearable device
            device_id: Device identifier
            auth_tokens: Authentication tokens for device API

        Returns:
            Device connection status and configuration
        """
        try:
            logger.info("Connecting wearable device", patient_id=patient_id, device_type=device_type)

            if device_type not in self.device_configs:
                return {"error": f"Unsupported device type: {device_type}"}

            device_config = self.device_configs[device_type]

            # Validate authentication tokens
            if not await self._validate_device_auth(device_type, auth_tokens):
                return {"error": "Invalid authentication tokens"}

            # Create device connection record
            device_connection = {
                "patient_id": patient_id,
                "device_type": device_type,
                "device_id": device_id,
                "connected_at": datetime.utcnow(),
                "auth_tokens": self._encrypt_auth_tokens(auth_tokens),
                "sync_status": "active",
                "last_sync": datetime.utcnow(),
                "data_types_enabled": device_config["supported_data_types"],
                "sync_frequency_minutes": 15,  # Default sync frequency
                "alert_thresholds": self._get_default_alert_thresholds(device_type),
                "privacy_settings": {
                    "data_sharing": "patient_consent",
                    "retention_days": 90,
                    "third_party_access": False
                }
            }

            # Test device connectivity
            connectivity_test = await self._test_device_connectivity(device_type, auth_tokens)

            if not connectivity_test["success"]:
                device_connection["sync_status"] = "error"
                device_connection["last_error"] = connectivity_test["error"]

            # Store device connection (in production, save to database)
            logger.info("Wearable device connected", patient_id=patient_id, device_type=device_type)

            return {
                "connection_status": "success" if connectivity_test["success"] else "error",
                "device_info": device_connection,
                "connectivity_test": connectivity_test,
                "next_sync_time": (datetime.utcnow() + timedelta(minutes=device_connection["sync_frequency_minutes"])).isoformat()
            }

        except Exception as e:
            logger.error("Error connecting wearable device", error=str(e), patient_id=patient_id)
            return {"error": "Failed to connect wearable device"}

    async def _validate_device_auth(self, device_type: str, auth_tokens: Dict[str, str]) -> bool:
        """Validate device authentication tokens."""

        try:
            device_config = self.device_configs[device_type]

            if device_config["authentication"] == "oauth2":
                # Validate OAuth2 tokens
                return "access_token" in auth_tokens and "refresh_token" in auth_tokens

            elif device_config["authentication"] == "api_key":
                # Validate API key
                return "api_key" in auth_tokens

            return False

        except Exception as e:
            logger.error("Error validating device auth", error=str(e))
            return False

    def _encrypt_auth_tokens(self, auth_tokens: Dict[str, str]) -> str:
        """Encrypt authentication tokens for storage."""

        # In production, use proper encryption
        token_json = json.dumps(auth_tokens)
        return base64.b64encode(token_json.encode()).decode()

    async def _test_device_connectivity(self, device_type: str, auth_tokens: Dict[str, str]) -> Dict[str, Any]:
        """Test connectivity to wearable device API."""

        try:
            # Test API connectivity with a simple request
            # In production, make actual API call to verify tokens

            return {
                "success": True,
                "response_time_ms": 150,
                "data_available": True
            }

        except Exception as e:
            logger.error("Error testing device connectivity", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }

    def _get_default_alert_thresholds(self, device_type: str) -> Dict[str, Any]:
        """Get default alert thresholds for device type."""

        # Base thresholds on device capabilities
        base_thresholds = {
            "heart_rate_alerts": True,
            "activity_alerts": True,
            "sleep_alerts": True
        }

        # Customize based on device type
        if device_type == "apple_health":
            base_thresholds["vital_signs_alerts"] = True

        elif device_type == "oura":
            base_thresholds["sleep_quality_alerts"] = True

        return base_thresholds

    async def sync_wearable_data(
        self,
        patient_id: int,
        device_type: str,
        sync_period_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Sync wearable device data for patient.

        Args:
            patient_id: Patient identifier
            device_type: Type of wearable device
            sync_period_hours: Hours of data to sync

        Returns:
            Sync results with health data
        """
        try:
            logger.info("Syncing wearable data", patient_id=patient_id, device_type=device_type)

            # Get device connection info
            device_connection = await self._get_device_connection(patient_id, device_type)
            if not device_connection:
                return {"error": "Device not connected"}

            # Decrypt auth tokens
            auth_tokens = self._decrypt_auth_tokens(device_connection["auth_tokens"])

            # Sync data from device API
            sync_results = await self._sync_from_device_api(
                device_type, auth_tokens, sync_period_hours
            )

            if "error" in sync_results:
                return sync_results

            # Process and analyze synced data
            processed_data = await self._process_synced_data(sync_results["raw_data"])

            # Generate health insights
            health_insights = await self._generate_health_insights(processed_data)

            # Check for alert conditions
            alerts = await self._check_alert_conditions(processed_data, device_connection["alert_thresholds"])

            return {
                "patient_id": patient_id,
                "device_type": device_type,
                "sync_timestamp": datetime.utcnow(),
                "sync_period_hours": sync_period_hours,
                "data_synced": {
                    "heart_rate_records": len(processed_data.get("heart_rate", [])),
                    "activity_records": len(processed_data.get("activity", [])),
                    "sleep_records": len(processed_data.get("sleep", [])),
                    "total_records": sum(len(v) for v in processed_data.values())
                },
                "processed_data": processed_data,
                "health_insights": health_insights,
                "alerts": alerts,
                "next_sync_time": (datetime.utcnow() + timedelta(minutes=device_connection["sync_frequency_minutes"])).isoformat()
            }

        except Exception as e:
            logger.error("Error syncing wearable data", error=str(e), patient_id=patient_id)
            return {"error": "Failed to sync wearable data"}

    async def _get_device_connection(self, patient_id: int, device_type: str) -> Optional[Dict[str, Any]]:
        """Get device connection information from database."""

        # In production, query database for device connection
        # For now, return mock data
        return {
            "patient_id": patient_id,
            "device_type": device_type,
            "auth_tokens": "encrypted_tokens",
            "alert_thresholds": self._get_default_alert_thresholds(device_type)
        }

    def _decrypt_auth_tokens(self, encrypted_tokens: str) -> Dict[str, str]:
        """Decrypt authentication tokens."""

        # In production, use proper decryption
        try:
            decrypted = base64.b64decode(encrypted_tokens.encode()).decode()
            return json.loads(decrypted)
        except:
            return {}

    async def _sync_from_device_api(
        self,
        device_type: str,
        auth_tokens: Dict[str, str],
        sync_period_hours: int
    ) -> Dict[str, Any]:
        """Sync data from device-specific API."""

        try:
            device_config = self.device_configs[device_type]

            if device_config["authentication"] == "oauth2":
                return await self._sync_oauth_device(device_type, auth_tokens, sync_period_hours)
            else:
                return await self._sync_api_key_device(device_type, auth_tokens, sync_period_hours)

        except Exception as e:
            logger.error(f"Error syncing from {device_type} API", error=str(e))
            return {"error": f"Failed to sync from {device_type}"}

    async def _sync_oauth_device(
        self,
        device_type: str,
        auth_tokens: Dict[str, str],
        sync_period_hours: int
    ) -> Dict[str, Any]:
        """Sync data from OAuth-based device API."""

        # In production, implement OAuth API calls
        # For now, return mock data

        mock_data = {
            "heart_rate": [
                {"timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat(), "value": 70 + i}
                for i in range(sync_period_hours)
            ],
            "steps": [
                {"timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat(), "value": 8000 + i * 100}
                for i in range(sync_period_hours)
            ],
            "sleep": [
                {
                    "date": (datetime.utcnow() - timedelta(days=i)).date().isoformat(),
                    "duration_hours": 7.5,
                    "quality_score": 85,
                    "stages": {"deep": 1.8, "rem": 1.2, "light": 4.5}
                }
                for i in range(min(sync_period_hours // 24, 7))
            ]
        }

        return {
            "raw_data": mock_data,
            "sync_success": True,
            "data_points": sum(len(v) for v in mock_data.values())
        }

    async def _sync_api_key_device(
        self,
        device_type: str,
        auth_tokens: Dict[str, str],
        sync_period_hours: int
    ) -> Dict[str, Any]:
        """Sync data from API key-based device."""

        # Similar to OAuth sync but with API key authentication
        return await self._sync_oauth_device(device_type, auth_tokens, sync_period_hours)

    async def _process_synced_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and normalize synced wearable data."""

        processed_data = {}

        # Process heart rate data
        if "heart_rate" in raw_data:
            processed_data["heart_rate"] = await self._process_heart_rate_data(raw_data["heart_rate"])

        # Process activity data
        if "steps" in raw_data:
            processed_data["activity"] = await self._process_activity_data(raw_data["steps"])

        # Process sleep data
        if "sleep" in raw_data:
            processed_data["sleep"] = await self._process_sleep_data(raw_data["sleep"])

        return processed_data

    async def _process_heart_rate_data(self, heart_rate_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process heart rate data with trend analysis."""

        if not heart_rate_data:
            return {}

        # Calculate statistics
        values = [record["value"] for record in heart_rate_data if "value" in record]

        if not values:
            return {}

        return {
            "readings": heart_rate_data,
            "statistics": {
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "resting_rate": min(values),  # Approximate resting rate
                "variability": self._calculate_heart_rate_variability(values)
            },
            "trends": await self._analyze_heart_rate_trends(heart_rate_data)
        }

    async def _process_activity_data(self, activity_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process activity and steps data."""

        if not activity_data:
            return {}

        daily_steps = {}
        for record in activity_data:
            date = datetime.fromisoformat(record["timestamp"]).date()
            if date not in daily_steps:
                daily_steps[date] = 0
            daily_steps[date] += record.get("value", 0)

        return {
            "daily_steps": [{"date": date.isoformat(), "steps": steps} for date, steps in daily_steps.items()],
            "average_daily_steps": sum(daily_steps.values()) / len(daily_steps) if daily_steps else 0,
            "goal_achievement_rate": sum(1 for steps in daily_steps.values() if steps >= 10000) / len(daily_steps) if daily_steps else 0
        }

    async def _process_sleep_data(self, sleep_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process sleep data with quality analysis."""

        if not sleep_data:
            return {}

        # Calculate sleep statistics
        durations = [record.get("duration_hours", 0) for record in sleep_data]
        quality_scores = [record.get("quality_score", 0) for record in sleep_data]

        return {
            "sleep_records": sleep_data,
            "statistics": {
                "average_duration": sum(durations) / len(durations) if durations else 0,
                "average_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                "total_sleep_hours": sum(durations),
                "deep_sleep_percentage": self._calculate_deep_sleep_percentage(sleep_data)
            },
            "trends": await self._analyze_sleep_trends(sleep_data)
        }

    def _calculate_heart_rate_variability(self, values: List[float]) -> float:
        """Calculate heart rate variability (simplified)."""

        if len(values) < 2:
            return 0.0

        # Simple standard deviation as HRV proxy
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    async def _analyze_heart_rate_trends(self, heart_rate_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze heart rate trends over time."""

        if len(heart_rate_data) < 24:
            return {"trend": "insufficient_data"}

        # Simple trend analysis
        recent_values = [record["value"] for record in heart_rate_data[-12:]]  # Last 12 hours
        older_values = [record["value"] for record in heart_rate_data[:12]]   # First 12 hours

        recent_avg = sum(recent_values) / len(recent_values)
        older_avg = sum(older_values) / len(older_values)

        if recent_avg > older_avg + 5:
            trend = "increasing"
        elif recent_avg < older_avg - 5:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "recent_average": recent_avg,
            "older_average": older_avg,
            "change_percent": ((recent_avg - older_avg) / older_avg) * 100 if older_avg > 0 else 0
        }

    async def _analyze_sleep_trends(self, sleep_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sleep trends over time."""

        if len(sleep_data) < 3:
            return {"trend": "insufficient_data"}

        # Analyze sleep duration trend
        durations = [record.get("duration_hours", 0) for record in sleep_data]

        if len(durations) >= 7:  # Weekly trend
            recent_week = durations[-7:]
            older_week = durations[:7] if len(durations) > 14 else durations[:len(durations)//2]

            recent_avg = sum(recent_week) / len(recent_week)
            older_avg = sum(older_week) / len(older_week)

            trend = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"

            return {
                "trend": trend,
                "recent_average": recent_avg,
                "older_average": older_avg,
                "consistency_score": self._calculate_sleep_consistency(durations)
            }

        return {"trend": "insufficient_data"}

    def _calculate_deep_sleep_percentage(self, sleep_data: List[Dict[str, Any]]) -> float:
        """Calculate average deep sleep percentage."""

        total_deep = 0
        total_sleep = 0

        for record in sleep_data:
            stages = record.get("stages", {})
            deep = stages.get("deep", 0)
            total_duration = sum(stages.values())

            if total_duration > 0:
                total_deep += deep
                total_sleep += total_duration

        return (total_deep / total_sleep) * 100 if total_sleep > 0 else 0

    def _calculate_sleep_consistency(self, durations: List[float]) -> float:
        """Calculate sleep duration consistency score."""

        if len(durations) < 2:
            return 0.0

        # Calculate coefficient of variation (lower is more consistent)
        mean = sum(durations) / len(durations)
        if mean == 0:
            return 0.0

        variance = sum((x - mean) ** 2 for x in durations) / len(durations)
        std_dev = variance ** 0.5
        cv = std_dev / mean  # Coefficient of variation

        # Convert to consistency score (0-1, higher is better)
        return max(0.0, 1.0 - cv)

    async def _generate_health_insights(self, processed_data: Dict[str, Any]) -> List[str]:
        """Generate health insights from wearable data."""

        insights = []

        # Heart rate insights
        if "heart_rate" in processed_data:
            hr_stats = processed_data["heart_rate"]["statistics"]
            avg_hr = hr_stats.get("average", 0)

            if avg_hr > 100:
                insights.append("Elevated average heart rate detected - consider consulting healthcare provider")
            elif avg_hr < 60:
                insights.append("Low average heart rate detected - monitor for bradycardia")

        # Activity insights
        if "activity" in processed_data:
            activity_stats = processed_data["activity"]
            avg_steps = activity_stats.get("average_daily_steps", 0)

            if avg_steps < 5000:
                insights.append("Low daily activity level - consider increasing physical activity")
            elif avg_steps > 15000:
                insights.append("High activity level - ensure adequate rest and recovery")

        # Sleep insights
        if "sleep" in processed_data:
            sleep_stats = processed_data["sleep"]["statistics"]
            avg_duration = sleep_stats.get("average_duration", 0)

            if avg_duration < 7:
                insights.append("Insufficient sleep duration - aim for 7-9 hours per night")
            elif avg_duration > 10:
                insights.append("Excessive sleep duration - consider sleep quality assessment")

        return insights

    async def _check_alert_conditions(
        self,
        processed_data: Dict[str, Any],
        alert_thresholds: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for alert conditions in wearable data."""

        alerts = []

        # Check heart rate alerts
        if "heart_rate" in processed_data and alert_thresholds.get("heart_rate_alerts", False):
            hr_alerts = await self._check_heart_rate_alerts(processed_data["heart_rate"])
            alerts.extend(hr_alerts)

        # Check activity alerts
        if "activity" in processed_data and alert_thresholds.get("activity_alerts", False):
            activity_alerts = await self._check_activity_alerts(processed_data["activity"])
            alerts.extend(activity_alerts)

        # Check sleep alerts
        if "sleep" in processed_data and alert_thresholds.get("sleep_alerts", False):
            sleep_alerts = await self._check_sleep_alerts(processed_data["sleep"])
            alerts.extend(sleep_alerts)

        return alerts

    async def _check_heart_rate_alerts(self, heart_rate_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for heart rate alert conditions."""

        alerts = []
        readings = heart_rate_data.get("readings", [])

        # Check recent readings for abnormal values
        recent_readings = readings[-10:] if len(readings) >= 10 else readings

        for reading in recent_readings:
            hr_value = reading.get("value", 0)

            if hr_value < 50:
                alerts.append({
                    "type": "bradycardia",
                    "severity": "medium",
                    "message": f"Low heart rate detected: {hr_value} bpm",
                    "timestamp": reading.get("timestamp"),
                    "recommendation": "Monitor for symptoms and consult provider if persistent"
                })

            elif hr_value > 120:
                alerts.append({
                    "type": "tachycardia",
                    "severity": "high",
                    "message": f"Elevated heart rate detected: {hr_value} bpm",
                    "timestamp": reading.get("timestamp"),
                    "recommendation": "Rest and monitor; seek care if symptoms persist"
                })

        return alerts

    async def _check_activity_alerts(self, activity_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for activity-related alerts."""

        alerts = []
        daily_steps = activity_data.get("daily_steps", [])

        # Check for sedentary behavior
        recent_days = daily_steps[-3:] if len(daily_steps) >= 3 else daily_steps

        for day in recent_days:
            if day["steps"] < 2000:
                alerts.append({
                    "type": "sedentary_behavior",
                    "severity": "low",
                    "message": f"Very low activity day: {day['steps']} steps",
                    "timestamp": day["date"],
                    "recommendation": "Consider increasing daily movement and activity"
                })

        return alerts

    async def _check_sleep_alerts(self, sleep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for sleep-related alerts."""

        alerts = []
        sleep_records = sleep_data.get("sleep_records", [])

        # Check recent sleep quality
        recent_sleep = sleep_records[-3:] if len(sleep_records) >= 3 else sleep_records

        for sleep in recent_sleep:
            quality = sleep.get("quality_score", 0)
            duration = sleep.get("duration_hours", 0)

            if quality < 60:
                alerts.append({
                    "type": "poor_sleep_quality",
                    "severity": "medium",
                    "message": f"Poor sleep quality: {quality}/100",
                    "timestamp": sleep.get("date"),
                    "recommendation": "Review sleep hygiene practices and environment"
                })

            if duration < 6:
                alerts.append({
                    "type": "insufficient_sleep",
                    "severity": "medium",
                    "message": f"Insufficient sleep: {duration} hours",
                    "timestamp": sleep.get("date"),
                    "recommendation": "Prioritize sleep and establish consistent sleep schedule"
                })

        return alerts

    async def get_wearable_insights(
        self,
        patient_id: int,
        time_range_days: int = 7
    ) -> Dict[str, Any]:
        """
        Get comprehensive health insights from wearable data.

        Args:
            patient_id: Patient identifier
            time_range_days: Time range for analysis

        Returns:
            Comprehensive health insights and trends
        """
        try:
            logger.info("Generating wearable insights", patient_id=patient_id, days=time_range_days)

            # Get all connected devices for patient
            connected_devices = await self._get_connected_devices(patient_id)

            if not connected_devices:
                return {"error": "No wearable devices connected"}

            # Aggregate data from all devices
            aggregated_data = {}

            for device in connected_devices:
                device_data = await self._get_device_data(patient_id, device["device_type"], time_range_days)

                if device_data:
                    for data_type, data in device_data.items():
                        if data_type not in aggregated_data:
                            aggregated_data[data_type] = []
                        aggregated_data[data_type].extend(data)

            # Generate comprehensive insights
            insights = {
                "patient_id": patient_id,
                "analysis_period_days": time_range_days,
                "connected_devices": len(connected_devices),
                "data_summary": await self._summarize_aggregated_data(aggregated_data),
                "health_trends": await self._analyze_health_trends(aggregated_data),
                "risk_assessment": await self._assess_health_risks(aggregated_data),
                "lifestyle_recommendations": await self._generate_lifestyle_recommendations(aggregated_data),
                "generated_at": datetime.utcnow()
            }

            logger.info("Wearable insights generated", patient_id=patient_id)
            return insights

        except Exception as e:
            logger.error("Error generating wearable insights", error=str(e), patient_id=patient_id)
            return {"error": "Failed to generate wearable insights"}

    async def _get_connected_devices(self, patient_id: int) -> List[Dict[str, Any]]:
        """Get list of connected wearable devices for patient."""

        # In production, query database for connected devices
        # For now, return mock data
        return [
            {"device_type": "fitbit", "device_id": "fitbit_123"},
            {"device_type": "apple_health", "device_id": "apple_456"}
        ]

    async def _get_device_data(
        self,
        patient_id: int,
        device_type: str,
        time_range_days: int
    ) -> Optional[Dict[str, Any]]:
        """Get historical data from specific device."""

        # In production, retrieve from database or sync from device
        # For now, return mock historical data
        return {
            "heart_rate": [
                {"timestamp": (datetime.utcnow() - timedelta(days=i)).isoformat(), "value": 70 + i}
                for i in range(time_range_days)
            ]
        }

    async def _summarize_aggregated_data(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize aggregated data from all devices."""

        summary = {}

        for data_type, data in aggregated_data.items():
            if data_type == "heart_rate":
                values = [record.get("value", 0) for record in data if "value" in record]
                summary["heart_rate"] = {
                    "average": sum(values) / len(values) if values else 0,
                    "data_points": len(values),
                    "time_range": f"{len(values)} readings"
                }

            elif data_type == "activity":
                total_steps = sum(record.get("steps", 0) for record in data if "steps" in record)
                summary["activity"] = {
                    "total_steps": total_steps,
                    "data_points": len(data),
                    "average_daily": total_steps / len(data) if data else 0
                }

        return summary

    async def _analyze_health_trends(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze health trends from aggregated data."""

        trends = {}

        # Heart rate trend
        if "heart_rate" in aggregated_data:
            hr_data = aggregated_data["heart_rate"]
            if len(hr_data) >= 7:  # Weekly trend
                recent_hr = [record["value"] for record in hr_data[-3:]]
                older_hr = [record["value"] for record in hr_data[:3]]

                recent_avg = sum(recent_hr) / len(recent_hr)
                older_avg = sum(older_hr) / len(older_hr)

                trends["heart_rate"] = {
                    "direction": "increasing" if recent_avg > older_avg else "decreasing",
                    "change_percent": abs(recent_avg - older_avg) / older_avg * 100 if older_avg > 0 else 0
                }

        return trends

    async def _assess_health_risks(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess health risks based on wearable data."""

        risks = {
            "overall_risk": "low",
            "risk_factors": [],
            "risk_score": 0
        }

        # Assess heart rate risks
        if "heart_rate" in aggregated_data:
            hr_avg = aggregated_data["heart_rate"]["statistics"].get("average", 0)

            if hr_avg > 100:
                risks["risk_factors"].append("Elevated heart rate")
                risks["risk_score"] += 30

            if hr_avg < 60:
                risks["risk_factors"].append("Bradycardia")
                risks["risk_score"] += 20

        # Assess activity risks
        if "activity" in aggregated_data:
            avg_steps = aggregated_data["activity"].get("average_daily_steps", 0)

            if avg_steps < 5000:
                risks["risk_factors"].append("Sedentary lifestyle")
                risks["risk_score"] += 25

        # Determine overall risk level
        if risks["risk_score"] > 60:
            risks["overall_risk"] = "high"
        elif risks["risk_score"] > 30:
            risks["overall_risk"] = "medium"

        return risks

    async def _generate_lifestyle_recommendations(self, aggregated_data: Dict[str, Any]) -> List[str]:
        """Generate lifestyle recommendations based on wearable data."""

        recommendations = []

        # Heart rate recommendations
        if "heart_rate" in aggregated_data:
            hr_avg = aggregated_data["heart_rate"]["statistics"].get("average", 0)

            if hr_avg > 100:
                recommendations.append("Consider stress management techniques to lower heart rate")

            if hr_avg < 60:
                recommendations.append("Monitor for symptoms of bradycardia")

        # Activity recommendations
        if "activity" in aggregated_data:
            avg_steps = aggregated_data["activity"].get("average_daily_steps", 0)

            if avg_steps < 5000:
                recommendations.append("Aim for 30 minutes of moderate activity daily")
                recommendations.append("Consider short walking breaks throughout the day")

        # Sleep recommendations
        if "sleep" in aggregated_data:
            avg_duration = aggregated_data["sleep"]["statistics"].get("average_duration", 0)

            if avg_duration < 7:
                recommendations.append("Establish consistent bedtime routine")
                recommendations.append("Create sleep-friendly environment")

        return recommendations


class WearableIntegrationService:
    """
    Comprehensive wearable device integration service.
    """

    def __init__(self):
        """Initialize wearable integration service."""
        self.device_manager = WearableDeviceManager()

    async def connect_patient_devices(
        self,
        patient_id: int,
        device_connections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Connect multiple wearable devices for a patient.

        Args:
            patient_id: Patient identifier
            device_connections: List of device connection requests

        Returns:
            Connection results for all devices
        """
        try:
            connection_results = []

            for device_connection in device_connections:
                result = await self.device_manager.connect_wearable_device(
                    patient_id=patient_id,
                    device_type=device_connection["device_type"],
                    device_id=device_connection["device_id"],
                    auth_tokens=device_connection["auth_tokens"]
                )

                connection_results.append({
                    "device_type": device_connection["device_type"],
                    "result": result
                })

            successful_connections = sum(1 for result in connection_results if result["result"].get("connection_status") == "success")

            return {
                "patient_id": patient_id,
                "total_devices": len(device_connections),
                "successful_connections": successful_connections,
                "connection_results": connection_results,
                "next_steps": "Set up data synchronization schedules" if successful_connections > 0 else "Review connection errors",
                "connected_at": datetime.utcnow()
            }

        except Exception as e:
            logger.error("Error connecting patient devices", error=str(e), patient_id=patient_id)
            return {"error": "Failed to connect patient devices"}

    async def get_patient_wearable_dashboard(
        self,
        patient_id: int,
        time_range_days: int = 7
    ) -> Dict[str, Any]:
        """
        Get comprehensive wearable data dashboard for patient.

        Args:
            patient_id: Patient identifier
            time_range_days: Time range for dashboard data

        Returns:
            Comprehensive dashboard with all wearable insights
        """
        try:
            # Get comprehensive insights
            insights = await self.device_manager.get_wearable_insights(patient_id, time_range_days)

            if "error" in insights:
                return insights

            # Get connected devices status
            connected_devices = await self.device_manager._get_connected_devices(patient_id)

            # Generate dashboard summary
            dashboard = {
                "patient_id": patient_id,
                "dashboard_timestamp": datetime.utcnow(),
                "time_range_days": time_range_days,
                "connected_devices": [
                    {
                        "device_type": device["device_type"],
                        "device_id": device["device_id"],
                        "status": "connected",
                        "last_sync": datetime.utcnow()  # Would get actual last sync time
                    }
                    for device in connected_devices
                ],
                "health_summary": insights.get("data_summary", {}),
                "trends": insights.get("health_trends", {}),
                "risk_assessment": insights.get("risk_assessment", {}),
                "recommendations": insights.get("lifestyle_recommendations", []),
                "alerts": await self._get_active_alerts(patient_id),
                "data_completeness": await self._calculate_data_completeness(connected_devices, time_range_days)
            }

            return dashboard

        except Exception as e:
            logger.error("Error generating wearable dashboard", error=str(e), patient_id=patient_id)
            return {"error": "Failed to generate wearable dashboard"}

    async def _get_active_alerts(self, patient_id: int) -> List[Dict[str, Any]]:
        """Get currently active health alerts for patient."""

        # In production, query database for active alerts
        # For now, return mock alerts
        return []

    async def _calculate_data_completeness(self, connected_devices: List[Dict[str, Any]], time_range_days: int) -> Dict[str, Any]:
        """Calculate data completeness score for dashboard."""

        if not connected_devices:
            return {"score": 0, "message": "No devices connected"}

        # Calculate based on device count and data availability
        expected_data_types = set()
        for device in connected_devices:
            device_type = device["device_type"]
            if device_type in self.device_manager.device_configs:
                expected_data_types.update(
                    self.device_manager.device_configs[device_type]["supported_data_types"]
                )

        # Mock completeness calculation
        completeness_score = min(1.0, len(expected_data_types) / 5.0)  # Normalize to 5 data types

        return {
            "score": completeness_score,
            "data_types_available": len(expected_data_types),
            "message": f"Data completeness: {completeness_score:.1%}"
        }


# Global wearable integration service
wearable_service = WearableIntegrationService()
