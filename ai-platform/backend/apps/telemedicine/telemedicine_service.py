"""
Telemedicine platform integration service
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json

import aiohttp
import structlog

from backend.core.config import settings

logger = structlog.get_logger()


class TelemedicinePlatform:
    """
    Integration with telemedicine platforms for virtual consultations.
    """

    def __init__(self):
        """Initialize telemedicine platform integrations."""
        self.platform_configs = {
            "teladoc": {
                "api_base": "https://api.teladoc.com/v1",
                "api_key": settings.TELADOC_API_KEY,
                "supported_services": ["urgent_care", "therapy", "dermatology"],
                "authentication": "oauth2"
            },
            "amwell": {
                "api_base": "https://api.amwell.com/v1",
                "api_key": settings.AMWELL_API_KEY,
                "supported_services": ["primary_care", "mental_health", "specialty"],
                "authentication": "api_key"
            },
            "doctor_on_demand": {
                "api_base": "https://api.doctorondemand.com/v1",
                "api_key": settings.DOCTOR_ON_DEMAND_API_KEY,
                "supported_services": ["urgent_care", "preventive_care", "chronic_care"],
                "authentication": "oauth2"
            },
            "mydoc": {
                "api_base": "https://api.mydoc.com/v1",
                "api_key": settings.MYDOC_API_KEY,
                "supported_services": ["family_medicine", "pediatrics", "internal_medicine"],
                "authentication": "api_key"
            }
        }

        # FHIR-based telemedicine servers
        self.fhir_telemedicine_servers = {
            "cerner": settings.CERNER_FHIR_URL,
            "epic": settings.EPIC_FHIR_URL,
            "allscripts": settings.ALLSCRIPTS_FHIR_URL
        }

    async def schedule_telemedicine_consultation(
        self,
        patient_id: int,
        consultation_request: Dict[str, Any],
        preferred_platform: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Schedule telemedicine consultation across multiple platforms.

        Args:
            patient_id: Patient identifier
            consultation_request: Consultation details (specialty, urgency, symptoms)
            preferred_platform: Preferred telemedicine platform

        Returns:
            Scheduled consultation details
        """
        try:
            logger.info("Scheduling telemedicine consultation", patient_id=patient_id, specialty=consultation_request.get("specialty"))

            # Determine best platform for the consultation
            selected_platform = await self._select_optimal_platform(
                consultation_request,
                preferred_platform
            )

            if not selected_platform:
                return {"error": "No suitable telemedicine platform available"}

            # Schedule consultation on selected platform
            platform_config = self.platform_configs[selected_platform]

            if platform_config["authentication"] == "oauth2":
                schedule_result = await self._schedule_via_oauth_api(
                    selected_platform, consultation_request
                )
            else:
                schedule_result = await self._schedule_via_api_key(
                    selected_platform, consultation_request
                )

            if "error" in schedule_result:
                return schedule_result

            # Create consultation record
            consultation_record = {
                "patient_id": patient_id,
                "platform": selected_platform,
                "consultation_id": schedule_result.get("consultation_id"),
                "scheduled_time": schedule_result.get("scheduled_time"),
                "provider_info": schedule_result.get("provider_info"),
                "meeting_link": schedule_result.get("meeting_link"),
                "instructions": schedule_result.get("instructions"),
                "scheduled_at": datetime.utcnow(),
                "status": "scheduled"
            }

            # Store consultation record (in production, save to database)
            logger.info("Telemedicine consultation scheduled", patient_id=patient_id, platform=selected_platform)

            return consultation_record

        except Exception as e:
            logger.error("Error scheduling telemedicine consultation", error=str(e), patient_id=patient_id)
            return {"error": "Failed to schedule telemedicine consultation"}

    async def _select_optimal_platform(
        self,
        consultation_request: Dict[str, Any],
        preferred_platform: Optional[str]
    ) -> Optional[str]:
        """Select optimal telemedicine platform for consultation."""

        specialty = consultation_request.get("specialty", "").lower()
        urgency = consultation_request.get("urgency", "routine")

        # Platform selection logic
        platform_scores = {}

        for platform, config in self.platform_configs.items():
            score = 0

            # Check if platform supports the specialty
            supported_services = config.get("supported_services", [])
            if specialty in supported_services:
                score += 10

            # Check urgency compatibility
            if urgency == "urgent" and "urgent_care" in supported_services:
                score += 5

            # Prefer user's preferred platform if it supports the service
            if preferred_platform == platform and score > 0:
                score += 3

            platform_scores[platform] = score

        # Return platform with highest score
        if platform_scores:
            return max(platform_scores.items(), key=lambda x: x[1])[0]

        return None

    async def _schedule_via_oauth_api(self, platform: str, consultation_request: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule consultation via OAuth-based API."""

        try:
            # In production, implement OAuth flow and API calls
            # For now, return mock scheduling result

            return {
                "consultation_id": f"{platform}_consult_{datetime.now().timestamp()}",
                "scheduled_time": (datetime.utcnow() + timedelta(hours=2)).isoformat(),
                "provider_info": {
                    "name": "Dr. Smith",
                    "specialty": consultation_request.get("specialty"),
                    "rating": 4.8
                },
                "meeting_link": f"https://{platform}.com/meeting/mock-link",
                "instructions": "Please be ready 5 minutes before your scheduled time"
            }

        except Exception as e:
            logger.error(f"Error scheduling via {platform} OAuth API", error=str(e))
            return {"error": f"Failed to schedule via {platform}"}

    async def _schedule_via_api_key(self, platform: str, consultation_request: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule consultation via API key-based platform."""

        try:
            # In production, implement API key authentication and calls
            # For now, return mock scheduling result

            return {
                "consultation_id": f"{platform}_consult_{datetime.now().timestamp()}",
                "scheduled_time": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
                "provider_info": {
                    "name": "Dr. Johnson",
                    "specialty": consultation_request.get("specialty"),
                    "rating": 4.7
                },
                "meeting_link": f"https://{platform}.com/meeting/mock-link",
                "instructions": "Join the meeting from a quiet, well-lit location"
            }

        except Exception as e:
            logger.error(f"Error scheduling via {platform} API key", error=str(e))
            return {"error": f"Failed to schedule via {platform}"}

    async def get_available_appointments(
        self,
        specialty: str,
        urgency: str = "routine",
        max_results: int = 10,
        preferred_platform: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get available telemedicine appointments across platforms.

        Args:
            specialty: Medical specialty needed
            urgency: Urgency level (routine, urgent, emergency)
            max_results: Maximum number of results
            preferred_platform: Preferred platform

        Returns:
            List of available appointments
        """
        try:
            available_appointments = []

            # Query each platform for availability
            for platform, config in self.platform_configs.items():
                if preferred_platform and platform != preferred_platform:
                    continue

                if specialty not in config.get("supported_services", []):
                    continue

                # Get platform availability
                platform_appointments = await self._get_platform_availability(
                    platform, specialty, urgency
                )

                available_appointments.extend(platform_appointments)

            # Sort by earliest availability
            available_appointments.sort(key=lambda x: x["available_time"])

            return available_appointments[:max_results]

        except Exception as e:
            logger.error("Error getting available appointments", error=str(e))
            return []

    async def _get_platform_availability(
        self,
        platform: str,
        specialty: str,
        urgency: str
    ) -> List[Dict[str, Any]]:
        """Get availability for specific platform."""

        try:
            # In production, query actual platform APIs
            # For now, return mock availability

            base_time = datetime.utcnow()

            appointments = [
                {
                    "platform": platform,
                    "specialty": specialty,
                    "available_time": (base_time + timedelta(hours=i)).isoformat(),
                    "provider_name": f"Dr. Provider {i}",
                    "provider_rating": 4.5 + (i % 5) * 0.1,
                    "estimated_wait": i * 5,  # minutes
                    "price": 50 + (i % 3) * 25,  # dollars
                    "insurance_accepted": i % 2 == 0,
                    "languages": ["English", "Spanish"] if i % 2 == 0 else ["English"]
                }
                for i in range(1, 6)  # Next 5 available slots
            ]

            return appointments

        except Exception as e:
            logger.error(f"Error getting {platform} availability", error=str(e))
            return []

    async def integrate_consultation_results(
        self,
        consultation_id: str,
        platform: str,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integrate telemedicine consultation results into patient record.

        Args:
            consultation_id: Consultation identifier
            platform: Telemedicine platform
            results: Consultation results and notes

        Returns:
            Integration status and updated patient data
        """
        try:
            logger.info("Integrating consultation results", consultation_id=consultation_id, platform=platform)

            # Parse consultation results
            clinical_notes = results.get("clinical_notes", "")
            diagnoses = results.get("diagnoses", [])
            medications = results.get("medications", [])
            follow_up = results.get("follow_up_instructions", "")

            # Create structured clinical summary
            clinical_summary = {
                "consultation_id": consultation_id,
                "platform": platform,
                "consultation_date": results.get("consultation_date", datetime.utcnow()),
                "provider_info": results.get("provider_info", {}),
                "chief_complaint": results.get("chief_complaint", ""),
                "assessment": diagnoses,
                "plan": {
                    "medications": medications,
                    "follow_up_instructions": follow_up,
                    "additional_tests": results.get("additional_tests", []),
                    "lifestyle_recommendations": results.get("lifestyle_recommendations", [])
                },
                "integrated_at": datetime.utcnow()
            }

            # Update patient record with consultation data
            integration_result = {
                "status": "integrated",
                "patient_updates": {
                    "new_diagnoses": diagnoses,
                    "new_medications": medications,
                    "follow_up_required": bool(follow_up)
                },
                "clinical_summary": clinical_summary,
                "next_steps": await self._generate_integration_next_steps(clinical_summary)
            }

            logger.info("Consultation results integrated", consultation_id=consultation_id)
            return integration_result

        except Exception as e:
            logger.error("Error integrating consultation results", error=str(e))
            return {"error": "Failed to integrate consultation results"}

    async def _generate_integration_next_steps(self, clinical_summary: Dict[str, Any]) -> List[str]:
        """Generate next steps after consultation integration."""

        next_steps = []

        # Check for new medications
        if clinical_summary.get("plan", {}).get("medications"):
            next_steps.append("Update medication list and check for interactions")

        # Check for follow-up requirements
        if clinical_summary.get("plan", {}).get("follow_up_instructions"):
            next_steps.append("Schedule follow-up consultation if required")

        # Check for additional tests
        if clinical_summary.get("plan", {}).get("additional_tests"):
            next_steps.append("Coordinate additional diagnostic tests")

        return next_steps


class FHIRTelemedicineIntegration:
    """
    FHIR-based telemedicine integration for healthcare systems.
    """

    def __init__(self):
        """Initialize FHIR telemedicine integration."""
        self.fhir_servers = {}

    async def schedule_fhir_consultation(
        self,
        patient_id: int,
        fhir_server: str,
        appointment_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Schedule consultation via FHIR Appointment resource.

        Args:
            patient_id: Patient identifier
            fhir_server: FHIR server URL
            appointment_request: Appointment details

        Returns:
            FHIR appointment creation result
        """
        try:
            # Create FHIR Appointment resource
            fhir_appointment = {
                "resourceType": "Appointment",
                "status": "proposed",
                "appointmentType": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0276",
                        "code": "WALKIN",
                        "display": "Walk In"
                    }]
                },
                "reasonCode": [{
                    "text": appointment_request.get("reason", "")
                }],
                "requestedPeriod": [{
                    "start": appointment_request.get("preferred_start"),
                    "end": appointment_request.get("preferred_end")
                }],
                "participant": [
                    {
                        "actor": {
                            "reference": f"Patient/{patient_id}",
                            "display": "Patient"
                        },
                        "status": "accepted"
                    }
                ]
            }

            # POST to FHIR server
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/fhir+json"}

                url = f"{fhir_server}/Appointment"
                async with session.post(url, json=fhir_appointment, headers=headers) as response:
                    if response.status in [200, 201]:
                        appointment_response = await response.json()

                        return {
                            "appointment_id": appointment_response.get("id"),
                            "status": appointment_response.get("status"),
                            "scheduled_time": appointment_response.get("start"),
                            "fhir_server": fhir_server,
                            "created_at": datetime.utcnow()
                        }
                    else:
                        return {"error": f"FHIR appointment creation failed: {response.status}"}

        except Exception as e:
            logger.error("Error scheduling FHIR consultation", error=str(e))
            return {"error": "Failed to schedule FHIR consultation"}

    async def get_fhir_appointment_history(self, patient_id: int, fhir_server: str) -> List[Dict[str, Any]]:
        """Get patient's telemedicine appointment history via FHIR."""

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/fhir+json"}

                # Query FHIR Appointment resources
                url = f"{fhir_server}/Appointment?patient={patient_id}"
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        bundle = await response.json()

                        appointments = []
                        for entry in bundle.get("entry", []):
                            appointment = entry.get("resource", {})

                            appointments.append({
                                "appointment_id": appointment.get("id"),
                                "status": appointment.get("status"),
                                "appointment_type": appointment.get("appointmentType", [{}])[0].get("display"),
                                "start_time": appointment.get("start"),
                                "end_time": appointment.get("end"),
                                "reason": appointment.get("reasonCode", [{}])[0].get("text"),
                                "location": appointment.get("location", {}).get("display")
                            })

                        return appointments

                    else:
                        return []

        except Exception as e:
            logger.error("Error getting FHIR appointment history", error=str(e))
            return []


class TelemedicineIntegrationService:
    """
    Comprehensive telemedicine integration service.
    """

    def __init__(self):
        """Initialize telemedicine integration service."""
        self.platform_integration = TelemedicinePlatform()
        self.fhir_integration = FHIRTelemedicineIntegration()

    async def find_telemedicine_options(
        self,
        patient_id: int,
        medical_need: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Find comprehensive telemedicine options for patient needs.

        Args:
            patient_id: Patient identifier
            medical_need: Description of medical need

        Returns:
            Available telemedicine options across platforms
        """
        try:
            logger.info("Finding telemedicine options", patient_id=patient_id)

            specialty = medical_need.get("specialty", "general")
            urgency = medical_need.get("urgency", "routine")

            # Get available appointments from all platforms
            available_appointments = await self.platform_integration.get_available_appointments(
                specialty=specialty,
                urgency=urgency,
                max_results=20
            )

            # Group by platform and time availability
            platform_options = {}
            time_slots = {"morning": [], "afternoon": [], "evening": []}

            for appointment in available_appointments:
                platform = appointment["platform"]

                if platform not in platform_options:
                    platform_options[platform] = {
                        "platform_name": appointment["platform"],
                        "available_slots": [],
                        "price_range": {"min": float('inf'), "max": 0},
                        "average_rating": 0,
                        "insurance_coverage": []
                    }

                platform_options[platform]["available_slots"].append(appointment)

                # Update price range
                price = appointment.get("price", 0)
                if price < platform_options[platform]["price_range"]["min"]:
                    platform_options[platform]["price_range"]["min"] = price
                if price > platform_options[platform]["price_range"]["max"]:
                    platform_options[platform]["price_range"]["max"] = price

                # Update insurance coverage
                if appointment.get("insurance_accepted"):
                    platform_options[platform]["insurance_coverage"].append("Accepted")

            # Calculate average ratings
            for platform in platform_options.values():
                slots = platform["available_slots"]
                if slots:
                    platform["average_rating"] = sum(slot.get("provider_rating", 0) for slot in slots) / len(slots)

            # Organize by time slots
            for appointment in available_appointments:
                appointment_time = datetime.fromisoformat(appointment["available_time"])
                hour = appointment_time.hour

                if 6 <= hour < 12:
                    time_slots["morning"].append(appointment)
                elif 12 <= hour < 18:
                    time_slots["afternoon"].append(appointment)
                else:
                    time_slots["evening"].append(appointment)

            return {
                "patient_id": patient_id,
                "medical_need": medical_need,
                "platform_options": list(platform_options.values()),
                "time_slots": time_slots,
                "total_options": len(available_appointments),
                "recommended_platform": await self._recommend_platform(available_appointments, medical_need),
                "search_timestamp": datetime.utcnow()
            }

        except Exception as e:
            logger.error("Error finding telemedicine options", error=str(e))
            return {"error": "Failed to find telemedicine options"}

    async def _recommend_platform(
        self,
        appointments: List[Dict[str, Any]],
        medical_need: Dict[str, Any]
    ) -> str:
        """Recommend best platform based on medical need."""

        if not appointments:
            return "teladoc"  # Default fallback

        # Simple recommendation logic
        urgency = medical_need.get("urgency", "routine")
        specialty = medical_need.get("specialty", "")

        # Recommend based on urgency
        if urgency == "urgent":
            return "amwell"  # Generally faster for urgent care

        # Recommend based on specialty
        specialty_platforms = {
            "mental_health": "amwell",
            "dermatology": "teladoc",
            "urgent_care": "doctor_on_demand"
        }

        return specialty_platforms.get(specialty, "teladoc")

    async def schedule_cross_platform_consultation(
        self,
        patient_id: int,
        consultation_request: Dict[str, Any],
        platform_preferences: List[str] = None
    ) -> Dict[str, Any]:
        """
        Schedule consultation across multiple platforms with fallback.

        Args:
            patient_id: Patient identifier
            consultation_request: Consultation details
            platform_preferences: Ordered list of preferred platforms

        Returns:
            Scheduled consultation or error with alternatives
        """
        try:
            if not platform_preferences:
                platform_preferences = ["teladoc", "amwell", "doctor_on_demand"]

            # Try each platform in order of preference
            for platform in platform_preferences:
                if platform in self.platform_integration.platform_configs:
                    result = await self.platform_integration.schedule_telemedicine_consultation(
                        patient_id=patient_id,
                        consultation_request=consultation_request,
                        preferred_platform=platform
                    )

                    if "error" not in result:
                        return result

            # If all platforms failed, return error with alternatives
            alternatives = await self.platform_integration.get_available_appointments(
                specialty=consultation_request.get("specialty", ""),
                urgency=consultation_request.get("urgency", "routine"),
                max_results=5
            )

            return {
                "error": "Unable to schedule with preferred platforms",
                "alternatives": alternatives,
                "suggested_next_steps": "Please select from available appointments above"
            }

        except Exception as e:
            logger.error("Error scheduling cross-platform consultation", error=str(e))
            return {"error": "Failed to schedule cross-platform consultation"}

    async def track_consultation_outcomes(
        self,
        consultation_id: str,
        outcome_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Track telemedicine consultation outcomes for analytics.

        Args:
            consultation_id: Consultation identifier
            outcome_data: Consultation outcome information

        Returns:
            Tracking status and analytics insights
        """
        try:
            # Store consultation outcome
            outcome_record = {
                "consultation_id": consultation_id,
                "outcome_type": outcome_data.get("outcome_type"),
                "patient_satisfaction": outcome_data.get("patient_satisfaction"),
                "clinical_effectiveness": outcome_data.get("clinical_effectiveness"),
                "follow_up_needed": outcome_data.get("follow_up_needed"),
                "recorded_at": datetime.utcnow()
            }

            # Generate analytics insights
            insights = await self._generate_outcome_insights(outcome_record)

            # Update platform performance metrics
            await self._update_platform_metrics(consultation_id, outcome_record)

            return {
                "consultation_id": consultation_id,
                "outcome_recorded": True,
                "insights": insights,
                "platform_performance_updated": True,
                "recorded_at": datetime.utcnow()
            }

        except Exception as e:
            logger.error("Error tracking consultation outcomes", error=str(e))
            return {"error": "Failed to track consultation outcomes"}

    async def _generate_outcome_insights(self, outcome_record: Dict[str, Any]) -> List[str]:
        """Generate insights from consultation outcomes."""

        insights = []

        satisfaction = outcome_record.get("patient_satisfaction", 0)
        if satisfaction > 4:
            insights.append("High patient satisfaction - platform performing well")
        elif satisfaction < 3:
            insights.append("Low patient satisfaction - review platform experience")

        if outcome_record.get("follow_up_needed"):
            insights.append("Follow-up consultation recommended for continued care")

        return insights

    async def _update_platform_metrics(self, consultation_id: str, outcome_record: Dict[str, Any]):
        """Update platform performance metrics."""

        # In production, update database with platform performance data
        logger.info("Platform metrics updated", consultation_id=consultation_id)


# Global telemedicine integration service
telemedicine_service = TelemedicineIntegrationService()
