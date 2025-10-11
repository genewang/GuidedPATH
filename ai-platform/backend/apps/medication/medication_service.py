"""
Medication management service with AI-powered features
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json

import httpx
from sentence_transformers import SentenceTransformer
import structlog

from backend.core.config import settings
from backend.apps.medication.models import Medication, MedicationRecord, MedicationInteraction, MedicationAdherence

logger = structlog.get_logger()


class MedicationService:
    """
    AI-powered medication management service.
    """

    def __init__(self):
        """Initialize the medication service."""
        self.embedding_model = SentenceTransformer(settings.DEFAULT_EMBEDDING_MODEL)
        self.drugbank_api_key = settings.DRUGBANK_API_KEY
        self.fhir_server_url = settings.FHIR_SERVER_URL

    async def check_drug_interactions(self, medication_names: List[str]) -> List[Dict[str, Any]]:
        """
        Check for drug interactions between medications.

        Args:
            medication_names: List of medication names to check

        Returns:
            List of potential interactions with severity and recommendations
        """
        try:
            logger.info("Checking drug interactions", medications=medication_names)

            # Get medication records for semantic matching
            medications = await self._find_medications_by_name(medication_names)

            if len(medications) < 2:
                return []

            interactions = []

            # Check each pair of medications
            for i, med1 in enumerate(medications):
                for med2 in medications[i+1:]:
                    interaction = await self._check_interaction_pair(med1, med2)
                    if interaction:
                        interactions.append(interaction)

            # Sort by severity
            severity_order = {"critical": 0, "high": 1, "moderate": 2, "low": 3}
            interactions.sort(key=lambda x: severity_order.get(x["severity"], 4))

            logger.info("Drug interaction check completed", interactions_found=len(interactions))
            return interactions

        except Exception as e:
            logger.error("Error checking drug interactions", error=str(e))
            return []

    async def _find_medications_by_name(self, medication_names: List[str]) -> List[Medication]:
        """Find medication records by name using semantic similarity."""
        # In production, query database and use embeddings for fuzzy matching
        # For now, return mock medications
        return []

    async def _check_interaction_pair(self, med1: Medication, med2: Medication) -> Optional[Dict[str, Any]]:
        """Check interaction between two specific medications."""
        # Query DrugBank API or use cached interaction data
        if self.drugbank_api_key:
            return await self._query_drugbank_api(med1, med2)
        else:
            return await self._check_local_interactions(med1, med2)

    async def _query_drugbank_api(self, med1: Medication, med2: Medication) -> Optional[Dict[str, Any]]:
        """Query DrugBank API for interaction information."""
        try:
            # This would use the actual DrugBank API
            # For now, return mock interaction data
            return {
                "medication1": med1.name,
                "medication2": med2.name,
                "severity": "moderate",
                "description": "May increase risk of side effects when taken together",
                "management": "Monitor for increased side effects",
                "contraindication": False
            }
        except Exception as e:
            logger.error("Error querying DrugBank API", error=str(e))
            return None

    async def _check_local_interactions(self, med1: Medication, med2: Medication) -> Optional[Dict[str, Any]]:
        """Check interactions using local knowledge base."""
        # Use rule-based system or ML model for interaction detection
        # For now, return None (no interaction found)
        return None

    async def generate_medication_schedule(self, patient_id: int, medications: List[MedicationRecord]) -> Dict[str, Any]:
        """
        Generate optimal medication schedule based on patient needs and drug properties.

        Args:
            patient_id: Patient identifier
            medications: List of patient's current medications

        Returns:
            Optimized schedule with timing recommendations
        """
        try:
            logger.info("Generating medication schedule", patient_id=patient_id, medication_count=len(medications))

            schedule = {
                "patient_id": patient_id,
                "generated_at": datetime.utcnow(),
                "schedule_by_time": {},
                "conflicts": [],
                "recommendations": []
            }

            # Group medications by administration time
            morning_meds = []
            afternoon_meds = []
            evening_meds = []
            bedtime_meds = []

            for med_record in medications:
                if med_record.status != "ACTIVE":
                    continue

                # Determine optimal timing based on drug properties and patient factors
                optimal_time = await self._determine_optimal_timing(med_record)

                if optimal_time == "morning":
                    morning_meds.append(med_record)
                elif optimal_time == "afternoon":
                    afternoon_meds.append(med_record)
                elif optimal_time == "evening":
                    evening_meds.append(med_record)
                elif optimal_time == "bedtime":
                    bedtime_meds.append(med_record)

            # Check for timing conflicts
            conflicts = await self._check_schedule_conflicts(morning_meds + afternoon_meds + evening_meds + bedtime_meds)

            # Generate final schedule
            schedule["schedule_by_time"] = {
                "morning": [self._format_medication_info(med) for med in morning_meds],
                "afternoon": [self._format_medication_info(med) for med in afternoon_meds],
                "evening": [self._format_medication_info(med) for med in evening_meds],
                "bedtime": [self._format_medication_info(med) for med in bedtime_meds]
            }

            schedule["conflicts"] = conflicts
            schedule["recommendations"] = await self._generate_schedule_recommendations(conflicts)

            logger.info("Medication schedule generated", patient_id=patient_id)
            return schedule

        except Exception as e:
            logger.error("Error generating medication schedule", error=str(e), patient_id=patient_id)
            return {"error": "Failed to generate schedule"}

    async def _determine_optimal_timing(self, med_record: MedicationRecord) -> str:
        """Determine optimal administration time for a medication."""
        # AI logic to determine best timing based on:
        # - Drug pharmacokinetics
        # - Food interactions
        # - Patient's sleep schedule
        # - Other medications

        # Simplified logic - in production, use ML model
        medication_name = med_record.medication.name.lower()

        if any(word in medication_name for word in ["prednisone", "methylprednisolone", "dexamethasone"]):
            return "morning"  # Steroids best taken in morning
        elif any(word in medication_name for word in ["sleep", "insomnia", "benadryl"]):
            return "bedtime"
        else:
            return "evening"  # Default to evening

    async def _check_schedule_conflicts(self, medications: List[MedicationRecord]) -> List[Dict[str, Any]]:
        """Check for potential conflicts in medication schedule."""
        conflicts = []

        # Check for timing conflicts
        medication_times = {}
        for med in medications:
            timing = await self._determine_optimal_timing(med)
            if timing in medication_times:
                conflicts.append({
                    "type": "timing_conflict",
                    "medications": [medication_times[timing], med.medication.name],
                    "recommendation": f"Consider spacing {timing} medications"
                })
            medication_times[timing] = med.medication.name

        # Check for food interactions
        for med in medications:
            if await self._has_food_interaction(med):
                conflicts.append({
                    "type": "food_interaction",
                    "medication": med.medication.name,
                    "recommendation": "Take on empty stomach or as directed"
                })

        return conflicts

    async def _has_food_interaction(self, med_record: MedicationRecord) -> bool:
        """Check if medication has food interactions."""
        # Query drug database for food interactions
        # For now, return False
        return False

    def _format_medication_info(self, med_record: MedicationRecord) -> Dict[str, Any]:
        """Format medication record for schedule display."""
        return {
            "id": med_record.id,
            "name": med_record.medication.name,
            "dosage": med_record.dosage,
            "instructions": med_record.instructions,
            "route": med_record.route,
            "frequency": med_record.frequency
        }

    async def _generate_schedule_recommendations(self, conflicts: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on schedule conflicts."""
        recommendations = []

        if conflicts:
            recommendations.append("Schedule reviewed for potential interactions")
            recommendations.append("Consider discussing timing with your healthcare provider")

        recommendations.append("Set up reminders for better adherence")
        recommendations.append("Report any side effects to your healthcare provider")

        return recommendations

    async def predict_adherence_risk(self, patient_id: int, medication_records: List[MedicationRecord]) -> Dict[str, Any]:
        """
        Predict risk of medication non-adherence using AI.

        Args:
            patient_id: Patient identifier
            medication_records: Patient's medication records

        Returns:
            Adherence risk assessment and recommendations
        """
        try:
            logger.info("Predicting adherence risk", patient_id=patient_id)

            risk_factors = []

            # Analyze medication complexity
            complex_meds = [med for med in medication_records if self._is_complex_medication(med)]
            if len(complex_meds) > 3:
                risk_factors.append("Multiple complex medications increase adherence risk")

            # Check dosing frequency
            frequent_meds = [med for med in medication_records if self._requires_frequent_dosing(med)]
            if frequent_meds:
                risk_factors.append("Frequent dosing schedule may impact adherence")

            # Check for side effects history
            side_effect_meds = [med for med in medication_records if med.side_effects]
            if side_effect_meds:
                risk_factors.append("History of side effects may reduce adherence")

            # Calculate overall risk score
            risk_score = min(1.0, len(risk_factors) * 0.2)  # Simple scoring

            risk_level = "low" if risk_score < 0.3 else "moderate" if risk_score < 0.7 else "high"

            return {
                "patient_id": patient_id,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "recommendations": await self._generate_adherence_recommendations(risk_factors),
                "predicted_at": datetime.utcnow()
            }

        except Exception as e:
            logger.error("Error predicting adherence risk", error=str(e), patient_id=patient_id)
            return {"error": "Failed to predict adherence risk"}

    def _is_complex_medication(self, med_record: MedicationRecord) -> bool:
        """Determine if medication regimen is complex."""
        # Complex if: multiple daily doses, special administration, or monitoring required
        return "injection" in (med_record.route or "").lower() or "infusion" in (med_record.route or "").lower()

    def _requires_frequent_dosing(self, med_record: MedicationRecord) -> bool:
        """Check if medication requires frequent dosing."""
        frequency = (med_record.frequency or "").lower()
        return any(word in frequency for word in ["hour", "hr", "every 4", "every 6"])

    async def _generate_adherence_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Generate recommendations for improving adherence."""
        recommendations = [
            "Use pill organizers or medication reminder apps",
            "Set up automatic refills with your pharmacy",
            "Discuss medication concerns with your healthcare provider"
        ]

        if any("complex" in factor for factor in risk_factors):
            recommendations.append("Consider simplified dosing regimens where possible")

        if any("frequent" in factor for factor in risk_factors):
            recommendations.append("Use medication timing aids and reminders")

        return recommendations

    async def sync_with_pharmacy(self, patient_id: int, pharmacy_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sync medication information with patient's pharmacy.

        Args:
            patient_id: Patient identifier
            pharmacy_info: Pharmacy contact and system information

        Returns:
            Sync status and any issues found
        """
        try:
            logger.info("Syncing with pharmacy", patient_id=patient_id)

            # This would integrate with pharmacy APIs (CVS, Walgreens, etc.)
            # For now, return mock sync status

            return {
                "patient_id": patient_id,
                "sync_status": "success",
                "last_sync": datetime.utcnow(),
                "medications_synced": 5,
                "refills_available": 3,
                "issues": []
            }

        except Exception as e:
            logger.error("Error syncing with pharmacy", error=str(e), patient_id=patient_id)
            return {
                "patient_id": patient_id,
                "sync_status": "error",
                "error": str(e)
            }


# Global medication service instance
medication_service = MedicationService()
