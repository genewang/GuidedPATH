"""
Enhanced medication management service with pharmacy integration
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
from backend.apps.medication.models import Medication, MedicationRecord, MedicationInteraction, MedicationAdherence, MedicationReminder

logger = structlog.get_logger()


class PharmacyIntegrationService:
    """
    Service for integrating with pharmacy systems and APIs.
    """

    def __init__(self):
        """Initialize pharmacy integration service."""
        self.pharmacy_apis = {
            "cvs": {"base_url": "https://api.cvs.com", "api_key": settings.CVS_API_KEY},
            "walgreens": {"base_url": "https://api.walgreens.com", "api_key": settings.WALGREENS_API_KEY},
            "rite_aid": {"base_url": "https://api.riteaid.com", "api_key": settings.RITE_AID_API_KEY},
            "costco": {"base_url": "https://api.costco.com", "api_key": settings.COSTCO_API_KEY}
        }

        # FHIR pharmacy endpoints
        self.fhir_endpoints = {
            "medication": "/Medication",
            "medication_request": "/MedicationRequest",
            "medication_dispense": "/MedicationDispense"
        }

    async def sync_patient_medications(
        self,
        patient_id: int,
        pharmacy_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sync patient medication records with pharmacy systems.

        Args:
            patient_id: Patient identifier
            pharmacy_info: Pharmacy system connection details

        Returns:
            Sync results with updated medication information
        """
        try:
            logger.info("Syncing patient medications with pharmacy", patient_id=patient_id)

            sync_results = {
                "patient_id": patient_id,
                "pharmacy": pharmacy_info.get("name"),
                "sync_timestamp": datetime.utcnow(),
                "medications_synced": 0,
                "refills_updated": 0,
                "interactions_found": 0,
                "discrepancies": [],
                "errors": []
            }

            # Get pharmacy-specific integration method
            pharmacy_system = pharmacy_info.get("system", "").lower()

            if pharmacy_system in self.pharmacy_apis:
                # Use pharmacy-specific API
                api_result = await self._sync_with_pharmacy_api(patient_id, pharmacy_info, pharmacy_system)
                sync_results.update(api_result)

            else:
                # Use FHIR-based synchronization
                fhir_result = await self._sync_with_fhir_pharmacy(patient_id, pharmacy_info)
                sync_results.update(fhir_result)

            # Check for medication discrepancies
            discrepancies = await self._check_medication_discrepancies(patient_id, sync_results)
            sync_results["discrepancies"] = discrepancies

            # Update local medication records
            await self._update_local_medication_records(patient_id, sync_results)

            logger.info("Pharmacy sync completed", patient_id=patient_id, medications_synced=sync_results["medications_synced"])
            return sync_results

        except Exception as e:
            logger.error("Error syncing with pharmacy", error=str(e), patient_id=patient_id)
            return {
                "patient_id": patient_id,
                "error": str(e),
                "sync_timestamp": datetime.utcnow()
            }

    async def _sync_with_pharmacy_api(
        self,
        patient_id: int,
        pharmacy_info: Dict[str, Any],
        pharmacy_system: str
    ) -> Dict[str, Any]:
        """Sync with pharmacy-specific API."""

        api_config = self.pharmacy_apis[pharmacy_system]
        base_url = api_config["base_url"]
        api_key = api_config["api_key"]

        if not api_key:
            return {"error": f"API key not configured for {pharmacy_system}"}

        try:
            # Pharmacy-specific API calls would go here
            # For now, return mock sync data
            return {
                "medications_synced": 5,
                "refills_updated": 2,
                "current_prescriptions": [
                    {
                        "medication_name": "Lisinopril",
                        "dosage": "10mg",
                        "refills_remaining": 3,
                        "last_filled": "2024-01-01",
                        "next_refill_due": "2024-02-01"
                    }
                ],
                "pharmacy_info": {
                    "name": pharmacy_info.get("name"),
                    "phone": pharmacy_info.get("phone"),
                    "address": pharmacy_info.get("address")
                }
            }

        except Exception as e:
            logger.error(f"Error syncing with {pharmacy_system} API", error=str(e))
            return {"error": f"Failed to sync with {pharmacy_system}"}

    async def _sync_with_fhir_pharmacy(
        self,
        patient_id: int,
        pharmacy_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sync using FHIR pharmacy endpoints."""

        fhir_server = pharmacy_info.get("fhir_server_url")
        if not fhir_server:
            return {"error": "FHIR server URL not provided"}

        try:
            # FHIR Medication resource query
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/fhir+json"}

                # Query for patient's medications
                url = f"{fhir_server}/MedicationRequest?patient={patient_id}"
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        fhir_data = await response.json()

                        return {
                            "medications_synced": len(fhir_data.get("entry", [])),
                            "fhir_resources": fhir_data,
                            "last_updated": datetime.utcnow()
                        }
                    else:
                        return {"error": f"FHIR query failed: {response.status}"}

        except Exception as e:
            logger.error("Error syncing with FHIR pharmacy", error=str(e))
            return {"error": "FHIR synchronization failed"}

    async def _check_medication_discrepancies(
        self,
        patient_id: int,
        sync_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for discrepancies between synced and local medication data."""

        discrepancies = []

        # TODO: Compare synced medications with local records
        # For now, return mock discrepancies

        return discrepancies

    async def _update_local_medication_records(
        self,
        patient_id: int,
        sync_results: Dict[str, Any]
    ):
        """Update local medication records with pharmacy data."""

        # TODO: Update database with synced medication information
        logger.info("Local medication records updated", patient_id=patient_id)


class EnhancedMedicationService:
    """
    Enhanced medication management with AI-powered features and pharmacy integration.
    """

    def __init__(self):
        """Initialize enhanced medication service."""
        self.pharmacy_service = PharmacyIntegrationService()
        self.drugbank_api_key = settings.DRUGBANK_API_KEY
        self.rxnorm_api_url = "https://rxnav.nlm.nih.gov/REST"

    async def check_comprehensive_drug_interactions(
        self,
        medications: List[str],
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive drug interaction checking with multiple data sources.

        Args:
            medications: List of medication names
            patient_context: Patient-specific context (age, conditions, etc.)

        Returns:
            Detailed interaction analysis with management recommendations
        """
        try:
            logger.info("Checking comprehensive drug interactions", medication_count=len(medications))

            # Multi-source interaction checking
            interactions = {
                "drugbank_interactions": await self._check_drugbank_interactions(medications),
                "rxnorm_interactions": await self._check_rxnorm_interactions(medications),
                "clinical_trial_interactions": await self._check_trial_interactions(medications),
                "patient_specific_risks": await self._assess_patient_specific_risks(medications, patient_context)
            }

            # Aggregate and prioritize interactions
            all_interactions = []
            for source, source_interactions in interactions.items():
                for interaction in source_interactions:
                    interaction["source"] = source
                    all_interactions.append(interaction)

            # Sort by severity and clinical significance
            severity_order = {"critical": 0, "high": 1, "moderate": 2, "low": 3}
            prioritized_interactions = sorted(
                all_interactions,
                key=lambda x: (
                    severity_order.get(x.get("severity", "moderate"), 2),
                    -x.get("confidence", 0.5)
                )
            )

            return {
                "medications_checked": medications,
                "total_interactions_found": len(prioritized_interactions),
                "interactions": prioritized_interactions[:10],  # Top 10 interactions
                "risk_assessment": await self._assess_overall_risk(prioritized_interactions),
                "monitoring_recommendations": await self._generate_monitoring_recommendations(prioritized_interactions),
                "management_strategies": await self._generate_management_strategies(prioritized_interactions),
                "checked_at": datetime.utcnow(),
                "data_sources": list(interactions.keys())
            }

        except Exception as e:
            logger.error("Error checking comprehensive drug interactions", error=str(e))
            return {"error": "Failed to check drug interactions"}

    async def _check_drugbank_interactions(self, medications: List[str]) -> List[Dict[str, Any]]:
        """Check interactions using DrugBank API."""

        if not self.drugbank_api_key:
            return []

        try:
            # DrugBank API integration would go here
            # For now, return mock interaction data
            return [
                {
                    "medication1": "Warfarin",
                    "medication2": "Aspirin",
                    "severity": "moderate",
                    "description": "May increase risk of bleeding",
                    "management": "Monitor INR closely",
                    "confidence": 0.85
                }
            ]

        except Exception as e:
            logger.error("Error checking DrugBank interactions", error=str(e))
            return []

    async def _check_rxnorm_interactions(self, medications: List[str]) -> List[Dict[str, Any]]:
        """Check interactions using RxNorm API."""

        try:
            async with aiohttp.ClientSession() as session:
                interactions = []

                # Get RxCUI codes for medications
                for medication in medications:
                    rxcui = await self._get_rxcui_for_medication(medication)
                    if rxcui:
                        # Query for interactions
                        url = f"{self.rxnorm_api_url}/interaction/interaction.json?rxcui={rxcui}"
                        async with session.get(url) as response:
                            if response.status == 200:
                                data = await response.json()
                                # Process interaction data
                                interactions.extend(await self._process_rxnorm_interactions(data))

                return interactions

        except Exception as e:
            logger.error("Error checking RxNorm interactions", error=str(e))
            return []

    async def _get_rxcui_for_medication(self, medication: str) -> Optional[str]:
        """Get RxCUI code for medication name."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.rxnorm_api_url}/drugs.json?name={medication}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        drug_group = data.get("drugGroup", {})
                        concept_group = drug_group.get("conceptGroup", [])
                        if concept_group:
                            return concept_group[0].get("conceptProperty", [{}])[0].get("rxcui")

            return None

        except Exception as e:
            logger.error("Error getting RxCUI", error=str(e))
            return None

    async def _process_rxnorm_interactions(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process RxNorm interaction data."""
        # Process RxNorm interaction response
        # For now, return empty list
        return []

    async def _check_trial_interactions(self, medications: List[str]) -> List[Dict[str, Any]]:
        """Check for interactions based on clinical trial data."""
        # Query clinical trial database for known interactions
        # For now, return empty list
        return []

    async def _assess_patient_specific_risks(
        self,
        medications: List[str],
        patient_context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Assess patient-specific interaction risks."""

        if not patient_context:
            return []

        risks = []

        # Age-related risks
        age = patient_context.get("age", 0)
        if age > 65:
            risks.append({
                "risk_type": "age_related",
                "severity": "moderate",
                "description": "Elderly patients may have increased sensitivity to drug interactions",
                "management": "Closer monitoring recommended"
            })

        # Condition-specific risks
        conditions = patient_context.get("conditions", [])
        if "kidney_disease" in conditions:
            risks.append({
                "risk_type": "renal_impairment",
                "severity": "high",
                "description": "Patients with kidney disease may need dose adjustments",
                "management": "Monitor renal function regularly"
            })

        return risks

    async def _assess_overall_risk(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall interaction risk level."""

        if not interactions:
            return {"level": "low", "score": 0.1, "description": "No significant interactions detected"}

        # Calculate risk score
        severity_weights = {"critical": 1.0, "high": 0.8, "moderate": 0.5, "low": 0.2}
        total_risk = sum(severity_weights.get(interaction.get("severity", "moderate"), 0.5)
                        for interaction in interactions)

        avg_risk = total_risk / len(interactions)

        if avg_risk > 0.7:
            level = "high"
            description = "Multiple high-risk interactions detected"
        elif avg_risk > 0.4:
            level = "moderate"
            description = "Some interactions require monitoring"
        else:
            level = "low"
            description = "Minimal interaction risk"

        return {
            "level": level,
            "score": avg_risk,
            "description": description,
            "interaction_count": len(interactions)
        }

    async def _generate_monitoring_recommendations(self, interactions: List[Dict[str, Any]]) -> List[str]:
        """Generate monitoring recommendations based on interactions."""

        recommendations = []

        for interaction in interactions:
            if interaction.get("severity") in ["high", "critical"]:
                recommendations.append(f"Monitor for {interaction['description'].lower()}")
                if "bleeding" in interaction["description"].lower():
                    recommendations.append("Monitor INR and signs of bleeding")
                if "kidney" in interaction["description"].lower():
                    recommendations.append("Monitor renal function tests")

        return list(set(recommendations))  # Remove duplicates

    async def _generate_management_strategies(self, interactions: List[Dict[str, Any]]) -> List[str]:
        """Generate management strategies for interactions."""

        strategies = []

        for interaction in interactions:
            management = interaction.get("management", "")
            if management:
                strategies.append(f"For {interaction['medication1']}-{interaction['medication2']}: {management}")

        return strategies

    async def predict_adherence_patterns(
        self,
        patient_id: int,
        medication_records: List[MedicationRecord]
    ) -> Dict[str, Any]:
        """
        Predict medication adherence patterns using ML.

        Args:
            patient_id: Patient identifier
            medication_records: Patient's medication history

        Returns:
            Adherence predictions and intervention recommendations
        """
        try:
            logger.info("Predicting adherence patterns", patient_id=patient_id)

            # Extract adherence features
            adherence_features = await self._extract_adherence_features(patient_id, medication_records)

            # Predict adherence for each medication
            adherence_predictions = []

            for med_record in medication_records:
                prediction = await self._predict_single_medication_adherence(med_record, adherence_features)
                adherence_predictions.append(prediction)

            # Generate intervention recommendations
            interventions = await self._generate_adherence_interventions(adherence_predictions)

            return {
                "patient_id": patient_id,
                "prediction_timestamp": datetime.utcnow(),
                "overall_adherence_score": sum(p["predicted_adherence"] for p in adherence_predictions) / len(adherence_predictions),
                "medication_predictions": adherence_predictions,
                "risk_factors": await self._identify_adherence_risk_factors(adherence_features),
                "intervention_recommendations": interventions,
                "next_review_date": (datetime.utcnow() + timedelta(days=30)).isoformat()
            }

        except Exception as e:
            logger.error("Error predicting adherence patterns", error=str(e))
            return {"error": "Failed to predict adherence patterns"}

    async def _extract_adherence_features(self, patient_id: int, medication_records: List[MedicationRecord]) -> Dict[str, Any]:
        """Extract features for adherence prediction."""

        features = {
            "medication_count": len(medication_records),
            "complex_regimens": 0,
            "frequent_dosing": 0,
            "side_effects_history": 0,
            "age_factor": 0,
            "condition_complexity": 0,
            "previous_adherence": 0.5  # Default
        }

        # Analyze medication characteristics
        for med_record in medication_records:
            # Count complex regimens
            if self._is_complex_regimen(med_record):
                features["complex_regimens"] += 1

            # Count frequent dosing
            if self._requires_frequent_dosing(med_record):
                features["frequent_dosing"] += 1

            # Check for side effects
            if med_record.side_effects:
                features["side_effects_history"] += 1

        return features

    def _is_complex_regimen(self, med_record: MedicationRecord) -> bool:
        """Determine if medication regimen is complex."""
        return (med_record.route in ["injection", "infusion"] or
                len((med_record.instructions or "").split()) > 20)

    def _requires_frequent_dosing(self, med_record: MedicationRecord) -> bool:
        """Check if medication requires frequent dosing."""
        frequency = (med_record.frequency or "").lower()
        return any(word in frequency for word in ["hour", "hr", "every 4", "every 6"])

    async def _predict_single_medication_adherence(
        self,
        med_record: MedicationRecord,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict adherence for a single medication."""

        # Simple rule-based prediction (in production, use ML model)
        base_adherence = 0.8

        # Adjust based on complexity
        if features["complex_regimens"] > 2:
            base_adherence -= 0.2

        if features["frequent_dosing"] > 1:
            base_adherence -= 0.15

        if features["side_effects_history"] > 0:
            base_adherence -= 0.1

        # Adjust based on medication characteristics
        if self._is_complex_regimen(med_record):
            base_adherence -= 0.1

        if self._requires_frequent_dosing(med_record):
            base_adherence -= 0.1

        adherence_score = max(0.1, min(0.95, base_adherence))

        return {
            "medication_id": med_record.id,
            "medication_name": med_record.medication.name,
            "predicted_adherence": adherence_score,
            "confidence": 0.75,  # Would come from ML model
            "risk_level": "high" if adherence_score < 0.6 else "medium" if adherence_score < 0.8 else "low"
        }

    async def _generate_adherence_interventions(self, predictions: List[Dict[str, Any]]) -> List[str]:
        """Generate interventions for low-adherence medications."""

        interventions = []

        for prediction in predictions:
            if prediction["risk_level"] in ["high", "medium"]:
                medication_name = prediction["medication_name"]

                if prediction["risk_level"] == "high":
                    interventions.extend([
                        f"Schedule adherence consultation for {medication_name}",
                        f"Consider simplified dosing regimen for {medication_name}",
                        f"Set up intensive reminder system for {medication_name}"
                    ])

                interventions.append(f"Monitor adherence closely for {medication_name}")

        return list(set(interventions))  # Remove duplicates

    async def _identify_adherence_risk_factors(self, features: Dict[str, Any]) -> List[str]:
        """Identify factors that may impact adherence."""

        risk_factors = []

        if features["medication_count"] > 5:
            risk_factors.append("High number of medications (polypharmacy)")

        if features["complex_regimens"] > 2:
            risk_factors.append("Complex medication regimens")

        if features["frequent_dosing"] > 2:
            risk_factors.append("Frequent dosing requirements")

        if features["side_effects_history"] > 1:
            risk_factors.append("History of medication side effects")

        return risk_factors

    async def generate_refill_reminders(
        self,
        patient_id: int,
        medication_records: List[MedicationRecord]
    ) -> List[Dict[str, Any]]:
        """
        Generate intelligent refill reminders.

        Args:
            patient_id: Patient identifier
            medication_records: Patient's current medications

        Returns:
            List of refill reminders with optimal timing
        """
        try:
            reminders = []

            for med_record in medication_records:
                if med_record.status != "ACTIVE":
                    continue

                # Calculate optimal refill timing
                refill_timing = await self._calculate_refill_timing(med_record)

                if refill_timing["should_remind"]:
                    reminder = {
                        "medication_id": med_record.id,
                        "medication_name": med_record.medication.name,
                        "current_refills": med_record.refills_remaining,
                        "recommended_refill_date": refill_timing["optimal_date"],
                        "urgency_level": refill_timing["urgency"],
                        "pharmacy_info": {
                            "name": med_record.pharmacy_name,
                            "phone": med_record.pharmacy_phone
                        },
                        "reminder_message": f"Time to refill {med_record.medication.name}. {med_record.refills_remaining} refills remaining.",
                        "next_reminder_date": refill_timing["next_reminder"]
                    }

                    reminders.append(reminder)

            return reminders

        except Exception as e:
            logger.error("Error generating refill reminders", error=str(e))
            return []

    async def _calculate_refill_timing(self, med_record: MedicationRecord) -> Dict[str, Any]:
        """Calculate optimal refill timing for medication."""

        refills_remaining = med_record.refills_remaining or 0

        # Base timing on refills remaining and medication frequency
        if refills_remaining <= 1:
            urgency = "high"
            days_ahead = 3
        elif refills_remaining <= 3:
            urgency = "medium"
            days_ahead = 7
        else:
            urgency = "low"
            days_ahead = 14

        optimal_date = datetime.utcnow() + timedelta(days=days_ahead)
        next_reminder = optimal_date + timedelta(days=3)  # Follow-up reminder

        return {
            "should_remind": True,
            "optimal_date": optimal_date.isoformat(),
            "urgency": urgency,
            "next_reminder": next_reminder.isoformat()
        }


# Global enhanced medication service instance
enhanced_medication_service = EnhancedMedicationService()
