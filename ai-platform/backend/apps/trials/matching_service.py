"""
AI-powered clinical trial matching service
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import structlog

from backend.core.config import settings
from backend.apps.trials.models import ClinicalTrial, TrialMatch, TrialSearchQuery

logger = structlog.get_logger()


class TrialMatchingService:
    """
    AI-powered service for matching patients with clinical trials.
    """

    def __init__(self):
        """Initialize the trial matching service."""
        self.embedding_model = SentenceTransformer(settings.DEFAULT_EMBEDDING_MODEL)
        self.trials_cache = {}  # In production, use Redis
        self.cache_timestamp = None

    def _extract_patient_profile(self, patient_record) -> Dict[str, Any]:
        """
        Extract relevant information from patient record for trial matching.

        Args:
            patient_record: Patient record from database

        Returns:
            Dict containing processed patient information
        """
        return {
            "conditions": patient_record.conditions or [],
            "age": patient_record.age,
            "gender": patient_record.gender,
            "performance_status": patient_record.performance_status,
            "current_treatments": patient_record.current_treatments or [],
            "treatment_history": patient_record.treatment_history or {},
            "biomarkers": patient_record.biomarkers or {},
            "comorbidities": patient_record.comorbidities or [],
            "location": patient_record.location,
            "willing_to_travel": patient_record.willing_to_travel or False
        }

    def _calculate_condition_similarity(self, patient_conditions: List[str], trial_conditions: List[str]) -> float:
        """
        Calculate semantic similarity between patient conditions and trial conditions.

        Args:
            patient_conditions: List of patient medical conditions
            trial_conditions: List of trial target conditions

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not patient_conditions or not trial_conditions:
            return 0.0

        # Create embeddings for conditions
        patient_text = " ".join(patient_conditions)
        trial_text = " ".join(trial_conditions)

        patient_embedding = self.embedding_model.encode([patient_text])
        trial_embedding = self.embedding_model.encode([trial_text])

        # Calculate cosine similarity
        similarity = cosine_similarity(patient_embedding, trial_embedding)[0][0]

        return float(max(0.0, similarity))  # Ensure non-negative

    def _check_eligibility_criteria(self, patient_profile: Dict, trial: ClinicalTrial) -> Tuple[float, List[str]]:
        """
        Check patient eligibility against trial criteria.

        Args:
            patient_profile: Processed patient information
            trial: Clinical trial object

        Returns:
            Tuple of (eligibility_score, list_of_unmet_criteria)
        """
        unmet_criteria = []
        met_criteria = 0
        total_criteria = 0

        if not trial.eligibility_criteria:
            return 1.0, []

        # Parse eligibility criteria (simplified - in production, use NLP)
        criteria_text = trial.eligibility_criteria.lower()

        # Basic eligibility checks (this should be much more sophisticated in production)
        basic_checks = [
            ("age", self._check_age_eligibility),
            ("performance_status", self._check_performance_status),
            ("treatment_history", self._check_treatment_history),
            ("biomarkers", self._check_biomarker_eligibility)
        ]

        for check_name, check_func in basic_checks:
            total_criteria += 1
            if check_func(patient_profile, trial):
                met_criteria += 1
            else:
                unmet_criteria.append(f"Does not meet {check_name} criteria")

        # Location eligibility
        total_criteria += 1
        if self._check_location_eligibility(patient_profile, trial):
            met_criteria += 1
        else:
            unmet_criteria.append("Location/travel requirements not met")

        eligibility_score = met_criteria / total_criteria if total_criteria > 0 else 0.0

        return eligibility_score, unmet_criteria

    def _check_age_eligibility(self, patient_profile: Dict, trial: ClinicalTrial) -> bool:
        """Check if patient meets age criteria."""
        # Simplified age check - in production, parse actual age ranges from criteria
        return True  # Placeholder

    def _check_performance_status(self, patient_profile: Dict, trial: ClinicalTrial) -> bool:
        """Check if patient meets performance status criteria."""
        # Simplified performance status check
        return True  # Placeholder

    def _check_treatment_history(self, patient_profile: Dict, trial: ClinicalTrial) -> bool:
        """Check if patient's treatment history is compatible."""
        # Check for exclusion criteria based on prior treatments
        return True  # Placeholder

    def _check_biomarker_eligibility(self, patient_profile: Dict, trial: ClinicalTrial) -> bool:
        """Check if patient meets biomarker requirements."""
        # Check for required biomarkers or genetic markers
        return True  # Placeholder

    def _check_location_eligibility(self, patient_profile: Dict, trial: ClinicalTrial) -> bool:
        """Check if trial location is accessible to patient."""
        patient_location = patient_profile.get("location", "")
        trial_locations = trial.locations or []

        if not trial_locations:
            return True  # No location restriction

        # Check if patient is willing to travel
        if patient_profile.get("willing_to_travel", False):
            return True

        # Check if any trial location is near patient
        # Simplified distance check - in production, use geocoding
        for location in trial_locations:
            if self._calculate_distance(patient_location, location) <= 100:  # 100 mile radius
                return True

        return False

    def _calculate_distance(self, location1: str, location2: str) -> float:
        """Calculate approximate distance between locations."""
        # Simplified distance calculation - in production, use proper geocoding
        return 50.0  # Placeholder distance

    def _calculate_phase_appropriateness(self, patient_profile: Dict, trial: ClinicalTrial) -> float:
        """
        Calculate how appropriate the trial phase is for the patient.

        Args:
            patient_profile: Patient information
            trial: Clinical trial

        Returns:
            Appropriateness score between 0.0 and 1.0
        """
        # Phase appropriateness based on treatment history and condition severity
        treatment_lines = len(patient_profile.get("treatment_history", {}).get("lines", []))

        phase_scores = {
            "phase_1": 0.3 if treatment_lines >= 2 else 0.8,
            "phase_1_2": 0.5 if treatment_lines >= 1 else 0.7,
            "phase_2": 0.7 if treatment_lines >= 1 else 0.5,
            "phase_2_3": 0.8 if treatment_lines >= 2 else 0.4,
            "phase_3": 0.9 if treatment_lines >= 2 else 0.3,
            "phase_4": 1.0 if treatment_lines >= 3 else 0.2
        }

        trial_phase = trial.phase.value
        return phase_scores.get(trial_phase, 0.5)

    async def find_matching_trials(
        self,
        patient_record,
        search_query: Optional[str] = None,
        max_results: int = 10,
        minimum_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Find clinical trials that match patient profile.

        Args:
            patient_record: Patient record from database
            search_query: Optional text search query
            max_results: Maximum number of results to return
            minimum_score: Minimum matching score threshold

        Returns:
            List of matching trials with scores and explanations
        """
        try:
            logger.info("Finding matching trials for patient", patient_id=patient_record.id)

            # Extract patient profile
            patient_profile = self._extract_patient_profile(patient_record)

            # Get available trials (in production, query database with caching)
            available_trials = await self._get_available_trials()

            if not available_trials:
                return []

            # Score trials
            scored_trials = []

            for trial in available_trials:
                # Skip trials not currently recruiting
                if trial.status != "RECRUITING":
                    continue

                # Calculate matching scores
                condition_similarity = self._calculate_condition_similarity(
                    patient_profile["conditions"],
                    trial.conditions
                )

                eligibility_score, unmet_criteria = self._check_eligibility_criteria(patient_profile, trial)

                location_score = 1.0 if self._check_location_eligibility(patient_profile, trial) else 0.0

                phase_score = self._calculate_phase_appropriateness(patient_profile, trial)

                # Weighted overall score
                overall_score = (
                    condition_similarity * 0.3 +
                    eligibility_score * 0.4 +
                    location_score * 0.2 +
                    phase_score * 0.1
                )

                if overall_score >= minimum_score:
                    scored_trials.append({
                        "trial": trial,
                        "overall_score": overall_score,
                        "condition_similarity": condition_similarity,
                        "eligibility_score": eligibility_score,
                        "location_score": location_score,
                        "phase_score": phase_score,
                        "unmet_criteria": unmet_criteria,
                        "explanation": self._generate_match_explanation(
                            overall_score, condition_similarity, eligibility_score, location_score, phase_score
                        )
                    })

            # Sort by overall score and return top results
            scored_trials.sort(key=lambda x: x["overall_score"], reverse=True)

            results = scored_trials[:max_results]

            # Log search for analytics
            await self._log_trial_search(patient_record.id, search_query, results)

            logger.info("Found matching trials", patient_id=patient_record.id, results_count=len(results))

            return results

        except Exception as e:
            logger.error("Error finding matching trials", error=str(e), patient_id=patient_record.id)
            return []

    async def _get_available_trials(self) -> List[ClinicalTrial]:
        """Get available clinical trials from database or cache."""
        # In production, this would query the database
        # For now, return mock data
        return []

    def _generate_match_explanation(
        self,
        overall_score: float,
        condition_similarity: float,
        eligibility_score: float,
        location_score: float,
        phase_score: float
    ) -> str:
        """Generate human-readable explanation for trial match."""
        explanations = []

        if condition_similarity > 0.7:
            explanations.append("Strong match with your medical condition")
        elif condition_similarity > 0.4:
            explanations.append("Moderate match with your medical condition")

        if eligibility_score > 0.8:
            explanations.append("You appear to meet most eligibility criteria")
        elif eligibility_score > 0.5:
            explanations.append("You may meet eligibility criteria")

        if location_score > 0.5:
            explanations.append("Trial location is accessible")
        else:
            explanations.append("May require travel for participation")

        if phase_score > 0.7:
            explanations.append("Trial phase appears appropriate for your situation")

        return ". ".join(explanations)

    async def _log_trial_search(self, patient_id: int, search_query: str, results: List[Dict]):
        """Log trial search for analytics and model improvement."""
        # TODO: Implement logging to database
        top_trial_ids = [result["trial"].id for result in results[:5]]

        logger.info("Trial search logged", patient_id=patient_id, query=search_query, results_count=len(results))


# Global trial matching service instance
trial_matching_service = TrialMatchingService()
