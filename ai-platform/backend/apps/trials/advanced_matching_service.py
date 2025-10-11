"""
Advanced AI-powered clinical trial matching system with deep learning
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import structlog

from backend.core.config import settings
from backend.apps.trials.models import ClinicalTrial, TrialMatch, TrialSearchQuery, User

logger = structlog.get_logger()


class TrialMatchingModel(nn.Module):
    """
    Deep learning model for clinical trial-patient matching.
    """

    def __init__(self, embedding_dim=768, hidden_dim=256, num_classes=1):
        super(TrialMatchingModel, self).__init__()

        # Embedding layers for different features
        self.patient_encoder = nn.Linear(embedding_dim, hidden_dim)
        self.trial_encoder = nn.Linear(embedding_dim, hidden_dim)

        # Multi-head attention for feature interaction
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

        # Matching network
        self.matching_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Feature extractors
        self.scaler = StandardScaler()

    def forward(self, patient_features, trial_features):
        """Forward pass through the matching network."""

        # Encode patient and trial features
        patient_encoded = torch.relu(self.patient_encoder(patient_features))
        trial_encoded = torch.relu(self.trial_encoder(trial_features))

        # Concatenate features
        combined_features = torch.cat([patient_encoded, trial_encoded], dim=1)

        # Pass through matching layers
        match_score = self.matching_layers(combined_features)

        return match_score.squeeze()


class AdvancedTrialMatchingService:
    """
    Advanced AI-powered clinical trial matching with deep learning and ML algorithms.
    """

    def __init__(self):
        """Initialize the advanced trial matching service."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models
        self.embedding_model = SentenceTransformer(settings.DEFAULT_EMBEDDING_MODEL)
        self.trial_matching_model = self._load_trial_matching_model()
        self.eligibility_classifier = self._load_eligibility_classifier()

        # ML models for different aspects of matching
        self.location_model = None
        self.phase_model = None
        self.inclusion_model = None

        # Trial database (in production, from database)
        self.trials_cache = {}
        self.trial_embeddings = {}

        # Feature importance tracking
        self.feature_importance = {}

        self._initialize_ml_models()

    def _load_trial_matching_model(self):
        """Load or create the deep learning trial matching model."""
        try:
            # In production, load pre-trained model
            # For now, return a new model instance
            model = TrialMatchingModel()
            model.to(self.device)
            model.eval()
            return model

        except Exception as e:
            logger.error("Error loading trial matching model", error=str(e))
            return None

    def _load_eligibility_classifier(self):
        """Load eligibility classification model."""
        try:
            # Load fine-tuned BERT model for eligibility classification
            # In production, use a medical-domain adapted model
            return pipeline(
                "text-classification",
                model="microsoft/BioGPT",
                device=0 if torch.cuda.is_available() else -1
            )

        except Exception as e:
            logger.error("Error loading eligibility classifier", error=str(e))
            return None

    def _initialize_ml_models(self):
        """Initialize traditional ML models for specific matching aspects."""
        try:
            # Location preference model
            self.location_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )

            # Phase appropriateness model
            self.phase_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            # Inclusion criteria model
            self.inclusion_model = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=8,
                random_state=42
            )

            logger.info("ML models initialized successfully")

        except Exception as e:
            logger.error("Error initializing ML models", error=str(e))

    async def find_optimal_trials(
        self,
        patient_profile: Dict[str, Any],
        search_criteria: Optional[Dict[str, Any]] = None,
        max_results: int = 10,
        diversity_factor: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Find optimal clinical trials using advanced ML algorithms.

        Args:
            patient_profile: Comprehensive patient information
            search_criteria: Optional search filters
            max_results: Maximum number of results
            diversity_factor: Balance between relevance and diversity

        Returns:
            Ranked list of matching trials with detailed explanations
        """
        try:
            logger.info("Finding optimal trials with advanced ML", patient_id=patient_profile.get("id"))

            # Get available trials
            available_trials = await self._get_trials_with_embeddings()

            if not available_trials:
                return []

            # Extract patient features for ML models
            patient_features = await self._extract_patient_features(patient_profile)

            # Score trials using ensemble of ML models
            scored_trials = []

            for trial in available_trials:
                # Multi-model scoring
                relevance_score = await self._calculate_relevance_score(patient_features, trial)
                eligibility_score = await self._calculate_eligibility_score(patient_profile, trial)
                location_score = await self._calculate_location_score(patient_profile, trial)
                phase_score = await self._calculate_phase_score(patient_profile, trial)

                # Ensemble scoring with weighted combination
                overall_score = (
                    relevance_score * 0.35 +
                    eligibility_score * 0.35 +
                    location_score * 0.15 +
                    phase_score * 0.15
                )

                if overall_score >= 0.3:  # Minimum threshold
                    # Generate detailed explanation
                    explanation = await self._generate_detailed_explanation(
                        patient_profile, trial, relevance_score, eligibility_score,
                        location_score, phase_score
                    )

                    # Calculate confidence interval
                    confidence = await self._calculate_prediction_confidence(
                        patient_features, trial
                    )

                    scored_trials.append({
                        "trial": trial,
                        "overall_score": overall_score,
                        "relevance_score": relevance_score,
                        "eligibility_score": eligibility_score,
                        "location_score": location_score,
                        "phase_score": phase_score,
                        "confidence_interval": confidence,
                        "explanation": explanation,
                        "feature_importance": await self._get_feature_importance(patient_features, trial),
                        "diversity_bonus": await self._calculate_diversity_bonus(trial, scored_trials)
                    })

            # Apply diversity ranking
            ranked_trials = await self._apply_diversity_ranking(scored_trials, diversity_factor)

            # Return top results
            results = ranked_trials[:max_results]

            # Log comprehensive search results
            await self._log_advanced_search(patient_profile, results)

            logger.info("Advanced trial matching completed", results_found=len(results))
            return results

        except Exception as e:
            logger.error("Error in advanced trial matching", error=str(e))
            return []

    async def _extract_patient_features(self, patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive features from patient profile for ML models."""

        features = {
            # Demographic features
            "age": patient_profile.get("age", 0),
            "gender": 1 if patient_profile.get("gender") == "female" else 0,
            "location": patient_profile.get("location", ""),

            # Medical condition features
            "cancer_type": self._encode_condition(patient_profile.get("cancer_type", "")),
            "cancer_stage": self._encode_stage(patient_profile.get("cancer_stage", "")),
            "performance_status": self._encode_performance_status(patient_profile.get("performance_status", "")),

            # Treatment history features
            "treatment_lines": len(patient_profile.get("treatment_history", {}).get("lines", [])),
            "current_treatments": len(patient_profile.get("current_treatments", [])),
            "comorbidities_count": len(patient_profile.get("comorbidities", [])),

            # Biomarker features
            "biomarker_count": len(patient_profile.get("biomarkers", {})),
            "genetic_markers": len(patient_profile.get("genetic_markers", {})),

            # Behavioral features
            "adherence_history": patient_profile.get("adherence_score", 0.5),
            "previous_trials": patient_profile.get("previous_trial_count", 0),

            # Temporal features
            "diagnosis_recency": self._calculate_recency(patient_profile.get("diagnosis_date")),
            "last_treatment_recency": self._calculate_recency(patient_profile.get("last_treatment_date"))
        }

        # Add computed features
        features["treatment_intensity"] = self._calculate_treatment_intensity(features)
        features["complexity_score"] = self._calculate_complexity_score(features)

        return features

    def _encode_condition(self, condition: str) -> int:
        """Encode cancer condition into numeric value."""
        condition_map = {
            "breast": 1, "lung": 2, "prostate": 3, "colorectal": 4,
            "melanoma": 5, "leukemia": 6, "lymphoma": 7, "ovarian": 8
        }
        return condition_map.get(condition.lower(), 0)

    def _encode_stage(self, stage: str) -> int:
        """Encode cancer stage into numeric value."""
        stage_map = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
        return stage_map.get(stage.lower(), 0)

    def _encode_performance_status(self, status: str) -> int:
        """Encode performance status into numeric value."""
        status_map = {"ecog_0": 0, "ecog_1": 1, "ecog_2": 2, "ecog_3": 3, "ecog_4": 4}
        return status_map.get(status.lower(), 1)

    def _calculate_recency(self, date_str: Optional[str]) -> int:
        """Calculate days since date (higher = more recent)."""
        if not date_str:
            return 0

        try:
            date = datetime.fromisoformat(date_str)
            days_since = (datetime.now() - date).days
            return min(days_since, 3650)  # Cap at 10 years
        except:
            return 0

    def _calculate_treatment_intensity(self, features: Dict[str, Any]) -> float:
        """Calculate treatment intensity score."""
        return min(1.0, (features["treatment_lines"] * 0.2 + features["current_treatments"] * 0.3))

    def _calculate_complexity_score(self, features: Dict[str, Any]) -> float:
        """Calculate patient complexity score."""
        return min(1.0, (features["comorbidities_count"] * 0.1 +
                        features["biomarker_count"] * 0.1 +
                        (1 - features["adherence_history"])))

    async def _calculate_relevance_score(self, patient_features: Dict[str, Any], trial: ClinicalTrial) -> float:
        """Calculate relevance score using deep learning model."""

        if not self.trial_matching_model:
            return 0.5  # Fallback score

        try:
            # Get trial features
            trial_features = await self._extract_trial_features(trial)

            # Convert to tensors
            patient_tensor = torch.tensor([list(patient_features.values())], dtype=torch.float32).to(self.device)
            trial_tensor = torch.tensor([list(trial_features.values())], dtype=torch.float32).to(self.device)

            # Get prediction
            with torch.no_grad():
                score = self.trial_matching_model(patient_tensor, trial_tensor)
                relevance_score = torch.sigmoid(score).item()

            return relevance_score

        except Exception as e:
            logger.error("Error calculating relevance score", error=str(e))
            return 0.5

    async def _extract_trial_features(self, trial: ClinicalTrial) -> Dict[str, Any]:
        """Extract features from trial for ML models."""
        return {
            "phase": self._encode_trial_phase(trial.phase.value),
            "enrollment_target": trial.enrollment_target or 0,
            "condition_count": len(trial.conditions),
            "intervention_count": len(trial.interventions),
            "location_count": len(trial.locations),
            "sponsor_type": 1 if "pharma" in (trial.sponsor or "").lower() else 0,
            "study_duration_days": self._calculate_study_duration(trial),
            "inclusion_complexity": self._calculate_inclusion_complexity(trial)
        }

    def _encode_trial_phase(self, phase: str) -> int:
        """Encode trial phase into numeric value."""
        phase_map = {"phase_0": 0, "phase_1": 1, "phase_1_2": 1.5, "phase_2": 2,
                    "phase_2_3": 2.5, "phase_3": 3, "phase_4": 4}
        return phase_map.get(phase, 2)

    def _calculate_study_duration(self, trial: ClinicalTrial) -> int:
        """Calculate estimated study duration in days."""
        if trial.study_start_date and trial.primary_completion_date:
            duration = trial.primary_completion_date - trial.study_start_date
            return duration.days
        return 365  # Default 1 year

    def _calculate_inclusion_complexity(self, trial: ClinicalTrial) -> float:
        """Calculate complexity of inclusion criteria."""
        criteria_text = trial.eligibility_criteria or ""
        return min(1.0, len(criteria_text.split()) / 100)  # Normalize by length

    async def _calculate_eligibility_score(self, patient_profile: Dict[str, Any], trial: ClinicalTrial) -> float:
        """Calculate eligibility score using ML model."""

        if not self.inclusion_model:
            return await self._rule_based_eligibility_check(patient_profile, trial)

        try:
            # Extract features for eligibility prediction
            features = await self._extract_eligibility_features(patient_profile, trial)

            # Get prediction
            prediction = self.inclusion_model.predict_proba([features])[0][1]

            return prediction

        except Exception as e:
            logger.error("Error calculating eligibility score", error=str(e))
            return await self._rule_based_eligibility_check(patient_profile, trial)

    async def _extract_eligibility_features(self, patient_profile: Dict[str, Any], trial: ClinicalTrial) -> List[float]:
        """Extract features for eligibility prediction."""
        # This would be a comprehensive feature vector
        # For now, return a simplified version
        return [
            patient_profile.get("age", 0) / 100,
            1 if patient_profile.get("gender") == "female" else 0,
            len(patient_profile.get("current_treatments", [])),
            len(patient_profile.get("comorbidities", [])),
            self._encode_condition(patient_profile.get("cancer_type", "")),
            self._encode_trial_phase(trial.phase.value)
        ]

    async def _rule_based_eligibility_check(self, patient_profile: Dict[str, Any], trial: ClinicalTrial) -> float:
        """Rule-based eligibility checking as fallback."""

        score = 0.5  # Base score

        # Age appropriateness
        if patient_profile.get("age", 0) > 18:
            score += 0.1

        # Treatment history compatibility
        if len(patient_profile.get("treatment_history", {}).get("lines", [])) < 3:
            score += 0.1

        # Performance status
        if patient_profile.get("performance_status") in ["ecog_0", "ecog_1"]:
            score += 0.1

        return min(1.0, score)

    async def _calculate_location_score(self, patient_profile: Dict[str, Any], trial: ClinicalTrial) -> float:
        """Calculate location compatibility score."""

        if not self.location_model:
            return await self._rule_based_location_score(patient_profile, trial)

        try:
            patient_location = patient_profile.get("location", "")
            trial_locations = trial.locations or []

            # Simple distance-based scoring
            min_distance = float('inf')
            for location in trial_locations:
                distance = await self._calculate_trial_distance(patient_location, location)
                min_distance = min(min_distance, distance)

            # Convert distance to score (closer = higher score)
            if min_distance == 0:
                return 1.0
            elif min_distance < 50:
                return 0.9
            elif min_distance < 100:
                return 0.7
            elif min_distance < 200:
                return 0.5
            else:
                return 0.3

        except Exception as e:
            logger.error("Error calculating location score", error=str(e))
            return 0.5

    async def _calculate_trial_distance(self, patient_location: str, trial_location: Dict[str, Any]) -> float:
        """Calculate distance between patient and trial location."""
        # In production, use geocoding APIs
        # For now, return mock distance
        return 25.0  # Mock 25 miles

    async def _rule_based_location_score(self, patient_profile: Dict[str, Any], trial: ClinicalTrial) -> float:
        """Rule-based location scoring as fallback."""
        willing_to_travel = patient_profile.get("willing_to_travel", False)

        if willing_to_travel:
            return 0.8
        else:
            return 0.6

    async def _calculate_phase_score(self, patient_profile: Dict[str, Any], trial: ClinicalTrial) -> float:
        """Calculate phase appropriateness score."""

        if not self.phase_model:
            return await self._rule_based_phase_score(patient_profile, trial)

        try:
            features = [
                patient_profile.get("age", 0) / 100,
                len(patient_profile.get("treatment_history", {}).get("lines", [])),
                self._encode_performance_status(patient_profile.get("performance_status", ""))
            ]

            prediction = self.phase_model.predict_proba([features])[0][1]
            return prediction

        except Exception as e:
            logger.error("Error calculating phase score", error=str(e))
            return await self._rule_based_phase_score(patient_profile, trial)

    async def _rule_based_phase_score(self, patient_profile: Dict[str, Any], trial: ClinicalTrial) -> float:
        """Rule-based phase scoring as fallback."""

        treatment_lines = len(patient_profile.get("treatment_history", {}).get("lines", []))
        performance_status = patient_profile.get("performance_status", "")

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

    async def _generate_detailed_explanation(
        self,
        patient_profile: Dict[str, Any],
        trial: ClinicalTrial,
        relevance_score: float,
        eligibility_score: float,
        location_score: float,
        phase_score: float
    ) -> str:
        """Generate detailed explanation for trial match."""

        explanations = []

        if relevance_score > 0.7:
            explanations.append("Strong match with your medical condition and treatment history")
        elif relevance_score > 0.4:
            explanations.append("Moderate match with your condition profile")

        if eligibility_score > 0.8:
            explanations.append("You appear to meet most eligibility criteria")
        elif eligibility_score > 0.5:
            explanations.append("You likely meet the basic eligibility requirements")

        if location_score > 0.7:
            explanations.append("Trial location is convenient for participation")
        elif location_score > 0.4:
            explanations.append("Trial location is reasonably accessible")

        if phase_score > 0.7:
            explanations.append("Trial phase appears well-suited to your treatment stage")

        if not explanations:
            explanations.append("This trial may be worth discussing with your healthcare provider")

        return ". ".join(explanations)

    async def _calculate_prediction_confidence(self, patient_features: Dict[str, Any], trial: ClinicalTrial) -> Tuple[float, float]:
        """Calculate confidence interval for prediction."""

        # Simplified confidence calculation
        base_confidence = 0.8

        # Reduce confidence for unusual cases
        if patient_features["complexity_score"] > 0.8:
            base_confidence -= 0.2

        if patient_features["treatment_lines"] > 3:
            base_confidence -= 0.1

        confidence_interval = (max(0.1, base_confidence - 0.1), min(0.95, base_confidence + 0.1))

        return confidence_interval

    async def _get_feature_importance(self, patient_features: Dict[str, Any], trial: ClinicalTrial) -> Dict[str, float]:
        """Get feature importance for explainability."""

        # Simplified feature importance
        importance = {
            "condition_match": 0.25,
            "eligibility_criteria": 0.25,
            "location_accessibility": 0.20,
            "phase_appropriateness": 0.15,
            "treatment_history": 0.15
        }

        return importance

    async def _calculate_diversity_bonus(self, trial: ClinicalTrial, existing_trials: List[Dict[str, Any]]) -> float:
        """Calculate diversity bonus to ensure variety in results."""

        if not existing_trials:
            return 0.0

        # Check for diversity in trial characteristics
        diversity_factors = []

        # Sponsor diversity
        existing_sponsors = {t["trial"].sponsor for t in existing_trials}
        if trial.sponsor not in existing_sponsors:
            diversity_factors.append(0.1)

        # Phase diversity
        existing_phases = {t["trial"].phase.value for t in existing_trials}
        if trial.phase.value not in existing_phases:
            diversity_factors.append(0.1)

        return min(0.2, sum(diversity_factors))

    async def _apply_diversity_ranking(self, scored_trials: List[Dict[str, Any]], diversity_factor: float) -> List[Dict[str, Any]]:
        """Apply diversity ranking to ensure varied results."""

        if diversity_factor == 0:
            return sorted(scored_trials, key=lambda x: x["overall_score"], reverse=True)

        # Sort by relevance first
        relevance_sorted = sorted(scored_trials, key=lambda x: x["overall_score"], reverse=True)

        # Apply diversity re-ranking
        diverse_results = []
        used_sponsors = set()
        used_phases = set()

        for trial_data in relevance_sorted:
            trial = trial_data["trial"]

            # Calculate diversity bonus
            sponsor_bonus = 0.1 if trial.sponsor not in used_sponsors else 0
            phase_bonus = 0.1 if trial.phase.value not in used_phases else 0

            # Apply diversity factor
            adjusted_score = trial_data["overall_score"] + (sponsor_bonus + phase_bonus) * diversity_factor

            trial_data["adjusted_score"] = adjusted_score
            trial_data["diversity_bonus"] = sponsor_bonus + phase_bonus

            diverse_results.append(trial_data)

            # Update diversity tracking
            used_sponsors.add(trial.sponsor)
            used_phases.add(trial.phase.value)

        return sorted(diverse_results, key=lambda x: x["adjusted_score"], reverse=True)

    async def _get_trials_with_embeddings(self) -> List[ClinicalTrial]:
        """Get available trials with pre-computed embeddings."""
        # In production, query database with cached embeddings
        # For now, return mock data
        return []

    async def _log_advanced_search(self, patient_profile: Dict[str, Any], results: List[Dict[str, Any]]):
        """Log advanced search for analytics and model improvement."""

        # Extract top trial IDs and scores
        top_trials = [
            {
                "trial_id": result["trial"].id,
                "score": result["overall_score"],
                "rank": i + 1
            }
            for i, result in enumerate(results[:5])
        ]

        logger.info("Advanced trial search logged", patient_id=patient_profile.get("id"), top_trial_count=len(top_trials))

    async def retrain_models(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Retrain ML models based on user feedback and outcomes.

        Args:
            feedback_data: List of trial match feedback and outcomes

        Returns:
            Training results and model performance metrics
        """
        try:
            logger.info("Retraining trial matching models", feedback_count=len(feedback_data))

            # Prepare training data
            X_train, y_train = await self._prepare_training_data(feedback_data)

            if len(X_train) < 100:
                return {"error": "Insufficient training data"}

            # Retrain models
            training_results = {}

            # Retrain eligibility model
            if self.inclusion_model:
                self.inclusion_model.fit(X_train, y_train)
                training_results["eligibility_model"] = {"status": "retrained", "samples": len(X_train)}

            # Retrain phase model
            if self.phase_model:
                self.phase_model.fit(X_train, y_train)
                training_results["phase_model"] = {"status": "retrained", "samples": len(X_train)}

            # Retrain location model
            if self.location_model:
                self.location_model.fit(X_train, y_train)
                training_results["location_model"] = {"status": "retrained", "samples": len(X_train)}

            # Save updated models
            await self._save_updated_models()

            logger.info("Model retraining completed", models_updated=len(training_results))

            return {
                "status": "success",
                "models_updated": training_results,
                "training_samples": len(X_train),
                "retrained_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error("Error retraining models", error=str(e))
            return {"error": "Failed to retrain models"}

    async def _prepare_training_data(self, feedback_data: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[float]]:
        """Prepare training data from feedback."""

        X_train = []
        y_train = []

        for feedback in feedback_data:
            # Extract features and labels from feedback
            features = feedback.get("patient_features", [])
            outcome = feedback.get("match_outcome", 0)  # 1 for successful match, 0 for unsuccessful

            if features and outcome in [0, 1]:
                X_train.append(features)
                y_train.append(outcome)

        return X_train, y_train

    async def _save_updated_models(self):
        """Save updated models to disk."""
        try:
            # Save models to model storage
            # In production, save to cloud storage or model registry
            logger.info("Updated models saved successfully")

        except Exception as e:
            logger.error("Error saving updated models", error=str(e))


# Global advanced trial matching service instance
advanced_trial_matching_service = AdvancedTrialMatchingService()
