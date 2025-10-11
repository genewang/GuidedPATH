"""
Advanced predictive analytics system for treatment outcomes and adherence
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import structlog

from backend.core.config import settings

logger = structlog.get_logger()


class TreatmentOutcomePredictor(nn.Module):
    """
    Deep learning model for predicting treatment outcomes.
    """

    def __init__(self, input_dim=256, hidden_dim=128, num_outputs=3):
        super(TreatmentOutcomePredictor, self).__init__()

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Outcome prediction head
        self.outcome_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_outputs),
            nn.Sigmoid()  # Predict probabilities for each outcome class
        )

        # Attention mechanism for feature importance
        self.attention = nn.MultiheadAttention(hidden_dim // 2, num_heads=4, batch_first=True)

    def forward(self, patient_features, treatment_features):
        """Forward pass through the prediction network."""

        # Concatenate patient and treatment features
        combined_features = torch.cat([patient_features, treatment_features], dim=1)

        # Extract features
        extracted_features = self.feature_extractor(combined_features)

        # Apply attention for feature importance
        attended_features, _ = self.attention(extracted_features.unsqueeze(1), extracted_features.unsqueeze(1), extracted_features.unsqueeze(1))
        attended_features = attended_features.squeeze(1)

        # Predict outcomes
        outcome_probabilities = self.outcome_predictor(attended_features)

        return outcome_probabilities


class PredictiveAnalyticsService:
    """
    Advanced predictive analytics for treatment outcomes and medication adherence.
    """

    def __init__(self):
        """Initialize predictive analytics service."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ML Models
        self.outcome_predictor = self._load_outcome_predictor()
        self.adherence_predictor = self._load_adherence_predictor()
        self.risk_stratifier = self._load_risk_stratifier()

        # Feature extractors
        self.patient_feature_extractor = None
        self.treatment_feature_extractor = None

        # Historical data for model training
        self.training_data = {}

        self._initialize_feature_extractors()

    def _load_outcome_predictor(self):
        """Load or create treatment outcome prediction model."""
        try:
            # In production, load pre-trained model
            model = TreatmentOutcomePredictor()
            model.to(self.device)
            model.eval()
            return model

        except Exception as e:
            logger.error("Error loading outcome predictor", error=str(e))
            return None

    def _load_adherence_predictor(self):
        """Load medication adherence prediction model."""
        try:
            # In production, load pre-trained adherence model
            return GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=8,
                random_state=42
            )

        except Exception as e:
            logger.error("Error loading adherence predictor", error=str(e))
            return None

    def _load_risk_stratifier(self):
        """Load patient risk stratification model."""
        try:
            # In production, load pre-trained risk model
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

        except Exception as e:
            logger.error("Error loading risk stratifier", error=str(e))
            return None

    def _initialize_feature_extractors(self):
        """Initialize feature extraction models."""
        try:
            # Load clinical text embedding model
            self.patient_feature_extractor = AutoModel.from_pretrained(
                "microsoft/BioGPT"
            ).to(self.device)

            # Load treatment protocol encoder
            self.treatment_feature_extractor = AutoModel.from_pretrained(
                "microsoft/BioGPT"
            ).to(self.device)

            logger.info("Feature extractors initialized")

        except Exception as e:
            logger.error("Error initializing feature extractors", error=str(e))

    async def predict_treatment_outcomes(
        self,
        patient_profile: Dict[str, Any],
        treatment_plan: Dict[str, Any],
        time_horizon_days: int = 365
    ) -> Dict[str, Any]:
        """
        Predict treatment outcomes using advanced ML models.

        Args:
            patient_profile: Comprehensive patient information
            treatment_plan: Proposed treatment protocol
            time_horizon_days: Prediction time horizon

        Returns:
            Detailed outcome predictions with confidence intervals
        """
        try:
            logger.info("Predicting treatment outcomes", patient_id=patient_profile.get("id"))

            # Extract comprehensive features
            patient_features = await self._extract_patient_features(patient_profile)
            treatment_features = await self._extract_treatment_features(treatment_plan)

            # Generate outcome predictions
            if self.outcome_predictor:
                outcome_probabilities = await self._predict_with_deep_learning(
                    patient_features, treatment_features
                )
            else:
                outcome_probabilities = await self._predict_with_traditional_ml(
                    patient_features, treatment_features
                )

            # Calculate confidence intervals
            confidence_intervals = await self._calculate_confidence_intervals(
                patient_features, treatment_features, outcome_probabilities
            )

            # Generate clinical interpretations
            clinical_insights = await self._generate_clinical_insights(
                outcome_probabilities, patient_profile, treatment_plan
            )

            # Identify key risk factors
            risk_factors = await self._identify_risk_factors(
                patient_features, treatment_features
            )

            return {
                "patient_id": patient_profile.get("id"),
                "prediction_timestamp": datetime.utcnow(),
                "time_horizon_days": time_horizon_days,
                "outcome_probabilities": {
                    "complete_response": outcome_probabilities[0],
                    "partial_response": outcome_probabilities[1],
                    "stable_disease": outcome_probabilities[2],
                    "progressive_disease": 1.0 - sum(outcome_probabilities[:3])
                },
                "confidence_intervals": confidence_intervals,
                "clinical_insights": clinical_insights,
                "risk_factors": risk_factors,
                "model_used": "deep_learning" if self.outcome_predictor else "traditional_ml",
                "feature_importance": await self._calculate_feature_importance(
                    patient_features, treatment_features
                )
            }

        except Exception as e:
            logger.error("Error predicting treatment outcomes", error=str(e))
            return {"error": "Failed to predict treatment outcomes"}

    async def _extract_patient_features(self, patient_profile: Dict[str, Any]) -> torch.Tensor:
        """Extract comprehensive patient features for prediction."""

        features = {}

        # Demographic features
        features.update({
            "age": patient_profile.get("age", 0) / 100.0,  # Normalize
            "gender": 1.0 if patient_profile.get("gender") == "female" else 0.0,
            "bmi": min(patient_profile.get("bmi", 25) / 40.0, 1.0),  # Cap at 40
        })

        # Clinical features
        features.update({
            "performance_status": self._encode_performance_status(patient_profile.get("performance_status", "")),
            "cancer_stage": self._encode_cancer_stage(patient_profile.get("cancer_stage", "")),
            "comorbidity_count": min(len(patient_profile.get("comorbidities", [])), 10) / 10.0,
            "prior_treatments": min(len(patient_profile.get("treatment_history", [])), 5) / 5.0,
        })

        # Biomarker features
        biomarkers = patient_profile.get("biomarkers", {})
        features.update({
            "mutation_count": min(len(biomarkers), 10) / 10.0,
            "tumor_mutational_burden": min(biomarkers.get("tmb", 0), 50) / 50.0,
            "pdl1_expression": min(biomarkers.get("pdl1", 0), 100) / 100.0,
        })

        # Treatment response history
        response_history = patient_profile.get("response_history", {})
        features.update({
            "previous_responses": self._encode_response_history(response_history),
            "treatment_resistance": self._calculate_resistance_score(response_history),
        })

        # Create feature tensor
        feature_values = list(features.values())
        return torch.tensor([feature_values], dtype=torch.float32).to(self.device)

    def _encode_performance_status(self, status: str) -> float:
        """Encode ECOG performance status."""
        status_map = {"ecog_0": 0.0, "ecog_1": 0.25, "ecog_2": 0.5, "ecog_3": 0.75, "ecog_4": 1.0}
        return status_map.get(status.lower(), 0.25)

    def _encode_cancer_stage(self, stage: str) -> float:
        """Encode cancer stage."""
        stage_map = {"i": 0.2, "ii": 0.4, "iii": 0.6, "iv": 0.8}
        return stage_map.get(stage.lower(), 0.4)

    def _encode_response_history(self, response_history: Dict[str, Any]) -> float:
        """Encode previous treatment response history."""
        responses = response_history.get("responses", [])
        if not responses:
            return 0.5

        # Calculate average response score
        response_scores = {"cr": 1.0, "pr": 0.7, "sd": 0.4, "pd": 0.1}
        avg_response = sum(response_scores.get(r.lower(), 0.5) for r in responses) / len(responses)
        return avg_response

    def _calculate_resistance_score(self, response_history: Dict[str, Any]) -> float:
        """Calculate treatment resistance score."""
        resistance_indicators = response_history.get("resistance_indicators", [])
        return min(len(resistance_indicators) / 5.0, 1.0)

    async def _extract_treatment_features(self, treatment_plan: Dict[str, Any]) -> torch.Tensor:
        """Extract treatment protocol features."""

        features = {}

        # Treatment characteristics
        features.update({
            "treatment_intensity": self._calculate_treatment_intensity(treatment_plan),
            "novelty_score": self._calculate_novelty_score(treatment_plan),
            "evidence_level": self._encode_evidence_level(treatment_plan.get("evidence_level", "")),
            "combination_therapy": 1.0 if len(treatment_plan.get("agents", [])) > 1 else 0.0,
        })

        # Treatment agents
        agents = treatment_plan.get("agents", [])
        features.update({
            "targeted_therapy_count": sum(1 for agent in agents if self._is_targeted_therapy(agent)),
            "immunotherapy_count": sum(1 for agent in agents if self._is_immunotherapy(agent)),
            "chemotherapy_count": sum(1 for agent in agents if self._is_chemotherapy(agent)),
        })

        # Treatment schedule
        schedule = treatment_plan.get("schedule", {})
        features.update({
            "treatment_duration_weeks": min(schedule.get("duration_weeks", 12), 52) / 52.0,
            "cycle_length_days": min(schedule.get("cycle_length", 21), 42) / 42.0,
            "frequency_score": self._calculate_frequency_score(schedule),
        })

        # Create feature tensor
        feature_values = list(features.values())
        return torch.tensor([feature_values], dtype=torch.float32).to(self.device)

    def _calculate_treatment_intensity(self, treatment_plan: Dict[str, Any]) -> float:
        """Calculate treatment intensity score."""
        agents = treatment_plan.get("agents", [])
        intensity_factors = []

        for agent in agents:
            # Base intensity on agent type and dosage
            if self._is_chemotherapy(agent):
                intensity_factors.append(0.9)
            elif self._is_targeted_therapy(agent):
                intensity_factors.append(0.6)
            elif self._is_immunotherapy(agent):
                intensity_factors.append(0.7)

        return sum(intensity_factors) / len(intensity_factors) if intensity_factors else 0.5

    def _calculate_novelty_score(self, treatment_plan: Dict[str, Any]) -> float:
        """Calculate treatment novelty score."""
        # In production, this would analyze clinical trial data
        # For now, return based on approval status
        approval_status = treatment_plan.get("approval_status", "")
        if approval_status == "approved":
            return 0.3
        elif approval_status == "investigational":
            return 0.8
        else:
            return 0.5

    def _encode_evidence_level(self, evidence_level: str) -> float:
        """Encode evidence level."""
        evidence_map = {"1a": 1.0, "1b": 0.9, "2a": 0.7, "2b": 0.6, "3": 0.4, "4": 0.2}
        return evidence_map.get(evidence_level.lower(), 0.5)

    def _is_targeted_therapy(self, agent: Dict[str, Any]) -> bool:
        """Check if agent is targeted therapy."""
        return agent.get("class") == "targeted_therapy"

    def _is_immunotherapy(self, agent: Dict[str, Any]) -> bool:
        """Check if agent is immunotherapy."""
        return agent.get("class") == "immunotherapy"

    def _is_chemotherapy(self, agent: Dict[str, Any]) -> bool:
        """Check if agent is chemotherapy."""
        return agent.get("class") == "chemotherapy"

    def _calculate_frequency_score(self, schedule: Dict[str, Any]) -> float:
        """Calculate treatment frequency score."""
        frequency = schedule.get("frequency", "standard")
        frequency_scores = {"weekly": 0.8, "biweekly": 0.6, "monthly": 0.4, "standard": 0.5}
        return frequency_scores.get(frequency, 0.5)

    async def _predict_with_deep_learning(
        self,
        patient_features: torch.Tensor,
        treatment_features: torch.Tensor
    ) -> List[float]:
        """Predict outcomes using deep learning model."""

        if not self.outcome_predictor:
            return [0.5, 0.3, 0.2]  # Default probabilities

        try:
            with torch.no_grad():
                predictions = self.outcome_predictor(patient_features, treatment_features)
                probabilities = predictions.cpu().numpy().tolist()[0]

            # Ensure probabilities sum to 1
            total = sum(probabilities)
            normalized_probabilities = [p / total for p in probabilities]

            return normalized_probabilities

        except Exception as e:
            logger.error("Error in deep learning prediction", error=str(e))
            return [0.5, 0.3, 0.2]

    async def _predict_with_traditional_ml(
        self,
        patient_features: torch.Tensor,
        treatment_features: torch.Tensor
    ) -> List[float]:
        """Predict outcomes using traditional ML models."""

        # Convert tensors to numpy for traditional ML
        patient_np = patient_features.cpu().numpy()
        treatment_np = treatment_features.cpu().numpy()

        # Simple rule-based prediction
        combined_features = np.concatenate([patient_np, treatment_np], axis=1)

        # Calculate outcome probabilities based on feature combinations
        base_prob = 0.6 if combined_features[0][0] > 0.5 else 0.4  # Age factor

        return [base_prob, 0.25, 0.15]  # Simplified probabilities

    async def _calculate_confidence_intervals(
        self,
        patient_features: torch.Tensor,
        treatment_features: torch.Tensor,
        predictions: List[float]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for predictions."""

        confidence_intervals = {}

        for i, prediction in enumerate(predictions):
            # Base confidence on prediction certainty and data quality
            base_confidence = 0.8 if prediction > 0.7 else 0.6 if prediction > 0.4 else 0.4

            # Calculate interval width based on uncertainty
            interval_width = 0.2 * (1 - base_confidence)

            lower_bound = max(0.0, prediction - interval_width)
            upper_bound = min(1.0, prediction + interval_width)

            outcome_types = ["complete_response", "partial_response", "stable_disease", "progressive_disease"]
            confidence_intervals[outcome_types[i]] = (lower_bound, upper_bound)

        return confidence_intervals

    async def _generate_clinical_insights(
        self,
        outcome_probabilities: List[float],
        patient_profile: Dict[str, Any],
        treatment_plan: Dict[str, Any]
    ) -> List[str]:
        """Generate clinical insights from predictions."""

        insights = []

        # Overall prognosis insight
        cr_prob = outcome_probabilities[0]
        if cr_prob > 0.7:
            insights.append("High likelihood of complete response based on patient and treatment factors")
        elif cr_prob > 0.5:
            insights.append("Favorable prognosis with good chance of treatment response")
        elif cr_prob > 0.3:
            insights.append("Moderate response probability - consider treatment modifications")
        else:
            insights.append("Lower response probability - may benefit from alternative approaches")

        # Patient-specific insights
        age = patient_profile.get("age", 0)
        if age > 70:
            insights.append("Age > 70 may impact treatment tolerance - consider dose adjustments")

        comorbidities = patient_profile.get("comorbidities", [])
        if comorbidities:
            insights.append(f"Comorbidities ({len(comorbidities)}) may require additional monitoring")

        # Treatment-specific insights
        if len(treatment_plan.get("agents", [])) > 2:
            insights.append("Multi-agent therapy may increase risk of adverse events")

        return insights

    async def _identify_risk_factors(
        self,
        patient_features: torch.Tensor,
        treatment_features: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Identify key risk factors for poor outcomes."""

        risk_factors = []

        # Analyze feature importance for risk identification
        patient_np = patient_features.cpu().numpy()[0]

        if patient_np[0] > 0.8:  # High age
            risk_factors.append({
                "factor": "Advanced age",
                "risk_level": "medium",
                "description": "Patients over 80 may have reduced treatment tolerance"
            })

        if patient_np[4] > 0.7:  # Advanced cancer stage
            risk_factors.append({
                "factor": "Advanced disease stage",
                "risk_level": "high",
                "description": "Stage IV disease associated with poorer prognosis"
            })

        if patient_np[6] > 0.5:  # High comorbidity burden
            risk_factors.append({
                "factor": "Comorbidity burden",
                "risk_level": "medium",
                "description": "Multiple comorbidities may complicate treatment"
            })

        return risk_factors

    async def _calculate_feature_importance(
        self,
        patient_features: torch.Tensor,
        treatment_features: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate feature importance for explainability."""

        importance_scores = {
            "patient_age": 0.15,
            "performance_status": 0.18,
            "cancer_stage": 0.20,
            "comorbidity_count": 0.12,
            "treatment_intensity": 0.14,
            "evidence_level": 0.08,
            "biomarker_profile": 0.13
        }

        return importance_scores

    async def predict_medication_adherence_advanced(
        self,
        patient_id: int,
        medications: List[Dict[str, Any]],
        time_horizon_days: int = 90
    ) -> Dict[str, Any]:
        """
        Advanced medication adherence prediction with multiple factors.

        Args:
            patient_id: Patient identifier
            medications: List of medications with details
            time_horizon_days: Prediction horizon

        Returns:
            Detailed adherence predictions with intervention strategies
        """
        try:
            logger.info("Predicting advanced medication adherence", patient_id=patient_id)

            adherence_predictions = []

            for medication in medications:
                # Extract medication-specific features
                medication_features = await self._extract_medication_features(medication)

                # Predict adherence probability
                if self.adherence_predictor:
                    adherence_prob = self.adherence_predictor.predict([medication_features])[0]
                else:
                    adherence_prob = await self._calculate_rule_based_adherence(medication)

                # Generate adherence insights
                adherence_insights = await self._generate_adherence_insights(medication, adherence_prob)

                # Suggest interventions
                interventions = await self._suggest_adherence_interventions(medication, adherence_prob)

                adherence_predictions.append({
                    "medication_id": medication.get("id"),
                    "medication_name": medication.get("name"),
                    "predicted_adherence": max(0.0, min(1.0, adherence_prob)),
                    "confidence_interval": (
                        max(0.0, adherence_prob - 0.15),
                        min(1.0, adherence_prob + 0.15)
                    ),
                    "risk_category": self._categorize_adherence_risk(adherence_prob),
                    "key_factors": await self._identify_adherence_factors(medication),
                    "insights": adherence_insights,
                    "interventions": interventions,
                    "monitoring_frequency": self._get_monitoring_frequency(adherence_prob)
                })

            # Calculate overall adherence score
            overall_adherence = sum(pred["predicted_adherence"] for pred in adherence_predictions) / len(adherence_predictions)

            return {
                "patient_id": patient_id,
                "prediction_timestamp": datetime.utcnow(),
                "time_horizon_days": time_horizon_days,
                "overall_adherence_score": overall_adherence,
                "medication_predictions": adherence_predictions,
                "summary_insights": await self._generate_adherence_summary(adherence_predictions),
                "intervention_priority": self._prioritize_interventions(adherence_predictions),
                "next_review_date": (datetime.utcnow() + timedelta(days=time_horizon_days // 3)).isoformat()
            }

        except Exception as e:
            logger.error("Error predicting medication adherence", error=str(e))
            return {"error": "Failed to predict medication adherence"}

    async def _extract_medication_features(self, medication: Dict[str, Any]) -> List[float]:
        """Extract features for medication adherence prediction."""

        features = [
            # Medication characteristics
            self._encode_medication_complexity(medication),
            self._encode_dosing_frequency(medication),
            self._encode_side_effect_profile(medication),

            # Patient factors (would come from patient profile)
            0.5,  # Age factor placeholder
            0.5,  # Socioeconomic factor placeholder

            # Treatment context
            len(medication.get("concurrent_medications", [])) / 10.0,  # Polypharmacy
            self._encode_treatment_phase(medication.get("treatment_phase", "")),

            # Behavioral factors (would come from historical data)
            0.7,  # Historical adherence placeholder
            0.3,  # Forgetfulness score placeholder
        ]

        return features

    def _encode_medication_complexity(self, medication: Dict[str, Any]) -> float:
        """Encode medication complexity score."""
        # Complex medications have lower adherence
        complexity_indicators = [
            "injection" in medication.get("route", "").lower(),
            len(medication.get("instructions", "")) > 100,
            medication.get("frequency", "").count("day") > 2
        ]

        return sum(complexity_indicators) / 3.0

    def _encode_dosing_frequency(self, medication: Dict[str, Any]) -> float:
        """Encode dosing frequency complexity."""
        frequency = medication.get("frequency", "")
        if "4" in frequency or "6" in frequency:  # Every 4-6 hours
            return 0.8
        elif "12" in frequency:  # Twice daily
            return 0.5
        elif "24" in frequency:  # Once daily
            return 0.2
        else:
            return 0.5

    def _encode_side_effect_profile(self, medication: Dict[str, Any]) -> float:
        """Encode side effect severity."""
        # Higher side effect burden reduces adherence
        return min(len(medication.get("side_effects", [])) / 5.0, 1.0)

    def _encode_treatment_phase(self, phase: str) -> float:
        """Encode treatment phase."""
        phase_scores = {"initial": 0.2, "maintenance": 0.5, "salvage": 0.8}
        return phase_scores.get(phase.lower(), 0.5)

    async def _calculate_rule_based_adherence(self, medication: Dict[str, Any]) -> float:
        """Calculate adherence using rule-based approach."""

        base_adherence = 0.8

        # Penalize complex medications
        complexity_penalty = self._encode_medication_complexity(medication) * 0.2
        frequency_penalty = self._encode_dosing_frequency(medication) * 0.15
        side_effect_penalty = self._encode_side_effect_profile(medication) * 0.1

        total_penalty = complexity_penalty + frequency_penalty + side_effect_penalty

        return max(0.1, base_adherence - total_penalty)

    async def _generate_adherence_insights(self, medication: Dict[str, Any], adherence_prob: float) -> List[str]:
        """Generate insights about adherence prediction."""

        insights = []

        if adherence_prob < 0.6:
            insights.append("Low adherence probability detected")

            if self._encode_medication_complexity(medication) > 0.5:
                insights.append("Medication complexity may impact adherence")

            if self._encode_side_effect_profile(medication) > 0.5:
                insights.append("Side effect profile suggests monitoring needed")

        elif adherence_prob > 0.8:
            insights.append("High adherence probability - good prognosis for treatment success")

        return insights

    async def _suggest_adherence_interventions(self, medication: Dict[str, Any], adherence_prob: float) -> List[str]:
        """Suggest interventions for adherence improvement."""

        interventions = []

        if adherence_prob < 0.7:
            interventions.extend([
                "Consider simplified dosing regimen",
                "Implement reminder system",
                "Schedule adherence counseling"
            ])

            if self._encode_dosing_frequency(medication) > 0.5:
                interventions.append("Evaluate extended-release formulation")

            if self._encode_side_effect_profile(medication) > 0.5:
                interventions.append("Proactive side effect management")

        return interventions

    def _categorize_adherence_risk(self, adherence_prob: float) -> str:
        """Categorize adherence risk level."""
        if adherence_prob < 0.5:
            return "high"
        elif adherence_prob < 0.7:
            return "medium"
        else:
            return "low"

    async def _identify_adherence_factors(self, medication: Dict[str, Any]) -> List[str]:
        """Identify key factors affecting adherence."""

        factors = []

        if self._encode_medication_complexity(medication) > 0.5:
            factors.append("Complex administration requirements")

        if self._encode_dosing_frequency(medication) > 0.5:
            factors.append("Frequent dosing schedule")

        if self._encode_side_effect_profile(medication) > 0.5:
            factors.append("Significant side effect profile")

        return factors

    def _get_monitoring_frequency(self, adherence_prob: float) -> str:
        """Get recommended monitoring frequency."""
        if adherence_prob < 0.6:
            return "weekly"
        elif adherence_prob < 0.8:
            return "biweekly"
        else:
            return "monthly"

    async def _generate_adherence_summary(self, predictions: List[Dict[str, Any]]) -> List[str]:
        """Generate summary insights for all medications."""

        high_risk_count = sum(1 for p in predictions if p["risk_category"] == "high")
        medium_risk_count = sum(1 for p in predictions if p["risk_category"] == "medium")

        insights = []

        if high_risk_count > 0:
            insights.append(f"{high_risk_count} medication(s) show high adherence risk")

        if medium_risk_count > 1:
            insights.append(f"{medium_risk_count} medications require adherence monitoring")

        overall_risk = sum(p["predicted_adherence"] for p in predictions) / len(predictions)
        if overall_risk > 0.8:
            insights.append("Overall good adherence prognosis")
        else:
            insights.append("Consider comprehensive adherence support program")

        return insights

    def _prioritize_interventions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize adherence interventions by urgency."""

        prioritized = []

        for prediction in predictions:
            if prediction["risk_category"] in ["high", "medium"]:
                priority_score = 1.0 - prediction["predicted_adherence"]

                prioritized.append({
                    "medication_name": prediction["medication_name"],
                    "priority_score": priority_score,
                    "interventions": prediction["interventions"],
                    "urgency": "high" if prediction["risk_category"] == "high" else "medium"
                })

        # Sort by priority score
        prioritized.sort(key=lambda x: x["priority_score"], reverse=True)

        return prioritized

    async def generate_risk_assessment_report(
        self,
        patient_id: int,
        assessment_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive risk assessment report.

        Args:
            patient_id: Patient identifier
            assessment_type: Type of risk assessment

        Returns:
            Detailed risk assessment with stratification
        """
        try:
            logger.info("Generating risk assessment report", patient_id=patient_id, assessment_type=assessment_type)

            # TODO: Get comprehensive patient data from database
            # patient_data = await get_comprehensive_patient_data(patient_id)

            # For now, use mock data
            patient_data = {"id": patient_id}

            # Generate risk scores
            treatment_risk = await self._assess_treatment_risk(patient_data)
            adherence_risk = await self._assess_adherence_risk(patient_data)
            progression_risk = await self._assess_progression_risk(patient_data)

            # Calculate overall risk score
            overall_risk = (treatment_risk["score"] + adherence_risk["score"] + progression_risk["score"]) / 3

            # Generate risk stratification
            risk_stratification = self._stratify_overall_risk(overall_risk)

            return {
                "patient_id": patient_id,
                "assessment_type": assessment_type,
                "assessment_timestamp": datetime.utcnow(),
                "risk_scores": {
                    "treatment_risk": treatment_risk,
                    "adherence_risk": adherence_risk,
                    "progression_risk": progression_risk,
                    "overall_risk": overall_risk
                },
                "risk_stratification": risk_stratification,
                "clinical_recommendations": await self._generate_risk_recommendations(
                    treatment_risk, adherence_risk, progression_risk
                ),
                "monitoring_plan": await self._generate_monitoring_plan(risk_stratification),
                "next_assessment_date": (datetime.utcnow() + timedelta(days=90)).isoformat()
            }

        except Exception as e:
            logger.error("Error generating risk assessment", error=str(e))
            return {"error": "Failed to generate risk assessment"}

    async def _assess_treatment_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess treatment-related risks."""

        # Simplified risk assessment
        return {
            "score": 0.6,
            "level": "medium",
            "factors": ["Age > 65", "Comorbidity burden"],
            "confidence": 0.75
        }

    async def _assess_adherence_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess medication adherence risks."""

        return {
            "score": 0.4,
            "level": "low",
            "factors": ["Simple regimen", "Good historical adherence"],
            "confidence": 0.8
        }

    async def _assess_progression_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess disease progression risks."""

        return {
            "score": 0.7,
            "level": "high",
            "factors": ["Advanced stage", "Aggressive histology"],
            "confidence": 0.7
        }

    def _stratify_overall_risk(self, overall_risk: float) -> Dict[str, Any]:
        """Stratify overall risk level."""

        if overall_risk > 0.7:
            return {
                "level": "high",
                "category": "High Risk",
                "description": "Requires intensive monitoring and intervention",
                "color_code": "red"
            }
        elif overall_risk > 0.4:
            return {
                "level": "medium",
                "category": "Moderate Risk",
                "description": "Requires regular monitoring",
                "color_code": "yellow"
            }
        else:
            return {
                "level": "low",
                "category": "Low Risk",
                "description": "Standard monitoring sufficient",
                "color_code": "green"
            }

    async def _generate_risk_recommendations(
        self,
        treatment_risk: Dict[str, Any],
        adherence_risk: Dict[str, Any],
        progression_risk: Dict[str, Any]
    ) -> List[str]:
        """Generate clinical recommendations based on risk assessment."""

        recommendations = []

        if treatment_risk["level"] == "high":
            recommendations.append("Consider dose modifications for elderly patients")

        if adherence_risk["level"] in ["medium", "high"]:
            recommendations.append("Implement adherence monitoring program")

        if progression_risk["level"] == "high":
            recommendations.append("Consider early alternative treatment options")

        return recommendations

    async def _generate_monitoring_plan(self, risk_stratification: Dict[str, Any]) -> Dict[str, Any]:
        """Generate monitoring plan based on risk level."""

        monitoring_frequency = {
            "high": "weekly",
            "medium": "monthly",
            "low": "quarterly"
        }

        return {
            "frequency": monitoring_frequency.get(risk_stratification["level"], "monthly"),
            "parameters": ["symptoms", "medication_adherence", "treatment_response"],
            "alert_thresholds": {
                "symptom_worsening": "immediate",
                "missed_doses": ">2 per week",
                "treatment_ineffectiveness": "after 2 cycles"
            }
        }


# Global predictive analytics service instance
predictive_analytics_service = PredictiveAnalyticsService()
