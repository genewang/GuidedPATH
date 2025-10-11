"""
Advanced personalization engine for adaptive user experiences
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import structlog

from backend.core.config import settings

logger = structlog.get_logger()


class PersonalizationModel(nn.Module):
    """
    Deep learning model for user personalization and preference learning.
    """

    def __init__(self, input_dim=512, hidden_dim=256, num_preferences=10):
        super(PersonalizationModel, self).__init__()

        # User profile encoder
        self.user_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Content encoder
        self.content_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Preference prediction
        self.preference_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_preferences),
            nn.Sigmoid()
        )

        # Attention mechanism for content relevance
        self.attention = nn.MultiheadAttention(hidden_dim // 2, num_heads=4, batch_first=True)

    def forward(self, user_profile, content_features):
        """Forward pass through personalization network."""

        # Encode user profile and content
        user_encoded = self.user_encoder(user_profile)
        content_encoded = self.content_encoder(content_features)

        # Apply attention
        attended_features, _ = self.attention(
            user_encoded.unsqueeze(1),
            content_encoded.unsqueeze(1),
            content_encoded.unsqueeze(1)
        )

        # Predict preferences
        preferences = self.preference_predictor(attended_features.squeeze(1))

        return preferences


class PersonalizationEngine:
    """
    Advanced personalization engine for adaptive healthcare experiences.
    """

    def __init__(self):
        """Initialize personalization engine."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ML Models
        self.personalization_model = self._load_personalization_model()
        self.user_segmentation_model = self._load_segmentation_model()
        self.preference_learning_model = self._load_preference_model()

        # User profiles and preferences
        self.user_profiles = {}
        self.user_preferences = {}
        self.user_behavior_history = {}

        # Content adaptation rules
        self.adaptation_rules = {}

        # Feature extractors
        self.text_encoder = None
        self.behavior_encoder = None

        self._initialize_encoders()

    def _load_personalization_model(self):
        """Load or create personalization model."""
        try:
            model = PersonalizationModel()
            model.to(self.device)
            model.eval()
            return model

        except Exception as e:
            logger.error("Error loading personalization model", error=str(e))
            return None

    def _load_segmentation_model(self):
        """Load user segmentation model."""
        try:
            return KMeans(n_clusters=5, random_state=42)

        except Exception as e:
            logger.error("Error loading segmentation model", error=str(e))
            return None

    def _load_preference_model(self):
        """Load preference learning model."""
        try:
            # In production, use collaborative filtering or deep preference learning
            return {}

        except Exception as e:
            logger.error("Error loading preference model", error=str(e))
            return {}

    def _initialize_encoders(self):
        """Initialize text and behavior encoders."""
        try:
            self.text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.text_encoder.to(self.device)

            logger.info("Personalization encoders initialized")

        except Exception as e:
            logger.error("Error initializing encoders", error=str(e))

    async def create_user_profile(
        self,
        user_id: int,
        initial_preferences: Optional[Dict[str, Any]] = None,
        demographic_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive user profile for personalization.

        Args:
            user_id: User identifier
            initial_preferences: Initial user preferences
            demographic_info: Demographic information

        Returns:
            Created user profile with personalization features
        """
        try:
            logger.info("Creating user profile", user_id=user_id)

            # Extract demographic features
            demographic_features = await self._extract_demographic_features(demographic_info)

            # Initialize preference profile
            preference_profile = await self._initialize_preference_profile(initial_preferences)

            # Create behavioral baseline
            behavioral_baseline = await self._create_behavioral_baseline()

            # Generate user segments
            user_segment = await self._determine_user_segment(demographic_features)

            user_profile = {
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "demographic_features": demographic_features,
                "preference_profile": preference_profile,
                "behavioral_baseline": behavioral_baseline,
                "user_segment": user_segment,
                "personalization_model": "hybrid_v1.0",
                "adaptation_capabilities": [
                    "content_filtering",
                    "communication_style",
                    "information_depth",
                    "timing_preferences",
                    "visual_preferences"
                ],
                "privacy_settings": {
                    "data_sharing": "minimal",
                    "personalization_level": "balanced",
                    "tracking_enabled": True
                }
            }

            # Store user profile
            self.user_profiles[user_id] = user_profile
            self.user_preferences[user_id] = preference_profile

            logger.info("User profile created successfully", user_id=user_id, segment=user_segment)
            return user_profile

        except Exception as e:
            logger.error("Error creating user profile", error=str(e), user_id=user_id)
            return {"error": "Failed to create user profile"}

    async def _extract_demographic_features(self, demographic_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract demographic features for personalization."""

        features = {
            "age_group": "adult",  # Would calculate from birth_date
            "education_level": "unknown",
            "health_literacy": "medium",
            "technology_comfort": "medium",
            "communication_preferences": {
                "preferred_language": "en",
                "communication_style": "balanced",
                "detail_level": "moderate"
            },
            "learning_style": "visual_mixed",
            "decision_making_style": "collaborative"
        }

        if demographic_info:
            # Update with provided information
            features.update(demographic_info)

        return features

    async def _initialize_preference_profile(self, initial_preferences: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Initialize user preference profile."""

        default_preferences = {
            "content_types": {
                "clinical_guidelines": 0.7,
                "clinical_trials": 0.8,
                "medication_info": 0.9,
                "symptom_tracking": 0.6,
                "mental_health": 0.5
            },
            "information_depth": {
                "medical_details": "moderate",
                "technical_terms": "minimal",
                "visual_aids": "preferred"
            },
            "communication_style": {
                "tone": "empathetic",
                "formality": "moderate",
                "length": "concise"
            },
            "timing_preferences": {
                "notification_timing": "morning",
                "frequency": "moderate",
                "urgency_handling": "immediate"
            }
        }

        if initial_preferences:
            # Merge with provided preferences
            for category, prefs in initial_preferences.items():
                if category in default_preferences:
                    default_preferences[category].update(prefs)
                else:
                    default_preferences[category] = prefs

        return default_preferences

    async def _create_behavioral_baseline(self) -> Dict[str, Any]:
        """Create behavioral baseline for new users."""

        return {
            "engagement_patterns": {
                "peak_hours": [9, 10, 14, 15],  # Default business hours
                "session_duration": "medium",
                "interaction_frequency": "daily"
            },
            "content_preferences": {
                "reading_speed": "normal",
                "visual_vs_text": "balanced",
                "interactive_elements": "moderate"
            },
            "decision_factors": {
                "evidence_weight": 0.8,
                "provider_opinion": 0.9,
                "peer_experiences": 0.3,
                "cost_considerations": 0.6
            }
        }

    async def _determine_user_segment(self, demographic_features: Dict[str, Any]) -> str:
        """Determine user segment for personalization."""

        # Simple rule-based segmentation (in production, use clustering)
        age_group = demographic_features.get("age_group", "adult")
        tech_comfort = demographic_features.get("technology_comfort", "medium")

        if tech_comfort == "high" and age_group in ["young_adult", "adult"]:
            return "tech_savvy"
        elif age_group in ["senior", "elderly"]:
            return "traditional"
        else:
            return "balanced"

    async def personalize_content(
        self,
        user_id: int,
        content: Dict[str, Any],
        content_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Personalize content based on user preferences and behavior.

        Args:
            user_id: User identifier
            content: Original content to personalize
            content_type: Type of content (guidelines, trials, medication, etc.)
            context: Additional context for personalization

        Returns:
            Personalized content with adaptations
        """
        try:
            logger.info("Personalizing content", user_id=user_id, content_type=content_type)

            # Get user profile
            user_profile = self.user_profiles.get(user_id)
            if not user_profile:
                # Return original content if no profile exists
                return content

            # Apply personalization adaptations
            personalized_content = await self._apply_content_adaptations(
                content, user_profile, content_type, context
            )

            # Update content relevance score
            relevance_score = await self._calculate_content_relevance(
                user_id, personalized_content, content_type
            )

            # Add personalization metadata
            personalized_content["personalization_metadata"] = {
                "user_id": user_id,
                "content_type": content_type,
                "adaptations_applied": await self._get_applied_adaptations(user_profile, content_type),
                "relevance_score": relevance_score,
                "personalized_at": datetime.utcnow(),
                "model_version": "personalization_v2.1"
            }

            # Log personalization for learning
            await self._log_personalization_event(user_id, content_type, relevance_score)

            return personalized_content

        except Exception as e:
            logger.error("Error personalizing content", error=str(e), user_id=user_id)
            return content  # Return original content on error

    async def _apply_content_adaptations(
        self,
        content: Dict[str, Any],
        user_profile: Dict[str, Any],
        content_type: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply personalization adaptations to content."""

        adapted_content = content.copy()

        # Get user preferences
        preferences = user_profile.get("preference_profile", {})
        user_segment = user_profile.get("user_segment", "balanced")

        # Apply content type specific adaptations
        if content_type == "clinical_guidelines":
            adapted_content = await self._adapt_guideline_content(adapted_content, preferences, user_segment)

        elif content_type == "clinical_trials":
            adapted_content = await self._adapt_trial_content(adapted_content, preferences, user_segment)

        elif content_type == "medication":
            adapted_content = await self._adapt_medication_content(adapted_content, preferences, user_segment)

        elif content_type == "symptoms":
            adapted_content = await self._adapt_symptom_content(adapted_content, preferences, user_segment)

        # Apply general adaptations
        adapted_content = await self._apply_general_adaptations(adapted_content, preferences, context)

        return adapted_content

    async def _adapt_guideline_content(
        self,
        content: Dict[str, Any],
        preferences: Dict[str, Any],
        user_segment: str
    ) -> Dict[str, Any]:
        """Adapt clinical guideline content."""

        # Adjust technical depth based on preferences
        info_depth = preferences.get("information_depth", {})
        technical_level = info_depth.get("technical_terms", "minimal")

        if technical_level == "minimal" and user_segment == "traditional":
            # Simplify technical terms
            content["summary"] = await self._simplify_medical_text(content.get("summary", ""))
            content["key_points"] = await self._simplify_key_points(content.get("key_points", []))

        elif technical_level == "detailed" and user_segment == "tech_savvy":
            # Add more technical details
            content["technical_details"] = await self._add_technical_details(content)

        return content

    async def _adapt_trial_content(
        self,
        content: Dict[str, Any],
        preferences: Dict[str, Any],
        user_segment: str
    ) -> Dict[str, Any]:
        """Adapt clinical trial content."""

        # Adjust based on user risk tolerance and information needs
        communication_style = preferences.get("communication_style", {})

        if communication_style.get("tone") == "conservative":
            # Emphasize risks and uncertainties
            content["risk_emphasis"] = "high"

        if communication_style.get("detail_level") == "comprehensive":
            # Add detailed trial information
            content["detailed_criteria"] = await self._expand_trial_criteria(content)

        return content

    async def _adapt_medication_content(
        self,
        content: Dict[str, Any],
        preferences: Dict[str, Any],
        user_segment: str
    ) -> Dict[str, Any]:
        """Adapt medication information content."""

        # Personalize based on health literacy and preferences
        info_depth = preferences.get("information_depth", {})

        if info_depth.get("medical_details") == "basic":
            # Simplify medication information
            content["instructions"] = await self._simplify_medication_instructions(content.get("instructions", ""))
            content["side_effects"] = await self._simplify_side_effects(content.get("side_effects", []))

        return content

    async def _adapt_symptom_content(
        self,
        content: Dict[str, Any],
        preferences: Dict[str, Any],
        user_segment: str
    ) -> Dict[str, Any]:
        """Adapt symptom assessment content."""

        # Adapt based on user anxiety level and communication style
        communication_style = preferences.get("communication_style", {})

        if communication_style.get("tone") == "reassuring":
            content["reassurance_level"] = "high"
            content["urgency_language"] = "calm"

        return content

    async def _apply_general_adaptations(
        self,
        content: Dict[str, Any],
        preferences: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply general personalization adaptations."""

        # Add visual preferences
        visual_prefs = preferences.get("information_depth", {}).get("visual_aids", "preferred")
        if visual_prefs == "preferred":
            content["include_visuals"] = True

        # Add timing context
        timing_prefs = preferences.get("timing_preferences", {})
        if context and context.get("urgency") == "high":
            content["highlight_urgency"] = True

        return content

    async def _simplify_medical_text(self, text: str) -> str:
        """Simplify medical text for general audiences."""

        # Simple text simplification (in production, use medical text simplification models)
        simplifications = {
            "myocardial infarction": "heart attack",
            "cerebrovascular accident": "stroke",
            "pulmonary embolism": "blood clot in lungs",
            "adenocarcinoma": "type of cancer"
        }

        for medical_term, simple_term in simplifications.items():
            text = text.replace(medical_term, simple_term)

        return text

    async def _simplify_key_points(self, key_points: List[str]) -> List[str]:
        """Simplify key points for better understanding."""

        simplified_points = []

        for point in key_points:
            simplified_point = await self._simplify_medical_text(point)
            simplified_points.append(simplified_point)

        return simplified_points

    async def _add_technical_details(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Add technical details for expert users."""

        technical_info = {
            "statistical_significance": "p < 0.05",
            "confidence_intervals": "95% CI",
            "study_power": "80%"
        }

        content["technical_details"] = technical_info
        return content

    async def _expand_trial_criteria(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Expand trial criteria with detailed information."""

        content["detailed_criteria"] = {
            "inclusion_criteria_detailed": content.get("inclusion_criteria", []),
            "exclusion_criteria_detailed": content.get("exclusion_criteria", []),
            "biomarker_requirements": content.get("biomarker_requirements", [])
        }

        return content

    async def _simplify_medication_instructions(self, instructions: str) -> str:
        """Simplify medication instructions."""

        # Break down complex instructions
        simplified = instructions.replace("Take with food", "Take with meals")
        simplified = simplified.replace("Do not crush or chew", "Swallow whole")

        return simplified

    async def _simplify_side_effects(self, side_effects: List[str]) -> List[str]:
        """Simplify side effect descriptions."""

        simplifications = {
            "nausea and vomiting": "upset stomach",
            "peripheral neuropathy": "numbness or tingling",
            "myelosuppression": "low blood cell counts"
        }

        simplified_effects = []

        for effect in side_effects:
            simplified = simplifications.get(effect.lower(), effect)
            simplified_effects.append(simplified)

        return simplified_effects

    async def _calculate_content_relevance(
        self,
        user_id: int,
        content: Dict[str, Any],
        content_type: str
    ) -> float:
        """Calculate content relevance score for user."""

        # Get user preferences
        user_prefs = self.user_preferences.get(user_id, {})

        # Base relevance on content type preference
        content_type_prefs = user_prefs.get("content_types", {})
        base_relevance = content_type_prefs.get(content_type, 0.5)

        # Adjust based on content characteristics
        content_complexity = await self._assess_content_complexity(content)

        # Users with lower health literacy prefer simpler content
        user_literacy = user_prefs.get("information_depth", {}).get("technical_terms", "minimal")

        if user_literacy == "minimal" and content_complexity == "high":
            base_relevance *= 0.8
        elif user_literacy == "detailed" and content_complexity == "low":
            base_relevance *= 0.9

        return min(1.0, base_relevance)

    async def _assess_content_complexity(self, content: Dict[str, Any]) -> str:
        """Assess content complexity level."""

        # Simple complexity assessment based on content length and technical terms
        text_length = len(str(content))
        technical_terms = ["statistically significant", "confidence interval", "p-value", "efficacy"]

        technical_count = sum(1 for term in technical_terms if term.lower() in str(content).lower())

        if text_length > 2000 or technical_count > 2:
            return "high"
        elif text_length > 1000 or technical_count > 0:
            return "medium"
        else:
            return "low"

    async def _get_applied_adaptations(
        self,
        user_profile: Dict[str, Any],
        content_type: str
    ) -> List[str]:
        """Get list of adaptations applied to content."""

        adaptations = []

        preferences = user_profile.get("preference_profile", {})
        user_segment = user_profile.get("user_segment", "balanced")

        # Content type specific adaptations
        if content_type == "clinical_guidelines":
            if preferences.get("information_depth", {}).get("technical_terms") == "minimal":
                adaptations.append("technical_simplification")

        if content_type == "medication":
            if preferences.get("information_depth", {}).get("medical_details") == "basic":
                adaptations.append("instruction_simplification")

        # General adaptations
        if user_segment == "traditional":
            adaptations.append("traditional_formatting")

        if preferences.get("information_depth", {}).get("visual_aids") == "preferred":
            adaptations.append("visual_enhancement")

        return adaptations

    async def _log_personalization_event(self, user_id: int, content_type: str, relevance_score: float):
        """Log personalization event for learning and improvement."""

        # Store in user behavior history
        if user_id not in self.user_behavior_history:
            self.user_behavior_history[user_id] = []

        event = {
            "timestamp": datetime.utcnow(),
            "content_type": content_type,
            "relevance_score": relevance_score,
            "user_segment": self.user_profiles.get(user_id, {}).get("user_segment", "unknown")
        }

        self.user_behavior_history[user_id].append(event)

        # Keep only recent events (last 100)
        if len(self.user_behavior_history[user_id]) > 100:
            self.user_behavior_history[user_id] = self.user_behavior_history[user_id][-100:]

    async def update_user_preferences(
        self,
        user_id: int,
        interaction_data: Dict[str, Any],
        feedback_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Update user preferences based on interaction patterns.

        Args:
            user_id: User identifier
            interaction_data: Recent interaction patterns
            feedback_score: Explicit feedback score (0-1)

        Returns:
            Updated preference profile
        """
        try:
            logger.info("Updating user preferences", user_id=user_id)

            if user_id not in self.user_profiles:
                return {"error": "User profile not found"}

            # Analyze interaction patterns
            behavior_insights = await self._analyze_behavior_patterns(user_id, interaction_data)

            # Update preferences based on insights
            updated_preferences = await self._update_preferences_from_behavior(
                self.user_preferences[user_id], behavior_insights, feedback_score
            )

            # Update user profile
            self.user_profiles[user_id]["preference_profile"] = updated_preferences
            self.user_profiles[user_id]["last_updated"] = datetime.utcnow()

            # Store updated preferences
            self.user_preferences[user_id] = updated_preferences

            logger.info("User preferences updated", user_id=user_id, insights_count=len(behavior_insights))
            return {
                "user_id": user_id,
                "updated_preferences": updated_preferences,
                "behavior_insights": behavior_insights,
                "updated_at": datetime.utcnow()
            }

        except Exception as e:
            logger.error("Error updating user preferences", error=str(e), user_id=user_id)
            return {"error": "Failed to update user preferences"}

    async def _analyze_behavior_patterns(self, user_id: int, interaction_data: Dict[str, Any]) -> List[str]:
        """Analyze user behavior patterns for preference updates."""

        insights = []

        # Analyze content engagement
        engagement = interaction_data.get("engagement", {})
        if engagement.get("average_session_time", 0) > 600:  # 10 minutes
            insights.append("prefers_detailed_content")
        elif engagement.get("average_session_time", 0) < 120:  # 2 minutes
            insights.append("prefers_concise_content")

        # Analyze content type preferences
        content_prefs = interaction_data.get("content_preferences", {})
        most_viewed = content_prefs.get("most_viewed_type", "")
        if most_viewed:
            insights.append(f"prefers_{most_viewed}_content")

        # Analyze timing patterns
        timing = interaction_data.get("timing_patterns", {})
        if timing.get("peak_usage_hour", 0) in [6, 7, 8]:
            insights.append("morning_person")
        elif timing.get("peak_usage_hour", 0) in [21, 22, 23]:
            insights.append("evening_person")

        return insights

    async def _update_preferences_from_behavior(
        self,
        current_preferences: Dict[str, Any],
        behavior_insights: List[str],
        feedback_score: Optional[float]
    ) -> Dict[str, Any]:
        """Update preferences based on behavior insights."""

        updated_preferences = current_preferences.copy()

        # Apply insights to preference updates
        for insight in behavior_insights:
            if insight == "prefers_detailed_content":
                updated_preferences["information_depth"]["medical_details"] = "comprehensive"
                updated_preferences["communication_style"]["length"] = "detailed"

            elif insight == "prefers_concise_content":
                updated_preferences["information_depth"]["medical_details"] = "basic"
                updated_preferences["communication_style"]["length"] = "concise"

            elif insight == "prefers_medication_content":
                updated_preferences["content_types"]["medication_info"] = min(1.0, updated_preferences["content_types"].get("medication_info", 0.5) + 0.1)

            elif insight == "morning_person":
                updated_preferences["timing_preferences"]["notification_timing"] = "morning"

            elif insight == "evening_person":
                updated_preferences["timing_preferences"]["notification_timing"] = "evening"

        # Incorporate explicit feedback if provided
        if feedback_score is not None:
            # Adjust content type preferences based on feedback
            for content_type in updated_preferences["content_types"]:
                if feedback_score > 0.8:
                    updated_preferences["content_types"][content_type] = min(1.0, updated_preferences["content_types"][content_type] + 0.05)
                elif feedback_score < 0.4:
                    updated_preferences["content_types"][content_type] = max(0.0, updated_preferences["content_types"][content_type] - 0.05)

        return updated_preferences

    async def generate_personalized_recommendations(
        self,
        user_id: int,
        available_content: List[Dict[str, Any]],
        recommendation_type: str = "comprehensive"
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized content recommendations.

        Args:
            user_id: User identifier
            available_content: Available content items
            recommendation_type: Type of recommendations

        Returns:
            Ranked list of personalized recommendations
        """
        try:
            logger.info("Generating personalized recommendations", user_id=user_id, content_count=len(available_content))

            if user_id not in self.user_profiles:
                # Return generic ranking if no profile
                return sorted(available_content, key=lambda x: x.get("relevance_score", 0.5), reverse=True)

            user_profile = self.user_profiles[user_id]
            user_prefs = user_profile.get("preference_profile", {})

            # Score each content item
            scored_content = []

            for content in available_content:
                content_type = content.get("content_type", "general")

                # Calculate personalized relevance score
                relevance_score = await self._calculate_content_relevance(user_id, content, content_type)

                # Apply user segment adjustments
                segment_multiplier = await self._get_segment_multiplier(user_profile.get("user_segment", "balanced"), content_type)
                relevance_score *= segment_multiplier

                # Apply timing preferences
                timing_bonus = await self._calculate_timing_bonus(user_id, content)
                relevance_score += timing_bonus

                scored_content.append({
                    "content": content,
                    "relevance_score": relevance_score,
                    "personalization_factors": {
                        "content_type_preference": user_prefs.get("content_types", {}).get(content_type, 0.5),
                        "segment_multiplier": segment_multiplier,
                        "timing_bonus": timing_bonus
                    }
                })

            # Sort by relevance score
            ranked_content = sorted(scored_content, key=lambda x: x["relevance_score"], reverse=True)

            # Return top recommendations with explanations
            recommendations = []
            for item in ranked_content[:10]:  # Top 10
                recommendation = item["content"].copy()
                recommendation["personalization_score"] = item["relevance_score"]
                recommendation["recommendation_factors"] = item["personalization_factors"]
                recommendation["recommendation_timestamp"] = datetime.utcnow()

                recommendations.append(recommendation)

            logger.info("Personalized recommendations generated", user_id=user_id, recommendation_count=len(recommendations))
            return recommendations

        except Exception as e:
            logger.error("Error generating personalized recommendations", error=str(e), user_id=user_id)
            return []

    async def _get_segment_multiplier(self, user_segment: str, content_type: str) -> float:
        """Get content multiplier based on user segment."""

        segment_multipliers = {
            "tech_savvy": {
                "clinical_guidelines": 1.2,
                "clinical_trials": 1.3,
                "medication_info": 1.1,
                "symptom_tracking": 1.0
            },
            "traditional": {
                "clinical_guidelines": 0.9,
                "clinical_trials": 0.8,
                "medication_info": 1.2,
                "symptom_tracking": 1.1
            },
            "balanced": {
                "clinical_guidelines": 1.0,
                "clinical_trials": 1.0,
                "medication_info": 1.0,
                "symptom_tracking": 1.0
            }
        }

        return segment_multipliers.get(user_segment, segment_multipliers["balanced"]).get(content_type, 1.0)

    async def _calculate_timing_bonus(self, user_id: int, content: Dict[str, Any]) -> float:
        """Calculate timing-based relevance bonus."""

        # Simple timing bonus based on current time
        current_hour = datetime.now().hour

        # Morning content bonus for morning people
        if 6 <= current_hour <= 12:
            if content.get("content_type") in ["medication_info", "symptom_tracking"]:
                return 0.1

        # Evening content bonus for evening people
        if 18 <= current_hour <= 23:
            if content.get("content_type") in ["mental_health", "clinical_guidelines"]:
                return 0.1

        return 0.0

    async def adapt_communication_style(
        self,
        user_id: int,
        message_content: str,
        target_audience: str = "patient"
    ) -> Dict[str, Any]:
        """
        Adapt communication style based on user preferences.

        Args:
            user_id: User identifier
            message_content: Original message content
            target_audience: Target audience type

        Returns:
            Adapted message with personalized communication style
        """
        try:
            user_profile = self.user_profiles.get(user_id)
            if not user_profile:
                return {"original_message": message_content, "adapted_message": message_content}

            preferences = user_profile.get("preference_profile", {})

            # Get communication style preferences
            comm_style = preferences.get("communication_style", {})

            # Adapt tone
            adapted_message = await self._adapt_message_tone(message_content, comm_style.get("tone", "empathetic"))

            # Adapt formality
            adapted_message = await self._adapt_message_formality(adapted_message, comm_style.get("formality", "moderate"))

            # Adapt length
            adapted_message = await self._adapt_message_length(adapted_message, comm_style.get("length", "concise"))

            return {
                "original_message": message_content,
                "adapted_message": adapted_message,
                "communication_style_applied": comm_style,
                "adaptations_made": await self._get_communication_adaptations(comm_style),
                "adapted_at": datetime.utcnow()
            }

        except Exception as e:
            logger.error("Error adapting communication style", error=str(e), user_id=user_id)
            return {"original_message": message_content, "adapted_message": message_content}

    async def _adapt_message_tone(self, message: str, tone: str) -> str:
        """Adapt message tone based on user preference."""

        if tone == "empathetic":
            # Add empathetic language
            if not any(word in message.lower() for word in ["sorry", "understand", "support"]):
                message = f"I understand this may be difficult. {message}"

        elif tone == "professional":
            # Ensure professional language
            message = message.replace("Hey", "Hello").replace("Yeah", "Yes")

        elif tone == "friendly":
            # Add friendly touches
            if not message.startswith(("Hi", "Hello", "Hey")):
                message = f"Hey! {message}"

        return message

    async def _adapt_message_formality(self, message: str, formality: str) -> str:
        """Adapt message formality level."""

        if formality == "formal":
            # Increase formality
            message = message.replace("you can", "you may")
            message = message.replace("let's", "let us")

        elif formality == "casual":
            # Increase casualness
            message = message.replace("you may", "you can")
            message = message.replace("please note", "just so you know")

        return message

    async def _adapt_message_length(self, message: str, length_preference: str) -> str:
        """Adapt message length based on preference."""

        words = message.split()

        if length_preference == "concise" and len(words) > 50:
            # Shorten message
            return " ".join(words[:30]) + "..."

        elif length_preference == "detailed" and len(words) < 20:
            # Expand message (in production, use content expansion)
            return message + " Please let me know if you need more details."

        return message

    async def _get_communication_adaptations(self, comm_style: Dict[str, Any]) -> List[str]:
        """Get list of communication adaptations applied."""

        adaptations = []

        if comm_style.get("tone"):
            adaptations.append(f"tone_{comm_style['tone']}")

        if comm_style.get("formality"):
            adaptations.append(f"formality_{comm_style['formality']}")

        if comm_style.get("length"):
            adaptations.append(f"length_{comm_style['length']}")

        return adaptations


# Global personalization engine instance
personalization_engine = PersonalizationEngine()
