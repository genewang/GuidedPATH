"""
AI-powered symptom checker and triage service
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import json

from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
import structlog

from backend.core.config import settings
from backend.apps.symptoms.models import SymptomReport, TriageResult, SymptomSeverity, TriageUrgency, SymptomCategory

logger = structlog.get_logger()


class SymptomCheckerService:
    """
    AI-powered symptom checker and triage service.
    """

    def __init__(self):
        """Initialize the symptom checker service."""
        self.embedding_model = SentenceTransformer(settings.DEFAULT_EMBEDDING_MODEL)

        # Initialize medical NLP models
        self.symptom_classifier = None
        self.triage_model = None
        self.severity_assessor = None

        # Load models (in production, these would be fine-tuned medical models)
        self._load_models()

    def _load_models(self):
        """Load pre-trained models for symptom analysis."""
        try:
            # Load symptom classification model
            # In production, use a fine-tuned BioBERT or similar model
            self.symptom_classifier = pipeline(
                "text-classification",
                model="microsoft/BioGPT",
                device=0 if torch.cuda.is_available() else -1
            )

            logger.info("Symptom checker models loaded successfully")

        except Exception as e:
            logger.error("Failed to load symptom checker models", error=str(e))

    async def analyze_symptoms(self, symptoms: List[str], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze symptoms and provide triage recommendations.

        Args:
            symptoms: List of symptom descriptions
            context: Additional patient context (age, conditions, etc.)

        Returns:
            Comprehensive symptom analysis with triage recommendations
        """
        try:
            logger.info("Analyzing symptoms", symptom_count=len(symptoms))

            # Extract symptom entities
            symptom_entities = await self._extract_symptom_entities(symptoms)

            # Classify symptom categories and severity
            categories = await self._classify_symptoms(symptom_entities)

            # Assess urgency level
            urgency_assessment = await self._assess_urgency(symptom_entities, categories, context)

            # Generate triage recommendations
            triage_recommendations = await self._generate_triage_recommendations(
                urgency_assessment, symptom_entities, context
            )

            # Identify potential conditions
            potential_conditions = await self._identify_conditions(symptom_entities, context)

            # Create comprehensive analysis
            analysis = {
                "symptoms_analyzed": symptoms,
                "symptom_entities": symptom_entities,
                "categories": categories,
                "urgency_assessment": urgency_assessment,
                "triage_recommendations": triage_recommendations,
                "potential_conditions": potential_conditions,
                "confidence_score": urgency_assessment["confidence"],
                "analyzed_at": datetime.utcnow(),
                "requires_immediate_attention": urgency_assessment["urgency"] in ["emergency", "immediate"]
            }

            # Log analysis for quality improvement
            await self._log_symptom_analysis(analysis)

            logger.info("Symptom analysis completed", urgency=urgency_assessment["urgency"])
            return analysis

        except Exception as e:
            logger.error("Error analyzing symptoms", error=str(e))
            return {
                "error": "Failed to analyze symptoms",
                "recommendation": "Please consult with a healthcare provider for symptom evaluation"
            }

    async def _extract_symptom_entities(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """Extract medical entities from symptom descriptions."""
        entities = []

        for symptom in symptoms:
            # Use NLP to extract medical entities
            # In production, use a medical NER model like BioBERT
            entity = {
                "original_text": symptom,
                "category": await self._categorize_symptom(symptom),
                "severity_indicators": await self._extract_severity_indicators(symptom),
                "duration_indicators": await self._extract_duration_indicators(symptom),
                "context_words": await self._extract_context_words(symptom)
            }
            entities.append(entity)

        return entities

    async def _categorize_symptom(self, symptom: str) -> str:
        """Categorize symptom into medical categories."""
        symptom_lower = symptom.lower()

        # Simple rule-based categorization (in production, use ML model)
        category_map = {
            "pain": SymptomCategory.MUSCULOSKELETAL,
            "chest": SymptomCategory.CARDIOVASCULAR,
            "breath": SymptomCategory.RESPIRATORY,
            "headache": SymptomCategory.NEUROLOGICAL,
            "nausea": SymptomCategory.GASTROINTESTINAL,
            "rash": SymptomCategory.DERMATOLOGICAL,
            "anxiety": SymptomCategory.PSYCHIATRIC,
            "fatigue": SymptomCategory.GENERAL
        }

        for keyword, category in category_map.items():
            if keyword in symptom_lower:
                return category.value

        return SymptomCategory.GENERAL.value

    async def _extract_severity_indicators(self, symptom: str) -> List[str]:
        """Extract words indicating symptom severity."""
        severity_words = {
            "severe": ["severe", "extreme", "intense", "terrible", "awful"],
            "moderate": ["moderate", "medium", "fairly", "quite"],
            "mild": ["mild", "slight", "minor", "light"]
        }

        indicators = []
        symptom_lower = symptom.lower()

        for level, words in severity_words.items():
            if any(word in symptom_lower for word in words):
                indicators.append(level)

        return indicators

    async def _extract_duration_indicators(self, symptom: str) -> List[str]:
        """Extract duration information from symptom description."""
        duration_words = [
            "sudden", "acute", "chronic", "ongoing", "persistent",
            "recent", "new", "worsening", "improving"
        ]

        indicators = []
        symptom_lower = symptom.lower()

        for word in duration_words:
            if word in symptom_lower:
                indicators.append(word)

        return indicators

    async def _extract_context_words(self, symptom: str) -> List[str]:
        """Extract medically relevant context words."""
        medical_context_words = [
            "fever", "chills", "sweating", "dizziness", "nausea", "vomiting",
            "diarrhea", "constipation", "bleeding", "bruising", "swelling",
            "numbness", "tingling", "weakness", "fatigue", "shortness of breath"
        ]

        context_words = []
        symptom_lower = symptom.lower()

        for word in medical_context_words:
            if word in symptom_lower:
                context_words.append(word)

        return context_words

    async def _classify_symptoms(self, symptom_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify symptoms and determine overall severity."""
        categories = {}
        severity_scores = []

        for entity in symptom_entities:
            category = entity["category"]
            severity_indicators = entity["severity_indicators"]

            if category not in categories:
                categories[category] = []

            categories[category].append(entity)

            # Calculate severity score
            if "severe" in severity_indicators:
                severity_scores.append(3)
            elif "moderate" in severity_indicators:
                severity_scores.append(2)
            elif "mild" in severity_indicators:
                severity_scores.append(1)
            else:
                severity_scores.append(2)  # Default to moderate

        # Determine overall severity
        avg_severity = sum(severity_scores) / len(severity_scores) if severity_scores else 2

        if avg_severity >= 2.5:
            overall_severity = "severe"
        elif avg_severity >= 1.5:
            overall_severity = "moderate"
        else:
            overall_severity = "mild"

        return {
            "categories": categories,
            "overall_severity": overall_severity,
            "severity_score": avg_severity,
            "symptom_count": len(symptom_entities)
        }

    async def _assess_urgency(self, symptom_entities: List[Dict[str, Any]], categories: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Assess the urgency level of symptoms."""

        # Emergency indicators
        emergency_indicators = [
            "chest pain", "severe shortness of breath", "sudden weakness",
            "severe headache", "vision loss", "seizure", "severe bleeding",
            "suicidal thoughts", "overdose"
        ]

        # Urgent indicators
        urgent_indicators = [
            "fever over 103", "severe pain", "vomiting blood",
            "severe abdominal pain", "confusion", "severe rash"
        ]

        symptom_text = " ".join([entity["original_text"] for entity in symptom_entities]).lower()

        # Check for emergency indicators
        for indicator in emergency_indicators:
            if indicator in symptom_text:
                return {
                    "urgency": "emergency",
                    "confidence": 0.9,
                    "reasoning": f"Emergency indicator '{indicator}' detected",
                    "red_flags": [indicator]
                }

        # Check for urgent indicators
        for indicator in urgent_indicators:
            if indicator in symptom_text:
                return {
                    "urgency": "urgent",
                    "confidence": 0.8,
                    "reasoning": f"Urgent indicator '{indicator}' detected",
                    "red_flags": [indicator]
                }

        # Consider patient context
        if context:
            if context.get("immunocompromised", False):
                return {
                    "urgency": "urgent",
                    "confidence": 0.7,
                    "reasoning": "Patient is immunocompromised - lower threshold for urgent care",
                    "red_flags": ["immunocompromised"]
                }

            if context.get("age", 0) > 65 or context.get("age", 0) < 2:
                return {
                    "urgency": "urgent",
                    "confidence": 0.6,
                    "reasoning": "Age factor increases urgency threshold",
                    "red_flags": ["age_factor"]
                }

        # Default to routine care
        return {
            "urgency": "routine",
            "confidence": 0.5,
            "reasoning": "No urgent or emergency indicators identified",
            "red_flags": []
        }

    async def _generate_triage_recommendations(self, urgency_assessment: Dict[str, Any], symptom_entities: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate triage recommendations based on urgency assessment."""

        urgency = urgency_assessment["urgency"]

        recommendations = {
            "urgency_level": urgency,
            "recommended_action": "",
            "time_frame": "",
            "self_care_advice": [],
            "when_to_seek_care": [],
            "emergency_indicators": []
        }

        if urgency == "emergency":
            recommendations.update({
                "recommended_action": "Seek emergency medical care immediately",
                "time_frame": "Immediately - call emergency services",
                "emergency_indicators": [
                    "Call 911 or go to nearest emergency room",
                    "Do not drive yourself if symptoms are severe",
                    "If possible, have someone stay with you"
                ]
            })

        elif urgency == "urgent":
            recommendations.update({
                "recommended_action": "Seek medical care within 24 hours",
                "time_frame": "Within 24 hours",
                "when_to_seek_care": [
                    "Contact your primary care provider",
                    "Visit urgent care center",
                    "Consider telemedicine consultation"
                ],
                "self_care_advice": [
                    "Rest and monitor symptoms",
                    "Stay hydrated",
                    "Take prescribed medications as directed"
                ]
            })

        else:  # routine
            recommendations.update({
                "recommended_action": "Monitor symptoms and seek care if they worsen",
                "time_frame": "Within 1-2 weeks if symptoms persist",
                "self_care_advice": [
                    "Rest and self-care measures",
                    "Monitor symptoms for changes",
                    "Maintain regular medications and treatments"
                ],
                "when_to_seek_care": [
                    "If symptoms worsen or new symptoms develop",
                    "If symptoms persist beyond expected duration",
                    "If concerned about symptom progression"
                ]
            })

        return recommendations

    async def _identify_conditions(self, symptom_entities: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Identify potential medical conditions based on symptoms."""

        # This would use a medical knowledge base or ML model
        # For now, return basic condition identification

        potential_conditions = []

        symptom_text = " ".join([entity["original_text"] for entity in symptom_entities]).lower()

        # Simple pattern matching (in production, use sophisticated ML)
        if "chest pain" in symptom_text:
            potential_conditions.append({
                "condition": "Possible cardiac or respiratory issue",
                "likelihood": "medium",
                "requires_evaluation": True,
                "specialist": "Cardiologist or Pulmonologist"
            })

        if "headache" in symptom_text and "severe" in symptom_text:
            potential_conditions.append({
                "condition": "Possible migraine or neurological issue",
                "likelihood": "medium",
                "requires_evaluation": True,
                "specialist": "Neurologist"
            })

        if not potential_conditions:
            potential_conditions.append({
                "condition": "Symptoms require clinical evaluation",
                "likelihood": "unknown",
                "requires_evaluation": True,
                "specialist": "Primary Care Provider"
            })

        return potential_conditions

    async def _log_symptom_analysis(self, analysis: Dict[str, Any]):
        """Log symptom analysis for quality improvement and analytics."""
        # TODO: Implement logging to database
        logger.info("Symptom analysis logged", urgency=analysis["urgency_assessment"]["urgency"])

    async def generate_triage_report(self, symptom_report: SymptomReport) -> TriageResult:
        """Generate a comprehensive triage report for a symptom report."""

        # Analyze the symptoms
        symptoms = [symptom_report.primary_symptom]
        if symptom_report.associated_symptoms:
            symptoms.extend(symptom_report.associated_symptoms)

        analysis = await self.analyze_symptoms(symptoms, {
            "age": 30,  # Would come from patient record
            "conditions": [],  # Would come from patient record
            "medications": symptom_report.current_medications or []
        })

        # Create triage result
        triage_result = TriageResult(
            symptom_report_id=symptom_report.id,
            urgency_level=TriageUrgency(analysis["urgency_assessment"]["urgency"]),
            recommended_action=analysis["triage_recommendations"]["recommended_action"],
            time_frame=analysis["triage_recommendations"]["time_frame"],
            reasoning=analysis["urgency_assessment"]["reasoning"],
            differential_diagnosis=[cond["condition"] for cond in analysis["potential_conditions"]],
            red_flags=analysis["urgency_assessment"]["red_flags"],
            self_care_advice=analysis["triage_recommendations"]["self_care_advice"],
            when_to_seek_care=analysis["triage_recommendations"]["when_to_seek_care"],
            emergency_indicators=analysis["triage_recommendations"]["emergency_indicators"],
            requires_clinician_review=analysis["requires_immediate_attention"],
            ai_model_used=settings.DEFAULT_LLM_MODEL,
            confidence_score=analysis["confidence_score"]
        )

        return triage_result

    async def track_symptom_patterns(self, patient_id: int, symptom_reports: List[SymptomReport]) -> List[Dict[str, Any]]:
        """Track and identify symptom patterns over time."""

        if len(symptom_reports) < 3:
            return []  # Need at least 3 reports to identify patterns

        # Group symptoms by category and time
        symptom_timeline = {}

        for report in symptom_reports:
            date_key = report.reported_at.date().isoformat()
            if date_key not in symptom_timeline:
                symptom_timeline[date_key] = []

            symptom_timeline[date_key].append({
                "symptom": report.primary_symptom,
                "category": report.category.value,
                "severity": report.severity.value
            })

        # Identify patterns (simplified logic)
        patterns = []

        # Check for recurring symptoms
        category_frequency = {}
        for date_reports in symptom_timeline.values():
            for report in date_reports:
                category = report["category"]
                category_frequency[category] = category_frequency.get(category, 0) + 1

        # Identify frequently occurring categories
        for category, frequency in category_frequency.items():
            if frequency >= 3:  # Appears in at least 3 reports
                patterns.append({
                    "pattern_type": "recurring",
                    "category": category,
                    "frequency": frequency,
                    "significance": "medium" if frequency >= 5 else "low"
                })

        return patterns


# Global symptom checker service instance
symptom_checker_service = SymptomCheckerService()
