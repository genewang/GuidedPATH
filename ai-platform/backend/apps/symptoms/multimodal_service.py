"""
Multimodal AI-powered symptom checker service
"""

import asyncio
import logging
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import io

import torch
import torchaudio
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
import whisper
import structlog

from backend.core.config import settings
from backend.apps.symptoms.models import SymptomReport, TriageResult, VitalSigns

logger = structlog.get_logger()


class MultimodalSymptomService:
    """
    Multimodal AI service for symptom analysis using text, voice, and image inputs.
    """

    def __init__(self):
        """Initialize multimodal symptom analysis service."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Text analysis models
        self.symptom_classifier = None
        self.severity_assessor = None

        # Voice analysis models
        self.whisper_model = None
        self.audio_classifier = None

        # Image analysis models
        self.image_processor = None
        self.skin_lesion_model = None
        self.wound_assessment_model = None

        # Multimodal fusion model
        self.multimodal_fusion_model = None

        self._load_multimodal_models()

    def _load_multimodal_models(self):
        """Load all multimodal AI models."""
        try:
            # Load Whisper for voice transcription
            if settings.ENABLE_VOICE_FEATURES:
                self.whisper_model = whisper.load_model("base", device=self.device)

            # Load image classification models
            if settings.ENABLE_VIDEO_FEATURES:
                self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
                self.skin_lesion_model = AutoModelForImageClassification.from_pretrained(
                    "microsoft/skin-lesion-classification"
                )

            # Load audio emotion classifier
            self.audio_classifier = pipeline(
                "audio-classification",
                model="MIT/ast-finetuned-audioset-10-10-0.4593",
                device=0 if torch.cuda.is_available() else -1
            )

            logger.info("Multimodal models loaded successfully")

        except Exception as e:
            logger.error("Error loading multimodal models", error=str(e))

    async def analyze_multimodal_symptoms(
        self,
        text_input: Optional[str] = None,
        voice_input: Optional[bytes] = None,
        image_input: Optional[bytes] = None,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze symptoms using multiple input modalities.

        Args:
            text_input: Text description of symptoms
            voice_input: Audio recording of symptom description
            image_input: Image of affected area (rash, wound, etc.)
            patient_context: Patient medical context

        Returns:
            Comprehensive multimodal symptom analysis
        """
        try:
            logger.info("Starting multimodal symptom analysis")

            # Process each input modality
            text_analysis = await self._analyze_text_symptoms(text_input) if text_input else {}
            voice_analysis = await self._analyze_voice_symptoms(voice_input) if voice_input else {}
            image_analysis = await self._analyze_image_symptoms(image_input) if image_input else {}

            # Fuse multimodal information
            fused_analysis = await self._fuse_multimodal_data(
                text_analysis, voice_analysis, image_analysis, patient_context
            )

            # Generate enhanced triage recommendations
            triage_result = await self._generate_multimodal_triage(fused_analysis)

            # Extract vital signs from voice/image if available
            vital_signs = await self._extract_vital_signs(voice_analysis, image_analysis)

            return {
                "analysis_timestamp": datetime.utcnow(),
                "modalities_analyzed": {
                    "text": text_input is not None,
                    "voice": voice_input is not None,
                    "image": image_input is not None
                },
                "text_analysis": text_analysis,
                "voice_analysis": voice_analysis,
                "image_analysis": image_analysis,
                "fused_analysis": fused_analysis,
                "triage_result": triage_result,
                "vital_signs": vital_signs,
                "confidence_score": fused_analysis.get("overall_confidence", 0.5),
                "requires_immediate_attention": triage_result.get("urgency") in ["emergency", "immediate"]
            }

        except Exception as e:
            logger.error("Error in multimodal symptom analysis", error=str(e))
            return {
                "error": "Failed to analyze symptoms",
                "recommendation": "Please consult with a healthcare provider for symptom evaluation"
            }

    async def _analyze_text_symptoms(self, text_input: str) -> Dict[str, Any]:
        """Analyze text-based symptom descriptions."""

        # Extract symptom entities from text
        symptoms = await self._extract_symptoms_from_text(text_input)

        # Classify symptom severity
        severity = await self._classify_text_severity(text_input)

        # Identify potential conditions
        potential_conditions = await self._identify_conditions_from_text(text_input)

        return {
            "input_text": text_input,
            "extracted_symptoms": symptoms,
            "severity_assessment": severity,
            "potential_conditions": potential_conditions,
            "confidence": 0.85
        }

    async def _extract_symptoms_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract symptom entities from text using NLP."""

        # Simple keyword-based extraction (in production, use medical NER)
        symptom_keywords = [
            "pain", "headache", "fever", "nausea", "fatigue", "dizziness",
            "shortness of breath", "chest pain", "abdominal pain", "rash"
        ]

        extracted_symptoms = []

        for keyword in symptom_keywords:
            if keyword in text.lower():
                extracted_symptoms.append({
                    "symptom": keyword,
                    "confidence": 0.8,
                    "context": text
                })

        return extracted_symptoms

    async def _classify_text_severity(self, text: str) -> Dict[str, Any]:
        """Classify symptom severity from text description."""

        severity_indicators = {
            "critical": ["severe", "extreme", "unbearable", "emergency"],
            "high": ["bad", "terrible", "intense", "acute"],
            "moderate": ["moderate", "fairly bad", "noticeable"],
            "mild": ["mild", "slight", "minor"]
        }

        text_lower = text.lower()

        for level, indicators in severity_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return {
                    "level": level,
                    "confidence": 0.7,
                    "indicators": indicators
                }

        return {"level": "moderate", "confidence": 0.5, "indicators": []}

    async def _identify_conditions_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Identify potential medical conditions from text."""

        # Simple pattern matching (in production, use medical knowledge base)
        condition_patterns = {
            "migraine": ["headache", "nausea", "sensitivity to light"],
            "flu": ["fever", "fatigue", "body aches"],
            "anxiety": ["racing heart", "sweating", "worried"]
        }

        potential_conditions = []

        for condition, symptoms in condition_patterns.items():
            if any(symptom in text.lower() for symptom in symptoms):
                potential_conditions.append({
                    "condition": condition,
                    "confidence": 0.6,
                    "matching_symptoms": symptoms
                })

        return potential_conditions

    async def _analyze_voice_symptoms(self, voice_data: bytes) -> Dict[str, Any]:
        """Analyze voice input for symptom assessment."""

        if not self.whisper_model:
            return {"error": "Voice analysis not available"}

        try:
            # Save audio data temporarily
            audio_path = f"/tmp/voice_input_{datetime.now().timestamp()}.wav"

            with open(audio_path, "wb") as f:
                f.write(voice_data)

            # Transcribe audio
            transcription_result = self.whisper_model.transcribe(audio_path)

            # Analyze emotional tone
            emotion_analysis = await self._analyze_voice_emotion(audio_path)

            # Extract speech characteristics
            speech_features = await self._extract_speech_features(audio_path)

            return {
                "transcription": transcription_result["text"],
                "confidence": transcription_result["avg_logprob"],
                "emotion_analysis": emotion_analysis,
                "speech_features": speech_features,
                "duration_seconds": transcription_result.get("duration", 0)
            }

        except Exception as e:
            logger.error("Error analyzing voice input", error=str(e))
            return {"error": "Failed to analyze voice input"}

    async def _analyze_voice_emotion(self, audio_path: str) -> Dict[str, Any]:
        """Analyze emotional tone from voice."""

        try:
            if self.audio_classifier:
                results = self.audio_classifier(audio_path)

                # Map audio classifications to emotions
                emotion_map = {
                    "speech": "neutral",
                    "music": "neutral",
                    "silence": "calm"
                }

                top_emotion = results[0]["label"]
                confidence = results[0]["score"]

                return {
                    "primary_emotion": emotion_map.get(top_emotion, "neutral"),
                    "confidence": confidence,
                    "all_emotions": results[:3]  # Top 3 emotions
                }

            return {"primary_emotion": "neutral", "confidence": 0.5}

        except Exception as e:
            logger.error("Error analyzing voice emotion", error=str(e))
            return {"error": "Failed to analyze voice emotion"}

    async def _extract_speech_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract speech characteristics for clinical assessment."""

        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_path)

            # Calculate basic audio features
            duration = waveform.shape[1] / sample_rate

            # Simple speech rate estimation (words per minute)
            # In production, use more sophisticated speech analysis
            estimated_words = duration * 2  # Rough estimate
            speech_rate = estimated_words / (duration / 60) if duration > 0 else 0

            return {
                "duration_seconds": duration,
                "sample_rate": sample_rate,
                "estimated_speech_rate": speech_rate,
                "volume_level": "normal",  # Would analyze actual volume
                "clarity": "good"  # Would assess speech clarity
            }

        except Exception as e:
            logger.error("Error extracting speech features", error=str(e))
            return {"error": "Failed to extract speech features"}

    async def _analyze_image_symptoms(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze medical images for symptom assessment."""

        if not self.image_processor or not self.skin_lesion_model:
            return {"error": "Image analysis not available"}

        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))

            # Preprocess image
            inputs = self.image_processor(image, return_tensors="pt")

            # Run classification
            with torch.no_grad():
                outputs = self.skin_lesion_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Get top predictions
            top_predictions = []
            for i in range(min(3, len(predictions[0]))):
                confidence = predictions[0][i].item()
                top_predictions.append({
                    "prediction": f"class_{i}",  # Would map to actual condition names
                    "confidence": confidence
                })

            return {
                "image_size": image.size,
                "image_format": image.format,
                "predictions": top_predictions,
                "analysis_type": "skin_lesion",  # Would detect image type automatically
                "confidence": top_predictions[0]["confidence"] if top_predictions else 0.0
            }

        except Exception as e:
            logger.error("Error analyzing image input", error=str(e))
            return {"error": "Failed to analyze image input"}

    async def _fuse_multimodal_data(
        self,
        text_analysis: Dict[str, Any],
        voice_analysis: Dict[str, Any],
        image_analysis: Dict[str, Any],
        patient_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fuse information from multiple modalities."""

        fused_data = {
            "symptoms": [],
            "severity_factors": [],
            "contextual_factors": [],
            "overall_confidence": 0.5
        }

        # Fuse symptom information
        if text_analysis and "extracted_symptoms" in text_analysis:
            fused_data["symptoms"].extend(text_analysis["extracted_symptoms"])

        if voice_analysis and "transcription" in voice_analysis:
            # Extract symptoms from voice transcription
            voice_symptoms = await self._extract_symptoms_from_text(voice_analysis["transcription"])
            fused_data["symptoms"].extend(voice_symptoms)

        # Fuse severity information
        if text_analysis and "severity_assessment" in text_analysis:
            fused_data["severity_factors"].append(text_analysis["severity_assessment"])

        if voice_analysis and "emotion_analysis" in voice_analysis:
            emotion = voice_analysis["emotion_analysis"].get("primary_emotion", "neutral")
            if emotion in ["agitated", "distressed"]:
                fused_data["severity_factors"].append({
                    "type": "voice_emotion",
                    "level": "high",
                    "indicator": emotion
                })

        # Add image-based findings
        if image_analysis and "predictions" in image_analysis:
            for prediction in image_analysis["predictions"]:
                if prediction["confidence"] > 0.7:
                    fused_data["symptoms"].append({
                        "symptom": f"visual_finding_{prediction['prediction']}",
                        "confidence": prediction["confidence"],
                        "modality": "image"
                    })

        # Calculate overall confidence
        confidence_factors = []
        if text_analysis:
            confidence_factors.append(text_analysis.get("confidence", 0.5))
        if voice_analysis and "error" not in voice_analysis:
            confidence_factors.append(voice_analysis.get("confidence", 0.5))
        if image_analysis and "error" not in image_analysis:
            confidence_factors.append(image_analysis.get("confidence", 0.5))

        fused_data["overall_confidence"] = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

        return fused_data

    async def _generate_multimodal_triage(self, fused_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate triage recommendations based on multimodal analysis."""

        # Enhanced triage logic considering multiple modalities
        urgency = "routine"
        confidence = fused_analysis.get("overall_confidence", 0.5)

        # Check for high-severity indicators across modalities
        severity_indicators = fused_analysis.get("severity_factors", [])

        for indicator in severity_indicators:
            if indicator.get("level") in ["critical", "high"]:
                urgency = "urgent"
                break

        # Check for emergency indicators
        symptoms = fused_analysis.get("symptoms", [])
        emergency_keywords = ["chest pain", "severe shortness of breath", "seizure", "severe bleeding"]

        for symptom in symptoms:
            symptom_text = symptom.get("symptom", "").lower()
            if any(keyword in symptom_text for keyword in emergency_keywords):
                urgency = "emergency"
                break

        # Generate triage recommendations
        triage_recommendations = {
            "urgency_level": urgency,
            "confidence": confidence,
            "recommended_action": self._get_urgency_action(urgency),
            "time_frame": self._get_urgency_timeframe(urgency),
            "monitoring_instructions": await self._get_monitoring_instructions(urgency, symptoms),
            "follow_up_recommendations": await self._get_follow_up_recommendations(urgency)
        }

        return triage_recommendations

    def _get_urgency_action(self, urgency: str) -> str:
        """Get recommended action based on urgency level."""
        actions = {
            "emergency": "Seek emergency medical care immediately",
            "urgent": "Seek medical care within 24 hours",
            "routine": "Monitor symptoms and seek care if they worsen"
        }
        return actions.get(urgency, actions["routine"])

    def _get_urgency_timeframe(self, urgency: str) -> str:
        """Get recommended timeframe for care."""
        timeframes = {
            "emergency": "Immediately - call emergency services",
            "urgent": "Within 24 hours",
            "routine": "Within 1-2 weeks if symptoms persist"
        }
        return timeframes.get(urgency, timeframes["routine"])

    async def _get_monitoring_instructions(self, urgency: str, symptoms: List[Dict[str, Any]]) -> List[str]:
        """Get monitoring instructions based on urgency and symptoms."""

        instructions = []

        if urgency == "emergency":
            instructions.extend([
                "Call 911 or go to nearest emergency room",
                "Do not drive yourself if symptoms are severe",
                "Monitor breathing and consciousness"
            ])

        elif urgency == "urgent":
            instructions.extend([
                "Monitor symptoms closely",
                "Note any changes in severity or new symptoms",
                "Prepare questions for healthcare provider"
            ])

        else:
            instructions.extend([
                "Track symptoms in a journal",
                "Note patterns or triggers",
                "Monitor for worsening symptoms"
            ])

        return instructions

    async def _get_follow_up_recommendations(self, urgency: str) -> List[str]:
        """Get follow-up recommendations."""

        if urgency == "emergency":
            return [
                "Follow up with primary care provider after emergency treatment",
                "Discuss symptoms and emergency visit with regular healthcare provider",
                "Consider preventive measures for similar episodes"
            ]

        elif urgency == "urgent":
            return [
                "Schedule appointment with healthcare provider",
                "Prepare symptom timeline and questions",
                "Consider keeping a symptom diary"
            ]

        else:
            return [
                "Continue monitoring symptoms",
                "Schedule routine check-up if symptoms persist",
                "Maintain healthy lifestyle habits"
            ]

    async def _extract_vital_signs(
        self,
        voice_analysis: Dict[str, Any],
        image_analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract vital signs from voice and image analysis."""

        vital_signs = {}

        # Extract from voice (speech patterns might indicate distress)
        if voice_analysis and "speech_features" in voice_analysis:
            speech_rate = voice_analysis["speech_features"].get("estimated_speech_rate", 0)

            # Very fast speech might indicate agitation
            if speech_rate > 200:  # words per minute
                vital_signs["heart_rate_estimate"] = 100  # Elevated heart rate estimate

        # Extract from image (if facial image available)
        if image_analysis and "predictions" in image_analysis:
            # Could analyze facial features for pallor, sweating, etc.
            # For now, return None as this would require facial recognition models
            pass

        return vital_signs if vital_signs else None

    async def process_voice_symptom_report(
        self,
        voice_data: bytes,
        patient_id: int,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process voice-recorded symptom report.

        Args:
            voice_data: Audio recording of symptom description
            patient_id: Patient identifier
            additional_context: Additional text context

        Returns:
            Processed voice symptom report with transcription and analysis
        """
        try:
            # Analyze voice input
            voice_analysis = await self._analyze_voice_symptoms(voice_data)

            if "error" in voice_analysis:
                return voice_analysis

            # Combine with text context if provided
            combined_text = voice_analysis["transcription"]
            if additional_context:
                combined_text += f" Additional context: {additional_context}"

            # Run full multimodal analysis
            analysis = await self.analyze_multimodal_symptoms(
                text_input=combined_text,
                voice_input=voice_data,
                patient_context={"patient_id": patient_id}
            )

            return {
                "voice_analysis": voice_analysis,
                "combined_analysis": analysis,
                "processing_timestamp": datetime.utcnow(),
                "input_method": "voice",
                "transcription_quality": "good" if voice_analysis.get("confidence", 0) > -1.0 else "poor"
            }

        except Exception as e:
            logger.error("Error processing voice symptom report", error=str(e))
            return {"error": "Failed to process voice symptom report"}

    async def process_image_symptom_report(
        self,
        image_data: bytes,
        description: str,
        patient_id: int
    ) -> Dict[str, Any]:
        """
        Process image-based symptom report (rashes, wounds, etc.).

        Args:
            image_data: Image of affected area
            description: Text description of the issue
            patient_id: Patient identifier

        Returns:
            Processed image symptom report with analysis
        """
        try:
            # Analyze image
            image_analysis = await self._analyze_image_symptoms(image_data)

            if "error" in image_analysis:
                return image_analysis

            # Combine image analysis with text description
            combined_analysis = await self.analyze_multimodal_symptoms(
                text_input=description,
                image_input=image_data,
                patient_context={"patient_id": patient_id}
            )

            return {
                "image_analysis": image_analysis,
                "combined_analysis": combined_analysis,
                "processing_timestamp": datetime.utcnow(),
                "input_method": "image",
                "image_quality": "good" if image_analysis.get("confidence", 0) > 0.5 else "poor"
            }

        except Exception as e:
            logger.error("Error processing image symptom report", error=str(e))
            return {"error": "Failed to process image symptom report"}


# Global multimodal symptom service instance
multimodal_symptom_service = MultimodalSymptomService()
