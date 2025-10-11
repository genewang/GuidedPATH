"""
AI-powered mental health support service
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import structlog

from backend.core.config import settings
from backend.apps.mental_health.models import (
    MentalHealthSession, ConversationMessage, CopingStrategy,
    CrisisResource, MoodLevel, CrisisLevel, ConversationType
)

logger = structlog.get_logger()


class MentalHealthService:
    """
    AI-powered mental health support service with crisis detection.
    """

    def __init__(self):
        """Initialize the mental health service."""
        self.crisis_detection_model = None
        self.sentiment_model = None
        self.conversation_model = None

        # Crisis indicators and keywords
        self.crisis_keywords = {
            "critical": [
                "suicide", "kill myself", "end it all", "not worth living",
                "harm myself", "hurt myself", "overdose", "jump",
                "hang myself", "shoot myself", "cut myself deeply"
            ],
            "severe": [
                "depressed", "hopeless", "worthless", "anxious",
                "panic attack", "can't cope", "losing control",
                "hearing voices", "seeing things", "paranoid"
            ],
            "moderate": [
                "sad", "lonely", "stressed", "worried", "overwhelmed",
                "tired", "unmotivated", "isolated", "empty"
            ]
        }

        # Coping strategies database (in production, from database)
        self.coping_strategies = {
            "mindfulness": [
                "Practice deep breathing exercises",
                "Try progressive muscle relaxation",
                "Use grounding techniques (5-4-3-2-1)",
                "Practice mindfulness meditation"
            ],
            "social": [
                "Reach out to a trusted friend or family member",
                "Join a support group",
                "Talk to a mental health professional",
                "Connect with others who understand your experience"
            ],
            "physical": [
                "Take a short walk outside",
                "Practice gentle yoga or stretching",
                "Engage in regular exercise",
                "Maintain good sleep hygiene"
            ]
        }

        self._load_models()

    def _load_models(self):
        """Load pre-trained models for mental health analysis."""
        try:
            # Load crisis detection model (would be fine-tuned in production)
            self.crisis_detection_model = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )

            # Load sentiment analysis model
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )

            logger.info("Mental health models loaded successfully")

        except Exception as e:
            logger.error("Failed to load mental health models", error=str(e))

    async def start_conversation(self, user_id: int, conversation_type: str = "general_support") -> Dict[str, Any]:
        """
        Start a mental health support conversation.

        Args:
            user_id: User identifier
            conversation_type: Type of conversation (check_in, crisis_support, etc.)

        Returns:
            Conversation session information
        """
        try:
            logger.info("Starting mental health conversation", user_id=user_id, type=conversation_type)

            # Create conversation session
            session = MentalHealthSession(
                user_id=user_id,
                session_type=ConversationType(conversation_type),
                ai_model=settings.DEFAULT_LLM_MODEL,
                started_at=datetime.utcnow()
            )

            # Initial greeting based on conversation type
            greeting = await self._generate_initial_greeting(conversation_type)

            return {
                "session_id": session.id,
                "conversation_type": conversation_type,
                "greeting": greeting,
                "started_at": session.started_at.isoformat(),
                "ai_model": session.ai_model,
                "guidelines": [
                    "I'm here to listen and support you",
                    "This is a safe, confidential space",
                    "I'm not a substitute for professional mental health care",
                    "If you're in crisis, I'll help you find immediate support"
                ]
            }

        except Exception as e:
            logger.error("Error starting conversation", error=str(e), user_id=user_id)
            return {"error": "Failed to start conversation"}

    async def _generate_initial_greeting(self, conversation_type: str) -> str:
        """Generate appropriate initial greeting based on conversation type."""
        greetings = {
            "check_in": "Hi! I'm glad you're taking time to check in with your mental health. How are you feeling today?",
            "crisis_support": "I'm here to help you through this difficult time. You're not alone. What's going on?",
            "therapy_session": "Welcome to our session. I'm here to support you in exploring your thoughts and feelings.",
            "coping_strategies": "Let's work together to find some helpful coping strategies. What would you like to focus on?",
            "general_support": "Hi! I'm here to listen and support you. How can I help you today?"
        }

        return greetings.get(conversation_type, greetings["general_support"])

    async def process_message(self, session_id: int, message: str, user_id: int) -> Dict[str, Any]:
        """
        Process a user message in a mental health conversation.

        Args:
            session_id: Conversation session identifier
            message: User's message text
            user_id: User identifier

        Returns:
            AI response with analysis and recommendations
        """
        try:
            logger.info("Processing mental health message", session_id=session_id, user_id=user_id)

            # Analyze message for crisis indicators
            crisis_analysis = await self._analyze_crisis_indicators(message)

            # Analyze sentiment
            sentiment_analysis = await self._analyze_sentiment(message)

            # Generate appropriate response
            response_text = await self._generate_response(message, crisis_analysis, sentiment_analysis)

            # Check if escalation is needed
            needs_escalation = crisis_analysis["crisis_level"] in ["severe", "critical"]

            # Create conversation message record
            conversation_message = ConversationMessage(
                session_id=session_id,
                message_text=message,
                is_from_user=True,
                sentiment_score=sentiment_analysis["score"],
                crisis_indicators=crisis_analysis["indicators"],
                timestamp=datetime.utcnow()
            )

            return {
                "session_id": session_id,
                "response": response_text,
                "crisis_analysis": crisis_analysis,
                "sentiment_analysis": sentiment_analysis,
                "needs_escalation": needs_escalation,
                "coping_strategies": await self._suggest_coping_strategies(crisis_analysis, sentiment_analysis),
                "resources": await self._get_relevant_resources(crisis_analysis) if needs_escalation else [],
                "processed_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error("Error processing message", error=str(e), session_id=session_id)
            return {
                "error": "Failed to process message",
                "response": "I'm having trouble processing your message. Please try again or contact a mental health professional directly."
            }

    async def _analyze_crisis_indicators(self, message: str) -> Dict[str, Any]:
        """Analyze message for crisis indicators and suicidal ideation."""
        message_lower = message.lower()
        indicators = []
        crisis_level = CrisisLevel.NONE

        # Check for crisis keywords
        for level, keywords in self.crisis_keywords.items():
            found_indicators = [keyword for keyword in keywords if keyword in message_lower]
            if found_indicators:
                indicators.extend(found_indicators)

                # Determine highest crisis level
                if level == "critical":
                    crisis_level = CrisisLevel.CRITICAL
                elif level == "severe" and crisis_level != CrisisLevel.CRITICAL:
                    crisis_level = CrisisLevel.SEVERE
                elif level == "moderate" and crisis_level == CrisisLevel.NONE:
                    crisis_level = CrisisLevel.MILD

        return {
            "crisis_level": crisis_level.value,
            "indicators": indicators,
            "requires_immediate_action": crisis_level in [CrisisLevel.SEVERE, CrisisLevel.CRITICAL]
        }

    async def _analyze_sentiment(self, message: str) -> Dict[str, Any]:
        """Analyze sentiment of the message."""
        try:
            if self.sentiment_model:
                result = self.sentiment_model(message)[0]
                # Convert to -1.0 to 1.0 scale
                score = 1.0 if result["label"] == "POSITIVE" else -1.0 if result["label"] == "NEGATIVE" else 0.0
                return {
                    "sentiment": result["label"],
                    "score": score,
                    "confidence": result["score"]
                }
            else:
                # Fallback sentiment analysis
                positive_words = ["good", "better", "hope", "happy", "grateful", "thankful"]
                negative_words = ["bad", "worse", "hopeless", "sad", "angry", "frustrated"]

                positive_count = sum(1 for word in positive_words if word in message.lower())
                negative_count = sum(1 for word in negative_words if word in message.lower())

                if positive_count > negative_count:
                    return {"sentiment": "POSITIVE", "score": 0.5, "confidence": 0.7}
                elif negative_count > positive_count:
                    return {"sentiment": "NEGATIVE", "score": -0.5, "confidence": 0.7}
                else:
                    return {"sentiment": "NEUTRAL", "score": 0.0, "confidence": 0.5}

        except Exception as e:
            logger.error("Error analyzing sentiment", error=str(e))
            return {"sentiment": "NEUTRAL", "score": 0.0, "confidence": 0.0}

    async def _generate_response(self, message: str, crisis_analysis: Dict[str, Any], sentiment_analysis: Dict[str, Any]) -> str:
        """Generate empathetic, supportive response based on analysis."""

        # Handle crisis situations
        if crisis_analysis["crisis_level"] == "critical":
            return (
                "I hear that you're in a lot of pain right now, and I'm really concerned about you. "
                "Please reach out for immediate help. You can call the National Suicide Prevention Lifeline at 988 (US), "
                "or text HOME to 741741 for the Crisis Text Line. If you're in immediate danger, call emergency services (911 in the US). "
                "You're not alone in this, and there are people who want to help you through this difficult time."
            )

        elif crisis_analysis["crisis_level"] == "severe":
            return (
                "I'm really worried about what you're going through. It sounds incredibly difficult. "
                "While I'm here to support you, I strongly recommend speaking with a mental health professional. "
                "Would you like resources for crisis support, or information about finding a therapist?"
            )

        # Generate supportive responses based on sentiment and content
        if sentiment_analysis["sentiment"] == "NEGATIVE":
            return (
                "I can hear that you're struggling right now, and I'm here to listen without judgment. "
                "It's okay to not be okay, and taking the step to talk about it is a sign of strength. "
                "What would be most helpful for you to talk about right now?"
            )

        elif "lonely" in message.lower() or "isolated" in message.lower():
            return (
                "Feeling lonely can be really tough. It's a common experience, especially during difficult times. "
                "Sometimes reaching out to others, even in small ways, can help. Have you considered "
                "connecting with a trusted friend, joining a support group, or talking to a mental health professional?"
            )

        elif "anxious" in message.lower() or "worried" in message.lower():
            return (
                "Anxiety can feel overwhelming, but you're taking a positive step by acknowledging it. "
                "Some people find that deep breathing, grounding techniques, or talking through their worries "
                "can help. Would you like to explore some coping strategies together?"
            )

        else:
            return (
                "Thank you for sharing that with me. I want you to know that your feelings are valid and important. "
                "Is there something specific you'd like support with, or would you like to talk about "
                "how you're feeling right now?"
            )

    async def _suggest_coping_strategies(self, crisis_analysis: Dict[str, Any], sentiment_analysis: Dict[str, Any]) -> List[str]:
        """Suggest appropriate coping strategies based on analysis."""

        strategies = []

        # Crisis-level strategies
        if crisis_analysis["crisis_level"] == "critical":
            strategies.extend([
                "Call emergency services (911) if in immediate danger",
                "Contact the National Suicide Prevention Lifeline at 988",
                "Reach out to a trusted friend or family member",
                "Go to your nearest emergency room"
            ])

        elif crisis_analysis["crisis_level"] == "severe":
            strategies.extend([
                "Contact a mental health crisis hotline",
                "Schedule an appointment with a mental health professional",
                "Reach out to a trusted support person",
                "Practice grounding techniques to manage intense emotions"
            ])

        # General coping strategies based on sentiment
        if sentiment_analysis["sentiment"] == "NEGATIVE":
            strategies.extend([
                "Practice deep breathing or mindfulness exercises",
                "Take a short walk or engage in light physical activity",
                "Journal your thoughts and feelings",
                "Connect with supportive people in your life"
            ])

        # Add general wellness strategies
        strategies.extend([
            "Maintain regular sleep and meal schedules",
            "Practice self-compassion and self-care",
            "Set small, achievable goals for the day",
            "Consider professional mental health support if needed"
        ])

        return strategies[:6]  # Return top 6 strategies

    async def _get_relevant_resources(self, crisis_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get relevant crisis resources based on analysis."""

        if crisis_analysis["crisis_level"] == "critical":
            return [
                {
                    "name": "National Suicide Prevention Lifeline",
                    "phone": "988",
                    "description": "24/7 confidential support for people in crisis",
                    "urgency": "immediate"
                },
                {
                    "name": "Crisis Text Line",
                    "phone": "Text HOME to 741741",
                    "description": "24/7 crisis counseling via text message",
                    "urgency": "immediate"
                }
            ]

        elif crisis_analysis["crisis_level"] == "severe":
            return [
                {
                    "name": "National Alliance on Mental Illness (NAMI) Helpline",
                    "phone": "1-800-950-NAMI (6264)",
                    "description": "Information and referral services",
                    "urgency": "urgent"
                }
            ]

        return []

    async def end_conversation(self, session_id: int, user_id: int) -> Dict[str, Any]:
        """
        End a mental health conversation and provide summary.

        Args:
            session_id: Conversation session identifier
            user_id: User identifier

        Returns:
            Conversation summary and follow-up recommendations
        """
        try:
            # Update session end time
            ended_at = datetime.utcnow()

            # Generate session summary
            summary = {
                "session_id": session_id,
                "ended_at": ended_at.isoformat(),
                "duration_minutes": 30,  # Would calculate actual duration
                "crisis_level_detected": CrisisLevel.NONE.value,
                "resources_shared": [],
                "follow_up_recommendations": [
                    "Continue practicing the coping strategies we discussed",
                    "Consider scheduling an appointment with a mental health professional",
                    "Reach out again if you need support",
                    "Remember that it's okay to ask for help"
                ]
            }

            logger.info("Mental health conversation ended", session_id=session_id, user_id=user_id)
            return summary

        except Exception as e:
            logger.error("Error ending conversation", error=str(e), session_id=session_id)
            return {"error": "Failed to end conversation"}

    async def get_mood_tracking_data(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """
        Get mood tracking data for trend analysis.

        Args:
            user_id: User identifier
            days: Number of days to look back

        Returns:
            Mood tracking trends and insights
        """
        try:
            # TODO: Get mood tracking data from database

            # Mock data for demonstration
            return {
                "user_id": user_id,
                "period_days": days,
                "average_mood_score": 6.5,
                "trend": "improving",
                "insights": [
                    "Your mood has been gradually improving over the past month",
                    "Sleep quality appears to correlate with mood fluctuations",
                    "Consider maintaining current coping strategies"
                ],
                "recommendations": [
                    "Continue tracking your mood daily",
                    "Identify patterns between sleep, stress, and mood",
                    "Celebrate small improvements in your mental health"
                ]
            }

        except Exception as e:
            logger.error("Error getting mood tracking data", error=str(e), user_id=user_id)
            return {"error": "Failed to get mood tracking data"}

    async def suggest_coping_strategies(self, current_mood: str, situation: str = "") -> List[Dict[str, Any]]:
        """
        Suggest personalized coping strategies.

        Args:
            current_mood: Current mood description
            situation: Optional context about current situation

        Returns:
            List of suggested coping strategies
        """
        try:
            strategies = []

            # Categorize mood and situation for strategy selection
            mood_lower = current_mood.lower()

            if any(word in mood_lower for word in ["anxious", "worried", "panicky"]):
                strategies.extend([
                    {
                        "category": "Immediate Relief",
                        "strategy": "4-7-8 Breathing",
                        "description": "Inhale for 4 seconds, hold for 7, exhale for 8",
                        "estimated_time": "2 minutes"
                    },
                    {
                        "category": "Grounding",
                        "strategy": "5-4-3-2-1 Technique",
                        "description": "Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste",
                        "estimated_time": "3 minutes"
                    }
                ])

            if any(word in mood_lower for word in ["sad", "depressed", "down"]):
                strategies.extend([
                    {
                        "category": "Mood Boost",
                        "strategy": "Gratitude Practice",
                        "description": "Write down 3 things you're grateful for today",
                        "estimated_time": "5 minutes"
                    },
                    {
                        "category": "Social Connection",
                        "strategy": "Reach Out",
                        "description": "Call or message someone you trust",
                        "estimated_time": "10 minutes"
                    }
                ])

            if any(word in mood_lower for word in ["angry", "frustrated", "irritated"]):
                strategies.extend([
                    {
                        "category": "Physical Release",
                        "strategy": "Progressive Muscle Relaxation",
                        "description": "Tense and release muscle groups from toes to head",
                        "estimated_time": "10 minutes"
                    }
                ])

            # Add general wellness strategies
            strategies.extend([
                {
                    "category": "Daily Practice",
                    "strategy": "Mindful Walking",
                    "description": "Take a 10-minute walk focusing on your surroundings",
                    "estimated_time": "10 minutes"
                },
                {
                    "category": "Self-Care",
                    "strategy": "Comfort Activity",
                    "description": "Engage in an activity that brings you comfort",
                    "estimated_time": "15 minutes"
                }
            ])

            return strategies[:5]  # Return top 5 strategies

        except Exception as e:
            logger.error("Error suggesting coping strategies", error=str(e))
            return []


# Global mental health service instance
mental_health_service = MentalHealthService()
