"""
Conversational AI assistant service
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import json

from langchain.chains import ConversationalRetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import structlog

from backend.core.config import settings
from backend.apps.chat.models import ChatSession, ChatMessage

logger = structlog.get_logger()


class ChatService:
    """
    Conversational AI assistant for comprehensive healthcare support.
    """

    def __init__(self):
        """Initialize the chat service."""
        self.conversation_chains = {}
        self.user_sessions = {}

        # Initialize conversation memory for each user
        self.memory_store = {}

        # Context awareness for different healthcare domains
        self.domain_contexts = {
            "general": "You are a helpful AI assistant for general healthcare questions and navigation.",
            "guidelines": "You are an expert in clinical guidelines and evidence-based medicine.",
            "trials": "You help patients find and understand clinical trials.",
            "medication": "You provide accurate information about medications and drug interactions.",
            "symptoms": "You help assess symptoms and provide triage recommendations.",
            "mental_health": "You provide supportive mental health guidance and crisis resources."
        }

    async def start_chat_session(self, user_id: int, initial_context: str = "general") -> Dict[str, Any]:
        """
        Start a new chat session with context awareness.

        Args:
            user_id: User identifier
            initial_context: Initial domain context for the conversation

        Returns:
            Chat session information
        """
        try:
            logger.info("Starting chat session", user_id=user_id, context=initial_context)

            # Create chat session
            session = ChatSession(
                user_id=user_id,
                context_domain=initial_context,
                started_at=datetime.utcnow(),
                ai_model=settings.DEFAULT_LLM_MODEL
            )

            # Initialize conversation memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

            # Store session and memory
            self.user_sessions[user_id] = session
            self.memory_store[user_id] = memory

            # Generate contextual greeting
            greeting = await self._generate_contextual_greeting(initial_context)

            return {
                "session_id": session.id,
                "context_domain": initial_context,
                "greeting": greeting,
                "started_at": session.started_at.isoformat(),
                "capabilities": [
                    "Answer healthcare questions",
                    "Provide evidence-based information",
                    "Help navigate treatment options",
                    "Connect to relevant services",
                    "Offer emotional support when appropriate"
                ]
            }

        except Exception as e:
            logger.error("Error starting chat session", error=str(e), user_id=user_id)
            return {"error": "Failed to start chat session"}

    async def _generate_contextual_greeting(self, context: str) -> str:
        """Generate greeting based on conversation context."""
        greetings = {
            "general": "Hi! I'm your AI healthcare assistant. I can help you with questions about your health, treatment options, and navigating the healthcare system. What would you like to know?",
            "guidelines": "Hello! I can help you understand clinical guidelines and evidence-based treatment recommendations. What specific area are you interested in learning about?",
            "trials": "Hi there! I can help you find and understand clinical trials that might be relevant to your condition. What type of treatment are you exploring?",
            "medication": "Hello! I can provide information about medications, potential interactions, and help you manage your medication schedule. What would you like to know?",
            "symptoms": "Hi! I can help you assess symptoms and provide guidance on when to seek medical care. Please describe what you're experiencing.",
            "mental_health": "Hello! I'm here to listen and support you. You can talk to me about how you're feeling, and I'll provide a safe, non-judgmental space. How are you doing today?"
        }

        return greetings.get(context, greetings["general"])

    async def process_chat_message(self, user_id: int, message: str, context_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a chat message and return AI response.

        Args:
            user_id: User identifier
            message: User's message text
            context_override: Optional context override for this message

        Returns:
            AI response with context and suggestions
        """
        try:
            logger.info("Processing chat message", user_id=user_id, message_length=len(message))

            # Get or create user session
            if user_id not in self.user_sessions:
                session = await self.start_chat_session(user_id)
                if "error" in session:
                    return session

            # Determine context for this message
            context = context_override or self.user_sessions[user_id].context_domain

            # Analyze message for intent and entities
            intent_analysis = await self._analyze_message_intent(message)

            # Generate contextual response
            response = await self._generate_chat_response(
                user_id, message, context, intent_analysis
            )

            # Check if context should be updated
            suggested_context = await self._suggest_context_update(message, intent_analysis)

            # Create chat message record
            chat_message = ChatMessage(
                session_id=self.user_sessions[user_id].id,
                message_text=message,
                is_from_user=True,
                context_used=context,
                intent_detected=intent_analysis["primary_intent"],
                entities_extracted=intent_analysis["entities"],
                timestamp=datetime.utcnow()
            )

            # Update conversation memory
            self.memory_store[user_id].chat_memory.add_user_message(message)
            self.memory_store[user_id].chat_memory.add_ai_message(response)

            return {
                "session_id": self.user_sessions[user_id].id,
                "response": response,
                "context_used": context,
                "suggested_context": suggested_context,
                "intent_analysis": intent_analysis,
                "entities_mentioned": intent_analysis["entities"],
                "follow_up_suggestions": await self._generate_follow_up_suggestions(
                    intent_analysis, context
                ),
                "processed_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error("Error processing chat message", error=str(e), user_id=user_id)
            return {
                "error": "Failed to process message",
                "response": "I'm having trouble understanding your message right now. Could you try rephrasing or ask me something else?"
            }

    async def _analyze_message_intent(self, message: str) -> Dict[str, Any]:
        """Analyze message to determine intent and extract entities."""

        message_lower = message.lower()

        # Intent classification (simplified - in production, use ML model)
        intent_keywords = {
            "question": ["what", "how", "why", "when", "where", "who", "can you", "do you"],
            "information": ["tell me about", "information about", "details on", "explain"],
            "help": ["help me", "i need", "can you help", "assist me", "support"],
            "search": ["find", "search for", "look for", "looking for"],
            "recommendation": ["recommend", "suggest", "what should", "advice"],
            "appointment": ["appointment", "schedule", "book", "see doctor"],
            "medication": ["medication", "drug", "prescription", "pill", "medicine"],
            "symptoms": ["symptom", "pain", "hurt", "feeling", "sick"],
            "mental_health": ["sad", "depressed", "anxious", "worried", "stressed", "mental health"]
        }

        # Find primary intent
        intent_scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > 0:
                intent_scores[intent] = score

        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else "general"

        # Extract entities (simplified - in production, use NER)
        entities = []

        # Medical condition entities
        conditions = ["cancer", "diabetes", "heart disease", "depression", "anxiety", "arthritis"]
        for condition in conditions:
            if condition in message_lower:
                entities.append({"type": "condition", "value": condition})

        # Medication entities
        medications = ["aspirin", "ibuprofen", "acetaminophen", "lisinopril", "metformin"]
        for medication in medications:
            if medication in message_lower:
                entities.append({"type": "medication", "value": medication})

        return {
            "primary_intent": primary_intent,
            "intent_confidence": min(1.0, sum(intent_scores.values()) / len(message.split())),
            "entities": entities,
            "message_length": len(message),
            "complexity_score": len([word for word in message.split() if len(word) > 6]) / len(message.split())
        }

    async def _generate_chat_response(self, user_id: int, message: str, context: str, intent_analysis: Dict[str, Any]) -> str:
        """Generate contextual chat response."""

        # Get conversation history
        memory = self.memory_store.get(user_id, ConversationBufferMemory(return_messages=True))
        chat_history = memory.chat_memory.messages if hasattr(memory, 'chat_memory') else []

        # Create context-aware prompt
        system_prompt = self._create_system_prompt(context, intent_analysis)

        # Generate response based on context and intent
        if intent_analysis["primary_intent"] == "medication":
            response = await self._handle_medication_query(message, intent_analysis)
        elif intent_analysis["primary_intent"] == "symptoms":
            response = await self._handle_symptom_query(message, intent_analysis)
        elif intent_analysis["primary_intent"] == "mental_health":
            response = await self._handle_mental_health_query(message, intent_analysis)
        elif intent_analysis["primary_intent"] == "question":
            response = await self._handle_general_question(message, context, intent_analysis)
        else:
            response = await self._handle_general_query(message, context, intent_analysis)

        return response

    def _create_system_prompt(self, context: str, intent_analysis: Dict[str, Any]) -> str:
        """Create system prompt based on context and intent."""

        base_prompt = f"""
        {self.domain_contexts.get(context, self.domain_contexts['general'])}

        Important guidelines:
        - Always provide evidence-based, accurate information
        - Never diagnose medical conditions
        - Recommend consulting healthcare providers for personal medical decisions
        - Be empathetic and supportive
        - If uncertain, admit limitations and suggest professional consultation
        - For crisis situations, provide immediate resources and encourage seeking help

        Current conversation intent: {intent_analysis['primary_intent']}
        """

        if intent_analysis["primary_intent"] == "mental_health":
            base_prompt += """
            Additional mental health guidelines:
            - Provide a safe, non-judgmental space
            - Listen actively and validate feelings
            - Suggest coping strategies when appropriate
            - Know when to recommend professional help
            - Have crisis resources ready if needed
            """

        return base_prompt.strip()

    async def _handle_medication_query(self, message: str, intent_analysis: Dict[str, Any]) -> str:
        """Handle medication-related queries."""
        return (
            "I can help you understand medications and their general uses, but I'm not a substitute for professional medical advice. "
            "For questions about your specific medications, dosages, or interactions, please consult your healthcare provider or pharmacist. "
            "Would you like general information about a type of medication, or help finding medication resources?"
        )

    async def _handle_symptom_query(self, message: str, intent_analysis: Dict[str, Any]) -> str:
        """Handle symptom-related queries."""
        return (
            "I'm here to help you understand symptoms in general terms. However, I cannot diagnose conditions or provide personalized medical advice. "
            "If you're experiencing concerning symptoms, please consult with a healthcare provider. "
            "Would you like me to help you prepare questions for your doctor, or provide general information about symptom management?"
        )

    async def _handle_mental_health_query(self, message: str, intent_analysis: Dict[str, Any]) -> str:
        """Handle mental health queries with appropriate care."""
        return (
            "I want you to know that your mental health matters and it's okay to talk about how you're feeling. "
            "I'm here to listen and provide general support and resources. If you're in crisis or need immediate help, "
            "please reach out to a crisis hotline or mental health professional. "
            "Would you like to talk about coping strategies, or need help finding mental health resources?"
        )

    async def _handle_general_question(self, message: str, context: str, intent_analysis: Dict[str, Any]) -> str:
        """Handle general healthcare questions."""
        return (
            "I can help you navigate healthcare information and connect you with appropriate resources. "
            "For personalized medical advice, please consult with your healthcare provider. "
            "What specific information are you looking for?"
        )

    async def _handle_general_query(self, message: str, context: str, intent_analysis: Dict[str, Any]) -> str:
        """Handle general queries."""
        return (
            "I'm here to help you with healthcare questions and navigation. "
            "What would you like to know more about?"
        )

    async def _suggest_context_update(self, message: str, intent_analysis: Dict[str, Any]) -> Optional[str]:
        """Suggest if conversation context should be updated."""

        # Analyze if message indicates a different domain
        message_lower = message.lower()

        context_keywords = {
            "medication": ["medication", "drug", "prescription", "pill", "medicine"],
            "symptoms": ["symptom", "pain", "hurt", "feeling", "sick"],
            "mental_health": ["sad", "depressed", "anxious", "worried", "stressed"],
            "trials": ["clinical trial", "research study", "experimental treatment"],
            "guidelines": ["guideline", "recommendation", "treatment protocol"]
        }

        # Find suggested context based on keywords
        for suggested_context, keywords in context_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return suggested_context

        return None

    async def _generate_follow_up_suggestions(self, intent_analysis: Dict[str, Any], context: str) -> List[str]:
        """Generate follow-up question suggestions."""

        suggestions = {
            "medication": [
                "What specific medication questions do you have?",
                "Would you like help understanding medication side effects?",
                "Are you looking for information about drug interactions?"
            ],
            "symptoms": [
                "Would you like to track your symptoms over time?",
                "Are you preparing questions for your healthcare provider?",
                "Would you like general information about symptom management?"
            ],
            "mental_health": [
                "Would you like to explore coping strategies?",
                "Are you interested in mental health resources?",
                "Would you like to talk about stress management techniques?"
            ],
            "general": [
                "What specific health topic interests you?",
                "Are you looking for treatment information?",
                "Would you like help finding healthcare resources?"
            ]
        }

        return suggestions.get(context, suggestions["general"])[:3]

    async def switch_context(self, user_id: int, new_context: str) -> Dict[str, Any]:
        """
        Switch conversation context for a user session.

        Args:
            user_id: User identifier
            new_context: New domain context

        Returns:
            Context switch confirmation
        """
        try:
            if user_id not in self.user_sessions:
                return {"error": "No active session found"}

            # Update session context
            self.user_sessions[user_id].context_domain = new_context

            # Generate context transition message
            transition_message = await self._generate_context_transition_message(new_context)

            return {
                "session_id": self.user_sessions[user_id].id,
                "previous_context": "general",  # Would track actual previous context
                "new_context": new_context,
                "transition_message": transition_message,
                "switched_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error("Error switching context", error=str(e), user_id=user_id)
            return {"error": "Failed to switch context"}

    async def _generate_context_transition_message(self, new_context: str) -> str:
        """Generate message for context transition."""

        transitions = {
            "medication": "Great! I'm switching to medication-focused assistance. I can help you understand medications, interactions, and management strategies.",
            "symptoms": "Perfect! I'll focus on symptom-related guidance. I can help you understand symptoms and when to seek care.",
            "mental_health": "Thank you for trusting me with this. I'm here to provide mental health support and resources in a safe space.",
            "trials": "Excellent! I can help you explore clinical trials and research opportunities that might be relevant to you.",
            "guidelines": "Great choice! I can provide information about clinical guidelines and evidence-based treatment approaches."
        }

        return transitions.get(new_context, "I've updated my focus to better assist you with this topic.")

    async def end_chat_session(self, user_id: int) -> Dict[str, Any]:
        """
        End a chat session and provide summary.

        Args:
            user_id: User identifier

        Returns:
            Session summary
        """
        try:
            if user_id not in self.user_sessions:
                return {"error": "No active session found"}

            session = self.user_sessions[user_id]

            # Calculate session metrics
            duration = datetime.utcnow() - session.started_at

            # Generate session summary
            summary = {
                "session_id": session.id,
                "duration_minutes": int(duration.total_seconds() / 60),
                "context_used": session.context_domain,
                "messages_exchanged": 10,  # Would count actual messages
                "topics_discussed": ["general_healthcare"],  # Would analyze actual topics
                "ended_at": datetime.utcnow().isoformat(),
                "follow_up_suggestions": [
                    "Continue exploring healthcare resources",
                    "Consider consulting healthcare providers for personalized advice",
                    "Reach out if you need more support"
                ]
            }

            # Clean up session data
            del self.user_sessions[user_id]
            del self.memory_store[user_id]

            logger.info("Chat session ended", user_id=user_id, duration_minutes=summary["duration_minutes"])
            return summary

        except Exception as e:
            logger.error("Error ending chat session", error=str(e), user_id=user_id)
            return {"error": "Failed to end chat session"}

    async def get_chat_history(self, user_id: int, limit: int = 50) -> Dict[str, Any]:
        """
        Get chat history for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of messages to return

        Returns:
            Chat history with summaries
        """
        try:
            # TODO: Get chat history from database

            return {
                "user_id": user_id,
                "total_sessions": 3,
                "recent_topics": ["medication", "symptoms", "general_healthcare"],
                "message_count": 25,
                "last_activity": "2024-01-01T00:00:00Z",
                "insights": [
                    "Frequently discusses medication questions",
                    "Shows interest in symptom tracking",
                    "Engages with mental health topics"
                ]
            }

        except Exception as e:
            logger.error("Error getting chat history", error=str(e), user_id=user_id)
            return {"error": "Failed to get chat history"}


# Global chat service instance
chat_service = ChatService()
