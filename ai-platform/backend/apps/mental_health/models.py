"""
Mental health support models
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import String, DateTime, Boolean, Integer, Text, JSON, Float, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from backend.core.database import Base


class MoodLevel(enum.Enum):
    """Mood severity levels."""
    VERY_LOW = "very_low"
    LOW = "low"
    NEUTRAL = "neutral"
    GOOD = "good"
    VERY_GOOD = "very_good"


class CrisisLevel(enum.Enum):
    """Crisis assessment levels."""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class ConversationType(enum.Enum):
    """Types of mental health conversations."""
    CHECK_IN = "check_in"
    CRISIS_SUPPORT = "crisis_support"
    THERAPY_SESSION = "therapy_session"
    COPING_STRATEGIES = "coping_strategies"
    GENERAL_SUPPORT = "general_support"


class MentalHealthSession(Base):
    """Mental health support session tracking."""
    __tablename__ = "mental_health_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)

    # Session details
    session_type: Mapped[ConversationType] = mapped_column(SQLEnum(ConversationType), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Mood and crisis assessment
    initial_mood: Mapped[Optional[MoodLevel]] = mapped_column(SQLEnum(MoodLevel))
    final_mood: Mapped[Optional[MoodLevel]] = mapped_column(SQLEnum(MoodLevel))
    crisis_level: Mapped[CrisisLevel] = mapped_column(SQLEnum(CrisisLevel), default=CrisisLevel.NONE)

    # AI model used
    ai_model: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[Optional[str]] = mapped_column(String(50))

    # Session outcomes
    goals_discussed: Mapped[Optional[List[str]]] = mapped_column(JSON)
    coping_strategies_shared: Mapped[Optional[List[str]]] = mapped_column(JSON)
    resources_provided: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Escalation tracking
    escalated_to_human: Mapped[bool] = mapped_column(Boolean, default=False)
    escalation_reason: Mapped[Optional[str]] = mapped_column(Text)
    human_intervention_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Session metrics
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_minutes: Mapped[Optional[int]] = mapped_column(Integer)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<MentalHealthSession(id={self.id}, type={self.session_type.value}, duration={self.duration_minutes})>"


class ConversationMessage(Base):
    """Individual messages in mental health conversations."""
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("mental_health_sessions.id"), nullable=False)

    # Message details
    message_text: Mapped[str] = mapped_column(Text, nullable=False)
    is_from_user: Mapped[bool] = mapped_column(Boolean, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # AI analysis
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)  # -1.0 to 1.0
    crisis_indicators: Mapped[Optional[List[str]]] = mapped_column(JSON)
    topics_discussed: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Response metadata
    response_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    ai_confidence: Mapped[Optional[float]] = mapped_column(Float)

    def __repr__(self) -> str:
        return f"<ConversationMessage(id={self.id}, session_id={self.session_id}, from_user={self.is_from_user})>"


class CopingStrategy(Base):
    """Coping strategies and techniques."""
    __tablename__ = "coping_strategies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    # Strategy classification
    category: Mapped[str] = mapped_column(String(100), nullable=False)  # mindfulness, exercise, social, etc.
    difficulty_level: Mapped[str] = mapped_column(String(50), nullable=False)  # beginner, intermediate, advanced
    time_required_minutes: Mapped[Optional[int]] = mapped_column(Integer)

    # Effectiveness tracking
    effectiveness_rating: Mapped[Optional[float]] = mapped_column(Float)  # 1.0 to 5.0
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[Optional[float]] = mapped_column(Float)  # 0.0 to 1.0

    # Content
    instructions: Mapped[str] = mapped_column(Text, nullable=False)
    tips: Mapped[Optional[List[str]]] = mapped_column(JSON)
    contraindications: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Evidence base
    evidence_level: Mapped[Optional[str]] = mapped_column(String(50))  # A, B, C
    references: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<CopingStrategy(id={self.id}, name={self.name}, category={self.category})>"


class CrisisResource(Base):
    """Crisis intervention resources and hotlines."""
    __tablename__ = "crisis_resources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    # Contact information
    phone_number: Mapped[Optional[str]] = mapped_column(String(20))
    website: Mapped[Optional[str]] = mapped_column(String(500))
    chat_url: Mapped[Optional[str]] = mapped_column(String(500))
    text_number: Mapped[Optional[str]] = mapped_column(String(20))

    # Coverage
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    region: Mapped[Optional[str]] = mapped_column(String(100))
    language: Mapped[str] = mapped_column(String(100), default="English")

    # Service details
    service_type: Mapped[str] = mapped_column(String(100), nullable=False)  # hotline, chat, text, etc.
    availability: Mapped[str] = mapped_column(String(200), nullable=False)  # 24/7, business hours, etc.
    target_audience: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Crisis indicators this resource helps with
    crisis_types: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    response_time: Mapped[Optional[str]] = mapped_column(String(100))  # immediate, within minutes, etc.

    # Quality metrics
    rating: Mapped[Optional[float]] = mapped_column(Float)  # 1.0 to 5.0
    review_count: Mapped[int] = mapped_column(Integer, default=0)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_verified: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<CrisisResource(id={self.id}, name={self.name}, country={self.country})>"


class MoodTracking(Base):
    """Daily mood tracking for patients."""
    __tablename__ = "mood_tracking"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)

    # Mood data
    mood_level: Mapped[MoodLevel] = mapped_column(SQLEnum(MoodLevel), nullable=False)
    mood_score: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-10 scale
    energy_level: Mapped[Optional[int]] = mapped_column(Integer)  # 1-10 scale

    # Context
    sleep_hours: Mapped[Optional[float]] = mapped_column(Float)
    stress_level: Mapped[Optional[int]] = mapped_column(Integer)  # 1-10 scale
    notes: Mapped[Optional[str]] = mapped_column(Text)

    # Factors
    positive_factors: Mapped[Optional[List[str]]] = mapped_column(JSON)
    negative_factors: Mapped[Optional[List[str]]] = mapped_column(JSON)
    coping_strategies_used: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Tracking date
    tracking_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<MoodTracking(id={self.id}, date={self.tracking_date.date()}, mood={self.mood_level.value})>"


class MentalHealthAssessment(Base):
    """Standardized mental health assessments."""
    __tablename__ = "mental_health_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)

    # Assessment details
    assessment_type: Mapped[str] = mapped_column(String(100), nullable=False)  # PHQ-9, GAD-7, etc.
    version: Mapped[str] = mapped_column(String(20), nullable=False)

    # Scores
    total_score: Mapped[int] = mapped_column(Integer, nullable=False)
    subscale_scores: Mapped[Optional[Dict[str, int]]] = mapped_column(JSON)
    severity_level: Mapped[str] = mapped_column(String(50), nullable=False)

    # Clinical interpretation
    interpretation: Mapped[str] = mapped_column(Text, nullable=False)
    recommendations: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    follow_up_needed: Mapped[bool] = mapped_column(Boolean, default=False)

    # Assessment metadata
    completed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    time_to_complete_minutes: Mapped[Optional[int]] = mapped_column(Integer)

    def __repr__(self) -> str:
        return f"<MentalHealthAssessment(id={self.id}, type={self.assessment_type}, score={self.total_score})>"
