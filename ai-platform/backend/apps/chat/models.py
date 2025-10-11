"""
Chat conversation models
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import String, DateTime, Boolean, Integer, Text, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from backend.core.database import Base


class ChatContext(enum.Enum):
    """Context domains for chat conversations."""
    GENERAL = "general"
    GUIDELINES = "guidelines"
    TRIALS = "trials"
    MEDICATION = "medication"
    SYMPTOMS = "symptoms"
    MENTAL_HEALTH = "mental_health"


class ChatSession(Base):
    """Chat session tracking."""
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)

    # Session details
    context_domain: Mapped[ChatContext] = mapped_column(SQLEnum(ChatContext), default=ChatContext.GENERAL)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # AI model information
    ai_model: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[Optional[str]] = mapped_column(String(50))

    # Session metrics
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_minutes: Mapped[Optional[int]] = mapped_column(Integer)
    topics_discussed: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    messages: Mapped[List["ChatMessage"]] = relationship("ChatMessage", back_populates="session")

    def __repr__(self) -> str:
        return f"<ChatSession(id={self.id}, context={self.context_domain.value}, duration={self.duration_minutes})>"


class ChatMessage(Base):
    """Individual chat messages."""
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("chat_sessions.id"), nullable=False)

    # Message details
    message_text: Mapped[str] = mapped_column(Text, nullable=False)
    is_from_user: Mapped[bool] = mapped_column(Boolean, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # AI analysis
    context_used: Mapped[Optional[str]] = mapped_column(String(50))
    intent_detected: Mapped[Optional[str]] = mapped_column(String(100))
    entities_extracted: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)

    # Response metadata
    response_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    ai_confidence: Mapped[Optional[float]] = mapped_column(Float)

    # Relationships
    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")

    def __repr__(self) -> str:
        return f"<ChatMessage(id={self.id}, session_id={self.session_id}, from_user={self.is_from_user})>"
