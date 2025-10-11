"""
Clinical trials models for trial matching system
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import String, DateTime, Boolean, Integer, Text, JSON, Float, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from backend.core.database import Base


class TrialStatus(enum.Enum):
    """Status of clinical trials."""
    RECRUITING = "recruiting"
    NOT_RECRUITING = "not_recruiting"
    ACTIVE_NOT_RECRUITING = "active_not_recruiting"
    COMPLETED = "completed"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    WITHDRAWN = "withdrawn"
    UNKNOWN = "unknown"


class TrialPhase(enum.Enum):
    """Phase of clinical trials."""
    PHASE_0 = "phase_0"
    PHASE_1 = "phase_1"
    PHASE_1_2 = "phase_1_2"
    PHASE_2 = "phase_2"
    PHASE_2_3 = "phase_2_3"
    PHASE_3 = "phase_3"
    PHASE_4 = "phase_4"
    NOT_APPLICABLE = "not_applicable"


class TrialType(enum.Enum):
    """Type of clinical trial."""
    INTERVENTIONAL = "interventional"
    OBSERVATIONAL = "observational"
    EXPANDED_ACCESS = "expanded_access"
    DIAGNOSTIC = "diagnostic"


class ClinicalTrial(Base):
    """Clinical trial model for trial matching."""
    __tablename__ = "clinical_trials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    nct_id: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)  # ClinicalTrials.gov ID
    title: Mapped[str] = mapped_column(String(1000), nullable=False)
    brief_summary: Mapped[Optional[str]] = mapped_column(Text)
    detailed_description: Mapped[Optional[str]] = mapped_column(Text)

    # Trial metadata
    status: Mapped[TrialStatus] = mapped_column(SQLEnum(TrialStatus), nullable=False)
    phase: Mapped[TrialPhase] = mapped_column(SQLEnum(TrialPhase), nullable=False)
    trial_type: Mapped[TrialType] = mapped_column(SQLEnum(TrialType), nullable=False)

    # Conditions and interventions
    conditions: Mapped[List[str]] = mapped_column(JSON, nullable=False)  # List of medical conditions
    interventions: Mapped[List[str]] = mapped_column(JSON, nullable=False)  # List of interventions
    keywords: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Eligibility criteria
    eligibility_criteria: Mapped[Optional[str]] = mapped_column(Text)
    inclusion_criteria: Mapped[Optional[List[str]]] = mapped_column(JSON)
    exclusion_criteria: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Trial details
    sponsor: Mapped[Optional[str]] = mapped_column(String(500))
    collaborators: Mapped[Optional[List[str]]] = mapped_column(JSON)
    study_start_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    primary_completion_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    study_completion_date: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Location information
    locations: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False)  # List of location objects
    country: Mapped[Optional[str]] = mapped_column(String(100))

    # Contact information
    contact_name: Mapped[Optional[str]] = mapped_column(String(200))
    contact_phone: Mapped[Optional[str]] = mapped_column(String(20))
    contact_email: Mapped[Optional[str]] = mapped_column(String(255))

    # Study details
    enrollment_target: Mapped[Optional[int]] = mapped_column(Integer)
    enrollment_actual: Mapped[Optional[int]] = mapped_column(Integer)
    study_design: Mapped[Optional[str]] = mapped_column(Text)
    primary_outcomes: Mapped[Optional[List[str]]] = mapped_column(JSON)
    secondary_outcomes: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Vector embeddings for semantic search
    title_embedding: Mapped[Optional[List[float]]] = mapped_column(JSON)
    description_embedding: Mapped[Optional[List[float]]] = mapped_column(JSON)
    criteria_embedding: Mapped[Optional[List[float]]] = mapped_column(JSON)

    # Metadata for search and filtering
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    source_url: Mapped[Optional[str]] = mapped_column(String(1000))
    data_source: Mapped[str] = mapped_column(String(50), default="clinicaltrials.gov")

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    matches: Mapped[List["TrialMatch"]] = relationship("TrialMatch", back_populates="trial")

    def __repr__(self) -> str:
        return f"<ClinicalTrial(id={self.id}, nct_id={self.nct_id}, title={self.title[:50]}...)>"


class TrialMatch(Base):
    """Trial-patient matching records."""
    __tablename__ = "trial_matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    trial_id: Mapped[int] = mapped_column(Integer, ForeignKey("clinical_trials.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)

    # Matching scores and criteria
    overall_match_score: Mapped[float] = mapped_column(Float, nullable=False)  # 0.0 to 1.0
    eligibility_score: Mapped[float] = mapped_column(Float, nullable=False)
    location_score: Mapped[float] = mapped_column(Float, nullable=False)
    phase_appropriateness_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Detailed matching analysis
    matching_criteria: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)  # Detailed criteria matching
    unmet_criteria: Mapped[List[str]] = mapped_column(JSON, nullable=False)  # Criteria not met
    risk_factors: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # User interaction
    user_interested: Mapped[Optional[bool]] = mapped_column(Boolean)
    contacted_trial: Mapped[bool] = mapped_column(Boolean, default=False)
    enrolled_in_trial: Mapped[Optional[bool]] = mapped_column(Boolean)

    # Matching metadata
    matched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    trial: Mapped["ClinicalTrial"] = relationship("ClinicalTrial", back_populates="matches")
    user: Mapped["User"] = relationship("User", back_populates="trial_matches")

    def __repr__(self) -> str:
        return f"<TrialMatch(id={self.id}, trial_id={self.trial_id}, user_id={self.user_id}, score={self.overall_match_score})>"


class TrialSearchQuery(Base):
    """Log of trial search queries for analytics."""
    __tablename__ = "trial_search_queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"))

    # Query details
    search_query: Mapped[str] = mapped_column(Text, nullable=False)
    search_filters: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Results
    results_returned: Mapped[int] = mapped_column(Integer, nullable=False)
    top_trial_ids: Mapped[List[int]] = mapped_column(JSON, nullable=False)

    # User feedback
    user_feedback: Mapped[Optional[int]] = mapped_column(Integer)  # 1-5 rating
    helpful_trials: Mapped[List[int]] = mapped_column(JSON, nullable=False)  # Trial IDs user found helpful

    # Patient context (anonymized)
    patient_context: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<TrialSearchQuery(id={self.id}, query={self.search_query[:50]}...)>"
