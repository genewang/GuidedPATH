"""
Symptom checker and triage models
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import String, DateTime, Boolean, Integer, Text, JSON, Float, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from backend.core.database import Base


class SymptomSeverity(enum.Enum):
    """Severity levels for symptoms."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class TriageUrgency(enum.Enum):
    """Urgency levels for triage recommendations."""
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"
    IMMEDIATE = "immediate"


class SymptomCategory(enum.Enum):
    """Categories of symptoms."""
    GENERAL = "general"
    CARDIOVASCULAR = "cardiovascular"
    RESPIRATORY = "respiratory"
    NEUROLOGICAL = "neurological"
    GASTROINTESTINAL = "gastrointestinal"
    MUSCULOSKELETAL = "musculoskeletal"
    DERMATOLOGICAL = "dermatological"
    PSYCHIATRIC = "psychiatric"
    ONCOLOGICAL = "oncological"
    IMMUNOLOGICAL = "immunological"


class SymptomReport(Base):
    """Patient symptom report model."""
    __tablename__ = "symptom_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_record_id: Mapped[int] = mapped_column(Integer, ForeignKey("patient_records.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)

    # Symptom details
    primary_symptom: Mapped[str] = mapped_column(String(200), nullable=False)
    symptom_description: Mapped[Optional[str]] = mapped_column(Text)
    severity: Mapped[SymptomSeverity] = mapped_column(SQLEnum(SymptomSeverity), nullable=False)
    category: Mapped[SymptomCategory] = mapped_column(SQLEnum(SymptomCategory), nullable=False)

    # Additional symptoms
    associated_symptoms: Mapped[Optional[List[str]]] = mapped_column(JSON)
    duration: Mapped[Optional[str]] = mapped_column(String(100))  # e.g., "3 days", "2 weeks"
    frequency: Mapped[Optional[str]] = mapped_column(String(100))  # e.g., "constant", "intermittent"

    # Context
    onset_timing: Mapped[Optional[str]] = mapped_column(String(100))  # sudden, gradual
    aggravating_factors: Mapped[Optional[List[str]]] = mapped_column(JSON)
    alleviating_factors: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Medical context
    current_medications: Mapped[Optional[List[str]]] = mapped_column(JSON)
    recent_treatments: Mapped[Optional[List[str]]] = mapped_column(JSON)
    relevant_history: Mapped[Optional[str]] = mapped_column(Text)

    # AI analysis
    ai_confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    ai_analysis_notes: Mapped[Optional[str]] = mapped_column(Text)

    # Timestamps
    reported_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient_record: Mapped["PatientRecord"] = relationship("PatientRecord", back_populates="symptom_reports")
    triage_result: Mapped["TriageResult"] = relationship("TriageResult", back_populates="symptom_report", uselist=False)

    def __repr__(self) -> str:
        return f"<SymptomReport(id={self.id}, symptom={self.primary_symptom}, severity={self.severity.value})>"


class TriageResult(Base):
    """AI-generated triage recommendation."""
    __tablename__ = "triage_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    symptom_report_id: Mapped[int] = mapped_column(Integer, ForeignKey("symptom_reports.id"), nullable=False)

    # Triage recommendation
    urgency_level: Mapped[TriageUrgency] = mapped_column(SQLEnum(TriageUrgency), nullable=False)
    recommended_action: Mapped[str] = mapped_column(Text, nullable=False)
    time_frame: Mapped[Optional[str]] = mapped_column(String(100))  # e.g., "within 24 hours"

    # Clinical reasoning
    reasoning: Mapped[Optional[str]] = mapped_column(Text)
    differential_diagnosis: Mapped[Optional[List[str]]] = mapped_column(JSON)
    red_flags: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Care recommendations
    self_care_advice: Mapped[Optional[List[str]]] = mapped_column(JSON)
    when_to_seek_care: Mapped[Optional[List[str]]] = mapped_column(JSON)
    emergency_indicators: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Escalation
    requires_clinician_review: Mapped[bool] = mapped_column(Boolean, default=False)
    clinician_notes: Mapped[Optional[str]] = mapped_column(Text)
    follow_up_recommended: Mapped[bool] = mapped_column(Boolean, default=False)

    # AI model information
    ai_model_used: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[Optional[str]] = mapped_column(String(50))

    # Timestamps
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Relationships
    symptom_report: Mapped["SymptomReport"] = relationship("SymptomReport", back_populates="triage_result")

    def __repr__(self) -> str:
        return f"<TriageResult(id={self.id}, urgency={self.urgency_level.value}, confidence={self.confidence_score})>"


class SymptomPattern(Base):
    """Pattern analysis for symptom tracking."""
    __tablename__ = "symptom_patterns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_record_id: Mapped[int] = mapped_column(Integer, ForeignKey("patient_records.id"), nullable=False)

    # Pattern identification
    pattern_type: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., "cyclic", "progressive"
    symptoms_involved: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    frequency: Mapped[Optional[str]] = mapped_column(String(100))

    # Pattern characteristics
    typical_duration: Mapped[Optional[str]] = mapped_column(String(100))
    typical_severity: Mapped[SymptomSeverity] = mapped_column(SQLEnum(SymptomSeverity), nullable=False)
    triggers: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Clinical significance
    clinical_significance: Mapped[Optional[str]] = mapped_column(Text)
    monitoring_recommendations: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # AI analysis
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    first_observed: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    last_observed: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Timestamps
    identified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<SymptomPattern(id={self.id}, type={self.pattern_type}, symptoms={len(self.symptoms_involved)})>"


class VitalSigns(Base):
    """Vital signs tracking for symptom context."""
    __tablename__ = "vital_signs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    symptom_report_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("symptom_reports.id"))
    patient_record_id: Mapped[int] = mapped_column(Integer, ForeignKey("patient_records.id"), nullable=False)

    # Vital signs
    temperature: Mapped[Optional[float]] = mapped_column(Float)  # Celsius
    heart_rate: Mapped[Optional[int]] = mapped_column(Integer)   # BPM
    blood_pressure_systolic: Mapped[Optional[int]] = mapped_column(Integer)
    blood_pressure_diastolic: Mapped[Optional[int]] = mapped_column(Integer)
    respiratory_rate: Mapped[Optional[int]] = mapped_column(Integer)  # Breaths per minute
    oxygen_saturation: Mapped[Optional[int]] = mapped_column(Integer)  # Percentage

    # Additional measurements
    weight: Mapped[Optional[float]] = mapped_column(Float)  # kg
    height: Mapped[Optional[float]] = mapped_column(Float)  # cm
    bmi: Mapped[Optional[float]] = mapped_column(Float)

    # Context
    measurement_method: Mapped[Optional[str]] = mapped_column(String(50))  # manual, device, estimated
    notes: Mapped[Optional[str]] = mapped_column(Text)

    # Timestamps
    measured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<VitalSigns(id={self.id}, measured_at={self.measured_at})>"


class SymptomQuery(Base):
    """Log of symptom checker queries for analytics."""
    __tablename__ = "symptom_queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"))

    # Query details
    symptoms_described: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    query_text: Mapped[Optional[str]] = mapped_column(Text)
    context_provided: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Results
    triage_urgency: Mapped[TriageUrgency] = mapped_column(SQLEnum(TriageUrgency), nullable=False)
    results_summary: Mapped[str] = mapped_column(Text, nullable=False)
    ai_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # User feedback
    user_feedback: Mapped[Optional[int]] = mapped_column(Integer)  # 1-5 rating
    helpful: Mapped[Optional[bool]] = mapped_column(Boolean)
    followed_recommendation: Mapped[Optional[bool]] = mapped_column(Boolean)

    # Timestamps
    queried_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<SymptomQuery(id={self.id}, urgency={self.triage_urgency.value}, symptoms={len(self.symptoms_described)})>"
