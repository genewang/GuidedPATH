"""
Medication management models
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import String, DateTime, Boolean, Integer, Text, JSON, Float, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from backend.core.database import Base


class MedicationStatus(enum.Enum):
    """Status of medication records."""
    ACTIVE = "active"
    DISCONTINUED = "discontinued"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"


class InteractionSeverity(enum.Enum):
    """Severity levels for drug interactions."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class Medication(Base):
    """Medication model for patient medication records."""
    __tablename__ = "medications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)  # Generic or brand name
    generic_name: Mapped[Optional[str]] = mapped_column(String(255))
    brand_names: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Drug identification
    rxnorm_code: Mapped[Optional[str]] = mapped_column(String(50))  # RxNorm identifier
    atc_code: Mapped[Optional[str]] = mapped_column(String(10))     # ATC classification
    drugbank_id: Mapped[Optional[str]] = mapped_column(String(20))

    # Drug classification
    drug_class: Mapped[Optional[str]] = mapped_column(String(100))
    therapeutic_class: Mapped[Optional[str]] = mapped_column(String(100))
    pharmacologic_class: Mapped[Optional[str]] = mapped_column(String(100))

    # Usage information
    indication: Mapped[Optional[str]] = mapped_column(Text)  # What it's used for
    contraindications: Mapped[Optional[List[str]]] = mapped_column(JSON)
    warnings: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Vector embeddings for semantic search
    name_embedding: Mapped[Optional[List[float]]] = mapped_column(JSON)
    indication_embedding: Mapped[Optional[List[float]]] = mapped_column(JSON)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient_records: Mapped[List["MedicationRecord"]] = relationship("MedicationRecord", back_populates="medication")
    interactions: Mapped[List["MedicationInteraction"]] = relationship("MedicationInteraction", back_populates="medication")

    def __repr__(self) -> str:
        return f"<Medication(id={self.id}, name={self.name})>"


class MedicationRecord(Base):
    """Patient-specific medication record."""
    __tablename__ = "medication_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_record_id: Mapped[int] = mapped_column(Integer, ForeignKey("patient_records.id"), nullable=False)
    medication_id: Mapped[int] = mapped_column(Integer, ForeignKey("medications.id"), nullable=False)

    # Prescription details
    prescribed_by: Mapped[Optional[str]] = mapped_column(String(200))  # Prescribing physician
    prescribed_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    dosage: Mapped[Optional[str]] = mapped_column(String(100))  # e.g., "50mg", "10mg/mL"
    frequency: Mapped[Optional[str]] = mapped_column(String(100))  # e.g., "twice daily", "every 6 hours"
    duration: Mapped[Optional[str]] = mapped_column(String(100))  # e.g., "30 days", "ongoing"

    # Administration
    route: Mapped[Optional[str]] = mapped_column(String(50))  # oral, IV, topical, etc.
    instructions: Mapped[Optional[str]] = mapped_column(Text)

    # Status and adherence
    status: Mapped[MedicationStatus] = mapped_column(SQLEnum(MedicationStatus), default=MedicationStatus.ACTIVE)
    start_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    end_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    adherence_score: Mapped[Optional[float]] = mapped_column(Float)  # 0.0 to 1.0

    # Pharmacy and refill information
    pharmacy_name: Mapped[Optional[str]] = mapped_column(String(200))
    pharmacy_phone: Mapped[Optional[str]] = mapped_column(String(20))
    refills_remaining: Mapped[Optional[int]] = mapped_column(Integer)
    last_refill_date: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Clinical notes
    notes: Mapped[Optional[str]] = mapped_column(Text)
    side_effects: Mapped[Optional[List[str]]] = mapped_column(JSON)
    effectiveness_notes: Mapped[Optional[str]] = mapped_column(Text)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient_record: Mapped["PatientRecord"] = relationship("PatientRecord", back_populates="medication_records")
    medication: Mapped["Medication"] = relationship("Medication", back_populates="patient_records")
    adherence_logs: Mapped[List["MedicationAdherence"]] = relationship("MedicationAdherence", back_populates="medication_record")

    def __repr__(self) -> str:
        return f"<MedicationRecord(id={self.id}, medication_id={self.medication_id}, status={self.status.value})>"


class MedicationInteraction(Base):
    """Drug interaction information."""
    __tablename__ = "medication_interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    medication_id: Mapped[int] = mapped_column(Integer, ForeignKey("medications.id"), nullable=False)
    interacting_medication_id: Mapped[int] = mapped_column(Integer, ForeignKey("medications.id"), nullable=False)

    # Interaction details
    severity: Mapped[InteractionSeverity] = mapped_column(SQLEnum(InteractionSeverity), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    mechanism: Mapped[Optional[str]] = mapped_column(Text)
    clinical_effects: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Management recommendations
    monitoring_needed: Mapped[bool] = mapped_column(Boolean, default=False)
    monitoring_parameters: Mapped[Optional[List[str]]] = mapped_column(JSON)
    management_strategies: Mapped[Optional[List[str]]] = mapped_column(JSON)
    contraindication: Mapped[bool] = mapped_column(Boolean, default=False)

    # Evidence and sources
    evidence_level: Mapped[Optional[str]] = mapped_column(String(50))  # A, B, C, etc.
    source_references: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    medication: Mapped["Medication"] = relationship("Medication", foreign_keys=[medication_id], back_populates="interactions")

    def __repr__(self) -> str:
        return f"<MedicationInteraction(id={self.id}, severity={self.severity.value})>"


class MedicationAdherence(Base):
    """Medication adherence tracking."""
    __tablename__ = "medication_adherence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    medication_record_id: Mapped[int] = mapped_column(Integer, ForeignKey("medication_records.id"), nullable=False)

    # Adherence tracking
    scheduled_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    taken_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    taken: Mapped[bool] = mapped_column(Boolean, default=False)

    # Context
    adherence_method: Mapped[Optional[str]] = mapped_column(String(50))  # manual, smart_pillbox, app
    notes: Mapped[Optional[str]] = mapped_column(Text)

    # Side effects tracking
    side_effects_reported: Mapped[Optional[List[str]]] = mapped_column(JSON)
    severity_rating: Mapped[Optional[int]] = mapped_column(Integer)  # 1-5 scale

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    medication_record: Mapped["MedicationRecord"] = relationship("MedicationRecord", back_populates="adherence_logs")

    def __repr__(self) -> str:
        return f"<MedicationAdherence(id={self.id}, scheduled_date={self.scheduled_date}, taken={self.taken})>"


class MedicationReminder(Base):
    """Medication reminders and scheduling."""
    __tablename__ = "medication_reminders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    medication_record_id: Mapped[int] = mapped_column(Integer, ForeignKey("medication_records.id"), nullable=False)

    # Reminder schedule
    reminder_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    reminder_type: Mapped[str] = mapped_column(String(50), nullable=False)  # scheduled, as_needed, refill
    message: Mapped[Optional[str]] = mapped_column(Text)

    # Notification settings
    notification_method: Mapped[str] = mapped_column(String(50), nullable=False)  # push, sms, email, voice
    notification_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    notification_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Response tracking
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    response: Mapped[Optional[str]] = mapped_column(String(100))  # taken, skipped, rescheduled

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    medication_record: Mapped["MedicationRecord"] = relationship("MedicationRecord", back_populates="reminders")

    def __repr__(self) -> str:
        return f"<MedicationReminder(id={self.id}, time={self.reminder_time}, type={self.reminder_type})>"
