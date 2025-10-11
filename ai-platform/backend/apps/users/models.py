"""
User models for the GuidedPATH platform
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, DateTime, Boolean, Text, JSON, Integer, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from backend.core.database import Base


class UserRole(enum.Enum):
    """User roles in the system."""
    PATIENT = "patient"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"
    ADMIN = "admin"


class UserStatus(enum.Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class User(Base):
    """User model for authentication and profile management."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    # Profile information
    first_name: Mapped[Optional[str]] = mapped_column(String(100))
    last_name: Mapped[Optional[str]] = mapped_column(String(100))
    phone_number: Mapped[Optional[str]] = mapped_column(String(20))
    date_of_birth: Mapped[Optional[datetime]] = mapped_column(DateTime)
    gender: Mapped[Optional[str]] = mapped_column(String(50))

    # Healthcare profile
    medical_record_number: Mapped[Optional[str]] = mapped_column(String(100))
    emergency_contact_name: Mapped[Optional[str]] = mapped_column(String(200))
    emergency_contact_phone: Mapped[Optional[str]] = mapped_column(String(20))
    primary_physician_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"))

    # Account management
    role: Mapped[UserRole] = mapped_column(SQLEnum(UserRole), default=UserRole.PATIENT)
    status: Mapped[UserStatus] = mapped_column(SQLEnum(UserStatus), default=UserStatus.ACTIVE)
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    phone_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    # Privacy and consent
    consent_given: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime)
    data_sharing_preferences: Mapped[Optional[dict]] = mapped_column(JSON)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Relationships
    primary_physician: Mapped[Optional["User"]] = relationship("User", remote_side=[id])
    patient_records: Mapped[List["PatientRecord"]] = relationship("PatientRecord", back_populates="patient")
    refresh_tokens: Mapped[List["RefreshToken"]] = relationship("RefreshToken", back_populates="user")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, role={self.role.value})>"


class PatientRecord(Base):
    """Detailed patient medical record information."""
    __tablename__ = "patient_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)

    # Medical condition information
    primary_diagnosis: Mapped[Optional[str]] = mapped_column(Text)
    secondary_diagnoses: Mapped[Optional[List[str]]] = mapped_column(JSON)
    current_treatments: Mapped[Optional[List[str]]] = mapped_column(JSON)
    treatment_history: Mapped[Optional[dict]] = mapped_column(JSON)

    # Clinical information
    cancer_stage: Mapped[Optional[str]] = mapped_column(String(50))
    inflammatory_condition: Mapped[Optional[str]] = mapped_column(String(100))
    comorbidities: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Genomic and biomarker data
    genetic_markers: Mapped[Optional[dict]] = mapped_column(JSON)
    biomarkers: Mapped[Optional[dict]] = mapped_column(JSON)

    # Current status
    performance_status: Mapped[Optional[str]] = mapped_column(String(50))  # ECOG, Karnofsky scale
    symptom_severity: Mapped[Optional[str]] = mapped_column(String(50))

    # Preferences and settings
    communication_preferences: Mapped[Optional[dict]] = mapped_column(JSON)
    notification_settings: Mapped[Optional[dict]] = mapped_column(JSON)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient: Mapped["User"] = relationship("User", back_populates="patient_records")
    symptom_reports: Mapped[List["SymptomReport"]] = relationship("SymptomReport", back_populates="patient_record")
    medication_records: Mapped[List["MedicationRecord"]] = relationship("MedicationRecord", back_populates="patient_record")

    def __repr__(self) -> str:
        return f"<PatientRecord(id={self.id}, user_id={self.user_id})>"


class RefreshToken(Base):
    """Refresh token model for authentication."""
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    token: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="refresh_tokens")

    def __repr__(self) -> str:
        return f"<RefreshToken(id={self.id}, user_id={self.user_id})>"
