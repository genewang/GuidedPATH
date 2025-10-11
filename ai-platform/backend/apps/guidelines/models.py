"""
Clinical guidelines models for RAG system
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import String, DateTime, Boolean, Integer, Text, JSON, Float, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from backend.core.database import Base


class GuidelineSource(enum.Enum):
    """Sources of clinical guidelines."""
    ASCO = "asco"
    ESMO = "esmo"
    EULAR = "eular"
    NCCN = "nccn"
    WHO = "who"
    FDA = "fda"
    EMA = "ema"
    CUSTOM = "custom"


class GuidelineStatus(enum.Enum):
    """Status of clinical guidelines."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"


class ClinicalGuideline(Base):
    """Clinical guideline model for storing guideline metadata."""
    __tablename__ = "clinical_guidelines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source: Mapped[GuidelineSource] = mapped_column(SQLEnum(GuidelineSource), nullable=False)
    source_url: Mapped[Optional[str]] = mapped_column(String(1000))

    # Content metadata
    publication_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    last_updated: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)

    # Classification
    cancer_type: Mapped[Optional[str]] = mapped_column(String(100))
    inflammatory_condition: Mapped[Optional[str]] = mapped_column(String(100))
    treatment_type: Mapped[Optional[str]] = mapped_column(String(100))
    guideline_type: Mapped[str] = mapped_column(String(100))  # diagnosis, treatment, follow-up, etc.

    # Status and approval
    status: Mapped[GuidelineStatus] = mapped_column(SQLEnum(GuidelineStatus), default=GuidelineStatus.ACTIVE)
    approved_by: Mapped[Optional[str]] = mapped_column(String(200))
    approval_date: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Content summary
    abstract: Mapped[Optional[str]] = mapped_column(Text)
    key_recommendations: Mapped[Optional[List[str]]] = mapped_column(JSON)
    target_population: Mapped[Optional[str]] = mapped_column(Text)

    # Metadata for vector search
    embedding_model: Mapped[str] = mapped_column(String(100), default="sentence-transformers/all-MiniLM-L6-v2")
    vector_id: Mapped[Optional[str]] = mapped_column(String(255))  # Pinecone vector ID

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    versions: Mapped[List["GuidelineVersion"]] = relationship("GuidelineVersion", back_populates="guideline")
    queries: Mapped[List["GuidelineQuery"]] = relationship("GuidelineQuery", back_populates="guideline")

    def __repr__(self) -> str:
        return f"<ClinicalGuideline(id={self.id}, title={self.title[:50]}...)>"


class GuidelineVersion(Base):
    """Version history for clinical guidelines."""
    __tablename__ = "guideline_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    guideline_id: Mapped[int] = mapped_column(Integer, ForeignKey("clinical_guidelines.id"), nullable=False)
    version_number: Mapped[str] = mapped_column(String(50), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(255), nullable=False)  # SHA-256 hash of content

    # Content storage
    content_path: Mapped[str] = mapped_column(String(500))  # Path to stored content file
    content_size: Mapped[int] = mapped_column(Integer)  # Size in bytes
    content_format: Mapped[str] = mapped_column(String(50), default="pdf")  # pdf, html, txt, etc.

    # Processing status
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False)
    processing_errors: Mapped[Optional[str]] = mapped_column(Text)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    guideline: Mapped["ClinicalGuideline"] = relationship("ClinicalGuideline", back_populates="versions")

    def __repr__(self) -> str:
        return f"<GuidelineVersion(id={self.id}, guideline_id={self.guideline_id}, version={self.version_number})>"


class GuidelineChunk(Base):
    """Chunked content from clinical guidelines for vector search."""
    __tablename__ = "guideline_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    guideline_id: Mapped[int] = mapped_column(Integer, ForeignKey("clinical_guidelines.id"), nullable=False)
    version_id: Mapped[int] = mapped_column(Integer, ForeignKey("guideline_versions.id"), nullable=False)

    # Chunk information
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)  # Position in document
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_title: Mapped[Optional[str]] = mapped_column(String(200))
    chunk_type: Mapped[str] = mapped_column(String(50), default="text")  # text, table, figure, etc.

    # Vector embeddings
    embedding_vector: Mapped[Optional[List[float]]] = mapped_column(JSON)  # 384-dimensional vector for MiniLM
    vector_id: Mapped[Optional[str]] = mapped_column(String(255))  # Pinecone vector ID

    # Metadata for better search
    section_path: Mapped[Optional[str]] = mapped_column(String(500))  # e.g., "1.2.3 Treatment Options"
    keywords: Mapped[Optional[List[str]]] = mapped_column(JSON)
    entities: Mapped[Optional[Dict[str, List[str]]]] = mapped_column(JSON)  # drugs, conditions, procedures

    # Quality metrics
    relevance_score: Mapped[Optional[float]] = mapped_column(Float)
    citation_count: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<GuidelineChunk(id={self.id}, guideline_id={self.guideline_id}, chunk_index={self.chunk_index})>"


class GuidelineQuery(Base):
    """Log of queries made against clinical guidelines."""
    __tablename__ = "guideline_queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    guideline_id: Mapped[int] = mapped_column(Integer, ForeignKey("clinical_guidelines.id"), nullable=False)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"))

    # Query information
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_embedding: Mapped[Optional[List[float]]] = mapped_column(JSON)

    # Results and feedback
    results_returned: Mapped[int] = mapped_column(Integer, default=0)
    top_result_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("guideline_chunks.id"))
    user_feedback: Mapped[Optional[int]] = mapped_column(Integer)  # 1-5 rating
    feedback_comment: Mapped[Optional[str]] = mapped_column(Text)

    # Context
    patient_context: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)  # anonymized patient info
    clinical_context: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<GuidelineQuery(id={self.id}, query_text={self.query_text[:50]}...)>"
