"""
API router for clinical guidelines service
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel
import structlog

from backend.core.database import get_db
from backend.apps.guidelines.rag_service import rag_service, GuidelineDocument

logger = structlog.get_logger()

router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for guideline queries."""
    query: str
    patient_context: Optional[Dict[str, Any]] = None
    top_k: int = 5


class QueryResponse(BaseModel):
    """Response model for guideline queries."""
    query: str
    answer: str
    source_chunks: List[Dict[str, Any]]
    total_results: int
    error: Optional[str] = None


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    title: str
    source: str
    publication_date: str
    version: str
    guideline_type: str
    target_conditions: List[str]


@router.post("/query", response_model=QueryResponse)
async def query_guidelines(
    request: QueryRequest,
    db = Depends(get_db)
):
    """
    Query clinical guidelines using RAG (Retrieval-Augmented Generation).

    This endpoint allows users to ask questions about clinical guidelines
    and receive evidence-based answers with source citations.
    """
    try:
        result = await rag_service.query_guidelines(
            query=request.query,
            patient_context=request.patient_context,
            top_k=request.top_k
        )

        return QueryResponse(**result)

    except Exception as e:
        logger.error("Error in guidelines query", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to process guidelines query"
        )


@router.post("/upload")
async def upload_guideline_document(
    request: DocumentUploadRequest,
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    """
    Upload and process a clinical guideline document.

    This endpoint accepts guideline document metadata and processes
    the document in the background for inclusion in the RAG system.
    """
    try:
        # TODO: Handle file upload
        # For now, create a mock document for testing
        document = GuidelineDocument(
            title=request.title,
            source=request.source,
            publication_date=request.publication_date,
            version=request.version,
            content_path="/tmp/mock_guideline.pdf",  # This should be the uploaded file path
            guideline_type=request.guideline_type,
            target_conditions=request.target_conditions
        )

        # Process document in background
        background_tasks.add_task(
            rag_service.process_guideline_document,
            document
        )

        return {
            "message": "Document upload initiated",
            "title": request.title,
            "status": "processing"
        }

    except Exception as e:
        logger.error("Error uploading guideline document", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to upload guideline document"
        )


@router.get("/sources")
async def list_guideline_sources():
    """List available guideline sources in the system."""
    # TODO: Query database for available sources
    return {
        "sources": [
            "ASCO",
            "ESMO",
            "EULAR",
            "NCCN",
            "WHO",
            "FDA"
        ]
    }


@router.get("/search")
async def search_guidelines(
    q: str = Query(..., description="Search query"),
    source: Optional[str] = Query(None, description="Filter by source"),
    condition: Optional[str] = Query(None, description="Filter by medical condition"),
    top_k: int = Query(5, description="Number of results to return"),
    db = Depends(get_db)
):
    """
    Search clinical guidelines with optional filters.

    This endpoint provides a simple search interface for finding
    relevant guidelines without full RAG processing.
    """
    try:
        # TODO: Implement search logic with database queries and vector similarity

        # For now, return mock results
        return {
            "query": q,
            "results": [
                {
                    "id": 1,
                    "title": "Sample Guideline",
                    "source": source or "ASCO",
                    "similarity_score": 0.85,
                    "snippet": "This is a sample guideline snippet..."
                }
            ],
            "total": 1
        }

    except Exception as e:
        logger.error("Error in guidelines search", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to search guidelines"
        )
