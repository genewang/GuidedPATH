"""
RAG (Retrieval-Augmented Generation) service for clinical guidelines
"""

import asyncio
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import os
from pathlib import Path

import aiofiles
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Pinecone
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
import numpy as np
import pinecone
from pydantic import BaseModel
import structlog

from backend.core.config import settings
from backend.apps.guidelines.models import ClinicalGuideline, GuidelineVersion, GuidelineChunk

logger = structlog.get_logger()


class GuidelineDocument(BaseModel):
    """Model for guideline document metadata."""
    title: str
    source: str
    publication_date: str
    version: str
    content_path: str
    guideline_type: str
    target_conditions: List[str]


class RAGService:
    """
    RAG service for processing and querying clinical guidelines.
    """

    def __init__(self):
        """Initialize the RAG service with embeddings and vector store."""
        self.embeddings_model = self._initialize_embeddings()
        self.vector_store = self._initialize_vector_store()
        self.text_splitter = self._initialize_text_splitter()
        self.llm_client = self._initialize_llm_client()

    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        if settings.OPENAI_API_KEY:
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=settings.OPENAI_API_KEY
            )
        else:
            return HuggingFaceEmbeddings(
                model_name=settings.DEFAULT_EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"}
            )

    def _initialize_vector_store(self):
        """Initialize Pinecone vector store."""
        if settings.PINECONE_API_KEY and settings.PINECONE_ENVIRONMENT:
            pinecone.init(
                api_key=settings.PINECONE_API_KEY,
                environment=settings.PINECONE_ENVIRONMENT
            )

            index_name = "guidedpath-guidelines"
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=384 if "MiniLM" in settings.DEFAULT_EMBEDDING_MODEL else 1536,
                    metric="cosine"
                )

            return Pinecone.from_existing_index(index_name, self.embeddings_model)

        return None

    def _initialize_text_splitter(self):
        """Initialize text splitter for document chunking."""
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _initialize_llm_client(self):
        """Initialize LLM client for answer generation."""
        if settings.ANTHROPIC_API_KEY:
            return ChatAnthropic(
                model=settings.DEFAULT_LLM_MODEL,
                temperature=0.1,
                max_tokens=settings.MAX_TOKENS_RESPONSE,
                anthropic_api_key=settings.ANTHROPIC_API_KEY
            )
        return None

    async def process_guideline_document(self, document: GuidelineDocument) -> bool:
        """
        Process a clinical guideline document and store in vector database.

        Args:
            document: Guideline document metadata

        Returns:
            bool: Success status
        """
        try:
            logger.info("Processing guideline document", title=document.title, source=document.source)

            # Load document content
            documents = await self._load_document(document.content_path)

            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)

            # Create guideline record
            guideline = await self._create_guideline_record(document)

            # Process and store chunks
            await self._process_chunks(chunks, guideline.id, document)

            logger.info("Successfully processed guideline document", guideline_id=guideline.id)
            return True

        except Exception as e:
            logger.error("Failed to process guideline document", error=str(e), title=document.title)
            return False

    async def _load_document(self, file_path: str) -> List[Document]:
        """Load document from file path."""
        file_extension = Path(file_path).suffix.lower()

        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        return await asyncio.get_event_loop().run_in_executor(None, loader.load)

    async def _create_guideline_record(self, document: GuidelineDocument) -> ClinicalGuideline:
        """Create a clinical guideline record in the database."""
        # Calculate content hash
        content_hash = await self._calculate_file_hash(document.content_path)

        # Create guideline record (this would typically be done with database session)
        # For now, return a mock object
        return ClinicalGuideline(
            title=document.title,
            source=document.source,
            publication_date=document.publication_date,
            version=document.version,
            guideline_type=document.guideline_type,
            # content_hash=content_hash
        )

    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file content."""
        hash_obj = hashlib.sha256()

        async with aiofiles.open(file_path, 'rb') as file:
            while chunk := await file.read(8192):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    async def _process_chunks(self, chunks: List[Document], guideline_id: int, document: GuidelineDocument):
        """Process document chunks and store in vector database."""
        for i, chunk in enumerate(chunks):
            # Create chunk record
            chunk_text = chunk.page_content
            chunk_title = chunk.metadata.get('title', '')

            # Generate embedding
            embedding_vector = await self._generate_embedding(chunk_text)

            # Store in vector database
            if self.vector_store:
                vector_id = f"{guideline_id}_chunk_{i}"
                metadata = {
                    "guideline_id": guideline_id,
                    "chunk_index": i,
                    "chunk_title": chunk_title,
                    "source": document.source,
                    "guideline_type": document.guideline_type,
                }

                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.vector_store.add_texts([chunk_text], metadatas=[metadata], ids=[vector_id])
                )

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        if isinstance(self.embeddings_model, OpenAIEmbeddings):
            # OpenAI embeddings
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.embeddings_model.embed_query(text)
            )
            return response
        else:
            # HuggingFace embeddings
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.embeddings_model.embed_query(text)
            )
            return response

    async def query_guidelines(self, query: str, patient_context: Optional[Dict] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Query clinical guidelines using RAG.

        Args:
            query: User query
            patient_context: Optional patient information for context
            top_k: Number of top results to return

        Returns:
            Dict containing results and generated answer
        """
        try:
            logger.info("Querying clinical guidelines", query=query[:100])

            # Generate query embedding
            query_embedding = await self._generate_embedding(query)

            # Search vector database
            search_results = await self._search_similar_chunks(query_embedding, query, top_k)

            # Generate contextualized answer
            answer = await self._generate_answer(query, search_results, patient_context)

            # Log query for analytics
            await self._log_query(query, search_results)

            return {
                "query": query,
                "answer": answer,
                "source_chunks": search_results,
                "total_results": len(search_results)
            }

        except Exception as e:
            logger.error("Failed to query guidelines", error=str(e), query=query[:100])
            return {
                "query": query,
                "answer": "I apologize, but I encountered an error while processing your query. Please try again or consult with your healthcare provider.",
                "source_chunks": [],
                "total_results": 0,
                "error": str(e)
            }

    async def _search_similar_chunks(self, query_embedding: List[float], query: str, top_k: int) -> List[Dict]:
        """Search for similar chunks in vector database."""
        if not self.vector_store:
            return []

        # Convert to numpy array for similarity search
        query_vector = np.array(query_embedding)

        # Search vector store
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.vector_store.similarity_search_with_score(query, k=top_k)
        )

        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score)
            })

        return formatted_results

    async def _generate_answer(self, query: str, search_results: List[Dict], patient_context: Optional[Dict] = None) -> str:
        """Generate contextualized answer using LLM."""
        if not self.llm_client or not search_results:
            return "I don't have enough information to provide a comprehensive answer to your question."

        # Prepare context from search results
        context = "\n\n".join([
            f"Source: {result['metadata'].get('source', 'Unknown')}\n{result['content']}"
            for result in search_results[:3]  # Use top 3 results for context
        ])

        # Create patient context if provided
        patient_info = ""
        if patient_context:
            relevant_info = []
            if "conditions" in patient_context:
                relevant_info.append(f"Medical conditions: {', '.join(patient_context['conditions'])}")
            if "age" in patient_context:
                relevant_info.append(f"Age: {patient_context['age']}")
            if "gender" in patient_context:
                relevant_info.append(f"Gender: {patient_context['gender']}")

            if relevant_info:
                patient_info = f"Patient information: {', '.join(relevant_info)}\n\n"

        # Create prompt
        prompt = f"""
        You are a helpful AI assistant providing information about clinical guidelines for cancer and inflammatory diseases.

        {patient_info}Question: {query}

        Based on the following clinical guideline information, provide an accurate, evidence-based answer:

        {context}

        Important guidelines:
        - Always reference the source of information
        - Provide balanced, factual information
        - Do not provide personalized medical advice
        - Recommend consulting healthcare providers for personal medical decisions
        - If information is insufficient, state that clearly

        Answer:"""

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm_client.invoke(prompt).content
            )
            return response.strip()

        except Exception as e:
            logger.error("Failed to generate answer", error=str(e))
            return "I apologize, but I was unable to generate a response. Please consult with your healthcare provider for medical advice."

    async def _log_query(self, query: str, results: List[Dict]):
        """Log query for analytics and model improvement."""
        # TODO: Implement query logging to database
        logger.info("Query logged", query_length=len(query), results_count=len(results))


# Global RAG service instance
rag_service = RAGService()
