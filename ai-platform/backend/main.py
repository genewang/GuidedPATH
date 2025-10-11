"""
GuidedPATH - AI-Powered Healthcare Platform
Main FastAPI application with comprehensive AI services for healthcare
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import structlog

from backend.core.config import settings
from backend.core.database import init_db, close_db
from backend.core.logging import setup_logging
from backend.apps.auth.router import router as auth_router
from backend.apps.guidelines.router import router as guidelines_router
from backend.apps.trials.router import router as trials_router
from backend.apps.medication.router import router as medication_router
from backend.apps.symptoms.router import router as symptoms_router
from backend.apps.mental_health.router import router as mental_health_router
from backend.apps.chat.router import router as chat_router
from backend.apps.users.router import router as users_router
from backend.core.security.middleware import SecurityMiddleware

# Setup structured logging
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting GuidedPATH AI Platform")

    # Initialize database connections
    await init_db()

    # Initialize AI model connections
    # TODO: Initialize AI model clients (OpenAI, Anthropic, etc.)

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down GuidedPATH AI Platform")

    # Close database connections
    await close_db()

    # Close AI model connections
    # TODO: Close AI model clients

    logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Set up CORS
    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Add trusted host middleware
    if not settings.DEBUG:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS,
        )

    # Add security middleware
    app.add_middleware(SecurityMiddleware)

    # Include routers
    app.include_router(auth_router, prefix=f"{settings.API_V1_STR}/auth", tags=["authentication"])
    app.include_router(users_router, prefix=f"{settings.API_V1_STR}/users", tags=["users"])
    app.include_router(guidelines_router, prefix=f"{settings.API_V1_STR}/guidelines", tags=["guidelines"])
    app.include_router(trials_router, prefix=f"{settings.API_V1_STR}/trials", tags=["clinical-trials"])
    app.include_router(medication_router, prefix=f"{settings.API_V1_STR}/medication", tags=["medication"])
    app.include_router(symptoms_router, prefix=f"{settings.API_V1_STR}/symptoms", tags=["symptoms"])
    app.include_router(mental_health_router, prefix=f"{settings.API_V1_STR}/mental-health", tags=["mental-health"])
    app.include_router(chat_router, prefix=f"{settings.API_V1_STR}/chat", tags=["chat"])

    @app.get("/health")
    async def health_check():
        """Health check endpoint for load balancers and monitoring."""
        return {"status": "healthy", "version": settings.VERSION}

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "description": settings.PROJECT_DESCRIPTION,
            "docs": "/docs",
            "health": "/health"
        }

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Middleware to log all requests."""
        start_time = time.time()

        response = await call_next(request)

        process_time = time.time() - start_time
        logger.info(
            "Request processed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=process_time,
        )

        return response

    return app


# Create the application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn
    import time

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
