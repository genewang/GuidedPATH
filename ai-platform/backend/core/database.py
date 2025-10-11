"""
Database connection management and initialization
"""

from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
import structlog

from backend.core.config import settings

logger = structlog.get_logger()


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    pool_size=20,
    max_overflow=30,
)

# Create async session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """
    Initialize database connections and create tables if needed.
    """
    try:
        # Import all models to ensure they are registered with SQLAlchemy
        from backend.apps.users.models import User  # noqa
        from backend.apps.auth.models import RefreshToken  # noqa
        from backend.apps.guidelines.models import ClinicalGuideline, GuidelineVersion  # noqa
        from backend.apps.trials.models import ClinicalTrial, TrialMatch  # noqa
        from backend.apps.medication.models import Medication, MedicationInteraction  # noqa
        from backend.apps.symptoms.models import SymptomReport, TriageResult  # noqa

        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error("Database initialization failed", error=str(e))
        raise


async def close_db() -> None:
    """
    Close database connections.
    """
    try:
        await engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error("Error closing database connections", error=str(e))


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database sessions.
    """
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# MongoDB connection (for unstructured data)
from motor.motor_asyncio import AsyncIOMotorClient

mongodb_client: Optional[AsyncIOMotorClient] = None
mongodb_database = None


async def init_mongodb() -> None:
    """
    Initialize MongoDB connection.
    """
    global mongodb_client, mongodb_database

    try:
        mongodb_client = AsyncIOMotorClient(settings.MONGODB_URL)
        mongodb_database = mongodb_client.guidedpath

        # Test connection
        await mongodb_client.admin.command('ping')
        logger.info("MongoDB connection established")

    except Exception as e:
        logger.error("MongoDB connection failed", error=str(e))
        raise


async def close_mongodb() -> None:
    """
    Close MongoDB connection.
    """
    global mongodb_client

    if mongodb_client:
        mongodb_client.close()
        logger.info("MongoDB connection closed")


async def get_mongodb():
    """
    Dependency for getting MongoDB database instance.
    """
    if mongodb_database is None:
        await init_mongodb()

    return mongodb_database
