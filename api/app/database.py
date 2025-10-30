import os
from sqlalchemy import create_engine, Column, String, DateTime, func, UUID as PG_UUID
from sqlalchemy.orm import sessionmaker, declarative_base
import uuid

# Get the database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set.")

# Create the SQLAlchemy engine, which is the entry point to the database.
engine = create_engine(DATABASE_URL)

# SessionLocal class will be used to create individual database sessions.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for our declarative models.
Base = declarative_base()

# --- Database Model Definition ---

class IngestionJob(Base):
    """
    SQLAlchemy model for the ingestion_jobs table.
    This class defines the schema for the table.
    """
    __tablename__ = "ingestion_jobs"

    # Columns
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column(String, nullable=False, unique=True)
    status = Column(String, nullable=False, default="PENDING")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<IngestionJob(id={self.id}, url='{self.url}', status='{self.status}')>"


def create_db_and_tables():
    """
    Function to create the database tables.
    This is called on API startup.
    """
    Base.metadata.create_all(bind=engine) 