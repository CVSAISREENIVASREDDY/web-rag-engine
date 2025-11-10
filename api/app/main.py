import uuid
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware # Import the middleware
from sqlalchemy.orm import Session

from fastapi import UploadFile, File
import tempfile
import shutil

from app.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.celery_client import celery_app
from app.database import SessionLocal, IngestionJob, create_db_and_tables
from app.query import query_rag_engine
from contextlib import asynccontextmanager

# Initialize the FastAPI application
@asynccontextmanager
async def lifespan(app):
    print("API is starting up. Creating database tables...")
    create_db_and_tables()
    print("Database tables created (if not existed).")
    yield

app = FastAPI(
    title="Scalable RAG Engine",
    description="An API for asynchronous ingestion and querying of web content.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Add CORS Middleware ---
# This is the fix for the "Failed to fetch" error in the browser UI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- API Endpoints ---

@app.get("/", tags=["Health Check"])
def read_root():
    return {"message": "RAG Engine API is running"}


@app.post("/ingest-url",
          response_model=IngestResponse,
          status_code=status.HTTP_202_ACCEPTED,
          tags=["Ingestion"])
def ingest_url(request: IngestRequest, db: Session = Depends(get_db)):
    """
    Accepts a URL, saves it to the database, and schedules it for processing.
    """
    try:
        existing_job = db.query(IngestionJob).filter(IngestionJob.url == str(request.url)).first()
        if existing_job:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"URL has already been submitted. Job ID: {existing_job.id}, Status: {existing_job.status}"
            )

        new_job = IngestionJob(url=str(request.url), status="PENDING")
        db.add(new_job)
        db.commit()
        db.refresh(new_job)
        
        job_id = new_job.id

        celery_app.send_task(
            "process_url_task",
            args=[str(job_id), str(request.url)]
        )
        
        return IngestResponse(job_id=job_id)

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create and schedule job: {str(e)}"
        )

@app.post("/query",
          response_model=QueryResponse,
          tags=["Query"])
def query(request: QueryRequest):
    """
    Accepts a user query and returns a grounded answer from the knowledge base.
    """
    try:
        result = query_rag_engine(request.query)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during the query process: {str(e)}"
        )



@app.post("/ingest-file",
          status_code=status.HTTP_202_ACCEPTED,
          tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Accepts a PDF or DOCX file, saves it, and schedules it for processing.
    """
    import os

    allowed_types = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF and DOCX files are supported."
        )

    try:
        # Save file to a temp location
        suffix = ".pdf" if file.content_type == "application/pdf" else ".docx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmpfile:
            shutil.copyfileobj(file.file, tmpfile)
            tmp_filepath = tmpfile.name

        job_filename = os.path.basename(tmp_filepath)
        from app.database import IngestionJob

        existing_job = db.query(IngestionJob).filter(IngestionJob.url == job_filename).first()
        if existing_job:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"File has already been submitted. Job ID: {existing_job.id}, Status: {existing_job.status}"
            )

        new_job = IngestionJob(url=job_filename, status="PENDING")
        db.add(new_job)
        db.commit()
        db.refresh(new_job)
        job_id = new_job.id

        celery_app.send_task(
            "process_file_task",
            args=[str(job_id), tmp_filepath, file.content_type]
        )

        return {
            "job_id": job_id,
            "status": "PENDING",
            "message": "File has been accepted and is pending processing."
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create and schedule file ingest job: {str(e)}"
        )

