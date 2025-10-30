import uuid
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware # Import the middleware
from sqlalchemy.orm import Session

from app.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.celery_client import celery_app
from app.database import SessionLocal, IngestionJob, create_db_and_tables
from app.query import query_rag_engine

# Initialize the FastAPI application
app = FastAPI(
    title="Scalable RAG Engine",
    description="An API for asynchronous ingestion and querying of web content.",
    version="1.0.0"
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


# --- Database Setup ---

@app.on_event("startup")
def on_startup():
    print("API is starting up. Creating database tables...")
    create_db_and_tables()
    print("Database tables created (if not existed).")

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