import uuid
from pydantic import BaseModel, Field, HttpUrl
from typing import Literal, List

# --- Ingestion Models ---

class IngestRequest(BaseModel):
    url: HttpUrl = Field(..., description="The URL of the web content to ingest.")

class IngestResponse(BaseModel):
    job_id: uuid.UUID = Field(..., description="The unique ID for the ingestion job.")
    status: Literal["PENDING"] = Field("PENDING", description="The initial status of the job.")
    message: str = Field("URL has been accepted and is pending processing.", description="A confirmation message.")


# --- Query Models ---

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="The question to ask the RAG engine.")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated, grounded answer.")
    sources: List[str] = Field(..., description="List of source URLs that contributed to the answer.")