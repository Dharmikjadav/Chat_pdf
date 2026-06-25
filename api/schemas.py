"""Pydantic request/response schemas for the PDF chatbot API."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    uploaded = "uploaded"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    deleted = "deleted"


class ProcessingStage(str, Enum):
    uploaded = "uploaded"
    rendering_pages = "rendering_pages"
    page_classification = "page_classification"
    opencv_region_detection = "opencv_region_detection"
    region_cropping = "region_cropping"
    table_extraction = "table_extraction"
    chart_extraction = "chart_extraction"
    json_validation = "json_validation"
    pageindex_building = "pageindex_building"
    completed = "completed"


class Document(BaseModel):
    doc_id: str
    filename: str
    status: DocumentStatus
    total_pages: int = 0
    processed_pages: int = 0
    progress_percent: float = 0.0
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None


class GetDocsResponse(BaseModel):
    documents: List[Document]


class PostDocResponse(BaseModel):
    doc_id: str
    filename: str
    status: DocumentStatus
    message: str


class ProcessingError(BaseModel):
    stage: Optional[str] = None
    message: str
    page_number: Optional[int] = None


class ProgressResponse(BaseModel):
    doc_id: str
    status: DocumentStatus
    total_pages: int = 0
    processed_pages: int = 0
    current_page: int = 0
    progress_percent: float = 0.0
    stage: ProcessingStage = ProcessingStage.uploaded
    errors: List[ProcessingError] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    doc_id: str
    deleted: bool
    message: str


class AskRequest(BaseModel):
    doc_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=50,
                                 description="How many ranked context nodes to retrieve (engine default if omitted).")


class Source(BaseModel):
    page_number: Optional[int] = None
    node_type: Optional[str] = None
    source: str = "pageindex"
    snippet: str = ""


class AskResponse(BaseModel):
    doc_id: str
    question: str
    answer: str
    sources: List[Source] = Field(default_factory=list)
