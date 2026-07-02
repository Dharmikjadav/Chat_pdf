"""FastAPI app exposing the offline PDF chatbot pipeline.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

import config  # noqa: F401 - importing loads .env (Ollama/model/device settings)
from api.schemas import (
    AskRequest,
    AskResponse,
    DeleteResponse,
    DocumentStatus,
    GetDocsResponse,
    PostDocResponse,
    ProgressResponse,
    Source,
)
from api.services import ask_service, document_service, processing_service
from api.services import storage_service as storage
from api.utils import file_utils
from utils.logger import setup_logging

logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    storage.init_storage()
    logger.info("PDF chatbot API ready (data dir: %s)", storage.DATA_DIR)
    try:
        yield
    finally:
        processing_service.shutdown()


app = FastAPI(
    title="Offline PDF Chatbot API",
    description="Upload PDFs, process them through the offline extraction pipeline, and ask questions.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return {"status": "ok", "docs": "/docs"}


@app.get("/get_doc", response_model=GetDocsResponse, tags=["documents"])
def get_doc() -> GetDocsResponse:
    """Return all processed (and in-progress) documents."""
    return GetDocsResponse(documents=document_service.list_documents())


@app.post("/post_doc", response_model=PostDocResponse, status_code=202, tags=["documents"])
async def post_doc(file: UploadFile = File(...)) -> PostDocResponse:
    """Upload a PDF and start background processing."""
    head = await file.read(2048)
    if not file_utils.is_pdf(file.filename, head):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF.")
    await file.seek(0)

    doc_id = file_utils.new_doc_id()
    while storage.upload_dir(doc_id).exists() or storage.get_document(doc_id) is not None:
        doc_id = file_utils.new_doc_id()  # 8-char ids are short; regenerate on the rare collision
    filename = file_utils.safe_filename(file.filename)
    dest = storage.upload_dir(doc_id) / filename
    file_utils.save_upload(file.file, str(dest))

    storage.register_upload(doc_id, filename)
    processing_service.submit(doc_id, str(dest))

    return PostDocResponse(
        doc_id=doc_id,
        filename=filename,
        status=DocumentStatus.uploaded,
        message="PDF uploaded successfully. Processing started.",
    )


@app.get("/process/{doc_id}", response_model=ProgressResponse, tags=["documents"])
def process(doc_id: str) -> ProgressResponse:
    """Return live processing progress for a document."""
    progress = document_service.get_progress(doc_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return progress


@app.delete("/delete_doc/{doc_id}", response_model=DeleteResponse, tags=["documents"])
def delete_doc(doc_id: str) -> DeleteResponse:
    """Delete a document and every artifact derived from it."""
    if not document_service.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return DeleteResponse(
        doc_id=doc_id,
        deleted=True,
        message="Document and all related data deleted successfully.",
    )


@app.post("/ask", response_model=AskResponse, tags=["chat"])
def ask(req: AskRequest) -> AskResponse:
    """Ask a question against a completed document's extracted data."""
    doc = document_service.get_document(req.doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if doc.status in (DocumentStatus.uploaded, DocumentStatus.processing):
        raise HTTPException(status_code=409, detail="Document is still processing.")
    if doc.status == DocumentStatus.failed:
        raise HTTPException(status_code=409, detail=f"Document processing failed: {doc.error or 'unknown error'}")
    if doc.status != DocumentStatus.completed:
        raise HTTPException(status_code=409, detail="Document is not ready.")

    try:
        result = ask_service.ask(req.doc_id, req.question, top_k=req.top_k)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="PageIndex for this document is missing.")
    except Exception as exc:  # noqa: BLE001 - surface a clean 500
        logger.exception("ask failed for %s", req.doc_id)
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}")

    return AskResponse(
        doc_id=req.doc_id,
        question=req.question,
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
    )
