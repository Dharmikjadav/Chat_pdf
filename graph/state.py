"""Shared state for the offline PDF-processing LangGraph pipeline.

The pipeline mirrors the project flow::

    PDF -> PyMuPDF -> Marker -> Markdown Parser -> Ollama Vision
        -> JSON Validator -> LangGraph -> PageIndex Storage -> Chatbot

`PDFPipelineState` is the single ``TypedDict`` threaded through every node.
Nodes return *partial* updates (a subset of keys); LangGraph merges them into
the running state. Most channels are last-write-wins, but a few are declared
with an append-only reducer (`reduce_append`) so that **parallel / fan-out**
nodes (e.g. one vision call per chart candidate via the ``Send`` API) can
accumulate results without clobbering each other.

Usage::

    from langgraph.graph import StateGraph
    from graph.state import PDFPipelineState, init_state

    graph = StateGraph(PDFPipelineState)
    ...
    app = graph.compile()
    result = app.invoke(init_state("data/raw_pdfs/report.pdf"))

The module deliberately has **no** runtime dependency on ``langgraph`` itself:
the reducer is a plain function read from the ``Annotated`` metadata when the
graph is built, so this state schema stays import-light and testable.
"""

from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional

from typing_extensions import NotRequired, TypedDict

__all__ = [
    "PDFPipelineState",
    "init_state",
    "reduce_append",
    # Record types
    "PageMetadata",
    "PageImage",
    "MarkerImage",
    "TextNode",
    "TableNode",
    "ChartNode",
    "VisionCandidate",
    "ValidatedNode",
    "PipelineError",
    "FinalOutput",
    # Literals
    "NodeType",
    "CandidateKind",
    "PipelineStage",
    "Status",
]


# --------------------------------------------------------------------------- #
# Literal vocabularies
# --------------------------------------------------------------------------- #
NodeType = Literal["text", "table", "chart", "figure", "heading", "kpi", "fact"]
CandidateKind = Literal["table", "chart", "kpi"]
PipelineStage = Literal[
    "ingestion",
    "marker",
    "parsing",
    "vision",
    "validation",
    "storage",
    "finalize",
]
Status = Literal["success", "partial", "failed"]


# --------------------------------------------------------------------------- #
# Channel reducer
# --------------------------------------------------------------------------- #
def reduce_append(existing: Optional[List[Any]], update: Optional[List[Any]]) -> List[Any]:
    """Append-only list reducer for fan-out / parallel channel writes.

    LangGraph calls this as ``reduce_append(current_value, node_update)`` to
    merge a node's returned list into the channel. It is null-safe so the
    channel can start empty and tolerate nodes that return ``None``.
    """
    if existing is None:
        existing = []
    if update is None:
        return list(existing)
    return [*existing, *update]


# --------------------------------------------------------------------------- #
# Record types (the payloads carried inside the state lists)
# --------------------------------------------------------------------------- #
class PageMetadata(TypedDict):
    """Per-page facts extracted by PyMuPDF. Page numbers are 1-based."""

    page_number: int
    width: float
    height: float
    rotation: int
    has_text_layer: bool
    char_count: int
    source: str  # document name / pdf path — preserves provenance
    image_count: NotRequired[int]
    word_count: NotRequired[int]
    block_count: NotRequired[int]
    numeric_token_ratio: NotRequired[float]
    image_area_ratio: NotRequired[float]
    needs_render: NotRequired[bool]


class PageImage(TypedDict):
    """A rendered full-page raster, used as a vision-model fallback."""

    page_number: int
    image_path: str
    dpi: int
    width: NotRequired[int]
    height: NotRequired[int]


class MarkerImage(TypedDict):
    """An image asset extracted by Marker (figure crop, embedded picture)."""

    name: str
    path: str
    page_number: NotRequired[Optional[int]]


class TextNode(TypedDict):
    """A prose / heading block parsed from Marker's Markdown."""

    node_id: str
    content: str
    page_number: int
    source: str
    heading_path: NotRequired[List[str]]  # breadcrumb of enclosing headings
    char_count: NotRequired[int]


class TableNode(TypedDict):
    """A table parsed from Markdown; may be flagged complex for vision."""

    node_id: str
    page_number: int
    source: str
    markdown: str  # the table rendered as Markdown
    caption: NotRequired[Optional[str]]
    n_rows: NotRequired[int]
    n_cols: NotRequired[int]
    is_complex: NotRequired[bool]
    image_path: NotRequired[Optional[str]]  # crop sent to the vision model


class ChartNode(TypedDict):
    """A chart / figure; `description` and `data` are filled by the vision model."""

    node_id: str
    page_number: int
    source: str
    image_path: str
    caption: NotRequired[Optional[str]]
    description: NotRequired[Optional[str]]
    data: NotRequired[Optional[Dict[str, Any]]]  # extracted series / values


class VisionCandidate(TypedDict):
    """An item queued for the Ollama vision model (complex table or chart)."""

    node_id: str
    kind: CandidateKind
    page_number: int
    source: str
    image_path: str
    context: NotRequired[Optional[str]]  # surrounding text for grounding
    reason: NotRequired[Optional[str]]  # why it was flagged for vision


class ValidatedNode(TypedDict):
    """A node that passed JSON-schema validation, ready for PageIndex storage."""

    node_id: str
    type: NodeType
    page_number: int
    source: str
    content: str
    valid: bool
    title: NotRequired[Optional[str]]
    structured: NotRequired[Optional[Dict[str, Any]]]  # validated vision payload


class PipelineError(TypedDict):
    """A non-fatal error captured by a node and accumulated on the state."""

    stage: PipelineStage
    message: str
    node_id: NotRequired[Optional[str]]
    page_number: NotRequired[Optional[int]]
    exception_type: NotRequired[Optional[str]]


class FinalOutput(TypedDict):
    """Summary written by the finalize node once the document is indexed."""

    document_name: str
    status: Status
    num_pages: int
    num_nodes: int
    num_tables: int
    num_charts: int
    num_errors: int
    pageindex_path: NotRequired[Optional[str]]


RegionType = Literal["table", "chart", "visual", "kpi_card", "dense_text"]


class DetectedRegion(TypedDict):
    """A region found by the OpenCV detector on a rendered page image."""

    type: RegionType
    bbox: List[int]  # [x1, y1, x2, y2] in page-image pixel coordinates
    confidence: float


class PageRegions(TypedDict):
    """All regions detected on a single page (Stage 1b, pre-Marker)."""

    page: int
    regions: List[DetectedRegion]
    skipped: NotRequired[Optional[str]]  # reason a page was skipped (e.g. "text_page")
    annotated_path: NotRequired[Optional[str]]  # debug overlay image path


class RegionCrop(TypedDict):
    """A saved crop of a detected region, routed to an extraction stage."""

    id: str
    type: RegionType
    page_number: int
    bbox: List[int]
    confidence: float
    image_path: str
    document_name: NotRequired[Optional[str]]
    source: NotRequired[Optional[str]]


# --------------------------------------------------------------------------- #
# Top-level pipeline state
# --------------------------------------------------------------------------- #
class PDFPipelineState(TypedDict):
    """End-to-end state for processing one PDF into a PageIndex tree.

    Channels marked ``Annotated[..., reduce_append]`` accumulate across
    parallel writes; all others are last-write-wins. Create instances with
    :func:`init_state` so every channel is initialised (lists to ``[]``,
    scalars to ``None``) before the graph runs.
    """

    # --- Input / identity -------------------------------------------------- #
    pdf_path: str
    document_name: str
    output_dir: str

    # --- Stage 1: PyMuPDF ingestion --------------------------------------- #
    page_metadata: List[PageMetadata]
    page_images: List[PageImage]
    page_texts: List[Dict[str, Any]]    # [{page_number, text, drawing_count}]

    # --- Stage 1a: Page classification ------------------------------------ #
    page_classes: List[Dict[str, Any]]  # [{page_number, page_class, signals}]

    # --- Stage 1b: OpenCV region detection (pre-Marker) ------------------- #
    page_regions: List[PageRegions]
    region_crops: List[RegionCrop]

    # --- Stage 2: Marker conversion --------------------------------------- #
    marker_md_path: Optional[str]
    marker_json_path: Optional[str]
    marker_images: List[MarkerImage]

    # --- Stage 3: Markdown parsing ---------------------------------------- #
    text_nodes: List[TextNode]
    table_nodes: List[TableNode]
    chart_nodes: List[ChartNode]
    visual_nodes: List[Dict[str, Any]]

    # --- Stage 4: Vision routing (items needing the vision model) --------- #
    complex_table_candidates: List[VisionCandidate]
    chart_candidates: List[VisionCandidate]
    visual_candidates: List[VisionCandidate]
    kpi_candidates: List[Dict[str, Any]]

    # --- Stage 4b: Extracted structured nodes ----------------------------- #
    kpi_nodes: List[Dict[str, Any]]
    fact_nodes: List[Dict[str, Any]]

    # --- Stage 5: Validation (accumulates across fan-out vision calls) ----- #
    validated_nodes: Annotated[List[ValidatedNode], reduce_append]

    # --- Stage 6: PageIndex storage --------------------------------------- #
    pageindex_path: Optional[str]
    cache_dir: Optional[str]

    # --- Cross-cutting: errors + per-stage timings ------------------------ #
    errors: Annotated[List[PipelineError], reduce_append]
    stage_timings: Annotated[List[Dict[str, Any]], reduce_append]

    # --- Stage 7: Finalize ------------------------------------------------ #
    final_output: Optional[FinalOutput]


# --------------------------------------------------------------------------- #
# State factory
# --------------------------------------------------------------------------- #
def init_state(
    pdf_path: str,
    *,
    output_dir: Optional[str] = None,
    document_name: Optional[str] = None,
) -> PDFPipelineState:
    """Build a fully-initialised :class:`PDFPipelineState` for one PDF.

    Args:
        pdf_path: Path to the source PDF.
        output_dir: Where this document's artifacts are written. Defaults to
            ``data/processed/<document_name>``.
        document_name: Logical document name. Defaults to the PDF file stem.

    Returns:
        A state dict with every channel present so nodes may index it directly.
    """
    path = Path(pdf_path)
    name = document_name or path.stem
    out_dir = output_dir or str(Path("data") / "processed" / name)

    return PDFPipelineState(
        pdf_path=str(path),
        document_name=name,
        output_dir=out_dir,
        page_metadata=[],
        page_images=[],
        page_texts=[],
        page_classes=[],
        page_regions=[],
        region_crops=[],
        marker_md_path=None,
        marker_json_path=None,
        marker_images=[],
        text_nodes=[],
        table_nodes=[],
        chart_nodes=[],
        visual_nodes=[],
        complex_table_candidates=[],
        chart_candidates=[],
        visual_candidates=[],
        kpi_candidates=[],
        kpi_nodes=[],
        fact_nodes=[],
        validated_nodes=[],
        pageindex_path=None,
        cache_dir=str(Path(out_dir) / "cache"),
        errors=[],
        stage_timings=[],
        final_output=None,
    )
