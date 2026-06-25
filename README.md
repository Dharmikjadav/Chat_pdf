# Offline PDF Chatbot

A **fully offline** chatbot that ingests complex PDFs — scanned documents,
financial reports, tables, and charts — and answers questions about them
using **PageIndex-style, vector-free retrieval**. No cloud APIs. Everything
runs locally on Windows (CPU or NVIDIA GPU).

> Status: **scaffolding**. This repo currently contains the folder structure,
> dependencies, and setup instructions only. Pipeline modules are added next.

---

## Pipeline

```
PDF
 └─ PyMuPDF            extract text, metadata, render page images
     └─ OpenCV Region Detector   find table/chart/visual regions (bbox + confidence)
         └─ Region Cropper        crop regions -> extracted_regions/page_NNN/*.png
             └─ Marker            PDF -> structured Markdown (tables, figures, equations)
                 └─ Markdown Parser     Markdown -> page-tagged blocks
                     └─ Ollama Vision Model   interpret charts / scans / complex tables
                         └─ JSON Validator     Pydantic-validated structured records
                             └─ LangGraph       orchestrate the stateful pipeline
                                 └─ PageIndex Storage   vector-free document tree (JSON)
                                     └─ Offline Chatbot   reasoning retrieval + cited answers
```

The **OpenCV stage** (`nodes/opencv_region_detector.py` + `utils/region_cropper.py`)
runs per rendered page: mostly-text pages skip it (Marker handles their text);
detected **table** crops route to table extraction, **chart** crops to chart
extraction, and ambiguous **visual** crops to the vision classifier. Set
`OPENCV_DEBUG=true` to save annotated overlays to `<output_dir>/debug/`.

Page numbers and source filenames are preserved end-to-end so every answer can
cite **which page of which document** it came from.

### Why these choices

| Concern              | Choice                          | Reason                                              |
| -------------------- | ------------------------------- | --------------------------------------------------- |
| OCR / layout         | **Marker** (not PaddleOCR)      | Better tables/equations, clean Markdown, offline    |
| LLM + vision         | **Ollama** (local)              | No cloud; runs `llama3.1` + `llama3.2-vision`        |
| Orchestration        | **LangGraph**                   | Stateful, resumable, inspectable pipeline           |
| Retrieval            | **PageIndex** (vector-free)     | Reasoning tree-search, no embeddings / vector DB     |

---

## Prerequisites

- **Windows 10/11**
- **Python 3.10 – 3.12**
- **[Ollama](https://ollama.com/download)** installed and running locally
- ~10 GB free disk for local models (more for larger LLMs)
- *(Optional)* NVIDIA GPU + CUDA for faster Marker OCR and vision inference

---

## Setup

### 1. Clone / open the project

```powershell
cd c:\LLM\DJ\Chat_pdf
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

CPU only:

```powershell
pip install -r requirements.txt
```

NVIDIA GPU (install CUDA PyTorch first, then the rest):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Pull the local models with Ollama

```powershell
ollama serve            # leave running in its own terminal
ollama pull qwen3:8b       # chat / answering   (OLLAMA_CHAT_MODEL)
ollama pull qwen2.5vl:7b   # vision: charts, scans, complex tables (OLLAMA_VISION_MODEL)
```

> Smaller hardware? Swap in lighter models (e.g. `qwen3:4b`,
> `minicpm-v`) and update `OLLAMA_CHAT_MODEL` / `OLLAMA_VISION_MODEL` in `.env`.

### 5. Configure environment

```powershell
Copy-Item .env.example .env
```

Edit `.env` to match your models, device (`TORCH_DEVICE=cpu|cuda`), and paths.

### 6. Verify the setup (quick checks)

```powershell
python -c "import fitz, marker, ollama, langgraph; print('deps OK')"
ollama list             # confirm the two models are present
```

---

## Going fully offline

1. Do steps 3–4 **once while online** so pip packages and Ollama models are
   cached locally.
2. Marker downloads its OCR/layout weights on first run — trigger one
   conversion while online, or pre-cache into `./models` (see `scripts/`).
3. After that, set `OFFLINE_MODE=true` in `.env`. The only network traffic is
   to your **local** Ollama server at `OLLAMA_BASE_URL`.

---

## Running on GPU

The two ML stages run on an NVIDIA GPU; PDF page rendering (PyMuPDF) is
CPU-only by design and cannot be GPU-accelerated.

| Stage | GPU? | How |
| --- | --- | --- |
| **Marker** (OCR/layout/tables) | ✅ | `TORCH_DEVICE=cuda` (needs CUDA-build PyTorch; auto-falls back to CPU if unavailable) |
| **Ollama** (vision + chat) | ✅ | `OLLAMA_NUM_GPU=999` offloads all layers; Ollama auto-detects CUDA |
| **OpenCV** (region detection) | ⚠️ | `cv2.cuda` if OpenCV is built with CUDA (not on PyPI); else CPU. Cheap either way. |
| **PyMuPDF** (page render) | ❌ | CPU rasterization — no GPU path exists |

Setup — **install the CUDA build of PyTorch matching your GPU** *before* the rest.
The pip `torch` from PyPI is CPU-only. Pick the index URL for your GPU:

```powershell
# Blackwell (RTX 50-series, e.g. 5070 Ti) -> CUDA 12.8 wheels
pip install "torch>=2.7,<3" --index-url https://download.pytorch.org/whl/cu128

# Ada/Ampere (RTX 30/40-series) -> CUDA 12.1 wheels
#   pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# .env.example is already GPU-configured: TORCH_DEVICE=cuda, OLLAMA_NUM_GPU=999

# Verify the GPU is visible
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> If `torch.cuda.is_available()` is `False`, Marker logs a warning and runs on
> CPU instead of crashing. OpenCV GPU needs a CUDA-built OpenCV (build from
> source); it falls back to CPU automatically.

> No GPU? Set `TORCH_DEVICE=cpu` and `OLLAMA_NUM_GPU=0` in `.env`.

---

## Project structure

```
Chat_pdf/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── raw_pdfs/        # input PDFs
│   ├── processed/       # Marker Markdown + extracted assets
│   ├── images/          # rendered pages / figures for the vision model
│   └── pageindex/       # PageIndex document trees (JSON)
├── models/              # local model cache (offline)
├── prompts/             # prompt templates
├── logs/                # runtime logs
├── scripts/             # ops scripts (pre-download models, batch ingest)
├── tests/
└── src/
    ├── config/          # typed settings from .env
    ├── ingestion/       # PyMuPDF + Marker
    ├── parsing/         # Markdown -> page-tagged blocks
    ├── vision/          # Ollama vision interpretation
    ├── validation/      # Pydantic JSON validation
    ├── workflow/        # LangGraph graphs
    ├── storage/         # PageIndex storage (vector-free)
    ├── retrieval/       # PageIndex tree-search retrieval
    ├── chatbot/         # offline chat (CLI + optional Streamlit UI)
    └── utils/           # logging, paths, Ollama client
```

---

## Usage

Make sure `ollama serve` is running and the two models are pulled, then:

```powershell
# 1) Ingest a PDF: PyMuPDF -> Marker -> parse -> vision -> validate -> PageIndex
python ingest.py data\raw_pdfs\report.pdf
#    Writes <output_dir>/extracted.json and <output_dir>/pageindex.json
#    (default output_dir = data\processed\report). Optional:
#    python ingest.py data\raw_pdfs\report.pdf --output-dir data\processed\report

# 2) Chat against the ingested document (cited answers)
python main.py data\processed\report\pageindex.json
#    Or let it auto-discover a single store:
python main.py
```

Inside the chat: `help`, `pages`, `sources`, `exit`. Logs stream to
`logs\pipeline.log`.

---

## License

TBD.
