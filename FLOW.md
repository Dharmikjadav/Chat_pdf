# Project Flow — Offline PDF Chatbot

End-to-end map of how a PDF becomes a queryable, cited knowledge store, and how
the chatbot answers from it. Everything runs locally (PyMuPDF + OpenCV + Marker
+ Ollama + LangGraph + a vector-free PageIndex). Two phases: **Ingest** and **Chat**.

```
                          ┌──────────────────────────┐
   report.pdf  ───────▶   │   INGEST  (ingest.py)    │  ───▶  pageindex.json
                          └──────────────────────────┘        extracted.json
                                                                    │
                          ┌──────────────────────────┐             ▼
   "What was revenue?" ▶  │   CHAT     (main.py)      │  ◀──── loads the store
                          └──────────────────────────┘
                                     │
                                     ▼
                  "Revenue ... 438,857 (source: report, page: 3)"
```

---

## Phase 1 — INGEST (LangGraph pipeline)

Entry: `python ingest.py <pdf>` → `ingest.main()` → `graph.pipeline.run_pipeline()`
→ `build_pipeline()` compiles a `StateGraph(PDFPipelineState)` and runs it.

### Node graph (with conditional routing)

```
START
  │
  ▼
pymupdf_process            render pages → PNGs + page metadata
  │
  ▼
detect_regions             OpenCV: find table/chart/visual regions per page
  │                        (mostly-text pages are skipped)
  ▼
crop_regions               crop regions → extracted_regions/page_NNN/*.png
  │                        → route crops into candidate channels
  ▼
marker_process             marker_single: PDF → Markdown (+ figures, json)
  │
  ▼
parse_marker_markdown      Markdown → text/table nodes; MERGE OpenCV + Marker
  │                        candidates (no overwrite)
  ▼
[route_after_parse] ───────────────────────────────────────────┐
  │ complex_table_candidates?  → process_complex_tables         │
  │ else chart_candidates?     → process_charts                 │
  │ else visual_candidates?    → process_visuals                │
  │ else                       → validate_all_json              │
  ▼                                                             │
process_complex_tables     Ollama vision → table JSON per crop  │
  │                                                             │
  ▼ [route_after_tables]                                        │
process_charts             Ollama vision → chart JSON per crop  │
  │                                                             │
  ▼ [route_after_charts]                                        │
process_visuals            Ollama classify_region per crop      │
  │                                                             │
  ▼  ◀───────────────────────────────────────────────────────-─┘
validate_all_json          repair + normalize + validate every node
  │                        (type, page_number, source, confidence, data)
  ▼
build_pageindex            build document→page→section→node tree, save JSON
  │
  ▼
END                        final_output summary
```

Routing rules (from the task spec):
- **Mostly-text page** (`char_count ≥ TEXT_PAGE_CHAR_THRESHOLD`) → skip OpenCV; Marker handles its text.
- **table** crop → table extraction · **chart** crop → chart extraction · **visual** crop → vision classifier.
- A vision stage runs **only if** its candidate list is non-empty (otherwise skipped).

### Stage → module → state channels written

| # | Node (`graph/pipeline.py`) | Module it calls | Writes to state |
|---|---|---|---|
| 1 | `pymupdf_process` | `nodes/pymupdf_node.py` | `page_metadata`, `page_images` |
| 1b | `detect_regions` | `nodes/opencv_region_detector.py` | `page_regions` |
| 1c | `crop_regions` | `utils/region_cropper.py` | `region_crops`, `complex_table_candidates`, `chart_candidates`, `visual_candidates` |
| 2 | `marker_process` | `nodes/marker_node.py` | `marker_md_path`, `marker_json_path`, `marker_images` |
| 3 | `parse_marker_markdown` | `utils/markdown_parser.py` | `text_nodes`, `table_nodes`, **+merges** `complex_table_candidates`, `chart_candidates` |
| 4a | `process_complex_tables` | `nodes/ollama_vision_node.py` (`extract_complex_table`) | `table_nodes` (appended) |
| 4b | `process_charts` | `nodes/ollama_vision_node.py` (`extract_chart`) | `chart_nodes` |
| 4c | `process_visuals` | `nodes/ollama_vision_node.py` (`classify_region`) | `visual_nodes` |
| 5 | `validate_all_json` | `utils/json_validator.py` (`validate_node`) | `validated_nodes` |
| 6 | `build_pageindex` | `pageindex_store/pageindex.py` (`PageIndex`) | `pageindex_path`, `final_output` |
| * | every node | — | `errors` (append-only reducer) |

State schema + factory: `graph/state.py` (`PDFPipelineState`, `init_state`).
`errors` and `validated_nodes` use an append reducer so parallel/fan-out writes
accumulate; all other channels are last-write-wins.

### What flows through (one region's journey)

```
page image (PNG)
  └─ detect_regions  →  {type:"table", bbox:[x1,y1,x2,y2], confidence:0.91}
       └─ crop_regions → extracted_regions/page_003/table_1.png  (+candidate)
            └─ process_complex_tables → Ollama vision → table JSON
                 └─ validate_all_json → {type, page_number:3, source, confidence, data}
                      └─ build_pageindex → table node under page-3 node in the tree
```

Page numbers + source are preserved at **every** hop, so answers can cite them.

### Ingest outputs (under `<output_dir>`, default `data/processed/<name>/`)

```
page_images/page_0001.png ...        rendered pages (PyMuPDF)
extracted_regions/page_001/*.png     OpenCV region crops
marker_output/<name>/<name>.md       Marker markdown (+ figures, *_meta.json)
debug/page_001_detected.png          OpenCV overlays (OPENCV_DEBUG=true)
pageindex.json                       the PageIndex store  ◀── chat reads this
extracted.json                       full dump: nodes, metadata, summary, errors
```

---

## Phase 2 — CHAT (vector-free retrieval + local LLM)

Entry: `python main.py <pageindex.json>` → `main._chat_loop()` →
`chatbot.chat_engine.ChatEngine`.

```
user question
  │
  ▼
ChatEngine.retrieve()         PageIndex.search_keyword()  ← keyword scoring,
  │                           no embeddings / vector DB
  ▼
ChatEngine._format_context()  top-K text/table/chart nodes →
  │                           "[i] source: <doc>, page: <n>\n<content>"
  ▼
ChatEngine._call_llm()        Ollama chat (qwen3:8b), system prompt =
  │                           "answer ONLY from context + cite page/source"
  ▼
answer + structured sources   <think> stripped; main.py prints
                              "Source: page 3, table: <title>."
```

Commands in `main.py`: `help`, `pages`, `sources`, `exit`.
Retrieval/storage: `pageindex_store/pageindex.py`
(`search_keyword`, `search_by_page`, `search_by_node_type`, `get_context_for_question`).

---

## Cross-cutting

| Concern | Where |
|---|---|
| Config (paths, models, DPI, GPU flags) | `config.py` (loads `.env`) |
| Logging → `logs/pipeline.log` | `utils/logger.py` |
| Prompts (table / chart / classifier, JSON-only) | `prompts/vision_prompts.py` |
| GPU | Marker `TORCH_DEVICE=cuda`, Ollama `OLLAMA_NUM_GPU`, OpenCV `cv2.cuda` (CPU fallback). PyMuPDF render is CPU-only. |

---

## File map (the flow in code)

```
ingest.py                         CLI: PDF → pipeline → save + summary
main.py                           CLI: chat REPL over a PageIndex store
config.py                         env/.env config
graph/
  state.py                        PDFPipelineState + record types + init_state
  pipeline.py                     LangGraph nodes, routing, build_pipeline/run_pipeline
nodes/
  pymupdf_node.py                 render pages, page metadata, crop_region()
  opencv_region_detector.py       OpenCV region detection (table/chart/visual)
  marker_node.py                  marker_single subprocess wrapper
  ollama_vision_node.py           extract_complex_table / extract_chart / classify_region
utils/
  region_cropper.py               crop_regions()
  markdown_parser.py              Markdown → page-tagged nodes
  json_validator.py               repair_json / normalize_* / validate_node
  logger.py                       file + console logging
prompts/vision_prompts.py         strict JSON-only vision prompts
pageindex_store/pageindex.py      vector-free document tree + retrieval
chatbot/chat_engine.py            retrieve → context → Ollama → cited answer
```

---

## Run

```powershell
ollama serve ; ollama pull qwen3:8b ; ollama pull qwen2.5vl:7b
Copy-Item .env.example .env
python ingest.py data\raw_pdfs\report.pdf     # PDF → pageindex.json
python main.py  data\processed\report\pageindex.json   # chat with citations
```
