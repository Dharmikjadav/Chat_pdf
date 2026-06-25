"""Strict JSON-only prompts for the local Ollama vision model.

Each prompt forces: single JSON object, preserved labels/numbers/units, preserved
row/column structure, null for anything unclear, and no explanation.
"""

__all__ = [
    "COMPLEX_TABLE_PROMPT",
    "CHART_PROMPT",
    "REGION_CLASSIFIER_PROMPT",
    "KPI_PROMPT",
    "PAGE_OCR_PROMPT",
    "VISION_PROMPTS",
]


PAGE_OCR_PROMPT = (
    "You are a precise OCR engine. Transcribe ALL text visible in this page image, "
    "exactly as written.\n"
    "- Preserve reading order, line breaks, headings, bullet points, and numbering.\n"
    "- Preserve every number, unit, currency symbol, and punctuation mark exactly.\n"
    "- Do NOT summarize, translate, paraphrase, or add any commentary, labels, or markdown fences.\n"
    "- If a region is illegible, write [illegible] in its place.\n"
    "Output only the transcribed plain text."
)


_OUTPUT_RULES = """
STRICT OUTPUT RULES (apply to every field):
- Respond with ONE valid JSON object only. Nothing else.
- Do NOT use markdown, code fences (```), comments, or any text before or after the JSON.
- Do NOT explain, describe, apologize, or add reasoning of any kind.
- Preserve every visible label, header, axis name, and category EXACTLY as written, including capitalization, punctuation, symbols, and footnote marks.
- Preserve every number EXACTLY as shown. Do not round, reformat, recompute, convert, or localize. Keep thousands separators, decimals, percent signs, currency symbols, leading zeros, and negatives (including parentheses for negatives) exactly.
- Preserve units and currency symbols. Keep them attached to the value or placed in the designated unit field; never silently drop them.
- Preserve the full row and column structure. Do not drop, merge, split, reorder, or invent rows or columns.
- If a value, label, or unit is missing, blank, illegible, or uncertain, use null. Never guess and never fabricate.
- Use null (not "", not "N/A", not 0, not "-") for anything you cannot read with confidence.
- Output the JSON object now.
""".rstrip() + "\n"


COMPLEX_TABLE_PROMPT = (
    """You are a precise financial table extraction engine.

The image contains one table or one table-like financial statement.

Return ONE valid JSON object only.

Schema:
{
  "type": "table",
  "title": null,
  "units": null,
  "columns": [],
  "rows": [
    {
      "section": null,
      "row_header": null,
      "row_level": 0,
      "cells": []
    }
  ],
  "footnotes": [],
  "confidence": 0.0
}

Rules:
- Preserve every visible row and column.
- Preserve merged headers and multi-level headers.
- Flatten stacked headers as "Parent - Child".
- Every row's cells array must match columns length.
- Keep ALL cell values as strings or null.
- Do not convert numbers to JSON numbers.
- Preserve commas, decimals, ₹, %, brackets, minus signs, and footnote marks.
- Use null only for blank or unreadable cells.
- For financial statements, detect sections like Assets, Liabilities, Income, Expenses.
- row_level shows hierarchy: 0 for section, 1 for main row, 2 for sub-row.
- Do not calculate, summarize, or infer missing values.
- No markdown. No explanation.
"""
    + _OUTPUT_RULES
)


CHART_PROMPT = (
    """You are a precise chart-data extraction engine. The image contains a single chart (bar, column, line, area, pie, scatter, or similar).

Extract the chart into a JSON object with EXACTLY this schema:
{
  "type": "chart",
  "chart_type": string | null,
  "title": string | null,
  "x_axis": { "label": string | null, "unit": string | null },
  "y_axis": { "label": string | null, "unit": string | null },
  "series": [
    {
      "name": string | null,
      "points": [
        { "label": string | null, "value": number | string | null, "unit": string | null }
      ]
    }
  ],
  "legend": [string, ...],
  "notes": null
}

CHART-SPECIFIC RULES:
- "chart_type" is one of "bar", "column", "line", "area", "pie", "scatter", or null if unclear.
- Read each value from its printed data label first. If a value is only implied by axis position, report it ONLY when gridlines make it unambiguous; otherwise use null.
- Preserve every category label, series name, and legend entry exactly.
- Keep axis units and value units (%, $, kg, etc.) in their unit fields and do not strip them from labeled values; put a value that still carries a symbol as a JSON string.
- One object in "series" per data series; for a single-series chart use one series with "name" null if unnamed.
- Do NOT compute totals, averages, percentages, or any derived values.
"""
    + _OUTPUT_RULES
)


REGION_CLASSIFIER_PROMPT = (
    """You are a precise document-region classifier. Look at the image region and classify what it primarily contains.

Return a JSON object with EXACTLY this schema:
{
  "type": "table" | "chart" | "figure" | "text" | "mixed",
  "subtype": string | null,
  "is_complex_table": boolean | null,
  "has_numbers": boolean | null,
  "title": string | null,
  "confidence": number | null
}

CLASSIFICATION-SPECIFIC RULES:
- Choose the single dominant content type for "type".
- Use "chart" for any data visualization, "table" for gridded tabular data, and "figure" for photos/diagrams/illustrations without tabular data.
- "subtype" gives a finer label when clear (e.g. "bar_chart", "line_chart", "financial_table", "diagram"), else null.
- "is_complex_table" is true ONLY when the region is a table with merged, multi-level, or nested structure; false for a simple grid; null when not a table.
- "confidence" is a number from 0.0 to 1.0.
- If the region is unreadable or ambiguous, set "type" to null and "confidence" to a low value.
"""
    + _OUTPUT_RULES
)


KPI_PROMPT = (
    """You are a precise KPI-card extraction engine. The image contains one or more KPI / highlight cards, each a large headline number with a short metric label.

Return ONE valid JSON object only with EXACTLY this schema:
{
  "kpis": [
    {
      "metric": string | null,
      "value": string | null,
      "unit": string | null,
      "period": string | null,
      "context": string | null,
      "confidence": number | null
    }
  ]
}

KPI-SPECIFIC RULES:
- One object in "kpis" per distinct card / headline number.
- "metric" is the short label (e.g. "Revenue", "EBITDA margin", "Total income").
- "value" is the exact visible number as a string, keeping symbols, commas, decimals, %, and ± / arrows for growth.
- "unit" is the unit or currency if shown (₹ Mn, %, x, days), else null.
- "period" is the year/period if shown (FY25, 2024-25, Q3), else null.
- "context" is any short qualifier printed on the card (e.g. "YoY growth", "consolidated"), else null.
- Do NOT compute, infer, or convert values. Use null for anything not clearly visible.
"""
    + _OUTPUT_RULES
)


VISION_PROMPTS = {
    "table": COMPLEX_TABLE_PROMPT,
    "complex_table": COMPLEX_TABLE_PROMPT,
    "chart": CHART_PROMPT,
    "classify": REGION_CLASSIFIER_PROMPT,
    "region": REGION_CLASSIFIER_PROMPT,
    "kpi": KPI_PROMPT,
}
