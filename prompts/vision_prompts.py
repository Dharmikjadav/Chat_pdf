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
    """You are a precise chart-data extraction engine. The image contains EXACTLY ONE chart. It may be simple (a single bar, column, line, area, pie, donut, or scatter) OR complex (stacked bar, 100%-stacked bar, grouped/clustered bar, multi-series line, combo bar+line on two axes, donut/pie, or waterfall), and it may have an adjacent legend or a separate data column printed beside it (for example a "CAGR %" column). Your job is to transcribe EVERY category, EVERY series, EVERY segment, and EVERY value with nothing dropped. Losing a stacked segment, a legend label, a forecast bar, a per-bar total, or a side data column is a FAILURE.

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
      "axis": "left" | "right" | null,
      "points": [
        { "label": string | null, "value": number | string | null, "unit": string | null }
      ]
    }
  ],
  "totals": [
    { "label": string | null, "value": number | string | null, "unit": string | null }
  ],
  "side_table": [
    { "name": string | null, "metric": string | null, "value": number | string | null, "unit": string | null }
  ],
  "legend": [string, ...],
  "n_categories": number | null,
  "n_series": number | null,
  "notes": string | null
}

UNIVERSAL MODEL -- every chart is a set of named series over a shared list of categories:
- A "category" is one position on the category axis: a year ("2024"), a period ("FY25"), a quarter, a region, a therapy area, or a pie/donut slice name. Identify the FULL ordered list of categories first and COUNT them; put that count in "n_categories".
- A "series" is one logical data set that spans the categories: one bar/segment color, one legend entry, one plotted line, one bar within a cluster, or the single set of a single-series chart.
- A "point" is the value of one series at one category. "label" is the category; "value" is that point's number; "unit" is the unit printed on that value if any.
- Use the SAME ordered set of category labels (same order, same count) across EVERY series. If a series has no bar/point at a category, still emit that point with "value": null. NEVER drop or reorder categories, and never let one series have fewer points than another.
- "name" is the series identity taken EXACTLY from the legend / segment color key / line label / cluster-bar label (e.g. "Filed", "Approved", "Asia-Pacific", "Revenue", "Forecast"). For a genuinely single unnamed series, use one series object with "name": null.
- Set "n_series" to the number of objects in "series"; it MUST equal the length of "series".

chart_type -- set the MOST SPECIFIC value that matches what you see:
- "bar" / "column" / "line" / "area" / "scatter" for simple single-series plots of that shape.
- "pie" for a full disc; "donut" for a ring with a hollow center.
- "multi_line" for two or more plotted lines.
- "stacked_bar" / "stacked_column" when segments stack within one bar; use "100_stacked_bar" / "100_stacked_column" when every stack always fills to 100%.
- "grouped_bar" / "grouped_column" when several bars sit side by side per category.
- "combo" when bars and a line share the plot (typically two y-axes).
- "waterfall" when bars float between running totals (increases/decreases bridging to a total).
- Use null ONLY if the shape is genuinely unreadable.

READING VALUES (apply to every series):
- Read every value from its PRINTED data label. If a value has no printed label and is only implied by axis position or bar height, use null for that value -- do NOT estimate from gridlines, height, or angle.
- Preserve every category label, series name, legend entry, axis title, unit, footnote mark, %, currency symbol, superscript, and parenthesis EXACTLY as written. A value that still carries a symbol (e.g. "33%", "(12)", "$1.2") must be a JSON string.
- Set "axis": "right" for any series read against a secondary/right-hand y-axis (common in combo charts); "left" otherwise; null if there is only one axis.

EXHAUSTIVE ENUMERATION (mandatory -- the most common failure is stopping early):
- Transcribe EVERY category and EVERY bar/column/segment/slice/point. Do not sample, do not summarize, do not stop after the first few. If there are 11 columns, output 11; if a stack has 10 segments, output all 10; if a pie has 8 slices, output 8.
- Include forecast / projected / estimated / shaded / dashed / "F" / "E" regions EXACTLY like the rest. Never drop a forecast bar and never treat it as a separate chart. If a bracket or band labels a range as forecast/estimate (e.g. "Forecast 2024-2028"), still emit every category inside it and record the bracket text in "notes".

STACKED bar/column (most error-prone -- follow EXACTLY):
- Emit ONE series PER stacked segment color. The series "name" is that segment's LEGEND label (the color key, e.g. "Asia-Pacific", "India", "Global"), NEVER the category/year.
- Read each bar from BOTTOM to TOP and map every colored segment to its legend label by matching color; the bottom segment's color = the legend entry drawn for that color. Keep this segment->legend mapping consistent for every category.
- Each segment series gets one point per category (the year/column). Use null where a category has no segment of that color.
- If a TOTAL is printed above each whole bar, put those totals in "totals" (one entry per category, "label" = the category, "value" = the printed total). Do NOT sum the segments yourself.
- For 100%-stacked charts, transcribe each segment's printed % as its value; do not normalize or recompute.

GROUPED / CLUSTERED bar/column:
- Emit ONE series per bar position within a cluster (each legend entry), each with one point per category. Example: two clustered bars "Filed" and "Approved" over FY21..FY25 -> two series, five points each.

COMBO bar+line:
- Emit the bar set(s) as series with "axis": "left" and the line as its own series with "axis": "right" (or whichever axis it is plotted against). Keep both over the same categories.

PIE / DONUT:
- Emit ONE series (name = chart title or null) whose points are the slices: "label" = slice name, "value" = the printed slice value or percentage (keep the "%" if shown). If both an absolute value and a percentage are printed for a slice, put the absolute number in "value" and append the percentage to the slice "label" (e.g. "India 33%"), so neither is lost. Mirror the slice names in "legend".

WATERFALL:
- Emit one series whose points follow the bars left to right (start, each increase/decrease, end). Use the printed step value for each; record any "Total"/"Subtotal" bars as their own points with the printed label.

ADJACENT LEGEND / SIDE DATA COLUMN (e.g. a "CAGR % 2024-2028" column printed to the right of the bars):
- This is REAL DATA, not decoration. Capture every row of that side column in "side_table": one entry per row, "name" = the row's category/region label, "metric" = the column header (e.g. "CAGR % 2024-2028"), "value" = the printed number, "unit" = its unit (e.g. "%") if shown.
- Still list those same labels in "legend" if they double as the chart legend.

DERIVED VALUES:
- Do NOT compute totals, averages, percentages, CAGRs, sums, or any derived value. Only transcribe numbers that are PRINTED on the chart. The only "totals" you may emit are per-bar totals that are PRINTED above the bars.

SELF-CHECK before you output (perform mentally, do not add extra keys):
- Does every series have the same number of points as there are categories? If not, add the missing null points.
- Did you map every stacked segment color to a legend label, bottom to top?
- Did you capture every forecast/projected category?
- Did you capture per-bar printed totals in "totals" and every side-column row in "side_table"?
- Does "n_series" equal len(series) and "n_categories" equal the number of categories?
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
- Use "chart" for ANYTHING that encodes numbers as VISUAL GEOMETRY -- bars, columns, stacked/grouped bars, lines, areas, pie/donut rings, scatter, waterfall, or KPI mini-bars -- EVEN IF the chart also prints its data labels and a small side table or legend column (such as a CAGR % column). The presence of printed numbers or an adjacent value column does NOT make it a "table".
- Use "table" ONLY when the dominant structure is a GRID of rows and columns with ruled/aligned cells and NO bars/slices/lines encoding the values. Use "figure" for photos/diagrams/illustrations without tabular data.
- When a region mixes a chart and a small data table (a chart with a side CAGR column, or a KPI card with a mini-bar), classify "type":"chart" and set "is_complex_table": false.
- "subtype" MUST name the chart family when type="chart": one of "bar","column","stacked_bar","stacked_column","100_stacked","grouped_bar","line","multi_line","area","combo","pie","donut","scatter","waterfall","kpi_card". For a table use a finer label when clear (e.g. "financial_table"), else null.
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
