"""
Rule-based Query Router for offline Annual Report PDF chatbot.

Purpose
-------
Classifies user questions into one of:
- table
- chart
- kpi
- financial
- governance
- esg
- text

Then returns recommended PageIndex node types and expanded query terms.

Designed for Indian annual reports such as:
- Sun Pharma
- Divi's Laboratories
- Apollo Hospitals

No external dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------------

FINANCIAL_KEYWORDS = {
    # core financial metrics
    "revenue", "revenue from operations", "total revenue", "income",
    "total income", "other income", "net sales", "sales",
    "profit", "net profit", "profit before tax", "profit after tax",
    "pbt", "pat", "ebitda", "ebit", "margin", "gross margin",
    "operating profit", "operating margin", "net margin",
    "eps", "earnings per share", "basic eps", "diluted eps",
    "dividend", "dividend payout", "dividend payout ratio",
    "cash flow", "operating cash flow", "free cash flow",
    "capex", "capital expenditure", "assets", "liabilities",
    "equity", "net worth", "borrowings", "debt", "cash",
    "reserves", "retained earnings", "share capital",
    "balance sheet", "statement of profit and loss", "p&l",
    "statement of cash flows", "changes in equity",
    "standalone", "consolidated", "financial statements",
    "financial highlights", "ten year financial highlights",
    "key performance indicators", "kpi",
    "fy", "f.y.", "financial year", "march 31", "year ended",
    "crore", "million", "₹", "rs", "inr",
}

TABLE_KEYWORDS = {
    "table", "details", "data", "figures", "values", "amount",
    "particulars", "schedule", "note", "notes", "statement",
    "as at", "for the year ended", "year ended", "ratio",
    "number of", "breakup", "break-up", "segment", "category",
    "subsidiary", "associate", "joint venture", "related party",
    "shareholding", "remuneration", "employee", "employees",
    "board meeting", "attendance", "composition", "committee",
    "audit committee", "csr committee", "risk management committee",
    "principle", "brsr", "business responsibility",
    "waste", "energy", "water", "emissions", "ghg",
}

CHART_KEYWORDS = {
    "chart", "graph", "trend", "growth", "visual", "infographic",
    "bar chart", "line chart", "pie chart", "donut", "percentage share",
    "share of", "mix", "contribution", "movement", "timeline",
    "compare", "comparison", "increase", "decrease", "year on year",
    "yoy", "over years", "last five years", "last 5 years",
}

KPI_KEYWORDS = {
    "kpi", "key performance indicator", "key performance indicators",
    "performance highlights", "highlights", "at a glance",
    "total income", "profit before tax", "profit after tax",
    "earnings per share", "eps", "ebitda", "ebitda margin",
    "return on capital employed", "roce", "roe", "roa",
    "safe man-hours", "countries reach", "manufacturing sites",
    "employees", "employee base", "csr commitment", "csr expenditure",
    "energy conserved", "water conserved", "plastic waste reduced",
    "ghg emissions", "female employees", "women directors",
    "independent directors", "shareholder grievances",
}

ESG_KEYWORDS = {
    "esg", "sustainability", "environment", "environmental",
    "social", "governance", "brsr", "business responsibility",
    "energy", "energy consumed", "energy conserved", "renewable",
    "water", "water conserved", "water recycled", "waste",
    "plastic waste", "emissions", "ghg", "scope 1", "scope 2",
    "tco2", "csr", "csr expenditure", "csr commitment",
    "community", "safe man-hours", "safety", "ehs", "training hours",
    "female employees", "women directors", "independent directors",
    "shareholder grievances", "board attendance",
}

GOVERNANCE_KEYWORDS = {
    "board", "board of directors", "director", "directors",
    "chairman", "managing director", "ceo", "cfo",
    "company secretary", "compliance officer", "committee",
    "audit committee", "nomination", "remuneration",
    "corporate governance", "shareholder", "shareholders",
    "agm", "annual general meeting", "notice", "resolution",
    "voting", "e-voting", "cut-off date", "scrip code",
    "trading symbol", "isin", "bse", "nse", "sebi",
}

TEXT_KEYWORDS = {
    "explain", "describe", "summary", "summarize", "overview",
    "what is", "who is", "why", "how", "business model",
    "strategy", "portfolio", "products", "therapies",
    "geographic presence", "business differentiators",
    "operating environment", "journey", "message",
    "chairman message", "managing director message",
    "management discussion", "risk", "outlook",
}

# Query synonyms improve retrieval without depending on vector DB.
QUERY_SYNONYMS = {
    "eps": ["earnings per share", "basic earnings per share", "diluted earnings per share"],
    "pat": ["profit after tax", "net profit", "profit for the year"],
    "pbt": ["profit before tax"],
    "ebitda": ["earnings before interest tax depreciation amortization"],
    "roce": ["return on capital employed"],
    "roe": ["return on equity"],
    "capex": ["capital expenditure", "capital work in progress"],
    "revenue": ["revenue from operations", "total revenue", "income from operations"],
    "sales": ["revenue from operations", "net sales"],
    "total income": ["income", "revenue from operations", "other income"],
    "dividend": ["final dividend", "interim dividend", "dividend payout ratio"],
    "brsr": ["business responsibility and sustainability report"],
    "csr": ["corporate social responsibility", "csr expenditure", "csr commitment"],
    "cashflow": ["cash flow", "statement of cash flows"],
    "cash flow": ["statement of cash flows", "net cash generated from operating activities"],
}


ROUTE_NODE_TYPES = {
    # Keep text fallback because annual report financial values are often stored
    # in OCR text/highlight pages, not only in table nodes.
    "financial": ("table", "text", "chart"),
    "table": ("table", "text"),
    "chart": ("chart", "text"),
    "kpi": ("table", "chart", "text"),
    "esg": ("table", "chart", "text"),
    "governance": ("text", "table"),
    "text": ("text",),
}


@dataclass
class QueryRoute:
    route: str
    node_types: Tuple[str, ...]
    scores: Dict[str, int]
    expanded_query: str
    matched_keywords: Dict[str, List[str]]
    top_k_multiplier: int = 4

    def to_dict(self) -> Dict:
        return asdict(self)


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("₹", " ₹ ")
    text = re.sub(r"[^a-z0-9₹%&.\-/ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_keyword(q: str, keyword: str) -> bool:
    k = _normalize(keyword)
    if " " in k:
        return k in q
    return re.search(rf"\b{re.escape(k)}\b", q) is not None


def _score_keywords(q: str, keywords: set[str]) -> tuple[int, list[str]]:
    matched = []
    score = 0
    for kw in keywords:
        if _contains_keyword(q, kw):
            matched.append(kw)
            # multi-word phrases are more specific
            score += 3 if " " in kw else 1
    return score, sorted(matched)


def expand_query(question: str) -> str:
    q = _normalize(question)
    extra = []
    for key, synonyms in QUERY_SYNONYMS.items():
        if _contains_keyword(q, key):
            extra.extend(synonyms)
    if extra:
        return question + " " + " ".join(dict.fromkeys(extra))
    return question


def route_question(question: str, *, default_top_k: int = 4) -> QueryRoute:
    """
    Classify question and return node types for PageIndex retrieval.

    Priority rule:
    - chart route wins if explicit chart/trend words exist.
    - kpi route wins for highlights/at-a-glance/performance cards.
    - financial route wins for finance metrics.
    - governance/esg routes are used for those sections.
    - otherwise text route.
    """
    q = _normalize(question)

    financial_score, financial_hits = _score_keywords(q, FINANCIAL_KEYWORDS)
    table_score, table_hits = _score_keywords(q, TABLE_KEYWORDS)
    chart_score, chart_hits = _score_keywords(q, CHART_KEYWORDS)
    kpi_score, kpi_hits = _score_keywords(q, KPI_KEYWORDS)
    esg_score, esg_hits = _score_keywords(q, ESG_KEYWORDS)
    governance_score, governance_hits = _score_keywords(q, GOVERNANCE_KEYWORDS)
    text_score, text_hits = _score_keywords(q, TEXT_KEYWORDS)

    # Strong signal when question asks exact figures/values by year/period.
    if re.search(r"\b(fy\s?\d{2,4}|20\d{2}[-–]\d{2}|march\s+31|year ended)\b", q):
        financial_score += 2
        table_score += 1

    if re.search(r"\b(value|amount|figure|figures|how much|what was|what were)\b", q):
        table_score += 1

    scores = {
        "financial": financial_score,
        "table": table_score,
        "chart": chart_score,
        "kpi": kpi_score,
        "esg": esg_score,
        "governance": governance_score,
        "text": text_score,
    }

    matched = {
        "financial": financial_hits,
        "table": table_hits,
        "chart": chart_hits,
        "kpi": kpi_hits,
        "esg": esg_hits,
        "governance": governance_hits,
        "text": text_hits,
    }

    # Explicit chart/trend request should not be swallowed by finance/table.
    if chart_score >= 2:
        route = "chart"
    elif kpi_score >= 3:
        route = "kpi"
    elif financial_score >= 2:
        route = "financial"
    elif esg_score >= 3:
        route = "esg"
    elif governance_score >= 3:
        route = "governance"
    elif table_score >= 2:
        route = "table"
    elif text_score > 0:
        route = "text"
    else:
        route = max(scores, key=scores.get)
        if scores[route] == 0:
            route = "text"

    # For finance/table questions, use wider retrieval window because answer-bearing
    # nodes can rank 4-12 in long annual reports.
    top_k_multiplier = 2 if route in {"financial", "table", "kpi", "chart", "esg"} else 1

    return QueryRoute(
        route=route,
        node_types=ROUTE_NODE_TYPES[route],
        scores=scores,
        expanded_query=expand_query(question),
        matched_keywords=matched,
        top_k_multiplier=top_k_multiplier,
    )


def get_node_types_for_question(question: str) -> Tuple[str, ...]:
    return route_question(question).node_types


if __name__ == "__main__":
    tests = [
        "What was revenue from operations in FY25?",
        "Show EPS for 2024-25",
        "What does the chart say about revenue growth?",
        "Give key performance indicators",
        "Who is the company secretary?",
        "Summarize the business model",
        "What was energy conserved?",
    ]
    for t in tests:
        print(t)
        print(route_question(t).to_dict())
        print("-" * 80)
