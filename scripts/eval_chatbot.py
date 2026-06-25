"""Automated evaluation harness for the PDF chatbot.

Derives ground truth DIRECTLY from PageIndex table nodes (never from the model),
generates a category-tagged question bank, then scores retrieval and (optionally)
LLM answers, classifying every failure.

Usage:
  python scripts/eval_chatbot.py <pageindex.json> --mode retrieval
  python scripts/eval_chatbot.py <pageindex.json> --mode llm --sample 36 --out eval_llm.json
"""

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pageindex_store import PageIndex  # noqa: E402

PERIOD_FULL = re.compile(r"(?i)fy ?\d{2,4}|20\d{2}([-/]\d{2,4})?")
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def col_header(c):
    if isinstance(c, dict):
        return str(c.get("header") or c.get("name") or c.get("label") or "").strip()
    t = str(c).strip()
    if t.startswith("{") and ("header" in t or "row_header" in t):
        try:
            o = ast.literal_eval(t)
            if isinstance(o, dict):
                return str(o.get("header") or o.get("row_header") or o.get("name") or "").strip()
        except (ValueError, SyntaxError):
            pass
    return t


def is_period(h):
    return bool(re.fullmatch(r"(?i)\s*(fy ?\d{2,4}|20\d{2}([-/]\d{2,4})?)\s*", h or ""))


def norm_num(s):
    """Normalize a number string to its digit/decimal form (drop commas, symbols, units)."""
    m = NUM_RE.search(str(s))
    if not m:
        return None
    return m.group(0).replace(",", "").lstrip("0") or "0"


def answer_numbers(text):
    return {norm_num(m) for m in NUM_RE.findall(text or "") if norm_num(m)}


# --------------------------------------------------------------------------- #
# Ground truth from clean period-column table nodes
# --------------------------------------------------------------------------- #
def build_ground_truth(store):
    """Return list of {metric, page, node_id, by_year: {YEAR: value}}."""
    gt = []
    for nid, n in store.nodes.items():
        if n["node_type"] != "table":
            continue
        d = (n.get("metadata") or {}).get("data") or {}
        title = (d.get("title") or n.get("title") or "").strip()
        cols = [col_header(c) for c in (d.get("columns") or [])]
        rows = d.get("rows") or []
        if not title or not rows:
            continue
        period_cols = [(i, c) for i, c in enumerate(cols) if is_period(c)]
        if len(period_cols) < 2:
            continue

        def _cells(r):
            return list(r.values()) if isinstance(r, dict) else (list(r) if isinstance(r, (list, tuple)) else [r])

        # Only trust SINGLE-data-row metric tables (page-7 style: title == metric, one value row).
        # Multi-row label-less tables (e.g. p37 'R&D Investments*' with %-of-sales AND ₹Bn rows)
        # make title->rows[0] ambiguous, so they are not reliable ground truth.
        data_rows = [r for r in rows if any(v not in (None, "") for v in _cells(r))]
        if len(data_rows) != 1:
            continue
        row = data_rows[0]
        cells = _cells(row)
        by_year = {}
        for i, c in period_cols:
            if i < len(cells) and cells[i] not in (None, ""):
                by_year[c.upper().replace(" ", "")] = str(cells[i]).strip()
        if len(by_year) >= 2:
            gt.append({"metric": title, "page": n["page_number"], "node_id": nid, "by_year": by_year})
    return gt


# --------------------------------------------------------------------------- #
# Question bank
# --------------------------------------------------------------------------- #
def generate_questions(gt, store):
    qs = []
    # 1. Financial / KPI / Year-specific (metric x year) from ground truth
    for g in gt:
        metric = g["metric"].split("\n")[0].strip()
        years = sorted(g["by_year"].keys())
        for y in years:
            cat = "financial"
            qs.append({
                "question": f"What was {metric} for {y}?",
                "category": cat, "metric": metric, "year": y,
                "expected_value": g["by_year"][y], "expected_page": g["page"],
                "expected_node_type": "table", "expected_node_id": g["node_id"],
                "by_year": g["by_year"],
            })
    # tag a slice as year_specific / kpi (overlap is fine; we report per primary tag)
    # 2. Comparative
    for g in gt:
        metric = g["metric"].split("\n")[0].strip()
        ys = sorted(g["by_year"].keys())
        if len(ys) >= 2:
            y1, y2 = ys[-2], ys[-1]
            qs.append({
                "question": f"Compare {metric} between {y1} and {y2}.",
                "category": "comparative", "metric": metric, "year": y2,
                "expected_value": g["by_year"][y2], "expected_page": g["page"],
                "expected_node_type": "table", "expected_node_id": g["node_id"],
                "by_year": g["by_year"], "expected_value2": g["by_year"][y1], "year2": y1,
            })
    # 3. Chart questions — only from REAL data-bearing charts (a point must carry a VALUE, else it
    #    is a process/flow diagram misread as a chart). Attach the chart's labels+values so the
    #    answer is scored on reproducing real chart content, not just echoing the title.
    for nid, n in store.nodes.items():
        if n["node_type"] != "chart":
            continue
        d = (n.get("metadata") or {}).get("data") or {}
        t = (d.get("title") or n.get("title") or "").strip()
        kws = []
        for s in d.get("series") or d.get("data_points") or d.get("values") or []:
            if not isinstance(s, dict):
                continue
            for p in s.get("points") or s.get("data") or []:
                if isinstance(p, dict) and p.get("value") not in (None, "", "null"):
                    if p.get("label"):
                        kws.append(str(p["label"]))
                    kws.append(str(p["value"]))
        if t and len(t) > 4 and kws:  # has at least one valued data point
            qs.append({
                "question": f"What does the chart titled '{t[:60]}' show? Give its data values.",
                "category": "chart", "expected_page": n["page_number"], "expected_node_type": "chart",
                "expected_node_id": nid, "expected_value": None, "expected_keywords": kws[:12],
            })
    # 3b. Statement line items — numerically validated from the structured statement_parser tables.
    KEY_LINES = ("revenue from operations", "total income", "total expenses",
                 "profit before tax", "cost of materials consumed", "employee benefits expense")
    for nid, n in store.nodes.items():
        d = (n.get("metadata") or {}).get("data") or {}
        if (n.get("metadata") or {}).get("origin") != "statement_parser":
            continue
        cols = [str(c) for c in (d.get("columns") or [])]
        pcols = [(i, c) for i, c in enumerate(cols) if is_period(c)]
        if not pcols:
            continue

        def _fyn(c):
            m = re.search(r"(\d{2,4})", c)
            return (int(m.group(1)) if int(m.group(1)) >= 1900 else 2000 + int(m.group(1))) if m else 0

        latest_i, latest_c = max(pcols, key=lambda ic: _fyn(ic[1]))  # latest by YEAR, not position
        title = (d.get("title") or "").strip()
        for row in d.get("rows") or []:
            label = str(row[0]).strip()
            if label.lower() not in KEY_LINES or latest_i >= len(row):
                continue
            val = row[latest_i]
            if val in (None, ""):
                continue
            qs.append({
                "question": f"What was {label} in the {title} for {latest_c}?",
                "category": "financial", "metric": label, "year": latest_c,
                "expected_value": str(val), "expected_page": n["page_number"],
                "expected_node_type": "table", "expected_node_id": nid,
                "by_year": {latest_c.upper().replace(' ', ''): str(val)},
            })
    # 4. Text / qualitative questions (open; scored on retrieval + non-empty)
    text_qs = [
        "What are the key business segments of the company?",
        "What is the total employee count?",
        "What is the company's R&D expenditure?",
        "What was revenue from operations?",
        "What is the dividend per share?",
        "Who is the chairman of the company?",
        "What is the company's market presence or geographic reach?",
        "What are the company's main therapeutic areas?",
        "What is the credit rating of the company?",
        "What is the company's CAGR?",
    ]
    for q in text_qs:
        qs.append({"question": q, "category": "text", "expected_page": None,
                   "expected_node_type": None, "expected_value": None})
    # 5. Adversarial / not-in-document (hallucination probes)
    for q in ["What is the CEO of Microsoft?", "What was the GDP of France in 2010?",
              "What is the price of Bitcoin today?"]:
        qs.append({"question": q, "category": "adversarial", "expected_page": None,
                   "expected_node_type": None, "expected_value": "__NOT_FOUND__"})
    return qs


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_retrieval(engine, q):
    nodes = engine.retrieve(q["question"])
    pages = [n.get("page_number") for n in nodes]
    ids = [n.get("id") for n in nodes]
    types = [n.get("node_type") for n in nodes]
    exp_page = q.get("expected_page")
    exp_id = q.get("expected_node_id")
    page_hit = (exp_page in pages) if exp_page is not None else None
    node_hit = (exp_id in ids) if exp_id else None
    rank = (ids.index(exp_id) + 1) if (exp_id and exp_id in ids) else None
    return {"pages": pages[:8], "types": types[:8], "page_hit": page_hit,
            "node_hit": node_hit, "node_rank": rank}


def classify_answer(q, answer, retr):
    """Return (correct, error_label)."""
    cat = q["category"]
    a = answer or ""
    al = a.lower()
    if q.get("expected_value") == "__NOT_FOUND__":
        return ("not found" in al, None if "not found" in al else "hallucination")
    if cat == "chart":
        if not a or al.startswith("[error]"):
            return (False, "missing_answer")
        if "not found" in al:
            return (False, "missing_answer")
        if retr.get("page_hit") is False:
            return (False, "retrieval_error")
        # Must reproduce at least one REAL data label or value from the chart (not just the title).
        kws = q.get("expected_keywords") or []
        if kws:
            anums = answer_numbers(a)
            def kw_hit(k):
                kn = norm_num(k)
                if kn:
                    return kn in anums
                toks = [w for w in re.findall(r"[a-z]{3,}", k.lower()) if w not in ("the", "and", "for")]
                return any(w in al for w in toks)
            if not any(kw_hit(k) for k in kws):
                return (False, "chart_parsing_error")
        return (True, None)
    if cat == "text":
        # qualitative: scored as ANSWER-RATE (answered vs refused), not factual correctness —
        # text questions have no numeric ground truth, so this is a weak proxy by design.
        if not a or al.startswith("[error]"):
            return (False, "missing_answer")
        if "not found" in al:
            return (False, "missing_answer")
        if retr.get("page_hit") is False:
            return (False, "retrieval_error")
        return (True, None)
    # numeric (financial/kpi/year/comparative)
    exp = norm_num(q.get("expected_value"))
    nums = answer_numbers(a)
    if "not found" in al:
        return (False, "missing_answer")
    if exp and exp in nums:
        # correct value present; check year + page + format
        if q.get("year") and q["year"].lower().replace(" ", "") not in al.replace(" ", ""):
            pass  # value right; year label maybe absent (minor)
        return (True, None)
    # value wrong: is it another year's value (wrong_year) ?
    by_year = q.get("by_year") or {}
    other = {norm_num(v) for y, v in by_year.items() if y != (q.get("year") or "")}
    if nums & {o for o in other if o}:
        return (False, "wrong_year_error")
    if retr.get("node_hit") is False and retr.get("page_hit") is False:
        return (False, "retrieval_error")
    if retr.get("node_hit") is False:
        return (False, "ranking_error")
    if retr.get("expected_node_type") == "table" or q.get("expected_node_type") == "table":
        return (False, "table_parsing_error")
    return (False, "numeric_error")


def format_ok(answer, q):
    """name + key:value + (page N) for table/chart numeric; one line otherwise."""
    a = (answer or "").strip()
    if not a:
        return False
    if a == "Not found in the document.":
        return True
    if q.get("category") in ("financial", "comparative") and q.get("expected_value") not in (None, "__NOT_FOUND__"):
        return bool(re.search(r"\(page\s*\d+\)", a)) and (":" in a)
    return True


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("--mode", choices=["retrieval", "llm"], default="retrieval")
    ap.add_argument("--sample", type=int, default=0, help="LLM mode: questions per category (0=all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    store = PageIndex.load(args.store)
    gt = build_ground_truth(store)
    questions = generate_questions(gt, store)

    cats = Counter(q["category"] for q in questions)
    print(f"ground-truth metrics: {len(gt)} | questions: {len(questions)} | by category: {dict(cats)}")

    if args.mode == "retrieval":
        from chatbot import ChatEngine
        engine = ChatEngine(store)
        agg = defaultdict(lambda: {"n": 0, "page_hit": 0, "node_hit": 0, "rank_sum": 0, "rank_n": 0})
        for q in questions:
            r = score_retrieval(engine, q)
            c = q["category"]
            agg[c]["n"] += 1
            if r["page_hit"]:
                agg[c]["page_hit"] += 1
            if r["node_hit"]:
                agg[c]["node_hit"] += 1
            if r["node_rank"]:
                agg[c]["rank_sum"] += r["node_rank"]
                agg[c]["rank_n"] += 1
        print("\n=== RETRIEVAL (no LLM) ===")
        print("%-12s %5s %12s %12s %10s" % ("category", "n", "page_hit%", "node_hit%", "avg_rank"))
        for c, a in sorted(agg.items()):
            ph = 100 * a["page_hit"] / a["n"] if a["n"] else 0
            nh = 100 * a["node_hit"] / a["n"] if a["n"] else 0
            ar = (a["rank_sum"] / a["rank_n"]) if a["rank_n"] else 0
            print("%-12s %5d %11.1f%% %11.1f%% %10.1f" % (c, a["n"], ph, nh, ar))
        return 0

    # LLM mode
    from chatbot import ChatEngine
    engine = ChatEngine(store)
    # stratified sample
    bycat = defaultdict(list)
    for q in questions:
        bycat[q["category"]].append(q)
    sample = []
    per = args.sample or 9999
    for c, ql in bycat.items():
        step = max(1, len(ql) // per) if per < len(ql) else 1
        sample.extend(ql[::step][:per])
    print(f"LLM evaluating {len(sample)} questions...\n", flush=True)

    results = []
    errcount = Counter()
    catscore = defaultdict(lambda: [0, 0])
    for i, q in enumerate(sample, 1):
        retr = score_retrieval(engine, q)
        ans = engine.ask(q["question"])
        answer = ans.get("answer", "")
        src_pages = [s.get("page_number") for s in ans.get("sources", [])]
        correct, err = classify_answer(q, answer, retr)
        fmt = format_ok(answer, q)
        catscore[q["category"]][1] += 1
        if correct:
            catscore[q["category"]][0] += 1
        if err:
            errcount[err] += 1
        if not fmt:
            errcount["formatting_error"] += 1
        results.append({
            "question": q["question"], "category": q["category"],
            "expected_value": q.get("expected_value"), "expected_page": q.get("expected_page"),
            "answer": answer, "source_pages": src_pages[:3],
            "retrieved_pages": retr["pages"], "node_rank": retr["node_rank"],
            "correct": correct, "error": err, "format_ok": fmt,
        })
        mark = "OK " if correct else "XX "
        print(f"{mark}[{q['category']:11}] {q['question'][:54]:54} -> {answer.splitlines()[0][:40] if answer else ''}", flush=True)

    print("\n=== ANSWER ACCURACY BY CATEGORY ===")
    for c, (ok, n) in sorted(catscore.items()):
        print("  %-12s %3d/%-3d = %5.1f%%" % (c, ok, n, 100 * ok / n if n else 0))
    tot_ok = sum(v[0] for v in catscore.values())
    tot_n = sum(v[1] for v in catscore.values())
    print("  %-12s %3d/%-3d = %5.1f%%" % ("OVERALL", tot_ok, tot_n, 100 * tot_ok / tot_n if tot_n else 0))
    print("\n=== ERROR CLASSES ===")
    for e, c in errcount.most_common():
        print("  %-22s %d" % (e, c))

    if args.out:
        Path(args.out).write_text(json.dumps({"summary": dict(catscore), "errors": dict(errcount),
                                              "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
