#!/usr/bin/env python3
"""Build a 100-qid LiveResearchBench-Full questions jsonl that's usable for
both the inference run_multi_react.py pipeline and the run_lrb_eval.sh
convert/eval pipeline.

Two modes for the "old 80" entries:

--mode from-old-jsonl
    Take the rows from an existing 80-qid jsonl verbatim. Use this when the
    model's iter*.jsonl outputs were generated on the same day as that jsonl
    (i.e. same {{date}} substitution), so the question strings line up exactly.

--mode recover-from-reports
    Recover (qid, question) pairs by matching each record's prediction string
    in <model-output-dir>/iter1.jsonl against content of
    <reports-dir>/<identifier>-iter1/qid_*_report.md (produced by a previous
    convert_to_lrb_format.py run). Use this when the iter question strings
    diverged from the current 80-qid jsonl (e.g. older {{date}}).

In both modes, the 20 new qids present in LiveResearchBenchFull but absent
from the old 80 are pulled from HF with {{current_year}}/{{date}} substitution.
"""

import argparse
import json
import os
import re
import sys


# Month + day + year in the exact format io_utils.replace_placeholders emits:
#   datetime.strftime("%B %d, %Y")  -> e.g. "April 09, 2026"
_MONTH_NAME = (r"(?:January|February|March|April|May|June|July|August|"
               r"September|October|November|December)")
_DATE_WORD_RE = re.compile(rf"\b{_MONTH_NAME}\s+\d{{1,2}},\s*\d{{4}}\b")
_DATE_ISO_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _date_normalize(s: str) -> str:
    """Collapse concrete dates to a single token so questions with different
    {{date}} substitutions (e.g. "April 09, 2026" vs "April 13, 2026") compare
    equal. Year alone is intentionally left untouched to avoid false merges on
    unrelated numbers."""
    s = _DATE_WORD_RE.sub("<DATE>", s)
    s = _DATE_ISO_RE.sub("<DATE>", s)
    return s


def _hf_login_if_possible():
    tok = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    if not tok:
        print("[prepare] Warning: no HF_TOKEN in env; gated dataset load "
              "will fail.")
        return
    try:
        from huggingface_hub import login
        login(token=tok, add_to_git_credential=False)
    except Exception as e:
        print(f"[prepare] HF login failed: {e}")


def _load_full_new_rows(old_qids, dataset_name, subset, split):
    lrb_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "LiveResearchBench")
    if lrb_root not in sys.path:
        sys.path.insert(0, lrb_root)
    from liveresearchbench.common.io_utils import load_liveresearchbench_dataset

    full = load_liveresearchbench_dataset(
        dataset_name=dataset_name,
        subset=subset,
        split=split,
        use_realtime=True,
    )
    if not full:
        raise RuntimeError("Full dataset loader returned empty")
    new = []
    for qid, entry in full.items():
        if qid in old_qids:
            continue
        q = entry.get("question", "")
        if qid and q:
            new.append({"id": qid, "question": q, "answer": ""})
    return new


def mode_from_old_jsonl(args):
    old_rows = []
    old_qids = set()
    with open(args.old_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("id") or row.get("qid")
            if qid:
                old_qids.add(qid)
            if "answer" not in row:
                row["answer"] = ""
            old_rows.append(row)
    new_rows = _load_full_new_rows(
        old_qids, args.full_dataset_name, args.full_subset, args.full_split)
    print(f"[prepare] from-old-jsonl: {len(old_rows)} old + "
          f"{len(new_rows)} new = {len(old_rows) + len(new_rows)}")
    return old_rows + new_rows


def mode_recover_from_reports(args):
    """Recover (qid, question) for the old 80 qids.

    Signal source: reports_<identifier>.json written by a prior preprocess.py.
    That file has the authoritative (query_id, query) pairs for every qid that
    was graded — distinct queries per qid, no hash collisions like the md
    content route would have on failure predictions.

    The only wrinkle is date drift: preprocess.py and inference both call
    replace_placeholders with datetime.now(), so the {{date}} substitution in
    the reports JSON's `query` can be a few days off from the matching
    iter1.jsonl question. We first pair by exact equality (covers the vast
    majority), then fall back to pairing the remainder by a date-normalized
    equality check.

    We ultimately keep the iter1 question string (so run_multi_react.py's
    question-based resume matches it verbatim) paired with the reports-side
    qid."""
    iter1_path = os.path.join(args.model_output_dir, "iter1.jsonl")
    if not os.path.isfile(iter1_path):
        raise RuntimeError(f"iter1.jsonl not found at {iter1_path}; "
                           "recover-from-reports needs prior inference output.")

    if not os.path.isfile(args.reports_json):
        raise RuntimeError(f"reports JSON not found: {args.reports_json}; "
                           "this should be the preprocess.py output from the "
                           "earlier 80-qid eval.")

    # 1) Load the prior reports JSON's authoritative (qid, query) pairs.
    with open(args.reports_json, "r", encoding="utf-8") as f:
        reports = json.load(f).get("reports", [])
    qid_to_reports_query = {}
    for rep in reports:
        qid = rep.get("query_id") or rep.get("qid")
        q = (rep.get("query") or "").strip()
        if qid and q and qid not in qid_to_reports_query:
            qid_to_reports_query[qid] = q
    if not qid_to_reports_query:
        raise RuntimeError(f"no (query_id, query) pairs found in "
                           f"{args.reports_json}")

    # 2) Load iter1.jsonl question strings (ordered, deduped).
    iter1_questions = []
    seen_q = set()
    with open(iter1_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in d:
                continue
            q = (d.get("question") or "").strip()
            if q and q not in seen_q:
                seen_q.add(q)
                iter1_questions.append(q)

    # 3) Pair qid -> iter1 question. Phase 1: exact match.
    qid_to_iter_q = {}
    iter_available = set(iter1_questions)
    for qid, rq in qid_to_reports_query.items():
        if rq in iter_available:
            qid_to_iter_q[qid] = rq
            iter_available.discard(rq)

    # 3b) Phase 2: date-normalized match for the remainder (handles {{date}}
    # drift between the earlier preprocess run and the inference run).
    rem_qids = [qid for qid in qid_to_reports_query
                if qid not in qid_to_iter_q]
    rem_iter = list(iter_available)
    norm_to_iter = {}
    for iq in rem_iter:
        norm_to_iter.setdefault(_date_normalize(iq), iq)
    for qid in rem_qids:
        norm_rq = _date_normalize(qid_to_reports_query[qid])
        iq = norm_to_iter.get(norm_rq)
        if iq is not None:
            qid_to_iter_q[qid] = iq
            del norm_to_iter[norm_rq]

    unmatched_qids = [qid for qid in qid_to_reports_query
                      if qid not in qid_to_iter_q]
    if unmatched_qids:
        print(f"[prepare] WARNING: {len(unmatched_qids)} qids could not be "
              f"paired to iter1 questions; they will be treated as new and "
              f"re-run in inference: {unmatched_qids}")

    recovered = [{"id": qid, "question": q, "answer": ""}
                 for qid, q in qid_to_iter_q.items()]
    matched = set(qid_to_iter_q)

    new_rows = _load_full_new_rows(
        matched, args.full_dataset_name, args.full_subset, args.full_split)
    exact_ct = sum(1 for qid in qid_to_iter_q
                   if qid_to_iter_q[qid] == qid_to_reports_query[qid])
    fuzzy_ct = len(qid_to_iter_q) - exact_ct
    print(f"[prepare] recover-from-reports: {exact_ct} exact + "
          f"{fuzzy_ct} date-normalized = {len(recovered)} recovered; "
          f"{len(new_rows)} new; total {len(recovered) + len(new_rows)}")
    return recovered + new_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["from-old-jsonl", "recover-from-reports"])
    p.add_argument("--output", required=True)
    p.add_argument("--full-dataset-name",
                   default="Salesforce/LiveResearchBenchFull")
    p.add_argument("--full-subset", default="question_with_checklist")
    p.add_argument("--full-split", default="test")
    p.add_argument("--old-jsonl",
                   help="[from-old-jsonl] existing 80-qid jsonl path")
    p.add_argument("--model-output-dir",
                   help="[recover-from-reports] dir containing iter1.jsonl "
                        "(inference output from the prior 80-qid run)")
    p.add_argument("--reports-json",
                   help="[recover-from-reports] path to the prior "
                        "extracted_reports/reports_<identifier>.json file "
                        "(authoritative qid↔query mapping)")
    args = p.parse_args()

    _hf_login_if_possible()

    if args.mode == "from-old-jsonl":
        if not args.old_jsonl:
            p.error("--old-jsonl is required for from-old-jsonl")
        rows = mode_from_old_jsonl(args)
    else:
        missing = [f for f in ("model_output_dir", "reports_json")
                   if not getattr(args, f)]
        if missing:
            p.error(f"missing for recover-from-reports: {missing}")
        rows = mode_recover_from_reports(args)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[prepare] wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
