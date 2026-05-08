#!/usr/bin/env python3
"""Convert QUEST inference iter*.jsonl outputs into LiveResearchBench
preprocess.py-compatible directory layout.

For each iter file, write reports to:
    <output_dir>/<identifier>-iter<N>/qid_<qid>_report.md

The qid is taken from the inference record's `id` field (which originated from
prepare_lrb_questions.py); the report body is the model `prediction`.
"""
import argparse
import json
import os
from pathlib import Path


def load_question_to_qid(questions_file: Path) -> dict:
    mapping = {}
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q = (r.get("question") or "").strip()
            qid = r.get("id") or r.get("qid")
            if q and qid:
                mapping[q] = qid
    return mapping


def convert_one(iter_file: Path, model_dir: Path, q2qid: dict) -> int:
    model_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(iter_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in rec:
                continue
            qid = rec.get("id") or rec.get("qid")
            if not qid:
                qid = q2qid.get((rec.get("question") or "").strip())
            prediction = rec.get("prediction", "")
            if not qid or not prediction:
                print(f"  skip (no qid or prediction): {rec.get('question','')[:60]}")
                continue
            out = model_dir / f"qid_{qid}_report.md"
            out.write_text(prediction, encoding="utf-8")
            n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, required=True,
                   help="Directory containing iter{N}.jsonl files from inference.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Root model_outputs dir consumed by preprocess.py.")
    p.add_argument("--identifier", type=str, required=True,
                   help="Subdirectory prefix; final dir is <identifier>-iter<N>.")
    p.add_argument("--iters", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--questions-file", type=Path, required=True,
                   help="The prepared LRB questions jsonl (provides question -> qid).")
    args = p.parse_args()

    q2qid = load_question_to_qid(args.questions_file)
    print(f"Loaded {len(q2qid)} question->qid mappings from {args.questions_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i in args.iters:
        iter_file = args.base_dir / f"iter{i}.jsonl"
        if not iter_file.exists():
            print(f"Warning: missing {iter_file}")
            continue
        model_dir = args.output_dir / f"{args.identifier}-iter{i}"
        n = convert_one(iter_file, model_dir, q2qid)
        print(f"  {iter_file.name} -> {model_dir} ({n} reports)")


if __name__ == "__main__":
    main()
