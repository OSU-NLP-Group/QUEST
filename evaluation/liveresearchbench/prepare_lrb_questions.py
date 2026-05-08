#!/usr/bin/env python3
"""Download LiveResearchBench questions from HuggingFace and dump as a jsonl
that the QUEST inference pipeline (run_multi_react.py) can consume.

Each output line: {"id": <qid>, "question": <question>}
"""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        required=True,
        help="Output jsonl path (id + question per line).",
    )
    parser.add_argument(
        "--subset",
        default="question_with_checklist",
        choices=["question_with_checklist", "question_only"],
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--use-realtime", action="store_true",
                        help="Replace {{current_year}}/{{date}} placeholders with today.")
    args = parser.parse_args()

    # LiveResearchBench is a gated HF dataset — make sure we're logged in.
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
        except Exception as e:
            print(f"[prepare_lrb_questions] huggingface_hub login failed: {e}")
    else:
        print("[prepare_lrb_questions] Warning: no HF_TOKEN in env; "
              "gated dataset download will fail.")

    # Prefer the in-repo loader so realtime substitution matches the eval side.
    lrb_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "LiveResearchBench")
    if lrb_root not in sys.path:
        sys.path.insert(0, lrb_root)

    try:
        from liveresearchbench.common.io_utils import load_liveresearchbench_dataset
        # Returns dict[qid -> {qid, question, checklists}]
        data = load_liveresearchbench_dataset(use_realtime=args.use_realtime)
        if not data:
            raise RuntimeError("loader returned empty dict")
        rows = []
        for qid, entry in data.items():
            question = entry.get("question", "")
            if qid and question:
                rows.append({"id": qid, "question": question, "answer": ""})
    except Exception as e:
        print(f"[prepare_lrb_questions] in-repo loader failed ({e}); "
              f"falling back to raw HuggingFace load.")
        from datasets import load_dataset
        ds = load_dataset("Salesforce/LiveResearchBenchFull", args.subset, split=args.split)
        rows = [{"id": r["qid"], "question": r["question"]} for r in ds]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} questions -> {args.output}")


if __name__ == "__main__":
    main()
