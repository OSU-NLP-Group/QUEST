#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""Evaluate WideSearch response files without importing the inference agent stack."""

import dataclasses
import json
import os
import traceback
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np

from src.evaluation.data_loader import (
    WideSearchDataLoaderHF,
    WideSearchResponseLoader,
)
from src.evaluation.evaluation import EvaluationResult, evaluate_single_query
from src.utils.logger import logger

logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")


class EvalTask:
    def __init__(
        self,
        query,
        response_path: str,
        result_save_path: str,
        eval_model_config_name: str,
        use_cache: bool,
    ):
        self.query = query
        self.response_path = response_path
        self.result_save_path = result_save_path
        self.eval_model_config_name = eval_model_config_name
        self.use_cache = use_cache
        self.eval_result_path = self.result_save_path.replace(".csv", ".json")

    def load_response(self):
        if not os.path.exists(self.response_path):
            raise FileNotFoundError(f"response_path {self.response_path} not found")
        return WideSearchResponseLoader.load_response(self.response_path)

    def eval(self):
        if os.path.exists(self.eval_result_path) and self.use_cache:
            with open(self.eval_result_path, encoding="utf-8") as f:
                return EvaluationResult(**json.load(f))

        if not os.path.exists(self.response_path):
            logger.error(f"response_path {self.response_path} not found, skip")
            response_list = [None]
        else:
            response_list = self.load_response()

        assert response_list, f"response is None, response_path: {self.response_path}"
        eval_result = evaluate_single_query(
            self.query,
            response_list[0],
            self.result_save_path,
            self.eval_model_config_name,
        )
        with open(self.eval_result_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(eval_result), f, ensure_ascii=False, indent=4)
        return eval_result


def calc_summary_results(tasks: list[EvalTask], summary_result_path: str, trial_num: int):
    metrics = [
        "score",
        "precision_by_row",
        "recall_by_row",
        "f1_by_row",
        "precision_by_item",
        "recall_by_item",
        "f1_by_item",
    ]

    all_results = {m: [] for m in metrics}
    id_to_task = {}
    for task in tasks:
        id_to_task.setdefault(task.query.instance_id, []).append(task)

    for iid, task_list in id_to_task.items():
        trial_metrics = {m: [] for m in metrics}
        for task in task_list:
            if not os.path.exists(task.eval_result_path):
                continue
            with open(task.eval_result_path, encoding="utf-8") as f:
                result = json.load(f)
            for metric in metrics:
                if metric in result:
                    trial_metrics[metric].append(result[metric])

        for metric in metrics:
            values = trial_metrics[metric]
            if not values or len(values) < trial_num:
                raise ValueError(
                    f"Not enough trials for metric {metric} on instance {iid}. "
                    f"Expected {trial_num}, got {len(values)}."
                )
            all_results[metric].append({
                "avg_n": float(np.mean(values)),
                "max_n": float(np.max(values)),
                "min_n": float(np.min(values)),
            })

    summary = {}
    for metric, vals in all_results.items():
        if not vals:
            continue
        summary[metric] = {
            "avg_n": float(np.mean([v["avg_n"] for v in vals])),
            "max_n": float(np.mean([v["max_n"] for v in vals])),
            "min_n": float(np.mean([v["min_n"] for v in vals])),
        }

    logger.info(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(summary_result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_config_name", type=str, default="quest")
    parser.add_argument("--response_root", type=str, required=True)
    parser.add_argument("--result_save_root", type=str, required=True)
    parser.add_argument("--eval_model_config_name", type=str, default="gpt-5-mini-eval")
    parser.add_argument("--trial_num", type=int, default=1)
    parser.add_argument("--instance_id", type=str, default="")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--thread_num", type=int, default=4)
    args = parser.parse_args()

    data_loader = WideSearchDataLoaderHF()
    instance_ids = data_loader.get_instance_id_list()
    requested = set(x.strip() for x in args.instance_id.split(",") if x.strip())

    os.makedirs(args.result_save_root, exist_ok=True)
    tasks = []
    for instance_id in instance_ids:
        if requested and instance_id not in requested:
            continue
        query = data_loader.load_query_by_instance_id(instance_id)
        for trial_idx in range(args.trial_num):
            response_path = (
                f"{args.response_root}/"
                f"{args.model_config_name}_{instance_id}_{trial_idx}_response.jsonl"
            )
            result_save_path = (
                f"{args.result_save_root}/"
                f"{args.model_config_name}_{instance_id}_{trial_idx}_eval_result.csv"
            )
            tasks.append(
                EvalTask(
                    query=deepcopy(query),
                    response_path=response_path,
                    result_save_path=result_save_path,
                    eval_model_config_name=args.eval_model_config_name,
                    use_cache=args.use_cache,
                )
            )

    logger.info(f"total task num: {len(tasks)}")
    with ThreadPoolExecutor(max_workers=args.thread_num) as executor:
        results = executor.map(lambda task: task.eval(), tasks)
        try:
            for result in results:
                logger.info(f"eval success, instance_id: {result.instance_id}")
        except Exception:
            logger.error(f"eval error: {traceback.format_exc()}")

    summary_result_path = (
        f"{args.result_save_root}/{args.model_config_name}_trial_num_{args.trial_num}_summary.json"
    )
    calc_summary_results(tasks=tasks, summary_result_path=summary_result_path, trial_num=args.trial_num)


if __name__ == "__main__":
    main()
