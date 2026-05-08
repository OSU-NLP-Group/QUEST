# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
This module provides data loaders for the WideSearch dataset.

It includes classes to load queries and responses from local files or Hugging Face datasets,
and to handle the extraction of dataframes from markdown responses.

Example:

```py
>>> data_loader = WideSearchDataLoaderHF()
>>> print(data_loader.load_query_by_instance_id("ws_en_001"))
```

"""

import json
import os
import re
from dataclasses import asdict, dataclass
from io import StringIO
from typing import Optional

import pandas as pd
from datasets import load_dataset
from huggingface_hub import snapshot_download, try_to_load_from_cache

from src.utils.logger import logger
from src.utils.utils import norm_column


@dataclass
class WideSearchQuery:
    instance_id: str
    query: str
    evaluation: dict
    answer: pd.DataFrame
    language: str


class WideSearchDataLoader:
    def __init__(self, data_path: str, answer_root: str):
        self.data = self.load_data(data_path, answer_root)

    def load_answer(self, answer_path, required_columns):
        if not os.path.exists(answer_path):
            logger.error(f"answer_path {answer_path} not found")
            return None
        answer = pd.read_csv(answer_path)
        answer.columns = [norm_column(col.strip()) for col in answer.columns]
        for col in required_columns:
            if col not in answer.columns:
                logger.error(
                    f"answer_path {answer_path} required_columns {required_columns} not found"
                )
                return None
        answer = answer[required_columns]
        return answer

    def load_data(self, data_path: str, answer_root: str):
        if not os.path.exists(data_path):
            logger.error(f"data_path {data_path} not found")
            return {}
        data = pd.read_json(data_path, lines=True).to_dict(orient="records")
        new_data = {}
        for item in data:
            answer_path = f"{answer_root}/{item['instance_id']}.csv"
            item["answer"] = self.load_answer(
                answer_path, item["evaluation"]["required"]
            )
            if item["answer"] is None:
                continue
            new_data[item["instance_id"]] = WideSearchQuery(**item)
        logger.info(f"load {len(new_data)} queries from {data_path}")
        return new_data

    def load_query_by_instance_id(self, instance_id: str):
        assert instance_id in self.data, f"instance_id {instance_id} not found"
        return self.data[instance_id]

    def get_instance_id_list(self):
        return list(self.data.keys())


class WideSearchDataLoaderHF:
    def __init__(
        self,
        repo_id: str = "ByteDance-Seed/WideSearch",
        answer_root: str = "widesearch_gold",
    ):
        self.repo_id = repo_id
        self.answer_root = answer_root
        snapshot_download(repo_id=self.repo_id, repo_type="dataset")
        self.data = self.load_data()

    def load_answer(self, instance_id, required_columns):
        answer_path = f"{self.answer_root}/{instance_id}.csv"
        cache_answer_path = try_to_load_from_cache(
            repo_id=self.repo_id, filename=answer_path, repo_type="dataset"
        )
        if cache_answer_path is None:
            return None
        try:
            answer = pd.read_csv(cache_answer_path)
        except Exception:
            return None
        answer.columns = [norm_column(col.strip()) for col in answer.columns]
        for col in required_columns:
            if col not in answer.columns:
                logger.error(
                    f"answer_path {answer_path} required_columns {col} not found in {answer.columns}"
                )
                return None
        answer = answer[required_columns]
        return answer

    def load_data(self):
        data = load_dataset(self.repo_id)["full"]
        new_data = {}
        for item in data:
            assert isinstance(item, dict)
            item["evaluation"] = json.loads(item["evaluation"])
            item["answer"] = self.load_answer(
                item["instance_id"], item["evaluation"]["required"]
            )

            if item["answer"] is None:
                continue
            new_data[item["instance_id"]] = WideSearchQuery(**item)
        logger.info(f"load {len(new_data)} queries from {self.repo_id}")
        return new_data

    def load_query_by_instance_id(self, instance_id: str):
        assert instance_id in self.data, f"instance_id {instance_id} not found"
        return self.data[instance_id]

    def get_instance_id_list(self):
        return list(self.data.keys())


@dataclass
class WideSearchResponse:
    instance_id: str
    response: str
    messages: Optional[list[dict]] = None
    trial_idx: Optional[int] = None

    def extract_dataframe(self) -> pd.DataFrame | None:
        response_df = None
        resp = self.response

        # Fix 1: Strip markdown links [text](url) -> text before parsing
        resp = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", resp)

        # Fix 1b: Strip bold/italic markdown (**text** -> text, *text* -> text)
        resp = re.sub(r"\*\*([^*]+)\*\*", r"\1", resp)
        resp = re.sub(r"\*([^*]+)\*", r"\1", resp)

        # Fix 2: Handle unclosed ```markdown blocks
        markdown_str = re.findall(r"```markdown(.*?)```", resp, re.DOTALL)
        if not markdown_str:
            markdown_str = re.findall(r"```markdown(.*)", resp, re.DOTALL)

        if not markdown_str:
            pipe_positions = [m.start() for m in re.finditer(r"\|", resp)]
            if len(pipe_positions) >= 4:
                first_pipe = pipe_positions[0]
                last_pipe = pipe_positions[-1]
                start = resp.rfind("\n", 0, first_pipe)
                start = 0 if start == -1 else start
                end = resp.find("\n", last_pipe)
                end = len(resp) if end == -1 else end
                table_candidate = resp[start:end]
                markdown_str = re.findall(r"((?:\|.*\n?)+)", table_candidate)

        if markdown_str:
            # Fix 3: Merge multiple tables with same column structure
            all_table_text = markdown_str[0].strip()
            if len(markdown_str) > 1:
                first_lines = all_table_text.split("\n")
                first_header = first_lines[0].replace(" ", "").lower() if first_lines else ""
                for extra in markdown_str[1:]:
                    extra = extra.strip()
                    if not extra:
                        continue
                    extra_lines = extra.split("\n")
                    extra_header = extra_lines[0].replace(" ", "").lower() if extra_lines else ""
                    if extra_header == first_header:
                        data_lines = []
                        for el in extra_lines:
                            el = el.strip()
                            if set(el).issubset(set("|- :")) or "|" not in el:
                                continue
                            if el.replace(" ", "").lower() == first_header:
                                continue
                            data_lines.append(el)
                        if data_lines:
                            all_table_text += "\n" + "\n".join(data_lines)

            logger.debug(f"find markdown_str {all_table_text[:64]} ...")
            lines = all_table_text.split("\n")
            lines[0] = lines[0].replace(" ", "").lower()  # columns
            lines = [line.strip() for line in lines]
            new_lines = []
            for line in lines:
                if set(line.strip()).issubset(set("|- :")) or "|" not in line:
                    continue
                new_lines.append("|".join([_line.strip() for _line in line.split("|")]))
            markdown_str_final = "\n".join(new_lines)
            response_df = pd.read_csv(StringIO(markdown_str_final), sep="|")
            response_df = response_df.loc[
                :, ~response_df.columns.str.startswith("Unnamed")
            ]

            # Fix 4: Forward-fill merged cells (empty cells inherit from above)
            response_df = response_df.replace(r"^\s*$", pd.NA, regex=True)
            response_df = response_df.ffill()

        else:
            logger.error(f"response {self.response} not found markdown_str")
        return response_df


class WideSearchResponseLoader:
    @staticmethod
    def load_response(response_path: str) -> list[WideSearchResponse]:
        response_list = pd.read_json(response_path, lines=True).to_dict(
            orient="records"
        )
        new_response_list = []
        for item in response_list:
            new_response_list.append(WideSearchResponse(**item))
        return new_response_list

    @staticmethod
    def dump_response(response_list: list[WideSearchResponse], response_path: str):
        new_response_list = [asdict(item) for item in response_list]
        pd.DataFrame(new_response_list).to_json(
            response_path, orient="records", lines=True, force_ascii=False
        )
        logger.info(f"dump {len(response_list)} responses to {response_path}")
        return
