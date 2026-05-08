# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os


def _azure_base_url() -> str:
    return os.environ.get("AZURE_OPENAI_ENDPOINT", "")


def _azure_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY", "")


def _azure_api_version() -> str:
    return os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")


def _azure_deployment(default: str) -> str:
    return os.environ.get("AZURE_OPENAI_DEPLOYMENT", default)


model_config = {
    "model_config_name": {
        "model_name": "MODEL_NAME",
        "base_url": "YOUR_BASE_URL",
        "api_key": "YOUR_API_KEY",
    },
    "gpt-4o": {
        "model_name": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "use_azure": False,
        "generate_kwargs": {"max_tokens": 16384},
    },
    "gpt-4.1": {
        "model_name": "gpt-4.1",
        "base_url": "https://api.openai.com/v1",
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "use_azure": False,
        "generate_kwargs": {"max_tokens": 32768},
    },
    "gpt-4.1-eval": {
        "model_name": _azure_deployment("gpt-4.1"),
        "base_url": _azure_base_url(),
        "api_key": _azure_api_key(),
        "use_azure": True,
        "azure_api_version": _azure_api_version(),
        "generate_kwargs": {"max_tokens": 10240},
        "temperature": 0,
    },
    "gpt-5-mini-eval": {
        "model_name": _azure_deployment("gpt-5-mini"),
        "base_url": _azure_base_url(),
        "api_key": _azure_api_key(),
        "use_azure": True,
        "azure_api_version": _azure_api_version(),
        "generate_kwargs": {"max_tokens": 10240},
        "temperature": 0,
    },
    "default_eval_config": {
        "model_name": _azure_deployment("gpt-5-mini"),
        "base_url": _azure_base_url(),
        "api_key": _azure_api_key(),
        "use_azure": True,
        "azure_api_version": _azure_api_version(),
        "generate_kwargs": {"max_tokens": 10240},
        "temperature": 0,
    },
    "quest": {
        "model_name": "deepresearch",
        "base_url": os.environ.get("QUEST_EVAL_BASE_URL", "http://localhost:6000/v1"),
        "api_key": "EMPTY",
        "use_azure": False,
        "generate_kwargs": {
            "max_tokens": 20000,
            "temperature": 0.6,
            "top_p": 0.95,
            "presence_penalty": 1.1,
        },
    },
}
