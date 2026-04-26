# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

from transformers import PreTrainedTokenizerBase, ProcessorMixin

from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True)
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True)
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def extract_system_prompt_and_generation(tokenizer):
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True)
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True)
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    )
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt


def apply_chat_template(
    processor: PreTrainedTokenizerBase | ProcessorMixin,
    messages: list[dict],
    *,
    tokenize: bool = True,
    add_generation_prompt: bool = True,
    tools=None,
    return_dict: bool = False,
    **kwargs,
) -> list[int] | str:
    """Apply chat template with a Qwen3.5-compatible fallback.

    Some Qwen3.5 templates require the conversation to contain at least one user
    message. When that constraint is violated, prepend an empty dummy user turn
    and strip its prefix from the final result.
    """
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
    except Exception:
        dummy_user_message = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        dummy_user_prefix = processor.apply_chat_template(
            dummy_user_message,
            tokenize=tokenize,
            add_generation_prompt=False,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
        output = processor.apply_chat_template(
            dummy_user_message + messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )

        if not tokenize:
            return output[len(dummy_user_prefix) :]
        if not return_dict:
            if isinstance(output[0], list):
                assert len(output) == 1, "output must be a list[int] or list[list[int]]"
                dummy_user_prefix = dummy_user_prefix[0]
                output = output[0]
            return normalize_token_ids(output[len(dummy_user_prefix) :])

        dummy_user_prefix = dict(dummy_user_prefix)
        output = dict(output)
        prefix_len = dummy_user_prefix["input_ids"].shape[1]
        output["input_ids"] = output["input_ids"][:, prefix_len:]
        output["attention_mask"] = output["attention_mask"][:, prefix_len:]
        if "mm_token_type_ids" in output:
            output["mm_token_type_ids"] = output["mm_token_type_ids"][:, prefix_len:]
        return output
