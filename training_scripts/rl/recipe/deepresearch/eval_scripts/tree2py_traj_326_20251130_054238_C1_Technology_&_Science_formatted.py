import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gpt5_pro_api_pricing"
TASK_DESCRIPTION = """
What are the current API pricing rates for OpenAI's GPT-5 Pro model? Provide both the input token price and output token price, expressed per million tokens in US dollars, and cite official OpenAI documentation as your source.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PricingExtraction(BaseModel):
    """
    Structured extraction of the pricing information present in the agent's answer.
    """
    model_name_mentioned: Optional[str] = None
    input_price_per_million_usd: Optional[str] = None
    output_price_per_million_usd: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pricing() -> str:
    return """
    Extract the pricing information for OpenAI's GPT-5 Pro model as stated in the answer.

    You must return the following fields:
    1. model_name_mentioned: The model name associated with the pricing in the answer (e.g., "GPT-5 Pro"). If multiple models are mentioned, choose the one explicitly tied to the provided prices. If none is stated, return null.
    2. input_price_per_million_usd: The input tokens price as expressed per one million (1,000,000) tokens in USD, exactly as stated in the answer text. Do NOT convert units. If the answer uses a different unit (e.g., per 1K tokens) or does not specify per million tokens, return null.
    3. output_price_per_million_usd: The output tokens price as expressed per one million (1,000,000) tokens in USD, exactly as stated in the answer text. Do NOT convert units. If the answer uses a different unit or does not specify per million tokens, return null.
    4. sources: Extract all URLs explicitly presented in the answer as sources for the pricing. Include markdown links' URL targets. If no URLs are provided, return an empty list.

    Notes:
    - Preserve the formatting of price strings as they appear (e.g., "$15", "USD 15", "$15.00").
    - If the answer provides ranges or approximate values, extract the exact string shown.
    - The sources must be valid URLs explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_official_openai_url(url: str) -> bool:
    """
    Check whether the URL appears to be from official OpenAI domains.
    Accept common OpenAI domains such as openai.com, platform.openai.com, help.openai.com, docs.openai.com.
    """
    u = (url or "").strip().lower()
    return (
        "openai.com" in u or
        "platform.openai.com" in u or
        "help.openai.com" in u or
        "docs.openai.com" in u
    )


def any_official_openai_source(urls: List[str]) -> bool:
    return any(is_official_openai_url(u) for u in urls)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_input_pricing(
    evaluator: Evaluator,
    parent_node,
    extracted: PricingExtraction,
) -> None:
    """
    Build and verify the Input Token Pricing subtree.
    All children under this critical node are marked critical to satisfy critical consistency.
    """
    input_node = evaluator.add_sequential(
        id="Input_Token_Pricing",
        desc="Verify that the input token pricing for GPT-5 Pro is provided, expressed per million tokens in US dollars, and supported by an official OpenAI source URL",
        parent=parent_node,
        critical=True
    )

    # 1) Price value must be provided in the answer (per 1M tokens)
    input_price_provided = evaluator.add_custom_node(
        result=bool(extracted.input_price_per_million_usd and extracted.input_price_per_million_usd.strip()),
        id="input_price_provided",
        desc="Input token price (per 1M tokens, USD) is provided in the answer",
        parent=input_node,
        critical=True
    )

    # 2) The answer explicitly expresses the input price per million tokens in USD
    input_unit_leaf = evaluator.add_leaf(
        id="input_price_unit_check",
        desc="The input price is expressed per one million tokens in US dollars in the answer",
        parent=input_node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, the input token price is explicitly expressed per one million (1,000,000) tokens in USD.",
        node=input_unit_leaf,
        additional_instruction=(
            "Check the answer text (not the sources) to see if the input price is clearly stated per 1M tokens in USD. "
            "Accept reasonable variants such as 'per 1M tokens', 'per million tokens', 'per 1,000,000 tokens', '$X per 1M', "
            "or 'USD per million tokens'."
        )
    )

    # 3) The answer must include at least one source URL
    input_sources_present = evaluator.add_custom_node(
        result=bool(extracted.sources and len(extracted.sources) > 0),
        id="input_sources_present",
        desc="At least one source URL is provided in the answer for pricing",
        parent=input_node,
        critical=True
    )

    # 4) At least one source must be official OpenAI documentation
    input_official_source = evaluator.add_custom_node(
        result=any_official_openai_source(extracted.sources),
        id="input_official_openai_source",
        desc="At least one pricing source is an official OpenAI documentation URL",
        parent=input_node,
        critical=True
    )

    # 5) The sources support the claimed input price for GPT-5 Pro (per 1M tokens, USD)
    input_supported_leaf = evaluator.add_leaf(
        id="input_price_supported_by_sources",
        desc="The input price for GPT-5 Pro is supported by the cited official documentation",
        parent=input_node,
        critical=True
    )
    claimed_input_price = extracted.input_price_per_million_usd or ""
    await evaluator.verify(
        claim=(
            f"The official OpenAI documentation states that the input token price for GPT-5 Pro is "
            f"{claimed_input_price} per 1M tokens (USD)."
        ),
        node=input_supported_leaf,
        sources=extracted.sources,
        additional_instruction=(
            "Verify on the provided official OpenAI documentation page(s) that the listed price corresponds specifically to GPT-5 Pro's input token rate. "
            "If the documentation lists prices per 1K tokens, you may proportionally scale to per 1M tokens to judge equivalence "
            "(e.g., multiply by 1000), allowing minor rounding differences. "
            "Reject if the page refers to a different model or if the price cannot be confirmed."
        )
    )


async def verify_output_pricing(
    evaluator: Evaluator,
    parent_node,
    extracted: PricingExtraction,
) -> None:
    """
    Build and verify the Output Token Pricing subtree.
    All children under this critical node are marked critical to satisfy critical consistency.
    """
    output_node = evaluator.add_sequential(
        id="Output_Token_Pricing",
        desc="Verify that the output token pricing for GPT-5 Pro is provided, expressed per million tokens in US dollars, and supported by an official OpenAI source URL",
        parent=parent_node,
        critical=True
    )

    # 1) Price value must be provided in the answer (per 1M tokens)
    output_price_provided = evaluator.add_custom_node(
        result=bool(extracted.output_price_per_million_usd and extracted.output_price_per_million_usd.strip()),
        id="output_price_provided",
        desc="Output token price (per 1M tokens, USD) is provided in the answer",
        parent=output_node,
        critical=True
    )

    # 2) The answer explicitly expresses the output price per million tokens in USD
    output_unit_leaf = evaluator.add_leaf(
        id="output_price_unit_check",
        desc="The output price is expressed per one million tokens in US dollars in the answer",
        parent=output_node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, the output token price is explicitly expressed per one million (1,000,000) tokens in USD.",
        node=output_unit_leaf,
        additional_instruction=(
            "Check the answer text (not the sources) to see if the output price is clearly stated per 1M tokens in USD. "
            "Accept reasonable variants such as 'per 1M tokens', 'per million tokens', 'per 1,000,000 tokens', '$X per 1M', "
            "or 'USD per million tokens'."
        )
    )

    # 3) The answer must include at least one source URL
    output_sources_present = evaluator.add_custom_node(
        result=bool(extracted.sources and len(extracted.sources) > 0),
        id="output_sources_present",
        desc="At least one source URL is provided in the answer for pricing",
        parent=output_node,
        critical=True
    )

    # 4) At least one source must be official OpenAI documentation
    output_official_source = evaluator.add_custom_node(
        result=any_official_openai_source(extracted.sources),
        id="output_official_openai_source",
        desc="At least one pricing source is an official OpenAI documentation URL",
        parent=output_node,
        critical=True
    )

    # 5) The sources support the claimed output price for GPT-5 Pro (per 1M tokens, USD)
    output_supported_leaf = evaluator.add_leaf(
        id="output_price_supported_by_sources",
        desc="The output price for GPT-5 Pro is supported by the cited official documentation",
        parent=output_node,
        critical=True
    )
    claimed_output_price = extracted.output_price_per_million_usd or ""
    await evaluator.verify(
        claim=(
            f"The official OpenAI documentation states that the output token price for GPT-5 Pro is "
            f"{claimed_output_price} per 1M tokens (USD)."
        ),
        node=output_supported_leaf,
        sources=extracted.sources,
        additional_instruction=(
            "Verify on the provided official OpenAI documentation page(s) that the listed price corresponds specifically to GPT-5 Pro's output token rate. "
            "If the documentation lists prices per 1K tokens, you may proportionally scale to per 1M tokens to judge equivalence "
            "(e.g., multiply by 1000), allowing minor rounding differences. "
            "Reject if the page refers to a different model or if the price cannot be confirmed."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the GPT-5 Pro API pricing task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Create the main rubric node under root (critical, parallel),
    # matching the JSON rubric: "GPT-5_Pro_API_Pricing"
    main_node = evaluator.add_parallel(
        id="GPT-5_Pro_API_Pricing",
        desc="Evaluate whether the solution provides complete and accurate API pricing information for OpenAI's GPT-5 Pro model",
        parent=root,
        critical=True
    )

    # Extract pricing info from the answer
    extracted_pricing = await evaluator.extract(
        prompt=prompt_extract_pricing(),
        template_class=PricingExtraction,
        extraction_name="pricing_extraction"
    )

    # Build verification subtrees for input and output pricing under the critical main node
    await verify_input_pricing(evaluator, main_node, extracted_pricing)
    await verify_output_pricing(evaluator, main_node, extracted_pricing)

    # Return summary with verification tree and extraction outcome
    return evaluator.get_summary()