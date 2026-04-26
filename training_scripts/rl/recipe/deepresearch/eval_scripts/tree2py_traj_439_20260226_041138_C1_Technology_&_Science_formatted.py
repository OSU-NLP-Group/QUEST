import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "oneplus13_verizon_mmwave"
TASK_DESCRIPTION = "Verify whether the OnePlus 13 supports Verizon's 5G Ultra Wideband mmWave bands (n260 and n261)."

BAND_INFO = {
    "n260": "37–40 GHz",
    "n261": "27.5–28.35 GHz",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BandSupportExtraction(BaseModel):
    """
    Extract band-related claims and all URLs the answer uses as evidence for
    OnePlus 13 network band support.
    """
    n260_claim: Optional[str] = None  # "supports" | "not_supported" | null if not stated
    n261_claim: Optional[str] = None  # "supports" | "not_supported" | null if not stated
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_band_support() -> str:
    return """
    Your task is to extract from the answer:
    1) Whether the answer explicitly claims that the OnePlus 13 supports 5G band n260.
       - Use the field 'n260_claim' with one of the following values:
         • "supports" if the answer clearly asserts support for band n260.
         • "not_supported" if the answer clearly asserts that band n260 is not supported.
         • null if the answer does not explicitly state either support or lack of support for n260.
    2) Whether the answer explicitly claims that the OnePlus 13 supports 5G band n261.
       - Use the field 'n261_claim' with one of the following values:
         • "supports" if the answer clearly asserts support for band n261.
         • "not_supported" if the answer clearly asserts that band n261 is not supported.
         • null if the answer does not explicitly state either support or lack of support for n261.
    3) Collect all URLs that the answer presents as evidence for OnePlus 13 network band support
       or connectivity specifications (e.g., official spec pages, carrier compatibility pages, GSMArena specs).
       - Put them into 'source_urls' as a list of URLs.
       - Only include actual URLs explicitly present in the answer (plain or markdown links).
       - If a link lacks a protocol, prepend 'http://'.
       - Do not invent or infer URLs.

    Return a JSON object with fields: n260_claim, n261_claim, source_urls.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _build_additional_instruction_for_band(band_code: str, ghz_desc: str, missing_sources: bool) -> str:
    base_rules = [
        f"Evaluate whether the provided webpages explicitly support that the OnePlus 13 supports 5G NR band {band_code} ({ghz_desc}).",
        "Pass only if at least one page clearly lists this band code (e.g., 'n260', 'n261') for the OnePlus 13 in its supported 5G NR bands.",
        "Accept reasonable synonyms or formatting variants such as 'mmWave 39 GHz (n260)' or 'mmWave 28 GHz (n261)'.",
        "The page must be about the OnePlus 13 (avoid mixing with similarly named models like OnePlus 13R, OnePlus 12, etc.).",
        "If multiple regional variants exist, acceptance requires that at least one documented OnePlus 13 variant supports the specified band (preferably a US or Verizon-compatible variant for this Verizon UW context).",
        "If the URLs are irrelevant (wrong device), inaccessible, or the band is absent/not mentioned in supported bands, conclude 'not supported' and return Incorrect.",
    ]
    if missing_sources:
        base_rules.append(
            "Important: The answer did not provide any source URLs for this verification. "
            "Because this is a factual, web-grounded claim, you must treat it as not supported and return Incorrect."
        )
    return " ".join(base_rules)


async def _verify_band_support(
    evaluator: Evaluator,
    parent_node,
    band_code: str,
    ghz_desc: str,
    sources: List[str],
) -> None:
    """
    Create a critical leaf node that verifies the OnePlus 13 supports a given 5G band,
    grounded by the URLs extracted from the answer.
    """
    leaf = evaluator.add_leaf(
        id=f"Band_{band_code}_Support",
        desc=f"The OnePlus 13 must support 5G band {band_code} ({ghz_desc})",
        parent=parent_node,
        critical=True,
    )

    claim = f"The OnePlus 13 supports 5G band {band_code} ({ghz_desc})."
    srcs = sources if sources else None
    add_ins = _build_additional_instruction_for_band(band_code, ghz_desc, missing_sources=(not sources))

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction=add_ins,
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate whether the answer correctly establishes that the OnePlus 13 supports Verizon UW mmWave bands n260 and n261,
    using only the URLs cited in the answer as evidence.
    """
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
        default_model=model,
    )

    # Extract band-related claims and evidence URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_band_support(),
        template_class=BandSupportExtraction,
        extraction_name="band_support_extraction",
    )

    # Grouping node representing the rubric root (critical, parallel aggregation)
    mmwave_group = evaluator.add_parallel(
        id="mmWave_Band_Support_Verification",
        desc="Verify whether the OnePlus 13 supports Verizon's 5G Ultra Wideband mmWave bands (n260 and n261)",
        parent=root,
        critical=True,
    )

    # Prepare sources for verifications (shared set for both bands)
    sources = extracted.source_urls if extracted and extracted.source_urls else []

    # Run critical checks for n260 and n261 support
    await _verify_band_support(
        evaluator=evaluator,
        parent_node=mmwave_group,
        band_code="n260",
        ghz_desc=BAND_INFO["n260"],
        sources=sources,
    )
    await _verify_band_support(
        evaluator=evaluator,
        parent_node=mmwave_group,
        band_code="n261",
        ghz_desc=BAND_INFO["n261"],
        sources=sources,
    )

    return evaluator.get_summary()