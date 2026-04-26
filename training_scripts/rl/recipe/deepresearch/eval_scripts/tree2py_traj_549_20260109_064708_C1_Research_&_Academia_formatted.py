import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nature_journal_metrics_2024"
TASK_DESCRIPTION = "What is the 2024 Journal Impact Factor and the publisher of Nature journal?"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NatureMetricsExtraction(BaseModel):
    """
    Structured extraction of the required fields from the agent's answer.
    Keep numbers as strings to be robust to formatting/rounding/annotation differences.
    """
    impact_factor_2024: Optional[str] = None
    jif_source_urls: List[str] = Field(default_factory=list)

    publisher_name: Optional[str] = None
    publisher_source_urls: List[str] = Field(default_factory=list)

    # Fallback/general sources when the answer doesn't clearly separate which source supports which fact
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_nature_metrics() -> str:
    return """
    Extract, from the provided answer, the following information specifically for the journal 'Nature' (the flagship multidisciplinary journal, not other 'Nature' titles like Nature Communications, Nature Physics, etc.):

    1) impact_factor_2024: The 2024 Journal Impact Factor value exactly as stated in the answer text. Keep it as a string (do not normalize). If not provided, return null.

    2) jif_source_urls: All URLs explicitly cited that support the stated 2024 Journal Impact Factor for Nature. Include official or authoritative sources (e.g., Clarivate Journal Citation Reports pages for Nature, the Nature journal page that shows the Impact Factor). If none are provided in the answer, return an empty list.

    3) publisher_name: The publisher name of the journal 'Nature' as stated in the answer text (e.g., 'Springer Nature', 'Nature Portfolio (part of Springer Nature)', or similar). Keep it as a string exactly as in the answer. If not provided, return null.

    4) publisher_source_urls: All URLs explicitly cited that support the publisher information for the journal 'Nature' (e.g., nature.com pages, springernature.com pages). If none are provided in the answer, return an empty list.

    5) general_sources: Any additional URLs listed in the answer that are intended as sources for these facts but are not clearly tied to either the Impact Factor or the publisher. If none, return an empty list.

    Rules for URL extraction:
    - Extract only URLs explicitly present in the answer text. Do not invent URLs.
    - Accept plain URLs or markdown links, but output the resolved URL string.
    - If a URL is missing a protocol, prepend http://.
    - Return all fields in a single JSON object following the specified schema.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _merge_sources(primary: List[str], fallback: List[str]) -> List[str]:
    return _dedup_preserve_order(list(primary or []) + list(fallback or []))


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, extracted: NatureMetricsExtraction) -> None:
    """
    Build the verification nodes according to the rubric and run verifications.
    """

    # Create the critical parallel node representing the rubric root
    metrics_node = evaluator.add_parallel(
        id="Nature_Journal_Metrics",
        desc="Provides complete and accurate information about Nature journal's 2024 Journal Impact Factor and publisher, with verifiable official/public sources",
        parent=evaluator.root,
        critical=True
    )

    # Prepare source lists
    jif_sources: List[str] = _merge_sources(extracted.jif_source_urls, extracted.general_sources)
    publisher_sources: List[str] = _merge_sources(extracted.publisher_source_urls, extracted.general_sources)

    # 1) Journal Impact Factor 2024 verification (critical leaf)
    jif_leaf = evaluator.add_leaf(
        id="Journal_Impact_Factor_2024",
        desc="Provides the correct 2024 Journal Impact Factor for Nature journal (as reported in official journal metrics)",
        parent=metrics_node,
        critical=True
    )

    jif_value = extracted.impact_factor_2024 or ""
    jif_claim = f"The 2024 Journal Impact Factor for the journal 'Nature' is '{jif_value}'."

    await evaluator.verify(
        claim=jif_claim,
        node=jif_leaf,
        sources=jif_sources if jif_sources else None,
        additional_instruction=(
            "Verify this specifically for the flagship journal 'Nature'. "
            "Accept minor rounding differences. For Clarivate JCR, the '2024 Journal Impact Factor' may be labeled "
            "as 'Journal Impact Factor (2023)' because JCR 2024 release reports 2023 IF values; treat this as equivalent "
            "as long as the page is clearly for Nature and the numeric value matches. "
            "If the value is missing or blank in the answer, judge this claim as incorrect."
        ),
    )

    # 2) Publisher verification (critical leaf)
    publisher_leaf = evaluator.add_leaf(
        id="Publisher_Name",
        desc="Provides the correct publisher name for Nature journal (verifiable from official journal sources)",
        parent=metrics_node,
        critical=True
    )

    publisher_name = extracted.publisher_name or ""
    publisher_claim = f"The publisher of the journal 'Nature' is '{publisher_name}'."

    await evaluator.verify(
        claim=publisher_claim,
        node=publisher_leaf,
        sources=publisher_sources if publisher_sources else None,
        additional_instruction=(
            "Verify the publisher for the flagship journal 'Nature' (nature.com/nature). "
            "Treat 'Springer Nature' and 'Nature Portfolio (part of Springer Nature)' as consistent when a page states "
            "that Nature is published by Nature Portfolio which is part of Springer Nature. "
            "Also note 'Nature Publishing Group' historically merged into Springer Nature; if a page explicitly indicates "
            "Nature is part of Springer Nature's portfolio, consider it consistent. "
            "If the provided name is missing or blank in the answer, judge this claim as incorrect."
        ),
    )

    # 3) Public/verifiable sources presence (critical leaf implemented as custom node)
    # The two verifications above already test that the provided sources actually support each claim.
    # Here we strictly check that the answer cited at least one source for JIF and at least one for publisher.
    sources_present = bool(jif_sources) and bool(publisher_sources)
    evaluator.add_custom_node(
        result=sources_present,
        id="Public_Verifiable_Sources",
        desc="Cites/links publicly accessible official sources that support both the stated 2024 Journal Impact Factor and the stated publisher",
        parent=metrics_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for Nature's 2024 Journal Impact Factor and publisher.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_nature_metrics(),
        template_class=NatureMetricsExtraction,
        extraction_name="nature_metrics_extraction"
    )

    # Build verification tree and run checks
    await build_and_verify(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()