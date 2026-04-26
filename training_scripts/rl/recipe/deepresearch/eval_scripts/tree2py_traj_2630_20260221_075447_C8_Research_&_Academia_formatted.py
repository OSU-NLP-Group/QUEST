import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ces2026_university_innovations"
TASK_DESCRIPTION = """
Identify four distinct universities that showcased research-driven innovations at CES 2026 (held January 6-9, 2026, in Las Vegas). For each university, provide: (1) The name of at least one specific innovation, startup, product, or technology that was showcased (with its exact name, not a general description); (2) The technology domain or application area of the innovation; (3) Verifiable funding information (such as specific grant amounts, investment details, or non-dilutive funding totals); and (4) A valid URL reference to an official source (university press release, news article, or announcement) that documents the university's CES 2026 participation and the provided details. The four universities must be distinct institutions, and each must have verifiable documentation for all four required elements above.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UnivEntry(BaseModel):
    university: Optional[str] = None
    innovation_name: Optional[str] = None
    technology_domain: Optional[str] = None
    funding_info: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    items: List[UnivEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to the first four (4) distinct universities that the answer claims showcased research-driven innovations at CES 2026 (January 6–9, 2026, Las Vegas).
    For each identified university, extract the following fields exactly as stated in the answer:
    - university: The exact institution name.
    - innovation_name: The exact specific name of the startup/product/project/technology (not a generic description).
    - technology_domain: The technology domain or application area (e.g., AI, digital health, robotics, wearable sensors, energy harvesting, etc.).
    - funding_info: Verifiable funding information mentioned (e.g., grant amounts, investment rounds, program names, non-dilutive funding totals). Copy the key figures/phrases exactly as stated. If none is provided, return null.
    - source_urls: All explicit URLs that the answer associates with this university’s CES 2026 participation and details (prefer official sources such as university press releases/news pages, or reputable verified news articles). Extract the actual URLs only; if none are present, return an empty array.

    Only extract information explicitly present in the provided answer text. Do not fabricate or infer missing information.
    Ensure that URLs are valid-looking (contain http or https and a domain). If URLs are in markdown format, output the actual URL.
    Return a JSON object with a top-level field "items" which is an array of up to 4 objects with the fields above.
    """


# --------------------------------------------------------------------------- #
# Helper: Build user-friendly text for universities list                      #
# --------------------------------------------------------------------------- #
def stringify_universities(univs: List[UnivEntry]) -> str:
    names = []
    for i, it in enumerate(univs):
        nm = (it.university or "").strip()
        if nm:
            names.append(nm)
        else:
            names.append(f"University_{i+1}_UNKNOWN")
    return ", ".join(names)


# --------------------------------------------------------------------------- #
# Verification for a single university participant                            #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    item: UnivEntry,
    idx: int,
) -> None:
    """
    Build verification sub-tree for one university participant.

    Leaves (all critical) under a parallel node:
      - Source reference validity and CES 2026 documentation
      - Named innovation support
      - Technology domain support
      - Funding information support

    We verify the source reference first, then use it as a prerequisite for the other checks so that
    if sources are invalid/irrelevant, subsequent factual checks are skipped (source-grounding policy).
    """
    # Create participant node
    part_node = evaluator.add_parallel(
        id=f"University_Participant_{idx+1}",
        desc=f"{['First','Second','Third','Fourth'][idx]} university participant with complete verifiable information about their CES 2026 research showcase.",
        parent=parent_node,
        critical=False,
    )

    univ = (item.university or "").strip()
    inno = (item.innovation_name or "").strip()
    domain = (item.technology_domain or "").strip()
    funding = (item.funding_info or "").strip()
    urls = item.source_urls or []

    # 1) Source Reference (Critical)
    src_leaf = evaluator.add_leaf(
        id=f"Univ{idx+1}_Source_Reference",
        desc=f"Provides a valid URL to an official source documenting {univ or 'the university'}'s CES 2026 participation and the claimed details.",
        parent=part_node,
        critical=True,
    )

    src_claim = (
        f"At least one of these URLs is an official or reputable source (e.g., a university press release or news page, "
        f"an official announcement, or a reputable verified news article) that documents {univ or 'the university'}'s "
        f"participation at CES 2026 (held January 6–9, 2026, in Las Vegas). "
        f"It also mentions at least one of the provided details such as the named innovation "
        f"'{inno}' or the technology domain '{domain}' or funding details like '{funding}'."
    )
    await evaluator.verify(
        claim=src_claim,
        node=src_leaf,
        sources=urls,
        additional_instruction=(
            "Check the page content and/or screenshot to confirm it's relevant to CES 2026 and the specified university. "
            "If none of the URLs are valid/relevant or there is no evidence about CES 2026 participation or the claimed details, return Incorrect."
        ),
    )

    # 2) Named Innovation (Critical)
    innovation_leaf = evaluator.add_leaf(
        id=f"Univ{idx+1}_Named_Innovation",
        desc=f"Identifies a specific named innovation showcased by {univ or 'the university'} at CES 2026.",
        parent=part_node,
        critical=True,
    )
    innovation_claim = (
        f"At least one of the provided URLs explicitly states that {univ or 'the university'} showcased at CES 2026 "
        f"a specific innovation named '{inno}'. The naming should be a concrete title (not a generic description), "
        f"allowing for minor formatting or casing variations only."
    )
    await evaluator.verify(
        claim=innovation_claim,
        node=innovation_leaf,
        sources=urls,
        additional_instruction=(
            "Verify that the exact (or trivially formatted) innovation name appears on the page and is tied to CES 2026. "
            "If the innovation name is missing, generic, or not supported by the page, return Incorrect."
        ),
        extra_prerequisites=[src_leaf],
    )

    # 3) Technology Domain (Critical)
    domain_leaf = evaluator.add_leaf(
        id=f"Univ{idx+1}_Technology_Domain",
        desc=f"Documents the technology domain/application area for the innovation from {univ or 'the university'}.",
        parent=part_node,
        critical=True,
    )
    domain_claim = (
        f"At least one of the provided URLs indicates that the innovation '{inno}' is in the technology domain or "
        f"application area '{domain}', or clearly describes functionality that aligns with this domain."
    )
    await evaluator.verify(
        claim=domain_claim,
        node=domain_leaf,
        sources=urls,
        additional_instruction=(
            "Look for explicit mentions of the domain (e.g., AI, digital health, robotics, etc.) or clear descriptions "
            "that match the provided domain. If the domain is not supported by the page, return Incorrect."
        ),
        extra_prerequisites=[src_leaf],
    )

    # 4) Funding Information (Critical)
    funding_leaf = evaluator.add_leaf(
        id=f"Univ{idx+1}_Funding_Information",
        desc=f"Provides verifiable funding information related to the innovation/team from {univ or 'the university'}.",
        parent=part_node,
        critical=True,
    )
    funding_claim = (
        f"At least one of the provided URLs explicitly mentions the funding information '{funding}' "
        f"(e.g., specific grant amounts, investment details, non-dilutive totals, or named grant programs) "
        f"associated with the innovation or team from {univ or 'the university'}."
    )
    await evaluator.verify(
        claim=funding_claim,
        node=funding_leaf,
        sources=urls,
        additional_instruction=(
            "Verify that the funding figures or program names are explicitly present on the page and tied to this "
            "innovation/team. If the funding info is missing or unsupported, return Incorrect."
        ),
        extra_prerequisites=[src_leaf],
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
    Evaluate an answer for the CES 2026 university innovations task.
    """
    # Initialize evaluator with root as parallel aggregation per rubric
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

    # 1) Extract up to four universities with required fields
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Normalize to exactly 4 entries (pad with empty if fewer, trim if more)
    items = (extracted.items or [])[:4]
    while len(items) < 4:
        items.append(UnivEntry())

    # Record some custom info for debugging/visibility
    evaluator.add_custom_info(
        {
            "count_provided": len(extracted.items or []),
            "used_first_n": 4,
            "university_names": [it.university for it in items],
        },
        info_type="extraction_summary",
    )

    # 2) Global check: Geographic diversity (Critical leaf per rubric)
    geo_leaf = evaluator.add_leaf(
        id="Geographic_Diversity",
        desc="The four identified universities collectively represent at least two different countries or regions.",
        parent=root,
        critical=True,
    )
    # Build claim listing the universities; instruct judge to use general world knowledge
    universities_str = stringify_universities(items)
    geo_claim = (
        f"Among these universities: {universities_str}, there are at least two different countries or regions represented."
    )
    await evaluator.verify(
        claim=geo_claim,
        node=geo_leaf,
        additional_instruction=(
            "Use general world knowledge to determine the primary country/region of each named university. "
            "If all are from the same country/region, the claim is Incorrect. If at least two different countries/regions "
            "are represented, the claim is Correct."
        ),
    )

    # 3) Per-university participant checks (four participants, non-critical at parent level)
    for idx in range(4):
        await verify_university(
            evaluator=evaluator,
            parent_node=root,
            item=items[idx],
            idx=idx,
        )

    # 4) Return structured summary
    return evaluator.get_summary()