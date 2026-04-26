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
TASK_ID = "tx_prop_mgmt_nrhp_leed_gold"
TASK_DESCRIPTION = """
Identify a property management company operating in Texas that manages at least one building which is both listed on the National Register of Historic Places and has achieved LEED Gold certification. Provide the name of the property management company and the name or address of the building that meets both criteria.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AnswerExtraction(BaseModel):
    # Company-related
    company_name: Optional[str] = None
    urls_company_texas_ops: List[str] = Field(default_factory=list)   # Evidence it operates in Texas
    urls_company_license: List[str] = Field(default_factory=list)     # TREC license lookup page, TREC rules, etc.

    # Building-related (select a single building that the answer claims meets both NRHP + LEED Gold)
    building_identifier: Optional[str] = None  # Name and/or address sufficient to identify
    urls_building_management: List[str] = Field(default_factory=list)  # Evidence the company manages the building
    urls_building_nrhp: List[str] = Field(default_factory=list)        # Evidence of NRHP listing
    urls_building_leed: List[str] = Field(default_factory=list)        # Evidence of LEED Gold certification
    urls_building_size: List[str] = Field(default_factory=list)        # Evidence of >= 1,000 sq ft (can reuse LEED or other specs)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_core() -> str:
    return """
    Extract exactly the following fields from the answer. Select one company and one building that the answer claims satisfy the task.

    COMPANY FIELDS:
    - company_name: The name of the property management company (string).
    - urls_company_texas_ops: Array of URLs that the answer cites as evidence that the company operates in Texas (e.g., Texas office page, Texas services page, portfolio page with Texas properties).
    - urls_company_license: Array of URLs the answer cites about Texas real estate broker licensing relevant to this company or to the requirement (e.g., TREC license lookup result for the company/broker, or TREC policy pages).

    BUILDING FIELDS (CHOOSE THE SINGLE BUILDING THE ANSWER ASSERTS MEETS BOTH NRHP + LEED GOLD):
    - building_identifier: The building name and/or address sufficient to uniquely identify it (string).
    - urls_building_management: Array of URLs cited that show the identified company manages this building (e.g., company site property page, building owner's page stating management, credible third-party listing).
    - urls_building_nrhp: Array of URLs cited that show the building is listed on the National Register of Historic Places (NRHP). Accept NPS NRHP, state registers referencing NRHP, Wikipedia, or reputable preservation databases.
    - urls_building_leed: Array of URLs cited that show the building has achieved LEED Gold certification. Accept USGBC/GBCI entries, owner/operator pages, credible press releases.
    - urls_building_size: Array of URLs cited that show the building has a gross floor area of at least 1,000 square feet. If the same LEED/owner page has area information, include it here too.

    RULES:
    - Only extract URLs that are explicitly present in the answer text (including markdown links). Do not invent or infer URLs.
    - If multiple buildings/companies are mentioned, extract only the first one that the answer claims meets the criteria. If unclear, extract the first property described.
    - If any field is missing, set it to null for strings or [] for arrays.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_company_checks(evaluator: Evaluator, parent_node, ex: AnswerExtraction) -> None:
    """
    Build and verify the 'Company' subtree:
    - Company_Name_Provided (custom existence)
    - Operates_In_Texas (verify with URLs)
    - Texas_Broker_License (verify with URLs)
    """
    company_node = evaluator.add_parallel(
        id="company",
        desc="Identify the property management company and verify it meets Texas-related constraints.",
        parent=parent_node,
        critical=True
    )

    # 1) Company name provided (existence)
    company_name_provided = evaluator.add_custom_node(
        result=bool(ex.company_name and ex.company_name.strip()),
        id="company_name_provided",
        desc="The answer provides the name of the property management company.",
        parent=company_node,
        critical=True
    )

    # 2) Operates in Texas
    operates_in_tx_node = evaluator.add_leaf(
        id="operates_in_texas",
        desc="The company operates in Texas.",
        parent=company_node,
        critical=True
    )
    # Sources: primarily company Texas ops URLs; allow fallback to building management URLs if they indicate Texas presence
    sources_operates = _dedupe_preserve_order(
        (ex.urls_company_texas_ops or []) + (ex.urls_building_management or [])
    )
    company_name = ex.company_name or "the company"
    claim_operates = f"The company {company_name} operates in Texas."
    await evaluator.verify(
        claim=claim_operates,
        node=operates_in_tx_node,
        sources=sources_operates,
        additional_instruction=(
            "Use the sources to confirm the company operates in Texas (e.g., Texas office, Texas services page, "
            "portfolio or property pages in Texas). If a source clearly shows the company managing or servicing a "
            "property in a Texas city, that counts as operating in Texas. Be flexible about naming variants."
        )
    )

    # 3) Texas Broker License (hold or requirement)
    license_node = evaluator.add_leaf(
        id="texas_broker_license",
        desc="The company holds or is required to hold a valid Texas real estate broker's license for property management operations (per TREC rules).",
        parent=company_node,
        critical=True
    )
    sources_license = _dedupe_preserve_order(ex.urls_company_license or [])
    claim_license = (
        f"The company {company_name} either holds a valid Texas real estate broker's license OR, under Texas TREC rules, "
        f"providing property management services in Texas requires a real estate broker's license."
    )
    await evaluator.verify(
        claim=claim_license,
        node=license_node,
        sources=sources_license,
        additional_instruction=(
            "This check should pass if any provided source shows: (a) an active Texas real estate broker license for "
            "the company or its responsible broker via a TREC license lookup; OR (b) an official Texas Real Estate "
            "Commission (TREC) rule or policy stating property management requires a broker's license in Texas. "
            "Do not fail due to missing company-specific license evidence if a TREC policy page clearly establishes the requirement."
        )
    )


async def build_building_checks(evaluator: Evaluator, parent_node, ex: AnswerExtraction) -> None:
    """
    Build and verify the 'Building' subtree:
    - Building_Identifier_Provided (custom existence)
    - Managed_By_Company (verify with URLs)
    - National_Register_Listing (verify with URLs)
    - LEED_Gold_Certification (verify with URLs)
    - LEED_Minimum_Size (verify with URLs)
    """
    building_node = evaluator.add_parallel(
        id="building",
        desc="Identify a building managed by the company and verify it satisfies the building constraints.",
        parent=parent_node,
        critical=True
    )

    # 1) Building identifier provided (existence)
    building_identifier_exists = evaluator.add_custom_node(
        result=bool(ex.building_identifier and ex.building_identifier.strip()),
        id="building_identifier_provided",
        desc="The answer provides the building name and/or address sufficient to identify the building.",
        parent=building_node,
        critical=True
    )

    # 2) Managed by company
    managed_node = evaluator.add_leaf(
        id="managed_by_company",
        desc="The provided building is managed by the identified property management company.",
        parent=building_node,
        critical=True
    )
    b_ident = ex.building_identifier or "the building"
    company_name = ex.company_name or "the company"
    claim_managed = f"{b_ident} is managed by {company_name}."
    await evaluator.verify(
        claim=claim_managed,
        node=managed_node,
        sources=_dedupe_preserve_order(ex.urls_building_management or []),
        additional_instruction=(
            "Accept evidence from a company portfolio/property page, the building owner's site, or a credible "
            "third-party listing explicitly stating that the building is managed by the named company. "
            "Look for phrasing such as 'managed by', 'property management by', or similar."
        )
    )

    # 3) NRHP listing
    nrhp_node = evaluator.add_leaf(
        id="nrhp_listing",
        desc="The building is listed on the National Register of Historic Places (NRHP).",
        parent=building_node,
        critical=True
    )
    claim_nrhp = f"{b_ident} is listed on the National Register of Historic Places."
    await evaluator.verify(
        claim=claim_nrhp,
        node=nrhp_node,
        sources=_dedupe_preserve_order(ex.urls_building_nrhp or []),
        additional_instruction=(
            "Confirm explicit mention that the building is listed on the National Register of Historic Places "
            "(NRHP). Accept synonyms such as 'on the National Register' or 'NRHP-listed'. Valid sources include "
            "the NPS NRHP database, state historic preservation listings referencing NRHP, Wikipedia, or reputable "
            "preservation databases."
        )
    )

    # 4) LEED Gold certification
    leed_gold_node = evaluator.add_leaf(
        id="leed_gold_certification",
        desc="The building has achieved LEED Gold certification.",
        parent=building_node,
        critical=True
    )
    claim_leed_gold = f"{b_ident} has achieved LEED Gold certification."
    await evaluator.verify(
        claim=claim_leed_gold,
        node=leed_gold_node,
        sources=_dedupe_preserve_order(ex.urls_building_leed or []),
        additional_instruction=(
            "Confirm that the building achieved LEED Gold. Accept expressions such as 'LEED Gold', 'LEED® Gold', "
            "'LEED-NC Gold', 'LEED v4 Gold', etc. Valid sources include USGBC/GBCI listings, the building owner/operator site, "
            "or credible press releases and case studies."
        )
    )

    # 5) LEED minimum size (>= 1,000 sq ft)
    leed_size_node = evaluator.add_leaf(
        id="leed_minimum_size",
        desc="The building meets the stated LEED minimum size requirement of at least 1,000 square feet of gross floor area (as given in the constraints).",
        parent=building_node,
        critical=True
    )
    # We combine multiple potential sources: size-specific URLs, plus LEED/owner pages if they include area
    size_sources = _dedupe_preserve_order((ex.urls_building_size or []) + (ex.urls_building_leed or []) + (ex.urls_building_management or []))
    claim_size = f"{b_ident} has a gross floor area of at least 1,000 square feet."
    await evaluator.verify(
        claim=claim_size,
        node=leed_size_node,
        sources=size_sources,
        additional_instruction=(
            "Verify that the building's gross floor area (GFA) is >= 1,000 square feet. Accept any explicit area "
            "statement that clearly exceeds 1,000 sq ft (e.g., '120,000 sq ft', '1.2 million sq ft', or metric equivalents "
            ">= ~93 m²). If a provided LEED/owner/portfolio page includes area data above this threshold, that suffices."
        )
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
    model: str = "o4-mini"
) -> Dict:
    """
    Entry point for evaluating an answer for the Texas property management + NRHP + LEED Gold task.
    """
    # Initialize evaluator with a parallel root, then create a critical sequential task root beneath it
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root created non-critical by framework
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

    # Add a critical sequential node to reflect the rubric's Root being critical + sequential
    task_root = evaluator.add_sequential(
        id="task_root",
        desc="Identify a Texas-operating property management company and a managed building that is both NRHP-listed and LEED Gold certified, and provide the required names/details.",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_core(),
        template_class=AnswerExtraction,
        extraction_name="selected_company_and_building"
    )

    # Build and verify Company subtree (first in sequence)
    await build_company_checks(evaluator, task_root, extraction)

    # Build and verify Building subtree (second in sequence; skipped automatically if Company fails)
    await build_building_checks(evaluator, task_root, extraction)

    # Return structured result
    return evaluator.get_summary()