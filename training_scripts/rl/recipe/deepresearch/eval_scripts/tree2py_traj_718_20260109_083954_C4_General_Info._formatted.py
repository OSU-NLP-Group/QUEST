import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "tsa_digital_id_eval"
TASK_DESCRIPTION = (
    "According to official TSA sources, provide the following information about the TSA Digital ID feature: "
    "(1) the month and year it was announced or launched, "
    "(2) the minimum number of airports where it is accepted, "
    "(3) the total number of participating U.S. states and territories, "
    "(4) the compliance requirement for the underlying driver's license or ID, "
    "(5) at least three digital wallet platforms or apps that support the feature, and "
    "(6) the primary purpose or use case for Digital ID at airports. Include reference URLs from official sources."
)


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class TSADigitalIDExtraction(BaseModel):
    # 1) Launch
    launch_month_year: Optional[str] = None  # e.g., "March 2025" or "May 2025"
    # 2) Airports acceptance minimum (text form, can include words like "250+")
    airports_minimum: Optional[str] = None
    # 3) Participating jurisdictions (text form, e.g., "22 states and 1 territory")
    jurisdictions_total: Optional[str] = None
    # 4) Compliance requirement (text form, e.g., "REAL ID-compliant or Enhanced ID")
    compliance_requirement: Optional[str] = None
    # 5) Supported platforms/apps
    wallet_platforms: List[str] = Field(default_factory=list)
    # 6) Primary purpose/use case
    primary_use_case: Optional[str] = None
    # Voluntary usage & carry physical ID statements (if stated in answer)
    voluntary_use_statement: Optional[str] = None
    carry_physical_id_statement: Optional[str] = None
    # All URLs cited in the answer
    sources: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_tsa_digital_id() -> str:
    return """
Extract the following fields EXACTLY as stated in the answer:

1) launch_month_year: The month and year the TSA Digital ID was officially announced or launched (e.g., "March 2025"). If not stated in month-year form, extract the closest month-year-like text provided. If missing, return null.

2) airports_minimum: The minimum number of U.S. airports where TSA Digital ID is accepted (as stated in the answer). This can be a number or a textual phrase like "250+", "more than 250", "over 250", etc. If missing, return null.

3) jurisdictions_total: The total number of participating U.S. states and territories (as stated in the answer). Return the stated total as a string (e.g., "23" or "22 states and 1 territory"). If missing, return null.

4) compliance_requirement: The compliance requirement for eligible Digital IDs (e.g., "REAL ID-compliant driver's licenses or Enhanced Driver's Licenses/IDs"). If missing, return null.

5) wallet_platforms: A list of at least three digital wallet platforms or apps mentioned as supporting TSA Digital ID (e.g., "Apple Wallet", "Google Wallet", "Samsung Wallet", "State Mobile ID app", "[State] Mobile ID"). Extract all listed names. If none, return an empty list.

6) primary_use_case: The primary purpose/use case at airports, as stated in the answer (e.g., "identity verification at TSA security checkpoints for domestic air travel"). If missing, return null.

7) voluntary_use_statement: If the answer states that using Digital ID is voluntary, extract the sentence/phrase. Else return null.

8) carry_physical_id_statement: If the answer states that travelers should still carry a physical ID, extract the sentence/phrase. Else return null.

9) sources: Extract ALL source URLs mentioned anywhere in the answer. Include plain URLs and URLs embedded in markdown links. Return only valid URLs. Do not fabricate URLs. If a URL is missing a protocol, prepend http://. Include URLs from all domains (we will filter for TSA later). If none, return an empty list.
    """.strip()


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def filter_tsa_urls(urls: List[str]) -> List[str]:
    out = []
    seen = set()
    for u in urls:
        if not isinstance(u, str):
            continue
        ul = u.strip()
        if not ul:
            continue
        low = ul.lower()
        if "tsa.gov" in low:
            # normalize duplicates by lowercase
            key = low
            if key not in seen:
                seen.add(key)
                out.append(ul)
    return out


def string_has_digits(s: Optional[str]) -> bool:
    if not s:
        return False
    return any(ch.isdigit() for ch in s)


def normalize_platforms(platforms: List[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for p in platforms:
        if not isinstance(p, str):
            continue
        name = p.strip()
        if not name:
            continue
        key = name.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(name)
    return cleaned


def includes_allowed_platform(platforms: List[str]) -> bool:
    """
    At least one platform should be among Apple Wallet, Google Wallet, Samsung Wallet,
    or clearly a state-specific mobile ID app (e.g., containing 'mobile id', 'mDL').
    """
    allowed_keywords = ["apple wallet", "google wallet", "samsung wallet"]
    for name in platforms:
        low = name.lower()
        if any(k in low for k in allowed_keywords):
            return True
        # state-specific mobile ID app
        if ("mobile id" in low) or ("mdl" in low) or ("mobile driver" in low) or ("state id app" in low):
            return True
    return False


def quoted_join(items: List[str]) -> str:
    # Join items as quoted comma-separated list for claims
    return ", ".join([f"'{x}'" for x in items])


# -----------------------------------------------------------------------------
# Verification subtrees
# -----------------------------------------------------------------------------
async def add_official_tsa_citations_check(
    evaluator: Evaluator,
    parent_node,
    tsa_urls: List[str],
) -> None:
    evaluator.add_custom_node(
        result=(len(tsa_urls) > 0),
        id="official_tsa_citations_present",
        desc="Includes reference URL(s) from official TSA sources (tsa.gov) that support the provided claims.",
        parent=parent_node,
        critical=True,
    )


async def add_launch_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: TSADigitalIDExtraction,
    tsa_urls: List[str],
) -> None:
    node = evaluator.add_sequential(
        id="launch_date_group",
        desc="Provides the month and year Digital ID was officially announced/launched, and corresponds to 2025.",
        parent=parent_node,
        critical=True,
    )

    # Existence of a month+year field
    evaluator.add_custom_node(
        result=bool(extracted.launch_month_year and extracted.launch_month_year.strip()),
        id="launch_date_provided",
        desc="Launch/announcement month and year is provided in the answer.",
        parent=node,
        critical=True,
    )

    # Must correspond to 2025 (requirement)
    evaluator.add_custom_node(
        result=("2025" in (extracted.launch_month_year or "")),
        id="launch_year_is_2025",
        desc="The provided launch/announcement month-year corresponds to year 2025.",
        parent=node,
        critical=True,
    )

    # Supported by TSA sources
    leaf = evaluator.add_leaf(
        id="launch_date_supported_by_tsa",
        desc="The provided launch/announcement month-year is supported by the official TSA source(s).",
        parent=node,
        critical=True,
    )
    claim = (
        f"TSA officially announced or launched the TSA Digital ID in {extracted.launch_month_year}."
        if extracted.launch_month_year else
        "TSA officially announced or launched the TSA Digital ID in the stated month and year."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=tsa_urls,
        additional_instruction=(
            "Verify from the TSA page(s) that the stated month and year match the official announcement or launch. "
            "Treat 'announced', 'introduced', or 'launched' as equivalent for this purpose. "
            "Use only the provided TSA source(s). If the TSA sources do not state the month/year, mark as not supported."
        ),
    )


async def add_airports_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: TSADigitalIDExtraction,
    tsa_urls: List[str],
) -> None:
    node = evaluator.add_sequential(
        id="airports_acceptance_group",
        desc="Provides the minimum number of airports and indicates acceptance is at more than 250 U.S. airports.",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(extracted.airports_minimum and string_has_digits(extracted.airports_minimum)),
        id="airports_minimum_provided",
        desc="The minimum number of U.S. airports accepting Digital ID is provided in the answer.",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="airports_over_250_supported_by_tsa",
        desc="Digital ID acceptance is at more than 250 U.S. airports (supported by TSA sources).",
        parent=node,
        critical=True,
    )
    claim = "TSA Digital ID is accepted at more than 250 U.S. airports."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=tsa_urls,
        additional_instruction=(
            "Look for wording like 'more than 250 airports', '250+ airports', or similar on the TSA page(s). "
            "Only consider the provided TSA source(s)."
        ),
    )


async def add_jurisdictions_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: TSADigitalIDExtraction,
    tsa_urls: List[str],
) -> None:
    node = evaluator.add_sequential(
        id="participating_jurisdictions_group",
        desc="Provides total participating U.S. states/territories and bases the count on TSA’s official listing.",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(extracted.jurisdictions_total and string_has_digits(extracted.jurisdictions_total)),
        id="jurisdictions_total_provided",
        desc="The total number of participating U.S. states and territories is provided in the answer.",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="jurisdictions_based_on_tsa_listing",
        desc="The total is based on TSA’s official participating states/territories listing (supported by TSA sources).",
        parent=node,
        critical=True,
    )
    claim = (
        "The count of participating U.S. states and territories for TSA Digital ID is based on the official TSA listing page."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=tsa_urls,
        additional_instruction=(
            "Confirm that the TSA page(s) provide an official listing of participating states/territories for Digital ID. "
            "You do not need to verify the exact numeric count, but the listing must be present on the TSA source(s)."
        ),
    )


async def add_compliance_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: TSADigitalIDExtraction,
    tsa_urls: List[str],
) -> None:
    node = evaluator.add_sequential(
        id="underlying_id_compliance_group",
        desc="States that eligible Digital IDs are based on REAL ID-compliant or Enhanced Driver’s Licenses/IDs.",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(extracted.compliance_requirement and extracted.compliance_requirement.strip()),
        id="compliance_requirement_provided",
        desc="The compliance requirement for eligible Digital IDs is provided in the answer.",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="compliance_supported_by_tsa",
        desc="Compliance requirement (REAL ID-compliant or Enhanced ID) is supported by TSA sources.",
        parent=node,
        critical=True,
    )
    claim = (
        "Eligible TSA Digital ID credentials are based on REAL ID-compliant driver’s licenses/IDs or Enhanced Driver’s Licenses/IDs."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=tsa_urls,
        additional_instruction=(
            "Check the TSA page(s) for statements that accepted Digital IDs originate from REAL ID-compliant IDs or Enhanced IDs (EDL/EID)."
        ),
    )


async def add_wallets_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: TSADigitalIDExtraction,
    tsa_urls: List[str],
) -> None:
    node = evaluator.add_sequential(
        id="supported_wallets_group",
        desc="Lists at least three supporting digital wallet platforms/apps with at least one allowed platform.",
        parent=parent_node,
        critical=True,
    )

    platforms = normalize_platforms(extracted.wallet_platforms)
    evaluator.add_custom_node(
        result=(len(platforms) >= 3),
        id="wallets_at_least_three",
        desc="At least three digital wallet platforms/apps are listed in the answer.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=includes_allowed_platform(platforms),
        id="wallets_includes_allowed",
        desc="The list includes at least one allowed platform: Apple Wallet, Google Wallet, Samsung Wallet, or a state-specific mobile ID app.",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="wallets_supported_by_tsa",
        desc="The listed digital wallet platforms/apps are supported for TSA Digital ID (as per TSA sources).",
        parent=node,
        critical=True,
    )

    if platforms:
        claim = (
            f"The following digital wallet platforms/apps support TSA Digital ID: {quoted_join(platforms)}."
        )
    else:
        claim = "The listed digital wallet platforms/apps support TSA Digital ID."

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=tsa_urls,
        additional_instruction=(
            "Check the TSA page(s) for mentions of supported digital wallets or mobile ID apps (e.g., Apple Wallet, Google Wallet, Samsung Wallet, or state mobile ID apps). "
            "It is acceptable if the TSA page provides representative examples rather than an exhaustive list, as long as the listed platforms are supported."
        ),
    )


async def add_primary_purpose_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: TSADigitalIDExtraction,
    tsa_urls: List[str],
) -> None:
    node = evaluator.add_sequential(
        id="primary_purpose_group",
        desc="Explains the primary purpose/use case at airports: identity verification at TSA security checkpoints for domestic travel.",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(extracted.primary_use_case and extracted.primary_use_case.strip()),
        id="primary_purpose_provided",
        desc="Primary purpose/use case is provided in the answer.",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="primary_purpose_supported_by_tsa",
        desc="Primary purpose/use case is supported by TSA sources.",
        parent=node,
        critical=True,
    )
    claim = (
        "TSA Digital ID is used for identity verification at TSA security checkpoints for domestic air travel."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=tsa_urls,
        additional_instruction=(
            "Confirm that the TSA page(s) describe Digital ID as being used at security checkpoints for identity verification for domestic flights."
        ),
    )


async def add_voluntary_and_physical_id_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: TSADigitalIDExtraction,
    tsa_urls: List[str],
) -> None:
    node = evaluator.add_sequential(
        id="voluntary_physical_id_group",
        desc="Use is voluntary, and travelers should still carry a physical ID.",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(extracted.voluntary_use_statement and extracted.voluntary_use_statement.strip()),
        id="voluntary_use_stated_in_answer",
        desc="The answer states that using Digital ID is voluntary.",
        parent=node,
        critical=True,
    )

    leaf1 = evaluator.add_leaf(
        id="voluntary_use_supported_by_tsa",
        desc="Use of TSA Digital ID is voluntary (supported by TSA sources).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Using TSA Digital ID is voluntary.",
        node=leaf1,
        sources=tsa_urls,
        additional_instruction="Verify that the TSA page(s) explicitly state that use of Digital ID is voluntary.",
    )

    evaluator.add_custom_node(
        result=bool(extracted.carry_physical_id_statement and extracted.carry_physical_id_statement.strip()),
        id="carry_physical_id_stated_in_answer",
        desc="The answer states that travelers should carry a physical ID.",
        parent=node,
        critical=True,
    )

    leaf2 = evaluator.add_leaf(
        id="carry_physical_id_supported_by_tsa",
        desc="Travelers should still carry a physical ID (supported by TSA sources).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Travelers should carry a physical ID even when using TSA Digital ID.",
        node=leaf2,
        sources=tsa_urls,
        additional_instruction="Verify that the TSA page(s) advise travelers to still carry a physical ID.",
    )


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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
        default_model=model,
    )

    # Extraction
    extracted: TSADigitalIDExtraction = await evaluator.extract(
        prompt=prompt_extract_tsa_digital_id(),
        template_class=TSADigitalIDExtraction,
        extraction_name="tsa_digital_id_extraction",
    )

    # Filter URLs to tsa.gov
    tsa_urls = filter_tsa_urls(extracted.sources)

    # Record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "all_extracted_sources": extracted.sources,
            "tsa_sources_only": tsa_urls,
            "wallet_platforms_extracted": extracted.wallet_platforms,
        },
        info_type="debug_info",
        info_name="extraction_postprocess",
    )

    # Build main critical node (as per rubric)
    tsa_info_node = evaluator.add_parallel(
        id="TSA_Digital_ID_Information",
        desc="Verify all required information about TSA Digital ID is provided and satisfies constraints, with citations from official TSA sources.",
        parent=root,
        critical=True,
    )

    # Add "Official TSA Citations" first so that it becomes a prerequisite for other checks
    await add_official_tsa_citations_check(evaluator, tsa_info_node, tsa_urls)

    # Add other groups (each critical as required by rubric)
    await add_launch_checks(evaluator, tsa_info_node, extracted, tsa_urls)
    await add_airports_checks(evaluator, tsa_info_node, extracted, tsa_urls)
    await add_jurisdictions_checks(evaluator, tsa_info_node, extracted, tsa_urls)
    await add_compliance_checks(evaluator, tsa_info_node, extracted, tsa_urls)
    await add_wallets_checks(evaluator, tsa_info_node, extracted, tsa_urls)
    await add_primary_purpose_checks(evaluator, tsa_info_node, extracted, tsa_urls)
    await add_voluntary_and_physical_id_checks(evaluator, tsa_info_node, extracted, tsa_urls)

    # Optionally add a concise summary of the provided fields (for visibility in results)
    evaluator.add_custom_info(
        info={
            "launch_month_year": extracted.launch_month_year,
            "airports_minimum": extracted.airports_minimum,
            "jurisdictions_total": extracted.jurisdictions_total,
            "compliance_requirement": extracted.compliance_requirement,
            "primary_use_case": extracted.primary_use_case,
            "wallet_platforms": normalize_platforms(extracted.wallet_platforms),
            "voluntary_use_statement": extracted.voluntary_use_statement,
            "carry_physical_id_statement": extracted.carry_physical_id_statement,
        },
        info_type="extracted_fields_preview",
    )

    # Return evaluation summary
    return evaluator.get_summary()