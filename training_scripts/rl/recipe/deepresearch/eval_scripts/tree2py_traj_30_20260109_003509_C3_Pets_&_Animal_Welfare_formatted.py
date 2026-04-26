import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "usda_favn_japan_lab"
TASK_DESCRIPTION = """
Identify a USDA-approved laboratory in the United States that performs FAVN (Fluorescent Antibody Virus Neutralization) rabies antibody titer tests for pets traveling to Japan. Provide the laboratory's official name, complete mailing address for submitting blood samples (including street address, city, state, and ZIP code), and the current FAVN test processing timeframe in calendar days.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LabAddress(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None


class LabSources(BaseModel):
    approvals_urls: List[str] = Field(default_factory=list)
    address_urls: List[str] = Field(default_factory=list)
    timeframe_urls: List[str] = Field(default_factory=list)
    lab_website_url: Optional[str] = None


class LabExtraction(BaseModel):
    official_name: Optional[str] = None
    address: LabAddress = Field(default_factory=LabAddress)
    processing_timeframe_days: Optional[str] = None  # Keep as free-form string; expect number or numeric range
    sources: LabSources = Field(default_factory=LabSources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_lab_info() -> str:
    return """
    You must extract structured information (exactly as stated in the answer) about ONE U.S. laboratory that performs FAVN rabies antibody titer testing for pets traveling to Japan.

    Extract the following fields:
    - official_name: The laboratory's official name (as written in the answer).
    - address.street: Street address for submitting blood samples.
    - address.city: City.
    - address.state: State (abbreviation or full state name).
    - address.zip_code: ZIP code (5-digit or ZIP+4 if provided).
    - processing_timeframe_days: The FAVN test processing timeframe as stated in the answer, preferably expressed in calendar days. If the answer uses business days, still extract exactly what is written (e.g., "7-10 business days").
    - sources.approvals_urls: All URLs that support the lab's USDA/Japan approval/acceptance status for FAVN testing (e.g., USDA APHIS pages, Japan MAFF/AQS pages, other official listings). Include only URLs that are explicitly present in the answer.
    - sources.address_urls: All URLs that support the provided mailing address (official sources or the lab’s own official website). Include only URLs explicitly present in the answer.
    - sources.timeframe_urls: All URLs that support the stated processing timeframe (official sources or the lab’s own official website). Include only URLs explicitly present in the answer.
    - sources.lab_website_url: The lab’s primary/official website URL if explicitly provided in the answer.

    Rules:
    1) Return only URLs explicitly present in the answer. Do not invent or infer URLs.
    2) If any field is missing, set it to null (for scalars) or [] (for URL lists).
    3) Do not normalize or change wording; extract exactly as written.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
ZIP_REGEX = re.compile(r"^\d{5}(-\d{4})?$")


def is_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        # Ensure scheme
        if not u.startswith("http://") and not u.startswith("https://"):
            u = "http://" + u
        out.append(u)
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for u in out:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def get_host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def registered_domain(host: str) -> str:
    """
    Very lightweight approximation of the registrable domain.
    Handles common public suffix cases (.go.jp, .co.uk, etc.) in a basic way.
    """
    host = host.lower()
    if not host:
        return host
    parts = host.split(".")
    if len(parts) <= 2:
        return host

    two_level_suffixes = {
        "co.uk", "ac.uk", "gov.uk",
        "go.jp", "co.jp", "ac.jp", "or.jp", "ne.jp", "ed.jp", "lg.jp",
        "go.kr", "co.in", "gov.in", "gov.au", "edu.au", "gov.cn", "edu.cn",
    }
    last_two = ".".join(parts[-2:])
    last_three = ".".join(parts[-3:])
    if last_two in two_level_suffixes and len(parts) >= 3:
        return last_three
    return last_two


def is_official_approvals_url(url: str) -> bool:
    """
    Approvals must be supported by USDA APHIS (.usda.gov) or Japanese authorities (.go.jp).
    """
    host = get_host(url)
    rdom = registered_domain(host)
    if not rdom:
        return False
    if rdom.endswith("usda.gov"):
        return True
    if rdom.endswith("go.jp"):
        return True
    return False


def is_official_or_lab_url(url: str, lab_domain: Optional[str] = None) -> bool:
    """
    Address/timeframe can be supported by official or lab website.
    - Official: any .gov, .go.jp, or .edu
    - Lab: matches lab's registrable domain (if provided). If lab domain is unknown, allow .org as a reasonable proxy for org sites.
    """
    host = get_host(url)
    if not host:
        return False
    rdom = registered_domain(host)

    # Official domains
    if rdom.endswith(".gov") or rdom.endswith("go.jp") or rdom.endswith(".edu"):
        return True

    # Lab domain match
    if lab_domain:
        lab_dom = registered_domain(lab_domain)
        if rdom == lab_dom or rdom.endswith("." + lab_dom) or lab_dom.endswith("." + rdom):
            return True

    # Fallback: Allow .org as acceptable lab/organization website if lab domain not known
    if (not lab_domain) and rdom.endswith(".org"):
        return True

    return False


def extract_lab_registered_domain(lab_website_url: Optional[str]) -> Optional[str]:
    if not lab_website_url:
        return None
    host = get_host(lab_website_url)
    if not host:
        return None
    return registered_domain(host)


def build_full_address(addr: LabAddress) -> str:
    parts = [addr.street or "", addr.city or "", addr.state or "", addr.zip_code or ""]
    return ", ".join([p for p in parts if p])


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_laboratory_eligibility(
    evaluator: Evaluator,
    parent_node,
    info: LabExtraction
) -> None:
    """
    LaboratoryEligibility:
    - OfficialNameProvided (existence)
    - LocatedInUnitedStates (verify with sources)
    - USDAApprovedForFAVN (verify with approvals URLs)
    - ListedForJapanFAVN (verify with approvals URLs)
    """
    node = evaluator.add_parallel(
        id="LaboratoryEligibility",
        desc="Laboratory is correctly identified and meets all approval/location requirements for Japan FAVN testing",
        parent=parent_node,
        critical=True
    )

    # 1) Official name provided (existence)
    evaluator.add_custom_node(
        result=is_nonempty(info.official_name),
        id="OfficialNameProvided",
        desc="Provides the laboratory's official name",
        parent=node,
        critical=True
    )

    # Prepare sources for following checks
    approvals_urls = normalize_urls(info.sources.approvals_urls)
    address_urls = normalize_urls(info.sources.address_urls)
    lab_site = info.sources.lab_website_url
    all_loc_sources: List[str] = []
    all_loc_sources.extend(address_urls)
    all_loc_sources.extend(approvals_urls)
    if lab_site:
        all_loc_sources.append(lab_site)
    all_loc_sources = normalize_urls(all_loc_sources)

    # 2) Located in United States (verify via any available official/lab sources)
    located_node = evaluator.add_leaf(
        id="LocatedInUnitedStates",
        desc="Laboratory is located in the United States",
        parent=node,
        critical=True
    )
    claim_loc = f"The laboratory '{info.official_name or ''}' with mailing address '{build_full_address(info.address)}' is located in the United States of America."
    await evaluator.verify(
        claim=claim_loc,
        node=located_node,
        sources=all_loc_sources or None,
        additional_instruction=(
            "Use the provided source(s). If the address clearly indicates a U.S. state or the website states the lab is in the U.S., this claim is supported. "
            "If sources are missing or irrelevant, do not support the claim."
        )
    )

    # 3) USDA-approved for FAVN (verify with approvals URLs)
    usda_node = evaluator.add_leaf(
        id="USDAApprovedForFAVN",
        desc="Laboratory is USDA-approved for performing FAVN rabies antibody titer testing",
        parent=node,
        critical=True
    )
    claim_usda = (
        f"The laboratory '{info.official_name or ''}' is USDA-approved (APHIS recognized) to perform FAVN rabies antibody titer testing."
    )
    await evaluator.verify(
        claim=claim_usda,
        node=usda_node,
        sources=approvals_urls or None,
        additional_instruction=(
            "Look for explicit language like 'USDA-approved', 'APHIS recognized/approved', or inclusion on an official approvals list for FAVN testing. "
            "If the provided URL is not an official approvals listing page or does not clearly state this, mark as not supported."
        )
    )

    # 4) Listed for Japan FAVN (verify with approvals URLs)
    japan_node = evaluator.add_leaf(
        id="ListedForJapanFAVN",
        desc="Laboratory is specifically listed as approved/accepted for FAVN testing for pets traveling to Japan by USDA APHIS or Japanese authorities",
        parent=node,
        critical=True
    )
    claim_japan = (
        f"The laboratory '{info.official_name or ''}' is listed by USDA APHIS or Japanese MAFF/AQS as approved/accepted for FAVN testing for pets traveling to Japan."
    )
    await evaluator.verify(
        claim=claim_japan,
        node=japan_node,
        sources=approvals_urls or None,
        additional_instruction=(
            "Confirm the lab appears on an official approvals/accepted labs list for Japan pet travel (e.g., APHIS pages or MAFF/AQS pages). "
            "The page must explicitly or clearly indicate acceptance for Japan FAVN. If unclear or unrelated, mark as not supported."
        )
    )


async def build_mailing_address(
    evaluator: Evaluator,
    parent_node,
    info: LabExtraction
) -> None:
    """
    MailingAddress:
    - StreetAddressProvided (existence)
    - CityProvided (existence)
    - StateProvided (existence)
    - ZIPProvided (existence and simple format check)
    """
    node = evaluator.add_parallel(
        id="MailingAddress",
        desc="Provides the complete mailing address for submitting blood samples",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_nonempty(info.address.street),
        id="StreetAddressProvided",
        desc="Provides the street address",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_nonempty(info.address.city),
        id="CityProvided",
        desc="Provides the city",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_nonempty(info.address.state),
        id="StateProvided",
        desc="Provides the state",
        parent=node,
        critical=True
    )

    zip_ok = is_nonempty(info.address.zip_code) and bool(ZIP_REGEX.match(info.address.zip_code.strip()))
    evaluator.add_custom_node(
        result=bool(zip_ok),
        id="ZIPProvided",
        desc="Provides the ZIP code",
        parent=node,
        critical=True
    )


async def build_processing_timeframe(
    evaluator: Evaluator,
    parent_node,
    info: LabExtraction
) -> None:
    """
    ProcessingTimeframe:
    - TimeframeProvidedInCalendarDays (verify using timeframe URLs; must be in calendar days)
    """
    node = evaluator.add_parallel(
        id="ProcessingTimeframe",
        desc="Provides the current FAVN test processing timeframe in calendar days",
        parent=parent_node,
        critical=True
    )

    timeframe_leaf = evaluator.add_leaf(
        id="TimeframeProvidedInCalendarDays",
        desc="States the current turnaround/processing timeframe explicitly in calendar days (a number or a numeric range)",
        parent=node,
        critical=True
    )

    tf_value = info.processing_timeframe_days or ""
    tf_sources = normalize_urls(info.sources.timeframe_urls)
    claim_tf = (
        f"The current processing timeframe for the FAVN test at '{info.official_name or ''}' is '{tf_value}' in calendar days."
    )

    await evaluator.verify(
        claim=claim_tf,
        node=timeframe_leaf,
        sources=tf_sources or None,
        additional_instruction=(
            "Check that the stated timeframe is explicitly in calendar days and is a number or numeric range (e.g., '7 days' or '7-10 days'). "
            "If the source only provides business days or turnaround phrased in business days without a clear calendar-day equivalent, then this claim is NOT supported. "
            "If the statement lacks numeric specificity (e.g., ‘within two weeks’ without explicit days), treat as not supported."
        )
    )


async def build_official_source_verification(
    evaluator: Evaluator,
    parent_node,
    info: LabExtraction
) -> None:
    """
    OfficialSourceVerification:
    - OfficialURLForApprovals (existence with official-domain check)
    - OfficialOrLabURLForAddress (existence with official-or-lab-domain check)
    - OfficialOrLabURLForTimeframe (existence with official-or-lab-domain check)
    """
    node = evaluator.add_parallel(
        id="OfficialSourceVerification",
        desc="All key claims are verifiable via reference URLs from official sources",
        parent=parent_node,
        critical=True
    )

    approvals_urls = normalize_urls(info.sources.approvals_urls)
    address_urls = normalize_urls(info.sources.address_urls)
    timeframe_urls = normalize_urls(info.sources.timeframe_urls)
    lab_domain = extract_lab_registered_domain(info.sources.lab_website_url)

    # 1) Approvals: require at least one official URL (USDA APHIS .usda.gov or Japanese authorities .go.jp)
    has_official_approvals = any(is_official_approvals_url(u) for u in approvals_urls)
    evaluator.add_custom_node(
        result=bool(approvals_urls) and has_official_approvals,
        id="OfficialURLForApprovals",
        desc="Provides at least one official-source URL supporting the lab's USDA/Japan approval/acceptance status",
        parent=node,
        critical=True
    )

    # 2) Address: require at least one official-or-lab URL
    has_official_or_lab_for_address = any(is_official_or_lab_url(u, lab_domain) for u in address_urls)
    evaluator.add_custom_node(
        result=bool(address_urls) and has_official_or_lab_for_address,
        id="OfficialOrLabURLForAddress",
        desc="Provides at least one official-source URL supporting the provided mailing address",
        parent=node,
        critical=True
    )

    # 3) Timeframe: require at least one official-or-lab URL
    has_official_or_lab_for_timeframe = any(is_official_or_lab_url(u, lab_domain) for u in timeframe_urls)
    evaluator.add_custom_node(
        result=bool(timeframe_urls) and has_official_or_lab_for_timeframe,
        id="OfficialOrLabURLForTimeframe",
        desc="Provides at least one official-source URL supporting the stated current processing timeframe",
        parent=node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the USDA-approved FAVN (Japan) lab task.
    """
    # Initialize evaluator with root as PARALLEL
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

    # Extract structured lab info from the answer
    lab_info = await evaluator.extract(
        prompt=prompt_extract_lab_info(),
        template_class=LabExtraction,
        extraction_name="lab_info_extraction"
    )

    # Build main rubric groups (as critical children of root)
    # Note: Root in framework is non-critical; we ensure all children are marked critical=True.
    # To preserve gating behavior across groups, evaluate OfficialSourceVerification first.
    await build_official_source_verification(evaluator, root, lab_info)
    await build_laboratory_eligibility(evaluator, root, lab_info)
    await build_mailing_address(evaluator, root, lab_info)
    await build_processing_timeframe(evaluator, root, lab_info)

    # Return the evaluation summary
    return evaluator.get_summary()