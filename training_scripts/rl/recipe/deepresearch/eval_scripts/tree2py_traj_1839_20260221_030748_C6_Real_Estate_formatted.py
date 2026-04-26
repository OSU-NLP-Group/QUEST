import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dc_reit_state_2025"
TASK_DESCRIPTION = """
Identify the US state that requires data centers to derive 100% of their energy from renewable or nuclear sources according to legislation introduced in 2025. Then, identify a publicly-traded Real Estate Investment Trust (REIT) whose primary business focus is data center infrastructure and that operates facilities in this state. For the identified REIT, verify that its data center facilities in this state meet the following technical specifications required for AI workloads: (1) Support power density of at least 60 kW per rack, (2) Provide liquid cooling technology capability, and (3) Have achieved Uptime Institute Tier certification (specify the tier level). Additionally, provide the following operational information about the REIT's data center portfolio in the identified state: the specific city or region where facilities are located, the total megawatt (MW) capacity of facilities in this state, and a description of the typical lease structure. Provide reference URLs documenting: (a) the state legislation requiring 100% renewable/nuclear energy, (b) the REIT's operations in the identified state, and (c) each technical specification (power density, liquid cooling, and Tier certification).
""".strip()


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class StateLegislationExtraction(BaseModel):
    state_name: Optional[str] = None
    legislation_name: Optional[str] = None
    bill_number: Optional[str] = None
    legislation_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class REITExtraction(BaseModel):
    reit_name: Optional[str] = None
    exchange: Optional[str] = None
    ticker: Optional[str] = None
    data_center_focus_desc: Optional[str] = None
    state_operations_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class TechnicalExtraction(BaseModel):
    power_density_value: Optional[str] = None
    power_density_statement: Optional[str] = None
    power_density_urls: List[str] = Field(default_factory=list)

    liquid_cooling_statement: Optional[str] = None
    liquid_cooling_urls: List[str] = Field(default_factory=list)

    uptime_tier_level: Optional[str] = None
    uptime_urls: List[str] = Field(default_factory=list)


class OperationalExtraction(BaseModel):
    locations: List[str] = Field(default_factory=list)
    total_mw_capacity: Optional[str] = None
    lease_structure: Optional[str] = None
    operations_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_state_legislation() -> str:
    return """
    Extract the single US state that is claimed to require data centers to derive 100% of their energy from renewable or nuclear sources according to legislation introduced in 2025.
    Return the following fields:
    - state_name: Full US state name (e.g., "Virginia").
    - legislation_name: The name or title of the legislation or bill (if provided).
    - bill_number: The bill identifier/number (e.g., "HB 1234" or "SB 567") if provided.
    - legislation_url: ONE primary URL that best documents this requirement.
    - additional_urls: An array of any other URLs cited as references for this requirement.
    If any field is missing in the answer, return null (or an empty list for URLs).
    """


def prompt_extract_reit_info() -> str:
    return """
    Identify a publicly-traded REIT focused on data center infrastructure that operates facilities in the identified state.
    Extract the following fields:
    - reit_name: The name of the REIT.
    - exchange: The US stock exchange (e.g., "NYSE", "Nasdaq") if provided.
    - ticker: The stock ticker symbol if provided.
    - data_center_focus_desc: A short description of the REIT's primary business focus (e.g., "data center infrastructure").
    - state_operations_url: ONE URL that directly indicates the REIT operates data center facilities in the identified state (e.g., locations page, press release, or product page specific to that state).
    - additional_urls: Any other corporate, investor relations, or factsheet URLs supporting trading status or business focus.
    If any field is missing in the answer, return null (or an empty list for URLs).
    """


def prompt_extract_technical_specs() -> str:
    return """
    For the identified REIT's facilities in the identified state, extract the technical specifications relevant to AI workloads:
    - power_density_value: The stated power density per rack (e.g., "60 kW per rack", "up to 80 kW/rack"), as a string exactly as in the answer.
    - power_density_statement: Any descriptive statement about power density support.
    - power_density_urls: Array of URLs that document the power density capability.
    - liquid_cooling_statement: Statement that liquid cooling infrastructure/capability is available.
    - liquid_cooling_urls: Array of URLs that document liquid cooling capability.
    - uptime_tier_level: The Uptime Institute Tier level (e.g., "Tier III") if specified.
    - uptime_urls: Array of URLs that document the Uptime Institute Tier certification.
    Return null for missing fields (empty list for URL arrays).
    """


def prompt_extract_operational_info() -> str:
    return """
    For the REIT's data center portfolio in the identified state, extract:
    - locations: Array of the specific city or region names where facilities are located in the state.
    - total_mw_capacity: The total MW capacity in that state (string, as presented).
    - lease_structure: A short description of the typical lease structure (e.g., "long-term, triple-net leases with built-in rent escalators").
    - operations_urls: Array of URLs that document the locations, capacity, or lease structure (can include corporate IR pages or product/location pages).
    Return null for missing fields (empty list for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _collect_urls(*args: Optional[List[str] | str]) -> List[str]:
    urls: List[str] = []
    for item in args:
        if not item:
            continue
        if isinstance(item, str):
            urls.append(item.strip())
        elif isinstance(item, list):
            for u in item:
                if u and isinstance(u, str):
                    urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification section builders                                               #
# --------------------------------------------------------------------------- #
async def build_state_section(
    evaluator: Evaluator,
    parent,
    state_info: StateLegislationExtraction,
):
    state_node = evaluator.add_parallel(
        id="state_identification",
        desc="Identify the US state that requires data centers to derive 100% of energy from renewable or nuclear sources",
        parent=parent,
        critical=False,
    )

    # Leaf: state_name (critical) – verify via legislation URLs that the requirement applies and was introduced in 2025
    state_name_leaf = evaluator.add_leaf(
        id="state_name",
        desc="Provide the correct state name that has the 100% renewable/nuclear energy requirement",
        parent=state_node,
        critical=True,
    )
    state_name = state_info.state_name or ""
    state_sources = _collect_urls(state_info.legislation_url, state_info.additional_urls)

    claim_state = (
        f"The referenced legislation requires data centers in {state_name} to derive 100% of "
        f"their energy from renewable or nuclear sources, and the legislation was introduced in 2025."
    )
    await evaluator.verify(
        claim=claim_state,
        node=state_name_leaf,
        sources=state_sources,
        additional_instruction="Verify that the legislation explicitly applies to data centers, demands 100% renewable or nuclear energy sourcing, and was introduced in 2025.",
    )

    # Leaf: legislative_reference (critical) – ensure at least one legislation URL is provided
    legislative_reference_leaf = evaluator.add_custom_node(
        result=len(state_sources) > 0,
        id="legislative_reference",
        desc="Provide reference to the specific legislation or bill requiring 100% renewable/nuclear energy, including a reference URL",
        parent=state_node,
        critical=True,
    )


async def build_reit_section(
    evaluator: Evaluator,
    parent,
    reit_info: REITExtraction,
    ops_info: OperationalExtraction,
    state_name: Optional[str],
):
    reit_node = evaluator.add_parallel(
        id="reit_identification",
        desc="Identify a publicly-traded data center REIT that operates facilities in the identified state",
        parent=parent,
        critical=False,
    )

    # Leaf: reit_name (critical) – existence check
    evaluator.add_custom_node(
        result=bool(reit_info and reit_info.reit_name and reit_info.reit_name.strip()),
        id="reit_name",
        desc="Provide the name of a publicly-traded REIT focused on data center infrastructure",
        parent=reit_node,
        critical=True,
    )

    # Leaf: state_operations (critical) – verify REIT operates in the identified state
    state_ops_leaf = evaluator.add_leaf(
        id="state_operations",
        desc="Verify that the REIT operates data center facilities in the identified state, with reference URL",
        parent=reit_node,
        critical=True,
    )
    reit_name = reit_info.reit_name or "the REIT"
    state_ops_sources = _collect_urls(reit_info.state_operations_url, ops_info.operations_urls)
    claim_ops = f"{reit_name} operates data center facilities in {state_name}."
    await evaluator.verify(
        claim=claim_ops,
        node=state_ops_leaf,
        sources=state_ops_sources,
        additional_instruction="Look for location pages, press releases, or facility listings that clearly show sites within the named state.",
    )

    # Leaf: public_trading_status (critical) – verify publicly traded
    public_leaf = evaluator.add_leaf(
        id="public_trading_status",
        desc="Verify that the REIT is publicly traded on a major US stock exchange",
        parent=reit_node,
        critical=True,
    )
    if reit_info.exchange and reit_info.ticker:
        claim_public = f"{reit_name} is publicly traded on the {reit_info.exchange} under the ticker symbol {reit_info.ticker}."
    else:
        claim_public = f"{reit_name} is publicly traded on a major US stock exchange (NYSE or Nasdaq)."
    public_sources = _collect_urls(reit_info.additional_urls, reit_info.state_operations_url)
    await evaluator.verify(
        claim=claim_public,
        node=public_leaf,
        sources=public_sources,
        additional_instruction="Investor relations, company overview, or exchange listing pages should confirm trading status.",
    )

    # Leaf: data_center_focus (critical) – verify primary focus is data center infrastructure
    focus_leaf = evaluator.add_leaf(
        id="data_center_focus",
        desc="Verify that the REIT's primary business focus is data center infrastructure",
        parent=reit_node,
        critical=True,
    )
    claim_focus = f"The primary business focus of {reit_name} is data center infrastructure (e.g., colocation, hyperscale, interconnection)."
    focus_sources = _collect_urls(reit_info.additional_urls, reit_info.state_operations_url)
    await evaluator.verify(
        claim=claim_focus,
        node=focus_leaf,
        sources=focus_sources,
        additional_instruction="Use company overview pages, fact sheets, or 10-K filings that describe the core business focus as data center infrastructure.",
    )


async def build_technical_section(
    evaluator: Evaluator,
    parent,
    tech_info: TechnicalExtraction,
    reit_name: Optional[str],
    state_name: Optional[str],
):
    tech_node = evaluator.add_parallel(
        id="technical_specifications",
        desc="Verify the REIT's data center facilities meet specific technical requirements for AI workloads",
        parent=parent,
        critical=False,
    )

    reit_display = reit_name or "the REIT"
    state_display = state_name or "the state"

    # Power density sub-node (critical)
    pd_node = evaluator.add_parallel(
        id="power_density",
        desc="Verify facilities support power density of at least 60 kW per rack for AI workloads",
        parent=tech_node,
        critical=True,
    )

    # power_capability leaf (critical) – verify >= 60 kW per rack
    pd_cap_leaf = evaluator.add_leaf(
        id="power_capability",
        desc="Confirm the power density specification meets or exceeds 60 kW per rack",
        parent=pd_node,
        critical=True,
    )
    claim_pd = (
        f"{reit_display}'s data center facilities in {state_display} support at least 60 kW per rack power density."
    )
    pd_sources = _collect_urls(tech_info.power_density_urls)
    await evaluator.verify(
        claim=claim_pd,
        node=pd_cap_leaf,
        sources=pd_sources,
        additional_instruction="Accept phrasing like '≥60 kW/rack', 'up to 80 kW/rack', or similar that imply equal to or greater than 60 kW per rack.",
    )

    # power_reference_url (critical) – existence of reference URL
    evaluator.add_custom_node(
        result=len(pd_sources) > 0,
        id="power_reference_url",
        desc="Provide a reference URL documenting the power density capability",
        parent=pd_node,
        critical=True,
    )

    # Cooling technology sub-node (critical)
    cool_node = evaluator.add_parallel(
        id="cooling_technology",
        desc="Verify facilities support liquid cooling technology for high-density workloads",
        parent=tech_node,
        critical=True,
    )

    # liquid_cooling_capability leaf (critical)
    lc_leaf = evaluator.add_leaf(
        id="liquid_cooling_capability",
        desc="Confirm the availability of liquid cooling infrastructure",
        parent=cool_node,
        critical=True,
    )
    claim_lc = f"{reit_display}'s facilities in {state_display} provide liquid cooling capability."
    lc_sources = _collect_urls(tech_info.liquid_cooling_urls)
    await evaluator.verify(
        claim=claim_lc,
        node=lc_leaf,
        sources=lc_sources,
        additional_instruction="Look for mentions of liquid cooling, direct-to-chip, rear-door heat exchangers, or similar liquid-based cooling systems.",
    )

    # cooling_reference_url (critical) – existence of reference URL
    evaluator.add_custom_node(
        result=len(lc_sources) > 0,
        id="cooling_reference_url",
        desc="Provide a reference URL documenting the liquid cooling capability",
        parent=cool_node,
        critical=True,
    )

    # Reliability certification sub-node (critical)
    cert_node = evaluator.add_parallel(
        id="reliability_certification",
        desc="Verify facilities have Uptime Institute Tier certification (any tier level)",
        parent=tech_node,
        critical=True,
    )

    # tier_certification_status leaf (critical)
    tier_status_leaf = evaluator.add_leaf(
        id="tier_certification_status",
        desc="Confirm the facility has achieved Uptime Institute Tier certification",
        parent=cert_node,
        critical=True,
    )
    tier_sources = _collect_urls(tech_info.uptime_urls)
    claim_tier_status = f"{reit_display}'s facilities in {state_display} have achieved Uptime Institute Tier certification."
    await evaluator.verify(
        claim=claim_tier_status,
        node=tier_status_leaf,
        sources=tier_sources,
        additional_instruction="Accept Tier I, II, III, or IV certifications and check the page for explicit mention of Uptime Institute certification.",
    )

    # tier_level leaf (critical) – verify stated level if provided, else generic
    tier_level_leaf = evaluator.add_leaf(
        id="tier_level",
        desc="Specify the Uptime Institute Tier level (I, II, III, or IV)",
        parent=cert_node,
        critical=True,
    )
    if tech_info.uptime_tier_level and tech_info.uptime_tier_level.strip():
        claim_tier_level = (
            f"{reit_display}'s facilities in {state_display} have Uptime Institute {tech_info.uptime_tier_level.strip()} certification."
        )
    else:
        claim_tier_level = (
            f"{reit_display}'s facilities in {state_display} have achieved some Uptime Institute Tier certification level."
        )
    await evaluator.verify(
        claim=claim_tier_level,
        node=tier_level_leaf,
        sources=tier_sources,
        additional_instruction="If a specific Tier level is mentioned (e.g., Tier III), verify that level; otherwise confirm the existence of any Tier level certification.",
    )

    # certification_reference_url (critical) – existence of reference URL
    evaluator.add_custom_node(
        result=len(tier_sources) > 0,
        id="certification_reference_url",
        desc="Provide a reference URL documenting the Tier certification",
        parent=cert_node,
        critical=True,
    )


async def build_operational_section(
    evaluator: Evaluator,
    parent,
    ops_info: OperationalExtraction,
    reit_info: REITExtraction,
    state_name: Optional[str],
):
    ops_node = evaluator.add_parallel(
        id="operational_information",
        desc="Provide additional operational details about the REIT's data center portfolio",
        parent=parent,
        critical=False,
    )

    state_display = state_name or "the state"
    reit_name = reit_info.reit_name or "the REIT"
    ops_sources = _collect_urls(ops_info.operations_urls, reit_info.state_operations_url, reit_info.additional_urls)

    # facility_location (non-critical)
    loc_leaf = evaluator.add_leaf(
        id="facility_location",
        desc="Identify the specific city or region where the REIT operates data center facilities in the identified state",
        parent=ops_node,
        critical=False,
    )
    if ops_info.locations:
        loc_list = ", ".join(sorted(set([l for l in ops_info.locations if l and isinstance(l, str)])))
        claim_loc = f"{reit_name} operates data center facilities in {state_display} in the following city/region locations: {loc_list}."
    else:
        claim_loc = f"{reit_name} operates data center facilities in {state_display}."
    await evaluator.verify(
        claim=claim_loc,
        node=loc_leaf,
        sources=ops_sources,
        additional_instruction="Confirm that the named cities/regions (if any) correspond to facilities in the identified state.",
    )

    # total_capacity (non-critical)
    cap_leaf = evaluator.add_leaf(
        id="total_capacity",
        desc="Provide information about the total MW capacity of the REIT's facilities in the identified state",
        parent=ops_node,
        critical=False,
    )
    if ops_info.total_mw_capacity and ops_info.total_mw_capacity.strip():
        claim_cap = f"The total MW capacity for {reit_name}'s facilities in {state_display} is {ops_info.total_mw_capacity.strip()}."
    else:
        claim_cap = f"The total MW capacity for {reit_name}'s facilities in {state_display} is stated on the provided sources."
    await evaluator.verify(
        claim=claim_cap,
        node=cap_leaf,
        sources=ops_sources,
        additional_instruction="Verify the MW capacity figure for the identified state if explicitly provided on the source pages.",
    )

    # lease_structure (non-critical)
    lease_leaf = evaluator.add_leaf(
        id="lease_structure",
        desc="Describe the typical lease structure (long-term contracts with built-in rent escalators)",
        parent=ops_node,
        critical=False,
    )
    if ops_info.lease_structure and ops_info.lease_structure.strip():
        claim_lease = f"{reit_name}'s typical lease structure is described as: {ops_info.lease_structure.strip()}."
    else:
        claim_lease = f"{reit_name}'s typical lease structure is long-term contracts, often with built-in rent escalators."
    await evaluator.verify(
        claim=claim_lease,
        node=lease_leaf,
        sources=ops_sources,
        additional_instruction="Use investor relations or corporate overview pages to confirm lease structure descriptions (e.g., long-term, triple-net, rent escalators).",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluation entry point for the multi-step data center REIT analysis.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Sequential: later steps depend on earlier correctness
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

    # ----------------------- Extraction Phase ---------------------------- #
    state_info = await evaluator.extract(
        prompt=prompt_extract_state_legislation(),
        template_class=StateLegislationExtraction,
        extraction_name="state_legislation",
    )

    identified_state = state_info.state_name or None
    add_ins_state = f"Identified state: {identified_state}" if identified_state else "Identified state: None"

    reit_info = await evaluator.extract(
        prompt=prompt_extract_reit_info(),
        template_class=REITExtraction,
        extraction_name="reit_info",
        additional_instruction=add_ins_state,
    )

    reit_name = reit_info.reit_name or None
    add_ins_reit = f"Identified state: {identified_state}; REIT: {reit_name}"

    tech_info = await evaluator.extract(
        prompt=prompt_extract_technical_specs(),
        template_class=TechnicalExtraction,
        extraction_name="technical_specs",
        additional_instruction=add_ins_reit,
    )

    ops_info = await evaluator.extract(
        prompt=prompt_extract_operational_info(),
        template_class=OperationalExtraction,
        extraction_name="operational_info",
        additional_instruction=add_ins_reit,
    )

    # Record lightweight custom info for debugging
    evaluator.add_custom_info(
        {
            "state": state_info.state_name,
            "legislation_url": state_info.legislation_url,
            "reit_name": reit_info.reit_name,
            "exchange": reit_info.exchange,
            "ticker": reit_info.ticker,
            "state_operations_url": reit_info.state_operations_url,
        },
        info_type="extraction_summary",
    )

    # --------------------- Verification Tree Build ---------------------- #
    # 1) State identification
    await build_state_section(evaluator, root, state_info)

    # 2) REIT identification
    await build_reit_section(evaluator, root, reit_info, ops_info, state_info.state_name)

    # 3) Technical specifications
    await build_technical_section(evaluator, root, tech_info, reit_info.reit_name, state_info.state_name)

    # 4) Operational information
    await build_operational_section(evaluator, root, ops_info, reit_info, state_info.state_name)

    # --------------------------- Return --------------------------------- #
    return evaluator.get_summary()