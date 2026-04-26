import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_semiconductor_major_investments_az_tx_ny"
TASK_DESCRIPTION = (
    "Identify 4 major semiconductor manufacturing companies that have announced investments of $10 billion or more "
    "(total commitment) for manufacturing facilities in Arizona, Texas, or New York. For each company, provide: "
    "(1) The company name, (2) At least one state (Arizona, Texas, or New York) where they have or are building a "
    "manufacturing facility, (3) A specific city within that state where their facility is located, (4) The total "
    "investment amount (in USD billions) they have committed for facilities in the specified state(s), "
    "(5) The primary facility type classification (must be either 'IDM Manufacturing' or 'Foundry Manufacturing'), "
    "(6) One key technical specification: either the wafer size (e.g., '300mm') or technology process node "
    "(e.g., '5nm', '3nm'), and (7) A reference URL that verifies the investment and facility information. "
    "Requirements: Each company must have at least one facility classified as 'IDM Manufacturing' or 'Foundry Manufacturing' "
    "(not only Materials, Equipment, or OSAT facilities). The $10 billion investment threshold refers to the company's total "
    "announced investment across all their facilities in Arizona, Texas, and/or New York combined. Use information available "
    "as of February 2026."
)

ALLOWED_STATES = {"arizona": "Arizona", "texas": "Texas", "new york": "New York", "ny": "New York", "az": "Arizona", "tx": "Texas"}
REQUIRED_CLASSIFICATIONS = {"foundry manufacturing": "Foundry Manufacturing", "idm manufacturing": "IDM Manufacturing"}
INVESTMENT_THRESHOLD_BILLIONS = 10.0


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Location(BaseModel):
    state: Optional[str] = None
    city: Optional[str] = None


class CompanyItem(BaseModel):
    company_name: Optional[str] = None
    locations: List[Location] = Field(default_factory=list)
    investment_total: Optional[str] = None  # Keep as string for flexibility; we'll parse
    facility_type: Optional[str] = None     # As stated in the answer; we will normalize
    technical_spec: Optional[str] = None    # Either wafer size (e.g., 300mm) or process node (e.g., 5nm)
    source_urls: List[str] = Field(default_factory=list)  # URLs cited for the company


class CompaniesExtraction(BaseModel):
    companies: List[CompanyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_companies() -> str:
    return """
    Extract up to all semiconductor manufacturing companies mentioned in the answer that are associated with investments
    and facilities in Arizona, Texas, or New York. For each company, extract the following fields exactly as presented:

    - company_name: The company’s name.
    - locations: An array of objects. For each object:
        - state: The U.S. state name where the facility is located (only extract if it is explicitly one of:
                 "Arizona", "Texas", or "New York"; also accept abbreviations AZ, TX, NY and normalize to full state
                 name if the answer clearly implies that; otherwise leave null).
        - city: The specific city within that state, if mentioned; otherwise null.
    - investment_total: The total investment amount across the stated facilities in those state(s), as a string
                        exactly as shown (e.g., "$25 billion", "at least $20B", "~$30bn", ">$10B").
    - facility_type: The facility type classification as described in the answer text for at least one facility in those states.
                     Examples: "Foundry Manufacturing", "IDM Manufacturing", "foundry", "IDM", etc. Do not invent; extract
                     the closest explicit term the answer uses.
    - technical_spec: One key technical specification from the answer text that can be associated with the facility,
                      either a wafer size (e.g., "300mm") or a process node (e.g., "5nm", "3nm", "18A").
    - source_urls: All URLs in the answer that are relevant to this company’s investment and facility information.
                   Include full URLs; accept plain links or markdown links. If none are present, return an empty list.

    Return the result as:
    {
      "companies": [
        {
          "company_name": ...,
          "locations": [{"state": ..., "city": ...}, ...],
          "investment_total": ...,
          "facility_type": ...,
          "technical_spec": ...,
          "source_urls": [...]
        }, ...
      ]
    }

    Notes:
    - Do NOT add information that is not explicitly in the answer.
    - If a field is missing for a company, set it to null (or empty array for lists).
    - Preserve the original order of companies as they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper normalization and parsing                                            #
# --------------------------------------------------------------------------- #
def _normalize_whitespace(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return re.sub(r"\s+", " ", s.strip())


def normalize_state_name(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = _normalize_whitespace(state).lower().replace(".", "")
    s = s.replace("st.", "saint")  # not relevant, but harmless
    # Normalize common abbreviations or variants
    if s in ALLOWED_STATES:
        return ALLOWED_STATES[s]
    # Handle partial matches (e.g., "AZ", "Ariz", "N.Y.")
    if s in {"ariz", "arizona"}:
        return "Arizona"
    if s in {"tx", "texas"}:
        return "Texas"
    if s in {"new york", "ny", "n y", "n.y"}:
        return "New York"
    return None


def normalize_classification(ftype: Optional[str]) -> Optional[str]:
    if not ftype:
        return None
    t = _normalize_whitespace(ftype).lower()
    # Map common synonyms
    if "foundry" in t or "contract manufacturer" in t or "pure-play" in t:
        return "Foundry Manufacturing"
    if "idm" in t or "integrated device manufacturer" in t:
        return "IDM Manufacturing"
    # Direct matches
    key = t if t in REQUIRED_CLASSIFICATIONS else None
    if key:
        return REQUIRED_CLASSIFICATIONS[key]
    return None


def parse_investment_billion(invest_str: Optional[str]) -> Optional[float]:
    """
    Parse an investment string and return a conservative minimum estimate in billions USD.
    Handles forms like:
    - "$25 billion", "$20B", "20bn", ">$10B", "approx $12b", "~$15bn", "15-20 billion", "2 trillion"
    - "30,000,000,000" (assume USD)
    Returns None if cannot parse any numeric cue.
    """
    if not invest_str:
        return None
    s = invest_str.lower()
    s = s.replace(",", "")
    # Replace unicode approximations
    s = s.replace("~", "").replace("≈", "").replace("about", "").replace("approx", "").replace("approximately", "")
    # Extract range if present (e.g., "15-20", "10 to 15")
    range_match = re.findall(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(trillion|tn|billion|bn|b|million|m)?", s)
    if range_match:
        lows = []
        for low, high, unit in range_match:
            try:
                low_v = float(low)
                unit_mult = 1.0
                if unit in {"trillion", "tn"}:
                    unit_mult = 1000.0
                elif unit in {"million", "m"}:
                    unit_mult = 0.001
                else:
                    unit_mult = 1.0  # billions
                lows.append(low_v * unit_mult)
            except:
                continue
        if lows:
            return min(lows)

    # Extract single amounts with unit
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(trillion|tn|billion|bn|b|million|m)?", s)
    values = []
    for num, unit in matches:
        try:
            val = float(num)
        except:
            continue
        unit_mult = 1.0
        if unit in {"trillion", "tn"}:
            unit_mult = 1000.0
        elif unit in {"million", "m"}:
            unit_mult = 0.001
        else:
            unit_mult = 1.0  # b/bn/billion default to billions
        values.append(val * unit_mult)

    if values:
        # Conservative: take the minimum plausible interpretation
        return min(values)

    # As a fallback, if huge raw number like 30000000000 appears, convert to billions
    big_num = re.findall(r"\$?(\d{8,})", s)
    if big_num:
        try:
            raw = float(big_num[0])
            return raw / 1e9
        except:
            pass

    return None


def looks_like_tech_spec(spec: Optional[str]) -> bool:
    if not spec:
        return False
    t = _normalize_whitespace(spec).lower()
    # Simple patterns: wafer sizes and process nodes
    if re.search(r"\b(200|300)\s*mm\b", t):
        return True
    if re.search(r"\b\d+(\.\d+)?\s*nm\b", t):
        return True
    if re.search(r"\b\d+\s*a\b", t):  # e.g., "18A"
        return True
    if "node" in t and re.search(r"\b\d+(\.\d+)?\b", t):
        return True
    return False


def select_primary_location(locations: List[Location]) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Select a primary location (state and city) among provided locations where state is in allowed states.
    Returns (state_full, city, all_allowed_states_list).
    """
    allowed_states = []
    chosen_state = None
    chosen_city = None
    for loc in locations:
        norm_state = normalize_state_name(loc.state)
        if norm_state:
            allowed_states.append(norm_state)
            if chosen_state is None:
                chosen_state = norm_state
                chosen_city = _normalize_whitespace(loc.city) if loc.city else None
    # Deduplicate states preserving order
    seen = set()
    uniq_states = []
    for s in allowed_states:
        if s not in seen:
            seen.add(s)
            uniq_states.append(s)
    return chosen_state, chosen_city, uniq_states


def dedup_urls(urls: List[str]) -> List[str]:
    out = []
    seen = set()
    for u in urls:
        if not u:
            continue
        uu = u.strip()
        if uu and uu not in seen:
            seen.add(uu)
            out.append(uu)
    return out


# --------------------------------------------------------------------------- #
# Verification logic for one company                                          #
# --------------------------------------------------------------------------- #
async def verify_company(
    evaluator: Evaluator,
    parent_node,
    company: CompanyItem,
    company_index: int,
) -> None:
    """
    Build and verify the tree for a single company as per rubric.
    """
    cname = _normalize_whitespace(company.company_name) if company else None
    primary_state, primary_city, all_states = select_primary_location(company.locations if company else [])
    states_str = ", ".join(all_states) if all_states else ""
    inv_str = _normalize_whitespace(company.investment_total) if company else None
    cls_norm = normalize_classification(company.facility_type if company else None)
    spec_str = _normalize_whitespace(company.technical_spec) if company else None
    urls = dedup_urls(company.source_urls if company else [])

    # Company node (parallel, non-critical as per rubric)
    comp_node = evaluator.add_parallel(
        id=f"company_{company_index}",
        desc=f"Semiconductor company #{company_index} verification",
        parent=parent_node,
        critical=False
    )

    # 1) Company_Name (critical): Provided
    evaluator.add_custom_node(
        result=bool(cname),
        id=f"company_{company_index}_name",
        desc="Company name is provided",
        parent=comp_node,
        critical=True
    )

    # 2) Geographic_Location (critical parallel)
    geo_node = evaluator.add_parallel(
        id=f"company_{company_index}_geo",
        desc="Verification of facility location in Arizona, Texas, or New York",
        parent=comp_node,
        critical=True
    )

    # 2.1) State_Identification (critical)
    evaluator.add_custom_node(
        result=bool(primary_state),
        id=f"company_{company_index}_state",
        desc="Company has facility in Arizona, Texas, or New York",
        parent=geo_node,
        critical=True
    )

    # 2.2) City_Location (critical)
    evaluator.add_custom_node(
        result=bool(primary_city),
        id=f"company_{company_index}_city",
        desc="Specific city within the state is provided",
        parent=geo_node,
        critical=True
    )

    # 2.3) Location_Reference_URL (critical, URL-verified)
    loc_ref_leaf = evaluator.add_leaf(
        id=f"company_{company_index}_location_ref",
        desc="URL reference confirms the facility location",
        parent=geo_node,
        critical=True
    )
    # If no URLs, mark fail immediately to enforce source grounding
    if urls:
        loc_claim_parts = []
        if cname:
            loc_claim_parts.append(f"{cname}")
        loc_claim_parts.append("has or is building a semiconductor wafer fabrication facility")
        if primary_city and primary_state:
            loc_claim_parts.append(f"in {primary_city}, {primary_state}")
        elif primary_state:
            loc_claim_parts.append(f"in {primary_state}")
        loc_claim = " ".join(loc_claim_parts) + "."
        await evaluator.verify(
            claim=loc_claim,
            node=loc_ref_leaf,
            sources=urls,
            additional_instruction=(
                "Verify that the page indicates a semiconductor wafer fabrication (front-end manufacturing) facility, "
                "not merely equipment, materials, or OSAT packaging/testing. Accept phrasing such as 'fab', 'fabrication plant', "
                "'chip factory', or 'semiconductor manufacturing facility'. Minor differences in city naming are acceptable if the "
                "state and metro area clearly match."
            )
        )
    else:
        loc_ref_leaf.score = 0.0
        loc_ref_leaf.status = "failed"

    # 3) Investment_Profile (critical parallel)
    invest_node = evaluator.add_parallel(
        id=f"company_{company_index}_investment",
        desc="Verification of investment amount meeting $10 billion threshold",
        parent=comp_node,
        critical=True
    )

    # 3.1) Investment_Amount (critical) — ensure answer states at least $10B (parsed from provided text)
    min_billions = parse_investment_billion(inv_str)
    evaluator.add_custom_node(
        result=(min_billions is not None and min_billions >= INVESTMENT_THRESHOLD_BILLIONS),
        id=f"company_{company_index}_investment_amount",
        desc="Total investment is at least $10 billion for facilities in specified states",
        parent=invest_node,
        critical=True
    )

    # 3.2) Investment_Reference_URL (critical, URL-verified)
    inv_ref_leaf = evaluator.add_leaf(
        id=f"company_{company_index}_investment_ref",
        desc="URL reference confirms the investment amount",
        parent=invest_node,
        critical=True
    )
    if urls:
        if states_str:
            inv_claim = (
                f"According to the cited sources, {cname} has announced total investment of at least $10 billion "
                f"for semiconductor manufacturing facilities in {states_str}."
            )
        else:
            inv_claim = (
                f"According to the cited sources, {cname} has announced total investment of at least $10 billion "
                f"for semiconductor manufacturing facilities in Arizona, Texas, or New York."
            )
        await evaluator.verify(
            claim=inv_claim,
            node=inv_ref_leaf,
            sources=urls,
            additional_instruction=(
                "Look for explicit investment figures for wafer fabrication facilities located in Arizona, Texas, or New York. "
                "It's acceptable if a single site meets the $10B threshold. If multiple sources cite different figures, consider "
                "whether any single figure is ≥ $10B; that suffices to support 'at least $10B'. Exclude investments clearly "
                "limited to packaging/test or materials/equipment supplier facilities."
            )
        )
    else:
        inv_ref_leaf.score = 0.0
        inv_ref_leaf.status = "failed"

    # 4) Manufacturing_Type (critical parallel)
    mtype_node = evaluator.add_parallel(
        id=f"company_{company_index}_mtype",
        desc="Verification of facility type classification",
        parent=comp_node,
        critical=True
    )

    # 4.1) Facility_Classification (critical) — ensure answer states IDM or Foundry (allow synonyms via normalization)
    evaluator.add_custom_node(
        result=bool(cls_norm),
        id=f"company_{company_index}_facility_classification",
        desc="Facility type is either 'IDM Manufacturing' or 'Foundry Manufacturing'",
        parent=mtype_node,
        critical=True
    )

    # 4.2) Facility_Type_Reference_URL (critical, URL-verified)
    mtype_ref_leaf = evaluator.add_leaf(
        id=f"company_{company_index}_facility_type_ref",
        desc="URL reference confirms the facility type",
        parent=mtype_node,
        critical=True
    )
    if urls and cls_norm:
        if primary_city and primary_state:
            cls_claim = (
                f"The {cname} facility in {primary_city}, {primary_state} is a {cls_norm} wafer fabrication facility."
            )
        elif primary_state:
            cls_claim = (
                f"A {cname} facility in {primary_state} is a {cls_norm} wafer fabrication facility."
            )
        else:
            cls_claim = (
                f"{cname} is operating or building a {cls_norm} wafer fabrication facility in Arizona, Texas, or New York."
            )
        await evaluator.verify(
            claim=cls_claim,
            node=mtype_ref_leaf,
            sources=urls,
            additional_instruction=(
                "Accept synonyms: 'foundry' for Foundry Manufacturing; 'integrated device manufacturer' or 'IDM' for IDM Manufacturing. "
                "Ensure that the page refers to wafer fabrication/production, not packaging, testing, equipment vendor, or materials supplier."
            )
        )
    else:
        mtype_ref_leaf.score = 0.0
        mtype_ref_leaf.status = "failed"

    # 5) Technical_Specification (critical parallel)
    spec_node = evaluator.add_parallel(
        id=f"company_{company_index}_techspec",
        desc="Verification of wafer size or technology node",
        parent=comp_node,
        critical=True
    )

    # 5.1) Specification_Detail (critical) — provided and looks like wafer size or process node
    evaluator.add_custom_node(
        result=looks_like_tech_spec(spec_str),
        id=f"company_{company_index}_spec_detail",
        desc="Wafer size (e.g., 300mm) or process node (e.g., 5nm) is provided",
        parent=spec_node,
        critical=True
    )

    # 5.2) Specification_Reference_URL (critical, URL-verified)
    spec_ref_leaf = evaluator.add_leaf(
        id=f"company_{company_index}_spec_ref",
        desc="URL reference confirms the technical specification",
        parent=spec_node,
        critical=True
    )
    if urls and spec_str:
        if primary_city and primary_state:
            spec_claim = (
                f"The {cname} facility in {primary_city}, {primary_state} is associated with the technical specification '{spec_str}'."
            )
        else:
            spec_claim = (
                f"A {cname} semiconductor manufacturing facility in Arizona, Texas, or New York is associated with '{spec_str}'."
            )
        await evaluator.verify(
            claim=spec_claim,
            node=spec_ref_leaf,
            sources=urls,
            additional_instruction=(
                "Confirm either a wafer size (e.g., 300mm) or a process node (e.g., 5nm, 3nm, 18A). Accept 'nm-class' phrasing. "
                "Ensure the specification pertains to wafer fabrication and is relevant to the cited facility."
            )
        )
    else:
        spec_ref_leaf.score = 0.0
        spec_ref_leaf.status = "failed"


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
) -> Dict:
    """
    Evaluate an answer against the rubric for identifying four semiconductor manufacturing companies
    with ≥$10B investments in Arizona, Texas, or New York, and verify required details with sources.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Companies evaluated independently
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

    # Record requirement context
    evaluator.add_custom_info(
        info={"allowed_states": ["Arizona", "Texas", "New York"], "investment_threshold_billions": INVESTMENT_THRESHOLD_BILLIONS,
              "as_of": "February 2026"},
        info_type="requirements",
        info_name="task_requirements"
    )

    # Extract companies from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_companies(),
        template_class=CompaniesExtraction,
        extraction_name="companies_extraction"
    )

    # Ensure exactly 4 entries (pad if fewer, slice if more)
    companies: List[CompanyItem] = (extracted.companies or [])[:4]
    while len(companies) < 4:
        companies.append(CompanyItem())

    # Create top-level node (parallel). Root is already parallel; we can add a descriptive grouping if desired.
    # But we can directly add company subtrees to root.

    # Build and verify each company subtree
    for idx in range(4):
        await verify_company(
            evaluator=evaluator,
            parent_node=root,
            company=companies[idx],
            company_index=idx + 1
        )

    # Return summary with verification tree and info
    return evaluator.get_summary()