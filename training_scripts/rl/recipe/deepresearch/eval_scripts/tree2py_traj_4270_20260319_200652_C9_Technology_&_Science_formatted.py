import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_semiconductor_facilities_tsmc_intel_samsung"
TASK_DESCRIPTION = """
Identify three major semiconductor manufacturing facilities currently operational or under construction in the United States, with one facility from each of the following companies: TSMC (Taiwan Semiconductor Manufacturing Company), Intel Corporation, and Samsung Electronics.

For each facility, provide:
1. Company Information: The full company name and a reference URL confirming their ownership/operation of the facility.
2. Location Details: The specific city and state where the facility is located, with reference URL(s) supporting this information.
3. Facility Specifications:
   - The total investment amount in USD (must be at least $10 billion)
   - Production capacity details (for TSMC: wafers per month at full capacity; for Intel: number of fabs planned; for Samsung: facility type)
   - Technology capabilities (process node size or production focus)
   - Reference URL(s) for these specifications
4. Timeline Information:
   - Current construction or operational status
   - Expected completion date or operational start date
   - Reference URL(s) for timeline details
5. Network Infrastructure:
   - Identification of at least one major U.S. wireless carrier (T-Mobile, AT&T, or Verizon) providing 5G coverage in the facility's area
   - Reference URL(s) confirming 5G coverage availability

Each facility must meet the following criteria:
- Located in the United States
- Represents an investment of at least $10 billion USD
- Capable of producing advanced semiconductor technology (7nm process node or smaller)
- Has publicly announced construction timelines or operational status
- Located in an area with 5G network coverage from major carriers

Provide all information with supporting reference URLs from reliable sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FacilityCommon(BaseModel):
    company_name: Optional[str] = None
    company_refs: List[str] = Field(default_factory=list)

    city: Optional[str] = None
    state: Optional[str] = None
    location_refs: List[str] = Field(default_factory=list)

    investment_amount_usd: Optional[str] = None

    timeline_status: Optional[str] = None
    timeline_date: Optional[str] = None
    timeline_refs: List[str] = Field(default_factory=list)

    network_carrier: Optional[str] = None  # Expected: "T-Mobile", "AT&T", or "Verizon"
    network_refs: List[str] = Field(default_factory=list)


class TSMCFacility(FacilityCommon):
    capacity_wpm: Optional[str] = None  # wafers per month at full capacity
    technology_node: Optional[str] = None  # e.g., 5nm, 3nm, N5, N3
    specs_refs: List[str] = Field(default_factory=list)


class IntelFacility(FacilityCommon):
    number_of_fabs: Optional[str] = None  # number of fabs planned at the site
    technology_capability: Optional[str] = None  # e.g., Intel 4/Intel 3/20A/18A
    specs_refs: List[str] = Field(default_factory=list)


class SamsungFacility(FacilityCommon):
    facility_type: Optional[str] = None  # e.g., advanced logic fab, foundry, semiconductor manufacturing facility
    production_focus: Optional[str] = None  # e.g., 4nm GAA, advanced logic nodes
    specs_refs: List[str] = Field(default_factory=list)


class FacilitiesExtraction(BaseModel):
    tsmc: Optional[TSMCFacility] = None
    intel: Optional[IntelFacility] = None
    samsung: Optional[SamsungFacility] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return """
Extract structured information about three major semiconductor manufacturing facilities in the United States, one each for TSMC (Taiwan Semiconductor Manufacturing Company), Intel Corporation, and Samsung Electronics (or Samsung Austin Semiconductor).

Requirements:
- Choose exactly one facility for each company (TSMC, Intel, Samsung). If multiple are mentioned, pick the principal one described.
- Extract only what is explicitly present in the answer. Do not fabricate or infer any information.
- For any missing field, return null (for strings) or an empty list (for arrays).
- All URLs must be valid and explicitly present in the answer.

Return a JSON object with three top-level fields: tsmc, intel, samsung.
Each field should be an object with the following fields (strings unless otherwise noted):

Common fields (for all three companies):
- company_name: The company name exactly as presented in the answer.
- company_refs: An array of URL strings that confirm the company owns/operates/is building the facility.
- city: The city where the facility is located.
- state: The U.S. state where the facility is located.
- location_refs: An array of URL strings supporting the location (city and state).
- investment_amount_usd: The total investment (e.g., "$40B", "over $20 billion", "USD 25 billion").
- timeline_status: Current status (e.g., "under construction", "operational", "planned").
- timeline_date: Expected completion or operational start date (can be a year or month+year).
- timeline_refs: An array of URL strings supporting status and/or dates.
- network_carrier: One of: "T-Mobile", "AT&T", or "Verizon" (the carrier identified in the answer as providing 5G coverage in the facility area).
- network_refs: An array of URL strings confirming 5G coverage for the facility area.

TSMC-specific fields (in addition to common fields):
- capacity_wpm: Wafers per month at full capacity (string).
- technology_node: Technology node capability (e.g., "5nm", "N3", "3nm").
- specs_refs: An array of URL strings supporting capacity, investment, and/or node.

Intel-specific fields:
- number_of_fabs: Number of fabrication plants planned at the site (string, e.g., "2 fabs").
- technology_capability: Advanced node capability (e.g., "Intel 4/Intel 3/20A/18A").
- specs_refs: An array of URL strings supporting the number of fabs, investment, and/or technology capability.

Samsung-specific fields:
- facility_type: Type of facility (e.g., "advanced logic fab", "foundry").
- production_focus: Production focus or technology (e.g., "4nm GAA", "advanced logic nodes").
- specs_refs: An array of URL strings supporting investment, facility type, and/or production focus.

If any field is missing in the answer, set it to null (or [] for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0)


def loc_str(city: Optional[str], state: Optional[str]) -> str:
    if is_non_empty(city) and is_non_empty(state):
        return f"{city}, {state}"
    if is_non_empty(state):
        return f"{state}, United States"
    return "the United States"


async def verify_with_urls(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    parent,
    claim: str,
    sources: Optional[List[str] | str],
    critical: bool = True,
    additional_instruction: str = "None"
):
    node = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=additional_instruction
    )
    return node


# --------------------------------------------------------------------------- #
# Verification builders for each company                                      #
# --------------------------------------------------------------------------- #
async def verify_tsmc(evaluator: Evaluator, root, fac: Optional[TSMCFacility]):
    tsmc_node = evaluator.add_parallel(
        id="Facility_from_TSMC",
        desc="A semiconductor manufacturing facility operated by TSMC in the United States",
        parent=root,
        critical=False
    )

    # Safeguard empty facility
    fac = fac or TSMCFacility()

    # Company Identification (critical)
    comp = evaluator.add_parallel(
        id="TSMC_Company_Identification",
        desc="Correctly identifies TSMC as the operating company",
        parent=tsmc_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.company_name),
        id="TSMC_Company_Name_present",
        desc="TSMC company name is provided",
        parent=comp,
        critical=True
    )

    await verify_with_urls(
        evaluator,
        node_id="TSMC_Company_Name",
        desc="Provides the full company name: Taiwan Semiconductor Manufacturing Company",
        parent=comp,
        claim=f"The provided company name '{fac.company_name}' refers to Taiwan Semiconductor Manufacturing Company (TSMC). Minor variants such as 'TSMC' or 'Taiwan Semiconductor Manufacturing Co., Ltd.' should be considered equivalent.",
        sources=None,
        additional_instruction="Judge whether the provided name clearly refers to Taiwan Semiconductor Manufacturing Company."
    )

    evaluator.add_custom_node(
        result=has_urls(fac.company_refs),
        id="TSMC_Company_References_provided",
        desc="Reference URL(s) confirming TSMC facility ownership/operation are provided",
        parent=comp,
        critical=True
    )

    await verify_with_urls(
        evaluator,
        node_id="TSMC_Company_References",
        desc="Provides valid reference URLs for TSMC facility information",
        parent=comp,
        claim=f"The sources confirm that TSMC (Taiwan Semiconductor Manufacturing Company) owns/operates or is building a semiconductor manufacturing facility in {loc_str(fac.city, fac.state)}.",
        sources=fac.company_refs,
        additional_instruction="Accept clear evidence such as official press releases, credible news, or government pages stating TSMC's facility at the cited location."
    )

    # Location Details (critical)
    loc = evaluator.add_parallel(
        id="TSMC_Location_Details",
        desc="Provides accurate location information for the TSMC facility",
        parent=tsmc_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.city),
        id="TSMC_City_present",
        desc="TSMC city is provided",
        parent=loc,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="TSMC_City",
        desc="Identifies the city where the facility is located",
        parent=loc,
        claim=f"The facility is located in the city of {fac.city}.",
        sources=fac.location_refs,
        additional_instruction="The page(s) should explicitly mention the city for the TSMC facility."
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.state),
        id="TSMC_State_present",
        desc="TSMC state is provided",
        parent=loc,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="TSMC_State",
        desc="Identifies the state where the facility is located",
        parent=loc,
        claim=f"The facility is located in the U.S. state of {fac.state}.",
        sources=fac.location_refs,
        additional_instruction="The page(s) should explicitly mention the U.S. state for the TSMC facility."
    )

    evaluator.add_custom_node(
        result=has_urls(fac.location_refs),
        id="TSMC_Location_References_provided",
        desc="Location reference URL(s) are provided",
        parent=loc,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="TSMC_Location_References",
        desc="Provides valid reference URLs for location information",
        parent=loc,
        claim=f"The provided sources support that the TSMC facility is located in {loc_str(fac.city, fac.state)}.",
        sources=fac.location_refs,
        additional_instruction="The sources should clearly indicate city and state for the TSMC facility."
    )

    # Facility Specifications (critical)
    spec = evaluator.add_parallel(
        id="TSMC_Facility_Specifications",
        desc="Provides technical specifications and investment details for the TSMC facility",
        parent=tsmc_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.capacity_wpm),
        id="TSMC_Production_Capacity_present",
        desc="TSMC wafers-per-month capacity is provided",
        parent=spec,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="TSMC_Production_Capacity",
        desc="States the production capacity in wafers per month at full capacity",
        parent=spec,
        claim=f"The TSMC facility at {loc_str(fac.city, fac.state)} has a full production capacity of about {fac.capacity_wpm} wafers per month.",
        sources=fac.specs_refs,
        additional_instruction="Look for explicit capacity mentions such as 'wafers per month (wpm)' at full ramp."
    )

    await verify_with_urls(
        evaluator,
        node_id="TSMC_Investment_Amount",
        desc="Confirms investment meets the $10 billion threshold",
        parent=spec,
        claim=f"The total investment for the TSMC facility at {loc_str(fac.city, fac.state)} is at least $10 billion USD.",
        sources=fac.specs_refs,
        additional_instruction="Accept if the source says '≥ $10B', 'over $10B', '$12B', '$40B', etc. If multiple figures, consider total investment at the site."
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.technology_node),
        id="TSMC_Technology_Node_present",
        desc="TSMC technology node is provided",
        parent=spec,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="TSMC_Technology_Node",
        desc="Specifies the semiconductor process technology node (7nm or smaller)",
        parent=spec,
        claim=f"The TSMC facility's technology capability is {fac.technology_node}, which is 7nm or smaller.",
        sources=fac.specs_refs,
        additional_instruction="Consider nodes like 7nm, 5nm, 4nm (N4/N5), 3nm (N3) as 7nm or smaller. Confirm from the page."
    )

    evaluator.add_custom_node(
        result=has_urls(fac.specs_refs),
        id="TSMC_Specifications_References_provided",
        desc="Specifications reference URL(s) are provided",
        parent=spec,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="TSMC_Specifications_References",
        desc="Provides valid reference URLs for facility specifications",
        parent=spec,
        claim=f"The provided sources contain facility specifications (capacity, investment, and/or technology node) for TSMC at {loc_str(fac.city, fac.state)}.",
        sources=fac.specs_refs,
        additional_instruction="At least one specification (capacity, node, investment) should be explicitly supported."
    )

    # Timeline Information (critical)
    tl = evaluator.add_parallel(
        id="TSMC_Timeline_Information",
        desc="Provides construction and operational timeline details",
        parent=tsmc_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.timeline_status),
        id="TSMC_Construction_Status_present",
        desc="TSMC construction/operational status is provided",
        parent=tl,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="TSMC_Construction_Status",
        desc="Indicates whether the facility is operational or under construction",
        parent=tl,
        claim=f"The TSMC facility at {loc_str(fac.city, fac.state)} is currently '{fac.timeline_status}'.",
        sources=fac.timeline_refs,
        additional_instruction="Accept phrasing like 'under construction', 'operational', 'online', or similar status language stated in the sources."
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.timeline_date),
        id="TSMC_Completion_Date_present",
        desc="TSMC completion/operational start date is provided",
        parent=tl,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="TSMC_Completion_Date",
        desc="Provides expected completion or operational start date",
        parent=tl,
        claim=f"The expected completion or operational start for the TSMC facility is {fac.timeline_date}.",
        sources=fac.timeline_refs,
        additional_instruction="The date can be a year or month+year; confirm the timeline reference supports it."
    )

    evaluator.add_custom_node(
        result=has_urls(fac.timeline_refs),
        id="TSMC_Timeline_References_provided",
        desc="Timeline reference URL(s) are provided",
        parent=tl,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="TSMC_Timeline_References",
        desc="Provides valid reference URLs for timeline information",
        parent=tl,
        claim="The provided sources include explicit timeline details (status and/or expected completion/operation dates) for the TSMC facility.",
        sources=fac.timeline_refs,
        additional_instruction="Ensure the sources mention timeline information."
    )

    # Network Infrastructure (critical)
    net = evaluator.add_parallel(
        id="TSMC_Network_Infrastructure",
        desc="Confirms 5G network coverage at the facility location",
        parent=tsmc_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.network_carrier),
        id="TSMC_5G_Coverage_present",
        desc="TSMC 5G carrier name is provided",
        parent=net,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="TSMC_5G_Coverage",
        desc="Identifies at least one major carrier providing 5G coverage in the facility area",
        parent=net,
        claim=f"{fac.network_carrier} provides 5G coverage in or around {loc_str(fac.city, fac.state)}.",
        sources=fac.network_refs,
        additional_instruction="Accept official coverage maps or reputable confirmations. Carrier must be one of T-Mobile, AT&T, or Verizon."
    )

    evaluator.add_custom_node(
        result=has_urls(fac.network_refs),
        id="TSMC_Network_References_provided",
        desc="Network coverage reference URL(s) are provided",
        parent=net,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="TSMC_Network_References",
        desc="Provides valid reference URLs for network coverage information",
        parent=net,
        claim=f"The provided sources confirm 5G coverage by {fac.network_carrier} in the area of {loc_str(fac.city, fac.state)}.",
        sources=fac.network_refs,
        additional_instruction="A coverage map, official carrier page, or credible coverage confirmation suffices."
    )


async def verify_intel(evaluator: Evaluator, root, fac: Optional[IntelFacility]):
    intel_node = evaluator.add_parallel(
        id="Facility_from_Intel",
        desc="A semiconductor manufacturing facility operated by Intel in the United States",
        parent=root,
        critical=False
    )

    fac = fac or IntelFacility()

    # Company Identification (critical)
    comp = evaluator.add_parallel(
        id="Intel_Company_Identification",
        desc="Correctly identifies Intel as the operating company",
        parent=intel_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.company_name),
        id="Intel_Company_Name_present",
        desc="Intel company name is provided",
        parent=comp,
        critical=True
    )

    await verify_with_urls(
        evaluator,
        node_id="Intel_Company_Name",
        desc="Provides the company name: Intel Corporation",
        parent=comp,
        claim=f"The provided company name '{fac.company_name}' refers to Intel Corporation. Minor variants like 'Intel' should be considered equivalent.",
        sources=None,
        additional_instruction="Judge whether the provided name refers to Intel Corporation."
    )

    evaluator.add_custom_node(
        result=has_urls(fac.company_refs),
        id="Intel_Company_References_provided",
        desc="Reference URL(s) confirming Intel facility ownership/operation are provided",
        parent=comp,
        critical=True
    )

    await verify_with_urls(
        evaluator,
        node_id="Intel_Company_References",
        desc="Provides valid reference URLs for Intel facility information",
        parent=comp,
        claim=f"The sources confirm that Intel owns/operates or is building a semiconductor manufacturing facility in {loc_str(fac.city, fac.state)}.",
        sources=fac.company_refs,
        additional_instruction="Accept official Intel pages, reliable news, or government announcements."
    )

    # Location Details (critical)
    loc = evaluator.add_parallel(
        id="Intel_Location_Details",
        desc="Provides accurate location information for the Intel facility",
        parent=intel_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.city),
        id="Intel_City_present",
        desc="Intel city is provided",
        parent=loc,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Intel_City",
        desc="Identifies the city where the facility is located",
        parent=loc,
        claim=f"The Intel facility is located in the city of {fac.city}.",
        sources=fac.location_refs,
        additional_instruction="The page(s) should explicitly mention the city."
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.state),
        id="Intel_State_present",
        desc="Intel state is provided",
        parent=loc,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Intel_State",
        desc="Identifies the state where the facility is located",
        parent=loc,
        claim=f"The Intel facility is located in the U.S. state of {fac.state}.",
        sources=fac.location_refs,
        additional_instruction="The page(s) should explicitly mention the state."
    )

    evaluator.add_custom_node(
        result=has_urls(fac.location_refs),
        id="Intel_Location_References_provided",
        desc="Location reference URL(s) are provided",
        parent=loc,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Intel_Location_References",
        desc="Provides valid reference URLs for location information",
        parent=loc,
        claim=f"The provided sources support that the Intel facility is located in {loc_str(fac.city, fac.state)}.",
        sources=fac.location_refs,
        additional_instruction="The sources should clearly indicate the city and state."
    )

    # Facility Specifications (critical)
    spec = evaluator.add_parallel(
        id="Intel_Facility_Specifications",
        desc="Provides technical specifications and investment details for the Intel facility",
        parent=intel_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.number_of_fabs),
        id="Intel_Number_of_Fabs_present",
        desc="Intel number of fabs is provided",
        parent=spec,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Intel_Number_of_Fabs",
        desc="Specifies the number of fabrication plants planned at this location",
        parent=spec,
        claim=f"At the Intel site in {loc_str(fac.city, fac.state)}, there are {fac.number_of_fabs} planned fabrication plant(s).",
        sources=fac.specs_refs,
        additional_instruction="Confirm that the number of fabs planned is stated (e.g., 'two new fabs')."
    )

    await verify_with_urls(
        evaluator,
        node_id="Intel_Investment_Amount",
        desc="Confirms investment meets the $10 billion threshold",
        parent=spec,
        claim=f"The total investment for the Intel facility at {loc_str(fac.city, fac.state)} is at least $10 billion USD.",
        sources=fac.specs_refs,
        additional_instruction="Accept formulations like '$20B', 'over $10B', or similar; consider total site investment."
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.technology_capability),
        id="Intel_Technology_Capability_present",
        desc="Intel technology capability is provided",
        parent=spec,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Intel_Technology_Capability",
        desc="Confirms capability to produce advanced semiconductor nodes",
        parent=spec,
        claim=f"The Intel facility's technology capability includes 7nm or smaller nodes (e.g., Intel 4/Intel 3/20A/18A) as stated: {fac.technology_capability}.",
        sources=fac.specs_refs,
        additional_instruction="Treat Intel 4 (≈7nm) and smaller (Intel 3/20A/18A) as meeting the ≤7nm criterion."
    )

    evaluator.add_custom_node(
        result=has_urls(fac.specs_refs),
        id="Intel_Specifications_References_provided",
        desc="Specifications reference URL(s) are provided",
        parent=spec,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Intel_Specifications_References",
        desc="Provides valid reference URLs for facility specifications",
        parent=spec,
        claim=f"The provided sources include facility specifications (number of fabs, investment, and/or technology capability) for Intel at {loc_str(fac.city, fac.state)}.",
        sources=fac.specs_refs,
        additional_instruction="At least one of the specification items should be explicitly supported."
    )

    # Timeline Information (critical)
    tl = evaluator.add_parallel(
        id="Intel_Timeline_Information",
        desc="Provides construction and operational timeline details",
        parent=intel_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.timeline_status),
        id="Intel_Construction_Status_present",
        desc="Intel construction status is provided",
        parent=tl,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Intel_Construction_Status",
        desc="Indicates current construction status of the facility",
        parent=tl,
        claim=f"The Intel facility at {loc_str(fac.city, fac.state)} is currently '{fac.timeline_status}'.",
        sources=fac.timeline_refs,
        additional_instruction="Look for explicit phrases indicating current construction/operational status."
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.timeline_date),
        id="Intel_Expected_Completion_present",
        desc="Intel expected completion date is provided",
        parent=tl,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Intel_Expected_Completion",
        desc="Provides expected completion dates for fab(s)",
        parent=tl,
        claim=f"The expected completion or operational start for the Intel facility is {fac.timeline_date}.",
        sources=fac.timeline_refs,
        additional_instruction="The page(s) should include a date reference (year or month+year)."
    )

    evaluator.add_custom_node(
        result=has_urls(fac.timeline_refs),
        id="Intel_Timeline_References_provided",
        desc="Intel timeline reference URL(s) are provided",
        parent=tl,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Intel_Timeline_References",
        desc="Provides valid reference URLs for timeline information",
        parent=tl,
        claim="The provided sources include explicit timeline details (status and/or expected completion/operation) for the Intel facility.",
        sources=fac.timeline_refs,
        additional_instruction="Ensure the sources mention timeline information."
    )

    # Network Infrastructure (critical)
    net = evaluator.add_parallel(
        id="Intel_Network_Infrastructure",
        desc="Confirms 5G network coverage at the facility location",
        parent=intel_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.network_carrier),
        id="Intel_5G_Coverage_present",
        desc="Intel 5G carrier name is provided",
        parent=net,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Intel_5G_Coverage",
        desc="Identifies at least one major carrier providing 5G coverage in the facility area",
        parent=net,
        claim=f"{fac.network_carrier} provides 5G coverage in or around {loc_str(fac.city, fac.state)}.",
        sources=fac.network_refs,
        additional_instruction="Accept official coverage maps or credible coverage confirmations for T-Mobile, AT&T, or Verizon."
    )

    evaluator.add_custom_node(
        result=has_urls(fac.network_refs),
        id="Intel_Network_References_provided",
        desc="Intel network coverage reference URL(s) are provided",
        parent=net,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Intel_Network_References",
        desc="Provides valid reference URLs for network coverage information",
        parent=net,
        claim=f"The provided sources confirm 5G coverage by {fac.network_carrier} in the area of {loc_str(fac.city, fac.state)}.",
        sources=fac.network_refs,
        additional_instruction="A coverage map, official carrier page, or credible confirmation suffices."
    )


async def verify_samsung(evaluator: Evaluator, root, fac: Optional[SamsungFacility]):
    samsung_node = evaluator.add_parallel(
        id="Facility_from_Samsung",
        desc="A semiconductor manufacturing facility operated by Samsung in the United States",
        parent=root,
        critical=False
    )

    fac = fac or SamsungFacility()

    # Company Identification (critical)
    comp = evaluator.add_parallel(
        id="Samsung_Company_Identification",
        desc="Correctly identifies Samsung as the operating company",
        parent=samsung_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.company_name),
        id="Samsung_Company_Name_present",
        desc="Samsung company name is provided",
        parent=comp,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Samsung_Company_Name",
        desc="Provides the company name: Samsung Electronics or Samsung Austin Semiconductor",
        parent=comp,
        claim=f"The provided company name '{fac.company_name}' refers to Samsung Electronics or Samsung Austin Semiconductor (SAS).",
        sources=None,
        additional_instruction="Judge whether the provided name clearly refers to Samsung Electronics or SAS."
    )

    evaluator.add_custom_node(
        result=has_urls(fac.company_refs),
        id="Samsung_Company_References_provided",
        desc="Reference URL(s) confirming Samsung facility ownership/operation are provided",
        parent=comp,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Samsung_Company_References",
        desc="Provides valid reference URLs for Samsung facility information",
        parent=comp,
        claim=f"The sources confirm that Samsung owns/operates or is building a semiconductor manufacturing facility in {loc_str(fac.city, fac.state)}.",
        sources=fac.company_refs,
        additional_instruction="Accept official Samsung pages, reliable news, or government announcements."
    )

    # Location Details (critical)
    loc = evaluator.add_parallel(
        id="Samsung_Location_Details",
        desc="Provides accurate location information for the Samsung facility",
        parent=samsung_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.city),
        id="Samsung_City_present",
        desc="Samsung city is provided",
        parent=loc,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Samsung_City",
        desc="Identifies the city where the facility is located",
        parent=loc,
        claim=f"The Samsung facility is located in the city of {fac.city}.",
        sources=fac.location_refs,
        additional_instruction="The page(s) should explicitly mention the city."
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.state),
        id="Samsung_State_present",
        desc="Samsung state is provided",
        parent=loc,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Samsung_State",
        desc="Identifies the state where the facility is located",
        parent=loc,
        claim=f"The Samsung facility is located in the U.S. state of {fac.state}.",
        sources=fac.location_refs,
        additional_instruction="The page(s) should explicitly mention the state."
    )

    evaluator.add_custom_node(
        result=has_urls(fac.location_refs),
        id="Samsung_Location_References_provided",
        desc="Location reference URL(s) are provided",
        parent=loc,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Samsung_Location_References",
        desc="Provides valid reference URLs for location information",
        parent=loc,
        claim=f"The provided sources support that the Samsung facility is located in {loc_str(fac.city, fac.state)}.",
        sources=fac.location_refs,
        additional_instruction="The sources should clearly indicate the city and state."
    )

    # Facility Specifications (critical)
    spec = evaluator.add_parallel(
        id="Samsung_Facility_Specifications",
        desc="Provides technical specifications and investment details for the Samsung facility",
        parent=samsung_node,
        critical=True
    )

    await verify_with_urls(
        evaluator,
        node_id="Samsung_Investment_Amount",
        desc="Confirms investment meets the $10 billion threshold",
        parent=spec,
        claim=f"The total investment for the Samsung facility at {loc_str(fac.city, fac.state)} is at least $10 billion USD.",
        sources=fac.specs_refs,
        additional_instruction="Accept formulations like '$17B', 'over $10B', etc.; consider total site investment."
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.facility_type),
        id="Samsung_Facility_Type_present",
        desc="Samsung facility type is provided",
        parent=spec,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Samsung_Facility_Type",
        desc="Specifies the type of semiconductor manufacturing facility",
        parent=spec,
        claim=f"The Samsung facility type is '{fac.facility_type}'.",
        sources=fac.specs_refs,
        additional_instruction="Look for phrases like 'foundry', 'advanced logic fab', 'semiconductor manufacturing facility', etc."
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.production_focus),
        id="Samsung_Production_Focus_present",
        desc="Samsung production focus/technology is provided",
        parent=spec,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Samsung_Production_Focus",
        desc="Describes the production focus or technology of the facility",
        parent=spec,
        claim=f"The Samsung facility's production focus/technology is '{fac.production_focus}', and it is 7nm or smaller (advanced node).",
        sources=fac.specs_refs,
        additional_instruction="Accept mentions of advanced nodes like 7nm, 5nm, 4nm, 3nm (e.g., '4nm GAA')."
    )

    evaluator.add_custom_node(
        result=has_urls(fac.specs_refs),
        id="Samsung_Specifications_References_provided",
        desc="Specifications reference URL(s) are provided",
        parent=spec,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Samsung_Specifications_References",
        desc="Provides valid reference URLs for facility specifications",
        parent=spec,
        claim=f"The provided sources include specifications (investment, facility type, and/or production focus) for the Samsung facility at {loc_str(fac.city, fac.state)}.",
        sources=fac.specs_refs,
        additional_instruction="At least one of these items should be explicitly supported."
    )

    # Timeline Information (critical)
    tl = evaluator.add_parallel(
        id="Samsung_Timeline_Information",
        desc="Provides construction and operational timeline details",
        parent=samsung_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.timeline_status),
        id="Samsung_Construction_Status_present",
        desc="Samsung construction/operational status is provided",
        parent=tl,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Samsung_Construction_Status",
        desc="Indicates current construction or operational status",
        parent=tl,
        claim=f"The Samsung facility at {loc_str(fac.city, fac.state)} is currently '{fac.timeline_status}'.",
        sources=fac.timeline_refs,
        additional_instruction="Accept phrasing like 'under construction', 'operational', etc., as stated."
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.timeline_date),
        id="Samsung_Expected_Operation_present",
        desc="Samsung expected operational start date is provided",
        parent=tl,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Samsung_Expected_Operation",
        desc="Provides expected operational start date or completion timeline",
        parent=tl,
        claim=f"The expected operational start/completion for the Samsung facility is {fac.timeline_date}.",
        sources=fac.timeline_refs,
        additional_instruction="The date can be year or month+year; confirm from the page(s)."
    )

    evaluator.add_custom_node(
        result=has_urls(fac.timeline_refs),
        id="Samsung_Timeline_References_provided",
        desc="Samsung timeline reference URL(s) are provided",
        parent=tl,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Samsung_Timeline_References",
        desc="Provides valid reference URLs for timeline information",
        parent=tl,
        claim="The provided sources include timeline details (status and/or expected completion/operation) for the Samsung facility.",
        sources=fac.timeline_refs,
        additional_instruction="Ensure the sources mention timeline information."
    )

    # Network Infrastructure (critical)
    net = evaluator.add_parallel(
        id="Samsung_Network_Infrastructure",
        desc="Confirms 5G network coverage at the facility location",
        parent=samsung_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(fac.network_carrier),
        id="Samsung_5G_Coverage_present",
        desc="Samsung 5G carrier name is provided",
        parent=net,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Samsung_5G_Coverage",
        desc="Identifies at least one major carrier providing 5G coverage in the facility area",
        parent=net,
        claim=f"{fac.network_carrier} provides 5G coverage in or around {loc_str(fac.city, fac.state)}.",
        sources=fac.network_refs,
        additional_instruction="Accept official carrier coverage pages or credible confirmations for T-Mobile, AT&T, or Verizon."
    )

    evaluator.add_custom_node(
        result=has_urls(fac.network_refs),
        id="Samsung_Network_References_provided",
        desc="Samsung network coverage reference URL(s) are provided",
        parent=net,
        critical=True
    )
    await verify_with_urls(
        evaluator,
        node_id="Samsung_Network_References",
        desc="Provides valid reference URLs for network coverage information",
        parent=net,
        claim=f"The provided sources confirm 5G coverage by {fac.network_carrier} in the area of {loc_str(fac.city, fac.state)}.",
        sources=fac.network_refs,
        additional_instruction="A coverage map, official carrier page, or credible coverage confirmation suffices."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict[str, Any]:
    """
    Entry point to evaluate an answer for the semiconductor facilities task.
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
        default_model=model
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction"
    )

    # Optional contextual info
    evaluator.add_ground_truth({
        "required_companies": ["TSMC (Taiwan Semiconductor Manufacturing Company)", "Intel Corporation", "Samsung Electronics / Samsung Austin Semiconductor"],
        "constraints": {
            "country": "United States",
            "min_investment_usd": ">= $10B",
            "advanced_node": "7nm or smaller",
            "timeline": "publicly announced status/date",
            "network": "5G coverage by T-Mobile / AT&T / Verizon"
        }
    })

    # Build verification subtrees
    await verify_tsmc(evaluator, root, extracted.tsmc)
    await verify_intel(evaluator, root, extracted.intel)
    await verify_samsung(evaluator, root, extracted.samsung)

    # Return structured result
    return evaluator.get_summary()