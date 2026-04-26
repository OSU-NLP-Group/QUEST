import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_tech_infra_facilities_3"
TASK_DESCRIPTION = """
Identify three major technology infrastructure facilities in the United States that each meet ALL of the following requirements:

Geographic Requirements:
- Must be located in the United States
- Provide the specific state and city/county location

Facility Type and Scale:
- Must be either a hyperscale data center OR a semiconductor manufacturing facility (fab)
- Must be operated by or built for a major technology company or recognized infrastructure provider
- For data centers: minimum power capacity of 100 MW
- For semiconductor fabs: must have 300mm wafer production capability

Investment Requirements:
- Minimum announced investment of $1 billion USD for the facility or campus
- Investment must be announced or committed between 2020-2026

Operational Status:
- Must be either currently operational, under construction, or with construction formally approved/announced
- For non-operational facilities: specify expected completion or production start date

Technical Specifications:
- Specify the primary purpose or production type (e.g., AI/cloud computing for data centers, or technology node/chip type for semiconductor fabs)
- Provide specific capacity metrics (power capacity in MW for data centers, OR production capacity/technology node for fabs)

Sustainability (recommended but not required):
- Include any evidence of sustainability commitments, renewable energy plans, or efficiency targets if available

For each facility, provide:
1. Facility name and operator/owner
2. Complete location (city/county and state)
3. Facility type (data center or semiconductor fab)
4. Investment amount and announcement date
5. Operational status and timeline
6. Technical specifications and capacity metrics
7. Supporting URL references for all major claims
"""


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class FacilityURLs(BaseModel):
    identification: List[str] = Field(default_factory=list)
    location: List[str] = Field(default_factory=list)
    type_and_scale: List[str] = Field(default_factory=list)
    investment: List[str] = Field(default_factory=list)
    status: List[str] = Field(default_factory=list)
    technical: List[str] = Field(default_factory=list)
    sustainability: List[str] = Field(default_factory=list)
    all: List[str] = Field(default_factory=list)


class FacilityItem(BaseModel):
    # Identification
    facility_name: Optional[str] = None
    operator: Optional[str] = None  # operator/owner or "built for" entity

    # Location
    location_state: Optional[str] = None
    location_city_or_county: Optional[str] = None
    location_country: Optional[str] = None  # should be "United States" / "USA" etc.

    # Facility type & scale
    facility_type: Optional[str] = None  # expected values like "data center" or "semiconductor fab"
    power_capacity_mw: Optional[str] = None  # string as extracted (e.g., "120 MW", "200+ MW")
    wafer_size_mm: Optional[str] = None  # e.g., "300mm"
    technology_node: Optional[str] = None  # e.g., "5nm", "3nm"
    production_capacity: Optional[str] = None  # textual capacity (e.g., "100k wpm")

    # Investment
    investment_amount: Optional[str] = None  # textual amount (e.g., "$1.2 billion")
    investment_announcement_date: Optional[str] = None  # textual date (e.g., "March 2023")

    # Status & timeline
    status: Optional[str] = None  # "operational", "under construction", "approved/announced", etc.
    timeline: Optional[str] = None  # expected completion/production start if not operational

    # Technical purpose
    purpose: Optional[str] = None  # e.g., "AI/cloud computing", "advanced semiconductor manufacturing"

    # Sustainability (optional)
    sustainability_summary: Optional[str] = None

    # URLs
    urls: FacilityURLs = Field(default_factory=FacilityURLs)


class FacilitiesExtraction(BaseModel):
    facilities: List[FacilityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return """
Extract up to five facilities described in the answer that match the task. For each facility, extract EXACTLY the following fields from the answer:

- facility_name: The name used in the answer (string or null)
- operator: The operator/owner or the company for whom the facility is being built (string or null)

- location_state: US state (string or null)
- location_city_or_county: City OR county as mentioned (string or null)
- location_country: Country (string or null)

- facility_type: Either "data center" (or similar wording) OR "semiconductor fab" (or synonyms like "fabrication plant", "foundry"). Use the exact wording from the answer when possible (string or null)

- power_capacity_mw: If a data center, extract the stated total/campus power capacity in MW as text (e.g., "120 MW", "over 150 MW") (string or null)
- wafer_size_mm: If a fab, extract wafer size text if present (e.g., "300mm", "12-inch") (string or null)
- technology_node: If a fab, extract node text if present (e.g., "5nm", "3nm") (string or null)
- production_capacity: If a fab, extract any production capacity metric text (e.g., "100k wafer starts per month") (string or null)

- investment_amount: The announced/committed investment amount as text (e.g., "$1.5 billion") (string or null)
- investment_announcement_date: The announcement or commitment date as text (string or null)

- status: One of "operational", "under construction", "approved/announced", or other short status text as in the answer (string or null)
- timeline: If not operational, extract expected completion or production start date text (string or null)

- purpose: The primary purpose or production type (e.g., "AI/cloud computing", "advanced semiconductor manufacturing") (string or null)
- sustainability_summary: Any sustainability/renewable/efficiency info (string or null)

- urls:
  - identification: URLs (array) that specifically support facility name and operator
  - location: URLs (array) that support the geographic location
  - type_and_scale: URLs (array) that support facility type and scale (MW or 300mm)
  - investment: URLs (array) that support investment amount and date
  - status: URLs (array) that support operational status/timeline
  - technical: URLs (array) that support primary purpose and capacity metrics
  - sustainability: URLs (array) that support sustainability claims
  - all: ALL URLs for this facility from the answer (array). It should include every URL cited for this facility.

RULES:
- Only extract information explicitly present in the answer text.
- For any field not present, set it to null (or empty array for URL lists).
- URLs must be actual URLs cited in the answer (plain or markdown links). Do not infer or invent URLs.
Return JSON as:
{
  "facilities": [
    {
      ... fields above ...
    },
    ...
  ]
}
If the answer contains more than three facilities, still extract them, we will use the first three later.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_type(facility_type: Optional[str]) -> Optional[str]:
    if not facility_type:
        return None
    s = facility_type.strip().lower()
    if "data" in s and "center" in s:
        return "data_center"
    if "fab" in s or "foundry" in s or "fabrication" in s or "semiconductor" in s:
        return "semiconductor_fab"
    return None


def _pick_sources(*url_lists: List[str]) -> List[str]:
    """Pick the first non-empty URL list."""
    for lst in url_lists:
        if lst:
            return lst
    return []


def _non_empty_text(x: Optional[str]) -> bool:
    return bool(x and str(x).strip())


# --------------------------------------------------------------------------- #
# Verification builders per facility                                          #
# --------------------------------------------------------------------------- #
async def verify_facility(
    evaluator: Evaluator,
    parent_node,
    facility: FacilityItem,
    index_1based: int,
) -> None:
    """
    Build verification tree and run checks for a single facility.
    """
    f_id = f"facility_{index_1based}"

    # Create a facility-level container (parallel, non-critical as a whole)
    facility_node = evaluator.add_parallel(
        id=f_id,
        desc=f"{['First','Second','Third','Fourth','Fifth'][index_1based-1] if index_1based<=5 else f'#{index_1based}'} qualifying facility identification and verification",
        parent=parent_node,
        critical=False
    )

    urls = facility.urls or FacilityURLs()
    typed_type = _normalize_type(facility.facility_type)
    is_dc = typed_type == "data_center"
    is_fab = typed_type == "semiconductor_fab"

    # ------------------ Identification ------------------
    ident_node = evaluator.add_parallel(
        id=f"{f_id}_identification",
        desc=f"Facility name and operator identification for {f_id}",
        parent=facility_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_text(facility.facility_name),
        id=f"{f_id}_name_provided",
        desc=f"Facility name must be clearly stated",
        parent=ident_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_text(facility.operator),
        id=f"{f_id}_operator_provided",
        desc=f"Operator or owner name must be clearly stated and must be a major technology company or recognized infrastructure provider",
        parent=ident_node,
        critical=True
    )

    ident_ref_leaf = evaluator.add_leaf(
        id=f"{f_id}_identification_reference",
        desc="URL reference confirming facility name and operator",
        parent=ident_node,
        critical=True
    )
    ident_sources = _pick_sources(urls.identification, urls.all)
    ident_claim = (
        f"The cited source(s) confirm that the facility named '{facility.facility_name or ''}' "
        f"is operated by or built for '{facility.operator or ''}', and that this operator is a major technology company "
        f"or a recognized infrastructure/data center provider."
    )
    await evaluator.verify(
        claim=ident_claim,
        node=ident_ref_leaf,
        sources=ident_sources,
        additional_instruction="Accept reasonable name/operator variants (e.g., LLC/Inc. suffixes). "
                               "Recognized providers include companies like Amazon, Google, Microsoft, Meta, Apple, Nvidia, "
                               "TSMC, Intel, Micron, Samsung, GlobalFoundries, Texas Instruments, Qualcomm, Broadcom, "
                               "as well as large colocation/DC providers such as Equinix, Digital Realty, QTS, Switch, etc."
    )

    # ------------------ Location ------------------
    loc_node = evaluator.add_parallel(
        id=f"{f_id}_location",
        desc=f"Geographic and location requirements for {f_id}",
        parent=facility_node,
        critical=True
    )

    us_loc_leaf = evaluator.add_leaf(
        id=f"{f_id}_us_location",
        desc="Facility must be located in the United States",
        parent=loc_node,
        critical=True
    )
    loc_sources = _pick_sources(urls.location, urls.identification, urls.all)
    claim_us = "The facility is located in the United States."
    await evaluator.verify(
        claim=claim_us,
        node=us_loc_leaf,
        sources=loc_sources,
        additional_instruction="Treat it as located in the US if a US city/state is clearly shown."
    )

    evaluator.add_custom_node(
        result=_non_empty_text(facility.location_state),
        id=f"{f_id}_state_specified",
        desc="Specific US state location must be identified",
        parent=loc_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_text(facility.location_city_or_county),
        id=f"{f_id}_city_specified",
        desc="Specific city or county location must be identified",
        parent=loc_node,
        critical=True
    )

    loc_ref_leaf = evaluator.add_leaf(
        id=f"{f_id}_location_reference",
        desc="URL reference confirming geographic location",
        parent=loc_node,
        critical=True
    )
    loc_desc = f"{(facility.location_city_or_county or '').strip()}, {(facility.location_state or '').strip()} in the United States"
    claim_loc = f"The cited source(s) confirm that the facility is located in {loc_desc}."
    await evaluator.verify(
        claim=claim_loc,
        node=loc_ref_leaf,
        sources=loc_sources,
        additional_instruction="Allow minor formatting variations (e.g., county vs. city)."
    )

    # ------------------ Type and Scale ------------------
    tas_node = evaluator.add_parallel(
        id=f"{f_id}_type_and_scale",
        desc=f"Facility type classification and scale requirements for {f_id}",
        parent=facility_node,
        critical=True
    )

    type_leaf = evaluator.add_leaf(
        id=f"{f_id}_type_classification",
        desc="Facility must be classified as either a hyperscale data center or a semiconductor manufacturing facility (fab)",
        parent=tas_node,
        critical=True
    )
    tas_sources = _pick_sources(urls.type_and_scale, urls.technical, urls.identification, urls.all)
    if is_dc:
        claim_type = "The facility is a hyperscale data center (or a large-scale data center)."
    elif is_fab:
        claim_type = "The facility is a semiconductor manufacturing facility (fab/foundry/fabrication plant)."
    else:
        claim_type = "The facility is either a hyperscale data center or a semiconductor manufacturing fab."
    await evaluator.verify(
        claim=claim_type,
        node=type_leaf,
        sources=tas_sources,
        additional_instruction="Accept synonyms (e.g., 'foundry', 'fabrication plant', 'wafer fab' for fabs; 'hyperscale' optional for large data centers)."
    )

    scale_leaf = evaluator.add_leaf(
        id=f"{f_id}_scale_requirement",
        desc="For data centers: minimum 100 MW power capacity OR for semiconductor fabs: 300mm wafer production capability",
        parent=tas_node,
        critical=True
    )
    if is_dc:
        claim_scale = "The facility has a power capacity of at least 100 MW (facility or campus total)."
    elif is_fab:
        claim_scale = "The fab has 300mm (12-inch) wafer production capability."
    else:
        claim_scale = "The facility meets one of the following: data center power capacity >= 100 MW OR semiconductor fab with 300mm wafer production."
    await evaluator.verify(
        claim=claim_scale,
        node=scale_leaf,
        sources=tas_sources,
        additional_instruction="For data centers, aggregated campus capacity or 'over 100 MW' counts. "
                               "For fabs, phrases like '300mm', '12-inch wafers' count."
    )

    type_ref_leaf = evaluator.add_leaf(
        id=f"{f_id}_type_reference",
        desc="URL reference confirming facility type and scale specifications",
        parent=tas_node,
        critical=True
    )
    claim_tas_ref = "The cited sources explicitly support the facility's type and the specific scale requirement stated above."
    await evaluator.verify(
        claim=claim_tas_ref,
        node=type_ref_leaf,
        sources=tas_sources,
        additional_instruction="Confirm both type and the applicable scale metric."
    )

    # ------------------ Investment ------------------
    inv_node = evaluator.add_parallel(
        id=f"{f_id}_investment",
        desc=f"Investment and economic impact requirements for {f_id}",
        parent=facility_node,
        critical=True
    )

    inv_sources = _pick_sources(urls.investment, urls.identification, urls.all)

    inv_thresh_leaf = evaluator.add_leaf(
        id=f"{f_id}_investment_threshold",
        desc="Minimum announced investment of $1 billion USD for the facility or campus",
        parent=inv_node,
        critical=True
    )
    claim_inv_thresh = "The announced or committed investment for this facility/campus is at least $1 billion USD."
    await evaluator.verify(
        claim=claim_inv_thresh,
        node=inv_thresh_leaf,
        sources=inv_sources,
        additional_instruction="Consider numbers like '$1B', '$1 billion', 'USD 1 billion', or amounts greater than or equal to $1B."
    )

    inv_time_leaf = evaluator.add_leaf(
        id=f"{f_id}_investment_timeframe",
        desc="Investment must be announced or committed between 2020-2026",
        parent=inv_node,
        critical=True
    )
    inv_date_hint = facility.investment_announcement_date or ""
    claim_time = (
        f"The investment for this facility was announced or committed between 2020 and 2026 inclusive. "
        f"If available, the announcement date is '{inv_date_hint}'."
    )
    await evaluator.verify(
        claim=claim_time,
        node=inv_time_leaf,
        sources=inv_sources,
        additional_instruction="Treat valid dates in the range 2020-01-01 to 2026-12-31 (inclusive) as compliant."
    )

    inv_ref_leaf = evaluator.add_leaf(
        id=f"{f_id}_investment_reference",
        desc="URL reference confirming investment amount and timeframe",
        parent=inv_node,
        critical=True
    )
    claim_inv_ref = "The cited sources support both the stated investment amount and that it was announced/committed during 2020–2026."
    await evaluator.verify(
        claim=claim_inv_ref,
        node=inv_ref_leaf,
        sources=inv_sources,
        additional_instruction="Both amount (>= $1B) and the 2020–2026 timeframe must be supported."
    )

    # ------------------ Operational Status ------------------
    # Parent set to non-critical due to mixed criticality of children (timeline is non-critical).
    op_node = evaluator.add_parallel(
        id=f"{f_id}_operational_status",
        desc=f"Operational status and timeline requirements for {f_id}",
        parent=facility_node,
        critical=False
    )

    status_sources = _pick_sources(urls.status, urls.identification, urls.all)

    status_leaf = evaluator.add_leaf(
        id=f"{f_id}_status_verification",
        desc="Facility must be either currently operational, in construction, or with construction approved/announced",
        parent=op_node,
        critical=True
    )
    status_txt = (facility.status or "").strip().lower()
    if status_txt:
        claim_status = f"The facility is '{status_txt}' (interpretable as currently operational, under construction, or construction approved/announced)."
    else:
        claim_status = "The facility is currently operational, under construction, or construction has been formally approved/announced."
    await evaluator.verify(
        claim=claim_status,
        node=status_leaf,
        sources=status_sources,
        additional_instruction="Treat clear statements of operations, construction activity, or formal approvals/announcements as compliant."
    )

    # Timeline (non-critical existence if not operational)
    need_timeline = (status_txt and status_txt not in ("operational", "in operation", "operating"))
    evaluator.add_custom_node(
        result=(True if not need_timeline else _non_empty_text(facility.timeline)),
        id=f"{f_id}_timeline_specified",
        desc="For non-operational facilities: expected completion or production start date must be specified",
        parent=op_node,
        critical=False
    )

    status_ref_leaf = evaluator.add_leaf(
        id=f"{f_id}_status_reference",
        desc="URL reference confirming operational status",
        parent=op_node,
        critical=True
    )
    claim_status_ref = "The cited sources confirm the facility's operational status and, if applicable, mention construction/approval."
    await evaluator.verify(
        claim=claim_status_ref,
        node=status_ref_leaf,
        sources=status_sources,
        additional_instruction="Cross-check the status language on the page."
    )

    # ------------------ Technical Capabilities ------------------
    tech_node = evaluator.add_parallel(
        id=f"{f_id}_technical_capabilities",
        desc=f"Technical specifications and capabilities for {f_id}",
        parent=facility_node,
        critical=True
    )

    tech_sources = _pick_sources(urls.technical, urls.type_and_scale, urls.identification, urls.all)

    primary_purpose_leaf = evaluator.add_leaf(
        id=f"{f_id}_primary_purpose",
        desc="Primary purpose or production type must be specified (e.g., AI/cloud computing, advanced semiconductor manufacturing, etc.)",
        parent=tech_node,
        critical=True
    )
    if _non_empty_text(facility.purpose):
        claim_purpose = f"The sources indicate the facility's primary purpose/production type is '{facility.purpose}'."
    else:
        # If not provided in answer, verify that the sources state a clear purpose typical for such facilities.
        claim_purpose = "The sources state a clear primary purpose/production type for this facility (e.g., AI/cloud computing for data centers, or advanced semiconductor manufacturing for fabs)."
    await evaluator.verify(
        claim=claim_purpose,
        node=primary_purpose_leaf,
        sources=tech_sources,
        additional_instruction="Minor wording differences are acceptable if the meaning matches."
    )

    capacity_leaf = evaluator.add_leaf(
        id=f"{f_id}_capacity_metric",
        desc="For data centers: power capacity in MW OR for fabs: production capacity or technology node",
        parent=tech_node,
        critical=True
    )
    if is_dc:
        if _non_empty_text(facility.power_capacity_mw):
            claim_capacity = f"The sources mention a power capacity metric for this data center, specifically '{facility.power_capacity_mw}'."
        else:
            claim_capacity = "The sources mention a specific power capacity (in MW) for this data center."
    elif is_fab:
        if _non_empty_text(facility.technology_node) or _non_empty_text(facility.wafer_size_mm) or _non_empty_text(facility.production_capacity):
            metrics_text = ", ".join(
                [t for t in [facility.technology_node, facility.wafer_size_mm, facility.production_capacity] if _non_empty_text(t)]
            )
            claim_capacity = f"The sources mention fab capacity metrics such as technology node/wafer size/throughput, specifically: {metrics_text}."
        else:
            claim_capacity = "The sources mention fab capacity metrics such as technology node (e.g., 5nm), wafer size (300mm), or production throughput."
    else:
        claim_capacity = "The sources provide specific capacity metrics appropriate to the facility type."
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_leaf,
        sources=tech_sources,
        additional_instruction="For data centers, look for MW figures; for fabs, look for node, wafer size (300mm/12-inch), or throughput."
    )

    tech_ref_leaf = evaluator.add_leaf(
        id=f"{f_id}_technical_reference",
        desc="URL reference confirming technical specifications",
        parent=tech_node,
        critical=True
    )
    claim_tech_ref = "The cited sources support the facility's technical specifications and capacity metrics."
    await evaluator.verify(
        claim=claim_tech_ref,
        node=tech_ref_leaf,
        sources=tech_sources,
        additional_instruction="Confirm that at least one concrete technical detail is present and aligned with the answer."
    )

    # ------------------ Sustainability (optional) ------------------
    sus_node = evaluator.add_parallel(
        id=f"{f_id}_sustainability",
        desc=f"Environmental and sustainability considerations for {f_id}",
        parent=facility_node,
        critical=False
    )

    sus_sources = _pick_sources(urls.sustainability, urls.all)

    sus_commit_leaf = evaluator.add_leaf(
        id=f"{f_id}_sustainability_commitment",
        desc="Evidence of sustainability commitment, renewable energy plans, or efficiency targets",
        parent=sus_node,
        critical=False
    )
    claim_sus = "The sources mention sustainability commitments, renewable energy plans, or efficiency/ PUE targets related to this facility."
    await evaluator.verify(
        claim=claim_sus,
        node=sus_commit_leaf,
        sources=sus_sources,
        additional_instruction="Any credible sustainability-related statement for the facility counts."
    )

    sus_ref_leaf = evaluator.add_leaf(
        id=f"{f_id}_sustainability_reference",
        desc="URL reference for sustainability information",
        parent=sus_node,
        critical=False
    )
    claim_sus_ref = "The cited source(s) are relevant sustainability references for this facility."
    await evaluator.verify(
        claim=claim_sus_ref,
        node=sus_ref_leaf,
        sources=sus_sources,
        additional_instruction="Verify the link is actually about sustainability aspects of this specific facility or campus."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 'US tech infrastructure facilities' task using the Mind2Web2 evaluation framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # facilities evaluated independently
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

    # Root should be non-critical to allow partial credit across facilities
    # (We intentionally override JSON root criticality to satisfy framework constraints and allow partial credit.)
    root.critical = False

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction"
    )

    # Prepare up to 3 facilities (pad with placeholders if fewer)
    facilities: List[FacilityItem] = list(extracted.facilities[:3])
    while len(facilities) < 3:
        facilities.append(FacilityItem())

    # Build verification subtrees for each of the three required facilities
    for i, fac in enumerate(facilities, start=1):
        await verify_facility(evaluator, root, fac, i)

    # Return summary
    return evaluator.get_summary()