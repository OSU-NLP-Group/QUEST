import asyncio
import logging
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "us_mfg_2025"
TASK_DESCRIPTION = """
In 2025, several major semiconductor and technology manufacturing initiatives were announced or became operational in the United States as part of efforts to strengthen domestic production capabilities. Identify two specific manufacturing facilities or partnerships that meet the following criteria:

1. Arizona Facility: Identify a semiconductor manufacturing facility located in Arizona that was operational or producing chips in 2025. The facility must:
   - Produce advanced semiconductors or AI chips
   - Serve at least one major technology company as a client
   - Represent significant scale (either through multi-billion dollar investment or production capacity of thousands of wafers per month)

2. Texas Partnership: Identify a technology manufacturing facility or partnership located in Texas that was announced in 2025. The partnership must:
   - Produce AI systems, semiconductors, or advanced electronics
   - Involve a named manufacturing partner company
   - Specify the city within Texas where it is located

For each facility or partnership, provide:
- The facility/partnership name
- The production technology or product type
- The major client or partner company
- Relevant scale or investment information
- The specific city location (for Texas)
- A reference URL that verifies the information
"""


# ----------------------------- Data Models -------------------------------- #
class ArizonaFacility(BaseModel):
    name: Optional[str] = None
    location_state: Optional[str] = None
    city: Optional[str] = None
    production_type: Optional[str] = None
    major_client: Optional[str] = None
    operational_2025: Optional[str] = None  # statement or phrase that indicates operational/production in 2025
    scale_info: Optional[str] = None  # investment amount or capacity details
    reference_urls: List[str] = Field(default_factory=list)


class TexasPartnership(BaseModel):
    name: Optional[str] = None
    location_state: Optional[str] = None
    city: Optional[str] = None
    production_type: Optional[str] = None
    manufacturing_partner: Optional[str] = None
    major_company: Optional[str] = None
    announced_2025: Optional[str] = None  # statement or phrase indicating 2025 announcement
    scale_or_investment_info: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class ManufacturingInitiativesExtraction(BaseModel):
    arizona: Optional[ArizonaFacility] = None
    texas: Optional[TexasPartnership] = None


# --------------------------- Extraction Prompt ---------------------------- #
def prompt_extract_initiatives() -> str:
    return """
    Extract exactly one Arizona semiconductor facility and exactly one Texas facility/partnership as presented in the answer, prioritizing the first qualifying items mentioned. Return a JSON object with two top-level fields: 'arizona' and 'texas'.

    For 'arizona', extract:
    - name: the facility name or identifier
    - location_state: the state (should be "Arizona")
    - city: the city (if mentioned)
    - production_type: the production technology or product type (e.g., 3nm, AI chips, HBM, advanced packaging)
    - major_client: name of at least one major technology company served as a client (e.g., Apple, NVIDIA, AMD)
    - operational_2025: a phrase or statement indicating it was operational or producing chips in 2025
    - scale_info: a phrase indicating significant scale (e.g., multi-billion dollar investment OR capacity of thousands of wafers per month)
    - reference_urls: list of URLs that verify the Arizona facility information (include all URLs mentioned for this facility)

    For 'texas', extract:
    - name: the facility or partnership name/identifier
    - location_state: the state (should be "Texas")
    - city: the specific city in Texas
    - production_type: the production technology or product type (AI systems, semiconductors, advanced electronics)
    - manufacturing_partner: the named manufacturing partner company involved (e.g., Foxconn, Samsung)
    - major_company: the major client or partner company associated (beyond just location)
    - announced_2025: a phrase indicating the announcement occurred in 2025
    - scale_or_investment_info: relevant scale or investment information (e.g., dollar amount, capacity, size)
    - reference_urls: list of URLs that verify the Texas facility/partnership information (include all URLs mentioned for this item)

    Rules:
    - Extract only what is explicitly stated in the answer.
    - If a field is not mentioned, set it to null (for strings) or [] for URLs.
    - For URLs, include full explicit URLs (plain or markdown links); ignore malformed URLs.
    - If multiple items are given per state, choose the first that appears to satisfy the criteria.
    """


# ---------------------------- Helper Functions ---------------------------- #
def _safe(val: Optional[str]) -> str:
    return val.strip() if isinstance(val, str) and val.strip() else ""


def _sources_or_none(urls: Optional[List[str]]) -> List[str] | None:
    if not urls:
        return None
    # Filter obviously invalid
    cleaned = [u for u in urls if isinstance(u, str) and u.strip()]
    return cleaned if cleaned else None


# --------------------------- Verification Logic --------------------------- #
async def verify_arizona_facility(evaluator: Evaluator, parent_node, az: Optional[ArizonaFacility]) -> None:
    az_node = evaluator.add_parallel(
        id="arizona_facility",
        desc="Arizona semiconductor manufacturing facility that meets the Arizona criteria and includes all required outputs.",
        parent=parent_node,
        critical=False
    )

    # Reference URL existence first to gate subsequent verifications via preconditions
    az_urls = az.reference_urls if az else []
    ref_url_result = bool(az_urls)
    evaluator.add_custom_node(
        result=ref_url_result,
        id="az_reference_url",
        desc="Provides at least one reference URL that verifies the Arizona facility information presented.",
        parent=az_node,
        critical=True
    )

    # Facility name provided
    evaluator.add_custom_node(
        result=bool(_safe(az.name) if az else False),
        id="az_facility_name",
        desc="Provides the name/identifier of the Arizona semiconductor manufacturing facility.",
        parent=az_node,
        critical=True
    )

    # Facility is located in Arizona
    loc_node = evaluator.add_leaf(
        id="az_location_state",
        desc="Facility is located in Arizona.",
        parent=az_node,
        critical=True
    )
    loc_claim = f"The facility '{_safe(az.name) if az else ''}' is located in the state of Arizona."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=_sources_or_none(az_urls),
        additional_instruction="Verify the facility location on the provided pages. Mentions of Arizona or Arizona cities (e.g., Phoenix) are acceptable."
    )

    # Operational or producing chips in 2025
    op_node = evaluator.add_leaf(
        id="az_operational_2025",
        desc="Facility was operational or producing chips in 2025.",
        parent=az_node,
        critical=True
    )
    op_claim = "This facility was operational or producing chips in 2025."
    await evaluator.verify(
        claim=op_claim,
        node=op_node,
        sources=_sources_or_none(az_urls),
        additional_instruction="Check timeline statements on the source pages for 2025 operation or production (including pilot, ramp, or regular production)."
    )

    # Production type provided and qualifies as advanced semiconductors or AI chips
    prod_node = evaluator.add_leaf(
        id="az_production_type_provided_and_advanced",
        desc="Provides the production technology or product type, and it qualifies as advanced semiconductors or AI chips.",
        parent=az_node,
        critical=True
    )
    prod_claim = f"The facility produces '{_safe(az.production_type) if az else ''}', which qualifies as advanced semiconductors or AI chips."
    await evaluator.verify(
        claim=prod_claim,
        node=prod_node,
        sources=_sources_or_none(az_urls),
        additional_instruction="Look for terms like '2nm/3nm/5nm', 'HBM', 'CoWoS', 'advanced packaging', 'AI accelerators/GPUs' to qualify as advanced or AI chips."
    )

    # Major technology client served
    client_node = evaluator.add_leaf(
        id="az_major_client",
        desc="Identifies at least one major technology company served as a client.",
        parent=az_node,
        critical=True
    )
    client_claim = f"The facility serves {_safe(az.major_client) if az else ''} as a client, which is a major technology company."
    await evaluator.verify(
        claim=client_claim,
        node=client_node,
        sources=_sources_or_none(az_urls),
        additional_instruction="Confirm that at least one named major tech company (e.g., Apple, NVIDIA, AMD, etc.) is a client of this facility."
    )

    # Significant scale threshold
    scale_node = evaluator.add_leaf(
        id="az_scale_threshold",
        desc="Demonstrates significant scale via either multi-billion-dollar investment OR production capacity of thousands of wafers per time period.",
        parent=az_node,
        critical=True
    )
    scale_claim = (
        f"The facility meets a significant scale threshold via multi-billion-dollar investment "
        f"or production capacity of thousands of wafers per month. Details: '{_safe(az.scale_info) if az else ''}'."
    )
    await evaluator.verify(
        claim=scale_claim,
        node=scale_node,
        sources=_sources_or_none(az_urls),
        additional_instruction="Accept the claim if sources indicate investment ≥ $1B (multi-billion) or capacity of thousands of wafers per month/quarter. Look for explicit numeric statements."
    )


async def verify_texas_partnership(evaluator: Evaluator, parent_node, tx: Optional[TexasPartnership]) -> None:
    tx_node = evaluator.add_parallel(
        id="texas_partnership",
        desc="Texas manufacturing facility/partnership that meets the Texas criteria and includes all required outputs.",
        parent=parent_node,
        critical=False
    )

    # Reference URL existence first
    tx_urls = tx.reference_urls if tx else []
    evaluator.add_custom_node(
        result=bool(tx_urls),
        id="tx_reference_url",
        desc="Provides at least one reference URL that verifies the Texas facility/partnership information presented.",
        parent=tx_node,
        critical=True
    )

    # Partnership/facility name provided
    evaluator.add_custom_node(
        result=bool(_safe(tx.name) if tx else False),
        id="tx_partnership_name",
        desc="Provides the name/identifier of the Texas facility or partnership.",
        parent=tx_node,
        critical=True
    )

    # Located in Texas
    tx_loc_node = evaluator.add_leaf(
        id="tx_location_state",
        desc="Facility/partnership is located in Texas.",
        parent=tx_node,
        critical=True
    )
    tx_loc_claim = f"The facility/partnership '{_safe(tx.name) if tx else ''}' is located in the state of Texas."
    await evaluator.verify(
        claim=tx_loc_claim,
        node=tx_loc_node,
        sources=_sources_or_none(tx_urls),
        additional_instruction="Verify that the state is Texas; mentions of Texas or Texas cities are acceptable."
    )

    # City specified (existence check)
    evaluator.add_custom_node(
        result=bool(_safe(tx.city) if tx else False),
        id="tx_city_specified",
        desc="Specifies the city within Texas where the facility/partnership is located.",
        parent=tx_node,
        critical=True
    )

    # Announced in 2025
    tx_ann_node = evaluator.add_leaf(
        id="tx_announced_2025",
        desc="Facility/partnership was announced in 2025.",
        parent=tx_node,
        critical=True
    )
    tx_ann_claim = "This Texas facility/partnership was announced in 2025."
    await evaluator.verify(
        claim=tx_ann_claim,
        node=tx_ann_node,
        sources=_sources_or_none(tx_urls),
        additional_instruction="Confirm press release, announcement article, or official documentation dated in 2025."
    )

    # Production type provided and qualifies as AI systems, semiconductors, or advanced electronics
    tx_prod_node = evaluator.add_leaf(
        id="tx_production_type_provided_and_advanced",
        desc="Provides the production technology or product type, and it qualifies as AI systems, semiconductors, or advanced electronics.",
        parent=tx_node,
        critical=True
    )
    tx_prod_claim = f"The Texas facility/partnership produces '{_safe(tx.production_type) if tx else ''}', which qualifies as AI systems, semiconductors, or advanced electronics."
    await evaluator.verify(
        claim=tx_prod_claim,
        node=tx_prod_node,
        sources=_sources_or_none(tx_urls),
        additional_instruction="Look for terms like AI systems, semiconductor manufacturing/packaging, HBM, advanced electronics, or similar."
    )

    # Named manufacturing partner company involved
    tx_partner_node = evaluator.add_leaf(
        id="tx_named_manufacturing_partner",
        desc="Names at least one manufacturing partner company involved in the Texas partnership.",
        parent=tx_node,
        critical=True
    )
    tx_partner_claim = f"The partnership involves a named manufacturing partner company: '{_safe(tx.manufacturing_partner) if tx else ''}'."
    await evaluator.verify(
        claim=tx_partner_claim,
        node=tx_partner_node,
        sources=_sources_or_none(tx_urls),
        additional_instruction="Confirm that a manufacturing partner company is explicitly named (e.g., Foxconn, Samsung, etc.)."
    )

    # Major client or partner company identified
    tx_major_node = evaluator.add_leaf(
        id="tx_major_client_or_partner_company",
        desc="Identifies the major client and/or partner company associated with the Texas facility/partnership (beyond just the location).",
        parent=tx_node,
        critical=True
    )
    tx_major_claim = f"The major client or partner company associated with the partnership is '{_safe(tx.major_company) if tx else ''}'."
    await evaluator.verify(
        claim=tx_major_claim,
        node=tx_major_node,
        sources=_sources_or_none(tx_urls),
        additional_instruction="Confirm at least one named major company associated beyond the manufacturing partner, if applicable."
    )

    # Scale or investment information provided
    tx_scale_node = evaluator.add_leaf(
        id="tx_scale_or_investment_info",
        desc="Provides relevant scale or investment information for the Texas facility/partnership.",
        parent=tx_node,
        critical=True
    )
    tx_scale_claim = f"Scale or investment information is provided: '{_safe(tx.scale_or_investment_info) if tx else ''}'."
    await evaluator.verify(
        claim=tx_scale_claim,
        node=tx_scale_node,
        sources=_sources_or_none(tx_urls),
        additional_instruction="Confirm the presence of relevant scale or investment details (e.g., dollar amounts, capacity, footprint)."
    )


# ------------------------- Main Evaluation Function ------------------------ #
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

    # Extract structured data for Arizona and Texas entries
    extracted = await evaluator.extract(
        prompt=prompt_extract_initiatives(),
        template_class=ManufacturingInitiativesExtraction,
        extraction_name="manufacturing_initiatives"
    )

    # Build verification subtrees
    await verify_arizona_facility(evaluator, root, extracted.arizona)
    await verify_texas_partnership(evaluator, root, extracted.texas)

    # Return summary
    return evaluator.get_summary()