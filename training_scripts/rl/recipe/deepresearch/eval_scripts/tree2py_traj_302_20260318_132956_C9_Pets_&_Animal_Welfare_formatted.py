import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wildlife_rehab_centers_us_three_states"
TASK_DESCRIPTION = (
    "Identify three wildlife rehabilitation centers in the United States, one in each of Colorado, California, and New York. "
    "For each center, provide comprehensive operational information demonstrating professional wildlife rehabilitation standards: "
    "state licensing, species categories (mammals, birds, indication about reptiles/amphibians), emergency contact phone, operating schedule, "
    "physical location in-state, general contact method, service area, coordination/recognition by state wildlife agency, public accessibility "
    "for wildlife emergencies, emergency response protocol/intake process, and a website/official listing URL supporting the information. "
    "Centers must be legitimate, operational, accept wildlife from the public, and maintain proper state licensing/permits."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CenterInfo(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    website_urls: List[str] = Field(default_factory=list)
    support_urls: List[str] = Field(default_factory=list)

    license_statement: Optional[str] = None
    mammals_statement: Optional[str] = None
    birds_statement: Optional[str] = None
    reptiles_amphibians_statement: Optional[str] = None  # e.g., “we handle reptiles” or “we do not accept reptiles”

    emergency_phone: Optional[str] = None
    operating_hours: Optional[str] = None
    address: Optional[str] = None
    general_contact: List[str] = Field(default_factory=list)  # phone/email/contact-form URL

    service_area: Optional[str] = None
    state_agency_coordination: Optional[str] = None
    public_accessibility: Optional[str] = None
    emergency_protocol: Optional[str] = None


class ThreeCentersExtraction(BaseModel):
    colorado: Optional[CenterInfo] = None
    california: Optional[CenterInfo] = None
    new_york: Optional[CenterInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_centers() -> str:
    return """
Extract exactly one wildlife rehabilitation center for each of the following states from the provided answer text: Colorado, California, and New York.

For each state, extract the following as a JSON object with the fields defined below. Only extract information explicitly present in the answer text. Do not invent or infer missing fields.

For each center, extract:
- name: The organization/center name.
- state: The state name (e.g., "Colorado", "California", "New York").
- website_urls: Array of center website URLs or official listings explicitly included in the answer. Include only URLs present in the answer text.
- support_urls: Array of any additional URLs from the answer that support licensing or operations (e.g., state agency listings).
- license_statement: A quoted or paraphrased line from the answer indicating state wildlife rehabilitation license or permit.
- mammals_statement: A quoted/paraphrased statement showing they rehabilitate mammals (if present).
- birds_statement: A quoted/paraphrased statement showing they rehabilitate birds (if present).
- reptiles_amphibians_statement: A statement that indicates whether they handle reptiles and/or amphibians (explicitly "yes" or "no"/"do not accept" both acceptable).
- emergency_phone: An emergency contact phone number for wildlife emergencies (if provided).
- operating_hours: Operating hours, days, or availability schedule (if provided).
- address: Physical address or clear in-state location description (e.g., "Boulder, CO" or a street address).
- general_contact: Array with at least one general contact method (phone, email, or a contact form URL) if present in the answer.
- service_area: Geographic service area/coverage region.
- state_agency_coordination: A statement showing coordination/recognition with the appropriate state wildlife agency.
- public_accessibility: A line showing the public can reach the center (e.g., instructions for people who find injured wildlife).
- emergency_protocol: A statement describing emergency response protocol or intake process.

Output JSON shape:
{
  "colorado": { CenterInfo fields ... } or null,
  "california": { CenterInfo fields ... } or null,
  "new_york": { CenterInfo fields ... } or null
}

Rules:
- Return null for a state if the answer did not provide a center for it.
- For URL fields, include only valid URLs explicitly present in the answer. Do not invent any URLs.
- If a field is missing in the answer, set it to null or an empty list as appropriate.
"""


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _combine_urls(info: Optional[CenterInfo]) -> List[str]:
    urls: List[str] = []
    if not info:
        return urls
    for u in (info.website_urls or []):
        if u and u not in urls:
            urls.append(u)
    for u in (info.support_urls or []):
        if u and u not in urls:
            urls.append(u)
    return urls


async def _verify_with_urls_or_fail(
    evaluator: Evaluator,
    node,
    claim: str,
    urls: List[str],
    additional_instruction: str,
) -> None:
    if urls:
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction=additional_instruction,
        )
    else:
        node.score = 0.0
        node.status = "failed"


def _state_meta(state_key: str) -> Dict[str, str]:
    if state_key == "colorado":
        return {
            "node_id": "Colorado_Center",
            "desc": "A wildlife rehabilitation center located in Colorado that meets all specified requirements",
            "state_full": "Colorado",
            "agency": "Colorado Parks & Wildlife",
            "abbr": "CO",
            "agency_short": "CPW",
        }
    if state_key == "california":
        return {
            "node_id": "California_Center",
            "desc": "A wildlife rehabilitation center located in California that meets all specified requirements",
            "state_full": "California",
            "agency": "California Department of Fish and Wildlife",
            "abbr": "CA",
            "agency_short": "CDFW",
        }
    if state_key == "new_york":
        return {
            "node_id": "New_York_Center",
            "desc": "A wildlife rehabilitation center located in New York that meets all specified requirements",
            "state_full": "New York",
            "agency": "New York State Department of Environmental Conservation",
            "abbr": "NY",
            "agency_short": "DEC",
        }
    return {
        "node_id": f"{state_key}_Center",
        "desc": f"A wildlife rehabilitation center located in {state_key} that meets all specified requirements",
        "state_full": state_key,
        "agency": "State wildlife agency",
        "abbr": state_key[:2].upper(),
        "agency_short": "AGENCY",
    }


# --------------------------------------------------------------------------- #
# Verification for one center (one state)                                     #
# --------------------------------------------------------------------------- #
async def verify_center_for_state(
    evaluator: Evaluator,
    parent_node,
    state_key: str,
    info: Optional[CenterInfo],
) -> None:
    meta = _state_meta(state_key)
    state_node = evaluator.add_parallel(
        id=meta["node_id"],
        desc=meta["desc"],
        parent=parent_node,
        critical=False  # allow partial credit across states
    )

    all_urls = _combine_urls(info)
    name = info.name if info and info.name else "the organization"
    state_full = meta["state_full"]
    agency = meta["agency"]
    abbr = meta["abbr"]
    agency_short = meta["agency_short"]

    # 1) State License
    leaf = evaluator.add_leaf(
        id=f"{abbr}_State_License",
        desc=f"The center holds a valid {state_full} wildlife rehabilitation license or permit issued by {agency}",
        parent=state_node,
        critical=True,
    )
    claim = f"For {name} in {state_full}, there is evidence that it holds a valid wildlife rehabilitation license/permit issued by {agency}."
    await _verify_with_urls_or_fail(
        evaluator,
        leaf,
        claim,
        all_urls,
        additional_instruction=(
            f"Verify on the provided pages that the organization explicitly indicates having a valid state wildlife rehabilitation license/permit "
            f"from {agency} (e.g., 'licensed/permitted by {agency}', a permit number, or by appearing on an official {agency_short} listing). "
            f"Minor wording variations are acceptable; ensure the agency and state match {state_full}."
        ),
    )

    # 2) Mammals
    leaf = evaluator.add_leaf(
        id=f"{abbr}_Mammals",
        desc="The center explicitly states they rehabilitate mammals",
        parent=state_node,
        critical=True,
    )
    claim = f"{name} explicitly indicates that it rehabilitates mammals."
    await _verify_with_urls_or_fail(
        evaluator,
        leaf,
        claim,
        all_urls,
        additional_instruction="Look for species categories, intake instructions, or services that clearly include mammals."
    )

    # 3) Birds
    leaf = evaluator.add_leaf(
        id=f"{abbr}_Birds",
        desc="The center explicitly states they rehabilitate birds",
        parent=state_node,
        critical=True,
    )
    claim = f"{name} explicitly indicates that it rehabilitates birds."
    await _verify_with_urls_or_fail(
        evaluator,
        leaf,
        claim,
        all_urls,
        additional_instruction="Look for species categories, avian pages, raptors, songbirds, or text that clearly includes birds."
    )

    # 4) Emergency Contact
    leaf = evaluator.add_leaf(
        id=f"{abbr}_Emergency_Contact",
        desc="The center provides an emergency contact phone number",
        parent=state_node,
        critical=True,
    )
    claim = f"{name} provides an emergency contact phone number for wildlife emergencies."
    await _verify_with_urls_or_fail(
        evaluator,
        leaf,
        claim,
        all_urls,
        additional_instruction="Accept clear emergency phone labels such as 'Emergency', 'Wildlife Emergency', 'Hotline', or similar."
    )

    # 5) Operating Hours
    leaf = evaluator.add_leaf(
        id=f"{abbr}_Operating_Hours",
        desc="The center specifies operating hours, days, or availability schedule",
        parent=state_node,
        critical=True,
    )
    claim = f"{name} specifies its operating hours, days, or availability schedule."
    await _verify_with_urls_or_fail(
        evaluator,
        leaf,
        claim,
        all_urls,
        additional_instruction="Look for 'Hours', 'Open', '24/7', 'After-hours', or text that describes when they accept calls/intakes."
    )

    # 6) Physical Location
    leaf = evaluator.add_leaf(
        id=f"{abbr}_Physical_Location",
        desc=f"The center provides a physical address or location in {state_full}",
        parent=state_node,
        critical=True,
    )
    addr_hint = info.address if info and info.address else ""
    claim = f"{name} provides a physical address or location in {state_full}. Example/Hint: {addr_hint}"
    await _verify_with_urls_or_fail(
        evaluator,
        leaf,
        claim,
        all_urls,
        additional_instruction=f"Confirm that the listed address/location is within {state_full} (city/county or street address in-state is acceptable)."
    )

    # 7) General Contact
    leaf = evaluator.add_leaf(
        id=f"{abbr}_General_Contact",
        desc="The center provides general contact information (phone, email, or contact form)",
        parent=state_node,
        critical=True,
    )
    claim = f"{name} provides at least one general contact method such as phone, email, or a contact form (not only emergency)."
    await _verify_with_urls_or_fail(
        evaluator,
        leaf,
        claim,
        all_urls,
        additional_instruction="Look for any general inquiry method, including non-emergency phone, an email address, or a contact form link."
    )

    # 8) Service Area
    leaf = evaluator.add_leaf(
        id=f"{abbr}_Service_Area",
        desc="The center indicates their service area or geographic coverage",
        parent=state_node,
        critical=True,
    )
    claim = f"{name} states its geographic service area or coverage region."
    await _verify_with_urls_or_fail(
        evaluator,
        leaf,
        claim,
        all_urls,
        additional_instruction="Look for phrases like 'serves', 'service area', counties/regions covered, or similar wording."
    )

    # 9) State Agency Coordination
    coord_leaf_id = {
        "CO": f"{abbr}_CPW_Coordination",
        "CA": f"{abbr}_CDFW_Coordination",
        "NY": f"{abbr}_DEC_Coordination"
    }.get(abbr, f"{abbr}_Agency_Coordination")

    leaf = evaluator.add_leaf(
        id=coord_leaf_id,
        desc=f"The center demonstrates coordination with or recognition by {agency}",
        parent=state_node,
        critical=True,
    )
    claim = f"{name} shows coordination with or recognition by {agency} (e.g., permitted/licensed by the agency, or listed on an official agency page)."
    await _verify_with_urls_or_fail(
        evaluator,
        leaf,
        claim,
        all_urls,
        additional_instruction=f"Accept explicit mentions of {agency}, permit numbers, 'licensed/permitted by', or an official {agency} listing that names the organization."
    )

    # 10) Public Accessibility
    leaf = evaluator.add_leaf(
        id=f"{abbr}_Public_Accessibility",
        desc="The center has publicly accessible contact information for public wildlife emergencies",
        parent=state_node,
        critical=True,
    )
    claim = f"{name} provides publicly accessible contact information for members of the public who find injured wildlife."
    await _verify_with_urls_or_fail(
        evaluator,
        leaf,
        claim,
        all_urls,
        additional_instruction="Look for clear directions to the public to call or contact the center when finding injured wildlife."
    )

    # 11) Emergency Response Protocol
    leaf = evaluator.add_leaf(
        id=f"{abbr}_Emergency_Protocol",
        desc="The center describes their emergency response protocol or intake process",
        parent=state_node,
        critical=True,
    )
    claim = f"{name} describes its emergency response protocol or intake process for handling wildlife emergencies."
    await _verify_with_urls_or_fail(
        evaluator,
        leaf,
        claim,
        all_urls,
        additional_instruction="Look for instructions such as 'what to do', intake drop-off directions, call-first guidance, or triage steps."
    )

    # 12) Reptile/Amphibian indication
    leaf = evaluator.add_leaf(
        id=f"{abbr}_Reptile_Amphibian",
        desc="The center indicates whether they handle reptiles or amphibians",
        parent=state_node,
        critical=True,
    )
    claim = f"{name} indicates whether it handles reptiles and/or amphibians (either accepts them or explicitly does not)."
    await _verify_with_urls_or_fail(
        evaluator,
        leaf,
        claim,
        all_urls,
        additional_instruction="Either a clear 'yes/accepts' or 'no/does not accept' for reptiles/amphibians satisfies this requirement, as long as it is explicit."
    )

    # 13) URL Reference
    url_ref_leaf = evaluator.add_leaf(
        id=f"{abbr}_URL_Reference",
        desc="A valid URL reference supporting the center's information is provided",
        parent=state_node,
        critical=True,
    )
    # Use a single primary URL if available (prefer website_urls first)
    primary_url: Optional[str] = None
    if info:
        if info.website_urls:
            primary_url = info.website_urls[0]
        elif info.support_urls:
            primary_url = info.support_urls[0]

    url_claim = f"The provided URL is the official website or an official listing for {name} in {state_full}."
    if primary_url:
        await evaluator.verify(
            claim=url_claim,
            node=url_ref_leaf,
            sources=primary_url,
            additional_instruction="Confirm that the page is the organization's own website or an official state/municipal/partner listing naming the organization."
        )
    else:
        url_ref_leaf.score = 0.0
        url_ref_leaf.status = "failed"


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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # top-level parallel across states
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_centers(),
        template_class=ThreeCentersExtraction,
        extraction_name="wildlife_centers_extraction",
    )

    # Optional ground truth-like meta for context
    evaluator.add_ground_truth({
        "required_states": ["Colorado", "California", "New York"],
        "agencies": {
            "Colorado": "Colorado Parks & Wildlife",
            "California": "California Department of Fish and Wildlife",
            "New York": "New York State Department of Environmental Conservation"
        }
    })

    # Add the task node (non-critical to allow partial credit across states given framework constraint)
    task_node = evaluator.add_parallel(
        id="Wildlife_Rehabilitation_Centers_Task",
        desc="Identify three wildlife rehabilitation centers, one in each of Colorado, California, and New York, that meet comprehensive operational and licensing requirements",
        parent=root,
        critical=False
    )

    # Verify each state center
    await verify_center_for_state(evaluator, task_node, "colorado", extracted.colorado)
    await verify_center_for_state(evaluator, task_node, "california", extracted.california)
    await verify_center_for_state(evaluator, task_node, "new_york", extracted.new_york)

    return evaluator.get_summary()