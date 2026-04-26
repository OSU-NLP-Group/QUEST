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
TASK_ID = "career_centers_accessibility_verification"
TASK_DESCRIPTION = """
A student organization is creating a resource guide for members who need career counseling services. They want to identify career centers at the following institutions that meet specific accessibility and service criteria: Indiana University of Pennsylvania, Grayson College, and Dallas College (including its Mountain View campus).

For each institution's career center, determine whether it meets ALL of the following requirements:
1. Has publicly documented specific operating hours (actual days and times, not just 'contact us for hours')
2. Offers drop-in or walk-in services where students can receive assistance without a pre-scheduled appointment
3. Explicitly provides mock interview services or interview preparation assistance
4. Provides complete contact information including a full physical address (building name and street address), direct phone number, and email address
5. Is located in Pennsylvania, Texas, or Ohio
6. All information is verifiable through official institutional websites or publicly accessible sources

Identify which career center(s) meet ALL six requirements and provide the following information for each qualifying center:
- Institution name and career center name
- Complete physical address
- Phone number and email address
- Specific operating hours (regular hours and drop-in hours if different)
- Confirmation of mock interview service availability with supporting reference URL
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CenterExtraction(BaseModel):
    # Identity
    institution: Optional[str] = None
    center_name: Optional[str] = None
    center_homepage_url: Optional[str] = None

    # Sources (by facet)
    general_source_urls: List[str] = Field(default_factory=list)

    operating_hours_text: Optional[str] = None
    operating_hours_urls: List[str] = Field(default_factory=list)

    dropin_services_text: Optional[str] = None
    dropin_hours_text: Optional[str] = None  # if provided distinctly
    dropin_urls: List[str] = Field(default_factory=list)

    mock_interview_text: Optional[str] = None
    mock_interview_urls: List[str] = Field(default_factory=list)

    # Contact details
    address_full: Optional[str] = None
    building_name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    contact_urls: List[str] = Field(default_factory=list)

    # Qualification claim in the answer
    qualifies_all_six: Optional[bool] = None
    qualifies_rationale: Optional[str] = None

    # Dallas-specific
    dallas_mountain_view_included: Optional[bool] = None


class CentersExtraction(BaseModel):
    iup: Optional[CenterExtraction] = None
    grayson: Optional[CenterExtraction] = None
    dallas: Optional[CenterExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_centers() -> str:
    return """
    Extract the structured information that the answer provides for each of the following institutions' career centers:
    - iup (Indiana University of Pennsylvania)
    - grayson (Grayson College)
    - dallas (Dallas College, explicitly including Mountain View campus if mentioned)

    IMPORTANT RULES:
    - Extract ONLY what is explicitly present in the answer text. Do not infer or invent.
    - When extracting URLs, return full absolute URLs as they appear (plain or in markdown).
    - If a field is not provided in the answer, return null (for strings) or an empty list (for URL arrays).

    For each institution object (iup, grayson, dallas), extract the following fields:
    1) Identity:
       - institution: the institution name as written in the answer (can be abbreviations like "IUP" if used in the answer)
       - center_name: the specific career center/service unit name
       - center_homepage_url: a main official page URL for the center/unit, if present
       - general_source_urls: list of any other general/source URLs the answer cites for this center

    2) Operating hours:
       - operating_hours_text: the specific operating hours text provided (days / times)
       - operating_hours_urls: list of URLs where the hours are documented

    3) Drop-in/walk-in services:
       - dropin_services_text: text indicating drop-in/walk-in/express service availability (no appointment required)
       - dropin_hours_text: drop-in hours text if distinct from regular hours (else null)
       - dropin_urls: list of URLs that document drop-in/walk-in

    4) Mock interviews:
       - mock_interview_text: text confirming mock interview/interview preparation availability
       - mock_interview_urls: list of URLs confirming mock interview/interview prep

    5) Contact information:
       - address_full: the full address string as written in the answer
       - building_name: building name (if provided as a distinct part)
       - street_address: street address (if provided as a distinct part)
       - city, state, zip_code: as provided
       - phone: direct phone number
       - email: direct email address
       - contact_urls: list of URLs that show contact info

    6) Qualification claim in the answer:
       - qualifies_all_six: boolean true/false if the answer EXPLICITLY claims the center meets ALL SIX requirements; null if not explicitly stated
       - qualifies_rationale: brief snippet from the answer that supports the claim (if any)

    7) Dallas-specific:
       - dallas_mountain_view_included: boolean true/false ONLY for Dallas; set to true if the answer explicitly says the Dallas College evaluation includes the Mountain View campus; otherwise false if explicitly not included; null if unclear or not explicitly addressed.

    Return a single JSON object with keys "iup", "grayson", and "dallas", each being a CenterExtraction object as defined. If the answer provides nothing for an institution, still include the object with all fields null/empty.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _dedup(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not _non_empty(u):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def candidate_urls(center: Optional[CenterExtraction], prefer: Optional[str] = None) -> List[str]:
    """
    Build a candidate URL list for verification, preferring facet-specific URLs,
    and falling back to general/contact/homepage URLs.
    """
    if not center:
        return []
    facet: List[str] = []
    if prefer == "hours":
        facet = center.operating_hours_urls
    elif prefer == "dropin":
        facet = center.dropin_urls
    elif prefer == "mock":
        facet = center.mock_interview_urls
    elif prefer == "contact":
        facet = center.contact_urls

    base = []
    if _non_empty(center.center_homepage_url):
        base.append(center.center_homepage_url)

    all_urls = facet + center.general_source_urls + center.contact_urls + base
    return _dedup(all_urls)


def address_is_complete(center: Optional[CenterExtraction]) -> bool:
    """
    Consider address 'complete' if both building_name and street_address are provided
    OR if address_full is present and both building and street could be reasonably parsed into separate fields by the extractor.
    We rely on extraction to fill building_name and street_address when possible.
    """
    if not center:
        return False
    if _non_empty(center.building_name) and _non_empty(center.street_address):
        return True
    # Fallback: if address_full present and either city/state present, still require street_address OR building_name
    if _non_empty(center.address_full):
        # Conservative: require also at least one of building_name or street_address separately present
        if _non_empty(center.building_name) or _non_empty(center.street_address):
            return True
    return False


def all_contact_fields_present(center: Optional[CenterExtraction]) -> bool:
    if not center:
        return False
    return address_is_complete(center) and _non_empty(center.phone) and _non_empty(center.email)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_criteria(
    evaluator: Evaluator,
    inst_node,
    criteria_parent,
    inst_key: str,
    center: Optional[CenterExtraction],
) -> None:
    """
    Build 6 requirement checks (R1-R6) for a given institution, and verify with URLs where applicable.
    This node is critical in the rubric.
    """
    center_label = (center.center_name or "the career center") if center else "the career center"
    inst_label = (center.institution or inst_key) if center else inst_key

    # R1: Specific operating hours (days and times)
    r1 = evaluator.add_leaf(
        id=f"{inst_key}_R1_Specific_Operating_Hours",
        desc="Center has publicly documented specific operating hours (actual days and times; not only 'contact us for hours').",
        parent=criteria_parent,
        critical=True,
    )
    r1_claim = f"{center_label} at {inst_label} publishes specific operating hours with explicit days and times on the cited page(s)."
    await evaluator.verify(
        claim=r1_claim,
        node=r1,
        sources=candidate_urls(center, "hours"),
        additional_instruction="Confirm the page shows concrete days and times (e.g., Mon–Fri 8am–5pm). Statements like 'contact us for hours' do NOT satisfy this."
    )

    # R2: Drop-in / Walk-in
    r2 = evaluator.add_leaf(
        id=f"{inst_key}_R2_DropIn_WalkIn",
        desc="Center explicitly offers drop-in/walk-in/express services without a pre-scheduled appointment.",
        parent=criteria_parent,
        critical=True,
    )
    r2_claim = f"{center_label} at {inst_label} explicitly offers drop-in or walk-in services (no appointment required)."
    await evaluator.verify(
        claim=r2_claim,
        node=r2,
        sources=candidate_urls(center, "dropin"),
        additional_instruction="Look for exact phrases like 'drop-in', 'walk-in', 'express advising', or equivalent statements that clearly indicate no appointment is required."
    )

    # R3: Mock interviews / interview prep
    r3 = evaluator.add_leaf(
        id=f"{inst_key}_R3_Mock_Interviews",
        desc="Center explicitly provides mock interview services or interview preparation assistance.",
        parent=criteria_parent,
        critical=True,
    )
    r3_claim = f"{center_label} at {inst_label} provides mock interview services or interview preparation assistance."
    await evaluator.verify(
        claim=r3_claim,
        node=r3,
        sources=candidate_urls(center, "mock"),
        additional_instruction="Verify that the page explicitly mentions 'mock interview', 'practice interview', or structured interview preparation offered by the center."
    )

    # R4: Complete contact info on official page(s)
    r4 = evaluator.add_leaf(
        id=f"{inst_key}_R4_Complete_Contact_Info",
        desc="Center provides complete contact info: full physical address (building + street), direct phone number, and email address.",
        parent=criteria_parent,
        critical=True,
    )
    r4_claim = f"The cited official page(s) for {center_label} at {inst_label} list a full physical address including building name and street address, a direct phone number, and an email address."
    await evaluator.verify(
        claim=r4_claim,
        node=r4,
        sources=candidate_urls(center, "contact"),
        additional_instruction="The page should show all three: (1) a building name AND street address, (2) a direct phone number, and (3) an email address for the center."
    )

    # R5: Allowed state (PA, TX, or OH)
    r5 = evaluator.add_leaf(
        id=f"{inst_key}_R5_Allowed_State",
        desc="Center is located in Pennsylvania, Texas, or Ohio.",
        parent=criteria_parent,
        critical=True,
    )
    r5_claim = f"{center_label} at {inst_label} is located in Pennsylvania, Texas, or Ohio."
    await evaluator.verify(
        claim=r5_claim,
        node=r5,
        sources=candidate_urls(center, "contact"),
        additional_instruction="Determine the state from the address or explicit campus location references on the cited official page(s)."
    )

    # R6: Verifiable via official or public sources
    r6 = evaluator.add_leaf(
        id=f"{inst_key}_R6_Verifiable_Public_Sources",
        desc="All claimed services and operational details are verifiable via official institutional webpages or other publicly accessible sources.",
        parent=criteria_parent,
        critical=True,
    )
    # Build a union of all URLs we have
    all_urls = candidate_urls(center, None)
    r6_claim = f"The cited URLs for {center_label} at {inst_label} are official institutional webpages or publicly accessible pages that document the services and operational details (hours, drop-in, mock interviews, contact)."
    await evaluator.verify(
        claim=r6_claim,
        node=r6,
        sources=all_urls,
        additional_instruction="Check whether the provided URLs are official (.edu or institution-owned subdomains) or clearly public sources that directly substantiate the claims."
    )


def _add_center_identified_node(
    evaluator: Evaluator, parent_node, inst_key: str, desc: str, center: Optional[CenterExtraction]
):
    identified = bool(center) and (_non_empty(center.center_name) or _non_empty(center.center_homepage_url))
    return evaluator.add_custom_node(
        result=identified,
        id=f"{inst_key}_Center_Identified",
        desc=desc,
        parent=parent_node,
        critical=True
    )


def _add_qualifying_status_stated_node(
    evaluator: Evaluator, parent_node, inst_key: str, center: Optional[CenterExtraction]
):
    stated = bool(center) and (center.qualifies_all_six is not None)
    return evaluator.add_custom_node(
        result=stated,
        id=f"{inst_key}_Qualifying_Status_Stated",
        desc=f"Response indicates whether the {inst_key} center meets ALL six requirements (qualifying vs not qualifying / not confirmed).",
        parent=parent_node,
        critical=True
    )


def _add_dallas_mountain_view_node(
    evaluator: Evaluator, parent_node, center: Optional[CenterExtraction]
):
    included = bool(center and center.dallas_mountain_view_included is True)
    return evaluator.add_custom_node(
        result=included,
        id="Dallas_MountainView_Included",
        desc="Response’s Dallas College evaluation explicitly includes the Mountain View campus.",
        parent=parent_node,
        critical=True
    )


def _add_qualifies_true_guard(
    evaluator: Evaluator, parent_node, inst_key: str, center: Optional[CenterExtraction]
):
    """
    A guard node to control relevance of output-fields checks.
    If the answer claims the center qualifies (meets all six), pass; otherwise this fails and,
    due to sequential aggregation, subsequent output-field checks will be skipped.
    """
    is_qualifying = bool(center and center.qualifies_all_six is True)
    return evaluator.add_custom_node(
        result=is_qualifying,
        id=f"{inst_key}_Qualifies_True_Guard",
        desc=f"Guard: response claims the {inst_key} center qualifies (meets all six) — required to proceed to output fields.",
        parent=parent_node,
        critical=False  # Non‑critical; used only to gate subsequent steps in the sequential chain
    )


def _add_output_fields_checks(
    evaluator: Evaluator, parent_node, inst_key: str, center: Optional[CenterExtraction]
):
    out_parent = evaluator.add_parallel(
        id=f"{inst_key}_If_Qualifying_Output_Fields",
        desc=f"If the response claims the {inst_key} center qualifies, the response provides all required output fields for that center.",
        parent=parent_node,
        critical=True
    )

    # All these nodes judge presence-in-answer only (output fields), hence custom nodes are appropriate.
    inst_center_ok = evaluator.add_custom_node(
        result=bool(center and _non_empty(center.institution) and _non_empty(center.center_name)),
        id=f"{inst_key}_Out_Institution_And_Center_Name",
        desc="Provides institution name and career center name.",
        parent=out_parent,
        critical=True
    )
    address_ok = evaluator.add_custom_node(
        result=address_is_complete(center),
        id=f"{inst_key}_Out_Address",
        desc="Provides complete physical address (building + street address).",
        parent=out_parent,
        critical=True
    )
    phone_ok = evaluator.add_custom_node(
        result=bool(center and _non_empty(center.phone)),
        id=f"{inst_key}_Out_Phone",
        desc="Provides a direct phone number.",
        parent=out_parent,
        critical=True
    )
    email_ok = evaluator.add_custom_node(
        result=bool(center and _non_empty(center.email)),
        id=f"{inst_key}_Out_Email",
        desc="Provides an email address.",
        parent=out_parent,
        critical=True
    )
    hours_ok = evaluator.add_custom_node(
        result=bool(center and _non_empty(center.operating_hours_text)),
        id=f"{inst_key}_Out_Operating_Hours",
        desc="Provides specific operating hours (and drop-in hours if different).",
        parent=out_parent,
        critical=True
    )
    mock_url_ok = evaluator.add_custom_node(
        result=bool(center and len(center.mock_interview_urls) > 0),
        id=f"{inst_key}_Out_MockInterview_Confirmation_With_URL",
        desc="Provides confirmation of mock interview/interview prep availability and includes a supporting reference URL.",
        parent=out_parent,
        critical=True
    )

    return out_parent


async def verify_institution_block(
    evaluator: Evaluator,
    parent,
    inst_key: str,
    center: Optional[CenterExtraction],
    is_dallas: bool = False,
):
    """
    Build the full evaluation subtree for a given institution.
    """
    # 1) Institution sequential evaluation container
    inst_node = evaluator.add_sequential(
        id=f"{inst_key}_Evaluation",
        desc=f"{('Dallas College (including Mountain View campus)' if is_dallas else inst_key.replace('_', ' ').title())}: evaluate the career center identified in the response.",
        parent=parent,
        critical=False
    )

    # 2) Center identified
    _add_center_identified_node(
        evaluator,
        inst_node,
        inst_key if not is_dallas else "Dallas",
        "Response identifies a specific career center/unit being evaluated (name or unambiguous unit/page).",
        center
    )

    # Dallas-specific requirement about Mountain View campus
    if is_dallas:
        _add_dallas_mountain_view_node(evaluator, inst_node, center)

    # 3) Criteria checks (R1–R6), critical
    criteria_parent = evaluator.add_parallel(
        id=f"{inst_key}_Criteria_Check",
        desc=f"Check whether the identified {('Dallas' if is_dallas else inst_key)} center meets each of the six stated requirements.",
        parent=inst_node,
        critical=True
    )
    await build_and_verify_criteria(
        evaluator=evaluator,
        inst_node=inst_node,
        criteria_parent=criteria_parent,
        inst_key=("Dallas" if is_dallas else inst_key),
        center=center
    )

    # 4) Qualifying status stated
    _add_qualifying_status_stated_node(
        evaluator,
        inst_node,
        inst_key if not is_dallas else "Dallas",
        center
    )

    # 5) Guard for output fields (only proceed if the answer claims 'qualifies_all_six' == True)
    _add_qualifies_true_guard(
        evaluator,
        inst_node,
        inst_key if not is_dallas else "Dallas",
        center
    )

    # 6) If qualifying: required output fields provided (parallel, critical)
    _add_output_fields_checks(
        evaluator,
        inst_node,
        inst_key if not is_dallas else "Dallas",
        center
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
    Evaluate an answer for the career center accessibility and service criteria task.
    """
    # Initialize evaluator (root node will be created internally as non-critical)
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

    # Create a task root node according to rubric (parallel). Set to non-critical to allow partial credit across institutions.
    task_root = evaluator.add_parallel(
        id="Career_Center_Qualification_Verification",
        desc="For each specified institution, evaluate whether the career center identified in the response meets all six requirements; and for each center the response claims qualifies, verify the required output fields are provided.",
        parent=root,
        critical=False
    )

    # Extract structured information from the answer
    centers = await evaluator.extract(
        prompt=prompt_extract_centers(),
        template_class=CentersExtraction,
        extraction_name="centers_extraction",
    )

    # Add minor helpful context for debugging in report
    evaluator.add_custom_info(
        {
            "allowed_states": ["Pennsylvania", "Texas", "Ohio"],
            "notes": "Output field checks judge presence-in-answer; requirement checks are URL-verified when sources are provided."
        },
        info_type="policy",
        info_name="evaluation_policy"
    )

    # Build evaluation per institution
    await verify_institution_block(
        evaluator=evaluator,
        parent=task_root,
        inst_key="IUP",
        center=centers.iup
    )

    await verify_institution_block(
        evaluator=evaluator,
        parent=task_root,
        inst_key="Grayson",
        center=centers.grayson
    )

    await verify_institution_block(
        evaluator=evaluator,
        parent=task_root,
        inst_key="Dallas",
        center=centers.dallas,
        is_dallas=True
    )

    # Return the structured evaluation summary
    return evaluator.get_summary()