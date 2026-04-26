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
TASK_ID = "tx_superintendent_2022_window"
TASK_DESCRIPTION = (
    "Identify the full name of the superintendent who leads a Texas public school district that meets all of the following criteria as of February 2026:\n\n"
    "1. The district is located in Texas\n"
    "2. The district serves between 40,000 and 55,000 students as of the 2024-25 school year\n"
    "3. The superintendent was appointed or officially named to the position between January 2022 and December 2022 (inclusive)\n"
    "4. The superintendent is currently still serving in that position as of February 2026\n"
    "5. The district is a traditional public school district (not a charter school system)\n\n"
    "Provide the superintendent's full name and the name of the school district they lead."
)

AS_OF_MONTH = "February"
AS_OF_YEAR = 2026
ENROLLMENT_MIN = 40000
ENROLLMENT_MAX = 55000
APPOINTMENT_YEAR_REQUIRED = 2022


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class SuperintendentExtraction(BaseModel):
    """
    Structured extraction of the candidate superintendent and district details
    as presented in the agent's answer.
    """
    superintendent_full_name: Optional[str] = None
    district_name: Optional[str] = None

    # Optional helpful snippets exactly as mentioned in the answer (free-form strings)
    state_or_location_text: Optional[str] = None
    enrollment_2024_25_text: Optional[str] = None
    appointment_date_text: Optional[str] = None
    current_status_text: Optional[str] = None
    district_type_text: Optional[str] = None

    # URL sources grouped by purpose; if the answer doesn't separate them,
    # the extractor should put links into sources_general.
    sources_general: List[str] = Field(default_factory=list)
    sources_location: List[str] = Field(default_factory=list)
    sources_enrollment: List[str] = Field(default_factory=list)
    sources_appointment: List[str] = Field(default_factory=list)
    sources_current: List[str] = Field(default_factory=list)
    sources_district_type: List[str] = Field(default_factory=list)
    sources_superintendent_bio: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendent() -> str:
    return (
        "Extract the superintendent and district details from the answer.\n"
        "Return the following fields exactly as they appear in the answer when applicable:\n"
        "- superintendent_full_name: Full name of the superintendent identified.\n"
        "- district_name: Full official name of the school district identified.\n"
        "- state_or_location_text: Any explicit mention indicating the district is in Texas (e.g., 'Texas', city, or state reference).\n"
        "- enrollment_2024_25_text: The stated enrollment or phrasing for the 2024–25 school year (if given).\n"
        "- appointment_date_text: The stated month/year or date range for when the superintendent was appointed/named.\n"
        "- current_status_text: Any phrase indicating the superintendent is currently serving as of now.\n"
        "- district_type_text: Any explicit statement that the district is a traditional public school district (not a charter), or the presence of 'ISD'/'CISD'.\n"
        "\n"
        "Also extract URLs grouped by purpose when the answer provides them:\n"
        "- sources_general: Any URLs cited for this item that do not clearly map to a single category below.\n"
        "- sources_location: URLs that support the district being in Texas.\n"
        "- sources_enrollment: URLs that support the 2024–25 enrollment figures.\n"
        "- sources_appointment: URLs that support the superintendent's appointment/naming and its date.\n"
        "- sources_current: URLs that support that the superintendent is still serving currently (as of Feb 2026).\n"
        "- sources_district_type: URLs that support that the district is a traditional public school district (not a charter system).\n"
        "- sources_superintendent_bio: URLs that describe superintendent background/bio/certifications (if provided).\n"
        "\n"
        "Special rules for URL extraction:\n"
        "- Extract only actual URLs present in the answer (including markdown links). Do not invent URLs.\n"
        "- Return full URLs. If protocol missing, prepend http://.\n"
        "- If the answer lists multiple URLs for a category, include them all.\n"
        "- If a category has no URLs, return an empty list for that category.\n"
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def pick_sources(ext: SuperintendentExtraction, preferred_keys: List[str]) -> List[str]:
    """
    Pick sources from extraction in priority order of the preferred_keys
    and fall back to sources_general. Deduplicate results and cap to a sane size.
    """
    collected: List[str] = []
    for key in preferred_keys:
        lst = getattr(ext, key, None)
        if isinstance(lst, list):
            collected.extend(lst)
    if not collected:
        collected.extend(ext.sources_general or [])
    collected = _dedup_urls(collected)
    # cap to at most 8 URLs to keep verification manageable
    return collected[:8]


async def verify_with_sources_or_mark_failed(
    evaluator: Evaluator,
    claim: str,
    node,
    sources: List[str],
    add_ins: str,
    extra_prereq: Optional[List[Any]] = None,
) -> bool:
    """
    Verify the claim against provided sources. If no sources are provided,
    mark the node as failed (source-grounding policy).
    """
    if not sources:
        node.score = 0.0
        node.status = "failed"
        evaluator.add_custom_info(
            {"claim": claim, "reason": "no_sources_provided"}, info_type="missing_sources"
        )
        return False

    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=add_ins,
        extra_prerequisites=extra_prereq,
    )


# --------------------------------------------------------------------------- #
# Build verification nodes                                                    #
# --------------------------------------------------------------------------- #
async def build_and_verify_criteria(
    evaluator: Evaluator,
    parent_node,
    ext: SuperintendentExtraction,
) -> None:
    # Identification checks (custom/binary existence)
    superintendent_present = bool(ext.superintendent_full_name and ext.superintendent_full_name.strip())
    district_present = bool(ext.district_name and ext.district_name.strip())

    node_super = evaluator.add_custom_node(
        result=superintendent_present,
        id="superintendent_identified",
        desc="A specific superintendent is clearly identified by full name",
        parent=parent_node,
        critical=True
    )

    node_district = evaluator.add_custom_node(
        result=district_present,
        id="district_identified",
        desc="The school district is clearly identified by name",
        parent=parent_node,
        critical=True
    )

    prereq = [node_super, node_district]

    # 1) Texas location
    node_tx = evaluator.add_leaf(
        id="texas_location",
        desc="The school district is located in Texas",
        parent=parent_node,
        critical=True
    )
    district_display = ext.district_name or "the identified district"
    claim_tx = f"The school district named '{district_display}' is located in the state of Texas."
    addins_tx = (
        "Verify from official district or authoritative sources that the district is in Texas. "
        "Evidence could include the district name (e.g., 'ISD' in Texas), Texas address, or explicit mention of Texas."
    )
    await verify_with_sources_or_mark_failed(
        evaluator,
        claim_tx,
        node_tx,
        pick_sources(ext, ["sources_location", "sources_district_type"]),
        addins_tx,
        extra_prereq=prereq
    )

    # 2) Enrollment range (40,000–55,000) as of 2024–25
    node_enr = evaluator.add_leaf(
        id="enrollment_range",
        desc="The district serves between 40,000 and 55,000 students as of the 2024-25 school year",
        parent=parent_node,
        critical=True
    )
    claim_enr = (
        f"As of the 2024–25 school year, '{district_display}' served between {ENROLLMENT_MIN} and {ENROLLMENT_MAX} students (inclusive)."
    )
    addins_enr = (
        "Use the cited page(s) to check enrollment for the 2024–25 school year. "
        "Slight phrasing variations like 'students served' or 'district enrollment' are acceptable. "
        "If a specific number falls within the range, consider this supported."
    )
    await verify_with_sources_or_mark_failed(
        evaluator,
        claim_enr,
        node_enr,
        pick_sources(ext, ["sources_enrollment"]),
        addins_enr,
        extra_prereq=prereq
    )

    # 3) Appointment timeframe between Jan 2022 and Dec 2022 (inclusive)
    node_appt = evaluator.add_leaf(
        id="appointment_timeframe",
        desc="The superintendent was appointed or named to the position between January 2022 and December 2022 (inclusive)",
        parent=parent_node,
        critical=True
    )
    sup_display = ext.superintendent_full_name or "the identified superintendent"
    claim_appt = (
        f"{sup_display} was appointed or officially named superintendent of '{district_display}' "
        f"between January {APPOINTMENT_YEAR_REQUIRED} and December {APPOINTMENT_YEAR_REQUIRED} (inclusive)."
    )
    addins_appt = (
        "Check official announcements, board minutes, or credible news releases. "
        "In Texas, 'lone finalist' naming followed by board approval is common; either the official naming/appointment in 2022 qualifies."
    )
    await verify_with_sources_or_mark_failed(
        evaluator,
        claim_appt,
        node_appt,
        pick_sources(ext, ["sources_appointment"]),
        addins_appt,
        extra_prereq=prereq
    )

    # 4) Currently serving as of Feb 2026
    node_current = evaluator.add_leaf(
        id="current_service",
        desc="The superintendent is still serving in that position as of February 2026",
        parent=parent_node,
        critical=True
    )
    claim_cur = (
        f"As of {AS_OF_MONTH} {AS_OF_YEAR}, {sup_display} is still serving as superintendent of '{district_display}'."
    )
    addins_cur = (
        "Prefer an official district leadership/staff page, superintendent page, or current directory indicating the individual holds the role. "
        "Recent updates in late 2025 or 2026 are ideal. If a page clearly shows them as the current superintendent, consider it supported."
    )
    await verify_with_sources_or_mark_failed(
        evaluator,
        claim_cur,
        node_current,
        pick_sources(ext, ["sources_current", "sources_superintendent_bio"]),
        addins_cur,
        extra_prereq=prereq
    )

    # 5) Traditional public school district (not a charter system)
    node_type = evaluator.add_leaf(
        id="traditional_public_district",
        desc="The district is a traditional public school district, not a charter school system",
        parent=parent_node,
        critical=True
    )
    claim_type = (
        f"'{district_display}' is a traditional public school district (e.g., an ISD/CISD), not a charter school system."
    )
    addins_type = (
        "Check for 'Independent School District' (ISD) or similar designation and that it is a standard public school district rather than a charter operator."
    )
    await verify_with_sources_or_mark_failed(
        evaluator,
        claim_type,
        node_type,
        pick_sources(ext, ["sources_district_type", "sources_location"]),
        addins_type,
        extra_prereq=prereq
    )

    # 6) Texas certification requirements (non-critical; rubric included but not in original task)
    node_cert = evaluator.add_leaf(
        id="texas_certification_requirements",
        desc="The superintendent meets TEA requirements including holding a master's degree and an appropriate certificate",
        parent=parent_node,
        critical=False  # Not part of the original task requirements; treat as non-critical partial credit
    )
    claim_cert = (
        f"{sup_display} meets Texas Education Agency superintendent requirements (e.g., holds a master's degree and "
        "an appropriate principal/superintendent certificate or equivalent)."
    )
    addins_cert = (
        "Verify via official bio pages or credible sources. If no explicit evidence of credentials is provided in the cited URLs, this should not be supported."
    )
    await verify_with_sources_or_mark_failed(
        evaluator,
        claim_cert,
        node_cert,
        pick_sources(ext, ["sources_superintendent_bio", "sources_general"]),
        addins_cert,
        extra_prereq=prereq
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
) -> Dict:
    """
    Evaluate an answer for the Texas superintendent identification task.
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

    # Extract candidate information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_superintendent(),
        template_class=SuperintendentExtraction,
        extraction_name="superintendent_candidate"
    )

    # Record evaluation context info
    evaluator.add_custom_info(
        {
            "as_of_month": AS_OF_MONTH,
            "as_of_year": AS_OF_YEAR,
            "enrollment_range_required": [ENROLLMENT_MIN, ENROLLMENT_MAX],
            "appointment_year_required": APPOINTMENT_YEAR_REQUIRED
        },
        info_type="evaluation_constraints",
        info_name="constraints"
    )

    # Build the tree and run verifications
    await build_and_verify_criteria(evaluator, root, extraction)

    # Return standardized summary
    return evaluator.get_summary()