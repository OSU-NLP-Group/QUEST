import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "us_r1_presidents_2025_window"
TASK_DESCRIPTION = (
    "Identify three university presidents in the United States who satisfy ALL of the following criteria:\n\n"
    "1. The president serves at an institution that holds the R1 (Research 1: Very High Spending and Doctorate Production) "
    "designation in the 2025 Carnegie Classifications.\n"
    "2. The institution is a public university or public university system (not a private institution).\n"
    "3. The institution has undergraduate enrollment of at least 30,000 students as reported in publicly available fall enrollment data.\n"
    "4. The institution is regionally accredited by one of the six recognized regional accrediting agencies in the United States.\n"
    "5. The president (or chancellor, if that is the institution's chief executive title) was appointed to their current position between "
    "January 1, 2024 and December 31, 2025 (inclusive).\n"
    "6. The president holds a doctoral degree, which may include a Ph.D., Ed.D., M.D., J.D., or other equivalent terminal degree.\n"
    "7. Before being appointed to their current presidential position, the president previously served as a provost, executive vice president "
    "for academic affairs, or chief academic officer at a college or university.\n\n"
    "For each of the three presidents, provide:\n"
    "- The president's full name and official title\n"
    "- The full name of the institution\n"
    "- Evidence that the institution meets the Carnegie R1, public status, enrollment threshold, and regional accreditation requirements\n"
    "- The month and year of the president's appointment to their current position\n"
    "- Information about the president's doctoral degree\n"
    "- Information about the president's previous role as a provost or chief academic officer, including the institution where they held that position\n"
    "- Reference URLs that support each piece of information"
)

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class PresidentItem(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None  # e.g., President, Chancellor (chief executive)
    institution: Optional[str] = None  # Full institution name
    appointment_month_year: Optional[str] = None  # e.g., "March 2025"
    doctoral_degree: Optional[str] = None  # e.g., "Ph.D. in Economics (University of X)"
    prior_role_title: Optional[str] = None  # e.g., "Provost", "EVP for Academic Affairs", "Chief Academic Officer"
    prior_role_institution: Optional[str] = None  # e.g., "University of Y"
    urls: List[str] = Field(default_factory=list)  # All reference URLs cited for this president


class PresidentsExtraction(BaseModel):
    presidents: List[PresidentItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_presidents() -> str:
    return """
    Extract up to the first three (3) U.S. university presidents (or chancellors, if that is the institution's chief executive title) described in the answer.
    For each president, extract the following fields exactly as stated in the answer:
    - name: The person's full name.
    - title: Their official current title at the institution (e.g., President, Chancellor).
    - institution: The full name of the institution they lead.
    - appointment_month_year: The month and year they were appointed or named to their current role (e.g., "March 2025"). If a specific day is provided, include only month and year.
    - doctoral_degree: A brief description of their doctoral/terminal degree (e.g., "Ph.D. in Biology (University of Z)", "J.D. (Harvard Law School)").
    - prior_role_title: The title of their prior role as a provost, executive vice president for academic affairs, or chief academic officer (or a clear equivalent).
    - prior_role_institution: The institution where they held that prior role.
    - urls: An array of all reference URLs the answer cites for this president. Include every URL associated with this president and their institution/evidence; collect both general references and criterion-specific links. If no URLs are cited, return an empty array.

    Return a JSON object with a 'presidents' array of up to three PresidentItem objects. If a field is not provided in the answer, set it to null. If there are more than three presidents in the answer, include only the first three in order of appearance.
    """


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _urls_or_empty(p: PresidentItem) -> List[str]:
    return [u for u in (p.urls or []) if _non_empty(u)]


# --------------------------------------------------------------------------- #
# Verification for one president                                              #
# --------------------------------------------------------------------------- #
async def verify_one_president(
    evaluator: Evaluator,
    parent_node,
    pres: PresidentItem,
    idx: int,
) -> None:
    """
    Build the verification subtree for a single president and run verifications.
    The node structure mirrors the provided rubric. All leaf nodes are binary.
    """
    pnum = idx + 1
    pres_node = evaluator.add_parallel(
        id=f"President_{pnum}",
        desc=(
            "First university president meeting all criteria" if pnum == 1 else
            "Second university president meeting all criteria" if pnum == 2 else
            "Third university president meeting all criteria"
        ),
        parent=parent_node,
        critical=False
    )

    urls = _urls_or_empty(pres)

    # 1) Reference URLs existence (critical)
    ref_urls_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"Reference_URLs_P{pnum}",
        desc="Provide reference URLs supporting the identification and verification of this president",
        parent=pres_node,
        critical=True
    )

    # 2) Personal Identification (critical): name + official title at institution
    personal_node = evaluator.add_leaf(
        id=f"Personal_Identification_P{pnum}",
        desc="Provide the president's full name and official title",
        parent=pres_node,
        critical=True
    )
    name = pres.name or ""
    title = pres.title or "president or chancellor"
    inst = pres.institution or "the institution"

    personal_claim = (
        f"{name} is the official {title} (the institution's chief executive) at {inst}."
    )
    await evaluator.verify(
        claim=personal_claim,
        node=personal_node,
        sources=urls,
        additional_instruction=(
            "Verify the person's full name and that they hold the exact or equivalent chief executive title "
            "(e.g., 'President' or 'Chancellor') at the named institution. Allow minor name variants and "
            "reasonable title phrasing (e.g., 'University President', 'Chancellor of X')."
        ),
    )

    # 3) Institution Name (critical)
    inst_name_node = evaluator.add_leaf(
        id=f"Institution_Name_P{pnum}",
        desc="Provide the full name of the institution where this president serves",
        parent=pres_node,
        critical=True
    )
    inst_name_claim = (
        f"The full official (or commonly recognized formal) name of the institution led by {name} is '{inst}'."
    )
    await evaluator.verify(
        claim=inst_name_claim,
        node=inst_name_node,
        sources=urls,
        additional_instruction=(
            "Confirm that the referenced pages clearly indicate the institution's full name (as used on its official site, "
            "brand/identity pages, or authoritative profiles). Minor variations like leading 'The' or state designation usage are acceptable."
        ),
    )

    # 4) Carnegie R1 in 2025 (critical)
    r1_node = evaluator.add_leaf(
        id=f"Carnegie_R1_P{pnum}",
        desc="Verify the institution holds R1 (Research 1: Very High Spending and Doctorate Production) designation in the 2025 Carnegie Classifications",
        parent=pres_node,
        critical=True
    )
    r1_claim = (
        f"{inst} is classified as R1 (Doctoral Universities – Very High Research Activity) in the 2025 Carnegie Classifications."
    )
    await evaluator.verify(
        claim=r1_claim,
        node=r1_node,
        sources=urls,
        additional_instruction=(
            "Look for explicit mention of 'R1', 'Very High Research Activity', or equivalent wording tied to the 2025 Carnegie Classifications. "
            "University news, Carnegie site, or other credible sources are acceptable, but the year should be 2025 for the classification."
        ),
    )

    # 5) Public Institution (critical)
    public_node = evaluator.add_leaf(
        id=f"Public_Institution_P{pnum}",
        desc="Verify the institution is a public university or public university system",
        parent=pres_node,
        critical=True
    )
    public_claim = f"{inst} is a public university or public university system (not a private institution)."
    await evaluator.verify(
        claim=public_claim,
        node=public_node,
        sources=urls,
        additional_instruction=(
            "Confirm the institution is publicly funded/state-operated or part of a public university system. "
            "Look for phrases like 'public research university', 'state university', or 'public university system'."
        ),
    )

    # 6) Enrollment threshold (critical): undergraduate >= 30,000
    enroll_node = evaluator.add_leaf(
        id=f"Enrollment_Threshold_P{pnum}",
        desc="Verify the institution has undergraduate enrollment of at least 30,000 students",
        parent=pres_node,
        critical=True
    )
    enroll_claim = (
        f"{inst} has undergraduate enrollment of at least 30,000 students according to publicly available fall enrollment data."
    )
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_node,
        sources=urls,
        additional_instruction=(
            "Check official facts pages, Common Data Set, or credible institutional statistics for undergraduate enrollment. "
            "The value must be undergraduate-specific and >= 30,000. If only total (UG+Grad) is shown without an undergraduate breakdown, "
            "that is insufficient. If a credible page clearly states 'undergraduate enrollment' >= 30,000, it qualifies."
        ),
    )

    # 7) Regional accreditation (critical)
    accred_node = evaluator.add_leaf(
        id=f"Regional_Accreditation_P{pnum}",
        desc="Verify the institution is regionally accredited by one of the six recognized regional accrediting agencies",
        parent=pres_node,
        critical=True
    )
    accred_claim = (
        f"{inst} is regionally accredited by one of the six recognized U.S. regional accrediting agencies "
        "(HLC, SACSCOC, NECHE, MSCHE, WSCUC, or NWCCU)."
    )
    await evaluator.verify(
        claim=accred_claim,
        node=accred_node,
        sources=urls,
        additional_instruction=(
            "Accept evidence of accreditation by any of: Higher Learning Commission (HLC), Southern Association of Colleges and Schools "
            "Commission on Colleges (SACSCOC), New England Commission of Higher Education (NECHE), Middle States Commission on Higher Education (MSCHE), "
            "WASC Senior College and University Commission (WSCUC), or Northwest Commission on Colleges and Universities (NWCCU). "
            "Professional/specialized accreditations do not count."
        ),
    )

    # 8) Appointment timeline (critical): appointed between 2024-01-01 and 2025-12-31 (inclusive)
    appt_node = evaluator.add_leaf(
        id=f"Appointment_Timeline_P{pnum}",
        desc="Verify the president was appointed to their current position between January 1, 2024 and December 31, 2025",
        parent=pres_node,
        critical=True
    )
    appt_when = pres.appointment_month_year or "an appointment date in 2024 or 2025"
    appt_claim = (
        f"{name} was appointed as {title} at {inst} in {appt_when}, and this appointment date falls between "
        f"January 1, 2024 and December 31, 2025 inclusive."
    )
    await evaluator.verify(
        claim=appt_claim,
        node=appt_node,
        sources=urls,
        additional_instruction=(
            "Check official announcements, press releases, governing board actions, or trusted news confirming the appointment date. "
            "The relevant date is when they were appointed/named/selected (not necessarily the start date if different). "
            "It must be in 2024 or 2025."
        ),
    )

    # 9) Doctoral degree (critical)
    degree_node = evaluator.add_leaf(
        id=f"Doctoral_Degree_P{pnum}",
        desc="Verify the president holds a doctoral degree (Ph.D., Ed.D., M.D., J.D., or equivalent terminal degree)",
        parent=pres_node,
        critical=True
    )
    degree_text = pres.doctoral_degree or "a doctoral/terminal degree (e.g., Ph.D., Ed.D., M.D., or J.D.)"
    degree_claim = f"{name} holds {degree_text}."
    await evaluator.verify(
        claim=degree_claim,
        node=degree_node,
        sources=urls,
        additional_instruction=(
            "Verify that the individual holds a terminal doctoral degree. Accept Ph.D., Ed.D., M.D., J.D., or other equivalent terminal doctorates. "
            "If only master's degrees or non-terminal degrees are listed, this does not satisfy the requirement."
        ),
    )

    # 10) Prior provost/CAO experience (critical)
    provost_node = evaluator.add_leaf(
        id=f"Prior_Provost_Experience_P{pnum}",
        desc="Verify the president previously served as a provost, executive vice president for academic affairs, or chief academic officer at a college or university",
        parent=pres_node,
        critical=True
    )
    prior_role = pres.prior_role_title or "a provost or chief academic officer role"
    prior_inst = pres.prior_role_institution or "a university"
    provost_claim = (
        f"Before being appointed {title} at {inst}, {name} served as {prior_role} at {prior_inst}, "
        f"which is a provost/Chief Academic Officer-equivalent role at a college or university."
    )
    await evaluator.verify(
        claim=provost_claim,
        node=provost_node,
        sources=urls,
        additional_instruction=(
            "Titles that qualify include 'Provost', 'Executive Vice President for Academic Affairs', 'Senior Vice President for Academic Affairs', "
            "'Chief Academic Officer', or closely equivalent institutional CAO roles. Confirm that this was at a college or university."
        ),
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
    Evaluate an answer for the 'three U.S. R1 presidents (2025 window)' task.
    Returns a structured summary with a full verification tree.
    """
    # Initialize evaluator (root is non-critical and parallel by default)
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

    # Optional: record policy parameters for transparency
    evaluator.add_custom_info(
        info={
            "carnegie_year": 2025,
            "enrollment_min_undergrad": 30000,
            "appointment_window_inclusive": ["2024-01-01", "2025-12-31"],
            "regional_agencies": ["HLC", "SACSCOC", "NECHE", "MSCHE", "WSCUC", "NWCCU"],
        },
        info_type="policy",
        info_name="verification_policies",
    )

    # Add a top-level rubric node mirroring the JSON's root (non-critical, parallel)
    top_node = evaluator.add_parallel(
        id="Identify_Three_University_Presidents",
        desc="Identify three university presidents who meet all specified criteria",
        parent=root,
        critical=False,
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_presidents(),
        template_class=PresidentsExtraction,
        extraction_name="presidents_extraction",
    )

    # Use only the first three records, pad with empties if fewer than 3
    presidents: List[PresidentItem] = list(extracted.presidents[:3])
    while len(presidents) < 3:
        presidents.append(PresidentItem())

    # Build and verify each president subtree
    for i, pres in enumerate(presidents):
        await verify_one_president(evaluator, top_node, pres, i)

    # Return full summary
    return evaluator.get_summary()