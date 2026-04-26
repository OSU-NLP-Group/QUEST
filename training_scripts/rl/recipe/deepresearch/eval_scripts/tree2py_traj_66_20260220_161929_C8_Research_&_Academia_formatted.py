import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "early_career_ai_ml_faculty_3x"
TASK_DESCRIPTION = """Identify 3 early-career faculty members in artificial intelligence or machine learning who meet ALL of the following criteria:

1. NSF CAREER Award: Received a National Science Foundation (NSF) CAREER award in 2024 or 2025 in the field of computer science, artificial intelligence, or machine learning. Provide the NSF Award Search URL confirming this award.

2. Current Academic Position: Currently holds an assistant professor position (or equivalent early-career tenure-track position) at a top-tier US university in a computer science, artificial intelligence, or related department. Provide the university faculty directory URL confirming their current position and rank.

3. Conference Leadership Service: Has served as an Area Chair, Senior Area Chair, or Program Committee member at a major AI/ML conference (NeurIPS, ICML, ICLR, or AAAI) in 2025 or 2026. Provide the conference website URL confirming this service role.

4. Research Impact: Has an h-index of at least 30 on Google Scholar. Provide the Google Scholar profile URL showing the h-index.

5. Research Group Affiliation: Is affiliated with an AI/ML research laboratory, research group, or research center at their institution. Provide the research lab/group website URL confirming this affiliation.

For each of the 3 faculty members, provide:
- Full name
- Current institution
- NSF Award Search URL
- Faculty directory URL
- Conference committee URL
- Google Scholar profile URL
- Research lab/group URL

All information must be verifiable through publicly accessible sources as of February 2026.
"""


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class FacultyItem(BaseModel):
    full_name: Optional[str] = None
    institution: Optional[str] = None
    nsf_award_url: Optional[str] = None
    directory_url: Optional[str] = None
    conference_url: Optional[str] = None
    scholar_url: Optional[str] = None
    lab_url: Optional[str] = None
    top_tier_support_urls: List[str] = Field(default_factory=list)


class FacultyList(BaseModel):
    faculty: List[FacultyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_faculty_list() -> str:
    return """
    Extract up to the first 5 faculty candidates listed in the answer (maintain the answer's order).
    For each candidate, extract the following fields exactly as stated/provided in the answer:
    - full_name: The person's full name
    - institution: The current institution name
    - nsf_award_url: The URL to the NSF Award Search page confirming a CAREER award in 2024 or 2025
    - directory_url: The URL to the university faculty directory (or official personal faculty page) confirming current rank and department
    - conference_url: The URL to an official NeurIPS/ICML/ICLR/AAAI page confirming the person's service as Area Chair / Senior Area Chair / Program Committee in 2025 or 2026
    - scholar_url: The URL to the Google Scholar profile for the person
    - lab_url: The URL to the research lab/group/center page at the stated institution confirming affiliation
    - top_tier_support_urls: An array of any URLs provided in the answer that help support that the institution is a top-tier US university (e.g., AAU membership page, major rankings, institutional status pages). If none are provided, return an empty array.

    RULES:
    - Extract only URLs explicitly present in the answer; do not infer or fabricate URLs.
    - If any field is missing for a candidate, set it to null (or empty array for top_tier_support_urls).
    - Do not include more than 5 candidates in the output. Preserve the order from the answer.
    - Do NOT try to validate; only extract what is present.
    """

# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_faculty_member(
    evaluator: Evaluator,
    parent_node,
    item: FacultyItem,
    idx: int,
) -> None:
    """
    Build and execute the verification sub-tree for a single faculty member.
    """
    # Container node for this faculty member (Sequential to gate later checks on presence)
    member_node = evaluator.add_sequential(
        id=f"fm_{idx}",
        desc=f"Faculty member #{idx + 1} (must meet all criteria; include all required fields/URLs)",
        parent=parent_node,
        critical=False
    )

    # 1) Presence checks (gate)
    presence_node = evaluator.add_parallel(
        id=f"fm_{idx}_presence",
        desc=f"Faculty member #{idx + 1} – required fields/URLs are provided",
        parent=member_node,
        critical=False
    )

    name_present = evaluator.add_custom_node(
        result=bool(item.full_name and item.full_name.strip()),
        id=f"fm_{idx}_name_present",
        desc=f"Faculty member #{idx + 1}: Full name provided",
        parent=presence_node,
        critical=True
    )
    inst_present = evaluator.add_custom_node(
        result=bool(item.institution and item.institution.strip()),
        id=f"fm_{idx}_inst_present",
        desc=f"Faculty member #{idx + 1}: Institution provided",
        parent=presence_node,
        critical=True
    )
    nsf_present = evaluator.add_custom_node(
        result=bool(item.nsf_award_url and item.nsf_award_url.strip()),
        id=f"fm_{idx}_nsf_url_present",
        desc=f"Faculty member #{idx + 1}: NSF CAREER award URL provided",
        parent=presence_node,
        critical=True
    )
    dir_present = evaluator.add_custom_node(
        result=bool(item.directory_url and item.directory_url.strip()),
        id=f"fm_{idx}_dir_url_present",
        desc=f"Faculty member #{idx + 1}: Faculty directory URL provided",
        parent=presence_node,
        critical=True
    )
    conf_present = evaluator.add_custom_node(
        result=bool(item.conference_url and item.conference_url.strip()),
        id=f"fm_{idx}_conf_url_present",
        desc=f"Faculty member #{idx + 1}: Conference service URL provided",
        parent=presence_node,
        critical=True
    )
    scholar_present = evaluator.add_custom_node(
        result=bool(item.scholar_url and item.scholar_url.strip()),
        id=f"fm_{idx}_scholar_url_present",
        desc=f"Faculty member #{idx + 1}: Google Scholar URL provided",
        parent=presence_node,
        critical=True
    )
    lab_present = evaluator.add_custom_node(
        result=bool(item.lab_url and item.lab_url.strip()),
        id=f"fm_{idx}_lab_url_present",
        desc=f"Faculty member #{idx + 1}: Lab/group URL provided",
        parent=presence_node,
        critical=True
    )

    # 2) Parallel verification checks (executed after presence)
    checks_node = evaluator.add_parallel(
        id=f"fm_{idx}_checks",
        desc=f"Faculty member #{idx + 1} – verification checks",
        parent=member_node,
        critical=False
    )

    # 2.1 Full name consistency with directory page
    name_match_node = evaluator.add_leaf(
        id=f"fm_{idx}_name_match",
        desc=f"Full name is consistent with the faculty directory page",
        parent=checks_node,
        critical=True
    )
    name_claim = f"The faculty directory page shows the person's name as '{item.full_name}', allowing minor variations (middle initials, diacritics, hyphenation, casing) that clearly refer to the same person."
    # 2.2 Institution consistency with directory page (and hosted by institution)
    inst_match_node = evaluator.add_leaf(
        id=f"fm_{idx}_inst_match",
        desc=f"Institution matches and directory page confirms the person is at the stated institution",
        parent=checks_node,
        critical=True
    )
    inst_claim = f"The faculty directory page is for '{item.institution}' and indicates that {item.full_name} is at this institution (accept reasonable abbreviations like 'UC Berkeley' vs 'University of California, Berkeley')."

    # 2.3 Assistant Professor (or equivalent early-career tenure-track) in CS/AI/related dept
    position_node = evaluator.add_leaf(
        id=f"fm_{idx}_assistant_prof_role",
        desc=f"Currently an assistant professor (or equivalent early-career tenure-track) in a CS/AI/related department",
        parent=checks_node,
        critical=True
    )
    position_claim = (
        f"The faculty directory page indicates that {item.full_name} currently holds an Assistant Professor "
        f"(or equivalent early-career tenure-track) position in a computer science, artificial intelligence, "
        f"or closely related department. Accept variations like 'Assistant Professor of Computer Science' or "
        f"'Assistant Professor in Electrical Engineering and Computer Science'. Do not accept 'Associate Professor', "
        f"'Adjunct', 'Visiting', 'Lecturer', or non-tenure-track roles as equivalent."
    )

    # 2.4 Top-tier US university check (requires supporting URLs if provided)
    top_tier_node = evaluator.add_leaf(
        id=f"fm_{idx}_top_tier_us",
        desc=f"The institution is a top-tier US university (supported by provided evidence)",
        parent=checks_node,
        critical=True
    )
    top_tier_sources: List[str] = []
    if item.top_tier_support_urls:
        top_tier_sources.extend(item.top_tier_support_urls)
    if item.directory_url:
        top_tier_sources.append(item.directory_url)
    top_tier_claim = (
        f"'{item.institution}' is a top-tier US university. The provided pages must themselves explicitly show or strongly imply "
        f"top-tier standing (e.g., AAU membership page, top rankings pages like US News / THE / QS) and that the institution is in the United States. "
        f"If the available pages do not provide such evidence, this claim is not supported."
    )

    # 2.5 NSF CAREER award in 2024 or 2025, CS/AI/ML field (via NSF Award Search)
    nsf_node = evaluator.add_leaf(
        id=f"fm_{idx}_nsf_career",
        desc=f"NSF Award Search page confirms a CAREER award in 2024 or 2025 in CS/AI/ML",
        parent=checks_node,
        critical=True
    )
    nsf_claim = (
        f"The NSF Award Search page confirms that {item.full_name} received a CAREER award in 2024 or 2025. "
        f"The page should show 'CAREER' and an award year of 2024 or 2025, and it should be in the computer and information science/engineering area "
        f"(e.g., Directorate for CISE such as CCF, CNS, IIS, OAC) or the project clearly centers on AI/ML."
    )

    # 2.6 Conference leadership/service (AC/SAC/PC) in 2025 or 2026 at NeurIPS/ICML/ICLR/AAAI
    conf_node = evaluator.add_leaf(
        id=f"fm_{idx}_conf_service",
        desc=f"Conference page confirms service as AC/SAC/PC for NeurIPS/ICML/ICLR/AAAI in 2025 or 2026",
        parent=checks_node,
        critical=True
    )
    conf_claim = (
        f"The conference webpage shows that {item.full_name} served as an Area Chair, Senior Area Chair, or Program Committee member "
        f"at NeurIPS, ICML, ICLR, or AAAI in either 2025 or 2026. Accept committee/area chair lists or program committee rosters."
    )

    # 2.7 Google Scholar h-index ≥ 30
    hindex_node = evaluator.add_leaf(
        id=f"fm_{idx}_scholar_hindex",
        desc=f"Google Scholar profile shows h-index of at least 30",
        parent=checks_node,
        critical=True
    )
    hindex_claim = (
        f"The Google Scholar profile for {item.full_name} shows an h-index of at least 30. "
        f"If the profile loads dynamically, rely on the screenshots; minor parsing differences are acceptable."
    )

    # 2.8 Research lab/group affiliation at the institution
    lab_node = evaluator.add_leaf(
        id=f"fm_{idx}_lab_affiliation",
        desc=f"Lab/group page confirms affiliation with AI/ML lab/group/center at the stated institution",
        parent=checks_node,
        critical=True
    )
    lab_claim = (
        f"The lab/group/center webpage indicates that {item.full_name} is affiliated with the group and that the group is at {item.institution}. "
        f"Accept roles like member, PI, faculty, or affiliated faculty."
    )

    # Prepare batch verifications (they will auto-skip if presence preconditions failed)
    claims_and_sources = [
        (name_claim, item.directory_url, name_match_node, "Allow minor spelling/casing variants and middle initials when matching names."),
        (inst_claim, item.directory_url, inst_match_node, "Treat common abbreviations as equivalent; verify the site is an official page of the stated institution."),
        (position_claim, item.directory_url, position_node, "Confirm 'Assistant Professor' or equivalent early-career tenure-track in CS/AI/related department on the page."),
        (top_tier_claim, top_tier_sources, top_tier_node, "Only support if evidence is explicitly present on provided pages (e.g., AAU membership page, top rankings) and indicates US-based institution."),
        (nsf_claim, item.nsf_award_url, nsf_node, "Verify the award is labeled 'CAREER', year is 2024 or 2025, and falls under CS/AI/ML (e.g., CISE divisions like CCF/CNS/IIS/OAC)."),
        (conf_claim, item.conference_url, conf_node, "Verify the page lists the person as AC/SAC/PC for NeurIPS/ICML/ICLR/AAAI in 2025 or 2026."),
        (hindex_claim, item.scholar_url, hindex_node, "Verify that the 'h-index' displayed is >= 30. Allow small formatting differences."),
        (lab_claim, item.lab_url, lab_node, "Verify the page lists the person as affiliated with the lab/group at the stated institution."),
    ]

    await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the early-career AI/ML faculty task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Members evaluated independently
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

    # Extract the faculty list from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_faculty_list(),
        template_class=FacultyList,
        extraction_name="extracted_faculty_list",
    )

    # Keep exactly 3 (pad with empty if fewer)
    faculty_items: List[FacultyItem] = list(extracted.faculty)[:3]
    while len(faculty_items) < 3:
        faculty_items.append(FacultyItem())

    evaluator.add_custom_info(
        info={
            "total_candidates_extracted": len(extracted.faculty),
            "candidates_used": 3
        },
        info_type="extraction_stats",
    )

    # Create container for the three members under root (parallel)
    members_root = evaluator.add_parallel(
        id="members",
        desc="Three faculty members verification",
        parent=root,
        critical=False
    )

    # Verify each member
    for i in range(3):
        await verify_faculty_member(
            evaluator=evaluator,
            parent_node=members_root,
            item=faculty_items[i],
            idx=i
        )

    return evaluator.get_summary()