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
TASK_ID = "nyu_stanford_ai_nsf"
TASK_DESCRIPTION = """Identify four tenure-track faculty members (Assistant Professor, Associate Professor, or Full Professor—not visiting, adjunct, or emeritus positions) from either New York University or Stanford University who meet all of the following criteria:

1. They must be affiliated with a Computer Science department:
   - For NYU: Either the Courant Institute of Mathematical Sciences Computer Science Department or the Tandon School of Engineering Computer Science and Engineering Department
   - For Stanford: The Computer Science Department

2. They must list artificial intelligence (AI) or machine learning (ML) as one of their primary research areas on their official university faculty profile page.

3. They must have received at least one active research grant from the National Science Foundation's (NSF) Computer and Information Science and Engineering (CISE) directorate, where the grant was either awarded or renewed in 2024 or 2025.

For each of the four faculty members, provide:
- Their full name
- Their current academic title (e.g., Assistant Professor, Associate Professor, Professor)
- Their university affiliation and specific department name
- Their primary research area(s) related to AI or ML
- A direct link to their official university faculty profile page
- The title of at least one NSF CISE grant they hold that was awarded or renewed in 2024 or 2025
- The NSF award number for that grant
- A direct link to the corresponding NSF award page
"""

ALLOWED_DEPARTMENT_INSTRUCTIONS = (
    "Allowed departments for this task are strictly limited to:\n"
    "- New York University (NYU) – Courant Institute of Mathematical Sciences – Department of Computer Science\n"
    "- New York University (NYU) – Tandon School of Engineering – Department of Computer Science and Engineering\n"
    "- Stanford University – Department of Computer Science\n"
    "Consider reasonable textual variants (e.g., 'Dept.' vs 'Department', "
    "'Computer Science and Engineering' vs 'CSE', or inclusions like 'at Courant').\n"
    "Affiliations such as Mathematics, EE, ECE, or generic 'Engineering' do NOT qualify unless the page clearly indicates the CS department as above.\n"
)

TENURE_TRACK_INSTRUCTIONS = (
    "Tenure-track titles for this task include exactly: 'Assistant Professor', 'Associate Professor', or 'Professor' "
    "(i.e., Full Professor). Exclude any that are 'Adjunct', 'Clinical', 'Visiting', 'Emeritus/Emerita', "
    "'Lecturer', 'Teaching Professor', 'Professor (Research)', or 'Research Professor'. "
    "If modifiers like 'of Computer Science' are attached (e.g., 'Assistant Professor of Computer Science'), they still qualify as tenure-track."
)

AI_ML_INSTRUCTIONS = (
    "Confirm that the official profile explicitly lists 'Artificial Intelligence' or 'Machine Learning' (or synonymous primary areas) "
    "as a research area. Accept common synonyms that unambiguously fall under AI/ML, such as 'AI', 'ML', 'Deep Learning', "
    "'Reinforcement Learning', 'Natural Language Processing', 'Computer Vision' if clearly framed as a primary research area. "
    "Do not accept merely incidental mentions (e.g., within publication titles) unless the profile has a dedicated 'Research'/'Interests' section indicating it."
)

PROFILE_URL_INSTRUCTIONS = (
    "Judge whether this URL is an official faculty profile page for the named person at NYU or Stanford. "
    "Common official domains include cs.nyu.edu, cims.nyu.edu, courant.nyu.edu, engineering.nyu.edu, and cs.stanford.edu or profiles.stanford.edu. "
    "Accept other official subdomains of nyu.edu or stanford.edu that clearly indicate an official faculty profile. "
    "The page should display the person's name and institutional affiliation. Reject non-official domains or lab pages if they are not the official faculty profile."
)

NSF_FUNDING_INSTRUCTIONS = (
    "Verify on the NSF award page that: "
    "1) The award belongs to the 'Directorate for Computer and Information Science and Engineering (CISE)'. "
    "2) The named faculty member appears as PI or Co-PI (or Senior Personnel) on the award page. "
    "3) The award was awarded or renewed in 2024 or 2025 (e.g., via 'Start Date', 'Last Amendment/Modification Date', 'Award Date', or similar). "
    "4) The award is 'Active' (status) OR the performance period includes the current date; reasonable interpretations of 'active' based on page metadata are acceptable."
)

NSF_GRANT_DETAILS_INSTRUCTIONS = (
    "Check that the award number and title in the claim match what is shown on the NSF award page. "
    "Allow minor punctuation/case differences, but the core title and exact numeric award ID should be consistent."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GrantInfo(BaseModel):
    title: Optional[str] = None
    award_number: Optional[str] = None
    nsf_url: Optional[str] = None


class FacultyInfo(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    university: Optional[str] = None
    department: Optional[str] = None
    research_areas: List[str] = Field(default_factory=list)
    profile_url: Optional[str] = None
    grant: Optional[GrantInfo] = None


class FacultyList(BaseModel):
    faculties: List[FacultyInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_faculty() -> str:
    return (
        "Extract up to four faculty entries from the answer. For each entry, return the following fields:\n"
        "- name: Full name of the faculty member.\n"
        "- title: Current academic title exactly as stated (e.g., 'Assistant Professor', 'Associate Professor', 'Professor').\n"
        "- university: The university name (e.g., 'New York University', 'Stanford University').\n"
        "- department: The specific department (e.g., 'Department of Computer Science', 'Department of Computer Science and Engineering', "
        "'Computer Science (Courant)').\n"
        "- research_areas: A list of the primary research areas related to AI or ML as the answer states (e.g., ['Artificial Intelligence', 'Machine Learning']).\n"
        "- profile_url: A direct URL to the official university faculty profile page for this person.\n"
        "- grant: An object with fields:\n"
        "   - title: The NSF award title for at least one NSF CISE grant mentioned.\n"
        "   - award_number: The NSF award number (digits) for that grant.\n"
        "   - nsf_url: A direct URL to the NSF award page for that grant.\n\n"
        "If any field is missing in the answer, set it to null (or an empty list for research_areas). "
        "If multiple NSF grants are mentioned, pick one that appears to be from the CISE directorate and was awarded or renewed in 2024 or 2025, if available."
    )


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _ordinal(idx: int) -> str:
    return ["First", "Second", "Third", "Fourth"][idx] if 0 <= idx < 4 else f"#{idx+1}"


def _safe_join(items: Optional[List[str]]) -> str:
    return ", ".join(items) if items else ""


# --------------------------------------------------------------------------- #
# Verification for a single faculty member                                    #
# --------------------------------------------------------------------------- #
async def verify_one_faculty(
    evaluator: Evaluator,
    parent_node,
    faculty: FacultyInfo,
    idx: int,
) -> None:
    ordinal = _ordinal(idx)
    fac_parent = evaluator.add_parallel(
        id=f"faculty_{idx+1}",
        desc=f"{ordinal} faculty member meeting all criteria",
        parent=parent_node,
        critical=False,
    )

    # 1) Name provided (existence check)
    name_present = bool(faculty and faculty.name and faculty.name.strip())
    evaluator.add_custom_node(
        result=name_present,
        id=f"faculty_{idx+1}_name",
        desc="Full name of the faculty member is provided",
        parent=fac_parent,
        critical=True,
    )

    # 2) Profile URL validity - verify via URL if present; else fail
    if faculty and faculty.profile_url and faculty.profile_url.strip():
        profile_node = evaluator.add_leaf(
            id=f"faculty_{idx+1}_profile_url",
            desc="Valid URL to faculty member's official university profile page is provided",
            parent=fac_parent,
            critical=True,
        )
        # Compose a cautious claim focusing on official profile nature and identity
        claimed_uni = faculty.university or "New York University or Stanford University"
        name_for_claim = faculty.name or "the faculty member"
        dept_fragment = f" in the {faculty.department}" if faculty and faculty.department else ""
        claim_profile = (
            f"This webpage is an official faculty profile page for {name_for_claim} at {claimed_uni}{dept_fragment}."
        )
        await evaluator.verify(
            claim=claim_profile,
            node=profile_node,
            sources=faculty.profile_url,
            additional_instruction=PROFILE_URL_INSTRUCTIONS,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"faculty_{idx+1}_profile_url",
            desc="Valid URL to faculty member's official university profile page is provided",
            parent=fac_parent,
            critical=True,
        )

    # 3) University + Department in allowed CS list — verify on profile page
    uni_node = evaluator.add_leaf(
        id=f"faculty_{idx+1}_university",
        desc="Faculty member is affiliated with NYU (Courant or Tandon CS) or Stanford CS Department",
        parent=fac_parent,
        critical=True,
    )
    uni = faculty.university or ""
    dept = faculty.department or ""
    claim_uni = (
        f"The official profile page indicates that this person is affiliated with {uni}, {dept}, "
        f"and that department is within the allowed list for the task."
    )
    await evaluator.verify(
        claim=claim_uni,
        node=uni_node,
        sources=faculty.profile_url if faculty and faculty.profile_url else None,
        additional_instruction=ALLOWED_DEPARTMENT_INSTRUCTIONS,
    )

    # 4) Tenure-track position — verify via profile page
    pos_node = evaluator.add_leaf(
        id=f"faculty_{idx+1}_position",
        desc="Faculty member holds a tenure-track position (Assistant/Associate/Full Professor, not visiting/adjunct/emeritus)",
        parent=fac_parent,
        critical=True,
    )
    title_str = faculty.title or ""
    claim_pos = (
        f"The current academic title on the profile is '{title_str}', and this title corresponds to a tenure-track rank "
        f"(Assistant Professor, Associate Professor, or Professor) and not visiting, adjunct, clinical, research, or emeritus."
    )
    await evaluator.verify(
        claim=claim_pos,
        node=pos_node,
        sources=faculty.profile_url if faculty and faculty.profile_url else None,
        additional_instruction=TENURE_TRACK_INSTRUCTIONS,
    )

    # 5) Research area AI/ML — verify via profile page
    ra_node = evaluator.add_leaf(
        id=f"faculty_{idx+1}_research_area",
        desc="Faculty member lists AI or ML as a primary research area on their official university profile",
        parent=fac_parent,
        critical=True,
    )
    ra_list = faculty.research_areas or []
    ra_text = _safe_join(ra_list)
    claim_ra = (
        f"The official faculty profile lists at least one of the following as a primary research area: {ra_text}. "
        f"In particular, Artificial Intelligence or Machine Learning (or clear synonyms) should be included."
    )
    await evaluator.verify(
        claim=claim_ra,
        node=ra_node,
        sources=faculty.profile_url if faculty and faculty.profile_url else None,
        additional_instruction=AI_ML_INSTRUCTIONS,
    )

    # 6) NSF grant details (title + award number + valid NSF URL) — verify against NSF page
    # If missing URL or award number/title, fail this leaf.
    has_grant = bool(faculty and faculty.grant)
    has_nsf_url = bool(has_grant and faculty.grant and faculty.grant.nsf_url and faculty.grant.nsf_url.strip())
    has_award_num = bool(has_grant and faculty.grant and faculty.grant.award_number and faculty.grant.award_number.strip())
    has_award_title = bool(has_grant and faculty.grant and faculty.grant.title and faculty.grant.title.strip())

    if has_nsf_url and has_award_num and has_award_title:
        grant_details_node = evaluator.add_leaf(
            id=f"faculty_{idx+1}_grant_details",
            desc="NSF grant title and award number are provided with valid NSF award page URL",
            parent=fac_parent,
            critical=True,
        )
        claim_grant_details = (
            f"The NSF award page shows award number '{faculty.grant.award_number}' and a title that matches "
            f"'{faculty.grant.title}'."
        )
        await evaluator.verify(
            claim=claim_grant_details,
            node=grant_details_node,
            sources=faculty.grant.nsf_url,
            additional_instruction=NSF_GRANT_DETAILS_INSTRUCTIONS,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"faculty_{idx+1}_grant_details",
            desc="NSF grant title and award number are provided with valid NSF award page URL",
            parent=fac_parent,
            critical=True,
        )

    # 7) NSF funding criteria — verify via NSF award page
    nsf_node = evaluator.add_leaf(
        id=f"faculty_{idx+1}_nsf_funding",
        desc="Faculty member has at least one active NSF CISE grant awarded or renewed in 2024 or 2025",
        parent=fac_parent,
        critical=True,
    )
    # Construct a robust claim including directorate, year, activity, and investigator identity
    person_name = faculty.name or ""
    uni_for_nsf = faculty.university or ""
    award_num_for_claim = faculty.grant.award_number if (faculty and faculty.grant and faculty.grant.award_number) else ""
    claim_nsf = (
        f"The NSF award page indicates that award {award_num_for_claim} is within the Directorate for Computer and "
        f"Information Science and Engineering (CISE), lists {person_name} as PI or Co-PI (or similar investigator), "
        f"was awarded or renewed in 2024 or 2025, and is currently active (or within an active performance period)."
    )
    await evaluator.verify(
        claim=claim_nsf,
        node=nsf_node,
        sources=faculty.grant.nsf_url if (faculty and faculty.grant and faculty.grant.nsf_url) else None,
        additional_instruction=NSF_FUNDING_INSTRUCTIONS,
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

    # Record allowed department policy for transparency
    evaluator.add_custom_info(
        info={
            "allowed_departments_policy": ALLOWED_DEPARTMENT_INSTRUCTIONS,
            "tenure_track_policy": TENURE_TRACK_INSTRUCTIONS,
            "ai_ml_policy": AI_ML_INSTRUCTIONS,
            "nsf_funding_policy": NSF_FUNDING_INSTRUCTIONS,
        },
        info_type="policies",
        info_name="evaluation_policies",
    )

    # Extract up to 4 faculty entries from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_faculty(),
        template_class=FacultyList,
        extraction_name="extracted_faculty_list",
    )

    faculties = list(extracted.faculties) if extracted and extracted.faculties else []
    # Keep only the first 4; pad with empty placeholders if fewer
    faculties = faculties[:4]
    while len(faculties) < 4:
        faculties.append(FacultyInfo())

    # Build verification subtrees for each faculty
    for i in range(4):
        await verify_one_faculty(evaluator, root, faculties[i], i)

    return evaluator.get_summary()