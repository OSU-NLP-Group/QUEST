import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "career_directors_diverse"
TASK_DESCRIPTION = (
    "Identify three career services professionals who currently hold director-level positions at three different "
    "universities in three different U.S. states. For each professional, provide university information (name, state, "
    "direct staff directory URL), professional information (full name, exact position title, functional area/specialty, "
    "official university email as listed on the career center website), and LinkedIn profile (URL, confirms current position, "
    "publicly accessible). Additional requirements: the three universities must be in three different U.S. states; each "
    "professional must work in a different functional area; all information must be verifiable from publicly accessible "
    "sources (university websites and LinkedIn)."
)

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class StaffMember(BaseModel):
    university_name: Optional[str] = None
    state: Optional[str] = None
    staff_page_url: Optional[str] = None

    full_name: Optional[str] = None
    title: Optional[str] = None
    functional_area: Optional[str] = None
    email: Optional[str] = None

    linkedin_url: Optional[str] = None


class StaffExtraction(BaseModel):
    people: List[StaffMember] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_staff() -> str:
    return """
    Extract up to three career services professionals exactly as presented in the answer text. For each professional, return an object with these fields:

    1) university_name: The university name.
    2) state: The U.S. state of the university (full name or two-letter abbreviation).
    3) staff_page_url: A direct URL that leads to either the career center staff directory listing or the individual's staff profile page within the university's career services/career center site.
    4) full_name: The staff member's full name, exactly as listed on the university website (if provided in the answer).
    5) title: The staff member's exact position title as listed on the university website (if provided in the answer). This title must include "Career" or "Careers" AND include one of: Executive Director, Director, Associate Director, or Assistant Director.
    6) functional_area: The specific functional area/specialty (e.g., Employer Relations, Career Education, Career Readiness, Healthcare Careers, Alumni Career Engagement, etc.), if mentioned.
    7) email: The official university email address for the staff member, as claimed in the answer. If not provided, return null.
    8) linkedin_url: The LinkedIn profile URL, in the format linkedin.com/in/[username], if provided.

    Return a JSON object:
    {
      "people": [
         { ... }, { ... }, { ... }
      ]
    }

    Rules:
    - Extract only what is explicitly present in the answer; do not invent or infer.
    - If a field is missing for a person, set it to null.
    - If more than 3 professionals are listed, include only the first three in the final JSON.
    - For URLs, include the full URL with protocol; if missing protocol, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_nonempty(text: Optional[str]) -> bool:
    return bool(text and str(text).strip())


def normalize_key(text: Optional[str]) -> Optional[str]:
    if not is_nonempty(text):
        return None
    return re.sub(r"\s+", " ", text.strip().lower())


def linkedin_in_profile_format(url: Optional[str]) -> bool:
    if not is_nonempty(url):
        return False
    # Accept http or https, with or without www, but must include /in/
    pattern = r"^(https?://)?(www\.)?linkedin\.com/in/[^/\s]+/?$"
    return re.match(pattern, url.strip()) is not None


def get_person_or_generic_name(sm: StaffMember) -> str:
    return sm.full_name.strip() if is_nonempty(sm.full_name) else "the staff member"


# --------------------------------------------------------------------------- #
# Verification for a single staff member                                      #
# --------------------------------------------------------------------------- #
async def verify_staff_member(
    evaluator: Evaluator,
    parent_node,
    sm: StaffMember,
    index: int
) -> None:
    idx = index + 1

    # Create Staff_Member_i sequential node
    staff_node = evaluator.add_sequential(
        id=f"Staff_Member_{idx}",
        desc=f"{['First','Second','Third'][index]} career services professional identified with complete information",
        parent=parent_node,
        critical=False
    )

    # --- University Verification (parallel, critical) ---
    uni_node = evaluator.add_parallel(
        id=f"SM{idx}_University_Verification",
        desc="Verify the university identification and accessible career center website with staff directory",
        parent=staff_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_nonempty(sm.university_name),
        id=f"SM{idx}_University_Name",
        desc="The name of the university is explicitly provided",
        parent=uni_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_nonempty(sm.state),
        id=f"SM{idx}_University_State",
        desc="The state where the university is located is explicitly provided",
        parent=uni_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_nonempty(sm.staff_page_url),
        id=f"SM{idx}_Website_URL",
        desc="A direct URL to the career center staff directory or staff page is provided",
        parent=uni_node,
        critical=True
    )

    # Accessible career center page (uses URL)
    uni_access_leaf = evaluator.add_leaf(
        id=f"SM{idx}_Career_Center_Website",
        desc="The university has a publicly accessible career services or career center website",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided URL is a publicly accessible career center staff directory or staff profile page on the university website.",
        node=uni_access_leaf,
        sources=sm.staff_page_url,
        additional_instruction="Confirm the page loads with visible content and does not require login to view basic staff information. A page hosted under an official university domain (often .edu) is expected."
    )

    # Staff directory or staff profile page check
    dir_leaf = evaluator.add_leaf(
        id=f"SM{idx}_Staff_Directory",
        desc="The career center website includes a public staff directory listing",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim="This URL points to either a staff directory that lists multiple staff members and titles OR an individual staff profile page within the career center site.",
        node=dir_leaf,
        sources=sm.staff_page_url,
        additional_instruction="If the page shows multiple staff entries, treat it as a directory. If it shows one staff member with name/title/email or bio, treat it as a staff profile page. Either satisfies the requirement."
    )

    # --- Staff Information (parallel, critical) ---
    info_node = evaluator.add_parallel(
        id=f"SM{idx}_Staff_Information",
        desc="Complete information about the identified staff member",
        parent=staff_node,
        critical=True
    )

    # Role Requirements (parallel, critical)
    role_node = evaluator.add_parallel(
        id=f"SM{idx}_Role_Requirements",
        desc="The staff member meets all role-level requirements",
        parent=info_node,
        critical=True
    )

    # Director-level check on the staff page
    director_leaf = evaluator.add_leaf(
        id=f"SM{idx}_Director_Level",
        desc="The position title includes one of the following: Executive Director, Director, Associate Director, or Assistant Director",
        parent=role_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The staff page shows that {get_person_or_generic_name(sm)} holds a director-level role (Executive Director, Director, Associate Director, or Assistant Director).",
        node=director_leaf,
        sources=sm.staff_page_url,
        additional_instruction="Accept plausible variants like 'Senior Director', 'Assistant Director', 'Associate Director', 'Co-Director'. Titles like 'Manager' or 'Coordinator' should NOT qualify as director-level."
    )

    # "Career" in title check on staff page
    career_in_title_leaf = evaluator.add_leaf(
        id=f"SM{idx}_Career_In_Title",
        desc="The position title explicitly includes the word 'Career' or 'Careers'",
        parent=role_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The staff member's title on the staff page contains the word 'Career' or 'Careers'.",
        node=career_in_title_leaf,
        sources=sm.staff_page_url,
        additional_instruction="Check the exact title text shown for the staff member. The word 'Career' or 'Careers' must appear."
    )

    # Functional area presence on staff page
    functional_leaf = evaluator.add_leaf(
        id=f"SM{idx}_Functional_Area",
        desc="A specific functional area or specialty is identified (e.g., Employer Relations, Career Education, Healthcare Careers, etc.)",
        parent=role_node,
        critical=True
    )
    await evaluator.verify(
        claim="The staff page indicates a specific functional area or specialty for the staff member (e.g., Employer Relations, Career Education, Career Readiness, Healthcare Careers, Alumni Career Engagement, etc.).",
        node=functional_leaf,
        sources=sm.staff_page_url,
        additional_instruction="Look for a subdivision, team, focus area, or specialty label associated with this staff member. It may appear in the title, department line, or bio."
    )

    # Contact Information (parallel, critical)
    contact_node = evaluator.add_parallel(
        id=f"SM{idx}_Contact_Information",
        desc="Complete and verified contact information for the staff member",
        parent=info_node,
        critical=True
    )

    # Full name exact on staff page
    name_leaf = evaluator.add_leaf(
        id=f"SM{idx}_Full_Name",
        desc="The staff member's full name is provided exactly as listed on the university website",
        parent=contact_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The staff page lists the staff member with the full name '{sm.full_name}'.",
        node=name_leaf,
        sources=sm.staff_page_url,
        additional_instruction="Match exactly as it appears on the page; allow only minor punctuation/casing differences. If multiple names are shown, ensure the claimed full name corresponds to the staff member on this page."
    )

    # Email presence (existence)
    evaluator.add_custom_node(
        result=is_nonempty(sm.email),
        id=f"SM{idx}_Email",
        desc="The staff member's official university email address is provided",
        parent=contact_node,
        critical=True
    )

    # Email appears on staff page
    email_source_leaf = evaluator.add_leaf(
        id=f"SM{idx}_Email_Source",
        desc="The email address is verified to appear on the official university career center website",
        parent=contact_node,
        critical=True
    )
    email_claim = (
        f"The email address '{sm.email}' appears on the provided career center staff page."
        if is_nonempty(sm.email) else
        "The staff page displays an official university email address for the staff member."
    )
    await evaluator.verify(
        claim=email_claim,
        node=email_source_leaf,
        sources=sm.staff_page_url,
        additional_instruction="Accept explicit email text, mailto links, or slightly obfuscated formats (e.g., 'name [at] university.edu'). The email domain should match the university."
    )

    # Exact position title appears on staff page
    title_exact_leaf = evaluator.add_leaf(
        id=f"SM{idx}_Title_Exact",
        desc="The exact position title as listed on the university website is provided",
        parent=contact_node,
        critical=True
    )
    title_claim = (
        f"The exact position title '{sm.title}' appears on the staff page for {get_person_or_generic_name(sm)}."
        if is_nonempty(sm.title) else
        "The staff page shows the staff member's exact position title."
    )
    await evaluator.verify(
        claim=title_claim,
        node=title_exact_leaf,
        sources=sm.staff_page_url,
        additional_instruction="Match the title string shown on the page; small punctuation or casing variations are acceptable."
    )

    # LinkedIn Profile (sequential, critical)
    linkedin_node = evaluator.add_sequential(
        id=f"SM{idx}_LinkedIn_Profile",
        desc="LinkedIn profile information for the staff member",
        parent=info_node,
        critical=True
    )

    # LinkedIn URL format check (simple verify)
    linkedin_url_leaf = evaluator.add_leaf(
        id=f"SM{idx}_LinkedIn_URL",
        desc="A LinkedIn profile URL is provided in the standard format (linkedin.com/in/[username])",
        parent=linkedin_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LinkedIn URL '{sm.linkedin_url}' is in the format linkedin.com/in/[username].",
        node=linkedin_url_leaf,
        additional_instruction="Allow optional protocol (http/https) and optional trailing slash. The path must include '/in/'."
    )

    # LinkedIn current position check (by URL)
    linkedin_position_leaf = evaluator.add_leaf(
        id=f"SM{idx}_LinkedIn_Current_Position",
        desc="The LinkedIn profile lists the current position at the identified university",
        parent=linkedin_node,
        critical=True
    )
    li_position_claim = (
        f"The LinkedIn profile shows that {get_person_or_generic_name(sm)} currently holds a position at {sm.university_name}, "
        f"consistent with the title '{sm.title}'."
        if is_nonempty(sm.university_name) and is_nonempty(sm.title)
        else "The LinkedIn profile shows that the staff member currently holds a role at the identified university."
    )
    await evaluator.verify(
        claim=li_position_claim,
        node=linkedin_position_leaf,
        sources=sm.linkedin_url,
        additional_instruction="Check the Experience section or headline for the current role and employer. Allow reasonable title variants; the university name must match or closely match."
    )

    # LinkedIn public accessibility check (by URL)
    linkedin_access_leaf = evaluator.add_leaf(
        id=f"SM{idx}_LinkedIn_Accessible",
        desc="The LinkedIn profile is publicly accessible (not requiring login to view basic information)",
        parent=linkedin_node,
        critical=True
    )
    await evaluator.verify(
        claim="The LinkedIn profile is publicly accessible and displays at least basic information without requiring login.",
        node=linkedin_access_leaf,
        sources=sm.linkedin_url,
        additional_instruction="If the page presents only a sign-in wall without any profile content, consider it not publicly accessible."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    # Initialize evaluator and root node
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

    # Create explicit Task_Completion node mirroring the rubric root
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Successfully identify three career services director-level professionals at three different universities in three different U.S. states, each working in distinct functional areas, with complete information including verified contact details and LinkedIn profiles",
        parent=root,
        critical=False
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_staff(),
        template_class=StaffExtraction,
        extraction_name="extracted_staff"
    )

    # Keep only first three people; pad if fewer
    people: List[StaffMember] = list(extracted.people[:3])
    while len(people) < 3:
        people.append(StaffMember())

    # Geographic Diversity (critical)
    states_norm = [normalize_key(p.state) for p in people if is_nonempty(p.state)]
    unique_states = set(s for s in states_norm if s)
    evaluator.add_custom_node(
        result=(len(unique_states) == 3),
        id="Geographic_Diversity",
        desc="The three universities are located in three different U.S. states",
        parent=task_node,
        critical=True
    )

    # Functional Area Diversity (critical)
    areas_norm = [normalize_key(p.functional_area) for p in people if is_nonempty(p.functional_area)]
    unique_areas = set(a for a in areas_norm if a)
    evaluator.add_custom_node(
        result=(len(unique_areas) == 3),
        id="Functional_Area_Diversity",
        desc="The three identified staff members work in three different functional areas within career services",
        parent=task_node,
        critical=True
    )

    # Verify each staff member subtree
    for i, sm in enumerate(people):
        await verify_staff_member(evaluator, task_node, sm, i)

    # Optional: record diversity details as custom info for transparency
    evaluator.add_custom_info(
        info={
            "states": [p.state for p in people],
            "unique_states_count": len(unique_states),
            "functional_areas": [p.functional_area for p in people],
            "unique_areas_count": len(unique_areas)
        },
        info_type="diversity_check",
        info_name="diversity_summary"
    )

    # Return evaluation summary
    return evaluator.get_summary()