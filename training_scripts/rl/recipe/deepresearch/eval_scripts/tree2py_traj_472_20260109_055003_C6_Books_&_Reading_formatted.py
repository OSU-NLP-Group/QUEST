import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cf_first_novel_prize_2024_mfa_midwest_constraints"
TASK_DESCRIPTION = (
    "Identify the author who won the Center for Fiction First Novel Prize in 2024 for their debut novel, "
    "who currently teaches writing/creative writing at a U.S. college or university. Provide: "
    "1) author's full name, 2) title of their prize-winning debut novel, 3) current college/university where they teach as writing faculty, "
    "4) the publisher of their debut novel (imprint + parent publishing group), "
    "5) the U.S. university where they obtained their MFA between 2015 and 2020 inclusive. "
    "The MFA university must: have an official Christian religious affiliation; be located in a Midwestern U.S. state; "
    "be a member of NCAA Division I athletics; have been founded between 1800 and 1899 inclusive. "
    "Each piece of information must include supporting reference URL(s)."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PrizeInfo(BaseModel):
    author_name: Optional[str] = None
    novel_title: Optional[str] = None
    winner_urls: List[str] = Field(default_factory=list)
    novel_urls: List[str] = Field(default_factory=list)
    debut_urls: List[str] = Field(default_factory=list)


class TeachingInfo(BaseModel):
    institution: Optional[str] = None
    role: Optional[str] = None
    teaching_urls: List[str] = Field(default_factory=list)


class PublisherInfo(BaseModel):
    imprint: Optional[str] = None
    parent_group: Optional[str] = None
    publisher_urls: List[str] = Field(default_factory=list)
    major_group_urls: List[str] = Field(default_factory=list)


class MFAInfo(BaseModel):
    university: Optional[str] = None
    completion_year: Optional[str] = None
    degree_urls: List[str] = Field(default_factory=list)
    affiliation_urls: List[str] = Field(default_factory=list)
    location_state: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)
    ncaa_urls: List[str] = Field(default_factory=list)
    founding_urls: List[str] = Field(default_factory=list)
    founded_year: Optional[str] = None


class OverallExtraction(BaseModel):
    prize: Optional[PrizeInfo] = None
    teaching: Optional[TeachingInfo] = None
    publisher: Optional[PublisherInfo] = None
    mfa: Optional[MFAInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract the structured information below strictly from the provided answer text. Do not invent any values. 
    Return null for any missing field, and return empty arrays for missing URL lists.

    Prize section (the 2024 Center for Fiction First Novel Prize):
    - prize.author_name: full name of the author identified as the 2024 winner
    - prize.novel_title: title of the prize-winning debut novel
    - prize.winner_urls: list of URL(s) that confirm the author won the Center for Fiction First Novel Prize in 2024
    - prize.novel_urls: list of URL(s) that support the prize-winning novel title and association with the author
    - prize.debut_urls: list of URL(s) confirming the book is the author's debut/first novel

    Current teaching position:
    - teaching.institution: name of the current U.S. college/university where the author teaches
    - teaching.role: the current teaching/faculty role (e.g., assistant professor, lecturer, writer-in-residence) indicating teaching in writing/creative writing or in English teaching writing
    - teaching.teaching_urls: URL(s) that confirm the author currently teaches at that institution and role

    Publisher details:
    - publisher.imprint: the publisher imprint of the debut novel
    - publisher.parent_group: the parent publishing group that owns the imprint
    - publisher.publisher_urls: URL(s) supporting the imprint and its relationship to the novel
    - publisher.major_group_urls: URL(s) supporting that the parent group is a major publishing group and/or that the imprint is owned by that parent

    MFA details and university constraints:
    - mfa.university: the U.S. university where the author obtained their MFA
    - mfa.completion_year: the MFA completion year (4-digit) if explicitly provided; otherwise null
    - mfa.degree_urls: URL(s) confirming the MFA credential and the completion timeframe (between 2015 and 2020 inclusive)
    - mfa.affiliation_urls: URL(s) confirming the university has an official Christian religious affiliation (Catholic/Protestant/other Christian)
    - mfa.location_state: the U.S. state of the university (full name or 2-letter code) if provided
    - mfa.location_urls: URL(s) confirming the university's location in a Midwestern U.S. state
    - mfa.ncaa_urls: URL(s) confirming the university is a member of NCAA Division I
    - mfa.founding_urls: URL(s) confirming the university was founded between 1800 and 1899 inclusive
    - mfa.founded_year: the founding year (4-digit) if explicitly provided; otherwise null

    Important URL rules:
    - Only extract URLs explicitly present in the answer text. If the answer references a source without a concrete URL, set the corresponding URL list to empty.
    - Accept URLs in plain form or embedded in markdown links.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _nonempty_urls(urls: Optional[List[str]]) -> bool:
    return isinstance(urls, list) and len([u for u in urls if _nonempty_str(u)]) > 0


def _join_sources(*url_lists: List[str]) -> List[str]:
    result: List[str] = []
    for ul in url_lists:
        if isinstance(ul, list):
            for u in ul:
                if _nonempty_str(u):
                    result.append(u)
    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for u in result:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def verify_prize_winner_and_book(evaluator: Evaluator, parent_node, data: Optional[PrizeInfo]) -> None:
    node = evaluator.add_parallel(
        id="Prize_Winner_And_Book",
        desc="Correctly identify the 2024 prize-winning author and their debut novel, with citations.",
        parent=parent_node,
        critical=True,
    )

    author_name = data.author_name if data else None
    novel_title = data.novel_title if data else None
    winner_urls = data.winner_urls if data else []
    novel_urls = data.novel_urls if data else []
    debut_urls = data.debut_urls if data else []

    # Critical gating nodes to enforce presence of essential info and URLs
    evaluator.add_custom_node(
        result=_nonempty_str(author_name),
        id="Prize_Author_Name_Present",
        desc="Author name for prize winner is present in the answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_urls(winner_urls),
        id="Winner_URLs_Provided",
        desc="At least one URL provided to support that the author won the 2024 Center for Fiction First Novel Prize",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_str(novel_title),
        id="Prize_Novel_Title_Present",
        desc="Prize-winning novel title is present in the answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_urls(novel_urls),
        id="Prize_Novel_Title_URLs_Provided",
        desc="At least one URL provided to support the prize-winning novel title",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_urls(debut_urls),
        id="Debut_Status_URLs_Provided",
        desc="At least one URL provided to confirm the book is the author's debut/first novel",
        parent=node,
        critical=True
    )

    # Leaf: Winner 2024 with full name and URL
    leaf_winner = evaluator.add_leaf(
        id="Winner_2024_With_Full_Name_And_URL",
        desc="Provide the author's full name and confirm via URL(s) that they won the Center for Fiction First Novel Prize in 2024.",
        parent=node,
        critical=True,
    )
    winner_claim = f"The author {_safe_val(author_name)} won the Center for Fiction First Novel Prize in 2024."
    await evaluator.verify(
        claim=winner_claim,
        node=leaf_winner,
        sources=winner_urls,
        additional_instruction="Verify the specific award (Center for Fiction First Novel Prize) and year 2024. Accept minor phrasing variants (e.g., 'First Novel Prize')."
    )

    # Leaf: Prize-winning novel title with URL
    leaf_title = evaluator.add_leaf(
        id="Prize_Winning_Novel_Title_With_URL",
        desc="Provide the title of the prize-winning novel and support it with URL(s).",
        parent=node,
        critical=True,
    )
    title_claim = f"The title of the author's prize-winning debut novel is '{_safe_val(novel_title)}'."
    await evaluator.verify(
        claim=title_claim,
        node=leaf_title,
        sources=novel_urls,
        additional_instruction="Check that the page associates the novel title with the author and the Center for Fiction First Novel Prize recognition."
    )

    # Leaf: Debut novel confirmed with URL
    leaf_debut = evaluator.add_leaf(
        id="Debut_Novel_Confirmed_With_URL",
        desc="Confirm via URL(s) that the prize-winning book is the author's debut/first novel.",
        parent=node,
        critical=True,
    )
    debut_claim = f"'{_safe_val(novel_title)}' is the author's debut (first) novel."
    await evaluator.verify(
        claim=debut_claim,
        node=leaf_debut,
        sources=debut_urls,
        additional_instruction="Look for explicit phrases like 'debut novel' or 'first novel'."
    )


async def verify_current_teaching_position(evaluator: Evaluator, parent_node, data: Optional[TeachingInfo], author_name: Optional[str]) -> None:
    node = evaluator.add_parallel(
        id="Current_Teaching_Position",
        desc="Verify the author currently teaches writing/creative writing (or writing within an English department) at a U.S. college/university, with citations.",
        parent=parent_node,
        critical=True,
    )

    institution = data.institution if data else None
    role = data.role if data else None
    teaching_urls = data.teaching_urls if data else []

    evaluator.add_custom_node(
        result=_nonempty_str(institution),
        id="Teaching_Institution_Present",
        desc="Teaching institution name is present in the answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_urls(teaching_urls),
        id="Teaching_URLs_Provided",
        desc="At least one URL provided to confirm current teaching role and institution",
        parent=node,
        critical=True
    )

    leaf_teach = evaluator.add_leaf(
        id="US_Teaching_Institution_And_Role_With_URL",
        desc="Provide the name of the current U.S. college/university where the author teaches and confirm via URL(s) that the role is a current teaching/faculty position in writing/creative writing (or English department teaching writing).",
        parent=node,
        critical=True,
    )
    role_str = role if _nonempty_str(role) else "a teaching/faculty role"
    teach_claim = (
        f"As of now, {_safe_val(author_name)} holds {role_str} in writing/creative writing (or teaches writing within an English department) "
        f"at {_safe_val(institution)}, which is a U.S. college or university."
    )
    await evaluator.verify(
        claim=teach_claim,
        node=leaf_teach,
        sources=teaching_urls,
        additional_instruction="Confirm that this is a current teaching/faculty position in writing/creative writing (or English teaching writing). Ignore alumni or past roles."
    )


async def verify_publisher_details(evaluator: Evaluator, parent_node, data: Optional[PublisherInfo], novel_title: Optional[str]) -> None:
    node = evaluator.add_parallel(
        id="Publisher_Details",
        desc="Provide publisher details for the debut novel (imprint + parent group) and confirm major publishing group status, with citations.",
        parent=parent_node,
        critical=True,
    )

    imprint = data.imprint if data else None
    parent_group = data.parent_group if data else None
    pub_urls = data.publisher_urls if data else []
    major_urls = data.major_group_urls if data else []

    evaluator.add_custom_node(
        result=_nonempty_str(imprint) and _nonempty_str(parent_group),
        id="Publisher_Fields_Present",
        desc="Publisher imprint and parent group are present in the answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_urls(pub_urls),
        id="Publisher_URLs_Provided",
        desc="At least one URL provided to support imprint and its association with the novel",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_urls(major_urls) or _nonempty_urls(pub_urls),
        id="Major_Group_URLs_Provided",
        desc="At least one URL provided to support parent group ownership and major publishing group status",
        parent=node,
        critical=True
    )

    # Leaf: Imprint and parent group with URL
    leaf_imprint = evaluator.add_leaf(
        id="Imprint_And_Parent_Group_With_URL",
        desc="Provide the publisher imprint and the parent publishing group for the debut novel, supported by URL(s).",
        parent=node,
        critical=True,
    )
    imprint_claim = (
        f"The debut novel '{_safe_val(novel_title)}' was published by {_safe_val(imprint)}, "
        f"an imprint of {_safe_val(parent_group)}."
    )
    await evaluator.verify(
        claim=imprint_claim,
        node=leaf_imprint,
        sources=_join_sources(pub_urls, major_urls),
        additional_instruction="Confirm that the page states the novel's publisher imprint and that the imprint is owned by the specified parent group."
    )

    # Leaf: Major publishing group constraint met with URL
    leaf_major = evaluator.add_leaf(
        id="Major_Publishing_Group_Constraint_Met_With_URL",
        desc="Confirm via URL(s) that the publishing house is part of a major publishing group (i.e., a large publishing group owning the imprint), as required by the constraints.",
        parent=node,
        critical=True,
    )
    major_claim = (
        f"{_safe_val(parent_group)} is a major publishing group and owns the {_safe_val(imprint)} imprint."
    )
    await evaluator.verify(
        claim=major_claim,
        node=leaf_major,
        sources=_join_sources(major_urls, pub_urls),
        additional_instruction="Accept sources that explicitly state the parent group is one of the Big Five or otherwise a major publishing group, or that it is a large publishing conglomerate owning the imprint."
    )


async def verify_mfa_degree_and_constraints(evaluator: Evaluator, parent_node, data: Optional[MFAInfo], author_name: Optional[str]) -> None:
    node = evaluator.add_parallel(
        id="MFA_Degree_And_University_Constraints",
        desc="Provide the author's MFA-granting university and verify all required MFA timing and university characteristics, with citations.",
        parent=parent_node,
        critical=True,
    )

    university = data.university if data else None
    completion_year = data.completion_year if data else None
    degree_urls = data.degree_urls if data else []
    affiliation_urls = data.affiliation_urls if data else []
    location_state = data.location_state if data else None
    location_urls = data.location_urls if data else []
    ncaa_urls = data.ncaa_urls if data else []
    founding_urls = data.founding_urls if data else []
    founded_year = data.founded_year if data else None

    # Gating nodes for presence of MFA key info and URLs
    evaluator.add_custom_node(
        result=_nonempty_str(university),
        id="MFA_University_Name_Present",
        desc="MFA university name is present in the answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_urls(degree_urls),
        id="MFA_Degree_URLs_Provided",
        desc="At least one URL provided to confirm the MFA degree and completion window",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_urls(affiliation_urls),
        id="MFA_Affiliation_URLs_Provided",
        desc="At least one URL provided to confirm Christian affiliation",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_urls(location_urls),
        id="MFA_Location_URLs_Provided",
        desc="At least one URL provided to confirm Midwestern location",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_urls(ncaa_urls),
        id="MFA_NCAA_URLs_Provided",
        desc="At least one URL provided to confirm NCAA Division I membership",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty_urls(founding_urls),
        id="MFA_Founding_URLs_Provided",
        desc="At least one URL provided to confirm 19th-century founding",
        parent=node,
        critical=True
    )

    # Leaf: MFA University name, US, and completion window
    leaf_mfa_name = evaluator.add_leaf(
        id="MFA_University_Name_US_And_Completion_Window_With_URL",
        desc="Provide the name of the U.S. university where the author obtained their MFA and confirm via URL(s) that completion occurred between 2015 and 2020 inclusive.",
        parent=node,
        critical=True,
    )
    # Compose claim with optional explicit year if provided
    if _nonempty_str(completion_year):
        mfa_claim = (
            f"{_safe_val(author_name)} obtained an MFA from {_safe_val(university)}, a U.S. university, in {_safe_val(completion_year)}, "
            "which lies between 2015 and 2020 inclusive."
        )
    else:
        mfa_claim = (
            f"{_safe_val(author_name)} obtained an MFA from {_safe_val(university)}, a U.S. university, "
            "with completion occurring between 2015 and 2020 inclusive."
        )
    await evaluator.verify(
        claim=mfa_claim,
        node=leaf_mfa_name,
        sources=degree_urls,
        additional_instruction="Verify the MFA credential and ensure the completion year or timeframe is between 2015 and 2020 inclusive. Confirm that the institution is in the United States."
    )

    # Leaf: Christian affiliation
    leaf_affil = evaluator.add_leaf(
        id="MFA_University_Christian_Affiliation_With_URL",
        desc="Confirm via URL(s) that the MFA-granting university has an official Christian religious affiliation (Catholic/Protestant/other Christian denomination).",
        parent=node,
        critical=True,
    )
    affil_claim = f"{_safe_val(university)} has an official Christian religious affiliation (Catholic, Protestant, or another Christian denomination)."
    await evaluator.verify(
        claim=affil_claim,
        node=leaf_affil,
        sources=affiliation_urls,
        additional_instruction="The affiliation must be official, not merely historical or cultural. Look for explicit statements of religious affiliation on official or reliable pages."
    )

    # Leaf: Midwestern state
    leaf_midwest = evaluator.add_leaf(
        id="MFA_University_Midwestern_State_With_URL",
        desc="Confirm via URL(s) that the MFA-granting university is located in a Midwestern U.S. state as specified in the constraints.",
        parent=node,
        critical=True,
    )
    if _nonempty_str(location_state):
        midwest_claim = f"{_safe_val(university)} is located in {_safe_val(location_state)}, a Midwestern U.S. state."
    else:
        midwest_claim = f"{_safe_val(university)} is located in a Midwestern U.S. state."
    await evaluator.verify(
        claim=midwest_claim,
        node=leaf_midwest,
        sources=location_urls,
        additional_instruction="Use the provided page(s) to confirm the university's state and that it is considered part of the U.S. Midwest."
    )

    # Leaf: NCAA Division I membership
    leaf_ncaa = evaluator.add_leaf(
        id="MFA_University_NCAA_Division_I_With_URL",
        desc="Confirm via URL(s) that the MFA-granting university is a member of NCAA Division I athletics.",
        parent=node,
        critical=True,
    )
    ncaa_claim = f"{_safe_val(university)} is a member of NCAA Division I athletics."
    await evaluator.verify(
        claim=ncaa_claim,
        node=leaf_ncaa,
        sources=ncaa_urls,
        additional_instruction="The source should clearly indicate NCAA Division I membership."
    )

    # Leaf: Founded between 1800 and 1899 inclusive
    leaf_founded = evaluator.add_leaf(
        id="MFA_University_Founded_1800_1899_With_URL",
        desc="Confirm via URL(s) that the MFA-granting university was founded between 1800 and 1899 inclusive.",
        parent=node,
        critical=True,
    )
    if _nonempty_str(founded_year):
        founded_claim = f"{_safe_val(university)} was founded in {_safe_val(founded_year)}, which is between 1800 and 1899 inclusive."
    else:
        founded_claim = f"{_safe_val(university)} was founded between 1800 and 1899 inclusive."
    await evaluator.verify(
        claim=founded_claim,
        node=leaf_founded,
        sources=founding_urls,
        additional_instruction="Confirm the founding year lies within the 19th century (1800–1899 inclusive)."
    )


def _safe_val(value: Optional[str]) -> str:
    return value if _nonempty_str(value) else ""


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
    Evaluate an answer for the 2024 Center for Fiction First Novel Prize task with MFA and university constraints.
    """
    # Initialize evaluator and root
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

    # Add top-level critical node mirroring rubric "Task_Completion"
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Identify the 2024 Center for Fiction First Novel Prize winner meeting all constraints and provide all required fields with supporting reference URLs.",
        parent=root,
        critical=True
    )

    # Extract all structured info in one pass
    extracted: OverallExtraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=OverallExtraction,
        extraction_name="extracted_structured_info"
    )

    prize = extracted.prize or PrizeInfo()
    teaching = extracted.teaching or TeachingInfo()
    publisher = extracted.publisher or PublisherInfo()
    mfa = extracted.mfa or MFAInfo()

    # Build and verify each major subtree
    await verify_prize_winner_and_book(evaluator, task_node, prize)
    await verify_current_teaching_position(evaluator, task_node, teaching, prize.author_name)
    await verify_publisher_details(evaluator, task_node, publisher, prize.novel_title)
    await verify_mfa_degree_and_constraints(evaluator, task_node, mfa, prize.author_name)

    # Return evaluation summary
    return evaluator.get_summary()