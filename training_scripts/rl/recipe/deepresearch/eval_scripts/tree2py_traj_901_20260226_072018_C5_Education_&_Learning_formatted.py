import asyncio
import logging
from typing import Any, List, Optional, Dict, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "conf_membership_west_2023_2024"
TASK_DESCRIPTION = (
    "Identify at least four universities that joined either the Big Ten Conference or the Atlantic Coast Conference (ACC) as full members, "
    "where the official membership start date occurred in 2023 or 2024, and the institution is located in a state west of the Mississippi River. "
    "For each university you identify, provide: (1) the university name, (2) which conference they joined, (3) the official membership start date, "
    "(4) the state where the university is located, and (5) reference URLs documenting these details."
)

# -----------------------------------------------------------------------------
# Geography helpers: states west of the Mississippi River
# -----------------------------------------------------------------------------
STATE_ABBR: Dict[str, str] = {
    "ALABAMA": "AL","ALASKA": "AK","ARIZONA": "AZ","ARKANSAS": "AR","CALIFORNIA": "CA","COLORADO": "CO",
    "CONNECTICUT": "CT","DELAWARE": "DE","FLORIDA": "FL","GEORGIA": "GA","HAWAII": "HI","IDAHO": "ID",
    "ILLINOIS": "IL","INDIANA": "IN","IOWA": "IA","KANSAS": "KS","KENTUCKY": "KY","LOUISIANA": "LA",
    "MAINE": "ME","MARYLAND": "MD","MASSACHUSETTS": "MA","MICHIGAN": "MI","MINNESOTA": "MN","MISSISSIPPI": "MS",
    "MISSOURI": "MO","MONTANA": "MT","NEBRASKA": "NE","NEVADA": "NV","NEW HAMPSHIRE": "NH","NEW JERSEY": "NJ",
    "NEW MEXICO": "NM","NEW YORK": "NY","NORTH CAROLINA": "NC","NORTH DAKOTA": "ND","OHIO": "OH","OKLAHOMA": "OK",
    "OREGON": "OR","PENNSYLVANIA": "PA","RHODE ISLAND": "RI","SOUTH CAROLINA": "SC","SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN","TEXAS": "TX","UTAH": "UT","VERMONT": "VT","VIRGINIA": "VA","WASHINGTON": "WA",
    "WEST VIRGINIA": "WV","WISCONSIN": "WI","WYOMING": "WY","DISTRICT OF COLUMBIA": "DC"
}

# States considered west of the Mississippi River (majority of territory lies west of the river).
WEST_OF_MISS_STATES_ABBR: set = {
    "WA","OR","CA","NV","ID","MT","WY","UT","AZ","NM","CO","ND","SD","NE","KS","OK","TX","IA","MN","MO","AR","LA","AK","HI"
}

def normalize_state_name(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip().upper()
    # If already a two-letter abbreviation, return as is if valid
    if len(s) == 2 and s in STATE_ABBR.values():
        return s
    # Try direct full name
    if s in STATE_ABBR:
        return STATE_ABBR[s]
    # Handle common abbreviations or alternate forms
    s = s.replace(".", "").replace("STATE OF ", "").strip()
    if s in STATE_ABBR:
        return STATE_ABBR[s]
    # Try first word (e.g., "Washington State")
    first_word = s.split()[0]
    if first_word in STATE_ABBR:
        return STATE_ABBR[first_word]
    return None

def is_state_west_of_mississippi(state: Optional[str]) -> bool:
    abbr = normalize_state_name(state)
    if abbr is None:
        return False
    return abbr in WEST_OF_MISS_STATES_ABBR

# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class UniversityEntry(BaseModel):
    university_name: Optional[str] = None
    conference: Optional[str] = None  # Expected: "Big Ten" or "ACC" (allow variants like "Big Ten Conference", "Atlantic Coast Conference")
    membership_start_date: Optional[str] = None  # e.g., "August 2, 2024" or "2024-08-02"
    state: Optional[str] = None  # Full state name or abbreviation
    reference_urls: List[str] = Field(default_factory=list)  # URLs provided for this university
    announcement_date: Optional[str] = None  # If the answer cites the official announcement date, extract it


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityEntry] = Field(default_factory=list)

# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_universities() -> str:
    return (
        "Extract up to five universities listed in the answer that joined either the Big Ten Conference or the Atlantic Coast Conference (ACC). "
        "For each university, return a JSON object with:\n"
        "1) university_name: the university name as stated.\n"
        "2) conference: the conference they joined, as stated (e.g., 'Big Ten Conference', 'Big Ten', 'ACC', 'Atlantic Coast Conference').\n"
        "3) membership_start_date: the official membership start date as stated in the answer.\n"
        "4) state: the U.S. state where the university is located, as stated (full name or 2-letter abbreviation).\n"
        "5) reference_urls: an array of all URLs cited for this university in the answer (include conference announcements, school news releases, Wikipedia, or other pages explicitly mentioned).\n"
        "6) announcement_date: if the answer cites the official announcement date for the move, extract it; otherwise return null.\n\n"
        "Rules:\n"
        "- Only extract universities explicitly mentioned in the answer text.\n"
        "- Do not invent any fields. If a field is not present, return null (or empty array for URLs).\n"
        "- If the answer lists more than five universities, include only the first five as they appear.\n"
        "- Preserve the exact formatting for dates as provided (do not normalize).\n"
        "- For URLs provided in markdown links, extract the actual URL.\n"
    )

# -----------------------------------------------------------------------------
# Verification helpers
# -----------------------------------------------------------------------------
def make_required_fields_node(evaluator: Evaluator, parent: VerificationNode, entry: UniversityEntry, idx: int) -> VerificationNode:
    req_node = evaluator.add_parallel(
        id=f"univ_{idx}_required_output_fields",
        desc="All required output fields for this university are provided.",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(entry.university_name and entry.university_name.strip()),
        id=f"univ_{idx}_name_provided",
        desc="University name is provided.",
        parent=req_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(entry.conference and entry.conference.strip()),
        id=f"univ_{idx}_conference_provided",
        desc="Conference joined is provided.",
        parent=req_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(entry.membership_start_date and entry.membership_start_date.strip()),
        id=f"univ_{idx}_start_date_provided",
        desc="Official membership start date is provided.",
        parent=req_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(entry.state and entry.state.strip()),
        id=f"univ_{idx}_state_provided",
        desc="State where the university is located is provided.",
        parent=req_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(entry.reference_urls and len(entry.reference_urls) > 0),
        id=f"univ_{idx}_urls_provided",
        desc="At least one reference URL is provided for this university.",
        parent=req_node,
        critical=True
    )
    return req_node


async def make_eligibility_node(evaluator: Evaluator, parent: VerificationNode, entry: UniversityEntry, idx: int) -> VerificationNode:
    elig_node = evaluator.add_parallel(
        id=f"univ_{idx}_eligibility_criteria",
        desc="This university meets all eligibility constraints.",
        parent=parent,
        critical=True
    )

    # Conference is Big Ten or ACC
    conf_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_conference_is_bigten_or_acc",
        desc="University joined either the Big Ten Conference or the Atlantic Coast Conference (ACC).",
        parent=elig_node,
        critical=True
    )
    conf_claim = (
        f"The conference '{entry.conference}' refers to either the Big Ten Conference or the Atlantic Coast Conference (ACC)."
        if entry.conference else "The conference is either Big Ten or ACC."
    )
    await evaluator.verify(
        claim=conf_claim,
        node=conf_leaf,
        additional_instruction=(
            "Allow variants and synonyms: 'Big Ten', 'Big Ten Conference', 'ACC', 'Atlantic Coast Conference'. "
            "Consider case-insensitive matching and minor formatting differences."
        ),
    )

    # Full Member (not affiliate-only) - verify via URLs
    full_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_full_member_not_affiliate",
        desc="University joined as a full member (not an affiliate member for specific sports only).",
        parent=elig_node,
        critical=True
    )
    full_claim = (
        f"{entry.university_name} joined the {entry.conference} as a full member (not affiliate-only)."
        if entry.university_name and entry.conference else "The university joined as a full member."
    )
    await evaluator.verify(
        claim=full_claim,
        node=full_leaf,
        sources=entry.reference_urls,
        additional_instruction=(
            "Use the provided URLs (conference announcements, institutional releases, or credible reports) to confirm full membership status."
        ),
    )

    # Start date in 2023 or 2024 (logic check)
    start_year_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_start_date_in_2023_or_2024",
        desc="Official membership start date occurred in 2023 or 2024.",
        parent=elig_node,
        critical=True
    )
    start_year_claim = (
        f"The official membership start date '{entry.membership_start_date}' occurred in 2023 or 2024."
        if entry.membership_start_date else "The official membership start date occurred in 2023 or 2024."
    )
    await evaluator.verify(
        claim=start_year_claim,
        node=start_year_leaf,
        additional_instruction=(
            "Focus on the year component of the date given. Accept reasonable formats (e.g., 'Aug 2, 2024', '2024-08-02')."
        ),
    )

    # State west of Mississippi (deterministic check)
    state_west_result = is_state_west_of_mississippi(entry.state)
    evaluator.add_custom_node(
        result=state_west_result,
        id=f"univ_{idx}_state_west_of_mississippi",
        desc="University is located in a state west of the Mississippi River.",
        parent=elig_node,
        critical=True
    )

    # Sponsors NCAA Division I FBS football - verify via URLs
    fbs_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_sponsors_ncaa_divi_fbs_football",
        desc="University sponsors NCAA Division I FBS football.",
        parent=elig_node,
        critical=True
    )
    fbs_claim = (
        f"{entry.university_name} sponsors NCAA Division I FBS football."
        if entry.university_name else "The university sponsors NCAA Division I FBS football."
    )
    await evaluator.verify(
        claim=fbs_claim,
        node=fbs_leaf,
        sources=entry.reference_urls,
        additional_instruction=(
            "Look for terms like 'FBS', 'Football Bowl Subdivision', or indications that the school's football program competes at the NCAA Division I FBS level."
        ),
    )

    return elig_node


async def make_documentation_node(evaluator: Evaluator, parent: VerificationNode, entry: UniversityEntry, idx: int) -> VerificationNode:
    doc_node = evaluator.add_parallel(
        id=f"univ_{idx}_documentation_requirements",
        desc="Provided URLs publicly document the required facts for this university.",
        parent=parent,
        critical=True
    )

    # URL supports conference affiliation
    conf_doc_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_url_documents_conference_affiliation",
        desc="At least one provided URL supports which conference the university joined (Big Ten or ACC).",
        parent=doc_node,
        critical=True
    )
    conf_doc_claim = (
        f"{entry.university_name} joined the {entry.conference}."
        if entry.university_name and entry.conference else "The university joined the specified conference."
    )
    await evaluator.verify(
        claim=conf_doc_claim,
        node=conf_doc_leaf,
        sources=entry.reference_urls,
        additional_instruction=(
            "Verify that at least one provided URL explicitly states the conference affiliation (Big Ten or ACC). "
            "Conference office announcements or official institutional communications are preferred."
        ),
    )

    # URL supports full membership status
    full_doc_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_url_documents_full_membership_status",
        desc="At least one provided URL supports that the university joined as a full member (not affiliate-only).",
        parent=doc_node,
        critical=True
    )
    full_doc_claim = (
        f"{entry.university_name} joined as a full member of the {entry.conference}."
        if entry.university_name and entry.conference else "The university joined as a full member."
    )
    await evaluator.verify(
        claim=full_doc_claim,
        node=full_doc_leaf,
        sources=entry.reference_urls,
        additional_instruction=(
            "Confirm that the membership status is full conference membership, not a sport-specific affiliate arrangement."
        ),
    )

    # URL supports official membership start date
    start_doc_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_url_documents_membership_start_date",
        desc="At least one provided URL supports the official membership start date.",
        parent=doc_node,
        critical=True
    )
    start_doc_claim = (
        f"The official membership start date for {entry.university_name} is {entry.membership_start_date}."
        if entry.university_name and entry.membership_start_date else "The official membership start date is as stated in the answer."
    )
    await evaluator.verify(
        claim=start_doc_claim,
        node=start_doc_leaf,
        sources=entry.reference_urls,
        additional_instruction=(
            "Verify that the provided URLs explicitly state the official membership start date."
        ),
    )

    # URL supports official announcement date (and it is cited)
    # If announcement_date is missing in the answer, mark as failed explicitly.
    if entry.announcement_date and entry.announcement_date.strip():
        announce_doc_leaf = evaluator.add_leaf(
            id=f"univ_{idx}_url_documents_official_announcement_date",
            desc="At least one provided URL supports the official announcement date (and it is cited).",
            parent=doc_node,
            critical=True
        )
        announce_doc_claim = (
            f"The official announcement date for {entry.university_name} joining the {entry.conference} is {entry.announcement_date}."
        )
        await evaluator.verify(
            claim=announce_doc_claim,
            node=announce_doc_leaf,
            sources=entry.reference_urls,
            additional_instruction=(
                "Confirm that at least one provided URL states the official announcement date and matches the date cited in the answer."
            ),
        )
    else:
        # Add a failed leaf when the announcement date is not cited in the answer
        evaluator.add_custom_node(
            result=False,
            id=f"univ_{idx}_url_documents_official_announcement_date",
            desc="At least one provided URL supports the official announcement date (and it is cited).",
            parent=doc_node,
            critical=True
        )

    # URL supports state location
    state_doc_leaf = evaluator.add_leaf(
        id=f"univ_{idx}_url_documents_state_location",
        desc="At least one provided URL supports the university's state location (or supports the location claim used for the west-of-Mississippi constraint).",
        parent=doc_node,
        critical=True
    )
    state_doc_claim = (
        f"{entry.university_name} is located in the state of {entry.state}."
        if entry.university_name and entry.state else "The university is located in the stated U.S. state."
    )
    await evaluator.verify(
        claim=state_doc_claim,
        node=state_doc_leaf,
        sources=entry.reference_urls,
        additional_instruction=(
            "Verify that at least one provided URL supports the university's state location."
        ),
    )

    return doc_node


async def verify_university_entry(
    evaluator: Evaluator,
    task_node: VerificationNode,
    entry: UniversityEntry,
    idx: int,
) -> VerificationNode:
    """
    Build the full verification subtree for one university entry and trigger all verifications.
    """
    uni_node = evaluator.add_parallel(
        id=f"University_Entry_{idx+1}",
        desc=f"University entry #{idx+1} (if provided) is internally complete and eligible; counts toward the 4 if it passes all critical checks in this node.",
        parent=task_node,
        critical=False
    )

    # Required fields
    make_required_fields_node(evaluator, uni_node, entry, idx+1)

    # Eligibility criteria
    await make_eligibility_node(evaluator, uni_node, entry, idx+1)

    # Documentation requirements
    await make_documentation_node(evaluator, uni_node, entry, idx+1)

    return uni_node


def count_qualifying_universities(evaluator: Evaluator, task_node: VerificationNode, extracted: List[UniversityEntry]) -> Tuple[int, List[int]]:
    """
    Count how many University_Entry_i nodes fully pass (i.e., aggregated score == 1.0),
    and ensure they are distinct by university_name (case-insensitive).
    """
    # Compute/upsert scores for the subtree to ensure statuses are finalized
    task_node.compute_score(mutate=True)

    qualified_indices: List[int] = []
    seen_names: set = set()

    for i in range(5):
        node_id = f"University_Entry_{i+1}"
        node = evaluator.find_node(node_id)
        if not node:
            continue
        node.compute_score(mutate=True)
        passed = (node.score == 1.0 and node.status == "passed")
        name = None
        if i < len(extracted) and extracted[i] and extracted[i].university_name:
            name = extracted[i].university_name.strip().lower()
        if passed and name and name not in seen_names:
            seen_names.add(name)
            qualified_indices.append(i+1)

    return len(qualified_indices), qualified_indices

# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the conference membership west-of-Mississippi 2023/2024 task.
    """
    # Initialize evaluator
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

    # Top-level task node (non-critical to allow partial credit; 'at least four' will be a critical gate under it)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Identify universities that joined the Big Ten or ACC as full members with official membership start dates in 2023 or 2024, located west of the Mississippi River, and provide required fields and supporting URLs.",
        parent=root,
        critical=False
    )

    # Extract universities
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Keep up to 5 entries, pad with empty if fewer
    entries: List[UniversityEntry] = list(extracted.universities[:5])
    while len(entries) < 5:
        entries.append(UniversityEntry())

    # Build verification subtrees for up to 5 entries
    uni_nodes: List[VerificationNode] = []
    for i, entry in enumerate(entries):
        node = await verify_university_entry(evaluator, task_node, entry, i)
        uni_nodes.append(node)

    # Compute counts of qualifying universities and add the critical gate node
    # Ensure subtree scores are finalized before counting
    task_node.compute_score(mutate=True)
    qualifying_count, qualifying_indices = count_qualifying_universities(evaluator, task_node, entries)

    evaluator.add_custom_node(
        result=qualifying_count >= 4,
        id="At_Least_Four_Qualifying_Universities",
        desc="The response contains at least four DISTINCT universities that each satisfy all eligibility constraints and documentation requirements (i.e., they are qualifying universities).",
        parent=task_node,
        critical=True
    )

    # Add helpful debug info
    evaluator.add_custom_info(
        info={
            "qualified_count": qualifying_count,
            "qualified_indices": qualifying_indices,
            "west_of_miss_states_abbr": sorted(list(WEST_OF_MISS_STATES_ABBR))
        },
        info_type="custom",
        info_name="qualification_summary"
    )

    # Return evaluation summary
    return evaluator.get_summary()