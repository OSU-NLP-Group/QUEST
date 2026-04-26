import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "big_ten_oc_candidates"
TASK_DESCRIPTION = (
    "Identify three qualified candidates who could be considered for a Big Ten offensive coordinator position "
    "based on the following career requirements: Each candidate must (1) currently hold a coaching position at an FBS "
    "collegiate football program as of the 2025 or 2026 season, (2) have at least 5 years of total collegiate coaching "
    "experience, (3) have experience coaching at least one offensive position group (quarterbacks, offensive line, "
    "wide receivers, tight ends, or running backs) for a minimum of 2 seasons, (4) have prior experience as an offensive "
    "coordinator at any collegiate level (FBS, FCS, Division II, or Division III) for at least 1 season, (5) have coached "
    "at a minimum of 2 different collegiate institutions during their career, and (6) have experience at a Power 5/Power 4 "
    "conference school (ACC, Big Ten, Big 12, Pac-12, or SEC) for at least 1 season in any coaching role. For each candidate, "
    "provide their full name, current position title, current institution, and supporting evidence including URL references "
    "that verify their qualifications."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OCExperience(BaseModel):
    institution: Optional[str] = None
    level: Optional[str] = None  # e.g., FBS, FCS, Division II, Division III
    seasons: List[str] = Field(default_factory=list)  # e.g., ["2019", "2020"]
    role_title: Optional[str] = None  # e.g., "Offensive Coordinator", "Co-Offensive Coordinator"


class CandidateIdentity(BaseModel):
    name: Optional[str] = None
    current_title: Optional[str] = None
    current_institution: Optional[str] = None
    current_season: Optional[str] = None  # e.g., "2025", "2026", or text indicating active season
    identity_urls: List[str] = Field(default_factory=list)  # URLs verifying current role and institution


class CandidateExperience(BaseModel):
    total_years_college_coaching: Optional[str] = None  # e.g., "7", "8+ years", etc.
    offensive_position_groups: List[str] = Field(default_factory=list)  # e.g., ["QB", "WR"]
    offensive_position_seasons: Optional[str] = None  # e.g., "Coached QB for 2019-2020; WR for 2021-2022"
    oc_experience: List[OCExperience] = Field(default_factory=list)
    experience_urls: List[str] = Field(default_factory=list)  # URLs documenting experience timeline and roles


class CandidateHistory(BaseModel):
    institutions: List[str] = Field(default_factory=list)  # all collegiate institutions the candidate coached at
    power_conference_school: Optional[str] = None  # e.g., "Ohio State" (Big Ten)
    power_conference_seasons: List[str] = Field(default_factory=list)  # e.g., ["2021", "2022 season"]
    position_groups_identified: List[str] = Field(default_factory=list)  # redundancy to ensure identification
    oc_documentation: List[str] = Field(default_factory=list)  # textual documentation snippets (from answer)
    history_urls: List[str] = Field(default_factory=list)  # URLs supporting career history details


class CandidateRecord(BaseModel):
    identity: Optional[CandidateIdentity] = None
    experience: Optional[CandidateExperience] = None
    history: Optional[CandidateHistory] = None


class CandidatesExtraction(BaseModel):
    candidates: List[CandidateRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_candidates() -> str:
    return """
    Extract up to five candidate records presented in the answer for Big Ten offensive coordinator consideration.
    For each candidate, extract the following structured fields:

    identity:
      - name: Full name of the candidate.
      - current_title: Current coaching position title at their institution.
      - current_institution: Name of current institution (school).
      - current_season: The active season year mentioned (aim for "2025" or "2026" if specified).
      - identity_urls: A list of URLs that verify the candidate’s current position and institution (staff bio pages, official releases, etc.).

    experience:
      - total_years_college_coaching: Total collegiate coaching experience in years (string; do not convert to number).
      - offensive_position_groups: A list of offensive position groups coached (choose among QB, OL, WR, TE, RB; use these abbreviations).
      - offensive_position_seasons: Text summary of seasons/years coaching those offensive positions (keep as free text).
      - oc_experience: A list where each item includes:
          * institution: the school where the candidate served as an offensive coordinator (or co-offensive coordinator).
          * level: one of FBS, FCS, Division II, Division III (if available).
          * seasons: list of season years (e.g., ["2019","2020"]).
          * role_title: e.g., "Offensive Coordinator" or "Co-Offensive Coordinator".
      - experience_urls: A list of URLs documenting the candidate’s experience timeline and roles.

    history:
      - institutions: A list of collegiate institutions the candidate has coached at (include current and past).
      - power_conference_school: One school (if any) that belongs to ACC, Big Ten, Big 12, Pac-12, or SEC where the candidate coached.
      - power_conference_seasons: A list of seasons/years the candidate coached at that power-conference school.
      - position_groups_identified: Offensive position groups coached (redundant for identification).
      - oc_documentation: Text snippets or brief notes indicating OC/Co-OC experience with institution and seasons (free text).
      - history_urls: A list of URLs that support the candidate’s career history.

    Rules:
    - Only extract information explicitly present in the answer. Do not invent or infer missing details.
    - For all URL fields, include only valid URLs explicitly present in the answer. If a URL is missing a protocol, prepend "http://".
    - If any field is missing, set it to null (for strings) or an empty list (for lists).
    - Return a JSON object with a single field "candidates" which is an array of candidate records as defined.

    We only need the first three qualified candidates for evaluation, but extract all mentioned up to five so that we can select the first three later.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def sanitize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u.strip() for u in urls if isinstance(u, str) and u.strip().startswith(("http://", "https://"))]


def gather_all_urls(candidate: CandidateRecord) -> List[str]:
    urls: List[str] = []
    if candidate.identity and candidate.identity.identity_urls:
        urls.extend(candidate.identity.identity_urls)
    if candidate.experience and candidate.experience.experience_urls:
        urls.extend(candidate.experience.experience_urls)
    if candidate.history and candidate.history.history_urls:
        urls.extend(candidate.history.history_urls)
    # Deduplicate while preserving order
    seen = set()
    merged: List[str] = []
    for u in sanitize_urls(urls):
        if u not in seen:
            seen.add(u)
            merged.append(u)
    return merged


def list_to_str(items: Optional[List[str]]) -> str:
    return ", ".join(items) if items else ""


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_candidate(
    evaluator: Evaluator,
    parent_node,
    candidate: CandidateRecord,
    idx: int,
) -> None:
    """
    Build verification tree and run checks for a single candidate.
    idx is 0-based; for node IDs we will use 1-based indexing to match natural naming.
    """
    cid = idx + 1
    label = f"candidate_{cid}"

    # Top-level candidate node (non-critical to allow partial credit across candidates)
    cand_node = evaluator.add_parallel(
        id=label,
        desc=f"{['First','Second','Third','Fourth','Fifth'][idx]} qualified candidate meeting all requirements",
        parent=parent_node,
        critical=False,
    )

    # Qualifications umbrella (critical)
    qual_node = evaluator.add_parallel(
        id=f"{label}_qualifications",
        desc="Complete verification of candidate's qualifications and background",
        parent=cand_node,
        critical=True,
    )

    # -------------------- Identity --------------------
    identity_node = evaluator.add_parallel(
        id=f"{label}_identity",
        desc="Candidate identity and current position",
        parent=qual_node,
        critical=True,
    )

    name = candidate.identity.name if candidate.identity else None
    title = candidate.identity.current_title if candidate.identity else None
    institution = candidate.identity.current_institution if candidate.identity else None
    season = candidate.identity.current_season if candidate.identity else None
    identity_urls = sanitize_urls(candidate.identity.identity_urls if candidate.identity else [])

    # Existence: name provided
    evaluator.add_custom_node(
        result=bool(name and name.strip()),
        id=f"{label}_name",
        desc="Full name of the candidate is provided",
        parent=identity_node,
        critical=True,
    )

    # Existence: current title specified
    evaluator.add_custom_node(
        result=bool(title and title.strip()),
        id=f"{label}_current_title",
        desc="Current position title is specified",
        parent=identity_node,
        critical=True,
    )

    # Existence: current institution specified
    inst_exists_node = evaluator.add_custom_node(
        result=bool(institution and institution.strip()),
        id=f"{label}_current_institution_exists",
        desc="Current institution is specified",
        parent=identity_node,
        critical=True,
    )

    # Verification: Institution is an FBS program (with URLs)
    institution_fbs_node = evaluator.add_leaf(
        id=f"{label}_current_institution",
        desc="Current institution is specified and is an FBS program",
        parent=identity_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The institution {institution or ''} competes in NCAA Division I FBS football.",
        node=institution_fbs_node,
        sources=gather_all_urls(candidate),
        additional_instruction=(
            "Verify that the named institution is an NCAA Division I FBS program. Membership in ACC, Big Ten, Big 12, "
            "Pac-12, or SEC implies FBS. If the provided sources are insufficient or irrelevant, mark as not supported."
        ),
    )

    # Verification: Current employment is active in 2025 or 2026 season (with URLs)
    current_season_node = evaluator.add_leaf(
        id=f"{label}_current_season",
        desc="Current employment is verified as active in 2025 or 2026 season",
        parent=identity_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"As of the 2025 or 2026 college football season, {name or ''} holds the position '{title or ''}' at {institution or ''}."
        ),
        node=current_season_node,
        sources=identity_urls or gather_all_urls(candidate),
        additional_instruction=(
            "Use the provided URLs (e.g., staff bios, official school pages, or roster/staff announcements) to verify the "
            "candidate's current employment during the 2025 or 2026 season. If the page clearly indicates the staff for 2025 or "
            "2026, consider it supported; otherwise, not supported."
        ),
    )

    # Existence: Identity reference URL provided
    evaluator.add_custom_node(
        result=bool(identity_urls),
        id=f"{label}_identity_reference",
        desc="URL reference provided to verify candidate's current position and institution",
        parent=identity_node,
        critical=True,
    )

    # -------------------- Experience Requirements --------------------
    exp_node = evaluator.add_parallel(
        id=f"{label}_experience_requirements",
        desc="Candidate meets minimum experience requirements",
        parent=qual_node,
        critical=True,
    )

    total_years = candidate.experience.total_years_college_coaching if candidate.experience else None
    pos_groups = candidate.experience.offensive_position_groups if candidate.experience else []
    pos_seasons_text = candidate.experience.offensive_position_seasons if candidate.experience else None
    oc_exp = candidate.experience.oc_experience if candidate.experience else []
    exp_urls = sanitize_urls(candidate.experience.experience_urls if candidate.experience else [])

    # Verification: At least 5 years collegiate coaching
    total_exp_node = evaluator.add_leaf(
        id=f"{label}_total_experience",
        desc="Candidate has at least 5 years of total collegiate coaching experience",
        parent=exp_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name or 'The candidate'} has at least five years of collegiate coaching experience.",
        node=total_exp_node,
        sources=exp_urls or gather_all_urls(candidate),
        additional_instruction=(
            "Evaluate career timeline across provided sources. Consider seasons or years explicitly shown. If total years are "
            "clearly >= 5, mark supported; otherwise not supported."
        ),
    )

    # Verification: Offensive position group coaching for >= 2 seasons
    position_coaching_node = evaluator.add_leaf(
        id=f"{label}_position_coaching",
        desc="Candidate has coached at least one offensive position group (QB, OL, WR, TE, or RB) for a minimum of 2 seasons",
        parent=exp_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"{name or 'The candidate'} has coached at least one offensive position group (QB/OL/WR/TE/RB) for two or more seasons. "
            f"Groups noted: {list_to_str(pos_groups)}; details: {pos_seasons_text or ''}"
        ),
        node=position_coaching_node,
        sources=exp_urls or gather_all_urls(candidate),
        additional_instruction=(
            "Confirm any offensive position group coaching totaling >= 2 seasons. Aggregating seasons across different offensive "
            "position groups is acceptable. If insufficient evidence, mark as not supported."
        ),
    )

    # Verification: Prior offensive coordinator experience (>= 1 season) at any collegiate level
    coord_exp_node = evaluator.add_leaf(
        id=f"{label}_coordinator_experience",
        desc="Candidate has prior offensive coordinator experience at any collegiate level for at least 1 season",
        parent=exp_node,
        critical=True,
    )
    oc_descriptions = "; ".join(
        [
            f"{oe.role_title or 'OC'} at {oe.institution or ''} ({list_to_str(oe.seasons)}) [{oe.level or 'college'}]"
            for oe in (oc_exp or [])
        ]
    )
    await evaluator.verify(
        claim=(
            f"{name or 'The candidate'} has at least one season of offensive coordinator (or co-offensive coordinator) experience "
            f"at a collegiate level. Details: {oc_descriptions}"
        ),
        node=coord_exp_node,
        sources=exp_urls or gather_all_urls(candidate),
        additional_instruction=(
            "Only count offensive coordinator or co-offensive coordinator roles at collegiate levels (FBS, FCS, DII, DIII). "
            "Analyst or non-OC coordinator (e.g., passing game coordinator) alone does not satisfy this requirement."
        ),
    )

    # Existence: Experience reference URLs provided
    evaluator.add_custom_node(
        result=bool(exp_urls),
        id=f"{label}_experience_reference",
        desc="URL reference provided documenting the candidate's career history and experience levels",
        parent=exp_node,
        critical=True,
    )

    # -------------------- Career History --------------------
    hist_node = evaluator.add_parallel(
        id=f"{label}_career_history",
        desc="Candidate's career history demonstrates required breadth",
        parent=qual_node,
        critical=True,
    )

    institutions = candidate.history.institutions if candidate.history else []
    power_school = candidate.history.power_conference_school if candidate.history else None
    power_seasons = candidate.history.power_conference_seasons if candidate.history else []
    hist_pos_groups = candidate.history.position_groups_identified if candidate.history else []
    oc_docs = candidate.history.oc_documentation if candidate.history else []
    hist_urls = sanitize_urls(candidate.history.history_urls if candidate.history else [])

    # Verification: Coached at >= 2 different collegiate institutions
    multi_inst_node = evaluator.add_leaf(
        id=f"{label}_multiple_institutions",
        desc="Candidate has coached at a minimum of 2 different collegiate institutions",
        parent=hist_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name or 'The candidate'} has coached at two or more different collegiate institutions: {list_to_str(institutions)}.",
        node=multi_inst_node,
        sources=hist_urls or gather_all_urls(candidate),
        additional_instruction=(
            "Confirm at least two distinct collegiate institutions across the candidate's career. If unclear or only one institution "
            "is evidenced, mark as not supported."
        ),
    )

    # Verification: Experience at a Power 5/Power 4 conference school (>= 1 season)
    power_conf_node = evaluator.add_leaf(
        id=f"{label}_power_conference",
        desc="Candidate has experience at a Power 5/Power 4 conference school for at least 1 season",
        parent=hist_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"{name or 'The candidate'} has coaching experience at a Power 5/Power 4 conference school "
            f"(ACC, Big Ten, Big 12, Pac-12, SEC), e.g., {power_school or ''} during {list_to_str(power_seasons)}."
        ),
        node=power_conf_node,
        sources=hist_urls or gather_all_urls(candidate),
        additional_instruction=(
            "Verify the school belongs to ACC, Big Ten, Big 12, Pac-12, or SEC and that the candidate coached there for at least "
            "one season. If sources do not support this, mark as not supported."
        ),
    )

    # Existence: At least one previous institution identified
    evaluator.add_custom_node(
        result=bool(institutions),
        id=f"{label}_prior_institution_1",
        desc="At least one previous institution where candidate coached is identified",
        parent=hist_node,
        critical=True,
    )

    # Existence: Offensive position groups identified
    # Allow fallback to experience list if history list is empty
    combined_pos_groups = hist_pos_groups or pos_groups
    evaluator.add_custom_node(
        result=bool(combined_pos_groups),
        id=f"{label}_position_groups",
        desc="Specific offensive position group(s) coached by candidate are identified",
        parent=hist_node,
        critical=True,
    )

    # Verification: Prior OC experience documented with institution and season
    oc_doc_node = evaluator.add_leaf(
        id=f"{label}_oc_documentation",
        desc="Prior offensive coordinator experience is documented with institution and season",
        parent=hist_node,
        critical=True,
    )
    # Use either structured oc_experience or oc_documentation text
    if oc_exp:
        first_oc = oc_exp[0]
        oc_claim_detail = (
            f"{name or 'The candidate'} served as {first_oc.role_title or 'Offensive Coordinator'} at "
            f"{first_oc.institution or ''} during {list_to_str(first_oc.seasons)}."
        )
    else:
        oc_claim_detail = f"{name or 'The candidate'} has documented offensive coordinator experience (institution and season noted): {list_to_str(oc_docs)}."
    await evaluator.verify(
        claim=oc_claim_detail,
        node=oc_doc_node,
        sources=gather_all_urls(candidate),
        additional_instruction=(
            "Confirm the offensive coordinator (or co-offensive coordinator) role with at least one season, including the institution "
            "and season(s) as documented in the provided URLs."
        ),
    )

    # Existence: History reference URLs provided
    evaluator.add_custom_node(
        result=bool(hist_urls),
        id=f"{label}_history_reference",
        desc="URL reference provided supporting the candidate's career history details",
        parent=hist_node,
        critical=True,
    )


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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Big Ten offensive coordinator candidates task.
    """
    # Initialize evaluator (non-critical root to allow partial credit across candidates)
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

    # Extract candidate records
    extracted = await evaluator.extract(
        prompt=prompt_extract_candidates(),
        template_class=CandidatesExtraction,
        extraction_name="candidates_extraction",
    )

    # Keep only the first 3 candidates; pad with empty records if fewer
    candidates: List[CandidateRecord] = list(extracted.candidates[:3])
    while len(candidates) < 3:
        candidates.append(CandidateRecord())

    # Build verification subtrees for each candidate
    tasks = []
    for i, cand in enumerate(candidates):
        tasks.append(verify_candidate(evaluator, root, cand, i))
    await asyncio.gather(*tasks)

    # Return structured summary
    return evaluator.get_summary()