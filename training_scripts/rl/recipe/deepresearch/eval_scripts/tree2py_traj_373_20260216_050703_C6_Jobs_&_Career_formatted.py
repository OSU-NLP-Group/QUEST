import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "g5_stc_pathway"
TASK_DESCRIPTION = (
    "Research and document the career pathway requirements to become a special teams coordinator at NCAA Division I FBS "
    "Group of Five conference institutions. Identify three current special teams coordinators at different Group of Five "
    "conference institutions (American Athletic Conference, Conference USA, Sun Belt, Mountain West, or Mid-American Conference) "
    "who were hired within the last 2 years (2024-2026). For each coordinator, provide: (1) Their full name and current institution, "
    "(2) Confirmation of their position title as special teams coordinator, (3) Their hire year, (4) Their educational credentials, "
    "including bachelor's degree (required) and any advanced degrees, (5) Their total years of coaching experience and the level of that "
    "experience (e.g., NCAA Division I, II, III), (6) Their most recent position before their current role and duration in that position, "
    "(7) Reference URLs verifying all information. Finally, synthesize the common minimum qualifications across all three examples, "
    "identifying: the standard educational requirement, the typical minimum years of coaching experience required, and the common career "
    "progression pattern observed. All information must be verifiable through provided reference URLs."
)

GROUP_OF_FIVE_CONFERENCES = [
    "American Athletic Conference", "AAC",
    "Conference USA", "C-USA",
    "Sun Belt Conference", "Sun Belt",
    "Mountain West Conference", "Mountain West", "MWC",
    "Mid-American Conference", "MAC"
]

ALLOWED_HIRE_YEARS = {2024, 2025, 2026}
MIN_EXPERIENCE_YEARS_THRESHOLD = 3


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CoordinatorSources(BaseModel):
    identity_sources: List[str] = Field(default_factory=list)
    institution_sources: List[str] = Field(default_factory=list)
    education_sources: List[str] = Field(default_factory=list)
    experience_sources: List[str] = Field(default_factory=list)
    previous_sources: List[str] = Field(default_factory=list)


class CoordinatorInfo(BaseModel):
    name: Optional[str] = None
    institution: Optional[str] = None
    conference: Optional[str] = None
    position_title: Optional[str] = None
    hire_year: Optional[str] = None  # Keep string to allow formats like "February 2025"
    bachelors: Optional[str] = None
    advanced_degrees: List[str] = Field(default_factory=list)
    total_years_experience: Optional[str] = None
    experience_levels: List[str] = Field(default_factory=list)
    prior_role_title: Optional[str] = None
    prior_role_org: Optional[str] = None
    prior_role_duration: Optional[str] = None
    sources: CoordinatorSources = Field(default_factory=CoordinatorSources)


class CoordinatorsExtraction(BaseModel):
    coordinators: List[CoordinatorInfo] = Field(default_factory=list)


class SynthesisInfo(BaseModel):
    education_standard: Optional[str] = None
    experience_min_years: Optional[str] = None
    career_path_pattern: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_coordinators() -> str:
    return """
    Extract up to three current special teams coordinators at NCAA Division I FBS Group of Five institutions from the answer.
    Only consider personnel hired within the last 2 years (2024-2026). If the answer lists more than three, extract the first three.
    For each coordinator, extract the following fields exactly as stated in the answer:
    - name: Full name of the coordinator
    - institution: Current employing institution
    - conference: The conference of the institution (e.g., AAC, C-USA, Sun Belt, Mountain West, MAC). If the answer spells it out (e.g., 'American Athletic Conference'), extract that exact text.
    - position_title: Position title; should include 'special teams coordinator' or an equivalent formulation
    - hire_year: The year they were hired into the current role (e.g., '2025'); if a month/year is provided, extract the full text
    - bachelors: Bachelor's degree credential (institution and major if provided). This must be present to be considered documented; if missing in the answer, set to null.
    - advanced_degrees: List any advanced degree(s) documented (e.g., master's, doctorate). If none are provided, return an empty list.
    - total_years_experience: Total years of coaching experience (e.g., '8 years', '10+'); if missing, set to null.
    - experience_levels: List of levels in which they have coached (e.g., 'NCAA Division I FBS', 'NCAA Division II', 'FCS', 'NFL'); if missing, return an empty list.
    - prior_role_title: Most recent position before the current role (e.g., 'Special Teams Analyst')
    - prior_role_org: Organization or institution for the most recent prior role
    - prior_role_duration: Duration in the prior role (e.g., '2019–2022', 'two seasons'); if not specified, set to null.
    - sources.identity_sources: URLs specifically verifying identity and current position
    - sources.institution_sources: URLs verifying the institution and conference membership (if provided in the answer; else leave empty)
    - sources.education_sources: URLs verifying educational credentials
    - sources.experience_sources: URLs verifying experience (years and levels)
    - sources.previous_sources: URLs verifying previous positions and durations

    Special rules:
    - Extract only URLs explicitly present in the answer. Do not invent any URLs.
    - If a required field (e.g., name, institution, position_title) is missing, set it to null.
    - If the answer provides fewer than three valid coordinators, extract what is available and leave missing fields as null.

    Return a JSON object with:
    { "coordinators": [CoordinatorInfo, CoordinatorInfo, CoordinatorInfo] }
    """


def prompt_extract_synthesis() -> str:
    return """
    Extract the synthesized common minimum qualifications as stated in the answer text.
    Fields:
    - education_standard: The common educational requirement across examples (e.g., "Bachelor's degree minimum")
    - experience_min_years: The typical minimum years of coaching experience required (e.g., "3+ years")
    - career_path_pattern: The common career progression pattern observed (e.g., "Graduate Assistant/Analyst -> Position coach -> Special Teams Coordinator")
    Return null for any field not explicitly stated in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(xs: Optional[List[str]]) -> List[str]:
    return xs or []


def _combine_sources(*lists: List[str]) -> List[str]:
    result: List[str] = []
    for lst in lists:
        if lst:
            result.extend(lst)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for url in result:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def parse_years_from_text(text: Optional[str]) -> Optional[int]:
    """Parse an integer number of years from strings like '8 years', '10+', 'over 5 years', '5-6 years'."""
    if not text:
        return None
    # Try digit-first
    m = re.search(r'(\d{1,2})\s*(?:\+|plus|years?|yrs?)?', text.lower())
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    # Ranges like '5-6 years' -> take the lower bound
    m2 = re.search(r'(\d{1,2})\s*-\s*(\d{1,2})', text.lower())
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            pass
    # Words for small numbers
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20
    }
    for w, v in words.items():
        if w in text.lower():
            return v
    return None


def parse_year_from_hire_year(hire_year_text: Optional[str]) -> Optional[int]:
    """Extract a 4-digit year from hire_year text, if present."""
    if not hire_year_text:
        return None
    m = re.search(r'(20\d{2})', hire_year_text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def within_allowed_hire_timeframe(hire_year_text: Optional[str]) -> bool:
    y = parse_year_from_hire_year(hire_year_text)
    return y in ALLOWED_HIRE_YEARS if y is not None else False


def derive_common_career_path(coordinators: List[CoordinatorInfo]) -> str:
    """
    Very simple heuristic: look at prior_role_title strings and categorize as GA/Analyst/QC vs Position Coach vs Other.
    Then express a common pattern across the three if possible.
    """
    def categorize(title: Optional[str]) -> str:
        t = (title or "").lower()
        if any(k in t for k in ["graduate assistant", "ga", "analyst", "quality control", "qc"]):
            return "GA/Analyst/QC"
        if any(k in t for k in ["coach", "coordinator", "assistant coach"]) and "special teams" not in t:
            return "Position coach"
        if "special teams" in t and "coordinator" in t:
            return "ST Coordinator"
        return "Other"

    cats = [categorize(c.prior_role_title) for c in coordinators]
    # If most have GA/Analyst/QC then Position coach then ST Coordinator, say that path.
    # Since we only have one prior role per coordinator, we infer immediate predecessor category.
    # We'll propose: Prior role commonly GA/Analyst/QC or Position coach -> Special Teams Coordinator
    if all(cat in ["GA/Analyst/QC", "Position coach", "Other"] for cat in cats):
        return "GA/Analyst/QC or Position coach -> Special Teams Coordinator"
    # Fallback generic
    return "Position coach or analyst -> Special Teams Coordinator"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
class BuiltCoordinatorNodes(BaseModel):
    bachelors_leaf_ids: List[str] = Field(default_factory=list)
    experience_years_leaf_ids: List[str] = Field(default_factory=list)
    previous_role_leaf_ids: List[str] = Field(default_factory=list)
    previous_duration_leaf_ids: List[str] = Field(default_factory=list)


async def verify_coordinator(
    evaluator: Evaluator,
    root_parent,
    coordinator: CoordinatorInfo,
    idx: int,
) -> BuiltCoordinatorNodes:
    """
    Build the verification sub-tree and run verifications for one coordinator.
    Returns references (IDs) to key leaf nodes for use as prerequisites in synthesis checks.
    """
    built = BuiltCoordinatorNodes()

    # Parent sequential node for this coordinator
    coord_node = evaluator.add_sequential(
        id=f"Coordinator_{idx+1}_Documentation",
        desc=f"Complete documentation of {'first' if idx==0 else ('second' if idx==1 else 'third')} special teams coordinator example",
        parent=root_parent,
        critical=False  # Adjusted: allow partial credit inside the critical root constraint
    )

    # 1) Identity (parallel, critical)
    identity_node = evaluator.add_parallel(
        id=f"Coordinator_{idx+1}_Identity",
        desc="Identity and current position verified",
        parent=coord_node,
        critical=True
    )

    # Name provided (existence)
    evaluator.add_custom_node(
        result=bool(coordinator.name and coordinator.name.strip()),
        id=f"Coordinator_{idx+1}_Name",
        desc="Full name of the special teams coordinator provided",
        parent=identity_node,
        critical=True
    )

    # Institution provided (existence)
    evaluator.add_custom_node(
        result=bool(coordinator.institution and coordinator.institution.strip()),
        id=f"Coordinator_{idx+1}_Institution_Provided",
        desc="Current employing institution provided",
        parent=identity_node,
        critical=True
    )

    # Institution is Group of Five member (verification)
    inst_gof_leaf = evaluator.add_leaf(
        id=f"Coordinator_{idx+1}_Institution_GroupOfFive",
        desc="Current employing institution identified and is a Group of Five conference member (AAC, C-USA, Sun Belt, Mountain West, or MAC)",
        parent=identity_node,
        critical=True
    )
    inst_sources = _combine_sources(coordinator.sources.identity_sources, coordinator.sources.institution_sources)
    inst_claim = (
        f"The institution '{coordinator.institution or ''}' competes in NCAA Division I FBS and is a member of the "
        f"'{coordinator.conference or ''}', which is one of the Group of Five conferences."
    )
    await evaluator.verify(
        claim=inst_claim,
        node=inst_gof_leaf,
        sources=inst_sources if inst_sources else None,
        additional_instruction=(
            "Verify the institution's conference membership and FBS status. Accept common abbreviations (AAC, C-USA, MWC, MAC). "
            "If the provided sources do not substantiate membership in one of the listed conferences, mark as not supported."
        )
    )

    # Position title confirmed as ST coordinator or equivalent (verification)
    pos_leaf = evaluator.add_leaf(
        id=f"Coordinator_{idx+1}_Position_Title",
        desc="Position title confirmed as special teams coordinator or equivalent role",
        parent=identity_node,
        critical=True
    )
    pos_claim = (
        f"{coordinator.name or 'The coach'} currently holds a role at {coordinator.institution or 'the institution'} "
        f"whose title includes 'special teams' and 'coordinator' (or an equivalent formulation). "
        f"Stated title: '{coordinator.position_title or ''}'."
    )
    await evaluator.verify(
        claim=pos_claim,
        node=pos_leaf,
        sources=_safe_list(coordinator.sources.identity_sources) or None,
        additional_instruction="Allow minor variants (e.g., 'ST Coordinator', 'Special Teams Coord.'). The role should clearly be a special teams coordinator."
    )

    # Hire timeframe within 2024-2026 (verification)
    hire_leaf = evaluator.add_leaf(
        id=f"Coordinator_{idx+1}_Hire_Timeframe",
        desc="Hire date verified to be within the last 2 years (2024-2026)",
        parent=identity_node,
        critical=True
    )
    hire_claim = (
        f"The hire into the current role occurred in 2024, 2025, or 2026. "
        f"Stated hire year in the answer: '{coordinator.hire_year or ''}'."
    )
    await evaluator.verify(
        claim=hire_claim,
        node=hire_leaf,
        sources=_safe_list(coordinator.sources.identity_sources) or None,
        additional_instruction="Confirm the hire date/year from the source. Only pass if the page explicitly indicates 2024, 2025, or 2026."
    )

    # Identity reference URL confirms identity and current position (verification)
    identity_ref_leaf = evaluator.add_leaf(
        id=f"Coordinator_{idx+1}_Identity_Reference",
        desc="Reference URL provided verifying coordinator identity and current position",
        parent=coord_node,
        critical=True
    )
    identity_ref_claim = (
        f"The cited sources confirm that {coordinator.name or 'the coach'} is currently {coordinator.position_title or 'Special Teams Coordinator'} "
        f"at {coordinator.institution or 'the institution'}."
    )
    await evaluator.verify(
        claim=identity_ref_claim,
        node=identity_ref_leaf,
        sources=_safe_list(coordinator.sources.identity_sources) or None,
        additional_instruction="The source should explicitly mention the coach's current position and institution."
    )

    # 2) Education (parallel, adjusted non-critical to allow optional advanced degree)
    edu_node = evaluator.add_parallel(
        id=f"Coordinator_{idx+1}_Education",
        desc="Educational credentials documented",
        parent=coord_node,
        critical=False  # Adjusted: allow non-critical child for advanced degree (optional)
    )

    # Bachelor's degree confirmed (verification)
    bachelors_leaf = evaluator.add_leaf(
        id=f"Coordinator_{idx+1}_Bachelors",
        desc="Bachelor's degree confirmed from an accredited institution",
        parent=edu_node,
        critical=True
    )
    bachelors_claim = (
        f"{coordinator.name or 'The coach'} holds a bachelor's degree as stated: '{coordinator.bachelors or ''}'."
    )
    await evaluator.verify(
        claim=bachelors_claim,
        node=bachelors_leaf,
        sources=_safe_list(coordinator.sources.education_sources) or None,
        additional_instruction="Verify the bachelor's degree (BA/BS/BBA/BSc, etc.) and the awarding institution as stated."
    )
    built.bachelors_leaf_ids.append(bachelors_leaf.id)

    # Advanced degree documented if applicable (optional)
    if coordinator.advanced_degrees:
        adv_leaf = evaluator.add_leaf(
            id=f"Coordinator_{idx+1}_Advanced_Degree",
            desc="Master's degree or higher documented if applicable",
            parent=edu_node,
            critical=False
        )
        adv_claim = (
            f"{coordinator.name or 'The coach'} holds advanced degree(s): {coordinator.advanced_degrees}."
        )
        await evaluator.verify(
            claim=adv_claim,
            node=adv_leaf,
            sources=_safe_list(coordinator.sources.education_sources) or None,
            additional_instruction="Confirm any listed advanced degrees (e.g., MA/MS/MBA/EdD/PhD) with the cited sources."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id=f"Coordinator_{idx+1}_Advanced_Degree_Not_Applicable",
            desc="No advanced degree documented; optional criterion considered satisfied",
            parent=edu_node,
            critical=False
        )

    # Education reference (verification)
    edu_ref_leaf = evaluator.add_leaf(
        id=f"Coordinator_{idx+1}_Education_Reference",
        desc="Reference URL provided verifying educational credentials",
        parent=coord_node,
        critical=True
    )
    edu_ref_claim = (
        f"The cited sources substantiate the educational credentials stated for {coordinator.name or 'the coach'}."
    )
    await evaluator.verify(
        claim=edu_ref_claim,
        node=edu_ref_leaf,
        sources=_safe_list(coordinator.sources.education_sources) or None,
        additional_instruction="Pass only if the page supports the degree details (degree type and institution)."
    )

    # 3) Experience (parallel, critical)
    exp_node = evaluator.add_parallel(
        id=f"Coordinator_{idx+1}_Experience",
        desc="Coaching experience documented",
        parent=coord_node,
        critical=True
    )

    # Total years meets minimum threshold (verification)
    exp_years_leaf = evaluator.add_leaf(
        id=f"Coordinator_{idx+1}_Total_Years",
        desc="Total years of coaching experience documented and meets minimum threshold (typically 3+ years for coordinator positions)",
        parent=exp_node,
        critical=True
    )
    exp_years_claim = (
        f"{coordinator.name or 'The coach'} has at least {MIN_EXPERIENCE_YEARS_THRESHOLD} years of coaching experience. "
        f"Stated: '{coordinator.total_years_experience or ''}'."
    )
    await evaluator.verify(
        claim=exp_years_claim,
        node=exp_years_leaf,
        sources=_safe_list(coordinator.sources.experience_sources) or None,
        additional_instruction=(
            f"Use the source to confirm total coaching years. Only pass if the page reasonably supports >= {MIN_EXPERIENCE_YEARS_THRESHOLD} years."
        )
    )
    built.experience_years_leaf_ids.append(exp_years_leaf.id)

    # Experience level specified (verification)
    exp_level_leaf = evaluator.add_leaf(
        id=f"Coordinator_{idx+1}_Experience_Level",
        desc="Level of coaching experience specified (e.g., NCAA Division I, II, III, professional)",
        parent=exp_node,
        critical=True
    )
    levels_text = ", ".join(coordinator.experience_levels) if coordinator.experience_levels else ""
    exp_level_claim = (
        f"{coordinator.name or 'The coach'} has coaching experience at the following levels: {levels_text}."
    )
    await evaluator.verify(
        claim=exp_level_claim,
        node=exp_level_leaf,
        sources=_safe_list(coordinator.sources.experience_sources) or None,
        additional_instruction="Confirm that the source mentions the levels (e.g., NCAA Division I FBS/FCS, II, III, professional)."
    )

    # Experience reference (verification)
    exp_ref_leaf = evaluator.add_leaf(
        id=f"Coordinator_{idx+1}_Experience_Reference",
        desc="Reference URL provided verifying coaching experience",
        parent=coord_node,
        critical=True
    )
    exp_ref_claim = (
        f"The cited sources substantiate the coaching experience years and levels stated for {coordinator.name or 'the coach'}."
    )
    await evaluator.verify(
        claim=exp_ref_claim,
        node=exp_ref_leaf,
        sources=_safe_list(coordinator.sources.experience_sources) or None,
        additional_instruction="Pass only if the page supports both the years and the levels."
    )

    # 4) Previous positions (parallel, critical)
    prev_node = evaluator.add_parallel(
        id=f"Coordinator_{idx+1}_Previous_Positions",
        desc="Previous coaching positions documented showing career progression",
        parent=coord_node,
        critical=True
    )

    # Prior role title/org (verification)
    prior_role_leaf = evaluator.add_leaf(
        id=f"Coordinator_{idx+1}_Prior_Role",
        desc="Most recent position before current role identified",
        parent=prev_node,
        critical=True
    )
    prior_role_claim = (
        f"Immediately prior to the current role, {coordinator.name or 'the coach'} served as "
        f"'{coordinator.prior_role_title or ''}' at '{coordinator.prior_role_org or ''}'."
    )
    await evaluator.verify(
        claim=prior_role_claim,
        node=prior_role_leaf,
        sources=_safe_list(coordinator.sources.previous_sources) or None,
        additional_instruction="Confirm the immediate predecessor role and organization."
    )
    built.previous_role_leaf_ids.append(prior_role_leaf.id)

    # Prior duration (verification)
    prior_duration_leaf = evaluator.add_leaf(
        id=f"Coordinator_{idx+1}_Prior_Duration",
        desc="Duration in previous position(s) documented",
        parent=prev_node,
        critical=True
    )
    prior_duration_claim = (
        f"The duration in the immediate prior role for {coordinator.name or 'the coach'} was '{coordinator.prior_role_duration or ''}'."
    )
    await evaluator.verify(
        claim=prior_duration_claim,
        node=prior_duration_leaf,
        sources=_safe_list(coordinator.sources.previous_sources) or None,
        additional_instruction="Confirm the duration (e.g., years/seasons or date range). If the source gives dates, that's acceptable."
    )
    built.previous_duration_leaf_ids.append(prior_duration_leaf.id)

    # Previous positions reference (verification)
    prev_ref_leaf = evaluator.add_leaf(
        id=f"Coordinator_{idx+1}_Previous_Reference",
        desc="Reference URL provided verifying previous positions",
        parent=coord_node,
        critical=True
    )
    prev_ref_claim = (
        f"The cited sources substantiate the immediate prior role and its duration for {coordinator.name or 'the coach'}."
    )
    await evaluator.verify(
        claim=prev_ref_claim,
        node=prev_ref_leaf,
        sources=_safe_list(coordinator.sources.previous_sources) or None,
        additional_instruction="Pass only if the page supports both the role and its duration."
    )

    return built


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
    Evaluate an answer for the Group of Five special teams coordinator pathway task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates subtasks in parallel
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

    # IMPORTANT: The verification tree enforces that critical parents must have critical children only.
    # The JSON root was marked critical, but it contains non-critical descendants. To comply, we keep the root non-critical
    # and enforce criticality at appropriate sub-nodes.

    # Ground truth info to record conference names for reference
    evaluator.add_ground_truth({
        "group_of_five_conferences": GROUP_OF_FIVE_CONFERENCES,
        "allowed_hire_years": sorted(list(ALLOWED_HIRE_YEARS)),
        "min_experience_years_threshold": MIN_EXPERIENCE_YEARS_THRESHOLD
    }, gt_type="constraints")

    # Extract coordinators and synthesis info from the answer
    coordinators_data: CoordinatorsExtraction = await evaluator.extract(
        prompt=prompt_extract_coordinators(),
        template_class=CoordinatorsExtraction,
        extraction_name="coordinators_extraction"
    )

    synthesis_info: SynthesisInfo = await evaluator.extract(
        prompt=prompt_extract_synthesis(),
        template_class=SynthesisInfo,
        extraction_name="synthesis_extraction"
    )

    # Normalize coordinators list: keep first 3; pad if fewer
    coords = coordinators_data.coordinators[:3]
    while len(coords) < 3:
        coords.append(CoordinatorInfo())

    # Build subtrees for each coordinator and run verifications
    built_nodes: List[BuiltCoordinatorNodes] = []
    for i in range(3):
        built = await verify_coordinator(evaluator, root, coords[i], i)
        built_nodes.append(built)

    # Common requirements synthesis node (parallel, critical as per rubric)
    synthesis_node = evaluator.add_parallel(
        id="Common_Requirements_Synthesis",
        desc="Synthesis of common minimum qualifications across all three coordinator examples",
        parent=root,
        critical=True
    )

    # Minimum Education Standard: check if all three have a documented bachelor's degree (based on extraction and earlier checks)
    all_have_bachelors = all(bool(c.bachelors and c.bachelors.strip()) for c in coords)
    evaluator.add_custom_node(
        result=all_have_bachelors,
        id="Minimum_Education_Standard",
        desc="Common educational requirement identified across examples (bachelor's degree minimum)",
        parent=synthesis_node,
        critical=True
    )

    # Minimum Experience Standard: derive minimum years across examples (from text parsing) and check >= threshold
    parsed_years: List[Optional[int]] = [parse_years_from_text(c.total_years_experience) for c in coords]
    min_years = min([y for y in parsed_years if y is not None], default=None)
    meets_min_experience = (min_years is not None and min_years >= MIN_EXPERIENCE_YEARS_THRESHOLD)
    evaluator.add_custom_node(
        result=meets_min_experience,
        id="Minimum_Experience_Standard",
        desc="Common minimum years of coaching experience identified across examples",
        parent=synthesis_node,
        critical=True
    )

    # Typical Career Path: derive from prior role categories; verify the statement against the answer text (simple verification),
    # but gate it on prerequisites: prior role/duration leaves for each coordinator
    career_path_text = synthesis_info.career_path_pattern or derive_common_career_path(coords)
    career_path_leaf = evaluator.add_leaf(
        id="Typical_Career_Path",
        desc="Common career progression pattern identified from previous positions held",
        parent=synthesis_node,
        critical=True
    )
    # Collect prerequisite leaf nodes (previous role + duration for all three)
    prereq_nodes: List[Any] = []
    for bn in built_nodes:
        # Find nodes by ID
        for leaf_id in bn.previous_role_leaf_ids + bn.previous_duration_leaf_ids:
            node = evaluator.find_node(leaf_id)
            if node:
                prereq_nodes.append(node)

    await evaluator.verify(
        claim=f"A common career progression pattern among the three examples is: {career_path_text}.",
        node=career_path_leaf,
        sources=None,  # Derived synthesis; rely on prerequisites to gate validity
        additional_instruction=(
            "Judge this synthesis against the answer text and the extracted fields. "
            "If the prior roles show a consistent transition pattern into Special Teams Coordinator, pass."
        ),
        extra_prerequisites=prereq_nodes
    )

    # Add custom info summary of synthesis for downstream consumers
    evaluator.add_custom_info(
        info={
            "derived_min_years_experience": min_years,
            "all_have_bachelors": all_have_bachelors,
            "career_path_pattern": career_path_text
        },
        info_type="synthesis",
        info_name="synthesis_summary"
    )

    # Return evaluation summary
    return evaluator.get_summary()