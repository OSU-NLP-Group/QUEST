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
TASK_ID = "nasa_admin_mission_research_2026"
TASK_DESCRIPTION = (
    "Who is NASA's current Administrator (as of February 2026)? Identify the private space mission they previously "
    "commanded, and provide the following details about that mission: (1) the approximate number of scientific "
    "research experiments conducted, and (2) the approximate number of partner institutions involved in the research "
    "portfolio. Include reference URLs for each piece of information."
)

AS_OF_DATE_TEXT = "February 2026"
EXPECTED_ADMIN_NAME = "Jared Isaacman"
EXPECTED_MISSION_NAME = "Polaris Dawn"
EXPECTED_EXPERIMENTS_RANGE_DESC = "nearly 40 (approximately 40) experiments, or any specific number in the range 36–40"
EXPECTED_INSTITUTIONS_RANGE_DESC = "over 30 partner institutions (30+), or any specific number in the range 30–35"
EXPECTED_CONFIRMATION_TIMING_DESC = "confirmed December 17, 2025 and sworn in December 18, 2025"
EXPECTED_MISSION_DATES_DESC = "September 10–15, 2024"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AdminInfo(BaseModel):
    """Administrator identity and appointment timing, with sources extracted from the answer."""
    name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    confirmation_timing: Optional[str] = None
    confirmation_urls: List[str] = Field(default_factory=list)


class MissionInfo(BaseModel):
    """Private mission identification and dates, with sources extracted from the answer."""
    name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    dates: Optional[str] = None
    date_urls: List[str] = Field(default_factory=list)


class ResearchInfo(BaseModel):
    """Research portfolio counts for the mission, with sources extracted from the answer."""
    experiments_count: Optional[str] = None
    experiments_urls: List[str] = Field(default_factory=list)
    institutions_count: Optional[str] = None
    institutions_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_admin_info() -> str:
    return (
        "From the answer, extract the current NASA Administrator identity and appointment timing details. "
        "Return JSON with fields:\n"
        "• name: administrator name as stated in the answer (string or null)\n"
        "• urls: array of URLs that the answer cites specifically to support the administrator identity (array; empty if none)\n"
        "• confirmation_timing: text about timing of confirmation/swearing-in (e.g., 'December 2025', or specific dates) (string or null)\n"
        "• confirmation_urls: array of URLs cited to support the timing (array; empty if none)\n"
        "Follow URL extraction rules strictly: include only URLs explicitly present in the answer (plain or markdown)."
    )


def prompt_extract_mission_info() -> str:
    return (
        "From the answer, extract the previously commanded private space mission and its dates. "
        "Return JSON with fields:\n"
        "• name: mission name (string or null)\n"
        "• urls: array of URLs cited to support the mission identification and commander relationship (array; empty if none)\n"
        "• dates: mission dates text as stated in the answer (string or null), e.g., 'September 10–15, 2024'\n"
        "• date_urls: array of URLs cited to support the mission dates (array; empty if none)\n"
        "Extract only URLs explicitly present in the answer."
    )


def prompt_extract_research_info() -> str:
    return (
        "From the answer, extract research portfolio counts for the mission. "
        "Return JSON with fields:\n"
        "• experiments_count: text indicating the approximate number of scientific experiments (e.g., 'nearly 40', 'about 38') (string or null)\n"
        "• experiments_urls: array of URLs cited to support the experiments count (array; empty if none)\n"
        "• institutions_count: text indicating the approximate number of partner institutions (e.g., 'over 30', '30+') (string or null)\n"
        "• institutions_urls: array of URLs cited to support the institutions count (array; empty if none)\n"
        "Extract only URLs explicitly present in the answer."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _non_empty_text(s: Optional[str]) -> bool:
    return bool(s) and bool(s.strip())


def _non_empty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_administrator_section(evaluator: Evaluator, parent_node, admin: AdminInfo) -> None:
    """
    Build Administrator section under the critical Complete_Answer node.
    """
    admin_section = evaluator.add_sequential(
        id="Administrator_Section",
        desc="Provides complete information about NASA's current Administrator",
        parent=parent_node,
        critical=True,
    )

    identity_node = evaluator.add_parallel(
        id="Administrator_Identity",
        desc="Correctly identifies the current NASA Administrator with verification",
        parent=admin_section,
        critical=True,
    )

    name_criterion = evaluator.add_parallel(
        id="Administrator_Name_Criterion",
        desc="States the administrator's name",
        parent=identity_node,
        critical=True,
    )

    # Existence checks (critical gating)
    evaluator.add_custom_node(
        result=_non_empty_text(admin.name),
        id="Administrator_Name_Provided",
        desc="Administrator name is provided in the answer",
        parent=name_criterion,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty_urls(admin.urls),
        id="Administrator_Name_Reference_Provided",
        desc="Reference URLs confirming the administrator are provided",
        parent=name_criterion,
        critical=True,
    )

    # Leaf: Name Value (match to expected)
    name_value_leaf = evaluator.add_leaf(
        id="Administrator_Name_Value",
        desc="Correctly identifies Jared Isaacman as NASA's current (15th) Administrator",
        parent=name_criterion,
        critical=True,
    )
    claim_match = (
        f"The administrator name provided in the answer ('{admin.name or ''}') matches '{EXPECTED_ADMIN_NAME}'. "
        "Treat case-insensitive equivalence and allow minor formatting differences (e.g., middle initials)."
    )
    await evaluator.verify(
        claim=claim_match,
        node=name_value_leaf,
        additional_instruction="Judge only the name equality; ignore unrelated content.",
    )

    # Leaf: Name Reference (supported by URLs)
    name_ref_leaf = evaluator.add_leaf(
        id="Administrator_Name_Reference",
        desc="Provides a valid reference URL confirming Jared Isaacman as NASA Administrator",
        parent=name_criterion,
        critical=True,
    )
    claim_admin_supported = f"As of {AS_OF_DATE_TEXT}, {EXPECTED_ADMIN_NAME} is the NASA Administrator."
    await evaluator.verify(
        claim=claim_admin_supported,
        node=name_ref_leaf,
        sources=admin.urls,
        additional_instruction=(
            "Confirm using the provided URLs (e.g., NASA.gov announcements or credible news) that Jared Isaacman is the NASA Administrator as of February 2026. "
            "If URLs are invalid, irrelevant, or do not state this, mark as not supported."
        ),
    )


async def build_mission_section(evaluator: Evaluator, parent_node, mission: MissionInfo) -> None:
    """
    Build Mission section under the critical Complete_Answer node.
    """
    mission_section = evaluator.add_sequential(
        id="Mission_Section",
        desc="Provides complete information about the private space mission previously commanded",
        parent=parent_node,
        critical=True,
    )

    identity_node = evaluator.add_parallel(
        id="Mission_Identity",
        desc="Correctly identifies the mission with verification",
        parent=mission_section,
        critical=True,
    )

    mission_criterion = evaluator.add_parallel(
        id="Mission_Name_Criterion",
        desc="States the mission name",
        parent=identity_node,
        critical=True,
    )

    # Existence checks (critical gating)
    evaluator.add_custom_node(
        result=_non_empty_text(mission.name),
        id="Mission_Name_Provided",
        desc="Mission name is provided in the answer",
        parent=mission_criterion,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty_urls(mission.urls),
        id="Mission_Reference_Provided",
        desc="Reference URLs about the mission are provided",
        parent=mission_criterion,
        critical=True,
    )

    # Leaf: Mission Name Value (match to expected)
    mission_value_leaf = evaluator.add_leaf(
        id="Mission_Name_Value",
        desc="Correctly identifies Polaris Dawn as the mission commanded by Jared Isaacman",
        parent=mission_criterion,
        critical=True,
    )
    claim_mission_match = (
        f"The mission name provided in the answer ('{mission.name or ''}') matches '{EXPECTED_MISSION_NAME}'. "
        "Treat case-insensitive equivalence and allow minor formatting differences."
    )
    await evaluator.verify(
        claim=claim_mission_match,
        node=mission_value_leaf,
        additional_instruction="Judge only the mission name equality; ignore unrelated content.",
    )

    # Leaf: Mission Reference (supported by URLs, including commander relationship)
    mission_ref_leaf = evaluator.add_leaf(
        id="Mission_Name_Reference",
        desc="Provides a valid reference URL about the Polaris Dawn mission",
        parent=mission_criterion,
        critical=True,
    )
    claim_mission_supported = (
        f"{EXPECTED_ADMIN_NAME} commanded the {EXPECTED_MISSION_NAME} mission (a private space mission)."
    )
    await evaluator.verify(
        claim=claim_mission_supported,
        node=mission_ref_leaf,
        sources=mission.urls,
        additional_instruction=(
            "Confirm that the provided URLs explicitly state that Jared Isaacman commanded Polaris Dawn. "
            "If URLs are irrelevant or do not state this, mark as not supported."
        ),
    )


async def build_research_section(evaluator: Evaluator, parent_node, research: ResearchInfo) -> None:
    """
    Build Research Portfolio section under the critical Complete_Answer node.
    """
    research_section = evaluator.add_parallel(
        id="Research_Portfolio_Section",
        desc="Provides complete information about the research conducted during the mission",
        parent=parent_node,
        critical=True,
    )

    # Experiments
    experiments_info = evaluator.add_sequential(
        id="Experiments_Information",
        desc="Provides information about the number of scientific research experiments",
        parent=research_section,
        critical=True,
    )

    # Existence gating as first child in sequential branch
    evaluator.add_custom_node(
        result=_non_empty_text(research.experiments_count) and _non_empty_urls(research.experiments_urls),
        id="Experiments_Info_Provided",
        desc="Experiments count text and supporting URLs are provided",
        parent=experiments_info,
        critical=True,
    )

    experiments_criterion = evaluator.add_parallel(
        id="Experiments_Count_Criterion",
        desc="States the approximate number of experiments with verification",
        parent=experiments_info,
        critical=True,
    )

    # Leaf: Experiments Count Value (format/range check via simple verify)
    experiments_value_leaf = evaluator.add_leaf(
        id="Experiments_Count_Value",
        desc=("Provides the correct count of experiments (nearly 40, approximately 40, or a specific number in the range 36-40)"),
        parent=experiments_criterion,
        critical=True,
    )
    claim_exp_value = (
        f"The extracted experiments count text '{research.experiments_count or ''}' indicates approximately 40 experiments "
        "(acceptable phrasing: 'nearly 40', 'approximately 40', 'about 40') or a specific number in the range 36–40."
    )
    await evaluator.verify(
        claim=claim_exp_value,
        node=experiments_value_leaf,
        additional_instruction=(
            "Judge based solely on the extracted text whether it communicates ~40 or a number between 36 and 40."
        ),
    )

    # Leaf: Experiments Count Reference (supported by URLs)
    experiments_ref_leaf = evaluator.add_leaf(
        id="Experiments_Count_Reference",
        desc="Provides a valid reference URL confirming the number of experiments conducted during Polaris Dawn",
        parent=experiments_criterion,
        critical=True,
    )
    claim_exp_supported = (
        f"The {EXPECTED_MISSION_NAME} mission conducted nearly 40 (approximately 40) scientific research experiments."
    )
    await evaluator.verify(
        claim=claim_exp_supported,
        node=experiments_ref_leaf,
        sources=research.experiments_urls,
        additional_instruction=(
            "Accept explicit statements like 'nearly 40' or 'approximately 40', and specific counts 36–40 on the provided URLs."
        ),
    )

    # Institutions
    institutions_info = evaluator.add_sequential(
        id="Institutions_Information",
        desc="Provides information about the number of partner institutions",
        parent=research_section,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_non_empty_text(research.institutions_count) and _non_empty_urls(research.institutions_urls),
        id="Institutions_Info_Provided",
        desc="Institutions count text and supporting URLs are provided",
        parent=institutions_info,
        critical=True,
    )

    institutions_criterion = evaluator.add_parallel(
        id="Institutions_Count_Criterion",
        desc="States the approximate number of partner institutions with verification",
        parent=institutions_info,
        critical=True,
    )

    # Leaf: Institutions Count Value (format/range check via simple verify)
    institutions_value_leaf = evaluator.add_leaf(
        id="Institutions_Count_Value",
        desc=("Provides the correct count of partner institutions (over 30, or 30+, or a specific number in the range 30-35)"),
        parent=institutions_criterion,
        critical=True,
    )
    claim_inst_value = (
        f"The extracted institutions count text '{research.institutions_count or ''}' indicates over 30 institutions "
        "(acceptable phrasing: 'over 30', '30+', 'more than 30') or a specific number in the range 30–35."
    )
    await evaluator.verify(
        claim=claim_inst_value,
        node=institutions_value_leaf,
        additional_instruction=(
            "Judge based solely on the extracted text whether it communicates 30+ or a number between 30 and 35."
        ),
    )

    # Leaf: Institutions Count Reference (supported by URLs)
    institutions_ref_leaf = evaluator.add_leaf(
        id="Institutions_Count_Reference",
        desc="Provides a valid reference URL confirming the number of partner institutions involved in Polaris Dawn research",
        parent=institutions_criterion,
        critical=True,
    )
    claim_inst_supported = (
        f"Over 30 partner institutions participated in the {EXPECTED_MISSION_NAME} mission research portfolio."
    )
    await evaluator.verify(
        claim=claim_inst_supported,
        node=institutions_ref_leaf,
        sources=research.institutions_urls,
        additional_instruction=(
            "Accept phrasing like 'over 30' or '30+', and specific counts in 30–35 on the provided URLs."
        ),
    )


async def build_optional_context(evaluator: Evaluator, root_node, admin: AdminInfo, mission: MissionInfo) -> None:
    """
    Build optional context branch (non-critical) under the root to avoid violating critical-children constraints.
    """
    optional_ctx = evaluator.add_parallel(
        id="Supplementary_Context",
        desc="Additional contextual information about appointment and mission timeline (optional)",
        parent=root_node,
        critical=False,
    )

    # Administrator context: confirmation timing
    admin_ctx = evaluator.add_parallel(
        id="Administrator_Context",
        desc="Provides contextual information about the administrator's appointment",
        parent=optional_ctx,
        critical=False,
    )

    confirmation_leaf = evaluator.add_leaf(
        id="Confirmation_Timing",
        desc=("Provides the confirmation timing (December 2025, specifically confirmed December 17 and sworn in December 18, 2025)"),
        parent=admin_ctx,
        critical=False,
    )
    claim_confirmation = (
        f"{EXPECTED_ADMIN_NAME} was confirmed on December 17, 2025 and sworn in on December 18, 2025 as NASA Administrator."
    )
    await evaluator.verify(
        claim=claim_confirmation,
        node=confirmation_leaf,
        sources=admin.confirmation_urls if _non_empty_urls(admin.confirmation_urls) else None,
        additional_instruction=(
            "If the provided URL states just 'December 2025' without specific dates but is clearly about the appointment, consider it reasonably supportive."
        ),
    )

    # Mission context: dates
    mission_ctx = evaluator.add_parallel(
        id="Mission_Context",
        desc="Provides contextual information about the mission",
        parent=optional_ctx,
        critical=False,
    )

    mission_dates_leaf = evaluator.add_leaf(
        id="Mission_Dates",
        desc=("Provides the mission dates (September 10-15, 2024)"),
        parent=mission_ctx,
        critical=False,
    )
    claim_mission_dates = f"The {EXPECTED_MISSION_NAME} mission occurred during September 10–15, 2024."
    await evaluator.verify(
        claim=claim_mission_dates,
        node=mission_dates_leaf,
        sources=mission.date_urls if _non_empty_urls(mission.date_urls) else None,
        additional_instruction=(
            "If the provided URL lists a near-identical date range for the mission timeline, consider it supportive."
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
    Evaluate an answer for the NASA Administrator and mission research portfolio task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root stays non-critical, parallel aggregation
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

    # Extract structured data concurrently
    admin_info_task = evaluator.extract(
        prompt=prompt_extract_admin_info(),
        template_class=AdminInfo,
        extraction_name="admin_info",
    )
    mission_info_task = evaluator.extract(
        prompt=prompt_extract_mission_info(),
        template_class=MissionInfo,
        extraction_name="mission_info",
    )
    research_info_task = evaluator.extract(
        prompt=prompt_extract_research_info(),
        template_class=ResearchInfo,
        extraction_name="research_info",
    )

    admin_info, mission_info, research_info = await asyncio.gather(
        admin_info_task, mission_info_task, research_info_task
    )

    # Add ground truth info (for transparency)
    evaluator.add_ground_truth({
        "as_of_date": AS_OF_DATE_TEXT,
        "expected_administrator": EXPECTED_ADMIN_NAME,
        "expected_mission": EXPECTED_MISSION_NAME,
        "expected_experiments": EXPECTED_EXPERIMENTS_RANGE_DESC,
        "expected_institutions": EXPECTED_INSTITUTIONS_RANGE_DESC,
        "expected_confirmation_timing": EXPECTED_CONFIRMATION_TIMING_DESC,
        "expected_mission_dates": EXPECTED_MISSION_DATES_DESC,
    })

    # Build the "Complete_Answer" critical node (mirrors rubric tree root)
    complete_node = evaluator.add_parallel(
        id="Complete_Answer",
        desc=("The answer provides comprehensive information about NASA's current Administrator, their previous space "
              "mission command experience, and the research conducted during that mission"),
        parent=root,
        critical=True,
    )

    # Build three critical sections under Complete_Answer
    await build_administrator_section(evaluator, complete_node, admin_info)
    await build_mission_section(evaluator, complete_node, mission_info)
    await build_research_section(evaluator, complete_node, research_info)

    # Build optional context branch under root to avoid critical-child constraint violations
    await build_optional_context(evaluator, root, admin_info, mission_info)

    # Return structured evaluation summary
    return evaluator.get_summary()