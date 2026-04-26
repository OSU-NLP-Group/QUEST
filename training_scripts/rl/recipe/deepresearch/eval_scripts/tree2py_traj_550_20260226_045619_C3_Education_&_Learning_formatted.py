import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "iowa_career_progression_2026"
TASK_DESCRIPTION = (
    "Identify the individual who satisfies all of the following career progression criteria: "
    "(1) Played as a three-year letterwinner linebacker at the University of Iowa from 1998-2000; "
    "(2) Played professionally in the NFL for seven seasons (2001-2007); "
    "(3) Joined the University of Iowa football staff as an administrative assistant in 2008; "
    "(4) Was promoted to an on-field coaching position at Iowa in 2012, coaching linebackers while assisting with special teams; "
    "(5) Was named Iowa's special teams coordinator in 2017; "
    "(6) Served as Iowa's full-time special teams coordinator from 2018 through 2025; "
    "(7) Spent a total of 18 years on Iowa's staff under head coach Kirk Ferentz; "
    "(8) Joined Michigan State in January 2026 as Assistant Head Coach/Special Teams Coordinator; "
    "(9) Led Iowa special teams units that were consistently ranked among national leaders in punt and kick return metrics; "
    "(10) Coached multiple All-Big Ten and All-American special teams specialists during their coordinator tenure. "
    "Provide the individual's full name and include URL references that verify each stage of their career progression "
    "(playing career, early coaching roles, coordinator tenure, current position, and performance achievements)."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class CareerExtraction(BaseModel):
    """
    Extract the individual's full name and URL references grouped by each stage
    of the career progression listed in the task.
    """
    full_name: Optional[str] = None

    # Stage-specific URL lists; must be URLs explicitly present in the answer
    college_career_urls: List[str] = Field(default_factory=list)
    nfl_career_urls: List[str] = Field(default_factory=list)
    administrative_role_urls: List[str] = Field(default_factory=list)
    coaching_promotion_urls: List[str] = Field(default_factory=list)
    coordinator_appointment_urls: List[str] = Field(default_factory=list)
    coordinator_tenure_urls: List[str] = Field(default_factory=list)
    current_position_urls: List[str] = Field(default_factory=list)
    performance_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_career_sources() -> str:
    return """
    Extract the individual's full name and all URL references grouped by each specified career stage. 
    Only extract URLs that are explicitly mentioned in the provided answer. Do not invent any URLs.

    Return a JSON object with the following fields:
    - full_name: the individual's full name as presented in the answer (string or null)
    - college_career_urls: URLs that support the individual's college playing career at Iowa (array)
    - nfl_career_urls: URLs that support the individual's NFL playing career (array)
    - administrative_role_urls: URLs that support that the individual joined Iowa's staff as an administrative assistant in 2008 (array)
    - coaching_promotion_urls: URLs that support that the individual was promoted to an on-field coaching role in 2012 coaching linebackers and assisting special teams (array)
    - coordinator_appointment_urls: URLs that support that the individual was named Iowa's special teams coordinator in 2017 (array)
    - coordinator_tenure_urls: URLs that support the full-time coordinator tenure from 2018 through 2025 and/or total Iowa service years (array)
    - current_position_urls: URLs that support the individual's current position at Michigan State effective January 2026 as Assistant Head Coach/Special Teams Coordinator (array)
    - performance_urls: URLs that support the individual's special teams performance achievements (national rankings, All-Big Ten/All-American specialists) (array)

    SPECIAL RULES:
    - Extract only actual URLs present in the answer (plain URL or markdown link target). If none are provided for a category, return an empty array for that category.
    - Do not duplicate URLs within a category.
    - Include the full URL with protocol. If a URL is missing a protocol, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _valid_urls(urls: Optional[List[str]]) -> List[str]:
    """Filter to plausible URLs to avoid passing garbage to the verifier."""
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        # Normalize protocol if missing
        if not s.startswith("http://") and not s.startswith("https://"):
            s = "http://" + s
        # Basic heuristic: must contain at least one dot and no spaces
        if "." in s and " " not in s:
            cleaned.append(s)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for s in cleaned:
        if s not in seen:
            deduped.append(s)
            seen.add(s)
    return deduped


def _has_any_url(urls: Optional[List[str]]) -> bool:
    """Check whether at least one valid-looking URL is provided."""
    return len(_valid_urls(urls)) > 0


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_name_verification(evaluator: Evaluator, parent, extraction: CareerExtraction) -> None:
    node = evaluator.add_parallel(
        id="Name_Verification",
        desc="Verify that the individual's full name is provided",
        parent=parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(extraction.full_name and extraction.full_name.strip() and " " in extraction.full_name.strip()),
        id="Full_Name_Provided",
        desc="The solution must provide the individual's full name",
        parent=node,
        critical=True
    )


async def build_playing_career_verification(evaluator: Evaluator, parent, extraction: CareerExtraction) -> None:
    pname = extraction.full_name or "the individual"

    play_root = evaluator.add_sequential(
        id="Playing_Career_Verification",
        desc="Verify the individual's playing career from college through professional level",
        parent=parent,
        critical=True
    )

    # College career
    college = evaluator.add_parallel(
        id="College_Career",
        desc="Verify the individual's collegiate playing career at the University of Iowa",
        parent=play_root,
        critical=True
    )

    # Reference existence (critical)
    evaluator.add_custom_node(
        result=_has_any_url(extraction.college_career_urls),
        id="College_Career_Reference",
        desc="Provide a URL reference that confirms the individual's playing career at Iowa (1998-2000)",
        parent=college,
        critical=True
    )

    # Details verification (critical)
    college_details = evaluator.add_leaf(
        id="College_Playing_Details",
        desc="The individual was a three-year letterwinner as a linebacker at the University of Iowa from 1998-2000",
        parent=college,
        critical=True
    )
    college_claim = (
        f"{pname} was a three-year letterwinner as a linebacker at the University of Iowa from 1998 through 2000."
    )
    await evaluator.verify(
        claim=college_claim,
        node=college_details,
        sources=_valid_urls(extraction.college_career_urls),
        additional_instruction=(
            "Confirm the page explicitly supports each element: "
            "three-year letterwinner, linebacker, University of Iowa, and the timeframe 1998–2000 (inclusive). "
            "Treat 'lettered three years' and 'three-time letterwinner' as equivalent phrasing."
        ),
    )

    # NFL professional career
    nfl = evaluator.add_parallel(
        id="Professional_Career",
        desc="Verify the individual's professional NFL playing career",
        parent=play_root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_any_url(extraction.nfl_career_urls),
        id="NFL_Career_Reference",
        desc="Provide a URL reference that confirms the individual's NFL career (2001-2007)",
        parent=nfl,
        critical=True
    )

    nfl_details = evaluator.add_leaf(
        id="NFL_Career_Details",
        desc="The individual played professionally in the NFL for seven seasons from 2001-2007",
        parent=nfl,
        critical=True
    )
    nfl_claim = (
        f"{pname} played professionally in the NFL for seven seasons from 2001 through 2007 (inclusive)."
    )
    await evaluator.verify(
        claim=nfl_claim,
        node=nfl_details,
        sources=_valid_urls(extraction.nfl_career_urls),
        additional_instruction=(
            "Support can be explicit (stating seven seasons, 2001–2007) or implicit via listed team seasons that total seven "
            "within 2001–2007 inclusive. The evidence must clearly indicate the span and professional NFL status."
        ),
    )


async def build_early_coaching_verification(evaluator: Evaluator, parent, extraction: CareerExtraction) -> None:
    pname = extraction.full_name or "the individual"

    early_root = evaluator.add_sequential(
        id="Early_Coaching_Career_Verification",
        desc="Verify the individual's early coaching positions at the University of Iowa",
        parent=parent,
        critical=True
    )

    # Administrative role (2008)
    admin = evaluator.add_parallel(
        id="Administrative_Role",
        desc="Verify the individual's initial administrative position at Iowa",
        parent=early_root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_any_url(extraction.administrative_role_urls),
        id="Administrative_Role_Reference",
        desc="Provide a URL reference that confirms the individual's administrative assistant role starting in 2008",
        parent=admin,
        critical=True
    )

    admin_details = evaluator.add_leaf(
        id="Administrative_Position_Details",
        desc="The individual joined the University of Iowa football staff as an administrative assistant in 2008",
        parent=admin,
        critical=True
    )
    admin_claim = f"{pname} joined the University of Iowa football staff as an administrative assistant in 2008."
    await evaluator.verify(
        claim=admin_claim,
        node=admin_details,
        sources=_valid_urls(extraction.administrative_role_urls),
        additional_instruction=(
            "Accept synonymous phrasing like 'administrative assistant for football' or 'joined staff in 2008 in an "
            "administrative role'. The year must be 2008."
        ),
    )

    # First coaching promotion (2012)
    promo = evaluator.add_parallel(
        id="First_Coaching_Promotion",
        desc="Verify the individual's promotion to on-field coaching",
        parent=early_root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_any_url(extraction.coaching_promotion_urls),
        id="Coaching_Promotion_Reference",
        desc="Provide a URL reference that confirms the individual's promotion to on-field coaching in 2012",
        parent=promo,
        critical=True
    )

    promo_details = evaluator.add_leaf(
        id="Coaching_Promotion_Details",
        desc="The individual was promoted to an on-field coaching position in 2012, coaching linebackers while assisting with special teams",
        parent=promo,
        critical=True
    )
    promo_claim = (
        f"In 2012, {pname} was promoted to an on-field coaching role at Iowa, coaching linebackers while assisting with special teams."
    )
    await evaluator.verify(
        claim=promo_claim,
        node=promo_details,
        sources=_valid_urls(extraction.coaching_promotion_urls),
        additional_instruction=(
            "Evidence should reflect a 2012 promotion to an on-field staff role involving linebackers and assisting with "
            "special teams. Accept close synonyms (e.g., 'assistant coach with linebackers and special teams')."
        ),
    )


async def build_coordinator_career_verification(evaluator: Evaluator, parent, extraction: CareerExtraction) -> None:
    pname = extraction.full_name or "the individual"

    coord_root = evaluator.add_sequential(
        id="Coordinator_Career_Verification",
        desc="Verify the individual's special teams coordinator appointment and tenure at Iowa",
        parent=parent,
        critical=True
    )

    # Appointment (2017)
    appoint = evaluator.add_parallel(
        id="Coordinator_Appointment",
        desc="Verify the individual's appointment as special teams coordinator",
        parent=coord_root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_any_url(extraction.coordinator_appointment_urls),
        id="Coordinator_Appointment_Reference",
        desc="Provide a URL reference that confirms the individual's appointment as special teams coordinator in 2017",
        parent=appoint,
        critical=True
    )

    appoint_details = evaluator.add_leaf(
        id="Coordinator_Appointment_Details",
        desc="The individual was named Iowa's special teams coordinator in 2017",
        parent=appoint,
        critical=True
    )
    appoint_claim = f"{pname} was named Iowa's special teams coordinator in 2017."
    await evaluator.verify(
        claim=appoint_claim,
        node=appoint_details,
        sources=_valid_urls(extraction.coordinator_appointment_urls),
        additional_instruction="The page should clearly indicate the appointment year 2017 and the title 'special teams coordinator'.",
    )

    # Full-time tenure (2018–2025) and total service (18 years, 2008–2025)
    tenure = evaluator.add_parallel(
        id="Full_Time_Coordinator_Tenure",
        desc="Verify the individual's full-time coordinator tenure and total Iowa service",
        parent=coord_root,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_any_url(extraction.coordinator_tenure_urls),
        id="Coordinator_Tenure_Reference",
        desc="Provide a URL reference that confirms the individual's full-time special teams coordinator tenure (2018-2025) and total Iowa service",
        parent=tenure,
        critical=True
    )

    tenure_full_time = evaluator.add_leaf(
        id="Full_Time_Period",
        desc="The individual served as Iowa's full-time special teams coordinator from 2018 through 2025",
        parent=tenure,
        critical=True
    )
    full_time_claim = f"{pname} served as Iowa's full-time special teams coordinator from 2018 through 2025 (inclusive)."
    await evaluator.verify(
        claim=full_time_claim,
        node=tenure_full_time,
        sources=_valid_urls(extraction.coordinator_tenure_urls),
        additional_instruction=(
            "Support may appear as annual staff listings or a biography stating continuous tenure as special teams "
            "coordinator across 2018–2025 inclusive."
        ),
    )

    tenure_total_service = evaluator.add_leaf(
        id="Total_Iowa_Service",
        desc="The individual spent a total of 18 years on Iowa's staff under head coach Kirk Ferentz (2008-2025)",
        parent=tenure,
        critical=True
    )
    total_service_claim = (
        f"{pname} spent a total of 18 years on Iowa's staff under head coach Kirk Ferentz from 2008 through 2025 (inclusive)."
    )
    await evaluator.verify(
        claim=total_service_claim,
        node=tenure_total_service,
        sources=_valid_urls(extraction.coordinator_tenure_urls),
        additional_instruction=(
            "The evidence should explicitly or implicitly support the sum of years 2008–2025 (inclusive) = 18 seasons/years "
            "under Kirk Ferentz."
        ),
    )


async def build_current_position_verification(evaluator: Evaluator, parent, extraction: CareerExtraction) -> None:
    pname = extraction.full_name or "the individual"

    current = evaluator.add_parallel(
        id="Current_Position_Verification",
        desc="Verify the individual's current coaching position at Michigan State",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_any_url(extraction.current_position_urls),
        id="Current_Position_Reference",
        desc="Provide a URL reference that confirms the individual's current position at Michigan State as Assistant Head Coach/Special Teams Coordinator effective January 2026",
        parent=current,
        critical=True
    )

    current_details = evaluator.add_leaf(
        id="Michigan_State_Position",
        desc="The individual joined Michigan State in January 2026 as Assistant Head Coach/Special Teams Coordinator",
        parent=current,
        critical=True
    )
    current_claim = (
        f"{pname} joined Michigan State in January 2026 as Assistant Head Coach/Special Teams Coordinator."
    )
    await evaluator.verify(
        claim=current_claim,
        node=current_details,
        sources=_valid_urls(extraction.current_position_urls),
        additional_instruction=(
            "Prefer official or reputable sources (e.g., MSU Athletics, press releases). "
            "The page should explicitly state January 2026 and the title 'Assistant Head Coach/Special Teams Coordinator'."
        ),
    )


async def build_performance_verification(evaluator: Evaluator, parent, extraction: CareerExtraction) -> None:
    pname = extraction.full_name or "the individual"

    perf = evaluator.add_parallel(
        id="Performance_Verification",
        desc="Verify the individual's achievements and performance as special teams coordinator at Iowa",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_any_url(extraction.performance_urls),
        id="Performance_Reference",
        desc="Provide a URL reference that confirms the individual's special teams performance metrics and player development achievements at Iowa",
        parent=perf,
        critical=True
    )

    rankings = evaluator.add_leaf(
        id="National_Rankings_Achievement",
        desc="The individual's special teams units at Iowa were consistently ranked among national leaders in punt and kick return metrics",
        parent=perf,
        critical=True
    )
    rankings_claim = (
        f"Under {pname}, Iowa special teams units were consistently ranked among national leaders in punt and kick return metrics."
    )
    await evaluator.verify(
        claim=rankings_claim,
        node=rankings,
        sources=_valid_urls(extraction.performance_urls),
        additional_instruction=(
            "The page(s) should indicate that Iowa's special teams (during the coordinator's tenure) repeatedly ranked among "
            "national leaders for punt return, kickoff return, coverage, or related efficiency metrics across multiple seasons."
        ),
    )

    dev = evaluator.add_leaf(
        id="Player_Development_Achievement",
        desc="The individual coached multiple All-Big Ten and All-American special teams specialists during their tenure as coordinator",
        parent=perf,
        critical=True
    )
    dev_claim = (
        f"During the coordinator tenure, {pname} coached multiple All-Big Ten and All-American special teams specialists."
    )
    await evaluator.verify(
        claim=dev_claim,
        node=dev,
        sources=_valid_urls(extraction.performance_urls),
        additional_instruction=(
            "Accept if sources list multiple specialists (e.g., kickers, punters, returners) earning All-Big Ten and/or "
            "All-America honors during the coordinator's tenure."
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
    Evaluate an answer for the Iowa career progression identification task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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
    # Reflect rubric: Root is critical; ensure all added children will be critical too
    root.critical = True

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_career_sources(),
        template_class=CareerExtraction,
        extraction_name="career_extraction"
    )

    # Record criteria as "ground truth template" for transparency (not absolute ground truth)
    evaluator.add_ground_truth({
        "criteria": [
            "Three-year letterwinner linebacker at Iowa (1998–2000)",
            "NFL seven seasons (2001–2007)",
            "Joined Iowa staff as administrative assistant in 2008",
            "Promoted to on-field role in 2012 (linebackers, assist special teams)",
            "Named Iowa special teams coordinator in 2017",
            "Full-time ST coordinator 2018–2025",
            "18 total Iowa staff years 2008–2025 (under Kirk Ferentz)",
            "Joined Michigan State in Jan 2026 as Assistant Head Coach/Special Teams Coordinator",
            "Iowa ST units consistently among national leaders (punt/kick return metrics)",
            "Coached multiple All-Big Ten and All-American ST specialists"
        ]
    })

    # Build verification tree according to rubric (all critical, root sequential)
    await build_name_verification(evaluator, root, extraction)
    await build_playing_career_verification(evaluator, root, extraction)
    await build_early_coaching_verification(evaluator, root, extraction)
    await build_coordinator_career_verification(evaluator, root, extraction)
    await build_current_position_verification(evaluator, root, extraction)
    await build_performance_verification(evaluator, root, extraction)

    # Optional: record extracted quick summary
    evaluator.add_custom_info(
        info={
            "extracted_full_name": extraction.full_name,
            "url_counts": {
                "college": len(_valid_urls(extraction.college_career_urls)),
                "nfl": len(_valid_urls(extraction.nfl_career_urls)),
                "admin": len(_valid_urls(extraction.administrative_role_urls)),
                "promotion": len(_valid_urls(extraction.coaching_promotion_urls)),
                "coord_appt": len(_valid_urls(extraction.coordinator_appointment_urls)),
                "coord_tenure": len(_valid_urls(extraction.coordinator_tenure_urls)),
                "current": len(_valid_urls(extraction.current_position_urls)),
                "performance": len(_valid_urls(extraction.performance_urls)),
            }
        },
        info_type="extraction_summary"
    )

    # Return structured evaluation summary
    return evaluator.get_summary()