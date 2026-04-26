import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "woodworking_advanced_certificate_ne_spring_2026"
TASK_DESCRIPTION = """
Identify an advanced woodworking certificate program in New England that begins in spring 2026 (March through May), requires prior woodworking experience or basic certificate completion as a prerequisite, can be completed in 30 class days or fewer, and is approved for GI Bill or VA educational benefits. Provide the program name, hosting institution, location (city and state), specific start date, duration in class days, and confirmation of GI Bill approval.
"""

NEW_ENGLAND_STATES = {"CT", "ME", "MA", "NH", "RI", "VT"}
SPRING_2026_START = datetime(2026, 3, 1)
SPRING_2026_END = datetime(2026, 5, 31)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramExtraction(BaseModel):
    program_name: Optional[str] = None
    institution_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Prefer two-letter abbreviation if available
    start_date: Optional[str] = None  # As written in the answer
    duration_days: Optional[str] = None  # As written in the answer, e.g., "24 class days"
    gi_bill_approval_text: Optional[str] = None  # Explicit confirmation text from the answer
    program_url: Optional[str] = None  # Primary program page URL (if any)
    source_urls: List[str] = Field(default_factory=list)  # All URLs cited in the answer relevant to this program


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program_info() -> str:
    return """
Extract the details of a single woodworking certificate program from the answer that the author claims meets these constraints:
- Advanced-level
- Located in New England (CT, ME, MA, NH, RI, or VT)
- Starts between March 1 and May 31, 2026 (a specific start date in that range)
- Can be completed in 30 class days or fewer (duration expressed in class days)
- Approved for GI Bill or VA educational benefits

Return the following fields exactly as they appear in the answer (do not infer or calculate):
- program_name: The program's name
- institution_name: The hosting institution's name
- city: The city where the program is held
- state: The two-letter state code if provided; otherwise, the state name
- start_date: The specific start date referenced in the answer for spring 2026 (March–May 2026). If multiple dates are present, select the one in this window.
- duration_days: The duration in class days as written (e.g., '24 class days'). If only weeks are mentioned and class days are not explicitly stated, set to null.
- gi_bill_approval_text: The phrase in the answer that explicitly confirms GI Bill or VA educational benefits (e.g., 'GI Bill approved', 'VA educational benefits accepted').
- program_url: The primary URL of the program page, if one is clearly identifiable among the cited links.
- source_urls: A list of all URLs cited in the answer that support any of the above information (including the program_url if present). Extract only actual URLs present in the answer text, including those within markdown links. Do not invent URLs.
"""


# --------------------------------------------------------------------------- #
# Helper utility functions                                                    #
# --------------------------------------------------------------------------- #
def _dedupe_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def combine_all_urls(extracted: ProgramExtraction) -> List[str]:
    urls: List[str] = []
    if extracted.program_url:
        urls.append(extracted.program_url)
    if extracted.source_urls:
        urls.extend(extracted.source_urls)
    return _dedupe_preserve_order(urls)


def parse_int_from_string(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    num = ""
    in_number = False
    for ch in s:
        if ch.isdigit():
            num += ch
            in_number = True
        else:
            if in_number:
                break
    try:
        return int(num) if num else None
    except Exception:
        return None


def try_parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    patterns = [
        "%B %d, %Y",      # March 15, 2026
        "%b %d, %Y",      # Mar 15, 2026
        "%B %d %Y",       # March 15 2026
        "%b %d %Y",       # Mar 15 2026
        "%m/%d/%Y",       # 03/15/2026
        "%m/%d/%y",       # 03/15/26
        "%Y-%m-%d",       # 2026-03-15
        "%d %B %Y",       # 15 March 2026
        "%d %b %Y",       # 15 Mar 2026
    ]
    for p in patterns:
        try:
            return datetime.strptime(date_str.strip(), p)
        except Exception:
            continue
    # Try a lax fallback for cases like "March 2026" → assume 1st of the month
    try:
        for month_fmt in ["%B %Y", "%b %Y"]:
            try_dt = datetime.strptime(date_str.strip(), month_fmt)
            return datetime(try_dt.year, try_dt.month, 1)
    except Exception:
        pass
    return None


def is_in_new_england(state: Optional[str]) -> bool:
    if not state:
        return False
    s = state.strip().upper()
    # Map full names to abbreviations if needed
    full_to_abbrev = {
        "CONNECTICUT": "CT",
        "MAINE": "ME",
        "MASSACHUSETTS": "MA",
        "NEW HAMPSHIRE": "NH",
        "RHODE ISLAND": "RI",
        "VERMONT": "VT",
    }
    if s in NEW_ENGLAND_STATES:
        return True
    if s in full_to_abbrev:
        return full_to_abbrev[s] in NEW_ENGLAND_STATES
    return False


def date_in_spring_2026(date_str: Optional[str]) -> bool:
    d = try_parse_date(date_str)
    if not d:
        return False
    return SPRING_2026_START <= d <= SPRING_2026_END


# --------------------------------------------------------------------------- #
# Verification tree construction & checks                                     #
# --------------------------------------------------------------------------- #
async def build_and_verify_constraints(evaluator: Evaluator, parent_node, extracted: ProgramExtraction):
    """
    Build the 'Constraint_Verification' subtree and run URL-grounded checks.
    """
    # Aggregate all URLs
    urls = combine_all_urls(extracted)

    # Constraint_Verification (critical, parallel)
    constraints_node = evaluator.add_parallel(
        id="constraint_verification",
        desc="Verify the identified program satisfies all six mandatory constraints",
        parent=parent_node,
        critical=True
    )

    # 1) Program_Level_Prerequisites (critical, parallel)
    plp_node = evaluator.add_parallel(
        id="program_level_prerequisites",
        desc="The program is classified as advanced-level AND requires prior woodworking experience or basic certificate completion as a prerequisite",
        parent=constraints_node,
        critical=True
    )

    # 1.a Advanced level supported by sources
    adv_leaf = evaluator.add_leaf(
        id="advanced_level_supported",
        desc="The program is advanced-level as supported by cited sources",
        parent=plp_node,
        critical=True
    )
    if not urls:
        adv_leaf.score = 0.0
        adv_leaf.status = "failed"
    else:
        prog_name = extracted.program_name or "the program"
        claim = f"The program '{prog_name}' is an advanced-level woodworking certificate program (not beginner)."
        await evaluator.verify(
            claim=claim,
            node=adv_leaf,
            sources=urls,
            additional_instruction="Accept clear synonyms like 'Advanced', 'Level II/III', 'Advanced Certificate', or 'Professional' that explicitly indicate advanced level. Do not accept 'beginner' or purely 'intermediate' unless explicitly equated to advanced."
        )

    # 1.b Prerequisites supported by sources
    prereq_leaf = evaluator.add_leaf(
        id="prerequisites_supported",
        desc="The program requires prior woodworking experience or completion of a basic certificate as a prerequisite",
        parent=plp_node,
        critical=True
    )
    if not urls:
        prereq_leaf.score = 0.0
        prereq_leaf.status = "failed"
    else:
        prog_name = extracted.program_name or "the program"
        claim = f"The program '{prog_name}' requires prior woodworking experience or completion of a basic/foundational woodworking certificate as a prerequisite (strong requirement, not just a recommendation)."
        await evaluator.verify(
            claim=claim,
            node=prereq_leaf,
            sources=urls,
            additional_instruction="Look for explicit prerequisite language such as 'prerequisite: prior woodworking experience', 'completion of Basic Woodworking required', or similar. Recommendations without requirement do NOT satisfy this."
        )

    # 2) Geographic_Temporal_Constraints (critical, parallel)
    gtc_node = evaluator.add_parallel(
        id="geographic_temporal_constraints",
        desc="The program is located in a New England state AND has a confirmed session starting between March 1 and May 31, 2026",
        parent=constraints_node,
        critical=True
    )

    # 2.a Location supported by sources
    loc_leaf = evaluator.add_leaf(
        id="location_supported",
        desc="The program's city and state location are supported by cited sources",
        parent=gtc_node,
        critical=True
    )
    if not urls:
        loc_leaf.score = 0.0
        loc_leaf.status = "failed"
    else:
        city = extracted.city or ""
        state = extracted.state or ""
        prog_name = extracted.program_name or "the program"
        claim = f"The program '{prog_name}' is located in {city}, {state}."
        await evaluator.verify(
            claim=claim,
            node=loc_leaf,
            sources=urls,
            additional_instruction="Verify the city and state location from the program or institution webpage. Minor formatting differences in city names are acceptable if clearly the same location."
        )

    # 2.b State is in New England (logic check)
    ne_state_leaf = evaluator.add_custom_node(
        result=is_in_new_england(extracted.state),
        id="state_in_new_england",
        desc="The program's state is in New England (CT, ME, MA, NH, RI, or VT)",
        parent=gtc_node,
        critical=True
    )

    # 2.c Start date supported by sources
    start_leaf = evaluator.add_leaf(
        id="start_date_supported",
        desc="The program has a confirmed session starting on the provided start date",
        parent=gtc_node,
        critical=True
    )
    if not urls or not extracted.start_date:
        start_leaf.score = 0.0
        start_leaf.status = "failed"
    else:
        prog_name = extracted.program_name or "the program"
        claim = f"The program '{prog_name}' has a confirmed session that starts on {extracted.start_date}."
        await evaluator.verify(
            claim=claim,
            node=start_leaf,
            sources=urls,
            additional_instruction="Verify the session start date for spring 2026 (March–May 2026) on the program or institution page, schedule page, or official calendar."
        )

    # 2.d Start date within Mar 1 – May 31, 2026 (logic check)
    in_range_leaf = evaluator.add_custom_node(
        result=date_in_spring_2026(extracted.start_date),
        id="start_date_in_range",
        desc="The start date falls between March 1 and May 31, 2026 (inclusive)",
        parent=gtc_node,
        critical=True
    )

    # 3) Duration_Approval_Constraints (critical, parallel)
    dac_node = evaluator.add_parallel(
        id="duration_approval_constraints",
        desc="The program can be completed in 30 class days or fewer AND is approved for GI Bill or VA educational benefits",
        parent=constraints_node,
        critical=True
    )

    # 3.a Duration supported by sources
    duration_leaf = evaluator.add_leaf(
        id="duration_supported",
        desc="The duration in class days is supported by cited sources",
        parent=dac_node,
        critical=True
    )
    if not urls or not extracted.duration_days:
        duration_leaf.score = 0.0
        duration_leaf.status = "failed"
    else:
        prog_name = extracted.program_name or "the program"
        claim = f"The program '{prog_name}' has a duration of {extracted.duration_days}."
        await evaluator.verify(
            claim=claim,
            node=duration_leaf,
            sources=urls,
            additional_instruction="Confirm that the program duration is expressed in class days (or clearly convertible to class days as described). Prefer explicit 'class days' statements; if only weeks are shown with a clear day-per-week schedule, reasonable interpretation is allowed."
        )

    # 3.b Duration ≤ 30 class days (logic check)
    days_num = parse_int_from_string(extracted.duration_days)
    duration_leq_leaf = evaluator.add_custom_node(
        result=(days_num is not None and days_num <= 30),
        id="duration_leq_30",
        desc="The duration is 30 class days or fewer",
        parent=dac_node,
        critical=True
    )

    # 3.c GI Bill/VA approval supported by sources
    gi_leaf = evaluator.add_leaf(
        id="gi_bill_supported",
        desc="The program is approved for GI Bill or VA educational benefits, supported by cited sources",
        parent=dac_node,
        critical=True
    )
    if not urls:
        gi_leaf.score = 0.0
        gi_leaf.status = "failed"
    else:
        prog_name = extracted.program_name or "the program"
        claim = f"The program '{prog_name}' is approved for GI Bill or VA educational benefits."
        await evaluator.verify(
            claim=claim,
            node=gi_leaf,
            sources=urls,
            additional_instruction="Accept clear statements such as 'GI Bill approved', 'VA educational benefits accepted', or references to VA approval/WEAMS that explicitly apply to this program or the certificate offering."
        )


def build_and_check_information_provision(evaluator: Evaluator, parent_node, extracted: ProgramExtraction):
    """
    Build the 'Information_Provision' subtree and run existence checks.
    """
    info_node = evaluator.add_parallel(
        id="information_provision",
        desc="Verify all requested information fields are explicitly provided in the answer",
        parent=parent_node,
        critical=True
    )

    # Program identity (program name + institution name)
    id_node = evaluator.add_parallel(
        id="program_identity_information",
        desc="The answer provides the program name AND the hosting institution name",
        parent=info_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.program_name and extracted.program_name.strip()),
        id="program_name_provided",
        desc="Program name is provided in the answer",
        parent=id_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.institution_name and extracted.institution_name.strip()),
        id="institution_name_provided",
        desc="Hosting institution name is provided in the answer",
        parent=id_node,
        critical=True
    )

    # Location (city + state)
    loc_node = evaluator.add_parallel(
        id="location_information",
        desc="The answer provides both the city and state where the program is located",
        parent=info_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.city and extracted.city.strip()),
        id="city_provided",
        desc="City is provided in the answer",
        parent=loc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.state and extracted.state.strip()),
        id="state_provided",
        desc="State is provided in the answer",
        parent=loc_node,
        critical=True
    )

    # Schedule (start date + duration in class days)
    sched_node = evaluator.add_parallel(
        id="schedule_information",
        desc="The answer provides the specific start date AND the duration expressed in class days",
        parent=info_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.start_date and extracted.start_date.strip()),
        id="start_date_provided",
        desc="Specific start date is provided in the answer",
        parent=sched_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.duration_days and extracted.duration_days.strip()),
        id="duration_days_provided",
        desc="Duration in class days is provided in the answer",
        parent=sched_node,
        critical=True
    )

    # GI Bill confirmation explicitly provided in answer
    evaluator.add_custom_node(
        result=bool(extracted.gi_bill_approval_text and extracted.gi_bill_approval_text.strip()),
        id="gi_bill_confirmation_provided",
        desc="The answer explicitly confirms the program's GI Bill approval status",
        parent=info_node,
        critical=True
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
    Evaluate an answer for the advanced woodworking certificate program task.
    """
    # Initialize evaluator (root is always non-critical; we'll add a critical child node)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Program identification flow is sequential
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

    # Extract structured program information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_program_info(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction",
    )

    # Add custom info for transparency
    evaluator.add_custom_info(
        {"new_england_states": sorted(list(NEW_ENGLAND_STATES)),
         "spring_2026_window": {"start": SPRING_2026_START.strftime("%Y-%m-%d"),
                                "end": SPRING_2026_END.strftime("%Y-%m-%d")}},
        info_type="constraints_context",
        info_name="constraints_context"
    )

    # Program Identification main node (critical, sequential)
    program_node = evaluator.add_sequential(
        id="program_identification",
        desc="Identify and verify a woodworking certificate program that meets all specified criteria and provide all requested information",
        parent=root,
        critical=True
    )

    # Build constraints subtree and run verifications
    await build_and_verify_constraints(evaluator, program_node, extracted)

    # Build information provision subtree (sequential parent ensures it may be skipped if constraints fail)
    build_and_check_information_provision(evaluator, program_node, extracted)

    # Return evaluation summary
    return evaluator.get_summary()