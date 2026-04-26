import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "qualified_online_mba_programs"
TASK_DESCRIPTION = """
Identify online Master of Business Administration (MBA) programs offered by U.S. institutions that meet ALL of the following requirements:

1. Listed in U.S. News & World Report's 2025 Best Online MBA Programs ranking (top 50)
2. Tuition rate of $800 or less per credit hour
3. Accredited by the Association to Advance Collegiate Schools of Business (AACSB)
4. Offered by a regionally accredited institution
5. Require 36 credit hours or fewer for degree completion
6. Offer rolling admissions or have at least 3 start dates per year

For each qualifying program, provide the institution name, its U.S. News ranking position, the specific tuition rate per credit hour, the number of credit hours required, and reference URLs documenting each requirement.
"""

# Recognized regional accreditors (guidance for verification)
RECOGNIZED_REGIONALS = [
    "Higher Learning Commission (HLC)",
    "Middle States Commission on Higher Education (MSCHE)",
    "New England Commission of Higher Education (NECHE)",
    "Northwest Commission on Colleges and Universities (NWCCU)",
    "Southern Association of Colleges and Schools Commission on Colleges (SACSCOC)",
    "WASC Senior College and University Commission (WSCUC)",
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramSources(BaseModel):
    """URLs to support each requirement for a specific program."""
    program_overview_urls: List[str] = Field(default_factory=list)
    ranking_urls: List[str] = Field(default_factory=list)
    tuition_urls: List[str] = Field(default_factory=list)
    aacsb_urls: List[str] = Field(default_factory=list)
    regional_accreditation_urls: List[str] = Field(default_factory=list)
    credit_hours_urls: List[str] = Field(default_factory=list)
    admissions_urls: List[str] = Field(default_factory=list)


class ProgramEntry(BaseModel):
    """Extraction for one online MBA program entry provided in the answer."""
    institution_name: Optional[str] = None
    program_name: Optional[str] = None
    is_online_mba: Optional[str] = None  # Prefer strings: "yes"/"no"/"unknown"
    is_us_institution: Optional[str] = None  # "yes"/"no"/"unknown"
    us_news_2025_ranking_position: Optional[str] = None  # e.g., "42", "T-38"
    tuition_per_credit: Optional[str] = None  # e.g., "$650", "USD 780"
    credit_hours_required: Optional[str] = None  # e.g., "30", "36"
    admissions_policy: Optional[str] = None  # e.g., "rolling admissions", "3 start dates per year"
    sources: ProgramSources = Field(default_factory=ProgramSources)


class MBAProgramsExtraction(BaseModel):
    """Extraction container holding up to 5 program entries."""
    programs: List[ProgramEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_mba_programs() -> str:
    return """
    Extract up to FIVE online MBA program entries presented in the answer, preserving the order in which they appear.
    For EACH program entry, extract the following fields exactly as stated in the answer:

    1. institution_name: The U.S. institution's name offering the program.
    2. program_name: The program's name. It should be an MBA delivered online (distance/online modality).
    3. is_online_mba: Whether the answer indicates the program is an ONLINE MBA (return "yes", "no", or "unknown").
    4. is_us_institution: Whether the answer indicates the institution is U.S.-based (return "yes", "no", or "unknown").
    5. us_news_2025_ranking_position: The stated U.S. News & World Report 2025 Best Online MBA Programs ranking position (e.g., "42", "T-38"). If missing, return null.
    6. tuition_per_credit: The specific tuition rate per credit hour as shown in the answer (e.g., "$650", "USD 780"). If a range or multiple rates are shown, choose the one explicitly tied to per-credit tuition for the online MBA; otherwise return the most specific single value mentioned. If missing, return null.
    7. credit_hours_required: The number of credit hours required to complete the degree (e.g., "30", "36"). If missing, return null.
    8. admissions_policy: Text indicating "rolling admissions" or the number of start dates per year (e.g., "3 start dates per year"). If missing, return null.

    Also extract the reference URL(s) cited for each requirement into the 'sources' object (arrays of URLs):
    - program_overview_urls: General program page(s) confirming it is an online MBA offered by the institution.
    - ranking_urls: URLs that substantiate the program is included in U.S. News & World Report's 2025 Best Online MBA Programs (top 50) and the stated ranking position.
    - tuition_urls: URLs supporting the specific per-credit tuition rate.
    - aacsb_urls: URLs showing AACSB accreditation (e.g., AACSB's official directory or the school's accreditation page explicitly mentioning AACSB).
    - regional_accreditation_urls: URLs confirming the institution has regional accreditation by a recognized U.S. accrediting body (e.g., HLC, MSCHE, NECHE, NWCCU, SACSCOC, WSCUC).
    - credit_hours_urls: URLs supporting the total credit hours required for completion.
    - admissions_urls: URLs supporting rolling admissions OR at least 3 start dates per year.

    IMPORTANT URL RULES:
    - Extract only URLs explicitly mentioned in the answer (including markdown links). Do not infer or create URLs.
    - If a URL appears without a protocol, prepend "http://".
    - If a field is missing, return null; if a URL category has no sources, return an empty list.

    Return a JSON object:
    {
      "programs": [
        {
          "institution_name": "...",
          "program_name": "...",
          "is_online_mba": "yes|no|unknown",
          "is_us_institution": "yes|no|unknown",
          "us_news_2025_ranking_position": "...",
          "tuition_per_credit": "...",
          "credit_hours_required": "...",
          "admissions_policy": "...",
          "sources": {
            "program_overview_urls": [...],
            "ranking_urls": [...],
            "tuition_urls": [...],
            "aacsb_urls": [...],
            "regional_accreditation_urls": [...],
            "credit_hours_urls": [...],
            "admissions_urls": [...]
          }
        }
      ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _to_bool(s: Optional[str]) -> bool:
    """Convert a yes/no-like string to boolean."""
    if s is None:
        return False
    v = s.strip().lower()
    return v in {"yes", "true", "y", "t", "1"}


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0


# --------------------------------------------------------------------------- #
# Verification for a single program                                           #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    parent_node,
    program: ProgramEntry,
    idx: int,
) -> None:
    """
    Build verification nodes and perform checks for one program entry.
    """
    prog_id = f"Program_{idx + 1}"

    # Program node (non-critical; allows partial credit across multiple programs)
    program_node = evaluator.add_parallel(
        id=prog_id,
        desc=f"{idx + 1}st program entry in the answer (if present)." if idx == 0 else
             (f"{idx + 1}nd program entry in the answer (if present)." if idx == 1 else
              (f"{idx + 1}rd program entry in the answer (if present)." if idx == 2 else
               f"{idx + 1}th program entry in the answer (if present).")),
        parent=parent_node,
        critical=False
    )

    # --------------------- Existence/identification check ---------------------
    # Leaf: US_Institution_And_Online_MBA_Identified (custom existence check)
    us_online_exists = (
        (program.institution_name is not None and program.institution_name.strip() != "") and
        _to_bool(program.is_online_mba) and
        _to_bool(program.is_us_institution)
    )
    evaluator.add_custom_node(
        result=us_online_exists,
        id=f"{prog_id}_US_Institution_And_Online_MBA_Identified",
        desc="Entry identifies the institution name and indicates the program is an online MBA offered by a U.S. institution.",
        parent=program_node,
        critical=True
    )

    # --------------------- Reference URLs existence gate ---------------------
    # Create a critical parallel subnode to ensure each requirement has at least one URL cited
    refs_node = evaluator.add_parallel(
        id=f"{prog_id}_Reference_URLs_Document_Each_Requirement",
        desc=("Entry includes reference URL(s) that document each requirement: U.S. News ranking/top-50, "
              "tuition/credit hour, AACSB accreditation, regional accreditation, credit-hour requirement, "
              "and admissions/start-date policy."),
        parent=program_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(program.sources.ranking_urls),
        id=f"{prog_id}_refs_ranking_present",
        desc="Ranking source URL(s) provided",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(program.sources.tuition_urls),
        id=f"{prog_id}_refs_tuition_present",
        desc="Tuition source URL(s) provided",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(program.sources.aacsb_urls),
        id=f"{prog_id}_refs_aacsb_present",
        desc="AACSB accreditation source URL(s) provided",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(program.sources.regional_accreditation_urls),
        id=f"{prog_id}_refs_regional_present",
        desc="Regional accreditation source URL(s) provided",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(program.sources.credit_hours_urls),
        id=f"{prog_id}_refs_credits_present",
        desc="Credit hours requirement source URL(s) provided",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(program.sources.admissions_urls),
        id=f"{prog_id}_refs_admissions_present",
        desc="Admissions/start dates policy source URL(s) provided",
        parent=refs_node,
        critical=True
    )

    # Prepare common strings
    inst = program.institution_name or ""
    prog_name = program.program_name or ""
    rank_pos = program.us_news_2025_ranking_position or "unknown"
    tuition_val = program.tuition_per_credit or "unknown"
    credits_val = program.credit_hours_required or "unknown"

    # --------------------- US News 2025 top 50 + position --------------------
    rank_leaf = evaluator.add_leaf(
        id=f"{prog_id}_US_News_2025_Top_50_And_Ranking_Position_Provided",
        desc="Entry states a U.S. News 2025 Best Online MBA Programs ranking position and that the program is within the top 50.",
        parent=program_node,
        critical=True
    )
    rank_claim = (
        f"The online MBA program '{prog_name}' at {inst} is listed in U.S. News & World Report's 2025 Best Online MBA Programs "
        f"top 50, with a ranking position of {rank_pos}."
    )
    await evaluator.verify(
        claim=rank_claim,
        node=rank_leaf,
        sources=program.sources.ranking_urls,
        additional_instruction=(
            "Confirm that the page(s) clearly belong to U.S. News & World Report's 2025 Best Online MBA Programs list. "
            "A ranking position must be visible and the program must be within the top 50. Allow ties (e.g., 'T-38'). "
            "Reject pages for other years or other program categories."
        )
    )

    # --------------------- Tuition per credit <= $800 ------------------------
    tuition_leaf = evaluator.add_leaf(
        id=f"{prog_id}_Tuition_Per_Credit_Provided_And_Leq_800",
        desc="Entry provides a specific tuition rate per credit hour and it is ≤ $800.",
        parent=program_node,
        critical=True
    )
    tuition_claim = (
        f"The tuition rate per credit hour for the online MBA program '{prog_name}' at {inst} is {tuition_val}, "
        f"which is less than or equal to $800."
    )
    await evaluator.verify(
        claim=tuition_claim,
        node=tuition_leaf,
        sources=program.sources.tuition_urls,
        additional_instruction=(
            "Verify that the cited tuition is explicitly a 'per credit hour' rate and that its numeric value is ≤ $800. "
            "If multiple rates exist (e.g., in-state vs. out-of-state), accept the rate referenced in the answer if it is ≤ $800. "
            "Reject per-course or per-semester amounts unless they directly state an equivalent per-credit rate."
        )
    )

    # --------------------- AACSB accreditation -------------------------------
    aacsb_leaf = evaluator.add_leaf(
        id=f"{prog_id}_AACSB_Accredited",
        desc="Entry indicates the program (or business school) is AACSB-accredited.",
        parent=program_node,
        critical=True
    )
    aacsb_claim = (
        f"The business school or the online MBA program at {inst} is accredited by AACSB."
    )
    await evaluator.verify(
        claim=aacsb_claim,
        node=aacsb_leaf,
        sources=program.sources.aacsb_urls,
        additional_instruction=(
            "The evidence must explicitly indicate AACSB accreditation (e.g., AACSB's official directory entry or the school's "
            "accreditation page explicitly mentioning 'AACSB'). Do not accept other accreditations (e.g., ACBSP, IACBE) "
            "as fulfilling this requirement."
        )
    )

    # --------------------- Regional accreditation ----------------------------
    regional_leaf = evaluator.add_leaf(
        id=f"{prog_id}_Regionally_Accredited_Institution",
        desc="Entry indicates the institution is regionally accredited by a recognized U.S. accrediting body.",
        parent=program_node,
        critical=True
    )
    regional_claim = (
        f"The institution {inst} is regionally accredited by a recognized U.S. accrediting body."
    )
    await evaluator.verify(
        claim=regional_claim,
        node=regional_leaf,
        sources=program.sources.regional_accreditation_urls,
        additional_instruction=(
            "Accept only U.S. regional accreditors such as: HLC, MSCHE, NECHE, NWCCU, SACSCOC, WSCUC. "
            "The page should clearly indicate the institution holds regional accreditation from one of these bodies."
        )
    )

    # --------------------- Credit hours <= 36 --------------------------------
    credits_leaf = evaluator.add_leaf(
        id=f"{prog_id}_Credit_Hours_Provided_And_Leq_36",
        desc="Entry provides the number of credit hours required for completion and it is ≤ 36.",
        parent=program_node,
        critical=True
    )
    credits_claim = (
        f"The online MBA program '{prog_name}' at {inst} requires {credits_val} credit hours to complete, "
        f"which is less than or equal to 36."
    )
    await evaluator.verify(
        claim=credits_claim,
        node=credits_leaf,
        sources=program.sources.credit_hours_urls,
        additional_instruction=(
            "Verify that the required total credit hours are explicitly stated and that the value is ≤ 36. "
            "Accept reasonable variants (e.g., tracks with 30–36 credits) if the answer's stated value is supported."
        )
    )

    # --------------------- Admissions policy ---------------------------------
    admissions_leaf = evaluator.add_leaf(
        id=f"{prog_id}_Rolling_Admissions_Or_At_Least_3_Start_Dates",
        desc="Entry states the program offers rolling admissions OR has at least 3 start dates per year.",
        parent=program_node,
        critical=True
    )
    admissions_claim = (
        f"The online MBA program '{prog_name}' at {inst} offers rolling admissions OR has at least three start dates per year."
    )
    await evaluator.verify(
        claim=admissions_claim,
        node=admissions_leaf,
        sources=program.sources.admissions_urls,
        additional_instruction=(
            "Confirm the page indicates rolling admissions or clearly lists three or more distinct start dates/intakes per year. "
            "Start windows spanning multiple months count as a single intake unless explicitly separated."
        )
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the online MBA qualification task and return the structured summary.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Programs evaluated independently
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

    # Root node reflecting rubric root; set non-critical to allow partial credit across programs.
    qualified_root = evaluator.add_parallel(
        id="Qualified_Online_MBA_Programs",
        desc=("Evaluate up to 5 provided U.S.-based online MBA program entries for compliance with all stated "
              "constraints and required reported fields/citations."),
        parent=root,
        critical=False
    )

    # Extract up to 5 program entries from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_mba_programs(),
        template_class=MBAProgramsExtraction,
        extraction_name="extracted_programs"
    )

    # Limit to first 5 items; pad if fewer
    programs = list(extracted.programs[:5])
    while len(programs) < 5:
        programs.append(ProgramEntry())

    # Add helpful guidance as custom info
    evaluator.add_custom_info(
        info={"recognized_regional_accreditors": RECOGNIZED_REGIONALS},
        info_type="guidance",
        info_name="regional_accreditors_reference"
    )

    # Verify each program
    for idx, program in enumerate(programs):
        await verify_program(evaluator, qualified_root, program, idx)

    # Return standard summary
    return evaluator.get_summary()