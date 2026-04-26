import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "wayne_scech_renewal_plan"
TASK_DESCRIPTION = (
    "An elementary school teacher in Wayne County, Michigan needs to renew their Standard Teaching Certificate, "
    "which requires completing 150 State Continuing Education Clock Hours (SCECHs) of professional development "
    "appropriate to their certification level and content area.\n\n"
    "Your task is to:\n"
    "1. Identify the Intermediate School District (ISD) that serves Wayne County, Michigan\n"
    "2. Verify that this ISD is an approved SCECH sponsor by the Michigan Department of Education\n"
    "3. Provide the official website URL for this ISD\n"
    "4. From this ISD's SCECH-approved professional development offerings, identify two specific programs that together provide at least 150 SCECHs total\n"
    "5. Ensure that at least one of the selected programs includes a focus on reading instruction or literacy (as required for elementary teacher renewal in Michigan)\n"
    "6. Ensure both programs are appropriate for elementary education teachers\n\n"
    "For each of the two programs you identify, provide:\n"
    "- The complete program name or title\n"
    "- The number of SCECH hours the program offers\n"
    "- A valid reference URL where the program information can be verified"
)

# ------------------------------
# Data models for extraction
# ------------------------------
class WayneISDInfo(BaseModel):
    isd_name: Optional[str] = None
    official_name: Optional[str] = None
    isd_url: Optional[str] = None
    isd_sources: List[str] = Field(default_factory=list)
    sponsor_verification_urls: List[str] = Field(default_factory=list)


class ProgramInfo(BaseModel):
    title: Optional[str] = None
    scech_hours: Optional[str] = None
    url: Optional[str] = None


class ProgramsExtraction(BaseModel):
    programs: List[ProgramInfo] = Field(default_factory=list)


class PDPlanExtraction(BaseModel):
    isd: Optional[WayneISDInfo] = None
    programs: List[ProgramInfo] = Field(default_factory=list)


# ------------------------------
# Extraction prompts
# ------------------------------
def prompt_extract_isd_and_programs() -> str:
    return (
        "Extract the following fields from the answer:\n"
        "A) ISD Information for Wayne County, Michigan\n"
        "- isd.isd_name: The ISD name as given (e.g., 'Wayne RESA')\n"
        "- isd.official_name: The full official name (e.g., 'Wayne Regional Educational Service Agency')\n"
        "- isd.isd_url: The official website URL of the ISD\n"
        "- isd.isd_sources: All URLs cited in the answer that support the ISD serving Wayne County\n"
        "- isd.sponsor_verification_urls: All URLs cited in the answer that support the ISD being an MDE-approved SCECH sponsor\n"
        "\n"
        "B) Professional Development Programs (Extract in the order they appear; include at least two if available)\n"
        "For each program, extract:\n"
        "- title: Complete program name or title\n"
        "- scech_hours: The stated SCECH hours offered (string, exactly as written, e.g., '75 SCECHs')\n"
        "- url: The reference URL where the program can be verified\n"
        "\n"
        "Return JSON with fields: 'isd' and 'programs' (array of program objects). If any field is missing, set it to null. "
        "Extract URLs exactly as presented (plain or markdown). Do not invent URLs."
    )


# ------------------------------
# Helpers
# ------------------------------
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if isinstance(u, str) and u.strip():
            # normalize minimal: ensure protocol if missing
            if not re.match(r'^[a-zA-Z]+://', u.strip()):
                cleaned.append(f"http://{u.strip()}")
            else:
                cleaned.append(u.strip())
    return cleaned


def _parse_hours_to_number(text: Optional[str]) -> float:
    """
    Best-effort parser: extract numeric hours from a free-form string.
    Strategy:
    - Prefer numbers near tokens (scech|hour|hours|clock) within ±30 chars.
    - Fallback to max numeric in the string.
    - Clamp to reasonable range (0..500). If none found, return 0.0.
    """
    if not text:
        return 0.0
    s = text.lower()
    tokens = ["scech", "hour", "hours", "clock"]
    positions = []
    for t in tokens:
        for m in re.finditer(re.escape(t), s):
            positions.append(m.start())

    def nums_in_window(pos: int, win: int = 30) -> List[float]:
        start = max(0, pos - win)
        end = min(len(s), pos + win)
        window = s[start:end]
        return [float(x) for x in re.findall(r'\d+(?:\.\d+)?', window)]

    near_token_nums: List[float] = []
    for p in positions:
        near_token_nums.extend(nums_in_window(p))

    candidate_nums = near_token_nums if near_token_nums else [float(x) for x in re.findall(r'\d+(?:\.\d+)?', s)]
    if not candidate_nums:
        return 0.0
    hours = max(candidate_nums)
    if hours < 0 or hours > 500:
        return 0.0
    return hours


def _program_total_hours(p1: ProgramInfo, p2: ProgramInfo) -> float:
    return _parse_hours_to_number(p1.scech_hours) + _parse_hours_to_number(p2.scech_hours)


# ------------------------------
# Verification builders
# ------------------------------
async def build_isd_verification(evaluator: Evaluator, parent_node, isd: WayneISDInfo) -> None:
    isd_node = evaluator.add_parallel(
        id="Regional_ISD_Identification_and_Verification",
        desc="Correctly identify and verify the Intermediate School District serving Wayne County and confirm its status as an MDE-approved SCECH sponsor",
        parent=parent_node,
        critical=True
    )

    # Wayne RESA as Serving ISD
    node_isd_serving = evaluator.add_leaf(
        id="Wayne_RESA_as_Serving_ISD",
        desc="Correctly identify Wayne RESA (Wayne Regional Educational Service Agency) as the Intermediate School District serving Wayne County",
        parent=isd_node,
        critical=True
    )
    isd_name = isd.isd_name or ""
    official_name = isd.official_name or ""
    claim_isd_serving = (
        f"{official_name or isd_name} is the Intermediate School District (ISD) serving Wayne County, Michigan."
    )
    sources_isd = []
    if isd.isd_url:
        sources_isd.append(isd.isd_url)
    sources_isd.extend(_safe_urls(isd.isd_sources))
    await evaluator.verify(
        claim=claim_isd_serving,
        node=node_isd_serving,
        sources=sources_isd,
        additional_instruction=(
            "Verify from the cited page(s) that Wayne RESA (Wayne Regional Educational Service Agency) is the ISD for Wayne County. "
            "Accept phrasing such as 'Wayne County Regional Educational Service Agency' or equivalent language indicating ISD service."
        )
    )

    # MDE SCECH Sponsor Approval Status
    node_sponsor = evaluator.add_leaf(
        id="MDE_SCECH_Sponsor_Approval_Status",
        desc="Verify that Wayne RESA is an approved SCECH sponsor by the Michigan Department of Education",
        parent=isd_node,
        critical=True
    )
    claim_sponsor = "Wayne RESA is an approved SCECH sponsor by the Michigan Department of Education (MDE)."
    sponsor_urls = _safe_urls(isd.sponsor_verification_urls)
    await evaluator.verify(
        claim=claim_sponsor,
        node=node_sponsor,
        sources=sponsor_urls,
        additional_instruction=(
            "Prefer official MDE webpages listing approved SCECH sponsors. If the page is on Wayne RESA and clearly states MDE-approved SCECH sponsor "
            "status, that is acceptable as evidence."
        )
    )

    # Wayne RESA Official Website URL
    node_isd_url = evaluator.add_leaf(
        id="Wayne_RESA_Official_Website_URL",
        desc="Provide the official website URL for Wayne RESA",
        parent=isd_node,
        critical=True
    )
    claim_official_site = "This URL is the official website of Wayne RESA (Wayne Regional Educational Service Agency)."
    await evaluator.verify(
        claim=claim_official_site,
        node=node_isd_url,
        sources=isd.isd_url,
        additional_instruction=(
            "Confirm the site branding, organization name, and contact details clearly indicate the official Wayne RESA website "
            "(often on the domain 'resa.net')."
        )
    )


async def build_program_verification(evaluator: Evaluator, parent_node, program: ProgramInfo, idx: int) -> None:
    prog_node = evaluator.add_parallel(
        id=f"Program_{idx}_Verification",
        desc=f"Identify and verify the {'first' if idx == 1 else 'second'} SCECH-approved professional development program from Wayne RESA",
        parent=parent_node,
        critical=True
    )

    # Title verification via URL
    node_title = evaluator.add_leaf(
        id=f"Program_{idx}_Title",
        desc=f"Provide the complete program name or title for Program {idx}",
        parent=prog_node,
        critical=True
    )
    title_text = program.title or ""
    claim_title = f"The program name or title is '{title_text}'."
    await evaluator.verify(
        claim=claim_title,
        node=node_title,
        sources=program.url,
        additional_instruction="Allow minor variations (case, punctuation). The page should clearly show the program title."
    )

    # SCECH hours offered verification via URL
    node_hours = evaluator.add_leaf(
        id=f"Program_{idx}_SCECH_Hours_Offered",
        desc=f"Correctly state the number of SCECH hours offered by Program {idx}",
        parent=prog_node,
        critical=True
    )
    hours_text = program.scech_hours or ""
    claim_hours = f"This program offers {hours_text} State Continuing Education Clock Hours (SCECHs)."
    await evaluator.verify(
        claim=claim_hours,
        node=node_hours,
        sources=program.url,
        additional_instruction="Check that the page explicitly states SCECH hours. Equivalents like 'SCECH' or 'clock hours' are acceptable."
    )

    # Elementary content alignment
    node_elem = evaluator.add_leaf(
        id=f"Program_{idx}_Elementary_Content_Alignment",
        desc=f"Verify that Program {idx} is appropriate for elementary education teachers",
        parent=prog_node,
        critical=True
    )
    claim_elem = "This program is appropriate for elementary education teachers."
    await evaluator.verify(
        claim=claim_elem,
        node=node_elem,
        sources=program.url,
        additional_instruction=(
            "Look for indications such as 'elementary', 'K-5', 'K-6', 'K-8', or explicit mentions that the audience includes elementary teachers."
        )
    )

    # Reference URL validity
    node_ref = evaluator.add_leaf(
        id=f"Program_{idx}_Reference_URL",
        desc=f"Provide a valid reference URL where Program {idx} information can be verified",
        parent=prog_node,
        critical=True
    )
    claim_ref = (
        f"This URL provides the official program information page for '{title_text}', clearly describing the program details."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=node_ref,
        sources=program.url,
        additional_instruction=(
            "The page should be from Wayne RESA or a recognized partner and contain program details (title, description, and ideally SCECH information)."
        )
    )


async def build_portfolio_verification(
    evaluator: Evaluator,
    parent_node,
    program1: ProgramInfo,
    program2: ProgramInfo
) -> None:
    portfolio_node = evaluator.add_parallel(
        id="Professional_Development_Program_Portfolio",
        desc="Identify and verify professional development programs that meet Michigan's certificate renewal requirements",
        parent=parent_node,
        critical=True
    )

    # Aggregate requirements
    agg_node = evaluator.add_parallel(
        id="Portfolio_Aggregate_Requirements",
        desc="Verify that the selected portfolio of programs meets all aggregate requirements for Michigan elementary teacher certificate renewal",
        parent=portfolio_node,
        critical=True
    )

    # Combined total hours >= 150
    total_hours = _program_total_hours(program1, program2)
    node_total = evaluator.add_custom_node(
        result=(total_hours >= 150.0),
        id="Combined_Total_Meets_150_Minimum",
        desc=f"Verify that the combined SCECH hours from Program 1 and Program 2 equals or exceeds 150 hours (computed total: {total_hours})",
        parent=agg_node,
        critical=True
    )

    # Reading instruction / literacy component included (at least one program)
    node_reading = evaluator.add_leaf(
        id="Reading_Instruction_Component_Included",
        desc="Verify that at least one of the selected programs includes a reading instruction or literacy component, as required for elementary teachers in Michigan",
        parent=agg_node,
        critical=True
    )
    urls_for_reading = []
    if program1.url:
        urls_for_reading.append(program1.url)
    if program2.url:
        urls_for_reading.append(program2.url)
    claim_reading = "At least one of the selected programs includes reading instruction or literacy content."
    await evaluator.verify(
        claim=claim_reading,
        node=node_reading,
        sources=urls_for_reading,
        additional_instruction=(
            "Accept terms like 'literacy', 'reading instruction', 'phonics', 'reading comprehension', 'science of reading', etc. "
            "Verification passes if any one program page clearly includes such focus."
        )
    )

    # Program 1 verification
    await build_program_verification(evaluator, portfolio_node, program1, idx=1)
    # Program 2 verification
    await build_program_verification(evaluator, portfolio_node, program2, idx=2)


# ------------------------------
# Main evaluation entry point
# ------------------------------
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=(
            "Develop a complete professional development plan for renewing a Michigan elementary teaching certificate in Wayne County, "
            "identifying appropriate SCECH-approved programs from the regional ISD"
        ),
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract ISD + programs
    extraction = await evaluator.extract(
        prompt=prompt_extract_isd_and_programs(),
        template_class=PDPlanExtraction,
        extraction_name="pd_plan_extraction"
    )

    # Ground truth hint (non-binding)
    evaluator.add_ground_truth({
        "expected_isd_name": "Wayne RESA",
        "expected_official_name": "Wayne Regional Educational Service Agency",
        "renewal_requirement": "150 SCECHs minimum; include reading/literacy; appropriate for elementary teachers"
    })

    # Build tree: Root critical sequential
    plan_root = evaluator.add_sequential(
        id="Professional_Development_Plan_for_Certificate_Renewal",
        desc="Develop a complete professional development plan for renewing a Michigan elementary teaching certificate in Wayne County, identifying appropriate SCECH-approved programs from the regional ISD",
        parent=root,
        critical=True
    )

    # ISD verification
    isd_info = extraction.isd or WayneISDInfo()
    await build_isd_verification(evaluator, plan_root, isd_info)

    # Prepare two programs (first two only; pad if needed)
    programs = extraction.programs or []
    p1 = programs[0] if len(programs) > 0 else ProgramInfo()
    p2 = programs[1] if len(programs) > 1 else ProgramInfo()

    # Custom info summary
    evaluator.add_custom_info(
        {
            "program_1": {"title": p1.title, "hours": p1.scech_hours, "url": p1.url},
            "program_2": {"title": p2.title, "hours": p2.scech_hours, "url": p2.url},
            "computed_total_hours": _program_total_hours(p1, p2),
        },
        info_type="portfolio_summary"
    )

    # Portfolio verification
    await build_portfolio_verification(evaluator, plan_root, p1, p2)

    return evaluator.get_summary()