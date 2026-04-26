import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "affordable_online_certificates_4"
TASK_DESCRIPTION = (
    "I am looking for affordable professional development certificate programs that can help me advance my career "
    "skills through flexible online learning. Please identify four online certificate programs that meet ALL of the "
    "following requirements:\n\n"
    "1. The program must be 100% online and offered in a self-paced or asynchronous format (no required live class attendance)\n"
    "2. The program can be completed in 8 months or less from the start date\n"
    "3. The total program cost must be $2,500 or less\n"
    "4. The program must be from an institution that is either regionally accredited OR accredited by a recognized accrediting agency (such as DEAC or ACCSC)\n"
    "5. The program must be open to individuals without a bachelor's degree (only high school diploma or equivalent required)\n"
    "6. The program must be in one of these career fields: Project Management, Data Analytics, Cybersecurity, or Business Management\n"
    "7. The program must prepare students for an industry-recognized certification exam OR award credits that can be applied toward a bachelor's degree\n"
    "8. The official program webpage must clearly state or allow calculation of the completion time\n"
    "9. The official program webpage must clearly state the total program cost or provide sufficient pricing information to calculate the total cost\n\n"
    "For each program, provide:\n"
    "- The program name and institution\n"
    "- The specific field (Project Management, Data Analytics, Cybersecurity, or Business Management)\n"
    "- The stated or estimated completion time\n"
    "- The total program cost\n"
    "- The type of accreditation (regional, DEAC, ACCSC, or other recognized accreditor)\n"
    "- Whether it prepares for an industry certification (specify which one) OR awards credits toward a degree (specify how many credits)\n"
    "- A direct link to the official program page where completion time and cost information can be verified"
)

ALLOWED_FIELDS = {"project management", "data analytics", "cybersecurity", "business management"}
MAX_MONTHS = 8.0
MAX_TOTAL_COST = 2500.0

RECOGNIZED_ACCREDITORS_HINT = (
    "Recognized accreditation includes: any US regional accreditor (HLC, MSCHE, NECHE, SACSCOC, WSCUC, NWCCU), "
    "and national accreditors such as DEAC or ACCSC, or any agency recognized by the U.S. Department of Education or CHEA."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramItem(BaseModel):
    program_name: Optional[str] = None
    institution_name: Optional[str] = None
    field: Optional[str] = None  # Expect one of ALLOWED_FIELDS (case-insensitive)
    completion_time: Optional[str] = None  # e.g., "6 months", "24 weeks", "4–8 months"
    total_cost: Optional[str] = None  # e.g., "$2,400", "$2,000 + $200 fee"
    accreditation_type: Optional[str] = None  # e.g., "regional (HLC)", "DEAC", "ACCSC"
    admission_requirement: Optional[str] = None  # e.g., "High school diploma or equivalent"
    cert_exam_or_credits: Optional[str] = None  # e.g., "Prepares for CompTIA Security+" or "Awards 12 credits..."
    official_program_url: Optional[str] = None  # direct program page for time and cost verification
    additional_urls: List[str] = Field(default_factory=list)  # any other URLs cited in the answer for this program


class ProgramsExtraction(BaseModel):
    programs: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return (
        "Extract the list of certificate programs described in the answer. For each program, extract the following "
        "fields exactly as stated in the answer:\n"
        "1. program_name\n"
        "2. institution_name\n"
        "3. field (must be one of: Project Management, Data Analytics, Cybersecurity, Business Management)\n"
        "4. completion_time (as stated, e.g., '6 months', '24 weeks', '4–8 months')\n"
        "5. total_cost (as stated, e.g., '$2,400', '$2,000 + $200 fee', or a clear textual description)\n"
        "6. accreditation_type (e.g., 'regional (HLC)', 'DEAC', 'ACCSC', or other recognized accreditor)\n"
        "7. admission_requirement (what is required to enroll; should reflect that a bachelor's degree is not required if stated)\n"
        "8. cert_exam_or_credits (specify the certification exam prepared for OR credits awarded and how many)\n"
        "9. official_program_url (a direct URL to the official program page used to verify completion time and cost)\n"
        "10. additional_urls (an array of any other URLs cited in the answer relevant for accreditation, admission, or certification/credits)\n\n"
        "Important rules:\n"
        "- Only extract URLs explicitly present in the answer.\n"
        "- If any field is missing for a program, set the field to null (or an empty array for additional_urls).\n"
        "- Preserve the textual form of completion_time and total_cost; do not convert formats.\n"
        "- Return an object with a 'programs' array of program objects with the above fields."
    )


# --------------------------------------------------------------------------- #
# Helpers: parsing completion time and cost                                   #
# --------------------------------------------------------------------------- #
def _to_number(num_str: str) -> Optional[float]:
    try:
        return float(num_str.replace(",", ""))
    except Exception:
        return None


def parse_duration_to_months(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    s = raw.strip().lower()
    s = s.replace("–", "-").replace("—", "-").replace("‑", "-")
    # under/less than/up to
    m = re.search(r"(under|less than|≤|up to)\s*(\d+(?:\.\d+)?)\s*(mo|mos|month|months)\b", s)
    if m:
        val = _to_number(m.group(2))
        return val if val is not None else None

    # months range, take max
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(mo|mos|month|months)\b", s)
    if m:
        lo = _to_number(m.group(1))
        hi = _to_number(m.group(2))
        if lo is not None and hi is not None:
            return max(lo, hi)

    # single months
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mo|mos|month|months)\b", s)
    if m:
        val = _to_number(m.group(1))
        return val

    # weeks range -> months (max)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(wk|wks|week|weeks)\b", s)
    if m:
        lo = _to_number(m.group(1))
        hi = _to_number(m.group(2))
        if lo is not None and hi is not None:
            return max(lo, hi) / 4.345

    # single weeks
    m = re.search(r"(\d+(?:\.\d+)?)\s*(wk|wks|week|weeks)\b", s)
    if m:
        val = _to_number(m.group(1))
        return (val / 4.345) if val is not None else None

    # days range -> months (max)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(day|days)\b", s)
    if m:
        lo = _to_number(m.group(1))
        hi = _to_number(m.group(2))
        if lo is not None and hi is not None:
            return max(lo, hi) / 30.0

    # days
    m = re.search(r"(\d+(?:\.\d+)?)\s*(day|days)\b", s)
    if m:
        val = _to_number(m.group(1))
        return (val / 30.0) if val is not None else None

    # 'X mo' shorthand
    m = re.search(r"(\d+(?:\.\d+)?)\s*mo\b", s)
    if m:
        val = _to_number(m.group(1))
        return val

    # If only 'hours' mentioned, not safely convertible
    if re.search(r"\bhour|hours|hr|hrs\b", s):
        return None

    return None


def parse_cost_to_usd(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    s = raw.lower()

    # Capture $ amounts or 'usd/dollars' amounts
    amounts: List[float] = []
    for m in re.findall(r"\$[\s]*([\d,]+(?:\.\d+)?)", s):
        val = _to_number(m)
        if val is not None:
            amounts.append(val)
    for m in re.findall(r"([\d,]+(?:\.\d+)?)\s*(?:usd|us\$|dollars?)", s):
        val = _to_number(m)
        if val is not None:
            amounts.append(val)

    if not amounts:
        return None

    # Heuristics:
    # - If contains "plus" or '+' with multiple amounts -> sum
    if ("+" in s or " plus " in s or " + " in s) and len(amounts) >= 2:
        return sum(amounts)

    # - If contains explicit range (e.g., "$2000-$2500"): take upper bound
    if any(sep in s for sep in ["-", "–", " to "]):
        return max(amounts)

    # - If mentions "per credit/course" and only one amount -> cannot compute safely
    if len(amounts) == 1 and any(kw in s for kw in ["per credit", "per-course", "per course", "per module", "per class"]):
        return None

    # Otherwise, if multiple amounts but not explicit plus, take max to be conservative
    if len(amounts) >= 2:
        return max(amounts)

    # Single amount
    return amounts[0]


def normalize_field_name(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().lower()
    # normalize common variants
    mapping = {
        "project management": "project management",
        "pm": "project management",
        "data analytics": "data analytics",
        "data analysis": "data analytics",
        "cybersecurity": "cybersecurity",
        "cyber security": "cybersecurity",
        "business management": "business management",
        "business admin": "business management",
        "business administration": "business management",
    }
    return mapping.get(s, s)


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _url_or_none(u: Optional[str]) -> Optional[str]:
    if not _nonempty(u):
        return None
    return u.strip()


def _merge_urls(primary: Optional[str], extras: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    if _url_or_none(primary):
        out.append(primary.strip())
    if extras:
        for e in extras:
            if _nonempty(e):
                out.append(e.strip())
    # deduplicate preserving order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


# --------------------------------------------------------------------------- #
# Build and verify per-program                                                #
# --------------------------------------------------------------------------- #
async def verify_one_program(
    evaluator: Evaluator,
    parent_node,
    program: ProgramItem,
    program_index: int,
) -> None:
    """
    Build the verification subtree for a single program and launch verifications.
    program_index is 0-based; user-facing numbering is index+1.
    """
    n = program_index + 1
    prog_node = evaluator.add_parallel(
        id=f"program_{n}",
        desc=f"Program {n} satisfies all constraints and required reporting fields.",
        parent=parent_node,
        critical=False,
    )

    # 1) Name and institution provided (critical, existence)
    name_inst_ok = _nonempty(program.program_name) and _nonempty(program.institution_name)
    evaluator.add_custom_node(
        result=name_inst_ok,
        id=f"p{n}_name_and_institution_provided",
        desc="Provides the program name and the institution name.",
        parent=prog_node,
        critical=True,
    )

    # 2) Field in allowed set and stated (critical, custom)
    normalized_field = normalize_field_name(program.field)
    field_ok = _nonempty(normalized_field) and (normalized_field in ALLOWED_FIELDS)
    evaluator.add_custom_node(
        result=bool(field_ok),
        id=f"p{n}_field_in_allowed_set_and_stated",
        desc="States the program field and it is one of: Project Management, Data Analytics, Cybersecurity, Business Management.",
        parent=prog_node,
        critical=True,
    )

    # 3) Official program URL provided (critical, existence)
    official_url_ok = _nonempty(program.official_program_url)
    evaluator.add_custom_node(
        result=official_url_ok,
        id=f"p{n}_official_program_url_provided",
        desc="Provides a direct link to the official program webpage used to verify completion time and cost.",
        parent=prog_node,
        critical=True,
    )

    # 4) Completion time <= 8 months (critical, custom)
    months = parse_duration_to_months(program.completion_time)
    months_ok = (months is not None) and (months <= MAX_MONTHS)
    evaluator.add_custom_node(
        result=months_ok,
        id=f"p{n}_completion_time_le_8_months",
        desc="Program completion time is 8 months or less.",
        parent=prog_node,
        critical=True,
    )

    # 5) Total cost <= $2,500 (critical, custom)
    cost_val = parse_cost_to_usd(program.total_cost)
    cost_ok = (cost_val is not None) and (cost_val <= MAX_TOTAL_COST)
    evaluator.add_custom_node(
        result=cost_ok,
        id=f"p{n}_total_cost_le_2500",
        desc="Total program cost is $2,500 or less.",
        parent=prog_node,
        critical=True,
    )

    # Prepare verification leaves that require webpages
    urls_all = _merge_urls(program.official_program_url, program.additional_urls)
    official_only = _merge_urls(program.official_program_url, [])

    # 6) Delivery: 100% online and self-paced/asynchronous (critical, verify with URLs)
    node_delivery = evaluator.add_leaf(
        id=f"p{n}_delivery_100pct_online_and_async",
        desc="Program is 100% online and self-paced/asynchronous with no required live attendance.",
        parent=prog_node,
        critical=True,
    )
    claim_delivery = (
        "The official information indicates the program is fully online (100% online) and offered in a self-paced or "
        "asynchronous format with no required live class attendance."
    )
    add_ins_delivery = (
        "Look for phrases like '100% online', 'fully online', 'asynchronous', 'self-paced', 'on-demand', "
        "'no required live sessions', or similar. Optional live/virtual sessions are acceptable, "
        "but required live meetings would not meet the requirement."
    )

    # 7) Completion time info verifiable on official page (critical, verify with official URL only)
    node_time_verif = evaluator.add_leaf(
        id=f"p{n}_completion_time_info_verifiable_on_official_page",
        desc="Official program webpage clearly states or allows calculation of completion time.",
        parent=prog_node,
        critical=True,
    )
    completion_time_text = program.completion_time or "(not provided)"
    claim_time = (
        f"The official program page clearly states (or provides enough information to calculate) the program's completion time, "
        f"which the answer summarized as: {completion_time_text}."
    )
    add_ins_time = (
        "Accept explicit statements like 'complete in X months/weeks' or ranges like '4–6 months'. "
        "Also accept sufficient modular details (e.g., number of courses with typical duration) that allow calculation "
        "of a total completion time. If this is not clearly stated or cannot be reasonably derived from the page, "
        "the claim is not supported."
    )

    # 8) Cost info verifiable on official page (critical, verify with official URL only)
    node_cost_verif = evaluator.add_leaf(
        id=f"p{n}_cost_info_verifiable_on_official_page",
        desc="Official program webpage clearly states or allows calculation of total program cost.",
        parent=prog_node,
        critical=True,
    )
    total_cost_text = program.total_cost or "(not provided)"
    claim_cost = (
        f"The official program page clearly states (or provides enough information to calculate) the total program cost, "
        f"which the answer summarized as: {total_cost_text}."
    )
    add_ins_cost = (
        "Accept explicit 'total cost/tuition' statements or a clear fee/tuition breakdown that enables computing a total. "
        "If only 'per credit' or 'per course' pricing is shown without sufficient quantity information to compute a total, "
        "this should be considered not clearly stated or calculable."
    )

    # 9) Accreditation recognized and type stated (critical, verify with URLs)
    node_accred = evaluator.add_leaf(
        id=f"p{n}_accreditation_recognized_and_type_stated",
        desc="Institution is accredited (regional or other recognized accreditor such as DEAC/ACCSC/USDE-recognized) and the accreditation type is stated.",
        parent=prog_node,
        critical=True,
    )
    accred_text = program.accreditation_type or "(not provided)"
    inst_text = program.institution_name or "(institution not provided)"
    claim_accred = (
        f"The institution '{inst_text}' holds recognized accreditation, specifically: {accred_text}. "
        f"Recognized accreditation includes regional accreditors or nationally recognized agencies such as DEAC or ACCSC."
    )
    add_ins_accred = (
        f"{RECOGNIZED_ACCREDITORS_HINT} "
        "The verification can be from the official institution site (e.g., accreditation page) or official accreditor listings. "
        "The page(s) must clearly indicate the accreditation and its type."
    )

    # 10) Open to individuals without a bachelor's degree (critical, verify with URLs)
    node_no_bach = evaluator.add_leaf(
        id=f"p{n}_no_bachelors_required",
        desc="Open to individuals without a bachelor's degree (high school diploma or equivalent only).",
        parent=prog_node,
        critical=True,
    )
    claim_no_bach = (
        "Enrollment does not require a bachelor's degree; a high school diploma or equivalent (e.g., GED) is sufficient for this program."
    )
    add_ins_no_bach = (
        "Check admissions/eligibility/prerequisites on the program or institution site. "
        "If it explicitly requires a bachelor's degree, fail. If it states HS diploma or equivalent is sufficient, pass. "
        "If unclear, treat as not supported."
    )

    # 11) Cert exam prep OR degree credits specified (critical, verify with URLs)
    node_cert_or_credit = evaluator.add_leaf(
        id=f"p{n}_cert_exam_prep_or_degree_credits_and_specified",
        desc="Either (a) prepares for an industry-recognized certification exam (specified) OR (b) awards credits applicable toward a bachelor's degree (credits/how specified).",
        parent=prog_node,
        critical=True,
    )
    cert_or_credit_text = program.cert_exam_or_credits or "(not provided)"
    claim_cert_or_credit = (
        f"The program either prepares students for a named industry-recognized certification exam OR awards credits applicable "
        f"toward a bachelor's degree. The answer states: {cert_or_credit_text}. At least one of these two must be explicitly supported."
    )
    add_ins_cert_or_credit = (
        "Look for explicit mentions like 'prepares for CompTIA Security+', 'PMI CAPM exam', 'eligible for X credits applicable toward a bachelor's degree', etc. "
        "At least one of the two conditions must be clearly indicated."
    )

    # Prepare batch verifications
    claims_and_sources: List[Tuple[str, List[str] | str | None, Any, Optional[str]]] = [
        (claim_delivery, urls_all if urls_all else None, node_delivery, add_ins_delivery),
        (claim_time, official_only if official_only else None, node_time_verif, add_ins_time),
        (claim_cost, official_only if official_only else None, node_cost_verif, add_ins_cost),
        (claim_accred, urls_all if urls_all else None, node_accred, add_ins_accred),
        (claim_no_bach, urls_all if urls_all else None, node_no_bach, add_ins_no_bach),
        (claim_cert_or_credit, urls_all if urls_all else None, node_cert_or_credit, add_ins_cert_or_credit),
    ]

    # Run all verifications for this program in parallel
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the affordable online certificates task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root parallel per rubric
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

    # Extract program items from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction",
    )

    # Keep exactly 4 programs (pad with empty if fewer; truncate if more)
    programs: List[ProgramItem] = list(extraction.programs or [])
    if len(programs) < 4:
        programs.extend([ProgramItem() for _ in range(4 - len(programs))])
    else:
        programs = programs[:4]

    # Add some custom info for transparency
    evaluator.add_custom_info(
        {
            "allowed_fields": sorted(list(ALLOWED_FIELDS)),
            "max_months": MAX_MONTHS,
            "max_total_cost_usd": MAX_TOTAL_COST,
        },
        info_type="constraints_summary",
        info_name="constraints_summary",
    )

    # Build and verify each program subtree
    tasks = []
    for idx in range(4):
        tasks.append(verify_one_program(evaluator, root, programs[idx], idx))
    await asyncio.gather(*tasks)

    return evaluator.get_summary()