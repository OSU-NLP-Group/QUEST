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
TASK_ID = "certificate_program_search"
TASK_DESCRIPTION = """
I am interested in pursuing a professional certificate in data analytics or business analytics to advance my career. I need to find four different online certificate programs that meet the following criteria:

1. Accreditation: The program must be offered by an institution that holds regional accreditation (from HLC, MSCHE, SACSCOC, NEASC, WSCUC, or NWCCU) or DEAC accreditation.

2. Delivery Format: The program must be offered 100% online.

3. Duration: The program must be completable within 12 months or less.

4. Course Structure: The program must consist of 3 to 6 courses, or require 12 to 18 credit hours for completion.

5. Subject Area: The program must specifically focus on data analytics, business analytics, or a directly related quantitative field.

6. Cost Transparency: The program must have publicly available cost information (total program cost, per-course cost, or per-credit-hour cost).

For each of the four programs, please provide:
- The program name and institution name
- The institution's accreditation type and a reference URL verifying the accreditation
- The official completion time as stated on the program webpage
- The number of courses or credit hours required
- The cost information as listed on the institution's website
- The direct URL to the program's official webpage

Each program must be from a different institution (no duplicate institutions).
"""

# Allowed accreditors (case-insensitive matching, with common synonyms)
ALLOWED_ACCREDITORS = {
    "hlc", "higher learning commission",
    "msche", "middle states commission on higher education",
    "sacs", "sacscoc", "southern association of colleges and schools commission on colleges",
    "neasc", "neche", "new england commission of higher education",
    "wscuc", "wasc senior college and university commission",
    "nwccu", "northwest commission on colleges and universities",
    "deac", "distance education accrediting commission",
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProgramItem(BaseModel):
    program_name: Optional[str] = None
    institution_name: Optional[str] = None
    program_url: Optional[str] = None

    accreditation_type: Optional[str] = None
    accreditation_verification_url: Optional[str] = None

    completion_time: Optional[str] = None  # e.g., "8 months", "2 semesters", "within 1 year"

    courses_count: Optional[str] = None    # e.g., "4 courses"
    credit_hours: Optional[str] = None     # e.g., "15 credits"

    subject_focus: Optional[str] = None    # short description or keywords from the page

    cost_info_text: Optional[str] = None   # extracted cost text/snippet from official site
    cost_urls: List[str] = Field(default_factory=list)  # additional official cost pages


class ProgramsExtraction(BaseModel):
    programs: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    Extract up to four certificate programs mentioned in the answer that relate to data analytics or business analytics.
    For each program, extract the following fields exactly as presented in the answer:
    - program_name: The official program name.
    - institution_name: The institution offering the program.
    - program_url: A direct URL to the program’s official webpage on the institution's domain (prefer .edu or official subdomains; avoid third-party aggregators if a direct official URL is present in the answer).
    - accreditation_type: The institution’s accreditation type (e.g., HLC, MSCHE, SACSCOC, NEASC/NECHE, WSCUC, NWCCU, DEAC). Use whatever label is in the answer (exact text).
    - accreditation_verification_url: A URL to the accrediting body directory entry or the institution’s accreditation page that verifies the accreditation.
    - completion_time: The official completion time text (e.g., "8 months", "two semesters", "within one year") as stated in the answer for this program.
    - courses_count: The number of required courses if stated (e.g., "4 courses"). If absent, return null.
    - credit_hours: The number of required credits if stated (e.g., "12 credits"). If absent, return null.
    - subject_focus: A brief phrase capturing the program’s focus (e.g., "business analytics", "data analytics").
    - cost_info_text: The cost information text (e.g., "Tuition is $X per credit" or "Total program cost is $Y") as provided in the answer from the official site.
    - cost_urls: Any official institution URLs (tuition/fees pages) mentioned that provide cost details for this program.

    Return a JSON object with a field 'programs' that is an array of up to 4 such objects.
    If the answer contains more than 4 programs, include only the first 4.
    If any field is missing for a program, set it to null (or empty list for 'cost_urls').
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_institution_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return " ".join(name.strip().lower().split())


def _is_accreditation_type_eligible(acc_type: Optional[str]) -> bool:
    if not acc_type:
        return False
    text = acc_type.strip().lower()
    # Accept if any allowed token appears in text
    return any(token in text for token in ALLOWED_ACCREDITORS)


def _combine_sources(*args: Optional[List[str] | str]) -> List[str]:
    urls: List[str] = []
    for a in args:
        if isinstance(a, list):
            urls.extend([u for u in a if isinstance(u, str) and u.strip()])
        elif isinstance(a, str) and a.strip():
            urls.append(a)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    parent_node,
    program: ProgramItem,
    idx: int,
) -> None:
    """
    Build verification subtree and run checks for a single program (idx: 1..4).
    """
    # Create the program node (non-critical; parallel aggregation for partial credit)
    pg_node = evaluator.add_parallel(
        id=f"Program_{idx}",
        desc=f"Evaluate program #{idx} against all constraints and required output fields.",
        parent=parent_node,
        critical=False
    )

    # Existence: name & institution (critical)
    name_inst_ok = bool(program.program_name and program.program_name.strip()) and \
                   bool(program.institution_name and program.institution_name.strip())
    evaluator.add_custom_node(
        result=name_inst_ok,
        id=f"P{idx}_Name_and_Institution",
        desc="Provides program name and institution name.",
        parent=pg_node,
        critical=True
    )

    # Existence: program URL provided (used to gate multiple verifications)
    url_ok = bool(program.program_url and program.program_url.strip())
    url_exist_node = evaluator.add_custom_node(
        result=url_ok,
        id=f"P{idx}_URL_Provided",
        desc="Program official webpage URL is provided.",
        parent=pg_node,
        critical=True
    )

    # Certificate vs Degree (critical leaf)
    cert_leaf = evaluator.add_leaf(
        id=f"P{idx}_Certificate_Not_Degree",
        desc="Program is a professional certificate or graduate certificate (not a full degree program).",
        parent=pg_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The '{program.program_name or 'program'}' is a certificate program (e.g., professional certificate or graduate certificate) and not a degree program.",
        node=cert_leaf,
        sources=program.program_url,
        additional_instruction="Check the page for a 'certificate' designation. If it states 'Bachelor', 'Master', 'MBA', 'MS', 'MA', or otherwise indicates a degree program rather than a certificate, the claim is incorrect."
    )

    # Official program webpage (critical leaf)
    official_leaf = evaluator.add_leaf(
        id=f"P{idx}_Official_Program_Webpage_URL",
        desc="Provides a direct URL to the program’s official webpage (institution-controlled domain).",
        parent=pg_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This URL is the institution's official webpage for the program '{program.program_name or ''}' at '{program.institution_name or ''}', hosted on an institution-controlled domain (e.g., .edu or official subdomain), not a third-party aggregator or marketing site.",
        node=official_leaf,
        sources=program.program_url,
        additional_instruction="Confirm the domain belongs to the institution (official .edu or institution subdomain). If the page is a third-party site (e.g., Coursera, edX, random marketing/aggregator), mark as incorrect."
    )

    # Accreditation checks
    # Existence: accreditation type string provided (critical to gate verification)
    acc_type_provided = bool(program.accreditation_type and program.accreditation_type.strip())
    evaluator.add_custom_node(
        result=acc_type_provided,
        id=f"P{idx}_Accreditation_Type_Provided",
        desc="Accreditation type string is provided.",
        parent=pg_node,
        critical=True
    )
    # Eligibility: accreditation type is among allowed (critical)
    evaluator.add_custom_node(
        result=_is_accreditation_type_eligible(program.accreditation_type),
        id=f"P{idx}_Accreditation_Type_Eligible",
        desc="Accreditation type is eligible (regional or DEAC).",
        parent=pg_node,
        critical=True
    )
    # Existence: accreditation verification URL (critical)
    acc_url_ok = bool(program.accreditation_verification_url and program.accreditation_verification_url.strip())
    evaluator.add_custom_node(
        result=acc_url_ok,
        id=f"P{idx}_Accreditation_Verification_URL_Provided",
        desc="Accreditation verification URL is provided.",
        parent=pg_node,
        critical=True
    )
    # Verify accreditation via URL (critical leaf)
    acc_leaf = evaluator.add_leaf(
        id=f"P{idx}_Accreditation_Eligible_Type_and_Verification_URL",
        desc="Institution holds eligible accreditation and provides a verification URL.",
        parent=pg_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The institution '{program.institution_name or 'the institution'}' is accredited by {program.accreditation_type or 'an eligible accreditor'}.",
        node=acc_leaf,
        sources=program.accreditation_verification_url,
        additional_instruction="Use the accrediting body directory entry or the institution’s accreditation page to confirm. Accept NEASC/NECHE equivalently. Eligible accreditors: HLC, MSCHE, SACSCOC, NEASC/NECHE, WSCUC, NWCCU, DEAC."
    )

    # Online-only (critical leaf)
    online_leaf = evaluator.add_leaf(
        id=f"P{idx}_Online_Only",
        desc="Program is offered 100% online with no in-person attendance requirements (supported by official program/institution page).",
        parent=pg_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program is offered fully online (100% online) and does not require any in-person campus attendance.",
        node=online_leaf,
        sources=program.program_url,
        additional_instruction="Look for phrases like '100% online', 'fully online', 'no campus visits required'. If the page indicates hybrid, on-campus, or optional but required in-person elements, mark as incorrect."
    )

    # Duration <= 12 months (critical leaf)
    duration_leaf = evaluator.add_leaf(
        id=f"P{idx}_Duration_Leq_12_Months",
        desc="Official completion time is stated and is 12 months or less (supported by official program page).",
        parent=pg_node,
        critical=True
    )
    duration_text = program.completion_time or ""
    await evaluator.verify(
        claim=f"Based on the official program page, the program can be completed within 12 months or less. Stated completion time text: '{duration_text}'.",
        node=duration_leaf,
        sources=program.program_url,
        additional_instruction="Interpret durations like 'two semesters', 'three quarters', '8–12 months', 'within one year' as needed. If any stated typical/official timeline exceeds 12 months, mark as incorrect."
    )

    # Course structure range: 3–6 courses OR 12–18 credits (critical leaf)
    course_struct_leaf = evaluator.add_leaf(
        id=f"P{idx}_Course_Structure_Range",
        desc="Program consists of 3–6 courses OR requires 12–18 credit hours (supported by official program page).",
        parent=pg_node,
        critical=True
    )
    courses_text = program.courses_count or ""
    credits_text = program.credit_hours or ""
    await evaluator.verify(
        claim=f"The program's official requirements fall within 3–6 courses OR 12–18 credits. Extracted (if any): courses='{courses_text}', credits='{credits_text}'.",
        node=course_struct_leaf,
        sources=program.program_url,
        additional_instruction="Check the program page for the number of courses and/or total credits. Treat synonyms like 'units' or 'hours' appropriately. The requirement passes if either count is within the ranges."
    )

    # Subject area fit (critical leaf)
    subject_leaf = evaluator.add_leaf(
        id=f"P{idx}_Subject_Area_Fit",
        desc="Program specifically focuses on data analytics, business analytics, or a directly related quantitative field (supported by official program description).",
        parent=pg_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program specifically focuses on data analytics or business analytics (or a directly related quantitative field such as applied analytics).",
        node=subject_leaf,
        sources=program.program_url,
        additional_instruction="Use the program description, learning outcomes, or curriculum overview to confirm that the primary focus aligns with data/business analytics rather than unrelated fields."
    )

    # Cost transparency (critical leaf)
    cost_leaf = evaluator.add_leaf(
        id=f"P{idx}_Cost_Transparency",
        desc="Provides publicly available cost information (total/per-course/per-credit) accessible on the institution’s official website.",
        parent=pg_node,
        critical=True
    )
    cost_sources = _combine_sources(program.program_url, program.cost_urls)
    cost_snippet = program.cost_info_text or ""
    await evaluator.verify(
        claim=f"The institution's official website provides publicly available cost information for this program (e.g., total cost, per-course, or per-credit). Example text (if provided): '{cost_snippet}'.",
        node=cost_leaf,
        sources=cost_sources,
        additional_instruction="Verify on official institution pages (program page or tuition/fees pages) that cost details are present. Third-party aggregators do not count."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate the provided answer for the certificate program search task.
    """
    # Initialize evaluator (root is parallel to allow partial credit across four programs)
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

    # Extract programs
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction",
    )

    # Keep exactly the first 4 programs; pad with empty items if fewer are provided
    programs: List[ProgramItem] = list(extracted.programs[:4])
    while len(programs) < 4:
        programs.append(ProgramItem())

    # Global constraint: different institutions (critical leaf via custom node)
    inst_names_norm = [
        _normalize_institution_name(p.institution_name) for p in programs
    ]
    inst_unique_ok = all(n is not None and n != "" for n in inst_names_norm) and \
                     (len(set(inst_names_norm)) == 4)
    evaluator.add_custom_node(
        result=inst_unique_ok,
        id="Global_Different_Institutions",
        desc="All four programs are from different institutions (no duplicate institutions).",
        parent=root,
        critical=True
    )

    # Build verification subtree for each program
    for i, program in enumerate(programs, start=1):
        await verify_program(evaluator, root, program, i)

    # Return summary
    return evaluator.get_summary()