import asyncio
import logging
import re
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_cc_adt_4"
TASK_DESCRIPTION = """
Identify four California community colleges that meet all of the following criteria:

1. Location: The college must be located in Los Angeles County, Orange County, or San Diego County, California.

2. Accreditation: The college must be accredited by the Accrediting Commission for Community and Junior Colleges (ACCJC).

3. UC TAG Participation: The college must participate in the University of California Transfer Admission Guarantee (TAG) program.

4. ADT Program Offerings: The college must offer Associate Degree for Transfer (ADT) programs in at least three of the following four fields:
   - Biology
   - Business Administration
   - Computer Science
   - Psychology

For each of the four colleges you identify, provide:
- The college's official name
- The complete mailing address (street address, city, state, and ZIP code)
- The official website URL
- The community college district to which it belongs
- A list of which ADT programs from the specified four fields (Biology, Business Administration, Computer Science, Psychology) are offered at that college
- A reference URL that verifies the ADT program offerings for that college
"""

ALLOWED_ADT_FIELDS = [
    "Biology",
    "Business Administration",
    "Computer Science",
    "Psychology",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CollegeItem(BaseModel):
    name: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    district: Optional[str] = None

    # If the answer explicitly mentions a county (e.g., "Los Angeles County")
    county: Optional[str] = None

    # Only include items from the four requested fields that the answer explicitly claims as ADT
    adt_programs: List[str] = Field(default_factory=list)

    # URL(s) provided in the answer that verify ADT offerings for this college
    adt_reference_urls: List[str] = Field(default_factory=list)

    # Optional: URLs in the answer that support accreditation or UC TAG participation
    accreditation_reference_urls: List[str] = Field(default_factory=list)
    tag_reference_urls: List[str] = Field(default_factory=list)

    # Optional: a URL in the answer that shows the address (e.g., contact page)
    address_reference_url: Optional[str] = None


class CollegesExtraction(BaseModel):
    colleges: List[CollegeItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_colleges() -> str:
    return """
    Extract information about community colleges listed in the answer. Return an array 'colleges' where each element contains:
    - name: the official name of the college (string)
    - website: the official college website URL (string or null)
    - address: the full mailing address including street address, city, state (CA), and ZIP code (string or null)
    - district: the community college district name as stated (string or null)
    - county: the county name if it is explicitly mentioned in the answer text (e.g., "Los Angeles County", "Orange County", or "San Diego County"); otherwise null
    - adt_programs: a list of ADT programs from ONLY these four fields that the answer explicitly claims the college offers: ["Biology", "Business Administration", "Computer Science", "Psychology"].
                     Include each field name exactly as one of those four if and only if the answer explicitly claims an ADT for that field.
    - adt_reference_urls: a list of URLs that the answer cites specifically to verify the ADT offerings for this college
    - accreditation_reference_urls: a list of URLs (if any) in the answer that support the claim that the college is accredited by ACCJC
    - tag_reference_urls: a list of URLs (if any) in the answer that support the claim that the college participates in the UC TAG program
    - address_reference_url: a URL (if any) in the answer that shows the mailing address of the college (e.g., contact page). If none is present, return null.

    RULES:
    - Do NOT invent information. Only extract what is explicitly present in the answer.
    - For any missing item, set it to null (or [] for lists).
    - For URLs, extract the actual URL(s) exactly as shown in the answer. If a URL lacks protocol, prepend http://
    - Return all colleges present in the answer; do not filter. The evaluator will select the first four later.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"][n - 1] if 1 <= n <= 6 else f"#{n}"


def normalize_adt_fields(raw_fields: List[str]) -> List[str]:
    """Normalize ADT field names to the allowed canonical values."""
    canon_map = {
        "biology": "Biology",
        "business administration": "Business Administration",
        "computer science": "Computer Science",
        "psychology": "Psychology",
    }
    normalized: List[str] = []
    for item in raw_fields or []:
        s = (item or "").strip().lower()
        # Light normalization for common shorthand
        if s in canon_map:
            normalized.append(canon_map[s])
            continue
        if s in {"business", "business admin"}:
            normalized.append("Business Administration")
            continue
        if s in {"cs"}:
            normalized.append("Computer Science")
            continue
        if s in {"psych", "psychological science"}:
            normalized.append("Psychology")
            continue
        # As-is if it exactly matches (case-insensitive) any allowed label
        for allowed in ALLOWED_ADT_FIELDS:
            if s == allowed.lower():
                normalized.append(allowed)
                break
    # Deduplicate preserving order
    seen = set()
    result = []
    for v in normalized:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")


def is_complete_address(addr: Optional[str]) -> bool:
    """Heuristic check for a complete US mailing address string within CA."""
    if not addr:
        return False
    has_zip = bool(ZIP_RE.search(addr))
    has_ca = (" CA " in f" {addr} ") or ("California" in addr)
    has_number = bool(re.search(r"\b\d{1,6}\b", addr))
    has_city_comma = "," in addr
    # Require key elements: CA + ZIP + a number; comma is a soft signal
    return has_ca and has_zip and has_number and has_city_comma


def collect_sources(*args: Optional[List[str] | str]) -> List[str]:
    """Collect non-empty string(s) and flatten into a list."""
    urls: List[str] = []
    for item in args:
        if not item:
            continue
        if isinstance(item, list):
            urls.extend([u for u in item if isinstance(u, str) and u.strip()])
        elif isinstance(item, str) and item.strip():
            urls.append(item)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification for one college                                                #
# --------------------------------------------------------------------------- #
async def verify_one_college(
    evaluator: Evaluator,
    parent_node,
    college: CollegeItem,
    idx: int,
) -> None:
    """
    Build the verification subtree for one college following the rubric.
    """
    label = ordinal(idx + 1)
    college_node = evaluator.add_parallel(
        id=f"college_{idx + 1}",
        desc=f"{label} community college meeting all specified criteria",
        parent=parent_node,
        critical=False,  # Allow partial credit across colleges
    )

    # 1) Name (critical): provided official name
    evaluator.add_custom_node(
        result=bool(college.name and college.name.strip()),
        id=f"college_{idx + 1}_name",
        desc="Provides the official name of the college",
        parent=college_node,
        critical=True,
    )

    # 2) Location (critical): college is in LA/Orange/SD County
    loc_node = evaluator.add_leaf(
        id=f"college_{idx + 1}_location",
        desc="College is located in Los Angeles County, Orange County, or San Diego County, California",
        parent=college_node,
        critical=True,
    )
    # Build claim using the address; allow LLM to use common knowledge about cities/counties.
    county_hint = (college.county or "").strip()
    allowed_counties = "Los Angeles County, Orange County, or San Diego County"
    if county_hint:
        location_claim = f"The college's address is '{college.address}'. This college is located in {county_hint}, which is one of: {allowed_counties}."
    else:
        location_claim = (
            f"The college's address is '{college.address}'. Determine whether this address is in one of the following counties in California: {allowed_counties}."
        )
    await evaluator.verify(
        claim=location_claim,
        node=loc_node,
        sources=collect_sources(college.address_reference_url, college.website),
        additional_instruction=(
            "Use the address shown on the provided page(s). If the county name is not explicitly shown, "
            "infer the county from the city using your general knowledge. Accept if it is clearly within "
            "Los Angeles County, Orange County, or San Diego County."
        ),
    )

    # 3) Accreditation by ACCJC (critical)
    accred_node = evaluator.add_leaf(
        id=f"college_{idx + 1}_accreditation",
        desc="College is accredited by the Accrediting Commission for Community and Junior Colleges (ACCJC)",
        parent=college_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The college named '{college.name}' is accredited by the Accrediting Commission for Community and Junior Colleges (ACCJC)."
        ),
        node=accred_node,
        sources=collect_sources(college.accreditation_reference_urls, college.website),
        additional_instruction=(
            "Look for explicit mentions like 'Accredited by ACCJC', 'WASC ACCJC', or 'Accrediting Commission for Community and Junior Colleges'. "
            "Minor variations in phrasing are acceptable."
        ),
    )

    # 4) UC TAG participation (critical)
    tag_node = evaluator.add_leaf(
        id=f"college_{idx + 1}_tag_participation",
        desc="College participates in the UC Transfer Admission Guarantee (TAG) program",
        parent=college_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The college named '{college.name}' participates in the University of California Transfer Admission Guarantee (TAG) program."
        ),
        node=tag_node,
        sources=collect_sources(college.tag_reference_urls, college.website),
        additional_instruction=(
            "Look for 'UC TAG' or 'Transfer Admission Guarantee' information on the provided page(s). "
            "It's sufficient if the college officially describes participation or services specific to UC TAG."
        ),
    )

    # 5) ADT programs (critical, parallel group of checks)
    adt_group = evaluator.add_parallel(
        id=f"college_{idx + 1}_adt_programs",
        desc="College offers ADT programs in at least three of the specified fields (Biology, Business Administration, Computer Science, or Psychology)",
        parent=college_node,
        critical=True,
    )

    normalized_adt = normalize_adt_fields(college.adt_programs)
    adt_list_str = ", ".join(normalized_adt) if normalized_adt else "None"

    # 5.1) ADT list provided (critical)
    evaluator.add_custom_node(
        result=bool(normalized_adt),
        id=f"college_{idx + 1}_adt_list",
        desc="Provides a list of which specific ADT programs from the four fields are offered at the college",
        parent=adt_group,
        critical=True,
    )

    # 5.2) ADT count >= 3 (critical)
    evaluator.add_custom_node(
        result=(len(normalized_adt) >= 3),
        id=f"college_{idx + 1}_adt_count",
        desc="The list includes at least three ADT programs from the specified fields",
        parent=adt_group,
        critical=True,
    )

    # 5.3) ADT verification against reference URL(s) (critical)
    adt_verify_node = evaluator.add_leaf(
        id=f"college_{idx + 1}_adt_verification",
        desc="Each claimed ADT program is verifiable through the provided reference URL or official college catalog",
        parent=adt_group,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The college named '{college.name}' offers Associate Degree for Transfer (ADT) programs in the following fields: {adt_list_str}. "
            "This should be supported by the provided page(s)."
        ),
        node=adt_verify_node,
        sources=collect_sources(college.adt_reference_urls),
        additional_instruction=(
            "Look specifically for ADT indicators such as 'ADT', 'AA-T', 'AS-T', or explicit statements like "
            "'Associate in Arts for Transfer' or 'Associate in Science for Transfer' in Biology, Business Administration, "
            "Computer Science, or Psychology. Minor naming variations are acceptable if they clearly denote ADT."
        ),
    )

    # 6) Contact info (critical, parallel: website + address completeness)
    contact_group = evaluator.add_parallel(
        id=f"college_{idx + 1}_contact_info",
        desc="Provides complete and accurate contact information including official website and full mailing address",
        parent=college_node,
        critical=True,
    )

    # 6.1) Website URL is provided and accessible/official (critical)
    website_node = evaluator.add_leaf(
        id=f"college_{idx + 1}_website",
        desc="Official website URL is provided and accessible",
        parent=contact_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This is the official website of the college named '{college.name}'.",
        node=website_node,
        sources=college.website if (college.website and college.website.strip()) else None,
        additional_instruction=(
            "Pass if the site clearly represents the named college (e.g., college name in the page title/header/footer). "
            "If no URL is provided, this should fail."
        ),
    )

    # 6.2) Address completeness (critical)
    evaluator.add_custom_node(
        result=is_complete_address(college.address),
        id=f"college_{idx + 1}_address",
        desc="Complete mailing address including street address, city, state, and ZIP code is provided",
        parent=contact_group,
        critical=True,
    )

    # 7) District (non-critical)
    district_node = evaluator.add_leaf(
        id=f"college_{idx + 1}_district",
        desc="Correctly identifies the community college district to which the college belongs",
        parent=college_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The college named '{college.name}' belongs to the community college district '{college.district}'.",
        node=district_node,
        sources=collect_sources(college.website, college.adt_reference_urls),
        additional_instruction=(
            "Look for mentions of the governing district on the website (often in the footer/about pages). "
            "Accept if the district name appears associated with the college."
        ),
    )

    # 8) Reference URL presence for ADTs (critical)
    evaluator.add_custom_node(
        result=bool(college.adt_reference_urls),
        id=f"college_{idx + 1}_reference_url",
        desc="Provides a reference URL that verifies the ADT program offerings for the college",
        parent=college_node,
        critical=True,
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
    Evaluate an answer for the California community colleges with ADT programs task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Colleges are independent; allow partial credit across them
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

    # Extract structured college information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_colleges(),
        template_class=CollegesExtraction,
        extraction_name="colleges_extraction",
    )

    # Record allowed ADT fields for reference
    evaluator.add_ground_truth({
        "allowed_adt_fields": ALLOWED_ADT_FIELDS,
        "requirement": "Each selected college must offer ADT programs in at least three of the allowed fields.",
    })

    # Select up to the first four colleges; pad with empty items if fewer
    colleges: List[CollegeItem] = list(extracted.colleges[:4])
    while len(colleges) < 4:
        colleges.append(CollegeItem())

    # Build verification subtrees for each of the four colleges
    for i in range(4):
        await verify_one_college(evaluator, root, colleges[i], i)

    # Return structured summary
    return evaluator.get_summary()