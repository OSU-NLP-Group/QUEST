import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ut_system_universities_requirements_eval"
TASK_DESCRIPTION = """
I am a high school senior researching Texas public universities for college applications and need to find institutions that meet specific academic and campus criteria. Please identify 3 universities within the University of Texas System that meet ALL of the following requirements:

1. Offer an undergraduate honors program or honors college
2. Have at least one LEED-certified building on campus (any certification level: Certified, Silver, Gold, or Platinum)
3. Compete in NCAA Division I athletics
4. Provide a writing center that offers free tutoring services for undergraduate students
5. Have a priority application deadline on or before December 1 for freshman admission or scholarship consideration

For each university, provide:
- The university's full official name
- Confirmation that it meets each of the 5 requirements listed above
- A reference URL for each requirement showing evidence that the criterion is met
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    full_official_name: Optional[str] = None

    # A URL that evidences the institution is part of the University of Texas System (if the answer provides it)
    system_urls: List[str] = Field(default_factory=list)

    # One or more URLs the answer cites for each requirement
    honors_urls: List[str] = Field(default_factory=list)
    leed_urls: List[str] = Field(default_factory=list)
    ncaa_urls: List[str] = Field(default_factory=list)
    writing_urls: List[str] = Field(default_factory=list)
    deadline_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
Extract up to three universities listed in the answer that the answer claims are within the University of Texas System and meet the specified five requirements. Return a JSON object with a 'universities' array. For EACH university, extract the following:

- full_official_name: The university's full official name as written in the answer.
- system_urls: A list of URLs in the answer that explicitly indicate the school is in the University of Texas System (e.g., UT System institutions page or any authoritative page that states UT System membership). If none are provided, return an empty list.
- honors_urls: A list of URLs in the answer that support the existence of an undergraduate honors program or honors college.
- leed_urls: A list of URLs in the answer that support the existence of at least one LEED-certified building on campus (Certified/Silver/Gold/Platinum).
- ncaa_urls: A list of URLs in the answer that support NCAA Division I athletics status.
- writing_urls: A list of URLs in the answer that support a writing center offering free tutoring for undergraduate students (e.g., "free", "no charge", "no-cost").
- deadline_urls: A list of URLs in the answer that support a priority application deadline on or before December 1 for freshman admission, scholarship consideration, or honors admission.

Rules:
- Extract ONLY URLs that are explicitly present in the answer. Do not invent or infer any URL.
- Include full URLs. If a URL in the answer lacks a protocol, prepend "http://".
- If the answer lists more than three universities, include only the first three in the 'universities' array.
- If a given field is not supported by any URL in the answer for that university, return an empty list for that field.
""".strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_urls(urls: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for u in urls or []:
        if not u:
            continue
        u = u.strip()
        # Basic validity: must start with http or https
        if not (u.lower().startswith("http://") or u.lower().startswith("https://")):
            continue
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _has_valid_urls(urls: List[str]) -> bool:
    return len(_normalize_urls(urls)) > 0


def _ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][n] if 0 <= n < 5 else f"#{n + 1}"


def _combine_all_sources(u: UniversityItem) -> List[str]:
    combined = (
        (u.system_urls or []) +
        (u.honors_urls or []) +
        (u.leed_urls or []) +
        (u.ncaa_urls or []) +
        (u.writing_urls or []) +
        (u.deadline_urls or [])
    )
    return _normalize_urls(combined)


# --------------------------------------------------------------------------- #
# Verification per university                                                 #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent: VerificationNode,
    uni: UniversityItem,
    idx: int
) -> None:
    ord_name = _ordinal(idx)
    uni_node = evaluator.add_parallel(
        id=f"university_{idx + 1}",
        desc=f"{ord_name} qualifying university with all requirements met",
        parent=parent,
        critical=False
    )

    uni_name = (uni.full_official_name or "").strip()

    # 1) Identification (full official name provided)
    evaluator.add_custom_node(
        result=bool(uni_name),
        id=f"u{idx + 1}_identification",
        desc="University is properly identified with its full official name",
        parent=uni_node,
        critical=True
    )

    # 2) UT System membership (verify via any provided membership URL or other authoritative URLs)
    membership_sources = _normalize_urls(uni.system_urls)
    if not membership_sources:
        # Fall back to any other provided sources that might explicitly state UT System membership
        membership_sources = _combine_all_sources(uni)

    if _has_valid_urls(membership_sources) and uni_name:
        sys_leaf = evaluator.add_leaf(
            id=f"u{idx + 1}_system_membership",
            desc="University is part of the University of Texas System",
            parent=uni_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The institution '{uni_name}' is a member of the University of Texas System.",
            node=sys_leaf,
            sources=membership_sources,
            additional_instruction=(
                "Verify explicit UT System membership. Prefer the official UT System site. "
                "Authoritative university pages that clearly state UT System membership also count. "
                "Do not assume membership based solely on the institution's name; rely on explicit statements."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"u{idx + 1}_system_membership",
            desc="University is part of the University of Texas System",
            parent=uni_node,
            critical=True
        )

    # 3) Honors program / honors college
    honors_node = evaluator.add_sequential(
        id=f"u{idx + 1}_honors_program",
        desc="University offers an undergraduate honors program or honors college",
        parent=uni_node,
        critical=True
    )
    honors_sources = _normalize_urls(uni.honors_urls)
    if _has_valid_urls(honors_sources) and uni_name:
        honors_exists_leaf = evaluator.add_leaf(
            id=f"u{idx + 1}_honors_exists",
            desc="An established honors program or honors college exists for undergraduates",
            parent=honors_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The institution '{uni_name}' offers an undergraduate honors program or an honors college.",
            node=honors_exists_leaf,
            sources=honors_sources,
            additional_instruction=(
                "Evidence should clearly refer to undergraduate honors at this university "
                "(e.g., 'Honors College', 'University Honors Program', 'Honors & Scholars')."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"u{idx + 1}_honors_exists",
            desc="An established honors program or honors college exists for undergraduates",
            parent=honors_node,
            critical=True
        )

    evaluator.add_custom_node(
        result=_has_valid_urls(honors_sources),
        id=f"u{idx + 1}_honors_url",
        desc="Valid reference URL provided for the honors program",
        parent=honors_node,
        critical=True
    )

    # 4) LEED-certified building
    leed_node = evaluator.add_sequential(
        id=f"u{idx + 1}_leed_building",
        desc="University has at least one LEED-certified building",
        parent=uni_node,
        critical=True
    )
    leed_sources = _normalize_urls(uni.leed_urls)
    if _has_valid_urls(leed_sources) and uni_name:
        leed_exists_leaf = evaluator.add_leaf(
            id=f"u{idx + 1}_leed_exists",
            desc="At least one building has LEED certification (any level: Certified, Silver, Gold, or Platinum)",
            parent=leed_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"At least one campus building at '{uni_name}' has an official LEED certification (Certified, Silver, Gold, or Platinum).",
            node=leed_exists_leaf,
            sources=leed_sources,
            additional_instruction=(
                "Accept official campus sustainability/facilities pages or USGBC entries that explicitly state LEED certification status. "
                "Reject claims that only say 'built to LEED standards' or 'seeking LEED' without a confirmed certification."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"u{idx + 1}_leed_exists",
            desc="At least one building has LEED certification (any level: Certified, Silver, Gold, or Platinum)",
            parent=leed_node,
            critical=True
        )

    evaluator.add_custom_node(
        result=_has_valid_urls(leed_sources),
        id=f"u{idx + 1}_leed_url",
        desc="Valid reference URL provided for LEED certification",
        parent=leed_node,
        critical=True
    )

    # 5) NCAA Division I athletics
    ncaa_node = evaluator.add_sequential(
        id=f"u{idx + 1}_ncaa_division_i",
        desc="University competes in NCAA Division I athletics",
        parent=uni_node,
        critical=True
    )
    ncaa_sources = _normalize_urls(uni.ncaa_urls)
    if _has_valid_urls(ncaa_sources) and uni_name:
        ncaa_status_leaf = evaluator.add_leaf(
            id=f"u{idx + 1}_ncaa_status",
            desc="University has current NCAA Division I status",
            parent=ncaa_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The institution '{uni_name}' competes in NCAA Division I athletics (not Division II or III).",
            node=ncaa_status_leaf,
            sources=ncaa_sources,
            additional_instruction=(
                "Accept NCAA.org, the university's official athletics site, or authoritative conference pages "
                "that clearly indicate Division I membership."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"u{idx + 1}_ncaa_status",
            desc="University has current NCAA Division I status",
            parent=ncaa_node,
            critical=True
        )

    evaluator.add_custom_node(
        result=_has_valid_urls(ncaa_sources),
        id=f"u{idx + 1}_ncaa_url",
        desc="Valid reference URL provided for NCAA Division I status",
        parent=ncaa_node,
        critical=True
    )

    # 6) Writing center free tutoring for undergraduates
    writing_node = evaluator.add_sequential(
        id=f"u{idx + 1}_writing_center",
        desc="University provides a writing center with free tutoring for undergraduates",
        parent=uni_node,
        critical=True
    )
    writing_sources = _normalize_urls(uni.writing_urls)
    if _has_valid_urls(writing_sources) and uni_name:
        writing_exists_leaf = evaluator.add_leaf(
            id=f"u{idx + 1}_writing_center_exists",
            desc="Writing center offers free tutoring services to undergraduate students",
            parent=writing_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The institution '{uni_name}' provides a writing center (or equivalent) that offers free tutoring services to undergraduate students.",
            node=writing_exists_leaf,
            sources=writing_sources,
            additional_instruction=(
                "Look for explicit phrases like 'free', 'no charge', or 'no-cost' for tutoring/writing consultations "
                "available to currently enrolled undergraduate students. Synonyms: writing lab, writing support center."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"u{idx + 1}_writing_center_exists",
            desc="Writing center offers free tutoring services to undergraduate students",
            parent=writing_node,
            critical=True
        )

    evaluator.add_custom_node(
        result=_has_valid_urls(writing_sources),
        id=f"u{idx + 1}_writing_center_url",
        desc="Valid reference URL provided for writing center services",
        parent=writing_node,
        critical=True
    )

    # 7) Priority deadline on or before December 1
    deadline_node = evaluator.add_sequential(
        id=f"u{idx + 1}_priority_deadline",
        desc="University has a priority deadline on or before December 1",
        parent=uni_node,
        critical=True
    )
    deadline_sources = _normalize_urls(uni.deadline_urls)
    if _has_valid_urls(deadline_sources) and uni_name:
        deadline_leaf = evaluator.add_leaf(
            id=f"u{idx + 1}_deadline_date",
            desc="Priority deadline for freshman admission, scholarship, or honors consideration is on or before December 1",
            parent=deadline_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"For '{uni_name}', there is at least one stated priority deadline on or before December 1 for freshman admission, scholarship consideration, or honors admission.",
            node=deadline_leaf,
            sources=deadline_sources,
            additional_instruction=(
                "Accept earlier dates as satisfying the requirement. If multiple deadlines are listed, "
                "confirm that at least one applicable to first-year/freshman, scholarships, or honors is on/before Dec 1."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"u{idx + 1}_deadline_date",
            desc="Priority deadline for freshman admission, scholarship, or honors consideration is on or before December 1",
            parent=deadline_node,
            critical=True
        )

    evaluator.add_custom_node(
        result=_has_valid_urls(deadline_sources),
        id=f"u{idx + 1}_deadline_url",
        desc="Valid reference URL provided for the priority deadline",
        parent=deadline_node,
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
    model: str = "o4-mini"
) -> Dict:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find 3 universities in the University of Texas System that meet all specified criteria",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Ensure exactly 3 entries for evaluation (pad with empty if fewer)
    universities: List[UniversityItem] = list(extraction.universities[:3])
    while len(universities) < 3:
        universities.append(UniversityItem())

    # Add a container node for the universities (parallel aggregation, allow partial credit)
    universities_node = evaluator.add_parallel(
        id="universities",
        desc="Identify exactly 3 qualifying universities",
        parent=root,
        critical=False
    )

    # Verify each university
    for i, uni in enumerate(universities):
        await verify_university(evaluator, universities_node, uni, i)

    return evaluator.get_summary()