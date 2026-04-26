import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_clear_150_clock_hours_provider_course"
TASK_DESCRIPTION = (
    "For a California teacher holding a Preliminary Multiple Subject Teaching Credential who wishes to upgrade to a Clear credential "
    "using the 150 clock hours pathway (rather than completing a Teacher Induction Program or Master's degree), identify one approved "
    "online professional development provider and one specific graduate-level course offering from that provider that would count toward "
    "the 150 clock hour requirement. Provide the provider name, verification of the provider's approval status with the California "
    "Commission on Teacher Credentialing, the specific course title, the number of graduate credits or clock hours the course provides, "
    "and reference URLs for both the provider and the course."
)

CONVERSION_NOTE = "Use 1 semester credit = 15 clock hours for equivalence."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProviderInfo(BaseModel):
    name: Optional[str] = None
    # URLs explicitly provided in the answer for the provider (homepage, provider course catalog, etc.)
    provider_urls: List[str] = Field(default_factory=list)
    # URLs explicitly provided in the answer that relate to CTC approval/recognition/listing for this provider
    ctc_approval_urls: List[str] = Field(default_factory=list)


class CourseInfo(BaseModel):
    title: Optional[str] = None
    # Primary official course offering page URL provided in the answer
    url: Optional[str] = None
    # As stated in the answer text (keep as free-form string for robustness)
    credits: Optional[str] = None
    # As stated in the answer text (could be a number or phrase; keep as string)
    clock_hours: Optional[str] = None


class SolutionExtraction(BaseModel):
    provider: Optional[ProviderInfo] = None
    course: Optional[CourseInfo] = None
    # Optional snippet the answer uses to frame the 150-hour pathway (if present)
    pathway_fit_text: Optional[str] = None
    # Optional snippet mentioning 5-year validity window (if present)
    five_year_window_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_solution() -> str:
    return """
    Extract exactly one provider and one course from the answer that are proposed for satisfying the California CTC 150 clock-hours pathway
    toward clearing a Preliminary Multiple Subject Teaching Credential. If multiple are mentioned, select the first provider and the first
    course mentioned in the answer.

    Return a JSON object with:
    - provider:
        - name: The provider's name as stated in the answer (string or null).
        - provider_urls: An array of URLs explicitly included in the answer that point to the provider's official site or pages.
        - ctc_approval_urls: An array of URLs explicitly included in the answer that point to California Commission on Teacher Credentialing
          (ctc.ca.gov) pages or other official CTC documentation/listings (if any).
    - course:
        - title: The specific course title or code as stated in the answer (string or null).
        - url: The course offering page URL (string or null). If multiple course URLs are present, choose the first one.
        - credits: The number of graduate credits (semester credits) the answer claims the course provides (string or null).
        - clock_hours: The number of clock hours the answer states for the course or the stated equivalent hours (string or null).
    - pathway_fit_text: Short quote or phrase from the answer that indicates the response is about the 150 clock-hour pathway (not induction or master's). If not found, return null.
    - five_year_window_text: Short quote or phrase from the answer that mentions that the 150 hours are completed within the 5-year validity window of the Preliminary credential. If not found, return null.

    IMPORTANT URL RULES:
    - Include only URLs that are explicitly present in the answer text (including markdown links).
    - Extract complete URLs. If missing protocol, prepend http://
    - Do not fabricate URLs.

    If anything is not mentioned in the answer, return null for that field or an empty array for URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
_num_pattern = re.compile(r"(\d+(\.\d+)?)")


def parse_number(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    m = _num_pattern.search(value)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def nearly_equal(a: float, b: float, tol: float = 1.0) -> bool:
    return abs(a - b) <= tol


def compute_conversion_check(credits_str: Optional[str], hours_str: Optional[str]) -> bool:
    """
    Return True if:
      - credits are not provided (conversion not required); or
      - credits and hours are both provided and hours ~= credits*15.
    Return False if:
      - credits provided but hours missing; or
      - credits provided and hours provided but mismatch numerically.
    """
    credit_num = parse_number(credits_str)
    hours_num = parse_number(hours_str)

    # If no credits mentioned, this check is not applicable -> pass
    if credit_num is None:
        return True

    # Credits mentioned but no hours -> require conversion to be stated
    if hours_num is None:
        return False

    expected_hours = credit_num * 15.0
    return nearly_equal(expected_hours, hours_num, tol=1.0)


def first_non_empty(*args: Optional[List[str]]) -> List[str]:
    """Return the first non-empty list among args, else empty list."""
    for arr in args:
        if arr and len(arr) > 0:
            return arr
    return []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_pathway_fit_checks(evaluator: Evaluator, parent_node) -> None:
    """
    Build the 'Pathway_Fit' check ensuring the answer is framed for the 150 clock-hour pathway
    (not Teacher Induction Program or a Master's degree).
    """
    node = evaluator.add_leaf(
        id="pathway_fit",
        desc="Response is framed for the 150 clock-hour pathway (not a Teacher Induction Program or a Master's degree pathway).",
        parent=parent_node,
        critical=True,
    )
    claim = (
        "The response focuses on earning/using 150 clock hours to clear a California Preliminary Multiple Subject Teaching Credential. "
        "It does not direct the teacher to complete a Teacher Induction Program or a Master's degree as the pathway."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction=(
            "Judge only from the answer text. Consider this correct if the answer explicitly frames the 150 clock-hour pathway for upgrading "
            "to a Clear credential and avoids recommending Teacher Induction or a master's as the chosen route."
        ),
    )


async def build_provider_checks(evaluator: Evaluator, parent_node, data: SolutionExtraction) -> None:
    """
    Build the 'Provider_Information' subtree:
    - Provider_Name (existence)
    - CTC_Approval_or_Recognition (supported by sources)
    - Online_Availability (supported by sources)
    - Provider_Reference_URL (existence)
    """
    provider_node = evaluator.add_parallel(
        id="provider_information",
        desc="Information about the professional development provider.",
        parent=parent_node,
        critical=True,
    )

    provider = data.provider or ProviderInfo()
    provider_name = provider.name or ""
    provider_urls = provider.provider_urls or []
    ctc_urls = provider.ctc_approval_urls or []
    all_provider_related_urls = (ctc_urls + provider_urls) if (ctc_urls or provider_urls) else []

    # Provider_Name existence
    evaluator.add_custom_node(
        result=bool(provider_name.strip()),
        id="provider_name",
        desc="Provider name is stated.",
        parent=provider_node,
        critical=True,
    )

    # Provider_Reference_URL existence
    evaluator.add_custom_node(
        result=len(all_provider_related_urls) > 0,
        id="provider_reference_url",
        desc="At least one reference URL is provided for the provider (e.g., official provider page and/or CTC-related listing/documentation).",
        parent=provider_node,
        critical=True,
    )

    # CTC_Approval_or_Recognition - verify via URLs
    ctc_support_node = evaluator.add_leaf(
        id="ctc_approval_or_recognition",
        desc="Provider is stated to be approved or recognized by the California Commission on Teacher Credentialing (CTC) for applicable credential-related coursework (claim is verifiable).",
        parent=provider_node,
        critical=True,
    )
    ctc_claim = (
        f"The provider '{provider_name}' is approved or recognized by the California Commission on Teacher Credentialing (CTC) "
        f"for applicable credential-related professional development/coursework that can count toward California credential requirements."
    )
    await evaluator.verify(
        claim=ctc_claim,
        node=ctc_support_node,
        sources=first_non_empty(ctc_urls, provider_urls),
        additional_instruction=(
            "Prefer support from ctc.ca.gov or official CTC documentation/listings. If not present, the provider's own official statements "
            "explicitly indicating CTC approval/recognition for credential-related coursework may be acceptable."
        ),
    )

    # Online_Availability - verify via provider URLs
    online_node = evaluator.add_leaf(
        id="online_availability",
        desc="Provider is stated to offer online courses (verifiable).",
        parent=provider_node,
        critical=True,
    )
    online_claim = f"The provider '{provider_name}' offers online courses or online professional development."
    await evaluator.verify(
        claim=online_claim,
        node=online_node,
        sources=first_non_empty(provider_urls, all_provider_related_urls),
        additional_instruction=(
            "Support can include phrases such as 'online', 'self-paced online', 'distance learning', 'virtual course', etc."
        ),
    )


async def build_course_checks(evaluator: Evaluator, parent_node, data: SolutionExtraction) -> None:
    """
    Build the 'Course_Information' subtree:
    - Course_Title (existence)
    - Course_Offered_By_Provider (URL verification)
    - Graduate_Level_or_Equivalent (URL verification)
    - CSTP_Alignment (URL verification; allow provider-wide CSTP alignment page if course page lacks explicit mention)
    - Credits_or_Clock_Hours_and_Conversion (split into verification + conversion check)
    - Course_Reference_URL (existence)
    """
    course_node = evaluator.add_parallel(
        id="course_information",
        desc="Information about one specific qualifying course from the named provider.",
        parent=parent_node,
        critical=True,
    )

    provider = data.provider or ProviderInfo()
    provider_name = provider.name or ""
    provider_urls = provider.provider_urls or []
    ctc_urls = provider.ctc_approval_urls or []
    all_provider_urls = (provider_urls + ctc_urls) if (provider_urls or ctc_urls) else []

    course = data.course or CourseInfo()
    course_title = course.title or ""
    course_url = course.url or ""
    credits_str = course.credits
    hours_str = course.clock_hours

    # Course_Title existence
    evaluator.add_custom_node(
        result=bool(course_title.strip()),
        id="course_title",
        desc="Course title (and/or course code) is stated.",
        parent=course_node,
        critical=True,
    )

    # Course_Reference_URL existence
    evaluator.add_custom_node(
        result=bool(course_url.strip()),
        id="course_reference_url",
        desc="A reference URL is provided for the course offering page (or equivalent official listing) that supports the course details.",
        parent=course_node,
        critical=True,
    )

    # Course_Offered_By_Provider - verify using course page
    offered_by_node = evaluator.add_leaf(
        id="course_offered_by_provider",
        desc="It is clear the named course is offered by the identified provider.",
        parent=course_node,
        critical=True,
    )
    offered_by_claim = (
        f"The course page indicates that the course '{course_title}' is offered by the provider '{provider_name}'. "
        "If the page shows a university partner involved, it should still identify the provider as the offering organization or sponsor."
    )
    await evaluator.verify(
        claim=offered_by_claim,
        node=offered_by_node,
        sources=course_url if course_url else None,
        additional_instruction="Look for the provider name/logo or explicit text attributing the course to the provider.",
    )

    # Graduate_Level_or_Equivalent - verify via course URL (and possibly provider URLs)
    grad_level_node = evaluator.add_leaf(
        id="graduate_level_or_equivalent",
        desc="Course is stated to be graduate-level or equivalent for credential-related requirements (verifiable).",
        parent=course_node,
        critical=True,
    )
    grad_claim = (
        f"The course '{course_title}' is described as graduate-level or provides graduate credit (or equivalent post-baccalaureate credit) "
        "suitable for credential-related professional development."
    )
    await evaluator.verify(
        claim=grad_claim,
        node=grad_level_node,
        sources=[u for u in [course_url, *all_provider_urls] if u],
        additional_instruction="Accept terms like 'graduate credit', 'graduate-level', '500-level', or equivalent language.",
    )

    # CSTP_Alignment - verify via course URL and/or provider URL
    cstp_node = evaluator.add_leaf(
        id="cstp_alignment",
        desc="Course is stated to align with the California Standards for the Teaching Profession (CSTP) (verifiable).",
        parent=course_node,
        critical=True,
    )
    cstp_claim = (
        f"The course '{course_title}' or the provider explicitly states alignment with the California Standards for the Teaching Profession (CSTP)."
    )
    await evaluator.verify(
        claim=cstp_claim,
        node=cstp_node,
        sources=[u for u in [course_url, *all_provider_urls] if u],
        additional_instruction="If the course page is silent, a provider page that clearly indicates CSTP alignment for its offerings is acceptable.",
    )

    # Credits_or_Clock_Hours_and_Conversion - split into two leaves under a critical parallel aggregator
    credits_main = evaluator.add_parallel(
        id="credits_or_clock_hours_and_conversion",
        desc="Course provides a specified number of semester credits or clock hours; and, if semester credits are used, uses 1 credit = 15 hours to express the equivalent.",
        parent=course_node,
        critical=True,
    )

    # 1) Verify credits or hours as stated are supported by the course URL
    credits_or_hours_node = evaluator.add_leaf(
        id="course_value_supported",
        desc="The stated credits and/or clock hours for the course are supported by the course page.",
        parent=credits_main,
        critical=True,
    )
    # Build claim depending on what's present in the answer
    if credits_str and hours_str:
        value_claim = (
            f"The course page indicates that the course '{course_title}' provides {credits_str} (graduate) semester credits "
            f"and/or {hours_str} clock hours (or an equivalent hours figure)."
        )
    elif credits_str:
        value_claim = f"The course page indicates that the course '{course_title}' provides {credits_str} (graduate) semester credits."
    elif hours_str:
        value_claim = f"The course page indicates that the course '{course_title}' provides {hours_str} clock hours (or equivalent)."
    else:
        # If neither is provided in the answer, make a minimal claim that will likely fail (gated by other critical sibling nodes)
        value_claim = f"The course page indicates the credits or clock hours for the course '{course_title}'."

    await evaluator.verify(
        claim=value_claim,
        node=credits_or_hours_node,
        sources=course_url if course_url else None,
        additional_instruction="Match the numbers or language on the official course page. Allow reasonable numeric formatting differences.",
    )

    # 2) Conversion check (custom computation). Pass if not applicable.
    conversion_ok = compute_conversion_check(credits_str, hours_str)
    evaluator.add_custom_node(
        result=conversion_ok,
        id="credits_to_hours_conversion_correct",
        desc="If semester credits are used in the answer, the conversion 1 credit = 15 hours is applied correctly and matches the provided hours.",
        parent=credits_main,
        critical=True,
    )


async def build_five_year_context_check(evaluator: Evaluator, parent_node) -> None:
    """
    Non-critical context node verifying the answer mentions the 150 hours are completed within
    the 5-year validity period of the Preliminary credential.
    """
    node = evaluator.add_leaf(
        id="hours_within_5_years_context",
        desc="Mentions that the 150 clock hours are completed within the 5-year validity period of the Preliminary credential.",
        parent=parent_node,
        critical=False,
    )
    claim = (
        "The response mentions that the 150 clock hours must be completed within the five-year validity window of the California Preliminary credential."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Judge only from the answer text.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the California 150 clock-hours pathway: one provider + one course verification.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Children evaluated independently
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

    # Record conversion rule as custom info
    evaluator.add_custom_info({"conversion_rule": CONVERSION_NOTE}, info_type="note", info_name="conversion_rule")

    # IMPORTANT: Root must be non-critical to allow non-critical children under it per framework constraint
    # The rubric labels the overall solution as critical, but we relax root criticality to satisfy framework rule:
    # "If parent is critical, all children must be critical." We have a non-critical child (5-year context).
    # Therefore, we keep root as non-critical.

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_solution(),
        template_class=SolutionExtraction,
        extraction_name="solution_extraction",
    )

    # Build verification tree according to rubric
    # Solution root-level grouping (acts as logical container)
    solution_node = evaluator.add_parallel(
        id="solution",
        desc="Identify one CTC-approved/recognized online provider and one qualifying course from that provider that can count toward the 150 clock-hour pathway, with required details and citations.",
        parent=root,
        critical=False,  # non-critical to allow non-critical children (5-year context) downstream
    )

    # 1) Pathway Fit (critical)
    await build_pathway_fit_checks(evaluator, solution_node)

    # 2) Provider Information (critical subtree)
    await build_provider_checks(evaluator, solution_node, extracted)

    # 3) Course Information (critical subtree)
    await build_course_checks(evaluator, solution_node, extracted)

    # 4) 150 hours within 5 years (non-critical)
    await build_five_year_context_check(evaluator, solution_node)

    return evaluator.get_summary()