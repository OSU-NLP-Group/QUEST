import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "career_services_director_ca_2025_2026"
TASK_DESCRIPTION = (
    "Identify a Director-level career services or career center position at a California public university "
    "(either in the California State University system or the University of California system) that was posted or "
    "made available during the 2025-2026 academic year. For this position, provide the following information with "
    "supporting URL references: (1) The name of the specific institution and confirmation of its CSU or UC system "
    "membership, (2) The exact position title and a link to the official job posting, (3) The minimum educational "
    "requirement (must include Master's degree), (4) The minimum years of experience required, (5) Confirmation that "
    "the position includes supervisory or management responsibilities, (6) The posting date or availability timeframe, "
    "(7) Salary range or compensation information (if publicly available), and (8) Key responsibilities of the role. "
    "All information must be supported by valid reference URLs from official sources."
)

# Academic year boundaries (inclusive)
ACADEMIC_YEAR_START = "2025-08-01"
ACADEMIC_YEAR_END = "2026-08-31"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PositionExtraction(BaseModel):
    # Core identification
    institution_name: Optional[str] = None
    system: Optional[str] = None  # Expected values: "CSU" or "UC"
    position_title: Optional[str] = None
    posting_url: Optional[str] = None

    # Dates / timeframe
    posting_date_or_timeframe: Optional[str] = None  # e.g., "Posted September 15, 2025" or "Open until filled, Oct 2025"
    timeframe_support_urls: List[str] = Field(default_factory=list)

    # Membership support
    membership_urls: List[str] = Field(default_factory=list)

    # Education requirement
    education_requirement_text: Optional[str] = None
    education_support_urls: List[str] = Field(default_factory=list)

    # Experience requirement
    min_experience_years: Optional[str] = None  # Keep as free text for robustness, e.g., "5+ years"
    experience_field_text: Optional[str] = None  # e.g., "career services, higher education, student affairs"
    experience_support_urls: List[str] = Field(default_factory=list)

    # Supervisory responsibilities
    supervisory_responsibilities_text: Optional[str] = None
    supervision_support_urls: List[str] = Field(default_factory=list)

    # Compensation
    compensation_text: Optional[str] = None  # e.g., "$100,000-$130,000" or "not publicly listed"
    compensation_support_urls: List[str] = Field(default_factory=list)

    # Responsibilities
    responsibilities: List[str] = Field(default_factory=list)
    responsibilities_support_urls: List[str] = Field(default_factory=list)

    # Title support
    title_support_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_position() -> str:
    return """
    Extract ONE Director-level career services or career center position at a California public university
    (California State University or University of California) that was posted or made available during the
    2025-2026 academic year (Aug 1, 2025 through Aug 31, 2026).

    STRICT RULES:
    - Extract only what is explicitly present in the answer. Do not invent any information or URLs.
    - If any field is missing, return null (or an empty array for list fields).
    - For all URL fields, include only valid URLs explicitly cited in the answer (plain URLs or markdown links).
    - The "system" must be either "CSU" or "UC" based on the institution named in the answer.

    Extract the following fields:
    1) institution_name: The specific campus name (e.g., "UC San Diego", "California State University Long Beach").
    2) system: "CSU" or "UC" corresponding to the institution.
    3) position_title: The exact position title as shown in the posting or announcement.
    4) posting_url: The official job posting or announcement URL.
    5) posting_date_or_timeframe: The posting date or availability timeframe as stated (if present).
    6) timeframe_support_urls: All cited URLs that support the posting date/timeframe (include posting_url if it supports).
    7) membership_urls: All cited URLs that confirm CSU or UC system membership for the institution (e.g., official system campus listing pages).
    8) education_requirement_text: Minimum education requirement verbatim or summarized (must include Master's if stated).
    9) education_support_urls: All cited URLs supporting the education requirement (include posting_url if it supports).
    10) min_experience_years: The minimum years of experience required (e.g., "5 years", "5+ years") as stated.
    11) experience_field_text: The domain area of experience required (e.g., "higher education, career services, student affairs").
    12) experience_support_urls: All cited URLs supporting the experience requirement (include posting_url if it supports).
    13) supervisory_responsibilities_text: Text confirming supervisory/management responsibilities (if present).
    14) supervision_support_urls: All cited URLs supporting supervisory responsibilities (include posting_url if it supports).
    15) compensation_text: Salary range or compensation details if publicly provided; if not listed, the answer should explicitly state that it is not listed—extract such a phrasing if present.
    16) compensation_support_urls: All cited URLs supporting compensation information or its absence (include posting_url if it supports).
    17) responsibilities: Key responsibilities of the role as stated (extract a list of short bullet points or phrases).
    18) responsibilities_support_urls: All cited URLs supporting the listed responsibilities (include posting_url if it supports).
    19) title_support_urls: All cited URLs supporting the exact position title (include posting_url).

    Reminder: If a field is not present in the answer, return null (or an empty array for the URLs/responsibilities).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_non_empty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not _is_non_empty_str(u):
            continue
        # Normalize minimal: strip spaces
        uu = u.strip()
        if uu not in seen:
            seen.add(uu)
            out.append(uu)
    return out


def _aggregate_all_urls(ex: PositionExtraction) -> List[str]:
    urls = []
    if _is_non_empty_str(ex.posting_url):
        urls.append(ex.posting_url.strip())

    urls.extend(ex.membership_urls or [])
    urls.extend(ex.title_support_urls or [])
    urls.extend(ex.timeframe_support_urls or [])
    urls.extend(ex.education_support_urls or [])
    urls.extend(ex.experience_support_urls or [])
    urls.extend(ex.supervision_support_urls or [])
    urls.extend(ex.compensation_support_urls or [])
    urls.extend(ex.responsibilities_support_urls or [])
    return _dedupe_urls(urls)


def _short_responsibilities(resps: List[str], k: int = 6) -> List[str]:
    if not resps:
        return []
    # Limit to first k concise items
    return [r.strip() for r in resps[:k] if _is_non_empty_str(r)]


def _is_official_or_reliable_domain(url: str) -> bool:
    """
    Heuristic check for official/reliable sources:
    - Official university domains (.edu) are accepted.
    - CSU/UC official/system domains (calstate.edu, csu.edu, ucop.edu) are accepted.
    - Official campus-hosted career portals sometimes use Workday (myworkdayjobs.com) with campus identifiers.
      Accept myworkdayjobs.com when the URL clearly references a UC/CSU campus in its subdomain or path.
    """
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()

        if host.endswith(".edu"):
            return True

        if ("calstate.edu" in host) or ("csu.edu" in host) or ("ucop.edu" in host):
            return True

        if "myworkdayjobs.com" in host:
            campus_markers = [
                "berkeley", "ucla", "ucsd", "uci", "ucr", "ucsb", "ucsc", "ucdavis", "ucsf", "ucmerced", "ucop",
                "cpp", "calpoly", "fullerton", "sfsu", "sfstate", "sjsu", "csula", "longbeach", "csulb", "chico",
                "humboldt", "sonoma", "csus", "sacstate", "fresnostate", "csufresno", "stanislaus", "csustan",
                "monterey", "csumb", "bakersfield", "csumb", "dominguez", "csudh", "channelislands", "csuci",
                "sanbernardino", "csusb", "csusm", "sanmarcos", "sanluisobispo", "csusobispo"
            ]
            marker_in_host = any(m in host for m in campus_markers)
            marker_in_path = any(m in path for m in campus_markers)
            return marker_in_host or marker_in_path

        return False
    except Exception:
        return False


async def _check_urls_resolvable(evaluator: Evaluator, urls: List[str]) -> bool:
    """
    Try fetching each URL via evaluator.extractor.get_page_info.
    Consider resolvable only if all URLs can be retrieved (text or screenshot present).
    """
    if not urls:
        return False

    for url in urls:
        try:
            screenshot_b64, web_text = await evaluator.extractor.get_page_info(url)
            if screenshot_b64 is None or web_text is None:
                return False
        except Exception:
            return False
    return True


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_identify_qualifying_posting(
    evaluator: Evaluator,
    parent: Any,
    ex: PositionExtraction,
) -> Any:
    node = evaluator.add_parallel(
        id="IdentifyQualifyingPosting",
        desc="Identify the specific qualifying institution and job posting that meets scope (CSU/UC in CA; Director-level career services; 2025-2026 AY).",
        parent=parent,
        critical=True,
    )

    # InstitutionNameProvided (existence)
    evaluator.add_custom_node(
        result=_is_non_empty_str(ex.institution_name),
        id="InstitutionNameProvided",
        desc="Provides the name of the specific institution.",
        parent=node,
        critical=True,
    )

    # CSUorUC_CaliforniaCampusConfirmed (URL-supported)
    sys_claim = f"The institution '{ex.institution_name or ''}' is a member campus of the {'UC' if (ex.system or '').upper()=='UC' else 'CSU'} system and is located in California."
    sys_node = evaluator.add_leaf(
        id="CSUorUC_CaliforniaCampusConfirmed",
        desc="Confirms the institution is a CSU or UC campus located in California.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=sys_claim,
        node=sys_node,
        sources=_dedupe_urls(ex.membership_urls or ([_ for _ in [ex.posting_url] if _is_non_empty_str(_)])),
        additional_instruction="Confirm the institution is listed among CSU or UC campuses in California using official system/University pages.",
    )

    # ExactPositionTitleProvided (existence)
    evaluator.add_custom_node(
        result=_is_non_empty_str(ex.position_title),
        id="ExactPositionTitleProvided",
        desc="Provides the exact position title as shown on the posting.",
        parent=node,
        critical=True,
    )

    # OfficialJobPostingURLProvided (existence)
    evaluator.add_custom_node(
        result=_is_non_empty_str(ex.posting_url),
        id="OfficialJobPostingURLProvided",
        desc="Provides a URL to the official job posting/announcement.",
        parent=node,
        critical=True,
    )

    # DirectorLevelCareerServicesConfirmed (URL-supported)
    director_claim = (
        "This role is a Director-level position within career services or a career center, based on the posting title/description."
    )
    director_node = evaluator.add_leaf(
        id="DirectorLevelCareerServicesConfirmed",
        desc="Confirms the role is Director-level and in career services/career center (based on posting title/description).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=director_claim,
        node=director_node,
        sources=ex.posting_url,
        additional_instruction="Look for 'Director' in the title or hierarchical description and confirm the functional area is career services/career center.",
    )

    # PostedWithin2025_2026AcademicYear (URL-supported)
    ay_claim = (
        f"The posting date or availability timeframe shown on the cited page(s) indicates the role was posted/available during the 2025-2026 academic year "
        f"(between {ACADEMIC_YEAR_START} and {ACADEMIC_YEAR_END}). Extracted timeframe: '{ex.posting_date_or_timeframe or ''}'."
    )
    ay_node = evaluator.add_leaf(
        id="PostedWithin2025_2026AcademicYear",
        desc="Confirms the posting date or availability timeframe is during the 2025-2026 academic year.",
        parent=node,
        critical=True,
    )
    timeframe_sources = _dedupe_urls((ex.timeframe_support_urls or []) + ([ex.posting_url] if _is_non_empty_str(ex.posting_url) else []))
    await evaluator.verify(
        claim=ay_claim,
        node=ay_node,
        sources=timeframe_sources,
        additional_instruction="Interpret academic year 2025-2026 as Aug 1, 2025 through Aug 31, 2026. Accept 'open until filled' if initial posting is within the window.",
    )

    return node


async def build_required_attributes_extracted(
    evaluator: Evaluator,
    parent: Any,
    ex: PositionExtraction,
) -> Any:
    node = evaluator.add_parallel(
        id="RequiredAttributesExtracted",
        desc="Extract and report the required job attributes from the posting/materials.",
        parent=parent,
        critical=True,
    )

    # MinimumEducationIncludesMasters
    edu_claim = (
        f"The minimum educational requirement includes a Master's degree as stated in the posting. Extracted education text: '{ex.education_requirement_text or ''}'."
    )
    edu_node = evaluator.add_leaf(
        id="MinimumEducationIncludesMasters",
        desc="States the minimum educational requirement and it includes a Master's degree as the minimum qualification.",
        parent=node,
        critical=True,
    )
    edu_sources = _dedupe_urls((ex.education_support_urls or []) + ([ex.posting_url] if _is_non_empty_str(ex.posting_url) else []))
    await evaluator.verify(
        claim=edu_claim,
        node=edu_node,
        sources=edu_sources,
        additional_instruction="Confirm explicit mention of a Master's degree (or equivalent) as a minimum requirement.",
    )

    # MinimumExperienceYearsAtLeast5
    exp_claim = (
        "The minimum years of experience required is at least 5 years, as stated in the posting. "
        f"Extracted minimum experience: '{ex.min_experience_years or ''}'."
    )
    exp_node = evaluator.add_leaf(
        id="MinimumExperienceYearsAtLeast5",
        desc="States the minimum years of experience required and it is at least 5 years.",
        parent=node,
        critical=True,
    )
    exp_sources = _dedupe_urls((ex.experience_support_urls or []) + ([ex.posting_url] if _is_non_empty_str(ex.posting_url) else []))
    await evaluator.verify(
        claim=exp_claim,
        node=exp_node,
        sources=exp_sources,
        additional_instruction="Allow phrasing such as '5 years', '5+ years', 'five years', or ≥5.",
    )

    # ExperienceIsRelevantDomain
    domain_claim = (
        "The required experience is relevant to higher education, career services, or student affairs as stated in the posting. "
        f"Extracted domain text: '{ex.experience_field_text or ''}'."
    )
    domain_node = evaluator.add_leaf(
        id="ExperienceIsRelevantDomain",
        desc="Confirms the required experience is relevant to higher education, career services, or student affairs (as stated in the posting).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=domain_claim,
        node=domain_node,
        sources=exp_sources,
        additional_instruction="Confirm that the specified experience domain includes higher education OR career services OR student affairs.",
    )

    # SupervisoryResponsibilitiesConfirmed
    sup_claim = (
        "The position includes supervisory or staff management responsibilities as described in the posting. "
        f"Extracted supervisory text: '{ex.supervisory_responsibilities_text or ''}'."
    )
    sup_node = evaluator.add_leaf(
        id="SupervisoryResponsibilitiesConfirmed",
        desc="Confirms the position includes supervisory or staff management responsibilities.",
        parent=node,
        critical=True,
    )
    sup_sources = _dedupe_urls((ex.supervision_support_urls or []) + ([ex.posting_url] if _is_non_empty_str(ex.posting_url) else []))
    await evaluator.verify(
        claim=sup_claim,
        node=sup_node,
        sources=sup_sources,
        additional_instruction="Look for phrases indicating supervising staff, managing teams, direct reports, or equivalent.",
    )

    # CompensationAddressed
    comp_sources = _dedupe_urls((ex.compensation_support_urls or []) + ([ex.posting_url] if _is_non_empty_str(ex.posting_url) else []))
    if _is_non_empty_str(ex.compensation_text):
        comp_claim = (
            f"The posting includes publicly available compensation information such as '{ex.compensation_text}'."
        )
    else:
        comp_claim = (
            "The cited posting materials do not list any salary range or compensation details."
        )

    comp_node = evaluator.add_leaf(
        id="CompensationAddressed",
        desc="Provides salary range/compensation information if publicly available; otherwise explicitly states it is not publicly listed in the cited materials.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=comp_claim,
        node=comp_node,
        sources=comp_sources,
        additional_instruction="If compensation is present, confirm content. If absent, confirm the posting does not show salary info.",
    )

    # KeyResponsibilitiesProvided
    top_resps = _short_responsibilities(ex.responsibilities or [])
    resp_text = "; ".join(top_resps) if top_resps else ""
    resp_claim = f"The posting lists key responsibilities such as: {resp_text}."
    resp_node = evaluator.add_leaf(
        id="KeyResponsibilitiesProvided",
        desc="Lists key responsibilities of the role.",
        parent=node,
        critical=True,
    )
    resp_sources = _dedupe_urls((ex.responsibilities_support_urls or []) + ([ex.posting_url] if _is_non_empty_str(ex.posting_url) else []))
    await evaluator.verify(
        claim=resp_claim,
        node=resp_node,
        sources=resp_sources,
        additional_instruction="Match the responsibilities phrasing approximately; minor wording differences are acceptable.",
    )

    return node


async def build_url_support_and_quality(
    evaluator: Evaluator,
    parent: Any,
    ex: PositionExtraction,
) -> Any:
    node = evaluator.add_parallel(
        id="URLSupportAndSourceQuality",
        desc="Provide URL support for each required attribute and ensure sources meet the stated standard.",
        parent=parent,
        critical=True,
    )

    # InstitutionSystemMembershipSupportedByURL
    inst_sys_claim = (
        f"The institution '{ex.institution_name or ''}' is confirmed as a member of the {'UC' if (ex.system or '').upper()=='UC' else 'CSU'} system."
    )
    inst_sys_node = evaluator.add_leaf(
        id="InstitutionSystemMembershipSupportedByURL",
        desc="Provides at least one URL supporting the institution’s CSU/UC membership (or equivalent official confirmation).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=inst_sys_claim,
        node=inst_sys_node,
        sources=_dedupe_urls(ex.membership_urls or []),
        additional_instruction="Use official system pages or campus homepages/campus listing pages.",
    )

    # PositionTitleSupportedByURL
    title_claim = f"The official posting shows the exact position title '{ex.position_title or ''}' or an equivalent minor variation."
    title_node = evaluator.add_leaf(
        id="PositionTitleSupportedByURL",
        desc="Provides at least one URL supporting the exact position title (typically the official posting URL).",
        parent=node,
        critical=True,
    )
    title_sources = _dedupe_urls((ex.title_support_urls or []) + ([ex.posting_url] if _is_non_empty_str(ex.posting_url) else []))
    await evaluator.verify(
        claim=title_claim,
        node=title_node,
        sources=title_sources,
        additional_instruction="Allow case-insensitive matching and minor variants (e.g., inclusion/exclusion of 'of', punctuation).",
    )

    # PostingTimeframeSupportedByURL
    timeframe_claim = (
        f"The posting date or availability timeframe is within Aug 1, 2025 to Aug 31, 2026. Extracted: '{ex.posting_date_or_timeframe or ''}'."
    )
    timeframe_node = evaluator.add_leaf(
        id="PostingTimeframeSupportedByURL",
        desc="Provides at least one URL supporting the posting date or availability timeframe being within the 2025-2026 academic year.",
        parent=node,
        critical=True,
    )
    timeframe_sources = _dedupe_urls((ex.timeframe_support_urls or []) + ([ex.posting_url] if _is_non_empty_str(ex.posting_url) else []))
    await evaluator.verify(
        claim=timeframe_claim,
        node=timeframe_node,
        sources=timeframe_sources,
        additional_instruction="Interpret academic year 2025-2026 as Aug 1, 2025 through Aug 31, 2026.",
    )

    # EducationRequirementSupportedByURL
    edu_support_claim = (
        f"The minimum education requirement in the posting includes a Master's degree. Extracted: '{ex.education_requirement_text or ''}'."
    )
    edu_support_node = evaluator.add_leaf(
        id="EducationRequirementSupportedByURL",
        desc="Provides at least one URL supporting the stated minimum education requirement (including Master's).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=edu_support_claim,
        node=edu_support_node,
        sources=_dedupe_urls((ex.education_support_urls or []) + ([ex.posting_url] if _is_non_empty_str(ex.posting_url) else [])),
        additional_instruction="Confirm explicit mention of a Master's degree (or equivalent).",
    )

    # ExperienceRequirementSupportedByURL
    exp_support_claim = (
        f"The posting specifies minimum experience of at least 5 years. Extracted minimum experience: '{ex.min_experience_years or ''}'."
    )
    exp_support_node = evaluator.add_leaf(
        id="ExperienceRequirementSupportedByURL",
        desc="Provides at least one URL supporting the stated minimum years of experience requirement (>= 5).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=exp_support_claim,
        node=exp_support_node,
        sources=_dedupe_urls((ex.experience_support_urls or []) + ([ex.posting_url] if _is_non_empty_str(ex.posting_url) else [])),
        additional_instruction="Accept '5 years', '5+ years', 'five years', or any wording indicating ≥5.",
    )

    # SupervisionSupportedByURL
    sup_support_claim = (
        f"The posting confirms supervisory/management responsibilities. Extracted: '{ex.supervisory_responsibilities_text or ''}'."
    )
    sup_support_node = evaluator.add_leaf(
        id="SupervisionSupportedByURL",
        desc="Provides at least one URL supporting the supervisory/management responsibility claim.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=sup_support_claim,
        node=sup_support_node,
        sources=_dedupe_urls((ex.supervision_support_urls or []) + ([ex.posting_url] if _is_non_empty_str(ex.posting_url) else [])),
        additional_instruction="Look for 'supervise', 'manage', 'direct reports', 'lead staff', or equivalent phrases.",
    )

    # CompensationSupportedByURL
    if _is_non_empty_str(ex.compensation_text):
        comp_support_claim = f"The posting provides compensation info such as '{ex.compensation_text}'."
    else:
        comp_support_claim = "The posting does not list salary range or compensation details."

    comp_support_node = evaluator.add_leaf(
        id="CompensationSupportedByURL",
        desc="Provides at least one URL supporting the compensation information, or supporting that compensation is not listed in the cited materials.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=comp_support_claim,
        node=comp_support_node,
        sources=_dedupe_urls((ex.compensation_support_urls or []) + ([ex.posting_url] if _is_non_empty_str(ex.posting_url) else [])),
        additional_instruction="If present, confirm salary content; if absent, confirm that the posting does not contain compensation details.",
    )

    # ResponsibilitiesSupportedByURL
    resp_support_claim = f"The posting includes responsibilities such as: {'; '.join(_short_responsibilities(ex.responsibilities or []))}."
    resp_support_node = evaluator.add_leaf(
        id="ResponsibilitiesSupportedByURL",
        desc="Provides at least one URL supporting the listed key responsibilities.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=resp_support_claim,
        node=resp_support_node,
        sources=_dedupe_urls((ex.responsibilities_support_urls or []) + ([ex.posting_url] if _is_non_empty_str(ex.posting_url) else [])),
        additional_instruction="Match responsibilities approximately; minor wording/ordering differences are acceptable.",
    )

    # SourcesAreOfficialOrReliable (custom check across all URLs)
    all_urls = _aggregate_all_urls(ex)
    official_result = (len(all_urls) > 0) and all(_is_official_or_reliable_domain(u) for u in all_urls)
    evaluator.add_custom_node(
        result=official_result,
        id="SourcesAreOfficialOrReliable",
        desc="All reference URLs are from official university/CSU/UC domains or other reliable sources permitted by the constraints.",
        parent=node,
        critical=True,
    )

    # URLsAreResolvable (custom resolvability check across all URLs)
    resolvable_result = await _check_urls_resolvable(evaluator, all_urls)
    evaluator.add_custom_node(
        result=resolvable_result,
        id="URLsAreResolvable",
        desc="All provided reference URLs are valid/resolvable.",
        parent=node,
        critical=True,
    )

    return node


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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the California Director-level career services/career center posting (AY 2025-2026).
    """
    # Initialize evaluator
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

    # Extract structured info from the answer
    ex = await evaluator.extract(
        prompt=prompt_extract_position(),
        template_class=PositionExtraction,
        extraction_name="position_extraction",
    )

    # Add top-level critical sequential node mirroring rubric root
    csd_node = evaluator.add_sequential(
        id="CareerServicesDirectorPosition",
        desc="Identify ONE Director-level career services/career center position at a California CSU or UC campus, posted/available during the 2025-2026 academic year, and provide required attributes with official/reliable URL support.",
        parent=root,
        critical=True,
    )

    # Build subtrees according to rubric
    await build_identify_qualifying_posting(evaluator, csd_node, ex)
    await build_required_attributes_extracted(evaluator, csd_node, ex)
    await build_url_support_and_quality(evaluator, csd_node, ex)

    # Record some custom info
    evaluator.add_custom_info(
        info={
            "institution_name": ex.institution_name,
            "system": ex.system,
            "position_title": ex.position_title,
            "posting_url": ex.posting_url,
            "all_urls_checked": _aggregate_all_urls(ex),
        },
        info_type="extraction_summary",
    )

    # Return evaluation summary
    return evaluator.get_summary()