import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task metadata
# -----------------------------------------------------------------------------
TASK_ID = "research_planning_2026"
TASK_DESCRIPTION = """
You are planning to submit research to ACL 2026 and want to understand the full academic ecosystem for your work. Please provide comprehensive information organized as follows:

1. ACL 2026 Conference Details:
   - What are the page limits for long papers and short papers submitted to ACL 2026? Include both submission and camera-ready versions.
   - What are the key dates for ACL 2026, including: the ARR submission deadline, commitment deadline to ACL 2026, notification of acceptance date, and camera-ready deadline?
   - What review system does ACL 2026 use, and what is the nature of the review process (e.g., single-blind, double-blind)?

2. Target Universities:
   - Identify three universities that appear in the QS World University Rankings 2026. For each university, provide:
     - University name and location
     - QS World University Rankings 2026 position
     - The university's Research & Discovery lens score (which accounts for 50% of the overall QS ranking methodology)

3. Fellowship Opportunities:
   - What is the annual stipend amount for NSF postdoctoral fellowships?
   - What is the submission deadline for the NSF Postdoctoral Research Fellowships in Biology (PRFB) program in 2026?
   - According to the QS World University Rankings methodology, how does QS define a "sustained international research partnership" (in terms of joint papers and time period)?

4. Open Access Journal Requirements:
   - What are the DOAJ (Directory of Open Access Journals) requirements regarding a journal's publishing history or minimum number of articles for a journal to be eligible for indexing?
   - What types of Creative Commons licenses does DOAJ accept for indexed journals?
   - What is DOAJ's definition of "fully open access" for journal eligibility?

For each piece of information, include a reference URL that supports your answer.
""".strip()


# -----------------------------------------------------------------------------
# Extraction Models
# -----------------------------------------------------------------------------
class ACLPaperLimits(BaseModel):
    long_submission_pages: Optional[str] = None
    long_camera_ready_pages: Optional[str] = None
    long_urls: List[str] = Field(default_factory=list)

    short_submission_pages: Optional[str] = None
    short_camera_ready_pages: Optional[str] = None
    short_urls: List[str] = Field(default_factory=list)


class ACLTimeline(BaseModel):
    arr_submission_deadline: Optional[str] = None
    commitment_deadline: Optional[str] = None
    notification_date: Optional[str] = None
    camera_ready_deadline: Optional[str] = None
    timeline_urls: List[str] = Field(default_factory=list)


class ACLReview(BaseModel):
    review_system: Optional[str] = None
    review_anonymity: Optional[str] = None
    review_urls: List[str] = Field(default_factory=list)


class ACLDetailsExtraction(BaseModel):
    paper_limits: Optional[ACLPaperLimits] = None
    timeline: Optional[ACLTimeline] = None
    review: Optional[ACLReview] = None


class UniversityItem(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    qs_position: Optional[str] = None
    research_discovery_score: Optional[str] = None
    ranking_url: Optional[str] = None


class QSUniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


class NSFFellowshipExtraction(BaseModel):
    annual_stipend: Optional[str] = None
    prfb_deadline_2026: Optional[str] = None
    nsf_urls: List[str] = Field(default_factory=list)

    qs_partnership_definition: Optional[str] = None
    qs_partnership_urls: List[str] = Field(default_factory=list)


class DOAJRequirementsExtraction(BaseModel):
    publishing_history_requirement: Optional[str] = None
    history_urls: List[str] = Field(default_factory=list)

    cc_licenses: List[str] = Field(default_factory=list)
    license_urls: List[str] = Field(default_factory=list)

    fully_open_access_definition: Optional[str] = None
    oa_urls: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction Prompts
# -----------------------------------------------------------------------------
def prompt_extract_acl_details() -> str:
    return """
Extract ACL 2026 conference details exactly as stated in the answer. Do not infer. Return:
- paper_limits:
  - long_submission_pages: numeric string for submission page limit of long papers (e.g., "8") or null
  - long_camera_ready_pages: numeric string for camera-ready page limit of long papers (e.g., "9") or null
  - long_urls: URLs cited in the answer that specifically support long paper limits
  - short_submission_pages: numeric string for submission page limit of short papers (e.g., "4") or null
  - short_camera_ready_pages: numeric string for camera-ready page limit of short papers (e.g., "5") or null
  - short_urls: URLs cited in the answer that specifically support short paper limits
- timeline:
  - arr_submission_deadline: date string exactly as presented for ARR submission deadline (e.g., "January 5, 2026") or null
  - commitment_deadline: date string for commitment deadline to ACL 2026 or null
  - notification_date: date string for notification of acceptance or null
  - camera_ready_deadline: date string for camera-ready deadline or null
  - timeline_urls: URLs cited that support any/all of the above dates
- review:
  - review_system: e.g., "ACL Rolling Review (ARR)" or "ARR", exactly as stated, or null
  - review_anonymity: e.g., "double-blind", exactly as stated, or null
  - review_urls: URLs cited to support the review system/anonymity
Only extract URLs explicitly included in the answer. If not provided, leave the array empty.
    """.strip()


def prompt_extract_qs_universities() -> str:
    return """
From the answer, extract up to three universities that the answer claims appear in the QS World University Rankings 2026. For each, return:
- name: university name (as in the answer)
- location: city/country or region (as in the answer)
- qs_position: ranking position string as claimed (e.g., "#12" or "12") 
- research_discovery_score: the Research & Discovery lens score exactly as stated (e.g., "95.6", "95.6%", or "95.6/100")
- ranking_url: a single URL from topuniversities.com (QS site) cited for that university's 2026 ranking; if multiple provided, pick the most specific per-university page; if none, set to null
If fewer than three are provided, return as many as available.
    """.strip()


def prompt_extract_nsf_and_qs_methodology() -> str:
    return """
Extract fellowship and methodology details exactly as stated in the answer. Return:
- annual_stipend: the annual stipend amount for NSF postdoctoral fellowships as claimed (e.g., "$78,000", "78k") or null
- prfb_deadline_2026: the 2026 submission deadline for NSF PRFB as claimed (e.g., "September 29, 2026") or null
- nsf_urls: URLs from nsf.gov cited that support stipend and/or PRFB deadline
- qs_partnership_definition: the QS definition of a "sustained international research partnership" as claimed (e.g., "3 or more joint papers in a 5-year period") or null
- qs_partnership_urls: URLs from topuniversities.com cited that support the partnership definition
Only return URLs explicitly present in the answer.
    """.strip()


def prompt_extract_doaj_requirements() -> str:
    return """
Extract DOAJ-related requirements exactly as stated in the answer. Return:
- publishing_history_requirement: the requirement about publishing history or minimum number of articles (verbatim or faithfully summarized from the answer)
- history_urls: URLs from doaj.org cited to support the publishing history requirement
- cc_licenses: list of Creative Commons license codes the answer claims DOAJ accepts (e.g., ["CC BY","CC BY-NC","CC0"]); if not provided, return an empty list
- license_urls: URLs from doaj.org cited to support licensing requirements
- fully_open_access_definition: how DOAJ defines "fully open access" (verbatim or faithful from answer) or null
- oa_urls: URLs from doaj.org cited to support the fully open access requirement
Only include URLs that appear in the answer text.
    """.strip()


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _filter_urls_by_domains(urls: List[str], domain_keywords: List[str]) -> List[str]:
    if not urls:
        return []
    lowered = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
    out: List[str] = []
    for u in lowered:
        lu = u.lower()
        if any(dk in lu for dk in domain_keywords):
            out.append(u)
    return out


def _has_domain(urls: List[str], domain_keywords: List[str]) -> bool:
    return len(_filter_urls_by_domains(urls, domain_keywords)) > 0


def _coalesce_url(url: Optional[str]) -> List[str]:
    return [url] if url else []


# -----------------------------------------------------------------------------
# Verification builders
# -----------------------------------------------------------------------------
async def build_acl_conference_details(
    evaluator: Evaluator,
    parent,
    acl: ACLDetailsExtraction
) -> None:
    node_acl = evaluator.add_parallel(
        id="acl_2026_conference_details",
        desc="Accurate information about ACL 2026 submission requirements and timeline",
        parent=parent,
        critical=True
    )

    # 1) Paper format requirements
    node_format = evaluator.add_parallel(
        id="paper_format_requirements",
        desc="Correct specification of ACL 2026 paper types and page limits",
        parent=node_acl,
        critical=True
    )

    # Long paper specs
    node_long = evaluator.add_parallel(
        id="long_paper_specifications",
        desc="Specification of long paper page limits",
        parent=node_format,
        critical=True
    )

    long_urls_all = acl.paper_limits.long_urls if (acl and acl.paper_limits and acl.paper_limits.long_urls) else []
    long_acl_urls = _filter_urls_by_domains(long_urls_all, ["aclweb.org"])

    # Reference domain check (aclweb.org)
    evaluator.add_custom_node(
        result=_has_domain(long_urls_all, ["aclweb.org"]),
        id="long_paper_reference",
        desc="Provides a valid reference URL from aclweb.org domain confirming long paper requirements",
        parent=node_long,
        critical=True
    )

    # Content verification
    long_claim = (
        f"For ACL 2026, the long paper page limits are: "
        f"{acl.paper_limits.long_submission_pages or 'UNKNOWN'} pages for submission and "
        f"{acl.paper_limits.long_camera_ready_pages or 'UNKNOWN'} pages for the camera-ready version."
    )
    leaf_long_content = evaluator.add_leaf(
        id="long_paper_content",
        desc="Long papers must be specified as 8 pages for submission (9 pages for camera-ready)",
        parent=node_long,
        critical=True
    )
    await evaluator.verify(
        claim=long_claim,
        node=leaf_long_content,
        sources=long_acl_urls if long_acl_urls else long_urls_all,
        additional_instruction="Verify the exact page limits for ACL 2026 long papers. Accept numeric formatting variants, "
                               "but ensure both submission and camera-ready limits are explicitly supported by the cited page(s)."
    )

    # Short paper specs
    node_short = evaluator.add_parallel(
        id="short_paper_specifications",
        desc="Specification of short paper page limits",
        parent=node_format,
        critical=True
    )

    short_urls_all = acl.paper_limits.short_urls if (acl and acl.paper_limits and acl.paper_limits.short_urls) else []
    short_acl_urls = _filter_urls_by_domains(short_urls_all, ["aclweb.org"])

    evaluator.add_custom_node(
        result=_has_domain(short_urls_all, ["aclweb.org"]),
        id="short_paper_reference",
        desc="Provides a valid reference URL from aclweb.org domain confirming short paper requirements",
        parent=node_short,
        critical=True
    )

    short_claim = (
        f"For ACL 2026, the short paper page limits are: "
        f"{acl.paper_limits.short_submission_pages or 'UNKNOWN'} pages for submission and "
        f"{acl.paper_limits.short_camera_ready_pages or 'UNKNOWN'} pages for the camera-ready version."
    )
    leaf_short_content = evaluator.add_leaf(
        id="short_paper_content",
        desc="Short papers must be specified as 4 pages for submission (5 pages for camera-ready)",
        parent=node_short,
        critical=True
    )
    await evaluator.verify(
        claim=short_claim,
        node=leaf_short_content,
        sources=short_acl_urls if short_acl_urls else short_urls_all,
        additional_instruction="Verify the exact page limits for ACL 2026 short papers. Accept numeric formatting variants, "
                               "but ensure both submission and camera-ready limits are explicitly supported by the cited page(s)."
    )

    # 2) Submission timeline
    node_timeline = evaluator.add_parallel(
        id="submission_timeline",
        desc="Accurate reporting of ACL 2026 key dates",
        parent=node_acl,
        critical=True
    )
    timeline_urls_all = acl.timeline.timeline_urls if (acl and acl.timeline and acl.timeline.timeline_urls) else []
    timeline_acl_urls = _filter_urls_by_domains(timeline_urls_all, ["aclweb.org"])

    # Each sub-item as its own grouping node with a single leaf for clarity
    node_arr = evaluator.add_parallel(
        id="arr_submission_deadline",
        desc="ARR submission deadline specification",
        parent=node_timeline,
        critical=True
    )
    leaf_arr = evaluator.add_leaf(
        id="arr_deadline_content",
        desc="ARR submission deadline correctly stated as January 5, 2026",
        parent=node_arr,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ACL 2026 ARR submission deadline is {acl.timeline.arr_submission_deadline or 'UNKNOWN'}.",
        node=leaf_arr,
        sources=timeline_acl_urls if timeline_acl_urls else timeline_urls_all,
        additional_instruction="Match the ARR submission deadline exactly as shown on the ACL 2026 official page. "
                               "Allow minor date-format variations (e.g., abbreviations)."
    )

    node_commit = evaluator.add_parallel(
        id="commitment_deadline",
        desc="ACL 2026 commitment deadline specification",
        parent=node_timeline,
        critical=True
    )
    leaf_commit = evaluator.add_leaf(
        id="commitment_deadline_content",
        desc="ACL 2026 commitment deadline correctly stated as March 14, 2026",
        parent=node_commit,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ACL 2026 commitment deadline is {acl.timeline.commitment_deadline or 'UNKNOWN'}.",
        node=leaf_commit,
        sources=timeline_acl_urls if timeline_acl_urls else timeline_urls_all,
        additional_instruction="Verify the commitment deadline exactly as per ACL 2026 official schedule."
    )

    node_notify = evaluator.add_parallel(
        id="notification_date",
        desc="Notification of acceptance date specification",
        parent=node_timeline,
        critical=True
    )
    leaf_notify = evaluator.add_leaf(
        id="notification_date_content",
        desc="Notification of acceptance correctly stated as April 4, 2026",
        parent=node_notify,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ACL 2026 notification of acceptance date is {acl.timeline.notification_date or 'UNKNOWN'}.",
        node=leaf_notify,
        sources=timeline_acl_urls if timeline_acl_urls else timeline_urls_all,
        additional_instruction="Verify the acceptance notification date using the official ACL 2026 timeline page."
    )

    node_camera = evaluator.add_parallel(
        id="camera_ready_deadline",
        desc="Camera-ready deadline specification",
        parent=node_timeline,
        critical=True
    )
    leaf_camera = evaluator.add_leaf(
        id="camera_ready_deadline_content",
        desc="Camera-ready deadline correctly stated as April 19, 2026",
        parent=node_camera,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ACL 2026 camera-ready deadline is {acl.timeline.camera_ready_deadline or 'UNKNOWN'}.",
        node=leaf_camera,
        sources=timeline_acl_urls if timeline_acl_urls else timeline_urls_all,
        additional_instruction="Verify the camera-ready deadline on the official ACL 2026 page."
    )

    evaluator.add_custom_node(
        result=_has_domain(timeline_urls_all, ["aclweb.org"]),
        id="timeline_reference_url",
        desc="Provides a valid reference URL from aclweb.org domain confirming the timeline",
        parent=node_timeline,
        critical=True
    )

    # 3) Review process
    node_review = evaluator.add_parallel(
        id="review_process",
        desc="Correct description of ACL 2026 review mechanism",
        parent=node_acl,
        critical=True
    )
    review_urls_all = acl.review.review_urls if (acl and acl.review and acl.review.review_urls) else []

    node_arr_sys = evaluator.add_parallel(
        id="arr_review_system",
        desc="Specification of review system",
        parent=node_review,
        critical=True
    )
    leaf_arr_sys = evaluator.add_leaf(
        id="arr_system_content",
        desc="Correctly identifies that ACL 2026 uses ACL Rolling Review (ARR) as the reviewing system",
        parent=node_arr_sys,
        critical=True
    )
    await evaluator.verify(
        claim=f"ACL 2026 uses {acl.review.review_system or 'UNKNOWN'} as the reviewing system.",
        node=leaf_arr_sys,
        sources=review_urls_all,
        additional_instruction="Confirm that ACL 2026 uses ACL Rolling Review (ARR). Accept minor name variations such as 'ARR'."
    )

    node_blind = evaluator.add_parallel(
        id="double_blind_review",
        desc="Specification of review anonymity",
        parent=node_review,
        critical=True
    )
    leaf_blind = evaluator.add_leaf(
        id="double_blind_content",
        desc="Correctly states that the review process is double-blind",
        parent=node_blind,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ACL 2026 review process is {acl.review.review_anonymity or 'UNKNOWN'}.",
        node=leaf_blind,
        sources=review_urls_all,
        additional_instruction="Verify the anonymity policy (e.g., double-blind) on the official review policy or call for papers."
    )

    evaluator.add_custom_node(
        result=bool(review_urls_all and len(review_urls_all) > 0),
        id="review_process_reference_url",
        desc="Provides a valid reference URL confirming the review process details",
        parent=node_review,
        critical=True
    )


async def build_target_universities(
    evaluator: Evaluator,
    parent,
    qs: QSUniversitiesExtraction
) -> None:
    node_unis = evaluator.add_parallel(
        id="target_universities",
        desc="Identification of universities meeting specified QS ranking criteria",
        parent=parent,
        critical=False  # allow partial credit across universities
    )

    # Use at most the first 3 universities; pad with empty if fewer
    items = qs.universities[:3] if qs and qs.universities else []
    while len(items) < 3:
        items.append(UniversityItem())

    for idx, uni in enumerate(items, start=1):
        u_node = evaluator.add_parallel(
            id=f"university_{idx}",
            desc=f"{['First','Second','Third'][idx-1]} university meeting all QS ranking criteria",
            parent=node_unis,
            critical=False
        )

        attrs_node = evaluator.add_parallel(
            id=f"university_{idx}_attributes",
            desc=f"Core attributes of the {['first','second','third'][idx-1]} university",
            parent=u_node,
            critical=True
        )

        # University identification (name and location)
        leaf_ident = evaluator.add_leaf(
            id=f"university_{idx}_identification",
            desc="Clear identification of the university name and location",
            parent=attrs_node,
            critical=True
        )
        claim_ident = (
            f"The QS World University Rankings 2026 page shows the university named "
            f"'{uni.name or 'UNKNOWN'}' located in '{uni.location or 'UNKNOWN'}'."
        )
        await evaluator.verify(
            claim=claim_ident,
            node=leaf_ident,
            sources=_coalesce_url(uni.ranking_url),
            additional_instruction="Verify the university's name and location as displayed on the QS 2026 ranking page. "
                                   "Allow minor formatting differences and regional naming variants."
        )

        # QS ranking position
        leaf_pos = evaluator.add_leaf(
            id=f"university_{idx}_qs_ranking_position",
            desc="Provides the university's QS World University Rankings 2026 position",
            parent=attrs_node,
            critical=True
        )
        claim_pos = (
            f"The QS World University Rankings 2026 position for {uni.name or 'this university'} is "
            f"'{uni.qs_position or 'UNKNOWN'}'."
        )
        await evaluator.verify(
            claim=claim_pos,
            node=leaf_pos,
            sources=_coalesce_url(uni.ranking_url),
            additional_instruction="Confirm the 2026 QS position as shown for the university. "
                                   "Accept reasonable variants like '#12' vs '12'."
        )

        # Research & Discovery lens score
        leaf_rd = evaluator.add_leaf(
            id=f"university_{idx}_research_discovery_score",
            desc="Reports the university's Research & Discovery lens score (weighted 50% in QS methodology)",
            parent=attrs_node,
            critical=True
        )
        claim_rd = (
            f"The QS 2026 page shows the Research & Discovery lens score for {uni.name or 'this university'} "
            f"as '{uni.research_discovery_score or 'UNKNOWN'}'."
        )
        await evaluator.verify(
            claim=claim_rd,
            node=leaf_rd,
            sources=_coalesce_url(uni.ranking_url),
            additional_instruction="Verify the 'Research & Discovery' lens score on the QS page. "
                                   "Accept decimals and percent-like formats."
        )

        # Reference domain check (topuniversities.com)
        evaluator.add_custom_node(
            result=bool(uni.ranking_url and "topuniversities.com" in uni.ranking_url.lower()),
            id=f"university_{idx}_ranking_reference_url",
            desc="Provides a valid reference URL from topuniversities.com confirming the ranking information",
            parent=u_node,
            critical=True
        )


async def build_fellowship_opportunities(
    evaluator: Evaluator,
    parent,
    nsf_qs: NSFFellowshipExtraction
) -> None:
    node_fellow = evaluator.add_parallel(
        id="fellowship_opportunities",
        desc="Identification of postdoctoral fellowship programs meeting specified criteria",
        parent=parent,
        critical=False  # allow partial credit within this group (QS methodology is ancillary)
    )

    # NSF details
    node_nsf = evaluator.add_parallel(
        id="nsf_fellowship_details",
        desc="Accurate information about NSF postdoctoral fellowship programs",
        parent=node_fellow,
        critical=True
    )

    # Program specifications (stipend + PRFB deadline)
    node_prog = evaluator.add_parallel(
        id="program_specifications",
        desc="NSF fellowship program specifications",
        parent=node_nsf,
        critical=True
    )

    # Annual stipend
    leaf_stipend = evaluator.add_leaf(
        id="annual_stipend",
        desc="NSF postdoctoral fellowship annual stipend correctly stated as $78,000",
        parent=node_prog,
        critical=True
    )
    await evaluator.verify(
        claim=f"The annual stipend amount for NSF postdoctoral fellowships is {nsf_qs.annual_stipend or 'UNKNOWN'}.",
        node=leaf_stipend,
        sources=nsf_qs.nsf_urls,
        additional_instruction="Verify the stipend on nsf.gov pages. Accept equivalent monetary formatting (e.g., $78k ≈ $78,000)."
    )

    # PRFB deadline 2026
    leaf_prfb = evaluator.add_leaf(
        id="prfb_deadline",
        desc="NSF PRFB (Postdoctoral Research Fellowships in Biology) deadline correctly stated as September 29, 2026",
        parent=node_prog,
        critical=True
    )
    await evaluator.verify(
        claim=f"The submission deadline for the NSF PRFB program in 2026 is {nsf_qs.prfb_deadline_2026 or 'UNKNOWN'}.",
        node=leaf_prfb,
        sources=nsf_qs.nsf_urls,
        additional_instruction="Verify the 2026 PRFB deadline on nsf.gov. Minor time/zone annotations are acceptable."
    )

    # Reference domain check (nsf.gov)
    evaluator.add_custom_node(
        result=_has_domain(nsf_qs.nsf_urls or [], ["nsf.gov"]),
        id="fellowship_reference_url",
        desc="Provides a valid reference URL from nsf.gov domain confirming fellowship details",
        parent=node_nsf,
        critical=True
    )

    # QS methodology - sustained partnerships
    node_internat = evaluator.add_parallel(
        id="international_research_network_criteria",
        desc="Understanding of sustained international research partnerships",
        parent=node_fellow,
        critical=False
    )

    node_partner = evaluator.add_parallel(
        id="partnership_specification",
        desc="QS partnership definition specification",
        parent=node_internat,
        critical=True
    )

    leaf_partner = evaluator.add_leaf(
        id="partnership_definition",
        desc="Correctly defines sustained partnerships as those resulting in 3 or more joint papers in a 5-year period (per QS methodology)",
        parent=node_partner,
        critical=True
    )
    await evaluator.verify(
        claim=f"According to QS methodology, a sustained international research partnership is defined as: "
              f"{nsf_qs.qs_partnership_definition or 'UNKNOWN'}.",
        node=leaf_partner,
        sources=nsf_qs.qs_partnership_urls,
        additional_instruction="Verify the QS methodology definition on topuniversities.com. "
                               "Accept equivalent wordings that clearly mean ≥3 joint papers within 5 years."
    )

    evaluator.add_custom_node(
        result=_has_domain(nsf_qs.qs_partnership_urls or [], ["topuniversities.com"]),
        id="partnership_reference_url",
        desc="Provides a valid reference URL from topuniversities.com confirming the partnership definition",
        parent=node_internat,
        critical=True
    )


async def build_open_access_requirements(
    evaluator: Evaluator,
    parent,
    doaj: DOAJRequirementsExtraction
) -> None:
    node_oa = evaluator.add_parallel(
        id="open_access_journal_requirements",
        desc="Understanding of DOAJ indexing requirements for open access journals",
        parent=parent,
        critical=True
    )

    # Publishing history requirement
    node_hist = evaluator.add_parallel(
        id="publishing_history_requirement",
        desc="Correct statement of DOAJ publishing history requirement",
        parent=node_oa,
        critical=True
    )
    node_hist_spec = evaluator.add_parallel(
        id="history_specification",
        desc="DOAJ publishing history specification",
        parent=node_hist,
        critical=True
    )
    leaf_hist = evaluator.add_leaf(
        id="one_year_history_or_ten_articles",
        desc="Correctly states that journals must have publishing history of more than one year OR have published at least 10 open access research articles",
        parent=node_hist_spec,
        critical=True
    )
    await evaluator.verify(
        claim=f"DOAJ requires that journals have the following publishing history or article count: "
              f"{doaj.publishing_history_requirement or 'UNKNOWN'}.",
        node=leaf_hist,
        sources=doaj.history_urls,
        additional_instruction="Verify on doaj.org that the publishing history criterion is either more than one year of publishing "
                               "or at least 10 open access research articles. Accept equivalent, explicit statements."
    )
    evaluator.add_custom_node(
        result=_has_domain(doaj.history_urls or [], ["doaj.org"]),
        id="doaj_reference_url",
        desc="Provides a valid reference URL from doaj.org domain confirming the requirement",
        parent=node_hist,
        critical=True
    )

    # Licensing requirements
    node_lic = evaluator.add_parallel(
        id="licensing_requirements",
        desc="Correct specification of DOAJ licensing requirements",
        parent=node_oa,
        critical=True
    )
    node_lic_spec = evaluator.add_parallel(
        id="license_specification",
        desc="DOAJ license specification",
        parent=node_lic,
        critical=True
    )
    leaf_cc = evaluator.add_leaf(
        id="creative_commons_licenses",
        desc="Correctly identifies that DOAJ requires Creative Commons licensing (CC BY, CC BY-NC, or CC0)",
        parent=node_lic_spec,
        critical=True
    )
    licenses_list = doaj.cc_licenses or []
    licenses_str = ", ".join(licenses_list) if licenses_list else "UNKNOWN"
    await evaluator.verify(
        claim=f"DOAJ accepts the following Creative Commons licenses for indexed journals: {licenses_str}.",
        node=leaf_cc,
        sources=doaj.license_urls,
        additional_instruction="Verify on doaj.org the acceptable CC licenses for journals (e.g., CC BY, CC BY-NC, CC0). "
                               "Allow minor formatting variants (hyphens, spaces)."
    )
    evaluator.add_custom_node(
        result=_has_domain(doaj.license_urls or [], ["doaj.org"]),
        id="licensing_reference_url",
        desc="Provides a valid reference URL from doaj.org domain confirming licensing requirements",
        parent=node_lic,
        critical=True
    )

    # Fully open access definition
    node_def = evaluator.add_parallel(
        id="open_access_definition",
        desc="Understanding that DOAJ only indexes fully open access journals",
        parent=node_oa,
        critical=True
    )
    node_def_spec = evaluator.add_parallel(
        id="oa_specification",
        desc="DOAJ open access specification",
        parent=node_def,
        critical=True
    )
    leaf_foa = evaluator.add_leaf(
        id="fully_open_access",
        desc="Correctly states that journals must be fully open access (all content immediately and freely available)",
        parent=node_def_spec,
        critical=True
    )
    await evaluator.verify(
        claim=f"DOAJ defines eligible journals as fully open access such that: "
              f"{doaj.fully_open_access_definition or 'UNKNOWN'}.",
        node=leaf_foa,
        sources=doaj.oa_urls,
        additional_instruction="Verify on doaj.org that 'fully open access' means all content is immediately and freely available."
    )
    evaluator.add_custom_node(
        result=_has_domain(doaj.oa_urls or [], ["doaj.org"]),
        id="oa_reference_url",
        desc="Provides a valid reference URL from doaj.org domain confirming the open access requirement",
        parent=node_def,
        critical=True
    )


# -----------------------------------------------------------------------------
# Main evaluation entrypoint
# -----------------------------------------------------------------------------
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
        strategy=AggregationStrategy.PARALLEL,   # root parallel; allow independent credit across sections
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

    # Top-level container node (non-critical to avoid critical-child constraint issues across mixed subtrees)
    top = evaluator.add_parallel(
        id="research_planning_2026",
        desc="Comprehensive research planning for ACL 2026 submission including conference requirements, target universities, fellowship opportunities, and publication venues",
        parent=root,
        critical=False
    )

    # Run extractions (can be parallelized)
    acl_task = evaluator.extract(
        prompt=prompt_extract_acl_details(),
        template_class=ACLDetailsExtraction,
        extraction_name="acl_details"
    )
    qs_task = evaluator.extract(
        prompt=prompt_extract_qs_universities(),
        template_class=QSUniversitiesExtraction,
        extraction_name="qs_universities"
    )
    nsf_task = evaluator.extract(
        prompt=prompt_extract_nsf_and_qs_methodology(),
        template_class=NSFFellowshipExtraction,
        extraction_name="nsf_and_qs_methodology"
    )
    doaj_task = evaluator.extract(
        prompt=prompt_extract_doaj_requirements(),
        template_class=DOAJRequirementsExtraction,
        extraction_name="doaj_requirements"
    )

    acl_details, qs_unis, nsf_qs, doaj_reqs = await asyncio.gather(acl_task, qs_task, nsf_task, doaj_task)

    # Build section verifications
    await build_acl_conference_details(evaluator, top, acl_details or ACLDetailsExtraction())
    await build_target_universities(evaluator, top, qs_unis or QSUniversitiesExtraction())
    await build_fellowship_opportunities(evaluator, top, nsf_qs or NSFFellowshipExtraction())
    await build_open_access_requirements(evaluator, top, doaj_reqs or DOAJRequirementsExtraction())

    return evaluator.get_summary()