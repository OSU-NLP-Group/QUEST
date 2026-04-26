import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.verification_tree import VerificationNode


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "conference_selection_2026"
TASK_DESCRIPTION = """A research group has four papers at different stages of completion that they plan to submit to academic conferences in 2026. For each paper, identify one suitable conference that meets all specified requirements.

Paper A (Short Paper, 4 pages): A completed technical contribution requiring rapid publication with early 2026 submission deadline. Requirements: (1) Accepts short papers of 4 pages in IEEE or ACM format, (2) Has submission deadline in January-February 2026, (3) Provides review decisions within 8-12 weeks, (4) Publishes proceedings in an indexed digital library (IEEE Xplore, ACM Digital Library, or equivalent), (5) Offers 10-15 minute oral presentation slots.

Paper B (Full Paper, 8-10 pages): A comprehensive research study requiring open-access publication and flexible submission timeline. Requirements: (1) Accepts full papers of 8-10 pages, (2) Provides open-access publication option, (3) Has submission deadline between February-May 2026, (4) Requires abstract submission (200-300 words) before or with full paper, (5) Conference takes place in second half of 2026 (July-December).

Paper C (Regular Paper, 6 pages): Work-in-progress needing constructive peer feedback with mid-2026 conference dates. Requirements: (1) Accepts regular papers of 6 pages, (2) Uses double-blind peer review process, (3) Has submission deadline in March-April 2026, (4) Conference occurs between June-September 2026, (5) Provides detailed review feedback within 10-14 weeks, (6) Publishes proceedings in indexed databases.

Paper D (Extended Abstract, 2-3 pages): Preliminary findings suitable for spring 2026 presentation with poster session option. Requirements: (1) Accepts extended abstracts or short papers of 2-4 pages, (2) Offers poster presentation option, (3) Conference dates in April-June 2026, (4) Has submission deadline at least 2 months (60 days) before conference start date, (5) Does not require full paper for abstract submissions.

For each of the four papers, provide: (1) Conference name and acronym, (2) Conference dates in 2026, (3) Submission deadline, (4) Conference location or format, (5) URL to call for papers or submission guidelines demonstrating that all requirements are met.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PaperInfo(BaseModel):
    conference_name: Optional[str] = None
    acronym: Optional[str] = None
    conference_dates: Optional[str] = None
    submission_deadline: Optional[str] = None
    location_or_format: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class ConferenceSelections(BaseModel):
    paper_a: Optional[PaperInfo] = None
    paper_b: Optional[PaperInfo] = None
    paper_c: Optional[PaperInfo] = None
    paper_d: Optional[PaperInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conferences() -> str:
    return """
Extract the conference selection details for four papers (Paper A, Paper B, Paper C, Paper D) as presented in the answer. For each paper, extract ONLY the first clearly associated conference and its details. If multiple conferences are mentioned for a paper, select the first one that the answer explicitly links to that paper.

For each paper, extract:
- conference_name: The official conference name (string)
- acronym: The conference acronym (string), if available
- conference_dates: The specific 2026 conference dates or month range as provided in the answer (string)
- submission_deadline: The submission deadline date or date range as provided in the answer (string)
- location_or_format: The specified location (city, country) or "virtual"/"hybrid" if stated (string)
- source_urls: An array of all URLs in the answer that are specifically used to justify this paper's conference choice (e.g., CFP pages, author guidelines, submission information). Include every relevant URL mentioned for that paper.

Important:
- Extract URLs exactly as they appear (accept plain or markdown links).
- Do not invent or infer any URLs; only include those explicitly shown in the answer.
- If any field is missing, set it to null (use an empty array for source_urls if none).

Return a JSON object with keys: paper_a, paper_b, paper_c, paper_d, each an object with the fields listed above.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _urls(info: Optional[PaperInfo]) -> List[str]:
    if not info or not info.source_urls:
        return []
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in info.source_urls:
        if isinstance(u, str):
            uu = u.strip()
            if uu and uu not in seen:
                seen.add(uu)
                out.append(uu)
    return out


async def _add_and_verify(
    evaluator: Evaluator,
    *,
    parent: VerificationNode,
    node_id: str,
    desc: str,
    claim: str,
    sources: List[str],
    critical: bool,
    additional_instruction: str,
    extra_prereq_nodes: Optional[List[VerificationNode]] = None
) -> VerificationNode:
    leaf = evaluator.add_leaf(id=node_id, desc=desc, parent=parent, critical=critical)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources if sources else None,
        additional_instruction=additional_instruction,
        extra_prerequisites=extra_prereq_nodes or []
    )
    return leaf


def _add_url_presence_gate(
    evaluator: Evaluator,
    *,
    parent: VerificationNode,
    node_id: str,
    desc: str,
    info: Optional[PaperInfo]
) -> VerificationNode:
    urls = _urls(info)
    return evaluator.add_custom_node(
        result=bool(urls),
        id=node_id,
        desc=desc,
        parent=parent,
        critical=True
    )


def _add_existence_node(
    evaluator: Evaluator,
    *,
    parent: VerificationNode,
    node_id: str,
    desc: str,
    condition: bool,
    critical: bool
) -> VerificationNode:
    return evaluator.add_custom_node(
        result=bool(condition),
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )


# --------------------------------------------------------------------------- #
# Verification builders for each paper                                        #
# --------------------------------------------------------------------------- #
async def verify_paper_a(evaluator: Evaluator, parent: VerificationNode, info: Optional[PaperInfo]) -> None:
    paper_node = evaluator.add_parallel(
        id="Paper_A_Conference_Match",
        desc="Suitable conference identified for Paper A (4-page short paper, early 2026)",
        parent=parent,
        critical=False
    )

    urls = _urls(info)

    # A.1 Submission Requirements (critical)
    sub_req = evaluator.add_parallel(
        id="Paper_A_Submission_Requirements",
        desc="Conference meets Paper A submission format and timeline requirements",
        parent=paper_node,
        critical=True
    )
    sub_req_url = _add_url_presence_gate(
        evaluator,
        parent=sub_req,
        node_id="Paper_A_Submission_URL",
        desc="URL reference documenting submission requirements",
        info=info
    )

    await _add_and_verify(
        evaluator,
        parent=sub_req,
        node_id="Paper_A_Page_Limit",
        desc="Conference accepts short papers of exactly 4 pages or allows 4-page submissions in short paper category",
        claim="The conference explicitly accepts short papers of 4 pages (exactly 4 pages or a short-paper track that allows 4 pages, excluding references if commonly excluded).",
        sources=urls,
        critical=True,
        additional_instruction="Look for 'short paper' or 'short submission' length policies; accept if 4 pages is allowed as a maximum or exact length for the short-paper category; minor variants like '4 pages + references' are acceptable.",
        extra_prereq_nodes=[sub_req_url]
    )

    await _add_and_verify(
        evaluator,
        parent=sub_req,
        node_id="Paper_A_Format_Standard",
        desc="Conference uses IEEE or ACM standard format templates",
        claim="The conference requires the IEEE or ACM formatting/templates for submissions.",
        sources=urls,
        critical=True,
        additional_instruction="Accept IEEE or ACM format (including ACM SIGCONF, sig-alternate, or IEEE conference templates).",
        extra_prereq_nodes=[sub_req_url]
    )

    await _add_and_verify(
        evaluator,
        parent=sub_req,
        node_id="Paper_A_Deadline_Timing",
        desc="Submission deadline falls between January 1 and February 28, 2026",
        claim="According to the official pages, the submission deadline is between January 1 and February 28, 2026.",
        sources=urls,
        critical=True,
        additional_instruction="Check key dates/CFP for the deadline; accept if any listed deadline relevant to paper submission lies between 2026-01-01 and 2026-02-28.",
        extra_prereq_nodes=[sub_req_url]
    )

    # A.2 Review Process (critical)
    review = evaluator.add_parallel(
        id="Paper_A_Review_Process",
        desc="Conference review process meets Paper A timeline needs",
        parent=paper_node,
        critical=True
    )
    review_url = _add_url_presence_gate(
        evaluator,
        parent=review,
        node_id="Paper_A_Review_URL",
        desc="URL reference documenting review timeline",
        info=info
    )

    await _add_and_verify(
        evaluator,
        parent=review,
        node_id="Paper_A_Review_Timeline",
        desc="Conference provides review decisions within 8-12 weeks of submission deadline",
        claim="The conference indicates that review decisions are provided within approximately 8 to 12 weeks from the submission deadline.",
        sources=urls,
        critical=True,
        additional_instruction="Look for explicit wording of review timeline, decision notification dates vs submission deadlines; allow reasonable phrasing such as 'about 2–3 months' or '8–12 weeks'.",
        extra_prereq_nodes=[review_url]
    )

    await _add_and_verify(
        evaluator,
        parent=review,
        node_id="Paper_A_Timeline_Documentation",
        desc="Review timeline is explicitly stated on conference website or call for papers",
        claim="The conference website or call-for-papers page explicitly states the review decision timeline.",
        sources=urls,
        critical=True,
        additional_instruction="Verify that an explicit statement exists (not inferred) about when decisions will be communicated.",
        extra_prereq_nodes=[review_url]
    )

    # A.3 Publication Standards (critical)
    pub = evaluator.add_parallel(
        id="Paper_A_Publication_Standards",
        desc="Conference meets Paper A publication and indexing requirements",
        parent=paper_node,
        critical=True
    )
    pub_url = _add_url_presence_gate(
        evaluator,
        parent=pub,
        node_id="Paper_A_Publication_URL",
        desc="URL reference documenting proceedings publication and indexing",
        info=info
    )

    await _add_and_verify(
        evaluator,
        parent=pub,
        node_id="Paper_A_Proceedings_Publication",
        desc="Conference publishes accepted papers in formal proceedings",
        claim="Accepted papers are published in formal conference proceedings.",
        sources=urls,
        critical=True,
        additional_instruction="Accept if proceedings are published through recognized publishers (IEEE, ACM, Springer, etc.).",
        extra_prereq_nodes=[pub_url]
    )

    await _add_and_verify(
        evaluator,
        parent=pub,
        node_id="Paper_A_Digital_Indexing",
        desc="Proceedings are published in an indexed digital library (IEEE Xplore, ACM Digital Library, or equivalent)",
        claim="Proceedings are included in a well-known indexed digital library, such as IEEE Xplore or the ACM Digital Library (or an equivalent indexed database).",
        sources=urls,
        critical=True,
        additional_instruction="Accept equivalent indexing (e.g., Springer LNCS indexed, Scopus/DBLP indexing mentions that clearly imply indexed discoverability).",
        extra_prereq_nodes=[pub_url]
    )

    # A.4 Presentation Format (critical)
    pres = evaluator.add_parallel(
        id="Paper_A_Presentation_Format",
        desc="Conference presentation format meets Paper A requirements",
        parent=paper_node,
        critical=True
    )
    pres_url = _add_url_presence_gate(
        evaluator,
        parent=pres,
        node_id="Paper_A_Presentation_URL",
        desc="URL reference documenting presentation format and duration",
        info=info
    )

    await _add_and_verify(
        evaluator,
        parent=pres,
        node_id="Paper_A_Presentation_Duration",
        desc="Oral presentations are allocated 10-15 minutes per paper",
        claim="Oral presentation slots are within 10 to 15 minutes per paper.",
        sources=urls,
        critical=True,
        additional_instruction="Look for presentation guidelines or program format; allow phrasing like 'approximately 10–15 minutes'.",
        extra_prereq_nodes=[pres_url]
    )

    await _add_and_verify(
        evaluator,
        parent=pres,
        node_id="Paper_A_Presentation_Type",
        desc="Conference offers oral presentation sessions for short papers",
        claim="The conference offers oral presentation sessions for short papers.",
        sources=urls,
        critical=True,
        additional_instruction="Accept if short papers can be presented orally (not only poster-only).",
        extra_prereq_nodes=[pres_url]
    )

    # A.5 Basic conference details (non-critical, existence checks)
    details = evaluator.add_parallel(
        id="Paper_A_Conference_Details",
        desc="Basic conference information for Paper A",
        parent=paper_node,
        critical=False
    )
    _add_existence_node(
        evaluator,
        parent=details,
        node_id="Paper_A_Conference_Name",
        desc="Official conference name and acronym provided",
        condition=(info is not None and bool(info.conference_name) and bool(info.acronym)),
        critical=False
    )
    _add_existence_node(
        evaluator,
        parent=details,
        node_id="Paper_A_Conference_Dates",
        desc="Conference dates in 2026 provided",
        condition=(info is not None and bool(info.conference_dates) and "2026" in (info.conference_dates or "")),
        critical=False
    )
    _add_existence_node(
        evaluator,
        parent=details,
        node_id="Paper_A_Conference_Location",
        desc="Conference location (city, country) or virtual format specified",
        condition=(info is not None and bool(info.location_or_format)),
        critical=False
    )


async def verify_paper_b(evaluator: Evaluator, parent: VerificationNode, info: Optional[PaperInfo]) -> None:
    paper_node = evaluator.add_parallel(
        id="Paper_B_Conference_Match",
        desc="Suitable conference identified for Paper B (8-10 page full paper, open access)",
        parent=parent,
        critical=False
    )

    urls = _urls(info)

    # B.1 Submission Requirements (critical)
    sub_req = evaluator.add_parallel(
        id="Paper_B_Submission_Requirements",
        desc="Conference meets Paper B submission format and timeline requirements",
        parent=paper_node,
        critical=True
    )
    sub_req_url = _add_url_presence_gate(
        evaluator,
        parent=sub_req,
        node_id="Paper_B_Submission_URL",
        desc="URL reference documenting submission requirements",
        info=info
    )

    await _add_and_verify(
        evaluator,
        parent=sub_req,
        node_id="Paper_B_Page_Limit",
        desc="Conference accepts full papers of 8-10 pages",
        claim="The conference accepts full papers with a length in the range of 8 to 10 pages (excluding references if commonly excluded).",
        sources=urls,
        critical=True,
        additional_instruction="Accept if guidelines specify 8–10 pages or a range that includes 8–10 pages as the allowed length for full papers.",
        extra_prereq_nodes=[sub_req_url]
    )

    await _add_and_verify(
        evaluator,
        parent=sub_req,
        node_id="Paper_B_Deadline_Timing",
        desc="Submission deadline falls between February 1 and May 31, 2026",
        claim="According to official pages, the submission deadline is between February 1 and May 31, 2026.",
        sources=urls,
        critical=True,
        additional_instruction="Check CFP key dates for a deadline within 2026-02-01 to 2026-05-31.",
        extra_prereq_nodes=[sub_req_url]
    )

    await _add_and_verify(
        evaluator,
        parent=sub_req,
        node_id="Paper_B_Abstract_Requirement",
        desc="Conference requires abstract submission (200-300 words) before or with full paper",
        claim="The conference requires an abstract submission of roughly 200–300 words, either prior to or together with the full paper submission.",
        sources=urls,
        critical=True,
        additional_instruction="Accept if the abstract length requirement is around 200–300 words (e.g., 150–300 or ≤300 words), and if it is required at or before full paper submission.",
        extra_prereq_nodes=[sub_req_url]
    )

    # B.2 Conference Timing (critical)
    timing = evaluator.add_parallel(
        id="Paper_B_Conference_Timing",
        desc="Conference dates meet Paper B timeline requirements",
        parent=paper_node,
        critical=True
    )
    timing_url = _add_url_presence_gate(
        evaluator,
        parent=timing,
        node_id="Paper_B_Timing_URL",
        desc="URL reference documenting conference dates",
        info=info
    )

    await _add_and_verify(
        evaluator,
        parent=timing,
        node_id="Paper_B_Conference_Period",
        desc="Conference takes place between July 1 and December 31, 2026",
        claim="The conference dates fall between July 1 and December 31, 2026.",
        sources=urls,
        critical=True,
        additional_instruction="Verify the scheduled conference dates/months occur in the second half of 2026.",
        extra_prereq_nodes=[timing_url]
    )

    # B.3 Publication Standards (critical)
    pub = evaluator.add_parallel(
        id="Paper_B_Publication_Standards",
        desc="Conference meets Paper B open access and publication requirements",
        parent=paper_node,
        critical=True
    )
    pub_url = _add_url_presence_gate(
        evaluator,
        parent=pub,
        node_id="Paper_B_Publication_URL",
        desc="URL reference documenting open access and proceedings",
        info=info
    )

    await _add_and_verify(
        evaluator,
        parent=pub,
        node_id="Paper_B_Open_Access",
        desc="Conference provides open-access publication option for accepted papers",
        claim="The conference offers an open-access publication option for accepted papers (e.g., gold OA, hybrid OA, or free OA).",
        sources=urls,
        critical=True,
        additional_instruction="Accept hybrid or optional OA where authors can choose to make papers open access (fees may apply).",
        extra_prereq_nodes=[pub_url]
    )

    await _add_and_verify(
        evaluator,
        parent=pub,
        node_id="Paper_B_Proceedings_Publication",
        desc="Conference publishes formal proceedings",
        claim="Accepted papers are published in formal conference proceedings.",
        sources=urls,
        critical=True,
        additional_instruction="Proceedings through recognized publishers (IEEE/ACM/Springer/etc.) are acceptable.",
        extra_prereq_nodes=[pub_url]
    )

    # B.4 Format Compliance (critical)
    fmt = evaluator.add_parallel(
        id="Paper_B_Format_Compliance",
        desc="Conference format requirements are compatible with Paper B",
        parent=paper_node,
        critical=True
    )
    fmt_url = _add_url_presence_gate(
        evaluator,
        parent=fmt,
        node_id="Paper_B_Format_URL",
        desc="URL reference documenting format requirements",
        info=info
    )

    await _add_and_verify(
        evaluator,
        parent=fmt,
        node_id="Paper_B_Format_Standard",
        desc="Conference uses standard academic format (IEEE, ACM, Springer, or similar)",
        claim="The conference uses a standard academic manuscript format such as IEEE, ACM, Springer LNCS, or similar.",
        sources=urls,
        critical=True,
        additional_instruction="Accept any mainstream standard format template and style guidelines.",
        extra_prereq_nodes=[fmt_url]
    )

    await _add_and_verify(
        evaluator,
        parent=fmt,
        node_id="Paper_B_Template_Availability",
        desc="Conference provides downloadable format templates",
        claim="The conference provides downloadable author templates for the required format.",
        sources=urls,
        critical=True,
        additional_instruction="Templates may be hosted externally (e.g., IEEE/ACM/Springer sites) but must be referenced.",
        extra_prereq_nodes=[fmt_url]
    )

    # B.5 Basic conference details (non-critical)
    details = evaluator.add_parallel(
        id="Paper_B_Conference_Details",
        desc="Basic conference information for Paper B",
        parent=paper_node,
        critical=False
    )
    _add_existence_node(
        evaluator,
        parent=details,
        node_id="Paper_B_Conference_Name",
        desc="Official conference name and acronym provided",
        condition=(info is not None and bool(info.conference_name) and bool(info.acronym)),
        critical=False
    )
    _add_existence_node(
        evaluator,
        parent=details,
        node_id="Paper_B_Conference_Dates",
        desc="Specific conference dates in 2026 provided",
        condition=(info is not None and bool(info.conference_dates) and "2026" in (info.conference_dates or "")),
        critical=False
    )
    _add_existence_node(
        evaluator,
        parent=details,
        node_id="Paper_B_Conference_Location",
        desc="Conference location or virtual format specified",
        condition=(info is not None and bool(info.location_or_format)),
        critical=False
    )


async def verify_paper_c(evaluator: Evaluator, parent: VerificationNode, info: Optional[PaperInfo]) -> None:
    paper_node = evaluator.add_parallel(
        id="Paper_C_Conference_Match",
        desc="Suitable conference identified for Paper C (6-page regular paper, mid-2026)",
        parent=parent,
        critical=False
    )

    urls = _urls(info)

    # C.1 Submission Requirements (critical)
    sub_req = evaluator.add_parallel(
        id="Paper_C_Submission_Requirements",
        desc="Conference meets Paper C submission format and timeline requirements",
        parent=paper_node,
        critical=True
    )
    sub_req_url = _add_url_presence_gate(
        evaluator,
        parent=sub_req,
        node_id="Paper_C_Submission_URL",
        desc="URL reference documenting submission requirements",
        info=info
    )

    await _add_and_verify(
        evaluator,
        parent=sub_req,
        node_id="Paper_C_Page_Limit",
        desc="Conference accepts regular papers of exactly 6 pages or 6 pages within acceptable range",
        claim="The conference accepts regular papers at a length of 6 pages, or a range that includes 6 pages.",
        sources=urls,
        critical=True,
        additional_instruction="Accept '6 pages + references' or ranges including 6 pages (e.g., 5–6 or 6–8 for 'regular' category if 6 pages is permitted).",
        extra_prereq_nodes=[sub_req_url]
    )

    await _add_and_verify(
        evaluator,
        parent=sub_req,
        node_id="Paper_C_Deadline_Timing",
        desc="Submission deadline falls between March 1 and April 30, 2026",
        claim="According to official pages, the submission deadline is between March 1 and April 30, 2026.",
        sources=urls,
        critical=True,
        additional_instruction="Check CFP key dates for a deadline within 2026-03-01 to 2026-04-30.",
        extra_prereq_nodes=[sub_req_url]
    )

    # C.2 Review Process (critical)
    review = evaluator.add_parallel(
        id="Paper_C_Review_Process",
        desc="Conference review process meets Paper C quality feedback requirements",
        parent=paper_node,
        critical=True
    )
    review_url = _add_url_presence_gate(
        evaluator,
        parent=review,
        node_id="Paper_C_Review_URL",
        desc="URL reference documenting review process and timeline",
        info=info
    )

    await _add_and_verify(
        evaluator,
        parent=review,
        node_id="Paper_C_Review_Type",
        desc="Conference uses double-blind peer review process",
        claim="The conference uses a double-blind peer review process.",
        sources=urls,
        critical=True,
        additional_instruction="Look for explicit mention of double-blind review policy.",
        extra_prereq_nodes=[review_url]
    )

    await _add_and_verify(
        evaluator,
        parent=review,
        node_id="Paper_C_Review_Timeline",
        desc="Conference provides review decisions within 10-14 weeks of submission",
        claim="The conference indicates decisions are provided within approximately 10 to 14 weeks from submission.",
        sources=urls,
        critical=True,
        additional_instruction="Allow phrasing like 'around 10–14 weeks' or 'about 2.5–3.5 months'.",
        extra_prereq_nodes=[review_url]
    )

    await _add_and_verify(
        evaluator,
        parent=review,
        node_id="Paper_C_Feedback_Quality",
        desc="Conference is documented to provide detailed review feedback (not just accept/reject)",
        claim="The conference provides detailed reviewer feedback (not just accept/reject decisions).",
        sources=urls,
        critical=True,
        additional_instruction="Look for language indicating detailed comments or constructive feedback from reviewers.",
        extra_prereq_nodes=[review_url]
    )

    # C.3 Conference Timing (critical)
    timing = evaluator.add_parallel(
        id="Paper_C_Conference_Timing",
        desc="Conference dates meet Paper C timeline requirements",
        parent=paper_node,
        critical=True
    )
    timing_url = _add_url_presence_gate(
        evaluator,
        parent=timing,
        node_id="Paper_C_Timing_URL",
        desc="URL reference documenting conference dates",
        info=info
    )

    await _add_and_verify(
        evaluator,
        parent=timing,
        node_id="Paper_C_Conference_Period",
        desc="Conference takes place between June 1 and September 30, 2026",
        claim="The conference dates fall between June 1 and September 30, 2026.",
        sources=urls,
        critical=True,
        additional_instruction="Verify scheduled conference dates occur within mid-2026 (June–September).",
        extra_prereq_nodes=[timing_url]
    )

    # C.4 Publication Standards (critical)
    pub = evaluator.add_parallel(
        id="Paper_C_Publication_Standards",
        desc="Conference meets Paper C publication requirements",
        parent=paper_node,
        critical=True
    )
    pub_url = _add_url_presence_gate(
        evaluator,
        parent=pub,
        node_id="Paper_C_Publication_URL",
        desc="URL reference documenting publication and indexing",
        info=info
    )

    await _add_and_verify(
        evaluator,
        parent=pub,
        node_id="Paper_C_Proceedings_Publication",
        desc="Conference publishes formal proceedings for regular papers",
        claim="Accepted regular papers are published in formal conference proceedings.",
        sources=urls,
        critical=True,
        additional_instruction="Recognized publishers acceptable (IEEE/ACM/Springer/etc.).",
        extra_prereq_nodes=[pub_url]
    )

    await _add_and_verify(
        evaluator,
        parent=pub,
        node_id="Paper_C_Indexing",
        desc="Proceedings are indexed in academic databases",
        claim="The proceedings are indexed in established academic databases.",
        sources=urls,
        critical=True,
        additional_instruction="Accept indexing mentions such as IEEE Xplore, ACM DL, Springer indexed series, Scopus, DBLP, etc.",
        extra_prereq_nodes=[pub_url]
    )

    # C.5 Basic conference details (non-critical)
    details = evaluator.add_parallel(
        id="Paper_C_Conference_Details",
        desc="Basic conference information for Paper C",
        parent=paper_node,
        critical=False
    )
    _add_existence_node(
        evaluator,
        parent=details,
        node_id="Paper_C_Conference_Name",
        desc="Official conference name and acronym provided",
        condition=(info is not None and bool(info.conference_name) and bool(info.acronym)),
        critical=False
    )
    _add_existence_node(
        evaluator,
        parent=details,
        node_id="Paper_C_Conference_Dates",
        desc="Specific conference dates in 2026 provided",
        condition=(info is not None and bool(info.conference_dates) and "2026" in (info.conference_dates or "")),
        critical=False
    )
    _add_existence_node(
        evaluator,
        parent=details,
        node_id="Paper_C_Conference_Location",
        desc="Conference location or virtual format specified",
        condition=(info is not None and bool(info.location_or_format)),
        critical=False
    )


async def verify_paper_d(evaluator: Evaluator, parent: VerificationNode, info: Optional[PaperInfo]) -> None:
    paper_node = evaluator.add_parallel(
        id="Paper_D_Conference_Match",
        desc="Suitable conference identified for Paper D (2-3 page extended abstract, spring 2026)",
        parent=parent,
        critical=False
    )

    urls = _urls(info)

    # D.1 Submission Requirements (critical)
    sub_req = evaluator.add_parallel(
        id="Paper_D_Submission_Requirements",
        desc="Conference meets Paper D submission format requirements",
        parent=paper_node,
        critical=True
    )
    sub_req_url = _add_url_presence_gate(
        evaluator,
        parent=sub_req,
        node_id="Paper_D_Submission_URL",
        desc="URL reference documenting submission requirements",
        info=info
    )

    await _add_and_verify(
        evaluator,
        parent=sub_req,
        node_id="Paper_D_Abstract_Length",
        desc="Conference accepts extended abstracts or short papers of 2-4 pages",
        claim="The conference accepts extended abstracts or short papers of length 2 to 4 pages.",
        sources=urls,
        critical=True,
        additional_instruction="Accept if guidelines explicitly allow 2–4 pages for extended abstracts or short papers.",
        extra_prereq_nodes=[sub_req_url]
    )

    await _add_and_verify(
        evaluator,
        parent=sub_req,
        node_id="Paper_D_No_Full_Paper_Requirement",
        desc="Conference does not require full paper submission for abstract-only submissions",
        claim="The conference does not require a full paper for extended-abstract-only submissions.",
        sources=urls,
        critical=True,
        additional_instruction="Look for explicit statements that extended abstracts suffice for submission and presentation without a full paper.",
        extra_prereq_nodes=[sub_req_url]
    )

    # D.2 Presentation Format (critical)
    pres = evaluator.add_parallel(
        id="Paper_D_Presentation_Format",
        desc="Conference presentation format meets Paper D requirements",
        parent=paper_node,
        critical=True
    )
    pres_url = _add_url_presence_gate(
        evaluator,
        parent=pres,
        node_id="Paper_D_Presentation_URL",
        desc="URL reference documenting presentation options",
        info=info
    )

    await _add_and_verify(
        evaluator,
        parent=pres,
        node_id="Paper_D_Poster_Option",
        desc="Conference offers poster presentation sessions",
        claim="The conference offers poster presentation sessions.",
        sources=urls,
        critical=True,
        additional_instruction="Look for 'poster session' options on the website/CFP.",
        extra_prereq_nodes=[pres_url]
    )

    await _add_and_verify(
        evaluator,
        parent=pres,
        node_id="Paper_D_Poster_Acceptance",
        desc="Extended abstracts are explicitly accepted for poster presentations",
        claim="Extended abstracts are explicitly eligible for poster presentations.",
        sources=urls,
        critical=True,
        additional_instruction="Accept if extended abstracts can be presented as posters per guidelines.",
        extra_prereq_nodes=[pres_url]
    )

    # D.3 Conference Timing (critical)
    timing = evaluator.add_parallel(
        id="Paper_D_Conference_Timing",
        desc="Conference dates and deadlines meet Paper D timeline requirements",
        parent=paper_node,
        critical=True
    )
    timing_url = _add_url_presence_gate(
        evaluator,
        parent=timing,
        node_id="Paper_D_Timing_URL",
        desc="URL reference documenting conference dates and submission deadline",
        info=info
    )

    await _add_and_verify(
        evaluator,
        parent=timing,
        node_id="Paper_D_Conference_Period",
        desc="Conference takes place between April 1 and June 30, 2026",
        claim="The conference dates fall between April 1 and June 30, 2026.",
        sources=urls,
        critical=True,
        additional_instruction="Verify scheduled dates occur within spring 2026 (April–June).",
        extra_prereq_nodes=[timing_url]
    )

    await _add_and_verify(
        evaluator,
        parent=timing,
        node_id="Paper_D_Deadline_Advance",
        desc="Submission deadline is at least 2 months (60 days) before conference start date",
        claim="The submission deadline occurs at least 60 days before the conference start date.",
        sources=urls,
        critical=True,
        additional_instruction="Use the page's listed deadline and start date; verify the deadline is ≥60 days earlier. If exact dates are present, this should be explicit or calculable.",
        extra_prereq_nodes=[timing_url]
    )

    # D.4 Publication expectations (non-critical)
    pub = evaluator.add_parallel(
        id="Paper_D_Publication_Standards",
        desc="Conference meets Paper D publication expectations",
        parent=paper_node,
        critical=False
    )
    _add_existence_node(
        evaluator,
        parent=pub,
        node_id="Paper_D_Abstract_Proceedings",
        desc="Conference publishes extended abstracts in proceedings or online repository",
        condition=True if urls else False,  # Weak non-critical placeholder: require presence of URLs to potentially document this
        critical=False
    )
    _add_existence_node(
        evaluator,
        parent=pub,
        node_id="Paper_D_Publication_URL",
        desc="URL reference documenting abstract publication",
        condition=bool(urls),
        critical=False
    )

    # D.5 Basic conference details (non-critical)
    details = evaluator.add_parallel(
        id="Paper_D_Conference_Details",
        desc="Basic conference information for Paper D",
        parent=paper_node,
        critical=False
    )
    _add_existence_node(
        evaluator,
        parent=details,
        node_id="Paper_D_Conference_Name",
        desc="Official conference name and acronym provided",
        condition=(info is not None and bool(info.conference_name) and bool(info.acronym)),
        critical=False
    )
    _add_existence_node(
        evaluator,
        parent=details,
        node_id="Paper_D_Conference_Dates",
        desc="Specific conference dates in 2026 provided",
        condition=(info is not None and bool(info.conference_dates) and "2026" in (info.conference_dates or "")),
        critical=False
    )
    _add_existence_node(
        evaluator,
        parent=details,
        node_id="Paper_D_Conference_Location",
        desc="Conference location or virtual format specified",
        condition=(info is not None and bool(info.location_or_format)),
        critical=False
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
    # Initialize evaluator and root as parallel (per rubric)
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
        default_model=model
    )

    # Extract the conference selections for four papers
    selections = await evaluator.extract(
        prompt=prompt_extract_conferences(),
        template_class=ConferenceSelections,
        extraction_name="conference_selections"
    )

    # Build top-level node (root parallel already set up by initialize)
    top_desc = "Complete conference submission strategy for all four research papers"
    root.desc = top_desc  # Update root description for clarity

    # Invoke per-paper verifications
    await verify_paper_a(evaluator, root, selections.paper_a)
    await verify_paper_b(evaluator, root, selections.paper_b)
    await verify_paper_c(evaluator, root, selections.paper_c)
    await verify_paper_d(evaluator, root, selections.paper_d)

    # Return full evaluation summary
    return evaluator.get_summary()