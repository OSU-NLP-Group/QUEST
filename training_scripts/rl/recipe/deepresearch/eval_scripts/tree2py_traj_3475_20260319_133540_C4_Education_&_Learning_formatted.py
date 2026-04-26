import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "villanova_expert_evidence_2007"
TASK_DESCRIPTION = (
    "In Volume 52, Issue 4 of the Villanova Law Review, published in 2007, an article titled "
    "'Idealizing Science and Demonizing Experts: An Intellectual History of Expert Evidence' was published. "
    "Please provide the following information: (1) The full name of the author of this article; "
    "(2) The starting page number of this article in the law review; "
    "(3) The institution where this author obtained their PhD degree; "
    "(4) The position this author was appointed to, effective July 1, 2026; "
    "(5) The minimum GPA requirement for Villanova University's DIS Stockholm study abroad program. "
    "For each piece of information, please provide supporting reference URL(s)."
)

EXPECTED_TITLE = "Idealizing Science and Demonizing Experts: An Intellectual History of Expert Evidence"
EXPECTED_JOURNAL = "Villanova Law Review"
EXPECTED_VOLUME = "52"
EXPECTED_ISSUE = "4"
EXPECTED_YEAR = "2007"
EXPECTED_START_PAGE = "763"
EXPECTED_PHD_INSTITUTION = "Massachusetts Institute of Technology (MIT)"
EXPECTED_APPOINTED_POSITION = "President of Columbia University"
EXPECTED_APPOINTMENT_EFFECTIVE_DATE = "July 1, 2026"
EXPECTED_MIN_GPA = "3.0"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FieldWithSources(BaseModel):
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ArticleInfo(BaseModel):
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    year: Optional[str] = None
    title: Optional[str] = None
    focus: Optional[str] = None  # One-sentence summary of article focus as stated in the answer
    urls: List[str] = Field(default_factory=list)  # URLs the answer cites for the article


class ExtractionResult(BaseModel):
    article: Optional[ArticleInfo] = None

    author_full_name: Optional[FieldWithSources] = None
    author_chancellor: Optional[FieldWithSources] = None  # statement + supporting URLs

    start_page: Optional[FieldWithSources] = None

    phd_institution: Optional[FieldWithSources] = None

    appointed_position: Optional[FieldWithSources] = None
    appointed_effective_date: Optional[FieldWithSources] = None

    dis_stockholm_min_gpa: Optional[FieldWithSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract the following fields strictly from the provided answer text. Do NOT invent anything. For each sub-field that requests URL(s), extract only URLs that appear in the answer. If a field is missing, set it to null; if URLs are missing, return an empty list for that URLs field.

Return a single JSON object with this structure:

{
  "article": {
    "journal": string|null,
    "volume": string|null,
    "issue": string|null,
    "year": string|null,
    "title": string|null,
    "focus": string|null,
    "urls": string[]   // URLs cited in the answer that are used to identify/describe the article
  },
  "author_full_name": { "value": string|null, "urls": string[] },
  "author_chancellor": { "value": string|null, "urls": string[] },     // The answer's statement about the author serving as chancellor of UW–Madison starting in 2022, plus supporting URL(s)
  "start_page": { "value": string|null, "urls": string[] },            // Starting page number of the specified article, as stated in the answer, plus supporting URL(s)
  "phd_institution": { "value": string|null, "urls": string[] },       // Institution where the author obtained their PhD, plus supporting URL(s)
  "appointed_position": { "value": string|null, "urls": string[] },    // The position the author was appointed to (as stated in the answer), plus supporting URL(s)
  "appointed_effective_date": { "value": string|null, "urls": string[] },  // The effective date of the appointment (as stated in the answer), plus supporting URL(s)
  "dis_stockholm_min_gpa": { "value": string|null, "urls": string[] }  // The minimum GPA for Villanova's DIS Stockholm, plus supporting URL(s)
}

Special guidance:
- URLs can be plain links or embedded in markdown. Extract the actual URL string.
- If a URL is missing a protocol, prepend http://.
- For start_page.value, extract exactly what the answer states (e.g., "763" or similar).
- For author_chancellor.value, copy the phrasing from the answer (e.g., "Served as chancellor of UW–Madison starting in 2022").
- Do not normalize names or numbers; just extract as stated in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_urls(*url_lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for urls in url_lists:
        if not urls:
            continue
        for u in urls:
            if isinstance(u, str):
                uu = u.strip()
                if uu and uu not in seen:
                    seen.add(uu)
                    merged.append(uu)
    return merged


def _require_url_support_instruction(extra: str = "") -> str:
    base = (
        "You must verify the claim solely based on the provided URL source(s). "
        "If no valid URL is provided or the URL content is irrelevant/inaccessible, return Incorrect. "
    )
    return base + (extra or "")


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_article_identification_checks(
    evaluator: Evaluator,
    parent_node,
    article: Optional[ArticleInfo],
):
    node = evaluator.add_parallel(
        id="article_identification",
        desc="The response targets the correct Villanova Law Review article as specified (metadata/title/focus constraints).",
        parent=parent_node,
        critical=True,
    )

    article_urls = article.urls if article else []

    # 1) Volume/Issue/Year
    leaf_meta = evaluator.add_leaf(
        id="journal_volume_issue_year",
        desc="Article is in Villanova Law Review, Volume 52, Issue 4, published in 2007.",
        parent=node,
        critical=True,
    )
    claim_meta = f"This source shows Villanova Law Review, Volume {EXPECTED_VOLUME}, Issue {EXPECTED_ISSUE}, published in {EXPECTED_YEAR}."
    await evaluator.verify(
        claim=claim_meta,
        node=leaf_meta,
        sources=article_urls,
        additional_instruction=_require_url_support_instruction(
            "The metadata can be on the article page or the table of contents. Minor formatting variants are acceptable, but volume=52, issue=4, and year=2007 must be clear."
        ),
    )

    # 2) Exact title match
    leaf_title = evaluator.add_leaf(
        id="article_title_match",
        desc=f"Article title matches exactly '{EXPECTED_TITLE}'.",
        parent=node,
        critical=True,
    )
    claim_title = f"The article title is exactly '{EXPECTED_TITLE}'."
    await evaluator.verify(
        claim=claim_title,
        node=leaf_title,
        sources=article_urls,
        additional_instruction=_require_url_support_instruction(
            "Require an exact title match as printed on the page. Ignore case-only differences are acceptable, but punctuation, wording, and ordering must match; do not accept paraphrases."
        ),
    )

    # 3) Focus constraint
    leaf_focus = evaluator.add_leaf(
        id="article_focus_constraint",
        desc="Article focuses on the intellectual history of expert evidence.",
        parent=node,
        critical=True,
    )
    claim_focus = "This article focuses on the intellectual history of expert evidence."
    await evaluator.verify(
        claim=claim_focus,
        node=leaf_focus,
        sources=article_urls,
        additional_instruction=_require_url_support_instruction(
            "Look for abstract, introduction, or headings that clearly indicate the historical/intellectual-history treatment of expert evidence."
        ),
    )


async def build_requested_info_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: ExtractionResult,
):
    node = evaluator.add_parallel(
        id="requested_information_with_urls",
        desc="The response provides all five requested pieces of information, each with supporting reference URL(s), and satisfies any additional author constraints.",
        parent=parent_node,
        critical=True,
    )

    article_urls = extracted.article.urls if (extracted.article and extracted.article.urls) else []

    # ---------------------- Author full name ---------------------- #
    author_full = extracted.author_full_name or FieldWithSources()
    author_group = evaluator.add_parallel(
        id="author_full_name",
        desc="Provides the full name of the author of the specified article, with supporting URL(s).",
        parent=node,
        critical=True,
    )

    # author name value check (verify against article page and provided URLs)
    leaf_author_value = evaluator.add_leaf(
        id="author_name_value",
        desc="Author full name is correctly stated.",
        parent=author_group,
        critical=True,
    )
    claim_author_value = (
        f"The author of the article titled '{EXPECTED_TITLE}' is '{author_full.value}'."
    )
    await evaluator.verify(
        claim=claim_author_value,
        node=leaf_author_value,
        sources=_merge_urls(article_urls, author_full.urls),
        additional_instruction=_require_url_support_instruction(
            "Treat minor name variations (e.g., middle initials) as equivalent if they refer to the same person."
        ),
    )

    # author name URL support (at least one provided URL supports it)
    leaf_author_url = evaluator.add_leaf(
        id="author_name_url_support",
        desc="At least one provided reference URL credibly supports the stated author full name.",
        parent=author_group,
        critical=True,
    )
    claim_author_url = (
        f"This source confirms that the author of the article titled '{EXPECTED_TITLE}' is '{author_full.value}'."
    )
    await evaluator.verify(
        claim=claim_author_url,
        node=leaf_author_url,
        sources=author_full.urls,
        additional_instruction=_require_url_support_instruction(
            "Pass only if at least one of these provided URLs directly shows the author's name for the specified article."
        ),
    )

    # ---------------------- Chancellor constraint ---------------------- #
    chancellor = extracted.author_chancellor or FieldWithSources()
    leaf_chancellor = evaluator.add_leaf(
        id="author_chancellor_constraint",
        desc="Satisfies the stated constraint that the author served as chancellor of the University of Wisconsin–Madison starting in 2022 (the response must state this fact).",
        parent=node,
        critical=True,
    )
    # We require BOTH: the answer explicitly states this fact AND at least one URL corroborates it.
    # The verifier sees the full answer as context; we instruct to fail if not stated in the answer or no supporting URL.
    claim_chancellor = (
        f"{author_full.value} served as Chancellor of the University of Wisconsin–Madison starting in 2022."
    )
    await evaluator.verify(
        claim=claim_chancellor,
        node=leaf_chancellor,
        sources=chancellor.urls,
        additional_instruction=_require_url_support_instruction(
            "Two conditions must be met to pass: (1) the answer text explicitly states this exact fact; "
            "(2) at least one provided URL corroborates it. If either is missing, return Incorrect."
        ),
    )

    # ---------------------- Start page ---------------------- #
    start_page = extracted.start_page or FieldWithSources()
    sp_group = evaluator.add_parallel(
        id="article_start_page",
        desc="Provides the starting page number of the article, with supporting URL(s), satisfying the stated start-page constraint.",
        parent=node,
        critical=True,
    )

    leaf_sp_value = evaluator.add_leaf(
        id="start_page_value",
        desc=f"Starting page number is {EXPECTED_START_PAGE}.",
        parent=sp_group,
        critical=True,
    )
    claim_sp_value = f"The starting page of the article titled '{EXPECTED_TITLE}' is {EXPECTED_START_PAGE}."
    await evaluator.verify(
        claim=claim_sp_value,
        node=leaf_sp_value,
        sources=_merge_urls(article_urls, start_page.urls),
        additional_instruction=_require_url_support_instruction(
            f"Verify that the article's first page is {EXPECTED_START_PAGE}. Look for page headers, PDF first page markers, or table of contents."
        ),
    )

    leaf_sp_url = evaluator.add_leaf(
        id="start_page_url_support",
        desc="At least one provided reference URL credibly supports the stated starting page number.",
        parent=sp_group,
        critical=True,
    )
    claim_sp_url = f"This source shows that the article starts on page {EXPECTED_START_PAGE}."
    await evaluator.verify(
        claim=claim_sp_url,
        node=leaf_sp_url,
        sources=start_page.urls,
        additional_instruction=_require_url_support_instruction(
            "Pass only if at least one provided URL directly indicates the article's starting page."
        ),
    )

    # ---------------------- PhD institution ---------------------- #
    phd = extracted.phd_institution or FieldWithSources()
    phd_group = evaluator.add_parallel(
        id="phd_institution",
        desc="Provides the institution where the author obtained their PhD, with supporting URL(s), satisfying the stated institution constraint.",
        parent=node,
        critical=True,
    )

    leaf_phd_value = evaluator.add_leaf(
        id="phd_institution_value",
        desc=f"PhD institution is {EXPECTED_PHD_INSTITUTION}.",
        parent=phd_group,
        critical=True,
    )
    claim_phd_value = (
        f"{author_full.value} obtained their PhD from the Massachusetts Institute of Technology (MIT)."
    )
    await evaluator.verify(
        claim=claim_phd_value,
        node=leaf_phd_value,
        sources=phd.urls,
        additional_instruction=_require_url_support_instruction(
            "Accept 'Massachusetts Institute of Technology' and 'MIT' as equivalent. The page must clearly indicate PhD (doctoral) degree from MIT."
        ),
    )

    leaf_phd_url = evaluator.add_leaf(
        id="phd_institution_url_support",
        desc="At least one provided reference URL credibly supports the stated PhD institution.",
        parent=phd_group,
        critical=True,
    )
    claim_phd_url = (
        f"This source confirms that {author_full.value} earned a PhD from MIT (Massachusetts Institute of Technology)."
    )
    await evaluator.verify(
        claim=claim_phd_url,
        node=leaf_phd_url,
        sources=phd.urls,
        additional_instruction=_require_url_support_instruction(
            "Pass only if at least one provided URL explicitly mentions MIT as the PhD-awarding institution."
        ),
    )

    # ---------------------- Appointed position + effective date ---------------------- #
    appointed_pos = extracted.appointed_position or FieldWithSources()
    appointed_date = extracted.appointed_effective_date or FieldWithSources()
    appoint_group = evaluator.add_parallel(
        id="appointed_position_effective_date",
        desc="Provides the position the author was appointed to effective July 1, 2026, with supporting URL(s), satisfying the stated appointment constraint.",
        parent=node,
        critical=True,
    )

    combined_appointment_urls = _merge_urls(appointed_pos.urls, appointed_date.urls)

    leaf_app_pos = evaluator.add_leaf(
        id="appointed_position_value",
        desc=f"Appointed position is {EXPECTED_APPOINTED_POSITION}.",
        parent=appoint_group,
        critical=True,
    )
    claim_app_pos = f"{author_full.value} was appointed {EXPECTED_APPOINTED_POSITION}."
    await evaluator.verify(
        claim=claim_app_pos,
        node=leaf_app_pos,
        sources=combined_appointment_urls,
        additional_instruction=_require_url_support_instruction(
            "The page must explicitly state the appointment to this position for the same person."
        ),
    )

    leaf_app_date = evaluator.add_leaf(
        id="appointed_effective_date_value",
        desc=f"Effective date is {EXPECTED_APPOINTMENT_EFFECTIVE_DATE}.",
        parent=appoint_group,
        critical=True,
    )
    claim_app_date = f"The effective date of this appointment is {EXPECTED_APPOINTMENT_EFFECTIVE_DATE}."
    await evaluator.verify(
        claim=claim_app_date,
        node=leaf_app_date,
        sources=combined_appointment_urls,
        additional_instruction=_require_url_support_instruction(
            "The page must clearly indicate the appointment's effective date."
        ),
    )

    leaf_app_url = evaluator.add_leaf(
        id="appointment_url_support",
        desc="At least one provided reference URL credibly supports the appointment position and effective date.",
        parent=appoint_group,
        critical=True,
    )
    claim_app_url = (
        f"This source confirms both that {author_full.value} was appointed {EXPECTED_APPOINTED_POSITION} "
        f"and that the effective date is {EXPECTED_APPOINTMENT_EFFECTIVE_DATE}."
    )
    await evaluator.verify(
        claim=claim_app_url,
        node=leaf_app_url,
        sources=combined_appointment_urls,
        additional_instruction=_require_url_support_instruction(
            "Both elements (position and effective date) must be present on the same source or clearly corroborated; otherwise, return Incorrect."
        ),
    )

    # ---------------------- DIS Stockholm minimum GPA ---------------------- #
    gpa = extracted.dis_stockholm_min_gpa or FieldWithSources()
    gpa_group = evaluator.add_parallel(
        id="dis_stockholm_min_gpa",
        desc="Provides the minimum GPA requirement for Villanova University's DIS Stockholm study abroad program, with supporting URL(s), satisfying the stated GPA constraint.",
        parent=node,
        critical=True,
    )

    leaf_gpa_value = evaluator.add_leaf(
        id="min_gpa_value",
        desc=f"Minimum GPA requirement is {EXPECTED_MIN_GPA}.",
        parent=gpa_group,
        critical=True,
    )
    claim_gpa_value = (
        f"The minimum GPA requirement for Villanova University's DIS Stockholm study abroad program is {EXPECTED_MIN_GPA}."
    )
    await evaluator.verify(
        claim=claim_gpa_value,
        node=leaf_gpa_value,
        sources=gpa.urls,
        additional_instruction=_require_url_support_instruction(
            "Prefer an official Villanova University or affiliated program page mentioning 'DIS Stockholm' and its minimum GPA."
        ),
    )

    leaf_gpa_url = evaluator.add_leaf(
        id="min_gpa_url_support",
        desc="At least one provided reference URL credibly supports the stated DIS Stockholm minimum GPA requirement.",
        parent=gpa_group,
        critical=True,
    )
    claim_gpa_url = (
        f"This source confirms that the DIS Stockholm minimum GPA requirement for Villanova students is {EXPECTED_MIN_GPA}."
    )
    await evaluator.verify(
        claim=claim_gpa_url,
        node=leaf_gpa_url,
        sources=gpa.urls,
        additional_instruction=_require_url_support_instruction(
            "Pass only if at least one provided URL clearly states this minimum GPA."
        ),
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
    # Initialize evaluator with root sequential strategy (as rubric)
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

    # Extraction
    extracted: ExtractionResult = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=ExtractionResult,
        extraction_name="extracted_fields",
    )

    # Add ground-truth expectations (for transparency in summary; not used for gating)
    evaluator.add_ground_truth(
        {
            "expected_article": {
                "journal": EXPECTED_JOURNAL,
                "volume": EXPECTED_VOLUME,
                "issue": EXPECTED_ISSUE,
                "year": EXPECTED_YEAR,
                "title": EXPECTED_TITLE,
                "start_page": EXPECTED_START_PAGE,
                "focus": "intellectual history of expert evidence",
            },
            "expected_author_constraints": {
                "chancellor_fact": "Chancellor of UW–Madison starting in 2022",
                "phd_institution": EXPECTED_PHD_INSTITUTION,
                "appointment": {
                    "position": EXPECTED_APPOINTED_POSITION,
                    "effective_date": EXPECTED_APPOINTMENT_EFFECTIVE_DATE,
                },
            },
            "expected_program_requirement": {"DIS_Stockholm_min_GPA": EXPECTED_MIN_GPA},
        },
        gt_type="ground_truth",
    )

    # Build verification tree according to rubric
    # 1) Article identification (critical, parallel)
    await build_article_identification_checks(evaluator, root, extracted.article)

    # 2) Requested information with URLs (critical, parallel)
    await build_requested_info_checks(evaluator, root, extracted)

    # Return final summary
    return evaluator.get_summary()