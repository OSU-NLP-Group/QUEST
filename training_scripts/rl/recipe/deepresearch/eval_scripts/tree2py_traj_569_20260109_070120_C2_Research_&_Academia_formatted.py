import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "uw_prof_2024_acm_dda_advisor"
TASK_DESCRIPTION = (
    "Who is the University of Washington professor who advised the PhD student that won the 2024 ACM Doctoral Dissertation Award "
    "for a dissertation focusing on human-AI collaboration to support mental health and well-being?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ExtractedURLs(BaseModel):
    # URLs that directly support advisor-advisee relationship (e.g., lab page, advisor/advisee bio, university news)
    advisor_relation_urls: List[str] = Field(default_factory=list)
    # URLs that support professor's UW affiliation (e.g., UW profile, Allen School page)
    professor_profile_urls: List[str] = Field(default_factory=list)
    # URLs for award announcements or related news (e.g., ACM, UW news)
    award_announcement_urls: List[str] = Field(default_factory=list)
    # URLs that discuss the dissertation focus/topic/title (may overlap with award announcements)
    dissertation_focus_urls: List[str] = Field(default_factory=list)
    # Any additional or uncategorized URLs included in the answer
    all_urls: List[str] = Field(default_factory=list)


class AnswerExtraction(BaseModel):
    # The professor named by the answer as the advisor
    professor_name: Optional[str] = None
    # The PhD student named by the answer as the 2024 ACM Doctoral Dissertation Award winner
    student_name: Optional[str] = None
    # Optional string describing the stated dissertation focus/theme from the answer
    dissertation_focus: Optional[str] = None
    # The affiliation string if explicitly mentioned (e.g., "University of Washington")
    professor_affiliation: Optional[str] = None
    # Grouped URLs
    urls: ExtractedURLs = Field(default_factory=ExtractedURLs)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_core() -> str:
    return """
    From the provided answer, extract the following fields related to the question:

    Required text fields:
    - professor_name: The full name of the professor claimed to be the advisor.
    - student_name: The full name of the PhD student claimed to be the 2024 ACM Doctoral Dissertation Award winner.
    - dissertation_focus: A short phrase or sentence that summarizes the dissertation's focus/theme if mentioned (e.g., "human-AI collaboration to support mental health and well-being"). If not mentioned, return null.
    - professor_affiliation: If the answer explicitly mentions the professor's affiliation (e.g., "University of Washington"), extract it. Otherwise, return null.

    URLs (extract explicitly listed URLs only; include full protocols):
    - urls.advisor_relation_urls: URLs that indicate or support the advisor-advisee relationship (e.g., advisor's lab page listing the student, student's page listing the advisor, university news noting the advisor).
    - urls.professor_profile_urls: URLs that show the professor is faculty at the University of Washington (e.g., UW / Allen School profile).
    - urls.award_announcement_urls: URLs that announce or reference the 2024 ACM Doctoral Dissertation Award (e.g., ACM official page, news articles).
    - urls.dissertation_focus_urls: URLs that describe the dissertation topic/focus/title (may overlap with award announcements).
    - urls.all_urls: A catch-all list containing every URL mentioned in the answer (including the above). Do not invent new URLs.

    If any field is missing from the answer, set it to null (for text fields) or an empty list (for URL lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        v = u.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def gather_sources_for_advisorship(info: AnswerExtraction) -> List[str]:
    urls = (
        info.urls.advisor_relation_urls
        + info.urls.award_announcement_urls
        + info.urls.professor_profile_urls
        + info.urls.dissertation_focus_urls
        + info.urls.all_urls
    )
    return _dedup_urls(urls)


def gather_sources_for_affiliation(info: AnswerExtraction) -> List[str]:
    urls = info.urls.professor_profile_urls + info.urls.all_urls
    return _dedup_urls(urls)


def gather_sources_for_focus(info: AnswerExtraction) -> List[str]:
    urls = info.urls.dissertation_focus_urls + info.urls.award_announcement_urls + info.urls.all_urls
    return _dedup_urls(urls)


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extraction: AnswerExtraction) -> None:
    # Create top-level rubric node: "Professor Identification" (critical, parallel)
    prof_ident_node = evaluator.add_parallel(
        id="professor_identification",
        desc="Identify the professor who advised the 2024 ACM Doctoral Dissertation Award winner",
        parent=evaluator.root,
        critical=True,
    )

    # Sub-node: "Award Advisorship Verification" (critical, parallel)
    advisorship_node = evaluator.add_parallel(
        id="award_advisorship_verification",
        desc="Verify the professor advised the 2024 ACM Doctoral Dissertation Award winner and institutional affiliation",
        parent=prof_ident_node,
        critical=True,
    )

    # Existence checks under advisorship node (critical)
    names_present = evaluator.add_custom_node(
        result=bool(extraction.professor_name and extraction.professor_name.strip())
        and bool(extraction.student_name and extraction.student_name.strip()),
        id="names_present",
        desc="Professor and student names are provided in the answer",
        parent=advisorship_node,
        critical=True,
    )

    advisorship_sources = gather_sources_for_advisorship(extraction)
    sources_present = evaluator.add_custom_node(
        result=len(advisorship_sources) > 0,
        id="advisorship_sources_present",
        desc="At least one source URL is provided to support advisorship/award claims",
        parent=advisorship_node,
        critical=True,
    )

    # Leaf: "Advised 2024 ACM Doctoral Dissertation Award Winner" (critical)
    advised_award_leaf = evaluator.add_leaf(
        id="advised_2024_acm_doctoral_dissertation_award_winner",
        desc="The professor was the PhD dissertation advisor of the student who won the 2024 ACM Doctoral Dissertation Award",
        parent=advisorship_node,
        critical=True,
    )
    claim_advisorship = (
        f"{extraction.professor_name or 'UNKNOWN PROFESSOR'} was the PhD dissertation advisor of "
        f"{extraction.student_name or 'UNKNOWN STUDENT'}, who won the 2024 ACM Doctoral Dissertation Award."
    )
    await evaluator.verify(
        claim=claim_advisorship,
        node=advised_award_leaf,
        sources=advisorship_sources,
        additional_instruction=(
            "Verify on a single provided page that (1) the named student won the 2024 ACM Doctoral Dissertation Award, "
            "and (2) the named professor is that student's PhD (doctoral) dissertation advisor. "
            "Allow reasonable synonyms like 'advisor', 'adviser', or 'supervisor'. If a page clearly states both facts, "
            "the claim is supported."
        ),
    )

    # Leaf: "University of Washington Affiliation" (critical)
    uw_affil_leaf = evaluator.add_leaf(
        id="university_of_washington_affiliation",
        desc="The professor is a faculty member at the University of Washington",
        parent=advisorship_node,
        critical=True,
    )
    uw_sources = gather_sources_for_affiliation(extraction)
    claim_affiliation = (
        f"{extraction.professor_name or 'UNKNOWN PROFESSOR'} is a faculty member at the University of Washington."
    )
    await evaluator.verify(
        claim=claim_affiliation,
        node=uw_affil_leaf,
        sources=uw_sources,
        additional_instruction=(
            "Accept evidence from official UW or departmental pages (e.g., Paul G. Allen School) showing the person holds a "
            "faculty position (assistant/associate/full professor, affiliate, or adjunct). Minor variations in title are acceptable."
        ),
    )

    # Leaf under top-level: "Dissertation Research Focus" (critical)
    focus_leaf = evaluator.add_leaf(
        id="dissertation_research_focus",
        desc="The advised dissertation focused on human-AI collaboration to support mental health and well-being",
        parent=prof_ident_node,
        critical=True,
    )
    focus_sources = gather_sources_for_focus(extraction)
    # Build a descriptive claim tying names if available (helps LLM focus)
    if extraction.professor_name and extraction.student_name:
        claim_focus = (
            f"The dissertation by {extraction.student_name}, advised by {extraction.professor_name}, "
            f"focused on human-AI collaboration to support mental health and well-being."
        )
    else:
        claim_focus = (
            "The dissertation focused on human-AI collaboration to support mental health and well-being."
        )

    await evaluator.verify(
        claim=claim_focus,
        node=focus_leaf,
        sources=focus_sources,
        additional_instruction=(
            "Support is sufficient if the page clearly states that the dissertation's topic/theme is about human-AI "
            "collaboration (or human–AI interaction) aimed at supporting mental health and/or wellbeing. "
            "Allow paraphrases such as 'mental wellbeing', 'mental health support', or similar wording."
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
    """
    Evaluate the answer for the UW advisor of the 2024 ACM Doctoral Dissertation Award winner task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
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

    # 1) Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_core(),
        template_class=AnswerExtraction,
        extraction_name="extracted_answer_entities_and_urls",
    )

    # 2) Record key extracted info for transparency
    evaluator.add_custom_info(
        info={
            "professor_name": extraction.professor_name,
            "student_name": extraction.student_name,
            "dissertation_focus": extraction.dissertation_focus,
            "professor_affiliation_text": extraction.professor_affiliation,
            "url_counts": {
                "advisor_relation_urls": len(extraction.urls.advisor_relation_urls),
                "professor_profile_urls": len(extraction.urls.professor_profile_urls),
                "award_announcement_urls": len(extraction.urls.award_announcement_urls),
                "dissertation_focus_urls": len(extraction.urls.dissertation_focus_urls),
                "all_urls": len(extraction.urls.all_urls),
            },
        },
        info_type="extraction_overview",
        info_name="extraction_overview",
    )

    # 3) Build verification tree and run checks
    await build_and_verify_tree(evaluator, extraction)

    # 4) Return summary
    return evaluator.get_summary()