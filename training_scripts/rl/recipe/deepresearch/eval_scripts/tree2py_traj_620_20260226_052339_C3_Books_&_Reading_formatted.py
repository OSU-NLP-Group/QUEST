import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "identify_masters_university_award_constrained"
TASK_DESCRIPTION = (
    "Identify the university where the following author earned their master's degree: "
    "The author won both the 2024 National Book Award for Fiction and the 2025 Pulitzer Prize for Fiction. "
    "The author holds a master's degree specifically in fiction, fiction writing, or creative writing, which was completed in 1982. "
    "The author's first tenure-track teaching position began within 5 years after completing their master's degree (between 1982 and 1987, inclusive), "
    "and this first tenure-track position was at the rank of assistant professor, associate professor, or professor. "
    "What is the name of the university where this author earned their master's degree?"
)


# =========================
# Data Models (Extraction)
# =========================

class AwardVerification(BaseModel):
    award_name: Optional[str] = None
    year: Optional[str] = None
    category: Optional[str] = None
    novel_title: Optional[str] = None
    author_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AwardsExtraction(BaseModel):
    nba_2024: Optional[AwardVerification] = None
    pulitzer_2025: Optional[AwardVerification] = None
    same_novel_sources: List[str] = Field(default_factory=list)


class EducationVerification(BaseModel):
    masters_university: Optional[str] = None
    masters_degree_field: Optional[str] = None
    masters_degree_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TeachingVerification(BaseModel):
    institution: Optional[str] = None
    start_year: Optional[str] = None
    rank: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CoreInfo(BaseModel):
    author_name: Optional[str] = None
    novel_title: Optional[str] = None


# =========================
# Extraction Prompts
# =========================

def prompt_extract_core() -> str:
    return (
        "Extract the core identity details of the author and the novel that (according to the answer) won both awards.\n"
        "Return JSON with fields:\n"
        "- author_name: the identified author name (string or null)\n"
        "- novel_title: the novel's title that is claimed to have won both the 2024 National Book Award for Fiction and the 2025 Pulitzer Prize for Fiction (string or null)\n"
        "If multiple novel titles are mentioned, choose the one claimed to have won both awards; otherwise return null."
    )


def prompt_extract_awards() -> str:
    return (
        "Extract award verification details for two awards from the answer. Return JSON with fields:\n"
        "- nba_2024: object with keys {award_name, year, category, novel_title, author_name, sources[]} for the 2024 National Book Award for Fiction\n"
        "- pulitzer_2025: object with keys {award_name, year, category, novel_title, author_name, sources[]} for the 2025 Pulitzer Prize for Fiction\n"
        "- same_novel_sources: array of URLs that explicitly state that both the 2024 National Book Award for Fiction and the 2025 Pulitzer Prize for Fiction were awarded for the same novel and/or to the same author\n"
        "SPECIAL RULES FOR URL SOURCES EXTRACTION:\n"
        "- Extract only URLs explicitly present in the answer (including markdown links). Do not invent.\n"
        "- If no URL is provided for an item, return an empty list for its 'sources'."
    )


def prompt_extract_education() -> str:
    return (
        "Extract the author's master's degree information and sources from the answer. Return JSON:\n"
        "- masters_university: name of the university where the master's degree was earned (string or null)\n"
        "- masters_degree_field: field/major of the master's degree (e.g., 'creative writing', 'fiction', 'fiction writing') (string or null)\n"
        "- masters_degree_year: completion year of the master's degree (string or null)\n"
        "- sources: array of URLs that support the master's degree information (university, field, and year)\n"
        "Only include URLs explicitly present in the answer."
    )


def prompt_extract_teaching() -> str:
    return (
        "Extract the author's first tenure-track teaching position details and sources from the answer. Return JSON:\n"
        "- institution: the university/college where the first tenure-track position began (string or null)\n"
        "- start_year: the year this first tenure-track position began (string or null)\n"
        "- rank: the academic rank at the start (e.g., 'assistant professor', 'associate professor', or 'professor') (string or null)\n"
        "- sources: array of URLs that support these details\n"
        "Only include URLs explicitly present in the answer."
    )


# =========================
# Helper Utilities
# =========================

def _safe(s: Optional[str], fallback: str = "") -> str:
    return (s or "").strip() or fallback


def _merge_urls(*url_lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str):
                u2 = u.strip()
                if u2 and u2 not in merged:
                    merged.append(u2)
    return merged


# =========================
# Verification Builders
# =========================

async def build_author_award_verification(
    evaluator: Evaluator,
    parent_node,
    core: CoreInfo,
    awards: AwardsExtraction,
) -> None:
    """
    Build and run the 'Author_Award_Verification' subtree.
    """
    author_award_node = evaluator.add_sequential(
        id="Author_Award_Verification",
        desc="The identified author must have won both specified fiction awards for the same novel.",
        parent=parent_node,
        critical=True
    )

    # Both_Awards_Won (parallel)
    both_awards_node = evaluator.add_parallel(
        id="Both_Awards_Won",
        desc="The author won both the 2024 National Book Award for Fiction and the 2025 Pulitzer Prize for Fiction.",
        parent=author_award_node,
        critical=True
    )

    # Extract NBA 2024 info
    nba = awards.nba_2024 or AwardVerification()
    pul = awards.pulitzer_2025 or AwardVerification()

    # Create source existence gates (critical siblings to gate verifications)
    nba_sources_exist = evaluator.add_custom_node(
        result=bool(nba.sources),
        id="National_Book_Award_2024_Sources_Provided",
        desc="Sources provided for the 2024 National Book Award for Fiction claim",
        parent=both_awards_node,
        critical=True
    )
    pul_sources_exist = evaluator.add_custom_node(
        result=bool(pul.sources),
        id="Pulitzer_Prize_2025_Sources_Provided",
        desc="Sources provided for the 2025 Pulitzer Prize for Fiction claim",
        parent=both_awards_node,
        critical=True
    )

    # Leaves for each award
    nba_leaf = evaluator.add_leaf(
        id="National_Book_Award_2024",
        desc="The author won the 2024 National Book Award for Fiction.",
        parent=both_awards_node,
        critical=True
    )
    pul_leaf = evaluator.add_leaf(
        id="Pulitzer_Prize_2025",
        desc="The author won the 2025 Pulitzer Prize for Fiction.",
        parent=both_awards_node,
        critical=True
    )

    # Build claims using available info
    author_name = _safe(nba.author_name, _safe(pul.author_name, _safe(core.author_name, "the author")))
    novel_from_core = _safe(core.novel_title, "")
    novel_nba = _safe(nba.novel_title, "")
    novel_pul = _safe(pul.novel_title, "")
    novel_for_awards = novel_nba or novel_pul or novel_from_core

    nba_claim = (
        f"{author_name} won the 2024 National Book Award for Fiction"
        + (f" for the novel '{novel_for_awards}'." if novel_for_awards else ".")
    )
    pul_claim = (
        f"{author_name} won the 2025 Pulitzer Prize for Fiction"
        + (f" for the novel '{novel_for_awards}'." if novel_for_awards else ".")
    )

    # Verify awards in parallel
    await evaluator.batch_verify([
        (
            nba_claim,
            nba.sources,
            nba_leaf,
            "Verify on the provided page(s) that the person indeed won the 2024 National Book Award for Fiction. "
            "If a novel is specified in the claim, ensure it matches the page. Allow minor name variants."
        ),
        (
            pul_claim,
            pul.sources,
            pul_leaf,
            "Verify on the provided page(s) that the person indeed won the 2025 Pulitzer Prize for Fiction. "
            "If a novel is specified in the claim, ensure it matches the page. Allow minor name variants."
        ),
    ])

    # Same_Novel_Verification (sequential next step)
    # Prepare sources: prefer explicit same_novel_sources; otherwise combine award sources
    combined_same_sources = _merge_urls(awards.same_novel_sources, nba.sources, pul.sources)
    same_sources_exist = evaluator.add_custom_node(
        result=bool(combined_same_sources),
        id="Same_Novel_Sources_Provided",
        desc="Sources provided to support that both awards were for the same novel",
        parent=author_award_node,
        critical=True
    )

    same_novel_leaf = evaluator.add_leaf(
        id="Same_Novel_Verification",
        desc="Both the 2024 National Book Award for Fiction and the 2025 Pulitzer Prize for Fiction were awarded for the same novel.",
        parent=author_award_node,
        critical=True
    )

    # Claim for same novel
    same_novel_claim = (
        f"Both the 2024 National Book Award for Fiction and the 2025 Pulitzer Prize for Fiction were awarded "
        f"to {author_name}" + (f" for the same novel titled '{novel_for_awards}'." if novel_for_awards else " for the same novel.")
    )

    await evaluator.verify(
        claim=same_novel_claim,
        node=same_novel_leaf,
        sources=combined_same_sources,
        additional_instruction=(
            "The evidence must clearly indicate that BOTH awards were for the same work. "
            "Accept pages that explicitly mention both awards for the same novel, or corroborating sources that jointly show the same novel title for both awards. "
            "If the page only shows one award without linking to the other, that page alone is insufficient unless it explicitly mentions both."
        ),
    )


async def build_education_and_teaching_verification(
    evaluator: Evaluator,
    parent_node,
    education: EducationVerification,
    teaching: TeachingVerification,
    core: CoreInfo
) -> None:
    """
    Build and run the 'Educational_Background_Verification' subtree.
    """
    edu_main = evaluator.add_parallel(
        id="Educational_Background_Verification",
        desc="The author's master's degree and first teaching position must meet the specified criteria.",
        parent=parent_node,
        critical=True
    )

    # Degree_Specifications (parallel)
    degree_specs = evaluator.add_parallel(
        id="Degree_Specifications",
        desc="The master's degree must be in the correct field and completed in the correct year.",
        parent=edu_main,
        critical=True
    )

    # Gate: degree sources provided
    degree_sources_exist = evaluator.add_custom_node(
        result=bool(education.sources),
        id="Degree_Sources_Provided",
        desc="Sources provided for master's degree details (university, field, year)",
        parent=degree_specs,
        critical=True
    )

    # Masters_University (additional leaf to ensure task output is validated)
    masters_uni_leaf = evaluator.add_leaf(
        id="Masters_University",
        desc="The author earned their master's degree at the specified university.",
        parent=degree_specs,
        critical=True
    )
    uni_claim_author = _safe(core.author_name, "The author")
    uni_claim = (
        f"{uni_claim_author} earned their master's degree at {_safe(education.masters_university, 'the specified university')}."
    )
    await evaluator.verify(
        claim=uni_claim,
        node=masters_uni_leaf,
        sources=education.sources,
        additional_instruction=(
            "Confirm the master's degree institution. "
            "Allow reasonable naming variants (e.g., 'University of X' vs. 'X University')."
        )
    )

    # Degree_Field leaf
    degree_field_leaf = evaluator.add_leaf(
        id="Degree_Field",
        desc="The master's degree is in fiction, fiction writing, or creative writing.",
        parent=degree_specs,
        critical=True
    )
    field_text = _safe(education.masters_degree_field, "")
    field_claim = (
        f"{uni_claim_author} earned a master's degree in {field_text}, which falls within fiction, fiction writing, or creative writing."
        if field_text else
        f"{uni_claim_author} earned a master's degree in a field that falls within fiction, fiction writing, or creative writing."
    )
    await evaluator.verify(
        claim=field_claim,
        node=degree_field_leaf,
        sources=education.sources,
        additional_instruction=(
            "Accept synonyms such as 'MFA in Creative Writing' or 'Master's in Fiction Writing' as compliant fields. "
            "Reject unrelated fields. The page must support the stated field."
        )
    )

    # Degree_Year leaf
    degree_year_leaf = evaluator.add_leaf(
        id="Degree_Year",
        desc="The master's degree was completed in 1982.",
        parent=degree_specs,
        critical=True
    )
    degree_year_claim = f"The master's degree was completed in 1982."
    await evaluator.verify(
        claim=degree_year_claim,
        node=degree_year_leaf,
        sources=education.sources,
        additional_instruction="Verify that the completion/award year of the master's degree is 1982."
    )

    # First_Teaching_Position (parallel)
    ftp_node = evaluator.add_parallel(
        id="First_Teaching_Position",
        desc="The author's first tenure-track teaching position must meet timeline and rank requirements.",
        parent=edu_main,
        critical=True
    )

    # Gate: teaching sources exist
    teaching_sources_exist = evaluator.add_custom_node(
        result=bool(teaching.sources),
        id="Teaching_Sources_Provided",
        desc="Sources provided for first tenure-track teaching position (institution, start year, rank)",
        parent=ftp_node,
        critical=True
    )

    # Position_Timeline leaf
    pos_timeline_leaf = evaluator.add_leaf(
        id="Position_Timeline",
        desc="The first tenure-track position began within 5 years after completing the master's degree (1982-1987, inclusive).",
        parent=ftp_node,
        critical=True
    )
    start_year_text = _safe(teaching.start_year, "")
    timeline_claim = (
        f"The author's first tenure-track teaching position began in {start_year_text}, which is between 1982 and 1987 inclusive."
        if start_year_text else
        "The author's first tenure-track teaching position began between 1982 and 1987 inclusive."
    )
    await evaluator.verify(
        claim=timeline_claim,
        node=pos_timeline_leaf,
        sources=teaching.sources,
        additional_instruction=(
            "Confirm that the earliest tenure-track appointment date falls within 1982–1987 inclusive. "
            "Assistant/Associate/Professor ranks at appointment imply tenure-track unless explicitly stated otherwise. "
            "If multiple positions are listed, consider the earliest tenure-track one."
        )
    )

    # Academic_Rank leaf
    rank_leaf = evaluator.add_leaf(
        id="Academic_Rank",
        desc="The first tenure-track position was at the rank of assistant professor, associate professor, or professor.",
        parent=ftp_node,
        critical=True
    )
    rank_text = _safe(teaching.rank, "")
    rank_claim = (
        f"The first tenure-track position rank was {rank_text}, which is one of: assistant professor, associate professor, or professor."
        if rank_text else
        "The first tenure-track position rank is one of: assistant professor, associate professor, or professor."
    )
    await evaluator.verify(
        claim=rank_claim,
        node=rank_leaf,
        sources=teaching.sources,
        additional_instruction=(
            "Accept reasonable variants (e.g., 'Asst. Professor', 'Prof.'). "
            "Ensure the page supports the rank at the time the first tenure-track position began."
        )
    )


# =========================
# Main Evaluation
# =========================

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

    # Run extractions in parallel
    core_task = evaluator.extract(
        prompt=prompt_extract_core(),
        template_class=CoreInfo,
        extraction_name="core_info",
    )
    awards_task = evaluator.extract(
        prompt=prompt_extract_awards(),
        template_class=AwardsExtraction,
        extraction_name="awards_info",
    )
    education_task = evaluator.extract(
        prompt=prompt_extract_education(),
        template_class=EducationVerification,
        extraction_name="education_info",
    )
    teaching_task = evaluator.extract(
        prompt=prompt_extract_teaching(),
        template_class=TeachingVerification,
        extraction_name="teaching_info",
    )

    core, awards, education, teaching = await asyncio.gather(core_task, awards_task, education_task, teaching_task)

    # Build the rubric tree according to JSON with a critical top-level node
    complete_task = evaluator.add_sequential(
        id="Complete_Identification_Task",
        desc="Correctly identify the author who won both the 2024 National Book Award for Fiction and the 2025 Pulitzer Prize for Fiction for the same novel, and identify the university where this author earned their master's degree.",
        parent=root,
        critical=True
    )

    # 1) Author Award Verification subtree
    await build_author_award_verification(
        evaluator=evaluator,
        parent_node=complete_task,
        core=core or CoreInfo(),
        awards=awards or AwardsExtraction(),
    )

    # 2) Educational Background Verification subtree (includes university check)
    await build_education_and_teaching_verification(
        evaluator=evaluator,
        parent_node=complete_task,
        education=education or EducationVerification(),
        teaching=teaching or TeachingVerification(),
        core=core or CoreInfo(),
    )

    return evaluator.get_summary()