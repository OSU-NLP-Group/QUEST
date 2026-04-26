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
TASK_ID = "beardsley_doctorate_credentials"
TASK_DESCRIPTION = """
Scott Beardsley was appointed as the 10th President of the University of Virginia in December 2025, taking office in January 2026. Before this role, he served as Dean of the UVA Darden School of Business beginning in August 2015. Provide the following details about his educational credentials: (1) What type of doctoral degree did he earn (specify Ed.D., Ph.D., or other terminal degree type)? (2) What was the specific field or program name of this doctoral degree? (3) From which institution did he receive this doctoral degree? (4) In what year was this degree completed, and confirm whether it was earned before or during his appointment as Darden dean in 2015? Include reference URLs from official or credible sources to support each piece of information.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BeardsleyDoctoralExtraction(BaseModel):
    # Core info from the answer
    subject_name: Optional[str] = None
    degree_type: Optional[str] = None  # e.g., "Ed.D.", "Doctor of Education", "Ph.D."
    degree_is_earned: Optional[bool] = None  # True if explicitly earned/awarded (not honorary), False if honorary, null if unclear
    field_program: Optional[str] = None  # e.g., "Higher Education Management"
    awarding_institution: Optional[str] = None  # e.g., "University of Pennsylvania"
    completion_year: Optional[str] = None  # e.g., "2013"
    timing_relative_to_2015: Optional[str] = None  # one of: "before", "during", "after", or null if not stated

    # URLs cited in the answer for supporting each part
    degree_type_urls: List[str] = Field(default_factory=list)
    field_urls: List[str] = Field(default_factory=list)
    institution_urls: List[str] = Field(default_factory=list)
    accreditation_urls: List[str] = Field(default_factory=list)
    completion_year_urls: List[str] = Field(default_factory=list)

    # All URLs mentioned anywhere in the answer (if the answer provided a unified sources list)
    all_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_beardsley_doctoral_details() -> str:
    return """
    Extract Scott Beardsley’s doctoral-degree details exactly as stated in the provided answer text. Return a single JSON object with the fields below. Do not invent or infer information beyond what is in the answer.

    Required JSON fields to extract:
    - subject_name: The person whose credentials are described (e.g., "Scott Beardsley"). If not explicitly named, put null.
    - degree_type: The type of doctoral/terminal degree named (e.g., "Ed.D.", "Doctor of Education", "Ph.D.", "DBA"). Use the exact wording shown in the answer. If not stated, null.
    - degree_is_earned: true if the answer explicitly indicates this is an earned/awarded doctoral degree (or clearly implies normal earned study, not honorary). false if the answer explicitly indicates it is honorary (e.g., "honorary", "honoris causa"). null if unclear.
    - field_program: The specific field or program name of the doctoral degree (e.g., "Higher Education Management", "Educational Leadership"). If not stated, null.
    - awarding_institution: The institution that awarded the doctoral degree. If not stated, null.
    - completion_year: The year the degree was completed (4-digit string if present, e.g., "2013"). If not stated, null.
    - timing_relative_to_2015: If the answer explicitly confirms timing relative to 2015 Darden deanship, use one of "before", "during", or "after". If not explicitly confirmed but a year is stated, you may fill by logic: if year < 2015 then "before"; if year == 2015 then "during"; if year > 2015 then "after". If no year and no explicit timing, null.

    Also extract URL lists that the answer cites to support each item. Only include URLs explicitly present in the answer:
    - degree_type_urls: URLs specifically supporting the degree type and (if present) earned/not honorary status.
    - field_urls: URLs supporting the field/program name.
    - institution_urls: URLs supporting the awarding institution.
    - accreditation_urls: URLs supporting that the awarding institution is accredited/recognized (e.g., official university pages, accreditor listings, government/official recognition pages). If none are cited, return [].
    - completion_year_urls: URLs supporting the completion year and/or timing vs 2015.
    - all_urls: All URLs mentioned anywhere in the answer (including those in the above lists).

    Important rules:
    - Extract only what appears in the answer text. If a field is missing, set it to null (or [] for URL lists).
    - For URLs, capture complete valid URLs as they appear (markdown links should be unwrapped to their actual URL).
    - Do not deduplicate URLs across lists; include them wherever they are cited. The 'all_urls' list should include every URL mentioned in the answer, including duplicates if they appear duplicated.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _parse_year_int(year_str: Optional[str]) -> Optional[int]:
    if not _is_nonempty(year_str):
        return None
    m = re.search(r"\b(19|20)\d{2}\b", year_str.strip())
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


def _compute_timing_from_year(year_int: Optional[int]) -> Optional[str]:
    if year_int is None:
        return None
    if year_int < 2015:
        return "before"
    if year_int == 2015:
        return "during"
    return "after"


def _unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        u = (u or "").strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _collect_all_urls(ex: BeardsleyDoctoralExtraction) -> List[str]:
    combined = []
    combined.extend(ex.all_urls or [])
    combined.extend(ex.degree_type_urls or [])
    combined.extend(ex.field_urls or [])
    combined.extend(ex.institution_urls or [])
    combined.extend(ex.accreditation_urls or [])
    combined.extend(ex.completion_year_urls or [])
    return _unique_urls(combined)


def _pick_sources(primary: List[str], fallback: List[str]) -> List[str]:
    if primary and len(primary) > 0:
        return primary
    return fallback


# --------------------------------------------------------------------------- #
# Verification tree builder                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: BeardsleyDoctoralExtraction) -> None:
    """
    Build the rubric-based verification tree and execute verifications.
    All nodes in this task are critical as per the rubric.
    """
    # Create top-level critical node under evaluator.root
    top = evaluator.add_parallel(
        id="Beardsley_Doctoral_Credentials",
        desc="Verify the answer provides all required doctoral-degree details for Scott Beardsley and includes credible supporting URLs for each requested detail.",
        parent=evaluator.root,
        critical=True
    )

    # Prepare a fallback pool of all URLs
    all_urls = _collect_all_urls(extraction)

    # ------------------------- Subject Identity --------------------------- #
    subject_node = evaluator.add_parallel(
        id="Subject_Identity",
        desc="The answer is about Scott Beardsley (the subject named in the prompt).",
        parent=top,
        critical=True
    )

    subj_leaf = evaluator.add_leaf(
        id="Identity_Is_Scott_Beardsley",
        desc="The subject whose credentials are provided is Scott Beardsley.",
        parent=subject_node,
        critical=True
    )
    extracted_subject = extraction.subject_name or ""
    subj_claim = (
        f"The subject whose doctoral credentials are described in the answer is Scott Beardsley."
    )
    await evaluator.verify(
        claim=subj_claim,
        node=subj_leaf,
        additional_instruction=(
            f"Use the answer context to judge identity. The extracted subject_name (if any) is: '{extracted_subject}'. "
            "Allow reasonable pronoun/coreference if the answer clearly centers on Scott Beardsley."
        )
    )

    # ---------------------- Doctoral Degree Type -------------------------- #
    degree_type_node = evaluator.add_parallel(
        id="Doctoral_Degree_Type",
        desc="Provide the type of doctoral/terminal degree he earned, with citation.",
        parent=top,
        critical=True
    )

    # Degree_Type_Provided (existence)
    evaluator.add_custom_node(
        result=_is_nonempty(extraction.degree_type),
        id="Degree_Type_Provided",
        desc="States the doctoral degree type (e.g., Ed.D., Ph.D., or other terminal doctoral degree type).",
        parent=degree_type_node,
        critical=True
    )

    # Degree_Is_Earned (answer explicitly indicates earned vs honorary)
    deg_earned_leaf = evaluator.add_leaf(
        id="Degree_Is_Earned",
        desc="Indicates the doctorate was earned/awarded (not merely honorary).",
        parent=degree_type_node,
        critical=True
    )
    if extraction.degree_is_earned is True:
        earned_claim = "The doctoral degree for Scott Beardsley is an earned/awarded degree (not honorary)."
    else:
        # If False or None, we still assert the positive claim; simple verification should fail when unsupported.
        earned_claim = "The doctoral degree for Scott Beardsley is an earned/awarded degree (not honorary)."
    await evaluator.verify(
        claim=earned_claim,
        node=deg_earned_leaf,
        additional_instruction=(
            "Judge using ONLY the provided answer text. Look for cues like 'earned', 'completed', 'awarded' "
            "versus 'honorary', 'honoris causa'. If unclear or not stated, mark incorrect."
        )
    )

    # Degree_Type_Supported_By_URL (URL-backed support)
    deg_type_support_leaf = evaluator.add_leaf(
        id="Degree_Type_Supported_By_URL",
        desc="Provides at least one official/credible reference URL that supports the stated degree type and earned status.",
        parent=degree_type_node,
        critical=True
    )
    deg_type_sources = _pick_sources(extraction.degree_type_urls or [], all_urls)
    deg_type_text = extraction.degree_type or "the doctoral degree"
    deg_type_claim = (
        f"The stated doctoral degree type for Scott Beardsley ('{deg_type_text}') is explicitly supported by at least one of the cited URLs, "
        f"and the doctorate is a non-honorary (earned) degree."
    )
    await evaluator.verify(
        claim=deg_type_claim,
        node=deg_type_support_leaf,
        sources=deg_type_sources,
        additional_instruction=(
            "Check whether the page explicitly states the degree type (e.g., Ed.D./Doctor of Education, Ph.D./Doctor of Philosophy, etc.) "
            "for Scott Beardsley. Treat 'Ed.D.' and 'Doctor of Education' as equivalent, and 'Ph.D.' and 'Doctor of Philosophy' as equivalent. "
            "If no URLs are provided in the answer, mark incorrect."
        )
    )

    # ----------------- Doctoral Degree Field/Program ---------------------- #
    field_node = evaluator.add_parallel(
        id="Doctoral_Degree_Field_Or_Program",
        desc="Provide the specific doctoral field/program name and satisfy the relevance constraint, with citation.",
        parent=top,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty(extraction.field_program),
        id="Field_Program_Name_Provided",
        desc="States the specific field or program name of the doctoral degree.",
        parent=field_node,
        critical=True
    )

    field_relevance_leaf = evaluator.add_leaf(
        id="Field_Relevance_Satisfied",
        desc="The stated field/program satisfies the relevance constraint to higher-education leadership.",
        parent=field_node,
        critical=True
    )
    field_name = extraction.field_program or ""
    relevance_claim = (
        f"The doctoral field/program '{field_name}' is relevant to higher-education leadership, administration, policy, "
        f"or management (i.e., it aligns with leadership in higher education)."
    )
    await evaluator.verify(
        claim=relevance_claim,
        node=field_relevance_leaf,
        additional_instruction=(
            "Base the judgment primarily on the program/field name semantics. Examples of relevant names include: "
            "'Higher Education', 'Higher Education Management/Administration', 'Educational Leadership', 'Education Leadership & Policy', etc. "
            "If the name is clearly unrelated (e.g., 'Electrical Engineering'), mark incorrect."
        )
    )

    field_support_leaf = evaluator.add_leaf(
        id="Field_Supported_By_URL",
        desc="Provides at least one official/credible reference URL supporting the stated field/program name.",
        parent=field_node,
        critical=True
    )
    field_sources = _pick_sources(extraction.field_urls or [], all_urls)
    field_claim = (
        f"The cited page(s) explicitly state that Scott Beardsley's doctoral field/program was '{field_name}' (or an equivalent naming)."
    )
    await evaluator.verify(
        claim=field_claim,
        node=field_support_leaf,
        sources=field_sources,
        additional_instruction=(
            "Accept close naming variants that are obviously equivalent (e.g., 'Higher Education' vs 'Higher Education Management'). "
            "If the answer provides no URLs, mark incorrect."
        )
    )

    # ------ Doctoral Degree Institution & Accreditation/Recognition ------- #
    inst_node = evaluator.add_parallel(
        id="Doctoral_Degree_Institution_And_Accreditation",
        desc="Provide the awarding institution and meet the accredited-institution constraint, with citation.",
        parent=top,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty(extraction.awarding_institution),
        id="Awarding_Institution_Provided",
        desc="States the institution that awarded the doctoral degree.",
        parent=inst_node,
        critical=True
    )

    inst_accred_leaf = evaluator.add_leaf(
        id="Awarding_Institution_Accredited",
        desc="Provides evidence via a credible URL that the awarding institution is an accredited/recognized higher-education institution.",
        parent=inst_node,
        critical=True
    )
    accred_sources = _pick_sources(extraction.accreditation_urls or [], _pick_sources(extraction.institution_urls or [], all_urls))
    inst_name = extraction.awarding_institution or "the stated awarding institution"
    accred_claim = (
        f"The awarding institution '{inst_name}' is an accredited or otherwise recognized institution of higher education."
    )
    await evaluator.verify(
        claim=accred_claim,
        node=inst_accred_leaf,
        sources=accred_sources,
        additional_instruction=(
            "Evidence may include: official university pages (especially .edu/.ac.uk), official accreditor/government listings, "
            "or internationally recognized official sources. If the URL is an official university domain, this typically suffices. "
            "If only non-credible sources are cited, mark incorrect."
        )
    )

    inst_support_leaf = evaluator.add_leaf(
        id="Institution_Supported_By_URL",
        desc="Provides at least one official/credible reference URL supporting that the stated institution awarded the doctorate.",
        parent=inst_node,
        critical=True
    )
    inst_sources = _pick_sources(extraction.institution_urls or [], all_urls)
    inst_claim = f"At least one cited page explicitly states that Scott Beardsley received the doctoral degree from '{inst_name}'."
    await evaluator.verify(
        claim=inst_claim,
        node=inst_support_leaf,
        sources=inst_sources,
        additional_instruction="Look for explicit phrasing that the degree was awarded by this institution. If no URLs are provided, mark incorrect."
    )

    # ------------- Completion Year & Timing vs 2015 Deanship -------------- #
    timing_node = evaluator.add_parallel(
        id="Doctoral_Degree_Completion_Year_And_Timing",
        desc="Provide the completion year and confirm whether it was earned before or during his 2015 Darden dean appointment, with citation.",
        parent=top,
        critical=True
    )

    year_int = _parse_year_int(extraction.completion_year)
    evaluator.add_custom_node(
        result=(year_int is not None),
        id="Completion_Year_Provided",
        desc="States the year the doctoral degree was completed.",
        parent=timing_node,
        critical=True
    )

    timing_leaf = evaluator.add_leaf(
        id="Timing_Relative_To_2015_Deanship_Confirmed",
        desc="Explicitly confirms whether the degree was earned before or during (not after) his 2015 appointment as Darden dean.",
        parent=timing_node,
        critical=True
    )
    # Prefer explicit timing if provided; otherwise infer from year when possible.
    extracted_timing = (extraction.timing_relative_to_2015 or "").strip().lower() if extraction.timing_relative_to_2015 else None
    inferred_timing = _compute_timing_from_year(year_int)
    chosen_timing = extracted_timing or inferred_timing

    if chosen_timing in ("before", "during"):
        timing_claim = (
            "Based on the answer's information (explicit timing statement and/or the stated completion year), "
            "the doctoral degree was earned before or during 2015 (not after)."
        )
    else:
        # If 'after' or unknown, we still assert the required condition; the check should fail if unsupported.
        timing_claim = (
            "Based on the answer's information (explicit timing statement and/or the stated completion year), "
            "the doctoral degree was earned before or during 2015 (not after)."
        )
    await evaluator.verify(
        claim=timing_claim,
        node=timing_leaf,
        additional_instruction=(
            f"Completion year extracted: {extraction.completion_year or 'null'}. "
            "If the year < 2015, treat as 'before 2015'. If the year == 2015, treat as 'during 2015'. "
            "If the year > 2015 or no clear year/timing is given, mark incorrect. "
            "If the answer provides month/day details confirming 'before/during 2015', also accept."
        )
    )

    timing_support_leaf = evaluator.add_leaf(
        id="Completion_Timing_Supported_By_URL",
        desc="Provides at least one official/credible reference URL supporting the completion year and the timing claim relative to the 2015 dean appointment.",
        parent=timing_node,
        critical=True
    )
    timing_sources = _pick_sources(extraction.completion_year_urls or [], all_urls)
    year_text = extraction.completion_year or "the stated year"
    timing_support_claim = (
        f"The cited page(s) support that Scott Beardsley completed the doctoral degree in {year_text} "
        f"(or earlier in a way consistent with that year) and therefore before or during 2015."
    )
    await evaluator.verify(
        claim=timing_support_claim,
        node=timing_support_leaf,
        sources=timing_sources,
        additional_instruction=(
            "Confirm the completion year on the page. If it shows a year < 2015, that implies 'before 2015'. "
            "If it shows 2015 (and not later), that implies 'during 2015'. If the answer provides no URLs, mark incorrect."
        )
    )

    # ------------------------- Source Credibility ------------------------- #
    src_node = evaluator.add_parallel(
        id="Source_Credibility",
        desc="Reference URLs used are official or otherwise credible biographical/academic-profile sources.",
        parent=top,
        critical=True
    )

    src_leaf = evaluator.add_leaf(
        id="URLs_Are_Official_Or_Credible",
        desc="Cited URLs are from official university pages, academic profiles, or other credible biographical sources.",
        parent=src_node,
        critical=True
    )
    urls_for_cred_check = all_urls
    cred_claim = (
        f"All of the following cited URLs are official or credible sources for biographical/academic information about Scott Beardsley: "
        f"{urls_for_cred_check}."
    )
    await evaluator.verify(
        claim=cred_claim,
        node=src_leaf,
        additional_instruction=(
            "Evaluate the entire list. If the list is empty, mark incorrect. Consider official university domains (.edu, .ac.uk), "
            "official school subdomains (e.g., darden.virginia.edu), recognized accreditors/regulators (.gov, official .org), "
            "and reputable mainstream outlets as credible. If any listed URL is clearly low-credibility or self-published/unverifiable, "
            "mark incorrect."
        )
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
    Evaluate an answer for Scott Beardsley's doctoral credentials.
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

    # Extract structured details from the answer
    extraction: BeardsleyDoctoralExtraction = await evaluator.extract(
        prompt=prompt_extract_beardsley_doctoral_details(),
        template_class=BeardsleyDoctoralExtraction,
        extraction_name="doctoral_details_extraction"
    )

    # Optionally record some custom stats/info
    evaluator.add_custom_info(
        info={
            "all_urls_count": len(_collect_all_urls(extraction)),
            "degree_type": extraction.degree_type,
            "field_program": extraction.field_program,
            "awarding_institution": extraction.awarding_institution,
            "completion_year": extraction.completion_year
        },
        info_type="extraction_summary",
        info_name="extraction_summary"
    )

    # Build and execute verification tree
    await build_verification_tree(evaluator, extraction)

    # Return final structured summary
    return evaluator.get_summary()