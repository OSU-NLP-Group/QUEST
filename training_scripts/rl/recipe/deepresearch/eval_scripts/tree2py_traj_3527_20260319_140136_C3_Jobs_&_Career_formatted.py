import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


TASK_ID = "jmu_president_2025"
TASK_DESCRIPTION = """
James Madison University announced a new president in March 2025 who began serving on July 1, 2025. Identify this president and verify their qualifications by providing: (1) their terminal degree type, field of study, and the institution that granted it; and (2) their most recent prior senior administrative position, including the exact title and the institution where they held that position. Include URL references that document both their educational credentials and their prior administrative role.
"""


# =========================
# Data Models (Extraction)
# =========================
class IdentificationExtraction(BaseModel):
    president_full_name: Optional[str] = None
    announcement_date_text: Optional[str] = None  # e.g., "March 5, 2025" or "March 2025"
    service_start_date_text: Optional[str] = None  # e.g., "July 1, 2025"
    identification_urls: List[str] = Field(default_factory=list)


class EducationExtraction(BaseModel):
    terminal_degree_type: Optional[str] = None  # e.g., "Ph.D." or "Ed.D."
    terminal_degree_field_of_study: Optional[str] = None  # e.g., "Higher Education Administration"
    terminal_degree_granting_institution: Optional[str] = None  # e.g., "University of X"
    education_urls: List[str] = Field(default_factory=list)


class PriorPositionExtraction(BaseModel):
    prior_position_exact_title: Optional[str] = None  # e.g., "Provost and Senior Vice President for Academic Affairs"
    prior_position_institution: Optional[str] = None  # e.g., "University of Y"
    prior_position_urls: List[str] = Field(default_factory=list)


# =========================
# Extraction Prompts
# =========================
def prompt_extract_identification() -> str:
    return """
    Extract the president identification details as presented in the answer.

    Fields to extract:
    - president_full_name: The full name of the person identified as the new president of James Madison University (JMU).
    - announcement_date_text: The announcement date as text exactly as written in the answer (e.g., "March 5, 2025" or "March 2025"). If not provided, return null.
    - service_start_date_text: The service start date as text exactly as written in the answer (e.g., "July 1, 2025"). If not provided, return null.
    - identification_urls: All URLs that the answer cites to document the president’s identity and/or the March 2025 announcement and/or the July 1, 2025 start date.

    URL extraction rules:
    - Extract only URLs that are explicitly present in the answer (plain URLs or Markdown links).
    - Return complete URLs including http:// or https://.
    - Do not fabricate or infer URLs.

    If a field is missing in the answer, set it to null (for single value fields) or [] (for URL arrays).
    """


def prompt_extract_education() -> str:
    return """
    Extract the terminal degree credentials for the identified president.

    Fields to extract:
    - terminal_degree_type: The terminal degree type (e.g., "Ph.D.", "PhD", "Ed.D.", "EdD"). If multiple degrees are listed, choose the terminal academic degree (PhD or EdD) if present. If unclear, return whatever the answer labels as the terminal degree.
    - terminal_degree_field_of_study: The field or discipline of that terminal degree (e.g., "Educational Leadership", "Biology").
    - terminal_degree_granting_institution: The institution that conferred the terminal degree.
    - education_urls: All URLs that the answer cites to document the terminal degree type, field, and granting institution.

    URL extraction rules:
    - Extract only URLs that are explicitly present in the answer (plain URLs or Markdown links).
    - Return complete URLs including http:// or https://.
    - Do not fabricate or infer URLs.

    If a field is missing in the answer, set it to null (for single value fields) or [] (for URL arrays).
    """


def prompt_extract_prior_position() -> str:
    return """
    Extract the most recent prior senior administrative position held by the identified president, immediately before becoming JMU president.

    Fields to extract:
    - prior_position_exact_title: The exact official title of that most recent prior senior administrative role (e.g., "Provost and Senior Vice President for Academic Affairs").
    - prior_position_institution: The institution where that role was held.
    - prior_position_urls: All URLs cited in the answer that document the prior role (title, institution) and/or that it was the most recent role prior to JMU.

    URL extraction rules:
    - Extract only URLs that are explicitly present in the answer (plain URLs or Markdown links).
    - Return complete URLs including http:// or https://.
    - Do not fabricate or infer URLs.

    If a field is missing in the answer, set it to null (for single value fields) or [] (for URL arrays).
    """


# =========================
# Helper Functions
# =========================
def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u2 = u.strip()
        if not u2:
            continue
        if not (u2.startswith("http://") or u2.startswith("https://")):
            u2 = "http://" + u2
        if u2 not in seen:
            seen.add(u2)
            out.append(u2)
    return out


def _union_urls(*lists: Optional[List[str]]) -> List[str]:
    union: List[str] = []
    seen = set()
    for lst in lists:
        for u in _normalize_urls(lst or []):
            if u not in seen:
                seen.add(u)
                union.append(u)
    return union


# =========================
# Verification Subtrees
# =========================
async def verify_identification(
    evaluator: Evaluator,
    parent,
    id_info: IdentificationExtraction
):
    node = evaluator.add_parallel(
        id="President_Identification",
        desc="Correctly identify the president and confirm the announcement/start timing with documentation.",
        parent=parent,
        critical=True
    )

    id_urls = _normalize_urls(id_info.identification_urls)

    # President_Full_Name
    leaf_full_name = evaluator.add_leaf(
        id="President_Full_Name",
        desc="Provide the president's full name.",
        parent=node,
        critical=True
    )
    claim_full_name = (
        f"The next president of James Madison University is {id_info.president_full_name}."
        if id_info.president_full_name else
        "The next president of James Madison University is <missing>."
    )
    await evaluator.verify(
        claim=claim_full_name,
        node=leaf_full_name,
        sources=id_urls,
        additional_instruction="Verify identity only. Accept if the page clearly names this person as the next or new president of JMU."
    )

    # Announcement_Timing
    leaf_announce = evaluator.add_leaf(
        id="Announcement_Timing",
        desc="Confirm the president was announced in March 2025.",
        parent=node,
        critical=True
    )
    nm = id_info.president_full_name or "the president"
    claim_announce = f"The announcement that {nm} would be JMU's next president occurred in March 2025."
    await evaluator.verify(
        claim=claim_announce,
        node=leaf_announce,
        sources=id_urls,
        additional_instruction="Accept if a cited page date (press release, news article, etc.) is in March 2025; any day in March 2025 counts."
    )

    # Service_Start_Date
    leaf_start = evaluator.add_leaf(
        id="Service_Start_Date",
        desc="Confirm the president began serving on July 1, 2025.",
        parent=node,
        critical=True
    )
    claim_start = f"{nm} began serving as JMU president on July 1, 2025."
    await evaluator.verify(
        claim=claim_start,
        node=leaf_start,
        sources=id_urls,
        additional_instruction="Accept common phrasings like 'effective July 1, 2025' or 'began on July 1, 2025'."
    )

    # Identification_URL_References -> split into three concrete checks under a container for better diagnostics
    refs_container = evaluator.add_parallel(
        id="Identification_URL_References",
        desc="Provide URL reference(s) that collectively document the president’s identity and the March 2025 announcement and July 1, 2025 start date.",
        parent=node,
        critical=True
    )

    # At least one URL documents identity
    leaf_refs_identity = evaluator.add_leaf(
        id="Identification_URL_References_identity_supported",
        desc="At least one cited URL documents the president’s identity.",
        parent=refs_container,
        critical=True
    )
    claim_refs_identity = f"At least one of these URLs states that {nm} is (or will be) the president of James Madison University: {id_urls}."
    await evaluator.verify(
        claim=claim_refs_identity,
        node=leaf_refs_identity,
        sources=id_urls,
        additional_instruction="Pass if any single URL explicitly identifies this person as JMU's next/new president."
    )

    # At least one URL documents March 2025 announcement
    leaf_refs_announce = evaluator.add_leaf(
        id="Identification_URL_References_announcement_supported",
        desc="At least one cited URL documents the March 2025 announcement.",
        parent=refs_container,
        critical=True
    )
    claim_refs_announce = f"At least one of these URLs shows that the announcement occurred in March 2025: {id_urls}."
    await evaluator.verify(
        claim=claim_refs_announce,
        node=leaf_refs_announce,
        sources=id_urls,
        additional_instruction="Pass if any single URL has a publication or announcement date in March 2025."
    )

    # At least one URL documents July 1, 2025 start date
    leaf_refs_start = evaluator.add_leaf(
        id="Identification_URL_References_startdate_supported",
        desc="At least one cited URL documents the July 1, 2025 start date.",
        parent=refs_container,
        critical=True
    )
    claim_refs_start = f"At least one of these URLs states that the start date was July 1, 2025: {id_urls}."
    await evaluator.verify(
        claim=claim_refs_start,
        node=leaf_refs_start,
        sources=id_urls,
        additional_instruction="Pass if any single URL explicitly mentions July 1, 2025 as the start date."
    )


async def verify_education(
    evaluator: Evaluator,
    parent,
    edu_info: EducationExtraction,
    id_info: IdentificationExtraction
):
    node = evaluator.add_parallel(
        id="Educational_Credentials",
        desc="Provide and verify the president’s terminal degree details (type, field, institution) with documentation.",
        parent=parent,
        critical=True
    )

    edu_urls = _normalize_urls(edu_info.education_urls)

    # Terminal_Degree_Type
    leaf_type = evaluator.add_leaf(
        id="Terminal_Degree_Type",
        desc="State the terminal degree type and ensure it is a PhD or EdD.",
        parent=node,
        critical=True
    )
    deg_type = edu_info.terminal_degree_type or "<missing>"
    nm = id_info.president_full_name or "the president"
    claim_type = f"The terminal degree type for {nm} is {deg_type}, and it is one of PhD (Ph.D.) or EdD (Ed.D.)."
    await evaluator.verify(
        claim=claim_type,
        node=leaf_type,
        sources=edu_urls,
        additional_instruction="Verify that the page explicitly lists a terminal academic degree of type PhD/Ph.D. or EdD/Ed.D. If the degree type is not one of these, mark incorrect."
    )

    # Terminal_Degree_Field_of_Study
    leaf_field = evaluator.add_leaf(
        id="Terminal_Degree_Field_of_Study",
        desc="State the terminal degree field of study.",
        parent=node,
        critical=True
    )
    field = edu_info.terminal_degree_field_of_study or "<missing>"
    claim_field = f"The field of study for {nm}'s terminal degree is {field}."
    await evaluator.verify(
        claim=claim_field,
        node=leaf_field,
        sources=edu_urls,
        additional_instruction="Verify the field or discipline as explicitly stated on the cited page(s); allow minor wording variations."
    )

    # Terminal_Degree_Granting_Institution
    leaf_inst = evaluator.add_leaf(
        id="Terminal_Degree_Granting_Institution",
        desc="State the institution that granted the terminal degree.",
        parent=node,
        critical=True
    )
    inst = edu_info.terminal_degree_granting_institution or "<missing>"
    claim_inst = f"The institution that granted {nm}'s terminal degree is {inst}."
    await evaluator.verify(
        claim=claim_inst,
        node=leaf_inst,
        sources=edu_urls,
        additional_instruction="Verify the granting institution name as shown on the cited page(s); allow minor name variants (e.g., with/without 'University')."
    )

    # Education_URL_References -> split into concrete checks
    refs_container = evaluator.add_parallel(
        id="Education_URL_References",
        desc="Provide URL reference(s) that collectively document the terminal degree type, field of study, and granting institution.",
        parent=node,
        critical=True
    )

    leaf_refs_type = evaluator.add_leaf(
        id="Education_URL_References_type_supported",
        desc="At least one cited URL documents the terminal degree type.",
        parent=refs_container,
        critical=True
    )
    claim_refs_type = f"At least one of these URLs documents the terminal degree type '{deg_type}': {edu_urls}."
    await evaluator.verify(
        claim=claim_refs_type,
        node=leaf_refs_type,
        sources=edu_urls,
        additional_instruction="Pass if any single URL explicitly states the terminal degree type."
    )

    leaf_refs_field = evaluator.add_leaf(
        id="Education_URL_References_field_supported",
        desc="At least one cited URL documents the terminal degree field of study.",
        parent=refs_container,
        critical=True
    )
    claim_refs_field = f"At least one of these URLs documents the terminal degree field '{field}': {edu_urls}."
    await evaluator.verify(
        claim=claim_refs_field,
        node=leaf_refs_field,
        sources=edu_urls,
        additional_instruction="Pass if any single URL explicitly states the degree field/discipline."
    )

    leaf_refs_inst = evaluator.add_leaf(
        id="Education_URL_References_institution_supported",
        desc="At least one cited URL documents the terminal degree granting institution.",
        parent=refs_container,
        critical=True
    )
    claim_refs_inst = f"At least one of these URLs documents the granting institution '{inst}': {edu_urls}."
    await evaluator.verify(
        claim=claim_refs_inst,
        node=leaf_refs_inst,
        sources=edu_urls,
        additional_instruction="Pass if any single URL explicitly names the granting institution for the terminal degree."
    )


async def verify_prior_position(
    evaluator: Evaluator,
    parent,
    prior_info: PriorPositionExtraction,
    id_info: IdentificationExtraction
):
    node = evaluator.add_parallel(
        id="Prior_Senior_Administrative_Position",
        desc="Provide and verify the president’s most recent prior senior administrative position (title and institution) with documentation.",
        parent=parent,
        critical=True
    )

    prior_urls = _normalize_urls(prior_info.prior_position_urls)
    id_urls = _normalize_urls(id_info.identification_urls)
    combined_urls = _union_urls(prior_urls, id_urls)

    nm = id_info.president_full_name or "the president"
    title = prior_info.prior_position_exact_title or "<missing>"
    inst = prior_info.prior_position_institution or "<missing>"

    # Prior_Position_Exact_Title
    leaf_title = evaluator.add_leaf(
        id="Prior_Position_Exact_Title",
        desc="Provide the exact title of the most recent prior senior administrative position.",
        parent=node,
        critical=True
    )
    claim_title = f"The most recent prior senior administrative position held by {nm} was titled '{title}'."
    await evaluator.verify(
        claim=claim_title,
        node=leaf_title,
        sources=combined_urls,
        additional_instruction="Verify the exact official title as written on the cited page(s); minor punctuation/casing variants acceptable."
    )

    # Prior_Position_Institution
    leaf_prior_inst = evaluator.add_leaf(
        id="Prior_Position_Institution",
        desc="Provide the institution where the most recent prior senior administrative position was held.",
        parent=node,
        critical=True
    )
    claim_prior_inst = f"The institution where {nm} held the most recent prior senior administrative position was {inst}."
    await evaluator.verify(
        claim=claim_prior_inst,
        node=leaf_prior_inst,
        sources=combined_urls,
        additional_instruction="Verify the institution name associated with the prior position as shown on the cited page(s)."
    )

    # Prior_Position_Senior_Admin_Level
    leaf_senior = evaluator.add_leaf(
        id="Prior_Position_Senior_Admin_Level",
        desc="Confirm the prior position qualifies as a senior administrative role (e.g., chancellor, provost, dean, vice president, or equivalent).",
        parent=node,
        critical=True
    )
    claim_senior = (
        f"The role '{title}' at {inst} qualifies as a senior administrative position (e.g., chancellor, provost, dean, vice president, or equivalent)."
    )
    await evaluator.verify(
        claim=claim_senior,
        node=leaf_senior,
        sources=combined_urls,
        additional_instruction="Judge based on the title; positions like Chancellor, President, Provost, Vice President, Vice Chancellor, Dean, or equivalent C-level university roles count as senior administrative."
    )

    # Prior_Position_Is_Most_Recent
    leaf_recent = evaluator.add_leaf(
        id="Prior_Position_Is_Most_Recent",
        desc="Confirm the provided senior administrative position is the most recent one held immediately prior to becoming JMU president.",
        parent=node,
        critical=True
    )
    claim_recent = f"The position '{title}' at {inst} was the most recent senior administrative role held by {nm} immediately prior to becoming JMU president."
    await evaluator.verify(
        claim=claim_recent,
        node=leaf_recent,
        sources=combined_urls,
        additional_instruction="Often the JMU announcement page states the person's then-current role; accept if the cited page(s) indicate this was the position held immediately before JMU."
    )

    # Prior_Position_URL_References -> split into concrete checks
    refs_container = evaluator.add_parallel(
        id="Prior_Position_URL_References",
        desc="Provide URL reference(s) that collectively document the prior position’s exact title, institution, and that it was the most recent prior senior administrative role.",
        parent=node,
        critical=True
    )

    # Title documented
    leaf_refs_title = evaluator.add_leaf(
        id="Prior_Position_URL_References_title_supported",
        desc="At least one cited URL documents the exact prior position title.",
        parent=refs_container,
        critical=True
    )
    claim_refs_title = f"At least one of these URLs documents the prior position title '{title}': {combined_urls}."
    await evaluator.verify(
        claim=claim_refs_title,
        node=leaf_refs_title,
        sources=combined_urls,
        additional_instruction="Pass if any single URL explicitly states the exact prior position title."
    )

    # Institution documented
    leaf_refs_inst = evaluator.add_leaf(
        id="Prior_Position_URL_References_institution_supported",
        desc="At least one cited URL documents the institution of the prior position.",
        parent=refs_container,
        critical=True
    )
    claim_refs_inst = f"At least one of these URLs documents the institution '{inst}' for the prior position: {combined_urls}."
    await evaluator.verify(
        claim=claim_refs_inst,
        node=leaf_refs_inst,
        sources=combined_urls,
        additional_instruction="Pass if any single URL explicitly states the institution for that prior role."
    )

    # Most recent documented
    leaf_refs_recent = evaluator.add_leaf(
        id="Prior_Position_URL_References_most_recent_supported",
        desc="At least one cited URL documents that this position was the most recent prior role before JMU presidency.",
        parent=refs_container,
        critical=True
    )
    claim_refs_recent = f"At least one of these URLs indicates that '{title}' at {inst} was the most recent senior administrative role before becoming JMU president: {combined_urls}."
    await evaluator.verify(
        claim=claim_refs_recent,
        node=leaf_refs_recent,
        sources=combined_urls,
        additional_instruction="Pass if any single URL explicitly states or clearly implies it was the immediately prior role."
    )


async def verify_source_quality(
    evaluator: Evaluator,
    parent,
    all_urls: List[str]
):
    node = evaluator.add_parallel(
        id="Source_Quality_Check",
        desc="Ensure all cited URLs are from official university pages or credible news outlets.",
        parent=parent,
        critical=True
    )

    # Guard: ensure we actually have URLs to check
    urls_present = evaluator.add_custom_node(
        result=len(all_urls) > 0,
        id="URLs_present_for_quality_check",
        desc="At least one URL was provided for quality checking.",
        parent=node,
        critical=True
    )

    # Add one leaf per URL to verify credibility/official status
    for idx, url in enumerate(all_urls):
        leaf = evaluator.add_leaf(
            id=f"URLs_Are_Official_or_Credible_{idx+1}",
            desc=f"URL is official university or credible news outlet: {url}",
            parent=node,
            critical=True
        )
        claim = (
            f"The URL {url} is either an official university webpage (e.g., .edu or an official institutional domain) "
            f"or a credible news outlet."
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=url,
            additional_instruction=(
                "Use the domain and page context to judge: official university (.edu, jmu.edu, or an institution's official domain) "
                "or reputable news organizations (AP, Reuters, major newspapers, recognized local/regional outlets, NPR/PBS affiliates). "
                "Do NOT count self-published blogs, generic content farms, or random wikis as credible. Official university newsroom/press pages count."
            )
        )


# =========================
# Main Evaluation Function
# =========================
async def evaluate_answer(
    client: LLMClient,
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
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract information in parallel
    id_task = evaluator.extract(
        prompt=prompt_extract_identification(),
        template_class=IdentificationExtraction,
        extraction_name="identification"
    )
    edu_task = evaluator.extract(
        prompt=prompt_extract_education(),
        template_class=EducationExtraction,
        extraction_name="education"
    )
    prior_task = evaluator.extract(
        prompt=prompt_extract_prior_position(),
        template_class=PriorPositionExtraction,
        extraction_name="prior_position"
    )

    identification_info, education_info, prior_info = await asyncio.gather(id_task, edu_task, prior_task)

    # Build main critical sequential node according to rubric
    main = evaluator.add_sequential(
        id="President_Identification_and_Verification",
        desc="Identify the JMU president announced in March 2025 who began serving July 1, 2025, and provide verified terminal-degree and most-recent prior senior administrative-position details with supporting URLs.",
        parent=root,
        critical=True
    )

    # 1) Identification subtree
    await verify_identification(evaluator, main, identification_info)

    # 2) Educational credentials subtree
    await verify_education(evaluator, main, education_info, identification_info)

    # 3) Prior senior administrative position subtree
    await verify_prior_position(evaluator, main, prior_info, identification_info)

    # 4) Source quality check subtree
    all_cited_urls = _union_urls(
        identification_info.identification_urls,
        education_info.education_urls,
        prior_info.prior_position_urls
    )
    await verify_source_quality(evaluator, main, all_cited_urls)

    return evaluator.get_summary()