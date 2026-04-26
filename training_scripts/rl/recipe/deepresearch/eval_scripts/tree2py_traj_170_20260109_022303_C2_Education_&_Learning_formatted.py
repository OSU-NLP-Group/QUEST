import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hlc_online_edlead_masters"
TASK_DESCRIPTION = (
    "Identify a university that is regionally accredited by the Higher Learning Commission (HLC) and "
    "offers a fully online master's degree program in Educational Leadership."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramTaskExtraction(BaseModel):
    """
    Extracted facts from the agent's answer needed for verification.
    """
    university_name: Optional[str] = None
    program_name: Optional[str] = None

    # Free-text descriptors to help verification
    degree_level: Optional[str] = None          # e.g., "Master of Education (M.Ed.)", "MA", "MS", "Master's"
    degree_field: Optional[str] = None          # e.g., "Educational Leadership", "School Leadership", "Educational Administration"
    online_format_description: Optional[str] = None  # e.g., "fully online", "100% online", etc.

    # URLs explicitly cited in the answer
    hlc_accreditation_urls: List[str] = Field(default_factory=list)  # URLs supporting HLC accreditation
    program_urls: List[str] = Field(default_factory=list)            # URLs supporting program details (online format, field, degree level)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program_task() -> str:
    return """
    From the provided answer, extract the following information about ONE university (if multiple are listed, pick the first one) and its program:

    Required string fields (return null if missing):
    - university_name: The name of the institution.
    - program_name: The exact name of the master's program the answer proposes.
    - degree_level: The degree level as written (e.g., "Master of Education", "M.Ed.", "MA", "MS", "Master's").
    - degree_field: The field (e.g., "Educational Leadership", "School Leadership", "Educational Administration").
    - online_format_description: The exact wording about the program's online delivery (e.g., "fully online", "100% online", "no campus visits required").

    Required URL lists (only include URLs explicitly present in the answer):
    - hlc_accreditation_urls: All URLs in the answer that support that the university is regionally accredited by the Higher Learning Commission (HLC). These could include the HLC member directory page or the university's accreditation page explicitly citing HLC.
    - program_urls: All URLs in the answer that describe the program and its online format (university program page, program overview, catalog page, etc.).

    IMPORTANT:
    - Do NOT invent or infer URLs. Include only URLs explicitly present in the answer (plain URLs or markdown links).
    - Always include full URLs with protocol. If a URL is missing a protocol, prepend http://.
    - If a required string field is not stated, return null for that field.
    - If no URLs of a required list are present, return an empty list for that list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*url_lists: Optional[List[str]]) -> List[str]:
    """Combine multiple URL lists, deduplicate, and keep order."""
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


def _sources_or_none(urls: List[str]) -> Optional[List[str]]:
    """Return None if empty to allow simple verify fallback (though multi-URL verify is preferred)."""
    return urls if urls else None


def _safe(val: Optional[str], fallback: str) -> str:
    """Return a safe non-empty string for claim construction."""
    return val.strip() if isinstance(val, str) and val.strip() else fallback


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: ProgramTaskExtraction) -> None:
    """
    Build the verification tree according to the rubric and run all verifications.
    """
    root = evaluator.root

    # ProgramMeetsAllRequirements (critical, parallel)
    node_all = evaluator.add_parallel(
        id="ProgramMeetsAllRequirements",
        desc="The identified university and program satisfy all specified requirements for an online master's degree in Educational Leadership with HLC accreditation",
        parent=root,
        critical=True,
    )

    # Child 1: HLCAccreditation (leaf, critical)
    node_hlc = evaluator.add_leaf(
        id="HLCAccreditation",
        desc="The institution is regionally accredited by the Higher Learning Commission (HLC)",
        parent=node_all,
        critical=True,
    )

    uni_name = _safe(extracted.university_name, "the institution")
    hlc_sources = _combine_sources(extracted.hlc_accreditation_urls, extracted.program_urls)

    claim_hlc = f"The institution {uni_name} is regionally accredited by the Higher Learning Commission (HLC)."
    await evaluator.verify(
        claim=claim_hlc,
        node=node_hlc,
        sources=_sources_or_none(hlc_sources),
        additional_instruction=(
            "Verify that the cited page(s) explicitly indicate regional accreditation by the Higher Learning Commission. "
            "Look for phrases like 'Higher Learning Commission', 'HLC', or links to hlcommission.org. "
            "If the page only states general accreditation without mentioning HLC specifically, consider the claim not supported."
        ),
    )

    # Child 2: ProgramSpecifications (critical, parallel)
    node_specs = evaluator.add_parallel(
        id="ProgramSpecifications",
        desc="The program meets all required characteristics for format, level, and field of study",
        parent=node_all,
        critical=True,
    )

    program_sources = _combine_sources(extracted.program_urls)

    # OnlineFormat (leaf, critical)
    node_online = evaluator.add_leaf(
        id="OnlineFormat",
        desc="The program is offered in a fully online format (100% online)",
        parent=node_specs,
        critical=True,
    )

    program_name = _safe(extracted.program_name, "the program")
    claim_online = (
        f"The {program_name} program at {uni_name} is offered fully online (i.e., 100% online with no required on-campus attendance)."
    )
    await evaluator.verify(
        claim=claim_online,
        node=node_online,
        sources=_sources_or_none(program_sources),
        additional_instruction=(
            "Confirm that the page states 'fully online', '100% online', 'delivered fully online', or equivalent language. "
            "Accept phrases such as 'no campus visits required' or 'entirely online'. "
            "Do NOT accept hybrid, blended, mostly/partially online, or programs requiring residency/onsite intensives."
        ),
    )

    # DegreeLevel (leaf, critical)
    node_level = evaluator.add_leaf(
        id="DegreeLevel",
        desc="The program is at the master's degree level (e.g., MEd or MA)",
        parent=node_specs,
        critical=True,
    )

    level_desc = _safe(extracted.degree_level, "a master's degree")
    claim_level = (
        f"The {program_name} at {uni_name} is a master's degree program (e.g., Master of Education (M.Ed.), MA, MS in Education)."
    )
    await evaluator.verify(
        claim=claim_level,
        node=node_level,
        sources=_sources_or_none(program_sources),
        additional_instruction=(
            "Verify that the program is a master's degree (Master of Education/M.Ed., Master of Arts/MA, Master of Science/MS, etc.). "
            "Do NOT accept graduate certificates, post-master's (Ed.S.), or doctoral degrees (Ed.D./Ph.D.). "
            "If the page is ambiguous or only says 'graduate' without indicating 'master's', consider the claim unsupported."
        ),
    )

    # DegreeField (leaf, critical)
    node_field = evaluator.add_leaf(
        id="DegreeField",
        desc="The program is in the field of Educational Leadership",
        parent=node_specs,
        critical=True,
    )

    field_desc = _safe(extracted.degree_field, "Educational Leadership")
    claim_field = (
        f"The {program_name} at {uni_name} is in the field of Educational Leadership."
    )
    await evaluator.verify(
        claim=claim_field,
        node=node_field,
        sources=_sources_or_none(program_sources),
        additional_instruction=(
            "Check that the program field is Educational Leadership. "
            "Allow close, commonly accepted synonyms such as 'Educational Administration', 'School Leadership', or 'Education Leadership' "
            "when they clearly denote the same field. If the page indicates a different field (e.g., curriculum, counseling), do not accept."
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
    Evaluate an answer for the HLC-accredited fully online Educational Leadership master's program task.
    Returns the evaluation summary dictionary.
    """
    # Initialize evaluator (root is non-critical; we add a critical child as per rubric)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_program_task(),
        template_class=ProgramTaskExtraction,
        extraction_name="program_task_extraction",
    )

    # Optional: record a compact custom info summary
    evaluator.add_custom_info(
        info={
            "university_name": extracted.university_name,
            "program_name": extracted.program_name,
            "degree_level": extracted.degree_level,
            "degree_field": extracted.degree_field,
            "online_format_description": extracted.online_format_description,
            "hlc_accreditation_urls_count": len(extracted.hlc_accreditation_urls or []),
            "program_urls_count": len(extracted.program_urls or []),
        },
        info_type="extracted_summary",
    )

    # Build tree and run verifications
    await build_and_verify_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()