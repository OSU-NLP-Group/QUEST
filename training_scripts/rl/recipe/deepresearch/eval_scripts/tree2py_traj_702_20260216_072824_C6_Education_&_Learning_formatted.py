import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "uva_presidency_accreditation_pathway"
TASK_DESCRIPTION = """
The University of Virginia appointed its 10th president in January 2026. This president previously served as dean of UVA's Darden School of Business since 2015 and holds a doctorate in higher education management from a Pennsylvania university. Identify: (1) the president's name, (2) the Pennsylvania institution where they earned their doctorate and that institution's regional accrediting body, (3) a different Pennsylvania university accredited by the same regional body that offers at least 10 doctoral programs, and (4) whether that second university offers online graduate degree programs.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PresidentInfo(BaseModel):
    name: Optional[str] = None
    appointment_timing_text: Optional[str] = None  # e.g., "January 2026"
    darden_dean_since: Optional[str] = None        # e.g., "2015"
    reference_urls: List[str] = Field(default_factory=list)


class DoctorateInfo(BaseModel):
    institution_name: Optional[str] = None
    field: Optional[str] = None  # e.g., "higher education management"
    state: Optional[str] = None  # Expect "Pennsylvania" (case-insensitive allowed)
    reference_urls: List[str] = Field(default_factory=list)


class AccreditorInfo(BaseModel):
    accreditor_name: Optional[str] = None
    is_regional: Optional[bool] = None
    reference_urls: List[str] = Field(default_factory=list)


class ComparableUniversityInfo(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None  # Expect "Pennsylvania"
    reference_urls: List[str] = Field(default_factory=list)         # general refs
    accreditor_urls: List[str] = Field(default_factory=list)        # accreditation proof
    doctoral_programs_urls: List[str] = Field(default_factory=list) # program list/catalog refs
    doctoral_programs_count_claim: Optional[str] = None             # e.g., "at least 12", "≥10", "10+"


class OnlineProgramsInfo(BaseModel):
    offers_online_graduate_degrees: Optional[str] = None  # "yes"/"no"/"true"/"false"
    reference_urls: List[str] = Field(default_factory=list)


class AllExtraction(BaseModel):
    president: Optional[PresidentInfo] = None
    doctorate: Optional[DoctorateInfo] = None
    accreditor: Optional[AccreditorInfo] = None
    comparable_university: Optional[ComparableUniversityInfo] = None
    online: Optional[OnlineProgramsInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract structured information from the answer. Return a JSON object strictly following the schema below. 
    Only extract what is explicitly present in the answer text, including URLs exactly as written (convert markdown links to bare URLs). 
    If something is missing in the answer, set the corresponding field to null or an empty array as appropriate.

    Required JSON schema:
    {
      "president": {
        "name": string | null,
        "appointment_timing_text": string | null,   // e.g., "January 2026"
        "darden_dean_since": string | null,         // e.g., "2015"
        "reference_urls": string[]                  // URLs that support the president identification/background
      },
      "doctorate": {
        "institution_name": string | null,          // the Pennsylvania institution where the president earned the doctorate
        "field": string | null,                     // the doctorate field, e.g., "higher education management"
        "state": string | null,                     // the stated location state of that institution if mentioned (aim for "Pennsylvania")
        "reference_urls": string[]                  // URLs that support the doctoral institution and field
      },
      "accreditor": {
        "accreditor_name": string | null,           // the regional accrediting body's name for the doctoral institution
        "is_regional": boolean | null,              // whether it is a regional (institutional) accreditor (not programmatic)
        "reference_urls": string[]                  // URLs that support the accreditor information
      },
      "comparable_university": {
        "name": string | null,                      // a different Pennsylvania university
        "state": string | null,                     // its state if mentioned
        "reference_urls": string[],                 // general URLs about this university (e.g., overview/about page)
        "accreditor_urls": string[],                // URLs supporting accreditation by the same regional accreditor
        "doctoral_programs_urls": string[],         // URLs supporting that it offers at least 10 doctoral programs
        "doctoral_programs_count_claim": string | null  // the claimed count phrasing present in the answer (e.g., "at least 10", "12")
      },
      "online": {
        "offers_online_graduate_degrees": string | null, // yes/no/true/false (any reasonable yes/no variants)
        "reference_urls": string[]                  // URLs supporting whether the comparable university offers online graduate degree programs
      }
    }

    Additional instructions:
    - For URLs, extract only ones explicitly present in the answer; do not invent or infer URLs.
    - If a URL is written in markdown link format [text](url), extract the url part.
    - Do not normalize names; extract them as written.
    - If the answer lists multiple URLs for a subpart, include them all in the corresponding array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_yes_no(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"yes", "true", "y", "1"}:
        return True
    if v in {"no", "false", "n", "0"}:
        return False
    return None


def _combine_urls(*url_lists: Optional[List[str]]) -> List[str]:
    """
    Combine multiple url lists into a single deduplicated list,
    filtering out obviously invalid entries.
    """
    seen = set()
    result: List[str] = []
    for urls in url_lists:
        if not urls:
            continue
        for u in urls:
            if not u:
                continue
            u2 = u.strip()
            if not u2:
                continue
            # Accept http(s) and common www prefixes; other schemes ignored
            if not (u2.startswith("http://") or u2.startswith("https://")):
                # Allow missing protocol by prepending http://
                if u2.startswith("www."):
                    u2 = f"http://{u2}"
                else:
                    # skip malformed
                    continue
            if u2 not in seen:
                seen.add(u2)
                result.append(u2)
    return result


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_president(
    evaluator: Evaluator,
    parent_node,
    data: AllExtraction,
) -> None:
    """
    identify_uva_president:
      - president_reference_urls (existence)
      - president_core_info (parallel):
          - president_full_name
          - president_appointment_timing (Jan 2026 and "10th president")
          - president_darden_since_2015
    """
    node = evaluator.add_sequential(
        id="identify_uva_president",
        desc="Correctly identify UVA's 10th president who took office in January 2026",
        parent=parent_node,
        critical=True,
    )

    pres = data.president or PresidentInfo()
    pres_urls = _combine_urls(pres.reference_urls)

    # Existence of reference URLs
    evaluator.add_custom_node(
        result=bool(pres_urls),
        id="president_reference_urls",
        desc="Provide valid URL references supporting president identification",
        parent=node,
        critical=True
    )

    core = evaluator.add_parallel(
        id="president_core_info",
        desc="Provide president's name and verify key background details",
        parent=node,
        critical=True
    )

    # Leaf: president_full_name
    full_name_node = evaluator.add_leaf(
        id="president_full_name",
        desc="Provide the president's full name",
        parent=core,
        critical=True
    )
    name_for_claim = pres.name or ""
    await evaluator.verify(
        claim=f"The individual identified in these sources as the University of Virginia's new president is named '{name_for_claim}'.",
        node=full_name_node,
        sources=pres_urls,
        additional_instruction="Verify that the person's full name appears clearly on the provided sources; minor variations such as middle initials should be accepted."
    )

    # Leaf: appointment timing + 10th president
    appoint_node = evaluator.add_leaf(
        id="president_appointment_timing",
        desc="Verify appointment timing (January 2026) and being the 10th president",
        parent=core,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_for_claim} took office as the University of Virginia's 10th president in January 2026.",
        node=appoint_node,
        sources=pres_urls,
        additional_instruction="Look for explicit mentions of '10th president' and the appointment taking office in January 2026."
    )

    # Leaf: Darden dean since 2015
    darden_node = evaluator.add_leaf(
        id="president_darden_since_2015",
        desc="Verify previous role as Darden dean since 2015",
        parent=core,
        critical=True
    )
    await evaluator.verify(
        claim=f"Before becoming president, {name_for_claim} served as dean of UVA's Darden School of Business since 2015.",
        node=darden_node,
        sources=pres_urls,
        additional_instruction="Allow wording like 'has served as dean since 2015' or similar; confirm the start year is 2015."
    )


async def verify_doctoral_institution(
    evaluator: Evaluator,
    parent_node,
    data: AllExtraction
) -> None:
    """
    identify_doctoral_institution:
      - doctoral_reference_urls (existence)
      - institution_details (parallel):
          - institution_name
          - pennsylvania_location
          - higher_ed_management_field
    """
    node = evaluator.add_sequential(
        id="identify_doctoral_institution",
        desc="Identify the Pennsylvania institution where the president earned their doctorate in higher education management",
        parent=parent_node,
        critical=True
    )

    pres = data.president or PresidentInfo()
    doc = data.doctorate or DoctorateInfo()
    doc_urls = _combine_urls(doc.reference_urls)

    evaluator.add_custom_node(
        result=bool(doc_urls),
        id="doctoral_reference_urls",
        desc="Provide valid URL references supporting doctoral institution identification",
        parent=node,
        critical=True
    )

    details = evaluator.add_parallel(
        id="institution_details",
        desc="Provide complete institution information",
        parent=node,
        critical=True
    )

    # Leaf: institution_name
    inst_name_node = evaluator.add_leaf(
        id="institution_name",
        desc="Provide the institution name",
        parent=details,
        critical=True
    )
    inst_name = doc.institution_name or ""
    await evaluator.verify(
        claim=f"{pres.name or 'The president'} earned a doctoral degree from {inst_name}.",
        node=inst_name_node,
        sources=doc_urls,
        additional_instruction="The source(s) should explicitly associate the named person with a doctoral degree from the specified institution."
    )

    # Leaf: pennsylvania_location
    pa_loc_node = evaluator.add_leaf(
        id="pennsylvania_location",
        desc="Verify Pennsylvania location",
        parent=details,
        critical=True
    )
    await evaluator.verify(
        claim=f"{inst_name} is located in Pennsylvania.",
        node=pa_loc_node,
        sources=doc_urls,
        additional_instruction="Accept explicit mention of the state or a campus address clearly in Pennsylvania."
    )

    # Leaf: higher_ed_management_field
    field_node = evaluator.add_leaf(
        id="higher_ed_management_field",
        desc="Verify doctorate is in higher education management",
        parent=details,
        critical=True
    )
    await evaluator.verify(
        claim=f"{pres.name or 'The president'} earned a doctorate in higher education management at {inst_name}.",
        node=field_node,
        sources=doc_urls,
        additional_instruction="Wording like 'Ed.D. in Higher Education Management' or similar should count as a match."
    )


async def verify_accreditor(
    evaluator: Evaluator,
    parent_node,
    data: AllExtraction
) -> None:
    """
    identify_regional_accreditor:
      - accreditation_reference_urls (existence)
      - accreditor_details (parallel):
          - accreditor_full_name
          - regional_accreditor_type
    """
    node = evaluator.add_sequential(
        id="identify_regional_accreditor",
        desc="Identify the regional accrediting body for the doctoral institution",
        parent=parent_node,
        critical=True
    )

    doc = data.doctorate or DoctorateInfo()
    accr = data.accreditor or AccreditorInfo()
    accr_urls = _combine_urls(accr.reference_urls)

    evaluator.add_custom_node(
        result=bool(accr_urls),
        id="accreditation_reference_urls",
        desc="Provide valid URL references supporting accreditation information",
        parent=node,
        critical=True
    )

    details = evaluator.add_parallel(
        id="accreditor_details",
        desc="Provide accreditor name and verify it is regional",
        parent=node,
        critical=True
    )

    # Leaf: accreditor_full_name
    accred_name_node = evaluator.add_leaf(
        id="accreditor_full_name",
        desc="Provide the full name of the regional accrediting body",
        parent=details,
        critical=True
    )
    accred_name = accr.accreditor_name or ""
    await evaluator.verify(
        claim=f"{doc.institution_name or 'The institution'} is accredited by {accred_name}.",
        node=accred_name_node,
        sources=accr_urls,
        additional_instruction="Look for institutional accreditation statements on the accreditor's site or the institution's accreditation page."
    )

    # Leaf: regional_accreditor_type
    regional_type_node = evaluator.add_leaf(
        id="regional_accreditor_type",
        desc="Verify it is a regional (not specialized) accrediting body",
        parent=details,
        critical=True
    )
    await evaluator.verify(
        claim=f"{accred_name} is a regional institutional accrediting body in the United States (not a specialized/programmatic accreditor).",
        node=regional_type_node,
        sources=accr_urls,
        additional_instruction="Accept authoritative statements (e.g., accreditor's own description or recognized listings) that classify it as a regional, institutional accreditor."
    )


async def verify_comparable_university(
    evaluator: Evaluator,
    parent_node,
    data: AllExtraction
) -> None:
    """
    find_comparable_university:
      - university_reference_urls (existence)
      - university_identification_and_verification (parallel):
          - university_name
          - university_in_pennsylvania
          - university_different_from_doctoral (custom logic)
          - same_accreditor_verification
          - minimum_doctoral_programs
    """
    node = evaluator.add_sequential(
        id="find_comparable_university",
        desc="Find a different Pennsylvania university with same accreditor offering at least 10 doctoral programs",
        parent=parent_node,
        critical=True
    )

    doc = data.doctorate or DoctorateInfo()
    accr = data.accreditor or AccreditorInfo()
    comp = data.comparable_university or ComparableUniversityInfo()

    combined_refs = _combine_urls(
        comp.reference_urls,
        comp.accreditor_urls,
        comp.doctoral_programs_urls,
    )

    evaluator.add_custom_node(
        result=bool(combined_refs),
        id="university_reference_urls",
        desc="Provide valid URL references supporting comparable university identification",
        parent=node,
        critical=True
    )

    verify_group = evaluator.add_parallel(
        id="university_identification_and_verification",
        desc="Identify qualifying university and verify all requirements",
        parent=node,
        critical=True
    )

    # Leaf: university_name
    uni_name_node = evaluator.add_leaf(
        id="university_name",
        desc="Provide the name of a qualifying university",
        parent=verify_group,
        critical=True
    )
    uni_name = comp.name or ""
    await evaluator.verify(
        claim=f"The qualifying university is {uni_name}.",
        node=uni_name_node,
        sources=combined_refs,
        additional_instruction="Confirm the university's name is clearly stated on the provided sources."
    )

    # Leaf: university_in_pennsylvania
    uni_pa_node = evaluator.add_leaf(
        id="university_in_pennsylvania",
        desc="Verify university is in Pennsylvania",
        parent=verify_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} is located in Pennsylvania.",
        node=uni_pa_node,
        sources=combined_refs,
        additional_instruction="Accept explicit mentions of the state or addresses clearly in Pennsylvania."
    )

    # Leaf: university_different_from_doctoral (custom logic)
    evaluator.add_custom_node(
        result=bool(uni_name) and bool(doc.institution_name) and (uni_name.strip().lower() != (doc.institution_name or "").strip().lower()),
        id="university_different_from_doctoral",
        desc="Verify university is a different institution from the doctoral institution",
        parent=verify_group,
        critical=True
    )

    # Leaf: same_accreditor_verification
    same_accr_node = evaluator.add_leaf(
        id="same_accreditor_verification",
        desc="Verify accreditation by the same regional body",
        parent=verify_group,
        critical=True
    )
    accred_name = accr.accreditor_name or ""
    same_accr_sources = _combine_urls(comp.accreditor_urls, accr.reference_urls)
    await evaluator.verify(
        claim=f"{uni_name} is accredited by {accred_name}.",
        node=same_accr_node,
        sources=same_accr_sources,
        additional_instruction="Confirm institutional accreditation by the same regional accreditor identified earlier."
    )

    # Leaf: minimum_doctoral_programs
    min_doc_node = evaluator.add_leaf(
        id="minimum_doctoral_programs",
        desc="Verify at least 10 doctoral programs are offered",
        parent=verify_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} offers at least 10 distinct doctoral degree programs.",
        node=min_doc_node,
        sources=_combine_urls(comp.doctoral_programs_urls),
        additional_instruction="Count only doctoral programs (e.g., PhD, EdD, DNP, DBA). Summaries or program lists/curriculum catalogs that show 10 or more programs satisfy this."
    )


async def verify_online_offerings(
    evaluator: Evaluator,
    parent_node,
    data: AllExtraction
) -> None:
    """
    verify_online_offerings:
      - online_reference_urls (existence)
      - online_program_availability (yes/no)
    """
    node = evaluator.add_sequential(
        id="verify_online_offerings",
        desc="Determine whether the identified comparable university offers online graduate degree programs",
        parent=parent_node,
        critical=True
    )

    comp = data.comparable_university or ComparableUniversityInfo()
    online = data.online or OnlineProgramsInfo()

    online_urls = _combine_urls(online.reference_urls)

    evaluator.add_custom_node(
        result=bool(online_urls),
        id="online_reference_urls",
        desc="Provide valid URL references supporting online program information",
        parent=node,
        critical=True
    )

    avail_node = evaluator.add_leaf(
        id="online_program_availability",
        desc="Verify whether online graduate degree programs are offered (yes/no answer required)",
        parent=node,
        critical=True
    )

    uni_name = comp.name or "the university"
    yn = _normalize_yes_no(online.offers_online_graduate_degrees)
    if yn is True:
        claim = f"{uni_name} offers online graduate degree programs."
        add_ins = "Accept master's or doctoral degree programs that can be completed fully online or with mostly online delivery."
    elif yn is False:
        claim = f"{uni_name} does not offer any online graduate degree programs."
        add_ins = "Confirm that no master's or doctoral degree programs are offered online; hybrid or partially online programs should not count as 'online graduate degree programs'."
    else:
        # If unclear from extraction, still perform verification, but phrase in a neutral way
        claim = f"Determine whether {uni_name} offers online graduate degree programs."
        add_ins = "If any online graduate degree programs are offered, treat as 'yes'; otherwise 'no'."

    await evaluator.verify(
        claim=claim,
        node=avail_node,
        sources=online_urls,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Entry point for evaluating an answer to the UVA presidency + accreditation + comparable institution task.
    """

    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root strategy; we'll add a single critical sequential child under root
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AllExtraction,
        extraction_name="extracted_entities"
    )

    # Build the top-level critical sequential node (as per rubric)
    root_crit = evaluator.add_sequential(
        id="complete_educational_pathway",
        desc="Successfully identify and verify all required information about university leadership, doctoral institution, accreditation, and comparable institutions",
        parent=evaluator.root,
        critical=True
    )

    # Sequentially verify each major component (each child is critical)
    await verify_president(evaluator, root_crit, extracted)
    await verify_doctoral_institution(evaluator, root_crit, extracted)
    await verify_accreditor(evaluator, root_crit, extracted)
    await verify_comparable_university(evaluator, root_crit, extracted)
    await verify_online_offerings(evaluator, root_crit, extracted)

    # Add a compact custom info summary for convenience (optional)
    evaluator.add_custom_info(
        info={
            "president_name": (extracted.president.name if extracted.president else None),
            "doctoral_institution": (extracted.doctorate.institution_name if extracted.doctorate else None),
            "accreditor": (extracted.accreditor.accreditor_name if extracted.accreditor else None),
            "comparable_university": (extracted.comparable_university.name if extracted.comparable_university else None),
            "online_grad_degrees_claim": (extracted.online.offers_online_graduate_degrees if extracted.online else None),
        },
        info_type="extraction_summary"
    )

    return evaluator.get_summary()