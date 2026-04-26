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
TASK_ID = "us_zoo_multi_cert_bcpsa_ssp_2026"
TASK_DESCRIPTION = (
    "Identify a zoo or aquarium facility in the United States that meets all of the following requirements as of February 2026:\n"
    "1. The facility must hold current accreditation from the Association of Zoos and Aquariums (AZA).\n"
    "2. The facility must also hold American Humane Certification for zoos and aquariums.\n"
    "3. The facility must be authorized as a qualified exempt entity under the Big Cat Public Safety Act to legally possess big cats.\n"
    "4. The facility must actively participate in at least one Species Survival Plan (SSP) program managed by AZA.\n\n"
    "Provide the name of one such facility along with supporting evidence including relevant URLs that verify each requirement."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SSPProgram(BaseModel):
    program_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FacilityExtraction(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)

    aza_urls: List[str] = Field(default_factory=list)

    american_humane_urls: List[str] = Field(default_factory=list)

    bcpsa_urls: List[str] = Field(default_factory=list)

    ssp_programs: List[SSPProgram] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facility() -> str:
    return """
    Extract the single U.S. facility (zoo or aquarium) that the answer presents as satisfying ALL of the following, as of February 2026:
    - AZA accreditation (current)
    - American Humane Certification (for zoos & aquariums)
    - Authorized as a qualified exempt entity under the Big Cat Public Safety Act (BCPSA) to possess big cats
    - Participation in at least one AZA Species Survival Plan (SSP) program

    If multiple facilities are discussed, choose the first one that is claimed to satisfy all requirements. If none are explicitly stated to satisfy all, choose the first facility mentioned and still extract any evidence URLs provided.

    Return the following fields:
    - name: facility name as written
    - state: U.S. state (if stated; else null)
    - location_urls: ALL URLs in the answer that support the facility’s U.S. location (address/contact/about pages, Wikipedia, government, etc.)
    - aza_urls: ALL URLs that specifically support AZA accreditation status (AZA member directory, AZA pages, or the facility page explicitly showing 'AZA accredited')
    - american_humane_urls: ALL URLs that support American Humane certification for zoos & aquariums (American Humane website listings, certification pages, or official facility pages showing the certification)
    - bcpsa_urls: ALL URLs that support authorization as a 'qualified exempt entity' under the Big Cat Public Safety Act (e.g., U.S. Fish & Wildlife Service official list or other authoritative government pages)
    - ssp_programs: a list of objects, each with:
        - program_name: the SSP program name (e.g., 'Amur Tiger SSP'); if multiple are mentioned, include each; if not specified, set to null
        - sources: ALL URLs that support the participation in this SSP program (facility conservation pages, AZA program pages listing participants, press releases, etc.)

    IMPORTANT:
    - Only extract URLs explicitly present in the answer. Do not invent.
    - Include full URLs. If missing protocol, prepend http://
    - If a category has no URLs mentioned, return an empty list for that category.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if isinstance(u, str):
                u = u.strip()
                if not u:
                    continue
                # basic normalization: ensure protocol present as extractor promised
                if u not in seen:
                    seen.add(u)
                    deduped.append(u)
    return deduped


def _flatten_program_sources(programs: List[SSPProgram]) -> List[str]:
    all_urls: List[str] = []
    for p in programs or []:
        all_urls.extend(p.sources or [])
    return _unique_urls(all_urls)


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: FacilityExtraction) -> None:
    # Create the rubric tree per the provided JSON

    # Facility_Identification (critical, parallel)
    facility_node = evaluator.add_parallel(
        id="Facility_Identification",
        desc="Identifies a U.S. zoo or aquarium facility that meets all specified accreditation and conservation program requirements",
        parent=evaluator.root,
        critical=True,
    )

    name = extracted.name or ""
    state = extracted.state or ""
    # Aggregate sources for various checks
    loc_sources = _unique_urls(extracted.location_urls, extracted.aza_urls, extracted.bcpsa_urls, extracted.american_humane_urls)

    # 1) US_Location (leaf, critical)
    us_loc_leaf = evaluator.add_leaf(
        id="US_Location",
        desc="The facility is located in a U.S. state",
        parent=facility_node,
        critical=True,
    )
    us_loc_claim = (
        f"The facility named '{name}' is physically located in the United States"
        + (f", specifically in the state of {state}." if state else ".")
    )
    await evaluator.verify(
        claim=us_loc_claim,
        node=us_loc_leaf,
        sources=loc_sources,
        additional_instruction=(
            "Use the provided webpages to confirm the facility is in the United States (one of the 50 states or the District of Columbia). "
            "Accept official pages, address/contact pages, or reputable directories that show a U.S. address or clearly indicate a U.S. location. "
            "Minor facility name variations are acceptable."
        ),
    )

    # 2) AZA_Accreditation (critical, sequential)
    aza_node = evaluator.add_sequential(
        id="AZA_Accreditation",
        desc="Verifies the facility holds current AZA (Association of Zoos and Aquariums) accreditation as of February 2026",
        parent=facility_node,
        critical=True,
    )
    # 2.1) AZA_Status_Confirmed (leaf, critical)
    aza_status_leaf = evaluator.add_leaf(
        id="AZA_Status_Confirmed",
        desc="The facility is listed as an AZA-accredited member institution",
        parent=aza_node,
        critical=True,
    )
    aza_status_claim = f"The facility '{name}' is an AZA-accredited member institution."
    await evaluator.verify(
        claim=aza_status_claim,
        node=aza_status_leaf,
        sources=extracted.aza_urls,
        additional_instruction=(
            "Look for AZA's official member directory or AZA pages explicitly indicating 'AZA Accredited', or an official facility page displaying 'AZA Accredited'. "
            "The claim should be explicitly supported by the source content."
        ),
    )
    # 2.2) AZA_Currency (leaf, critical)
    aza_currency_leaf = evaluator.add_leaf(
        id="AZA_Currency",
        desc="The accreditation status is current and valid as of February 2026",
        parent=aza_node,
        critical=True,
    )
    aza_currency_claim = f"As of February 2026, the facility '{name}' holds current AZA accreditation (not expired or suspended)."
    await evaluator.verify(
        claim=aza_currency_claim,
        node=aza_currency_leaf,
        sources=extracted.aza_urls,
        additional_instruction=(
            "Confirm that the accreditation is current as of February 2026. "
            "Evidence can include AZA's current member listings, pages indicating accreditation validity through 2026, or other explicit current-status indicators. "
            "If a date or 'current list' is present, rely on that. If the page shows outdated accreditation or no indication of current status, do not support the claim."
        ),
    )

    # 3) American_Humane_Certification (critical, sequential)
    ah_node = evaluator.add_sequential(
        id="American_Humane_Certification",
        desc="Verifies the facility holds American Humane Certification for zoos and aquariums as of February 2026",
        parent=facility_node,
        critical=True,
    )
    # 3.1) AH_Certification_Confirmed (leaf, critical)
    ah_cert_leaf = evaluator.add_leaf(
        id="AH_Certification_Confirmed",
        desc="The facility is listed as American Humane Certified",
        parent=ah_node,
        critical=True,
    )
    ah_cert_claim = (
        f"The facility '{name}' is certified by American Humane for zoos and aquariums "
        "(commonly branded within American Humane's Humane Conservation program)."
    )
    await evaluator.verify(
        claim=ah_cert_claim,
        node=ah_cert_leaf,
        sources=extracted.american_humane_urls,
        additional_instruction=(
            "Verify via American Humane's official listings or authoritative pages that the facility is certified for zoos & aquariums "
            "(Humane Conservation/American Humane Certified). A facility page displaying the official certification is also acceptable if clear."
        ),
    )
    # 3.2) AH_Currency (leaf, critical)
    ah_currency_leaf = evaluator.add_leaf(
        id="AH_Currency",
        desc="The certification status is current and valid as of February 2026",
        parent=ah_node,
        critical=True,
    )
    ah_currency_claim = f"As of February 2026, the facility '{name}' holds current American Humane certification for zoos & aquariums."
    await evaluator.verify(
        claim=ah_currency_claim,
        node=ah_currency_leaf,
        sources=extracted.american_humane_urls,
        additional_instruction=(
            "Confirm that the certification is current as of February 2026. "
            "Look for current listings, validity-through dates, or other explicit signals that the certification is active as of that time."
        ),
    )

    # 4) Conservation_Programs (critical, parallel)
    cons_node = evaluator.add_parallel(
        id="Conservation_Programs",
        desc="Verifies the facility's compliance with big cat regulations and participation in conservation programs",
        parent=facility_node,
        critical=True,
    )
    # 4.1) Big_Cat_Authorization (leaf, critical)
    bcpsa_leaf = evaluator.add_leaf(
        id="Big_Cat_Authorization",
        desc="The facility is authorized as a qualified exempt entity under the Big Cat Public Safety Act to possess big cats",
        parent=cons_node,
        critical=True,
    )
    bcpsa_claim = (
        f"The facility '{name}' is listed as a qualified exempt entity under the Big Cat Public Safety Act (BCPSA) and is authorized to possess big cats."
    )
    await evaluator.verify(
        claim=bcpsa_claim,
        node=bcpsa_leaf,
        sources=extracted.bcpsa_urls,
        additional_instruction=(
            "Prefer an official U.S. Fish & Wildlife Service (FWS) or DOI page listing 'qualified exempt entities' under the BCPSA. "
            "The facility's legal name should appear on that list (minor name variations acceptable). "
            "If only a non-official source is provided without clear official backing, the claim should not be supported."
        ),
    )

    # 4.2) SSP_Participation (leaf, critical)
    ssp_leaf = evaluator.add_leaf(
        id="SSP_Participation",
        desc="The facility participates in at least one AZA Species Survival Plan (SSP) program",
        parent=cons_node,
        critical=True,
    )
    # Build claim text with program names if present
    program_names = [p.program_name for p in extracted.ssp_programs if p.program_name]
    if program_names:
        program_list_text = "; ".join(program_names)
        ssp_claim = (
            f"The facility '{name}' participates in at least one AZA Species Survival Plan (SSP) program, such as: {program_list_text}."
        )
    else:
        ssp_claim = f"The facility '{name}' participates in at least one AZA Species Survival Plan (SSP) program."

    ssp_sources = _flatten_program_sources(extracted.ssp_programs)
    # Allow additional corroboration from AZA URLs if helpful
    ssp_sources_all = _unique_urls(ssp_sources, extracted.aza_urls)

    await evaluator.verify(
        claim=ssp_claim,
        node=ssp_leaf,
        sources=ssp_sources_all,
        additional_instruction=(
            "Confirm that the facility participates in at least one AZA-managed SSP program. "
            "Accept AZA program pages that list participating institutions, or the facility's official conservation pages that clearly state participation in a named SSP."
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
        prompt=prompt_extract_facility(),
        template_class=FacilityExtraction,
        extraction_name="facility_extraction",
    )

    # Build tree and verify
    await build_and_verify_tree(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()