import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "interstellar_institutions"
TASK_DESCRIPTION = """In the ongoing study of interstellar objects, three universities have made significant contributions to research in astronomy, artificial intelligence education, and generational learning patterns.

First, identify the university whose physics department researchers published water detection findings in the interstellar comet 3I/ATLAS on February 11, 2026. Provide: (1) the name of the lead researcher on this water detection study, (2) the specific department this researcher works in, (3) the NASA observatory instrument used for detecting water in 3I/ATLAS, and (4) reference URLs for each piece of information.

Second, identify the university whose Institute for Astronomy participated in 3I/ATLAS observations using a 2.2-meter telescope for spectroscopic analysis. Provide: (1) the name of a key researcher from the Institute for Astronomy involved in 3I/ATLAS research, (2) the specific organizational unit (institute/center) this researcher is affiliated with, (3) the exact size designation of the telescope used (in meters or inches), and (4) reference URLs for each piece of information.

Third, identify the university that launched a 3-course Explainable AI (XAI) specialization in 2024. Provide: (1) the exact number of courses in this specialization, (2) the year it was launched or first offered, (3) the name of the primary instructor or course leader, and (4) reference URLs for each piece of information.

For each of the three universities, you must provide the institution's name and reference URLs supporting all claims.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AuburnInfo(BaseModel):
    institution_name: Optional[str] = None
    institution_sources: List[str] = Field(default_factory=list)

    lead_researcher_name: Optional[str] = None
    researcher_department: Optional[str] = None
    researcher_title: Optional[str] = None
    researcher_sources: List[str] = Field(default_factory=list)

    publication_date: Optional[str] = None
    publication_sources: List[str] = Field(default_factory=list)

    instrument_name: Optional[str] = None
    observation_type: Optional[str] = None
    instrument_sources: List[str] = Field(default_factory=list)


class HawaiiInfo(BaseModel):
    institution_name: Optional[str] = None
    institution_sources: List[str] = Field(default_factory=list)

    researcher_name: Optional[str] = None
    researcher_affiliation: Optional[str] = None
    researcher_role: Optional[str] = None
    researcher_sources: List[str] = Field(default_factory=list)

    telescope_size: Optional[str] = None
    telescope_application: Optional[str] = None
    telescope_sources: List[str] = Field(default_factory=list)

    contribution_description: Optional[str] = None
    contribution_sources: List[str] = Field(default_factory=list)


class DukeInfo(BaseModel):
    institution_name: Optional[str] = None
    institution_sources: List[str] = Field(default_factory=list)

    course_count: Optional[str] = None
    launch_year: Optional[str] = None
    program_structure_sources: List[str] = Field(default_factory=list)

    instructor_name: Optional[str] = None
    instructor_sources: List[str] = Field(default_factory=list)

    delivery_format: Optional[str] = None
    content_coverage: Optional[str] = None
    delivery_sources: List[str] = Field(default_factory=list)


class InstitutionsExtraction(BaseModel):
    auburn: Optional[AuburnInfo] = None
    hawaii: Optional[HawaiiInfo] = None
    duke: Optional[DukeInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_institutions() -> str:
    return """
    Extract structured information for three institutions from the answer text. Return JSON with keys 'auburn', 'hawaii', and 'duke' corresponding to the following:

    auburn:
      - institution_name: The university name associated with publishing water detection findings in comet 3I/ATLAS (e.g., Auburn University).
      - institution_sources: All URLs provided in the answer that support the institution's involvement in the 3I/ATLAS water detection research.
      - lead_researcher_name: The lead researcher for the water detection study (e.g., Dennis Bodewits).
      - researcher_department: The department the researcher works in (e.g., Department of Physics).
      - researcher_title: The researcher's title/position if stated (e.g., associate professor, professor). If not mentioned, return null.
      - researcher_sources: URLs that support the identity, department, and role of the researcher.
      - publication_date: The exact publication date for the findings (e.g., February 11, 2026).
      - publication_sources: URLs that support the publication date.
      - instrument_name: The NASA observatory instrument used (e.g., Neil Gehrels Swift Observatory, Swift satellite).
      - observation_type: The type of observations used (e.g., ultraviolet imaging, UV observations). If not mentioned, return null.
      - instrument_sources: URLs that support the instrument and observation details.

    hawaii:
      - institution_name: The university whose Institute for Astronomy participated in 3I/ATLAS observations (e.g., University of Hawaii / University of Hawaiʻi).
      - institution_sources: URLs supporting the institution's participation in 3I/ATLAS observations.
      - researcher_name: Name of a key researcher involved (e.g., Karen Meech or Karen J. Meech).
      - researcher_affiliation: Organizational unit (e.g., Institute for Astronomy).
      - researcher_role: Role/position if stated (e.g., astronomer, faculty chair). If not mentioned, return null.
      - researcher_sources: URLs supporting the researcher identity and affiliation.
      - telescope_size: Exact size designation (e.g., "2.2-meter", "2.2 m", "88-inch").
      - telescope_application: Use-case (e.g., spectroscopic observations).
      - telescope_sources: URLs supporting telescope size and application.
      - contribution_description: Short description of the contribution (e.g., observations, characterization, initial spectroscopy). If not mentioned, return null.
      - contribution_sources: URLs supporting the contribution details.

    duke:
      - institution_name: The university offering the XAI specialization (e.g., Duke University).
      - institution_sources: URLs confirming the XAI specialization is offered by Duke.
      - course_count: The number of courses in the specialization (e.g., "3" or "three").
      - launch_year: The year launched or first offered (e.g., "2024").
      - program_structure_sources: URLs supporting course count and launch year (e.g., Coursera/Duke pages).
      - instructor_name: Primary instructor or course leader (e.g., Dr. Brinnae Bent). If not mentioned, return null.
      - instructor_sources: URLs supporting instructor identity.
      - delivery_format: Delivery method (e.g., online/Coursera). If not mentioned, return null.
      - content_coverage: A brief summary of topics covered (e.g., interpretable ML, XAI techniques). If not mentioned, return null.
      - delivery_sources: URLs supporting delivery format and content coverage.

    IMPORTANT:
    - Extract ONLY what appears in the answer. Do not infer or invent any values.
    - For each '*_sources' field, include all valid URLs explicitly present in the answer that support the corresponding information.
    - When a specific item is missing in the answer, set it to null (for strings) or [] (for URL lists).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_nonempty_str(s: Optional[str]) -> bool:
    return bool(s) and bool(str(s).strip())


def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


# --------------------------------------------------------------------------- #
# Verification functions: Auburn                                              #
# --------------------------------------------------------------------------- #
async def verify_institution_auburn(evaluator: Evaluator, parent_node, auburn: Optional[AuburnInfo]) -> None:
    inst_node = evaluator.add_parallel(
        id="institution_auburn",
        desc="Institution that published water detection findings in interstellar comet 3I/ATLAS in February 2026",
        parent=parent_node,
        critical=False,
    )

    # Basic info
    basic_node = evaluator.add_parallel(
        id="auburn_basic_info",
        desc="Basic institutional identification",
        parent=inst_node,
        critical=False,
    )
    evaluator.add_custom_node(
        result=_has_nonempty_str(auburn.institution_name) if auburn else False,
        id="auburn_basic_info_exists",
        desc="Auburn basic info exists (name provided)",
        parent=basic_node,
        critical=True,
    )

    leaf_name = evaluator.add_leaf(
        id="auburn_name",
        desc="Institution name is Auburn University",
        parent=basic_node,
        critical=True,
    )
    name_claim = f"The institution identified is '{auburn.institution_name}' which equals 'Auburn University'."
    await evaluator.verify(
        claim=name_claim,
        node=leaf_name,
        additional_instruction="Judge if the provided institution name matches 'Auburn University' allowing minor variants (e.g., casing, punctuation).",
    )

    leaf_name_ref = evaluator.add_leaf(
        id="auburn_name_reference",
        desc="URL reference confirming Auburn University published 3I/ATLAS water detection research",
        parent=basic_node,
        critical=True,
    )
    ref_claim = "Auburn University published water detection findings related to the interstellar comet 3I/ATLAS."
    await evaluator.verify(
        claim=ref_claim,
        node=leaf_name_ref,
        sources=(auburn.institution_sources if auburn else []),
        additional_instruction="Confirm the page(s) associate Auburn University with publishing water detection findings in 3I/ATLAS.",
    )

    # Lead researcher
    lead_node = evaluator.add_parallel(
        id="auburn_lead_researcher",
        desc="Lead researcher on water detection study",
        parent=inst_node,
        critical=False,
    )
    evaluator.add_custom_node(
        result=(auburn is not None and _has_nonempty_str(auburn.lead_researcher_name) and _has_sources(auburn.researcher_sources)),
        id="auburn_lead_researcher_exists",
        desc="Lead researcher info exists (name and sources provided)",
        parent=lead_node,
        critical=True,
    )

    ident_node = evaluator.add_parallel(
        id="researcher_identity",
        desc="Researcher identification details",
        parent=lead_node,
        critical=False,
    )

    leaf_r_name = evaluator.add_leaf(
        id="researcher_name",
        desc="Researcher name is Dennis Bodewits",
        parent=ident_node,
        critical=True,
    )
    rname_claim = f"The lead researcher is '{auburn.lead_researcher_name}', which equals 'Dennis Bodewits'."
    await evaluator.verify(
        claim=rname_claim,
        node=leaf_r_name,
        sources=(auburn.researcher_sources if auburn else []),
        additional_instruction="Allow minor variations (middle initials, casing). Check pages for mention of 'Dennis Bodewits'.",
    )

    leaf_r_dept = evaluator.add_leaf(
        id="researcher_department",
        desc="Researcher works in Department of Physics",
        parent=ident_node,
        critical=True,
    )
    rdept_claim = f"Dennis Bodewits works in the Department of Physics at Auburn University."
    await evaluator.verify(
        claim=rdept_claim,
        node=leaf_r_dept,
        sources=(auburn.researcher_sources if auburn else []),
        additional_instruction="Verify the page(s) explicitly associate Dennis Bodewits with the Department of Physics at Auburn.",
    )

    leaf_r_title = evaluator.add_leaf(
        id="researcher_title",
        desc="Researcher holds position of associate professor or higher",
        parent=ident_node,
        critical=False,
    )
    rtitle_claim = "Dennis Bodewits holds a position of associate professor or higher (e.g., associate professor, professor, endowed chair)."
    await evaluator.verify(
        claim=rtitle_claim,
        node=leaf_r_title,
        sources=(auburn.researcher_sources if auburn else []),
        additional_instruction="Accept 'associate professor', 'professor', or equivalent senior academic titles.",
    )

    leaf_r_ref = evaluator.add_leaf(
        id="researcher_reference",
        desc="URL reference for researcher information",
        parent=lead_node,
        critical=True,
    )
    rref_claim = "The provided pages explicitly mention Dennis Bodewits and his role/affiliation relevant to the 3I/ATLAS water detection study."
    await evaluator.verify(
        claim=rref_claim,
        node=leaf_r_ref,
        sources=(auburn.researcher_sources if auburn else []),
        additional_instruction="Check the pages for clear mention of the researcher's identity and affiliation (Auburn Physics).",
    )

    # Publication details
    pub_node = evaluator.add_parallel(
        id="auburn_publication_details",
        desc="Publication details for water detection findings",
        parent=inst_node,
        critical=False,
    )
    evaluator.add_custom_node(
        result=(auburn is not None and _has_nonempty_str(auburn.publication_date) and _has_sources(auburn.publication_sources)),
        id="auburn_publication_exists",
        desc="Publication date info exists (date and sources provided)",
        parent=pub_node,
        critical=True,
    )

    timing_node = evaluator.add_parallel(
        id="publication_timing",
        desc="Publication date information",
        parent=pub_node,
        critical=False,
    )

    leaf_exact_date = evaluator.add_leaf(
        id="exact_date",
        desc="Publication date is February 11, 2026",
        parent=timing_node,
        critical=True,
    )
    date_claim = "The publication date for the water detection findings is February 11, 2026."
    await evaluator.verify(
        claim=date_claim,
        node=leaf_exact_date,
        sources=(auburn.publication_sources if auburn else []),
        additional_instruction="Accept minor formatting variants like 'Feb. 11, 2026' or 'February 11, 2026'.",
    )

    leaf_date_source = evaluator.add_leaf(
        id="date_source",
        desc="Source is ScienceDaily or equivalent authoritative source",
        parent=timing_node,
        critical=False,
    )
    source_claim = "At least one of the provided sources is ScienceDaily or an equivalent authoritative outlet (e.g., NASA.gov, Phys.org, a university press release)."
    await evaluator.verify(
        claim=source_claim,
        node=leaf_date_source,
        sources=(auburn.publication_sources if auburn else []),
        additional_instruction="Check the domain/branding on the page to confirm an authoritative source.",
    )

    leaf_pub_ref = evaluator.add_leaf(
        id="publication_reference",
        desc="URL reference for publication date",
        parent=pub_node,
        critical=True,
    )
    pub_ref_claim = "The provided URLs explicitly support the claimed publication date for the water detection findings."
    await evaluator.verify(
        claim=pub_ref_claim,
        node=leaf_pub_ref,
        sources=(auburn.publication_sources if auburn else []),
        additional_instruction="Confirm the date is clearly shown or stated on the page(s).",
    )

    # Observation method / instrumentation
    obs_node = evaluator.add_parallel(
        id="auburn_observation_method",
        desc="Observation methodology and instrumentation",
        parent=inst_node,
        critical=False,
    )
    evaluator.add_custom_node(
        result=(auburn is not None and _has_nonempty_str(auburn.instrument_name) and _has_sources(auburn.instrument_sources)),
        id="auburn_instrument_exists",
        desc="Instrument info exists (name and sources provided)",
        parent=obs_node,
        critical=True,
    )

    inst_details_node = evaluator.add_parallel(
        id="instrument_details",
        desc="Instrument used for observations",
        parent=obs_node,
        critical=False,
    )

    leaf_inst_name = evaluator.add_leaf(
        id="instrument_name",
        desc="Instrument is NASA's Neil Gehrels Swift Observatory or Swift satellite",
        parent=inst_details_node,
        critical=True,
    )
    inst_claim = "The instrument used to detect water in 3I/ATLAS was NASA's Neil Gehrels Swift Observatory (Swift satellite)."
    await evaluator.verify(
        claim=inst_claim,
        node=leaf_inst_name,
        sources=(auburn.instrument_sources if auburn else []),
        additional_instruction="Allow variants like 'Swift Observatory', 'Swift satellite', or 'Neil Gehrels Swift (Swift)'.",
    )

    leaf_obs_type = evaluator.add_leaf(
        id="observation_type",
        desc="Used ultraviolet imaging or UV observations",
        parent=inst_details_node,
        critical=False,
    )
    obs_type_claim = "Ultraviolet imaging or UV observations were used in detecting water in 3I/ATLAS."
    await evaluator.verify(
        claim=obs_type_claim,
        node=leaf_obs_type,
        sources=(auburn.instrument_sources if auburn else []),
        additional_instruction="Confirm that UV imaging/observations are explicitly mentioned.",
    )

    leaf_inst_ref = evaluator.add_leaf(
        id="instrument_reference",
        desc="URL reference for instrument information",
        parent=obs_node,
        critical=True,
    )
    inst_ref_claim = "The provided URLs clearly support the instrument used (Neil Gehrels Swift Observatory) and its role in the observations."
    await evaluator.verify(
        claim=inst_ref_claim,
        node=leaf_inst_ref,
        sources=(auburn.instrument_sources if auburn else []),
        additional_instruction="Check for explicit mention of the instrument and its usage in the study.",
    )


# --------------------------------------------------------------------------- #
# Verification functions: Hawaii                                              #
# --------------------------------------------------------------------------- #
async def verify_institution_hawaii(evaluator: Evaluator, parent_node, hawaii: Optional[HawaiiInfo]) -> None:
    inst_node = evaluator.add_parallel(
        id="institution_hawaii",
        desc="Institution whose Institute for Astronomy participated in 3I/ATLAS comet observations using a 2.2-meter telescope",
        parent=parent_node,
        critical=False,
    )

    # Basic info
    basic_node = evaluator.add_parallel(
        id="hawaii_basic_info",
        desc="Basic institutional identification",
        parent=inst_node,
        critical=False,
    )
    evaluator.add_custom_node(
        result=_has_nonempty_str(hawaii.institution_name) if hawaii else False,
        id="hawaii_basic_info_exists",
        desc="Hawaii basic info exists (name provided)",
        parent=basic_node,
        critical=True,
    )

    leaf_name = evaluator.add_leaf(
        id="hawaii_name",
        desc="Institution name is University of Hawaii",
        parent=basic_node,
        critical=True,
    )
    name_claim = f"The institution identified is '{hawaii.institution_name}', which equals 'University of Hawaii' (including 'University of Hawaiʻi')."
    await evaluator.verify(
        claim=name_claim,
        node=leaf_name,
        additional_instruction="Allow 'University of Hawaiʻi' spelling with diacritics, case variants, and official naming variations.",
    )

    leaf_name_ref = evaluator.add_leaf(
        id="hawaii_name_reference",
        desc="URL reference confirming University of Hawaii participated in 3I/ATLAS research",
        parent=basic_node,
        critical=True,
    )
    ref_claim = "The University of Hawaii (Institute for Astronomy) participated in observations related to the interstellar comet 3I/ATLAS."
    await evaluator.verify(
        claim=ref_claim,
        node=leaf_name_ref,
        sources=(hawaii.institution_sources if hawaii else []),
        additional_instruction="Confirm the page(s) document UH/IfA involvement in 3I/ATLAS observations.",
    )

    # Researcher
    researcher_node = evaluator.add_parallel(
        id="hawaii_researcher",
        desc="Key researcher involved in 3I/ATLAS observations",
        parent=inst_node,
        critical=False,
    )
    evaluator.add_custom_node(
        result=(hawaii is not None and _has_nonempty_str(hawaii.researcher_name) and _has_sources(hawaii.researcher_sources)),
        id="hawaii_researcher_exists",
        desc="Hawaii researcher info exists (name and sources provided)",
        parent=researcher_node,
        critical=True,
    )

    ident_node = evaluator.add_parallel(
        id="hawaii_researcher_identity",
        desc="Researcher identification details",
        parent=researcher_node,
        critical=False,
    )

    leaf_r_name = evaluator.add_leaf(
        id="hawaii_researcher_name",
        desc="Researcher name is Karen Meech or Karen J. Meech",
        parent=ident_node,
        critical=True,
    )
    rname_claim = f"The key researcher is '{hawaii.researcher_name}', which equals 'Karen Meech' (allow 'Karen J. Meech')."
    await evaluator.verify(
        claim=rname_claim,
        node=leaf_r_name,
        sources=(hawaii.researcher_sources if hawaii else []),
        additional_instruction="Allow middle initial variants ('Karen Meech' vs 'Karen J. Meech'), casing differences.",
    )

    leaf_affil = evaluator.add_leaf(
        id="hawaii_researcher_affiliation",
        desc="Researcher affiliated with Institute for Astronomy",
        parent=ident_node,
        critical=True,
    )
    affil_claim = "Karen Meech is affiliated with the University of Hawaii Institute for Astronomy."
    await evaluator.verify(
        claim=affil_claim,
        node=leaf_affil,
        sources=(hawaii.researcher_sources if hawaii else []),
        additional_instruction="Page(s) should explicitly connect Karen Meech to UH/IfA.",
    )

    leaf_role = evaluator.add_leaf(
        id="hawaii_researcher_role",
        desc="Researcher holds faculty chair position or astronomer position",
        parent=ident_node,
        critical=False,
    )
    role_claim = "Karen Meech holds a faculty chair or astronomer position (or equivalent senior role)."
    await evaluator.verify(
        claim=role_claim,
        node=leaf_role,
        sources=(hawaii.researcher_sources if hawaii else []),
        additional_instruction="Accept chair titles, astronomer roles, or similar senior roles.",
    )

    leaf_r_ref = evaluator.add_leaf(
        id="hawaii_researcher_reference",
        desc="URL reference for researcher information",
        parent=researcher_node,
        critical=True,
    )
    rref_claim = "The provided URLs explicitly mention Karen Meech and her UH/IfA affiliation related to 3I/ATLAS."
    await evaluator.verify(
        claim=rref_claim,
        node=leaf_r_ref,
        sources=(hawaii.researcher_sources if hawaii else []),
        additional_instruction="Confirm explicit mentions of the researcher and affiliation.",
    )

    # Telescope info
    tele_node = evaluator.add_parallel(
        id="hawaii_telescope_info",
        desc="Telescope used for observations",
        parent=inst_node,
        critical=False,
    )
    evaluator.add_custom_node(
        result=(hawaii is not None and _has_nonempty_str(hawaii.telescope_size) and _has_sources(hawaii.telescope_sources)),
        id="hawaii_telescope_exists",
        desc="Hawaii telescope info exists (size and sources provided)",
        parent=tele_node,
        critical=True,
    )

    specs_node = evaluator.add_sequential(
        id="telescope_specifications",
        desc="Telescope specification details",
        parent=tele_node,
        critical=False,
    )

    leaf_size = evaluator.add_leaf(
        id="telescope_size",
        desc="Telescope size is 2.2-meter or 2.2-m or 88-inch",
        parent=specs_node,
        critical=True,
    )
    size_claim = "The telescope used is the 2.2-meter (2.2 m) telescope, also known as the 88-inch telescope."
    await evaluator.verify(
        claim=size_claim,
        node=leaf_size,
        sources=(hawaii.telescope_sources if hawaii else []),
        additional_instruction="Accept any of the forms: '2.2-meter', '2.2 m', or '88-inch'.",
    )

    leaf_appl = evaluator.add_leaf(
        id="telescope_application",
        desc="Used for spectroscopic observations or spectroscopy",
        parent=specs_node,
        critical=True,
    )
    appl_claim = "The telescope was used for spectroscopic observations (spectroscopy)."
    await evaluator.verify(
        claim=appl_claim,
        node=leaf_appl,
        sources=(hawaii.telescope_sources if hawaii else []),
        additional_instruction="The page(s) should explicitly mention spectroscopic use.",
    )

    leaf_tel_ref = evaluator.add_leaf(
        id="telescope_reference",
        desc="URL reference for telescope information",
        parent=tele_node,
        critical=True,
    )
    tel_ref_claim = "The provided URLs support the telescope size and its spectroscopic use in 3I/ATLAS observations."
    await evaluator.verify(
        claim=tel_ref_claim,
        node=leaf_tel_ref,
        sources=(hawaii.telescope_sources if hawaii else []),
        additional_instruction="Confirm both size and spectroscopy usage.",
    )

    # Contribution (non-critical)
    contrib_node = evaluator.add_parallel(
        id="hawaii_research_contribution",
        desc="Nature of research contribution",
        parent=inst_node,
        critical=False,
    )
    evaluator.add_custom_node(
        result=(hawaii is not None and _has_nonempty_str(hawaii.contribution_description) and _has_sources(hawaii.contribution_sources)),
        id="hawaii_contribution_exists",
        desc="Contribution info exists (description and sources provided)",
        parent=contrib_node,
        critical=False,
    )

    contrib_desc_node = evaluator.add_parallel(
        id="contribution_description",
        desc="Contribution details",
        parent=contrib_node,
        critical=False,
    )
    leaf_contrib_type = evaluator.add_leaf(
        id="contribution_type",
        desc="Participated in observations, characterization, or initial spectroscopy",
        parent=contrib_desc_node,
        critical=False,
    )
    contrib_claim = "The University of Hawaii/IfA participated in observations, characterization, or initial spectroscopy for 3I/ATLAS."
    await evaluator.verify(
        claim=contrib_claim,
        node=leaf_contrib_type,
        sources=(hawaii.contribution_sources if hawaii else []),
        additional_instruction="Confirm one or more of: observations, characterization, initial spectroscopy.",
    )

    leaf_contrib_ref = evaluator.add_leaf(
        id="contribution_reference",
        desc="URL reference for contribution information",
        parent=contrib_node,
        critical=False,
    )
    contrib_ref_claim = "The provided URLs support the described contribution by UH/IfA in 3I/ATLAS research."
    await evaluator.verify(
        claim=contrib_ref_claim,
        node=leaf_contrib_ref,
        sources=(hawaii.contribution_sources if hawaii else []),
        additional_instruction="Confirm the contribution details are explicitly stated.",
    )


# --------------------------------------------------------------------------- #
# Verification functions: Duke                                                #
# --------------------------------------------------------------------------- #
async def verify_institution_duke(evaluator: Evaluator, parent_node, duke: Optional[DukeInfo]) -> None:
    inst_node = evaluator.add_parallel(
        id="institution_duke",
        desc="Institution that offers a 3-course XAI (Explainable AI) specialization launched in 2024",
        parent=parent_node,
        critical=False,
    )

    # Basic info
    basic_node = evaluator.add_parallel(
        id="duke_basic_info",
        desc="Basic institutional identification",
        parent=inst_node,
        critical=True,  # Make critical to enforce correctness of institution identification
    )

    evaluator.add_custom_node(
        result=_has_nonempty_str(duke.institution_name) if duke else False,
        id="duke_basic_info_exists",
        desc="Duke basic info exists (name provided)",
        parent=basic_node,
        critical=True,
    )

    leaf_name = evaluator.add_leaf(
        id="duke_name",
        desc="Institution name is Duke University",
        parent=basic_node,
        critical=True,
    )
    name_claim = f"The institution identified is '{duke.institution_name}', which equals 'Duke University'."
    await evaluator.verify(
        claim=name_claim,
        node=leaf_name,
        additional_instruction="Judge if the provided institution name matches 'Duke University' allowing minor variants.",
    )

    leaf_name_ref = evaluator.add_leaf(
        id="duke_name_reference",
        desc="URL reference confirming Duke University offers XAI specialization",
        parent=basic_node,
        critical=True,
    )
    ref_claim = "Duke University offers an Explainable AI (XAI) specialization."
    await evaluator.verify(
        claim=ref_claim,
        node=leaf_name_ref,
        sources=(duke.institution_sources if duke else []),
        additional_instruction="Confirm the page(s) explicitly tie the XAI specialization to Duke University.",
    )

    # Program structure
    prog_node = evaluator.add_parallel(
        id="duke_xai_program_structure",
        desc="XAI program structural details",
        parent=inst_node,
        critical=True,  # Structure must be correct
    )

    evaluator.add_custom_node(
        result=(duke is not None and _has_nonempty_str(duke.course_count) and _has_nonempty_str(duke.launch_year) and _has_sources(duke.program_structure_sources)),
        id="duke_program_structure_exists",
        desc="Program structure info exists (course count, year, and sources provided)",
        parent=prog_node,
        critical=True,
    )

    comp_node = evaluator.add_parallel(
        id="program_composition",
        desc="Program composition details",
        parent=prog_node,
        critical=True,
    )

    leaf_count = evaluator.add_leaf(
        id="course_count",
        desc="Specialization consists of 3 courses",
        parent=comp_node,
        critical=True,
    )
    count_claim = "This XAI specialization consists of exactly 3 courses."
    await evaluator.verify(
        claim=count_claim,
        node=leaf_count,
        sources=(duke.program_structure_sources if duke else []),
        additional_instruction="Accept numeric and word forms ('3' or 'three') if clearly indicated on the page(s).",
    )

    leaf_launch = evaluator.add_leaf(
        id="launch_timing",
        desc="Program launched or offered in 2024",
        parent=comp_node,
        critical=True,
    )
    launch_claim = "This XAI specialization was launched or first offered in 2024."
    await evaluator.verify(
        claim=launch_claim,
        node=leaf_launch,
        sources=(duke.program_structure_sources if duke else []),
        additional_instruction="Verify the year 2024 is explicitly stated for launch or initial offering.",
    )

    leaf_prog_ref = evaluator.add_leaf(
        id="program_structure_reference",
        desc="URL reference for program structure information",
        parent=prog_node,
        critical=True,
    )
    prog_ref_claim = "The provided URLs support both the 3-course composition and the 2024 launch timing."
    await evaluator.verify(
        claim=prog_ref_claim,
        node=leaf_prog_ref,
        sources=(duke.program_structure_sources if duke else []),
        additional_instruction="Confirm both the course count and the launch year on the page(s).",
    )

    # Instructor (non-critical)
    instr_node = evaluator.add_parallel(
        id="duke_xai_instructor",
        desc="Program instructor information",
        parent=inst_node,
        critical=False,
    )
    evaluator.add_custom_node(
        result=(duke is not None and _has_nonempty_str(duke.instructor_name) and _has_sources(duke.instructor_sources)),
        id="duke_instructor_exists",
        desc="Instructor info exists (name and sources provided)",
        parent=instr_node,
        critical=False,
    )

    ident_node = evaluator.add_parallel(
        id="instructor_identity",
        desc="Instructor identification details",
        parent=instr_node,
        critical=False,
    )

    leaf_instr_name = evaluator.add_leaf(
        id="instructor_name",
        desc="Primary instructor is Dr. Brinnae Bent",
        parent=ident_node,
        critical=False,
    )
    instr_claim = f"The primary instructor for the XAI specialization is '{duke.instructor_name}', which equals 'Dr. Brinnae Bent'."
    await evaluator.verify(
        claim=instr_claim,
        node=leaf_instr_name,
        sources=(duke.instructor_sources if duke else []),
        additional_instruction="Allow minor variants such as 'Brinnae Bent, PhD', casing differences.",
    )

    leaf_instr_ref = evaluator.add_leaf(
        id="instructor_reference",
        desc="URL reference for instructor information",
        parent=instr_node,
        critical=False,
    )
    instr_ref_claim = "The provided URLs support the instructor identity (Dr. Brinnae Bent) for the XAI specialization."
    await evaluator.verify(
        claim=instr_ref_claim,
        node=leaf_instr_ref,
        sources=(duke.instructor_sources if duke else []),
        additional_instruction="Confirm explicit mention of the instructor's name and role.",
    )

    # Delivery (non-critical)
    delivery_node = evaluator.add_parallel(
        id="duke_program_delivery",
        desc="Program delivery method and accessibility",
        parent=inst_node,
        critical=False,
    )
    evaluator.add_custom_node(
        result=_has_sources(duke.delivery_sources) if duke else False,
        id="duke_delivery_exists",
        desc="Delivery info sources exist",
        parent=delivery_node,
        critical=False,
    )

    format_node = evaluator.add_parallel(
        id="delivery_format",
        desc="Delivery format details",
        parent=delivery_node,
        critical=False,
    )

    leaf_format = evaluator.add_leaf(
        id="format_type",
        desc="Available online or through Coursera platform",
        parent=format_node,
        critical=False,
    )
    format_claim = "This XAI specialization is available online and/or through the Coursera platform."
    await evaluator.verify(
        claim=format_claim,
        node=leaf_format,
        sources=(duke.delivery_sources if duke else []),
        additional_instruction="Confirm availability online and/or via Coursera on the page(s).",
    )

    leaf_coverage = evaluator.add_leaf(
        id="content_coverage",
        desc="Covers XAI concepts, interpretable machine learning, and advanced explainability techniques",
        parent=format_node,
        critical=False,
    )
    coverage_claim = "The specialization covers XAI concepts, interpretable machine learning, and advanced explainability techniques."
    await evaluator.verify(
        claim=coverage_claim,
        node=leaf_coverage,
        sources=(duke.delivery_sources if duke else []),
        additional_instruction="Confirm coverage of XAI fundamentals, interpretability, and advanced techniques.",
    )

    leaf_delivery_ref = evaluator.add_leaf(
        id="delivery_reference",
        desc="URL reference for delivery information",
        parent=delivery_node,
        critical=False,
    )
    delivery_ref_claim = "The provided URLs support the program delivery format and topical coverage."
    await evaluator.verify(
        claim=delivery_ref_claim,
        node=leaf_delivery_ref,
        sources=(duke.delivery_sources if duke else []),
        additional_instruction="Confirm both delivery platform/method and topic coverage.",
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
    Evaluate an answer for the interstellar institutions task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root: parallel aggregation across institutions
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

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_institutions(),
        template_class=InstitutionsExtraction,
        extraction_name="institutions_extraction",
    )

    # Ground truth info (for summary)
    evaluator.add_ground_truth({
        "expected_universities": [
            "Auburn University",
            "University of Hawaii (Institute for Astronomy)",
            "Duke University (XAI specialization)"
        ],
        "expected_details": {
            "auburn": {
                "lead_researcher": "Dennis Bodewits",
                "department": "Department of Physics",
                "instrument": "Neil Gehrels Swift Observatory (Swift satellite)",
                "publication_date": "February 11, 2026"
            },
            "hawaii": {
                "key_researcher": "Karen Meech (Karen J. Meech)",
                "affiliation": "Institute for Astronomy",
                "telescope_size": "2.2-meter (88-inch)",
                "application": "Spectroscopy"
            },
            "duke": {
                "course_count": "3",
                "launch_year": "2024",
                "primary_instructor": "Dr. Brinnae Bent"
            }
        }
    })

    # Build institution subtrees
    await verify_institution_auburn(evaluator, root, extracted.auburn or AuburnInfo())
    await verify_institution_hawaii(evaluator, root, extracted.hawaii or HawaiiInfo())
    await verify_institution_duke(evaluator, root, extracted.duke or DukeInfo())

    # Return summary
    return evaluator.get_summary()