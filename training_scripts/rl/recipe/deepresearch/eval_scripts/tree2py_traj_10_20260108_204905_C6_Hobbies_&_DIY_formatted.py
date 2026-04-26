import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "woodworking_program_and_project_specs"
TASK_DESCRIPTION = """
You are planning to enroll in a woodworking certificate program and complete a capstone dining table project. Identify a suitable program and provide complete project specifications:

Program Requirements:
- The program must be located in either the Rocky Mountain region (Colorado) or the Midwest (Michigan)
- Must offer a certificate specifically in fine woodworking or furniture making
- Must be offered by an accredited institution or established woodworking school
- Provide the program's duration and tuition/cost information

Project Specifications for Indoor Dining Table with Drawers:
- Select a North American hardwood species appropriate for furniture construction and provide its Janka hardness rating in lbf
- Specify the joinery method for the table frame and leg connections, explaining why it provides adequate structural strength for a dining table
- Specify the joinery method for drawer box assembly, explaining why it is suitable for drawer construction
- Select a wood finish appropriate for indoor dining furniture and confirm it complies with EPA VOC content standards (≤450 g/L for wood coatings)
- Document the ventilation requirements for finish application and necessary personal protective equipment

For all specifications, provide URL references to support your selections.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramInfo(BaseModel):
    program_name: Optional[str] = None
    institution_name: Optional[str] = None
    location_state: Optional[str] = None  # e.g., "Colorado" or "Michigan"
    city: Optional[str] = None
    certificate_type: Optional[str] = None  # e.g., "Certificate in Fine Woodworking"
    institution_status: Optional[str] = None  # e.g., "accredited institution", "established woodworking school"
    duration: Optional[str] = None  # e.g., "1 year", "9 months", "2 semesters"
    tuition_cost: Optional[str] = None  # e.g., "$4,500 per semester", "$9,200 program cost"
    program_urls: List[str] = Field(default_factory=list)


class ProjectSpecs(BaseModel):
    species_name: Optional[str] = None
    species_urls: List[str] = Field(default_factory=list)

    janka_hardness_lbf: Optional[str] = None
    janka_urls: List[str] = Field(default_factory=list)

    frame_joinery_method: Optional[str] = None
    frame_joinery_explanation: Optional[str] = None
    frame_joinery_urls: List[str] = Field(default_factory=list)

    drawer_joinery_method: Optional[str] = None
    drawer_joinery_explanation: Optional[str] = None
    drawer_joinery_urls: List[str] = Field(default_factory=list)

    finish_type: Optional[str] = None
    finish_urls: List[str] = Field(default_factory=list)  # Manufacturer/product pages, SDS/TDS

    voc_standard_urls: List[str] = Field(default_factory=list)  # URLs that state/confirm ≤450 g/L limit

    ventilation_requirements: Optional[str] = None
    ventilation_urls: List[str] = Field(default_factory=list)

    ppe_requirements: Optional[str] = None
    ppe_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_program_info() -> str:
    return """
Extract the woodworking program details as explicitly stated in the answer. Return a JSON with:

- program_name: The name of the program (if provided)
- institution_name: The name of the provider (college/school)
- location_state: The U.S. state for the program location (verbatim from the answer, e.g., "Colorado" or "Michigan" or abbreviations "CO"/"MI")
- city: The city (if provided)
- certificate_type: The credential name (ensure it is specifically a certificate in fine woodworking or furniture making if the answer claims so)
- institution_status: Summarize as one of: "accredited institution", "established woodworking school", or null if unclear in the answer
- duration: Program duration as stated (e.g., "9 months", "two semesters", etc.)
- tuition_cost: Tuition/cost information exactly as stated in the answer (could be a number, range, per-term, etc.)
- program_urls: All URLs cited that support the program details (official program pages strongly preferred). Extract all distinct URLs mentioned for the program.

Rules:
- Extract only what is present in the answer.
- For URLs, include full URLs. If a URL lacks a protocol, prepend http://.
- If a field is not mentioned, set it to null; if no URLs are present, return an empty list for program_urls.
""".strip()


def prompt_extract_project_specs() -> str:
    return """
Extract the dining table project specifications and their supporting URLs from the answer. Return a JSON with:

- species_name: Selected North American hardwood species (as named in the answer)
- species_urls: URLs that support the claim that it is a North American hardwood appropriate for furniture
- janka_hardness_lbf: The Janka hardness value in lbf exactly as the answer states it (prefer numeric form, but keep as-is)
- janka_urls: URLs cited for the Janka value

- frame_joinery_method: The joinery method used for table frame/leg connections (e.g., "mortise and tenon", "domino", etc.)
- frame_joinery_explanation: The explanation text (from the answer) for why this joinery provides adequate strength for a dining table
- frame_joinery_urls: URLs supporting the method’s suitability for table frames/legs

- drawer_joinery_method: The joinery method used for drawer box assembly (e.g., "half-blind dovetails", "locking rabbet", etc.)
- drawer_joinery_explanation: The explanation text for why the joinery is suitable for drawers
- drawer_joinery_urls: URLs supporting the drawer joinery choice

- finish_type: The selected finish appropriate for indoor dining furniture (e.g., "oil-based polyurethane", product name)
- finish_urls: URLs supporting that this finish is appropriate for indoor dining furniture and/or providing product specs (SDS/TDS)

- voc_standard_urls: URLs that state the EPA (or authoritative) VOC content standard/limit for wood/architectural coatings (≤450 g/L as claimed in the answer)

- ventilation_requirements: The ventilation guidance text as given in the answer
- ventilation_urls: URLs supporting ventilation requirements for the chosen finish/product category

- ppe_requirements: The PPE text as given in the answer
- ppe_urls: URLs supporting PPE guidance for applying the chosen finish

Rules:
- Extract only what is present in the answer.
- Keep values verbatim from the answer (e.g., do not normalize units).
- For all URL arrays, include all distinct URLs mentioned. If none given, return an empty list.
""".strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_text(value: Optional[str]) -> bool:
    return bool(value and isinstance(value, str) and value.strip())


def _has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0)


def _looks_like_number_in_text(text: Optional[str]) -> bool:
    if not _has_text(text):
        return False
    return bool(re.search(r"\d", text or ""))


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_program(evaluator: Evaluator, parent_node, program: ProgramInfo) -> None:
    """
    Build and verify the 'program_identification' subtree.
    All children must be critical due to rubric (and parent is critical).
    """
    prog_node = evaluator.add_parallel(
        id="program_identification",
        desc="Identify an appropriate woodworking certificate program meeting all stated constraints",
        parent=parent_node,
        critical=True
    )

    # 1) Location: Colorado or Michigan (verify by URLs)
    node_loc = evaluator.add_leaf(
        id="program_location",
        desc="Program is located in either Colorado or Michigan",
        parent=prog_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program is located in either Colorado or Michigan (accept city/state variations and abbreviations CO or MI).",
        node=node_loc,
        sources=program.program_urls,
        additional_instruction="Check the program page(s) to confirm its physical location is in Colorado or Michigan."
    )

    # 2) Certificate type: fine woodworking or furniture making (verify by URLs)
    node_cert = evaluator.add_leaf(
        id="program_certificate_type",
        desc="Program offers a certificate specifically in fine woodworking or furniture making",
        parent=prog_node,
        critical=True
    )
    await evaluator.verify(
        claim="This program offers a certificate specifically in fine woodworking or furniture making.",
        node=node_cert,
        sources=program.program_urls,
        additional_instruction="Look for the credential type; 'Certificate' (not just classes), and subject area explicitly mentions 'fine woodworking' or 'furniture making' (allow close synonyms such as 'furniture design' if clearly part of furniture-making training)."
    )

    # 3) Institution status: accredited institution or established woodworking school (verify by URLs)
    node_inst = evaluator.add_leaf(
        id="program_institution_status",
        desc="Program is offered by an accredited institution or established woodworking school",
        parent=prog_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program is provided by an accredited institution or an established woodworking school.",
        node=node_inst,
        sources=program.program_urls,
        additional_instruction="Accept community colleges/universities with recognized accreditation (e.g., HLC, MSCHE, SACSCOC) or a dedicated woodworking school with a clear, established program."
    )

    # 4) Duration provided (existence)
    evaluator.add_custom_node(
        result=_has_text(program.duration),
        id="program_duration_provided",
        desc="Program duration is provided",
        parent=prog_node,
        critical=True
    )

    # 5) Cost provided (existence)
    evaluator.add_custom_node(
        result=_has_text(program.tuition_cost),
        id="program_cost_provided",
        desc="Program tuition/cost information is provided",
        parent=prog_node,
        critical=True
    )

    # 6) URL reference(s) provided (existence)
    evaluator.add_custom_node(
        result=_has_any_url(program.program_urls),
        id="program_url_reference",
        desc="URL reference(s) are provided supporting the program claims (e.g., location, credential type, duration, and cost)",
        parent=prog_node,
        critical=True
    )


async def verify_project(evaluator: Evaluator, parent_node, specs: ProjectSpecs) -> None:
    """
    Build and verify the 'project_specifications' subtree.
    All children must be critical due to rubric (and parent is critical).
    """
    proj_node = evaluator.add_parallel(
        id="project_specifications",
        desc="Provide complete dining table (with drawers) specifications meeting all stated constraints",
        parent=parent_node,
        critical=True
    )

    # 1) Wood species provided (existence)
    evaluator.add_custom_node(
        result=_has_text(specs.species_name),
        id="wood_species",
        desc="Select and name a North American hardwood species appropriate for furniture construction",
        parent=proj_node,
        critical=True
    )

    # 2) Wood species URL support (verify by URLs)
    node_species_ref = evaluator.add_leaf(
        id="wood_species_url_reference",
        desc="Provide URL reference(s) supporting the wood species selection as a North American hardwood appropriate for furniture use",
        parent=proj_node,
        critical=True
    )
    species = specs.species_name or "the selected species"
    await evaluator.verify(
        claim=f"{species} is a North American hardwood and is appropriate for furniture construction.",
        node=node_species_ref,
        sources=specs.species_urls,
        additional_instruction="Confirm both hardwood classification and typical furniture usage. Allow common synonyms (e.g., 'Northern red oak' vs 'red oak')."
    )

    # 3) Janka hardness provided (existence, ensure digits present)
    evaluator.add_custom_node(
        result=_looks_like_number_in_text(specs.janka_hardness_lbf),
        id="janka_hardness_lbf",
        desc="Provide the selected species' Janka hardness rating in lbf",
        parent=proj_node,
        critical=True
    )

    # 4) Janka hardness URL support (verify by URLs)
    node_janka_ref = evaluator.add_leaf(
        id="janka_url_reference",
        desc="Provide URL reference(s) supporting the Janka hardness value",
        parent=proj_node,
        critical=True
    )
    janka_val = specs.janka_hardness_lbf or ""
    await evaluator.verify(
        claim=f"The Janka hardness rating (lbf) for {species} is {janka_val}.",
        node=node_janka_ref,
        sources=specs.janka_urls,
        additional_instruction="Verify the numeric value (allow reasonable rounding). Prefer authoritative sources (e.g., Wood Database, USDA FPL, manufacturer/industry references)."
    )

    # 5) Frame joinery method provided (existence)
    evaluator.add_custom_node(
        result=_has_text(specs.frame_joinery_method),
        id="frame_joinery_method",
        desc="Specify the joinery method for the table frame and leg connections",
        parent=proj_node,
        critical=True
    )

    # 6) Frame joinery strength explanation provided (existence)
    evaluator.add_custom_node(
        result=_has_text(specs.frame_joinery_explanation),
        id="frame_joinery_strength_explanation",
        desc="Explain why the frame/leg joinery provides adequate structural strength for a dining table",
        parent=proj_node,
        critical=True
    )

    # 7) Frame joinery URL support (verify by URLs)
    node_frame_ref = evaluator.add_leaf(
        id="frame_joinery_url_reference",
        desc="Provide URL reference(s) supporting the frame/leg joinery method information",
        parent=proj_node,
        critical=True
    )
    frame_joinery = specs.frame_joinery_method or "the specified frame/leg joinery"
    await evaluator.verify(
        claim=f"{frame_joinery} is an appropriate, strong joint for table frames and leg connections.",
        node=node_frame_ref,
        sources=specs.frame_joinery_urls,
        additional_instruction="Look for statements about strength for table bases/frames (e.g., mortise & tenon, bridle, floating tenon/domino). Support should indicate suitability for load-bearing furniture."
    )

    # 8) Drawer joinery method provided (existence)
    evaluator.add_custom_node(
        result=_has_text(specs.drawer_joinery_method),
        id="drawer_joinery_method",
        desc="Specify the joinery method for drawer box assembly",
        parent=proj_node,
        critical=True
    )

    # 9) Drawer joinery explanation provided (existence)
    evaluator.add_custom_node(
        result=_has_text(specs.drawer_joinery_explanation),
        id="drawer_joinery_suitability_explanation",
        desc="Explain why the drawer joinery is suitable for drawer construction",
        parent=proj_node,
        critical=True
    )

    # 10) Drawer joinery URL support (verify by URLs)
    node_drawer_ref = evaluator.add_leaf(
        id="drawer_joinery_url_reference",
        desc="Provide URL reference(s) supporting the drawer joinery method information",
        parent=proj_node,
        critical=True
    )
    drawer_joinery = specs.drawer_joinery_method or "the specified drawer joinery"
    await evaluator.verify(
        claim=f"{drawer_joinery} is a suitable joint for drawer box construction.",
        node=node_drawer_ref,
        sources=specs.drawer_joinery_urls,
        additional_instruction="Support should indicate typical use in drawer boxes (e.g., dovetails, locking rabbet), focusing on mechanical strength and longevity."
    )

    # 11) Finish type provided (existence)
    evaluator.add_custom_node(
        result=_has_text(specs.finish_type),
        id="finish_type",
        desc="Select a wood finish appropriate for indoor dining furniture",
        parent=proj_node,
        critical=True
    )

    # 12) Finish type suitability URL support (verify by URLs)
    node_finish_ref = evaluator.add_leaf(
        id="finish_type_url_reference",
        desc="Provide URL reference(s) supporting that the selected finish is appropriate for indoor dining furniture",
        parent=proj_node,
        critical=True
    )
    finish_type = specs.finish_type or "the selected finish"
    await evaluator.verify(
        claim=f"{finish_type} is appropriate for indoor dining furniture use.",
        node=node_finish_ref,
        sources=specs.finish_urls,
        additional_instruction="Look for properties such as durability, chemical/water resistance, and suitability for tables (may include manufacturer product pages or expert articles)."
    )

    # 13) Finish VOC compliance confirmed (verify by product URLs)
    node_voc_ok = evaluator.add_leaf(
        id="finish_voc_compliance_confirmed",
        desc="Confirm the finish complies with EPA VOC content standards (≤450 g/L for wood coatings)",
        parent=proj_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The finish '{finish_type}' has a VOC content ≤ 450 g/L (complies with EPA VOC content standards for wood/architectural coatings).",
        node=node_voc_ok,
        sources=specs.finish_urls,
        additional_instruction="Verify VOC content from SDS/TDS or official product specs; accept 'VOC (less water/exempt solvents)' if ≤450 g/L. If multiple values exist, use the most conservative/highest and ensure it’s ≤450 g/L."
    )

    # 14) VOC limit reference URL(s) (verify that a cited URL states ≤450 g/L limit)
    node_voc_ref = evaluator.add_leaf(
        id="finish_voc_url_reference",
        desc="Provide URL reference(s) supporting the VOC limit and the basis/evidence for the finish compliance claim",
        parent=proj_node,
        critical=True
    )
    await evaluator.verify(
        claim="The EPA (or authoritative regulatory) VOC content standard for wood/architectural coatings includes a limit of ≤ 450 g/L.",
        node=node_voc_ref,
        sources=specs.voc_standard_urls,
        additional_instruction="Confirm that at least one cited regulation or authoritative standard explicitly states a ≤450 g/L VOC content limit for relevant wood/architectural coatings categories."
    )

    # 15) Ventilation requirements provided (existence)
    evaluator.add_custom_node(
        result=_has_text(specs.ventilation_requirements),
        id="ventilation_requirements",
        desc="Document ventilation requirements for finish application",
        parent=proj_node,
        critical=True
    )

    # 16) Ventilation requirements URL support (verify by URLs)
    node_vent_ref = evaluator.add_leaf(
        id="ventilation_url_reference",
        desc="Provide URL reference(s) supporting the ventilation guidance for finish application",
        parent=proj_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided sources include ventilation requirements for applying {finish_type} (e.g., use in a well-ventilated area, provide adequate airflow, avoid ignition sources as applicable).",
        node=node_vent_ref,
        sources=specs.ventilation_urls,
        additional_instruction="Prefer SDS/TDS or official safety guidance for the chosen finish/product category."
    )

    # 17) PPE requirements provided (existence)
    evaluator.add_custom_node(
        result=_has_text(specs.ppe_requirements),
        id="ppe_requirements",
        desc="Specify necessary personal protective equipment for finish application",
        parent=proj_node,
        critical=True
    )

    # 18) PPE requirements URL support (verify by URLs)
    node_ppe_ref = evaluator.add_leaf(
        id="ppe_url_reference",
        desc="Provide URL reference(s) supporting the PPE guidance for finish application",
        parent=proj_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided sources include PPE guidance appropriate for applying {finish_type} (e.g., gloves, eye protection, and respirator as needed).",
        node=node_ppe_ref,
        sources=specs.ppe_urls,
        additional_instruction="Prefer SDS/TDS or official product/application guidance referencing PPE requirements."
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
    Evaluate an answer for the woodworking program + project specifications task.
    """
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
        default_model=model,
    )

    # Run extractions (in parallel)
    program_task = evaluator.extract(
        prompt=prompt_extract_program_info(),
        template_class=ProgramInfo,
        extraction_name="program_info"
    )
    project_task = evaluator.extract(
        prompt=prompt_extract_project_specs(),
        template_class=ProjectSpecs,
        extraction_name="project_specs"
    )

    program_info, project_specs = await asyncio.gather(program_task, project_task)

    # Build verification tree:
    # According to rubric, both main branches are critical to overall success.
    # We will attach their subtrees under the root and mark them critical.
    # Note: children of these critical nodes must also be critical (enforced by framework).

    # Program verification
    await verify_program(evaluator, root, program_info)

    # Project verification
    await verify_project(evaluator, root, project_specs)

    # Return structured summary
    return evaluator.get_summary()