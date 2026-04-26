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
TASK_ID = "illinois_edd_caep_ra11"
TASK_DESCRIPTION = """
Identify a Doctor of Education (EdD) program offered by a CAEP-accredited institution in Illinois that demonstrates compliance with CAEP Standard RA.1.1 regarding research methodology training. Specifically, verify that the program: (1) is offered by an institution in Illinois that holds current CAEP accreditation for its educator preparation programs, (2) provides a Doctor of Education (EdD) degree at the post-baccalaureate or graduate level, (3) includes required coursework in research methodologies as part of its curriculum, and (4) addresses candidates' understanding and use of qualitative, quantitative, and/or mixed methods research methodologies. Provide the name of the institution, the EdD program, and URL-based evidence for each verification point above.
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ProgramSelection(BaseModel):
    institution_name: Optional[str] = None
    edd_program_name: Optional[str] = None


class EvidenceExtraction(BaseModel):
    # Point 1: split into IL location and CAEP accreditation URLs
    point1_location_urls: List[str] = Field(default_factory=list)
    point1_caep_urls: List[str] = Field(default_factory=list)

    # Point 2–4 URLs
    point2_urls: List[str] = Field(default_factory=list)  # EdD at graduate level
    point3_urls: List[str] = Field(default_factory=list)  # Required research methodology coursework
    point4_urls: List[str] = Field(default_factory=list)  # Qual/Quant/Mixed methods addressed

    # Constraint-related URLs (optional per rubric wording)
    leads_to_licensure_urls: List[str] = Field(default_factory=list)
    ra1_to_ra5_urls: List[str] = Field(default_factory=list)
    admissions_criteria_urls: List[str] = Field(default_factory=list)
    ra11_other_proficiency_urls: List[str] = Field(default_factory=list)
    ra22_urls: List[str] = Field(default_factory=list)
    ra33_urls: List[str] = Field(default_factory=list)
    ra34_urls: List[str] = Field(default_factory=list)
    ra12_alignment_urls: List[str] = Field(default_factory=list)

    # Any extras not easily classified
    additional_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_program_selection() -> str:
    return """
    From the answer, extract the single institution and single EdD program that are being evaluated.

    Return JSON with:
    - institution_name: The exact institution name chosen (e.g., "University of Illinois Urbana-Champaign").
    - edd_program_name: The exact EdD program name chosen (e.g., "EdD in Educational Leadership").
    
    If either item is not explicitly provided, return null for that field.
    """


def prompt_extract_evidence_urls() -> str:
    return """
    Extract all URLs explicitly cited in the answer and categorize them according to the following evidence buckets. 
    Include only URLs that the answer appears to use for that category. Accept both plain URLs and markdown links.
    
    Required categories:
    - point1_location_urls: URLs that demonstrate the institution is located in Illinois (IL), e.g., official "About" pages, campus address pages, or authoritative profiles.
    - point1_caep_urls: URLs that demonstrate the institution/EPP currently holds CAEP accreditation for educator preparation programs (e.g., CAEP official provider listing or institution CAEP page).
    - point2_urls: URLs that demonstrate the institution offers a Doctor of Education (EdD) at the post-baccalaureate/graduate level (e.g., program page, catalog).
    - point3_urls: URLs that demonstrate the EdD curriculum includes required (not optional) coursework in research methodologies (e.g., curriculum page, catalog requirement page).
    - point4_urls: URLs that demonstrate the program addresses candidates' understanding and use of qualitative, quantitative, and/or mixed methods research methodologies (e.g., program outcomes, curriculum framework, course descriptions).

    Constraint-related (optional; include if provided in the answer):
    - leads_to_licensure_urls
    - ra1_to_ra5_urls
    - admissions_criteria_urls
    - ra11_other_proficiency_urls
    - ra22_urls
    - ra33_urls
    - ra34_urls
    - ra12_alignment_urls

    If a category has no URLs in the answer, return an empty list for that category.
    Put any remaining, unclassified but mentioned URLs into 'additional_urls'.
    
    Do not invent URLs. Return exactly what the answer provides.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_program_identification(
    evaluator: Evaluator,
    parent_node,
    selection: ProgramSelection
) -> None:
    """
    Build Program Identification subtree:
    - Provides_Institution_Name (critical, existence)
    - Provides_EdD_Program_Name (critical, existence)
    """
    node = evaluator.add_parallel(
        id="Program_Identification",
        desc="Select and clearly name a single institution and a single EdD program to be evaluated.",
        parent=parent_node,
        critical=True
    )

    inst_exists = bool(selection.institution_name and selection.institution_name.strip())
    prog_exists = bool(selection.edd_program_name and selection.edd_program_name.strip())

    evaluator.add_custom_node(
        result=inst_exists,
        id="Provides_Institution_Name",
        desc="Answer provides the institution name.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=prog_exists,
        id="Provides_EdD_Program_Name",
        desc="Answer provides the EdD program name.",
        parent=node,
        critical=True
    )


async def build_point1_il_and_caep(
    evaluator: Evaluator,
    parent_node,
    selection: ProgramSelection,
    evidence: EvidenceExtraction
) -> None:
    """
    Point 1: Institution in Illinois AND current CAEP accreditation.
    We enforce URL presence for both sub-points and verify both with provided URLs.
    """
    point1 = evaluator.add_parallel(
        id="Point1_Illinois_And_Current_CAEP_Accreditation_With_URL",
        desc="Provides URL evidence that the institution is located in Illinois AND holds current CAEP accreditation for its educator preparation programs.",
        parent=parent_node,
        critical=True
    )

    # Enforce URL presence for both parts
    evaluator.add_custom_node(
        result=len(evidence.point1_location_urls) > 0,
        id="Point1_Location_URLs_Provided",
        desc="URL(s) provided to evidence that the institution is located in Illinois.",
        parent=point1,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(evidence.point1_caep_urls) > 0,
        id="Point1_CAEP_URLs_Provided",
        desc="URL(s) provided to evidence current CAEP accreditation.",
        parent=point1,
        critical=True
    )

    # Verify IL location
    il_loc_leaf = evaluator.add_leaf(
        id="Point1_IL_Location_Verify",
        desc="Institution is located in Illinois (IL).",
        parent=point1,
        critical=True
    )
    inst_name = selection.institution_name or "the institution"
    claim_il = f"The institution {inst_name} is located in Illinois (IL)."
    await evaluator.verify(
        claim=claim_il,
        node=il_loc_leaf,
        sources=evidence.point1_location_urls,
        additional_instruction="Look for the state 'Illinois' or 'IL' on the referenced page(s), e.g., addresses, campus location, statewide identification. Accept official pages or authoritative organizational profiles."
    )

    # Verify current CAEP accreditation
    caep_leaf = evaluator.add_leaf(
        id="Point1_Current_CAEP_Verify",
        desc="Institution holds current CAEP accreditation for its educator preparation programs.",
        parent=point1,
        critical=True
    )
    claim_caep = f"The institution {inst_name} currently holds CAEP accreditation for educator preparation programs (EPP)."
    await evaluator.verify(
        claim=claim_caep,
        node=caep_leaf,
        sources=evidence.point1_caep_urls,
        additional_instruction="Use CAEP official provider listings or the institution's official accreditation page. 'Current' should indicate the accreditation is active (not expired)."
    )


async def build_point2_edd_grad_level(
    evaluator: Evaluator,
    parent_node,
    selection: ProgramSelection,
    evidence: EvidenceExtraction
) -> None:
    """
    Point 2: EdD at post-baccalaureate/graduate level with URLs.
    Enforce URL presence, then verify via URLs.
    """
    point2 = evaluator.add_parallel(
        id="Point2_EdD_Graduate_Level_With_URL",
        desc="Provides URL evidence that the institution offers a Doctor of Education (EdD) at the post-baccalaureate/graduate level.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(evidence.point2_urls) > 0,
        id="Point2_URLs_Provided",
        desc="URL(s) provided to evidence EdD at graduate level.",
        parent=point2,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Point2_EdD_Verify",
        desc="EdD degree is offered at the post-baccalaureate/graduate level.",
        parent=point2,
        critical=True
    )
    inst = selection.institution_name or "the institution"
    program = selection.edd_program_name or "the EdD program"
    claim = f"{inst} offers the program '{program}' that confers a Doctor of Education (EdD) degree at the graduate (post-baccalaureate) level."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=evidence.point2_urls,
        additional_instruction="Confirm the degree is specifically 'Doctor of Education (EdD)' and that it is a graduate-level program. Use program page or official catalog."
    )


async def build_point3_required_research_methods(
    evaluator: Evaluator,
    parent_node,
    selection: ProgramSelection,
    evidence: EvidenceExtraction
) -> None:
    """
    Point 3: Required coursework in research methodologies with URLs.
    Enforce URL presence, then verify via URLs.
    """
    point3 = evaluator.add_parallel(
        id="Point3_Required_Research_Methodology_Coursework_With_URL",
        desc="Provides URL evidence that the EdD curriculum includes required (not optional) coursework in research methodologies.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(evidence.point3_urls) > 0,
        id="Point3_URLs_Provided",
        desc="URL(s) provided to evidence required research methodology coursework.",
        parent=point3,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Point3_Required_Methods_Verify",
        desc="EdD curriculum includes required (not optional) coursework in research methodologies.",
        parent=point3,
        critical=True
    )
    program = selection.edd_program_name or "the EdD program"
    claim = f"The curriculum for {program} includes required (not elective) coursework in research methodologies."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=evidence.point3_urls,
        additional_instruction="Look for language indicating 'required' courses in research methods (e.g., 'Research Methods', 'Quantitative Methods', 'Qualitative Methods', 'Mixed Methods', 'Doctoral Research Methods'). Distinguish required vs elective."
    )


async def build_point4_qual_quant_mixed(
    evaluator: Evaluator,
    parent_node,
    selection: ProgramSelection,
    evidence: EvidenceExtraction
) -> None:
    """
    Point 4: Addresses qualitative, quantitative, and/or mixed methods with URLs.
    Enforce URL presence, then verify via URLs.
    """
    point4 = evaluator.add_parallel(
        id="Point4_Qual_Quant_AndOr_Mixed_Methods_Addressed_With_URL",
        desc="Provides URL evidence that the program addresses candidates' understanding and use of qualitative, quantitative, and/or mixed methods research methodologies.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(evidence.point4_urls) > 0,
        id="Point4_URLs_Provided",
        desc="URL(s) provided to evidence qualitative/quantitative/mixed methods are addressed.",
        parent=point4,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Point4_Methods_Foci_Verify",
        desc="Program addresses candidates' understanding and use of qualitative, quantitative, and/or mixed methods research methodologies.",
        parent=point4,
        critical=True
    )
    program = selection.edd_program_name or "the EdD program"
    claim = f"{program} addresses candidates' understanding and use of qualitative, quantitative, and/or mixed methods research methodologies."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=evidence.point4_urls,
        additional_instruction="Confirm explicit mention of qualitative, quantitative, and/or mixed methods in program learning outcomes, curriculum descriptions, or course descriptions."
    )


async def build_constraints(
    evaluator: Evaluator,
    parent_node,
    selection: ProgramSelection,
    evidence: EvidenceExtraction
) -> None:
    """
    Build constraint verification subtree. These are critical per rubric, but do not require URLs.
    If URLs are provided in the answer, use them; otherwise fall back to simple verification against the answer text.
    """
    constraints = evaluator.add_parallel(
        id="Constraint_Verification",
        desc="Verify all additional required constraints provided in the constraints list (no extra requirement that these must be supported by URLs).",
        parent=parent_node,
        critical=True
    )

    inst = selection.institution_name or "the institution"
    program = selection.edd_program_name or "the EdD program"

    # 1) Licensure/Certification/Endorsement
    c1 = evaluator.add_leaf(
        id="Leads_To_Licensure_Certification_Or_Endorsement",
        desc="Program leads to licensure, certification, or endorsement.",
        parent=constraints,
        critical=True
    )
    claim1 = f"{program} leads to a licensure, certification, or endorsement outcome for candidates."
    await evaluator.verify(
        claim=claim1,
        node=c1,
        sources=evidence.leads_to_licensure_urls if evidence.leads_to_licensure_urls else None,
        additional_instruction="Accept if the program directly states it leads to licensure/certification/endorsement or is an approved/recognized pathway."
    )

    # 2) Addresses all five CAEP RA.1–RA.5
    c2 = evaluator.add_leaf(
        id="Addresses_All_Five_Core_CAEP_Advanced_Standards_RA1_to_RA5",
        desc="Program addresses all five core CAEP advanced-level standards (RA.1 through RA.5).",
        parent=constraints,
        critical=True
    )
    claim2 = f"{program} addresses all five CAEP Advanced Standards RA.1 through RA.5."
    await evaluator.verify(
        claim=claim2,
        node=c2,
        sources=evidence.ra1_to_ra5_urls if evidence.ra1_to_ra5_urls else None,
        additional_instruction="Look for alignment matrices, accreditation narratives, or EPP assurances explicitly covering RA.1, RA.2, RA.3, RA.4, and RA.5."
    )

    # 3) Admissions minimum criteria (CAEP)
    c3 = evaluator.add_leaf(
        id="Admission_Minimum_Criteria_Met",
        desc="Admissions meet CAEP minimum criteria (group avg GPA ≥ 3.0 OR top 50th percentile nationally normed assessments OR highest state/grad school minimums).",
        parent=constraints,
        critical=True
    )
    claim3 = "Admissions requirements meet CAEP minimum criteria (group average GPA ≥ 3.0 or group average performance in the top 50th percentile on nationally normed assessments or the highest of state/graduate school minimums)."
    await evaluator.verify(
        claim=claim3,
        node=c3,
        sources=evidence.admissions_criteria_urls if evidence.admissions_criteria_urls else None,
        additional_instruction="Accept explicit statements of meeting CAEP minimums or equivalent evidence indicating thresholds are met for advanced programs."
    )

    # 4) RA.1.1 other proficiencies (beyond methods)
    c4 = evaluator.add_leaf(
        id="RA11_Other_Proficiency_Areas_Addressed",
        desc="Program ensures candidates demonstrate other RA.1.1 proficiency areas: data literacy; data analysis for equity; collaborative activities; technology applications; professional dispositions/ethics.",
        parent=constraints,
        critical=True
    )
    claim4 = f"{program} addresses data literacy, data analysis for equity, collaborative activities, technology applications, and professional dispositions/ethics, as required by CAEP RA.1.1."
    await evaluator.verify(
        claim=claim4,
        node=c4,
        sources=evidence.ra11_other_proficiency_urls if evidence.ra11_other_proficiency_urls else None,
        additional_instruction="Look for program outcomes/assessments referring to data literacy, equity analysis, collaboration, technology, and professional dispositions/ethics."
    )

    # 5) RA.2.2 Culminating experiences
    c5 = evaluator.add_leaf(
        id="RA22_Culminating_Experiences",
        desc="Program provides culminating experiences where candidates demonstrate proficiencies via problem-based tasks or research (qualitative, quantitative, mixed methods, or action research) as specified in RA.2.2.",
        parent=constraints,
        critical=True
    )
    claim5 = f"{program} includes culminating experiences where candidates demonstrate proficiencies via problem-based tasks or research (qualitative/quantitative/mixed/action research) per RA.2.2."
    await evaluator.verify(
        claim=claim5,
        node=c5,
        sources=evidence.ra22_urls if evidence.ra22_urls else None,
        additional_instruction="Confirm existence of dissertation, capstone, action research, or applied research culminating products aligned to RA.2.2."
    )

    # 6) RA.3.3 Disaggregated progression data
    c6 = evaluator.add_leaf(
        id="RA33_Disaggregated_Progression_Data",
        desc="Program uses disaggregated data (by race, ethnicity, and other relevant categories) to monitor candidate progression from admission through completion (RA.3.3).",
        parent=constraints,
        critical=True
    )
    claim6 = f"{program} uses disaggregated data (e.g., by race/ethnicity) to monitor candidate progression from admission to completion, consistent with RA.3.3."
    await evaluator.verify(
        claim=claim6,
        node=c6,
        sources=evidence.ra33_urls if evidence.ra33_urls else None,
        additional_instruction="Look for dashboard or assessment system documentation mentioning disaggregated reporting of candidate progression data."
    )

    # 7) RA.3.4 Multiple measures, disaggregated analysis
    c7 = evaluator.add_leaf(
        id="RA34_Multiple_Measures_Disaggregated_Analysis",
        desc="Program provides multiple measures of candidate competency with disaggregated analysis (RA.3.4).",
        parent=constraints,
        critical=True
    )
    claim7 = f"{program} uses multiple measures of candidate competency and provides disaggregated analysis as required by RA.3.4."
    await evaluator.verify(
        claim=claim7,
        node=c7,
        sources=evidence.ra34_urls if evidence.ra34_urls else None,
        additional_instruction="Seek evidence of multiple assessments (e.g., course assessments, practica, capstone) analyzed with disaggregation."
    )

    # 8) RA.1.2 Alignment with state/national standards
    c8 = evaluator.add_leaf(
        id="RA12_Alignment_With_State_Or_National_Standards",
        desc="Program aligns with approved state and/or national discipline-specific standards (RA.1.2).",
        parent=constraints,
        critical=True
    )
    claim8 = f"{program} aligns with approved state and/or national discipline-specific standards consistent with RA.1.2."
    await evaluator.verify(
        claim=claim8,
        node=c8,
        sources=evidence.ra12_alignment_urls if evidence.ra12_alignment_urls else None,
        additional_instruction="Look for alignment matrices or explicit references to state or national standards for the discipline (e.g., leadership, curriculum, specializations)."
    )


async def build_proposed_question_verifications(
    evaluator: Evaluator,
    parent_node,
    selection: ProgramSelection,
    evidence: EvidenceExtraction
) -> None:
    """
    Build the 'Proposed Question Verifications With URL Evidence' subtree.
    Contains Points 1–4, each critical. Each Point enforces URL presence.
    """
    verif = evaluator.add_parallel(
        id="Proposed_Question_Verifications_With_URL_Evidence",
        desc="For each of the 4 verification points explicitly stated in the proposed question, provide URL-based evidence supporting the claim.",
        parent=parent_node,
        critical=True
    )

    await build_point1_il_and_caep(evaluator, verif, selection, evidence)
    await build_point2_edd_grad_level(evaluator, verif, selection, evidence)
    await build_point3_required_research_methods(evaluator, verif, selection, evidence)
    await build_point4_qual_quant_mixed(evaluator, verif, selection, evidence)


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
    Evaluate an answer for the Illinois EdD CAEP RA.1.1 verification task.
    """
    # Initialize evaluator
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

    # Extract key selections and evidence URLs
    selection, evidence = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_program_selection(),
            template_class=ProgramSelection,
            extraction_name="program_selection"
        ),
        evaluator.extract(
            prompt=prompt_extract_evidence_urls(),
            template_class=EvidenceExtraction,
            extraction_name="evidence_urls"
        ),
    )

    # Add a critical main task node under root to reflect rubric root criticality
    main_task = evaluator.add_sequential(
        id="Illinois_EdD_CAEP_RA11_Task",
        desc="Identify one Illinois EdD program at a CAEP-accredited institution and verify (with URLs for the 4 explicitly listed verification points) the proposed-question requirements, plus satisfy all provided constraints.",
        parent=root,
        critical=True
    )

    # Build subtrees
    await build_program_identification(evaluator, main_task, selection)
    await build_proposed_question_verifications(evaluator, main_task, selection, evidence)
    await build_constraints(evaluator, main_task, selection, evidence)

    # Optional: record ground truth expectations template for transparency
    evaluator.add_ground_truth({
        "expected_points": [
            "Point1: Institution in Illinois and currently CAEP-accredited (URL-based)",
            "Point2: Offers EdD at graduate level (URL-based)",
            "Point3: Required coursework in research methodology (URL-based)",
            "Point4: Addresses qualitative/quantitative/mixed methods (URL-based)"
        ],
        "constraints_expected": [
            "Leads to licensure/certification/endorsement",
            "Addresses all five CAEP Advanced Standards RA.1–RA.5",
            "Admissions meet CAEP minimum criteria",
            "RA.1.1: other proficiencies (data literacy; equity analysis; collaboration; technology; dispositions/ethics)",
            "RA.2.2: culminating experiences",
            "RA.3.3: disaggregated progression data",
            "RA.3.4: multiple measures with disaggregated analysis",
            "RA.1.2: alignment with state/national standards"
        ]
    }, gt_type="rubric_requirements")

    return evaluator.get_summary()