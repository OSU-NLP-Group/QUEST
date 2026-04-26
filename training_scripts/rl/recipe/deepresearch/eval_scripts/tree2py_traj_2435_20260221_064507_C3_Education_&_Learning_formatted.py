import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "uva_president_doctorate"
TASK_DESCRIPTION = (
    "Identify the current president of the University of Virginia who previously served as dean of the Darden School of Business. "
    "Then, provide comprehensive information about his doctoral education by answering the following: "
    "(1) What university did he attend for his doctorate? "
    "(2) What was the specific name of the doctoral program, and what type of doctoral degree (e.g., Ph.D., Ed.D., etc.) did he earn? "
    "(3) Was the degree awarded with any honors or distinctions? If so, specify what distinction. "
    "(4) In what year was the degree conferred? "
    "(5) What is the complete title of his dissertation? "
    "(6) Is the doctoral program he completed described as a two-year program? "
    "(7) Did he receive any distinguished alumni award from that doctoral institution after earning his degree? If so, what award and in what year? "
    "For each piece of information, provide at least one reference URL that supports your answer."
)

# Ground truth anchors (used for simple correctness checks)
EXPECTED_IDENTITY_NAME = "Scott C. Beardsley"
EXPECTED_INSTITUTION = "University of Pennsylvania"
EXPECTED_DEGREE_TYPE = "Ed.D."
EXPECTED_PROGRAM_NAME_HINTS = [
    "Executive Doctorate in Higher Education Management",
    "Higher Education Management",
    "Executive Doctorate in Higher Education Management (EDHEM)"
]
EXPECTED_CONFERRED_YEAR = "2015"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class IdentityInfo(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class InstitutionInfo(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ProgramInfo(BaseModel):
    program_name: Optional[str] = None
    degree_type: Optional[str] = None
    distinction: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DissertationInfo(BaseModel):
    conferred_year: Optional[str] = None
    title: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AdditionalInfo(BaseModel):
    is_two_year_program: Optional[str] = None  # Expected values: "yes", "no", "true", "false", or descriptive
    two_year_sources: List[str] = Field(default_factory=list)

    alumni_award_name: Optional[str] = None
    alumni_award_year: Optional[str] = None
    alumni_award_sources: List[str] = Field(default_factory=list)


class DoctoralEducationExtraction(BaseModel):
    academic_leader_identity: Optional[IdentityInfo] = None
    doctoral_institution: Optional[InstitutionInfo] = None
    doctoral_program: Optional[ProgramInfo] = None
    dissertation_details: Optional[DissertationInfo] = None
    additional_verifications: Optional[AdditionalInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_doctoral_education() -> str:
    return """
    Extract structured information from the answer about the identified academic leader and his doctoral education. Return a JSON object with these fields:

    academic_leader_identity:
      - name: the full name of the identified UVA president who previously served as Darden dean (string or null)
      - sources: list of URL(s) that confirm his identity, current role (UVA president), and the fact that he previously served as Darden dean

    doctoral_institution:
      - name: the university where he earned his doctoral degree (string or null)
      - sources: list of URL(s) confirming the doctoral institution

    doctoral_program:
      - program_name: the specific name of the doctoral program (e.g., "Executive Doctorate in Higher Education Management") (string or null)
      - degree_type: the type of doctoral degree (e.g., "Ed.D.", "Ph.D.") (string or null)
      - distinction: if any honor/distinction was mentioned for the degree (e.g., "with distinction"); otherwise null
      - sources: list of URL(s) confirming the program name, degree type, and distinction (if any)

    dissertation_details:
      - conferred_year: the year when the degree was conferred (string or null; keep as written in the answer)
      - title: the complete dissertation title as stated in the answer (string or null)
      - sources: list of URL(s) confirming the conferred year and the dissertation title

    additional_verifications:
      - is_two_year_program: whether the program is described as a two-year program ("yes"/"no"/"true"/"false"/other descriptive string; null if not stated)
      - two_year_sources: list of URL(s) supporting the program's duration description, if provided
      - alumni_award_name: name of any distinguished alumni award he received from the doctoral institution after earning the degree (string or null)
      - alumni_award_year: year of that alumni award (string or null)
      - alumni_award_sources: list of URL(s) supporting the alumni award details

    Special rules for URLs:
    - Only include valid URLs explicitly present in the answer.
    - If a URL is missing a protocol, prepend "http://".
    - If no URLs are provided for a section in the answer, return an empty list for that section's sources.

    If any field is not mentioned in the answer, set it to null (or [] for lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _union_sources(*lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not url or not isinstance(url, str):
                continue
            u = url.strip()
            if u and u not in seen:
                merged.append(u)
                seen.add(u)
    return merged


def _truthy_flag(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    val = s.strip().lower()
    if val in {"yes", "true", "y", "t"}:
        return True
    if val in {"no", "false", "n", "f"}:
        return False
    return None  # Unknown/ambiguous description


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_identity_subtree(evaluator: Evaluator, parent, data: DoctoralEducationExtraction) -> None:
    node = evaluator.add_sequential(
        id="academic_leader_identity",
        desc="Correctly identify the current University of Virginia president who previously served as dean of the Darden School of Business",
        parent=parent,
        critical=True
    )

    ident = data.academic_leader_identity or IdentityInfo()

    # Leaf: identity_verification
    leaf_identity = evaluator.add_leaf(
        id="identity_verification",
        desc="The identified individual is Scott C. Beardsley",
        parent=node,
        critical=True
    )
    claim_identity = f"The name '{ident.name}' and '{EXPECTED_IDENTITY_NAME}' refer to the same person."
    await evaluator.verify(
        claim=claim_identity,
        node=leaf_identity,
        additional_instruction="Allow minor variations (middle initials, casing). Treat 'Scott Beardsley' and 'Scott C. Beardsley' as equivalent."
    )

    # Leaf: identity_source_url
    leaf_identity_src = evaluator.add_leaf(
        id="identity_source_url",
        desc="Provide a reference URL confirming the identity",
        parent=node,
        critical=True
    )
    claim_identity_src = (
        "Scott C. Beardsley is the current president of the University of Virginia and previously served as dean of the Darden School of Business."
    )
    await evaluator.verify(
        claim=claim_identity_src,
        node=leaf_identity_src,
        sources=ident.sources,
        additional_instruction="The URL(s) should explicitly confirm both roles: current UVA president and past Darden dean."
    )


async def build_institution_subtree(evaluator: Evaluator, parent, data: DoctoralEducationExtraction) -> None:
    node = evaluator.add_sequential(
        id="doctoral_institution",
        desc="Identify the university where the doctoral degree was earned",
        parent=parent,
        critical=True
    )

    inst = data.doctoral_institution or InstitutionInfo()

    # Leaf: institution_verification
    leaf_inst = evaluator.add_leaf(
        id="institution_verification",
        desc="The doctoral degree was earned from the University of Pennsylvania",
        parent=node,
        critical=True
    )
    claim_inst = f"The doctoral institution is '{inst.name}', and it is the University of Pennsylvania."
    await evaluator.verify(
        claim=claim_inst,
        node=leaf_inst,
        additional_instruction="Check if the provided institution corresponds to 'University of Pennsylvania' allowing minor naming variations (e.g., 'Penn', 'UPenn')."
    )

    # Leaf: institution_source_url
    leaf_inst_src = evaluator.add_leaf(
        id="institution_source_url",
        desc="Provide a reference URL confirming the doctoral institution",
        parent=node,
        critical=True
    )
    claim_inst_src = "Scott C. Beardsley earned his doctoral degree from the University of Pennsylvania."
    await evaluator.verify(
        claim=claim_inst_src,
        node=leaf_inst_src,
        sources=inst.sources,
        additional_instruction="The URL(s) should explicitly state University of Pennsylvania as the doctoral institution."
    )


async def build_program_details_subtree(evaluator: Evaluator, parent, data: DoctoralEducationExtraction) -> None:
    node = evaluator.add_sequential(
        id="doctoral_program_details",
        desc="Verify specific details about the doctoral program",
        parent=parent,
        critical=True
    )

    prog = data.doctoral_program or ProgramInfo()

    # Child: program_specifics (parallel)
    specifics = evaluator.add_parallel(
        id="program_specifics",
        desc="Verify the program name and degree type",
        parent=node,
        critical=True
    )

    # Leaf: program_name
    leaf_prog_name = evaluator.add_leaf(
        id="program_name",
        desc="The program is the Executive Doctorate in Higher Education Management (or Higher Education Management)",
        parent=specifics,
        critical=True
    )
    claim_prog_name = (
        f"The doctoral program completed is '{prog.program_name}', which corresponds to the Executive Doctorate in Higher Education Management (Higher Education Management)."
    )
    await evaluator.verify(
        claim=claim_prog_name,
        node=leaf_prog_name,
        sources=prog.sources,
        additional_instruction="Consider variants or abbreviations (e.g., EDHEM). Treat 'Higher Education Management' and 'Executive Doctorate in Higher Education Management' as referring to the same program when clearly indicated."
    )

    # Leaf: degree_type
    leaf_degree_type = evaluator.add_leaf(
        id="degree_type",
        desc="The degree type is Ed.D. (Doctor of Education)",
        parent=specifics,
        critical=True
    )
    claim_degree_type = f"The degree type is '{prog.degree_type}', and it is an Ed.D. (Doctor of Education)."
    await evaluator.verify(
        claim=claim_degree_type,
        node=leaf_degree_type,
        sources=prog.sources,
        additional_instruction="Allow 'EdD' vs 'Ed.D.' formatting variants; confirm it is Doctor of Education."
    )

    # Leaf: distinction_awarded
    leaf_dist = evaluator.add_leaf(
        id="distinction_awarded",
        desc="The degree was awarded with distinction",
        parent=node,
        critical=True
    )
    claim_dist = "The doctoral degree was awarded with distinction."
    await evaluator.verify(
        claim=claim_dist,
        node=leaf_dist,
        sources=prog.sources,
        additional_instruction="The source should explicitly indicate 'with distinction' or equivalent honor for the doctoral degree."
    )

    # Leaf: program_source_url
    leaf_prog_src = evaluator.add_leaf(
        id="program_source_url",
        desc="Provide a reference URL confirming the program details and distinction",
        parent=node,
        critical=True
    )
    claim_prog_src = (
        "These sources confirm that Scott C. Beardsley completed the Executive Doctorate in Higher Education Management at the University of Pennsylvania, "
        "earned an Ed.D. (Doctor of Education), and that the degree was awarded with distinction."
    )
    await evaluator.verify(
        claim=claim_prog_src,
        node=leaf_prog_src,
        sources=prog.sources,
        additional_instruction="At least one URL should jointly or collectively support the program name, degree type, and distinction."
    )


async def build_dissertation_details_subtree(evaluator: Evaluator, parent, data: DoctoralEducationExtraction) -> None:
    node = evaluator.add_parallel(
        id="dissertation_details",
        desc="Verify dissertation information",
        parent=parent,
        critical=True
    )

    diss = data.dissertation_details or DissertationInfo()

    # Leaf: dissertation_source_url (create first so others can depend on it if needed)
    leaf_diss_src = evaluator.add_leaf(
        id="dissertation_source_url",
        desc="Provide a reference URL confirming the year and dissertation title",
        parent=node,
        critical=True
    )
    claim_diss_src = (
        f"The source(s) confirm that the doctoral degree was conferred in {EXPECTED_CONFERRED_YEAR} and that the dissertation title is '{diss.title}'."
    )
    await evaluator.verify(
        claim=claim_diss_src,
        node=leaf_diss_src,
        sources=diss.sources,
        additional_instruction="The URL(s) should explicitly state the conferral year and present the complete dissertation title; small punctuation or casing variations are acceptable."
    )

    # Leaf: year_conferred
    leaf_year = evaluator.add_leaf(
        id="year_conferred",
        desc="The degree was conferred in 2015",
        parent=node,
        critical=True
    )
    claim_year = f"Scott C. Beardsley's doctoral degree was conferred in {EXPECTED_CONFERRED_YEAR}."
    await evaluator.verify(
        claim=claim_year,
        node=leaf_year,
        sources=diss.sources,
        additional_instruction="Confirm the conferral year is 2015. Use the dissertation or program source page that explicitly states the year."
    )

    # Leaf: dissertation_title
    leaf_title = evaluator.add_leaf(
        id="dissertation_title",
        desc="Provide the complete dissertation title",
        parent=node,
        critical=True
    )
    claim_title = f"The complete dissertation title is '{diss.title}'."
    await evaluator.verify(
        claim=claim_title,
        node=leaf_title,
        sources=diss.sources,
        additional_instruction="Verify the title matches the source text; minor punctuation/casing differences are acceptable."
    )


async def build_additional_verifications_subtree(evaluator: Evaluator, parent, data: DoctoralEducationExtraction) -> None:
    node = evaluator.add_parallel(
        id="additional_verifications",
        desc="Verify program structure and post-doctoral recognition",
        parent=parent,
        critical=False
    )

    addi = data.additional_verifications or AdditionalInfo()
    prog = data.doctoral_program or ProgramInfo()

    # Leaf: program_duration
    leaf_duration = evaluator.add_leaf(
        id="program_duration",
        desc="Confirm whether the Executive Doctorate program is described as a two-year program",
        parent=node,
        critical=False
    )
    # Determine boolean intent from extracted flag
    is_two_year = _truthy_flag(addi.is_two_year_program)
    duration_sources = _union_sources(addi.two_year_sources, prog.sources)
    if is_two_year is True:
        claim_duration = "The Executive Doctorate in Higher Education Management program is described as a two-year program."
    elif is_two_year is False:
        claim_duration = "The Executive Doctorate in Higher Education Management program is not described as a two-year program."
    else:
        claim_duration = (
            "The provided sources describe the Executive Doctorate in Higher Education Management program's duration; assess whether it is characterized as a two-year program."
        )
    await evaluator.verify(
        claim=claim_duration,
        node=leaf_duration,
        sources=duration_sources,
        additional_instruction="Check the program overview/details page for explicit wording about a two-year structure; allow reasonable paraphrases."
    )

    # Leaf: alumni_award
    leaf_award = evaluator.add_leaf(
        id="alumni_award",
        desc="Identify any distinguished alumni award received from the University of Pennsylvania after earning the doctorate",
        parent=node,
        critical=False
    )
    if addi.alumni_award_name and addi.alumni_award_year:
        claim_award = (
            f"Scott C. Beardsley received the '{addi.alumni_award_name}' in {addi.alumni_award_year} from the University of Pennsylvania after earning his doctorate."
        )
    elif addi.alumni_award_name:
        claim_award = f"Scott C. Beardsley received the '{addi.alumni_award_name}' from the University of Pennsylvania after earning his doctorate."
    else:
        claim_award = "The provided sources indicate whether Scott C. Beardsley received a distinguished alumni award from the University of Pennsylvania after earning his doctorate."
    await evaluator.verify(
        claim=claim_award,
        node=leaf_award,
        sources=addi.alumni_award_sources,
        additional_instruction="Confirm the award name and year if provided; ensure the awarding institution is University of Pennsylvania (e.g., Penn GSE)."
    )

    # Leaf: additional_source_urls
    leaf_add_sources = evaluator.add_leaf(
        id="additional_source_urls",
        desc="Provide reference URLs for program duration and alumni award information",
        parent=node,
        critical=False
    )
    add_sources_union = _union_sources(addi.two_year_sources, addi.alumni_award_sources)
    claim_add_sources = "The provided source URLs substantiate the program duration description and the alumni award details."
    await evaluator.verify(
        claim=claim_add_sources,
        node=leaf_add_sources,
        sources=add_sources_union,
        additional_instruction="At least one URL should support each subtopic (duration, alumni award); collectively they should cover both."
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
    Evaluate an answer for the UVA president doctoral education task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Complete verification of doctoral education for UVA's current president who previously served as Darden dean",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_doctoral_education(),
        template_class=DoctoralEducationExtraction,
        extraction_name="doctoral_education"
    )

    # Optional: record ground truth anchors for transparency
    evaluator.add_ground_truth({
        "expected_identity_name": EXPECTED_IDENTITY_NAME,
        "expected_doctoral_institution": EXPECTED_INSTITUTION,
        "expected_degree_type": EXPECTED_DEGREE_TYPE,
        "expected_program_name_hints": EXPECTED_PROGRAM_NAME_HINTS,
        "expected_conferred_year": EXPECTED_CONFERRED_YEAR
    }, gt_type="expected_values")

    # Build and verify rubric subtrees (sequential at root)
    await build_identity_subtree(evaluator, root, extracted)
    await build_institution_subtree(evaluator, root, extracted)
    await build_program_details_subtree(evaluator, root, extracted)
    await build_dissertation_details_subtree(evaluator, root, extracted)
    await build_additional_verifications_subtree(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()