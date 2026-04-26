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
TASK_ID = "unc_iprc_director_education"
TASK_DESCRIPTION = (
    "A person was appointed as the director of the University of North Carolina Injury Prevention Research Center "
    "(UNC IPRC), with the appointment becoming effective on September 1, 2024. Prior to this permanent appointment, "
    "this individual served as the interim director of UNC IPRC starting in February 2024. What is the full name of "
    "this person? Additionally, provide details about their educational background, including: (1) the institution "
    "where they obtained their undergraduate degree, the field of study, and any additional certificates earned during "
    "undergraduate studies; and (2) the institution(s) where they obtained their graduate degrees, specifying both "
    "their Master's and Doctoral degree programs."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class IdentityAndSources(BaseModel):
    full_name: Optional[str] = None
    timeline_sources: List[str] = Field(default_factory=list)
    bio_sources: List[str] = Field(default_factory=list)
    other_sources: List[str] = Field(default_factory=list)


class UndergraduateDetails(BaseModel):
    institution: Optional[str] = None
    field_of_study: Optional[str] = None
    certificate: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class MastersDetails(BaseModel):
    institution: Optional[str] = None
    program: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DoctoralDetails(BaseModel):
    institution: Optional[str] = None
    program: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_identity_sources() -> str:
    return """
    Extract the identity of the person and any URLs cited in the answer that support the stated roles/timeline and biography.
    Return a JSON object with:
    - full_name: The person's full name (at least first and last name) as stated in the answer.
    - timeline_sources: Array of URLs cited in the answer that support one or more of these facts:
        • appointed as UNC Injury Prevention Research Center (UNC IPRC) director effective September 1, 2024
        • served as interim director of UNC IPRC starting in February 2024
        • was associate director of UNC IPRC since 2017
    - bio_sources: Array of URLs cited in the answer that support one or more of these facts:
        • associate professor in the Department of Health Behavior at the UNC Gillings School of Global Public Health
        • serves as the University of North Carolina's Chair of the Faculty
        • joined the Gillings faculty in 2008
        • research focuses on the primary and secondary prevention of gender-based violence
        • under interim leadership, UNC IPRC renewed its funding as a CDC Injury Control Research Center (ICRC)
    - other_sources: any additional URLs cited in the answer that may be relevant to these facts but not clearly categorized above.

    IMPORTANT:
    • Extract only URLs explicitly present in the answer (including ones in markdown).
    • Do not fabricate URLs.
    """


def prompt_extract_undergrad_details() -> str:
    return """
    Extract the undergraduate education details for the identified person from the answer.
    Return a JSON object with:
    - institution: The institution where the undergraduate degree was obtained (string or null).
    - field_of_study: The undergraduate major/field of study (string or null).
    - certificate: Any additional undergraduate certificate mentioned (string or null). If none is stated, return null.
    - sources: Array of URLs cited in the answer that support any of the above undergraduate details.
    """


def prompt_extract_masters_details() -> str:
    return """
    Extract the Master's degree details from the answer.
    Return a JSON object with:
    - institution: The institution where the Master's degree was obtained (string or null).
    - program: The Master's degree program/field (string or null).
    - sources: Array of URLs cited in the answer that support the Master's degree details.
    """


def prompt_extract_doctoral_details() -> str:
    return """
    Extract the Doctoral degree details from the answer.
    Return a JSON object with:
    - institution: The institution where the Doctoral degree was obtained (string or null).
    - program: The Doctoral degree program/field (string or null).
    - sources: Array of URLs cited in the answer that support the Doctoral degree details.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(*lists: List[str]) -> List[str]:
    seen = set()
    combined = []
    for lst in lists:
        for url in lst or []:
            if not url:
                continue
            key = url.strip()
            if key and key not in seen:
                seen.add(key)
                combined.append(key)
    return combined


def safe_str(x: Optional[str]) -> str:
    return x if isinstance(x, str) else ""


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_person_identification_tree(
    evaluator: Evaluator,
    parent_node,
    identity: IdentityAndSources,
) -> None:
    """
    Build and verify the 'person_identification' subtree.
    """
    name = safe_str(identity.full_name)
    # We will use all related identity sources to improve the chance the verifier finds support
    timeline_src = identity.timeline_sources
    bio_src = identity.bio_sources
    other_src = identity.other_sources
    all_id_sources = combine_sources(timeline_src, bio_src, other_src)

    # person_identification (critical, parallel)
    pid_node = evaluator.add_parallel(
        id="person_identification",
        desc="Identify the correct individual described by the role/timeline/biographical constraints.",
        parent=parent_node,
        critical=True,
    )

    # full_name_provided (critical leaf via custom node)
    evaluator.add_custom_node(
        result=bool(name.strip()),
        id="full_name_provided",
        desc="Response includes the person’s full name (at least first and last name).",
        parent=pid_node,
        critical=True
    )

    # role_and_timeline_constraints_match (critical, parallel)
    role_node = evaluator.add_parallel(
        id="role_and_timeline_constraints_match",
        desc="The named person matches the specified director/interim/associate-director timeline constraints.",
        parent=pid_node,
        critical=True
    )

    # director_appointment_effective_date_match (critical leaf)
    director_leaf = evaluator.add_leaf(
        id="director_appointment_effective_date_match",
        desc="The named person is the UNC IPRC director whose appointment is effective September 1, 2024.",
        parent=role_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} was appointed as the Director of the UNC Injury Prevention Research Center (UNC IPRC), "
              f"with the appointment effective September 1, 2024.",
        node=director_leaf,
        sources=combine_sources(timeline_src, all_id_sources),
        additional_instruction="Verify that the page explicitly states that the person named was appointed "
                               "as UNC IPRC Director effective September 1, 2024. Allow minor name variations."
    )

    # interim_director_start_match (critical leaf)
    interim_leaf = evaluator.add_leaf(
        id="interim_director_start_match",
        desc="The named person served as interim director of UNC IPRC starting in February 2024.",
        parent=role_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} served as interim director of the UNC IPRC starting in February 2024.",
        node=interim_leaf,
        sources=combine_sources(timeline_src, all_id_sources),
        additional_instruction="Check for language indicating they began an interim director role in February 2024; "
                               "phrases like 'starting in February 2024' or 'since February 2024' are acceptable."
    )

    # associate_director_since_2017_match (critical leaf)
    assoc_dir_leaf = evaluator.add_leaf(
        id="associate_director_since_2017_match",
        desc="Before the interim director role, the named person held the position of associate director of UNC IPRC since 2017.",
        parent=role_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Before becoming interim director, {name} had been an associate director of the UNC IPRC since 2017.",
        node=assoc_dir_leaf,
        sources=combine_sources(timeline_src, all_id_sources),
        additional_instruction="Look for explicit mention of associate director role starting in 2017 (e.g., 'since 2017')."
    )

    # biographical_constraints_match (critical, parallel)
    bio_node = evaluator.add_parallel(
        id="biographical_constraints_match",
        desc="The named person matches the specified non-education biographical/professional constraints.",
        parent=pid_node,
        critical=True
    )

    # associate_professor_match
    assoc_prof_leaf = evaluator.add_leaf(
        id="associate_professor_match",
        desc="The named person is an associate professor in the Department of Health Behavior at the UNC Gillings School of Global Public Health.",
        parent=bio_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} is an associate professor in the Department of Health Behavior at the UNC Gillings School of Global Public Health.",
        node=assoc_prof_leaf,
        sources=combine_sources(bio_src, all_id_sources),
        additional_instruction="Confirm the exact department 'Health Behavior' at 'UNC Gillings School of Global Public Health'."
    )

    # chair_of_faculty_match
    chair_leaf = evaluator.add_leaf(
        id="chair_of_faculty_match",
        desc="The named person serves as the University of North Carolina's Chair of the Faculty.",
        parent=bio_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} serves as the University of North Carolina's Chair of the Faculty.",
        node=chair_leaf,
        sources=combine_sources(bio_src, all_id_sources),
        additional_instruction="This refers to the campus-wide Chair of the Faculty role at UNC–Chapel Hill, not a department chair."
    )

    # joined_faculty_2008_match
    joined_leaf = evaluator.add_leaf(
        id="joined_faculty_2008_match",
        desc="The named person joined the faculty at the UNC Gillings School of Global Public Health in 2008.",
        parent=bio_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} joined the faculty at the UNC Gillings School of Global Public Health in 2008.",
        node=joined_leaf,
        sources=combine_sources(bio_src, all_id_sources),
        additional_instruction="Look for phrasing like 'joined the Gillings faculty in 2008' or equivalent."
    )

    # research_specialization_match
    research_leaf = evaluator.add_leaf(
        id="research_specialization_match",
        desc="The named person’s research specialization focuses on the primary and secondary prevention of gender-based violence.",
        parent=bio_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name}'s research focuses on the primary and secondary prevention of gender-based violence.",
        node=research_leaf,
        sources=combine_sources(bio_src, all_id_sources),
        additional_instruction="Allow synonyms such as 'intimate partner violence', 'sexual violence', "
                               "and phrasing indicating prevention research on gender-based violence."
    )

    # funding_renewal_under_interim_leadership_match
    renewal_leaf = evaluator.add_leaf(
        id="funding_renewal_under_interim_leadership_match",
        desc="Under the named person’s interim leadership, UNC IPRC renewed its funding as a CDC Injury Control Research Center (ICRC).",
        parent=bio_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Under {name}'s interim leadership, the UNC IPRC renewed its funding as a CDC Injury Control Research Center (ICRC).",
        node=renewal_leaf,
        sources=combine_sources(timeline_src, bio_src, all_id_sources),
        additional_instruction="Look for explicit linkage between the person's interim leadership and UNC IPRC's CDC ICRC funding renewal."
    )


async def build_education_tree(
    evaluator: Evaluator,
    parent_node,
    identity: IdentityAndSources,
    undergrad: UndergraduateDetails,
    masters: MastersDetails,
    doctoral: DoctoralDetails,
) -> None:
    """
    Build and verify the 'educational_background' subtree.
    """
    name = safe_str(identity.full_name)

    # In case some specific sources are missing, fall back to identity sources
    id_sources_union = combine_sources(identity.timeline_sources, identity.bio_sources, identity.other_sources)

    edu_node = evaluator.add_parallel(
        id="educational_background",
        desc="Provide the required undergraduate and graduate education details for the identified person.",
        parent=parent_node,
        critical=True
    )

    # Undergraduate details (critical, parallel)
    ug_node = evaluator.add_parallel(
        id="undergraduate_details",
        desc="Undergraduate education details: institution, field of study, and any undergraduate certificate.",
        parent=edu_node,
        critical=True
    )

    ug_sources = combine_sources(undergrad.sources, id_sources_union)

    # undergrad_institution
    ug_inst_leaf = evaluator.add_leaf(
        id="undergrad_institution",
        desc="Correctly states the institution where the undergraduate degree was obtained.",
        parent=ug_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} obtained an undergraduate degree from {safe_str(undergrad.institution)}.",
        node=ug_inst_leaf,
        sources=ug_sources,
        additional_instruction="Verify the undergraduate institution. If the institution is missing or incorrect, mark as not supported."
    )

    # undergrad_field_of_study
    ug_field_leaf = evaluator.add_leaf(
        id="undergrad_field_of_study",
        desc="Correctly states the undergraduate field of study/major.",
        parent=ug_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name}'s undergraduate major/field of study was {safe_str(undergrad.field_of_study)}.",
        node=ug_field_leaf,
        sources=ug_sources,
        additional_instruction="Verify the undergraduate major/field from the provided source(s)."
    )

    # undergrad_certificate
    ug_cert_leaf = evaluator.add_leaf(
        id="undergrad_certificate",
        desc="Correctly states any additional certificate earned during undergraduate studies.",
        parent=ug_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"During undergraduate studies, {name} earned a certificate in {safe_str(undergrad.certificate)}.",
        node=ug_cert_leaf,
        sources=ug_sources,
        additional_instruction="If the answer claims an undergraduate certificate, verify it. "
                               "If no certificate was mentioned in the answer, this claim should not be supported."
    )

    # Graduate details (critical, parallel)
    grad_node = evaluator.add_parallel(
        id="graduate_details",
        desc="Graduate education details: institution(s) and programs for both the Master's and Doctoral degrees.",
        parent=edu_node,
        critical=True
    )

    # Master's
    ms_sources = combine_sources(masters.sources, id_sources_union)

    ms_inst_leaf = evaluator.add_leaf(
        id="masters_institution",
        desc="Correctly states the institution where the Master's degree was obtained.",
        parent=grad_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} obtained a Master's degree from {safe_str(masters.institution)}.",
        node=ms_inst_leaf,
        sources=ms_sources,
        additional_instruction="Verify the Master's degree institution."
    )

    ms_prog_leaf = evaluator.add_leaf(
        id="masters_program",
        desc="Correctly states the Master's degree program.",
        parent=grad_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name}'s Master's program/field was {safe_str(masters.program)}.",
        node=ms_prog_leaf,
        sources=ms_sources,
        additional_instruction="Verify the Master's program/field (e.g., MPH in Health Behavior)."
    )

    # Doctoral
    phd_sources = combine_sources(doctoral.sources, id_sources_union)

    phd_inst_leaf = evaluator.add_leaf(
        id="doctoral_institution",
        desc="Correctly states the institution where the doctoral degree was obtained.",
        parent=grad_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} obtained a doctoral degree from {safe_str(doctoral.institution)}.",
        node=phd_inst_leaf,
        sources=phd_sources,
        additional_instruction="Verify the doctoral institution."
    )

    phd_prog_leaf = evaluator.add_leaf(
        id="doctoral_program",
        desc="Correctly states the doctoral degree program/field.",
        parent=grad_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name}'s doctoral program/field was {safe_str(doctoral.program)}.",
        node=phd_prog_leaf,
        sources=phd_sources,
        additional_instruction="Verify the doctoral program/field (e.g., PhD in Health Behavior)."
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
    Evaluate an answer for the UNC IPRC director identification and education details task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # High-level sequential: identify person first, then verify education details
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

    # Top-level critical sequential node to mirror rubric root criticality
    task_root = evaluator.add_sequential(
        id="task_root",
        desc="Provide the full name of the person described and the requested undergraduate and graduate education details.",
        parent=root,
        critical=True
    )

    # 1) Extract identity + sources
    identity = await evaluator.extract(
        prompt=prompt_extract_identity_sources(),
        template_class=IdentityAndSources,
        extraction_name="identity_and_sources"
    )

    # 2) Extract education pieces
    undergrad, masters, doctoral = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_undergrad_details(),
            template_class=UndergraduateDetails,
            extraction_name="undergraduate_details"
        ),
        evaluator.extract(
            prompt=prompt_extract_masters_details(),
            template_class=MastersDetails,
            extraction_name="masters_details"
        ),
        evaluator.extract(
            prompt=prompt_extract_doctoral_details(),
            template_class=DoctoralDetails,
            extraction_name="doctoral_details"
        ),
    )

    # Optional: record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "extracted_name": identity.full_name,
            "counts": {
                "timeline_sources": len(identity.timeline_sources or []),
                "bio_sources": len(identity.bio_sources or []),
                "other_sources": len(identity.other_sources or []),
                "ug_sources": len(undergrad.sources or []),
                "ms_sources": len(masters.sources or []),
                "phd_sources": len(doctoral.sources or []),
            }
        },
        info_type="extraction_metadata",
        info_name="extraction_overview"
    )

    # Build and verify the person identification subtree
    await build_person_identification_tree(
        evaluator=evaluator,
        parent_node=task_root,
        identity=identity
    )

    # Build and verify the educational background subtree
    await build_education_tree(
        evaluator=evaluator,
        parent_node=task_root,
        identity=identity,
        undergrad=undergrad,
        masters=masters,
        doctoral=doctoral
    )

    return evaluator.get_summary()