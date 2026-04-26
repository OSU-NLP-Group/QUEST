import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_r1_three_universities"
TASK_DESCRIPTION = """
A prospective doctoral student in the behavioral or biomedical sciences is seeking PhD programs at research-intensive universities in California. Identify three universities in California that meet ALL of the following requirements:

1. Geographic Location: The university must be located in California, United States.
2. Research Classification: The university must hold Carnegie Classification R1 status (Doctoral Universities - Highest/Very High Research Activity).
3. Doctoral Program: The university must offer PhD programs in Psychology, Clinical Psychology, or related biomedical/behavioral sciences fields.
4. Dissertation Committee: The university's graduate school must require a minimum of 4 members on doctoral dissertation committees.
5. IRB Training: The university must require completion of the CITI program for human subjects research, with a passing score of 80% or better, as part of IRB compliance.
6. Dissertation Publishing: The university must use ProQuest for electronic dissertation/thesis submission and publishing.

For each of the three universities identified, provide:
- The university name
- URL evidence verifying its California location and R1 Carnegie Classification status
- URL evidence confirming it offers PhD programs in relevant fields
- URL evidence documenting the minimum 4-member dissertation committee requirement, CITI training requirement (with 80%+ passing score), and ProQuest submission requirement.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    name: Optional[str] = None

    # Institutional characteristics evidence
    location_urls: List[str] = Field(default_factory=list)
    carnegie_r1_urls: List[str] = Field(default_factory=list)

    # Doctoral program evidence
    relevant_programs: List[str] = Field(default_factory=list)  # e.g., "PhD in Psychology"
    program_urls: List[str] = Field(default_factory=list)

    # Standard doctoral requirements evidence
    committee_requirement_urls: List[str] = Field(default_factory=list)
    citi_requirement_urls: List[str] = Field(default_factory=list)
    proquest_requirement_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to the first three universities mentioned in the answer that are proposed as meeting the stated criteria.
    For each university, extract the following fields exactly as provided in the answer. Only return URLs that are explicitly present in the answer.

    For each university, extract:
    - name: The university's name (string).
    - location_urls: All URLs used as evidence that the university is located in California, United States.
    - carnegie_r1_urls: All URLs that support or explicitly state the university holds the Carnegie Classification R1 designation. These can include the official Carnegie Classifications page, institutional research pages, Wikipedia infoboxes, etc.
    - relevant_programs: A list of program names/titles (strings) mentioned that are relevant PhD programs (e.g., "PhD in Psychology", "PhD in Clinical Psychology", "PhD in Neuroscience", "PhD in Cognitive Science", "PhD in Biomedical Sciences").
    - program_urls: All URLs that support that the university offers at least one relevant PhD program in psychology, clinical psychology, or closely related biomedical/behavioral sciences.
    - committee_requirement_urls: All URLs that document the graduate school's dissertation committee membership policy (we are looking for policies that require a minimum of 4 members).
    - citi_requirement_urls: All URLs that document an IRB/Human Research Protections policy requiring CITI training for human subjects research, including a passing score threshold of at least 80%.
    - proquest_requirement_urls: All URLs that document that the university uses ProQuest (e.g., ProQuest/UMI, ETD Administrator) for electronic thesis/dissertation submission/publishing.

    Important extraction rules:
    - Only include URLs that are explicitly mentioned in the answer. Do not invent or infer URLs.
    - Include the full URL string. If a URL lacks a protocol, prepend http://
    - If a particular category of URL is not mentioned for a university, return an empty list for that category.
    - If more than three universities are presented in the answer, only extract the first three in the order they appear.
    - If fewer than three are presented, just extract those available.

    Return a JSON object with a single top-level key "universities" which is an array of university objects following the schema above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(u: UniversityItem, idx: int) -> str:
    base = u.name.strip() if (u and u.name) else f"University #{idx + 1}"
    return base


def _merge_urls(*url_lists: List[str]) -> List[str]:
    """Merge multiple URL lists, de-duplicate while preserving order."""
    seen = set()
    merged: List[str] = []
    for urls in url_lists:
        for url in urls or []:
            if url and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification sub-tree for one university                                    #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    index: int,
) -> None:
    uni_name = _safe_name(uni, index)

    # Top-level node for this university (non-critical, to allow partial credit across universities)
    uni_node = evaluator.add_parallel(
        id=f"university_{index+1}",
        desc=f"{uni_name} - verification of all specified criteria",
        parent=parent_node,
        critical=False,
    )

    # -------------------- Institutional Characteristics -------------------- #
    inst_node = evaluator.add_parallel(
        id=f"uni_{index+1}_institutional",
        desc="Basic institutional qualifications including location and research classification",
        parent=uni_node,
        critical=True,
    )

    # Existence of reference URLs for institutional characteristics
    inst_ref_exists = evaluator.add_custom_node(
        result=(bool(uni.location_urls) and bool(uni.carnegie_r1_urls)),
        id=f"uni_{index+1}_institutional_refs",
        desc="URL evidence supporting location and R1 status claims",
        parent=inst_node,
        critical=True,
    )

    # Located in California (verify using location and possibly institutional pages)
    located_leaf = evaluator.add_leaf(
        id=f"uni_{index+1}_located_ca",
        desc="University is physically located in the state of California, United States",
        parent=inst_node,
        critical=True,
    )
    located_claim = f"The university {uni_name} is located in the state of California, United States."
    await evaluator.verify(
        claim=located_claim,
        node=located_leaf,
        sources=_merge_urls(uni.location_urls, uni.carnegie_r1_urls),
        additional_instruction=(
            "Verify that the referenced page(s) clearly indicate the campus is in California (CA). "
            "Accept 'California' or 'CA' in address/contact/footer/about pages. If the institution is a multi-campus system, "
            "the cited page must clearly correspond to the California campus named in the answer."
        ),
    )

    # R1 Carnegie Classification
    r1_leaf = evaluator.add_leaf(
        id=f"uni_{index+1}_carnegie_r1",
        desc="University holds R1 designation (Doctoral Universities - Very High/Highest Research Activity)",
        parent=inst_node,
        critical=True,
    )
    r1_claim = (
        f"The university {uni_name} holds the Carnegie Classification R1 designation "
        f"(Doctoral Universities – Very High/Highest Research Activity)."
    )
    await evaluator.verify(
        claim=r1_claim,
        node=r1_leaf,
        sources=_merge_urls(uni.carnegie_r1_urls, uni.location_urls),
        additional_instruction=(
            "Look for explicit statements like 'R1', 'Very High Research Activity (R1)', or historical phrasing 'Highest Research Activity'. "
            "Authoritative sources include the Carnegie Classifications website or other credible references."
        ),
    )

    # -------------------- Doctoral Program Offering ------------------------ #
    prog_node = evaluator.add_parallel(
        id=f"uni_{index+1}_doctoral_program",
        desc="Availability of relevant doctoral programs in the biomedical or behavioral sciences",
        parent=uni_node,
        critical=True,
    )

    prog_ref_exists = evaluator.add_custom_node(
        result=bool(uni.program_urls),
        id=f"uni_{index+1}_program_refs",
        desc="URL evidence supporting the existence of relevant doctoral programs",
        parent=prog_node,
        critical=True,
    )

    prog_leaf = evaluator.add_leaf(
        id=f"uni_{index+1}_program_offered",
        desc="University offers PhD programs in Psychology, Clinical Psychology, or related biomedical/behavioral sciences",
        parent=prog_node,
        critical=True,
    )
    # Make a descriptive claim mentioning an example program if provided
    exemplar_prog = uni.relevant_programs[0] if uni.relevant_programs else "a relevant PhD program"
    prog_claim = (
        f"The university {uni_name} offers {exemplar_prog}, which is a PhD program in Psychology, Clinical Psychology, "
        f"or a closely related biomedical/behavioral sciences field."
    )
    await evaluator.verify(
        claim=prog_claim,
        node=prog_leaf,
        sources=uni.program_urls,
        additional_instruction=(
            "Accept fields such as Psychology, Clinical Psychology, Neuroscience, Cognitive Science, Biomedical Sciences, "
            "Behavioral Neuroscience, or other clearly related behavioral/biomedical PhD programs. "
            "The page should explicitly indicate a doctoral (Ph.D./PhD) program."
        ),
    )

    # -------------------- Standard Doctoral Requirements ------------------- #
    req_node = evaluator.add_parallel(
        id=f"uni_{index+1}_requirements",
        desc="Institutional compliance with standard graduate school requirements for doctoral education",
        parent=uni_node,
        critical=True,
    )

    req_ref_exists = evaluator.add_custom_node(
        result=(bool(uni.committee_requirement_urls) and bool(uni.citi_requirement_urls) and bool(uni.proquest_requirement_urls)),
        id=f"uni_{index+1}_requirements_refs",
        desc="URL evidence supporting institutional requirements (committee size, CITI, ProQuest)",
        parent=req_node,
        critical=True,
    )

    # Committee size >= 4
    committee_leaf = evaluator.add_leaf(
        id=f"uni_{index+1}_committee_min4",
        desc="University requires minimum of 4 members on doctoral dissertation committees",
        parent=req_node,
        critical=True,
    )
    committee_claim = (
        f"The university {uni_name}'s graduate policy requires doctoral dissertation committees to have at least four members "
        f"(minimum of 4 committee members)."
    )
    await evaluator.verify(
        claim=committee_claim,
        node=committee_leaf,
        sources=uni.committee_requirement_urls,
        additional_instruction=(
            "Accept phrasings like 'at least four', 'minimum of four members', 'chair plus three additional members', or similar. "
            "Departmental or Graduate Division/GPS policies are acceptable as long as they are official."
        ),
    )

    # CITI training with >= 80% passing score
    citi_leaf = evaluator.add_leaf(
        id=f"uni_{index+1}_citi_80",
        desc="University requires CITI program completion with 80%+ passing score for IRB compliance",
        parent=req_node,
        critical=True,
    )
    citi_claim = (
        f"The university {uni_name} requires completion of the CITI human subjects research training with a passing score "
        f"threshold of at least 80% as part of IRB compliance."
    )
    await evaluator.verify(
        claim=citi_claim,
        node=citi_leaf,
        sources=uni.citi_requirement_urls,
        additional_instruction=(
            "Accept any explicit passing threshold that is >= 80% (e.g., 80%, 85%, 90%). "
            "Look for IRB/Human Research Protections/Research Compliance pages that specify CITI training and the passing score."
        ),
    )

    # ProQuest ETD submission/publishing
    proquest_leaf = evaluator.add_leaf(
        id=f"uni_{index+1}_proquest",
        desc="University uses ProQuest for electronic dissertation submission and publishing",
        parent=req_node,
        critical=True,
    )
    proquest_claim = (
        f"The university {uni_name} requires or uses ProQuest (e.g., ProQuest/UMI, ProQuest ETD Administrator) for "
        f"electronic thesis/dissertation submission and publishing."
    )
    await evaluator.verify(
        claim=proquest_claim,
        node=proquest_leaf,
        sources=uni.proquest_requirement_urls,
        additional_instruction=(
            "Look for language such as 'submit via ProQuest', 'ProQuest ETD Administrator', 'ProQuest/UMI', or official graduate division instructions "
            "that direct students to ProQuest for ETD submission."
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
    Evaluate an answer for identifying three California R1 universities with compliant doctoral programs.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Universities evaluated independently
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

    # Extract structured university info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Normalize to exactly 3 entries (pad with empty placeholders if needed)
    universities: List[UniversityItem] = list(extraction.universities[:3])
    while len(universities) < 3:
        universities.append(UniversityItem())

    # Root grouping node reflecting the rubric root
    root_group = evaluator.add_parallel(
        id="three_california_r1_universities",
        desc="Three California R1 Universities with Compliant Doctoral Programs",
        parent=root,
        critical=False,
    )

    # Build and verify each university subtree
    for idx in range(3):
        await verify_university(
            evaluator=evaluator,
            parent_node=root_group,
            uni=universities[idx],
            index=idx,
        )

    # Optionally record custom info
    evaluator.add_custom_info(
        info={
            "requested_universities": 3,
            "extracted_universities_count": len(extraction.universities),
            "evaluated_universities": 3,
        },
        info_type="stats",
        info_name="evaluation_stats",
    )

    return evaluator.get_summary()