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
TASK_ID = "academic_admin_identification"
TASK_DESCRIPTION = """
An academic administrator was born in Sri Lanka and later pursued higher education in the United States. This person earned a Bachelor of Science degree in Astronomy and Physics from Yale University in 1994, followed by a Ph.D. in Astronomy from Harvard University in 2000. After completing doctoral studies, this individual embarked on an academic career that eventually led to serving as the Harold Tanner Dean of the College of Arts and Sciences at Cornell University, a position that began on September 1, 2018. Following this role, the person was appointed as Provost of Johns Hopkins University, starting October 15, 2023. Most recently, this administrator was named as the 10th President of the California Institute of Technology, with the term set to begin on July 1, 2026. Who is this person?
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EducationEntry(BaseModel):
    degree: Optional[str] = None
    field: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PositionEntry(BaseModel):
    title: Optional[str] = None
    institution: Optional[str] = None
    start_date: Optional[str] = None
    ordinal: Optional[str] = None  # e.g., "10th" for "10th President"
    sources: List[str] = Field(default_factory=list)


class PersonExtraction(BaseModel):
    name: Optional[str] = None

    birth_country: Optional[str] = None
    birth_sources: List[str] = Field(default_factory=list)

    yale: Optional[EducationEntry] = None
    harvard: Optional[EducationEntry] = None

    cornell: Optional[PositionEntry] = None
    johns_hopkins: Optional[PositionEntry] = None
    caltech: Optional[PositionEntry] = None

    # Optionally, collect any other URLs cited in the answer
    other_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_person_info() -> str:
    return """
    Extract structured information about the person identified in the answer. Only extract information explicitly present in the answer text. Do not infer or invent anything.

    Required fields:
    - name: The full name of the person identified as the answer to "Who is this person?"
    - birth_country: The country where the person was born (e.g., "Sri Lanka"), if stated.
    - birth_sources: A list of URLs explicitly cited in the answer that directly support the birth country claim. If none are cited, return an empty list.

    Education entries (extract if present in the answer):
    - yale:
        - degree: e.g., "Bachelor of Science", "B.S.", etc.
        - field: e.g., "Astronomy and Physics"
        - institution: Should be "Yale University" if present.
        - year: e.g., "1994"
        - sources: URLs explicitly cited in the answer that support this Yale undergraduate credential. Return an empty list if none.
    - harvard:
        - degree: e.g., "Ph.D."
        - field: e.g., "Astronomy"
        - institution: Should be "Harvard University" if present.
        - year: e.g., "2000"
        - sources: URLs explicitly cited in the answer that support this Harvard doctoral credential. Return an empty list if none.

    Career positions (extract if present in the answer):
    - cornell:
        - title: e.g., "Harold Tanner Dean of the College of Arts and Sciences"
        - institution: Should be "Cornell University" if present.
        - start_date: e.g., "September 1, 2018" or similar
        - ordinal: leave null
        - sources: URLs explicitly cited in the answer that support this role and start date. Return an empty list if none.
    - johns_hopkins:
        - title: e.g., "Provost"
        - institution: Should be "Johns Hopkins University" if present.
        - start_date: e.g., "October 15, 2023" or similar
        - ordinal: leave null
        - sources: URLs explicitly cited in the answer that support this role and start date. Return an empty list if none.
    - caltech:
        - title: e.g., "President"
        - institution: Should be "California Institute of Technology" if present.
        - start_date: e.g., "July 1, 2026" or similar
        - ordinal: e.g., "10th" if explicitly mentioned (otherwise null)
        - sources: URLs explicitly cited in the answer that support the presidency appointment, ordinal, and start date. Return an empty list if none.

    - other_sources: Any other URLs cited in the answer (not already listed in the above source lists).

    Notes:
    - For all 'sources' fields, only include URLs that are explicitly present in the answer (including markdown links). If none are provided, return an empty list.
    - If any subfield is not present in the answer, return null for that subfield (or empty list for sources).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def person_subject(extracted: PersonExtraction) -> str:
    if extracted and extracted.name and extracted.name.strip():
        return extracted.name.strip()
    return "the person identified in the answer"


def non_empty_sources(sources: Optional[List[str]]) -> bool:
    return bool(sources) and len(sources) > 0


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, root, info: PersonExtraction) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    Note: The JSON marks Person_Identification as critical but includes non-critical children,
    which violates the framework constraint that critical parents must have all-critical children.
    Therefore, we set the top-level Person_Identification node to non-critical to preserve mixed child criticalities.
    """

    # Top-level container corresponding to "Person_Identification" (parallel, non-critical due to framework constraint)
    person_node = evaluator.add_parallel(
        id="Person_Identification",
        desc="Identify the academic administrator who meets all specified criteria",
        parent=root,
        critical=False
    )

    # --------------------------- Birth Country ---------------------------- #
    # Gating node (non-critical) to require sources for birth claim
    birth_gate = evaluator.add_custom_node(
        result=(info is not None and info.birth_country is not None and "sri lanka" in info.birth_country.lower() and non_empty_sources(info.birth_sources)),
        id="Birth_Country_sources_present",
        desc="Sources provided for the birth country claim (Sri Lanka)",
        parent=person_node,
        critical=False
    )

    birth_leaf = evaluator.add_leaf(
        id="Birth_Country",
        desc="The person was born in Sri Lanka",
        parent=person_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{person_subject(info)} was born in Sri Lanka.",
        node=birth_leaf,
        sources=info.birth_sources if info else [],
        additional_instruction="Verify that the cited source(s) explicitly indicate Sri Lanka as the person's birth country. Allow minor variations in phrasing.",
        extra_prerequisites=[birth_gate]
    )

    # ----------------------- Educational Background ---------------------- #
    edu_node = evaluator.add_parallel(
        id="Educational_Background",
        desc="Verify the person's educational credentials from Yale and Harvard",
        parent=person_node,
        critical=False
    )

    # Yale Undergraduate
    yale_sources = info.yale.sources if (info and info.yale) else []
    yale_gate = evaluator.add_custom_node(
        result=non_empty_sources(yale_sources),
        id="Yale_Undergraduate_sources_present",
        desc="Sources provided for Yale undergraduate credential",
        parent=edu_node,
        critical=False
    )
    yale_leaf = evaluator.add_leaf(
        id="Yale_Undergraduate",
        desc="The person earned a B.S. in Astronomy and Physics from Yale University, graduating in 1994",
        parent=edu_node,
        critical=True
    )
    yale_claim = (
        f"{person_subject(info)} earned a Bachelor of Science (B.S.) in Astronomy and Physics from Yale University in 1994."
    )
    await evaluator.verify(
        claim=yale_claim,
        node=yale_leaf,
        sources=yale_sources,
        additional_instruction="Confirm degree level (Bachelor/B.S.), field (Astronomy and Physics), institution (Yale University), and year (1994). Allow reasonable synonyms like 'B.Sc.' for 'B.S.' and minor phrasing variations.",
        extra_prerequisites=[yale_gate]
    )

    # Harvard Doctorate
    harvard_sources = info.harvard.sources if (info and info.harvard) else []
    harvard_gate = evaluator.add_custom_node(
        result=non_empty_sources(harvard_sources),
        id="Harvard_Doctorate_sources_present",
        desc="Sources provided for Harvard doctoral credential",
        parent=edu_node,
        critical=False
    )
    harvard_leaf = evaluator.add_leaf(
        id="Harvard_Doctorate",
        desc="The person earned a Ph.D. in Astronomy from Harvard University, graduating in 2000",
        parent=edu_node,
        critical=True
    )
    harvard_claim = (
        f"{person_subject(info)} earned a Ph.D. in Astronomy from Harvard University in 2000."
    )
    await evaluator.verify(
        claim=harvard_claim,
        node=harvard_leaf,
        sources=harvard_sources,
        additional_instruction="Confirm doctoral degree (Ph.D.), field (Astronomy), institution (Harvard University), and year (2000). Allow minor formatting variations.",
        extra_prerequisites=[harvard_gate]
    )

    # -------------------------- Career Positions ------------------------- #
    career_node = evaluator.add_parallel(
        id="Career_Positions",
        desc="Verify the person's career positions at Cornell, Johns Hopkins, and Caltech",
        parent=person_node,
        critical=False
    )

    # Cornell Deanship
    cornell_sources = info.cornell.sources if (info and info.cornell) else []
    cornell_gate = evaluator.add_custom_node(
        result=non_empty_sources(cornell_sources),
        id="Cornell_Deanship_sources_present",
        desc="Sources provided for Cornell deanship",
        parent=career_node,
        critical=False
    )
    cornell_leaf = evaluator.add_leaf(
        id="Cornell_Deanship",
        desc="The person served as Harold Tanner Dean of the College of Arts and Sciences at Cornell University, starting September 1, 2018",
        parent=career_node,
        critical=True
    )
    cornell_claim = (
        f"{person_subject(info)} served as the Harold Tanner Dean of the College of Arts and Sciences at Cornell University, "
        f"starting on September 1, 2018."
    )
    await evaluator.verify(
        claim=cornell_claim,
        node=cornell_leaf,
        sources=cornell_sources,
        additional_instruction="Confirm the exact title and the start date. If a source states 'September 2018' without a day, consider it consistent with September 1, 2018, unless contradicted.",
        extra_prerequisites=[cornell_gate]
    )

    # Johns Hopkins Provost
    jhu_sources = info.johns_hopkins.sources if (info and info.johns_hopkins) else []
    jhu_gate = evaluator.add_custom_node(
        result=non_empty_sources(jhu_sources),
        id="Johns_Hopkins_Provost_sources_present",
        desc="Sources provided for Johns Hopkins provost role",
        parent=career_node,
        critical=False
    )
    jhu_leaf = evaluator.add_leaf(
        id="Johns_Hopkins_Provost",
        desc="The person served as Provost of Johns Hopkins University, starting October 15, 2023",
        parent=career_node,
        critical=True
    )
    jhu_claim = (
        f"{person_subject(info)} was appointed Provost of Johns Hopkins University, starting on October 15, 2023."
    )
    await evaluator.verify(
        claim=jhu_claim,
        node=jhu_leaf,
        sources=jhu_sources,
        additional_instruction="Confirm appointment as Provost and the start date of October 15, 2023. Accept 'effective' date phrasing.",
        extra_prerequisites=[jhu_gate]
    )

    # Caltech Presidency
    caltech_sources = info.caltech.sources if (info and info.caltech) else []
    caltech_gate = evaluator.add_custom_node(
        result=non_empty_sources(caltech_sources),
        id="Caltech_Presidency_sources_present",
        desc="Sources provided for Caltech presidency",
        parent=career_node,
        critical=False
    )
    caltech_leaf = evaluator.add_leaf(
        id="Caltech_Presidency",
        desc="The person was appointed as the 10th President of the California Institute of Technology, with the term beginning July 1, 2026",
        parent=career_node,
        critical=True
    )
    # Compose ordinal part if provided, else default to '10th'
    ordinal_text = None
    if info and info.caltech and info.caltech.ordinal:
        ordinal_text = info.caltech.ordinal.strip()
    ordinal_text = ordinal_text or "10th"
    caltech_claim = (
        f"{person_subject(info)} was named/appointed the {ordinal_text} President of the California Institute of Technology, "
        f"with the term beginning July 1, 2026."
    )
    await evaluator.verify(
        claim=caltech_claim,
        node=caltech_leaf,
        sources=caltech_sources,
        additional_instruction="Confirm both the presidency at Caltech (including the ordinal, e.g., '10th President') and that the term begins July 1, 2026. Allow reasonable phrasing variations (e.g., 'named', 'appointed').",
        extra_prerequisites=[caltech_gate]
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
    Evaluate an answer for the academic administrator identification task.
    """
    # Initialize evaluator (root is always non-critical; choose parallel aggregation)
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
        default_model=model
    )

    # Extract structured information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_person_info(),
        template_class=PersonExtraction,
        extraction_name="person_extraction"
    )

    # Optional: record expected ground truth hint (not used for scoring)
    evaluator.add_ground_truth({
        "expected_traits": {
            "birth_country": "Sri Lanka",
            "yale_year": "1994",
            "harvard_year": "2000",
            "cornell_start": "September 1, 2018",
            "jhu_start": "October 15, 2023",
            "caltech_start": "July 1, 2026",
            "caltech_ordinal": "10th"
        }
    })

    # Build verification tree and verify claims
    await build_and_verify(evaluator, root, extracted_info)

    # Return structured summary
    return evaluator.get_summary()