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
TASK_ID = "recital_hall_capacity_chain"
TASK_DESCRIPTION = (
    "A pianist won a major international piano competition between 2020 and 2025, becoming the youngest winner in that "
    "competition's history at age 18. This pianist's primary teacher is a concert pianist who later joined the faculty "
    "at a major U.S. conservatory. The teacher graduated from the New England Conservatory of Music with an Artist "
    "Diploma in 2004. In 2008, the teacher released a debut recording album on the Honens label, featuring Liszt "
    "transcriptions and études. On October 8, 2009, the teacher made a debut performance at a specific recital hall within "
    "Carnegie Hall in New York City. What is the seating capacity of this recital hall?"
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class PianistExtraction(BaseModel):
    pianist_name: Optional[str] = None
    competition_name: Optional[str] = None
    competition_year: Optional[str] = None
    youngest_winner_age: Optional[str] = None
    pianist_sources: List[str] = Field(default_factory=list)


class TeacherExtraction(BaseModel):
    teacher_name: Optional[str] = None
    primary_teacher_statement: Optional[str] = None
    conservatory_faculty_name: Optional[str] = None
    concert_pianist_statement: Optional[str] = None
    teacher_sources: List[str] = Field(default_factory=list)


class EducationExtraction(BaseModel):
    nec_credential: Optional[str] = None
    nec_year: Optional[str] = None
    education_sources: List[str] = Field(default_factory=list)


class RecordingExtraction(BaseModel):
    album_title: Optional[str] = None
    album_year: Optional[str] = None
    label: Optional[str] = None
    features_liszt_transcriptions_and_etudes: Optional[bool] = None
    recording_sources: List[str] = Field(default_factory=list)


class CarnegieDebutExtraction(BaseModel):
    debut_date: Optional[str] = None  # e.g., "October 8, 2009"
    specific_recital_hall_name: Optional[str] = None  # e.g., "Weill Recital Hall"
    venue_sources: List[str] = Field(default_factory=list)


class CapacityExtraction(BaseModel):
    seating_capacity: Optional[str] = None
    capacity_sources: List[str] = Field(default_factory=list)


class FullExtraction(BaseModel):
    pianist: Optional[PianistExtraction] = None
    teacher: Optional[TeacherExtraction] = None
    education: Optional[EducationExtraction] = None
    recording: Optional[RecordingExtraction] = None
    carnegie_debut: Optional[CarnegieDebutExtraction] = None
    capacity: Optional[CapacityExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_core() -> str:
    return """
    Extract the specific entities and facts the answer claims, along with all URLs cited to support each part. 
    Return a single JSON object with the following structure (use null where unknown/missing, and empty arrays for missing URL lists):

    {
      "pianist": {
        "pianist_name": string|null,
        "competition_name": string|null,
        "competition_year": string|null,
        "youngest_winner_age": string|null,
        "pianist_sources": string[]    // URLs that support the pianist/competition/age claims
      },
      "teacher": {
        "teacher_name": string|null,
        "primary_teacher_statement": string|null,         // Phrase in the answer that indicates 'primary teacher'
        "conservatory_faculty_name": string|null,         // Name of the major U.S. conservatory the teacher joined
        "concert_pianist_statement": string|null,         // Phrase indicating the person is a 'concert pianist'
        "teacher_sources": string[]                       // URLs that support the teacher relationship and credential/faculty claims
      },
      "education": {
        "nec_credential": string|null,    // e.g., "Artist Diploma"
        "nec_year": string|null,          // e.g., "2004"
        "education_sources": string[]     // URLs that support the NEC credential/year
      },
      "recording": {
        "album_title": string|null, 
        "album_year": string|null,        // e.g., "2008"
        "label": string|null,             // e.g., "Honens"
        "features_liszt_transcriptions_and_etudes": boolean|null,  // true if the debut album features Liszt transcriptions and études
        "recording_sources": string[]     // URLs that support debut album details (year, label, content)
      },
      "carnegie_debut": {
        "debut_date": string|null,                // e.g., "October 8, 2009" (if different format provided, keep as-is)
        "specific_recital_hall_name": string|null,// e.g., "Weill Recital Hall" or "Zankel Hall"
        "venue_sources": string[]                 // URLs that support the Carnegie Hall debut date and hall
      },
      "capacity": {
        "seating_capacity": string|null,          // e.g., "268", "599", "2804", etc. Keep as in the answer (string).
        "capacity_sources": string[]              // URLs specifically for seating capacity (if provided). If not, leave empty.
      }
    }

    Rules:
    - Extract only what the answer explicitly claims. Do not invent any data.
    - Always extract the actual URLs (from plain text or markdown links). If none are provided for a section, return an empty array.
    - Preserve the answer's phrasing for dates/names when possible. Do not normalize unless the answer clearly normalizes itself.
    - Booleans: if the answer clearly states that the debut album features Liszt transcriptions and études, set true; if it clearly denies, set false; if unclear, set null.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(xs: Optional[List[str]]) -> List[str]:
    return xs if isinstance(xs, list) else []


def _merge_sources(*lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        for u in _safe_list(lst):
            if isinstance(u, str) and u.strip():
                merged.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in merged:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_identify_pianist_nodes(evaluator: Evaluator, parent, extracted: FullExtraction) -> None:
    pianist = extracted.pianist or PianistExtraction()
    pianist_node = evaluator.add_parallel(
        id="Identify_Pianist",
        desc="Identify the pianist who satisfies the competition constraints",
        parent=parent,
        critical=True
    )

    # Leaf: Won_Competition_2020_2025
    won_node = evaluator.add_leaf(
        id="Won_Competition_2020_2025",
        desc="The pianist won a major international piano competition between 2020 and 2025",
        parent=pianist_node,
        critical=True
    )
    claim_won = (
        f"{pianist.pianist_name or 'The pianist'} won the {pianist.competition_name or 'named competition'} "
        f"in {pianist.competition_year or 'the stated year'}, which falls between 2020 and 2025 (inclusive)."
    )
    await evaluator.verify(
        claim=claim_won,
        node=won_node,
        sources=_safe_list(pianist.pianist_sources),
        additional_instruction="Verify that the named person indeed won the specified competition in the stated year. "
                               "It is acceptable to infer that a year like 2022 lies between 2020 and 2025."
    )

    # Leaf: Youngest_Winner_Age_18
    youngest_node = evaluator.add_leaf(
        id="Youngest_Winner_Age_18",
        desc="The pianist became the youngest winner in that competition's history at age 18",
        parent=pianist_node,
        critical=True
    )
    claim_youngest = (
        f"{pianist.pianist_name or 'The pianist'} became the youngest winner in the history of "
        f"{pianist.competition_name or 'the competition'} at age 18."
    )
    await evaluator.verify(
        claim=claim_youngest,
        node=youngest_node,
        sources=_safe_list(pianist.pianist_sources),
        additional_instruction="Treat phrasings like '18-year-old' as equivalent to 'age 18'. Minor wording variations are acceptable."
    )


async def build_identify_teacher_nodes(evaluator: Evaluator, parent, extracted: FullExtraction) -> None:
    pianist = extracted.pianist or PianistExtraction()
    teacher = extracted.teacher or TeacherExtraction()

    teacher_node = evaluator.add_parallel(
        id="Identify_Teacher",
        desc="Identify the pianist's primary teacher and verify the teacher satisfies the stated constraints",
        parent=parent,
        critical=True
    )

    # Leaf: Teacher_Is_Primary_Teacher
    primary_node = evaluator.add_leaf(
        id="Teacher_Is_Primary_Teacher",
        desc="The identified person is the pianist's primary teacher (not merely a coach/masterclass teacher), supported by evidence",
        parent=teacher_node,
        critical=True
    )
    claim_primary = (
        f"{teacher.teacher_name or 'The teacher'} is the primary teacher of {pianist.pianist_name or 'the pianist'}, "
        f"not merely a masterclass teacher or occasional coach."
    )
    await evaluator.verify(
        claim=claim_primary,
        node=primary_node,
        sources=_safe_list(teacher.teacher_sources),
        additional_instruction="Look for language such as 'primary teacher', 'main teacher', 'principal teacher', or equivalent "
                               "that clearly indicates an ongoing primary mentorship rather than a one-off class."
    )

    # Leaf: Teacher_ConcertPianist_And_ConservatoryFaculty
    cnsrv_node = evaluator.add_leaf(
        id="Teacher_ConcertPianist_And_ConservatoryFaculty",
        desc="The teacher is a concert pianist who later joined the faculty at a major U.S. conservatory",
        parent=teacher_node,
        critical=True
    )
    claim_conserv = (
        f"{teacher.teacher_name or 'The teacher'} is a concert pianist and later joined the faculty at a major U.S. conservatory"
        f"{(' (' + teacher.conservatory_faculty_name + ')') if (teacher.conservatory_faculty_name) else ''}."
    )
    await evaluator.verify(
        claim=claim_conserv,
        node=cnsrv_node,
        sources=_safe_list(teacher.teacher_sources),
        additional_instruction="Verify two things: (1) the person is a concert pianist, and (2) the person joined the faculty of a "
                               "major conservatory in the United States (e.g., schools like NEC, Juilliard, Curtis, etc.)."
    )


async def build_teacher_education_nodes(evaluator: Evaluator, parent, extracted: FullExtraction) -> None:
    teacher = extracted.teacher or TeacherExtraction()
    edu = extracted.education or EducationExtraction()

    edu_node = evaluator.add_parallel(
        id="Teacher_Education",
        desc="Verify the teacher's NEC credential requirement",
        parent=parent,
        critical=True
    )

    nec_node = evaluator.add_leaf(
        id="NEC_ArtistDiploma_2004",
        desc="The teacher graduated from the New England Conservatory of Music with an Artist Diploma in 2004",
        parent=edu_node,
        critical=True
    )
    claim_nec = (
        f"{teacher.teacher_name or 'The teacher'} graduated from the New England Conservatory of Music with an Artist Diploma in 2004."
    )
    await evaluator.verify(
        claim=claim_nec,
        node=nec_node,
        sources=_safe_list(edu.education_sources),
        additional_instruction="The page should explicitly indicate an Artist Diploma (AD) from NEC and the year 2004."
    )


async def build_debut_recording_nodes(evaluator: Evaluator, parent, extracted: FullExtraction) -> None:
    teacher = extracted.teacher or TeacherExtraction()
    rec = extracted.recording or RecordingExtraction()

    rec_node = evaluator.add_parallel(
        id="Debut_Recording",
        desc="Verify the teacher's debut recording album constraints",
        parent=parent,
        critical=True
    )

    # Leaf: Debut_Recording_2008_Honens
    debut_node = evaluator.add_leaf(
        id="Debut_Recording_2008_Honens",
        desc="The teacher released a debut recording album in 2008 on the Honens label",
        parent=rec_node,
        critical=True
    )
    claim_debut = (
        f"In 2008, {teacher.teacher_name or 'the teacher'} released a debut recording album on the Honens label."
    )
    await evaluator.verify(
        claim=claim_debut,
        node=debut_node,
        sources=_safe_list(rec.recording_sources),
        additional_instruction="The evidence should indicate the year 2008, that it was the debut recording, and that the label was Honens."
    )

    # Leaf: Recording_Features_Liszt
    liszt_node = evaluator.add_leaf(
        id="Recording_Features_Liszt",
        desc="The debut recording featured Liszt transcriptions and études",
        parent=rec_node,
        critical=True
    )
    claim_liszt = (
        f"The debut recording released by {teacher.teacher_name or 'the teacher'} on Honens in 2008 features Liszt transcriptions and études."
    )
    await evaluator.verify(
        claim=claim_liszt,
        node=liszt_node,
        sources=_safe_list(rec.recording_sources),
        additional_instruction="Look for track listings or descriptions explicitly mentioning Liszt transcriptions and études."
    )


async def build_carnegie_debut_nodes(evaluator: Evaluator, parent, extracted: FullExtraction) -> None:
    teacher = extracted.teacher or TeacherExtraction()
    debut = extracted.carnegie_debut or CarnegieDebutExtraction()

    ch_node = evaluator.add_parallel(
        id="Carnegie_Hall_Debut",
        desc="Identify the specific Carnegie Hall recital hall from the debut-performance constraints",
        parent=parent,
        critical=True
    )

    # Leaf: Debut_At_CarnegieHall_On_Date
    date_node = evaluator.add_leaf(
        id="Debut_At_CarnegieHall_On_Date",
        desc="The teacher made a debut performance at Carnegie Hall on October 8, 2009",
        parent=ch_node,
        critical=True
    )
    claim_date = (
        f"On October 8, 2009, {teacher.teacher_name or 'the teacher'} made a debut performance at Carnegie Hall in New York City."
    )
    await evaluator.verify(
        claim=claim_date,
        node=date_node,
        sources=_safe_list(debut.venue_sources),
        additional_instruction="Allow minor date formatting variants like 'Oct 8, 2009'. The meaning must match October 8, 2009."
    )

    # Leaf: Specific_Recital_Hall_Within_CarnegieHall
    hall_node = evaluator.add_leaf(
        id="Specific_Recital_Hall_Within_CarnegieHall",
        desc="The debut performance was at a specific recital hall within Carnegie Hall (the answer must name which hall), supported by evidence",
        parent=ch_node,
        critical=True
    )
    hall_name = debut.specific_recital_hall_name or "the specified recital hall"
    claim_hall = (
        f"The debut performance took place at {hall_name}, a specific recital hall within Carnegie Hall."
    )
    await evaluator.verify(
        claim=claim_hall,
        node=hall_node,
        sources=_safe_list(debut.venue_sources),
        additional_instruction="Carnegie Hall venues include Weill Recital Hall, Zankel Hall, and Stern Auditorium/Perelman Stage. "
                               "Verify the specific hall name cited in the answer."
    )


async def build_capacity_node(evaluator: Evaluator, parent, extracted: FullExtraction) -> None:
    debut = extracted.carnegie_debut or CarnegieDebutExtraction()
    cap = extracted.capacity or CapacityExtraction()

    # Single leaf at root level (critical)
    cap_node = evaluator.add_leaf(
        id="Venue_Capacity",
        desc="Provide the exact seating capacity of the identified Carnegie Hall recital hall (and ensure it matches the hall named in the prior step)",
        parent=parent,
        critical=True
    )

    hall_name = debut.specific_recital_hall_name or "the identified hall"
    capacity_text = cap.seating_capacity or "the stated capacity"
    sources = _merge_sources(cap.capacity_sources, debut.venue_sources)

    claim_capacity = f"The seating capacity of {hall_name} at Carnegie Hall is {capacity_text}."

    await evaluator.verify(
        claim=claim_capacity,
        node=cap_node,
        sources=sources,
        additional_instruction=(
            "Prefer official or authoritative sources (e.g., carnegiehall.org). Common capacities: Weill Recital Hall (~268), "
            "Zankel Hall (~599), Stern Auditorium/Perelman Stage (~2800+). Minor formatting (e.g., commas) should not affect correctness."
        )
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
) -> Dict[str, Any]:
    """
    Evaluation entry point for the recital hall capacity identification task.
    """
    # Initialize evaluator with a sequential root to respect ordering dependencies
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

    # Add a critical task root node to mirror the rubric's critical Root
    task_root = evaluator.add_sequential(
        id="Root",
        desc="Identify the recital hall implied by the given constraints and report that hall's seating capacity",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extracted: FullExtraction = await evaluator.extract(
        prompt=prompt_extract_core(),
        template_class=FullExtraction,
        extraction_name="core_extraction"
    )

    # Build and verify subtrees according to the rubric (all critical and sequential under Root)
    await build_identify_pianist_nodes(evaluator, task_root, extracted)
    await build_identify_teacher_nodes(evaluator, task_root, extracted)
    await build_teacher_education_nodes(evaluator, task_root, extracted)
    await build_debut_recording_nodes(evaluator, task_root, extracted)
    await build_carnegie_debut_nodes(evaluator, task_root, extracted)
    await build_capacity_node(evaluator, task_root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()