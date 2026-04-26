import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "who_is_this_person_uw_columbia_2026"
TASK_DESCRIPTION = """An individual was born in 1967 and earned three degrees from prestigious institutions: a BA from Harvard University in 1988, a JD from Yale Law School in 1995, and a PhD in history and social study of science and technology from MIT in 1999. This person began their academic career as a faculty member at the University of Virginia School of Law, serving from 1998 to 2005, before joining the faculty of UCLA School of Law in 2005.

After serving on the UCLA Law faculty for a decade, this individual was appointed as Dean of the UCLA School of Law in August 2015, succeeding Rachel F. Moran (who served from 2010 to 2015) and serving until June 2022, when they were succeeded by Russell Korobkin (who served as Interim Dean from 2022 to 2023).

Following their UCLA deanship, this person became the 30th Chancellor of the University of Wisconsin-Madison on August 4, 2022. They succeeded Rebecca Blank (who served from July 22, 2013 to May 31, 2022), with John Karl Scholz serving as interim chancellor from June 1, 2022 to August 3, 2022 between Blank and this individual.

This person was elected to the American Academy of Arts and Sciences on April 23, 2020. On January 25, 2026, it was announced that this individual would become the 21st President of Columbia University, with the appointment effective July 1, 2026.

Who is this person?"""

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class PersonExtraction(BaseModel):
    name: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_person() -> str:
    return """
    You must extract:
    1) name: The full name of the person explicitly identified in the answer text (not a description).
    2) source_urls: Every URL cited in the answer as evidence or sources (include plain URLs and markdown links).
    
    Return JSON with fields:
    - name: string or null if not explicitly stated
    - source_urls: array of strings (valid URLs). If none are present, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root: VerificationNode, extracted: PersonExtraction) -> None:
    # Main critical node representing the rubric root
    person_node = evaluator.add_parallel(
        id="Person_Identification",
        desc="Identify the person who satisfies all stated constraints",
        parent=root,
        critical=True,
    )

    # Critical: The answer must state a person's name
    provides_name_node = evaluator.add_custom_node(
        result=bool(extracted.name and extracted.name.strip()),
        id="Provides_Person_Name",
        desc="Response explicitly states the person's name/identity (not just a description)",
        parent=person_node,
        critical=True
    )

    # Prepare sources (may be empty). Verification will try to use webpages when present
    sources = extracted.source_urls

    person_ref = extracted.name if extracted.name else "the person identified in the answer"

    # Birth year
    birth_year_leaf = evaluator.add_leaf(
        id="Birth_Year",
        desc="Born in 1967",
        parent=person_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{person_ref} was born in 1967.",
        node=birth_year_leaf,
        sources=sources,
        additional_instruction="Verify using the provided source URLs. Accept minor phrasing differences (e.g., 'born 1967' or 'born in 1967'). If no URL explicitly supports this, mark as not supported."
    )

    # Degrees (critical parallel)
    degrees_node = evaluator.add_parallel(
        id="Degrees",
        desc="Educational degree constraints",
        parent=person_node,
        critical=True
    )

    ba_leaf = evaluator.add_leaf(
        id="BA_Harvard_1988",
        desc="Holds a BA from Harvard University earned in 1988",
        parent=degrees_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{person_ref} holds a BA from Harvard University awarded in 1988.",
        node=ba_leaf,
        sources=sources,
        additional_instruction="Check if the sources explicitly state a Bachelor of Arts from Harvard in 1988. Allow minor variations in formatting (e.g., 'A.B.')."
    )

    jd_leaf = evaluator.add_leaf(
        id="JD_Yale_1995",
        desc="Holds a JD from Yale Law School earned in 1995",
        parent=degrees_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{person_ref} holds a JD from Yale Law School awarded in 1995.",
        node=jd_leaf,
        sources=sources,
        additional_instruction="Verify explicitly that the person has a Juris Doctor from Yale Law School in 1995. Allow common abbreviations like 'J.D.'"
    )

    phd_leaf = evaluator.add_leaf(
        id="PhD_MIT_1999",
        desc="Holds a PhD in history and social study of science and technology from MIT earned in 1999",
        parent=degrees_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{person_ref} holds a PhD in history and social study of science and technology from MIT awarded in 1999.",
        node=phd_leaf,
        sources=sources,
        additional_instruction="Allow minor variations of the program's phrasing (e.g., 'History and Social Study of Science and Technology'). Verify PhD from MIT, 1999, matching this field."
    )

    # Faculty positions (critical parallel)
    faculty_node = evaluator.add_parallel(
        id="Faculty_Positions",
        desc="Faculty position constraints",
        parent=person_node,
        critical=True
    )

    uva_leaf = evaluator.add_leaf(
        id="UVA_Faculty_1998_2005",
        desc="Served as faculty at University of Virginia School of Law from 1998 to 2005",
        parent=faculty_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{person_ref} served as a faculty member at the University of Virginia School of Law from 1998 to 2005.",
        node=uva_leaf,
        sources=sources,
        additional_instruction="Confirm appointment at UVA School of Law during that range (1998–2005)."
    )

    ucla_join_leaf = evaluator.add_leaf(
        id="UCLA_Faculty_Joined_2005",
        desc="Joined UCLA School of Law faculty in 2005",
        parent=faculty_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{person_ref} joined the UCLA School of Law faculty in 2005.",
        node=ucla_join_leaf,
        sources=sources,
        additional_instruction="Confirm that the person joined UCLA Law in 2005 (not necessarily the same as becoming dean)."
    )

    # UCLA Deanship (critical parallel)
    ucla_dean_node = evaluator.add_parallel(
        id="UCLA_Dean",
        desc="UCLA School of Law deanship constraints",
        parent=person_node,
        critical=True
    )

    ucla_term_leaf = evaluator.add_leaf(
        id="UCLA_Dean_Term_Aug2015_Jun2022",
        desc="Served as Dean of UCLA School of Law from August 2015 to June 2022",
        parent=ucla_dean_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{person_ref} served as Dean of UCLA School of Law from August 2015 to June 2022.",
        node=ucla_term_leaf,
        sources=sources,
        additional_instruction="Verify both start month/year and end month/year."
    )

    ucla_pre_leaf = evaluator.add_leaf(
        id="UCLA_Dean_Predecessor_Rachel_Moran_2010_2015",
        desc="Immediate predecessor as UCLA Law Dean was Rachel F. Moran (served 2010–2015)",
        parent=ucla_dean_node,
        critical=True
    )
    await evaluator.verify(
        claim="The immediate predecessor as UCLA School of Law dean was Rachel F. Moran, who served from 2010 to 2015.",
        node=ucla_pre_leaf,
        sources=sources,
        additional_instruction="Check for predecessor listing and Moran's term (2010–2015)."
    )

    ucla_succ_leaf = evaluator.add_leaf(
        id="UCLA_Dean_Successor_Russell_Korobkin_Interim_2022_2023",
        desc="Immediate successor as UCLA Law Dean was Russell Korobkin (Interim, 2022–2023)",
        parent=ucla_dean_node,
        critical=True
    )
    await evaluator.verify(
        claim="The immediate successor as UCLA School of Law dean was Russell Korobkin, who served as Interim Dean from 2022 to 2023.",
        node=ucla_succ_leaf,
        sources=sources,
        additional_instruction="Verify that Korobkin served as interim dean in the stated period."
    )

    # UW–Madison Chancellorship (critical parallel)
    uw_node = evaluator.add_parallel(
        id="UW_Madison_Chancellor",
        desc="UW–Madison chancellorship constraints",
        parent=person_node,
        critical=True
    )

    uw_30th_leaf = evaluator.add_leaf(
        id="UW_30th_Chancellor",
        desc="Is the 30th Chancellor of the University of Wisconsin–Madison",
        parent=uw_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{person_ref} is the 30th Chancellor of the University of Wisconsin–Madison.",
        node=uw_30th_leaf,
        sources=sources,
        additional_instruction="Confirm ordinal numbering (30th) explicitly on an official or authoritative source."
    )

    uw_start_leaf = evaluator.add_leaf(
        id="UW_Chancellor_Start_Aug4_2022",
        desc="Assumed the UW–Madison chancellorship on August 4, 2022",
        parent=uw_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{person_ref} assumed the UW–Madison chancellorship on August 4, 2022.",
        node=uw_start_leaf,
        sources=sources,
        additional_instruction="Verify the exact start date of the chancellorship."
    )

    uw_pred_leaf = evaluator.add_leaf(
        id="UW_Predecessor_Rebecca_Blank_Jul22_2013_May31_2022",
        desc="Predecessor as UW–Madison Chancellor was Rebecca Blank (served July 22, 2013 to May 31, 2022)",
        parent=uw_node,
        critical=True
    )
    await evaluator.verify(
        claim="The predecessor as UW–Madison Chancellor was Rebecca Blank, who served from July 22, 2013 to May 31, 2022.",
        node=uw_pred_leaf,
        sources=sources,
        additional_instruction="Confirm predecessor name and service dates."
    )

    uw_interim_leaf = evaluator.add_leaf(
        id="UW_Interim_JohnKarlScholz_Jun1_Aug3_2022",
        desc="Between Rebecca Blank and the person, John Karl Scholz served as interim chancellor from June 1, 2022 to August 3, 2022",
        parent=uw_node,
        critical=True
    )
    await evaluator.verify(
        claim="Between Rebecca Blank and the person, John Karl Scholz served as interim chancellor from June 1, 2022 to August 3, 2022.",
        node=uw_interim_leaf,
        sources=sources,
        additional_instruction="Confirm the interim service and its exact dates."
    )

    # Academic Honors (critical parallel)
    honors_node = evaluator.add_parallel(
        id="Academic_Honors",
        desc="Academic honor constraints",
        parent=person_node,
        critical=True
    )

    aaas_leaf = evaluator.add_leaf(
        id="AAAS_Elected_Apr23_2020",
        desc="Elected to the American Academy of Arts and Sciences on April 23, 2020",
        parent=honors_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{person_ref} was elected to the American Academy of Arts and Sciences on April 23, 2020.",
        node=aaas_leaf,
        sources=sources,
        additional_instruction="Verify election to AAAS and the specific date."
    )

    # Columbia Presidency (critical parallel)
    columbia_node = evaluator.add_parallel(
        id="Columbia_Presidency",
        desc="Columbia University presidency constraints",
        parent=person_node,
        critical=True
    )

    columbia_21st_leaf = evaluator.add_leaf(
        id="Columbia_21st_President",
        desc="Named the 21st President of Columbia University",
        parent=columbia_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{person_ref} was named the 21st President of Columbia University.",
        node=columbia_21st_leaf,
        sources=sources,
        additional_instruction="Confirm the ordinal (21st) and position title explicitly."
    )

    columbia_announced_leaf = evaluator.add_leaf(
        id="Columbia_Announced_Jan25_2026",
        desc="Appointment announced on January 25, 2026",
        parent=columbia_node,
        critical=True
    )
    await evaluator.verify(
        claim="The appointment as Columbia University President was announced on January 25, 2026.",
        node=columbia_announced_leaf,
        sources=sources,
        additional_instruction="Verify the official announcement date from authoritative sources."
    )

    columbia_effective_leaf = evaluator.add_leaf(
        id="Columbia_Effective_Jul1_2026",
        desc="Appointment effective on July 1, 2026",
        parent=columbia_node,
        critical=True
    )
    await evaluator.verify(
        claim="The appointment as Columbia University President is effective July 1, 2026.",
        node=columbia_effective_leaf,
        sources=sources,
        additional_instruction="Verify the effective date from authoritative sources."
    )

    # Chronology Consistency (critical parallel) – logical check
    chronology_node = evaluator.add_parallel(
        id="Chronology_Consistency",
        desc="Chronological constraints across roles",
        parent=person_node,
        critical=True
    )

    chronology_leaf = evaluator.add_leaf(
        id="President_Announcement_During_Chancellorship",
        desc="Columbia presidency announcement (January 25, 2026) occurred during the UW–Madison chancellorship",
        parent=chronology_node,
        critical=True
    )
    await evaluator.verify(
        claim="Given the UW–Madison chancellorship began on August 4, 2022 and the Columbia presidency announcement was on January 25, 2026, the announcement occurred during the chancellorship.",
        node=chronology_leaf,
        # This is a logical consistency check; we do not need URLs here,
        # but it depends on the two earlier nodes being correct.
        sources=None,
        additional_instruction="This is a logical consistency check based on the previously verified dates. If either of the prerequisite date claims is not supported, this should not pass.",
        extra_prerequisites=[uw_start_leaf, columbia_announced_leaf]
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_person(),
        template_class=PersonExtraction,
        extraction_name="person_extraction"
    )

    # Optional: record custom info about extraction
    evaluator.add_custom_info(
        {"extracted_name": extracted.name, "num_source_urls": len(extracted.source_urls)},
        info_type="extraction_summary"
    )

    # Build and run verification tree
    await build_verification_tree(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()