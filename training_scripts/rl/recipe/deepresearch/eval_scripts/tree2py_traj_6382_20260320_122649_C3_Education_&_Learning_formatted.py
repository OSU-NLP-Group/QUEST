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
TASK_ID = "higher_ed_leader_identification"
TASK_DESCRIPTION = """
An individual in higher education leadership earned a Bachelor of Science degree from Yale University, with undergraduate studies in astronomy and physics, and subsequently earned a Ph.D. in astronomy from Harvard University. Early in their career, this person held a Miller Research Fellowship at the University of California, Berkeley. The individual later served as Dean of Science at York University, with this appointment announced or welcomed in June 2014. Following York University, the individual became Dean of the College of Arts and Sciences at Cornell University. In July 2023, this person was appointed as Provost of Johns Hopkins University and assumed that role in October 2023. Most recently, in January 2026, the individual was appointed as the 10th President of the California Institute of Technology (Caltech), with an official start date of July 1, 2026. Who is this individual?
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PersonCareerExtraction(BaseModel):
    """Extract the identified individual's name and per-claim supporting URLs."""
    name: Optional[str] = None

    # Education
    yale_urls: List[str] = Field(default_factory=list)      # B.S. from Yale; undergrad astronomy & physics
    harvard_urls: List[str] = Field(default_factory=list)   # Ph.D. in astronomy from Harvard

    # Early career
    miller_urls: List[str] = Field(default_factory=list)    # Miller Research Fellow at UC Berkeley

    # Leadership roles
    york_urls: List[str] = Field(default_factory=list)      # Dean of Science at York; announced/welcomed June 2014
    cornell_urls: List[str] = Field(default_factory=list)   # Dean of College of Arts & Sciences, Cornell
    hopkins_urls: List[str] = Field(default_factory=list)   # JHU Provost July 2023 appointment; Oct 2023 start
    caltech_urls: List[str] = Field(default_factory=list)   # 10th President of Caltech; announced Jan 2026; starts Jul 1, 2026


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_person_and_sources() -> str:
    return """
    Extract the individual's full name and the explicit URL sources cited in the answer that support each specific claim category below.

    Fields to extract:
    - name: The individual's full name as given in the answer.
    - yale_urls: URLs that specifically support that the person earned a Bachelor of Science (B.S.) from Yale University and studied astronomy and physics as an undergraduate.
    - harvard_urls: URLs that specifically support that the person earned a Ph.D. in astronomy from Harvard University.
    - miller_urls: URLs that specifically support that the person held a Miller Research Fellowship at the University of California, Berkeley (UC Berkeley).
    - york_urls: URLs that specifically support that the person served as Dean of Science at York University and that this appointment was announced or welcomed in June 2014.
    - cornell_urls: URLs that specifically support that the person served as Dean of the College of Arts and Sciences at Cornell University.
    - hopkins_urls: URLs that specifically support that the person was appointed Provost of Johns Hopkins University in July 2023 and assumed the role in October 2023.
    - caltech_urls: URLs that specifically support that the person was appointed as the 10th President of Caltech, announced in January 2026, with an official start date of July 1, 2026.

    Rules:
    - Only include URLs explicitly present in the answer. Do not invent URLs.
    - If the answer provides a single list of sources, assign each URL to the most relevant category based on its content as indicated in the answer text.
    - Exclude duplicates within each list.
    - If no URLs are provided for a category, return an empty list for that category.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _person_ref(name: Optional[str]) -> str:
    return name if name and name.strip() else "the identified individual"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_undergraduate_education(
    evaluator: Evaluator,
    parent,
    name: Optional[str],
    urls: List[str],
):
    node = evaluator.add_parallel(
        id="Undergraduate_Education",
        desc="The individual's undergraduate education at Yale University is verified, including degree type and fields of study",
        parent=parent,
        critical=True,
    )

    # Existence of at least one URL reference (critical)
    evaluator.add_custom_node(
        result=bool(urls),
        id="Undergraduate_Reference",
        desc="At least one URL reference verifies the Yale B.S. degree and astronomy/physics studies",
        parent=node,
        critical=True,
    )

    # Verify Yale B.S. with astronomy and physics
    leaf = evaluator.add_leaf(
        id="Yale_BS_Degree",
        desc="The individual earned a Bachelor of Science degree from Yale University with undergraduate studies in astronomy and physics",
        parent=node,
        critical=True,
    )
    claim = f"{_person_ref(name)} earned a Bachelor of Science (B.S.) degree from Yale University and studied astronomy and physics as an undergraduate."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "Treat 'Yale College' as equivalent to Yale University's undergraduate program. "
            "Allow reasonable phrasing variants (e.g., 'physics and astronomy'). "
            "The page should explicitly support both: the Yale B.S. degree and the astronomy/physics undergraduate studies."
        ),
    )


async def build_graduate_education(
    evaluator: Evaluator,
    parent,
    name: Optional[str],
    urls: List[str],
):
    node = evaluator.add_parallel(
        id="Graduate_Education",
        desc="The individual's doctoral education at Harvard University is verified, including field of study",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(urls),
        id="Graduate_Reference",
        desc="At least one URL reference verifies the Harvard Ph.D. in astronomy",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Harvard_PhD_Astronomy",
        desc="The individual earned a Ph.D. in astronomy from Harvard University",
        parent=node,
        critical=True,
    )
    claim = f"{_person_ref(name)} earned a Ph.D. in astronomy from Harvard University."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="Verify that the doctorate is specifically in astronomy and that it is from Harvard University.",
    )


async def build_credential_verification(
    evaluator: Evaluator,
    parent,
    extraction: PersonCareerExtraction,
):
    node = evaluator.add_sequential(
        id="Credential_Verification",
        desc="The individual's educational credentials are verified through multiple institutions",
        parent=parent,
        critical=True,
    )

    await build_undergraduate_education(evaluator, node, extraction.name, extraction.yale_urls)
    await build_graduate_education(evaluator, node, extraction.name, extraction.harvard_urls)


async def build_postdoctoral_research(
    evaluator: Evaluator,
    parent,
    name: Optional[str],
    urls: List[str],
):
    node = evaluator.add_parallel(
        id="Postdoctoral_Research",
        desc="The individual held a Miller Research Fellowship at the University of California, Berkeley",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(urls),
        id="Fellowship_Reference",
        desc="At least one URL reference verifies the Miller Research Fellowship at UC Berkeley",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Miller_Fellowship",
        desc="The individual was a Miller Research Fellow at UC Berkeley",
        parent=node,
        critical=True,
    )
    claim = f"{_person_ref(name)} held a Miller Research Fellowship at the University of California, Berkeley."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="The page should clearly indicate the individual was a Miller Research Fellow at UC Berkeley.",
    )


async def build_first_deanship(
    evaluator: Evaluator,
    parent,
    name: Optional[str],
    urls: List[str],
):
    node = evaluator.add_parallel(
        id="First_Deanship",
        desc="The individual served as Dean of Science at York University, with appointment announced or welcomed in June 2014",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(urls),
        id="York_Reference",
        desc="At least one URL reference verifies the York University Dean of Science position and June 2014 timing",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="York_Dean_Role_And_Timing",
        desc="The individual held the position of Dean of Science at York University and the appointment was announced or welcomed in June 2014",
        parent=node,
        critical=True,
    )
    claim = (
        f"{_person_ref(name)} served as Dean of Science at York University and the appointment was announced or "
        f"welcomed in June 2014."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "Confirm two parts: (1) the person was Dean of Science at York University; "
            "(2) the appointment announcement or welcome was in June 2014 (the post/article date should be June 2014). "
            "Allow reasonable phrasing such as 'welcomed as dean' or 'announced as dean' in June 2014."
        ),
    )


async def build_cornell_deanship(
    evaluator: Evaluator,
    parent,
    name: Optional[str],
    urls: List[str],
):
    node = evaluator.add_parallel(
        id="Cornell_Deanship",
        desc="The individual served as Dean of the College of Arts and Sciences at Cornell University",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(urls),
        id="Cornell_Reference",
        desc="At least one URL reference verifies the Cornell Dean of Arts and Sciences position",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Cornell_Dean_Role",
        desc="The individual held the position of Dean of the College of Arts and Sciences at Cornell University",
        parent=node,
        critical=True,
    )
    claim = f"{_person_ref(name)} served as Dean of the College of Arts and Sciences at Cornell University."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction="Verify that the individual was dean of Cornell University's College of Arts & Sciences.",
    )


async def build_hopkins_provost(
    evaluator: Evaluator,
    parent,
    name: Optional[str],
    urls: List[str],
):
    node = evaluator.add_parallel(
        id="Johns_Hopkins_Provost",
        desc="The individual was appointed as Provost of Johns Hopkins University in July 2023 and assumed the role in October 2023",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(urls),
        id="Hopkins_Reference",
        desc="At least one URL reference verifies the Johns Hopkins Provost appointment in July 2023 and/or October 2023 start date",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Hopkins_Appointment_And_Start",
        desc="The individual was appointed as Provost of Johns Hopkins University in July 2023 and assumed the role in October 2023",
        parent=node,
        critical=True,
    )
    claim = f"{_person_ref(name)} was appointed Provost of Johns Hopkins University in July 2023 and started in October 2023."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "Confirm both the July 2023 appointment announcement and the October 2023 start date. "
            "Allow references that explicitly provide either the appointment month (July 2023) and/or the start month (October 2023); "
            "ideally, the combined set of provided URLs should cover both facts."
        ),
    )


async def build_caltech_presidency(
    evaluator: Evaluator,
    parent,
    name: Optional[str],
    urls: List[str],
):
    node = evaluator.add_parallel(
        id="Caltech_Presidency",
        desc="The individual was appointed as the 10th President of Caltech, announced in January 2026, with a start date of July 1, 2026",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(urls),
        id="Caltech_Reference",
        desc="At least one URL reference verifies the Caltech 10th President appointment, January 2026 announcement, and July 1, 2026 start date",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Caltech_President_Details",
        desc="The individual was appointed as the 10th President of Caltech, the appointment was announced in January 2026, and the official start date is July 1, 2026",
        parent=node,
        critical=True,
    )
    claim = (
        f"In January 2026, {_person_ref(name)} was announced as the 10th President of the California Institute of "
        f"Technology (Caltech), with an official start date of July 1, 2026."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "Verify three details: (1) 10th President of Caltech, (2) announcement in January 2026, "
            "(3) start date July 1, 2026."
        ),
    )


async def build_academic_leadership_progression(
    evaluator: Evaluator,
    parent,
    extraction: PersonCareerExtraction,
):
    node = evaluator.add_sequential(
        id="Academic_Leadership_Progression",
        desc="The individual progressed through multiple dean-level positions at major universities",
        parent=parent,
        critical=True,
    )

    # First deanship at York (with June 2014 timing)
    await build_first_deanship(evaluator, node, extraction.name, extraction.york_urls)

    # Second deanship and beyond (Cornell -> Hopkins -> Caltech)
    second_node = evaluator.add_sequential(
        id="Second_Deanship_And_Beyond",
        desc="The individual advanced to dean position at Cornell University and then to provost and presidential roles",
        parent=node,
        critical=True,
    )

    await build_cornell_deanship(evaluator, second_node, extraction.name, extraction.cornell_urls)

    # Executive leadership (Provost -> President)
    exec_node = evaluator.add_sequential(
        id="Executive_Leadership",
        desc="The individual advanced to the highest levels of university leadership as provost and president",
        parent=second_node,
        critical=True,
    )

    await build_hopkins_provost(evaluator, exec_node, extraction.name, extraction.hopkins_urls)
    await build_caltech_presidency(evaluator, exec_node, extraction.name, extraction.caltech_urls)


async def build_professional_career(
    evaluator: Evaluator,
    parent,
    extraction: PersonCareerExtraction,
):
    node = evaluator.add_sequential(
        id="Professional_Career",
        desc="The individual's career progression through academic and administrative positions is verified",
        parent=parent,
        critical=True,
    )

    await build_postdoctoral_research(evaluator, node, extraction.name, extraction.miller_urls)
    await build_academic_leadership_progression(evaluator, node, extraction)


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the higher education leadership identification task.
    """
    # Initialize evaluator and root
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
        default_model=model,
    )

    # Extract the individual's name and categorized URLs from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_person_and_sources(),
        template_class=PersonCareerExtraction,
        extraction_name="person_and_sources",
    )

    # Build the verification tree according to the rubric
    # Top-level critical sequential node
    top = evaluator.add_sequential(
        id="Individual_Identification",
        desc="The solver correctly identifies an individual in higher education leadership who meets all specified educational and career criteria",
        parent=root,
        critical=True,
    )

    # Educational credentials (critical, sequential)
    await build_credential_verification(evaluator, top, extraction)

    # Professional career (critical, sequential)
    await build_professional_career(evaluator, top, extraction)

    # Return the structured evaluation summary
    return evaluator.get_summary()