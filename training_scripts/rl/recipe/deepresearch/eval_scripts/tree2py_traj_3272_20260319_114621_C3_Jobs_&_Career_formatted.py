import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "astro_career_identification"
TASK_DESCRIPTION = """
An astrophysicist who graduated with a B.S. in Astronomy and Physics from Yale University in 1994 and earned a Ph.D. in Astronomy from Harvard University in 2000 began their academic career with a Miller Research Fellowship at the University of California, Berkeley. After this postdoctoral position, they served as an Assistant Professor at the University of Michigan for approximately 2 years before joining the University of Toronto in 2004, where they spent exactly 10 years and held a Canada Research Chair in Observational Astrophysics (awarded in 2008). In 2014, they became the Dean of the Faculty of Science at York University in Toronto, serving for approximately 4 years. In 2018, they were appointed as the 22nd Dean of the College of Arts and Sciences at Cornell University (specifically as the Harold Tanner Dean) and were later named the Hans A. Bethe Professor in 2022. In 2023, they became the 16th Provost of Johns Hopkins University. In January 2026, this individual was named the 10th President of a major California research institution, with the position effective July 1, 2026. Who is this person, and what institution will they lead as president starting July 1, 2026?
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CareerExtraction(BaseModel):
    """
    Extract the identified individual's name, the target presidential institution, and
    source URLs that the answer provides for verifying each career milestone.
    """
    individual_name: Optional[str] = None
    presidential_institution: Optional[str] = None

    # Source URLs grouped by milestone (extract all URLs mentioned in the answer for each item)
    yale_bs_urls: List[str] = Field(default_factory=list)
    harvard_phd_urls: List[str] = Field(default_factory=list)
    berkeley_miller_urls: List[str] = Field(default_factory=list)
    michigan_assistant_prof_urls: List[str] = Field(default_factory=list)
    toronto_crc_urls: List[str] = Field(default_factory=list)
    york_dean_urls: List[str] = Field(default_factory=list)
    cornell_dean_bethe_urls: List[str] = Field(default_factory=list)
    jhu_provost_urls: List[str] = Field(default_factory=list)
    presidency_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_career_info() -> str:
    return """
    From the provided answer:
    1) Extract the individual's full name as 'individual_name'.
    2) Extract the California research institution they will lead as president starting July 1, 2026 as 'presidential_institution'.

    Also extract all URLs that the answer cites for supporting each of the following milestones.
    Only include URLs explicitly present in the answer (plain links or markdown links).
    Do not invent or infer any URLs.

    Return a JSON object with these fields (use empty list if none provided, and null for missing strings):
    - individual_name: string | null
    - presidential_institution: string | null
    - yale_bs_urls: string[]    # B.S. in Astronomy and Physics from Yale University in 1994
    - harvard_phd_urls: string[]  # Ph.D. in Astronomy from Harvard University in 2000
    - berkeley_miller_urls: string[]  # Miller Research Fellowship at UC Berkeley (circa 2000–2002)
    - michigan_assistant_prof_urls: string[]  # Assistant Professor at University of Michigan (circa 2002–2004)
    - toronto_crc_urls: string[]  # University of Toronto (joined 2004, 10 years to 2014) + Canada Research Chair in Observational Astrophysics (awarded 2008)
    - york_dean_urls: string[]  # Dean of Faculty of Science at York University (2014–2018)
    - cornell_dean_bethe_urls: string[]  # 22nd Harold Tanner Dean at Cornell in 2018; named Hans A. Bethe Professor in 2022
    - jhu_provost_urls: string[]  # 16th Provost of Johns Hopkins University (effective Oct 2023)
    - presidency_urls: string[]  # Named 10th President in Jan 2026, effective July 1, 2026, of the California institution
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _with_source_requirement(base_instruction: str, urls: List[str], name_present: bool = True) -> str:
    """
    Build additional instruction enforcing source-grounding:
    - If no URLs are provided, the judge must mark as NOT SUPPORTED (Incorrect).
    - If the individual's name is missing, emphasize that the claim should be marked Incorrect.
    """
    ins = base_instruction.strip()
    if not urls:
        ins += " Important: The answer provided NO URL(s) for this item. You MUST conclude the claim is NOT SUPPORTED and return Incorrect."
    if not name_present:
        ins += " Important: No individual's name was extracted from the answer; treat the claim as NOT SUPPORTED and return Incorrect."
    return ins


async def _verify_leaf_with_urls(
    evaluator: Evaluator,
    *,
    node_id: str,
    node_desc: str,
    parent,
    claim: str,
    urls: List[str],
    base_instruction: str,
    critical: bool = True,
) -> None:
    """
    Create a leaf node and verify a claim with URL evidence when available.
    Enforces failure if URLs are missing (per source-grounding policy) via additional instruction.
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=critical,
    )
    add_ins = _with_source_requirement(base_instruction, urls)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls if urls else None,
        additional_instruction=add_ins,
    )


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_and_verify_career_tree(evaluator: Evaluator, root: Any, ex: CareerExtraction) -> None:
    """
    Build the verification tree according to the rubric and run all checks.
    """

    # Root-critical node as per rubric
    cci_node = evaluator.add_parallel(
        id="Complete_Career_Identification",
        desc="Correctly identify both the individual whose career progression matches all specified milestones and the institution they will lead as president",
        parent=root,
        critical=True,
    )

    # Block A: Identify the individual and verify full career progression
    individual_node = evaluator.add_parallel(
        id="Individual_Name_Identification",
        desc="Provide the correct name of the individual whose career matches all specified educational and professional milestones",
        parent=cci_node,
        critical=True,
    )

    # Optional existence check (explicit leaf) to ensure the answer provides a name
    name_present = ex.individual_name is not None and str(ex.individual_name).strip() != ""
    evaluator.add_custom_node(
        result=name_present,
        id="Name_Provided",
        desc="The individual's name is provided in the answer",
        parent=individual_node,
        critical=True,
    )

    # Sequential career verification
    career_seq = evaluator.add_sequential(
        id="Career_Progression_Verification",
        desc="Verify that the identified individual's complete career progression matches all specified milestones in chronological sequence",
        parent=individual_node,
        critical=True,
    )

    # 1) Educational Background (parallel)
    edu_parallel = evaluator.add_parallel(
        id="Educational_Background",
        desc="Verify the individual's educational credentials match the specified undergraduate and doctoral degrees",
        parent=career_seq,
        critical=True,
    )

    # 1.a Yale B.S. 1994
    await _verify_leaf_with_urls(
        evaluator,
        node_id="Yale_Undergraduate_Degree",
        node_desc="Verify B.S. in Astronomy and Physics from Yale University in 1994",
        parent=edu_parallel,
        claim=f"The individual named '{ex.individual_name}' earned a B.S. (or equivalent, e.g., B.Sc.) in Astronomy and Physics from Yale University in 1994.",
        urls=ex.yale_bs_urls,
        base_instruction="Confirm the undergraduate degree field (Astronomy and Physics), institution (Yale University), and year (1994). Minor naming variations (e.g., 'Astronomy & Physics') are acceptable.",
        critical=True,
    )

    # 1.b Harvard Ph.D. 2000
    await _verify_leaf_with_urls(
        evaluator,
        node_id="Harvard_Doctoral_Degree",
        node_desc="Verify Ph.D. in Astronomy from Harvard University in 2000",
        parent=edu_parallel,
        claim=f"The individual named '{ex.individual_name}' earned a Ph.D. in Astronomy from Harvard University in 2000.",
        urls=ex.harvard_phd_urls,
        base_instruction="Verify doctoral degree discipline (Astronomy), institution (Harvard University), and year (2000).",
        critical=True,
    )

    # 2) Early Career Positions (sequential)
    early_seq = evaluator.add_sequential(
        id="Early_Career_Positions",
        desc="Verify the individual's early career positions following PhD completion",
        parent=career_seq,
        critical=True,
    )

    # 2.a Berkeley Miller Fellow 2000–2002
    await _verify_leaf_with_urls(
        evaluator,
        node_id="Berkeley_Postdoc",
        node_desc="Verify Miller Research Fellowship at UC Berkeley (2000-2002)",
        parent=early_seq,
        claim=f"From 2000 to 2002, the individual named '{ex.individual_name}' held a Miller Research Fellowship at the University of California, Berkeley.",
        urls=ex.berkeley_miller_urls,
        base_instruction="Confirm the postdoctoral title (Miller Research Fellow), institution (UC Berkeley), and approximate time window (2000–2002).",
        critical=True,
    )

    # 2.b Michigan Assistant Professor ~2002–2004
    await _verify_leaf_with_urls(
        evaluator,
        node_id="Michigan_Faculty",
        node_desc="Verify Assistant Professor position at University of Michigan for approximately 2 years (2002-2004)",
        parent=early_seq,
        claim=f"Between 2002 and 2004 (approximately two years), the individual named '{ex.individual_name}' served as an Assistant Professor at the University of Michigan.",
        urls=ex.michigan_assistant_prof_urls,
        base_instruction="Verify rank (Assistant Professor), institution (University of Michigan), and approximate period (circa 2002–2004).",
        critical=True,
    )

    # 3) Canadian Phase (sequential)
    canada_seq = evaluator.add_sequential(
        id="Canadian_Academic_Phase",
        desc="Verify the individual's Canadian academic positions and tenure",
        parent=career_seq,
        critical=True,
    )

    # 3.a University of Toronto tenure + CRC (2008)
    await _verify_leaf_with_urls(
        evaluator,
        node_id="Toronto_Faculty_Tenure",
        node_desc="Verify joining University of Toronto in 2004, spending exactly 10 years (2004-2014), and holding Canada Research Chair in Observational Astrophysics (awarded 2008)",
        parent=canada_seq,
        claim=f"The individual named '{ex.individual_name}' joined the University of Toronto in 2004, spent exactly ten years there (2004–2014), and held a Canada Research Chair in Observational Astrophysics awarded in 2008.",
        urls=ex.toronto_crc_urls,
        base_instruction="Confirm: (1) joined year 2004, (2) tenure length 2004–2014 (10 years), and (3) CRC in Observational Astrophysics, awarded in 2008.",
        critical=True,
    )

    # 3.b York University Dean (2014–2018)
    await _verify_leaf_with_urls(
        evaluator,
        node_id="York_Deanship",
        node_desc="Verify Dean of Faculty of Science at York University from July 2014 to 2018 (approximately 4 years)",
        parent=canada_seq,
        claim=f"From July 2014 to 2018 (approximately four years), the individual named '{ex.individual_name}' served as Dean of the Faculty of Science at York University (Toronto).",
        urls=ex.york_dean_urls,
        base_instruction="Verify the role (Dean of Faculty of Science), institution (York University), and approximate service window (2014–2018).",
        critical=True,
    )

    # 4) U.S. Senior Leadership (sequential)
    us_senior_seq = evaluator.add_sequential(
        id="US_Senior_Leadership_Positions",
        desc="Verify the individual's progression through senior academic leadership positions at major U.S. research universities",
        parent=career_seq,
        critical=True,
    )

    # 4.a Cornell Dean (2018) + Hans A. Bethe Professor (2022)
    await _verify_leaf_with_urls(
        evaluator,
        node_id="Cornell_Dean_Position",
        node_desc="Verify appointment as 22nd Harold Tanner Dean of College of Arts and Sciences at Cornell in 2018 and appointment as Hans A. Bethe Professor in 2022",
        parent=us_senior_seq,
        claim=f"In 2018, the individual named '{ex.individual_name}' was appointed the 22nd Harold Tanner Dean of the College of Arts and Sciences at Cornell University, and in 2022 was named the Hans A. Bethe Professor.",
        urls=ex.cornell_dean_bethe_urls,
        base_instruction="Confirm both elements: (1) 22nd Harold Tanner Dean of A&S at Cornell in 2018; (2) Hans A. Bethe Professor title in 2022.",
        critical=True,
    )

    # 4.b Johns Hopkins Provost (effective Oct 2023)
    await _verify_leaf_with_urls(
        evaluator,
        node_id="Johns_Hopkins_Provost_Position",
        node_desc="Verify appointment as 16th Provost of Johns Hopkins University, effective October 2023",
        parent=us_senior_seq,
        claim=f"In 2023, the individual named '{ex.individual_name}' was appointed the 16th Provost of Johns Hopkins University, effective October 2023.",
        urls=ex.jhu_provost_urls,
        base_instruction="Verify the ordinal (16th Provost), the institution (Johns Hopkins University), and the effective date (October 2023).",
        critical=True,
    )

    # Block B: Presidential institution identification (sibling under CCI root)
    await _verify_leaf_with_urls(
        evaluator,
        node_id="Presidential_Institution_Identification",
        node_desc="Correctly identify the California research institution where the individual was appointed as 10th President in January 2026, with position effective July 1, 2026",
        parent=cci_node,
        claim=f"In January 2026, the individual named '{ex.individual_name}' was named the 10th President of '{ex.presidential_institution}', with the position effective July 1, 2026.",
        urls=ex.presidency_urls,
        base_instruction="Confirm the institution name, the ordinal (10th President), the appointment timing (January 2026), and the effective date (July 1, 2026).",
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Build the full verification tree and evaluate the agent's answer.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall rubric root operates in parallel as per spec
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

    # Extraction
    ex: CareerExtraction = await evaluator.extract(
        prompt=prompt_extract_career_info(),
        template_class=CareerExtraction,
        extraction_name="career_extraction",
    )

    # Build tree and verify
    await build_and_verify_career_tree(evaluator, root, ex)

    # Return structured summary
    return evaluator.get_summary()