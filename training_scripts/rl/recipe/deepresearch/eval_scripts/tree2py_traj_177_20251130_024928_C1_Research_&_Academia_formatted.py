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
TASK_ID = "cu_boulder_noaa_coop_institute"
TASK_DESCRIPTION = """
I am researching university partnerships with federal agencies in environmental sciences. Identify the cooperative institute at the University of Colorado Boulder that partners with NOAA (National Oceanic and Atmospheric Administration) for environmental sciences research. Provide the full official name of the institute and state the year it was established.
"""

EXPECTED_INSTITUTE_NAME_HINT = "Cooperative Institute for Research in Environmental Sciences (CIRES)"
EXPECTED_FOUNDING_YEAR = "1967"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class InstituteExtraction(BaseModel):
    """
    Extracted fields from the answer.
    - institute_name: The full official institute name as presented in the answer (spelled out, not just acronym).
    - founding_year: The year the institute was established, as stated in the answer (e.g., "1967").
    - source_urls: Any URLs cited in the answer that are about the institute (official institute/NOAA/CU Boulder pages, press releases, about pages),
                   or that explicitly state the institute’s official name, NOAA partnership, affiliation with CU Boulder, environmental science focus, or founding year.
    """
    institute_name: Optional[str] = None
    founding_year: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_institute() -> str:
    return """
    Your job is to extract from the answer the cooperative institute associated with NOAA based at the University of Colorado Boulder,
    along with the year it was established, and any URLs the answer provides as evidence.

    Extract the following fields exactly as they appear in the answer text:
    1) institute_name:
       - The full official name of the institute as stated in the answer (spelled out, not just an acronym).
       - If the answer only shows an acronym (e.g., "CIRES") and does NOT spell it out, return that acronym as-is (do NOT invent or expand).
       - If multiple forms are present, prefer the fully spelled-out official version.
    2) founding_year:
       - The establishment/founding year explicitly stated in the answer for the institute.
       - Return only the 4-digit year string (e.g., "1967"). If missing, return null.
    3) source_urls:
       - All URLs in the answer that are specifically about the institute or used as evidence for its name, NOAA partnership, CU Boulder affiliation,
         environmental sciences/Earth system focus, or founding year.
       - Include official institute pages, NOAA pages, or CU Boulder pages, plus any other cited sources relevant to this institute.
       - Return an array of valid, complete URLs. If none are present, return an empty array.

    Do NOT invent or infer any information that is not present in the answer.
    If any requested field is missing in the answer, return null for that field (or empty array for URLs).
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_institute(evaluator: Evaluator, parent_node, extracted: InstituteExtraction) -> None:
    """
    Build the verification tree based on the rubric and run checks.
    """
    # Top-level critical parallel node matching the rubric root
    inst_root = evaluator.add_parallel(
        id="Institute_Identification",
        desc="Identifies the cooperative institute at the University of Colorado Boulder that partners with NOAA for environmental sciences research, and provides required details.",
        parent=parent_node,
        critical=True
    )

    # Prepare common variables
    name = extracted.institute_name or ""
    year = (extracted.founding_year or "").strip()
    sources = extracted.source_urls if extracted.source_urls else []

    # 1) Institute_Affiliation_CU_Boulder (Critical Leaf)
    cu_aff_node = evaluator.add_leaf(
        id="Institute_Affiliation_CU_Boulder",
        desc="Institute is affiliated with (or based at) the University of Colorado Boulder.",
        parent=inst_root,
        critical=True,
    )
    claim_aff = (
        f"The institute named '{name}' is affiliated with or based at the University of Colorado Boulder (CU Boulder)."
        if name else
        "The institute is affiliated with or based at the University of Colorado Boulder (CU Boulder)."
    )
    await evaluator.verify(
        claim=claim_aff,
        node=cu_aff_node,
        sources=sources,
        additional_instruction="Look for explicit statements such as 'based at the University of Colorado Boulder', 'affiliated with CU Boulder', "
                              "'at the University of Colorado Boulder', or similar phrasing."
    )

    # 2) NOAA_Partnership (Critical Leaf)
    noaa_node = evaluator.add_leaf(
        id="NOAA_Partnership",
        desc="Institute is a cooperative institute/partnership that includes NOAA (National Oceanic and Atmospheric Administration) as a partner.",
        parent=inst_root,
        critical=True,
    )
    claim_noaa = (
        f"'{name}' is a NOAA cooperative institute (or joint/cooperative partnership) that includes NOAA as a partner."
        if name else
        "The institute is a NOAA cooperative institute (or joint/cooperative partnership) that includes NOAA as a partner."
    )
    await evaluator.verify(
        claim=claim_noaa,
        node=noaa_node,
        sources=sources,
        additional_instruction="Accept phrasing like 'NOAA Cooperative Institute', 'cooperative institute with NOAA', 'NOAA-funded institute', or 'joint institute with NOAA'."
    )

    # 3) Environmental_Sciences_Focus (Critical Leaf)
    env_node = evaluator.add_leaf(
        id="Environmental_Sciences_Focus",
        desc="Institute focuses on environmental sciences and/or Earth system research.",
        parent=inst_root,
        critical=True,
    )
    claim_env = (
        f"The institute '{name}' focuses on environmental sciences and/or Earth system research (e.g., Earth system science, environmental science)."
        if name else
        "The institute focuses on environmental sciences and/or Earth system research (e.g., Earth system science, environmental science)."
    )
    await evaluator.verify(
        claim=claim_env,
        node=env_node,
        sources=sources,
        additional_instruction="Look for mission statements or descriptions mentioning environmental science, Earth system science, or similar wording."
    )

    # 4) Institute_Official_Full_Name (Critical Leaf)
    name_node = evaluator.add_leaf(
        id="Institute_Official_Full_Name",
        desc="Answer includes the complete official name of the institute.",
        parent=inst_root,
        critical=True,
    )
    claim_full_name = (
        f"The institute's full official name is '{name}', not just an acronym."
        if name else
        "The institute's full official name is explicitly stated (spelled out), not just an acronym."
    )
    await evaluator.verify(
        claim=claim_full_name,
        node=name_node,
        sources=sources,
        additional_instruction=(
            "Confirm that the provided name is the official, fully spelled-out name shown on the cited pages. "
            "If the answer only uses an acronym (e.g., 'CIRES') without spelling it out ('Cooperative Institute for Research in Environmental Sciences'), "
            "treat this as not supported."
        )
    )

    # 5) Founding_Year (Critical Group -> split into two single-step leaves)
    #    We split into two leaves to ensure single-step checks:
    #    (a) The answer explicitly states 1967; (b) 1967 is supported by sources.
    fy_group = evaluator.add_sequential(
        id="Founding_Year",
        desc="States the founding/established year correctly as 1967.",
        parent=inst_root,
        critical=True
    )

    # 5a) Year stated in answer as 1967 (custom boolean check on the answer content)
    year_stated = evaluator.add_custom_node(
        result=(year != "" and "1967" in year),
        id="Founding_Year_Stated_1967",
        desc="The answer explicitly states the institute was established in 1967.",
        parent=fy_group,
        critical=True
    )

    # 5b) Year supported by sources (verification against provided URLs, if any)
    year_supported_node = evaluator.add_leaf(
        id="Founding_Year_Supported",
        desc="The institute's establishment year is 1967, supported by cited sources.",
        parent=fy_group,
        critical=True
    )
    claim_year = (
        f"The institute '{name}' was established in 1967."
        if name else
        "The institute was established in 1967."
    )
    await evaluator.verify(
        claim=claim_year,
        node=year_supported_node,
        sources=sources,
        additional_instruction="Accept synonyms like 'founded', 'established', or 'formed' in 1967."
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
    Evaluate an answer for the CU Boulder–NOAA cooperative institute identification task.
    """
    # Initialize evaluator (root is non-critical, we'll add a critical child per rubric)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall rubric is not order-dependent
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_institute(),
        template_class=InstituteExtraction,
        extraction_name="institute_extraction",
    )

    # Add ground truth info (for transparency; not used to auto-grade directly)
    evaluator.add_ground_truth({
        "expected_institute_name_hint": EXPECTED_INSTITUTE_NAME_HINT,
        "expected_established_year": EXPECTED_FOUNDING_YEAR
    })

    # Verification
    await verify_institute(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()