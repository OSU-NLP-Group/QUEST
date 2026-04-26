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
TASK_ID = "chips_megafab_ny_dram"
TASK_DESCRIPTION = """
A major semiconductor fabrication megafab project was announced in the United States with the following characteristics:

• Located in New York State
• Received CHIPS Act direct funding between $6.0 billion and $6.5 billion
• Represents a total private investment commitment of at least $90 billion over multiple decades
• Focuses specifically on leading-edge DRAM (Dynamic Random-Access Memory) chip production, not logic chips
• The initial construction phase includes building at least two high-volume manufacturing (HVM) fabrication facilities
• Each individual fabrication facility features approximately 600,000 square feet of cleanroom space
• The project has been officially described as creating "the largest amount of cleanroom space ever announced in the United States"
• Owned and operated by a U.S.-based semiconductor manufacturer
• Has received a "Final Award" status from the CHIPS Program Office

Identify the company developing this megafab project and specify the town or city in New York where it is located.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ProjectExtraction(BaseModel):
    company: Optional[str] = None
    city_or_town: Optional[str] = None
    state: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_project() -> str:
    return """
    From the answer, extract the following fields about the described megafab project:

    - company: The name of the U.S.-based semiconductor manufacturer developing/operating the project.
    - city_or_town: The specific town or city in New York where the project is located (e.g., "Clay", "Syracuse", etc.).
    - state: If a U.S. state is specified with the location, extract it (expect "New York"). Otherwise return null.
    - sources: Collect ALL URLs explicitly mentioned in the answer that are intended as supporting sources for this project and its attributes. 
               Include official pages (e.g., nist.gov, chips.gov, commerce.gov), company press releases, and relevant news articles, as long as they are explicitly cited in the answer.

    Rules:
    - Extract only what is explicitly present in the answer text.
    - Return null for missing fields.
    - For sources, include absolute URLs only. If a URL lacks protocol, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Utility                                                                     #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    results = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            u = "http://" + u
        if u not in seen:
            seen.add(u)
            results.append(u)
    return results


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_project(evaluator: Evaluator, parent_node, extraction: ProjectExtraction) -> None:
    """
    Build the verification tree under the critical 'Project_Identification' node and perform verifications.
    """
    # Normalize sources list (can be empty)
    sources = _dedup_urls(extraction.sources)

    # Create the critical parent node for this rubric
    project_node = evaluator.add_parallel(
        id="Project_Identification",
        desc="Identify the company and the New York town/city for the megafab project that satisfies all stated constraints.",
        parent=parent_node,
        critical=True
    )

    # 1) Answer must provide company
    evaluator.add_custom_node(
        result=bool(extraction.company and extraction.company.strip()),
        id="Answer_Provides_Company",
        desc="The response identifies the company developing/operating the project.",
        parent=project_node,
        critical=True
    )

    # 2) Answer must provide town or city
    evaluator.add_custom_node(
        result=bool(extraction.city_or_town and extraction.city_or_town.strip()),
        id="Answer_Provides_Town_or_City",
        desc="The response specifies the town or city in New York where the project is located.",
        parent=project_node,
        critical=True
    )

    # Prepare claims for constraints
    company = extraction.company or "the identified company"
    place = extraction.city_or_town or "the identified town or city"
    state = extraction.state or "New York"

    # Create all leaf nodes first
    node_loc = evaluator.add_leaf(
        id="Constraint_Project_Located_in_New_York_State",
        desc="The identified project is located in New York State.",
        parent=project_node,
        critical=True
    )

    node_chips_funding = evaluator.add_leaf(
        id="Constraint_CHIPS_Direct_Funding_Range",
        desc="The identified project received CHIPS Act direct funding between $6.0B and $6.5B (inclusive).",
        parent=project_node,
        critical=True
    )

    node_private_invest = evaluator.add_leaf(
        id="Constraint_Private_Investment_At_Least_90B",
        desc="The identified project represents at least $90B in total private investment commitment.",
        parent=project_node,
        critical=True
    )

    node_multi_decades = evaluator.add_leaf(
        id="Constraint_Investment_Multiple_Decades",
        desc="The identified project is described as a commitment spanning multiple decades.",
        parent=project_node,
        critical=True
    )

    node_dram = evaluator.add_leaf(
        id="Constraint_Technology_Focus_Leading_Edge_DRAM_Not_Logic",
        desc="The identified project focuses specifically on leading-edge DRAM chip production (and not logic chips).",
        parent=project_node,
        critical=True
    )

    node_two_hvm = evaluator.add_leaf(
        id="Constraint_Initial_Phase_At_Least_Two_HVM_Fabs",
        desc="The initial construction phase includes building at least two high-volume manufacturing (HVM) fabrication facilities.",
        parent=project_node,
        critical=True
    )

    node_cleanroom_size = evaluator.add_leaf(
        id="Constraint_Cleanroom_Approx_600k_SqFt_Per_Fab",
        desc="Each individual fabrication facility features approximately 600,000 square feet of cleanroom space.",
        parent=project_node,
        critical=True
    )

    node_largest_cleanroom = evaluator.add_leaf(
        id="Constraint_Largest_Cleanroom_Space_Claim_US",
        desc='The project has been officially described as creating the largest amount of cleanroom space ever announced in the United States.',
        parent=project_node,
        critical=True
    )

    node_us_based_owner = evaluator.add_leaf(
        id="Constraint_US_Based_Semiconductor_Manufacturer_Owner_Operator",
        desc="The project is owned and operated by a U.S.-based semiconductor manufacturer.",
        parent=project_node,
        critical=True
    )

    node_final_award = evaluator.add_leaf(
        id="Constraint_Final_Award_Status",
        desc='The project has received "Final Award" status from the CHIPS Program Office (as documented by NIST per the constraints).',
        parent=project_node,
        critical=True
    )

    # Build claims and additional instructions
    claim_loc = (
        f"The megafab project by {company} is located in {place}, {state}, United States."
        if extraction.city_or_town else
        f"The megafab project by {company} is located in New York State, United States."
    )
    addins_loc = (
        "Confirm that the project site is in New York State. If a town or city is mentioned "
        "in the answer (e.g., Clay or Syracuse), confirm it is in New York."
    )

    claim_funding = (
        "The CHIPS Program Office awarded direct funding between $6.0 billion and $6.5 billion (inclusive) for this specific megafab project, "
        "for example approximately $6.1 billion."
    )
    addins_funding = (
        "Look specifically for CHIPS Act 'direct funding' (grant) amounts for this project. "
        "Treat figures such as '$6.1B' or 'approximately $6.1 billion' as within range. "
        "If the amount is outside $6.0B–$6.5B, mark as not supported."
    )

    claim_private_invest = (
        "The total private investment commitment for the project is at least $90 billion (e.g., $100B) over the project's lifetime."
    )
    addins_private_invest = (
        "Look for phrases like '$90 billion', '$100 billion', 'at least $90B', or similar. "
        "Any figure >= $90B qualifies. If lower than $90B, fail."
    )

    claim_multi_decades = (
        "The project is described as a multi-decade commitment, such as 'over the next 20 years' or 'over multiple decades'."
    )
    addins_multi_decades = (
        "Look for language indicating the timeline spans multiple decades (e.g., 'over 20 years', 'multi-decade', 'over decades')."
    )

    claim_dram = (
        "The New York megafab focuses on leading-edge DRAM (Dynamic Random-Access Memory) manufacturing and is a memory fab, not a logic chip fab."
    )
    addins_dram = (
        "Confirm that the project is for DRAM production (memory). If the page emphasizes logic chips or foundry logic, mark as not supported."
    )

    claim_two_hvm = (
        "The initial construction phase of the project includes at least two high-volume manufacturing (HVM) fabrication facilities."
    )
    addins_two_hvm = (
        "Look for statements like 'two fabs', 'two facilities', or similar in the initial phase."
    )

    claim_cleanroom = (
        "Each individual fabrication facility (each fab building) will feature approximately 600,000 square feet of cleanroom space."
    )
    addins_cleanroom = (
        "Accept approximate sizes (e.g., 'around 600,000 sq. ft.'). The key is per-fab cleanroom area near 600k sq ft."
    )

    claim_largest_cleanroom = (
        'The project has been officially described as creating the largest amount of cleanroom space ever announced in the United States.'
    )
    addins_largest_cleanroom = (
        "Look for official phrasing indicating it is the largest cleanroom space ever announced in the U.S., or a direct equivalent."
    )

    claim_us_based_owner = (
        f"The project is owned and operated by {company}, which is a U.S.-based semiconductor manufacturer."
        if extraction.company else
        "The project is owned and operated by a U.S.-based semiconductor manufacturer."
    )
    addins_us_based_owner = (
        "Verify that the company is U.S.-based (headquartered in the United States) and that it owns and operates this project."
    )

    claim_final_award = (
        'The CHIPS Program Office (NIST/CHIPS for America) has issued a "Final Award" for this project.'
    )
    addins_final_award = (
        "This verification should be supported by an official U.S. government source (e.g., nist.gov or chips.gov). "
        "If the provided URLs are not official NIST/CHIPS domains and do not contain explicit 'Final Award' language, treat as not supported."
    )

    # Prepare batch verifications
    verifications = [
        (claim_loc, sources, node_loc, addins_loc),
        (claim_funding, sources, node_chips_funding, addins_funding),
        (claim_private_invest, sources, node_private_invest, addins_private_invest),
        (claim_multi_decades, sources, node_multi_decades, addins_multi_decades),
        (claim_dram, sources, node_dram, addins_dram),
        (claim_two_hvm, sources, node_two_hvm, addins_two_hvm),
        (claim_cleanroom, sources, node_cleanroom_size, addins_cleanroom),
        (claim_largest_cleanroom, sources, node_largest_cleanroom, addins_largest_cleanroom),
        (claim_us_based_owner, sources, node_us_based_owner, addins_us_based_owner),
        (claim_final_award, sources, node_final_award, addins_final_award),
    ]

    # Execute all verifications in parallel
    await evaluator.batch_verify(verifications)


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
    Evaluate an answer for the CHIPS megafab in New York (leading-edge DRAM) identification task.
    """
    # Initialize evaluator
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_project(),
        template_class=ProjectExtraction,
        extraction_name="project_extraction"
    )

    # Record some custom info for debugging
    evaluator.add_custom_info(
        {
            "extracted_company": extraction.company,
            "extracted_city_or_town": extraction.city_or_town,
            "extracted_state": extraction.state,
            "num_sources": len(extraction.sources or []),
        },
        info_type="extraction_summary",
        info_name="extraction_overview"
    )

    # Build verification nodes and run checks
    await build_and_verify_project(evaluator, root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()