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
TASK_ID = "multi_state_edu_info"
TASK_DESCRIPTION = """
Provide comprehensive educational information for two specific requirements:

First, list all credit requirements for high school graduation in Pennsylvania, including: (1) the total number of credits required, and (2) the specific number of credits required in each of the following subject areas: English, Mathematics, Science, Social Studies, and Arts or Humanities.

Second, provide a complete operational profile of Frisco Independent School District, including: (1) the total number of schools in the district, (2) the number of elementary schools, (3) the number of middle schools, (4) the number of high schools, (5) the number of trustees serving on the district's Board of Trustees, (6) the district's four-year graduation rate, and (7) the U.S. state in which the district is located.

For each piece of information provided, include a reference URL from an official or authoritative source.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PARequirements(BaseModel):
    total_credits: Optional[str] = None
    english_credits: Optional[str] = None
    mathematics_credits: Optional[str] = None
    science_credits: Optional[str] = None
    social_studies_credits: Optional[str] = None
    arts_humanities_credits: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FriscoProfile(BaseModel):
    total_schools: Optional[str] = None
    elementary_schools: Optional[str] = None
    middle_schools: Optional[str] = None
    high_schools: Optional[str] = None
    trustees_count: Optional[str] = None
    graduation_rate: Optional[str] = None
    state: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_pa_requirements() -> str:
    return """
Extract the Pennsylvania high school graduation credit requirements exactly as stated in the answer text.

Return a JSON object with:
- total_credits: The total number of credits required for Pennsylvania high school graduation as presented (keep as a string exactly as written, e.g., "22", "22 credits", or "twenty-two").
- english_credits: The number of English credits (string, as written).
- mathematics_credits: The number of Mathematics credits (string, as written).
- science_credits: The number of Science credits (string, as written).
- social_studies_credits: The number of Social Studies credits (string, as written).
- arts_humanities_credits: The number of Arts or Humanities credits (string, as written).
- sources: An array of all URL(s) explicitly provided in the answer that support Pennsylvania graduation requirements (extract real URLs only).

Rules:
- Extract exactly what the answer states. Do not infer or invent.
- If any field is missing from the answer, set it to null.
- For sources, include all URLs that are meant to support Pennsylvania graduation requirements.
"""


def prompt_extract_frisco_profile() -> str:
    return """
Extract the operational profile for Frisco Independent School District (Frisco ISD) exactly as stated in the answer text.

Return a JSON object with:
- total_schools: The total number of schools (string, as written).
- elementary_schools: The number of elementary schools (string, as written).
- middle_schools: The number of middle schools (string, as written).
- high_schools: The number of high schools (string, as written).
- trustees_count: The number of trustees on the Board of Trustees (string, as written).
- graduation_rate: The four-year graduation rate (string, as written, e.g., "98%", "98.0%", or "98 percent").
- state: The U.S. state where Frisco ISD is located (string, as written, e.g., "Texas").
- sources: An array of all URL(s) explicitly provided in the answer that support Frisco ISD information (extract real URLs only).

Rules:
- Extract exactly what the answer states. Do not infer or invent.
- If any field is missing from the answer, set it to null.
- For sources, include all URLs that are meant to support Frisco ISD statistics.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _ensure_list(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_pa_requirements(
    evaluator: Evaluator,
    parent_node,
    pa: PARequirements
) -> None:
    """
    Build and verify the Pennsylvania graduation requirements subtree.
    """
    pa_node = evaluator.add_parallel(
        id="Pennsylvania_Graduation_Requirements",
        desc="Provide all required credit information for Pennsylvania high school graduation",
        parent=parent_node,
        critical=False
    )

    pa_sources = _ensure_list(pa.sources)

    # Critical existence check for source URL(s)
    evaluator.add_custom_node(
        result=len(pa_sources) > 0,
        id="PA_Requirements_Source_URL",
        desc="Provide a valid reference URL supporting Pennsylvania graduation requirements",
        parent=pa_node,
        critical=True
    )

    # Total credits
    node_total = evaluator.add_leaf(
        id="PA_Total_Credits",
        desc="State the total number of credits required for high school graduation in Pennsylvania",
        parent=pa_node,
        critical=True
    )
    claim_total = f"The total number of credits required for high school graduation in Pennsylvania is '{pa.total_credits}'."
    await evaluator.verify(
        claim=claim_total,
        node=node_total,
        sources=pa_sources,
        additional_instruction="Confirm that the page explicitly states the statewide Pennsylvania high school graduation total credit requirement. Accept minor formatting variations (e.g., 'credit(s)', 'unit(s)'). If the page is district-specific, it must explicitly indicate statewide requirements to support the claim."
    )

    # English credits
    node_eng = evaluator.add_leaf(
        id="PA_English_Credits",
        desc="State the number of English credits required for graduation in Pennsylvania",
        parent=pa_node,
        critical=True
    )
    claim_eng = f"The required English credits for Pennsylvania high school graduation is '{pa.english_credits}'."
    await evaluator.verify(
        claim=claim_eng,
        node=node_eng,
        sources=pa_sources,
        additional_instruction="Verify the page shows the required number of English credits for Pennsylvania high school graduation. Allow 'units' vs 'credits'."
    )

    # Mathematics credits
    node_math = evaluator.add_leaf(
        id="PA_Mathematics_Credits",
        desc="State the number of Mathematics credits required for graduation in Pennsylvania",
        parent=pa_node,
        critical=True
    )
    claim_math = f"The required Mathematics credits for Pennsylvania high school graduation is '{pa.mathematics_credits}'."
    await evaluator.verify(
        claim=claim_math,
        node=node_math,
        sources=pa_sources,
        additional_instruction="Verify the page shows the required number of Mathematics credits for Pennsylvania high school graduation. Allow 'units' vs 'credits'."
    )

    # Science credits
    node_sci = evaluator.add_leaf(
        id="PA_Science_Credits",
        desc="State the number of Science credits required for graduation in Pennsylvania",
        parent=pa_node,
        critical=True
    )
    claim_sci = f"The required Science credits for Pennsylvania high school graduation is '{pa.science_credits}'."
    await evaluator.verify(
        claim=claim_sci,
        node=node_sci,
        sources=pa_sources,
        additional_instruction="Verify the page shows the required number of Science credits for Pennsylvania high school graduation. Allow 'units' vs 'credits'."
    )

    # Social Studies credits
    node_soc = evaluator.add_leaf(
        id="PA_Social_Studies_Credits",
        desc="State the number of Social Studies credits required for graduation in Pennsylvania",
        parent=pa_node,
        critical=True
    )
    claim_soc = f"The required Social Studies credits for Pennsylvania high school graduation is '{pa.social_studies_credits}'."
    await evaluator.verify(
        claim=claim_soc,
        node=node_soc,
        sources=pa_sources,
        additional_instruction="Verify the page shows the required number of Social Studies credits for Pennsylvania high school graduation. Allow 'units' vs 'credits' and inclusive phrasing (e.g., 'social studies including civics')."
    )

    # Arts/Humanities credits
    node_arts = evaluator.add_leaf(
        id="PA_Arts_Humanities_Credits",
        desc="State the number of Arts or Humanities credits required for graduation in Pennsylvania",
        parent=pa_node,
        critical=True
    )
    claim_arts = f"The required Arts or Humanities credits for Pennsylvania high school graduation is '{pa.arts_humanities_credits}'."
    await evaluator.verify(
        claim=claim_arts,
        node=node_arts,
        sources=pa_sources,
        additional_instruction="Verify the page shows the required number of Arts or Humanities credits for Pennsylvania high school graduation. Allow 'units' vs 'credits' and wording variants like 'arts/humanities'."
    )


async def verify_frisco_profile(
    evaluator: Evaluator,
    parent_node,
    frisco: FriscoProfile
) -> None:
    """
    Build and verify the Frisco ISD comprehensive profile subtree.
    """
    frisco_node = evaluator.add_parallel(
        id="Frisco_ISD_Comprehensive_Profile",
        desc="Provide complete operational statistics and information about Frisco Independent School District",
        parent=parent_node,
        critical=False
    )

    frisco_sources = _ensure_list(frisco.sources)

    # Critical existence check for source URL(s)
    evaluator.add_custom_node(
        result=len(frisco_sources) > 0,
        id="Frisco_ISD_Source_URL",
        desc="Provide a valid reference URL supporting Frisco ISD information",
        parent=frisco_node,
        critical=True
    )

    # Total schools
    node_total = evaluator.add_leaf(
        id="Frisco_Total_Schools",
        desc="State the total number of schools operated by Frisco ISD",
        parent=frisco_node,
        critical=True
    )
    claim_total = f"The total number of schools operated by Frisco Independent School District is '{frisco.total_schools}'."
    await evaluator.verify(
        claim=claim_total,
        node=node_total,
        sources=frisco_sources,
        additional_instruction="Verify the page states the total number of schools/campuses in Frisco ISD. Accept synonyms like 'campuses'."
    )

    # Elementary schools
    node_elem = evaluator.add_leaf(
        id="Frisco_Elementary_Schools",
        desc="State the number of elementary schools in Frisco ISD",
        parent=frisco_node,
        critical=True
    )
    claim_elem = f"Frisco ISD has '{frisco.elementary_schools}' elementary schools."
    await evaluator.verify(
        claim=claim_elem,
        node=node_elem,
        sources=frisco_sources,
        additional_instruction="Verify the page lists the number of elementary schools (or campuses) in Frisco ISD. Allow minor wording differences."
    )

    # Middle schools
    node_mid = evaluator.add_leaf(
        id="Frisco_Middle_Schools",
        desc="State the number of middle schools in Frisco ISD",
        parent=frisco_node,
        critical=True
    )
    claim_mid = f"Frisco ISD has '{frisco.middle_schools}' middle schools."
    await evaluator.verify(
        claim=claim_mid,
        node=node_mid,
        sources=frisco_sources,
        additional_instruction="Verify the page lists the number of middle schools (or campuses) in Frisco ISD. Allow minor wording differences."
    )

    # High schools
    node_high = evaluator.add_leaf(
        id="Frisco_High_Schools",
        desc="State the number of high schools in Frisco ISD",
        parent=frisco_node,
        critical=True
    )
    claim_high = f"Frisco ISD has '{frisco.high_schools}' high schools."
    await evaluator.verify(
        claim=claim_high,
        node=node_high,
        sources=frisco_sources,
        additional_instruction="Verify the page lists the number of high schools (or campuses) in Frisco ISD. Allow minor wording differences."
    )

    # Trustees
    node_trustees = evaluator.add_leaf(
        id="Frisco_Board_Trustees",
        desc="State the number of trustees on the Frisco ISD Board of Trustees",
        parent=frisco_node,
        critical=True
    )
    claim_trustees = f"The Frisco ISD Board of Trustees consists of '{frisco.trustees_count}' trustees."
    await evaluator.verify(
        claim=claim_trustees,
        node=node_trustees,
        sources=frisco_sources,
        additional_instruction="Verify the page states the number of board members/trustees for Frisco ISD. Accept synonyms like 'board members' or 'trustees'."
    )

    # Graduation rate
    node_grad = evaluator.add_leaf(
        id="Frisco_Graduation_Rate",
        desc="State the four-year graduation rate for Frisco ISD",
        parent=frisco_node,
        critical=True
    )
    claim_grad = f"Frisco ISD's four-year graduation rate is '{frisco.graduation_rate}'."
    await evaluator.verify(
        claim=claim_grad,
        node=node_grad,
        sources=frisco_sources,
        additional_instruction="Verify the page shows the four-year graduation rate for Frisco ISD. Allow rounding (e.g., 98% vs 98.0%)."
    )

    # State location
    node_state = evaluator.add_leaf(
        id="Frisco_State_Location",
        desc="Identify the U.S. state where Frisco ISD is located",
        parent=frisco_node,
        critical=True
    )
    claim_state = f"Frisco Independent School District is located in the U.S. state of '{frisco.state}'."
    await evaluator.verify(
        claim=claim_state,
        node=node_state,
        sources=frisco_sources,
        additional_instruction="Verify the page clearly indicates Frisco ISD is in the specified U.S. state."
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
    Evaluate an answer for the Multi-State Educational Information Task.
    """
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

    # Extract both sections (can run concurrently)
    pa_extraction_task = evaluator.extract(
        prompt=prompt_extract_pa_requirements(),
        template_class=PARequirements,
        extraction_name="pa_requirements"
    )
    frisco_extraction_task = evaluator.extract(
        prompt=prompt_extract_frisco_profile(),
        template_class=FriscoProfile,
        extraction_name="frisco_profile"
    )
    pa_req, frisco_prof = await asyncio.gather(pa_extraction_task, frisco_extraction_task)

    # Top-level aggregation node as in rubric (parallel, non-critical children)
    top_node = evaluator.add_parallel(
        id="Multi-State_Educational_Information_Task",
        desc="Provide comprehensive educational information including Pennsylvania high school graduation requirements and complete operational profile of Frisco Independent School District in Texas",
        parent=root,
        critical=False
    )

    # Build and verify subtrees
    await verify_pa_requirements(evaluator, top_node, pa_req or PARequirements())
    await verify_frisco_profile(evaluator, top_node, frisco_prof or FriscoProfile())

    return evaluator.get_summary()