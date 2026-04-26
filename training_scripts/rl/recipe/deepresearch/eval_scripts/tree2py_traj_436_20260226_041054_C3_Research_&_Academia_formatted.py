import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "artemis2_first_person_of_color"
TASK_DESCRIPTION = """NASA's Artemis II mission, scheduled to launch in early 2026, will carry four astronauts on a 10-day journey around the Moon aboard the Orion spacecraft named 'Integrity.' Among the crew is an astronaut who will make history as the first person of color to travel around the Moon.

Identify this crew member and provide the following information:
1. Their full name
2. Their official crew position on the Artemis II mission
3. The specific spacecraft handling task they will perform during the high Earth orbit phase (involving the spent Interim Cryogenic Propulsion Stage)
4. Whether this is their first or second spaceflight
5. A reference URL from an official NASA or space agency source confirming this information

Your answer must be grounded in official mission documentation.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CrewAnswerExtraction(BaseModel):
    """Fields we expect the agent to provide in the answer."""
    name: Optional[str] = None
    crew_position: Optional[str] = None
    icps_task: Optional[str] = None
    spaceflight_experience: Optional[str] = None  # e.g., "second spaceflight", "first", "second", etc.
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return """
Extract the following fields from the answer text. Return exactly these fields in JSON:
- name: The full name of the specific Artemis II crew member identified as the first person of color to travel around the Moon.
- crew_position: The astronaut’s official crew position on Artemis II (e.g., "Pilot", "Commander", "Mission Specialist").
- icps_task: The exact phrase(s) describing the specific spacecraft handling task in high Earth orbit that involves the spent Interim Cryogenic Propulsion Stage (ICPS). If multiple phrasings are provided, include the most detailed one from the answer.
- spaceflight_experience: Whether this is the astronaut’s "first" or "second" spaceflight, as stated in the answer. Use the exact wording from the answer if available (e.g., "second spaceflight", "second", etc.). If unclear or not stated, return null.
- urls: A list of all URLs explicitly included in the answer (include NASA or other space agency links if present). Extract the full URLs as they appear (plain or markdown). Do not invent any URLs.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
OFFICIAL_DOMAINS = [
    "nasa.gov",          # NASA (covers www.nasa.gov, blogs.nasa.gov, jpl.nasa.gov, etc.)
    "asc-csa.gc.ca",     # Canadian Space Agency (English/French domain)
    "csa-asc.gc.ca",     # Alternate CSA bilingual pattern (allow leniency)
    "esa.int",           # European Space Agency
    "jaxa.jp",           # JAXA
    "isro.gov.in",       # ISRO
    "dlr.de",            # DLR (German Aerospace Center)
    "cnsa.gov.cn",       # CNSA
]


def _get_domain(url: str) -> str:
    try:
        parsed = urlparse(url if (url.startswith("http://") or url.startswith("https://")) else f"http://{url}")
        return (parsed.netloc or "").lower()
    except Exception:
        return ""


def is_official_space_agency_url(url: str) -> bool:
    dom = _get_domain(url)
    return any(dom.endswith(d) for d in OFFICIAL_DOMAINS)


def filter_official_urls(urls: List[str]) -> List[str]:
    seen = set()
    official = []
    for u in urls:
        if u and u not in seen and is_official_space_agency_url(u):
            official.append(u)
            seen.add(u)
    return official


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_reference_nodes(evaluator: Evaluator, parent) -> Dict[str, Any]:
    """
    Create the Reference_Documentation subtree with:
    - URL_Provided (critical, custom)
    - URL_Validity (critical, custom)
    Returns a dict with references to the two leaves for dependency wiring.
    """
    ref_parent = evaluator.add_parallel(
        id="reference_documentation",
        desc="The answer provides at least one valid reference URL from an official source",
        parent=parent,
        critical=False
    )
    return {"parent": ref_parent}


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    Evaluate an answer for the Artemis II crew identification and related details.
    """
    # Initialize evaluator (root is parallel; non-critical to allow partial credit)
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

    # 1) Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=CrewAnswerExtraction,
        extraction_name="extracted_fields"
    )

    # Record some custom info for debugging
    evaluator.add_custom_info(
        {
            "extracted_name": extracted.name,
            "extracted_position": extracted.crew_position,
            "extracted_icps_task": extracted.icps_task,
            "extracted_experience": extracted.spaceflight_experience,
            "extracted_urls_count": len(extracted.urls),
        },
        info_type="extraction_overview"
    )

    # Compute official URLs
    official_urls = filter_official_urls(extracted.urls)
    evaluator.add_custom_info(
        {
            "official_urls": official_urls,
            "all_urls": extracted.urls
        },
        info_type="urls_info"
    )

    # ------------------------------------------------------------------- #
    # Build top-level categories as per rubric                            #
    # ------------------------------------------------------------------- #

    # A) Astronaut_Identification
    ident_parent = evaluator.add_parallel(
        id="astronaut_identification",
        desc="The answer correctly identifies the crew member who will become the first person of color to travel around the Moon on Artemis II",
        parent=root,
        critical=False
    )
    # Leaf: Name_Provided (existence check)
    name_provided_leaf = evaluator.add_custom_node(
        result=bool(extracted.name and extracted.name.strip()),
        id="name_provided",
        desc="The answer provides the full name of the astronaut",
        parent=ident_parent,
        critical=True
    )

    # B) Position_Information
    pos_parent = evaluator.add_parallel(
        id="position_information",
        desc="The answer correctly states the astronaut's official crew position on the Artemis II mission",
        parent=root,
        critical=False
    )
    # Leaf: Position_Stated (check the answer explicitly states 'Pilot')
    pos_stated_leaf = evaluator.add_leaf(
        id="position_stated",
        desc="The crew position (Pilot) is explicitly stated in the answer",
        parent=pos_parent,
        critical=True
    )
    # Verify via simple check on the answer content (LLM sees the full answer in context)
    await evaluator.verify(
        claim="In the provided answer, the astronaut's crew position on Artemis II is explicitly stated as 'Pilot' (accept reasonable wording like 'Orion pilot').",
        node=pos_stated_leaf,
        sources=None,
        additional_instruction="Focus only on the answer text above. Do not accept 'Commander' or 'Mission Specialist'. Accept variants like 'pilot of Orion' or 'Artemis II pilot'."
    )

    # C) Mission_Task_Information
    task_parent = evaluator.add_parallel(
        id="mission_task_information",
        desc="The answer correctly identifies the specific spacecraft handling task involving the ICPS",
        parent=root,
        critical=False
    )

    # We restructure the JSON's Task_Description (with a child) into a small parallel group
    task_group = evaluator.add_parallel(
        id="task_description_group",
        desc="Task description for the proximity operations with the spent ICPS",
        parent=task_parent,
        critical=False
    )

    # Leaf: Task_Description (critical) - Verify against official sources (preferably)
    task_desc_leaf = evaluator.add_leaf(
        id="task_desc_icps_prox_ops",
        desc="The answer describes the proximity operations task with the spent Interim Cryogenic Propulsion Stage",
        parent=task_group,
        critical=True
    )

    # Reference Documentation parent (we'll add after creating all leaves to wire dependencies)
    ref_nodes = build_reference_nodes  # to avoid forward reference confusion

    # D) Spaceflight_Experience
    exp_parent = evaluator.add_parallel(
        id="spaceflight_experience",
        desc="The answer provides accurate information about the astronaut's spaceflight experience",
        parent=root,
        critical=False
    )
    exp_stated_leaf = evaluator.add_leaf(
        id="experience_stated_second",
        desc="The answer correctly indicates this is the astronaut's second spaceflight",
        parent=exp_parent,
        critical=True
    )

    # E) Reference_Documentation
    reference_parent = evaluator.add_parallel(
        id="reference_documentation",
        desc="The answer provides at least one valid reference URL from an official source",
        parent=root,
        critical=False
    )

    # Leaf: URL_Provided (critical)
    url_provided_leaf = evaluator.add_custom_node(
        result=bool(extracted.urls and len(extracted.urls) > 0),
        id="url_provided",
        desc="At least one URL is included in the answer",
        parent=reference_parent,
        critical=True
    )

    # Leaf: URL_Validity (critical) - check at least one is from official NASA or space agency domain
    url_validity_leaf = evaluator.add_custom_node(
        result=bool(official_urls),
        id="url_validity",
        desc="At least one provided URL is from an official NASA or space agency source",
        parent=reference_parent,
        critical=True
    )

    # ------------------------------------------------------------------- #
    # Now perform the verifications that require sources                  #
    # ------------------------------------------------------------------- #

    # Task_Description verified by official sources (if provided and valid)
    # We require the URL_Provided and URL_Validity leaves as prerequisites to ensure source-grounding.
    await evaluator.verify(
        claim=(
            "Artemis II includes a proximity operations demonstration in high Earth orbit where Orion conducts "
            "relative navigation and formation-flying maneuvers with the spent Interim Cryogenic Propulsion Stage (ICPS) "
            "to evaluate Orion's handling qualities."
        ),
        node=task_desc_leaf,
        sources=official_urls if official_urls else extracted.urls,
        extra_prerequisites=[url_provided_leaf, url_validity_leaf, name_provided_leaf],
        additional_instruction=(
            "Accept wording such as 'proximity operations demonstration', 'proximity ops', 'relative navigation', "
            "'formation flying', or 'handling qualities' with the (spent) ICPS. The core idea must be that Orion "
            "approaches/tracks the jettisoned ICPS in high Earth orbit to evaluate handling/relative navigation."
        )
    )

    # Task_Detail (non-critical): confirm the answer mentions evaluation of handling qualities or formation flying aspects
    task_detail_leaf = evaluator.add_leaf(
        id="task_detail_handling",
        desc="The answer mentions the evaluation of Orion's handling qualities or the maneuvering/formation flying aspects",
        parent=task_group,
        critical=False
    )
    await evaluator.verify(
        claim=(
            "In the provided answer text, the task description explicitly mentions evaluating Orion's handling qualities "
            "or uses closely related phrasing such as 'manual piloting', 'formation flying', or 'relative navigation maneuvers'."
        ),
        node=task_detail_leaf,
        sources=None,
        additional_instruction="Focus only on the answer text; do not use external knowledge."
    )

    # Experience_Stated (critical): verify with official sources that it is the astronaut's second spaceflight
    await evaluator.verify(
        claim=(
            f"It is {extracted.name}'s second spaceflight (they have previously flown once before) as part of Artemis II."
            if extracted.name else
            "It is the astronaut's second spaceflight as part of Artemis II."
        ),
        node=exp_stated_leaf,
        sources=official_urls if official_urls else extracted.urls,
        extra_prerequisites=[url_provided_leaf, url_validity_leaf, name_provided_leaf],
        additional_instruction=(
            "Confirm that the identified astronaut will be on their second spaceflight on Artemis II. "
            "Minor wording differences are fine (e.g., 'second mission')."
        )
    )

    # ------------------------------------------------------------------- #
    # Optional: Add ground-truth context (non-enforced)                   #
    # ------------------------------------------------------------------- #
    evaluator.add_ground_truth(
        {
            "expected_role_for_target_astronaut": "Pilot",
            "expected_spaceflight_experience": "Second spaceflight",
            "task_keyword": "Proximity operations demonstration with spent ICPS to evaluate handling qualities"
        },
        gt_type="expected_facts"
    )

    # Return the final structured summary
    return evaluator.get_summary()