import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "az_mixed_use_project_screening"
TASK_DESCRIPTION = (
    "Identify the name of the mixed-use development project in Arizona that meets all of the following investment criteria: "
    "(1) total project cost of at least $500 million, "
    "(2) site size of at least 80 acres, "
    "(3) includes at least 700 residential units combining both traditional apartments and build-to-rent homes, "
    "(4) includes dedicated open space of at least 15 acres, "
    "(5) groundbreaking occurred between January 2023 and December 2024 inclusive, "
    "(6) includes a light industrial component in addition to residential and retail uses, "
    "(7) includes retail space, and "
    "(8) is being developed by a commercial real estate development company headquartered in a southeastern U.S. state."
)

PROJECT_IDENTIFICATION_DESC = "Identify a mixed-use development project in Arizona that satisfies all stated investment criteria."

SOUTHEASTERN_STATES = [
    "Alabama", "Arkansas", "Florida", "Georgia", "Kentucky", "Louisiana",
    "Mississippi", "North Carolina", "South Carolina", "Tennessee",
    "Virginia", "West Virginia"
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProjectExtraction(BaseModel):
    # Core identification
    project_name: Optional[str] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None

    # Key quantitative/qualitative program details (keep strings for flexibility)
    total_cost: Optional[str] = None                  # e.g., "$600 million", "approx. $0.8B"
    site_size_acres: Optional[str] = None             # e.g., "120 acres", "about 85 acres"
    residential_units_total: Optional[str] = None     # e.g., "700 units", "approx. 800"
    includes_apartments: Optional[str] = None         # e.g., "yes/no/unsure" as appears in text
    includes_build_to_rent: Optional[str] = None      # e.g., "yes/no/BTR/single-family rental"
    open_space_acres: Optional[str] = None            # e.g., "15 acres", "20+ acres of open space"

    groundbreaking_date: Optional[str] = None         # e.g., "Q2 2023", "November 2023"
    includes_light_industrial: Optional[str] = None   # e.g., "light industrial/flex industrial"
    includes_retail: Optional[str] = None             # e.g., "retail", "shops", "restaurant pads"

    developer_name: Optional[str] = None
    developer_type: Optional[str] = None              # e.g., "real estate developer"
    developer_hq_state: Optional[str] = None          # e.g., "Georgia", "Florida"

    # All URLs explicitly mentioned in the answer as supporting sources
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_project_info() -> str:
    return """
    Extract the single best candidate mixed-use development project in Arizona referenced in the answer that appears to satisfy the stated criteria. 
    If multiple projects are mentioned, choose the one that most closely matches all criteria and is presented as the answer's main candidate.

    Return the following fields exactly as they appear in the answer (use strings to preserve formatting/approximations). If unavailable, set to null.

    Required fields:
    - project_name: The official or commonly cited project name.
    - location_city: City or locality within Arizona, if provided.
    - location_state: State (expect "Arizona" or "AZ").
    - total_cost: Total project cost (e.g., "$600 million", "$0.7B", "approx. $550 million").
    - site_size_acres: Site size (e.g., "80 acres", "c. 120 acres").
    - residential_units_total: Total number of residential units (e.g., "700", "about 800").
    - includes_apartments: Whether the project includes traditional apartments; return a short string extracted from the answer (e.g., "apartments", "multifamily", "yes", or null if unknown).
    - includes_build_to_rent: Whether the project includes build-to-rent (BTR) homes; synonyms include "build-to-rent", "BTR", "single-family rental", "for-rent homes"; return a short string extracted (or null if unknown).
    - open_space_acres: Dedicated open space (e.g., "15 acres", "20+ acres of open space").
    - groundbreaking_date: When groundbreaking occurred (e.g., "January 2024", "late 2023", "Q1 2023").
    - includes_light_industrial: Indication of a light industrial component (e.g., "light industrial", "flex industrial", "light manufacturing").
    - includes_retail: Indication of retail (e.g., "retail", "shops", "restaurants").
    - developer_name: Developer company name.
    - developer_type: Type/description for the developer (e.g., "commercial real estate developer").
    - developer_hq_state: The developer's headquarters state, if provided.
    - sources: An array of all URLs explicitly mentioned in the answer; include any project/developer/news pages supporting the above facts.

    Rules:
    - Extract only what is explicitly present in the answer; do not infer or add content beyond the answer.
    - Preserve units, qualifiers, and approximations in strings (e.g., "$0.6B", "about 85 acres", "~750 units").
    - For includes_build_to_rent and includes_apartments, capture whatever short phrase indicates its presence (e.g., "BTR", "apartments", "yes"). Use null if the answer does not mention it.
    - For sources, collect every URL (including Markdown links) explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _collect_sources(extracted: ProjectExtraction) -> List[str]:
    if not extracted or not extracted.sources:
        return []
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in extracted.sources:
        if not u:
            continue
        u2 = u.strip()
        if u2 and u2 not in seen:
            out.append(u2)
            seen.add(u2)
    return out


def _southeastern_instruction() -> str:
    return (
        "A 'southeastern U.S. state' should be one of the following: "
        + ", ".join(SOUTHEASTERN_STATES)
        + ". If the company's HQ is in any of these, then it qualifies."
    )


def _yesno(val: Optional[str]) -> str:
    return val if val else "unspecified"


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_project_verification(
    evaluator: Evaluator,
    parent_node,
    extracted: ProjectExtraction
) -> None:
    """
    Build the verification tree under the Project_Identification node and run verifications.
    All child nodes here are critical because the rubric requires all criteria to be satisfied.
    """
    sources = _collect_sources(extracted)
    project_ref = extracted.project_name or "the project"

    # 1) Project Name: ensure provided AND verify via sources
    proj_name_node = evaluator.add_parallel(
        id="Project_Name",
        desc="The project has a designated official name (the answer provides the project name).",
        parent=parent_node,
        critical=True
    )
    name_exists = bool(extracted.project_name and extracted.project_name.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id="Project_Name_Provided",
        desc="Project name is provided in the answer.",
        parent=proj_name_node,
        critical=True
    )
    name_verify_leaf = evaluator.add_leaf(
        id="Project_Name_Verified",
        desc="The project's official name matches the provided name.",
        parent=proj_name_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The project's official name is '{extracted.project_name or ''}'.",
        node=name_verify_leaf,
        sources=sources,
        additional_instruction="Confirm the project name as cited by official or reputable sources (developer, city documents, or credible news). Allow minor variations (e.g., hyphens, capitalization)."
    )

    # 2) Location in Arizona
    loc_leaf = evaluator.add_leaf(
        id="Location_Arizona",
        desc="The project is located in Arizona.",
        parent=parent_node,
        critical=True
    )
    location_phrase = f"in {extracted.location_city}, Arizona" if extracted.location_city else "in Arizona"
    await evaluator.verify(
        claim=f"{project_ref} is located {location_phrase}.",
        node=loc_leaf,
        sources=sources,
        additional_instruction="Verify that the project is in the state of Arizona. If city is given, ensure that city is in AZ. Accept 'AZ' as equivalent."
    )

    # 3) Minimum Cost (>= $500 million)
    cost_leaf = evaluator.add_leaf(
        id="Minimum_Cost",
        desc="The total project cost is at least $500 million.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{project_ref} has a total project cost of at least $500 million.",
        node=cost_leaf,
        sources=sources,
        additional_instruction="Check that cost, investment value, or development budget meets or exceeds $500,000,000. Accept approximations like '$0.5B+', '$600M', '~$550 million'."
    )

    # 4) Minimum Acreage (>= 80 acres)
    acreage_leaf = evaluator.add_leaf(
        id="Minimum_Acreage",
        desc="The project site is at least 80 acres in size.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{project_ref} has a site size of at least 80 acres.",
        node=acreage_leaf,
        sources=sources,
        additional_instruction="Confirm total site area across phases/parcels if applicable. Accept approximations like 'about 80 acres' or '>80 acres'."
    )

    # 5) Residential Program: at least 700 units AND both apartments and BTR
    res_prog_node = evaluator.add_parallel(
        id="Residential_Program",
        desc="The development includes at least 700 total residential units AND includes both traditional apartments and build-to-rent homes.",
        parent=parent_node,
        critical=True
    )
    res_units_leaf = evaluator.add_leaf(
        id="Residential_Units_Count",
        desc="At least 700 total residential units.",
        parent=res_prog_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{project_ref} includes at least 700 total residential units.",
        node=res_units_leaf,
        sources=sources,
        additional_instruction="Look for a total residential unit count >= 700. Accept ranges or 'approximately'."
    )
    res_mix_leaf = evaluator.add_leaf(
        id="Residential_Mix_Types",
        desc="Includes both traditional apartments and build-to-rent (BTR) homes.",
        parent=res_prog_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{project_ref} includes both traditional apartments and build-to-rent (single-family rental) homes.",
        node=res_mix_leaf,
        sources=sources,
        additional_instruction="Confirm presence of both multifamily apartments and BTR/SFR homes. Synonyms: 'build-to-rent', 'BTR', 'single-family rental', 'for-rent homes'."
    )

    # 6) Open Space (>= 15 acres)
    open_space_leaf = evaluator.add_leaf(
        id="Open_Space",
        desc="The project includes at least 15 acres of dedicated open space.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{project_ref} includes at least 15 acres of dedicated open space.",
        node=open_space_leaf,
        sources=sources,
        additional_instruction="Open space may include parks, greenbelts, preserves, trails. Confirm dedicated open space area >= 15 acres."
    )

    # 7) Groundbreaking between Jan 2023 and Dec 2024 inclusive
    gb_leaf = evaluator.add_leaf(
        id="Groundbreaking_Timeline",
        desc="Groundbreaking occurred between January 2023 and December 2024 (inclusive).",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{project_ref} broke ground between January 2023 and December 2024 (inclusive).",
        node=gb_leaf,
        sources=sources,
        additional_instruction="Look for 'groundbreaking', 'broke ground', or 'construction started' dates. Accept any event within 2023 or 2024."
    )

    # 8) Mixed-Use Components: includes retail and light industrial
    mix_use_node = evaluator.add_parallel(
        id="Mixed_Use_Components",
        desc="The project is mixed-use and includes residential, retail, and light industrial components.",
        parent=parent_node,
        critical=True
    )
    retail_leaf = evaluator.add_leaf(
        id="Includes_Retail",
        desc="Includes retail space.",
        parent=mix_use_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{project_ref} includes retail space.",
        node=retail_leaf,
        sources=sources,
        additional_instruction="Retail can be described as shops, restaurants, commercial retail, or retail pads."
    )
    light_ind_leaf = evaluator.add_leaf(
        id="Includes_Light_Industrial",
        desc="Includes a light industrial component.",
        parent=mix_use_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{project_ref} includes a light industrial component.",
        node=light_ind_leaf,
        sources=sources,
        additional_instruction="Synonyms: 'light industrial', 'flex industrial', 'light manufacturing', 'industrial park'. Ensure the component is part of the project program."
    )

    # 9) Developer Type: a commercial real estate development company
    dev_type_leaf = evaluator.add_leaf(
        id="Developer_Type",
        desc="The developer is a commercial real estate development company.",
        parent=parent_node,
        critical=True
    )
    dev_name_for_claim = extracted.developer_name or "the developer"
    await evaluator.verify(
        claim=f"{dev_name_for_claim} is a commercial real estate development company.",
        node=dev_type_leaf,
        sources=sources,
        additional_instruction="Confirm that the company is a real estate developer (acceptable if described as 'real estate development company', 'developer', 'developer & owner'). Avoid firms that are solely architects, brokers, or property managers without development function."
    )

    # 10) Developer HQ in a southeastern U.S. state
    dev_hq_leaf = evaluator.add_leaf(
        id="Developer_Headquarters",
        desc="The developer company is headquartered in a southeastern U.S. state.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{dev_name_for_claim} is headquartered in a southeastern U.S. state.",
        node=dev_hq_leaf,
        sources=sources,
        additional_instruction=_southeastern_instruction()
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Entry point for evaluating an answer for the Arizona mixed-use project identification task.
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

    # Extract project information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_project_info(),
        template_class=ProjectExtraction,
        extraction_name="project_extraction"
    )

    # Add a critical node to aggregate all criteria (as per rubric root)
    proj_node = evaluator.add_parallel(
        id="Project_Identification",
        desc=PROJECT_IDENTIFICATION_DESC,
        parent=root,
        critical=True
    )

    # Build verification subtree and run checks
    await build_project_verification(evaluator, proj_node, extracted)

    # Optionally record helper info in summary for transparency
    evaluator.add_custom_info(
        info={
            "southeastern_states_list": SOUTHEASTERN_STATES,
            "extracted_snapshot": extracted.dict()
        },
        info_type="context",
        info_name="evaluation_context"
    )

    return evaluator.get_summary()