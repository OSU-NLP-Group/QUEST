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
TASK_ID = "bayarea_makerspace_cutting_board"
TASK_DESCRIPTION = """
I want to build my first wooden cutting board as a beginner woodworking project. Identify a community makerspace in the San Francisco Bay Area where I could complete this project, and provide the following information: 
1. The name and location of the makerspace 
2. Confirmation that they have the necessary woodworking equipment (table saw, planer, sanders) 
3. Their safety training or certification requirements 
4. Valid contact information (website, phone, or email) 
5. The monthly membership cost for woodshop access 
6. Their hours of operation or member access policy 
7. Any age requirements for membership 
8. Recommended food-safe hardwood species for the cutting board 
9. Appropriate lumber dimensions or thickness to purchase 
10. Suitable food-safe finishing products for the cutting board 
11. An estimated time to complete the project as a beginner 
12. Any additional relevant preparation details. Ensure all information is accurate and verifiable from reliable sources.
"""

# A helper hint list for Bay Area locality judgment in verification
BAY_AREA_HINT = (
    "Treat an address in any of these counties/cities as San Francisco Bay Area: "
    "Counties: San Francisco, San Mateo, Santa Clara, Alameda, Contra Costa, Marin, Napa, Sonoma, Solano. "
    "Representative cities include (but are not limited to): San Francisco, Oakland, Berkeley, San Jose, "
    "Palo Alto, Mountain View, Sunnyvale, Santa Clara, Redwood City, San Mateo, Daly City, Fremont, "
    "Hayward, Walnut Creek, Richmond, San Rafael."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MakerspaceExtraction(BaseModel):
    # Core makerspace identity and location
    name: Optional[str] = None
    city: Optional[str] = None
    address: Optional[str] = None
    website_url: Optional[str] = None

    # Contact info
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None

    # Membership/access/age
    membership_cost_monthly: Optional[str] = None
    access_hours_policy: Optional[str] = None
    min_age_requirement: Optional[str] = None

    # Woodshop/equipment/training
    woodshop_equipment: List[str] = Field(default_factory=list)
    safety_training_summary: Optional[str] = None

    # Woodworking guidance
    recommended_species: List[str] = Field(default_factory=list)
    lumber_dimensions: Optional[str] = None
    finishing_products: List[str] = Field(default_factory=list)
    time_estimate: Optional[str] = None

    # Source URLs explicitly cited in the answer (per-topic)
    location_sources: List[str] = Field(default_factory=list)
    equipment_sources: List[str] = Field(default_factory=list)
    safety_sources: List[str] = Field(default_factory=list)
    contact_sources: List[str] = Field(default_factory=list)
    membership_cost_sources: List[str] = Field(default_factory=list)
    access_hours_sources: List[str] = Field(default_factory=list)
    age_requirement_sources: List[str] = Field(default_factory=list)

    species_sources: List[str] = Field(default_factory=list)
    lumber_sources: List[str] = Field(default_factory=list)
    finish_sources: List[str] = Field(default_factory=list)
    time_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_makerspace() -> str:
    return """
    Extract information about ONE specific community makerspace in the San Francisco Bay Area mentioned in the answer. 
    If multiple are mentioned, extract the FIRST one only. Extract exactly what appears in the answer text.

    Required fields (set missing ones to null or empty list):
    - name: Full makerspace name.
    - city: City name (e.g., 'San Jose', 'Oakland') if provided.
    - address: The street address if provided.
    - website_url: The official website URL if provided.
    - contact_phone: A phone number if provided.
    - contact_email: An email address if provided.

    Membership/access/age:
    - membership_cost_monthly: The stated monthly membership cost for woodshop access (string as written).
    - access_hours_policy: Stated hours or access policy (e.g., '24/7 members access' or 'Mon–Fri 10am–6pm').
    - min_age_requirement: The minimum age requirement if mentioned.

    Woodshop/equipment/training:
    - woodshop_equipment: List of equipment specifically named in the answer (e.g., 'table saw', 'planer', 'belt sander').
    - safety_training_summary: Summary phrase if the answer states orientation/training/certification is required.

    Woodworking guidance for the cutting board:
    - recommended_species: List of hardwood species recommended in the answer (e.g., 'maple', 'walnut', 'cherry').
    - lumber_dimensions: Lumber dimension or thickness recommended (e.g., '4/4 (1 inch)').
    - finishing_products: List of food-safe finish products named (e.g., 'mineral oil', 'beeswax', 'butcher block conditioner').
    - time_estimate: The time estimate given in the answer (e.g., '4–6 hours', 'about 5 hours').

    For each of the following topics, also extract an array of URL sources that the answer explicitly cites for that topic.
    Only include actual URLs that appear in the answer text (markdown links OK).
    - location_sources
    - equipment_sources
    - safety_sources
    - contact_sources
    - membership_cost_sources
    - access_hours_sources
    - age_requirement_sources
    - species_sources
    - lumber_sources
    - finish_sources
    - time_sources

    Return a single JSON object with all fields above. For any fields not present in the answer, return null or [].
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _sources_or_fallback(primary_sources: List[str], fallback_url: Optional[str]) -> List[str]:
    """Prefer explicit sources extracted from the answer; otherwise fall back to website_url if available."""
    src = [u for u in (primary_sources or []) if _non_empty(u)]
    if not src and _non_empty(fallback_url):
        src = [fallback_url]  # Use official site if no topic-specific URL was cited
    return src


def _first_n(items: List[str], n: int) -> List[str]:
    return [x for x in items[:n] if _non_empty(x)]


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_makerspace_name(evaluator: Evaluator, root, ex: MakerspaceExtraction) -> None:
    node = evaluator.add_sequential(
        id="MakerspaceName_seq",
        desc="Provides the name of a specific community makerspace in the San Francisco Bay Area",
        parent=root,
        critical=True,  # Critical criterion
    )
    evaluator.add_custom_node(
        result=_non_empty(ex.name),
        id="MakerspaceName",
        desc="Makerspace name is provided",
        parent=node,
        critical=True
    )


async def verify_location(evaluator: Evaluator, root, ex: MakerspaceExtraction) -> None:
    node = evaluator.add_sequential(
        id="LocationVerification_seq",
        desc="Provides the city or specific address confirming the makerspace is in the San Francisco Bay Area",
        parent=root,
        critical=True,
    )

    exists = (_non_empty(ex.city) or _non_empty(ex.address)) and bool(
        _sources_or_fallback(ex.location_sources, ex.website_url)
    )
    evaluator.add_custom_node(
        result=exists,
        id="LocationVerification_exists",
        desc="Location info and at least one source are provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="LocationVerification",
        desc="Location is confirmed (Bay Area city/address) by cited sources",
        parent=node,
        critical=True
    )
    city = ex.city or ""
    address = ex.address or ""
    name = ex.name or "the makerspace"
    claim = f"The cited source(s) show that {name} is located at '{address}' in '{city}', which is in the San Francisco Bay Area."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_sources_or_fallback(ex.location_sources, ex.website_url),
        additional_instruction=(
            "You must confirm the specific address or city on the page. "
            "Then determine whether that city/address is within the San Francisco Bay Area. "
            f"{BAY_AREA_HINT} If the page clearly shows an address within one of these cities/counties, "
            "conclude that it is in the Bay Area."
        ),
    )


async def verify_equipment(evaluator: Evaluator, root, ex: MakerspaceExtraction) -> None:
    parent = evaluator.add_sequential(
        id="WoodshopEquipmentAvailability_seq",
        desc="Confirms the makerspace has woodworking equipment including table saw, planer, and sanders",
        parent=root,
        critical=True,
    )

    # Gate on having at least some source to check
    has_src = bool(_sources_or_fallback(ex.equipment_sources, ex.website_url))
    evaluator.add_custom_node(
        result=has_src,
        id="WoodshopEquipmentAvailability_sources_exist",
        desc="Equipment sources exist",
        parent=parent,
        critical=True
    )

    # Parallel verification of each required tool
    parallel_tools = evaluator.add_parallel(
        id="WoodshopEquipmentAvailability",
        desc="Verify required woodshop tools exist (table saw, planer, sanders)",
        parent=parent,
        critical=True
    )

    # 1) Table saw
    leaf_ts = evaluator.add_leaf(
        id="Equipment_TableSaw",
        desc="Table saw is available in the makerspace woodshop",
        parent=parallel_tools,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ex.name or 'The makerspace'} has a table saw available for member use.",
        node=leaf_ts,
        sources=_sources_or_fallback(ex.equipment_sources, ex.website_url),
        additional_instruction="Look for 'table saw' or equivalent wording on the woodshop/equipment pages."
    )

    # 2) Planer
    leaf_pl = evaluator.add_leaf(
        id="Equipment_Planer",
        desc="Planer (thickness planer) is available in the makerspace woodshop",
        parent=parallel_tools,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ex.name or 'The makerspace'} has a planer (thickness planer) available for member use.",
        node=leaf_pl,
        sources=_sources_or_fallback(ex.equipment_sources, ex.website_url),
        additional_instruction="Accept synonyms like 'thickness planer'."
    )

    # 3) Sanders
    leaf_sd = evaluator.add_leaf(
        id="Equipment_Sanders",
        desc="Sander(s) are available in the makerspace woodshop",
        parent=parallel_tools,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ex.name or 'The makerspace'} has sander(s) available for member use.",
        node=leaf_sd,
        sources=_sources_or_fallback(ex.equipment_sources, ex.website_url),
        additional_instruction="Accept common sander types (belt, disc, spindle, random orbital, drum sander, etc.)."
    )


async def verify_safety_training(evaluator: Evaluator, root, ex: MakerspaceExtraction) -> None:
    node = evaluator.add_sequential(
        id="SafetyTrainingRequirement_seq",
        desc="Identifies that the makerspace requires safety certification, orientation, or training before tool access",
        parent=root,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(_sources_or_fallback(ex.safety_sources, ex.website_url)),
        id="SafetyTrainingRequirement_sources_exist",
        desc="Safety/training sources exist",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="SafetyTrainingRequirement",
        desc="Safety training requirement is confirmed by cited sources",
        parent=node,
        critical=True
    )
    claim = (
        f"{ex.name or 'The makerspace'} requires safety orientation/training/certification before members can use "
        "the woodshop tools."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_sources_or_fallback(ex.safety_sources, ex.website_url),
        additional_instruction="Look for phrases such as 'orientation', 'woodshop safety', 'tool authorization', or 'certification required'."
    )


async def verify_contact(evaluator: Evaluator, root, ex: MakerspaceExtraction) -> None:
    node = evaluator.add_sequential(
        id="ContactInformation_seq",
        desc="Provides valid contact information (website, phone, or email) for the makerspace",
        parent=root,
        critical=True,
    )

    any_contact = _non_empty(ex.website_url) or _non_empty(ex.contact_phone) or _non_empty(ex.contact_email)
    evaluator.add_custom_node(
        result=any_contact and bool(_sources_or_fallback(ex.contact_sources, ex.website_url) or ex.website_url),
        id="ContactInformation_exists",
        desc="At least one contact method and a source are provided",
        parent=node,
        critical=True
    )

    # Choose best verification path by priority: website > phone > email
    if _non_empty(ex.website_url):
        leaf = evaluator.add_leaf(
            id="ContactInformation_website",
            desc="Official website is valid",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=f"This page is the official website of {ex.name or 'the makerspace'}: {ex.website_url}",
            node=leaf,
            sources=ex.website_url,
            additional_instruction="Confirm branding/name on the page matches the makerspace; homepage or primary domain is acceptable."
        )
    elif _non_empty(ex.contact_phone):
        leaf = evaluator.add_leaf(
            id="ContactInformation_phone",
            desc="Phone number is valid",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The phone number for {ex.name or 'the makerspace'} is {ex.contact_phone}.",
            node=leaf,
            sources=_sources_or_fallback(ex.contact_sources, ex.website_url),
            additional_instruction="Check contact/about pages for the exact phone number."
        )
    else:
        leaf = evaluator.add_leaf(
            id="ContactInformation_email",
            desc="Email address is valid",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The email address for {ex.name or 'the makerspace'} is {ex.contact_email or ''}.",
            node=leaf,
            sources=_sources_or_fallback(ex.contact_sources, ex.website_url),
            additional_instruction="Check contact/about pages for the exact email address."
        )


async def verify_membership_cost(evaluator: Evaluator, root, ex: MakerspaceExtraction) -> None:
    node = evaluator.add_sequential(
        id="MembershipCost_seq",
        desc="Specifies the monthly membership cost for woodshop access",
        parent=root,
        critical=False,
    )

    evaluator.add_custom_node(
        result=_non_empty(ex.membership_cost_monthly) and bool(_sources_or_fallback(ex.membership_cost_sources, ex.website_url)),
        id="MembershipCost_exists",
        desc="Membership cost and a source are provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="MembershipCost",
        desc="Monthly membership cost is supported by cited sources",
        parent=node,
        critical=True
    )
    claim = f"The monthly membership cost for woodshop access is stated as '{ex.membership_cost_monthly or ''}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_sources_or_fallback(ex.membership_cost_sources, ex.website_url),
        additional_instruction="Accept if the page shows a matching or clearly equivalent monthly price (tiers acceptable if the stated one is shown)."
    )


async def verify_access_hours(evaluator: Evaluator, root, ex: MakerspaceExtraction) -> None:
    node = evaluator.add_sequential(
        id="AccessHours_seq",
        desc="Specifies the makerspace's hours of operation or access policy",
        parent=root,
        critical=False,
    )

    evaluator.add_custom_node(
        result=_non_empty(ex.access_hours_policy) and bool(_sources_or_fallback(ex.access_hours_sources, ex.website_url)),
        id="AccessHours_exists",
        desc="Access hours/policy and a source are provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="AccessHours",
        desc="Access hours/policy is supported by cited sources",
        parent=node,
        critical=True
    )
    claim = f"The makerspace's hours of operation or member access policy is '{ex.access_hours_policy or ''}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_sources_or_fallback(ex.access_hours_sources, ex.website_url),
        additional_instruction="Accept wording variations such as '24/7 access for members' or listed daily hours."
    )


async def verify_age_requirement(evaluator: Evaluator, root, ex: MakerspaceExtraction) -> None:
    node = evaluator.add_sequential(
        id="MemberAgeRequirement_seq",
        desc="Identifies the minimum age requirement for membership or facility use",
        parent=root,
        critical=False,
    )

    evaluator.add_custom_node(
        result=_non_empty(ex.min_age_requirement) and bool(_sources_or_fallback(ex.age_requirement_sources, ex.website_url)),
        id="MemberAgeRequirement_exists",
        desc="Minimum age requirement and a source are provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="MemberAgeRequirement",
        desc="Minimum age requirement is supported by cited sources",
        parent=node,
        critical=True
    )
    claim = f"The minimum age requirement for membership or woodshop use is '{ex.min_age_requirement or ''}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_sources_or_fallback(ex.age_requirement_sources, ex.website_url),
        additional_instruction="Check membership policy, rules, or FAQs pages for stated age requirements."
    )


async def verify_species(evaluator: Evaluator, root, ex: MakerspaceExtraction) -> None:
    node = evaluator.add_sequential(
        id="WoodSpeciesSelection_seq",
        desc="Recommends appropriate food-safe hardwood species for cutting boards",
        parent=root,
        critical=False,
    )

    evaluator.add_custom_node(
        result=bool(ex.recommended_species) and bool(_sources_or_fallback(ex.species_sources, None)),
        id="WoodSpeciesSelection_exists",
        desc="At least one species and at least one source are provided",
        parent=node,
        critical=True
    )

    parallel = evaluator.add_parallel(
        id="WoodSpeciesSelection",
        desc="Verify each recommended hardwood species is appropriate for cutting boards",
        parent=node,
        critical=False
    )

    # Verify up to first 3 species for partial credit
    for idx, sp in enumerate(_first_n(ex.recommended_species, 3)):
        leaf = evaluator.add_leaf(
            id=f"WoodSpecies_{idx}",
            desc=f"Species '{sp}' is appropriate and food-safe for cutting boards",
            parent=parallel,
            critical=False
        )
        await evaluator.verify(
            claim=f"The hardwood species '{sp}' is appropriate and commonly recommended for food-safe cutting boards.",
            node=leaf,
            sources=_sources_or_fallback(ex.species_sources, None),
            additional_instruction="Look for guidance pages that list safe hardwoods for cutting boards (e.g., maple, walnut, cherry, etc.)."
        )


async def verify_lumber_dimensions(evaluator: Evaluator, root, ex: MakerspaceExtraction) -> None:
    node = evaluator.add_sequential(
        id="LumberDimensions_seq",
        desc="Specifies appropriate lumber dimensions or thickness for a cutting board project",
        parent=root,
        critical=False,
    )

    evaluator.add_custom_node(
        result=_non_empty(ex.lumber_dimensions) and bool(_sources_or_fallback(ex.lumber_sources, None)),
        id="LumberDimensions_exists",
        desc="Lumber thickness/dimensions and a source are provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="LumberDimensions",
        desc="Appropriate lumber dimension/thickness is supported by cited sources",
        parent=node,
        critical=True
    )
    claim = f"An appropriate lumber thickness/dimension for a cutting board project is '{ex.lumber_dimensions or ''}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_sources_or_fallback(ex.lumber_sources, None),
        additional_instruction="Common guidance includes 4/4 (approx. 1 inch) stock or finished thickness around 3/4–1 inch for beginners."
    )


async def verify_finishes(evaluator: Evaluator, root, ex: MakerspaceExtraction) -> None:
    node = evaluator.add_sequential(
        id="FoodSafeFinish_seq",
        desc="Identifies appropriate food-safe finishing products for cutting boards",
        parent=root,
        critical=False,
    )

    evaluator.add_custom_node(
        result=bool(ex.finishing_products) and bool(_sources_or_fallback(ex.finish_sources, None)),
        id="FoodSafeFinish_exists",
        desc="At least one finishing product and a source are provided",
        parent=node,
        critical=True
    )

    parallel = evaluator.add_parallel(
        id="FoodSafeFinish",
        desc="Verify each recommended finish is food-safe and suitable for cutting boards",
        parent=node,
        critical=False
    )

    for idx, finish in enumerate(_first_n(ex.finishing_products, 3)):
        leaf = evaluator.add_leaf(
            id=f"FoodSafeFinish_{idx}",
            desc=f"Finish '{finish}' is food-safe and appropriate for cutting boards",
            parent=parallel,
            critical=False
        )
        await evaluator.verify(
            claim=f"The finish '{finish}' is food-safe and suitable for protecting wooden cutting boards.",
            node=leaf,
            sources=_sources_or_fallback(ex.finish_sources, None),
            additional_instruction="Common acceptable finishes include mineral oil, beeswax, and butcher block conditioners."
        )


async def verify_time_estimate(evaluator: Evaluator, root, ex: MakerspaceExtraction) -> None:
    node = evaluator.add_sequential(
        id="ProjectTimeEstimate_seq",
        desc="Provides a realistic time estimate for completing a beginner cutting board project",
        parent=root,
        critical=False,
    )

    evaluator.add_custom_node(
        result=_non_empty(ex.time_estimate) and bool(_sources_or_fallback(ex.time_sources, None)),
        id="ProjectTimeEstimate_exists",
        desc="Time estimate and a source are provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="ProjectTimeEstimate",
        desc="Beginner time estimate is supported by cited sources",
        parent=node,
        critical=True
    )
    claim = f"A beginner can complete a basic wooden cutting board in about '{ex.time_estimate or ''}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_sources_or_fallback(ex.time_sources, None),
        additional_instruction="Typical guidance suggests roughly 4–6 hours; accept reasonable nearby ranges if the cited page supports it."
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize the evaluator. Root is non-critical to allow partial credit aggregation.
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent criteria
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

    # Create the rubric root node (non-critical to permit partial scoring on non-critical children)
    rubric_root = evaluator.add_parallel(
        id="MakerspaceAndProjectPreparation",
        desc="Evaluate whether the response correctly identifies a qualifying Bay Area makerspace and specifies proper cutting board project preparations",
        parent=root,
        critical=False
    )

    # Extract structured info from the answer
    ex: MakerspaceExtraction = await evaluator.extract(
        prompt=prompt_extract_makerspace(),
        template_class=MakerspaceExtraction,
        extraction_name="makerspace_extraction",
    )

    # Build verification subtrees according to rubric
    await verify_makerspace_name(evaluator, rubric_root, ex)
    await verify_location(evaluator, rubric_root, ex)
    await verify_equipment(evaluator, rubric_root, ex)
    await verify_safety_training(evaluator, rubric_root, ex)
    await verify_contact(evaluator, rubric_root, ex)

    await verify_membership_cost(evaluator, rubric_root, ex)
    await verify_access_hours(evaluator, rubric_root, ex)
    await verify_age_requirement(evaluator, rubric_root, ex)

    await verify_species(evaluator, rubric_root, ex)
    await verify_lumber_dimensions(evaluator, rubric_root, ex)
    await verify_finishes(evaluator, rubric_root, ex)
    await verify_time_estimate(evaluator, rubric_root, ex)

    # Return the full summary with verification tree and scores
    return evaluator.get_summary()