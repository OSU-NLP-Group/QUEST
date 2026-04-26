import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nds2025_bis_missouri_adoption_research"
TASK_DESCRIPTION = (
    "I watched the 2025 National Dog Show and want to adopt a dog of the breed that won Best in Show. "
    "I live in Missouri and need to research the following information before proceeding: "
    "(1) What breed won Best in Show at the 2025 National Dog Show, and which AKC group does this breed belong to? "
    "(2) What are the official height ranges (in inches) and weight ranges (in pounds) for both male and female dogs "
    "of this breed according to AKC standards? "
    "(3) For this breed's care requirements: What is the minimum daily exercise duration (in minutes) recommended for "
    "herding dogs, what are the three health evaluations recommended by the AKC or the breed's national club, and "
    "what is the regular grooming frequency outside of heavy shedding periods? "
    "(4) For adoption in Missouri: What is the minimum age requirement to adopt a dog from shelters or rescues, what "
    "type of identification is required, and can you provide the name and website of at least one dog adoption "
    "organization operating in Missouri?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BreedStandards(BaseModel):
    male_height_inches: Optional[str] = None
    male_weight_lbs: Optional[str] = None
    female_height_inches: Optional[str] = None
    female_weight_lbs: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CareRequirements(BaseModel):
    exercise_min_minutes: Optional[str] = None
    health_evaluations: List[str] = Field(default_factory=list)
    grooming_frequency: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AdoptionInfo(BaseModel):
    min_adopter_age: Optional[str] = None
    id_required: Optional[str] = None
    org_name: Optional[str] = None
    org_website: Optional[str] = None
    # Policy source URLs for age/ID. Can include shelter policy pages, state resources, etc.
    sources: List[str] = Field(default_factory=list)


class ResearchExtraction(BaseModel):
    winner_breed: Optional[str] = None
    winner_sources: List[str] = Field(default_factory=list)
    breed_group: Optional[str] = None
    group_sources: List[str] = Field(default_factory=list)

    standards: BreedStandards = Field(default_factory=BreedStandards)
    care: CareRequirements = Field(default_factory=CareRequirements)
    adoption: AdoptionInfo = Field(default_factory=AdoptionInfo)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_research() -> str:
    return """
Extract all requested details from the answer. Return a single JSON object matching the following schema:

{
  "winner_breed": string | null,
  "winner_sources": string[]  // URLs explicitly cited for Best in Show winner
  "breed_group": string | null,
  "group_sources": string[]   // URLs explicitly cited for AKC group of the winner breed

  "standards": {
    "male_height_inches": string | null,   // e.g., "24-26", "24 to 26", or similar string; do not convert to number
    "male_weight_lbs": string | null,      // e.g., "65-75", "65 to 75"
    "female_height_inches": string | null,
    "female_weight_lbs": string | null,
    "sources": string[]                    // URLs that support AKC physical standards (ideally AKC breed page)
  },

  "care": {
    "exercise_min_minutes": string | null, // minimum daily exercise duration in minutes; if the answer uses hours (e.g., "at least 1 hour"), convert to minutes string "60" or "60+" as string
    "health_evaluations": string[],        // three evaluations recommended by AKC or the national breed club (strings as stated)
    "grooming_frequency": string | null,   // regular grooming frequency outside heavy shedding (e.g., "weekly", "several times a week")
    "sources": string[]                    // URLs that support care requirements (AKC breed page or national club)
  },

  "adoption": {
    "min_adopter_age": string | null,      // minimum age to adopt from shelters/rescues in Missouri, as stated by cited sources
    "id_required": string | null,          // identification required, as stated (e.g., "government-issued photo ID")
    "org_name": string | null,             // name of at least one dog adoption organization in Missouri
    "org_website": string | null,          // website URL for that organization
    "sources": string[]                    // URLs to adoption policy pages (shelters/rescues/government) supporting age/ID info; may include the org website
  }
}

Rules:
- Extract only what is explicitly present in the answer.
- For any missing field, return null (or empty array for list fields).
- For any URL fields, include only actual URLs explicitly present in the answer (plain or markdown links).
- Keep ranges and durations as strings exactly as written or normalized (e.g., "1 hour" -> "60").
- Do not invent any URLs or facts.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(val: Optional[str]) -> str:
    return val if val else ""


def _combine_sources(*source_lists: Optional[List[str]]) -> List[str]:
    combined: List[str] = []
    seen = set()
    for lst in source_lists:
        if not lst:
            continue
        for url in lst:
            if not url:
                continue
            if url not in seen:
                combined.append(url)
                seen.add(url)
    return combined


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_show_winner_checks(
    evaluator: Evaluator,
    parent: VerificationNode,
    data: ResearchExtraction
) -> VerificationNode:
    """
    Build 'Show_Winner_Identification' subtree and run verifications.
    Returns the subtree node; the 'Winner_Breed' leaf will be findable via evaluator.find_node if needed.
    """
    node = evaluator.add_parallel(
        id="Show_Winner_Identification",
        desc="Identify the 2025 National Dog Show Best in Show winner breed and its AKC group",
        parent=parent,
        critical=True
    )

    # Winner breed
    winner_leaf = evaluator.add_leaf(
        id="Winner_Breed",
        desc="Correctly identify the breed that won Best in Show at the 2025 National Dog Show",
        parent=node,
        critical=True
    )
    winner_claim = f"The breed that won Best in Show at the 2025 National Dog Show was {_safe(data.winner_breed)}."
    await evaluator.verify(
        claim=winner_claim,
        node=winner_leaf,
        sources=data.winner_sources,
        additional_instruction="Verify via official results or credible news (e.g., National Dog Show or NBC Sports coverage). Allow minor title/capitalization variants."
    )

    # Breed group
    group_leaf = evaluator.add_leaf(
        id="Breed_Group",
        desc="Correctly identify which AKC group the winning breed belongs to",
        parent=node,
        critical=True
    )
    group_sources = _combine_sources(data.group_sources, data.standards.sources)
    group_claim = f"The breed {_safe(data.winner_breed)} belongs to the {_safe(data.breed_group)} group in the American Kennel Club classification."
    await evaluator.verify(
        claim=group_claim,
        node=group_leaf,
        sources=group_sources,
        additional_instruction="Check the AKC classification on the AKC breed page or AKC resources. Accept minor wording variants (e.g., 'Herding Group' vs 'herding')."
    )

    return node


async def build_breed_physical_standards_checks(
    evaluator: Evaluator,
    parent: VerificationNode,
    data: ResearchExtraction,
    prereq: Optional[VerificationNode] = None
) -> VerificationNode:
    """
    Build 'Breed_Physical_Standards' subtree and run verifications.
    """
    node = evaluator.add_parallel(
        id="Breed_Physical_Standards",
        desc="AKC physical standards for the winning breed (male/female height and weight ranges)",
        parent=parent,
        critical=True
    )
    sources = data.standards.sources

    # Male height
    male_height_leaf = evaluator.add_leaf(
        id="Male_Height_Range",
        desc="Provide the correct height range in inches for male dogs of the breed (AKC standard)",
        parent=node,
        critical=True
    )
    male_height_claim = (
        f"According to the AKC breed standard, the height for an adult male {_safe(data.winner_breed)} is "
        f"{_safe(data.standards.male_height_inches)} inches."
    )
    await evaluator.verify(
        claim=male_height_claim,
        node=male_height_leaf,
        sources=sources,
        additional_instruction="Verify the range on the AKC breed page/standard. Allow variants like '24-26 inches', '24 to 26 inches', or minor rounding.",
        extra_prerequisites=[prereq] if prereq else None
    )

    # Male weight
    male_weight_leaf = evaluator.add_leaf(
        id="Male_Weight_Range",
        desc="Provide the correct weight range in pounds for male dogs of the breed (AKC standard)",
        parent=node,
        critical=True
    )
    male_weight_claim = (
        f"According to the AKC breed standard, the weight for an adult male {_safe(data.winner_breed)} is "
        f"{_safe(data.standards.male_weight_lbs)} pounds."
    )
    await evaluator.verify(
        claim=male_weight_claim,
        node=male_weight_leaf,
        sources=sources,
        additional_instruction="Verify the weight range on the AKC breed page/standard. Allow 'lbs'/'pounds' variants and minor rounding.",
        extra_prerequisites=[prereq] if prereq else None
    )

    # Female height
    female_height_leaf = evaluator.add_leaf(
        id="Female_Height_Range",
        desc="Provide the correct height range in inches for female dogs of the breed (AKC standard)",
        parent=node,
        critical=True
    )
    female_height_claim = (
        f"According to the AKC breed standard, the height for an adult female {_safe(data.winner_breed)} is "
        f"{_safe(data.standards.female_height_inches)} inches."
    )
    await evaluator.verify(
        claim=female_height_claim,
        node=female_height_leaf,
        sources=sources,
        additional_instruction="Verify the range on the AKC breed page/standard. Allow notation and rounding variants.",
        extra_prerequisites=[prereq] if prereq else None
    )

    # Female weight
    female_weight_leaf = evaluator.add_leaf(
        id="Female_Weight_Range",
        desc="Provide the correct weight range in pounds for female dogs of the breed (AKC standard)",
        parent=node,
        critical=True
    )
    female_weight_claim = (
        f"According to the AKC breed standard, the weight for an adult female {_safe(data.winner_breed)} is "
        f"{_safe(data.standards.female_weight_lbs)} pounds."
    )
    await evaluator.verify(
        claim=female_weight_claim,
        node=female_weight_leaf,
        sources=sources,
        additional_instruction="Verify the weight range on the AKC breed page/standard. Allow 'lbs'/'pounds' variants and minor rounding.",
        extra_prerequisites=[prereq] if prereq else None
    )

    return node


async def build_health_care_checks(
    evaluator: Evaluator,
    parent: VerificationNode,
    data: ResearchExtraction,
    prereq: Optional[VerificationNode] = None
) -> VerificationNode:
    """
    Build 'Health_Care_Requirements' subtree and run verifications.
    """
    node = evaluator.add_parallel(
        id="Health_Care_Requirements",
        desc="Care requirements for the breed (exercise, health evaluations, grooming frequency)",
        parent=parent,
        critical=True
    )
    sources = _combine_sources(data.care.sources, data.standards.sources, data.group_sources)

    # Exercise duration
    exercise_leaf = evaluator.add_leaf(
        id="Exercise_Duration",
        desc="Specify the minimum daily exercise duration in minutes recommended for herding dogs (as required by the prompt/constraints)",
        parent=node,
        critical=True
    )
    exercise_claim = (
        f"The minimum daily exercise duration recommended for herding dogs (or for the {_safe(data.winner_breed)}) "
        f"is {_safe(data.care.exercise_min_minutes)} minutes per day."
    )
    await evaluator.verify(
        claim=exercise_claim,
        node=exercise_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that sources recommend at least the stated minutes per day. If a source states 'at least 1 hour', "
            "treat it as 60 minutes. Allow phrasing variants like 'minimum of 60–90 minutes' to match >= the claimed minimum."
        ),
        extra_prerequisites=[prereq] if prereq else None
    )

    # Health screenings (3)
    screenings_leaf = evaluator.add_leaf(
        id="Health_Screenings",
        desc="List the three health evaluations recommended by the AKC or the breed's national club",
        parent=node,
        critical=True
    )
    screenings_text = ", ".join(data.care.health_evaluations) if data.care.health_evaluations else ""
    screenings_claim = (
        f"The recommended health evaluations for {_safe(data.winner_breed)} are: {screenings_text}. "
        f"These come from AKC or the national breed club."
    )
    await evaluator.verify(
        claim=screenings_claim,
        node=screenings_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that AKC or the national breed club (parent club) explicitly lists these health tests. "
            "Accept synonyms (e.g., 'hip evaluation' vs 'OFA hips', 'eye exam' vs 'CERF/CAER'). "
            "There should be three specific evaluations matching those listed."
        ),
        extra_prerequisites=[prereq] if prereq else None
    )

    # Grooming frequency
    grooming_leaf = evaluator.add_leaf(
        id="Grooming_Frequency",
        desc="Describe the regular grooming frequency outside of heavy shedding periods",
        parent=node,
        critical=True
    )
    grooming_claim = (
        f"Outside heavy shedding periods, the regular grooming frequency for {_safe(data.winner_breed)} is "
        f"{_safe(data.care.grooming_frequency)}."
    )
    await evaluator.verify(
        claim=grooming_claim,
        node=grooming_leaf,
        sources=sources,
        additional_instruction=(
            "Focus on the normal routine (not seasonal blowout). Accept equivalent phrasing (e.g., 'weekly', 'once a week', "
            "'several times a week')."
        ),
        extra_prerequisites=[prereq] if prereq else None
    )

    return node


async def build_missouri_adoption_checks(
    evaluator: Evaluator,
    parent: VerificationNode,
    data: ResearchExtraction,
    prereq: Optional[VerificationNode] = None
) -> VerificationNode:
    """
    Build 'Missouri_Adoption_Requirements' subtree and run verifications.
    """
    node = evaluator.add_parallel(
        id="Missouri_Adoption_Requirements",
        desc="Missouri adoption requirements and at least one Missouri adoption organization (name + website)",
        parent=parent,
        critical=True
    )

    # Age requirement
    age_leaf = evaluator.add_leaf(
        id="Age_Requirement",
        desc="State the minimum age requirement to adopt a dog from shelters or rescues",
        parent=node,
        critical=True
    )
    age_claim = (
        f"The minimum age requirement to adopt a dog from shelters or rescues in Missouri is "
        f"{_safe(data.adoption.min_adopter_age)}."
    )
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        sources=data.adoption.sources,
        additional_instruction=(
            "Verify based on Missouri-based shelter/rescue policy pages or authoritative sources. "
            "Accept if the cited policy clearly states the minimum adopter age."
        ),
        extra_prerequisites=[prereq] if prereq else None
    )

    # ID requirement
    id_leaf = evaluator.add_leaf(
        id="ID_Requirement",
        desc="Specify the type of identification required to adopt",
        parent=node,
        critical=True
    )
    id_claim = (
        f"To adopt a dog in Missouri, the required identification is {_safe(data.adoption.id_required)}."
    )
    await evaluator.verify(
        claim=id_claim,
        node=id_leaf,
        sources=data.adoption.sources,
        additional_instruction=(
            "Verify on Missouri shelter/rescue policy pages or other cited authoritative sources. "
            "Accept common formulations such as 'government-issued photo ID' or 'photo ID with current address'."
        ),
        extra_prerequisites=[prereq] if prereq else None
    )

    # Missouri adoption organization (name + website)
    org_leaf = evaluator.add_leaf(
        id="Missouri_Adoption_Resource",
        desc="Provide the name and website of at least one dog adoption organization operating in Missouri",
        parent=node,
        critical=True
    )
    org_claim = (
        f"The organization named '{_safe(data.adoption.org_name)}' operates in Missouri and its website is "
        f"{_safe(data.adoption.org_website)}."
    )
    await evaluator.verify(
        claim=org_claim,
        node=org_leaf,
        sources=data.adoption.org_website,
        additional_instruction=(
            "Confirm from the provided website that this organization is a dog adoption/shelter/rescue operating in Missouri "
            "or serving Missouri (statewide or local)."
        ),
        extra_prerequisites=[prereq] if prereq else None
    )

    return node


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
    Evaluate an answer for the 2025 National Dog Show Best in Show research and Missouri adoption requirements.
    """
    # Initialize evaluator with a non-critical root by framework design.
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

    # Add a top-level critical node to aggregate all required sub-criteria (since root is non-critical)
    main = evaluator.add_parallel(
        id="Belgian_Sheepdog_Adoption_Research",
        desc="Comprehensive research for adopting the 2025 National Dog Show Best in Show winning breed in Missouri",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extracted: ResearchExtraction = await evaluator.extract(
        prompt=prompt_extract_research(),
        template_class=ResearchExtraction,
        extraction_name="research_extraction"
    )

    # Build subtrees and run verifications
    show_node = await build_show_winner_checks(evaluator, main, extracted)

    # Find the 'Winner_Breed' leaf to use as a prerequisite for other sections
    winner_breed_leaf = evaluator.find_node("Winner_Breed")

    await build_breed_physical_standards_checks(
        evaluator=evaluator,
        parent=main,
        data=extracted,
        prereq=winner_breed_leaf
    )

    await build_health_care_checks(
        evaluator=evaluator,
        parent=main,
        data=extracted,
        prereq=winner_breed_leaf
    )

    await build_missouri_adoption_checks(
        evaluator=evaluator,
        parent=main,
        data=extracted,
        prereq=winner_breed_leaf
    )

    # Return the evaluation summary
    return evaluator.get_summary()