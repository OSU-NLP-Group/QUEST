import asyncio
import logging
from typing import Optional, List, Dict, Set

import openai
from pydantic import BaseModel, Field

from mind2web2.eval_toolkit import create_evaluator, Extractor, Verifier
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import Evaluator
from mind2web2.verification_tree import VerificationNode, AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pet_food_vet"
TASK_DESCRIPTION = """
I'm a first-time cat owner preparing thoroughly before bringing a cat home. To ensure I have all essential information, please help me address the following points by finding detailed and credible online resources.:

Dietary Needs: Find one comprehensive article or blog that clearly outlines both the key nutritional requirements for a healthy adult indoor cat and foods that must be avoided. Summarize the main points briefly.

Health and Vaccinations: Provide a list of core vaccinations recommended for indoor cats based on reputable veterinary sources (e.g., AVMA, AAHA). Clearly indicate the typical age for the initial vaccination, the recommended intervals for booster shots.

Common Hazards: Identify five common household items or substances that are hazardous to cats.

Behavioral Needs: Find one authoritative webpage offering practical advice on preventing destructive cat behaviors, such as furniture scratching and aggressive playing.

Grooming Requirements: Describe the recommended grooming routine for short-haired cats, including suggested grooming frequency and recommended grooming tools or products.

Emergency Preparedness: Locate clear guidelines provided by veterinary or animal welfare organizations on managing cat-related medical emergencies (e.g., poisoning, injuries, sudden illnesses) before reaching veterinary care.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                     #
# --------------------------------------------------------------------------- #
class DietaryInfo(BaseModel):
    """Information about cat dietary needs and sources"""
    nutritional_requirements: Optional[str] = None
    foods_to_avoid: Optional[str] = None
    source_url: Optional[str] = None


class Vaccination(BaseModel):
    """Information about a specific vaccination"""
    name: Optional[str] = None
    initial_age: Optional[str] = None
    booster_intervals: Optional[str] = None


class VaccinationInfo(BaseModel):
    """Information about cat vaccinations and sources"""
    vaccinations: List[Vaccination] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class HazardInfo(BaseModel):
    """Information about household hazards for cats"""
    hazards: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class BehavioralInfo(BaseModel):
    """Information about cat behavior management"""
    behavioral_tips: Optional[str] = None
    source_url: Optional[str] = None


class GroomingInfo(BaseModel):
    """Information about cat grooming requirements"""
    frequency: Optional[str] = None
    tools: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class EmergencyInfo(BaseModel):
    """Information about cat emergency management"""
    guidelines: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class CatCareInfo(BaseModel):
    """Complete extracted information about cat care"""
    dietary: Optional[DietaryInfo] = None
    vaccinations: Optional[VaccinationInfo] = None
    hazards: Optional[HazardInfo] = None
    behavioral: Optional[BehavioralInfo] = None
    grooming: Optional[GroomingInfo] = None
    emergency: Optional[EmergencyInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_dietary_info() -> str:
    return """
    Extract information about cat dietary needs from the answer. Please identify:
    
    1. Summary of key nutritional requirements for healthy adult indoor cats
    2. Summary of foods that must be avoided
    3. The URL of the source article or blog mentioned both of them (if any)
    
    Return null for any missing information. Extract information exactly as it appears in the text without adding any additional information.
    """


def prompt_extract_vaccination_info() -> str:
    return """
    Extract information about cat vaccinations from the answer. Specifically:
    
    1. A list of core vaccinations recommended for indoor cats, including:
       - The name of each vaccination
       - The typical age for initial vaccination (if mentioned)
       - The recommended intervals for booster shots (if mentioned)
    2. All URLs mentioned for the information about cat vaccinations
    
    Return null for any missing information. Extract information exactly as it appears in the text without adding any additional information.
    """


def prompt_extract_hazard_info() -> str:
    return """
    Extract information about household hazards for cats from the answer. Specifically:
    
    1. A list of household items or substances identified as hazardous to cats
    2. The URLs of any sources mentioned for this information
    
    Return null for any missing information. Extract information exactly as it appears in the text without adding any additional information.
    """


def prompt_extract_behavioral_info() -> str:
    return """
    Extract information about preventing destructive cat behaviors from the answer. Specifically:
    
    1. Practical advice or tips for preventing destructive behaviors like furniture scratching and aggressive playing
    2. The URL of the authoritative webpage mentioned as a source
    
    Return null for any missing information. Extract information exactly as it appears in the text without adding any additional information.
    """


def prompt_extract_grooming_info() -> str:
    return """
    Extract information about grooming requirements for short-haired cats from the answer. Specifically:
    
    1. The recommended grooming frequency
    2. The recommended grooming tools or products
    3. The URLs of any sources mentioned for this information
    
    Return null for any missing information. Extract information exactly as it appears in the text without adding any additional information.
    """


def prompt_extract_emergency_info() -> str:
    return """
    Extract information about managing cat-related medical emergencies from the answer. Specifically:
    
    1. Guidelines for managing cat-related medical emergencies before reaching veterinary care
    2. The URLs of any veterinary or animal welfare organizations mentioned as sources
    
    Return null for any missing information. Extract information exactly as it appears in the text without adding any additional information.
    """


# --------------------------------------------------------------------------- #
# Verification functions for each section                                     #
# --------------------------------------------------------------------------- #
async def verify_dietary_section(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        dietary_info: DietaryInfo,
) -> None:
    """
    Verify the dietary section of the answer.
    """
    dietary_node = evaluator.add_parallel(
        id="dietary_section",
        desc="The answer provides comprehensive information about cat dietary needs with a credible source",
        parent=parent_node,
        critical=False,
    )

    # Check if dietary info and source exist
    existence_check = evaluator.add_custom_node(
        result=(dietary_info is not None and
                dietary_info.source_url is not None and
                dietary_info.nutritional_requirements is not None and 
                dietary_info.foods_to_avoid is not None),
        id="dietary_source_exists",
        desc="Dietary information with source URL is provided",
        parent=dietary_node,
        critical=True
    )

    # Verify the content from the source - comprehensive article check
    comprehensive_node = evaluator.add_leaf(
        id="dietary_comprehensive",
        desc="The source is a comprehensive article or blog that contains both nutritional requirements and foods to avoid for cats",
        parent=dietary_node,
        critical=True,
    )

    # Verify if the main points are supported by the source
    points_verified_node = evaluator.add_leaf(
        id="dietary_points_verified",
        desc="The main points summarized in the answer are supported by the source",
        parent=dietary_node,
        critical=True,
    )

    # Always call verify methods, let the framework handle missing data
    await evaluator.verify(
        claim="The provided URL is a comprehensive article or blog that contains information about both nutritional requirements for cats AND foods that should be avoided for cats",
        node=comprehensive_node,
        sources=dietary_info.source_url if dietary_info else None,
        additional_instruction="Check if the URL leads to a comprehensive article or blog that contains BOTH information about nutritional requirements for cats AND foods that should be avoided. Both aspects must be present to pass this verification."
    )

    # Create claim for points verification
    claim_parts = []
    if dietary_info and dietary_info.nutritional_requirements:
        claim_parts.append(f"Nutritional requirements for cats include: {dietary_info.nutritional_requirements}")
    if dietary_info and dietary_info.foods_to_avoid:
        claim_parts.append(f"Foods to avoid for cats include: {dietary_info.foods_to_avoid}")
    
    claim = " AND ".join(claim_parts) if claim_parts else "The dietary information is supported by the source"
    
    await evaluator.verify(
        claim=claim,
        node=points_verified_node,
        sources=dietary_info.source_url if dietary_info else None,
        additional_instruction="Verify if the main points mentioned in the answer are supported by the source. Check for both nutritional requirements and foods to avoid if they are mentioned in the claim."
    )


async def verify_vaccination_section(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        vaccination_info: VaccinationInfo,
) -> None:
    """
    Verify the vaccination section of the answer.
    """
    vaccination_node = evaluator.add_parallel(
        id="vaccination_section",
        desc="The answer provides information about core vaccinations for indoor cats from reputable sources",
        parent=parent_node,
        critical=False,
    )

    # Check if vaccination info exists with sources
    info_exists = evaluator.add_custom_node(
        result=(vaccination_info is not None and 
                vaccination_info.vaccinations and 
                len(vaccination_info.vaccinations) > 0 and
                vaccination_info.source_urls and 
                len(vaccination_info.source_urls) > 0 and
                all(v.initial_age is not None and v.booster_intervals is not None 
                    for v in vaccination_info.vaccinations)),
        id="vaccination_info_exists",
        desc="Vaccination list with source URLs is provided",
        parent=vaccination_node,
        critical=True
    )

    # Verify reputable source
    reputable_source_node = evaluator.add_leaf(
        id="vaccination_reputable_source",
        desc="At least one source is from a reputable veterinary organization",
        parent=vaccination_node,
        critical=True,
    )

    # Verify vaccination content
    vaccine_verification_node = evaluator.add_leaf(
        id="vaccination_content_verification",
        desc="The vaccination list and timing information are supported by the reputable sources",
        parent=vaccination_node,
        critical=True,
    )

    # Verify reputable source
    sources = vaccination_info.source_urls if vaccination_info and vaccination_info.source_urls else []
    await evaluator.verify(
        claim="This URL is from a reputable veterinary source such as a veterinary association, veterinary hospital, veterinary school, or established pet health organization",
        node=reputable_source_node,
        sources=sources,
        additional_instruction="Check if the URL belongs to a reputable veterinary source like AVMA, AAHA, university vet school, or established animal hospital/clinic."
    )

    # Verify vaccination content
    if vaccination_info and vaccination_info.vaccinations and reputable_source_node.status == "passed":
        # Create combined claim for all vaccinations
        verification_claims = []
        for vax in vaccination_info.vaccinations:
            claim_parts = [f"'{vax.name}' is a core vaccination for indoor cats"]
            if vax.initial_age:
                claim_parts.append(f"initial vaccination is typically at {vax.initial_age}")
            if vax.booster_intervals:
                claim_parts.append(f"booster shots are recommended {vax.booster_intervals}")
            verification_claims.append(" and ".join(claim_parts))
        
        # Verify each claim
        all_verified = True
        for i, claim in enumerate(verification_claims):
            verified = await evaluator.verify(
                claim=claim,
                node=None,
                sources=sources,
                additional_instruction="Verify if this vaccination information is supported by the sources. Check for both the vaccine name and timing details if provided."
            )
            if not verified:
                all_verified = False
                break
        vaccine_verification_node.score = 1.0 if all_verified else 0.0
        vaccine_verification_node.status = "passed" if all_verified else "failed"
    else:
        await evaluator.verify(
            claim="There is no vaccination information.",
            node=vaccine_verification_node,
            sources=sources,
            additional_instruction="Verify if this vaccination information is supported by the sources. Check for both the vaccine name and timing details if provided."
        )


async def verify_hazard_section(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        hazard_info: HazardInfo,
) -> None:
    """
    Verify the common hazards section of the answer.
    """
    hazard_node = evaluator.add_parallel(
        id="hazard_section",
        desc="The answer identifies five common household items or substances that are hazardous to cats",
        parent=parent_node,
        critical=False,
    )

    # Check completeness
    hazards_complete = evaluator.add_custom_node(
        result=(hazard_info is not None and 
                hazard_info.hazards and 
                len(hazard_info.hazards) >= 5 and
                hazard_info.source_urls and 
                len(hazard_info.source_urls) > 0),
        id="hazards_complete",
        desc="At least 5 hazards are identified with sources",
        parent=hazard_node,
        critical=True
    )

    # Verify the hazards
    all_hazards_node = evaluator.add_leaf(
        id="hazard_verification",
        desc="The five hazardous items identified are supported by the provided sources",
        parent=hazard_node,
        critical=True,
    )

    # Verify hazards
    sources = hazard_info.source_urls if hazard_info and hazard_info.source_urls else []
    
    if hazard_info and hazard_info.hazards:
        hazards_to_verify = hazard_info.hazards[:5]

        all_verified = True
        for i, hazard in enumerate(hazards_to_verify):
            claim = f"{hazard} is hazardous to cats."
            
            verified = await evaluator.verify(
                claim=claim,
                node=None,
                sources=sources,
                additional_instruction="Verify if the sources confirm that the item is hazardous to cats."
            )
            if not verified:
                all_verified = False
                break
        all_hazards_node.score = 1.0 if all_verified else 0.0
        all_hazards_node.status = "passed" if all_verified else "failed"
    else:
        combined_claim = "There is no hazardous item provided."
        await evaluator.verify(
            claim=combined_claim,
            node=all_hazards_node,
            sources=sources,
            additional_instruction="Verify if the sources confirm that these items are hazardous to cats."
        )


async def verify_behavioral_section(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        behavioral_info: BehavioralInfo,
) -> None:
    """
    Verify the behavioral section of the answer.
    """
    behavioral_node = evaluator.add_parallel(
        id="behavioral_section",
        desc="The answer provides practical advice on preventing destructive cat behaviors from an authoritative source",
        parent=parent_node,
        critical=False,
    )

    # Check if behavioral info exists
    info_exists = evaluator.add_custom_node(
        result=(behavioral_info is not None and behavioral_info.source_url is not None),
        id="behavioral_info_exists",
        desc="Behavioral information with source URL is provided",
        parent=behavioral_node,
        critical=True
    )

    # Verify practical advice
    practical_advice_node = evaluator.add_leaf(
        id="behavioral_practical_advice",
        desc="The source provides practical advice on preventing destructive cat behaviors",
        parent=behavioral_node,
        critical=True,
    )

    # Verify practical advice
    if behavioral_info and behavioral_info.behavioral_tips:
        claim = f"The source provides practical advice on preventing destructive cat behaviors, such as: {behavioral_info.behavioral_tips}"
    else:
        claim = "The source provides practical, actionable advice on preventing destructive cat behaviors such as furniture scratching and aggressive playing"
    
    await evaluator.verify(
        claim=claim,
        node=practical_advice_node,
        sources=behavioral_info.source_url if behavioral_info else None,
        additional_instruction="Check if the source provides specific, actionable advice on preventing destructive behaviors like furniture scratching and aggressive playing. The advice should be practical and implementable, not just theoretical explanations."
    )


async def verify_grooming_section(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        grooming_info: GroomingInfo,
) -> None:
    """
    Verify the grooming section of the answer.
    """
    grooming_node = evaluator.add_parallel(
        id="grooming_section",
        desc="The answer describes the recommended grooming routine for short-haired cats",
        parent=parent_node,
        critical=False,
    )

    # Check completeness
    grooming_complete = evaluator.add_custom_node(
        result=(grooming_info is not None and 
                grooming_info.frequency is not None and
                grooming_info.tools is not None and
                grooming_info.source_urls and 
                len(grooming_info.source_urls) > 0),
        id="grooming_complete",
        desc="Grooming frequency, tools, and sources are all provided",
        parent=grooming_node,
        critical=True
    )

    # Verify grooming information
    info_verification_node = evaluator.add_leaf(
        id="grooming_info_verification",
        desc="The grooming frequency and tools information is supported by the provided sources",
        parent=grooming_node,
        critical=True,
    )

    # Verify grooming info
    sources = grooming_info.source_urls if grooming_info and grooming_info.source_urls else []
    
    if grooming_info and grooming_info.frequency and grooming_info.tools:
        claim = f"For short-haired cats, recommended grooming frequency is {grooming_info.frequency} and recommended grooming tools include: {grooming_info.tools}"
    else:
        claim = "The grooming frequency and tools for short-haired cats are supported by the sources"
    
    await evaluator.verify(
        claim=claim,
        node=info_verification_node,
        sources=sources,
        additional_instruction="Verify if the sources confirm the recommended grooming frequency and at least some of the listed grooming tools for short-haired cats."
    )


async def verify_emergency_section(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        emergency_info: EmergencyInfo,
) -> None:
    """
    Verify the emergency preparedness section of the answer.
    """
    emergency_node = evaluator.add_parallel(
        id="emergency_section",
        desc="The answer provides clear guidelines on managing cat-related medical emergencies from veterinary or animal welfare organizations",
        parent=parent_node,
        critical=False,
    )

    # Check if info exists
    info_exists = evaluator.add_custom_node(
        result=(emergency_info is not None and 
                emergency_info.source_urls and 
                len(emergency_info.source_urls) > 0),
        id="emergency_info_exists",
        desc="Emergency information with source URLs is provided",
        parent=emergency_node,
        critical=True
    )

    # Verify organization source
    org_source_node = evaluator.add_leaf(
        id="emergency_org_source",
        desc="At least one source is from a veterinary or animal welfare organization",
        parent=emergency_node,
        critical=True,
    )

    # Verify guidelines content
    guidelines_node = evaluator.add_leaf(
        id="emergency_guidelines",
        desc="The sources provide guidelines for managing cat-related medical emergencies before reaching veterinary care",
        parent=emergency_node,
        critical=True,
    )

    # Verify organization source
    sources = emergency_info.source_urls if emergency_info and emergency_info.source_urls else []
    
    await evaluator.verify(
        claim="This URL is from a veterinary or animal welfare organization",
        node=org_source_node,
        sources=sources,
        additional_instruction="Check if the URL belongs to a veterinary organization, animal hospital, animal welfare group, or similar authority on pet health emergencies."
    )

    # Verify guidelines
    if emergency_info and emergency_info.guidelines:
        claim = f"The sources provide guidelines on managing cat emergencies before reaching veterinary care, including information on topics such as poisoning, injuries, or sudden illnesses. They should cover some of the points of the following guidelines: {emergency_info.guidelines}"
    else:
        claim = "The sources provide guidelines on managing cat emergencies before reaching veterinary care"
    
    await evaluator.verify(
        claim=claim,
        node=guidelines_node,
        sources=sources,
        additional_instruction="Check if the sources provide clear guidelines for what to do in case of cat medical emergencies BEFORE reaching a veterinarian or professional care. The guidelines should cover topics like poisoning, injuries, or sudden illnesses."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: openai.AsyncAzureOpenAI,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer to the pet_food_vet task and return a structured result dictionary.
    """
    # -------- 1. Set up evaluator ----------------------------- #
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

    # -------- 2. Extract structured information from each section -------- #
    # Extract dietary information
    dietary_info = await evaluator.extract(
        prompt=prompt_extract_dietary_info(),
        template_class=DietaryInfo,
        extraction_name="dietary_info"
    )
    
    # Extract vaccination information
    vaccination_info = await evaluator.extract(
        prompt=prompt_extract_vaccination_info(),
        template_class=VaccinationInfo,
        extraction_name="vaccination_info"
    )
    
    # Extract hazard information
    hazard_info = await evaluator.extract(
        prompt=prompt_extract_hazard_info(),
        template_class=HazardInfo,
        extraction_name="hazard_info"
    )
    
    # Extract behavioral information
    behavioral_info = await evaluator.extract(
        prompt=prompt_extract_behavioral_info(),
        template_class=BehavioralInfo,
        extraction_name="behavioral_info"
    )
    
    # Extract grooming information
    grooming_info = await evaluator.extract(
        prompt=prompt_extract_grooming_info(),
        template_class=GroomingInfo,
        extraction_name="grooming_info"
    )
    
    # Extract emergency information
    emergency_info = await evaluator.extract(
        prompt=prompt_extract_emergency_info(),
        template_class=EmergencyInfo,
        extraction_name="emergency_info"
    )

    # -------- 3. Build verification tree -------------------------------- #
    # Create verification nodes for each section
    await verify_dietary_section(evaluator, root, dietary_info)
    await verify_vaccination_section(evaluator, root, vaccination_info)
    await verify_hazard_section(evaluator, root, hazard_info)
    await verify_behavioral_section(evaluator, root, behavioral_info)
    await verify_grooming_section(evaluator, root, grooming_info)
    await verify_emergency_section(evaluator, root, emergency_info)

    # -------- 4. Aggregate score ---------------------------------------- #
    final_score = evaluator.score()  # triggers recursive aggregation

    # -------- 5. Return structured result ------------------------------- #
    return evaluator.get_summary()