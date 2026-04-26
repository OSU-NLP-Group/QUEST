import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "akc_bis_2025"
TASK_DESCRIPTION = (
    "For the three major American Kennel Club (AKC) sanctioned dog shows held in 2025—the National Dog Show Presented by Purina "
    "(Thanksgiving broadcast), the AKC National Championship Presented by Royal Canin (December), and the Westminster Kennel Club "
    "Dog Show Presented by Purina Pro Plan (February)—identify the Best in Show winner from each competition and provide the "
    "following information for each winning dog:\n\n"
    "1. Winner Identification: The dog's call name (not the full registered name), breed, and a reference URL confirming this information.\n"
    "2. Dog Details: The dog's age at the time of the show (or date of birth for Westminster), sex (male/female), and the AKC breed group to which "
    "the breed belongs, along with a reference URL.\n"
    "3. Handler Information: The full name of the professional handler who presented the dog during Best in Show judging, and (where available) the "
    "handler's location (city and state) or kennel program name, along with a reference URL.\n"
    "4. Additional Requirements:\n"
    "   - For the National Dog Show winner: Describe the breed's coat color according to AKC breed standard and key temperament characteristics, plus "
    "provide all co-owners' names and the breeder's name with reference URLs.\n"
    "   - For the AKC National Championship winner: Describe the dog's major competitive achievements prior to this win, including national specialty wins "
    "if applicable, plus provide all co-owners' names and breeder(s)' names with reference URLs.\n"
    "   - For the Westminster winner: Provide the dog's sire (father) and dam (mother) names from the pedigree, the AKC breed standard height range and "
    "weight range for the dog's sex, plus all co-owners' names and the breeder's name with reference URLs.\n\n"
    "All information must be verifiable through official dog show results, AKC resources, or credible dog sport news sources. Provide reference URLs for each category of information."
)

NATIONAL_DOG_SHOW_NAME = "2025 National Dog Show Presented by Purina"
AKC_NATIONALS_NAME = "2025 AKC National Championship Presented by Royal Canin"
WESTMINSTER_NAME = "2025 Westminster Kennel Club Dog Show Presented by Purina Pro Plan"

# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class Show1Extraction(BaseModel):
    # Winner identification
    call_name: Optional[str] = None
    breed: Optional[str] = None
    id_urls: List[str] = Field(default_factory=list)

    # Dog details
    age_at_show: Optional[str] = None  # e.g., "3 years", "3-year-old"
    sex: Optional[str] = None  # male/female
    group: Optional[str] = None  # AKC Group name (e.g., "Hound Group", "Working Group", etc.)
    details_urls: List[str] = Field(default_factory=list)

    # Handler info
    handler_name: Optional[str] = None
    handler_location: Optional[str] = None  # city, state OR kennel/program name
    handler_urls: List[str] = Field(default_factory=list)

    # Breed characteristics (per AKC standard)
    coat_color_standard: Optional[str] = None
    breed_temperament: Optional[str] = None
    breed_standard_urls: List[str] = Field(default_factory=list)

    # Ownership
    owners: List[str] = Field(default_factory=list)  # all co-owners
    breeders: List[str] = Field(default_factory=list)
    ownership_urls: List[str] = Field(default_factory=list)


class Show2Extraction(BaseModel):
    # Winner identification
    call_name: Optional[str] = None
    breed: Optional[str] = None
    id_urls: List[str] = Field(default_factory=list)

    # Dog details
    age_at_show: Optional[str] = None
    sex: Optional[str] = None
    group: Optional[str] = None
    show_date: Optional[str] = None
    details_urls: List[str] = Field(default_factory=list)

    # Handler info
    handler_name: Optional[str] = None
    handler_kennel: Optional[str] = None  # kennel/program name, or location if given as such
    handler_urls: List[str] = Field(default_factory=list)

    # Achievements
    previous_wins: List[str] = Field(default_factory=list)  # major wins before this BIS
    specialty_wins: List[str] = Field(default_factory=list)  # national specialty wins if any
    achievements_urls: List[str] = Field(default_factory=list)

    # Ownership
    owners: List[str] = Field(default_factory=list)
    breeders: List[str] = Field(default_factory=list)
    ownership_urls: List[str] = Field(default_factory=list)


class Show3Extraction(BaseModel):
    # Winner identification
    call_name: Optional[str] = None
    breed: Optional[str] = None
    id_urls: List[str] = Field(default_factory=list)

    # Dog details
    sex: Optional[str] = None
    birth_date: Optional[str] = None  # Date of Birth
    group: Optional[str] = None
    details_urls: List[str] = Field(default_factory=list)

    # Handler info
    handler_name: Optional[str] = None
    handler_location_or_kennel: Optional[str] = None
    handler_urls: List[str] = Field(default_factory=list)

    # Pedigree
    sire: Optional[str] = None
    dam: Optional[str] = None
    pedigree_urls: List[str] = Field(default_factory=list)

    # Breed standard specifics
    height_standard: Optional[str] = None  # Height range for this sex
    weight_standard: Optional[str] = None  # Weight range for this sex
    breed_standard_urls: List[str] = Field(default_factory=list)

    # Ownership
    owners: List[str] = Field(default_factory=list)
    breeders: List[str] = Field(default_factory=list)
    ownership_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_show1() -> str:
    return """
    Extract structured information about the Best in Show winner for the 2025 National Dog Show Presented by Purina (NBC Thanksgiving broadcast).
    IMPORTANT:
    - call_name is the everyday dog name used publicly (not the long AKC registered name).
    - Provide all relevant URLs explicitly mentioned in the answer for each category.

    Return a JSON object with these fields:
    1) call_name: string | null
    2) breed: string | null
    3) id_urls: array of strings (URLs confirming the winner's call name and breed)
    4) age_at_show: string | null (e.g., "3 years", "3-year-old")
    5) sex: string | null ("male" or "female", case-insensitive OK)
    6) group: string | null (AKC breed group, e.g., "Hound Group", "Working Group", "Non-Sporting Group", etc.)
    7) details_urls: array of strings (URLs confirming age/sex/group or other dog details; may include official result pages or credible news)
    8) handler_name: string | null (full professional handler name)
    9) handler_location: string | null (city and state or kennel/program name if that's how it's described)
    10) handler_urls: array of strings (URLs confirming handler info during Best in Show)
    11) coat_color_standard: string | null (AKC standard coat color description text/synopsis for this breed)
    12) breed_temperament: string | null (AKC standard temperament traits synopsis)
    13) breed_standard_urls: array of strings (AKC breed page(s) or official breed club standard pages; must be authoritative)
    14) owners: array of strings (all co-owners' names; split multi-name strings into separate items)
    15) breeders: array of strings (all breeders' names; split if multiple)
    16) ownership_urls: array of strings (URLs confirming ownership/breeder info)

    Notes:
    - Only include URLs explicitly present in the answer.
    - If an item is not present, set it to null or empty list as appropriate.
    """


def prompt_extract_show2() -> str:
    return """
    Extract structured information about the Best in Show winner for the 2025 AKC National Championship Presented by Royal Canin (December).
    IMPORTANT:
    - call_name is the everyday dog name used publicly (not the long AKC registered name).
    - Provide all relevant URLs explicitly mentioned in the answer for each category.

    Return a JSON object with these fields:
    1) call_name: string | null
    2) breed: string | null
    3) id_urls: array of strings
    4) age_at_show: string | null
    5) sex: string | null
    6) group: string | null
    7) show_date: string | null (date of Best in Show judging if provided)
    8) details_urls: array of strings (URLs confirming age/sex/group/show date, etc.)
    9) handler_name: string | null
    10) handler_kennel: string | null (kennel/program name or location if presented like that)
    11) handler_urls: array of strings
    12) previous_wins: array of strings (major achievements prior to this BIS; each win/achievement as one string)
    13) specialty_wins: array of strings (national specialty wins if any; each as separate string)
    14) achievements_urls: array of strings (URLs confirming achievements)
    15) owners: array of strings (all co-owners)
    16) breeders: array of strings (all breeders)
    17) ownership_urls: array of strings (URLs confirming ownership/breeder info)

    Notes:
    - Only include URLs explicitly present in the answer.
    - If an item is not present, set it to null or empty list as appropriate.
    """


def prompt_extract_show3() -> str:
    return """
    Extract structured information about the Best in Show winner for the 2025 Westminster Kennel Club Dog Show Presented by Purina Pro Plan (February).
    IMPORTANT:
    - call_name is the everyday dog name used publicly (not the long AKC registered name).
    - Provide all relevant URLs explicitly mentioned in the answer for each category.

    Return a JSON object with these fields:
    1) call_name: string | null
    2) breed: string | null
    3) id_urls: array of strings
    4) sex: string | null
    5) birth_date: string | null (the dog's date of birth)
    6) group: string | null
    7) details_urls: array of strings (URLs confirming sex/DOB/group, etc.)
    8) handler_name: string | null
    9) handler_location_or_kennel: string | null (city/state or kennel/program if applicable)
    10) handler_urls: array of strings
    11) sire: string | null (dog's father)
    12) dam: string | null (dog's mother)
    13) pedigree_urls: array of strings (URLs confirming sire/dam pedigree information)
    14) height_standard: string | null (AKC standard height range for the breed & sex, textual)
    15) weight_standard: string | null (AKC standard weight range for the breed & sex, textual)
    16) breed_standard_urls: array of strings (AKC breed standard URLs)
    17) owners: array of strings (all co-owners)
    18) breeders: array of strings (all breeders)
    19) ownership_urls: array of strings (URLs confirming ownership/breeder info)

    Notes:
    - Only include URLs explicitly present in the answer.
    - If an item is not present, set it to null or empty list as appropriate.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def combine_urls(*url_containers: Optional[List[str]]) -> List[str]:
    """Merge multiple url lists, removing empties and duplicates while preserving order."""
    seen = set()
    merged: List[str] = []
    for urls in url_containers:
        if not urls:
            continue
        for u in urls:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _readable_list(items: Optional[List[str]]) -> str:
    if not items:
        return ""
    return ", ".join([s for s in items if s])


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_show1(evaluator: Evaluator, parent: VerificationNode, data: Show1Extraction) -> None:
    """
    National Dog Show (Thanksgiving 2025) winner information
    """
    show_node = evaluator.add_sequential(
        id="show_1_national_dog_show",
        desc="National Dog Show (Thanksgiving 2025) winner information",
        parent=parent,
        critical=False
    )

    # 1) Winner Identification (critical group)
    ident_node = evaluator.add_parallel(
        id="show_1_winner_identification",
        desc="Correctly identify the Best in Show winner's name and breed",
        parent=show_node,
        critical=True
    )

    # Reference presence (critical)
    ref_present_node = evaluator.add_custom_node(
        result=bool(data.id_urls),
        id="show_1_reference",
        desc="Provide valid reference URL for the winner identification",
        parent=ident_node,
        critical=True
    )

    # Dog name (critical)
    name_node = evaluator.add_leaf(
        id="show_1_dog_name",
        desc="Provide the winner's call name",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Best in Show winner at the {NATIONAL_DOG_SHOW_NAME} has the call name '{data.call_name}'.",
        node=name_node,
        sources=data.id_urls,
        additional_instruction="Confirm the actual call name (not long registered name) on the cited pages, and that it refers to the 2025 National Dog Show BIS winner.",
        extra_prerequisites=[ref_present_node]
    )

    # Breed (critical)
    breed_node = evaluator.add_leaf(
        id="show_1_breed",
        desc="Provide the winner's breed",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Best in Show winner at the {NATIONAL_DOG_SHOW_NAME} is a {data.breed}.",
        node=breed_node,
        sources=data.id_urls,
        additional_instruction="Verify that the cited source identifies the BIS winner's breed for the 2025 National Dog Show.",
        extra_prerequisites=[ref_present_node]
    )

    # 2) Dog Details (non-critical group)
    details_node = evaluator.add_parallel(
        id="show_1_dog_details",
        desc="Provide accurate details about the winning dog",
        parent=show_node,
        critical=False
    )

    # Age (non-critical)
    age_node = evaluator.add_leaf(
        id="show_1_age",
        desc="Provide the dog's age at time of show",
        parent=details_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The dog's age at the time of the {NATIONAL_DOG_SHOW_NAME} was {data.age_at_show}.",
        node=age_node,
        sources=combine_urls(data.details_urls, data.id_urls),
        additional_instruction="Accept phrasing like '3-year-old' or 'age 3'. Ensure the source refers to the same BIS dog.",
        extra_prerequisites=[]
    )

    # Sex (non-critical)
    sex_node = evaluator.add_leaf(
        id="show_1_sex",
        desc="Provide the dog's sex (male/female)",
        parent=details_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The dog's sex is {data.sex}.",
        node=sex_node,
        sources=combine_urls(data.details_urls, data.id_urls),
        additional_instruction="Verify the dog's sex from credible pages mentioning the BIS dog.",
        extra_prerequisites=[]
    )

    # AKC Group (critical within details)
    group_node = evaluator.add_leaf(
        id="show_1_group",
        desc="Identify the AKC breed group the winner belongs to",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The {data.breed} belongs to the {data.group} in the American Kennel Club (AKC).",
        node=group_node,
        sources=combine_urls(data.breed_standard_urls, data.details_urls),
        additional_instruction="Prefer AKC breed page or official breed club standard confirming the AKC Group classification.",
        extra_prerequisites=[]
    )

    # Details reference presence (critical within details)
    details_ref_node = evaluator.add_custom_node(
        result=bool(data.details_urls),
        id="show_1_details_reference",
        desc="Provide valid reference URL for dog details",
        parent=details_node,
        critical=True
    )

    # 3) Handler Info (non-critical group)
    handler_node = evaluator.add_parallel(
        id="show_1_handler_info",
        desc="Provide accurate information about the handler",
        parent=show_node,
        critical=False
    )

    handler_ref_node = evaluator.add_custom_node(
        result=bool(data.handler_urls),
        id="show_1_handler_reference",
        desc="Provide valid reference URL for handler information",
        parent=handler_node,
        critical=True
    )

    handler_name_node = evaluator.add_leaf(
        id="show_1_handler_name",
        desc="Provide the handler's full name",
        parent=handler_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The dog was presented during Best in Show at the {NATIONAL_DOG_SHOW_NAME} by handler {data.handler_name}.",
        node=handler_name_node,
        sources=combine_urls(data.handler_urls, data.id_urls),
        additional_instruction="Verify the named person is the professional handler for the BIS ring.",
        extra_prerequisites=[handler_ref_node]
    )

    handler_loc_node = evaluator.add_leaf(
        id="show_1_handler_location",
        desc="Provide the handler's city and state",
        parent=handler_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The handler's location or kennel/program is '{data.handler_location}'.",
        node=handler_loc_node,
        sources=data.handler_urls,
        additional_instruction="If the answer presents a kennel/program instead of city/state, that is acceptable.",
        extra_prerequisites=[handler_ref_node]
    )

    # 4) Breed characteristics (non-critical group)
    breed_char_node = evaluator.add_parallel(
        id="show_1_breed_characteristics",
        desc="Provide breed-specific characteristics according to AKC breed standard",
        parent=show_node,
        critical=False
    )

    breed_std_ref_node = evaluator.add_custom_node(
        result=bool(data.breed_standard_urls),
        id="show_1_breed_reference",
        desc="Provide valid reference URL for breed standard information",
        parent=breed_char_node,
        critical=True
    )

    coat_color_node = evaluator.add_leaf(
        id="show_1_coat_color_standard",
        desc="Describe the breed's coat color according to AKC standard",
        parent=breed_char_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"According to the AKC standard for the {data.breed}, acceptable coat color description includes: {data.coat_color_standard}.",
        node=coat_color_node,
        sources=data.breed_standard_urls,
        additional_instruction="Check the AKC breed standard page for allowed colors/markings; paraphrase is acceptable if equivalent.",
        extra_prerequisites=[breed_std_ref_node]
    )

    temperament_node = evaluator.add_leaf(
        id="show_1_breed_temperament",
        desc="Describe key temperament characteristics from breed standard",
        parent=breed_char_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"According to the AKC standard for the {data.breed}, key temperament traits include: {data.breed_temperament}.",
        node=temperament_node,
        sources=data.breed_standard_urls,
        additional_instruction="Verify temperament descriptors appear on AKC or official breed standard resources.",
        extra_prerequisites=[breed_std_ref_node]
    )

    # 5) Ownership (non-critical group)
    ownership_node = evaluator.add_parallel(
        id="show_1_ownership",
        desc="Provide ownership information",
        parent=show_node,
        critical=False
    )

    ownership_ref_node = evaluator.add_custom_node(
        result=bool(data.ownership_urls),
        id="show_1_ownership_reference",
        desc="Provide valid reference URL for ownership information",
        parent=ownership_node,
        critical=True
    )

    owners_node = evaluator.add_leaf(
        id="show_1_owners",
        desc="List all co-owners of the dog",
        parent=ownership_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The co-owners of the Best in Show winner are: {_readable_list(data.owners)}.",
        node=owners_node,
        sources=data.ownership_urls,
        additional_instruction="Names may be separated by commas or 'and'; confirm all listed co-owners are present on the cited page(s).",
        extra_prerequisites=[ownership_ref_node]
    )

    breeder_node = evaluator.add_leaf(
        id="show_1_breeder",
        desc="Identify the dog's breeder",
        parent=ownership_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The breeder(s) of the dog are: {_readable_list(data.breeders)}.",
        node=breeder_node,
        sources=data.ownership_urls,
        additional_instruction="Confirm breeder name(s) on credible sources such as official results, AKC pages, or breeder/kennel pages.",
        extra_prerequisites=[ownership_ref_node]
    )


async def verify_show2(evaluator: Evaluator, parent: VerificationNode, data: Show2Extraction) -> None:
    """
    AKC National Championship (December 2025) winner information
    """
    show_node = evaluator.add_sequential(
        id="show_2_akc_national",
        desc="AKC National Championship (December 2025) winner information",
        parent=parent,
        critical=False
    )

    # 1) Winner Identification (critical group)
    ident_node = evaluator.add_parallel(
        id="show_2_winner_identification",
        desc="Correctly identify the Best in Show winner's name and breed",
        parent=show_node,
        critical=True
    )

    ref_present_node = evaluator.add_custom_node(
        result=bool(data.id_urls),
        id="show_2_reference",
        desc="Provide valid reference URL for the winner identification",
        parent=ident_node,
        critical=True
    )

    name_node = evaluator.add_leaf(
        id="show_2_dog_name",
        desc="Provide the winner's call name",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Best in Show winner at the {AKC_NATIONALS_NAME} has the call name '{data.call_name}'.",
        node=name_node,
        sources=data.id_urls,
        additional_instruction="Confirm actual call name (not long registered name) for the 2025 AKC National Championship BIS winner.",
        extra_prerequisites=[ref_present_node]
    )

    breed_node = evaluator.add_leaf(
        id="show_2_breed",
        desc="Provide the winner's breed",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Best in Show winner at the {AKC_NATIONALS_NAME} is a {data.breed}.",
        node=breed_node,
        sources=data.id_urls,
        additional_instruction="Verify the breed from official results or credible dog-sport news sources that explicitly reference the 2025 event.",
        extra_prerequisites=[ref_present_node]
    )

    # 2) Dog Details (non-critical group)
    details_node = evaluator.add_parallel(
        id="show_2_dog_details",
        desc="Provide accurate details about the winning dog",
        parent=show_node,
        critical=False
    )

    age_node = evaluator.add_leaf(
        id="show_2_age",
        desc="Provide the dog's age at time of show",
        parent=details_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The dog's age at the time of the {AKC_NATIONALS_NAME} was {data.age_at_show}.",
        node=age_node,
        sources=combine_urls(data.details_urls, data.id_urls),
        additional_instruction="Accept phrasing like '3-year-old' or 'age 3'; ensure it refers to this BIS winner.",
        extra_prerequisites=[]
    )

    sex_node = evaluator.add_leaf(
        id="show_2_sex",
        desc="Provide the dog's sex (male/female)",
        parent=details_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The dog's sex is {data.sex}.",
        node=sex_node,
        sources=combine_urls(data.details_urls, data.id_urls),
        additional_instruction="Confirm the dog's sex on cited pages mentioning the BIS dog for this event.",
        extra_prerequisites=[]
    )

    group_node = evaluator.add_leaf(
        id="show_2_group",
        desc="Identify the AKC breed group the winner belongs to",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The {data.breed} belongs to the {data.group} in the American Kennel Club (AKC).",
        node=group_node,
        sources=data.details_urls,
        additional_instruction="Prefer AKC breed page or authoritative breed club standard confirming the AKC Group classification.",
        extra_prerequisites=[]
    )

    show_date_node = evaluator.add_leaf(
        id="show_2_show_date",
        desc="Provide the date of Best in Show judging",
        parent=details_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The Best in Show judging took place on {data.show_date}.",
        node=show_date_node,
        sources=combine_urls(data.details_urls, data.id_urls),
        additional_instruction="Check official AKC National Championship schedule/result press or credible news reporting the BIS date.",
        extra_prerequisites=[]
    )

    details_ref_node = evaluator.add_custom_node(
        result=bool(data.details_urls),
        id="show_2_details_reference",
        desc="Provide valid reference URL for dog details",
        parent=details_node,
        critical=True
    )

    # 3) Handler Info (non-critical group)
    handler_node = evaluator.add_parallel(
        id="show_2_handler_info",
        desc="Provide accurate information about the handler",
        parent=show_node,
        critical=False
    )

    handler_ref_node = evaluator.add_custom_node(
        result=bool(data.handler_urls),
        id="show_2_handler_reference",
        desc="Provide valid reference URL for handler information",
        parent=handler_node,
        critical=True
    )

    handler_name_node = evaluator.add_leaf(
        id="show_2_handler_name",
        desc="Provide the handler's full name",
        parent=handler_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The dog was presented during Best in Show at the {AKC_NATIONALS_NAME} by handler {data.handler_name}.",
        node=handler_name_node,
        sources=combine_urls(data.handler_urls, data.id_urls),
        additional_instruction="Verify that this person is the BIS ring handler for the 2025 AKC National Championship.",
        extra_prerequisites=[handler_ref_node]
    )

    handler_kennel_node = evaluator.add_leaf(
        id="show_2_handler_kennel",
        desc="Identify the handler's kennel name/program if applicable",
        parent=handler_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The handler's kennel/program or location is '{data.handler_kennel}'.",
        node=handler_kennel_node,
        sources=data.handler_urls,
        additional_instruction="If the page states a program/kennel rather than a geographic location, accept that.",
        extra_prerequisites=[handler_ref_node]
    )

    # 4) Achievements (non-critical group)
    achieve_node = evaluator.add_parallel(
        id="show_2_achievements",
        desc="Provide information about the dog's competitive achievements",
        parent=show_node,
        critical=False
    )

    achieve_ref_node = evaluator.add_custom_node(
        result=bool(data.achievements_urls),
        id="show_2_achievements_reference",
        desc="Provide valid reference URL for achievement information",
        parent=achieve_node,
        critical=True
    )

    prev_wins_node = evaluator.add_leaf(
        id="show_2_previous_wins",
        desc="Describe the dog's major wins prior to this Best in Show",
        parent=achieve_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"Before winning the {AKC_NATIONALS_NAME} Best in Show, the dog had major competitive achievements including: {_readable_list(data.previous_wins)}.",
        node=prev_wins_node,
        sources=data.achievements_urls,
        additional_instruction="Confirm that each listed achievement refers to the same dog and predates the 2025 AKC National Championship BIS.",
        extra_prerequisites=[achieve_ref_node]
    )

    specialty_wins_node = evaluator.add_leaf(
        id="show_2_specialty_wins",
        desc="Mention national specialty wins if applicable",
        parent=achieve_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"National specialty wins (if any) include: {_readable_list(data.specialty_wins)}.",
        node=specialty_wins_node,
        sources=data.achievements_urls,
        additional_instruction="If specialty wins are listed, confirm they are accurately attributed to this dog and are recognized events.",
        extra_prerequisites=[achieve_ref_node]
    )

    # 5) Ownership (non-critical group)
    ownership_node = evaluator.add_parallel(
        id="show_2_ownership",
        desc="Provide ownership information",
        parent=show_node,
        critical=False
    )

    ownership_ref_node = evaluator.add_custom_node(
        result=bool(data.ownership_urls),
        id="show_2_ownership_reference",
        desc="Provide valid reference URL for ownership information",
        parent=ownership_node,
        critical=True
    )

    owners_node = evaluator.add_leaf(
        id="show_2_owners",
        desc="List all co-owners of the dog",
        parent=ownership_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The co-owners of the Best in Show winner are: {_readable_list(data.owners)}.",
        node=owners_node,
        sources=data.ownership_urls,
        additional_instruction="Confirm all listed co-owners are present on the cited page(s).",
        extra_prerequisites=[ownership_ref_node]
    )

    breeders_node = evaluator.add_leaf(
        id="show_2_breeders",
        desc="Identify the dog's breeder(s)",
        parent=ownership_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The breeder(s) of the dog are: {_readable_list(data.breeders)}.",
        node=breeders_node,
        sources=data.ownership_urls,
        additional_instruction="Confirm breeder name(s) on credible sources such as official results or AKC pages.",
        extra_prerequisites=[ownership_ref_node]
    )


async def verify_show3(evaluator: Evaluator, parent: VerificationNode, data: Show3Extraction) -> None:
    """
    Westminster Kennel Club Dog Show (February 2025) winner information
    """
    show_node = evaluator.add_sequential(
        id="show_3_westminster",
        desc="Westminster Kennel Club Dog Show (February 2025) winner information",
        parent=parent,
        critical=False
    )

    # 1) Winner Identification (critical group)
    ident_node = evaluator.add_parallel(
        id="show_3_winner_identification",
        desc="Correctly identify the Best in Show winner's name and breed",
        parent=show_node,
        critical=True
    )

    ref_present_node = evaluator.add_custom_node(
        result=bool(data.id_urls),
        id="show_3_reference",
        desc="Provide valid reference URL for the winner identification",
        parent=ident_node,
        critical=True
    )

    name_node = evaluator.add_leaf(
        id="show_3_dog_name",
        desc="Provide the winner's call name",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Best in Show winner at the {WESTMINSTER_NAME} has the call name '{data.call_name}'.",
        node=name_node,
        sources=data.id_urls,
        additional_instruction="Confirm the call name (not registered name) for the 2025 Westminster BIS dog.",
        extra_prerequisites=[ref_present_node]
    )

    breed_node = evaluator.add_leaf(
        id="show_3_breed",
        desc="Provide the winner's breed",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Best in Show winner at the {WESTMINSTER_NAME} is a {data.breed}.",
        node=breed_node,
        sources=data.id_urls,
        additional_instruction="Verify on official Westminster results or credible dog-sport news sources for the 2025 event.",
        extra_prerequisites=[ref_present_node]
    )

    # 2) Dog Details (non-critical group)
    details_node = evaluator.add_parallel(
        id="show_3_dog_details",
        desc="Provide accurate details about the winning dog",
        parent=show_node,
        critical=False
    )

    sex_node = evaluator.add_leaf(
        id="show_3_sex",
        desc="Provide the dog's sex (male/female)",
        parent=details_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The dog's sex is {data.sex}.",
        node=sex_node,
        sources=combine_urls(data.details_urls, data.id_urls),
        additional_instruction="Confirm the dog's sex for the Westminster BIS dog.",
        extra_prerequisites=[]
    )

    birth_node = evaluator.add_leaf(
        id="show_3_birth_date",
        desc="Provide the dog's date of birth",
        parent=details_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The dog's date of birth is {data.birth_date}.",
        node=birth_node,
        sources=combine_urls(data.details_urls, data.id_urls),
        additional_instruction="Confirm DOB from credible pedigree/registration sources or news citing breeder records.",
        extra_prerequisites=[]
    )

    group_node = evaluator.add_leaf(
        id="show_3_group",
        desc="Identify the AKC breed group the winner belongs to",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The {data.breed} belongs to the {data.group} in the American Kennel Club (AKC).",
        node=group_node,
        sources=combine_urls(data.breed_standard_urls, data.details_urls),
        additional_instruction="Prefer AKC breed page or authoritative breed club standard.",
        extra_prerequisites=[]
    )

    details_ref_node = evaluator.add_custom_node(
        result=bool(data.details_urls),
        id="show_3_details_reference",
        desc="Provide valid reference URL for dog details",
        parent=details_node,
        critical=True
    )

    # 3) Handler Info (non-critical group)
    handler_node = evaluator.add_parallel(
        id="show_3_handler_info",
        desc="Provide accurate information about the handler",
        parent=show_node,
        critical=False
    )

    handler_ref_node = evaluator.add_custom_node(
        result=bool(data.handler_urls),
        id="show_3_handler_reference",
        desc="Provide valid reference URL for handler information",
        parent=handler_node,
        critical=True
    )

    handler_name_node = evaluator.add_leaf(
        id="show_3_handler_name",
        desc="Provide the handler's full name",
        parent=handler_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The dog was presented during Best in Show at the {WESTMINSTER_NAME} by handler {data.handler_name}.",
        node=handler_name_node,
        sources=combine_urls(data.handler_urls, data.id_urls),
        additional_instruction="Verify that the named person is the BIS ring handler for the 2025 Westminster show.",
        extra_prerequisites=[handler_ref_node]
    )

    handler_loc_node = evaluator.add_leaf(
        id="show_3_handler_location",
        desc="Provide the handler's location (city and state) or kennel program name if available",
        parent=handler_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The handler's location or kennel/program is '{data.handler_location_or_kennel}'.",
        node=handler_loc_node,
        sources=data.handler_urls,
        additional_instruction="If the page lists kennel/program instead of city/state, accept that.",
        extra_prerequisites=[handler_ref_node]
    )

    # 4) Pedigree (non-critical group)
    pedigree_node = evaluator.add_parallel(
        id="show_3_pedigree",
        desc="Provide pedigree information",
        parent=show_node,
        critical=False
    )

    pedigree_ref_node = evaluator.add_custom_node(
        result=bool(data.pedigree_urls),
        id="show_3_pedigree_reference",
        desc="Provide valid reference URL for pedigree information",
        parent=pedigree_node,
        critical=True
    )

    sire_node = evaluator.add_leaf(
        id="show_3_sire",
        desc="Identify the dog's sire (father)",
        parent=pedigree_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The dog's sire (father) is '{data.sire}'.",
        node=sire_node,
        sources=data.pedigree_urls,
        additional_instruction="Confirm sire from pedigree/registration sources or credible breeder/kennel records.",
        extra_prerequisites=[pedigree_ref_node]
    )

    dam_node = evaluator.add_leaf(
        id="show_3_dam",
        desc="Identify the dog's dam (mother)",
        parent=pedigree_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The dog's dam (mother) is '{data.dam}'.",
        node=dam_node,
        sources=data.pedigree_urls,
        additional_instruction="Confirm dam from pedigree/registration sources or credible breeder/kennel records.",
        extra_prerequisites=[pedigree_ref_node]
    )

    # 5) Ownership (non-critical group)
    ownership_node = evaluator.add_parallel(
        id="show_3_ownership",
        desc="Provide ownership information",
        parent=show_node,
        critical=False
    )

    ownership_ref_node = evaluator.add_custom_node(
        result=bool(data.ownership_urls),
        id="show_3_ownership_reference",
        desc="Provide valid reference URL for ownership information",
        parent=ownership_node,
        critical=True
    )

    owners_node = evaluator.add_leaf(
        id="show_3_owners",
        desc="List all co-owners of the dog",
        parent=ownership_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The co-owners of the Best in Show winner are: {_readable_list(data.owners)}.",
        node=owners_node,
        sources=data.ownership_urls,
        additional_instruction="Confirm all listed co-owners are present on the cited page(s).",
        extra_prerequisites=[ownership_ref_node]
    )

    breeder_node = evaluator.add_leaf(
        id="show_3_breeder",
        desc="Identify the dog's breeder",
        parent=ownership_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The breeder(s) of the dog are: {_readable_list(data.breeders)}.",
        node=breeder_node,
        sources=data.ownership_urls,
        additional_instruction="Confirm breeder name(s) on credible sources such as official results, AKC pages, or breeder/kennel pages.",
        extra_prerequisites=[ownership_ref_node]
    )

    # 6) Breed Standard specifics (non-critical group)
    std_node = evaluator.add_parallel(
        id="show_3_breed_standard",
        desc="Provide breed-specific standard information",
        parent=show_node,
        critical=False
    )

    std_ref_node = evaluator.add_custom_node(
        result=bool(data.breed_standard_urls),
        id="show_3_standard_reference",
        desc="Provide valid reference URL for breed standard information",
        parent=std_node,
        critical=True
    )

    height_node = evaluator.add_leaf(
        id="show_3_height_standard",
        desc="Provide the AKC breed standard height range for the dog's sex",
        parent=std_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"According to the AKC breed standard for the {data.breed} ({data.sex}), the typical height range is: {data.height_standard}.",
        node=height_node,
        sources=data.breed_standard_urls,
        additional_instruction="Verify the height range for the given sex (when sex-specific); if unisex, the general breed range applies.",
        extra_prerequisites=[std_ref_node]
    )

    weight_node = evaluator.add_leaf(
        id="show_3_weight_standard",
        desc="Provide the AKC breed standard weight range for the dog's sex",
        parent=std_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"According to the AKC breed standard for the {data.breed} ({data.sex}), the typical weight range is: {data.weight_standard}.",
        node=weight_node,
        sources=data.breed_standard_urls,
        additional_instruction="Verify the weight range for the given sex (when sex-specific); if unisex, the general breed range applies.",
        extra_prerequisites=[std_ref_node]
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 2025 AKC major show BIS winners task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Three shows evaluated independently
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

    # Run extractions (in parallel)
    show1_task = evaluator.extract(
        prompt=prompt_extract_show1(),
        template_class=Show1Extraction,
        extraction_name="show_1_extraction"
    )
    show2_task = evaluator.extract(
        prompt=prompt_extract_show2(),
        template_class=Show2Extraction,
        extraction_name="show_2_extraction"
    )
    show3_task = evaluator.extract(
        prompt=prompt_extract_show3(),
        template_class=Show3Extraction,
        extraction_name="show_3_extraction"
    )

    show1_data, show2_data, show3_data = await asyncio.gather(show1_task, show2_task, show3_task)

    # Build verification subtrees for each show
    await verify_show1(evaluator, root, show1_data)
    await verify_show2(evaluator, root, show2_data)
    await verify_show3(evaluator, root, show3_data)

    # Summary
    return evaluator.get_summary()