import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "alabama_aza_tn_aquarium"
TASK_DESCRIPTION = """
Identify an AZA-accredited zoo located in Alabama that offers a Family membership level priced at $150. This membership must provide at least 50% off reciprocal admission at participating AZA-accredited institutions. Once you have identified this zoo, provide the name and location of the Tennessee Aquarium in Chattanooga, Tennessee, and state the standard adult admission price for the Tennessee Aquarium.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AlabamaZooMembership(BaseModel):
    zoo_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    family_membership_name: Optional[str] = None  # e.g., "Family"
    family_price: Optional[str] = None            # keep string to allow variants like "$150", "150", "$150.00"
    reciprocal_benefit_desc: Optional[str] = None # free text from answer describing reciprocal benefit
    reciprocal_percentage: Optional[str] = None   # e.g., "50%", "at least 50%"
    membership_urls: List[str] = Field(default_factory=list)       # membership page(s)
    accreditation_urls: List[str] = Field(default_factory=list)    # AZA accreditation proof or AZA directory
    location_urls: List[str] = Field(default_factory=list)         # contact/about/location page(s)
    reciprocal_urls: List[str] = Field(default_factory=list)       # any page stating AZA reciprocal discount specifics


class TennesseeAquariumDetails(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    adult_price: Optional[str] = None  # string, e.g., "$39.95"
    urls: List[str] = Field(default_factory=list)          # general site URLs (home, about, contact)
    pricing_urls: List[str] = Field(default_factory=list)  # dedicated ticket/pricing page(s)


class CombinedExtraction(BaseModel):
    alabama_zoo: Optional[AlabamaZooMembership] = None
    tennessee_aquarium: Optional[TennesseeAquariumDetails] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract the key entities and facts the answer provides for two parts:

    Part A: Alabama Zoo and Membership (must be AZA-accredited, located in Alabama, and offer a Family membership priced at $150 with at least 50% off AZA reciprocal admission).
    Return the following fields under "alabama_zoo":
    - zoo_name: the zoo's name identified for Alabama
    - city: the city of the Alabama zoo as provided
    - state: the state (should be "Alabama" or "AL") as provided
    - family_membership_name: the exact membership level name related to family (e.g., "Family", "Family Membership")
    - family_price: the Family membership price as provided (verbatim, e.g., "$150" or "150")
    - reciprocal_benefit_desc: the textual description provided about reciprocal admission benefits (verbatim)
    - reciprocal_percentage: the numeric percent string mentioned for reciprocal discount if present (e.g., "50%")
    - membership_urls: all URLs the answer cites that are relevant to membership levels, pricing, or benefits
    - accreditation_urls: all URLs the answer cites that can support AZA accreditation (e.g., AZA directory page or the zoo's site mentioning AZA)
    - location_urls: all URLs the answer cites that can support the zoo's location in Alabama (e.g., contact/about page)
    - reciprocal_urls: all URLs the answer cites that can support the stated reciprocal discount terms (e.g., AZA reciprocity list, zoo membership FAQ)

    Part B: Tennessee Aquarium details.
    Return the following fields under "tennessee_aquarium":
    - name: the institution name provided for the aquarium
    - city: the city for the Tennessee Aquarium (e.g., "Chattanooga")
    - state: the state for the Tennessee Aquarium (e.g., "Tennessee" or "TN")
    - adult_price: the standard adult admission price value as stated (verbatim string, e.g., "$39.95")
    - urls: any general URLs provided that are about the Tennessee Aquarium (homepage, about, contact)
    - pricing_urls: any URLs that specifically show ticket prices or admissions information for the Tennessee Aquarium

    GENERAL EXTRACTION RULES:
    - Extract only what is explicitly present in the answer text.
    - For any missing item, use null (or empty list for URL arrays).
    - For URL fields, extract actual URLs (including from markdown links).
    - Do not fabricate URLs or facts; keep the exact formatting of prices (including $ if shown).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_alabama_zoo(
    evaluator: Evaluator,
    parent_node,
    data: Optional[AlabamaZooMembership],
) -> None:
    """
    Build and verify the Alabama zoo constraints under a critical parallel node.
    """
    alabama_node = evaluator.add_parallel(
        id="Identify_Qualifying_Alabama_Zoo",
        desc="Provide a zoo that meets all stated constraints for the Alabama zoo and membership.",
        parent=parent_node,
        critical=True
    )

    zoo_name = (data.zoo_name if data and data.zoo_name else "") or ""
    # Collect sources
    membership_sources = data.membership_urls if data else []
    accreditation_sources = data.accreditation_urls if data else []
    location_sources = data.location_urls if data else []
    reciprocal_sources = data.reciprocal_urls if data else []

    # 1) Zoo_Is_Located_In_Alabama (critical leaf)
    loc_leaf = evaluator.add_leaf(
        id="Zoo_Is_Located_In_Alabama",
        desc="The identified zoo is located in Alabama.",
        parent=alabama_node,
        critical=True
    )
    loc_claim = f"The zoo named '{zoo_name}' is located in the U.S. state of Alabama."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=_unique_urls(location_sources, membership_sources, accreditation_sources),
        additional_instruction="Accept 'AL' as Alabama. Evidence can be the zoo's contact/about/location page or similar official page explicitly indicating Alabama."
    )

    # 2) Zoo_Is_AZA_Accredited (critical leaf)
    aza_leaf = evaluator.add_leaf(
        id="Zoo_Is_AZA_Accredited",
        desc="The identified zoo is AZA-accredited.",
        parent=alabama_node,
        critical=True
    )
    aza_claim = f"The zoo named '{zoo_name}' is accredited by the Association of Zoos & Aquariums (AZA)."
    await evaluator.verify(
        claim=aza_claim,
        node=aza_leaf,
        sources=_unique_urls(accreditation_sources, membership_sources),
        additional_instruction="Look for explicit phrasing such as 'AZA-accredited' on the zoo's site or in the AZA member directory. Equivalents like 'accredited by the Association of Zoos & Aquariums' should count."
    )

    # 3) Zoo_Offers_Family_Membership_Level (critical leaf)
    family_leaf = evaluator.add_leaf(
        id="Zoo_Offers_Family_Membership_Level",
        desc="The identified zoo offers a Family membership level.",
        parent=alabama_node,
        critical=True
    )
    family_claim = f"The zoo named '{zoo_name}' offers a membership level named 'Family' (or an exact 'Family' tier)."
    await evaluator.verify(
        claim=family_claim,
        node=family_leaf,
        sources=_unique_urls(membership_sources),
        additional_instruction="The page should list a membership tier explicitly named 'Family' (minor variations like 'Family Membership' acceptable). Tiers like 'Family Plus' are not the same as 'Family' unless the page also shows a tier exactly named 'Family'."
    )

    # 4) Family_Membership_Priced_150 (critical leaf)
    price_leaf = evaluator.add_leaf(
        id="Family_Membership_Priced_150",
        desc="The Family membership level is priced at $150.",
        parent=alabama_node,
        critical=True
    )
    price_claim = "The 'Family' membership price is $150 (USD)."
    await evaluator.verify(
        claim=price_claim,
        node=price_leaf,
        sources=_unique_urls(membership_sources),
        additional_instruction="Accept formatting variants like '$150.00'. Only verify the base price (exclude taxes/fees)."
    )

    # 5) Reciprocal_Benefit_At_Least_50_Percent_Off (critical leaf)
    reciprocal_leaf = evaluator.add_leaf(
        id="Reciprocal_Benefit_At_Least_50_Percent_Off",
        desc="The membership provides at least 50% off reciprocal admission at participating AZA-accredited institutions.",
        parent=alabama_node,
        critical=True
    )
    reciprocal_claim = (
        "The 'Family' membership includes a reciprocal admission benefit of at least 50% off at participating AZA-accredited institutions."
    )
    await evaluator.verify(
        claim=reciprocal_claim,
        node=reciprocal_leaf,
        sources=_unique_urls(reciprocal_sources, membership_sources),
        additional_instruction="Look for phrases like '50% off reciprocal admission' or 'AZA reciprocal benefits (50% discount)' for participating AZA-accredited institutions."
    )


async def verify_tennessee_aquarium(
    evaluator: Evaluator,
    parent_node,
    data: Optional[TennesseeAquariumDetails],
) -> None:
    """
    Build and verify the Tennessee Aquarium part under a critical parallel node.
    """
    tn_node = evaluator.add_parallel(
        id="Provide_Tennessee_Aquarium_Info",
        desc="Provide the requested Tennessee Aquarium identification and standard adult admission price.",
        parent=parent_node,
        critical=True
    )

    name = (data.name if data and data.name else "") or ""
    tn_urls = _unique_urls((data.urls if data else []), (data.pricing_urls if data else []))

    # 1) Aquarium_Name_Is_Tennessee_Aquarium (critical leaf)
    name_leaf = evaluator.add_leaf(
        id="Aquarium_Name_Is_Tennessee_Aquarium",
        desc="The named institution is the Tennessee Aquarium.",
        parent=tn_node,
        critical=True
    )
    name_claim = f"The provided aquarium name '{name}' refers to the 'Tennessee Aquarium'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=None,
        additional_instruction="Allow minor variations such as including 'The' or capitalization differences. This check confirms the answer indeed names the Tennessee Aquarium."
    )

    # 2) Aquarium_Location_Is_Chattanooga_Tennessee (critical leaf)
    loc_leaf = evaluator.add_leaf(
        id="Aquarium_Location_Is_Chattanooga_Tennessee",
        desc="The location given for the Tennessee Aquarium is Chattanooga, Tennessee.",
        parent=tn_node,
        critical=True
    )
    loc_claim = "The Tennessee Aquarium is located in Chattanooga, Tennessee (Chattanooga, TN)."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=tn_urls,
        additional_instruction="Accept 'Chattanooga, TN' as equivalent to 'Chattanooga, Tennessee'. Prefer official site or reliable directory pages."
    )

    # 3) Aquarium_Standard_Adult_Admission_Price_Is_39_95 (critical leaf)
    price_leaf = evaluator.add_leaf(
        id="Aquarium_Standard_Adult_Admission_Price_Is_39_95",
        desc="The standard adult admission price for the Tennessee Aquarium is stated as $39.95.",
        parent=tn_node,
        critical=True
    )
    price_claim = "The standard adult admission price for the Tennessee Aquarium is $39.95."
    await evaluator.verify(
        claim=price_claim,
        node=price_leaf,
        sources=(data.pricing_urls if data else []),
        additional_instruction="Verify the regular adult ticket price (exclude taxes/fees/add-ons). Accept formatting variants like '$39.95'. Prefer current pricing page from the official site."
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
    Evaluate an answer for the Alabama AZA zoo + Tennessee Aquarium task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # top-level flow is sequential per rubric
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

    # Extract structured info from the answer (single combined extraction)
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=CombinedExtraction,
        extraction_name="structured_answer"
    )

    # Build rubric root node (critical, sequential) under the non-critical framework root
    complete_task_node = evaluator.add_sequential(
        id="Complete_Task",
        desc="Identify a qualifying AZA-accredited Alabama zoo with the specified family membership and reciprocal benefit, then provide Tennessee Aquarium identification and its standard adult admission price.",
        parent=root,
        critical=True
    )

    # 1) Alabama zoo verification (parallel, critical)
    await verify_alabama_zoo(
        evaluator=evaluator,
        parent_node=complete_task_node,
        data=extracted.alabama_zoo
    )

    # 2) Tennessee Aquarium information verification (parallel, critical)
    await verify_tennessee_aquarium(
        evaluator=evaluator,
        parent_node=complete_task_node,
        data=extracted.tennessee_aquarium
    )

    # Return structured summary
    return evaluator.get_summary()