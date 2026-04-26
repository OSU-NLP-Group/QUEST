import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "national_dog_show_2025_bis"
TASK_DESCRIPTION = (
    "Identify the dog that won Best in Show at the 2025 National Dog Show held on Thanksgiving Day "
    "(November 27, 2025). Provide the following information about this winner: (1) The dog's breed and call name, "
    "(2) The dog's complete AKC registered show name, (3) The handler's full name and geographic location (city and state), "
    "(4) The breeder and co-owner's full name and geographic location (city and state), and (5) Reference URLs from official "
    "or reputable news sources that confirm this information. Please ensure all details are accurate and verifiable through the provided sources."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PersonLocation(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None


class BestInShowExtraction(BaseModel):
    breed: Optional[str] = None
    call_name: Optional[str] = None
    akc_show_name: Optional[str] = None
    handler: Optional[PersonLocation] = None
    breeder_coowner: Optional[PersonLocation] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_bis() -> str:
    return """
    Extract the requested details about the 2025 National Dog Show Best in Show winner as presented in the answer.

    Required fields:
    - breed: The winner dog's breed (e.g., "American Foxhound").
    - call_name: The dog's call name (the everyday name used, e.g., "Biscuit").
    - akc_show_name: The dog's complete AKC registered show name (the formal show name).
    - handler: An object with:
        - name: The handler's full name
        - city: The handler's city
        - state: The handler's state (use the standard state name or USPS abbreviation if given)
    - breeder_coowner: An object with:
        - name: The breeder and co-owner's full name (as provided; if separate breeder and co-owner are given, list both in one string as presented)
        - city: The breeder/co-owner city
        - state: The breeder/co-owner state
    - reference_urls: An array of all reference URLs explicitly cited in the answer that purportedly corroborate these details.
      Extract actual URLs only (including those in markdown links). Deduplicate if necessary. Do not invent URLs.

    Rules:
    - If any field is missing in the answer, set it to null (or empty array for reference_urls).
    - Do not infer missing locations; only extract explicitly provided locations.
    - Preserve names as they appear, allowing for initials or suffixes if present.
    """


# --------------------------------------------------------------------------- #
# Helper for safe string composition                                          #
# --------------------------------------------------------------------------- #
def _safe(val: Optional[str]) -> str:
    return val if val is not None else ""


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the 2025 National Dog Show Best in Show task according to the rubric.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as parallel; critical gating is handled by child
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

    # Extract structured information from the answer
    extracted: BestInShowExtraction = await evaluator.extract(
        prompt=prompt_extract_bis(),
        template_class=BestInShowExtraction,
        extraction_name="bis_extraction"
    )

    # Create the main critical node (must be critical, and all of its children must be critical)
    main_node = evaluator.add_parallel(
        id="2025_National_Dog_Show_Best_in_Show_Winner_Verification",
        desc="Verify all required details for the Best in Show winner of the 2025 National Dog Show specified in the prompt, including winner identity, associated people/locations, and supporting reputable/official references.",
        parent=root,
        critical=True
    )

    # -------------------------- References subtree -------------------------- #
    refs_node = evaluator.add_parallel(
        id="references",
        desc="Provide reference URLs from official or reputable news sources, and the URLs must corroborate the provided winner/handler/breeder details.",
        parent=main_node,
        critical=True
    )

    # 1) Reference URLs provided (existence check)
    refs_provided = evaluator.add_custom_node(
        result=bool(extracted.reference_urls),
        id="reference_urls_provided",
        desc="Includes one or more reference URLs.",
        parent=refs_node,
        critical=True
    )

    # 2) Sources from official or reputable outlets (single-leaf judgment)
    reputable_leaf = evaluator.add_leaf(
        id="sources_official_or_reputable",
        desc="All provided URLs are from official sources or reputable news outlets (not arbitrary/unsupported sources).",
        parent=refs_node,
        critical=True
    )
    reputable_claim = (
        f"Evaluate the list of URLs for reputability. All of these must be from official organizations (e.g., AKC, the event's official site) "
        f"or reputable news outlets (national networks, respected newspapers, local TV news sites, etc.): {extracted.reference_urls}. "
        f"If any URL is not from an official/reputable outlet, the claim should be considered incorrect."
    )
    await evaluator.verify(
        claim=reputable_claim,
        node=reputable_leaf,
        sources=None,
        additional_instruction="Judge reputability by domain recognition and typical news/official sites. Accept well-known sports networks, major newspapers, AKC/official orgs, and reputable local TV news sites. "
                               "If any domain appears untrustworthy or obscure without journalistic/official standing, mark as incorrect."
    )

    # -------------------------- Winner detail leaves ------------------------ #
    # 1) Breed & Call name
    breed_call_leaf = evaluator.add_leaf(
        id="winner_breed_and_call_name",
        desc="Provide the Best in Show winner's breed and call name.",
        parent=main_node,
        critical=True
    )
    breed = _safe(extracted.breed)
    call_name = _safe(extracted.call_name)
    claim_breed_call = (
        f"At the 2025 National Dog Show (Thanksgiving Day, November 27, 2025), the Best in Show winner is a {breed} "
        f"with the call name '{call_name}'."
    )
    await evaluator.verify(
        claim=claim_breed_call,
        node=breed_call_leaf,
        sources=extracted.reference_urls,
        additional_instruction="Verify that at least one of the provided sources explicitly states both the breed and the call name "
                               "for the 2025 National Dog Show Best in Show winner. Minor casing or formatting differences are acceptable. "
                               "If either the breed or call name is missing or unspecified, consider this incorrect."
    )

    # 2) AKC registered show name
    akc_leaf = evaluator.add_leaf(
        id="akc_registered_show_name",
        desc="Provide the winner dog's complete AKC registered show name.",
        parent=main_node,
        critical=True
    )
    akc_name = _safe(extracted.akc_show_name)
    claim_akc = (
        f"The complete AKC registered show name of the 2025 National Dog Show Best in Show winner is '{akc_name}'."
    )
    await evaluator.verify(
        claim=claim_akc,
        node=akc_leaf,
        sources=extracted.reference_urls,
        additional_instruction="Verify that the full AKC registered show name appears in at least one provided source (official AKC pages, "
                               "event pages, or reputable news often include this). Allow minor punctuation or formatting differences."
    )

    # 3) Handler full name and location
    handler_leaf = evaluator.add_leaf(
        id="handler_name_and_location",
        desc="Provide the handler's full name and geographic location (city and state).",
        parent=main_node,
        critical=True
    )
    h_name = _safe(extracted.handler.name if extracted.handler else None)
    h_city = _safe(extracted.handler.city if extracted.handler else None)
    h_state = _safe(extracted.handler.state if extracted.handler else None)
    claim_handler = (
        f"The handler of the 2025 National Dog Show Best in Show winner is {h_name} from {h_city}, {h_state}."
    )
    await evaluator.verify(
        claim=claim_handler,
        node=handler_leaf,
        sources=extracted.reference_urls,
        additional_instruction="Verify that a provided source corroborates the handler's full name and their location (city and state). "
                               "Accept standard state abbreviations (e.g., 'PA' for Pennsylvania). Minor variations in formatting are acceptable."
    )

    # 4) Breeder & co-owner full name and location
    breeder_leaf = evaluator.add_leaf(
        id="breeder_coowner_name_and_location",
        desc="Provide the breeder and co-owner's full name and geographic location (city and state).",
        parent=main_node,
        critical=True
    )
    b_name = _safe(extracted.breeder_coowner.name if extracted.breeder_coowner else None)
    b_city = _safe(extracted.breeder_coowner.city if extracted.breeder_coowner else None)
    b_state = _safe(extracted.breeder_coowner.state if extracted.breeder_coowner else None)
    claim_breeder = (
        f"The breeder and co-owner of the 2025 National Dog Show Best in Show winner is {b_name} from {b_city}, {b_state}."
    )
    await evaluator.verify(
        claim=claim_breeder,
        node=breeder_leaf,
        sources=extracted.reference_urls,
        additional_instruction="Verify that a provided source corroborates the breeder and co-owner's name and their location (city and state). "
                               "If the answer lists distinct breeder and co-owner, the combined provided name/location should be clearly supported by the sources."
    )

    # 3) URLs collectively corroborate all required details (custom pass/fail based on leaves and URLs)
    urls_corroborate_all = (
        bool(extracted.reference_urls)
        and breed_call_leaf.status == "passed"
        and akc_leaf.status == "passed"
        and handler_leaf.status == "passed"
        and breeder_leaf.status == "passed"
    )
    evaluator.add_custom_node(
        result=urls_corroborate_all,
        id="urls_corroborate_all_required_details",
        desc="The provided URLs collectively verify: breed & call name, AKC registered show name, handler name & city/state, and breeder/co-owner name & city/state.",
        parent=refs_node,
        critical=True
    )

    # Return the evaluator summary
    return evaluator.get_summary()