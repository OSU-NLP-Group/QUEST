import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "thanksgiving_chain_2025_waitlist_policy"
TASK_DESCRIPTION = (
    "For Thanksgiving Day 2025 dining in the United States, identify a national restaurant chain that meets all of the following criteria: "
    "(1) open for dine-in service on Thanksgiving Day, (2) does not accept traditional advance reservations for Thanksgiving, "
    "(3) offers an online waitlist system for Thanksgiving dining, (4) the online waitlist system has a documented maximum group size limit, "
    "and (5) provides a documented alternative accommodation method for groups that exceed the online waitlist maximum. "
    "Based on official sources, what is the maximum group size allowed for this chain's online waitlist on Thanksgiving, and what is the specific documented "
    "alternative method for accommodating groups larger than this maximum?"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ChainCandidate(BaseModel):
    """The primary chain candidate identified in the answer."""
    chain_name: Optional[str] = None

    # Statements explicitly claimed in the answer (verbatim or concise paraphrase from the answer)
    national_us_chain_statement: Optional[str] = None
    open_dinein_thanksgiving_statement: Optional[str] = None
    no_advance_reservations_statement: Optional[str] = None
    online_waitlist_thanksgiving_statement: Optional[str] = None
    alternative_involves_direct_contact_statement: Optional[str] = None

    # Requested explicit outputs (verbatim from the answer)
    waitlist_max_group_size_value: Optional[str] = None
    alternative_method_description: Optional[str] = None


class ChainSources(BaseModel):
    """Categorized official or documented source URLs referenced in the answer."""
    national_chain_sources: List[str] = Field(default_factory=list)
    thanksgiving_open_policy_sources: List[str] = Field(default_factory=list)
    no_reservations_policy_sources: List[str] = Field(default_factory=list)
    online_waitlist_sources: List[str] = Field(default_factory=list)
    max_group_size_limit_sources: List[str] = Field(default_factory=list)
    alternative_accommodation_sources: List[str] = Field(default_factory=list)
    direct_contact_requirement_sources: List[str] = Field(default_factory=list)


class ThanksgivingChainExtraction(BaseModel):
    """Complete extraction result for a single chain candidate."""
    chain: Optional[ChainCandidate] = None
    sources: Optional[ChainSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_thanksgiving_chain() -> str:
    return """
    Extract the single primary restaurant chain candidate (national US chain) that the answer is proposing for Thanksgiving Day 2025 dining. 
    If multiple chains are mentioned, choose the first one that is clearly presented as the main candidate.

    Return a JSON object with two top-level keys: "chain" and "sources".

    For "chain", extract the following fields exactly as presented in the answer (use verbatim phrases where possible, or concise paraphrases if needed):
    - chain_name: The restaurant chain's name.
    - national_us_chain_statement: The answer's statement that it is a national chain operating in the United States (if stated).
    - open_dinein_thanksgiving_statement: The answer's statement about being open for dine-in on Thanksgiving Day 2025.
    - no_advance_reservations_statement: The answer's statement about not accepting traditional advance reservations for Thanksgiving.
    - online_waitlist_thanksgiving_statement: The answer's statement about offering an online waitlist for Thanksgiving dining.
    - waitlist_max_group_size_value: The exact maximum group size value stated for the online waitlist (e.g., "6", "6 guests", "up to 6", etc.). If not provided, set to null.
    - alternative_method_description: The exact documented alternative method for groups exceeding the waitlist maximum (e.g., "call the restaurant", "speak with the manager at the host stand"). If not provided, set to null.
    - alternative_involves_direct_contact_statement: The answer's statement indicating that the over-limit method involves direct contact with the restaurant (if stated), otherwise set to null.

    For "sources", extract the URLs for each policy category (extract only actual URLs mentioned; if none provided for a category, use an empty list):
    - national_chain_sources: URLs supporting that it is a national US chain.
    - thanksgiving_open_policy_sources: URLs supporting Thanksgiving Day 2025 dine-in open policy.
    - no_reservations_policy_sources: URLs supporting the no traditional advance reservations policy for Thanksgiving 2025.
    - online_waitlist_sources: URLs supporting the existence of the online waitlist system for Thanksgiving dining.
    - max_group_size_limit_sources: URLs documenting the online waitlist maximum group size for Thanksgiving.
    - alternative_accommodation_sources: URLs documenting the alternative method for groups exceeding the maximum.
    - direct_contact_requirement_sources: URLs documenting that the alternative method involves direct contact with the restaurant.

    SPECIAL REQUIREMENTS:
    - Only extract URLs explicitly provided in the answer text (including markdown links). Do not invent URLs.
    - If a category is not supported by any URLs in the answer, return an empty list for that category.
    - Do not convert numbers; keep them as strings exactly as presented (e.g., "6", "six", "up to 6").

    If the answer does not clearly identify a single chain, set chain_name to null and still return the structure with nulls/empty lists for other fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if isinstance(u, str):
            s = u.strip()
            if s and (s.lower().startswith("http://") or s.lower().startswith("https://")):
                cleaned.append(s)
    # Deduplicate, preserve order
    seen = set()
    result = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _union_sources(*lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        merged.extend(lst or [])
    return _normalize_urls(merged)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_chain_candidate(
        evaluator: Evaluator,
        parent_complete_task_node,
        extraction: ThanksgivingChainExtraction,
) -> None:
    """
    Build and verify the rubric tree for the Thanksgiving chain candidate.
    """
    chain = extraction.chain or ChainCandidate()
    sources = extraction.sources or ChainSources()

    chain_name = chain.chain_name or ""

    # 1) Chain_Identified (critical leaf under Complete_Task)
    chain_identified_node = evaluator.add_custom_node(
        result=(bool(chain.chain_name) and chain.chain_name.strip() != ""),
        id="Chain_Identified",
        desc="Provide the name of a specific restaurant chain candidate.",
        parent=parent_complete_task_node,
        critical=True
    )

    # 2) Eligibility_Criteria (critical parallel under Complete_Task)
    eligibility_node = evaluator.add_parallel(
        id="Eligibility_Criteria",
        desc="Verify the identified chain satisfies every stated constraint for Thanksgiving Day 2025 dine-in in the United States.",
        parent=parent_complete_task_node,
        critical=True
    )

    # 2.a National_US_Chain
    national_chain_leaf = evaluator.add_leaf(
        id="National_US_Chain",
        desc="The restaurant is a national chain operating in the United States (not a single-location/local restaurant).",
        parent=eligibility_node,
        critical=True
    )
    national_claim = f"{chain_name} is a national restaurant chain operating in the United States."
    await evaluator.verify(
        claim=national_claim,
        node=national_chain_leaf,
        sources=_normalize_urls(sources.national_chain_sources),
        additional_instruction="Verify using official/documented chain sources (e.g., brand site indicating multiple US locations, store locator, 'nationwide' phrasing)."
    )

    # 2.b Open_DineIn_Thanksgiving_2025
    open_dinein_leaf = evaluator.add_leaf(
        id="Open_DineIn_Thanksgiving_2025",
        desc="The chain is open for dine-in service on Thanksgiving Day 2025.",
        parent=eligibility_node,
        critical=True
    )
    open_claim = f"{chain_name} is open for dine-in service on Thanksgiving Day 2025."
    await evaluator.verify(
        claim=open_claim,
        node=open_dinein_leaf,
        sources=_normalize_urls(sources.thanksgiving_open_policy_sources),
        additional_instruction="Confirm the page states Thanksgiving Day 2025 dine-in availability. Treat Nov 27, 2025 as Thanksgiving Day. Prefer official chain communications."
    )

    # 2.c No_Advance_Reservations_Thanksgiving_2025
    no_res_leaf = evaluator.add_leaf(
        id="No_Advance_Reservations_Thanksgiving_2025",
        desc="The chain does not accept traditional advance reservations for Thanksgiving Day 2025 dine-in.",
        parent=eligibility_node,
        critical=True
    )
    no_res_claim = f"{chain_name} does not accept traditional advance reservations for Thanksgiving Day 2025 dine-in."
    await evaluator.verify(
        claim=no_res_claim,
        node=no_res_leaf,
        sources=_normalize_urls(sources.no_reservations_policy_sources),
        additional_instruction="Look for explicit language like 'no reservations', 'walk-in only', or 'use waitlist instead' specific to Thanksgiving 2025."
    )

    # 2.d Online_Waitlist_Thanksgiving_2025
    waitlist_leaf = evaluator.add_leaf(
        id="Online_Waitlist_Thanksgiving_2025",
        desc="The chain offers an online waitlist system for Thanksgiving Day 2025 dining.",
        parent=eligibility_node,
        critical=True
    )
    waitlist_claim = f"{chain_name} offers an online waitlist system for Thanksgiving Day 2025 dining."
    await evaluator.verify(
        claim=waitlist_claim,
        node=waitlist_leaf,
        sources=_normalize_urls(sources.online_waitlist_sources),
        additional_instruction="Confirm the existence of an online waitlist (e.g., chain website, official app page, or waitlist partner page linked from the chain) applicable to Thanksgiving."
    )

    # 2.e Max_Group_Size_Limit_Documented
    max_limit_leaf = evaluator.add_leaf(
        id="Max_Group_Size_Limit_Documented",
        desc="The online waitlist system has a documented maximum group size limit for Thanksgiving dining.",
        parent=eligibility_node,
        critical=True
    )
    max_val = (chain.waitlist_max_group_size_value or "").strip()
    max_limit_claim = f"The online waitlist system for {chain_name} has a documented maximum group size limit of '{max_val}' for Thanksgiving dining."
    await evaluator.verify(
        claim=max_limit_claim,
        node=max_limit_leaf,
        sources=_normalize_urls(sources.max_group_size_limit_sources),
        additional_instruction="Verify the page explicitly states a maximum party/group size cap for the online waitlist (Thanksgiving context)."
    )

    # 2.f Alternative_Accommodation_Documented
    alt_method_leaf = evaluator.add_leaf(
        id="Alternative_Accommodation_Documented",
        desc="There is a documented alternative accommodation method for groups exceeding the online waitlist maximum.",
        parent=eligibility_node,
        critical=True
    )
    alt_desc = (chain.alternative_method_description or "").strip()
    alt_claim = f"For groups exceeding the online waitlist maximum at {chain_name}, the documented alternative accommodation method is: {alt_desc}."
    await evaluator.verify(
        claim=alt_claim,
        node=alt_method_leaf,
        sources=_normalize_urls(sources.alternative_accommodation_sources),
        additional_instruction="Confirm the page documents what guests should do if their group exceeds the waitlist cap (e.g., call the restaurant, coordinate with manager, split parties)."
    )

    # 2.g Alternative_Involves_Direct_Contact
    direct_contact_leaf = evaluator.add_leaf(
        id="Alternative_Involves_Direct_Contact",
        desc="The documented alternative accommodation method involves direct contact with the restaurant.",
        parent=eligibility_node,
        critical=True
    )
    direct_sources = _union_sources(sources.direct_contact_requirement_sources, sources.alternative_accommodation_sources)
    direct_claim = f"The alternative accommodation method for over-the-limit groups at {chain_name} involves directly contacting the restaurant."
    await evaluator.verify(
        claim=direct_claim,
        node=direct_contact_leaf,
        sources=direct_sources,
        additional_instruction="Look for instructions such as 'call the store', 'speak to the manager', 'visit the host stand', or equivalent direct contact with the restaurant."
    )

    # 3) Requested_Outputs (critical parallel under Complete_Task)
    outputs_node = evaluator.add_parallel(
        id="Requested_Outputs",
        desc="Provide the specific requested outputs (not just existence claims).",
        parent=parent_complete_task_node,
        critical=True
    )

    # 3.a Max_Waitlist_Group_Size_Value (check the answer explicitly states it)
    max_output_leaf = evaluator.add_leaf(
        id="Max_Waitlist_Group_Size_Value",
        desc="State the maximum group size allowed by the chain’s online waitlist for Thanksgiving dining.",
        parent=outputs_node,
        critical=True
    )
    max_output_claim = f"The answer explicitly states the maximum group size allowed for the online waitlist is '{max_val}'."
    await evaluator.verify(
        claim=max_output_claim,
        node=max_output_leaf,
        additional_instruction="Verify within the answer itself (not the sources) that the maximum group size value is explicitly provided."
    )

    # 3.b Over_Limit_Alternative_Method_Specifics (check the answer explicitly states it)
    alt_output_leaf = evaluator.add_leaf(
        id="Over_Limit_Alternative_Method_Specifics",
        desc="Describe the specific documented alternative method for accommodating groups larger than the waitlist maximum (what the customer should do).",
        parent=outputs_node,
        critical=True
    )
    alt_output_claim = f"The answer explicitly describes the alternative method for over-limit groups as: {alt_desc}."
    await evaluator.verify(
        claim=alt_output_claim,
        node=alt_output_leaf,
        additional_instruction="Verify within the answer itself that the alternative accommodation method is specifically and explicitly described."
    )

    # 4) Source_Verifiability (critical parallel under Complete_Task)
    src_node = evaluator.add_parallel(
        id="Source_Verifiability",
        desc="All key claims are supported by official restaurant sources or documented communications, as required.",
        parent=parent_complete_task_node,
        critical=True
    )

    # 4.a Source_For_Thanksgiving_Open_Policy
    src_open_leaf = evaluator.add_leaf(
        id="Source_For_Thanksgiving_Open_Policy",
        desc="Provide an official/documented source supporting the Thanksgiving Day 2025 dine-in open policy.",
        parent=src_node,
        critical=True
    )
    src_open_claim = f"There is an official or documented source that supports {chain_name} being open for dine-in on Thanksgiving Day 2025."
    await evaluator.verify(
        claim=src_open_claim,
        node=src_open_leaf,
        sources=_normalize_urls(sources.thanksgiving_open_policy_sources),
        additional_instruction="Only pass if at least one provided URL is an official chain page or documented communication that explicitly states Thanksgiving 2025 dine-in availability."
    )

    # 4.b Source_For_No_Reservations_Policy
    src_nores_leaf = evaluator.add_leaf(
        id="Source_For_No_Reservations_Policy",
        desc="Provide an official/documented source supporting the no traditional advance reservations policy for Thanksgiving Day 2025.",
        parent=src_node,
        critical=True
    )
    src_nores_claim = f"There is an official or documented source that supports {chain_name} not accepting traditional advance reservations for Thanksgiving Day 2025."
    await evaluator.verify(
        claim=src_nores_claim,
        node=src_nores_leaf,
        sources=_normalize_urls(sources.no_reservations_policy_sources),
        additional_instruction="Only pass if at least one provided URL is an official chain page or documented communication that explicitly states the no-reservations policy for Thanksgiving 2025."
    )

    # 4.c Source_For_Online_Waitlist
    src_waitlist_leaf = evaluator.add_leaf(
        id="Source_For_Online_Waitlist",
        desc="Provide an official/documented source supporting the existence of the online waitlist system for Thanksgiving dining.",
        parent=src_node,
        critical=True
    )
    src_waitlist_claim = f"There is an official or documented source that confirms {chain_name} offers an online waitlist system for Thanksgiving dining."
    await evaluator.verify(
        claim=src_waitlist_claim,
        node=src_waitlist_leaf,
        sources=_normalize_urls(sources.online_waitlist_sources),
        additional_instruction="Only pass if at least one provided URL is an official chain page or documented communication confirming the online waitlist for Thanksgiving."
    )

    # 4.d Source_For_Max_Group_Size_Limit
    src_max_leaf = evaluator.add_leaf(
        id="Source_For_Max_Group_Size_Limit",
        desc="Provide an official/documented source supporting the documented maximum group size limit for the online waitlist on Thanksgiving.",
        parent=src_node,
        critical=True
    )
    src_max_claim = f"There is an official or documented source that explicitly states the online waitlist maximum group size for {chain_name} on Thanksgiving is '{max_val}'."
    await evaluator.verify(
        claim=src_max_claim,
        node=src_max_leaf,
        sources=_normalize_urls(sources.max_group_size_limit_sources),
        additional_instruction="Only pass if at least one provided URL is official/documented and clearly states the maximum group size."
    )

    # 4.e Source_For_Alternative_Accommodation_Method
    src_alt_leaf = evaluator.add_leaf(
        id="Source_For_Alternative_Accommodation_Method",
        desc="Provide an official/documented source supporting the alternative accommodation method for groups exceeding the online waitlist maximum.",
        parent=src_node,
        critical=True
    )
    src_alt_claim = f"There is an official or documented source that explicitly describes the alternative method for over-limit groups at {chain_name}: {alt_desc}."
    await evaluator.verify(
        claim=src_alt_claim,
        node=src_alt_leaf,
        sources=_normalize_urls(sources.alternative_accommodation_sources),
        additional_instruction="Only pass if an official or documented chain source describes what guests should do when exceeding the waitlist cap."
    )

    # 4.f Source_For_Direct_Contact_Requirement
    src_direct_leaf = evaluator.add_leaf(
        id="Source_For_Direct_Contact_Requirement",
        desc="Provide an official/documented source supporting that the alternative accommodation method involves direct contact with the restaurant.",
        parent=src_node,
        critical=True
    )
    src_direct_claim = f"There is an official or documented source confirming that the alternative method at {chain_name} involves direct contact with the restaurant."
    await evaluator.verify(
        claim=src_direct_claim,
        node=src_direct_leaf,
        sources=_normalize_urls(direct_sources),
        additional_instruction="Only pass if a provided official/documented source indicates contacting the restaurant directly (call, host stand, manager, etc.)."
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
    Evaluate an answer for the Thanksgiving Day 2025 restaurant chain waitlist policy task.
    """
    # Initialize evaluator with a sequential root to reflect staged gating
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Create the critical Complete_Task sequential node under root
    complete_task_node = evaluator.add_sequential(
        id="Complete_Task",
        desc="Identify a qualifying national US restaurant chain for Thanksgiving Day 2025 and provide the requested waitlist limit and over-limit accommodation method, supported by official/documented sources.",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_thanksgiving_chain(),
        template_class=ThanksgivingChainExtraction,
        extraction_name="thanksgiving_chain_extraction"
    )

    # Build verification tree for the candidate chain
    await verify_chain_candidate(
        evaluator=evaluator,
        parent_complete_task_node=complete_task_node,
        extraction=extraction
    )

    # Return evaluation summary
    return evaluator.get_summary()