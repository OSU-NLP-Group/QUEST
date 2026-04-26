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
TASK_ID = "orchestra_chicago_constraints"
TASK_DESCRIPTION = """
Identify a professional symphony orchestra that meets all of the following criteria: The orchestra must be based in Chicago, Illinois. The orchestra must have appointed a music director designate who is scheduled to begin their tenure in the 2027/28 season. The music director designate must have been born in 1996 in Helsinki, Finland. The orchestra's primary concert hall must have a seating capacity of exactly 2,522 seats. The orchestra must offer season subscription packages that provide savings of up to 30% off single ticket prices. The orchestra must have a principal oboe player who joined the orchestra in 2021. Provide the name of the orchestra along with reference URLs confirming each of these details.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OrchestraDetailsExtraction(BaseModel):
    """
    Structured extraction of the orchestra identification answer.
    All URLs must be explicitly present in the answer (plain or markdown).
    """
    orchestra_name: Optional[str] = None

    # Based in Chicago, IL
    based_in_urls: List[str] = Field(default_factory=list)

    # Music Director Designate appointment and season
    md_designate_name: Optional[str] = None
    md_designate_season: Optional[str] = None
    md_designate_appointment_urls: List[str] = Field(default_factory=list)

    # Music Director Designate birth details
    md_designate_birth_year: Optional[str] = None
    md_designate_birth_city: Optional[str] = None
    md_designate_birth_country: Optional[str] = None
    md_birth_urls: List[str] = Field(default_factory=list)

    # Primary hall capacity
    primary_hall_name: Optional[str] = None
    primary_hall_capacity: Optional[str] = None
    hall_capacity_urls: List[str] = Field(default_factory=list)

    # Subscription savings
    subscription_savings_percent: Optional[str] = None
    subscription_urls: List[str] = Field(default_factory=list)

    # Principal oboe joined year
    principal_oboe_name: Optional[str] = None
    principal_oboe_join_year: Optional[str] = None
    principal_oboe_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_orchestra_details() -> str:
    return """
    Extract the orchestra identification details exactly as presented in the answer. You must not invent anything.

    Fields to extract:
    1) orchestra_name: The full name of the orchestra being proposed.

    2) based_in_urls: All URLs that explicitly confirm the orchestra is based in Chicago, Illinois.

    3) md_designate_name: The name of the music director designate, if provided.
       md_designate_season: The season text for the start (e.g., "2027/28" or "2027-28") if provided.
       md_designate_appointment_urls: All URLs that explicitly confirm the orchestra appointed a music director designate scheduled to begin in the 2027/28 season.

    4) md_designate_birth_year: The birth year for the music director designate, if provided (e.g., "1996").
       md_designate_birth_city: The birth city, if provided (e.g., "Helsinki").
       md_designate_birth_country: The birth country, if provided (e.g., "Finland").
       md_birth_urls: All URLs that explicitly confirm the music director designate was born in 1996 in Helsinki, Finland.

    5) primary_hall_name: The name of the orchestra's primary concert hall, if provided.
       primary_hall_capacity: The capacity value text if provided (e.g., "2,522").
       hall_capacity_urls: All URLs that explicitly confirm the primary concert hall has exactly 2,522 seats.

    6) subscription_savings_percent: The savings percent text for season subscriptions if provided (e.g., "up to 30%").
       subscription_urls: All URLs that explicitly confirm season subscription packages offer savings of up to 30% off single ticket prices.

    7) principal_oboe_name: The principal oboe player's name, if provided.
       principal_oboe_join_year: The year they joined the orchestra, if provided (e.g., "2021").
       principal_oboe_urls: All URLs that explicitly confirm the orchestra has a principal oboe player who joined in 2021.

    IMPORTANT URL RULES:
    - Extract only URLs explicitly present in the answer (plain URLs or markdown links).
    - Do not infer or invent URLs.
    - Include full URLs with protocol; if missing, prepend "http://".
    - If the answer provides multiple URLs for a criterion, include all of them in the corresponding list.
    - If the answer provides no URLs for a criterion, return an empty list for that criterion.

    If a field is not mentioned, set it to null (for strings) or [] (for URL lists).
    """


# --------------------------------------------------------------------------- #
# Verification helper                                                         #
# --------------------------------------------------------------------------- #
def _require_url_instruction(extra: str = "") -> str:
    base = (
        "You must judge support strictly based on the provided webpage(s). "
        "CRITICAL: If the answer did not provide any URL(s) for this criterion, "
        "you must mark the claim as NOT SUPPORTED (Incorrect). "
        "Allow reasonable wording variants (e.g., '2027-28' vs '2027/28'; 'up to 30%' vs 'save up to 30%'). "
    )
    if extra:
        return base + extra
    return base


# --------------------------------------------------------------------------- #
# Build verification tree and run checks                                      #
# --------------------------------------------------------------------------- #
async def verify_orchestra(
    evaluator: Evaluator,
    root: Any,
    details: OrchestraDetailsExtraction,
) -> None:
    """
    Construct the verification tree and run all checks according to the rubric.
    """

    # Add a critical parallel node for the orchestral identification (matches rubric root)
    main_node = evaluator.add_parallel(
        id="OrchestraIdentification",
        desc="Identify a professional symphony orchestra that satisfies all constraints and provide supporting reference URLs for each required detail",
        parent=root,
        critical=True
    )

    # 1) Orchestra name provided (existence check)
    name_provided = bool(details.orchestra_name and details.orchestra_name.strip())
    evaluator.add_custom_node(
        result=name_provided,
        id="OrchestraNameProvided",
        desc="Answer provides the name of the orchestra being proposed",
        parent=main_node,
        critical=True
    )

    # 2) Based in Chicago, Illinois (with URL)
    based_node = evaluator.add_leaf(
        id="BasedInChicagoIL_WithURL",
        desc="Provide at least one reference URL that explicitly confirms the orchestra is based in Chicago, Illinois",
        parent=main_node,
        critical=True
    )
    based_claim = f"This page explicitly confirms that {details.orchestra_name or 'the orchestra'} is based in Chicago, Illinois."
    await evaluator.verify(
        claim=based_claim,
        node=based_node,
        sources=details.based_in_urls,
        additional_instruction=_require_url_instruction(
            "The page should clearly indicate the orchestra's base/location as Chicago, Illinois."
        ),
    )

    # 3) Music Director Designate scheduled to begin in the 2027/28 season (with URL)
    md_app_node = evaluator.add_leaf(
        id="MusicDirectorDesignate_2027_28_WithURL",
        desc="Provide at least one reference URL that explicitly confirms the orchestra appointed a music director designate scheduled to begin in the 2027/28 season",
        parent=main_node,
        critical=True
    )
    if details.md_designate_name:
        md_app_claim = (
            f"This page explicitly confirms that {details.orchestra_name or 'the orchestra'} appointed "
            f"{details.md_designate_name} as Music Director Designate, scheduled to begin in the 2027/28 season."
        )
    else:
        md_app_claim = (
            f"This page explicitly confirms that {details.orchestra_name or 'the orchestra'} appointed a "
            f"Music Director Designate scheduled to begin in the 2027/28 season."
        )
    await evaluator.verify(
        claim=md_app_claim,
        node=md_app_node,
        sources=details.md_designate_appointment_urls,
        additional_instruction=_require_url_instruction(
            "Accept minor variations such as '2027-28'. The page must clearly indicate the designation and the start season."
        ),
    )

    # 4) Music Director Designate birth: 1996 in Helsinki, Finland (with URL)
    md_birth_node = evaluator.add_leaf(
        id="MusicDirectorDesignate_Born1996Helsinki_WithURL",
        desc="Provide at least one reference URL that explicitly confirms the music director designate was born in 1996 in Helsinki, Finland",
        parent=main_node,
        critical=True
    )
    if details.md_designate_name:
        md_birth_claim = (
            f"This page confirms that {details.md_designate_name} was born in 1996 in Helsinki, Finland."
        )
    else:
        md_birth_claim = "This page confirms that the music director designate was born in 1996 in Helsinki, Finland."
    await evaluator.verify(
        claim=md_birth_claim,
        node=md_birth_node,
        sources=details.md_birth_urls,
        additional_instruction=_require_url_instruction(
            "Allow formatting variants (e.g., 'born 1996', 'Helsinki (Finland)'). Both year 1996 and city/country must be present."
        ),
    )

    # 5) Primary concert hall capacity exactly 2,522 seats (with URL)
    hall_cap_node = evaluator.add_leaf(
        id="PrimaryHallCapacity2522_WithURL",
        desc="Provide at least one reference URL that explicitly confirms the orchestra's primary concert hall has exactly 2,522 seats",
        parent=main_node,
        critical=True
    )
    if details.primary_hall_name:
        hall_cap_claim = (
            f"This page confirms that the primary concert hall of {details.orchestra_name or 'the orchestra'}, "
            f"{details.primary_hall_name}, has exactly 2,522 seats."
        )
    else:
        hall_cap_claim = (
            f"This page confirms that the orchestra's primary concert hall has exactly 2,522 seats."
        )
    await evaluator.verify(
        claim=hall_cap_claim,
        node=hall_cap_node,
        sources=details.hall_capacity_urls,
        additional_instruction=_require_url_instruction(
            "The capacity must be exactly 2,522. Do not accept approximate values or different capacities."
        ),
    )

    # 6) Subscription savings up to 30% off single ticket prices (with URL)
    subs_node = evaluator.add_leaf(
        id="SubscriptionSavingsUpTo30Percent_WithURL",
        desc="Provide at least one reference URL that explicitly confirms season subscription packages offer savings of up to 30% off single ticket prices",
        parent=main_node,
        critical=True
    )
    subs_claim = (
        f"This page confirms that season subscription packages for {details.orchestra_name or 'the orchestra'} "
        f"offer savings of up to 30% off single ticket prices."
    )
    await evaluator.verify(
        claim=subs_claim,
        node=subs_node,
        sources=details.subscription_urls,
        additional_instruction=_require_url_instruction(
            "The page must explicitly mention 'up to 30%' savings or an equivalent phrase."
        ),
    )

    # 7) Principal oboe joined in 2021 (with URL)
    oboe_node = evaluator.add_leaf(
        id="PrincipalOboeJoined2021_WithURL",
        desc="Provide at least one reference URL that explicitly confirms the orchestra has a principal oboe player who joined in 2021",
        parent=main_node,
        critical=True
    )
    if details.principal_oboe_name:
        oboe_claim = (
            f"This page confirms that {details.principal_oboe_name} is the principal oboe of "
            f"{details.orchestra_name or 'the orchestra'} and joined in 2021."
        )
    else:
        oboe_claim = "This page confirms that the orchestra's principal oboe player joined in 2021."
    await evaluator.verify(
        claim=oboe_claim,
        node=oboe_node,
        sources=details.principal_oboe_urls,
        additional_instruction=_require_url_instruction(
            "Accept equivalent wording such as 'appointed in 2021', 'joined the orchestra in 2021', or 'since 2021'. "
            "The role must be principal oboe."
        ),
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
    Evaluate an answer for the Chicago orchestra identification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent checks, overall aggregation at root
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

    # Extract structured details from the answer
    details = await evaluator.extract(
        prompt=prompt_extract_orchestra_details(),
        template_class=OrchestraDetailsExtraction,
        extraction_name="orchestra_details",
    )

    # Build verification tree and run checks
    await verify_orchestra(evaluator, root, details)

    # Return summary
    return evaluator.get_summary()