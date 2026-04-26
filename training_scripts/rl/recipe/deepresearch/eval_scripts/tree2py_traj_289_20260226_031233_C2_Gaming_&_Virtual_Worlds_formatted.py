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
TASK_ID = "marvel_rivals_ignite_us_2025"
TASK_DESCRIPTION = """
Identify a Marvel Rivals IGNITE tournament held in the United States in 2025 that meets all of the following requirements:
- The tournament must have a total prize pool of at least $500,000 USD
- The tournament must be an offline event held at a physical venue
- The tournament must feature qualified teams from at least 3 different international regions

For the tournament you identify, provide the following information:
1. The full official tournament name
2. The complete date range of the tournament (start date and end date)
3. The exact total prize pool amount in USD
4. The official name of the venue facility
5. The complete street address of the venue, including street number, street name, city, state, and ZIP code
6. A reference URL to an official or authoritative source documenting the tournament details
7. A reference URL documenting the venue information
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TournamentExtraction(BaseModel):
    """Flat model for tournament and venue details extracted from the answer text."""
    tournament_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    prize_pool_usd: Optional[str] = None
    tournament_source_url: Optional[str] = None

    # Regions/qualification context
    participating_regions: List[str] = Field(default_factory=list)

    # Venue details
    venue_name: Optional[str] = None
    venue_address_full: Optional[str] = None
    venue_city: Optional[str] = None
    venue_state: Optional[str] = None
    venue_zip: Optional[str] = None
    venue_source_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tournament_info() -> str:
    return """
    Extract the details for a single Marvel Rivals IGNITE tournament described in the answer. If multiple tournaments are mentioned, extract the first one that appears to meet the criteria. Return fields exactly as specified.

    Required fields:
    - tournament_name: Full official tournament name.
    - start_date: The start date of the tournament as it appears in the answer (free-form string).
    - end_date: The end date of the tournament as it appears in the answer (free-form string).
    - prize_pool_usd: The total prize pool amount in USD (keep formatting as in the answer, e.g., "$500,000" or "USD 500,000").
    - tournament_source_url: A single URL that the answer cites as an official or authoritative source documenting the tournament details. If the answer lists multiple URLs, choose the most official/authoritative one. If no URL is provided, return null.

    Regions:
    - participating_regions: List the region names mentioned for qualified teams or participants (e.g., "North America", "EMEA", "APAC", "Latin America", etc.). If none are mentioned, return an empty list.

    Venue details:
    - venue_name: Official name of the venue facility (e.g., "Madison Square Garden"). If not provided, return null.
    - venue_address_full: The street number and street name for the venue (e.g., "4 Pennsylvania Plaza"). If not provided, return null.
    - venue_city: City name of the venue (e.g., "New York"). If not provided, return null.
    - venue_state: State/territory abbreviation or full name (e.g., "NY" or "New York"). If not provided, return null.
    - venue_zip: ZIP code (e.g., "10001" or ZIP+4). If not provided, return null.
    - venue_source_url: A single URL that the answer cites as a source for venue information. Prefer the venue's official site or an authoritative listing. If no URL is provided, return null.

    SPECIAL RULES FOR URL EXTRACTION:
    - Only extract URLs explicitly present in the answer text. Do not invent or infer URLs.
    - Accept URLs shown as plain text or markdown links. Normalize obvious malformed URLs when possible; if not possible, return null.
    - If a URL is missing protocol, prepend "http://".

    Return a single JSON object following the TournamentExtraction schema strictly. If any field is missing in the answer, set it to null or an empty list as instructed.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def build_full_address(ex: TournamentExtraction) -> str:
    """Build a printable full address string from extracted fields."""
    parts = []
    if ex.venue_address_full:
        parts.append(ex.venue_address_full.strip())
    city_state_zip = " ".join(
        [p for p in [
            (ex.venue_city or "").strip(),
            (ex.venue_state or "").strip()
        ] if p]
    )
    if city_state_zip:
        parts.append(city_state_zip)
    if ex.venue_zip and ex.venue_zip.strip():
        parts.append(ex.venue_zip.strip())
    return ", ".join(parts).strip()


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_tournament_identification(evaluator: Evaluator, parent_node, info: TournamentExtraction) -> None:
    """
    Build and verify the 'Tournament_Identification' subtree.
    """
    ti_node = evaluator.add_parallel(
        id="Tournament_Identification",
        desc="Correctly identify the specific Marvel Rivals IGNITE tournament meeting all specified criteria.",
        parent=parent_node,
        critical=True
    )

    # Leaf: Tournament_Source_Reference
    tsr_leaf = evaluator.add_leaf(
        id="Tournament_Source_Reference",
        desc="Provide a valid reference URL to an official or authoritative source documenting this tournament.",
        parent=ti_node,
        critical=True
    )
    # Determine verification route based on presence of URL
    if info.tournament_source_url:
        claim_tsr = (
            f"This webpage is an official or authoritative source that documents the Marvel Rivals IGNITE tournament "
            f"'{info.tournament_name or ''}' held in the United States in 2025, including core details (name, dates, prize pool, regions)."
        )
        await evaluator.verify(
            claim=claim_tsr,
            node=tsr_leaf,
            sources=info.tournament_source_url,
            additional_instruction=(
                "Judge whether the page is official (publisher/organizer site) or authoritative (recognized esports outlet, Liquipedia, "
                "major press) and whether it clearly documents the tournament's core details. If irrelevant or inaccessible, mark not supported."
            )
        )
    else:
        # Fall back to simple verification to check answer provided a URL at all
        claim_tsr = "The answer provides a valid reference URL to an official or authoritative source documenting the tournament's details."
        await evaluator.verify(
            claim=claim_tsr,
            node=tsr_leaf,
            sources=None,
            additional_instruction="Check the provided answer text for an actual URL. If no URL is present, mark incorrect."
        )

    # Node: Core_Tournament_Details
    ctd_node = evaluator.add_parallel(
        id="Core_Tournament_Details",
        desc="Provide accurate core details about the identified tournament.",
        parent=ti_node,
        critical=True
    )

    # Leaf: Tournament_Name
    tn_leaf = evaluator.add_leaf(
        id="Tournament_Name",
        desc="The full official name of the tournament.",
        parent=ctd_node,
        critical=True
    )
    claim_tn = f"The official tournament name is '{info.tournament_name or ''}'."
    await evaluator.verify(
        claim=claim_tn,
        node=tn_leaf,
        sources=info.tournament_source_url,
        additional_instruction=(
            "Verify the page explicitly shows the tournament's official name. Allow minor formatting variations or stylization."
        )
    )

    # Leaf: Tournament_Dates
    td_leaf = evaluator.add_leaf(
        id="Tournament_Dates",
        desc="The complete date range of the tournament (start date and end date).",
        parent=ctd_node,
        critical=True
    )
    claim_td = f"The tournament ran from {info.start_date or ''} to {info.end_date or ''} in 2025."
    await evaluator.verify(
        claim=claim_td,
        node=td_leaf,
        sources=info.tournament_source_url,
        additional_instruction=(
            "Confirm both the start and end dates match the page and that the event dates fall within the year 2025. "
            "Accept minor formatting variations (e.g., month name vs. numeric)."
        )
    )

    # Leaf: Prize_Pool_Amount
    pp_leaf = evaluator.add_leaf(
        id="Prize_Pool_Amount",
        desc="The total prize pool amount in USD, which must be at least $500,000.",
        parent=ctd_node,
        critical=True
    )
    claim_pp = f"The tournament's total prize pool was {info.prize_pool_usd or ''} USD, which is at least 500,000 USD."
    await evaluator.verify(
        claim=claim_pp,
        node=pp_leaf,
        sources=info.tournament_source_url,
        additional_instruction=(
            "Verify the page states the prize pool amount and determine whether it is ≥ $500,000 USD. "
            "Accept equivalent formats like '$500,000', 'USD 500,000', or textual statements like 'over $500,000'."
        )
    )


async def verify_venue_details(evaluator: Evaluator, parent_node, info: TournamentExtraction) -> None:
    """
    Build and verify the 'Venue_Details' subtree.
    """
    vd_node = evaluator.add_parallel(
        id="Venue_Details",
        desc="Provide complete information about the physical venue where the tournament is held.",
        parent=parent_node,
        critical=True
    )

    # Leaf: Venue_Source_Reference
    vsr_leaf = evaluator.add_leaf(
        id="Venue_Source_Reference",
        desc="Provide a valid reference URL documenting the venue information.",
        parent=vd_node,
        critical=True
    )
    full_addr = build_full_address(info)
    if info.venue_source_url:
        claim_vsr = (
            f"This webpage documents the official venue details for '{info.venue_name or ''}', including the full street address "
            f"'{full_addr}'."
        )
        await evaluator.verify(
            claim=claim_vsr,
            node=vsr_leaf,
            sources=info.venue_source_url,
            additional_instruction=(
                "Prefer the venue's official website or an authoritative listing. The page should clearly show the venue name and full postal address. "
                "If the page is irrelevant or inaccessible, mark not supported."
            )
        )
    else:
        claim_vsr = "The answer provides a valid reference URL that documents the venue's official name and full street address."
        await evaluator.verify(
            claim=claim_vsr,
            node=vsr_leaf,
            sources=None,
            additional_instruction="Check the answer text for an actual venue URL. If no URL is present, mark incorrect."
        )

    # Node: Physical_Venue_Information
    pvi_node = evaluator.add_parallel(
        id="Physical_Venue_Information",
        desc="Provide accurate venue details.",
        parent=vd_node,
        critical=True
    )

    # Leaf: Venue_Name
    vn_leaf = evaluator.add_leaf(
        id="Venue_Name",
        desc="The official name of the venue facility.",
        parent=pvi_node,
        critical=True
    )
    claim_vn = f"The official venue facility name is '{info.venue_name or ''}'."
    await evaluator.verify(
        claim=claim_vn,
        node=vn_leaf,
        sources=info.venue_source_url,
        additional_instruction="Confirm the page shows the venue's official name. Allow minor stylization differences."
    )

    # Leaf: Complete_Street_Address
    csa_leaf = evaluator.add_leaf(
        id="Complete_Street_Address",
        desc="The full street address of the venue including street number, street name, city, state, and ZIP code.",
        parent=pvi_node,
        critical=True
    )
    claim_csa = (
        f"The venue's complete street address is '{info.venue_address_full or ''}, "
        f"{(info.venue_city or '').strip()}, {(info.venue_state or '').strip()} {(info.venue_zip or '').strip()}' (United States)."
    )
    await evaluator.verify(
        claim=claim_csa,
        node=csa_leaf,
        sources=info.venue_source_url,
        additional_instruction=(
            "Verify the address includes street number and street name, city, state, and ZIP code. "
            "Accept ZIP+4. Allow minor punctuation or ordering variations typical for US addresses."
        )
    )


async def verify_multi_region(evaluator: Evaluator, parent_node, info: TournamentExtraction) -> None:
    """
    Build and verify the 'Multi_Region_Qualification' leaf.
    """
    mrq_leaf = evaluator.add_leaf(
        id="Multi_Region_Qualification",
        desc="Verify that the tournament features qualified teams from at least 3 different international regions.",
        parent=parent_node,
        critical=True
    )

    regions_list = ", ".join(info.participating_regions) if info.participating_regions else "N/A"
    claim_mrq = (
        f"The tournament features qualified teams from at least three distinct international regions "
        f"(reported regions: {regions_list})."
    )
    await evaluator.verify(
        claim=claim_mrq,
        node=mrq_leaf,
        sources=info.tournament_source_url,
        additional_instruction=(
            "Use the tournament source to confirm region representation. Count distinct region categories (e.g., NA, EMEA/Europe, APAC/Asia, "
            "Latin America, Middle East, Africa). Different countries within the same region count as one region. "
            "If the page does not support ≥3 distinct regions, mark not supported."
        )
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
    Evaluate an answer for the Marvel Rivals IGNITE (US, 2025) tournament identification task.
    """
    # Initialize evaluator (root is non-critical by framework design)
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

    # Add a critical top-level node mirroring the rubric root under the framework root
    top_node = evaluator.add_parallel(
        id="Marvel_Rivals_Tournament_US_2025",
        desc="Identify a Marvel Rivals IGNITE tournament held in the United States in 2025 with a prize pool of at least $500,000 USD that features teams from multiple international regions.",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_tournament_info(),
        template_class=TournamentExtraction,
        extraction_name="tournament_extraction"
    )

    # Build and verify subtrees
    await verify_tournament_identification(evaluator, top_node, extracted)
    await verify_venue_details(evaluator, top_node, extracted)
    await verify_multi_region(evaluator, top_node, extracted)

    # Return summary
    return evaluator.get_summary()