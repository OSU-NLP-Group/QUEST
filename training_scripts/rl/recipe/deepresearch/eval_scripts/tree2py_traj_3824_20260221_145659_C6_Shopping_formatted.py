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
TASK_ID = "pharmacy_holiday_2025"
TASK_DESCRIPTION = (
    "Identify four national retail pharmacy chains operating in the United States that satisfy all of the following criteria for the 2025 holiday season:\n\n"
    "1. The chain must provide pharmacy services in-store\n"
    "2. The chain must operate locations in at least three different U.S. states\n"
    "3. The chain must remain open on Christmas Day 2025 (December 25, 2025)\n"
    "4. The chain must close at or before 6:00 PM local time on Christmas Eve 2025 (December 24, 2025)\n"
    "5. The chain must operate at least one 24-hour location (stores that remain open 24 hours a day, 7 days a week)\n\n"
    "For each chain, provide:\n"
    "- The chain name\n"
    "- Evidence of pharmacy services\n"
    "- Evidence of multi-state operations\n"
    "- Documented Christmas Day 2025 operating hours\n"
    "- Documented Christmas Eve 2025 closing time\n"
    "- Evidence of 24-hour location availability\n\n"
    "All claims must be supported by valid reference URLs from official company sources or reliable news sources."
)
YEAR = 2025
XMAS_DAY = "December 25, 2025"
XMAS_EVE = "December 24, 2025"
EVE_DEADLINE = "6:00 PM"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ChainEvidence(BaseModel):
    name: Optional[str] = None
    identification_urls: List[str] = Field(default_factory=list)
    pharmacy_urls: List[str] = Field(default_factory=list)
    multi_state_urls: List[str] = Field(default_factory=list)
    christmas_day_urls: List[str] = Field(default_factory=list)
    christmas_eve_urls: List[str] = Field(default_factory=list)
    twentyfour_urls: List[str] = Field(default_factory=list)


class ChainsExtraction(BaseModel):
    chains: List[ChainEvidence] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_chains() -> str:
    return (
        "From the provided answer, extract up to FOUR U.S. retail pharmacy chains that the answer claims satisfy ALL of these criteria for the 2025 holiday season:\n"
        "• Provides pharmacy services in-store\n"
        "• Operates locations in at least three different U.S. states\n"
        f"• Open on Christmas Day {YEAR} ({XMAS_DAY})\n"
        f"• Closes at or before {EVE_DEADLINE} local time on Christmas Eve {YEAR} ({XMAS_EVE})\n"
        "• Operates at least one 24-hour location\n\n"
        "For each chain, extract:\n"
        "1) name: The chain name\n"
        "2) identification_urls: URLs that clearly identify the company as a U.S. retail pharmacy chain (prefer official company sites or reliable news)\n"
        "3) pharmacy_urls: URLs evidencing in-store pharmacy services (e.g., official pharmacy service pages, store pages listing 'Pharmacy')\n"
        "4) multi_state_urls: URLs evidencing presence in at least 3 U.S. states (e.g., store locator with states, corporate statement of nationwide operations)\n"
        f"5) christmas_day_urls: URLs that explicitly state the chain is open on {XMAS_DAY} and/or list Christmas Day {YEAR} hours\n"
        f"6) christmas_eve_urls: URLs that explicitly show Christmas Eve {YEAR} ({XMAS_EVE}) closing time at or before {EVE_DEADLINE}\n"
        "7) twentyfour_urls: URLs evidencing at least one 24-hour location (e.g., store locator filters for 24 hours, pages stating 'open 24 hours' or '24-hour pharmacy')\n\n"
        "STRICT SOURCE RULES:\n"
        "• Use only URLs explicitly present in the answer. Do not invent URLs.\n"
        "• Prefer official company domains or major reputable news; avoid personal blogs or unreliable sources.\n"
        "• Deduplicate URLs. Extract full URLs (include protocol), and ignore malformed ones.\n\n"
        "Return a JSON object with a 'chains' array of up to four items following the specified schema. If any field is missing for a chain, set it to null (for 'name') or an empty list for URL arrays."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def pad_or_trim_chains(extraction: ChainsExtraction, k: int = 4) -> List[ChainEvidence]:
    """Ensure exactly k chains: take first k and pad with empty chains if fewer."""
    items = extraction.chains[:k]
    while len(items) < k:
        items.append(ChainEvidence())
    return items


def clean_urls(urls: Optional[List[str]]) -> List[str]:
    """Normalize URL list."""
    if not urls:
        return []
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


def make_additional_instruction_for_sources(sources: List[str], guidance: str) -> str:
    """Build an instruction that enforces source-grounding; fail if no sources."""
    if sources and len(sources) > 0:
        return (
            guidance
            + " Use ONLY the provided URLs to make your judgment. If any URL is irrelevant or does not explicitly support the claim, judge the claim as not supported."
        )
    else:
        return (
            "No sources are provided for this verification—You MUST judge the claim as not supported and mark it incorrect. "
            + guidance
        )


# --------------------------------------------------------------------------- #
# Verification builder for one chain                                          #
# --------------------------------------------------------------------------- #
async def verify_chain(
    evaluator: Evaluator,
    parent_node,
    chain: ChainEvidence,
    chain_index: int,
) -> None:
    """
    Build verification sub-tree for a single chain and dispatch batch verifications.
    """
    idx_str = str(chain_index + 1)
    chain_name = (chain.name or "").strip() or f"Chain #{idx_str}"

    # Top-level node for this chain (non-critical to allow partial credit across chains)
    chain_node = evaluator.add_parallel(
        id=f"chain_{idx_str}",
        desc=[
            "First retail pharmacy chain meeting all criteria",
            "Second retail pharmacy chain meeting all criteria",
            "Third retail pharmacy chain meeting all criteria",
            "Fourth retail pharmacy chain meeting all criteria",
        ][chain_index],
        parent=parent_node,
        critical=False,
    )

    # ---------------- Identification group (Critical) ---------------- #
    ident_node = evaluator.add_parallel(
        id=f"chain_{idx_str}_identification",
        desc="Chain is identified as a retail pharmacy chain operating in the United States",
        parent=chain_node,
        critical=True,
    )

    # Pharmacy services evidence
    pharmacy_leaf = evaluator.add_leaf(
        id=f"chain_{idx_str}_pharmacy_service",
        desc="Chain provides pharmacy services in-store",
        parent=ident_node,
        critical=True,
    )

    # Multi-state operations evidence
    multi_state_leaf = evaluator.add_leaf(
        id=f"chain_{idx_str}_multi_state",
        desc="Chain operates locations in at least three different states",
        parent=ident_node,
        critical=True,
    )

    # Identification reference URL (valid official or reliable news)
    ident_ref_leaf = evaluator.add_leaf(
        id=f"chain_{idx_str}_reference_url",
        desc="Valid reference URL provided for chain identification",
        parent=ident_node,
        critical=True,
    )

    # ---------------- Christmas Day group (Critical) ---------------- #
    xday_node = evaluator.add_parallel(
        id=f"chain_{idx_str}_christmas_day",
        desc=f"Chain operates on Christmas Day {YEAR} (December 25)",
        parent=chain_node,
        critical=True,
    )

    xday_open_leaf = evaluator.add_leaf(
        id=f"chain_{idx_str}_christmas_day_open",
        desc=f"Chain stores are open on Christmas Day {YEAR}",
        parent=xday_node,
        critical=True,
    )

    xday_ref_leaf = evaluator.add_leaf(
        id=f"chain_{idx_str}_christmas_day_reference",
        desc="Valid reference URL provided for Christmas Day hours",
        parent=xday_node,
        critical=True,
    )

    # ---------------- Christmas Eve group (Critical) ---------------- #
    xeve_node = evaluator.add_parallel(
        id=f"chain_{idx_str}_christmas_eve",
        desc=f"Chain closes at or before {EVE_DEADLINE} on Christmas Eve {YEAR} (December 24)",
        parent=chain_node,
        critical=True,
    )

    xeve_time_leaf = evaluator.add_leaf(
        id=f"chain_{idx_str}_christmas_eve_time",
        desc=f"Chain closing time on Christmas Eve is {EVE_DEADLINE} or earlier",
        parent=xeve_node,
        critical=True,
    )

    xeve_ref_leaf = evaluator.add_leaf(
        id=f"chain_{idx_str}_christmas_eve_reference",
        desc="Valid reference URL provided for Christmas Eve hours",
        parent=xeve_node,
        critical=True,
    )

    # ---------------- 24-hour locations group (Critical) ------------- #
    twentyfour_node = evaluator.add_parallel(
        id=f"chain_{idx_str}_24hour",
        desc="Chain operates at least one 24-hour location",
        parent=chain_node,
        critical=True,
    )

    twentyfour_avail_leaf = evaluator.add_leaf(
        id=f"chain_{idx_str}_24hour_availability",
        desc="Chain has 24-hour locations in operation",
        parent=twentyfour_node,
        critical=True,
    )

    twentyfour_ref_leaf = evaluator.add_leaf(
        id=f"chain_{idx_str}_24hour_reference",
        desc="Valid reference URL provided for 24-hour location information",
        parent=twentyfour_node,
        critical=True,
    )

    # ---------------- Prepare claims and sources --------------------- #
    pharmacy_sources = clean_urls(chain.pharmacy_urls)
    multi_state_sources = clean_urls(chain.multi_state_urls)
    ident_sources = clean_urls(chain.identification_urls)
    xday_sources = clean_urls(chain.christmas_day_urls)
    xeve_sources = clean_urls(chain.christmas_eve_urls)
    tf_sources = clean_urls(chain.twentyfour_urls)

    claims_and_sources: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # Identification: pharmacy services
    pharmacy_claim = (
        f"{chain_name} provides in-store pharmacy services at its retail locations."
    )
    pharmacy_guidance = (
        "Confirm that the provided page(s) explicitly indicate in-store pharmacy services for the chain "
        "(e.g., 'Pharmacy', 'Pharmacy services', 'In-store Rx'). Official company pages or reliable news sources are acceptable."
    )
    claims_and_sources.append((
        pharmacy_claim,
        pharmacy_sources if pharmacy_sources else None,
        pharmacy_leaf,
        make_additional_instruction_for_sources(pharmacy_sources, pharmacy_guidance),
    ))

    # Identification: multi-state operations (≥3 states)
    multi_state_claim = (
        f"{chain_name} operates locations in at least three different U.S. states."
    )
    multi_state_guidance = (
        "Accept evidence such as a store locator listing multiple distinct states, corporate statements of nationwide operations, "
        "or reliable news indicating presence across multiple states. There must be clear support for at least three states."
    )
    claims_and_sources.append((
        multi_state_claim,
        multi_state_sources if multi_state_sources else None,
        multi_state_leaf,
        make_additional_instruction_for_sources(multi_state_sources, multi_state_guidance),
    ))

    # Identification: reference URL validity
    ident_ref_claim = (
        f"This webpage is an official {chain_name} company page or a reliable news source that clearly identifies the chain as a U.S. retail pharmacy chain."
    )
    ident_ref_guidance = (
        "Evaluate whether the URL is from an official company domain or a reputable news outlet and that the content clearly identifies the chain "
        "as a U.S. retail pharmacy chain. If the URL is irrelevant or low-credibility, mark not supported."
    )
    claims_and_sources.append((
        ident_ref_claim,
        ident_sources if ident_sources else None,
        ident_ref_leaf,
        make_additional_instruction_for_sources(ident_sources, ident_ref_guidance),
    ))

    # Christmas Day: open on Dec 25, 2025
    xday_open_claim = (
        f"{chain_name} stores are open on {XMAS_DAY} (Christmas Day)."
    )
    xday_open_guidance = (
        f"Verify that the page explicitly states being open on {XMAS_DAY} (e.g., holiday hours indicating open status). "
        "Statements like 'select stores open' or 'limited hours' are acceptable as long as the chain is open in some capacity."
    )
    claims_and_sources.append((
        xday_open_claim,
        xday_sources if xday_sources else None,
        xday_open_leaf,
        make_additional_instruction_for_sources(xday_sources, xday_open_guidance),
    ))

    # Christmas Day: reference page explicitly providing hours
    xday_ref_claim = (
        f"This webpage explicitly provides Christmas Day {YEAR} operating hours for {chain_name} stores."
    )
    xday_ref_guidance = (
        f"Confirm that the page explicitly mentions Christmas Day {YEAR} and provides operating hours or open status for that date."
    )
    claims_and_sources.append((
        xday_ref_claim,
        xday_sources if xday_sources else None,
        xday_ref_leaf,
        make_additional_instruction_for_sources(xday_sources, xday_ref_guidance),
    ))

    # Christmas Eve: closes at or before 6:00 PM local time on Dec 24, 2025
    xeve_time_claim = (
        f"{chain_name} stores close at or before {EVE_DEADLINE} local time on {XMAS_EVE}."
    )
    xeve_time_guidance = (
        f"Verify the page explicitly states closing time on {XMAS_EVE} at or before {EVE_DEADLINE} (e.g., 'closes at 4 PM', '5 PM', or '6 PM'). "
        "If the page shows later than 6 PM or does not specify the date/year, mark not supported."
    )
    claims_and_sources.append((
        xeve_time_claim,
        xeve_sources if xeve_sources else None,
        xeve_time_leaf,
        make_additional_instruction_for_sources(xeve_sources, xeve_time_guidance),
    ))

    # Christmas Eve: reference page explicitly providing hours/closing time
    xeve_ref_claim = (
        f"This webpage explicitly provides Christmas Eve {YEAR} closing times for {chain_name} stores."
    )
    xeve_ref_guidance = (
        f"Confirm that the page explicitly mentions Christmas Eve {YEAR} and provides closing times for that date."
    )
    claims_and_sources.append((
        xeve_ref_claim,
        xeve_sources if xeve_sources else None,
        xeve_ref_leaf,
        make_additional_instruction_for_sources(xeve_sources, xeve_ref_guidance),
    ))

    # 24-hour locations: availability
    tf_avail_claim = (
        f"{chain_name} operates at least one 24-hour store or 24-hour pharmacy location."
    )
    tf_avail_guidance = (
        "Verify that the page explicitly indicates 'open 24 hours', '24-hour store', or '24-hour pharmacy' for at least one location."
    )
    claims_and_sources.append((
        tf_avail_claim,
        tf_sources if tf_sources else None,
        twentyfour_avail_leaf,
        make_additional_instruction_for_sources(tf_sources, tf_avail_guidance),
    ))

    # 24-hour locations: reference validity
    tf_ref_claim = (
        f"This webpage explicitly indicates 24-hour location(s) available for {chain_name}."
    )
    tf_ref_guidance = (
        "Confirm the page clearly states 24-hour availability (store or pharmacy)."
    )
    claims_and_sources.append((
        tf_ref_claim,
        tf_sources if tf_sources else None,
        twentyfour_ref_leaf,
        make_additional_instruction_for_sources(tf_sources, tf_ref_guidance),
    ))

    # ---------------- Dispatch verifications in batch ----------------- #
    await evaluator.batch_verify(claims_and_sources)


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
    """
    Evaluate an answer for the 2025 holiday retail pharmacy chains task.
    """
    # 1) Initialize evaluator (root is non-critical by default per framework)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent verification for each chain
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

    # 2) Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_chains(),
        template_class=ChainsExtraction,
        extraction_name="chains_extraction",
    )

    # Ensure exactly 4 chains
    chains = pad_or_trim_chains(extraction, k=4)

    # 3) Build verification tree and verify each chain
    for i in range(4):
        await verify_chain(evaluator, root, chains[i], i)

    # 4) Add custom info for context
    evaluator.add_custom_info(
        info={"target_year": YEAR, "christmas_day": XMAS_DAY, "christmas_eve": XMAS_EVE, "eve_deadline": EVE_DEADLINE},
        info_type="task_context",
        info_name="holiday_parameters",
    )

    # 5) Return structured evaluation summary
    return evaluator.get_summary()