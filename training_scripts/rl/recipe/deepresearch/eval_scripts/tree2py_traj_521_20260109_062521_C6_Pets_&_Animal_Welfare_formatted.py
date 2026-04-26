import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "gfas_farm_sanctuaries_2"
TASK_DESCRIPTION = """
Identify two farm animal sanctuaries in the United States that meet all of the following requirements:

1. The sanctuary must hold current GFAS (Global Federation of Animal Sanctuaries) Accredited status (not just Verified status)
2. The sanctuary must maintain a no-breeding policy (does not intentionally breed animals in its care)
3. The sanctuary must maintain a no commercial trade policy (does not buy, sell, or trade animals or animal parts)
4. The sanctuary must prohibit public contact with animals (no petting, touching, or photo opportunities with the animals)
5. The sanctuary must hold 501(c)(3) tax-exempt status with the IRS
6. The sanctuary must have a board of directors with a minimum of 3 members
7. The sanctuary must be located in the United States
8. The sanctuary must provide care for farm animals (such as cattle, pigs, goats, sheep, chickens, or horses)

For each sanctuary, provide:
- The official name of the sanctuary
- The website URL
- The specific city and state location
- Reference URL(s) confirming each of the requirements listed above
"""


# ----------------------------- Data Models --------------------------------- #
class SanctuaryEntry(BaseModel):
    official_name: Optional[str] = None
    website_url: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

    # Per-requirement cited reference URLs
    req1_gfas_urls: List[str] = Field(default_factory=list)  # Accredited (not Verified)
    req2_no_breeding_urls: List[str] = Field(default_factory=list)
    req3_no_trade_urls: List[str] = Field(default_factory=list)
    req4_no_public_contact_urls: List[str] = Field(default_factory=list)
    req5_501c3_urls: List[str] = Field(default_factory=list)
    req6_board_min3_urls: List[str] = Field(default_factory=list)
    req7_located_us_urls: List[str] = Field(default_factory=list)
    req8_farm_animals_urls: List[str] = Field(default_factory=list)


class SanctuariesExtraction(BaseModel):
    sanctuaries: List[SanctuaryEntry] = Field(default_factory=list)


# --------------------------- Extraction Prompt ----------------------------- #
def prompt_extract_sanctuaries() -> str:
    return """
    Extract up to all sanctuaries mentioned in the answer that the agent proposes for this task.
    For each sanctuary, return a JSON object containing:

    1) official_name: The official name of the sanctuary.
    2) website_url: The sanctuary's website URL.
    3) city: The specific city of the sanctuary's location.
    4) state: The specific state of the sanctuary's location (use the state name or 2-letter abbreviation).
    5) req1_gfas_urls: Array of URL(s) that the answer cites to show GFAS Accredited status (not just Verified).
    6) req2_no_breeding_urls: Array of URL(s) cited to show a no-breeding policy.
    7) req3_no_trade_urls: Array of URL(s) cited to show a no commercial trade policy.
    8) req4_no_public_contact_urls: Array of URL(s) cited to show prohibition of public contact with animals.
    9) req5_501c3_urls: Array of URL(s) cited to confirm 501(c)(3) tax-exempt status.
    10) req6_board_min3_urls: Array of URL(s) cited to confirm a board of directors with at least 3 members.
    11) req7_located_us_urls: Array of URL(s) cited to confirm the sanctuary is located in the United States.
    12) req8_farm_animals_urls: Array of URL(s) cited to confirm care for farm animals (e.g., cattle, pigs, goats, sheep, chickens, horses).

    RULES:
    - Extract only information and URLs explicitly present in the provided answer; do not invent or infer anything.
    - For any field that is missing in the answer, set it to null (for strings) or an empty array (for URL lists).
    - For URLs, accept plain URLs or markdown links; always output the actual URLs. If a URL lacks protocol, prepend http:// as needed.
    - Preserve order of sanctuaries as presented in the answer.

    Return a JSON object with one field:
    sanctuaries: an array of sanctuary objects as described above.
    """


# ------------------------------ Helpers ------------------------------------ #
def _normalize_urls(urls: List[str]) -> List[str]:
    norm: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        v = u.strip()
        if not v:
            continue
        if not (v.startswith("http://") or v.startswith("https://")):
            v = "http://" + v
        norm.append(v)
    return norm


def _safe_name(s: SanctuaryEntry) -> str:
    return s.official_name.strip() if s.official_name else "the sanctuary"


# --------------------------- Verification Logic ---------------------------- #
async def _verify_requirement(
    evaluator: Evaluator,
    parent,
    id_prefix: str,
    desc_prefix: str,
    claim: str,
    sources: List[str],
    additional_instruction: str,
) -> None:
    """
    Build container + two leaves for a single requirement:
      - sources_provided (custom node, critical)
      - supported_by_sources (leaf verification, critical)
    """
    container = evaluator.add_parallel(
        id=id_prefix,
        desc=desc_prefix,
        parent=parent,
        critical=True,
    )

    # Existence of sources
    sources_exist = bool(sources) and len(sources) > 0
    evaluator.add_custom_node(
        result=sources_exist,
        id=f"{id_prefix}_sources_provided",
        desc=f"{desc_prefix} - sources are provided",
        parent=container,
        critical=True,
    )

    # Support check via URLs
    supported_node = evaluator.add_leaf(
        id=f"{id_prefix}_supported",
        desc=f"{desc_prefix} - claim supported by cited sources",
        parent=container,
        critical=True,
    )
    await evaluator.verify(
        claim=claim,
        node=supported_node,
        sources=_normalize_urls(sources),
        additional_instruction=additional_instruction,
    )


async def verify_sanctuary(
    evaluator: Evaluator,
    root_parent,
    sanctuary: SanctuaryEntry,
    index: int,
) -> None:
    """
    Construct verification subtree for one sanctuary.
    """
    s_node = evaluator.add_parallel(
        id=f"sanctuary_{index + 1}",
        desc=f"Sanctuary #{index + 1} (must satisfy all requirements and provide required fields)",
        parent=root_parent,
        critical=False,
    )

    # Identity fields group (critical)
    ident_node = evaluator.add_parallel(
        id=f"sanctuary_{index + 1}_identity_fields",
        desc=f"Provide required identifying information for sanctuary #{index + 1}",
        parent=s_node,
        critical=True,
    )

    # official_name provided
    evaluator.add_custom_node(
        result=bool(sanctuary.official_name and sanctuary.official_name.strip()),
        id=f"sanctuary_{index + 1}_official_name",
        desc="Provides the official name of the sanctuary",
        parent=ident_node,
        critical=True,
    )

    # website_url provided
    evaluator.add_custom_node(
        result=bool(sanctuary.website_url and sanctuary.website_url.strip()),
        id=f"sanctuary_{index + 1}_website_url",
        desc="Provides the sanctuary's website URL",
        parent=ident_node,
        critical=True,
    )

    # city and state provided
    evaluator.add_custom_node(
        result=bool(sanctuary.city and sanctuary.city.strip() and sanctuary.state and sanctuary.state.strip()),
        id=f"sanctuary_{index + 1}_city_and_state",
        desc="Provides the specific city and state location",
        parent=ident_node,
        critical=True,
    )

    # Requirement 1: GFAS Accredited (not Verified)
    await _verify_requirement(
        evaluator=evaluator,
        parent=s_node,
        id_prefix=f"sanctuary_{index + 1}_req_1_gfas_accredited",
        desc_prefix="GFAS Accredited status (not merely Verified)",
        claim=f"{_safe_name(sanctuary)} is GFAS Accredited (not just Verified), and the status appears current.",
        sources=sanctuary.req1_gfas_urls,
        additional_instruction=(
            "Confirm that the cited page(s) explicitly show GFAS Accredited (not Verified). "
            "If the page states only 'GFAS Verified' or lacks an 'Accredited' indicator, this fails. "
            "Prefer official GFAS profile pages or official statements; check that the status is not expired or revoked."
        ),
    )

    # Requirement 2: No-breeding policy
    await _verify_requirement(
        evaluator=evaluator,
        parent=s_node,
        id_prefix=f"sanctuary_{index + 1}_req_2_no_breeding",
        desc_prefix="No-breeding policy",
        claim=f"{_safe_name(sanctuary)} maintains a policy of no intentional breeding of animals in its care.",
        sources=sanctuary.req2_no_breeding_urls,
        additional_instruction=(
            "Look for explicit statements such as 'we do not breed animals', 'no breeding', or equivalent language. "
            "General rescue mission language is insufficient unless it clearly states no intentional breeding."
        ),
    )

    # Requirement 3: No commercial trade policy
    await _verify_requirement(
        evaluator=evaluator,
        parent=s_node,
        id_prefix=f"sanctuary_{index + 1}_req_3_no_commercial_trade",
        desc_prefix="No commercial trade policy",
        claim=f"{_safe_name(sanctuary)} does not buy, sell, or trade animals or animal parts (no commercial trade).",
        sources=sanctuary.req3_no_trade_urls,
        additional_instruction=(
            "Verify explicit policy language prohibiting buying, selling, or trading of animals or animal parts. "
            "Fundraising merchandise is unrelated; focus on animal or animal-part commerce."
        ),
    )

    # Requirement 4: No public contact with animals
    await _verify_requirement(
        evaluator=evaluator,
        parent=s_node,
        id_prefix=f"sanctuary_{index + 1}_req_4_no_public_contact",
        desc_prefix="No public contact with animals",
        claim=f"{_safe_name(sanctuary)} prohibits public contact with animals (no petting, touching, or photo ops with animals).",
        sources=sanctuary.req4_no_public_contact_urls,
        additional_instruction=(
            "Confirm policy barring direct physical contact. Allow for observational tours, but any petting/touching/photo-op with animals "
            "violates this requirement."
        ),
    )

    # Requirement 5: 501(c)(3) tax-exempt status
    await _verify_requirement(
        evaluator=evaluator,
        parent=s_node,
        id_prefix=f"sanctuary_{index + 1}_req_5_501c3",
        desc_prefix="501(c)(3) IRS tax-exempt status",
        claim=f"{_safe_name(sanctuary)} is a registered 501(c)(3) tax-exempt nonprofit organization.",
        sources=sanctuary.req5_501c3_urls,
        additional_instruction=(
            "Look for '501(c)(3)' explicitly or equivalent official tax-exemption statements. "
            "If the page references an EIN and states 501(c)(3) status, that suffices."
        ),
    )

    # Requirement 6: Board of directors with at least 3 members
    await _verify_requirement(
        evaluator=evaluator,
        parent=s_node,
        id_prefix=f"sanctuary_{index + 1}_req_6_board_min_3",
        desc_prefix="Board of directors has at least 3 members",
        claim=f"{_safe_name(sanctuary)} has a board of directors with at least three members.",
        sources=sanctuary.req6_board_min3_urls,
        additional_instruction=(
            "Find a 'Board of Directors' or governance page listing members. Count distinct individuals; "
            "if fewer than three are listed, this fails."
        ),
    )

    # Requirement 7: Located in the United States
    loc_city = sanctuary.city.strip() if sanctuary.city else None
    loc_state = sanctuary.state.strip() if sanctuary.state else None
    if loc_city and loc_state:
        loc_claim = f"{_safe_name(sanctuary)} is located in {loc_city}, {loc_state}, United States."
    else:
        loc_claim = f"{_safe_name(sanctuary)} is located in the United States."
    await _verify_requirement(
        evaluator=evaluator,
        parent=s_node,
        id_prefix=f"sanctuary_{index + 1}_req_7_located_in_us",
        desc_prefix="Located in the United States",
        claim=loc_claim,
        sources=sanctuary.req7_located_us_urls,
        additional_instruction=(
            "Confirm a U.S. location (city/state in the United States). "
            "State abbreviations are acceptable; international locations fail."
        ),
    )

    # Requirement 8: Provides care for farm animals
    await _verify_requirement(
        evaluator=evaluator,
        parent=s_node,
        id_prefix=f"sanctuary_{index + 1}_req_8_farm_animals",
        desc_prefix="Provides care for farm animals",
        claim=f"{_safe_name(sanctuary)} provides care for farm animals such as cattle, pigs, goats, sheep, chickens, or horses.",
        sources=sanctuary.req8_farm_animals_urls,
        additional_instruction=(
            "Verify that the sanctuary cares for farm animals (domestic livestock species). "
            "If only wildlife/exotics are mentioned without farm animals, this fails."
        ),
    )


# ----------------------------- Main Entry ---------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for identifying two U.S.-based GFAS-Accredited farm animal sanctuaries
    with required policies and evidence.
    """
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
        default_model=model,
    )

    # Extract sanctuaries and their referenced URLs per requirement
    extracted = await evaluator.extract(
        prompt=prompt_extract_sanctuaries(),
        template_class=SanctuariesExtraction,
        extraction_name="sanctuaries_extraction",
    )

    # Select exactly two sanctuaries (pad if fewer)
    selected: List[SanctuaryEntry] = list(extracted.sanctuaries[:2])
    while len(selected) < 2:
        selected.append(SanctuaryEntry())

    # Build verification tree for each sanctuary
    for idx, s in enumerate(selected):
        await verify_sanctuary(evaluator, root, s, idx)

    return evaluator.get_summary()