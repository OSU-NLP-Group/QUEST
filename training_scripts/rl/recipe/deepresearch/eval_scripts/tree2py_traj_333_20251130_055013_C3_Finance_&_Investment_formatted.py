import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "investor_1790s_boston"
TASK_DESCRIPTION = (
    "Among the top three largest institutional investors in the United States by assets under management, "
    "identify the investment management firm that was founded in the 1790s and is currently headquartered in "
    "Boston, Massachusetts. For this firm, provide: (1) the firm's name and exact founding year, "
    "(2) the complete street address of its current corporate headquarters, "
    "(3) the name of the historical predecessor bank chartered in 1792 from which this firm traces its origins, "
    "and (4) the full name and official title of the government official who granted the charter to that predecessor bank."
)


class FirmExtraction(BaseModel):
    firm_name: Optional[str] = None
    firm_identity_sources: List[str] = Field(default_factory=list)

    founding_year: Optional[str] = None
    founding_year_sources: List[str] = Field(default_factory=list)

    hq_address: Optional[str] = None
    hq_sources: List[str] = Field(default_factory=list)

    predecessor_bank_name: Optional[str] = None
    predecessor_bank_charter_year: Optional[str] = None
    predecessor_bank_sources: List[str] = Field(default_factory=list)

    charter_grantor_name: Optional[str] = None
    charter_grantor_title: Optional[str] = None
    charter_sources: List[str] = Field(default_factory=list)

    top3_by_aum_sources: List[str] = Field(default_factory=list)


def prompt_extract_firm_details() -> str:
    return """
Extract the details about the single investment management firm that the answer identifies as being:
- among the top three largest institutional investors in the United States by assets under management (AUM),
- founded in the 1790s,
- and currently headquartered in Boston, Massachusetts.

Return a single JSON object with the following fields:
- firm_name: the firm’s exact name as given in the answer (e.g., "State Street Corporation", "State Street", or "State Street Global Advisors").
- firm_identity_sources: an array of all URLs cited for the firm identity/name (e.g., official site, Wikipedia, reputable profiles). If none, return [].

- founding_year: the exact founding year (4 digits) mentioned for the firm. If the answer gives a range or multiple dates, select the specific year the answer claims is the founding year. If missing, return null.
- founding_year_sources: an array of all URLs cited to support the founding year. If none, return [].

- hq_address: the complete current corporate headquarters street address in a single line (include street number/name, city, state, and ZIP if available; e.g., "One Lincoln Street, Boston, MA 02111"). If only city/state are provided but not a full street address, still return whatever address line the answer gives. If missing, return null.
- hq_sources: an array of all URLs cited for the headquarters location/address. If none, return [].

- predecessor_bank_name: the name of the historical predecessor bank from which the firm traces its origins. If missing, return null.
- predecessor_bank_charter_year: the year (4 digits) the predecessor bank was chartered, as stated in the answer. If not mentioned, return null.
- predecessor_bank_sources: an array of URLs cited to support the predecessor bank and charter claims. If none, return [].

- charter_grantor_name: the full name of the government official who granted the 1792 charter to the predecessor bank. If missing, return null.
- charter_grantor_title: the official title held by that official at the time of granting the charter (e.g., "Governor of the Commonwealth of Massachusetts"). If missing, return null.
- charter_sources: an array of URLs cited for the charter/grantor details. If none, return [].

- top3_by_aum_sources: an array of URLs cited to demonstrate the firm is among the top three largest institutional investors by AUM in the United States. If none, return [].

Follow these rules strictly:
- Extract only what is explicitly present in the answer text.
- For URL fields, extract only actual URLs present in the answer (including markdown links).
- Do not invent or infer missing information. Use null or [] when appropriate.
"""


def _merge_sources(*lists_of_urls: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists_of_urls:
        if not lst:
            continue
        for url in lst:
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def _parse_first_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


async def build_verification_tree(evaluator: Evaluator, extracted: FirmExtraction) -> None:
    # Create a critical aggregation node under root to emulate a critical root
    main = evaluator.add_parallel(
        id="overall_critical",
        desc="Identify the qualifying US institutional investment management firm and provide all requested historical and headquarters details.",
        parent=evaluator.root,
        critical=True,
    )

    # firm_identity (leaf)
    firm_identity_node = evaluator.add_leaf(
        id="firm_identity",
        desc="Provide the firm’s exact name.",
        parent=main,
        critical=True,
    )
    firm_identity_claim = (
        f"The investment management firm identified is named '{extracted.firm_name}'."
        if extracted.firm_name else
        "The investment management firm identified has a specific, verifiable official name."
    )
    await evaluator.verify(
        claim=firm_identity_claim,
        node=firm_identity_node,
        sources=_merge_sources(extracted.firm_identity_sources),
        additional_instruction="Verify the firm's official or commonly recognized name from the provided sources. Allow standard variants (e.g., 'State Street' vs 'State Street Corporation', 'State Street Global Advisors' vs 'SSGA')."
    )

    # top3_by_aum (leaf)
    top3_node = evaluator.add_leaf(
        id="top3_by_aum",
        desc="Demonstrate that the firm is among the top three largest institutional investors in the United States by assets under management (AUM).",
        parent=main,
        critical=True,
    )
    top3_claim = (
        f"The firm '{extracted.firm_name}' is among the top three largest institutional investors in the United States by assets under management."
        if extracted.firm_name else
        "This firm is among the top three largest institutional investors in the United States by assets under management."
    )
    await evaluator.verify(
        claim=top3_claim,
        node=top3_node,
        sources=_merge_sources(extracted.top3_by_aum_sources, extracted.firm_identity_sources),
        additional_instruction=(
            "Confirm that the named firm ranks within the top three institutional investors by AUM in the United States. "
            "Accept credible rankings/listings showing BlackRock, Vanguard, and State Street (or their investment management arms); "
            "the firm under evaluation should be one of these."
        )
    )

    # founding_year_1790s (non-leaf: two checks)
    fy_parent = evaluator.add_parallel(
        id="founding_year_1790s",
        desc="Provide the firm’s exact founding year, and it must fall within the 1790s.",
        parent=main,
        critical=True,
    )
    # founding year supported by sources
    fy_supported = evaluator.add_leaf(
        id="founding_year_supported",
        desc="The firm's exact founding year is correctly stated and supported by sources.",
        parent=fy_parent,
        critical=True,
    )
    fy_claim = (
        f"The firm '{extracted.firm_name}' was founded in {extracted.founding_year}."
        if extracted.firm_name and extracted.founding_year
        else "The firm has a specific founding year that can be verified from the provided sources."
    )
    await evaluator.verify(
        claim=fy_claim,
        node=fy_supported,
        sources=_merge_sources(extracted.founding_year_sources, extracted.firm_identity_sources, extracted.predecessor_bank_sources),
        additional_instruction="Verify the precise founding year as a four-digit year as claimed in the answer."
    )
    # founding year is within 1790s (custom boolean)
    year_int = _parse_first_year(extracted.founding_year)
    in_1790s = year_int is not None and 1790 <= year_int <= 1799
    evaluator.add_custom_node(
        result=in_1790s,
        id="founding_year_within_1790s",
        desc="The firm’s founding year falls within 1790–1799 inclusive.",
        parent=fy_parent,
        critical=True
    )

    # hq_in_boston_ma (leaf)
    hq_boston_node = evaluator.add_leaf(
        id="hq_in_boston_ma",
        desc="State that the firm’s current corporate headquarters is located in Boston, Massachusetts.",
        parent=main,
        critical=True,
    )
    hq_boston_claim = (
        f"The current corporate headquarters of '{extracted.firm_name}' is located in Boston, Massachusetts."
        if extracted.firm_name else
        "The firm's current corporate headquarters is in Boston, Massachusetts."
    )
    await evaluator.verify(
        claim=hq_boston_claim,
        node=hq_boston_node,
        sources=_merge_sources(extracted.hq_sources, extracted.firm_identity_sources),
        additional_instruction="Look for the firm's official HQ location; confirm it is Boston, Massachusetts (allow 'Boston, MA')."
    )

    # hq_complete_street_address (leaf)
    hq_addr_node = evaluator.add_leaf(
        id="hq_complete_street_address",
        desc="Provide the complete current corporate headquarters street address (street number/name, city, state, ZIP).",
        parent=main,
        critical=True,
    )
    addr_text = extracted.hq_address if extracted.hq_address else ""
    hq_addr_claim = (
        f"The firm's current corporate headquarters street address is: {addr_text}"
        if addr_text else
        "The firm has a complete current corporate headquarters street address that includes street number/name, city, state, and ZIP."
    )
    await evaluator.verify(
        claim=hq_addr_claim,
        node=hq_addr_node,
        sources=_merge_sources(extracted.hq_sources, extracted.firm_identity_sources),
        additional_instruction="Verify the full HQ street address exactly as claimed, including street number/name, city, state (or abbreviation), and ZIP if available."
    )

    # predecessor_bank_1792 (non-leaf: identify bank and verify charter year 1792)
    pred_parent = evaluator.add_parallel(
        id="predecessor_bank_1792",
        desc="Identify the historical predecessor bank from which the firm traces its origins, and specify that this predecessor bank was chartered in 1792.",
        parent=main,
        critical=True,
    )
    pred_bank_leaf = evaluator.add_leaf(
        id="predecessor_bank_identified",
        desc="The firm traces its origins to the specified predecessor bank.",
        parent=pred_parent,
        critical=True,
    )
    pred_bank_claim = (
        f"The firm '{extracted.firm_name}' traces its origins to the predecessor bank named '{extracted.predecessor_bank_name}'."
        if extracted.firm_name and extracted.predecessor_bank_name
        else "The firm traces its origins to a specific predecessor bank, as claimed in the answer."
    )
    await evaluator.verify(
        claim=pred_bank_claim,
        node=pred_bank_leaf,
        sources=_merge_sources(extracted.predecessor_bank_sources, extracted.firm_identity_sources),
        additional_instruction="Verify that the provided predecessor bank is historically cited as the firm's origin."
    )
    pred_charter_leaf = evaluator.add_leaf(
        id="predecessor_chartered_1792",
        desc="The predecessor bank was chartered in 1792.",
        parent=pred_parent,
        critical=True,
    )
    pred_charter_claim = "This predecessor bank was chartered in 1792."
    await evaluator.verify(
        claim=pred_charter_claim,
        node=pred_charter_leaf,
        sources=_merge_sources(extracted.predecessor_bank_sources, extracted.charter_sources),
        additional_instruction="Confirm that the identified predecessor bank received its charter specifically in the year 1792."
    )

    # charter_grantor_name (leaf)
    grantor_name_node = evaluator.add_leaf(
        id="charter_grantor_name",
        desc="Provide the full name of the government official who granted the 1792 charter to the predecessor bank.",
        parent=main,
        critical=True,
    )
    grantor_name_claim = (
        f"The 1792 charter of the predecessor bank was granted by {extracted.charter_grantor_name}."
        if extracted.charter_grantor_name else
        "The 1792 charter of the predecessor bank was granted by a specific government official whose full name can be verified."
    )
    await evaluator.verify(
        claim=grantor_name_claim,
        node=grantor_name_node,
        sources=_merge_sources(extracted.charter_sources, extracted.predecessor_bank_sources),
        additional_instruction="Verify the exact full name of the official who granted or signed the 1792 charter for the predecessor bank."
    )

    # charter_grantor_title (leaf)
    grantor_title_node = evaluator.add_leaf(
        id="charter_grantor_title",
        desc="Provide the official title held by the charter grantor at the time the 1792 charter was granted.",
        parent=main,
        critical=True,
    )
    grantor_title_claim = (
        f"At the time of granting the 1792 charter, {extracted.charter_grantor_name} held the title '{extracted.charter_grantor_title}'."
        if extracted.charter_grantor_name and extracted.charter_grantor_title else
        "At the time of the 1792 charter, the grantor held a specific official title that can be verified."
    )
    await evaluator.verify(
        claim=grantor_title_claim,
        node=grantor_title_node,
        sources=_merge_sources(extracted.charter_sources, extracted.predecessor_bank_sources),
        additional_instruction="Verify the precise official title held by the named grantor at the time of the 1792 charter (e.g., Governor of the Commonwealth of Massachusetts)."
    )


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

    extracted = await evaluator.extract(
        prompt=prompt_extract_firm_details(),
        template_class=FirmExtraction,
        extraction_name="firm_extraction",
    )

    # Optionally log parsed founding year and simple checks
    evaluator.add_custom_info(
        info={
            "parsed_founding_year": _parse_first_year(extracted.founding_year),
            "hq_address_extracted": extracted.hq_address,
            "firm_name_extracted": extracted.firm_name
        },
        info_type="diagnostics",
        info_name="parsed_fields_overview"
    )

    await build_verification_tree(evaluator, extracted)

    return evaluator.get_summary()