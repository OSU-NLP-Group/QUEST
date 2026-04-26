import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "trump_eo_march_2026_fr_within_7days"
TASK_DESCRIPTION = (
    "Identify four executive orders signed by President Donald J. Trump in March 2026 that were "
    "published in the Federal Register within 7 calendar days of the signing date. For each executive "
    "order, provide the following information: (1) Executive Order Number, (2) Official Title, "
    "(3) Signing Date, (4) Federal Register Publication Date, (5) Federal Register Citation (Volume, "
    "Issue Number, and Page Range), (6) Federal Register Document Number, and (7) URL Link to the "
    "official Federal Register or GovInfo page for the executive order. All information must be "
    "verifiable from official Federal Register or GovInfo sources."
)

EO_MIN_NUMBER = 14372
EO_MAX_NUMBER = 14394
EO_MONTH = 3
EO_YEAR = 2026
WITHIN_DAYS = 7
OFFICIAL_DOMAINS = ("federalregister.gov", "govinfo.gov")


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class FRCitation(BaseModel):
    volume: Optional[str] = None
    issue_number: Optional[str] = None
    page_range: Optional[str] = None


class ExecutiveOrderItem(BaseModel):
    number: Optional[str] = None
    title: Optional[str] = None
    signing_date: Optional[str] = None
    fr_publication_date: Optional[str] = None
    fr_citation: Optional[FRCitation] = None
    fr_doc_number: Optional[str] = None
    official_url: Optional[str] = None
    other_urls: List[str] = Field(default_factory=list)


class ExecutiveOrdersExtraction(BaseModel):
    executive_orders: List[ExecutiveOrderItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_executive_orders() -> str:
    return """
Extract every executive order mentioned in the answer. For each one, return the following fields exactly as they appear in the answer text without inventing anything:
- number: The executive order number as written (e.g., "Executive Order 14375", "EO 14375", or similar). If missing, set to null.
- title: The official title as given in the answer. If missing, set to null.
- signing_date: The signing date string as given (e.g., "March 6, 2026" or "2026-03-06"). If missing, set to null.
- fr_publication_date: The Federal Register publication date string as given. If missing, set to null.
- fr_citation: An object with three fields as given in the answer:
  • volume (e.g., "91")
  • issue_number (e.g., "46")
  • page_range (e.g., "12345-12347" or "12345")
  If any of the three is missing in the answer, set only the missing ones to null.
- fr_doc_number: The Federal Register document number string (e.g., "2026-01234"). If missing, set to null.
- official_url: A single URL that is explicitly provided and that points to the official Federal Register (federalregister.gov) or GovInfo (govinfo.gov) page for the same executive order. If multiple such URLs are provided, pick one and put the rest into other_urls. If none provided, set to null.
- other_urls: All other URLs mentioned in the answer for this executive order (including any extra Federal Register or GovInfo links beyond the single chosen official_url, and also any non-official links). If none, return an empty array.

Important:
- Do not fabricate URLs or any data.
- Preserve the text exactly as provided by the answer for dates, numbers, and titles.
- Return a JSON object with an 'executive_orders' array.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_eo_number(eo_number_text: Optional[str]) -> Optional[int]:
    if not eo_number_text:
        return None
    m = re.search(r"(\d{3,5})", eo_number_text)
    try:
        return int(m.group(1)) if m else None
    except Exception:
        return None


def is_official_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    return any(d in url.lower() for d in OFFICIAL_DOMAINS)


def gather_official_urls(item: ExecutiveOrderItem) -> List[str]:
    urls: List[str] = []
    if is_official_url(item.official_url):
        urls.append(item.official_url.strip())
    for u in item.other_urls:
        if is_official_url(u):
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)
    return unique_urls


def try_parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str or not isinstance(date_str, str):
        return None
    # Common formats
    fmts = [
        "%B %d, %Y",   # March 6, 2026
        "%b %d, %Y",   # Mar 6, 2026
        "%b. %d, %Y",  # Mar. 6, 2026
        "%Y-%m-%d",    # 2026-03-06
        "%m/%d/%Y",    # 03/06/2026
        "%Y/%m/%d",    # 2026/03/06
        "%d %B %Y",    # 6 March 2026
        "%d %b %Y",    # 6 Mar 2026
    ]
    # Normalize some casual punctuation like extra commas or double spaces
    s = " ".join(date_str.replace("  ", " ").replace(" ,", ",").strip().split())
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except Exception:
            continue
    return None


def within_n_days(signing: Optional[str], publication: Optional[str], n: int) -> bool:
    s = try_parse_date(signing)
    p = try_parse_date(publication)
    if not s or not p:
        return False
    delta = (p - s).days
    return 0 <= delta <= n


def in_march_2026(signing: Optional[str]) -> bool:
    d = try_parse_date(signing)
    if not d:
        return False
    return d.year == EO_YEAR and d.month == EO_MONTH


def eo_signature(item: ExecutiveOrderItem) -> str:
    n = parse_eo_number(item.number)
    if n is not None:
        return f"EO-{n}"
    title_key = (item.title or "").strip().lower()
    url_key = (item.official_url or "").strip().lower()
    return f"title:{title_key}|url:{url_key}"


def normalize_pages(page_range: Optional[str]) -> Optional[str]:
    if not page_range:
        return None
    return page_range.strip()


# --------------------------------------------------------------------------- #
# Verification logic for a single executive order                             #
# --------------------------------------------------------------------------- #
async def verify_single_eo(
    evaluator: Evaluator,
    parent_node,
    item: ExecutiveOrderItem,
    idx: int,
) -> None:
    eo_node = evaluator.add_parallel(
        id=f"executive_order_{idx+1}",
        desc=f"Evaluate the {'first' if idx==0 else 'second' if idx==1 else 'third' if idx==2 else 'fourth'} executive order provided",
        parent=parent_node,
        critical=False
    )

    # eo_number: provided and within range
    number_int = parse_eo_number(item.number)
    number_in_range = number_int is not None and EO_MIN_NUMBER <= number_int <= EO_MAX_NUMBER
    evaluator.add_custom_node(
        result=number_in_range,
        id=f"eo{idx+1}_number",
        desc=f"The executive order number is provided and falls within EO {EO_MIN_NUMBER} through EO {EO_MAX_NUMBER}",
        parent=eo_node,
        critical=True
    )

    # official_url: must be working and point to FR or GovInfo for same EO
    official_urls = gather_official_urls(item)

    if not official_urls:
        evaluator.add_custom_node(
            result=False,
            id=f"eo{idx+1}_official_url",
            desc="A working URL is provided that links to the official Federal Register or GovInfo page for this same executive order",
            parent=eo_node,
            critical=True
        )
        official_url_leaf = None
    else:
        official_url_leaf = evaluator.add_leaf(
            id=f"eo{idx+1}_official_url",
            desc="A working URL is provided that links to the official Federal Register or GovInfo page for this same executive order",
            parent=eo_node,
            critical=True
        )

        number_part = item.number or ""
        title_part = item.title or ""
        claim = (
            f"This URL is an official Federal Register or GovInfo page for an executive order, "
            f"and it corresponds to the same executive order referenced in the answer"
            f"{' with number ' + number_part if number_part else ''}"
            f"{' and title ' + repr(title_part) if title_part else ''}."
        )

        await evaluator.verify(
            claim=claim,
            node=official_url_leaf,
            sources=official_urls,
            additional_instruction=(
                "Confirm that the URL is hosted on federalregister.gov or govinfo.gov and that the page is the "
                "official EO page for the same executive order. If an EO number/title is provided in the claim, "
                "confirm they match the page. The URL must be accessible."
            ),
        )

    # signed_by_trump: verify via official source
    if official_urls:
        signed_leaf = evaluator.add_leaf(
            id=f"eo{idx+1}_signed_by_trump",
            desc="The executive order is in fact signed by President Donald J. Trump (verifiable via official Federal Register or GovInfo record)",
            parent=eo_node,
            critical=True
        )
        await evaluator.verify(
            claim="This executive order was signed by President Donald J. Trump.",
            node=signed_leaf,
            sources=official_urls,
            additional_instruction="Locate the signer or presidential signature information on the page; confirm it is President Donald J. Trump.",
            extra_prerequisites=[official_url_leaf] if official_url_leaf else None
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"eo{idx+1}_signed_by_trump",
            desc="The executive order is in fact signed by President Donald J. Trump (verifiable via official Federal Register or GovInfo record)",
            parent=eo_node,
            critical=True
        )

    # title exact match
    if official_urls and (item.title is not None and item.title.strip() != ""):
        title_leaf = evaluator.add_leaf(
            id=f"eo{idx+1}_title",
            desc="The official title is provided and exactly matches the title as published in the Federal Register/GovInfo record",
            parent=eo_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The official title of this executive order exactly matches: {repr(item.title)}.",
            node=title_leaf,
            sources=official_urls,
            additional_instruction=(
                "Compare the provided title with the official title or heading shown on the Federal Register or "
                "GovInfo page. Treat differences in capitalization or minor typographical punctuation as mismatches "
                "unless they are clearly inconsequential formatting differences."
            ),
            extra_prerequisites=[official_url_leaf] if official_url_leaf else None
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"eo{idx+1}_title",
            desc="The official title is provided and exactly matches the title as published in the Federal Register/GovInfo record",
            parent=eo_node,
            critical=True
        )

    # signing_date accurate and in March 2026
    if official_urls and (item.signing_date is not None and item.signing_date.strip() != ""):
        signing_leaf = evaluator.add_leaf(
            id=f"eo{idx+1}_signing_date",
            desc="The signing date is provided, is accurate per the official record, and falls within March 2026",
            parent=eo_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The signing date of this executive order is {repr(item.signing_date)}, and it falls within March 2026.",
            node=signing_leaf,
            sources=official_urls,
            additional_instruction=(
                "Locate the 'Signed' date on the official page and confirm it matches exactly the provided date string "
                "and that the month is March and the year is 2026."
            ),
            extra_prerequisites=[official_url_leaf] if official_url_leaf else None
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"eo{idx+1}_signing_date",
            desc="The signing date is provided, is accurate per the official record, and falls within March 2026",
            parent=eo_node,
            critical=True
        )

    # fr_publication_date accurate
    if official_urls and (item.fr_publication_date is not None and item.fr_publication_date.strip() != ""):
        pub_leaf = evaluator.add_leaf(
            id=f"eo{idx+1}_fr_publication_date",
            desc="The Federal Register publication date is provided and is accurate per the official record",
            parent=eo_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The Federal Register publication date for this executive order is {repr(item.fr_publication_date)}.",
            node=pub_leaf,
            sources=official_urls,
            additional_instruction="Confirm the publication date as shown on the Federal Register or GovInfo page.",
            extra_prerequisites=[official_url_leaf] if official_url_leaf else None
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"eo{idx+1}_fr_publication_date",
            desc="The Federal Register publication date is provided and is accurate per the official record",
            parent=eo_node,
            critical=True
        )

    # within 7 days check (computed)
    evaluator.add_custom_node(
        result=within_n_days(item.signing_date, item.fr_publication_date, WITHIN_DAYS),
        id=f"eo{idx+1}_within_7_days",
        desc="The time gap between signing date and Federal Register publication date is ≤ 7 calendar days",
        parent=eo_node,
        critical=True
    )

    # fr_citation provided and correct
    fr_cit = item.fr_citation or FRCitation()
    if official_urls and (fr_cit.volume and fr_cit.issue_number and fr_cit.page_range):
        cit_leaf = evaluator.add_leaf(
            id=f"eo{idx+1}_fr_citation",
            desc="The Federal Register citation is provided and correct (volume number, issue number, and page range match the official record)",
            parent=eo_node,
            critical=True
        )
        vol = fr_cit.volume
        iss = fr_cit.issue_number
        pages = normalize_pages(fr_cit.page_range) or ""
        claim = (
            f"The Federal Register citation on this page indicates Volume {vol}, No. {iss}, "
            f"and pages {pages} for this executive order."
        )
        await evaluator.verify(
            claim=claim,
            node=cit_leaf,
            sources=official_urls,
            additional_instruction=(
                "On the Federal Register or GovInfo page, find the citation details (e.g., 'Vol. X, No. Y' and "
                "'Pages A-B'). Confirm all three components (volume, issue number, and page range) match the claim."
            ),
            extra_prerequisites=[official_url_leaf] if official_url_leaf else None
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"eo{idx+1}_fr_citation",
            desc="The Federal Register citation is provided and correct (volume number, issue number, and page range match the official record)",
            parent=eo_node,
            critical=True
        )

    # fr_doc_number provided and correct
    if official_urls and (item.fr_doc_number is not None and item.fr_doc_number.strip() != ""):
        doc_leaf = evaluator.add_leaf(
            id=f"eo{idx+1}_fr_doc_number",
            desc="The Federal Register document number (FR Doc. No.) is provided and correct per the official record",
            parent=eo_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The FR Doc. No. for this executive order is {repr(item.fr_doc_number)}.",
            node=doc_leaf,
            sources=official_urls,
            additional_instruction="Locate 'FR Doc. No.' on the page and confirm it matches the provided value exactly.",
            extra_prerequisites=[official_url_leaf] if official_url_leaf else None
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"eo{idx+1}_fr_doc_number",
            desc="The Federal Register document number (FR Doc. No.) is provided and correct per the official record",
            parent=eo_node,
            critical=True
        )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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

    # 1) Extract all executive orders mentioned in the answer
    extracted: ExecutiveOrdersExtraction = await evaluator.extract(
        prompt=prompt_extract_executive_orders(),
        template_class=ExecutiveOrdersExtraction,
        extraction_name="executive_orders_extraction"
    )

    all_items = extracted.executive_orders or []
    top4: List[ExecutiveOrderItem] = list(all_items[:4])
    # pad to 4 for consistent tree shape
    while len(top4) < 4:
        top4.append(ExecutiveOrderItem())

    # 2) Set-level checks (critical)
    set_node = evaluator.add_parallel(
        id="response_set_requirements",
        desc="Check set-level requirements for the executive orders provided",
        parent=root,
        critical=True
    )

    # exactly 4 EOs provided in the answer (not just evaluated)
    provides_four = (len(all_items) == 4)
    evaluator.add_custom_node(
        result=provides_four,
        id="provides_four_executive_orders",
        desc="The response provides exactly 4 executive orders",
        parent=set_node,
        critical=True
    )

    # all 4 distinct (use the 4 provided if exactly four; otherwise fail)
    if provides_four:
        signatures = [eo_signature(it) for it in all_items]
        distinct = len(set(signatures)) == 4
    else:
        distinct = False

    evaluator.add_custom_node(
        result=distinct,
        id="executive_orders_are_distinct",
        desc="All 4 executive orders are distinct (no duplicates)",
        parent=set_node,
        critical=True
    )

    # 3) Per-item verification (non-critical across items)
    for i in range(4):
        await verify_single_eo(evaluator, root, top4[i], i)

    # 4) Record some custom info
    parsed_numbers = [parse_eo_number(it.number) for it in all_items[:4]]
    evaluator.add_custom_info(
        info={
            "total_items_in_answer": len(all_items),
            "first4_parsed_numbers": parsed_numbers,
            "first4_signatures": [eo_signature(it) for it in top4],
            "eo_range_expected": [EO_MIN_NUMBER, EO_MAX_NUMBER],
            "require_month": "March",
            "require_year": EO_YEAR,
            "publish_within_days": WITHIN_DAYS,
        },
        info_type="diagnostics",
        info_name="eo_diagnostics"
    )

    return evaluator.get_summary()