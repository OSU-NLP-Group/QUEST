import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "cpb_dissolution_eval"
TASK_DESCRIPTION = "On what date did the Corporation for Public Broadcasting's board of directors vote to dissolve the organization, and how many years had the CPB been in operation before this dissolution vote? Please provide a reference URL from a reputable news source that confirms this information."

EXPECTED_FACTS = {
    "vote_date": "January 5, 2026",
    "years_operation": "58"
}

REPUTABLE_NEWS_DOMAINS = {
    "apnews.com",
    "reuters.com",
    "nytimes.com",
    "washingtonpost.com",
    "wsj.com",
    "bloomberg.com",
    "bbc.com",
    "bbc.co.uk",
    "npr.org",
    "pbs.org",
    "abcnews.go.com",
    "cbsnews.com",
    "nbcnews.com",
    "cnn.com",
    "theguardian.com",
    "politico.com",
    "thehill.com",
    "latimes.com",
    "usatoday.com",
    "axios.com",
    "fortune.com",
    "forbes.com",
    "time.com"
}


class CPBExtraction(BaseModel):
    vote_date: Optional[str] = None
    years_operation: Optional[str] = None
    official_announcement_date: Optional[str] = None
    official_announcement_urls: List[str] = Field(default_factory=list)
    funding_cuts_amount: Optional[str] = None
    funding_cuts_passage_date: Optional[str] = None
    bill_signer: Optional[str] = None
    reputable_news_urls: List[str] = Field(default_factory=list)
    all_urls: List[str] = Field(default_factory=list)


def prompt_extract_cpb() -> str:
    return (
        "Extract the following fields exactly as stated in the answer. If a field is not explicitly present, return null for that field. "
        "Also extract all URLs explicitly present in the answer and categorize reputable news URLs.\n"
        "Required fields:\n"
        "- vote_date: The date the CPB board voted to dissolve the organization (e.g., 'January 5, 2026').\n"
        "- years_operation: The number of years CPB had been in operation before the dissolution vote (e.g., '58').\n"
        "- official_announcement_date: The date of the official CPB news release announcing the dissolution decision, if mentioned.\n"
        "- official_announcement_urls: A list of URLs that are official CPB pages (cpb.org) related to the dissolution announcement.\n"
        "- funding_cuts_amount: The amount of funding cuts described (e.g., 'over $1 billion').\n"
        "- funding_cuts_passage_date: The date or month-year when Congress passed the CPB funding cuts (e.g., 'July 2025').\n"
        "- bill_signer: The person named as having signed the bill containing the CPB funding cuts (e.g., 'President Donald Trump').\n"
        "- reputable_news_urls: A list of reputable mainstream news URLs included in the answer. Only include URLs from well-known outlets such as AP, Reuters, BBC, NYTimes, Washington Post, WSJ, Bloomberg, NPR, PBS, ABC, CBS, NBC News, CNN, The Guardian, Politico, The Hill, USA Today, LA Times, Axios, etc. If unsure, leave this array empty.\n"
        "- all_urls: A list of all URLs explicitly mentioned in the answer (include every valid URL regardless of source).\n"
        "Rules:\n"
        "1) Extract only what is explicitly stated in the answer; do not infer or invent.\n"
        "2) URLs must be actual URLs present in the answer (plain or markdown). If a URL is missing protocol, prepend 'http://'.\n"
        "3) If a field is missing, return null. If a URL category has no entries, return an empty list."
    )


def dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not u or not isinstance(u, str):
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def get_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        return host
    except Exception:
        return ""


def filter_reputable_news_urls(urls: List[str]) -> List[str]:
    filtered = []
    for u in urls:
        d = get_domain(u)
        # Handle subdomains (e.g., www.reuters.com, edition.cnn.com, etc.)
        if d in REPUTABLE_NEWS_DOMAINS:
            filtered.append(u)
        else:
            # Check if any reputable domain is a suffix of d (e.g., subdomain.reuters.com)
            for base in REPUTABLE_NEWS_DOMAINS:
                if d.endswith("." + base):
                    filtered.append(u)
                    break
    return dedup_preserve_order(filtered)


def filter_official_cpb_urls(urls: List[str]) -> List[str]:
    filtered = []
    for u in urls:
        d = get_domain(u)
        if d == "cpb.org" or d.endswith(".cpb.org"):
            filtered.append(u)
    return dedup_preserve_order(filtered)


async def build_and_verify_tree(evaluator: Evaluator, extraction: CPBExtraction) -> None:
    root = evaluator.add_parallel(
        id="CPB_Dissolution_Information",
        desc="Verify the required dissolution-vote facts and required context per the given constraints, and include a reputable-news reference URL per the proposed question.",
        parent=None,
        critical=True
    )

    all_urls = dedup_preserve_order(extraction.all_urls or [])
    news_urls_extracted = dedup_preserve_order(extraction.reputable_news_urls or [])
    news_urls = filter_reputable_news_urls(news_urls_extracted or all_urls)
    cpb_urls = filter_official_cpb_urls(all_urls)

    vote_date = extraction.vote_date or ""
    years_op = extraction.years_operation or ""
    official_date = extraction.official_announcement_date or ""
    funding_amount = extraction.funding_cuts_amount or ""
    funding_passage = extraction.funding_cuts_passage_date or ""
    bill_signer = extraction.bill_signer or ""

    # Dissolution Vote Date
    node_vote_date = evaluator.add_leaf(
        id="Dissolution_Vote_Date",
        desc="States the date the CPB board voted to dissolve the organization as January 5, 2026.",
        parent=root,
        critical=True
    )
    claim_vote_date = f"The CPB board of directors voted to dissolve the organization on {vote_date}."
    add_ins_vote_date = (
        "Judge Correct only if at least one reputable news article among the provided URLs explicitly reports the vote occurred on the stated date. "
        "Accept minor formatting variants (e.g., 'Jan 5, 2026' vs 'January 5, 2026'). "
        "If the provided list of reputable news URLs is empty or the articles report a different date, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_vote_date,
        node=node_vote_date,
        sources=news_urls if news_urls else None,
        additional_instruction=add_ins_vote_date
    )

    # Years of Operation
    node_years = evaluator.add_leaf(
        id="Years_of_Operation",
        desc="States the CPB had been in operation for 58 years before the dissolution vote.",
        parent=root,
        critical=True
    )
    claim_years = f"CPB had been in operation for {years_op} years before the dissolution vote."
    add_ins_years = (
        "Judge Correct only if at least one reputable news article among the provided URLs explicitly states the years-in-operation number at the time of the dissolution vote. "
        "Allow minor phrasing variants like 'for 58 years' or 'a 58-year-old institution'. "
        "If no reputable news URLs are provided or the number differs from the claimed value, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_years,
        node=node_years,
        sources=news_urls if news_urls else None,
        additional_instruction=add_ins_years
    )

    # Official Announcement
    node_official = evaluator.add_leaf(
        id="Official_Announcement",
        desc="States the dissolution decision was announced through an official CPB news release on January 5, 2026.",
        parent=root,
        critical=True
    )
    claim_official = f"The dissolution decision was announced through an official CPB news release on {official_date}."
    add_ins_official = (
        "Judge Correct only if an official CPB webpage (cpb.org) indicates a news release announcing the dissolution decision on the stated date. "
        "If no CPB official URL is provided or the date on the page does not match, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_official,
        node=node_official,
        sources=cpb_urls if cpb_urls else None,
        additional_instruction=add_ins_official
    )

    # Funding Cuts Context
    node_funding = evaluator.add_leaf(
        id="Funding_Cuts_Context",
        desc="States the dissolution followed Congressional passage of funding cuts that stripped CPB of over $1 billion in funding, passed in July 2025.",
        parent=root,
        critical=True
    )
    claim_funding = f"In {funding_passage}, Congress passed funding cuts that stripped CPB of {funding_amount} in funding."
    add_ins_funding = (
        "Judge Correct if the provided sources show that Congress passed CPB funding cuts in July 2025 and the amount is described as over one billion dollars "
        "(accept variants like 'more than $1 billion', '$1+ billion'). "
        "If month/year or amount does not match the claim, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_funding,
        node=node_funding,
        sources=news_urls if news_urls else (all_urls if all_urls else None),
        additional_instruction=add_ins_funding
    )

    # Bill Signing
    node_signing = evaluator.add_leaf(
        id="Bill_Signing",
        desc="States President Donald Trump signed the bill containing the CPB funding cuts.",
        parent=root,
        critical=True
    )
    signer_display = bill_signer if bill_signer else "President Donald Trump"
    claim_signing = f"{signer_display} signed the bill containing the CPB funding cuts into law."
    add_ins_signing = (
        "Judge Correct only if the provided sources explicitly state that President Donald Trump signed the bill with CPB funding cuts into law. "
        "If sources indicate the signer was someone else or do not corroborate this, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_signing,
        node=node_signing,
        sources=news_urls if news_urls else (all_urls if all_urls else None),
        additional_instruction=add_ins_signing
    )

    # Reference URL from Reputable News corroborating both key facts
    node_ref = evaluator.add_leaf(
        id="Reference_URL_Reputable_News",
        desc="Provides at least one reference URL from a reputable news source that corroborates both the dissolution vote date and the years-in-operation claim.",
        parent=root,
        critical=True
    )
    claim_ref = (
        f"At least one reputable news article among the provided URLs confirms both that the CPB board voted to dissolve the organization on {vote_date} "
        f"and that CPB had been in operation for {years_op} years at the time."
    )
    add_ins_ref = (
        "Judge Correct only if at least one reputable mainstream news outlet in the list corroborates both facts together (vote date and years-in-operation). "
        "If the reputable news URL list is empty or articles corroborate only one of the two facts, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=node_ref,
        sources=news_urls if news_urls else None,
        additional_instruction=add_ins_ref
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
    evaluator.initialize(
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

    extraction = await evaluator.extract(
        prompt=prompt_extract_cpb(),
        template_class=CPBExtraction,
        extraction_name="cpb_extraction"
    )

    evaluator.add_ground_truth({
        "expected_vote_date": EXPECTED_FACTS["vote_date"],
        "expected_years_operation": EXPECTED_FACTS["years_operation"],
        "notes": "Expected facts included for reference; verification must be grounded in cited URLs."
    })

    await build_and_verify_tree(evaluator, extraction)

    return evaluator.get_summary()