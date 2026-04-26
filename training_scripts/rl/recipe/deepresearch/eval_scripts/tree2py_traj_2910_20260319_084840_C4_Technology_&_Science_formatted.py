import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tech_setup_2026_selection"
TASK_DESCRIPTION = (
    "A mobile professional is upgrading their technology setup in 2026 and needs to identify the best options across "
    "five categories based on specific performance criteria. For each category below, identify a product or service "
    "that meets the stated requirements:\n\n"
    "1. Smartphone: Identify a smartphone model that achieved at least 30 hours of battery life in a standardized "
    "practical battery test conducted by a reputable tech review site in 2026.\n\n"
    "2. Laptop Processor: Identify a laptop processor that has a benchmark score of at least 55,000 in recognized CPU "
    "benchmark testing.\n\n"
    "3. Mobile Carrier: Identify a U.S. mobile carrier that provides 5G coverage to at least 95% of the United States "
    "population according to 2026 coverage reports.\n\n"
    "4. Wireless Earbuds: Identify wireless earbuds that are ranked among the top 3 best wireless earbuds by at least "
    "two major tech review publications in 2026.\n\n"
    "5. Cloud Storage Service: Identify a cloud storage service that offers a business plan with at least 1TB of "
    "storage for no more than $6 per month.\n\n"
    "For each product or service, provide the specific model/name and include a reference URL that verifies it meets "
    "the stated criteria."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SmartphoneSelection(BaseModel):
    model: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LaptopProcessorSelection(BaseModel):
    processor: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class MobileCarrierSelection(BaseModel):
    carrier: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EarbudsSelection(BaseModel):
    model: Optional[str] = None
    sources: List[str] = Field(default_factory=list)  # Expect at least 2 distinct major pubs


class CloudStorageSelection(BaseModel):
    service: Optional[str] = None
    plan_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TechSelections(BaseModel):
    smartphone: Optional[SmartphoneSelection] = None
    laptop_processor: Optional[LaptopProcessorSelection] = None
    mobile_carrier: Optional[MobileCarrierSelection] = None
    earbuds: Optional[EarbudsSelection] = None
    cloud_storage: Optional[CloudStorageSelection] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tech_selections() -> str:
    return """
    Extract the selected product or service for each of the five categories below from the provided answer text, along with
    the explicit supporting source URLs that the answer cites for verification. If multiple candidates are mentioned for a
    category, pick the first one that appears.

    For each category, extract:
    - smartphone:
        - model: exact smartphone model name as stated
        - sources: all URLs cited in the answer that support the 2026 standardized practical battery test result
    - laptop_processor:
        - processor: exact laptop processor model name (not desktop)
        - sources: all URLs cited that show a recognized CPU benchmark score for this processor
    - mobile_carrier:
        - carrier: the U.S. mobile carrier name
        - sources: all URLs cited that support the 2026 5G population coverage percentage
    - earbuds:
        - model: exact wireless earbuds model name
        - sources: all URLs cited that show 2026 rankings; include at least two distinct publications if provided
    - cloud_storage:
        - service: the cloud storage service brand
        - plan_name: the specific business plan name if provided
        - sources: all URLs cited that support the business plan's storage capacity and price

    Rules:
    1) Only extract URLs that actually appear in the answer. Do not invent URLs.
    2) Keep URLs exactly as shown (plain or markdown). If missing protocol, prepend http://.
    3) If a field is missing, set it to null. If no URLs are provided for a category, return an empty list.
    4) Do not include unrelated URLs (ensure each extracted URL directly supports that category's requirement).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        if host.startswith("m."):
            host = host[2:]
        return host
    except Exception:
        return url


def _distinct_domains(urls: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for u in urls:
        d = _domain(u)
        if d and d not in seen:
            seen.add(d)
            deduped.append(u)
    return deduped


def _has_min_distinct_domains(urls: List[str], n: int) -> bool:
    return len({ _domain(u) for u in urls if isinstance(u, str) and u.strip() }) >= n


# --------------------------------------------------------------------------- #
# Verification functions per category                                         #
# --------------------------------------------------------------------------- #
async def verify_smartphone_battery(evaluator: Evaluator, parent_node, data: Optional[SmartphoneSelection]) -> None:
    node = evaluator.add_sequential(
        id="Smartphone_Battery_Life",
        desc="The identified smartphone must have achieved ≥30 hours battery life in a standardized practical battery test by a reputable tech review site in 2026",
        parent=parent_node,
        critical=False,
    )

    name = (data.model if data else None) or ""
    urls = (data.sources if data else []) or []

    # Existence and source availability (critical)
    exists = bool(name.strip()) and len(urls) > 0
    evaluator.add_custom_node(
        result=exists,
        id="smartphone_exists",
        desc="Smartphone name provided and at least one supporting source URL is cited",
        parent=node,
        critical=True
    )

    # Constraints (parallel)
    constraints = evaluator.add_parallel(
        id="smartphone_constraints",
        desc="Smartphone battery test constraints",
        parent=node,
        critical=False
    )

    # Battery duration ≥ 30h
    leaf_duration = evaluator.add_leaf(
        id="smartphone_battery_30h",
        desc="Battery life is at least 30 hours per the test result",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cited source(s) show that the smartphone model '{name}' achieved at least 30 hours of battery life in a practical hands-on or standardized battery test.",
        node=leaf_duration,
        sources=urls,
        additional_instruction="Confirm the number is from a test result on the page (not a marketing claim). Allow well-known standardized tests like Tom's Guide battery test, GSMArena endurance rating, DXOMARK Battery, Laptop Mag battery test, PCMark battery life, Rtings, etc."
    )

    # Year is 2026
    leaf_year = evaluator.add_leaf(
        id="smartphone_test_2026",
        desc="The battery test/review is from the year 2026",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim="The cited page shows the battery test/review publication date is in 2026.",
        node=leaf_year,
        sources=urls,
        additional_instruction="Use the article date or explicit 2026 references. If only older years are shown (e.g., 2024/2025), mark as not supported."
    )

    # Reputable tech review site
    leaf_reputable = evaluator.add_leaf(
        id="smartphone_reputable_site",
        desc="The test is from a reputable tech review publication/website",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim="The cited page belongs to a reputable tech review publication (e.g., GSMArena, The Verge, Tom's Guide, Laptop Mag, Rtings, TechRadar, PCMag, Engadget, Android Authority, Consumer Reports, DXOMARK, etc.).",
        node=leaf_reputable,
        sources=urls,
        additional_instruction="Use judgement based on brand recognition and editorial reputation; personal blogs, forums, or unknown affiliate-only pages should not count."
    )

    # Standardized practical test (not a claim/spec)
    leaf_standardized = evaluator.add_leaf(
        id="smartphone_standardized_test",
        desc="Result comes from a standardized practical battery test methodology (not a spec sheet or claim)",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim="The battery life figure for this phone on the cited page comes from a standardized practical battery test methodology applied across devices (not a manufacturer claim or spec sheet).",
        node=leaf_standardized,
        sources=urls,
        additional_instruction="Look for language indicating a consistent test procedure (e.g., specific test name, methodology details, or lab testing) used by the publication."
    )


async def verify_laptop_processor(evaluator: Evaluator, parent_node, data: Optional[LaptopProcessorSelection]) -> None:
    node = evaluator.add_sequential(
        id="Laptop_Processor_Performance",
        desc="The identified laptop processor must have a recognized benchmark score ≥ 55,000",
        parent=parent_node,
        critical=False,
    )

    name = (data.processor if data else None) or ""
    urls = (data.sources if data else []) or []

    exists = bool(name.strip()) and len(urls) > 0
    evaluator.add_custom_node(
        result=exists,
        id="cpu_exists",
        desc="Laptop processor name provided and at least one supporting source URL is cited",
        parent=node,
        critical=True
    )

    constraints = evaluator.add_parallel(
        id="cpu_constraints",
        desc="Laptop processor benchmark constraints",
        parent=node,
        critical=False
    )

    # Benchmark threshold
    leaf_score = evaluator.add_leaf(
        id="cpu_benchmark_55k",
        desc="Processor has a benchmark score of at least 55,000 in a recognized benchmark",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cited source(s) show a recognized CPU benchmark score for the laptop processor '{name}' that is at least 55,000.",
        node=leaf_score,
        sources=urls,
        additional_instruction="Recognized benchmarks include PassMark CPU Mark, Geekbench (Multi-Core), Cinebench, SPEC, etc. The page should explicitly show the benchmark name and a score ≥ 55,000 for this processor."
    )

    # Confirm it's a laptop/mobile processor
    leaf_laptop = evaluator.add_leaf(
        id="cpu_is_laptop_part",
        desc="Processor is a laptop/mobile/notebook part (not desktop)",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"The processor '{name}' is used in laptops/mobile notebooks (not a desktop-only CPU).",
        node=leaf_laptop,
        sources=urls,
        additional_instruction="Look for descriptors like 'mobile', 'laptop', 'notebook', or clear usage in laptops/portable systems."
    )


async def verify_mobile_carrier(evaluator: Evaluator, parent_node, data: Optional[MobileCarrierSelection]) -> None:
    node = evaluator.add_sequential(
        id="Mobile_Carrier_Coverage",
        desc="The identified U.S. mobile carrier must provide 5G coverage to at least 95% of the U.S. population per 2026 coverage reports",
        parent=parent_node,
        critical=False,
    )

    name = (data.carrier if data else None) or ""
    urls = (data.sources if data else []) or []

    exists = bool(name.strip()) and len(urls) > 0
    evaluator.add_custom_node(
        result=exists,
        id="carrier_exists",
        desc="Carrier name provided and at least one supporting source URL is cited",
        parent=node,
        critical=True
    )

    constraints = evaluator.add_parallel(
        id="carrier_constraints",
        desc="Carrier coverage constraints",
        parent=node,
        critical=False
    )

    leaf_coverage = evaluator.add_leaf(
        id="carrier_5g_95pct_population",
        desc="5G coverage reaches at least 95% of the U.S. population",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cited source(s) state that the U.S. carrier '{name}' provides 5G coverage to at least 95% of the U.S. population.",
        node=leaf_coverage,
        sources=urls,
        additional_instruction="Ensure the metric is population coverage (not land area). Accept reputable sources (e.g., carrier official reports, FCC, Ookla, Opensignal, RootMetrics, major press) that clearly state ≥95% population coverage for 5G."
    )

    leaf_year = evaluator.add_leaf(
        id="carrier_coverage_2026",
        desc="Coverage statistic is reported in 2026",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim="The cited page provides or references the 5G population coverage figure specifically for the year 2026.",
        node=leaf_year,
        sources=urls,
        additional_instruction="Look for a 2026 date on the report/page or explicit mention that the coverage figure pertains to 2026."
    )


async def verify_earbuds(evaluator: Evaluator, parent_node, data: Optional[EarbudsSelection]) -> None:
    node = evaluator.add_sequential(
        id="Wireless_Earbuds_Ranking",
        desc="The identified wireless earbuds must be top-3 in 2026 by at least two major tech review publications",
        parent=parent_node,
        critical=False,
    )

    name = (data.model if data else None) or ""
    urls_all = (data.sources if data else []) or []
    urls_distinct = _distinct_domains(urls_all)

    exists = bool(name.strip()) and len(urls_distinct) >= 2
    evaluator.add_custom_node(
        result=exists,
        id="earbuds_exists",
        desc="Earbuds model provided and at least two supporting source URLs from distinct publications are cited",
        parent=node,
        critical=True
    )

    # Additional structural sanity: at least 2 distinct domains
    evaluator.add_custom_node(
        result=_has_min_distinct_domains(urls_distinct, 2),
        id="earbuds_two_distinct_pubs",
        desc="At least two distinct publication domains are provided",
        parent=node,
        critical=True
    )

    constraints = evaluator.add_parallel(
        id="earbuds_constraints",
        desc="Earbuds ranking constraints across multiple publications",
        parent=node,
        critical=False
    )

    # Verify first two distinct sources
    for idx in range(min(2, len(urls_distinct))):
        url = urls_distinct[idx]
        leaf = evaluator.add_leaf(
            id=f"earbuds_top3_pub_{idx+1}",
            desc=f"Earbuds are ranked in the top 3 by a major tech review publication in 2026 (source #{idx+1})",
            parent=constraints,
            critical=True
        )
        await evaluator.verify(
            claim=f"The cited page ranks the wireless earbuds '{name}' within the top 3 (positions #1–#3) on a 'best wireless earbuds' list for 2026 by a major tech review publication.",
            node=leaf,
            sources=url,
            additional_instruction="Confirm 1) the site is a major tech review publication (e.g., Wirecutter/NYT, The Verge, CNET, PCMag, Tom's Guide, TechRadar, Rtings, SoundGuys, What Hi‑Fi?, Engadget, etc.), 2) the list is for 2026, and 3) the earbuds are in the top three of that list."
        )


async def verify_cloud_storage(evaluator: Evaluator, parent_node, data: Optional[CloudStorageSelection]) -> None:
    node = evaluator.add_sequential(
        id="Cloud_Storage_Plan",
        desc="The identified cloud storage service must offer a business plan with ≥1TB for ≤$6/month",
        parent=parent_node,
        critical=False,
    )

    name = (data.service if data else None) or ""
    plan = (data.plan_name if data else None) or ""
    urls = (data.sources if data else []) or []

    exists = bool(name.strip()) and len(urls) > 0
    evaluator.add_custom_node(
        result=exists,
        id="cloud_exists",
        desc="Cloud storage service (and optionally plan name) provided with at least one supporting source URL",
        parent=node,
        critical=True
    )

    constraints = evaluator.add_parallel(
        id="cloud_constraints",
        desc="Cloud storage business plan constraints",
        parent=node,
        critical=False
    )

    # Business plan
    leaf_business = evaluator.add_leaf(
        id="cloud_is_business_plan",
        desc="The cited plan is a business plan (not consumer/personal)",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cited page shows that the plan from '{name}'{(' named ' + plan) if plan else ''} is a business or business/teams plan (not a personal/consumer plan).",
        node=leaf_business,
        sources=urls,
        additional_instruction="Look for explicit 'Business', 'Business Standard', 'Teams', 'Work', 'Business Essentials', or similar language indicating a business plan."
    )

    # Storage capacity ≥ 1 TB
    leaf_storage = evaluator.add_leaf(
        id="cloud_storage_ge_1tb",
        desc="The business plan offers at least 1 TB of storage",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim="The cited page shows that the business plan includes storage capacity of at least 1 TB (≥ 1,000 GB) per user or for the plan.",
        node=leaf_storage,
        sources=urls,
        additional_instruction="Accept 1 TB, 2 TB, or higher. If only smaller amounts (e.g., 200 GB) are shown, do not pass."
    )

    # Price ≤ $6/month
    leaf_price = evaluator.add_leaf(
        id="cloud_price_le_6usd",
        desc="The business plan costs no more than $6 per month",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim="The cited page shows the price for the business plan is no more than $6 USD per month per user (monthly billing or the monthly equivalent of annual billing).",
        node=leaf_price,
        sources=urls,
        additional_instruction="If pricing is shown annually, convert to monthly equivalent and check ≤ $6/month. Ignore temporary trial pricing or limited-time promos. Ensure the currency is USD."
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel as per rubric
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_tech_selections(),
        template_class=TechSelections,
        extraction_name="tech_selections",
    )

    # Optional: record a quick summary of extracted names
    evaluator.add_custom_info(
        info={
            "smartphone": (extracted.smartphone.model if extracted.smartphone else None),
            "laptop_processor": (extracted.laptop_processor.processor if extracted.laptop_processor else None),
            "mobile_carrier": (extracted.mobile_carrier.carrier if extracted.mobile_carrier else None),
            "earbuds": (extracted.earbuds.model if extracted.earbuds else None),
            "cloud_storage_service": (extracted.cloud_storage.service if extracted.cloud_storage else None),
        },
        info_type="extracted_names",
    )

    # Build tree nodes per category (as children of root)
    # Smartphone
    await verify_smartphone_battery(evaluator, root, extracted.smartphone if extracted else None)

    # Laptop processor
    await verify_laptop_processor(evaluator, root, extracted.laptop_processor if extracted else None)

    # Mobile carrier
    await verify_mobile_carrier(evaluator, root, extracted.mobile_carrier if extracted else None)

    # Wireless earbuds
    await verify_earbuds(evaluator, root, extracted.earbuds if extracted else None)

    # Cloud storage
    await verify_cloud_storage(evaluator, root, extracted.cloud_storage if extracted else None)

    # Return structured evaluation summary
    return evaluator.get_summary()