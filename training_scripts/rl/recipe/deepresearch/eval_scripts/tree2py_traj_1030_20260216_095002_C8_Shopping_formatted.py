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
TASK_ID = "holiday_retailers_2025"
TASK_DESCRIPTION = """
A family is planning their 2025 holiday shopping strategy across major U.S. retail chains and needs to identify four different retailers that meet specific operational requirements during the Thanksgiving through Christmas period.

Identify four major U.S. retailers (each from chains such as Target, Walmart, Best Buy, Kohl's, Costco, Macy's, Home Depot, Lowe's, or similar national chains) where:

Retailer 1 must satisfy ALL of the following criteria:
- Closed on Thanksgiving Day 2025 (Thursday, November 27, 2025)
- Opens at 6:00 a.m. or earlier on Black Friday 2025 (Friday, November 28, 2025)
- Remains open until 10:00 p.m. or later on Black Friday 2025
- Operates with reduced hours on Christmas Eve 2025 (Tuesday, December 24, 2025), closing before its typical evening closing time
- Offers free curbside pickup service with a minimum order requirement of exactly $35 or less (including $0 minimum)

Retailer 2 must satisfy ALL of the following criteria:
- Closed on Thanksgiving Day 2025 (Thursday, November 27, 2025)
- Opens at 6:00 a.m. or earlier on Black Friday 2025 (Friday, November 28, 2025)
- Remains open until 10:00 p.m. or later on Black Friday 2025

Retailer 3 must satisfy ALL of the following criteria:
- Operates with reduced hours on Christmas Eve 2025 (Tuesday, December 24, 2025)
- Closes at or before 8:00 p.m. on Christmas Eve 2025

Retailer 4 must satisfy ALL of the following criteria:
- Closed on Thanksgiving Day 2025 (Thursday, November 27, 2025)
- Closes between 5:00 p.m. and 7:00 p.m. (inclusive) on Christmas Eve 2025 (Tuesday, December 24, 2025)

For each retailer identified, provide the specific retailer name and reference URL(s) that verify the operational hours and policies stated above.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Retailer(BaseModel):
    """Model to represent a single retailer and its cited reference URLs."""
    name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class RetailersExtraction(BaseModel):
    """Model for the extracted retailers (first four only)."""
    retailer1: Optional[Retailer] = None
    retailer2: Optional[Retailer] = None
    retailer3: Optional[Retailer] = None
    retailer4: Optional[Retailer] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_retailers() -> str:
    return """
    From the provided answer, extract up to four major U.S. retailers that the answer proposes, along with all reference URL(s) the answer cites to support their holiday operating hours and/or curbside pickup policy.

    Extraction rules:
    - Only include the first four retailers mentioned in the answer, in order.
    - For each retailer, extract:
      1) name: The retailer chain name (e.g., "Target", "Walmart", "Best Buy", "Costco").
      2) urls: A list of all URLs the answer cites as evidence for holiday hours (Thanksgiving 2025, Black Friday 2025, Christmas Eve 2025) or curbside pickup policy.
    - Extract URLs exactly as presented (plain URLs or inside markdown links). Normalize to include protocol (http:// or https://).
    - If the answer does not provide any URLs for a retailer, return an empty list for 'urls'.
    - If a retailer name is missing or unclear, set 'name' to null.

    Return a JSON object with fields:
    - retailer1: { name, urls }
    - retailer2: { name, urls }
    - retailer3: { name, urls }
    - retailer4: { name, urls }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _urls_present(urls: List[str]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls)


def _retailer_name_present(name: Optional[str]) -> bool:
    return bool(name) and bool(name.strip())


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_retailer_1(evaluator: Evaluator, parent_node, retailer: Optional[Retailer]) -> None:
    """
    Build and verify Retailer 1 subtree:
    Criteria:
    - Closed on Thanksgiving Day 2025
    - Opens at 6:00 a.m. or earlier on Black Friday 2025
    - Remains open until 10:00 p.m. or later on Black Friday 2025
    - Reduced hours on Christmas Eve 2025 (closing earlier than typical)
    - Free curbside pickup with minimum order requirement of $35 or less (including $0)
    - Provide reference URL(s)
    """
    node = evaluator.add_parallel(
        id="Retailer_1",
        desc="A retailer that is closed on Thanksgiving, operates extended Black Friday hours, has reduced Christmas Eve hours, and offers convenient curbside pickup",
        parent=parent_node,
        critical=False
    )

    name = retailer.name if retailer else None
    urls = retailer.urls if retailer else []

    # Existence checks (Critical)
    evaluator.add_custom_node(
        result=_retailer_name_present(name),
        id="R1_Name_Provided",
        desc="Retailer 1: Retailer name is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_present(urls),
        id="R1_URL_Reference",
        desc="Provide reference URL(s) that verify the retailer's operational hours and curbside pickup policy",
        parent=node,
        critical=True
    )

    # Build leaf nodes
    leaf_thanks = evaluator.add_leaf(
        id="R1_Thanksgiving_Closed",
        desc="The retailer is closed on Thanksgiving Day 2025 (Thursday, November 27, 2025)",
        parent=node,
        critical=True
    )
    leaf_bf_open = evaluator.add_leaf(
        id="R1_BlackFriday_Opens_Early",
        desc="The retailer opens at 6:00 a.m. or earlier on Black Friday 2025 (Friday, November 28, 2025)",
        parent=node,
        critical=True
    )
    leaf_bf_close = evaluator.add_leaf(
        id="R1_BlackFriday_Closes_Late",
        desc="The retailer remains open until 10:00 p.m. or later on Black Friday 2025",
        parent=node,
        critical=True
    )
    leaf_xmas_reduced = evaluator.add_leaf(
        id="R1_ChristmasEve_Reduced",
        desc="The retailer operates with reduced hours on Christmas Eve 2025 (Tuesday, December 24, 2025), closing before its typical evening closing time",
        parent=node,
        critical=True
    )
    leaf_curbside = evaluator.add_leaf(
        id="R1_Curbside_Minimum",
        desc="The retailer offers free curbside pickup service with a minimum order requirement of exactly $35 or less (including $0 minimum)",
        parent=node,
        critical=True
    )

    # Prepare claims
    retailer_name = name or ""
    claims_and_sources = [
        (
            f"{retailer_name} is closed on Thanksgiving Day 2025 (Thursday, November 27, 2025).",
            urls,
            leaf_thanks,
            "Verify that the source explicitly indicates the retailer is closed on Thanksgiving Day 2025. Corporate holiday hours pages or 2025-specific announcements are acceptable."
        ),
        (
            f"{retailer_name} opens at 6:00 a.m. or earlier on Black Friday 2025 (Friday, November 28, 2025).",
            urls,
            leaf_bf_open,
            "Confirm the Black Friday 2025 opening time is at or before 6:00 a.m. Accept wording like 'Doors open at 6am' or earlier."
        ),
        (
            f"{retailer_name} remains open until 10:00 p.m. or later on Black Friday 2025.",
            urls,
            leaf_bf_close,
            "Confirm the Black Friday 2025 closing time is at or after 10:00 p.m."
        ),
        (
            f"{retailer_name} has reduced hours on Christmas Eve 2025 and closes earlier than its typical evening closing time.",
            urls,
            leaf_xmas_reduced,
            "The source should indicate shorter/limited hours for Christmas Eve 2025 (e.g., closing at 6–8pm vs. typical later closing). Accept corporate holiday hours notices."
        ),
        (
            f"{retailer_name} offers free curbside pickup service with a minimum order requirement of $35 or less (including $0 minimum).",
            urls,
            leaf_curbside,
            "Check the curbside/drive-up policy page: service must be free to the customer and the minimum order requirement must be <= $35 (including $0). 'No minimum' qualifies as $0."
        ),
    ]

    # Execute verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


async def verify_retailer_2(evaluator: Evaluator, parent_node, retailer: Optional[Retailer]) -> None:
    """
    Build and verify Retailer 2 subtree:
    Criteria:
    - Closed on Thanksgiving Day 2025
    - Opens at 6:00 a.m. or earlier on Black Friday 2025
    - Remains open until 10:00 p.m. or later on Black Friday 2025
    - Provide reference URL(s)
    """
    node = evaluator.add_parallel(
        id="Retailer_2",
        desc="A second retailer that is closed on Thanksgiving and operates extended Black Friday hours",
        parent=parent_node,
        critical=False
    )

    name = retailer.name if retailer else None
    urls = retailer.urls if retailer else []

    # Existence checks (Critical)
    evaluator.add_custom_node(
        result=_retailer_name_present(name),
        id="R2_Name_Provided",
        desc="Retailer 2: Retailer name is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_present(urls),
        id="R2_URL_Reference",
        desc="Provide reference URL(s) that verify the retailer's operational hours",
        parent=node,
        critical=True
    )

    # Leaf nodes
    leaf_thanks = evaluator.add_leaf(
        id="R2_Thanksgiving_Closed",
        desc="The retailer is closed on Thanksgiving Day 2025 (Thursday, November 27, 2025)",
        parent=node,
        critical=True
    )
    leaf_bf_open = evaluator.add_leaf(
        id="R2_BlackFriday_Opens_Early",
        desc="The retailer opens at 6:00 a.m. or earlier on Black Friday 2025 (Friday, November 28, 2025)",
        parent=node,
        critical=True
    )
    leaf_bf_close = evaluator.add_leaf(
        id="R2_BlackFriday_Closes_Late",
        desc="The retailer remains open until 10:00 p.m. or later on Black Friday 2025",
        parent=node,
        critical=True
    )

    retailer_name = name or ""
    claims_and_sources = [
        (
            f"{retailer_name} is closed on Thanksgiving Day 2025 (Thursday, November 27, 2025).",
            urls,
            leaf_thanks,
            "Verify that the source explicitly indicates the retailer is closed on Thanksgiving Day 2025."
        ),
        (
            f"{retailer_name} opens at 6:00 a.m. or earlier on Black Friday 2025 (Friday, November 28, 2025).",
            urls,
            leaf_bf_open,
            "Confirm the Black Friday 2025 opening time is at or before 6:00 a.m."
        ),
        (
            f"{retailer_name} remains open until 10:00 p.m. or later on Black Friday 2025.",
            urls,
            leaf_bf_close,
            "Confirm the Black Friday 2025 closing time is at or after 10:00 p.m."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


async def verify_retailer_3(evaluator: Evaluator, parent_node, retailer: Optional[Retailer]) -> None:
    """
    Build and verify Retailer 3 subtree:
    Criteria:
    - Reduced hours on Christmas Eve 2025
    - Closes at or before 8:00 p.m. on Christmas Eve 2025
    - Provide reference URL(s)
    """
    node = evaluator.add_parallel(
        id="Retailer_3",
        desc="A third retailer that operates with specific reduced Christmas Eve hours",
        parent=parent_node,
        critical=False
    )

    name = retailer.name if retailer else None
    urls = retailer.urls if retailer else []

    # Existence checks (Critical)
    evaluator.add_custom_node(
        result=_retailer_name_present(name),
        id="R3_Name_Provided",
        desc="Retailer 3: Retailer name is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_present(urls),
        id="R3_URL_Reference",
        desc="Provide reference URL(s) that verify the retailer's Christmas Eve 2025 operating hours",
        parent=node,
        critical=True
    )

    # Leaf nodes
    leaf_xmas_reduced = evaluator.add_leaf(
        id="R3_ChristmasEve_Reduced",
        desc="The retailer operates with reduced hours on Christmas Eve 2025 (Tuesday, December 24, 2025)",
        parent=node,
        critical=True
    )
    leaf_xmas_8pm = evaluator.add_leaf(
        id="R3_ChristmasEve_Closes_8PM",
        desc="The retailer closes at or before 8:00 p.m. on Christmas Eve 2025",
        parent=node,
        critical=True
    )

    retailer_name = name or ""
    claims_and_sources = [
        (
            f"{retailer_name} operates with reduced hours on Christmas Eve 2025 (Tuesday, December 24, 2025).",
            urls,
            leaf_xmas_reduced,
            "Confirm the source indicates shorter/special hours for Christmas Eve 2025 compared to typical hours."
        ),
        (
            f"{retailer_name} closes at or before 8:00 p.m. on Christmas Eve 2025.",
            urls,
            leaf_xmas_8pm,
            "Check that the Christmas Eve closing time shown is 8:00 p.m. or earlier."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


async def verify_retailer_4(evaluator: Evaluator, parent_node, retailer: Optional[Retailer]) -> None:
    """
    Build and verify Retailer 4 subtree:
    Criteria:
    - Closed on Thanksgiving Day 2025
    - Closes between 5:00 p.m. and 7:00 p.m. (inclusive) on Christmas Eve 2025
    - Provide reference URL(s)
    """
    node = evaluator.add_parallel(
        id="Retailer_4",
        desc="A fourth retailer that is closed on Thanksgiving and closes early on Christmas Eve",
        parent=parent_node,
        critical=False
    )

    name = retailer.name if retailer else None
    urls = retailer.urls if retailer else []

    # Existence checks (Critical)
    evaluator.add_custom_node(
        result=_retailer_name_present(name),
        id="R4_Name_Provided",
        desc="Retailer 4: Retailer name is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_present(urls),
        id="R4_URL_Reference",
        desc="Provide reference URL(s) that verify the retailer's Christmas Eve 2025 operating hours",
        parent=node,
        critical=True
    )

    # Leaf nodes
    leaf_thanks = evaluator.add_leaf(
        id="R4_Thanksgiving_Closed",
        desc="The retailer is closed on Thanksgiving Day 2025 (Thursday, November 27, 2025)",
        parent=node,
        critical=True
    )
    leaf_xmas_range = evaluator.add_leaf(
        id="R4_ChristmasEve_Early",
        desc="The retailer closes between 5:00 p.m. and 7:00 p.m. on Christmas Eve 2025 (Tuesday, December 24, 2025)",
        parent=node,
        critical=True
    )

    retailer_name = name or ""
    claims_and_sources = [
        (
            f"{retailer_name} is closed on Thanksgiving Day 2025 (Thursday, November 27, 2025).",
            urls,
            leaf_thanks,
            "Verify that the source explicitly indicates the retailer is closed on Thanksgiving Day 2025."
        ),
        (
            f"{retailer_name} closes between 5:00 p.m. and 7:00 p.m. (inclusive) on Christmas Eve 2025 (Tuesday, December 24, 2025).",
            urls,
            leaf_xmas_range,
            "Check the Christmas Eve 2025 closing time; it must be within the inclusive window 5:00 p.m. to 7:00 p.m."
        ),
    ]

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
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 'holiday_retailers_2025' task.
    Builds a verification tree according to the rubric and returns a structured evaluation summary.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates retailers independently
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify four major U.S. retailers that meet specific holiday operational criteria for Thanksgiving 2025, Black Friday 2025, and Christmas Eve 2025",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract retailers and their URLs
    extracted = await evaluator.extract(
        prompt=prompt_extract_retailers(),
        template_class=RetailersExtraction,
        extraction_name="retailers_extraction"
    )

    # Build and verify retailer subtrees
    # Run each retailer subtree construction and verification (can be parallelized)
    tasks = [
        verify_retailer_1(evaluator, root, extracted.retailer1),
        verify_retailer_2(evaluator, root, extracted.retailer2),
        verify_retailer_3(evaluator, root, extracted.retailer3),
        verify_retailer_4(evaluator, root, extracted.retailer4),
    ]
    await asyncio.gather(*tasks)

    # Return structured result
    return evaluator.get_summary()