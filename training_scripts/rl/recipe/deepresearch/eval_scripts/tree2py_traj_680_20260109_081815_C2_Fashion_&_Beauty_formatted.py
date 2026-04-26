import asyncio
import logging
import re
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bcorp_uk_beauty_2000s_female_founder"
TASK_DESCRIPTION = (
    "Identify a B Corp certified beauty brand that has its headquarters in the United Kingdom "
    "and was founded in the 2000s (between 2000-2009, inclusive) by a female founder. Provide "
    "the brand name, the founder's full name, and the exact year the brand was founded."
)

START_YEAR = 2000
END_YEAR = 2009


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BrandCandidate(BaseModel):
    """
    Structured extraction for the user's chosen brand.
    All fields must be extracted strictly from the answer text.
    """
    brand_name: Optional[str] = None
    founder_full_name: Optional[str] = None
    founding_year: Optional[str] = None
    # Optional contextual fields (if the answer mentions them)
    industry: Optional[str] = None
    headquarters_location: Optional[str] = None
    # All URLs explicitly cited in the answer as sources (including markdown links)
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_brand_candidate() -> str:
    return """
    Extract the single brand candidate presented in the answer for this task.
    Return these fields:
    - brand_name: The chosen brand's name (not a parent company).
    - founder_full_name: The founder's full name as claimed in the answer. If multiple founders are mentioned, extract the female founder name that is used for satisfying the task (if stated).
    - founding_year: The exact founding year as a 4-digit string, if provided (e.g., "2003"). If the year is missing or ambiguous, return null.
    - industry: The industry/category described in the answer for the brand (e.g., skincare, cosmetics, beauty, personal care). If not clearly stated, return null.
    - headquarters_location: The HQ location mentioned in the answer (e.g., "London, United Kingdom"). If not stated, return null.
    - source_urls: Extract all URLs explicitly cited in the answer that support any of the claims (B Corp certification, industry, HQ, founder identity, founding year). Include plain URLs and URLs found in markdown links.
    
    IMPORTANT:
    - Only extract information explicitly present in the answer text. Do not infer or introduce new info.
    - For URLs, extract them exactly as they appear. If a URL is missing a protocol, prepend http://.
    - If any field is missing in the answer, return null for that field (or an empty list for URLs).
    """


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_brand_constraints(
    evaluator: Evaluator,
    parent_node,
    data: BrandCandidate
) -> None:
    """
    Build and verify the 'identify_qualifying_brand' parallel node with all critical leaf checks.
    """
    brand = (data.brand_name or "the brand").strip() or "the brand"
    founder = (data.founder_full_name or "the founder").strip() or "the founder"
    year_str = (data.founding_year or "").strip()
    sources = data.source_urls if data.source_urls else []

    node_main = evaluator.add_parallel(
        id="identify_qualifying_brand",
        desc="Chosen brand satisfies all eligibility constraints (organization + founder constraints used for identification)",
        parent=parent_node,
        critical=True
    )

    # 1) B Corp certification as of December 2023
    n_bcorp = evaluator.add_leaf(
        id="b_corp_certification_as_of_dec_2023",
        desc="Brand is B Corp certified as of December 2023",
        parent=node_main,
        critical=True
    )
    claim_bcorp = f"As of December 2023, {brand} is certified as a B Corporation (B Corp)."
    ins_bcorp = (
        "Use the provided URLs (e.g., brand website, B Lab directory, credible press) to check if the brand "
        "is a Certified B Corporation. The page does not need to explicitly mention 'December 2023'; treat this "
        "as a contemporaneous certification status circa 2023. If the sources are irrelevant or invalid, mark as not supported."
    )

    # 2) Beauty/personal care industry
    n_industry = evaluator.add_leaf(
        id="beauty_or_personal_care_industry",
        desc="Brand operates in beauty/personal care (e.g., skincare, cosmetics, related products)",
        parent=node_main,
        critical=True
    )
    claim_industry = (
        f"{brand} operates in the beauty or personal care sector (e.g., skincare, cosmetics, haircare, fragrance, hygiene)."
    )
    ins_industry = (
        "Check product categories or brand descriptions. The brand must clearly be a beauty/personal care company. "
        "Fashion-only or appliances-only companies do NOT qualify."
    )

    # 3) UK headquarters
    n_uk_hq = evaluator.add_leaf(
        id="uk_headquarters",
        desc="Brand headquarters is located in the United Kingdom",
        parent=node_main,
        critical=True
    )
    claim_uk_hq = f"The headquarters of {brand} is located in the United Kingdom (UK)."
    ins_uk_hq = (
        "Accept UK HQ if the city is in England, Scotland, Wales, or Northern Ireland. "
        "Phrases such as 'UK-based', 'British cosmetics company', or an HQ address in the UK are sufficient."
    )

    # 4) Founded in 2000–2009 inclusive
    n_year_range = evaluator.add_leaf(
        id="founded_2000_2009_inclusive",
        desc="Brand was founded between 2000 and 2009 inclusive",
        parent=node_main,
        critical=True
    )
    year_part = year_str if year_str else "an unspecified year"
    claim_year_range = (
        f"{brand} was founded in {year_part}, which lies between 2000 and 2009 inclusive."
    )
    ins_year_range = (
        "First confirm the founding year from the source(s). Then judge whether the founding year is within 2000–2009 inclusive. "
        "If the page suggests multiple years, use the one most clearly supported as the founding year."
    )

    # 5) Female founder
    n_female_founder = evaluator.add_leaf(
        id="female_founder",
        desc="Brand was founded by a woman",
        parent=node_main,
        critical=True
    )
    claim_female_founder = f"The brand {brand} was founded by a woman named {founder}."
    ins_female_founder = (
        "Use biography, pronouns, or reputable references to determine that the named founder is female. "
        "If multiple co-founders exist, it is sufficient that at least one founder is a woman and the answer selected her."
    )

    # Batch verify the 5 constraints (parallel)
    await evaluator.batch_verify([
        (claim_bcorp, sources, n_bcorp, ins_bcorp),
        (claim_industry, sources, n_industry, ins_industry),
        (claim_uk_hq, sources, n_uk_hq, ins_uk_hq),
        (claim_year_range, sources, n_year_range, ins_year_range),
        (claim_female_founder, sources, n_female_founder, ins_female_founder),
    ])


def add_required_outputs_provided(
    evaluator: Evaluator,
    parent_node,
    data: BrandCandidate
) -> None:
    """
    Add the 'required_outputs_provided' parallel node with critical existence checks.
    """
    node_main = evaluator.add_parallel(
        id="required_outputs_provided",
        desc="Response includes all required fields",
        parent=parent_node,
        critical=True
    )

    # Brand name provided
    brand_ok = bool(data.brand_name and data.brand_name.strip())
    evaluator.add_custom_node(
        result=brand_ok,
        id="brand_name_provided",
        desc="Brand name is provided",
        parent=node_main,
        critical=True
    )

    # Founder full name provided
    founder_ok = bool(data.founder_full_name and data.founder_full_name.strip())
    evaluator.add_custom_node(
        result=founder_ok,
        id="founder_full_name_provided",
        desc="Founder’s full name is provided",
        parent=node_main,
        critical=True
    )

    # Exact founding year provided (prefer a 4-digit year)
    year_ok = bool(data.founding_year and data.founding_year.strip())
    if year_ok:
        year_ok = bool(re.fullmatch(r"\d{4}", data.founding_year.strip()))
    evaluator.add_custom_node(
        result=year_ok,
        id="exact_founding_year_provided",
        desc="Exact founding year is provided",
        parent=node_main,
        critical=True
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
    Evaluate an answer for the B Corp UK beauty brand founded in 2000s by a female founder task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root stays non-critical in framework; we add a critical top-level node below
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

    # Top-level critical sequential node representing the entire task
    task_main = evaluator.add_sequential(
        id="task_main",
        desc="Identify a B Corp certified UK beauty/personal care brand founded in 2000–2009 by a female founder, and provide brand name, founder full name, and exact founding year",
        parent=root,
        critical=True
    )

    # Extract candidate info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_brand_candidate(),
        template_class=BrandCandidate,
        extraction_name="brand_candidate"
    )

    # Optional: record constraints in summary
    evaluator.add_ground_truth({
        "constraints": {
            "bcorp_certified": True,
            "industry": "beauty/personal care",
            "hq_country": "United Kingdom",
            "founding_year_range": [START_YEAR, END_YEAR],
            "female_founder_required": True
        }
    }, gt_type="task_constraints")

    # Build constraint verification (Step 1 in sequential flow)
    await verify_brand_constraints(evaluator, task_main, extracted)

    # Build required outputs existence checks (Step 2 in sequential flow)
    add_required_outputs_provided(evaluator, task_main, extracted)

    # Return evaluation summary
    return evaluator.get_summary()