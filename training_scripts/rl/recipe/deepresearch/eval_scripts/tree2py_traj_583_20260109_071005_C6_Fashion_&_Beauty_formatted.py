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
TASK_ID = "luxury_sustain_brand"
TASK_DESCRIPTION = """
Identify the luxury fashion brand that achieved all of the following sustainability milestones:
(1) was the first brand in the fashion industry to achieve Cradle to Cradle Certified Gold level specifically for wool yarn in 2017;
(2) obtained B Corp certification in 2022 with a B Impact Assessment score exceeding 90 points;
(3) had its founder/designer receive the CFDA Environmental Sustainability Award in 2023;
(4) maintains a publicly documented commitment to never using leather or fur in any of its products; and
(5) operates as a recognized luxury fashion brand.
Provide the brand name and reference URLs that verify each of the five criteria.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MilestoneEvidence(BaseModel):
    """One milestone's claim and its supporting URLs extracted from the answer."""
    claim_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class BrandSustainabilitySubmission(BaseModel):
    """Structured extraction of the answer for the sustainability milestones."""
    brand_name: Optional[str] = None

    c2c_gold_wool_2017_first: MilestoneEvidence = Field(default_factory=MilestoneEvidence)
    bcorp_2022_score_over_90: MilestoneEvidence = Field(default_factory=MilestoneEvidence)
    cfda_award_2023_founder_designer: MilestoneEvidence = Field(default_factory=MilestoneEvidence)
    never_uses_leather_or_fur: MilestoneEvidence = Field(default_factory=MilestoneEvidence)
    luxury_brand_status: MilestoneEvidence = Field(default_factory=MilestoneEvidence)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_brand_submission() -> str:
    return """
    Extract the information the answer provides about a single luxury fashion brand that meets all five specified sustainability milestones.
    You must extract strictly from the provided answer text (do not invent or infer beyond the answer).
    Return the following fields:

    1) brand_name: The single brand name explicitly stated in the answer. If multiple brands are mentioned, choose the one that the answer clearly identifies as the subject meeting all criteria. If no brand is clearly identified, return null.

    For each milestone below, extract:
    - claim_text: A short statement (paraphrase or direct quote) summarizing exactly what the answer claims for this milestone.
    - urls: A list of URLs that the answer explicitly provides to verify or support this milestone. Include only URLs present in the answer; accept both direct URLs and URLs embedded in markdown links. If no URLs are provided for a milestone, return an empty array.

    Milestones:
    a) c2c_gold_wool_2017_first:
       • The answer should claim that the brand was the first in the fashion industry to achieve Cradle to Cradle Certified Gold level specifically for wool yarn in 2017.
    b) bcorp_2022_score_over_90:
       • The answer should claim that the brand obtained B Corp certification in 2022 and that the B Impact Assessment score exceeds 90 points.
    c) cfda_award_2023_founder_designer:
       • The answer should claim that the brand’s founder/designer received the CFDA Environmental Sustainability Award in 2023.
    d) never_uses_leather_or_fur:
       • The answer should claim the brand publicly commits to never using leather or fur in any of its products.
    e) luxury_brand_status:
       • The answer should claim the brand is recognized/positioned as a luxury fashion brand.

    SPECIAL URL RULES:
    - Extract only URLs explicitly present in the answer text. Do not invent any URL.
    - Include complete URLs. If a URL misses the protocol, prepend http://.
    - If the answer provides a general source mention (e.g., “according to CFDA”), but no URL, return an empty URL list for that milestone.

    If any field cannot be found directly in the answer, set it to null (for text) or an empty list (for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _collect_all_urls(submission: BrandSustainabilitySubmission) -> List[str]:
    """Collect all unique URLs across milestones."""
    urls = []
    urls.extend(submission.c2c_gold_wool_2017_first.urls or [])
    urls.extend(submission.bcorp_2022_score_over_90.urls or [])
    urls.extend(submission.cfda_award_2023_founder_designer.urls or [])
    urls.extend(submission.never_uses_leather_or_fur.urls or [])
    urls.extend(submission.luxury_brand_status.urls or [])
    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            unique_urls.append(u)
    return unique_urls


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _add_brand_name_node(
    evaluator: Evaluator,
    parent_node,
    brand_name: Optional[str],
) -> Any:
    """
    Add and verify the Brand_Name_Provided leaf node (critical).
    """
    leaf = evaluator.add_leaf(
        id="Brand_Name_Provided",
        desc="Answer explicitly states a single brand name.",
        parent=parent_node,
        critical=True,
    )

    # Build claim focusing on the answer content
    if brand_name and brand_name.strip():
        claim = f"The answer explicitly states a single brand name: '{brand_name.strip()}'."
    else:
        claim = "The answer explicitly states a single brand name."

    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction=(
            "Check the answer text only. Pass if the answer clearly names one brand as the subject "
            "satisfying all criteria. If multiple brands are listed without a single clear subject, or "
            "if no brand is named, fail."
        ),
    )
    return leaf


async def _add_milestone_parallel(
    evaluator: Evaluator,
    parent_node,
    milestone_id: str,
    milestone_desc: str,
    satisfied_desc: str,
    evidence_desc: str,
    brand_name: Optional[str],
    claim_supported_by_urls: str,
    urls: List[str],
    brand_leaf_prereq: Any,
    additional_instruction: str,
) -> None:
    """
    Generic builder for a milestone node with two critical leaves:
      - Milestone_Satisfied (verified against provided URLs to ensure the claim is truly supported)
      - Evidence_URL_Provided (existence check for at least one URL)
    All children are critical as the parent milestone is critical in the rubric.
    """
    # Milestone parallel node (critical)
    milestone_node = evaluator.add_parallel(
        id=milestone_id,
        desc=milestone_desc,
        parent=parent_node,
        critical=True,
    )

    # Leaf 1: Milestone_Satisfied (verify claim using the provided URLs as evidence)
    satisfied_leaf = evaluator.add_leaf(
        id=f"{milestone_id}_Milestone_Satisfied",
        desc=satisfied_desc,
        parent=milestone_node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim_supported_by_urls,
        node=satisfied_leaf,
        sources=urls if urls else None,  # If empty, will route to simple verification; presence leaf below ensures URLs exist
        additional_instruction=additional_instruction,
        extra_prerequisites=[brand_leaf_prereq],  # Depend on brand name being provided
    )

    # Leaf 2: Evidence_URL_Provided (existence check of at least one URL)
    evidence_present = evaluator.add_custom_node(
        result=bool(urls) and len(urls) > 0,
        id=f"{milestone_id}_Evidence_URL_Provided",
        desc=evidence_desc,
        parent=milestone_node,
        critical=True,
    )

    # Note: We keep evidence_present as a pure existence check,
    # and we rely on the verified 'Milestone_Satisfied' above (with URLs) to ensure truthfulness.


async def _add_reference_urls_reliable(
    evaluator: Evaluator,
    parent_node,
    all_urls: List[str],
    brand_leaf_prereq: Any,
) -> None:
    """
    Add the Reference_URLs_Reliable leaf (critical).
    We verify at least one of the provided URLs is publicly accessible and from a reputable source suitable for verifying such milestones.
    """
    reliable_leaf = evaluator.add_leaf(
        id="Reference_URLs_Reliable",
        desc="Provided reference URLs are from reliable sources and are publicly accessible enough to verify the stated claims.",
        parent=parent_node,
        critical=True,
    )

    claim = (
        "At least one of the provided URLs is publicly accessible and from a reliable, authoritative source "
        "appropriate for verifying sustainability milestones (e.g., official organization pages such as Cradle to Cradle, "
        "B Corporation, CFDA; the brand's official website; or a reputable mainstream publication)."
    )

    await evaluator.verify(
        claim=claim,
        node=reliable_leaf,
        sources=all_urls if all_urls else None,
        additional_instruction=(
            "Consider reliability in terms of recognized institutions, official pages, or well‑established publications. "
            "If none of the URLs load or are clearly unreliable/irrelevant, fail."
        ),
        extra_prerequisites=[brand_leaf_prereq],
    )


# --------------------------------------------------------------------------- #
# Main verification orchestration                                             #
# --------------------------------------------------------------------------- #
async def verify_brand_identification(
    evaluator: Evaluator,
    root_node,
    submission: BrandSustainabilitySubmission,
) -> None:
    """
    Build the verification tree based on the rubric and run all milestone checks.
    The top-level node is critical and parallel; all its children are critical.
    """
    # Top-level critical parallel node
    top = evaluator.add_parallel(
        id="Luxury_Fashion_Brand_Identification",
        desc="Identify a luxury fashion brand that meets all specified sustainability milestones and provide URLs verifying each milestone.",
        parent=root_node,
        critical=True,
    )

    # 1) Brand name provided (critical leaf)
    brand_leaf = await _add_brand_name_node(evaluator, top, submission.brand_name)

    # 2) C2C Gold Wool Yarn 2017 First (critical parallel node with two critical leaves)
    await _add_milestone_parallel(
        evaluator=evaluator,
        parent_node=top,
        milestone_id="C2C_Gold_Wool_Yarn_2017_First",
        milestone_desc="Brand was the first in the fashion industry to achieve Cradle to Cradle Certified Gold specifically for wool yarn in 2017.",
        satisfied_desc="Claim includes all required parts: C2C Certified Gold, specific to wool yarn, achieved in 2017, and first in the fashion industry.",
        evidence_desc="At least one publicly accessible URL is provided that supports the full C2C milestone claim (or URLs collectively cover all parts).",
        brand_name=submission.brand_name,
        claim_supported_by_urls=(
            f"{submission.brand_name or 'The brand'} was the first in the fashion industry to achieve "
            f"Cradle to Cradle Certified Gold level specifically for wool yarn in 2017."
        ),
        urls=submission.c2c_gold_wool_2017_first.urls,
        brand_leaf_prereq=brand_leaf,
        additional_instruction=(
            "Verify ALL components: (1) Cradle to Cradle Certified Gold level; (2) specific to wool yarn; "
            "(3) achieved in 2017; and (4) first in the fashion industry. If any component is missing from the evidence, fail. "
            "Allow the URLs collectively to cover all parts."
        ),
    )

    # 3) B Corp 2022 score over 90 (critical parallel node with two critical leaves)
    await _add_milestone_parallel(
        evaluator=evaluator,
        parent_node=top,
        milestone_id="B_Corp_2022_Score_Over_90",
        milestone_desc="Brand obtained B Corp certification in 2022 with a B Impact Assessment score exceeding 90 points.",
        satisfied_desc="Claim includes both: B Corp certification year is 2022, and B Impact score is > 90.",
        evidence_desc="At least one publicly accessible URL is provided that supports the full B Corp milestone claim (or URLs collectively cover all parts).",
        brand_name=submission.brand_name,
        claim_supported_by_urls=(
            f"{submission.brand_name or 'The brand'} obtained B Corp certification in 2022 and has a B Impact "
            f"Assessment score exceeding 90 points."
        ),
        urls=submission.bcorp_2022_score_over_90.urls,
        brand_leaf_prereq=brand_leaf,
        additional_instruction=(
            "Verify BOTH elements from the evidence: certification year is 2022, AND the B Impact Assessment score is > 90. "
            "Numerical rounding is acceptable (e.g., 90.5 counts as > 90). If the year or score cannot be confirmed, fail."
        ),
    )

    # 4) CFDA Award 2023 founder/designer (critical parallel node with two critical leaves)
    await _add_milestone_parallel(
        evaluator=evaluator,
        parent_node=top,
        milestone_id="CFDA_Award_2023_Founder_Designer",
        milestone_desc="Brand’s founder/designer received the CFDA Environmental Sustainability Award in 2023.",
        satisfied_desc="Claim specifies: CFDA Environmental Sustainability Award, year 2023, and recipient is the brand’s founder/designer.",
        evidence_desc="At least one publicly accessible URL is provided that supports the full CFDA award milestone claim (or URLs collectively cover all parts).",
        brand_name=submission.brand_name,
        claim_supported_by_urls=(
            f"The founder/designer of {submission.brand_name or 'the brand'} received the CFDA Environmental "
            f"Sustainability Award in 2023."
        ),
        urls=submission.cfda_award_2023_founder_designer.urls,
        brand_leaf_prereq=brand_leaf,
        additional_instruction=(
            "Confirm the award title (CFDA Environmental Sustainability Award), the year (2023), and that the recipient is recognized "
            "as the brand’s founder or designer. If the evidence names a person (e.g., 'X') verify that 'X' is the founder/designer "
            "of the brand referenced."
        ),
    )

    # 5) Never uses leather or fur (critical parallel node with two critical leaves)
    await _add_milestone_parallel(
        evaluator=evaluator,
        parent_node=top,
        milestone_id="Never_Uses_Leather_Or_Fur",
        milestone_desc="Brand maintains a publicly documented commitment to never using leather or fur in any products.",
        satisfied_desc="Public commitment covers both: never uses leather and never uses fur.",
        evidence_desc="At least one publicly accessible URL is provided that supports the leather- and fur-free commitment (or separate URLs cover each).",
        brand_name=submission.brand_name,
        claim_supported_by_urls=(
            f"{submission.brand_name or 'The brand'} publicly commits to never using leather or fur in any of its products."
        ),
        urls=submission.never_uses_leather_or_fur.urls,
        brand_leaf_prereq=brand_leaf,
        additional_instruction=(
            "Confirm BOTH exclusions (no leather AND no fur). The evidence can be on an official brand policy page, credible certification or "
            "campaign pages, or reputable coverage directly quoting the brand policy."
        ),
    )

    # 6) Luxury brand status (critical parallel node with two critical leaves)
    await _add_milestone_parallel(
        evaluator=evaluator,
        parent_node=top,
        milestone_id="Luxury_Fashion_Brand_Status",
        milestone_desc="Brand is positioned and recognized as a luxury fashion brand.",
        satisfied_desc="Answer supports that the brand is recognized/positioned as a luxury fashion brand (not merely sustainable or mass-market).",
        evidence_desc="At least one publicly accessible URL is provided that supports luxury brand status.",
        brand_name=submission.brand_name,
        claim_supported_by_urls=(
            f"{submission.brand_name or 'The brand'} is recognized/positioned as a luxury fashion brand."
        ),
        urls=submission.luxury_brand_status.urls,
        brand_leaf_prereq=brand_leaf,
        additional_instruction=(
            "Confirm that the brand is characterized within reliable sources as 'luxury', 'luxury fashion', or equivalent, "
            "not simply sustainable or mass-market. Official brand positioning, industry analyses, or reputable publications suffice."
        ),
    )

    # 7) Reference URLs Reliable (critical leaf)
    all_urls = _collect_all_urls(submission)
    await _add_reference_urls_reliable(evaluator, top, all_urls, brand_leaf)


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
    Evaluate an answer for the luxury sustainability brand identification task
    and return a structured evaluation summary.
    """
    # Initialize evaluator with a parallel root (top-level rubric node is also parallel)
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

    # Extraction: Pull brand name and per-milestone claims and URLs from the answer
    submission = await evaluator.extract(
        prompt=prompt_extract_brand_submission(),
        template_class=BrandSustainabilitySubmission,
        extraction_name="brand_sustainability_submission",
    )

    # Build and run verification tree according to rubric
    await verify_brand_identification(evaluator, root, submission)

    # Return structured result
    return evaluator.get_summary()