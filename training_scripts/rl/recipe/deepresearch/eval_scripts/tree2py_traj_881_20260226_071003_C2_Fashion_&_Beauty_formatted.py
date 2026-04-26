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
TASK_ID = "first_joint_campaign_march_2025"
TASK_DESCRIPTION = "Which celebrity couple starred in their first joint fashion campaign together for an Italian fashion brand that launched in March 2025?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CampaignInfo(BaseModel):
    couple: List[str] = Field(default_factory=list)           # Expect up to two names
    brand: Optional[str] = None                               # Brand name
    launch_month: Optional[str] = None                        # e.g., "March"
    launch_year: Optional[str] = None                         # e.g., "2025"
    campaign_urls: List[str] = Field(default_factory=list)    # URLs supporting the campaign announcement/details
    brand_urls: List[str] = Field(default_factory=list)       # URLs about the brand itself (official site/Wikipedia/etc.)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campaign_info() -> str:
    return """
Extract from the answer all details about the celebrity couple’s first joint fashion campaign:

Fields to extract:
- couple: an array of up to two individual names that form the couple featured in the campaign (e.g., ["Name A", "Name B"]). 
  • If more than two names are mentioned, choose the two explicitly presented as the couple starring in this campaign.
  • If only one relevant name is present, include that one; otherwise return an empty array.
- brand: the fashion brand name associated with the campaign (e.g., "Versace", "Gucci").
- launch_month: the campaign launch month as written in the answer (prefer the exact string, e.g., "March", "Mar").
- launch_year: the campaign launch year as written in the answer (e.g., "2025").
- campaign_urls: all URLs cited in the answer that directly discuss/announce this specific campaign, the couple’s participation, and/or mention that it is their first campaign together.
- brand_urls: URLs cited in the answer that provide brand background or identity (e.g., official brand site, Wikipedia, reputable profiles).

Important rules:
- Return only what appears in the provided answer text exactly as written.
- For any missing field, return null (for single fields) or an empty list (for arrays).
- For URL fields, extract only valid URLs explicitly present in the answer (including those embedded in markdown links).
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _two_names_present(info: CampaignInfo) -> bool:
    return len(info.couple) >= 2 and all((n or "").strip() for n in info.couple[:2])


def _has_campaign_sources(info: CampaignInfo) -> bool:
    return bool(info.campaign_urls and len(info.campaign_urls) > 0)


def _brand_present(info: CampaignInfo) -> bool:
    return bool(info.brand and info.brand.strip())


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    root,
    info: CampaignInfo,
) -> None:
    """
    Build the verification tree following the rubric and run verifications.
    Root strategy is sequential, so we place two main children:
      1) Couple_Identification (parallel)
      2) Campaign_Details_Verification (parallel)
    """

    # ------------------------- Node 1: Couple Identification -------------------------
    couple_node = evaluator.add_parallel(
        id="Couple_Identification",
        desc="Identify the couple and verify they meet the celebrity and first-campaign requirements",
        parent=root,
        critical=False
    )

    # Existence gate: Couple names provided (custom critical node to gate related checks)
    couple_names_provided = evaluator.add_custom_node(
        result=_two_names_present(info),
        id="couple_names_provided",
        desc="Couple names are provided (two individual names identified)",
        parent=couple_node,
        critical=True
    )

    # Celebrity status verification (critical)
    celebrity_leaf = evaluator.add_leaf(
        id="Celebrity_Status_Verification",
        desc="Verify that both individuals in the couple are celebrities",
        parent=couple_node,
        critical=True
    )
    # Build claim for celebrity verification
    name1 = info.couple[0] if len(info.couple) >= 1 else ""
    name2 = info.couple[1] if len(info.couple) >= 2 else ""
    celeb_claim = f"Both individuals in the couple are celebrities: '{name1}' and '{name2}'. They are widely recognized public figures or entertainers."
    all_urls = list(dict.fromkeys((info.campaign_urls or []) + (info.brand_urls or [])))  # deduplicate while preserving order
    await evaluator.verify(
        claim=celeb_claim,
        node=celebrity_leaf,
        sources=all_urls if all_urls else None,
        additional_instruction=(
            "Treat 'celebrity' broadly as widely recognized public figures (e.g., actors, musicians, models, athletes, influencers). "
            "Use the provided webpages to confirm that both individuals are notable public figures, even if the pages describe them by profession."
        )
    )

    # We'll create the details node and gates before verifying 'first joint' so we can link as extra prerequisite
    details_node = evaluator.add_parallel(
        id="Campaign_Details_Verification",
        desc="Verify the campaign details match the specified requirements",
        parent=root,
        critical=False
    )

    # Sources gate for any campaign detail verifications
    campaign_sources_provided = evaluator.add_custom_node(
        result=_has_campaign_sources(info),
        id="campaign_sources_provided",
        desc="Campaign sources are provided (at least one URL that discusses/announces the campaign)",
        parent=details_node,
        critical=True
    )

    # Brand name availability (non-critical gate specific to brand nationality check)
    brand_name_provided = evaluator.add_custom_node(
        result=_brand_present(info),
        id="brand_name_provided",
        desc="Brand name is provided in the answer",
        parent=details_node,
        critical=False  # Not a global critical sibling; used as an explicit extra prerequisite for brand nationality check
    )

    # First joint campaign verification (critical under Couple_Identification)
    first_joint_leaf = evaluator.add_leaf(
        id="First_Joint_Campaign_Verification",
        desc="Verify this was the couple's first fashion campaign together",
        parent=couple_node,
        critical=True
    )
    # Build claim for first-joint verification
    if _brand_present(info):
        first_joint_claim = (
            f"This was the first fashion campaign together for {name1} and {name2}, specifically for the brand '{info.brand}'."
        )
    else:
        first_joint_claim = f"This was the first fashion campaign together for {name1} and {name2}."
    await evaluator.verify(
        claim=first_joint_claim,
        node=first_joint_leaf,
        sources=info.campaign_urls if info.campaign_urls else None,
        additional_instruction=(
            "Look for explicit language such as 'first joint campaign', 'first campaign together', or equivalent phrasing. "
            "General mentions of a campaign without indicating it's the first together are insufficient."
        ),
        extra_prerequisites=[campaign_sources_provided]  # Require campaign sources for this claim
    )

    # ------------------------- Node 2: Campaign Details -------------------------

    # Italian brand verification (critical)
    italian_brand_leaf = evaluator.add_leaf(
        id="Italian_Brand_Verification",
        desc="Verify the campaign was for an Italian fashion brand",
        parent=details_node,
        critical=True
    )
    brand = info.brand or ""
    brand_claim = f"'{brand}' is an Italian fashion brand."
    brand_sources = list(dict.fromkeys((info.brand_urls or []) + (info.campaign_urls or [])))
    await evaluator.verify(
        claim=brand_claim,
        node=italian_brand_leaf,
        sources=brand_sources if brand_sources else None,
        additional_instruction=(
            "Confirm that the brand is Italian (e.g., described as an Italian fashion house, founded/headquartered in Italy). "
            "Information from the brand's official site or Wikipedia counts as support. "
            "If the brand is not clearly indicated as Italian, mark as not supported."
        ),
        extra_prerequisites=[campaign_sources_provided, brand_name_provided]
    )

    # Launch timing verification (critical)
    launch_leaf = evaluator.add_leaf(
        id="Launch_Timing_Verification",
        desc="Verify the campaign launched in March 2025",
        parent=details_node,
        critical=True
    )
    # Build claim for launch timing
    if _brand_present(info) and _two_names_present(info):
        timing_claim = (
            f"The {brand} campaign featuring {name1} and {name2} launched in March 2025."
        )
    else:
        timing_claim = "The campaign launched in March 2025."
    await evaluator.verify(
        claim=timing_claim,
        node=launch_leaf,
        sources=info.campaign_urls if info.campaign_urls else None,
        additional_instruction=(
            "Verify that the cited page(s) indicate the campaign launch in March 2025. "
            "Accept equivalent phrasing like 'March 2025', 'launched in March 2025', 'released in March 2025'. "
            "Teasers or unrelated dates do not count."
        ),
        extra_prerequisites=[campaign_sources_provided]
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
    Entry point for evaluating the answer to the 'first joint campaign' task.
    Builds a verification tree and returns the evaluation summary.
    """
    # Initialize evaluator with a sequential root to mirror the rubric
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured campaign information from the answer
    info: CampaignInfo = await evaluator.extract(
        prompt=prompt_extract_campaign_info(),
        template_class=CampaignInfo,
        extraction_name="campaign_info_extraction"
    )

    # Optional: record extracted summary as custom info for debugging
    evaluator.add_custom_info(
        {
            "couple": info.couple,
            "brand": info.brand,
            "launch_month": info.launch_month,
            "launch_year": info.launch_year,
            "campaign_urls_count": len(info.campaign_urls),
            "brand_urls_count": len(info.brand_urls),
        },
        info_type="extraction_summary",
    )

    # Build the tree and run verifications
    await build_and_verify(evaluator, root, info)

    # Return structured evaluation summary
    return evaluator.get_summary()