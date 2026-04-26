import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "respawn_research"
TASK_DESCRIPTION = """
Respawn Entertainment has been a major force in the gaming industry. For two of their key titles, provide the following information:

1. For Apex Legends: What are the minimum RAM requirement (in GB) and the minimum GPU model required to run the game?

2. For Titanfall 2: What is the maximum framerate supported (in fps) when running with uncapped framerate, and in what year was the game released?

Additionally, answer the following:

3. What is the full name of the co-founder of Respawn Entertainment who served as the studio head and died on December 21, 2025? Provide the specific date of death and the exact vehicle model (including year and full model name) involved in the fatal crash.

4. What was the total prize pool amount for the Esports World Cup 2025?

For each piece of information, provide supporting reference URLs.
"""

# --------------------------------------------------------------------------- #
# Ground truth expectations (for transparency in the summary)                 #
# --------------------------------------------------------------------------- #
GROUND_TRUTH = {
    "apex_legends": {
        "min_ram_gb": "6 GB",
        "min_gpu_models": ["NVIDIA GeForce GT 640", "AMD Radeon HD 7730"]
    },
    "titanfall_2": {
        "max_uncapped_fps": "144 fps",
        "release_year": "2016"
    },
    "founder": {
        "acceptable_names": ["Vince Zampella", "Vincent Walter Zampella II"],
        "date_of_death": "December 21, 2025",
        "vehicle_model": "2026 Ferrari 296 GTS"
    },
    "ewc_2025": {
        "prize_pool_minimum": "$70,000,000",
        "common_phrasings": ["$70+ million", "over $70 million", "approximately $70.45 million"]
    }
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ApexRequirements(BaseModel):
    min_ram_gb: Optional[str] = None
    min_gpu_models: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class TitanfallSpecs(BaseModel):
    max_uncapped_fps: Optional[str] = None
    release_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FounderDetails(BaseModel):
    full_name: Optional[str] = None
    date_of_death: Optional[str] = None
    vehicle_model: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EWCPrize(BaseModel):
    total_prize_pool: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RespawnResearchExtraction(BaseModel):
    apex_legends: Optional[ApexRequirements] = None
    titanfall_2: Optional[TitanfallSpecs] = None
    founder_info: Optional[FounderDetails] = None
    ewc_prize_info: Optional[EWCPrize] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_respawn() -> str:
    return """
    Extract structured information from the answer for the following items. If any field is missing, return null (for strings) or an empty list (for arrays). Extract only what is explicitly present in the answer text.

    Schema:
    {
      "apex_legends": {
        "min_ram_gb": string | null,
        "min_gpu_models": string[]  // list of GPU model names shown as minimum requirement in the answer,
        "sources": string[]         // all URLs explicitly cited that support Apex Legends system requirements
      },
      "titanfall_2": {
        "max_uncapped_fps": string | null,   // e.g., "144 fps", "uncapped", etc., as stated in the answer
        "release_year": string | null,       // e.g., "2016"
        "sources": string[]                  // all URLs explicitly cited that support Titanfall 2 technical specs and release year
      },
      "founder_info": {
        "full_name": string | null,          // founder's full name as stated in the answer
        "date_of_death": string | null,      // specific date as stated in the answer, e.g., "December 21, 2025"
        "vehicle_model": string | null,      // full vehicle model with year, e.g., "2026 Ferrari 296 GTS"
        "sources": string[]                  // all URLs explicitly cited that support the founder identity and death details
      },
      "ewc_prize_info": {
        "total_prize_pool": string | null,   // prize pool as stated, e.g., "$70+ million", "$70,450,000"
        "sources": string[]                  // all URLs explicitly cited that support the EWC 2025 prize pool
      }
    }

    Special URL extraction rules:
    - Only include actual URLs shown in the answer. Extract valid HTTP/HTTPS URLs (including markdown links).
    - Do not invent URLs. If none are provided, keep the array empty.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _fmt_list(items: List[str]) -> str:
    if not items:
        return "[]"
    return "[" + "; ".join(items) + "]"


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_apex_legends(evaluator: Evaluator, parent_node, apex: ApexRequirements) -> None:
    # Create a container node for Apex Legends info (non-critical at top level, per rubric)
    apex_main = evaluator.add_parallel(
        id="apex_legends_info",
        desc="Verification of Apex Legends specifications",
        parent=parent_node,
        critical=False
    )

    # Critical group: system requirements
    apex_sys = evaluator.add_parallel(
        id="apex_system_requirements",
        desc="System requirements verification for Apex Legends",
        parent=apex_main,
        critical=True
    )

    # Sub-group for RAM (critical)
    ram_group = evaluator.add_parallel(
        id="apex_min_ram_group",
        desc="Minimum RAM requirement is 6 GB (with supporting references)",
        parent=apex_sys,
        critical=True
    )

    # Leaf: RAM value matches expected "6 GB"
    ram_leaf = evaluator.add_leaf(
        id="apex_min_ram",
        desc="Minimum RAM requirement is 6 GB",
        parent=ram_group,
        critical=True
    )
    ram_claim = f"According to the answer, the minimum RAM requirement for Apex Legends is '{apex.min_ram_gb}'. This matches '6 GB'."
    await evaluator.verify(
        claim=ram_claim,
        node=ram_leaf,
        additional_instruction="Treat '6GB', '6 GB', or '6 gigabytes' as equivalent. If the answer states a different amount (e.g., 8 GB), it does not match."
    )

    # Leaf: RAM supported by sources
    ram_ref_leaf = evaluator.add_leaf(
        id="apex_ram_reference",
        desc="Reference URL supporting Apex Legends RAM requirement",
        parent=ram_group,
        critical=True
    )
    await evaluator.verify(
        claim="Apex Legends minimum RAM requirement is 6 GB.",
        node=ram_ref_leaf,
        sources=apex.sources,
        additional_instruction="Use the provided URLs to confirm the minimum RAM requirement is explicitly 6 GB. Accept official game pages, store listings, or reputable tech sites."
    )

    # Sub-group for GPU (critical)
    gpu_group = evaluator.add_parallel(
        id="apex_min_gpu_group",
        desc="Minimum GPU is NVIDIA GeForce GT 640 or AMD Radeon HD 7730 (with supporting references)",
        parent=apex_sys,
        critical=True
    )

    # Leaf: GPU value matches one of the expected models
    gpu_leaf = evaluator.add_leaf(
        id="apex_min_gpu",
        desc="Minimum GPU is NVIDIA GeForce GT 640 or AMD Radeon HD 7730",
        parent=gpu_group,
        critical=True
    )
    gpu_list_str = _fmt_list(apex.min_gpu_models)
    gpu_claim = (
        f"According to the answer, the minimum GPU(s) listed for Apex Legends are: {gpu_list_str}. "
        f"This matches either 'NVIDIA GeForce GT 640' or 'AMD Radeon HD 7730'. Minor naming variations (e.g., including 'GeForce') are acceptable."
    )
    await evaluator.verify(
        claim=gpu_claim,
        node=gpu_leaf,
        additional_instruction="Consider model name variants and casing (e.g., 'GeForce GT 640', 'Radeon HD 7730'). If none of the extracted GPUs match, this should be incorrect."
    )

    # Leaf: GPU supported by sources
    gpu_ref_leaf = evaluator.add_leaf(
        id="apex_gpu_reference",
        desc="Reference URL supporting Apex Legends GPU requirement",
        parent=gpu_group,
        critical=True
    )
    await evaluator.verify(
        claim="Apex Legends minimum GPU requirement is NVIDIA GeForce GT 640 or AMD Radeon HD 7730.",
        node=gpu_ref_leaf,
        sources=apex.sources,
        additional_instruction="Verify that at least one provided URL explicitly lists either 'GeForce GT 640' or 'Radeon HD 7730' as the minimum GPU requirement."
    )


async def verify_titanfall_2(evaluator: Evaluator, parent_node, tf2: TitanfallSpecs) -> None:
    tf2_main = evaluator.add_parallel(
        id="titanfall_2_info",
        desc="Verification of Titanfall 2 specifications",
        parent=parent_node,
        critical=False
    )

    tf2_specs = evaluator.add_parallel(
        id="titanfall_specifications",
        desc="Technical specifications and features for Titanfall 2",
        parent=tf2_main,
        critical=True
    )

    # Framerate group
    fr_group = evaluator.add_parallel(
        id="titanfall_framerate_group",
        desc="Supports uncapped framerate up to 144fps (with supporting references)",
        parent=tf2_specs,
        critical=True
    )

    fr_leaf = evaluator.add_leaf(
        id="titanfall_framerate_support",
        desc="Supports uncapped framerate up to 144fps",
        parent=fr_group,
        critical=True
    )
    fr_claim = (
        f"According to the answer, Titanfall 2's uncapped framerate is stated as '{tf2.max_uncapped_fps}'. "
        f"This matches support up to 144 fps."
    )
    await evaluator.verify(
        claim=fr_claim,
        node=fr_leaf,
        additional_instruction="If the answer indicates '144 fps' or an uncapped framerate plausibly reaching 144 Hz monitors, consider it a match."
    )

    fr_ref_leaf = evaluator.add_leaf(
        id="titanfall_framerate_reference",
        desc="Reference URL supporting Titanfall 2 framerate capability",
        parent=fr_group,
        critical=True
    )
    await evaluator.verify(
        claim="Titanfall 2 supports an uncapped framerate up to 144 fps.",
        node=fr_ref_leaf,
        sources=tf2.sources,
        additional_instruction="Confirm via the provided URLs (e.g., PC platform details, performance guides, official posts) that 144 fps is supported or achievable."
    )

    # Release year group
    rel_group = evaluator.add_parallel(
        id="titanfall_release_group",
        desc="Released in 2016 (with supporting references)",
        parent=tf2_specs,
        critical=True
    )

    rel_leaf = evaluator.add_leaf(
        id="titanfall_release_year",
        desc="Released in 2016",
        parent=rel_group,
        critical=True
    )
    rel_claim = f"According to the answer, Titanfall 2 was released in '{tf2.release_year}'. This matches 2016."
    await evaluator.verify(
        claim=rel_claim,
        node=rel_leaf,
        additional_instruction="Accept '2016' even if the exact month/day is omitted. If the answer states a different year, it's incorrect."
    )

    rel_ref_leaf = evaluator.add_leaf(
        id="titanfall_release_reference",
        desc="Reference URL supporting Titanfall 2 release year",
        parent=rel_group,
        critical=True
    )
    await evaluator.verify(
        claim="Titanfall 2 was released in 2016.",
        node=rel_ref_leaf,
        sources=tf2.sources,
        additional_instruction="Verify via official announcements, store pages, or reputable sources that Titanfall 2 released in 2016."
    )


async def verify_founder(evaluator: Evaluator, parent_node, founder: FounderDetails) -> None:
    founder_main = evaluator.add_parallel(
        id="studio_founder_info",
        desc="Information about Respawn Entertainment's co-founder and studio head",
        parent=parent_node,
        critical=False
    )

    ident_group = evaluator.add_parallel(
        id="founder_identity",
        desc="Identity verification of the co-founder",
        parent=founder_main,
        critical=True
    )

    name_group = evaluator.add_parallel(
        id="founder_name_group",
        desc="Name is Vince Zampella (or Vincent Walter Zampella II) (with supporting references)",
        parent=ident_group,
        critical=True
    )

    name_leaf = evaluator.add_leaf(
        id="founder_name",
        desc="Name is Vince Zampella (or Vincent Walter Zampella II)",
        parent=name_group,
        critical=True
    )
    name_claim = (
        f"According to the answer, the founder mentioned is '{founder.full_name}'. "
        f"This matches 'Vince Zampella' or 'Vincent Walter Zampella II' (minor variations acceptable)."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        additional_instruction="Allow minor variants like 'Vincent Zampella'. He is a co-founder of Respawn and served as studio head."
    )

    name_ref_leaf = evaluator.add_leaf(
        id="founder_name_reference",
        desc="Reference URL supporting founder's name",
        parent=name_group,
        critical=True
    )
    await evaluator.verify(
        claim="Respawn Entertainment's co-founder and studio head is Vince Zampella (also known as Vincent Walter Zampella II).",
        node=name_ref_leaf,
        sources=founder.sources,
        additional_instruction="Use provided URLs to confirm Vince Zampella's identity and role."
    )

    death_group = evaluator.add_parallel(
        id="founder_death_details",
        desc="Details about the founder's death",
        parent=founder_main,
        critical=True
    )

    # Date sub-group
    dd_group = evaluator.add_parallel(
        id="death_date_group",
        desc="Death occurred on December 21, 2025 (with supporting references)",
        parent=death_group,
        critical=True
    )

    dd_leaf = evaluator.add_leaf(
        id="death_date",
        desc="Death occurred on December 21, 2025",
        parent=dd_group,
        critical=True
    )
    dd_claim = f"According to the answer, the date of death is '{founder.date_of_death}', which matches December 21, 2025."
    await evaluator.verify(
        claim=dd_claim,
        node=dd_leaf,
        additional_instruction="Match 'December 21, 2025'. Minor format variations (e.g., 'Dec 21, 2025') acceptable."
    )

    dd_ref_leaf = evaluator.add_leaf(
        id="death_date_reference",
        desc="Reference URL supporting death date",
        parent=dd_group,
        critical=True
    )
    await evaluator.verify(
        claim="Vince Zampella died on December 21, 2025.",
        node=dd_ref_leaf,
        sources=founder.sources,
        additional_instruction="Confirm exact date via the provided URLs (e.g., news reports, official statements)."
    )

    # Vehicle sub-group
    dv_group = evaluator.add_parallel(
        id="death_vehicle_group",
        desc="Vehicle involved was a 2026 Ferrari 296 GTS (with supporting references)",
        parent=death_group,
        critical=True
    )

    dv_leaf = evaluator.add_leaf(
        id="death_vehicle",
        desc="Vehicle involved was a 2026 Ferrari 296 GTS",
        parent=dv_group,
        critical=True
    )
    dv_claim = f"According to the answer, the vehicle involved was '{founder.vehicle_model}', which matches '2026 Ferrari 296 GTS' (year and full model name)."
    await evaluator.verify(
        claim=dv_claim,
        node=dv_leaf,
        additional_instruction="Ensure the year '2026' and model 'Ferrari 296 GTS' both match."
    )

    dv_ref_leaf = evaluator.add_leaf(
        id="death_vehicle_reference",
        desc="Reference URL supporting vehicle information",
        parent=dv_group,
        critical=True
    )
    await evaluator.verify(
        claim="The vehicle involved in the fatal crash was a 2026 Ferrari 296 GTS.",
        node=dv_ref_leaf,
        sources=founder.sources,
        additional_instruction="Verify the exact year and model from the provided URLs."
    )


async def verify_ewc_prize(evaluator: Evaluator, parent_node, ewc: EWCPrize) -> None:
    ewc_main = evaluator.add_parallel(
        id="ewc_prize_info",
        desc="Verification of Esports World Cup 2025 prize pool information",
        parent=parent_node,
        critical=False
    )

    ewc_data = evaluator.add_parallel(
        id="ewc_prize_data",
        desc="Prize pool amount verification",
        parent=ewc_main,
        critical=True
    )

    prize_group = evaluator.add_parallel(
        id="ewc_total_prize_group",
        desc="Total prize pool is over $70 million (with supporting references)",
        parent=ewc_data,
        critical=True
    )

    prize_leaf = evaluator.add_leaf(
        id="total_prize_amount",
        desc="Total prize pool is over $70 million (specifically over $70,450,000 or commonly stated as $70+ million)",
        parent=prize_group,
        critical=True
    )
    prize_claim = (
        f"According to the answer, the Esports World Cup 2025 prize pool is stated as '{ewc.total_prize_pool}'. "
        f"This is over $70 million (i.e., >= $70,000,000). Accept phrasings like '$70+ million' or '~$70.45 million'."
    )
    await evaluator.verify(
        claim=prize_claim,
        node=prize_leaf,
        additional_instruction="Judge whether the stated amount clearly indicates >= $70,000,000."
    )

    prize_ref_leaf = evaluator.add_leaf(
        id="ewc_prize_reference",
        desc="Reference URL supporting EWC prize information",
        parent=prize_group,
        critical=True
    )
    await evaluator.verify(
        claim="The Esports World Cup 2025 total prize pool is over $70 million.",
        node=prize_ref_leaf,
        sources=ewc.sources,
        additional_instruction="Verify via provided URLs that EWC 2025 prize pool is explicitly stated as > $70M (allow '$70+ million' or specific amounts like ~$70.45M)."
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
    # Initialize evaluator with root parallel aggregation
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_respawn(),
        template_class=RespawnResearchExtraction,
        extraction_name="respawn_research_extraction"
    )

    # Record ground truth expectations for transparency
    evaluator.add_ground_truth({"expected": GROUND_TRUTH}, gt_type="expected_values")

    # Build verification subtrees
    if extracted.apex_legends is None:
        extracted.apex_legends = ApexRequirements()
    if extracted.titanfall_2 is None:
        extracted.titanfall_2 = TitanfallSpecs()
    if extracted.founder_info is None:
        extracted.founder_info = FounderDetails()
    if extracted.ewc_prize_info is None:
        extracted.ewc_prize_info = EWCPrize()

    await verify_apex_legends(evaluator, root, extracted.apex_legends)
    await verify_titanfall_2(evaluator, root, extracted.titanfall_2)
    await verify_founder(evaluator, root, extracted.founder_info)
    await verify_ewc_prize(evaluator, root, extracted.ewc_prize_info)

    # Return standardized summary
    return evaluator.get_summary()