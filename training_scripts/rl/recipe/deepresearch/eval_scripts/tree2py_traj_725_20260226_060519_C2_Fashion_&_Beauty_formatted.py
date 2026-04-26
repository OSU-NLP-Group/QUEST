import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sustainable_cashmere_brand_sep_2022"
TASK_DESCRIPTION = (
    "A fashion industry analyst is researching celebrity-founded sustainable fashion brands for a market report. "
    "They need to identify which celebrity launched a fashion brand in September 2022 that specializes exclusively "
    "in 100% cashmere knitwear. Provide the celebrity's full name and the name of the brand they founded."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FashionBrandExtraction(BaseModel):
    """
    Extracted info structure from the agent's answer.
    """
    celebrity_full_name: Optional[str] = None
    brand_name: Optional[str] = None
    # Month-Year string as presented (e.g., "September 2022", "Sep 2022")
    launch_month_year: Optional[str] = None
    # Short phrase summarizing material specialization (e.g., "100% cashmere knitwear", "pure cashmere")
    material_specialization: Optional[str] = None
    # Roles description for the celebrity with respect to the brand (e.g., "founder and creative director")
    roles_text: Optional[str] = None
    # All URLs explicitly present in the answer that relate to the brand or claims
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_brand_info() -> str:
    return """
    Extract the key information about the celebrity-founded sustainable fashion brand mentioned in the answer.

    Return a JSON object with the following fields:
    - celebrity_full_name: The celebrity's full name, exactly as presented in the answer.
    - brand_name: The name of the brand they founded.
    - launch_month_year: The brand's stated launch timing (month and year) in the format as presented in the answer (e.g., "September 2022" or "Sep 2022"). If absent, return null.
    - material_specialization: A short phrase summarizing the brand's product/material specialization, focusing on whether it states "100% cashmere", "pure cashmere", or "exclusively cashmere knitwear". If absent, return null.
    - roles_text: A concise description of the celebrity's role(s) with the brand (e.g., "founder and creative director"). If absent, return null.
    - sources: An array of all URLs explicitly present in the answer that support any of these claims (official brand site, press releases, credible articles, etc.). If none are provided, return an empty array.

    Important:
    - Do not invent any information not present in the answer.
    - Extract URLs exactly as they appear; accept plain URLs or markdown links.
    - If a URL lacks a protocol, prepend "http://".
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _safe(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) else ""


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
    extracted: FashionBrandExtraction,
    logger: logging.Logger,
) -> None:
    """
    Build the verification tree and run all checks according to the rubric.
    """

    # Create the main critical aggregation node
    complete_node = evaluator.add_parallel(
        id="Complete_Answer_Provided",
        desc="The answer provides both the celebrity's full name and the brand name, with all information satisfying the specified constraints",
        parent=root_node,
        critical=True,
    )

    # -------------------- Brand Identification (Critical) -------------------- #
    brand_node = evaluator.add_parallel(
        id="Brand_Identification",
        desc="The brand name is provided and the brand meets the specified material and launch date requirements",
        parent=complete_node,
        critical=True,
    )

    # Existence check: brand name provided
    brand_exists_node = evaluator.add_custom_node(
        result=_nonempty_str(extracted.brand_name),
        id="Brand_Name_Provided",
        desc="The brand name is provided",
        parent=brand_node,
        critical=True,
    )

    # Material specification: exclusively 100% cashmere knitwear
    material_node = evaluator.add_leaf(
        id="Material_Specification",
        desc="The identified brand is composed entirely of 100% cashmere materials",
        parent=brand_node,
        critical=True,
    )
    brand_name = _safe(extracted.brand_name)
    material_claim = (
        f"The brand '{brand_name}' specializes exclusively in 100% cashmere knitwear (i.e., pure cashmere products, "
        f"with no other primary materials)."
    )
    await evaluator.verify(
        claim=material_claim,
        node=material_node,
        sources=extracted.sources,
        additional_instruction=(
            "Verify that the brand explicitly positions itself as 100% cashmere or pure cashmere, and that its core "
            "product offering is exclusively cashmere knitwear. Allow synonymous phrasing such as 'pure cashmere', "
            "'100% cashmere', or 'cashmere-only'. If sources indicate notable non-cashmere product categories, the claim should be considered not supported."
        ),
        extra_prerequisites=[brand_exists_node],
    )

    # Launch date specification: specifically September 2022
    launch_node = evaluator.add_leaf(
        id="Launch_Date_Specification",
        desc="The brand launched specifically in September 2022",
        parent=brand_node,
        critical=True,
    )
    launch_claim = f"The brand '{brand_name}' launched in September 2022."
    await evaluator.verify(
        claim=launch_claim,
        node=launch_node,
        sources=extracted.sources,
        additional_instruction=(
            "Confirm that the brand's debut/launch occurred in September 2022. Accept equivalent phrasing such as "
            "'launched in Sep 2022', 'debuted September 2022', or 'first drop in September 2022'. The month must be "
            "September and the year must be 2022."
        ),
        extra_prerequisites=[brand_exists_node],
    )

    # ----------------- Celebrity Identification (Critical) ------------------- #
    celeb_node = evaluator.add_sequential(
        id="Celebrity_Identification",
        desc="The celebrity's full name is provided and the celebrity is confirmed as the founder and creative director of the identified brand",
        parent=complete_node,
        critical=True,
    )

    # Existence check: celebrity full name provided
    celeb_exists_node = evaluator.add_custom_node(
        result=_nonempty_str(extracted.celebrity_full_name),
        id="Celebrity_Name_Provided",
        desc="The celebrity's full name is provided",
        parent=celeb_node,
        critical=True,
    )

    # Founder and creative director verification
    founder_cd_node = evaluator.add_leaf(
        id="Founder_Status_Verification",
        desc="The provided celebrity is confirmed as both founder and creative director of the brand",
        parent=celeb_node,
        critical=True,
    )
    celeb_name = _safe(extracted.celebrity_full_name)
    founder_cd_claim = (
        f"Celebrity '{celeb_name}' is both the founder and the creative director of the brand '{brand_name}'."
    )
    await evaluator.verify(
        claim=founder_cd_claim,
        node=founder_cd_node,
        sources=extracted.sources,
        additional_instruction=(
            "Explicitly verify BOTH roles (founder and creative director) using the provided sources. Accept reasonable "
            "variants such as 'creative lead' or 'serves as creative director'. If only one role is supported or "
            "neither is supported, the claim should be considered not supported."
        ),
        extra_prerequisites=[brand_exists_node, celeb_exists_node],
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the agent's answer for the September 2022 100% cashmere knitwear brand task.
    """

    # Initialize evaluator with a parallel root (top-level aggregation can be parallel)
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_brand_info(),
        template_class=FashionBrandExtraction,
        extraction_name="celebrity_cashmere_brand_extraction",
    )

    # Build the verification tree and run checks
    await build_verification_tree(evaluator, root, extracted, logger)

    # Return summary
    return evaluator.get_summary()