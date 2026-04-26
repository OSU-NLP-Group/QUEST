import asyncio
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "celebrity_fragrance_brand_2014_2018"
TASK_DESCRIPTION = (
    "Among the following celebrities—Sarah Jessica Parker, Sofia Vergara, Emily Ratajkowski, and Gwyneth Paltrow—"
    "identify the actress who founded and launched her own fragrance (perfume) brand with its first product released "
    "between 2014 and 2018, inclusive. Provide the following information: (1) The celebrity's full name, "
    "(2) The fragrance brand name, (3) The exact year the fragrance brand first launched, and (4) At least one reference "
    "URL that verifies this information. Note: The celebrity must have founded or launched the brand themselves, not "
    "merely served as a brand ambassador or spokesperson for another company's fragrance line."
)

ALLOWED_CELEBRITIES = [
    "Sarah Jessica Parker",
    "Sofia Vergara",
    "Emily Ratajkowski",
    "Gwyneth Paltrow",
]

YEAR_RANGE_MIN = 2014
YEAR_RANGE_MAX = 2018


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class CelebrityBrandInfo(BaseModel):
    celebrity_full_name: Optional[str] = None
    brand_name: Optional[str] = None
    launch_year: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_celebrity_brand() -> str:
    return """
    From the answer, extract exactly one celebrity-brand entry that the answer presents as the final result.
    If the answer mentions multiple possible celebrities/brands, select only the first one that the answer claims or implies is the correct result for the task.
    Return the following fields:
    - celebrity_full_name: the full name of the celebrity chosen in the answer (string; use exactly what the answer states).
    - brand_name: the name of the fragrance brand the answer associates with that celebrity (string).
    - launch_year: the year (as a 4-digit string) that the brand's FIRST product launched, according to the answer (e.g., "2014"). If a range or phrase is given, extract the single 4-digit year the answer asserts as the first launch year.
    - reference_urls: an array of all URLs the answer provides to support this claim (absolutely only extract URLs actually present in the answer; include Google links, news articles, official pages, etc.). If none are present, return an empty array.
    If any required field is missing in the answer, set it to null (or [] for reference_urls).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_name(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()


def _is_allowed_celebrity(name: Optional[str]) -> bool:
    norm = _normalize_name(name)
    for allowed in ALLOWED_CELEBRITIES:
        if _normalize_name(allowed) == norm:
            return True
    return False


def _extract_first_year_number(year_text: Optional[str]) -> Optional[int]:
    if not year_text:
        return None
    m = re.search(r"(19|20)\d{2}", year_text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_nodes(
    evaluator: Evaluator,
    parent_node,
    info: CelebrityBrandInfo
) -> None:
    """
    Build the verification tree under the critical aggregate node and run verifications.
    """
    # Existence checks (critical; run first so later leaves can be auto-skipped if they fail)
    evaluator.add_custom_node(
        result=bool(info.celebrity_full_name and info.celebrity_full_name.strip()),
        id="Celebrity_Full_Name_Provided",
        desc="The celebrity's full name is clearly stated in the answer",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.brand_name and info.brand_name.strip()),
        id="Brand_Name_Provided",
        desc="The specific name of the fragrance brand is provided",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.launch_year and info.launch_year.strip()),
        id="Launch_Year_Stated",
        desc="The exact year when the fragrance brand first launched is stated",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.reference_urls and len([u for u in info.reference_urls if isinstance(u, str) and u.strip()]) > 0),
        id="Reference_URL_Provided",
        desc="At least one verifiable reference URL that supports the provided information is included",
        parent=parent_node,
        critical=True
    )

    # Celebrity must be from provided list (critical)
    evaluator.add_custom_node(
        result=_is_allowed_celebrity(info.celebrity_full_name),
        id="Celebrity_from_Provided_List",
        desc="The identified celebrity is one of the following: Sarah Jessica Parker, Sofia Vergara, Emily Ratajkowski, or Gwyneth Paltrow",
        parent=parent_node,
        critical=True
    )

    # Actress qualification (critical) - verify with provided sources
    actress_node = evaluator.add_leaf(
        id="Actress_Qualification",
        desc="The identified celebrity is an actress by profession",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{info.celebrity_full_name} is an actress (or actor) by profession.",
        node=actress_node,
        sources=info.reference_urls,
        additional_instruction=(
            "Confirm that the person is identified as an 'actress' or 'actor' in the provided source(s). "
            "Variants such as 'actress and model' or 'actor and businesswoman' count as being an actress. "
            "If none of the sources identify her as an actress/actor, mark as not supported."
        )
    )

    # Founder/launcher role (critical) - verify with sources that it's her own brand, not an ambassador role
    founder_node = evaluator.add_leaf(
        id="Brand_Founder_Role",
        desc="The celebrity founded or launched their own fragrance brand, not merely served as a brand ambassador or spokesperson",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{info.celebrity_full_name} founded or launched her own fragrance brand named '{info.brand_name}', not merely serving as an ambassador or spokesperson.",
        node=founder_node,
        sources=info.reference_urls,
        additional_instruction=(
            "Verify that the source explicitly states the celebrity founded or launched her own fragrance brand "
            "(e.g., 'launched her own fragrance line', 'introduced her own perfume brand', 'founded brand X'). "
            "If the language indicates only an ambassador/spokesperson/endorsement role for another company's product, this is NOT acceptable."
        )
    )

    # Product category must be fragrance/perfume specifically (critical)
    category_node = evaluator.add_leaf(
        id="Product_Category_Fragrance",
        desc="The brand is specifically a fragrance/perfume line, not other beauty categories such as skincare, makeup, or fashion",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The brand '{info.brand_name}' associated with {info.celebrity_full_name} is specifically a fragrance/perfume brand.",
        node=category_node,
        sources=info.reference_urls,
        additional_instruction=(
            "Confirm that the brand is specifically a fragrance/perfume line (e.g., perfume, eau de parfum, eau de toilette). "
            "If the brand is primarily another category (fashion, general beauty, skincare, or makeup) and not specifically a fragrance/perfume line, mark as not supported."
        )
    )

    # Launch year in range and supported (critical)
    launch_year_claim_node = evaluator.add_leaf(
        id="Launch_Year_Range_2014_2018",
        desc="The fragrance brand's first product launched between 2014 and 2018, inclusive",
        parent=parent_node,
        critical=True
    )
    # Compose claim asserting both the year and the in-range constraint
    year_str = info.launch_year or ""
    await evaluator.verify(
        claim=(
            f"The first product of the fragrance brand '{info.brand_name}' launched in {year_str}, "
            f"which is between {YEAR_RANGE_MIN} and {YEAR_RANGE_MAX}, inclusive."
        ),
        node=launch_year_claim_node,
        sources=info.reference_urls,
        additional_instruction=(
            f"Verify that the source(s) explicitly support the FIRST launch year for this fragrance brand or its debut product "
            f"(e.g., 'launched', 'debuted', 'introduced', 'first released'). "
            f"The year must be between {YEAR_RANGE_MIN} and {YEAR_RANGE_MAX}, inclusive. "
            f"If the source suggests a different year outside this range or does not clearly refer to the FIRST launch, mark as not supported."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the celebrity fragrance brand identification task (2014–2018).
    """
    # Initialize evaluator with a parallel root
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

    # Extract core fields from the answer
    extracted: CelebrityBrandInfo = await evaluator.extract(
        prompt=prompt_extract_celebrity_brand(),
        template_class=CelebrityBrandInfo,
        extraction_name="celebrity_fragrance_brand_extraction"
    )

    # Add a critical aggregate node under root to mirror the rubric structure
    agg_node = evaluator.add_parallel(
        id="Celebrity_Fragrance_Brand_Identification",
        desc="Evaluates whether the correct celebrity from the provided list who launched a fragrance brand between 2014-2018 is identified with all required details",
        parent=root,
        critical=True
    )

    # Build and verify all rubric leaves
    await build_and_verify_nodes(evaluator, agg_node, extracted)

    # Optionally record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "allowed_celebrities": ALLOWED_CELEBRITIES,
            "extracted": extracted.dict()
        },
        info_type="debug",
        info_name="extraction_debug_info"
    )

    # Return evaluation summary
    return evaluator.get_summary()