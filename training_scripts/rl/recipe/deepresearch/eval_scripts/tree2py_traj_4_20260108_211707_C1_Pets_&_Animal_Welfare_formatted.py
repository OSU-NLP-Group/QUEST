import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "br_municipal_shelter_18plus"
TASK_DESCRIPTION = (
    "I am planning to adopt a pet in Baton Rouge, Louisiana, and want to find the official municipal animal shelter "
    "operated by the city or parish government (not a private rescue or shelter). Please identify this government-"
    "operated animal shelter and verify that it requires adopters to be at least 18 years old. Provide the shelter's "
    "official name and a reference URL to its official website or government page."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ShelterExtraction(BaseModel):
    """
    Extracted identifying info from the answer.
    - shelter_name: The official name of the municipal/government-operated shelter identified.
    - official_urls: All official/government URLs cited in the answer for the shelter.
    """
    shelter_name: Optional[str] = None
    official_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_shelter_info() -> str:
    return """
    Extract the identifying information for the official municipal/government-operated animal shelter in Baton Rouge, Louisiana,
    as presented in the answer.

    Return a JSON object with:
    - shelter_name: The official shelter name stated in the answer. This must refer to a government-operated municipal/parish shelter,
      not a private rescue or non-profit organization.
    - official_urls: An array of all URLs in the answer that are official government pages for the identified shelter.
      Guidelines for official URLs:
        • Prefer .gov domains or the City-Parish’s official domains (e.g., brla.gov or similar official city/parish sites).
        • Include only official government web pages for the identified shelter or division (e.g., Animal Control & Rescue Center).
        • Do not include private/non-profit rescue sites, third-party directories, social media pages, or news articles.

    If the answer contains multiple URLs, include all that meet the criteria above.
    If no official government URLs are present, return an empty array for official_urls.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    """Deduplicate while preserving order and filter out empty strings."""
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree_and_verify(evaluator: Evaluator, extracted: ShelterExtraction) -> None:
    """
    Build the verification tree according to the rubric JSON and perform verifications.
    The root is initialized as SEQUENTIAL in evaluator.initialize(). We add two critical children:
      1) Shelter_Identification_Output (parallel, critical)
      2) Constraint_Verification (parallel, critical)
    """

    root = evaluator.root  # Root already initialized with SEQUENTIAL aggregation

    # Prepare extracted values
    shelter_name = (extracted.shelter_name or "").strip()
    urls_to_use = _dedup_urls(extracted.official_urls or [])

    # ------------------------------------------------------------------- #
    # Node 1: Shelter_Identification_Output (critical, parallel)          #
    # ------------------------------------------------------------------- #
    node_ident_output = evaluator.add_parallel(
        id="Shelter_Identification_Output",
        desc="Provide the shelter’s identifying information required by the question.",
        parent=root,
        critical=True
    )

    # Leaf: Official_Shelter_Name_Provided (critical)
    name_provided = bool(shelter_name)
    evaluator.add_custom_node(
        result=name_provided,
        id="Official_Shelter_Name_Provided",
        desc="The answer states the official name of the identified government-operated municipal animal shelter.",
        parent=node_ident_output,
        critical=True
    )

    # Leaf: Official_Reference_URL_Provided (critical)
    # We verify this via simple verification against the answer text:
    # Claim: The answer includes at least one official government URL for the identified shelter.
    url_leaf = evaluator.add_leaf(
        id="Official_Reference_URL_Provided",
        desc="The answer includes a reference URL that is an official website or government page for the identified shelter (not solely a third-party/private site).",
        parent=node_ident_output,
        critical=True
    )

    urls_preview = ", ".join(urls_to_use) if urls_to_use else "(none found)"
    claim_official_url = (
        f"The answer includes at least one official government website URL for the identified shelter"
        f"{f' (shelter name: \"{shelter_name}\")' if shelter_name else ''}. "
        f"Extracted official URLs: {urls_preview}."
    )

    await evaluator.verify(
        claim=claim_official_url,
        node=url_leaf,
        # No sources: we want the judge to look at the answer itself for presence/officialness indication.
        sources=None,
        additional_instruction=(
            "Judge based on the answer text and the extracted URL(s). Mark as Incorrect if there is no URL in the answer "
            "or if none of the URLs appears to be an official government website/page for the municipal shelter. "
            "Official typically means a city/parish .gov site (e.g., brla.gov) or another clearly official City‑Parish page. "
            "Do NOT accept private non-profit or rescue websites (e.g., .org) or third‑party directories/social media as official."
        )
    )

    # ------------------------------------------------------------------- #
    # Node 2: Constraint_Verification (critical, parallel)                #
    # ------------------------------------------------------------------- #
    node_constraints = evaluator.add_parallel(
        id="Constraint_Verification",
        desc="Verify the identified shelter satisfies all stated constraints.",
        parent=root,
        critical=True
    )

    # Leaf: Location_Verification (critical)
    loc_leaf = evaluator.add_leaf(
        id="Location_Verification",
        desc="Verify the identified shelter is located in Baton Rouge, Louisiana.",
        parent=node_constraints,
        critical=True
    )
    claim_location = (
        f"The identified shelter{f' \"{shelter_name}\"' if shelter_name else ''} is located in Baton Rouge, Louisiana "
        f"(City of Baton Rouge / Parish of East Baton Rouge)."
    )
    await evaluator.verify(
        claim=claim_location,
        node=loc_leaf,
        sources=urls_to_use,
        additional_instruction=(
            "Use the provided official page(s). Look for address or jurisdiction statements indicating Baton Rouge, LA or "
            "East Baton Rouge Parish. Evidence can include an address in Baton Rouge, an official city/parish header, or "
            "explicit mention of jurisdiction. If the page does not clearly indicate Baton Rouge/East Baton Rouge Parish, mark as not supported."
        )
    )

    # Leaf: Government_Operation_Verification (critical)
    gov_leaf = evaluator.add_leaf(
        id="Government_Operation_Verification",
        desc="Verify the identified shelter is operated by city/parish/municipal government (not a private rescue/shelter).",
        parent=node_constraints,
        critical=True
    )
    claim_gov_run = (
        f"The identified shelter{f' \"{shelter_name}\"' if shelter_name else ''} is operated by the city/parish government "
        f"(City of Baton Rouge / Parish of East Baton Rouge) rather than a private rescue or non‑profit organization."
    )
    await evaluator.verify(
        claim=claim_gov_run,
        node=gov_leaf,
        sources=urls_to_use,
        additional_instruction=(
            "Rely on the official page(s). Accept if the shelter is presented as a city/parish division/department "
            "(e.g., Animal Control & Rescue Center under a government department). Do NOT accept if it is described as "
            "a private/non-profit rescue, contractor, or partner unless the page clearly states the shelter itself is a city/parish entity. "
            "If ambiguous, mark as not supported."
        )
    )

    # Leaf: Minimum_Adoption_Age_Requirement (critical)
    age_leaf = evaluator.add_leaf(
        id="Minimum_Adoption_Age_Requirement",
        desc="Verify the shelter requires adopters to be at least 18 years old.",
        parent=node_constraints,
        critical=True
    )
    claim_age = (
        "The shelter requires adopters to be at least 18 years old (e.g., '18 or older', 'must be 18+')."
    )
    await evaluator.verify(
        claim=claim_age,
        node=age_leaf,
        sources=urls_to_use,
        additional_instruction=(
            "Use the official page(s) to find adoption policy/requirements. Accept if wording like 'must be 18 or older' "
            "appears. If the requirement is a different minimum age (e.g., 21+) or not stated, mark as not supported."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Baton Rouge municipal shelter identification & 18+ adoption requirement task.
    """
    # Initialize evaluator with SEQUENTIAL root to honor ordering:
    # First identify & provide official info, then verify constraints.
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_shelter_info(),
        template_class=ShelterExtraction,
        extraction_name="shelter_info"
    )

    # Build tree and verify per rubric
    await build_verification_tree_and_verify(evaluator, extracted)

    # Optional: record a quick summary of extracted URLs
    evaluator.add_custom_info(
        info={"shelter_name_extracted": extracted.shelter_name, "official_urls_extracted": extracted.official_urls},
        info_type="extraction_summary",
        info_name="extraction_summary"
    )

    # Return final structured summary
    return evaluator.get_summary()