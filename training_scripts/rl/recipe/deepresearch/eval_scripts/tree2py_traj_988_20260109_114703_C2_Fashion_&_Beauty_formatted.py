import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gots_bcorp_brand"
TASK_DESCRIPTION = """
Identify a fashion clothing brand based in the United States or Canada that holds both Global Organic Textile Standard (GOTS) certification for its organic cotton products and is also a Certified B Corporation. Provide the brand name, its country of operation, and reference URLs that verify both certifications.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BrandCertificationExtraction(BaseModel):
    """
    Structured extraction for a single brand's certification info.
    """
    brand_name: Optional[str] = None
    country: Optional[str] = None
    gots_urls: List[str] = Field(default_factory=list)
    bcorp_urls: List[str] = Field(default_factory=list)
    gots_claim_snippet: Optional[str] = None
    bcorp_claim_snippet: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_brand_certification() -> str:
    return """
    Extract exactly one fashion clothing brand mentioned in the answer that is based in the United States or Canada and is stated to hold BOTH:
    (1) a Global Organic Textile Standard (GOTS) certification (for its organic cotton clothing/products), and
    (2) a Certified B Corporation certification.

    Return the following fields:
    - brand_name: The brand's name as written in the answer.
    - country: The country of operation mentioned for this brand (prefer canonical names "United States" or "Canada" if clearly implied; otherwise copy what the answer states).
    - gots_urls: An array of URLs that the answer provides to verify the GOTS certification for the brand's organic cotton clothing/products. These should be real, publicly accessible URLs (e.g., GOTS public database page, brand's certification page, etc.).
    - bcorp_urls: An array of URLs that the answer provides to verify the brand's Certified B Corporation status (e.g., B Lab Directory listing or official announcement).
    - gots_claim_snippet: Quote or summarize the exact sentence/construction from the answer where it states the brand is GOTS certified for organic cotton clothing/products (if present).
    - bcorp_claim_snippet: Quote or summarize the exact sentence/construction from the answer where it states the brand is a Certified B Corporation (if present).

    Rules:
    1) Extract only what is explicitly present in the answer.
    2) For URL fields, include full valid URLs (prepend http:// or https:// if missing).
    3) If multiple brands are mentioned, extract ONLY the first one that claims both certifications.
    4) If a field is missing in the answer, set it to null (for strings) or an empty list (for urls arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_us_or_canada(country: Optional[str]) -> bool:
    """
    Heuristic check whether a provided country string indicates the United States or Canada.
    """
    if not country:
        return False
    c = country.strip().lower()

    us_aliases = {
        "united states", "united states of america", "usa", "u.s.a", "u.s.", "us", "u.s",
        "america", "american"  # allow some common phrasing
    }
    ca_aliases = {
        "canada", "ca", "canadian"
    }

    # Direct match
    if c in us_aliases or c in ca_aliases:
        return True

    # Substring/phrase-based allowances
    for token in ["united states", "u.s.", "u.s", "usa", "u.s.a", "us"]:
        if token in c:
            return True
    if "canada" in c:
        return True

    return False


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_brand_tree(
    evaluator: Evaluator,
    parent_node,
    extracted: BrandCertificationExtraction
) -> None:
    """
    Build the rubric tree under a critical Brand_Identification node and run verifications.
    """
    # Create the top-level rubric node (critical, parallel as specified)
    brand_node = evaluator.add_parallel(
        id="Brand_Identification",
        desc="Identify one fashion clothing brand based in the United States or Canada that is both GOTS-certified for organic cotton clothing products and a Certified B Corporation, and provide required fields and verification URLs.",
        parent=parent_node,
        critical=True
    )

    # 1) Brand_Name_Provided (critical)
    evaluator.add_custom_node(
        result=bool(extracted.brand_name and extracted.brand_name.strip()),
        id="Brand_Name_Provided",
        desc="Answer provides the brand name.",
        parent=brand_node,
        critical=True
    )

    # 2) Country_of_Operation_Provided (critical)
    evaluator.add_custom_node(
        result=bool(extracted.country and extracted.country.strip()),
        id="Country_of_Operation_Provided",
        desc="Answer provides the brand's country of operation.",
        parent=brand_node,
        critical=True
    )

    # 3) Geographic_Location_Constraint (critical)
    evaluator.add_custom_node(
        result=_is_us_or_canada(extracted.country),
        id="Geographic_Location_Constraint",
        desc="Brand is based in either the United States or Canada (consistent with the provided country of operation).",
        parent=brand_node,
        critical=True
    )

    # 4) GOTS_Certification_Requirement (critical parent, parallel)
    gots_parent = evaluator.add_parallel(
        id="GOTS_Certification_Requirement",
        desc="Brand holds Global Organic Textile Standard (GOTS) certification that applies to its organic cotton clothing products.",
        parent=brand_node,
        critical=True
    )

    # 4.a) GOTS_Certified (critical leaf)
    gots_cert_leaf = evaluator.add_leaf(
        id="GOTS_Certified",
        desc="Answer asserts the brand is GOTS-certified (i.e., holds a valid GOTS certification).",
        parent=gots_parent,
        critical=True
    )
    gots_brand = extracted.brand_name or "the brand"
    gots_cert_claim = (
        f"The brand '{gots_brand}' is certified to the Global Organic Textile Standard (GOTS)."
    )
    await evaluator.verify(
        claim=gots_cert_claim,
        node=gots_cert_leaf,
        sources=extracted.gots_urls,
        additional_instruction=(
            "Use the provided URL(s) (e.g., GOTS Public Database listing, official certification pages) to confirm "
            "that the brand/company holds a valid GOTS certification. Allow minor name variants (case, punctuation). "
            "If no URL is provided, you must answer Incorrect."
        ),
    )

    # 4.b) GOTS_Applies_to_Organic_Cotton_Clothing (critical leaf)
    gots_scope_leaf = evaluator.add_leaf(
        id="GOTS_Applies_to_Organic_Cotton_Clothing",
        desc="Answer indicates the GOTS certification applies to the brand's organic cotton clothing products (not an unrelated material/category).",
        parent=gots_parent,
        critical=True
    )
    gots_scope_claim = (
        f"The GOTS certification for '{gots_brand}' applies to organic cotton clothing/apparel products."
    )
    await evaluator.verify(
        claim=gots_scope_claim,
        node=gots_scope_leaf,
        sources=extracted.gots_urls,
        additional_instruction=(
            "Check the scope/category on the referenced page(s) to ensure it is relevant to organic cotton apparel/clothing "
            "(e.g., certificate categories like 'Apparel' or page text stating 'GOTS-certified organic cotton' for clothing). "
            "If no URL is provided, you must answer Incorrect."
        ),
    )

    # 5) B_Corp_Certification_Requirement (critical leaf)
    bcorp_leaf = evaluator.add_leaf(
        id="B_Corp_Certification_Requirement",
        desc="Brand is a Certified B Corporation.",
        parent=brand_node,
        critical=True
    )
    bcorp_claim = f"The brand '{gots_brand}' is a Certified B Corporation (certified by B Lab)."
    await evaluator.verify(
        claim=bcorp_claim,
        node=bcorp_leaf,
        sources=extracted.bcorp_urls,
        additional_instruction=(
            "Verify that the company/brand is recognized as a Certified B Corporation by B Lab (e.g., via the official "
            "B Corp Directory listing or a reputable source that clearly states 'Certified B Corporation'). "
            "Do NOT confuse 'benefit corporation' (a legal corporate form) with 'Certified B Corporation'. "
            "If no URL is provided, you must answer Incorrect."
        ),
    )

    # 6) Reference_URLs_Verify_Certifications (critical parent, parallel)
    refs_parent = evaluator.add_parallel(
        id="Reference_URLs_Verify_Certifications",
        desc="Answer provides publicly available reference URL(s) that verify both certifications.",
        parent=brand_node,
        critical=True
    )

    # 6.a) Public_URL_Verifies_GOTS (critical leaf)
    url_gots_leaf = evaluator.add_leaf(
        id="Public_URL_Verifies_GOTS",
        desc="At least one publicly accessible URL is provided that supports/verifies the brand's GOTS certification (for organic cotton clothing/products).",
        parent=refs_parent,
        critical=True
    )
    url_gots_claim = (
        f"At least one of the provided URLs supports/verifies that '{gots_brand}' holds a GOTS certification relevant to organic cotton clothing/products."
    )
    await evaluator.verify(
        claim=url_gots_claim,
        node=url_gots_leaf,
        sources=extracted.gots_urls,
        additional_instruction=(
            "Judge based on the content of the provided page(s). If the list of URLs is empty or the pages do not "
            "explicitly support the GOTS certification for the brand, answer Incorrect."
        ),
    )

    # 6.b) Public_URL_Verifies_B_Corp (critical leaf)
    url_bcorp_leaf = evaluator.add_leaf(
        id="Public_URL_Verifies_B_Corp",
        desc="At least one publicly accessible URL is provided that supports/verifies the brand's Certified B Corporation status.",
        parent=refs_parent,
        critical=True
    )
    url_bcorp_claim = (
        f"At least one of the provided URLs supports/verifies that '{gots_brand}' is a Certified B Corporation."
    )
    await evaluator.verify(
        claim=url_bcorp_claim,
        node=url_bcorp_leaf,
        sources=extracted.bcorp_urls,
        additional_instruction=(
            "Judge based on the content of the provided page(s), preferably the official B Lab directory. "
            "If the list of URLs is empty or the pages do not explicitly support the Certified B Corporation status, answer Incorrect."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the GOTS + B Corp brand identification task.
    """
    # Initialize evaluator (root is a non-critical container by design)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall rubric is parallel at the top-level Brand_Identification node
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

    # Extract structured information
    extracted: BrandCertificationExtraction = await evaluator.extract(
        prompt=prompt_extract_brand_certification(),
        template_class=BrandCertificationExtraction,
        extraction_name="brand_certification_extraction"
    )

    # Build verification tree and run checks
    await build_and_verify_brand_tree(evaluator, root, extracted)

    # Return standardized summary
    return evaluator.get_summary()