import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "belgian_sheepdog_akc_recognition"
TASK_DESCRIPTION = "In what year was the Belgian Sheepdog first officially recognized by the American Kennel Club (AKC)? Provide the year and include a reference to an official source to verify this information."
EXPECTED_YEAR = "1912"

# Allowed official/authoritative domains
ALLOWED_OFFICIAL_DOMAINS = [
    "akc.org",      # American Kennel Club official site (includes subdomains)
    "bsca.us",      # Belgian Sheepdog Club of America (AKC parent club)
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AnswerInfo(BaseModel):
    # The year (4-digit) as stated in the answer for AKC recognition of Belgian Sheepdog
    year: Optional[str] = None
    # Breed terms explicitly mentioned in the answer (e.g., Belgian Sheepdog, Groenendael, Belgian Malinois)
    breed_terms: List[str] = Field(default_factory=list)
    # All URLs mentioned in the answer (sources/references)
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_answer_info() -> str:
    return """
    Extract from the answer the following fields related to the AKC recognition of the Belgian Sheepdog:
    1) year: The 4-digit year that the answer claims as the AKC first official recognition year for the Belgian Sheepdog (also known as the Groenendael). If multiple years are mentioned, choose the one explicitly tied to the Belgian Sheepdog's AKC recognition. If the answer is ambiguous or does not provide a clear 4-digit year, return null.
    2) breed_terms: A list of distinct breed names mentioned in the answer, especially among: "Belgian Sheepdog", "Groenendael", "Belgian Malinois", "Belgian Tervuren", "Belgian Laekenois". Include any synonyms or variants used in the answer text.
    3) source_urls: A list of all URLs (links) cited in the answer. Include any format such as raw URLs or markdown links; output the canonical URLs only.

    Do not invent or infer any data; extract only what appears in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def filter_official_urls(urls: List[str]) -> List[str]:
    """Return subset of urls that are from official/authoritative domains."""
    official = []
    for u in urls:
        if not u:
            continue
        try:
            parsed = urlparse(u if "://" in u else "http://" + u)
            host = parsed.netloc.lower()
            for dom in ALLOWED_OFFICIAL_DOMAINS:
                if host == dom or host.endswith("." + dom):
                    official.append(u)
                    break
        except Exception:
            continue
    # Deduplicate while preserving order
    seen = set()
    result = []
    for u in official:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, answer_info: AnswerInfo) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """
    # Top-level node (critical, parallel aggregation)
    top = evaluator.add_parallel(
        id="Belgian_Sheepdog_AKC_Recognition",
        desc="Verify the year the Belgian Sheepdog was officially recognized by the American Kennel Club and that proper documentation is provided",
        parent=evaluator.root,
        critical=True,
    )

    # 1) Recognition_Year (leaf; critical)
    recog_year_leaf = evaluator.add_leaf(
        id="Recognition_Year",
        desc="The answer provides the correct year of AKC recognition for the Belgian Sheepdog, which is 1912",
        parent=top,
        critical=True,
    )
    # Verify against the answer text that the stated year is 1912
    # This is a simple check of the answer content (not world verification).
    await evaluator.verify(
        claim="The answer explicitly states that the Belgian Sheepdog (Groenendael) was first officially recognized by the AKC in 1912.",
        node=recog_year_leaf,
        additional_instruction=(
            "Judge purely based on the provided answer text. "
            "Pass only if the answer clearly states the year 1912 for AKC first recognition of the Belgian Sheepdog. "
            "Minor phrasing variations like 'in 1912' are acceptable. "
            "If the answer lists another year or is ambiguous, mark as Incorrect."
        ),
    )

    # 2) Official_Source_Reference (critical group) with two critical leaves:
    #    - presence of an official/authoritative source URL
    #    - the official/authoritative source supports 'recognized in 1912'
    official_group = evaluator.add_parallel(
        id="Official_Source_Reference",
        desc="A verifiable reference to an official AKC source or authoritative breed documentation is provided to support the recognition year",
        parent=top,
        critical=True,
    )

    all_urls = answer_info.source_urls or []
    official_urls = filter_official_urls(all_urls)

    # 2a) Presence of an official/authoritative source URL (custom, critical)
    has_official_src = evaluator.add_custom_node(
        result=len(official_urls) > 0,
        id="official_source_present",
        desc="At least one cited source is an official AKC page (akc.org) or the AKC-recognized Belgian Sheepdog parent club (bsca.us)",
        parent=official_group,
        critical=True,
    )

    # 2b) The official/authoritative source supports the 1912 recognition year (leaf, critical)
    official_supports_year_leaf = evaluator.add_leaf(
        id="official_source_supports_1912",
        desc="An official or authoritative source explicitly supports that the Belgian Sheepdog (Groenendael) was first recognized by AKC in 1912",
        parent=official_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The cited official source explicitly states that the Belgian Sheepdog (Groenendael) was first recognized by the American Kennel Club in 1912.",
        node=official_supports_year_leaf,
        sources=official_urls,  # Verify against AKC/parent-club sources only
        additional_instruction=(
            "Check the provided official/authoritative page(s) (akc.org or the AKC-recognized parent club bsca.us). "
            "Confirm that the page explicitly indicates the AKC first recognition year is 1912 and that it refers to the Belgian Sheepdog "
            "(also called Groenendael), not the Belgian Malinois, Tervuren, or Laekenois. "
            "If the provided URLs are irrelevant, inaccessible, or fail to mention 1912 for Belgian Sheepdog, mark as Not Supported."
        ),
    )

    # 3) Breed_Identification (leaf; critical)
    breed_ident_leaf = evaluator.add_leaf(
        id="Breed_Identification",
        desc="The answer specifically refers to the 'Belgian Sheepdog' or 'Groenendael', not other Belgian herding breeds that were recognized separately",
        parent=top,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The recognition year stated in the answer clearly pertains to the Belgian Sheepdog breed "
            "(also known as the Groenendael), not to Belgian Malinois, Belgian Tervuren, or Belgian Laekenois."
        ),
        node=breed_ident_leaf,
        additional_instruction=(
            "Judge by reading the answer text. Pass only if the year claim is explicitly tied to 'Belgian Sheepdog' "
            "or 'Groenendael'. If the answer conflates the Belgian varieties or ties the year to a different Belgian breed, mark Incorrect."
        ),
    )

    # Record some custom info for transparency
    evaluator.add_ground_truth({"expected_year": EXPECTED_YEAR}, gt_type="ground_truth_expected")
    evaluator.add_custom_info(
        info={"all_cited_urls": all_urls, "official_urls_used": official_urls},
        info_type="debug_info",
        info_name="source_selection_details"
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
    Evaluate an answer for the Belgian Sheepdog AKC recognition year task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract answer info
    extracted = await evaluator.extract(
        prompt=prompt_extract_answer_info(),
        template_class=AnswerInfo,
        extraction_name="answer_info",
    )

    # Build tree and run verifications
    await build_verification_tree(evaluator, extracted)

    return evaluator.get_summary()