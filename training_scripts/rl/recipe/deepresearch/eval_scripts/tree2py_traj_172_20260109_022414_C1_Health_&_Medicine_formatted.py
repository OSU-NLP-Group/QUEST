import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nash_hq_city"
TASK_DESCRIPTION = """
What is the headquarters city of the pharmaceutical company that received FDA approval in 2024 for the first treatment specifically indicated for noncirrhotic nonalcoholic steatohepatitis (NASH) with moderate to advanced liver scarring?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NashHQExtraction(BaseModel):
    """
    Extracted information from the agent's answer needed for verification.
    """
    company_name: Optional[str] = None
    headquarters_city: Optional[str] = None
    drug_name: Optional[str] = None
    approval_year: Optional[str] = None
    sources_fda: List[str] = Field(default_factory=list)
    sources_hq: List[str] = Field(default_factory=list)
    other_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_nash_hq() -> str:
    return """
    Extract the specific information from the answer needed to verify the claim:

    Required fields:
    1) company_name: The pharmaceutical company's name that is claimed to have received FDA approval in 2024 for the first treatment specifically indicated for noncirrhotic NASH (also known as MASH) with moderate to advanced liver scarring.
    2) headquarters_city: The headquarters city of the identified company as stated by the answer.
    3) drug_name: The drug name associated with the FDA approval, if mentioned (otherwise null).
    4) approval_year: The approval year mentioned in the answer (as a string, e.g., "2024"; otherwise null).

    URL sources:
    - sources_fda: A list of URLs explicitly provided in the answer that support the FDA approval claim (including the "first treatment" specificity for noncirrhotic NASH/MASH with moderate to advanced liver scarring). Include only URLs that the answer indicates as evidence for the FDA approval part. If none are provided, return an empty list.
    - sources_hq: A list of URLs explicitly provided in the answer that support the company's headquarters city claim. Include only URLs that the answer indicates as evidence for HQ location. If none are provided, return an empty list.
    - other_sources: Any additional URLs mentioned in the answer that do not clearly map to one of the above categories. If none, return an empty list.

    Important rules:
    - Extract only URLs explicitly present in the answer (plain URLs or markdown links). Do not invent URLs.
    - If a field is not mentioned, set it to null (for strings) or an empty list (for URLs).
    - Do not include duplicate URLs; de-duplicate when possible.
    """


# --------------------------------------------------------------------------- #
# Verification assembly functions                                             #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    extraction: NashHQExtraction,
    parent_node
) -> None:
    """
    Build verification nodes according to the rubric and perform verifications.
    """

    # Create the main critical node that aggregates the three critical checks
    main_node = evaluator.add_parallel(
        id="Correct_Headquarters_City",
        desc=("The answer correctly identifies the headquarters city of the pharmaceutical company that received FDA "
              "approval in 2024 for the first treatment specifically indicated for noncirrhotic nonalcoholic "
              "steatohepatitis (NASH) with moderate to advanced liver scarring."),
        parent=parent_node,
        critical=True
    )

    # Prepare values
    company = extraction.company_name or ""
    hq_city = extraction.headquarters_city or ""
    drug = extraction.drug_name or ""
    year = extraction.approval_year or "2024"  # The task specifically mentions 2024

    # FDA approval verification leaf
    fda_leaf = evaluator.add_leaf(
        id="Company_FDA_Approval",
        desc=("The identified pharmaceutical company received FDA approval in 2024 for a drug that is the first "
              "FDA-approved treatment specifically indicated for noncirrhotic NASH with moderate to advanced liver scarring."),
        parent=main_node,
        critical=True
    )
    fda_claim_parts = [
        f"In {year}, {company} received FDA approval",
        "for the first treatment specifically indicated for noncirrhotic NASH (also called MASH)",
        "with moderate to advanced liver scarring (e.g., fibrosis stages F2–F3)"
    ]
    if drug.strip():
        fda_claim_parts.append(f"The drug name is '{drug}'.")
    fda_claim = ". ".join(fda_claim_parts)

    # Prefer sources_fda; if empty, try other_sources; if still empty, verify without sources
    fda_sources: List[str] = extraction.sources_fda or extraction.other_sources or []

    # HQ city verification leaf
    hq_leaf = evaluator.add_leaf(
        id="Headquarters_City_Match",
        desc="The provided city is the actual headquarters city of the identified pharmaceutical company.",
        parent=main_node,
        critical=True
    )
    hq_claim = f"The headquarters city of {company} is {hq_city}."

    # Prefer sources_hq; fallback to other_sources; else no sources
    hq_sources: List[str] = extraction.sources_hq or extraction.other_sources or []

    # Perform the two core verifications (batch to avoid sequential gating effects)
    await evaluator.batch_verify([
        (
            fda_claim,
            fda_sources if fda_sources else None,
            fda_leaf,
            ("Focus on verifying BOTH that the approval year is 2024 and that the product is the FIRST treatment "
             "specifically indicated for noncirrhotic NASH (also known as MASH) with moderate to advanced liver scarring "
             "(commonly described as F2–F3 fibrosis). Accept reasonable synonyms/variants and the renaming of NASH to MASH.")
        ),
        (
            hq_claim,
            hq_sources if hq_sources else None,
            hq_leaf,
            ("Verify explicitly that the page states the headquarters city for the company. Allow typical corporate phrasing "
             "like 'Headquarters' or an address that clearly indicates the city. Minor formatting differences are okay, "
             "but the city name should clearly match.")
        ),
    ])

    # Official URL references check (critical)
    # We require evidence for BOTH:
    # (a) FDA approval/first-indication claim from an OFFICIAL or authoritative source (e.g., fda.gov, company's official site, sec.gov).
    # (b) Headquarters city claim from an OFFICIAL or authoritative source (e.g., company's official website, sec.gov).
    has_official_fda = False
    has_official_hq = False

    if extraction.sources_fda:
        official_fda_claim = (
            f"This webpage is an official or authoritative source (e.g., fda.gov, the company's official website, or sec.gov) "
            f"and it clearly supports that in {year} {company} received FDA approval for the first treatment specifically indicated "
            f"for noncirrhotic NASH (MASH) with moderate to advanced liver scarring."
        )
        has_official_fda = await evaluator.verify(
            claim=official_fda_claim,
            node=None,  # standalone verification; result captured below via custom node
            sources=extraction.sources_fda,
            additional_instruction=("Treat a source as OFFICIAL if its domain is fda.gov, sec.gov, or the company's official domain "
                                   "(press releases or key pages on the company's website). The page must also substantively support the claim.")
        )

    if extraction.sources_hq:
        official_hq_claim = (
            f"This webpage is an official or authoritative source (e.g., the company's official website or sec.gov) and it "
            f"explicitly states that the headquarters city of {company} is {hq_city}."
        )
        has_official_hq = await evaluator.verify(
            claim=official_hq_claim,
            node=None,  # standalone verification
            sources=extraction.sources_hq,
            additional_instruction=("Treat a source as OFFICIAL if it is the company's official website (including investor relations) "
                                   "or sec.gov. The page must clearly indicate the HQ city.")
        )

    official_refs_node = evaluator.add_custom_node(
        result=(has_official_fda and has_official_hq),
        id="Official_URL_References",
        desc=("Verifiable URL references from official sources are provided to support both (a) the FDA approval/first-indication "
              "claim and (b) the company's headquarters city."),
        parent=main_node,
        critical=True
    )

    # Record custom info to help debugging
    evaluator.add_custom_info(
        info={
            "company_name": company,
            "headquarters_city": hq_city,
            "drug_name": drug,
            "approval_year": year,
            "counts": {
                "sources_fda_count": len(extraction.sources_fda),
                "sources_hq_count": len(extraction.sources_hq),
                "other_sources_count": len(extraction.other_sources)
            },
            "official_checks": {
                "has_official_fda_source": has_official_fda,
                "has_official_hq_source": has_official_hq
            }
        },
        info_type="debug",
        info_name="verification_context"
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
) -> Dict[str, Any]:
    """
    Evaluate an answer to the NASH headquarters city task.

    Returns a structured summary containing the verification tree and final score.
    """
    # Initialize evaluator
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

    # Extract necessary structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_nash_hq(),
        template_class=NashHQExtraction,
        extraction_name="extracted_nash_hq"
    )

    # Build the verification tree and run checks
    await build_and_verify(evaluator, extraction, root)

    # Return standard summary
    return evaluator.get_summary()