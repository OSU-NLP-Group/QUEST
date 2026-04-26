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
TASK_ID = "pulitzer_2025_breaking_news_photography_press_freedom"
TASK_DESCRIPTION = """
Identify the photographer who won the 2025 Pulitzer Prize for Breaking News Photography and provide the name of their affiliated news organization. Then, determine the ranking position of the country where this news organization is primarily based in the 2025 RSF (Reporters Without Borders) World Press Freedom Index. Provide reference URLs from official sources for both the Pulitzer Prize announcement and the RSF World Press Freedom Index.
""".strip()

RSF_YEAR = 2025

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PulitzerPressFreedomExtraction(BaseModel):
    """
    Structured information extracted from the agent's answer for:
    - Pulitzer: winner's photographer name, affiliated organization, and pulitzer.org URL(s)
    - RSF Index: country and its 2025 ranking with rsf.org URL(s)
    - Optional: any URL(s) the answer cites to support the organization's home country
    """
    photographer_name: Optional[str] = None
    organization_name: Optional[str] = None
    pulitzer_urls: List[str] = Field(default_factory=list)

    country: Optional[str] = None
    rsf_ranking: Optional[str] = None
    rsf_urls: List[str] = Field(default_factory=list)

    # Optional helper to verify organization's primary country (e.g., org About page or Wikipedia)
    org_country_source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return f"""
    Extract the following fields exactly as they appear in the provided answer. Do not invent or infer.

    Required fields:
    1) photographer_name: The name of the photographer who won the 2025 Pulitzer Prize for "Breaking News Photography".
    2) organization_name: The name of the photographer's affiliated news organization (as presented in the answer).
    3) pulitzer_urls: An array of all URLs from pulitzer.org that the answer cites as evidence for the 2025 Breaking News Photography winner.
    4) country: The country where the news organization is primarily based (as stated in the answer).
    5) rsf_ranking: The 2025 RSF World Press Freedom Index ranking position for the identified country (e.g., "58", "58th", or "58/180"). Keep as a string.
    6) rsf_urls: An array of all URLs from rsf.org that the answer cites for the 2025 World Press Freedom Index (the country page or ranking evidence).
    7) org_country_source_urls: Any URL(s) (official organization page or reliable reference like Wikipedia) cited in the answer that support the organization's home country. If none are cited, return an empty array.

    Special rules for URL extraction:
    - Only extract URLs explicitly present in the answer.
    - Include full URLs with protocol.
    - Accept URLs in plain form or embedded in markdown links.
    - For pulitzer_urls, include only URLs on pulitzer.org if present.
    - For rsf_urls, include only URLs on rsf.org if present.

    If any field is missing in the answer, set it to null (for strings) or [] (for arrays).
    """.strip()


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_pulitzer_section(
    evaluator: Evaluator,
    parent_node,
    data: PulitzerPressFreedomExtraction,
) -> None:
    """
    Build and verify the Pulitzer identification section:
    - Pulitzer_Reference_URL (critical)
    - Photographer_Name (critical)
    - Organization_Name (critical)
    """
    node = evaluator.add_parallel(
        id="Photographer_and_Organization_Identification",
        desc="Correctly identify both the photographer who won the 2025 Pulitzer Prize for Breaking News Photography and their affiliated news organization",
        parent=parent_node,
        critical=True,
    )

    # 1) Pulitzer Reference URL
    pulitzer_ref_node = evaluator.add_leaf(
        id="Pulitzer_Reference_URL",
        desc="Provide a reference URL from pulitzer.org that confirms the 2025 Breaking News Photography winner",
        parent=node,
        critical=True,
    )
    claim_pulitzer_ref = (
        "This page is on pulitzer.org and confirms the winner of the 2025 Pulitzer Prize for Breaking News Photography."
    )
    await evaluator.verify(
        claim=claim_pulitzer_ref,
        node=pulitzer_ref_node,
        sources=data.pulitzer_urls,
        additional_instruction="Ensure the page clearly indicates the 2025 'Breaking News Photography' winner. If the URL is not on pulitzer.org, this should be marked incorrect."
    )

    # 2) Photographer Name
    photographer_node = evaluator.add_leaf(
        id="Photographer_Name",
        desc="Correctly identify the name of the photographer who won the 2025 Pulitzer Prize for Breaking News Photography",
        parent=node,
        critical=True,
    )
    photographer_name = data.photographer_name or ""
    claim_photographer = (
        f"On this pulitzer.org page, the winner of the 2025 Pulitzer Prize for Breaking News Photography is {photographer_name}."
    )
    await evaluator.verify(
        claim=claim_photographer,
        node=photographer_node,
        sources=data.pulitzer_urls,
        additional_instruction="Allow minor variants (middle initials, diacritics). It must explicitly state this person is the winner (not just a finalist) for the 2025 'Breaking News Photography' category."
    )

    # 3) Organization Name
    organization_node = evaluator.add_leaf(
        id="Organization_Name",
        desc="Correctly identify the news organization with which the winning photographer is affiliated",
        parent=node,
        critical=True,
    )
    organization_name = data.organization_name or ""
    claim_org = (
        f"On this pulitzer.org page, the winner's affiliated news organization is {organization_name}."
    )
    await evaluator.verify(
        claim=claim_org,
        node=organization_node,
        sources=data.pulitzer_urls,
        additional_instruction="Look for the affiliation shown with the winner on the page. Allow common abbreviations (e.g., 'AP' vs 'Associated Press') if clearly equivalent."
    )


async def verify_press_freedom_section(
    evaluator: Evaluator,
    parent_node,
    data: PulitzerPressFreedomExtraction,
) -> None:
    """
    Build and verify the RSF press freedom ranking section:
    - Country_Identification (critical)
    - Ranking_Position (critical)
    - RSF_Reference_URL (critical)
    """
    node = evaluator.add_parallel(
        id="Press_Freedom_Ranking",
        desc="Correctly identify the country where the organization is based and provide its ranking in the 2025 RSF World Press Freedom Index",
        parent=parent_node,
        critical=True,
    )

    # 1) Country Identification (verify organization's primary base country)
    country_node = evaluator.add_leaf(
        id="Country_Identification",
        desc="Correctly identify the country where the news organization is primarily based",
        parent=node,
        critical=True,
    )
    org_name = data.organization_name or "the identified news organization"
    country_name = data.country or ""
    claim_country = f"The news organization {org_name} is primarily based in {country_name}."
    # Prefer explicit country-supporting URLs if provided; otherwise, proceed without URLs (simple verify)
    org_country_sources = data.org_country_source_urls if data.org_country_source_urls else None
    await evaluator.verify(
        claim=claim_country,
        node=country_node,
        sources=org_country_sources,
        additional_instruction="Verify the organization's primary headquarters/base country using an official source (e.g., About/Contact page) or a reliable reference (e.g., Wikipedia). If multiple locations exist, prefer the HQ country."
    )

    # 2) RSF Reference URL (verify page is on rsf.org and is for 2025)
    rsf_ref_node = evaluator.add_leaf(
        id="RSF_Reference_URL",
        desc="Provide a reference URL from rsf.org that confirms the 2025 World Press Freedom Index ranking for the identified country",
        parent=node,
        critical=True,
    )
    claim_rsf_ref = f"This page is on rsf.org and shows the {RSF_YEAR} World Press Freedom Index ranking for {country_name}."
    await evaluator.verify(
        claim=claim_rsf_ref,
        node=rsf_ref_node,
        sources=data.rsf_urls,
        additional_instruction=f"The page must clearly indicate the {RSF_YEAR} World Press Freedom Index and be relevant to the specified country. If not on rsf.org, mark as incorrect."
    )

    # 3) Ranking Position (verify the numeric ranking for 2025)
    ranking_node = evaluator.add_leaf(
        id="Ranking_Position",
        desc="Correctly provide the ranking position of the identified country in the 2025 RSF World Press Freedom Index",
        parent=node,
        critical=True,
    )
    ranking_value = data.rsf_ranking or ""
    claim_ranking = (
        f"In the {RSF_YEAR} RSF World Press Freedom Index, {country_name} is ranked {ranking_value}."
    )
    await evaluator.verify(
        claim=claim_ranking,
        node=ranking_node,
        sources=data.rsf_urls,
        additional_instruction="Accept equivalent formats, e.g., '58', '58th', or '58/180'. Ensure the ranking pertains to 2025 for the specified country."
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
    Evaluate an answer for the 2025 Pulitzer Breaking News Photography + Press Freedom task.
    Returns a structured summary with the verification tree and final score.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Use a synthetic non-critical root
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

    # Record RSF year metadata
    evaluator.add_custom_info({"rsf_year": RSF_YEAR}, info_type="meta", info_name="parameters")

    # 1) Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=PulitzerPressFreedomExtraction,
        extraction_name="pulitzer_press_freedom_extraction",
    )

    # 2) Build the rubric tree as per the JSON: a critical sequential node with two critical parallel children
    task_node = evaluator.add_sequential(
        id="2025_Pulitzer_Breaking_News_Photography_and_Press_Freedom",
        desc="Complete identification of the 2025 Pulitzer Prize for Breaking News Photography winner and their organization's home country press freedom ranking",
        parent=root,
        critical=True,
    )

    # 2.1) Pulitzer identification section (critical, parallel)
    await verify_pulitzer_section(evaluator, task_node, extracted)

    # 2.2) RSF press freedom ranking section (critical, parallel)
    # Note: Because the parent is sequential and critical, if the previous section fails,
    # subsequent leaves here will be auto-skipped during verification (precondition logic).
    await verify_press_freedom_section(evaluator, task_node, extracted)

    # 3) Return the evaluation summary
    return evaluator.get_summary()