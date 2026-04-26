import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "news_orgs_constraints_2020_2024_pulitzer_subscription_international_daily_reach"
TASK_DESCRIPTION = """
Identify three major U.S. news organizations that meet all of the following criteria:

1. Headquarters Location: The organization must be headquartered in one of the following U.S. cities: New York City, Washington D.C., Los Angeles, or Chicago.
2. Pulitzer Prize Recognition: The organization must have won at least one Pulitzer Prize in any journalism category between 2020 and 2024 (inclusive).
3. Subscription Model: The organization must operate a digital paywall or subscription-based model for accessing online news content.
4. International Presence: The organization must maintain international news bureaus or employ foreign correspondents.
5. Publication Frequency: The organization must publish news content on a daily basis.
6. Circulation/Reach: The organization must be either among the top 25 U.S. newspapers by circulation OR among the top 50 global news websites by monthly visitors.

For each of the three organizations you identify, provide the following information:

- Organization Name
- Headquarters Address (specific)
- Pulitzer Prize information (year and category, 2020–2024)
- Subscription details (paywall/subscription model)
- International operations (examples of bureaus or foreign correspondents)
- Circulation/readership data demonstrating reach threshold
- Reference URLs for each of the above information items
"""

ALLOWED_CITIES = [
    "New York City",
    "New York",
    "NYC",
    "Washington, D.C.",
    "Washington DC",
    "Washington",
    "Los Angeles",
    "LA",
    "Chicago"
]

PULITZER_YEAR_MIN = 2020
PULITZER_YEAR_MAX = 2024


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class PulitzerInfo(BaseModel):
    year: Optional[str] = None
    category: Optional[str] = None
    award_title: Optional[str] = None


class OrgReferenceURLs(BaseModel):
    hq_urls: List[str] = Field(default_factory=list)
    pulitzer_urls: List[str] = Field(default_factory=list)
    subscription_urls: List[str] = Field(default_factory=list)
    international_urls: List[str] = Field(default_factory=list)
    daily_publication_urls: List[str] = Field(default_factory=list)
    reach_urls: List[str] = Field(default_factory=list)


class OrganizationInfo(BaseModel):
    name: Optional[str] = None
    headquarters_address: Optional[str] = None
    headquarters_city: Optional[str] = None
    pulitzer: Optional[PulitzerInfo] = None
    subscription_details: Optional[str] = None
    international_operations: Optional[str] = None
    daily_publication_info: Optional[str] = None
    reach_data: Optional[str] = None
    references: Optional[OrgReferenceURLs] = None


class NewsOrganizationsExtraction(BaseModel):
    organizations: List[OrganizationInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_news_orgs() -> str:
    return """
    Extract up to three U.S. news organizations explicitly mentioned in the answer that are proposed to satisfy the task constraints.
    For each organization, extract the following fields exactly as stated in the answer text:

    1. name: The official organization name.
    2. headquarters_address: A specific street address of the headquarters, if provided.
    3. headquarters_city: The city of the headquarters as explicitly stated (e.g., New York City, Washington D.C., Los Angeles, Chicago).
    4. pulitzer:
       - year: The year of a Pulitzer Prize win explicitly mentioned (must be 2020–2024).
       - category: The category of the Pulitzer Prize win (e.g., Public Service, Investigative Reporting).
       - award_title: If an award/citation title or story name is mentioned, include it; else null.
    5. subscription_details: A description of the digital paywall or subscription model (e.g., metered paywall, hard paywall, pricing tiers).
    6. international_operations: Examples or statements evidencing international bureaus or foreign correspondents (e.g., bureau locations or named roles).
    7. daily_publication_info: The statement or description indicating the organization publishes news daily.
    8. reach_data: Circulation numbers or monthly visitor statistics, or ranking statements demonstrating the reach threshold (top 25 U.S. newspapers by circulation OR top 50 global news websites by monthly visitors).
    9. references: For each item above, extract all URLs provided in the answer that support that specific item. A URL may be a plain link or a markdown link; extract the raw URL.
       - hq_urls: URLs supporting headquarters address/city
       - pulitzer_urls: URLs supporting Pulitzer win info (year and category)
       - subscription_urls: URLs supporting paywall/subscription model
       - international_urls: URLs supporting international bureaus/correspondents
       - daily_publication_urls: URLs supporting daily publication claim
       - reach_urls: URLs supporting circulation/visitor ranking/statistics

    Rules:
    - Extract only what appears in the answer text. Do not invent or infer missing pieces.
    - If any required information is missing, set the field to null (or empty list for URLs).
    - Return at most three organizations in an array field 'organizations'. If the answer mentions more than three, include the first three.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def safe_str(s: Optional[str]) -> str:
    return s or ""


def has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0 and any(u.strip() for u in urls)


# --------------------------------------------------------------------------- #
# Verification for one organization                                           #
# --------------------------------------------------------------------------- #
async def verify_organization(
    evaluator: Evaluator,
    parent_node,
    org: OrganizationInfo,
    org_index: int,
) -> None:
    """
    Build verification subtree for a single organization and run checks.
    """
    display_idx = org_index + 1
    org_node = evaluator.add_parallel(
        id=f"organization_{display_idx}",
        desc=f"Organization {display_idx}: meets all criteria and provides all required fields with sources",
        parent=parent_node,
        critical=False
    )

    # Name provided (existence)
    name_exists = org is not None and org.name is not None and org.name.strip() != ""
    evaluator.add_custom_node(
        result=name_exists,
        id=f"org{display_idx}_name",
        desc="Provides the official organization name",
        parent=org_node,
        critical=True
    )

    # Headquarters: verify address and allowed city with sources
    hq_leaf = evaluator.add_leaf(
        id=f"org{display_idx}_headquarters",
        desc="Provides a specific headquarters address AND headquarters is in one of: New York City, Washington D.C., Los Angeles, Chicago",
        parent=org_node,
        critical=True
    )
    hq_addr = safe_str(org.headquarters_address)
    hq_city = safe_str(org.headquarters_city)
    hq_claim = (
        f"The headquarters of {safe_str(org.name)} is located at '{hq_addr}' in '{hq_city}', "
        f"and the city is one of the allowed cities: {', '.join(ALLOWED_CITIES)}."
    )
    hq_sources = org.references.hq_urls if org.references else []
    await evaluator.verify(
        claim=hq_claim,
        node=hq_leaf,
        sources=hq_sources,
        additional_instruction=(
            "Verify that the provided URL(s) explicitly indicate the organization's headquarters address or location. "
            "Additionally, confirm that the headquarters city is one of the allowed cities (New York City, Washington D.C., Los Angeles, Chicago). "
            "Minor formatting differences are acceptable, but the city must clearly match one of the allowed options."
        )
    )

    # Pulitzer Prize between 2020–2024 with year and category
    pul_leaf = evaluator.add_leaf(
        id=f"org{display_idx}_pulitzer",
        desc="Provides at least one Pulitzer Prize win between 2020–2024 inclusive, including the year and category",
        parent=org_node,
        critical=True
    )
    pul_year = safe_str(org.pulitzer.year if org.pulitzer else None)
    pul_cat = safe_str(org.pulitzer.category if org.pulitzer else None)
    pul_title = safe_str(org.pulitzer.award_title if org.pulitzer else None)
    pul_claim = (
        f"{safe_str(org.name)} won a Pulitzer Prize in {pul_year} in the category '{pul_cat}'. "
        f"Award title/context: '{pul_title}'. The win year must be between {PULITZER_YEAR_MIN} and {PULITZER_YEAR_MAX} inclusive."
    )
    pul_sources = org.references.pulitzer_urls if org.references else []
    await evaluator.verify(
        claim=pul_claim,
        node=pul_leaf,
        sources=pul_sources,
        additional_instruction=(
            f"Use the provided URL(s) to confirm a Pulitzer Prize win for the organization. "
            f"Ensure the year is within {PULITZER_YEAR_MIN}–{PULITZER_YEAR_MAX} inclusive and that a category is stated. "
            "Shared awards or team awards are acceptable if attributed to the organization (or its staff)."
        )
    )

    # Subscription model / paywall
    subs_leaf = evaluator.add_leaf(
        id=f"org{display_idx}_subscription_model",
        desc="Describes the organization's digital paywall/subscription-based model for online news access",
        parent=org_node,
        critical=True
    )
    subs_details = safe_str(org.subscription_details)
    subs_claim = (
        f"{safe_str(org.name)} operates a digital paywall or subscription-based model for online news access. "
        f"Details provided: {subs_details}"
    )
    subs_sources = org.references.subscription_urls if org.references else []
    await evaluator.verify(
        claim=subs_claim,
        node=subs_leaf,
        sources=subs_sources,
        additional_instruction=(
            "Verify that the organization has a paywall or subscription product for its digital content. "
            "Accept metered paywalls, hard paywalls, and subscription offerings with pricing tiers. "
            "Marketing or help pages explaining subscription access qualify as evidence."
        )
    )

    # International presence (bureaus or foreign correspondents)
    intl_leaf = evaluator.add_leaf(
        id=f"org{display_idx}_international_presence",
        desc="Provides evidence of international news bureaus and/or foreign correspondents, including examples",
        parent=org_node,
        critical=True
    )
    intl_info = safe_str(org.international_operations)
    intl_claim = (
        f"{safe_str(org.name)} maintains international news bureaus and/or employs foreign correspondents. "
        f"Examples or description: {intl_info}"
    )
    intl_sources = org.references.international_urls if org.references else []
    await evaluator.verify(
        claim=intl_claim,
        node=intl_leaf,
        sources=intl_sources,
        additional_instruction=(
            "Confirm that the organization has international bureau locations and/or foreign correspondents. "
            "Examples of bureau cities or staff role pages are acceptable. "
            "General claims without explicit evidence should not pass."
        )
    )

    # Daily publication
    daily_leaf = evaluator.add_leaf(
        id=f"org{display_idx}_daily_publication",
        desc="Shows the organization publishes news content daily",
        parent=org_node,
        critical=True
    )
    daily_info = safe_str(org.daily_publication_info)
    daily_claim = (
        f"{safe_str(org.name)} publishes news content on a daily basis. Evidence provided: {daily_info}"
    )
    daily_sources = org.references.daily_publication_urls if org.references else []
    await evaluator.verify(
        claim=daily_claim,
        node=daily_leaf,
        sources=daily_sources,
        additional_instruction=(
            "Look for explicit statements or reasonable evidence of daily publishing cadence (e.g., 'updated daily', "
            "'publishes every day', or a newsroom/about page indicating daily publication). "
            "A single day's content is not sufficient evidence without a general policy or cadence statement."
        )
    )

    # Reach threshold: top 25 US newspapers by circulation OR top 50 global news sites by visitors
    reach_leaf = evaluator.add_leaf(
        id=f"org{display_idx}_reach_threshold",
        desc="Shows the organization is either top 25 U.S. newspapers by circulation OR top 50 global news websites by monthly visitors, with supporting stats/ranking",
        parent=org_node,
        critical=True
    )
    reach_info = safe_str(org.reach_data)
    reach_claim = (
        f"{safe_str(org.name)} meets the reach threshold. Evidence: {reach_info}. "
        "It must be either among the top 25 U.S. newspapers by circulation or among the top 50 global news websites by monthly visitors."
    )
    reach_sources = org.references.reach_urls if org.references else []
    await evaluator.verify(
        claim=reach_claim,
        node=reach_leaf,
        sources=reach_sources,
        additional_instruction=(
            "Verify the ranking or statistics from reputable sources (industry reports, audit bureaus, analytics firms). "
            "Accept either criterion: top 25 U.S. newspapers by circulation OR top 50 global websites by monthly visitors. "
            "Reasonable recency is acceptable; ensure the ranking explicitly places the organization within the threshold."
        )
    )

    # Reference URLs existence checks (critical)
    ref_group = evaluator.add_parallel(
        id=f"org{display_idx}_reference_urls",
        desc=f"Provides reference URL(s) supporting each required piece of information for organization {display_idx}",
        parent=org_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_url(hq_sources),
        id=f"org{display_idx}_hq_url",
        desc="URL supports headquarters address/location",
        parent=ref_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_url(pul_sources),
        id=f"org{display_idx}_pulitzer_url",
        desc="URL supports Pulitzer win (year and category) between 2020–2024",
        parent=ref_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_url(subs_sources),
        id=f"org{display_idx}_subscription_url",
        desc="URL supports paywall/subscription model",
        parent=ref_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_url(intl_sources),
        id=f"org{display_idx}_international_url",
        desc="URL supports international bureaus/correspondents claim",
        parent=ref_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_url(daily_sources),
        id=f"org{display_idx}_daily_publication_url",
        desc="URL supports daily publication claim",
        parent=ref_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_url(reach_sources),
        id=f"org{display_idx}_reach_url",
        desc="URL supports circulation/visitors ranking/statistics meeting the threshold",
        parent=ref_group,
        critical=True
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
    Evaluate the answer for identifying three major U.S. news organizations satisfying all constraints.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Use parallel to allow partial credit across organizations
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

    # Record constraint info for transparency
    evaluator.add_custom_info(
        info={
            "allowed_cities": ALLOWED_CITIES,
            "pulitzer_year_range": [PULITZER_YEAR_MIN, PULITZER_YEAR_MAX],
            "reach_thresholds": {
                "us_newspapers_by_circulation": "top 25",
                "global_news_websites_by_monthly_visitors": "top 50"
            }
        },
        info_type="constraints",
        info_name="evaluation_constraints"
    )

    # Extract organizations data
    extraction = await evaluator.extract(
        prompt=prompt_extract_news_orgs(),
        template_class=NewsOrganizationsExtraction,
        extraction_name="news_organizations"
    )

    # Normalize to exactly 3 organizations (pad with blanks if fewer)
    orgs: List[OrganizationInfo] = list(extraction.organizations[:3])
    while len(orgs) < 3:
        orgs.append(OrganizationInfo())

    # Build verification for each organization
    for idx, org in enumerate(orgs):
        await verify_organization(evaluator, root, org, idx)

    # Return summary
    return evaluator.get_summary()