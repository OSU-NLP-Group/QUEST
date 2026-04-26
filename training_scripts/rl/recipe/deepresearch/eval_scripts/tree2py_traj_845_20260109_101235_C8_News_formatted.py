import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_news_orgs_trust"
TASK_DESCRIPTION = """
Identify three U.S.-based news organizations, each serving a different major city or metropolitan region, that meet all of the following professional journalism standards:

1. Trust and Transparency: The organization must have verifiable trust or transparency credentials, demonstrated by either participation in The Trust Project, a NewsGuard rating of 75 or higher, or Journalism Trust Initiative (JTI) certification.

2. Professional Membership: The organization must hold membership in a recognized professional journalism association such as the Society of Professional Journalists (SPJ), Associated Press (AP) membership, Investigative Reporters and Editors (IRE), or a similar professional journalism organization.

3. Digital Presence: The organization must maintain an active, publicly accessible news website with a regular publication schedule.

4. Published Ethical Standards: The organization must have a publicly available ethics policy, editorial standards document, or code of conduct for journalists.

5. Geographic Coverage: Each organization must serve a specific major U.S. city or metropolitan region (and the three organizations must serve different geographic areas).

For each news organization you identify, provide:
- The organization's name
- The city/region it serves
- Evidence of its trust/transparency credentials (specific rating, certification, or program participation)
- Evidence of its professional membership
- URL to its digital news website
- URL to its published ethical standards document
- Brief description of how it meets each criterion
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OrganizationItem(BaseModel):
    """One news organization with all required evidence fields."""
    name: Optional[str] = None
    city_region: Optional[str] = None

    website_url: Optional[str] = None
    ethics_url: Optional[str] = None

    trust_credential_type: Optional[str] = None  # e.g., "Trust Project", "NewsGuard", "JTI"
    trust_evidence_text: Optional[str] = None    # e.g., "NewsGuard score 89/100"
    trust_evidence_urls: List[str] = Field(default_factory=list)

    membership_body: Optional[str] = None        # e.g., "SPJ", "Associated Press", "IRE"
    membership_evidence_text: Optional[str] = None
    membership_evidence_urls: List[str] = Field(default_factory=list)

    brief_description: Optional[str] = None


class OrganizationsExtraction(BaseModel):
    """All organizations mentioned in the answer."""
    organizations: List[OrganizationItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_organizations() -> str:
    return """
    Extract all U.S.-based news organizations mentioned in the answer that are candidates for the task. 
    For each organization, extract the following fields exactly as presented in the answer:

    - name: The organization's full name.
    - city_region: The specific U.S. city or metropolitan region the organization serves (e.g., "Chicago", "Bay Area", "Greater Boston").
    - website_url: The URL to the organization's public news website.
    - ethics_url: The URL to the organization's published ethics policy, editorial standards, or code of conduct.
    - trust_credential_type: One of ["Trust Project", "NewsGuard", "JTI", "Other"]. If unspecified, use "Other".
    - trust_evidence_text: Any textual details provided in the answer about the trust/transparency credential (e.g., "NewsGuard rating 85/100", "Trust Project participant").
    - trust_evidence_urls: All URLs in the answer that support the trust/transparency credential (e.g., Trust Project partner listing, NewsGuard rating page, JTI certificate page).
    - membership_body: The professional journalism association cited (e.g., "SPJ", "Associated Press", "IRE", or similar).
    - membership_evidence_text: Any textual detail in the answer supporting the membership (e.g., "member since 2019").
    - membership_evidence_urls: All URLs in the answer that support the membership (e.g., membership directory listing, announcement).
    - brief_description: A brief explanation, as provided by the answer, of how the organization meets each criterion.

    Rules:
    - Extract only information explicitly present in the answer; do not invent missing fields.
    - If a field is missing, set it to null (or empty array for lists).
    - Include every organization mentioned; do NOT filter within extraction. Filtering to the first three will be done later.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_region(region: Optional[str]) -> str:
    if not region:
        return ""
    return "".join(ch.lower() for ch in region.strip())


def _merge_sources(*url_lists: List[Optional[str] | List[str]]) -> List[str]:
    """Merge multiple urls/lists into a flat unique list, skip None/empty."""
    collected: List[str] = []
    for item in url_lists:
        if item is None:
            continue
        if isinstance(item, list):
            for u in item:
                if u and isinstance(u, str) and u.strip():
                    collected.append(u.strip())
        elif isinstance(item, str):
            if item.strip():
                collected.append(item.strip())
    # De-duplicate while preserving order
    seen = set()
    result = []
    for u in collected:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _trust_claim_for_org(org: OrganizationItem) -> str:
    t = (org.trust_credential_type or "").lower()
    detail = org.trust_evidence_text or ""
    name = org.name or "the organization"
    if "newsguard" in t:
        return f"The organization '{name}' has a NewsGuard rating of 75 or higher. Evidence: {detail}"
    if "trust project" in t or "trustproject" in t:
        return f"The organization '{name}' participates in The Trust Project. Evidence: {detail}"
    if "jti" in t or "journalism trust initiative" in t:
        return f"The organization '{name}' holds Journalism Trust Initiative (JTI) certification. Evidence: {detail}"
    # Fallback general statement
    return f"The organization '{name}' has verifiable trust/transparency credentials documented by the provided evidence. Evidence: {detail}"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_organization(
    evaluator: Evaluator,
    parent_node,
    org: OrganizationItem,
    index: int
) -> None:
    """Build verification subtree and run checks for a single organization."""

    # Organization container node (non-critical to allow partial credit per org)
    org_node = evaluator.add_parallel(
        id=f"organization_{index+1}",
        desc=f"Organization {index+1}: {org.name or 'Unnamed'}",
        parent=parent_node,
        critical=False
    )

    # Information Completeness (non-critical)
    info_node = evaluator.add_parallel(
        id=f"org_{index+1}_information_completeness",
        desc=f"All required fields/evidence are provided for organization {index+1}",
        parent=org_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(org.name and org.name.strip()),
        id=f"org_{index+1}_name_provided",
        desc=f"Organization {index+1} name is provided",
        parent=info_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(org.city_region and org.city_region.strip()),
        id=f"org_{index+1}_city_region_provided",
        desc=f"Organization {index+1} city/region served is provided",
        parent=info_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(org.trust_evidence_text and org.trust_evidence_text.strip()) or bool(org.trust_evidence_urls),
        id=f"org_{index+1}_trust_evidence_provided",
        desc=f"Specific evidence of organization {index+1} trust/transparency credential is provided",
        parent=info_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(org.membership_body and org.membership_body.strip()) or bool(org.membership_evidence_urls),
        id=f"org_{index+1}_membership_evidence_provided",
        desc=f"Evidence of organization {index+1} professional membership is provided",
        parent=info_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(org.website_url and org.website_url.strip()),
        id=f"org_{index+1}_website_url_provided",
        desc=f"URL to organization {index+1} digital news website is provided",
        parent=info_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(org.ethics_url and org.ethics_url.strip()),
        id=f"org_{index+1}_ethics_url_provided",
        desc=f"URL to organization {index+1} published ethical standards document is provided",
        parent=info_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(org.brief_description and org.brief_description.strip()),
        id=f"org_{index+1}_brief_description_provided",
        desc=f"Brief description explains how organization {index+1} meets each criterion",
        parent=info_node,
        critical=False
    )

    # Criteria Compliance (critical – all must be satisfied for full pass of this subtree)
    criteria_node = evaluator.add_parallel(
        id=f"org_{index+1}_criteria_compliance",
        desc=f"Organization {index+1} meets all required standards",
        parent=org_node,
        critical=True
    )

    # US-Based
    us_leaf = evaluator.add_leaf(
        id=f"org_{index+1}_us_based",
        desc=f"Organization {index+1} is U.S.-based",
        parent=criteria_node,
        critical=True
    )
    us_sources = _merge_sources(org.website_url, org.membership_evidence_urls, org.trust_evidence_urls, org.ethics_url)
    us_claim = f"The organization '{org.name or ''}' is a U.S.-based news organization."
    await evaluator.verify(
        claim=us_claim,
        node=us_leaf,
        sources=us_sources,
        additional_instruction="Accept if the About/Contact/Imprint or coverage page shows a U.S. address or clearly states U.S. coverage."
    )

    # Trust & Transparency Credentials
    trust_leaf = evaluator.add_leaf(
        id=f"org_{index+1}_trust_transparency",
        desc=f"Organization {index+1} has verifiable trust/transparency credentials",
        parent=criteria_node,
        critical=True
    )
    trust_sources = _merge_sources(org.trust_evidence_urls, org.website_url)
    trust_claim = _trust_claim_for_org(org)
    await evaluator.verify(
        claim=trust_claim,
        node=trust_leaf,
        sources=trust_sources,
        additional_instruction=(
            "Any ONE of: Trust Project participation (partner list page), NewsGuard rating ≥ 75 (rating page), "
            "or JTI certification (certificate/listing page) is sufficient. Verify explicitly from the provided URLs."
        )
    )

    # Professional Membership
    membership_leaf = evaluator.add_leaf(
        id=f"org_{index+1}_professional_membership",
        desc=f"Organization {index+1} holds membership in a recognized professional journalism association",
        parent=criteria_node,
        critical=True
    )
    membership_sources = _merge_sources(org.membership_evidence_urls, org.website_url)
    membership_body = org.membership_body or "a recognized professional journalism association"
    membership_claim = f"The organization '{org.name or ''}' is a member of {membership_body}. Evidence: {org.membership_evidence_text or ''}"
    await evaluator.verify(
        claim=membership_claim,
        node=membership_leaf,
        sources=membership_sources,
        additional_instruction="Prefer official membership directories or announcement pages. Evidence from the organization's About page is acceptable if explicit."
    )

    # Digital Presence
    digital_leaf = evaluator.add_leaf(
        id=f"org_{index+1}_digital_presence",
        desc=f"Organization {index+1} maintains an active, publicly accessible news website with a regular publication schedule",
        parent=criteria_node,
        critical=True
    )
    digital_sources = _merge_sources(org.website_url)
    digital_claim = (
        f"The organization '{org.name or ''}' maintains an active, publicly accessible news website that publishes regularly."
    )
    await evaluator.verify(
        claim=digital_claim,
        node=digital_leaf,
        sources=digital_sources,
        additional_instruction="Check recent publication dates or sections showing ongoing updates (e.g., latest news, recent articles)."
    )

    # Published Ethical Standards
    ethics_leaf = evaluator.add_leaf(
        id=f"org_{index+1}_published_ethics",
        desc=f"Organization {index+1} has a publicly available ethics policy/editorial standards/code of conduct document",
        parent=criteria_node,
        critical=True
    )
    ethics_sources = _merge_sources(org.ethics_url)
    ethics_claim = (
        f"The organization '{org.name or ''}' publishes a publicly available ethics policy, editorial standards, or code of conduct."
    )
    await evaluator.verify(
        claim=ethics_claim,
        node=ethics_leaf,
        sources=ethics_sources,
        additional_instruction="Verify that the page explicitly contains ethics/editorial standards/code of conduct content, not just generic terms."
    )

    # Geographic Coverage
    geo_leaf = evaluator.add_leaf(
        id=f"org_{index+1}_geographic_coverage",
        desc=f"Organization {index+1} serves a specific major U.S. city or metropolitan region",
        parent=criteria_node,
        critical=True
    )
    geo_sources = _merge_sources(org.website_url)
    geo_claim = f"The organization '{org.name or ''}' serves the {org.city_region or ''} area."
    await evaluator.verify(
        claim=geo_claim,
        node=geo_leaf,
        sources=geo_sources,
        additional_instruction="Look for About/coverage pages stating the city/region served; headlines/sections naming the city are acceptable if coverage is clearly local."
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
    Evaluate an answer for the U.S.-based news organizations trust/compliance task.
    """
    # Initialize evaluator with a parallel root (allow partial credit across organizations)
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

    # Extract organizations mentioned in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_organizations(),
        template_class=OrganizationsExtraction,
        extraction_name="organizations_extraction"
    )

    # Record basic extraction info
    total_found = len(extracted.organizations)
    evaluator.add_custom_info(
        info={"total_organizations_mentioned": total_found},
        info_type="stats",
        info_name="extraction_stats"
    )

    # Gate: Provide at least three organizations (filtering will enforce using the first 3)
    valid_orgs = [o for o in extracted.organizations if o and o.name and o.name.strip()]
    evaluator.add_custom_node(
        result=len(valid_orgs) >= 3,
        id="Provide_Exactly_Three_Organizations",
        desc="Response identifies three (3) news organizations (not fewer or more)",
        parent=root,
        critical=True
    )

    # Use only the first three organizations for verification
    selected_orgs = extracted.organizations[:3]

    # Gate: Geographic uniqueness among the three selected organizations
    normalized_regions = [_normalize_region(o.city_region) for o in selected_orgs]
    unique_regions = set([r for r in normalized_regions if r])
    evaluator.add_custom_node(
        result=(len(selected_orgs) == 3 and len(unique_regions) == 3),
        id="Geographic_Uniqueness",
        desc="The three organizations serve three different major U.S. cities or metropolitan regions (no two serve the same area)",
        parent=root,
        critical=True
    )

    # Build per-organization subtrees
    for idx, org in enumerate(selected_orgs):
        # Create a container node for each organization under root
        org_container = evaluator.add_parallel(
            id=f"Organization_{idx+1}",
            desc=f"Organization {idx+1} verification",
            parent=root,
            critical=False
        )
        # Verify details for this organization
        await verify_organization(evaluator, org_container, org, idx)

    # Return standard summary
    return evaluator.get_summary()