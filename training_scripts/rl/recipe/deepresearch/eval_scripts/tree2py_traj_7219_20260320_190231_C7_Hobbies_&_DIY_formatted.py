import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "craft_subscription_boxes_2026"
TASK_DESCRIPTION = """
Identify three different craft subscription box services that are currently operating in the United States as of 2026. For each service, provide the following information: (1) The official company name and website URL, (2) Confirmation that the service ships to U.S. addresses, (3) The monthly subscription cost for the standard monthly plan, (4) Confirmation that each monthly box includes at least one complete project, (5) Confirmation that all necessary materials and supplies for the project(s) are provided in the box, (6) The target skill level (beginner, intermediate, advanced, or all levels), (7) The primary craft category or categories covered, (8) Confirmation that the service is actively accepting subscriptions as of 2026, (9) Confirmation that flexible subscription options are available (such as monthly, multi-month, or annual plans), (10) Information about whether reusable tools are included or only consumable materials, and (11) The typical number of projects included per monthly box. All three services must meet criteria 1-9, while criteria 10-11 are additional information to provide.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ServiceInfo(BaseModel):
    company_name: Optional[str] = None
    website_url: Optional[str] = None
    us_shipping: Optional[str] = None
    monthly_cost: Optional[str] = None
    includes_projects: Optional[str] = None
    materials_provided: Optional[str] = None
    skill_level: Optional[str] = None
    craft_category: Optional[str] = None
    active_as_of_2026: Optional[str] = None
    subscription_options: Optional[str] = None
    tools_policy: Optional[str] = None
    projects_per_box: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)


class ServicesExtraction(BaseModel):
    services: List[ServiceInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_services() -> str:
    return """
    Extract up to five craft subscription box services that the answer mentions. For each service, extract the following fields exactly as stated in the answer (do not invent anything not present in the answer text):
    - company_name: Official subscription service or brand name (string)
    - website_url: Official website URL for the service (string URL)
    - us_shipping: The answer's statement about U.S. shipping (string like "ships to U.S.", "U.S. only", "worldwide incl. U.S.", etc.), or null if not stated
    - monthly_cost: The monthly price for the standard month-to-month plan, as presented (e.g., "$29", "$29.99 + shipping", "from $25") or null if not stated
    - includes_projects: The answer's statement confirming each monthly box includes at least one complete project (string) or null if not stated
    - materials_provided: The answer's statement about providing all necessary materials/supplies (string) or null if not stated
    - skill_level: The target skill level as stated (e.g., "beginner", "intermediate", "advanced", "all levels") or null if not stated
    - craft_category: The primary craft category/categories (e.g., "knitting", "papercraft, embroidery") or null if not stated
    - active_as_of_2026: The answer's statement confirming the service is active/accepting subscriptions as of 2026 (string) or null if not stated
    - subscription_options: The answer's statement about plan flexibility (e.g., "monthly, 3-month, annual") or null if not stated
    - tools_policy: The answer's statement about whether reusable tools are included or only consumable materials (string) or null if not stated
    - projects_per_box: The answer's statement of the typical number of projects per box (string like "1", "2-3") or null if not stated
    - supporting_urls: Any additional URLs the answer cites for this service besides the main website_url (array of strings). If none, return an empty array.

    Return a JSON object with a single key "services" as an array of service objects. If fewer than three services are described, include as many as are present. If the answer provides more than three, still extract them all (we will use the first three distinct ones later). For any field not present in the answer, return null (or empty array for supporting_urls).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_domain(url: Optional[str]) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        host = parsed.hostname or ""
        # Strip common prefixes like www.
        return host.lower().lstrip("www.") if host else ""
    except Exception:
        return ""


def combined_sources(service: ServiceInfo) -> List[str]:
    urls: List[str] = []
    if service.website_url and service.website_url.strip():
        urls.append(service.website_url.strip())
    # Append any additional supporting URLs explicitly cited in the answer
    if service.supporting_urls:
        urls.extend([u for u in service.supporting_urls if isinstance(u, str) and u.strip()])
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def nth_service_label(idx: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][idx] if 0 <= idx < 5 else f"#{idx + 1}"


# --------------------------------------------------------------------------- #
# Verification for a single service                                           #
# --------------------------------------------------------------------------- #
async def verify_service(
    evaluator: Evaluator,
    parent_node,
    service: ServiceInfo,
    index: int,
) -> None:
    """
    Build verification subtree for one craft subscription box service.
    Criteria 1-9 are critical; 10-11 are non-critical (additional information).
    """
    desc = f"{nth_service_label(index)} qualifying craft subscription box service"
    svc_node = evaluator.add_parallel(
        id=f"subscription_box_{index + 1}",
        desc=desc,
        parent=parent_node,
        critical=False
    )

    # Basic required info (company name + official website URL) must exist to proceed
    required_ok = (
        (service.company_name is not None and service.company_name.strip() != "") and
        (service.website_url is not None and service.website_url.strip() != "")
    )
    evaluator.add_custom_node(
        result=required_ok,
        id=f"subscription_box_{index + 1}_required_info",
        desc="Official name and official website URL are provided in the answer",
        parent=svc_node,
        critical=True
    )

    sources = combined_sources(service)

    # 1) Ships to U.S.
    node_us = evaluator.add_leaf(
        id=f"subscription_box_{index + 1}_operates_in_us",
        desc="Service ships to addresses in the United States",
        parent=svc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The subscription box service ships to addresses in the United States.",
        node=node_us,
        sources=sources,
        additional_instruction="Check the shipping/delivery policy, FAQ, or checkout page to confirm U.S. shipping is available. If the site clearly shows 'Subscribe' for U.S. customers or lists U.S. shipping options, consider it supported."
    )

    # 2) Monthly subscription cost (standard monthly plan)
    node_cost = evaluator.add_leaf(
        id=f"subscription_box_{index + 1}_monthly_cost",
        desc="Provide the monthly subscription cost for the standard monthly plan",
        parent=svc_node,
        critical=True
    )
    cost_text = service.monthly_cost or ""
    await evaluator.verify(
        claim=f"The standard month-to-month plan costs '{cost_text}' per month (before shipping/taxes).",
        node=node_cost,
        sources=sources,
        additional_instruction="Verify that the official site lists this monthly price (or an equivalent like 'from $X per month'). Minor variations due to shipping/taxes or small rounding are acceptable."
    )

    # 3) Includes at least one complete project per monthly box
    node_projects_included = evaluator.add_leaf(
        id=f"subscription_box_{index + 1}_includes_projects",
        desc="Service includes at least one complete project per monthly box",
        parent=svc_node,
        critical=True
    )
    await evaluator.verify(
        claim="Each monthly box includes at least one complete craft project (a full kit with instructions).",
        node=node_projects_included,
        sources=sources,
        additional_instruction="Confirm the site states that a complete, ready-to-make project (or kit) is included each month. Accept synonyms like 'project kit', 'complete kit', or 'make-this-project'."
    )

    # 4) All necessary materials/supplies provided
    node_materials = evaluator.add_leaf(
        id=f"subscription_box_{index + 1}_materials_provided",
        desc="Service provides all necessary materials and supplies for the included project(s)",
        parent=svc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The box includes all necessary materials and supplies needed to complete the project(s).",
        node=node_materials,
        sources=sources,
        additional_instruction="Accept common exceptions like basic household tools (e.g., scissors, glue, ruler). The site should clearly indicate that materials needed for the project's core tasks are provided."
    )

    # 5) Target skill level
    node_skill = evaluator.add_leaf(
        id=f"subscription_box_{index + 1}_skill_level_specified",
        desc="Service clearly indicates the target skill level (beginner, intermediate, advanced, or all levels)",
        parent=svc_node,
        critical=True
    )
    skill_text = service.skill_level or ""
    await evaluator.verify(
        claim=f"The service specifies the target skill level as '{skill_text}' (or indicates it is suitable for all levels).",
        node=node_skill,
        sources=sources,
        additional_instruction="Look for explicit wording like 'beginner-friendly', 'intermediate', 'advanced', or 'all skill levels'. Synonyms or level ranges are acceptable."
    )

    # 6) Craft category
    node_category = evaluator.add_leaf(
        id=f"subscription_box_{index + 1}_craft_category",
        desc="Specify the primary craft category or categories covered by the service",
        parent=svc_node,
        critical=True
    )
    cat_text = service.craft_category or ""
    await evaluator.verify(
        claim=f"The primary craft category or categories include: {cat_text}.",
        node=node_category,
        sources=sources,
        additional_instruction="Check the product/plan pages for how the service describes its craft focus (e.g., knitting, embroidery, papercraft, mixed crafts)."
    )

    # 7) Active and accepting subscriptions as of 2026
    node_active = evaluator.add_leaf(
        id=f"subscription_box_{index + 1}_active_as_of_2026",
        desc="Service is actively operating and accepting subscriptions as of 2026",
        parent=svc_node,
        critical=True
    )
    await evaluator.verify(
        claim="As of 2026, the service is actively operating and accepting subscriptions.",
        node=node_active,
        sources=sources,
        additional_instruction="Evidence includes an active 'Subscribe', 'Join', or 'Add to cart' flow for subscriptions, up-to-date plan pages, or current cycle information. If the site indicates 'sold out' temporarily but generally still accepts subscriptions, consider it active."
    )

    # 8) Flexible subscription options (monthly, multi-month, annual, etc.)
    node_options = evaluator.add_leaf(
        id=f"subscription_box_{index + 1}_subscription_options",
        desc="Service offers flexible subscription options (monthly, multi-month, or annual plans)",
        parent=svc_node,
        critical=True
    )
    opts_text = service.subscription_options or ""
    await evaluator.verify(
        claim=f"The service offers flexible plan options such as monthly, multi-month bundles, or annual subscriptions. Example options stated: {opts_text}.",
        node=node_options,
        sources=sources,
        additional_instruction="Confirm presence of at least two different terms (e.g., monthly and 3/6/12-month) or mention of gift/prepay options that imply flexibility."
    )

    # 9) Official company/service name (supported by website)
    node_name = evaluator.add_leaf(
        id=f"subscription_box_{index + 1}_company_name",
        desc="Provide the official name of the subscription box service",
        parent=svc_node,
        critical=True
    )
    name_text = service.company_name or ""
    await evaluator.verify(
        claim=f"The official subscription service name/brand is '{name_text}'.",
        node=node_name,
        sources=sources,
        additional_instruction="The site should consistently show this brand/service name (e.g., on the header, footer, product pages, or about page). Allow minor casing or punctuation variations."
    )

    # 10) Official website URL correctness (source = the URL itself)
    node_site = evaluator.add_leaf(
        id=f"subscription_box_{index + 1}_website_url",
        desc="Provide the official website URL for the subscription service",
        parent=svc_node,
        critical=True
    )
    site_claim_name = name_text if name_text else "the subscription service"
    await evaluator.verify(
        claim=f"This URL is the official website for {site_claim_name}.",
        node=node_site,
        sources=service.website_url if service.website_url else None,
        additional_instruction="Judge whether the page appears to be the official site/landing page for the named subscription service (e.g., plan pages, branding alignment). If URL is missing/invalid, this should fail."
    )

    # 11) Tools provision policy (non-critical additional info)
    node_tools = evaluator.add_leaf(
        id=f"subscription_box_{index + 1}_tools_provision_policy",
        desc="Indicate whether the service includes reusable tools in boxes or only consumable materials",
        parent=svc_node,
        critical=False
    )
    tools_text = service.tools_policy or ""
    await evaluator.verify(
        claim=f"The tools provision policy is: '{tools_text}' (e.g., whether reusable tools are included or only consumable materials).",
        node=node_tools,
        sources=sources,
        additional_instruction="Look for mentions of included tools vs. requiring you to have your own. Accept partial inclusions (e.g., occasional tools included) if clearly stated."
    )

    # 12) Typical number of projects per monthly box (non-critical additional info)
    node_projects_count = evaluator.add_leaf(
        id=f"subscription_box_{index + 1}_projects_per_box",
        desc="Specify the typical number of projects included in each monthly box",
        parent=svc_node,
        critical=False
    )
    proj_text = service.projects_per_box or ""
    await evaluator.verify(
        claim=f"The typical number of projects per monthly box is '{proj_text}'.",
        node=node_projects_count,
        sources=sources,
        additional_instruction="Check plan/product descriptions. Accept ranges like '1-2' or '1+' if the site indicates variability."
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
    Evaluate an answer for the craft subscription boxes (2026) task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # three services evaluated independently
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

    # 1) Extract services from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_services(),
        template_class=ServicesExtraction,
        extraction_name="services_extraction"
    )

    # 2) Select up to three distinct services by website domain (first 3 distinct)
    distinct_services: List[ServiceInfo] = []
    seen_domains = set()
    for svc in extracted.services:
        domain = normalize_domain(svc.website_url)
        if domain and domain not in seen_domains:
            seen_domains.add(domain)
            distinct_services.append(svc)
        # Stop when we have 3 distinct
        if len(distinct_services) >= 3:
            break

    # If fewer than 3 distinct services present, pad with empty placeholders
    while len(distinct_services) < 3:
        distinct_services.append(ServiceInfo())

    # 3) Add uniqueness check (critical to the overall task requirement: "three different")
    uniq_domains = [normalize_domain(s.website_url) for s in distinct_services if s.website_url]
    uniq_ok = len([d for d in uniq_domains if d]) == 3 and len(set([d for d in uniq_domains if d])) == 3
    evaluator.add_custom_node(
        result=uniq_ok,
        id="distinct_services_check",
        desc="Three services are distinct by official website/domain",
        parent=root,
        critical=True
    )

    # Record some custom info for debugging/traceability
    evaluator.add_custom_info(
        {
            "selected_services": [
                {
                    "company_name": s.company_name,
                    "website_url": s.website_url,
                    "domain": normalize_domain(s.website_url)
                }
                for s in distinct_services
            ]
        },
        info_type="selection_info",
        info_name="selected_services_info"
    )

    # 4) Build verification subtrees for each of the three services
    for idx in range(3):
        await verify_service(evaluator, root, distinct_services[idx], idx)

    # 5) Return evaluation summary
    return evaluator.get_summary()