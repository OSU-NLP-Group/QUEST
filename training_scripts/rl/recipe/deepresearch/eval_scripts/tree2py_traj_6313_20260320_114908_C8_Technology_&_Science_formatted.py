import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "colo_provider_all_reqs_v1"
TASK_DESCRIPTION = """
Identify at least one colocation data center provider operating in the United States that satisfies ALL of the following requirements:

Infrastructure and Technical Requirements:
- Operates facilities certified as Uptime Institute Tier III or Tier IV
- Provides an uptime Service Level Agreement (SLA) of at least 99.98%
- Offers carrier-neutral network connectivity options

Certifications and Compliance Requirements:
- Holds SOC 2 Type II certification
- Maintains ISO 27001 certification
- Has SSAE 18 attestation documentation

Service Offerings Requirements:
- Provides colocation services for customer equipment
- Offers managed services for infrastructure management
- Includes 24/7 remote hands technical support
- Offers disaster recovery services
- Supports hybrid cloud integration capabilities

Geographic and Operational Requirements:
- Operates data center facilities in at least 3 different US states
- Has been operational as a company since before January 1, 2020

Provide the name of the qualifying provider along with URL references that verify each of the certifications, technical specifications, and operational criteria listed above.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProviderEvidence(BaseModel):
    provider_name: Optional[str] = None

    # Optional claims extracted from the answer (free-form text; for context only)
    tier_level_claim: Optional[str] = None
    uptime_sla_claim: Optional[str] = None
    founding_year_claim: Optional[str] = None
    states_claimed: List[str] = Field(default_factory=list)

    # URL evidences for each requirement
    tier_cert_urls: List[str] = Field(default_factory=list)
    uptime_sla_urls: List[str] = Field(default_factory=list)
    carrier_neutral_urls: List[str] = Field(default_factory=list)

    soc2_type2_urls: List[str] = Field(default_factory=list)
    iso27001_urls: List[str] = Field(default_factory=list)
    ssae18_urls: List[str] = Field(default_factory=list)

    colocation_urls: List[str] = Field(default_factory=list)
    managed_services_urls: List[str] = Field(default_factory=list)
    remote_hands_urls: List[str] = Field(default_factory=list)
    disaster_recovery_urls: List[str] = Field(default_factory=list)
    hybrid_cloud_urls: List[str] = Field(default_factory=list)

    multi_state_urls: List[str] = Field(default_factory=list)
    operational_history_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_provider_evidence() -> str:
    return """
    Extract a single colocation data center provider and all cited URLs from the answer that are intended to verify each requirement below. If multiple providers are mentioned, choose the first provider that appears to have the most complete supporting evidence across the criteria. Do not invent any URLs; extract only those explicitly present in the answer (including markdown links).

    Return a JSON object with the following fields:
    - provider_name: The provider/company name.
    - tier_level_claim: Optional free-text snippet from the answer describing the Tier level (e.g., "Tier III" or "Tier IV").
    - uptime_sla_claim: Optional free-text snippet from the answer describing the SLA (e.g., "99.99% uptime").
    - founding_year_claim: Optional free-text snippet from the answer with a founding/operational-since year if stated.
    - states_claimed: Array of any U.S. state names or abbreviations explicitly mentioned for locations (if provided in the answer).
    - tier_cert_urls: Array of URLs cited to prove Uptime Institute Tier III or Tier IV certification.
    - uptime_sla_urls: Array of URLs cited to prove an uptime SLA of at least 99.98%.
    - carrier_neutral_urls: Array of URLs cited to prove carrier-neutral network connectivity.
    - soc2_type2_urls: Array of URLs cited to prove SOC 2 Type II certification.
    - iso27001_urls: Array of URLs cited to prove ISO 27001 certification.
    - ssae18_urls: Array of URLs cited to prove SSAE 18 attestation (e.g., SOC 1 under SSAE 18).
    - colocation_urls: Array of URLs cited to prove colocation services for customer equipment.
    - managed_services_urls: Array of URLs cited to prove managed services for infrastructure management.
    - remote_hands_urls: Array of URLs cited to prove 24/7 remote hands technical support.
    - disaster_recovery_urls: Array of URLs cited to prove disaster recovery services.
    - hybrid_cloud_urls: Array of URLs cited to prove hybrid cloud integration capabilities.
    - multi_state_urls: Array of URLs cited to prove the provider operates facilities in at least 3 different U.S. states.
    - operational_history_urls: Array of URLs cited to prove the provider has been operational since before January 1, 2020.

    URL extraction rules:
    - Include only valid URLs explicitly present in the answer (plain or markdown).
    - If a URL is missing protocol, prepend http://
    - Remove duplicates within each array.

    If any required field is not present in the answer, set it to null (for single value fields) or an empty list (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls)


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def _add_requirement_check(
    evaluator: Evaluator,
    *,
    parent,
    req_id: str,
    req_desc: str,
    urls: Optional[List[str]],
    claim: str,
    add_ins: str,
) -> None:
    """
    Build a critical requirement node with:
      - a critical URL existence check
      - a critical verification leaf checking the claim against the provided URLs
    """
    # Parent requirement node (critical; parallel to allow its children to be evaluated independently)
    req_node = evaluator.add_parallel(
        id=req_id,
        desc=req_desc,
        parent=parent,
        critical=True,
    )

    # Existence check for URLs (critical)
    evaluator.add_custom_node(
        result=_has_urls(urls),
        id=f"{req_id}_urls_present",
        desc=f"At least one URL reference is provided for: {req_desc}",
        parent=req_node,
        critical=True,
    )

    # Verification leaf (critical) grounded by the provided URLs
    verify_node = evaluator.add_leaf(
        id=f"{req_id}_supported",
        desc=f"{req_desc} — supported by cited sources",
        parent=req_node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=urls or [],
        additional_instruction=add_ins,
    )


async def verify_provider_requirements(
    evaluator: Evaluator,
    root,
    evidence: ProviderEvidence,
) -> None:
    """
    Build and verify all requirement nodes for the extracted provider.
    """
    provider_name = (evidence.provider_name or "").strip() or "the provider"

    # A single critical gating node to ensure provider name is present
    evaluator.add_custom_node(
        result=bool(evidence.provider_name and evidence.provider_name.strip()),
        id="provider_name_provided",
        desc="Provider name is provided in the answer",
        parent=root,
        critical=True,
    )

    # 1) Uptime Institute Tier III/IV certification
    await _add_requirement_check(
        evaluator,
        parent=root,
        req_id="tier_certification",
        req_desc="Provider operates facilities certified as Uptime Institute Tier III or Tier IV",
        urls=evidence.tier_cert_urls,
        claim=f"{provider_name} operates at least one data center facility that is certified by the Uptime Institute as Tier III or Tier IV.",
        add_ins=(
            "The source must explicitly mention 'Uptime Institute' and 'Tier III' or 'Tier IV'. "
            "Accept clear phrases like 'Uptime Institute Tier III (or Tier IV) certified', "
            "'TIER III/IV Certified Constructed Facility (TCCF)', or 'Certification of Design Documents (TCDD)'. "
            "Reject vague statements like 'built to Tier III' or 'Tier III design' that do not clearly indicate Uptime Institute certification."
        ),
    )

    # 2) Uptime SLA ≥ 99.98%
    await _add_requirement_check(
        evaluator,
        parent=root,
        req_id="uptime_sla",
        req_desc="Provider offers uptime SLA of at least 99.98%",
        urls=evidence.uptime_sla_urls,
        claim=f"{provider_name} publicly offers an uptime Service Level Agreement (SLA) of at least 99.98%.",
        add_ins=(
            "Pass if the page states an availability/uptime guarantee of 99.98% or higher (e.g., 99.99%, 99.995%, 99.999%, or 'five nines'). "
            "If multiple service tiers are shown, at least one colocation-related SLA must be ≥ 99.98%."
        ),
    )

    # 3) Carrier-neutral network connectivity options
    await _add_requirement_check(
        evaluator,
        parent=root,
        req_id="carrier_neutral",
        req_desc="Provider offers carrier-neutral network connectivity",
        urls=evidence.carrier_neutral_urls,
        claim=f"{provider_name} provides carrier-neutral connectivity options (customers can choose among multiple network carriers/ISPs).",
        add_ins=(
            "Look for phrases like 'carrier-neutral' or an explicit list of multiple on-net carriers/ISPs/meet-me room options. "
            "Offerings limited to a single upstream provider should not pass."
        ),
    )

    # 4) SOC 2 Type II certification
    await _add_requirement_check(
        evaluator,
        parent=root,
        req_id="soc2_type2",
        req_desc="Provider holds SOC 2 Type II certification",
        urls=evidence.soc2_type2_urls,
        claim=f"{provider_name} holds SOC 2 Type II certification (or has a current SOC 2 Type II report).",
        add_ins=(
            "Specifically confirm SOC 2 Type II (not Type I). A statement such as 'SOC 2 Type II report available under NDA' is acceptable. "
            "Do not confuse SOC 2 with SOC 1; SOC 1 alone does not satisfy this requirement."
        ),
    )

    # 5) ISO 27001 certification
    await _add_requirement_check(
        evaluator,
        parent=root,
        req_id="iso_27001",
        req_desc="Provider maintains ISO 27001 certification",
        urls=evidence.iso27001_urls,
        claim=f"{provider_name} maintains ISO 27001 certification.",
        add_ins=(
            "Accept 'ISO/IEC 27001' or 'ISO 27001' certification, including 2013 or 2022 versions. "
            "Look for explicit certification statements or trust center/compliance pages."
        ),
    )

    # 6) SSAE 18 attestation
    await _add_requirement_check(
        evaluator,
        parent=root,
        req_id="ssae_18",
        req_desc="Provider has SSAE 18 attestation",
        urls=evidence.ssae18_urls,
        claim=f"{provider_name} has SSAE 18 attestation (e.g., SOC 1 under SSAE 18).",
        add_ins=(
            "Look for 'SSAE 18' or 'SOC 1' attestation aligned to SSAE 18. "
            "General references to older SSAE standards without 'SSAE 18' should not pass."
        ),
    )

    # 7) Colocation services
    await _add_requirement_check(
        evaluator,
        parent=root,
        req_id="colocation_services",
        req_desc="Provider offers colocation services",
        urls=evidence.colocation_urls,
        claim=f"{provider_name} provides colocation services for customer-owned equipment.",
        add_ins=(
            "Accept phrases such as 'colocation', 'colo', 'rack and cage colocation', or 'cabinet colocation'. "
            "Managed hosting alone without colocation should not pass."
        ),
    )

    # 8) Managed services
    await _add_requirement_check(
        evaluator,
        parent=root,
        req_id="managed_services",
        req_desc="Provider offers managed services",
        urls=evidence.managed_services_urls,
        claim=f"{provider_name} offers managed services for infrastructure management.",
        add_ins=(
            "Look for 'managed services' or specific examples such as managed monitoring, managed backup, network management, patching, or infrastructure management."
        ),
    )

    # 9) 24/7 remote hands
    await _add_requirement_check(
        evaluator,
        parent=root,
        req_id="remote_hands_247",
        req_desc="Provider includes 24/7 remote hands support",
        urls=evidence.remote_hands_urls,
        claim=f"{provider_name} includes 24/7 (24x7) remote hands technical support.",
        add_ins=(
            "Look for 'remote hands' combined with '24/7', '24x7', 'around-the-clock', or equivalent wording."
        ),
    )

    # 10) Disaster recovery services
    await _add_requirement_check(
        evaluator,
        parent=root,
        req_id="disaster_recovery",
        req_desc="Provider offers disaster recovery services",
        urls=evidence.disaster_recovery_urls,
        claim=f"{provider_name} offers disaster recovery services.",
        add_ins=(
            "Accept 'disaster recovery', 'DR', 'DRaaS', or clear business continuity/disaster recovery offerings."
        ),
    )

    # 11) Hybrid cloud integration
    await _add_requirement_check(
        evaluator,
        parent=root,
        req_id="hybrid_cloud",
        req_desc="Provider supports hybrid cloud integration",
        urls=evidence.hybrid_cloud_urls,
        claim=f"{provider_name} supports hybrid cloud integration capabilities.",
        add_ins=(
            "Evidence may include 'hybrid cloud', 'multicloud', 'cloud interconnect', or on-ramps such as AWS Direct Connect, Azure ExpressRoute, "
            "Google Cloud Interconnect, Megaport/Equinix Fabric, etc."
        ),
    )

    # 12) Multi-state presence (≥ 3 different U.S. states)
    await _add_requirement_check(
        evaluator,
        parent=root,
        req_id="multi_state_presence",
        req_desc="Provider operates data centers in at least 3 different US states",
        urls=evidence.multi_state_urls,
        claim=f"{provider_name} operates data center facilities in at least three different U.S. states.",
        add_ins=(
            "Verify the page(s) indicate locations in 3 or more distinct U.S. states. Cities must map to distinct states (e.g., Phoenix, AZ; Dallas, TX; Ashburn, VA). "
            "Multiple cities within the same state count as one state. If the evidence is ambiguous, fail."
        ),
    )

    # 13) Operational history (before 2020-01-01)
    await _add_requirement_check(
        evaluator,
        parent=root,
        req_id="operational_history",
        req_desc="Provider has been operational since before 2020",
        urls=evidence.operational_history_urls,
        claim=f"{provider_name} has been operational as a company since before January 1, 2020.",
        add_ins=(
            "Accept evidence such as 'founded in <year ≤ 2019>', 'operational since <year ≤ 2019>', or press/news archives before 2020. "
            "If only vague language with no dates is provided, do not pass."
        ),
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
    Evaluate an answer for the 'colocation provider with all requirements' task.
    """
    # Initialize evaluator with a parallel root so each requirement is checked independently
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

    # Extract structured evidence from the answer
    evidence: ProviderEvidence = await evaluator.extract(
        prompt=prompt_extract_provider_evidence(),
        template_class=ProviderEvidence,
        extraction_name="provider_evidence",
    )

    # Add a brief custom info section summarizing URL counts (optional)
    url_counts = {
        "tier_cert_urls": len(evidence.tier_cert_urls),
        "uptime_sla_urls": len(evidence.uptime_sla_urls),
        "carrier_neutral_urls": len(evidence.carrier_neutral_urls),
        "soc2_type2_urls": len(evidence.soc2_type2_urls),
        "iso27001_urls": len(evidence.iso27001_urls),
        "ssae18_urls": len(evidence.ssae18_urls),
        "colocation_urls": len(evidence.colocation_urls),
        "managed_services_urls": len(evidence.managed_services_urls),
        "remote_hands_urls": len(evidence.remote_hands_urls),
        "disaster_recovery_urls": len(evidence.disaster_recovery_urls),
        "hybrid_cloud_urls": len(evidence.hybrid_cloud_urls),
        "multi_state_urls": len(evidence.multi_state_urls),
        "operational_history_urls": len(evidence.operational_history_urls),
    }
    evaluator.add_custom_info(
        info={"provider_name": evidence.provider_name, "url_counts": url_counts},
        info_type="summary",
        info_name="extraction_summary",
    )

    # Build and run all verification checks
    await verify_provider_requirements(evaluator, root, evidence)

    # Return structured evaluation summary
    return evaluator.get_summary()