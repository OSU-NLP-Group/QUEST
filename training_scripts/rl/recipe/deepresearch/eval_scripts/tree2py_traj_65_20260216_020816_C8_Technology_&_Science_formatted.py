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
TASK_ID = "colocation_selection_us_markets"
TASK_DESCRIPTION = """
I am evaluating enterprise colocation options for our company's critical infrastructure deployment across multiple US markets. We need to identify three colocation data centers that meet our comprehensive technical, security, and operational requirements.

For each of the three data centers, please provide:

1. Facility Name and Location: The specific data center facility name and its location, which must be in one of the following major US markets: Northern Virginia, Chicago, Dallas, Atlanta, Phoenix, Los Angeles, Silicon Valley, New York Metro, Seattle, or Denver.

2. Tier Certification: The data center must hold Uptime Institute Tier III or Tier IV certification (either for Design or Constructed Facility). Provide the tier level and a reference URL confirming this certification.

3. Security Certifications: The data center must possess both SOC 2 Type 2 and ISO 27001 certifications. Provide a reference URL confirming these certifications.

4. Carrier Neutrality: Confirm that the data center is carrier-neutral, allowing us to choose from multiple network service providers.

5. Power Capacity: Confirm that the data center supports a minimum power density of at least 8 kW per rack.

6. Physical Security: Confirm that the data center implements biometric access control systems.

7. Compliance Support: Confirm that the data center supports PCI DSS compliance requirements for customers.

8. Remote Hands Services: Confirm that the data center offers 24/7 remote hands services.

9. Uptime SLA: Confirm that the data center guarantees a minimum of 99.99% uptime in its service level agreement.

10. Interconnection: Confirm that the data center provides meet-me room (MMR) or cross-connect services for interconnection.

11. Provider Reputation: Confirm that the data center is operated by an established colocation provider with a documented presence in multiple US markets.

12. Reference URL: Provide a primary reference URL (such as the facility's official page) confirming the data center's location.

All information must be verifiable through official provider websites, certification bodies, or reputable data center industry sources.
"""

ALLOWED_MARKETS = [
    "Northern Virginia",
    "Chicago",
    "Dallas",
    "Atlanta",
    "Phoenix",
    "Los Angeles",
    "Silicon Valley",
    "New York Metro",
    "Seattle",
    "Denver"
]

def market_membership_instruction() -> str:
    return (
        "Allowed major US markets: Northern Virginia, Chicago, Dallas, Atlanta, Phoenix, Los Angeles, Silicon Valley, New York Metro, Seattle, Denver. "
        "City-to-market mapping examples for judgment: "
        "Northern Virginia includes Ashburn, Reston, Manassas, Herndon, Chantilly, Sterling. "
        "Chicago includes Chicago, Elk Grove Village, Oak Brook. "
        "Dallas includes Dallas, Plano, Irving, Richardson, Carrollton. "
        "Atlanta includes Atlanta, Alpharetta, Marietta, Norcross. "
        "Phoenix includes Phoenix, Tempe, Chandler, Mesa. "
        "Los Angeles includes Los Angeles, El Segundo, Torrance, Irvine. "
        "Silicon Valley includes San Jose, Santa Clara, Sunnyvale, Milpitas, Fremont. "
        "New York Metro includes New York City, Manhattan, Brooklyn, Newark, Secaucus, Jersey City. "
        "Seattle includes Seattle, Tukwila, Redmond, Bellevue. "
        "Denver includes Denver, Englewood, Aurora. "
        "Use the provided facility/location page as evidence for the location, then decide whether it belongs to one of the allowed markets."
    )

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DataCenterItem(BaseModel):
    facility_name: Optional[str] = None
    location: Optional[str] = None  # City/state or market text from the answer
    location_ref_url: Optional[str] = None  # Primary facility page confirming location

    tier_cert_level: Optional[str] = None  # e.g., "Tier III", "Tier IV"
    tier_cert_kind: Optional[str] = None   # e.g., "Constructed Facility", "Design"
    tier_cert_url: Optional[str] = None    # URL confirming Tier certification

    soc2_type2: Optional[str] = None       # evidence text from answer, leave as string
    iso27001: Optional[str] = None         # evidence text from answer, leave as string
    security_cert_url: Optional[str] = None  # URL confirming SOC2 and ISO27001

    carrier_neutral: Optional[str] = None
    power_density_kw_per_rack: Optional[str] = None  # keep string to accept forms like "8-12 kW"
    biometric_access: Optional[str] = None
    pci_dss_support: Optional[str] = None
    remote_hands_24x7: Optional[str] = None
    uptime_sla: Optional[str] = None       # e.g., "99.99%"
    interconnection_services: Optional[str] = None  # e.g., "MMR", "Cross-connect"

    provider_name: Optional[str] = None
    provider_presence_url: Optional[str] = None  # provider locations or markets page

    extra_urls: List[str] = Field(default_factory=list)  # any additional official or reputable URLs


class ColocationExtraction(BaseModel):
    datacenters: List[DataCenterItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_datacenters() -> str:
    return """
    Extract up to three (3) colocation data centers as provided in the answer. For each data center, extract the following fields exactly as presented:

    1. facility_name: The specific facility name (e.g., "Digital Realty - CHI10").
    2. location: The city/state or region information mentioned for the facility (e.g., "Ashburn, VA" or "Santa Clara, CA").
    3. location_ref_url: The primary official facility page URL confirming the location. If multiple are provided, choose the most direct facility page URL. If none provided, set to null.
    4. tier_cert_level: The tier level mentioned (prefer "Tier III" or "Tier IV"). If absent, set to null.
    5. tier_cert_kind: The certification type if specified (e.g., "Constructed Facility", "Design", "TCCF", "TCDD"). If absent, set to null.
    6. tier_cert_url: A URL confirming Tier III or Tier IV certification (prefer official Uptime Institute certification listing or official provider page that references it). If none provided, set to null.
    7. soc2_type2: Text asserting SOC 2 Type 2 (if mentioned). If absent, set to null.
    8. iso27001: Text asserting ISO 27001 certification (if mentioned). If absent, set to null.
    9. security_cert_url: A URL that confirms SOC 2 Type 2 and/or ISO 27001 for the facility/provider (prefer official provider compliance page or official reports page). If none, set to null.
    10. carrier_neutral: Text asserting carrier neutrality. If absent, set to null.
    11. power_density_kw_per_rack: The stated per-rack power density (e.g., "8 kW per rack", "up to 10 kW"). If absent, set to null.
    12. biometric_access: Text indicating biometric access control. If absent, set to null.
    13. pci_dss_support: Text indicating PCI DSS support for customers. If absent, set to null.
    14. remote_hands_24x7: Text indicating 24/7 remote hands services. If absent, set to null.
    15. uptime_sla: The stated uptime guarantee (e.g., "99.99%"). If absent, set to null.
    16. interconnection_services: Text indicating meet-me room (MMR) or cross-connect services. If absent, set to null.
    17. provider_name: The colocation provider operating the facility (e.g., "Equinix", "Digital Realty"). If absent, set to null.
    18. provider_presence_url: A URL showing the provider has documented presence across multiple US markets (e.g., provider locations page listing multiple US cities/markets). If none, set to null.
    19. extra_urls: An array of any additional official or reputable URLs cited for this facility (e.g., product sheets, interconnection pages, SLA pages). If none, return an empty array.

    Return a JSON object with a 'datacenters' array of up to 3 objects. If the answer mentions more than three facilities, include only the first three. If fewer than three are mentioned, include those available.
    For URL fields, extract actual URLs explicitly mentioned in the answer; do not infer URLs. If a URL is missing, set to null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _collect_sources(*urls: Optional[str], extra: Optional[List[str]] = None) -> List[str]:
    out: List[str] = []
    for u in urls:
        if u and isinstance(u, str) and u.strip():
            out.append(u.strip())
    if extra:
        for u in extra:
            if u and isinstance(u, str) and u.strip():
                out.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    unique_out = []
    for u in out:
        if u not in seen:
            seen.add(u)
            unique_out.append(u)
    return unique_out


def _ordinal(idx_zero_based: int) -> str:
    return ["First", "Second", "Third"][idx_zero_based] if 0 <= idx_zero_based < 3 else f"#{idx_zero_based+1}"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_data_center(
    evaluator: Evaluator,
    root_node,
    dc: DataCenterItem,
    idx_zero_based: int
) -> None:
    """
    Build verification subtree and perform checks for one data center item.
    """
    ord_name = _ordinal(idx_zero_based)

    # Parent node for this data center (non-critical to allow partial credit across data centers)
    dc_node = evaluator.add_parallel(
        id=f"DataCenter_{idx_zero_based+1}",
        desc=f"{ord_name} qualifying colocation data center meeting all requirements",
        parent=root_node,
        critical=False
    )

    # 1) Facility Name existence (critical)
    evaluator.add_custom_node(
        result=bool(dc.facility_name and dc.facility_name.strip()),
        id=f"DC{idx_zero_based+1}_FacilityName",
        desc="Provide the specific facility name of the data center",
        parent=dc_node,
        critical=True
    )

    # 2) Location reference URL provided (critical)
    loc_ref_exists = evaluator.add_custom_node(
        result=bool(dc.location_ref_url and dc.location_ref_url.strip()),
        id=f"DC{idx_zero_based+1}_Location_Reference",
        desc="Provide URL reference confirming the data center's location",
        parent=dc_node,
        critical=True
    )

    # 3) Location is in allowed markets (critical, verified using the location page)
    loc_leaf = evaluator.add_leaf(
        id=f"DC{idx_zero_based+1}_Location",
        desc=("Data center is located in one of the following major US markets: "
              "Northern Virginia, Chicago, Dallas, Atlanta, Phoenix, Los Angeles, Silicon Valley, New York Metro, Seattle, or Denver"),
        parent=dc_node,
        critical=True
    )
    loc_claim = (
        f"The facility location '{dc.location or ''}' belongs to one of the allowed major US markets: "
        f"{', '.join(ALLOWED_MARKETS)}."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=dc.location_ref_url,
        additional_instruction=market_membership_instruction()
    )

    # 4) Tier certification URL exists (critical)
    tier_ref_exists = evaluator.add_custom_node(
        result=bool(dc.tier_cert_url and dc.tier_cert_url.strip()),
        id=f"DC{idx_zero_based+1}_TierCertification_Reference",
        desc="Provide URL reference confirming the Tier certification",
        parent=dc_node,
        critical=True
    )

    # 5) Tier certification is III or IV (critical, verified by the tier URL)
    tier_leaf = evaluator.add_leaf(
        id=f"DC{idx_zero_based+1}_TierCertification",
        desc="Data center holds Uptime Institute Tier III or Tier IV certification (either Design or Constructed Facility)",
        parent=dc_node,
        critical=True
    )
    tier_claim = (
        "This data center has Uptime Institute Tier III or Tier IV certification. "
        "Accept evidence of Design Documentation (TCDD) or Constructed Facility (TCCF) certifications."
    )
    await evaluator.verify(
        claim=tier_claim,
        node=tier_leaf,
        sources=dc.tier_cert_url,
        additional_instruction="Consider equivalent phrasing such as 'Tier 3', 'Tier 4', and official Uptime Institute certification listings."
    )

    # 6) Security compliance URL exists (critical)
    sec_ref_exists = evaluator.add_custom_node(
        result=bool(dc.security_cert_url and dc.security_cert_url.strip()),
        id=f"DC{idx_zero_based+1}_SecurityCompliance_Reference",
        desc="Provide URL reference confirming SOC 2 Type 2 and ISO 27001 certifications",
        parent=dc_node,
        critical=True
    )

    # 7) SOC 2 Type 2 (critical)
    soc_leaf = evaluator.add_leaf(
        id=f"DC{idx_zero_based+1}_SOC2",
        desc="Data center has SOC 2 Type 2 certification",
        parent=dc_node,
        critical=True
    )
    soc_claim = "The data center (facility/provider) has SOC 2 Type 2 certification."
    await evaluator.verify(
        claim=soc_claim,
        node=soc_leaf,
        sources=dc.security_cert_url,
        additional_instruction="Verify the presence of SOC 2 Type 2 (not Type 1). Accept provider-wide compliance pages if they explicitly cover colocation facilities."
    )

    # 8) ISO 27001 (critical)
    iso_leaf = evaluator.add_leaf(
        id=f"DC{idx_zero_based+1}_ISO27001",
        desc="Data center has ISO 27001 certification",
        parent=dc_node,
        critical=True
    )
    iso_claim = "The data center (facility/provider) has ISO 27001 certification."
    await evaluator.verify(
        claim=iso_claim,
        node=iso_leaf,
        sources=dc.security_cert_url,
        additional_instruction="Accept provider compliance pages or certification registries that explicitly list ISO 27001 coverage."
    )

    # Common sources to use when specific URLs are not provided beyond primary
    common_sources = _collect_sources(dc.location_ref_url, extra=dc.extra_urls)

    # 9) Carrier neutrality (critical)
    carrier_leaf = evaluator.add_leaf(
        id=f"DC{idx_zero_based+1}_CarrierNeutral",
        desc="Data center is carrier-neutral, allowing customers to choose from multiple network service providers",
        parent=dc_node,
        critical=True
    )
    carrier_claim = "This facility is carrier-neutral (offers access to multiple network service providers)."
    await evaluator.verify(
        claim=carrier_claim,
        node=carrier_leaf,
        sources=common_sources,
        additional_instruction="Look for phrases like 'carrier-neutral', 'multiple carriers', 'network-neutral', 'ecosystem of providers', or a carrier list."
    )

    # 10) Power density >= 8 kW per rack (critical)
    power_leaf = evaluator.add_leaf(
        id=f"DC{idx_zero_based+1}_PowerDensity",
        desc="Data center supports minimum power density of at least 8 kW per rack",
        parent=dc_node,
        critical=True
    )
    power_claim = "This facility supports a minimum power density of at least 8 kW per rack (>= 8 kW per rack)."
    await evaluator.verify(
        claim=power_claim,
        node=power_leaf,
        sources=common_sources,
        additional_instruction="Accept equivalent statements like '8 kVA per rack', '8+ kW per cabinet', 'up to 8 kW and above'."
    )

    # 11) Biometric access control (critical)
    bio_leaf = evaluator.add_leaf(
        id=f"DC{idx_zero_based+1}_Biometric",
        desc="Data center implements biometric access control systems for physical security",
        parent=dc_node,
        critical=True
    )
    bio_claim = "This facility implements biometric access control for physical security."
    await evaluator.verify(
        claim=bio_claim,
        node=bio_leaf,
        sources=common_sources,
        additional_instruction="Look for 'biometric' terms such as fingerprint, iris, palm-vein, or facial recognition in access control descriptions."
    )

    # 12) PCI DSS support (critical)
    pci_leaf = evaluator.add_leaf(
        id=f"DC{idx_zero_based+1}_PCICompliance",
        desc="Data center supports PCI DSS compliance requirements for customers",
        parent=dc_node,
        critical=True
    )
    pci_claim = "This facility/provider supports PCI DSS compliance requirements for customers."
    await evaluator.verify(
        claim=pci_claim,
        node=pci_leaf,
        sources=common_sources,
        additional_instruction="Accept explicit mentions of 'PCI DSS', 'PCI compliance support', or customer compliance enablement for PCI."
    )

    # 13) Remote hands 24/7 (critical)
    rh_leaf = evaluator.add_leaf(
        id=f"DC{idx_zero_based+1}_RemoteHands",
        desc="Data center offers 24/7 remote hands services",
        parent=dc_node,
        critical=True
    )
    rh_claim = "This facility offers 24/7 remote hands services."
    await evaluator.verify(
        claim=rh_claim,
        node=rh_leaf,
        sources=common_sources,
        additional_instruction="Look for 'remote hands', 'smart hands', '24x7 support' or similar wording."
    )

    # 14) Uptime SLA >= 99.99% (critical)
    sla_leaf = evaluator.add_leaf(
        id=f"DC{idx_zero_based+1}_UptimeSLA",
        desc="Data center guarantees minimum 99.99% uptime in service level agreement",
        parent=dc_node,
        critical=True
    )
    sla_claim = "This facility/provider guarantees a minimum of 99.99% uptime in its SLA."
    await evaluator.verify(
        claim=sla_claim,
        node=sla_leaf,
        sources=common_sources,
        additional_instruction="Accept phrases like 'four nines', '99.99% availability', or 'SLA 99.99%'."
    )

    # 15) Interconnection: MMR or cross-connect (critical)
    mmr_leaf = evaluator.add_leaf(
        id=f"DC{idx_zero_based+1}_MeetMeRoom",
        desc="Data center provides meet-me room (MMR) or cross-connect services for interconnection",
        parent=dc_node,
        critical=True
    )
    mmr_claim = "This facility provides a meet-me room (MMR) or cross-connect services for interconnection."
    await evaluator.verify(
        claim=mmr_claim,
        node=mmr_leaf,
        sources=common_sources,
        additional_instruction="Look for 'meet-me room', 'MMR', 'cross-connect', 'interconnection services', or similar."
    )

    # 16) Provider reputation: multi-market presence (critical)
    provider_leaf = evaluator.add_leaf(
        id=f"DC{idx_zero_based+1}_Provider",
        desc="Data center is operated by an established colocation provider with documented presence in multiple US markets",
        parent=dc_node,
        critical=True
    )
    provider_entity = dc.provider_name or "the facility's operator"
    provider_claim = (
        f"The data center is operated by {provider_entity} which has documented presence across multiple US markets in the United States."
    )
    provider_sources = _collect_sources(dc.provider_presence_url, dc.location_ref_url, extra=dc.extra_urls)
    await evaluator.verify(
        claim=provider_claim,
        node=provider_leaf,
        sources=provider_sources,
        additional_instruction="Accept provider 'locations' pages or market listings that clearly show multiple US cities/markets. Prefer official provider sources or highly reputable industry sources."
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
    Evaluate the answer against the colocation selection rubric.
    """
    # Initialize evaluator (root is non-critical parallel to allow partial credit across the three facilities)
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

    # Record allowed markets info for transparency
    evaluator.add_custom_info(
        info={"allowed_markets": ALLOWED_MARKETS},
        info_type="constraints",
        info_name="allowed_markets"
    )

    # Extract structured data centers
    extracted = await evaluator.extract(
        prompt=prompt_extract_datacenters(),
        template_class=ColocationExtraction,
        extraction_name="colocation_data_centers"
    )

    # Ensure we have exactly 3 items (pad with empty items if fewer)
    dcs: List[DataCenterItem] = list(extracted.datacenters[:3])
    while len(dcs) < 3:
        dcs.append(DataCenterItem())

    # Build verification subtrees for each data center
    for i, dc in enumerate(dcs):
        await verify_data_center(evaluator, root, dc, i)

    # Return structured evaluation summary
    return evaluator.get_summary()