import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cloud_gaming_us_multiplayer_infra"
TASK_DESCRIPTION = """
Design a cloud gaming infrastructure configuration for deploying competitive multiplayer game servers across the United States. Your solution must specify:
1. Cloud Provider: Select one major cloud provider (AWS, Azure, or Google Cloud) that offers dedicated game server hosting services.
2. Geographic Deployment: Identify specific data center regions from your chosen provider that provide coverage across at least three distinct US geographic areas (East Coast, West Coast, and Central US), ensuring low-latency access for players nationwide.
3. Server Instance Specifications: Specify the instance type or server configuration that meets the following technical requirements: Minimum 4 CPU cores (or 8+ for larger scale), Minimum 16GB RAM (or 32-64GB for larger scale), SSD or NVMe storage, Network capability to support <50-100ms latency for competitive gaming, Bandwidth to support minimum 3-5 Mbps upload per concurrent player.
4. Capacity Planning: Demonstrate that the infrastructure can handle 1000-2000 concurrent player connections per server instance (typical for MMORPG-style games) and supports horizontal scaling.
5. Supporting Documentation: Provide reference URLs from the cloud provider's official documentation to verify: Geographic data center locations and regional availability, Server instance specifications (CPU, RAM, storage type), Gaming service capabilities (if applicable, e.g., AWS GameLift, Azure PlayFab), Concurrent user capacity or scaling documentation.

Your answer should present a cohesive infrastructure configuration that balances technical performance requirements with geographic coverage needs for competitive multiplayer gaming deployment.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CloudInfraExtraction(BaseModel):
    # Provider and gaming service
    provider: Optional[str] = None
    gaming_service: Optional[str] = None

    # Geographic deployment
    east_regions: List[str] = Field(default_factory=list)
    central_regions: List[str] = Field(default_factory=list)
    west_regions: List[str] = Field(default_factory=list)
    latency_explanation: Optional[str] = None

    # Server instance / configuration
    instance_type: Optional[str] = None
    cpu_cores: Optional[str] = None
    ram: Optional[str] = None
    storage_type: Optional[str] = None
    network_latency_target_ms: Optional[str] = None
    per_player_upload_mbps: Optional[str] = None
    high_speed_connectivity: Optional[str] = None

    # Capacity and scaling
    per_instance_concurrency: Optional[str] = None
    horizontal_scaling: Optional[str] = None
    ram_scaling_rule: Optional[str] = None

    # Cost/Tradeoff
    cost_tradeoff: Optional[str] = None

    # URLs from official provider docs (as presented in the answer)
    regions_urls: List[str] = Field(default_factory=list)
    instance_specs_urls: List[str] = Field(default_factory=list)
    gaming_service_urls: List[str] = Field(default_factory=list)
    scaling_capacity_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_cloud_infra() -> str:
    return """
Extract the cloud gaming infrastructure details from the answer.

Return a JSON object with the following fields strictly as strings or string arrays where applicable. Extract text as written in the answer; do not infer or invent.

Provider and gaming service:
- provider: The single chosen cloud provider (e.g., "AWS", "Amazon Web Services", "Azure", "Microsoft Azure", "Google Cloud", "GCP")
- gaming_service: The named dedicated/managed game server hosting service (e.g., "AWS GameLift", "Azure PlayFab Multiplayer Servers", "Google Cloud Game Servers/Agones") if provided; else null

Geographic deployment (region codes or location names exactly as written in the answer):
- east_regions: array of provider region names/ids or locations explicitly assigned to cover the US East Coast (e.g., "us-east-1", "N. Virginia", "eastus", "us-east4")
- central_regions: array of provider region names/ids or locations explicitly assigned to cover the US Central (e.g., "us-central1", "Iowa", "centralus", "us-east-2 (Ohio)" if used as central by the answer)
- west_regions: array of provider region names/ids or locations explicitly assigned to cover the US West Coast (e.g., "us-west-1", "N. California", "westus2", "us-west2")
- latency_explanation: brief excerpt where the answer explains how these region choices reduce latency nationwide (or null)

Server instance/configuration:
- instance_type: the specific instance type or configuration name (e.g., "c6i.4xlarge", "E2-standard-16", "D8s v5") if present; else null
- cpu_cores: the stated CPU core or vCPU figure (as text, e.g., "8 vCPU", "16 cores") if present; else null
- ram: the stated RAM figure (as text, e.g., "32 GB", "64 GiB") if present; else null
- storage_type: the stated storage type (as text, e.g., "NVMe SSD", "EBS gp3 SSD") if present; else null
- network_latency_target_ms: the latency target (as text, e.g., "<100 ms", "50–80 ms") if present; else null
- per_player_upload_mbps: the stated per-player upstream bandwidth planning (as text, e.g., "3–5 Mbps per player", "≥3 Mbps") if present; else null
- high_speed_connectivity: any mention/justification of high‑speed connectivity (as text, e.g., "10/25/40 Gbps NIC", "gigabit uplinks") if present; else null

Capacity and scaling:
- per_instance_concurrency: the stated per-instance concurrent players target/capacity (as text, e.g., "1,500 CCU") if present; else null
- horizontal_scaling: brief excerpt describing the horizontal scaling approach (autoscaling, fleets, Kubernetes/Agones scaling, PlayFab MPS scaling, etc.) if present; else null
- ram_scaling_rule: excerpt showing the "≈1 GB RAM per 20 concurrent players" rule is addressed (or a reasoned alternative) if present; else null

Cost:
- cost_tradeoff: excerpt discussing cost vs performance/reliability tradeoffs, instance sizing choices, autoscaling, or purchase model (e.g., Reserved/Spot/Preemptible) if present; else null

Official provider documentation URLs (only official domains; extract all URLs from the answer and categorize):
- regions_urls: array of official provider URLs verifying regions/locations/availability
- instance_specs_urls: array of official provider URLs verifying instance specifications (CPU/RAM/storage/NIC)
- gaming_service_urls: array of official provider URLs for the gaming service capabilities (GameLift, PlayFab MPS, GCP Game Servers/Agones)
- scaling_capacity_urls: array of official provider URLs about scaling, autoscaling, capacity, or concurrency guidance

If something is not clearly present, set it to null or an empty array.
Only include URLs explicitly present in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
_ALLOWED_PROVIDERS = {
    "aws": "AWS",
    "amazon web services": "AWS",
    "amazon": "AWS",
    "azure": "Azure",
    "microsoft azure": "Azure",
    "google cloud": "Google Cloud",
    "google cloud platform": "Google Cloud",
    "gcp": "Google Cloud",
}


def normalize_provider_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    key = name.strip().lower()
    return _ALLOWED_PROVIDERS.get(key, None)


def extract_numbers_with_units(text: Optional[str]) -> List[float]:
    """
    Extract numeric values, supporting 'k' suffix (e.g., 1k = 1000).
    """
    if not text:
        return []
    nums = []
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(k|K|m|M)?', text):
        val = float(m.group(1))
        suf = m.group(2)
        if suf:
            if suf.lower() == 'k':
                val *= 1000.0
            elif suf.lower() == 'm':
                val *= 1_000_000.0
        nums.append(val)
    return nums


def parse_first_float(text: Optional[str]) -> Optional[float]:
    vals = extract_numbers_with_units(text)
    return vals[0] if vals else None


def parse_latency_upper_bound_ms(text: Optional[str]) -> Optional[float]:
    """
    Try to infer the maximum latency target from text like "<100 ms", "50-80ms", "50–100 ms".
    Returns the upper bound if a range is present; else the single value.
    """
    if not text:
        return None
    # Normalize unicode dashes
    s = text.lower().replace("–", "-").replace("—", "-").replace("to", "-")
    # If comparator like "<" or "<=", get the number
    m = re.search(r'(<\s*=?\s*)(\d+(?:\.\d+)?)\s*ms', s)
    if m:
        try:
            return float(m.group(2))
        except Exception:
            pass
    # Range like "50-80 ms"
    m = re.search(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*ms', s)
    if m:
        try:
            lo = float(m.group(1))
            hi = float(m.group(2))
            return max(lo, hi)
        except Exception:
            pass
    # Single value
    m = re.search(r'(\d+(?:\.\d+)?)\s*ms', s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return None


def storage_mentions_ssd_or_nvme(text: Optional[str]) -> bool:
    if not text:
        return False
    s = text.lower()
    return any(k in s for k in ["ssd", "nvme", "gp3", "gp2"])


def mentions_high_speed_connectivity(text: Optional[str]) -> bool:
    if not text:
        return False
    s = text.lower()
    return any(k in s for k in [
        "gbps", "gigabit", "10gbe", "25gbe", "40gbe", "100gbe", "10 gb", "25 gb", "40 gb", "100 gb",
        "10-gig", "25-gig", "40-gig", "100-gig", "nic", "ena", "accelerated networking"
    ])


def addresses_ram_scaling_rule(text: Optional[str]) -> bool:
    """
    True if explicitly mentions approx '1GB per 20 players' or a clear memory-per-player rationale.
    """
    if not text:
        return False
    s = text.lower().replace(" ", "")
    if "1gbper20" in s or "20playersper1gb" in s or "1gibper20" in s:
        return True
    # Generic memory-per-player rationale
    s2 = text.lower()
    if ("per player" in s2 or "per-player" in s2) and ("ram" in s2 or "memory" in s2) and re.search(r'\d+\s*(mb|gb|gib|mib)', s2):
        return True
    return False


def concurrency_in_required_range(text: Optional[str], lo: int = 1000, hi: int = 2000) -> bool:
    vals = extract_numbers_with_units(text)
    if not vals:
        return False
    # Accept if any numeric figure falls within [lo, hi]
    return any(lo <= v <= hi for v in vals)


def network_capability_ok(lat_text: Optional[str], up_text: Optional[str]) -> bool:
    """
    Check that latency target <= 100ms and per-player upload >= 3 Mbps.
    """
    lat = parse_latency_upper_bound_ms(lat_text)
    up = parse_first_float(up_text)
    return (lat is not None and lat <= 100.0) and (up is not None and up >= 3.0)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_provider_selection(evaluator: Evaluator, parent, extracted: CloudInfraExtraction):
    node = evaluator.add_parallel(
        id="provider_selection",
        desc="Selects an allowed major cloud provider and identifies its dedicated/managed game server hosting service",
        parent=parent,
        critical=True,
    )

    # provider_is_allowed
    canon = normalize_provider_name(extracted.provider)
    evaluator.add_custom_node(
        result=(canon is not None),
        id="provider_is_allowed",
        desc="Selects exactly one cloud provider and it is one of: AWS, Azure, or Google Cloud",
        parent=node,
        critical=True,
    )

    # dedicated_gaming_service_identified (verify with official URLs if provided)
    service_leaf = evaluator.add_leaf(
        id="dedicated_gaming_service_identified",
        desc="Identifies a dedicated/managed game server hosting service (or clearly equivalent offering) from the chosen provider",
        parent=node,
        critical=True,
    )
    service_name = extracted.gaming_service or ""
    provider_name = canon or (extracted.provider or "the chosen provider")
    await evaluator.verify(
        claim=f"The service named '{service_name}' is an official managed/dedicated game server hosting service (or equivalent offering) from {provider_name}. Examples include AWS GameLift, Azure PlayFab Multiplayer Servers, or Google Cloud Game Servers (Agones).",
        node=service_leaf,
        sources=extracted.gaming_service_urls,
        additional_instruction="Verify the page is an official provider documentation site (e.g., aws.amazon.com or docs.aws.amazon.com; azure.microsoft.com or learn.microsoft.com; cloud.google.com). The page should explicitly describe a managed/dedicated game server capability."
    )


async def build_geographic_deployment(evaluator: Evaluator, parent, extracted: CloudInfraExtraction):
    node = evaluator.add_parallel(
        id="geographic_deployment",
        desc="Deployment uses specific provider regions/locations to cover required US geographic areas with low-latency intent",
        parent=parent,
        critical=True,
    )

    # regions_cover_east_central_west
    regions_leaf = evaluator.add_leaf(
        id="regions_cover_east_central_west",
        desc="Specifies concrete provider regions/locations that collectively cover at least three distinct US geographic areas: East Coast, Central US, and West Coast",
        parent=node,
        critical=True,
    )
    east = extracted.east_regions or []
    central = extracted.central_regions or []
    west = extracted.west_regions or []
    await evaluator.verify(
        claim=(
            f"The deployment specifies provider regions/locations for three distinct US areas: "
            f"East Coast={east}, Central US={central}, West Coast={west}. "
            f"These are valid regions/locations of the chosen provider and collectively cover East, Central, and West US."
        ),
        node=regions_leaf,
        sources=extracted.regions_urls,
        additional_instruction=(
            "Judge whether each listed region/location is a real region/location for the chosen provider and that, together, "
            "they represent East, Central, and West coverage of the United States. Use reasonable geography (e.g., N. Virginia/East; "
            "Iowa/Central; N. California/West). Verify against official provider region/location documentation."
        )
    )

    # latency_consideration_explained
    latency_leaf = evaluator.add_leaf(
        id="latency_consideration_explained",
        desc="Explains how region/location choices are intended to reduce latency for players nationwide (e.g., proximity to player populations/traffic routing rationale)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer contains an explicit explanation of how the selected regions/locations reduce latency nationwide (e.g., proximity to player populations, peering, routing).",
        node=latency_leaf,
        additional_instruction="Base your judgment only on the provided answer text. A brief rationale is sufficient if it clearly links chosen regions to latency reduction."
    )


async def build_server_specs(evaluator: Evaluator, parent, extracted: CloudInfraExtraction):
    node = evaluator.add_parallel(
        id="server_instance_specifications",
        desc="Specifies server instance type/configuration meeting compute, memory, storage, and network/bandwidth requirements",
        parent=parent,
        critical=True,
    )

    # instance_type_or_config_named (existence)
    evaluator.add_custom_node(
        result=bool(extracted.instance_type and extracted.instance_type.strip()),
        id="instance_type_or_config_named",
        desc="Provides a specific instance type or clearly defined server configuration from the chosen provider/service",
        parent=node,
        critical=True,
    )

    # cpu_requirement_met (>=4 vCPU)
    cpu_leaf = evaluator.add_leaf(
        id="cpu_requirement_met",
        desc="Meets the CPU requirement: at least 4 CPU cores (or explicitly chooses/justifies 8+ cores for larger scale)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The chosen instance/configuration '{extracted.instance_type}' provides at least 4 vCPU/cores (8+ acceptable for larger scale) according to official provider specs.",
        node=cpu_leaf,
        sources=extracted.instance_specs_urls,
        additional_instruction="Confirm via the official instance specification page(s) that the instance size referenced in the answer has >= 4 vCPUs/cores."
    )

    # ram_requirement_met (>=16 GB)
    ram_leaf = evaluator.add_leaf(
        id="ram_requirement_met",
        desc="Meets the RAM requirement: at least 16GB RAM (or explicitly chooses/justifies 32–64GB for larger scale)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The chosen instance/configuration '{extracted.instance_type}' provides at least 16 GB of memory (32–64 GB acceptable for larger scale) according to official provider specs.",
        node=ram_leaf,
        sources=extracted.instance_specs_urls,
        additional_instruction="Confirm via the official instance specification page(s) that memory is >= 16 GB."
    )

    # storage_requirement_met (SSD or NVMe)
    storage_leaf = evaluator.add_leaf(
        id="storage_requirement_met",
        desc="Specifies SSD or NVMe storage",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The storage specified in the answer is SSD or NVMe (e.g., NVMe local SSD, EBS gp3/gp2 SSD) for the chosen instance/configuration '{extracted.instance_type}'.",
        node=storage_leaf,
        sources=extracted.instance_specs_urls,
        additional_instruction="Look for explicit mention of SSD/NVMe or EBS SSD families (gp3/gp2) in the official documentation for the instance/storage options used."
    )

    # network_capability_specified (latency target and per-player upload)
    net_ok = network_capability_ok(extracted.network_latency_target_ms, extracted.per_player_upload_mbps)
    evaluator.add_custom_node(
        result=net_ok,
        id="network_capability_specified",
        desc="Specifies/justifies networking capability/architecture intended to support competitive play latency (<50–100ms) and per-player bandwidth planning (≥3–5 Mbps upload per concurrent player)",
        parent=node,
        critical=True,
    )

    # high_speed_connectivity_addressed
    evaluator.add_custom_node(
        result=mentions_high_speed_connectivity(extracted.high_speed_connectivity),
        id="high_speed_connectivity_addressed",
        desc="Addresses the constraint about dedicated high-speed (gigabit-level) connectivity where relevant to the deployment (e.g., server networking/ingress/egress/venue uplinks as applicable)",
        parent=node,
        critical=True,
    )


async def build_capacity_and_scaling(evaluator: Evaluator, parent, extracted: CloudInfraExtraction):
    node = evaluator.add_parallel(
        id="capacity_and_scaling",
        desc="Addresses per-instance concurrent player capacity and horizontal scaling requirements",
        parent=parent,
        critical=True,
    )

    # per_instance_concurrency_target
    evaluator.add_custom_node(
        result=concurrency_in_required_range(extracted.per_instance_concurrency, 1000, 2000),
        id="per_instance_concurrency_target",
        desc="Demonstrates/plans for 1000–2000 concurrent player connections per server instance",
        parent=node,
        critical=True,
    )

    # horizontal_scaling_supported (verify with gaming service URLs if available)
    hscale_leaf = evaluator.add_leaf(
        id="horizontal_scaling_supported",
        desc="Describes a horizontal scaling approach appropriate to the chosen provider/service",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The chosen provider/service supports horizontal scaling (e.g., fleets/autoscaling for GameLift/PlayFab, Agones/Kubernetes scaling), matching the approach described in the answer.",
        node=hscale_leaf,
        sources=(extracted.gaming_service_urls or extracted.scaling_capacity_urls),
        additional_instruction="Verify via official provider docs that the referenced service or stack supports horizontal scaling/autoscaling as described."
    )

    # ram_scaling_rule_addressed
    evaluator.add_custom_node(
        result=addresses_ram_scaling_rule(extracted.ram_scaling_rule),
        id="ram_scaling_rule_addressed",
        desc="Addresses the constraint: approximately 1GB RAM per 20 concurrent players (uses it in sizing or explicitly explains why a different sizing model is used)",
        parent=node,
        critical=True,
    )


async def build_cost_optimization(evaluator: Evaluator, parent, extracted: CloudInfraExtraction):
    node = evaluator.add_parallel(
        id="cost_optimization",
        desc="Balances performance requirements against infrastructure costs as required by constraints",
        parent=parent,
        critical=True,
    )

    # cost_tradeoff_discussed
    cost_leaf = evaluator.add_leaf(
        id="cost_tradeoff_discussed",
        desc="Discusses cost vs performance/reliability tradeoffs in the proposed infrastructure (e.g., instance sizing choice, scaling strategy, purchase model)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer discusses cost versus performance/reliability tradeoffs (e.g., instance sizing, autoscaling strategy, Reserved/Spot/Preemptible choices).",
        node=cost_leaf,
        additional_instruction="Base solely on the provided answer text; a brief but explicit discussion satisfies this requirement."
    )


async def _verify_official_docs_category(
    evaluator: Evaluator,
    parent,
    id_prefix: str,
    desc: str,
    urls: List[str],
    claim: str,
    add_ins: str
):
    """
    Helper for supporting documentation checks. If urls list is empty, add a failing custom node;
    otherwise verify by URLs.
    """
    if not urls:
        evaluator.add_custom_node(
            result=False,
            id=f"{id_prefix}_present",
            desc=f"{desc} — at least one official documentation URL is provided",
            parent=parent,
            critical=True,
        )
        return

    leaf = evaluator.add_leaf(
        id=id_prefix,
        desc=desc,
        parent=parent,
        critical=True,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=add_ins
    )


async def build_supporting_documentation(evaluator: Evaluator, parent, extracted: CloudInfraExtraction):
    node = evaluator.add_parallel(
        id="supporting_documentation",
        desc="Provides official provider documentation URLs that verify required claims",
        parent=parent,
        critical=True,
    )

    # 1) Regions / locations
    await _verify_official_docs_category(
        evaluator,
        node,
        id_prefix="official_regions_docs",
        desc="Official provider documentation URL(s) verify regions/locations and regional availability",
        urls=extracted.regions_urls,
        claim="This page is an official provider documentation/resource listing cloud regions/locations/availability in the United States.",
        add_ins="Ensure the domain is official (aws.amazon.com / docs.aws.amazon.com; azure.microsoft.com / learn.microsoft.com; cloud.google.com). The content should list regions/locations."
    )

    # 2) Instance specifications
    await _verify_official_docs_category(
        evaluator,
        node,
        id_prefix="official_instance_specs_docs",
        desc="Official provider documentation URL(s) verify instance specifications (CPU, RAM, storage/NIC)",
        urls=extracted.instance_specs_urls,
        claim="This page is an official provider documentation/resource showing instance specifications such as vCPU/cores, memory (GB), storage type (SSD/NVMe/EBS), and/or network capabilities.",
        add_ins="Confirm it is an official provider page and includes instance spec tables or descriptions with CPU, RAM, storage, or NIC info."
    )

    # 3) Gaming service capabilities
    await _verify_official_docs_category(
        evaluator,
        node,
        id_prefix="official_gaming_service_docs",
        desc="Official provider documentation URL(s) verify gaming service capabilities",
        urls=extracted.gaming_service_urls,
        claim="This page is an official provider documentation/resource that describes the managed/dedicated game server hosting service capabilities (e.g., GameLift, PlayFab Multiplayer Servers, Google Cloud Game Servers/Agones).",
        add_ins="Confirm it is an official provider page and it explicitly describes the game server service, features, and/or usage."
    )

    # 4) Scaling/capacity guidance
    await _verify_official_docs_category(
        evaluator,
        node,
        id_prefix="official_scaling_capacity_docs",
        desc="Official provider documentation URL(s) verify scaling/capacity or autoscaling guidance",
        urls=extracted.scaling_capacity_urls,
        claim="This page is an official provider documentation/resource that discusses scaling (autoscaling, fleets, multi-server scaling) and/or capacity/concurrency guidance relevant to the proposed solution.",
        add_ins="Confirm it is an official provider page and it covers scaling/autoscaling or capacity considerations."
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
    Evaluate an answer for the cloud gaming infrastructure configuration task.
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

    # Use a critical top-level aggregator node under the framework's root
    critical_root = evaluator.add_parallel(
        id="root_checks",
        desc="Cloud gaming infrastructure configuration satisfies the proposed question and listed constraints",
        parent=root,
        critical=True,
    )

    # Extraction
    extracted: CloudInfraExtraction = await evaluator.extract(
        prompt=prompt_extract_cloud_infra(),
        template_class=CloudInfraExtraction,
        extraction_name="cloud_infra_extraction",
    )

    # Provider selection
    await build_provider_selection(evaluator, critical_root, extracted)

    # Geographic deployment
    await build_geographic_deployment(evaluator, critical_root, extracted)

    # Server instance specifications
    await build_server_specs(evaluator, critical_root, extracted)

    # Capacity and scaling
    await build_capacity_and_scaling(evaluator, critical_root, extracted)

    # Cost optimization
    await build_cost_optimization(evaluator, critical_root, extracted)

    # Supporting documentation
    await build_supporting_documentation(evaluator, critical_root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()