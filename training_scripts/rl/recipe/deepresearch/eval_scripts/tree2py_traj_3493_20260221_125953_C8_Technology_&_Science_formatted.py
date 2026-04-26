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
TASK_ID = "us_major_carrier_5g_2024_2025"
TASK_DESCRIPTION = """
Identify three major wireless telecommunications carriers operating nationwide in the United States that meet ALL of the following technical deployment and regulatory criteria as documented at the end of 2024 or beginning of 2025:

1. The carrier must have deployed 5G network services using mid-band spectrum, specifically either the 2.5 GHz frequency range (Band 41) or the C-band frequency range (approximately 3.5-3.7 GHz)
2. The carrier's 5G network must provide coverage to at least 300 million people across the United States
3. The carrier must operate 4G LTE services on the 700 MHz frequency band (Band 13 for lower 700 MHz, or Bands 12/17 for upper 700 MHz)
4. The carrier must have deployed Standalone 5G (5G SA) network capabilities for at least business or commercial use
5. The carrier must be recognized as one of the three major nationwide wireless carriers in the United States
6. The carrier's network specifications and frequency band deployments must be publicly documented on their official website or through verifiable telecommunications industry sources

For each identified carrier, provide the carrier's name, verification that all six criteria are met, and URL references that document the carrier's network specifications and deployment information.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CarrierItem(BaseModel):
    """Represents a single carrier and associated source URLs explicitly cited in the answer."""
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CarriersExtraction(BaseModel):
    """List of carriers extracted from the answer."""
    carriers: List[CarrierItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_carriers() -> str:
    return """
    Extract up to three (3) major US nationwide wireless carriers explicitly mentioned in the answer, along with all URLs the answer cites for each carrier.

    For each carrier, extract:
    - name: The carrier's name exactly as written in the answer (e.g., "Verizon", "AT&T", "T-Mobile")
    - sources: An array of URLs that the answer associates with that carrier to document network specifications and deployment details.
      Include official webpages (e.g., carrier.com, newsroom, support pages), and reputable telecom industry sources (e.g., FCC, CTIA, GSMA, 3GPP, credible trade publications).
      Only include actual URLs present in the answer (plain or markdown links). Do not invent URLs.

    Return a JSON object:
    {
      "carriers": [
        {"name": "...", "sources": ["url1", "url2", ...]},
        ...
      ]
    }

    Rules:
    - If a carrier's name is present but no URLs are provided for it, return an empty array for sources.
    - If fewer than three carriers are provided in the answer, return what is available.
    - If more than three are provided, include the first three in order of appearance.
    - Only include valid URLs; ignore obviously malformed ones.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _sanitize_sources(sources: List[str], max_urls: int = 12) -> List[str]:
    """Keep only plausible HTTP(S) URLs and cap the list length to avoid excessive verification calls."""
    valid = []
    for u in sources:
        if isinstance(u, str) and u.strip():
            url = u.strip()
            if url.startswith("http://") or url.startswith("https://"):
                valid.append(url)
            else:
                # Best-effort normalization: prepend http:// if it looks like a domain/path
                if "://" not in url and "." in url:
                    valid.append(f"http://{url}")
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in valid:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped[:max_urls]


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_carrier(
    evaluator: Evaluator,
    parent_node,
    carrier: CarrierItem,
    index: int
) -> None:
    """
    Build verification sub-tree and run checks for one carrier.
    Follows the rubric's structure for Carrier_1, Carrier_2, Carrier_3 with parallel aggregation.
    """
    carrier_idx = index + 1
    carrier_node = evaluator.add_parallel(
        id=f"Carrier_{carrier_idx}",
        desc=f"{['First','Second','Third'][index]} major US carrier meeting all specified 5G deployment and technical criteria",
        parent=parent_node,
        critical=False  # Allow partial credit per carrier under the main task
    )

    # Existence gate to avoid meaningless verifications when no info is provided
    has_name = bool(carrier.name and carrier.name.strip())
    sources = _sanitize_sources(carrier.sources or [])
    has_sources = len(sources) > 0

    evaluator.add_custom_node(
        result=has_name and has_sources,
        id=f"Carrier_{carrier_idx}_info_provided",
        desc=f"Carrier #{carrier_idx} has name and at least one URL source",
        parent=carrier_node,
        critical=True
    )

    # Leaf nodes per rubric criteria (all critical under each carrier)
    # 1) Mid-band 5G deployment (2.5 GHz Band 41 or C-band ~3.5–3.7 GHz)
    midband_node = evaluator.add_leaf(
        id=f"MidBand_5G_Deployment_C{carrier_idx}",
        desc="Carrier has deployed 5G services using mid-band spectrum (either 2.5 GHz Band 41 or C-band 3.5-3.7 GHz)",
        parent=carrier_node,
        critical=True
    )

    # 2) Coverage reach ≥ 300 million people
    coverage_node = evaluator.add_leaf(
        id=f"Coverage_Reach_C{carrier_idx}",
        desc="Carrier's 5G network provides coverage to at least 300 million people in the United States",
        parent=carrier_node,
        critical=True
    )

    # 3) 700 MHz LTE operations (Band 13, 12, or 17)
    lte700_node = evaluator.add_leaf(
        id=f"700MHz_LTE_Operation_C{carrier_idx}",
        desc="Carrier operates 4G LTE services on 700 MHz frequency band (Band 13, 12, or 17)",
        parent=carrier_node,
        critical=True
    )

    # 4) Standalone 5G (SA) capability
    sa_node = evaluator.add_leaf(
        id=f"Standalone_5G_Capability_C{carrier_idx}",
        desc="Carrier has deployed Standalone 5G (5G SA) network capabilities",
        parent=carrier_node,
        critical=True
    )

    # 5) Recognized as one of the three major nationwide carriers
    major_node = evaluator.add_leaf(
        id=f"Major_Nationwide_Status_C{carrier_idx}",
        desc="Carrier is recognized as one of the three major nationwide wireless carriers in the United States",
        parent=carrier_node,
        critical=True
    )

    # 6) URL reference quality: provided URLs document network specs/deployments
    url_ref_node = evaluator.add_leaf(
        id=f"URL_Reference_C{carrier_idx}",
        desc="Provide URL reference documenting the carrier's network specifications and deployment information",
        parent=carrier_node,
        critical=True
    )

    # Prepare claims
    cname = carrier.name or ""

    claims_and_sources = [
        (
            f"The carrier {cname} has deployed 5G services using mid-band spectrum, including either 2.5 GHz (Band 41, NR n41) or the C-band around 3.5–3.7 GHz (NR n77).",
            sources,
            midband_node,
            "Confirm explicit mention of either: (a) 2.5 GHz/Band 41/n41, or (b) C-band ~3.5–3.7 GHz/n77. Prefer official or reputable industry documentation. Timeframe: end of 2024 or early 2025."
        ),
        (
            f"The carrier {cname}'s 5G network provides coverage to at least 300 million people in the United States.",
            sources,
            coverage_node,
            "Look for coverage statements like '300+ million', 'over 300 million', or specific counts ≥ 300,000,000. Minor rounding is acceptable."
        ),
        (
            f"The carrier {cname} operates 4G LTE service using the 700 MHz band (Band 13 or Bands 12/17).",
            sources,
            lte700_node,
            "Accept references to LTE Band 13 (lower 700 MHz) and/or Bands 12/17 (upper 700 MHz). Documentation can be network specs, frequency band listings, or official tech pages."
        ),
        (
            f"The carrier {cname} has deployed Standalone 5G (SA) capability for at least business or commercial use.",
            sources,
            sa_node,
            "Look for mention of '5G SA', 'Standalone 5G', and indications that it is deployed/available (enterprise, commercial, or consumer). Announcements or official documentation acceptable."
        ),
        (
            f"The carrier {cname} is recognized as one of the three major nationwide wireless carriers in the United States.",
            sources,
            major_node,
            "Use official/regulatory bodies (FCC/CTIA) or reputable industry analyses to confirm that the carrier is one of the three nationwide majors (commonly AT&T, Verizon, T‑Mobile)."
        ),
        (
            f"At least one of the provided URLs is an official or reputable telecom industry source that documents {cname}'s network specifications and deployment information.",
            sources,
            url_ref_node,
            "The page(s) should describe network bands, deployment details, or technical coverage info. Marketing pages are acceptable if they include concrete specs or deployment confirmations."
        ),
    ]

    # Run verifications; auto precondition will skip if info_provided failed
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate identification of three US major carriers that meet specified 5G deployment and technical criteria.
    """
    # Initialize evaluator; root node must be non-critical (framework requires critical parents to have all critical children)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Carriers evaluated independently
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

    # Create main task node (non-critical to allow partial credit)
    main_node = evaluator.add_parallel(
        id="US_Major_Carrier_5G_Deployment_Analysis",
        desc="Evaluate the identification of three major US wireless telecommunications carriers that meet specified 5G deployment and technical criteria",
        parent=root,
        critical=False
    )

    # Extract carriers and sources from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_carriers(),
        template_class=CarriersExtraction,
        extraction_name="carriers_extraction"
    )

    # Normalize: take first three carriers, pad if fewer
    carriers = list(extracted.carriers[:3])
    while len(carriers) < 3:
        carriers.append(CarrierItem())

    # Build verification subtrees
    for i, carrier in enumerate(carriers):
        await verify_carrier(evaluator, main_node, carrier, i)

    # Add task info to summary
    evaluator.add_ground_truth({
        "criteria": [
            "5G mid-band (2.5 GHz Band 41 or C-band ~3.5–3.7 GHz)",
            "5G coverage ≥ 300 million people (US)",
            "4G LTE on 700 MHz (Band 13/12/17)",
            "Standalone 5G (SA) deployed",
            "Recognized as one of the three major nationwide carriers (US)",
            "Publicly documented specs/deployments via official or reputable sources"
        ],
        "timeframe": "End of 2024 or beginning of 2025"
    })

    return evaluator.get_summary()