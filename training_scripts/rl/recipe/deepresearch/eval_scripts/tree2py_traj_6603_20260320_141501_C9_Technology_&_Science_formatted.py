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
TASK_ID = "5g_multi_zone_deployment"
TASK_DESCRIPTION = """
A telecommunications infrastructure company is deploying a 5G network across a major US metropolitan area and must procure base station equipment that meets multiple technical requirements. The deployment spans three distinct coverage zones, each with different specifications:

Zone 1 - Urban High-Capacity Core:
- Must support the C-Band frequency range used by all three major US carriers (Verizon, AT&T, T-Mobile) for mid-band 5G deployment
- Requires the maximum channel bandwidth capability supported by this band according to 3GPP 5G NR specifications
- Must use the appropriate duplex mode for this frequency range
- Must support 5G standalone (SA) architecture
- Should support carrier aggregation capability

Zone 2 - Suburban Balanced Coverage:
- Must support T-Mobile's primary mid-band spectrum at 2.5 GHz (the band acquired from Sprint merger)
- Must also support the 600 MHz low-band spectrum used for wide-area coverage
- Requires minimum 80 MHz channel bandwidth capability for the mid-band
- Must support both FDD and TDD duplex modes (as appropriate for each band)
- Must maintain latency within 5G target specifications

Zone 3 - High-Density mmWave Hotspot:
- Must support the mmWave band operating in the 37-40 GHz frequency range
- Requires minimum 400 MHz channel bandwidth capability
- Must be compatible with at least two major US carriers' mmWave deployments
- Must use TDD duplex mode
- Should support FR2 frequency range specifications

For each zone, identify and specify:
1. The specific 5G NR band number(s) (using 'n' prefix notation)
2. The exact operating frequency range in MHz or GHz as defined in 3GPP specifications
3. The duplex mode (FDD or TDD)
4. The channel bandwidth specification in MHz
5. Which major US carriers (Verizon, AT&T, and/or T-Mobile) deploy these bands
6. Provide reference URLs supporting each specification
"""

# Some canonical 3GPP facts for reference logging/ground truth
GROUND_TRUTH_REFERENCE = {
    "zone1": {
        "expected_band": "n77",
        "operating_freq_mhz": "3300–4200 MHz",
        "duplex": "TDD",
        "max_channel_bw_mhz": "100 MHz",
        "required_carriers": ["Verizon", "AT&T", "T-Mobile"]
    },
    "zone2": {
        "n41": {
            "band": "n41",
            "operating_freq_mhz": "2496–2690 MHz",
            "duplex": "TDD",
            "min_channel_bw_mhz": ">= 80 MHz (typ. up to 100 MHz)"
        },
        "n71": {
            "band": "n71",
            "operating_freq_mhz_ul": "663–698 MHz (UL)",
            "operating_freq_mhz_dl": "617–652 MHz (DL)",
            "duplex": "FDD",
            "typ_max_channel_bw_mhz": "up to 20 MHz"
        },
        "latency_target": "1–10 ms (5G target range)"
    },
    "zone3": {
        "expected_band": "n260",
        "operating_freq_ghz": "37–40 GHz",
        "duplex": "TDD",
        "min_channel_bw_mhz": ">= 400 MHz",
        "fr_range": "FR2",
        "carrier_requirement": ">= 2 major US carriers"
    }
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Zone1Spec(BaseModel):
    bands: List[str] = Field(default_factory=list)
    frequency_range: Optional[str] = None
    duplex_mode: Optional[str] = None
    channel_bandwidth_mhz: Optional[str] = None
    sa_support: Optional[str] = None  # e.g., "SA", "Standalone", "Yes", "Supported"
    ca_support: Optional[str] = None  # e.g., "CA", "Carrier Aggregation", "Yes", "Supported"
    carriers: List[str] = Field(default_factory=list)  # e.g., ["Verizon", "AT&T", "T-Mobile"]
    urls: List[str] = Field(default_factory=list)


class Zone2BandSpec(BaseModel):
    bands: List[str] = Field(default_factory=list)
    frequency_range: Optional[str] = None  # For n41, a single range string (e.g., "2496–2690 MHz")
    frequency_range_ul: Optional[str] = None  # For n71, uplink range
    frequency_range_dl: Optional[str] = None  # For n71, downlink range
    duplex_mode: Optional[str] = None
    channel_bandwidth_mhz: Optional[str] = None
    carriers: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class Zone2Spec(BaseModel):
    mid_band_n41: Optional[Zone2BandSpec] = None
    low_band_n71: Optional[Zone2BandSpec] = None
    latency_spec: Optional[str] = None  # e.g., "1–10 ms", "sub-10ms", etc.
    urls: List[str] = Field(default_factory=list)  # Additional zone-wide references


class Zone3Spec(BaseModel):
    bands: List[str] = Field(default_factory=list)
    frequency_range: Optional[str] = None
    duplex_mode: Optional[str] = None
    channel_bandwidth_mhz: Optional[str] = None
    carriers: List[str] = Field(default_factory=list)
    fr2_compliance: Optional[str] = None  # e.g., "FR2", "Yes", "Supported"
    urls: List[str] = Field(default_factory=list)


class SpecExtraction(BaseModel):
    zone1: Optional[Zone1Spec] = None
    zone2: Optional[Zone2Spec] = None
    zone3: Optional[Zone3Spec] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_specs() -> str:
    return """
    Extract the 5G deployment specifications mentioned in the answer for each zone. Return only what is explicitly stated.

    For Zone 1 (Urban High-Capacity Core), extract:
    - bands: list of 5G NR band numbers (e.g., ["n77"])
    - frequency_range: the stated operating frequency range text (e.g., "3300–4200 MHz" or "3.3–4.2 GHz")
    - duplex_mode: stated duplex mode text ("TDD" or "FDD")
    - channel_bandwidth_mhz: stated channel bandwidth text in MHz (e.g., "100 MHz", "up to 100")
    - sa_support: text indicating 5G Standalone (SA) support if present (e.g., "SA", "Standalone", "supported"), otherwise null
    - ca_support: text indicating carrier aggregation (CA) support if present, otherwise null
    - carriers: list of US carriers explicitly associated with this zone (e.g., ["Verizon","AT&T","T-Mobile"])
    - urls: list of all reference URLs cited for Zone 1

    For Zone 2 (Suburban Balanced Coverage), extract:
    mid_band_n41:
      - bands: list of band numbers for this mid-band (expect n41 if present)
      - frequency_range: stated frequency range text for n41 (e.g., "2496–2690 MHz")
      - duplex_mode: stated duplex mode ("TDD" expected for n41)
      - channel_bandwidth_mhz: stated channel bandwidth text (e.g., "80 MHz", "100 MHz", ">=80 MHz")
      - carriers: list of carriers explicitly associated with n41 (must include T-Mobile if stated)
      - urls: list of reference URLs for n41
    low_band_n71:
      - bands: list of band numbers for this low-band (expect n71 if present)
      - frequency_range_ul: stated uplink frequency range text (e.g., "663–698 MHz")
      - frequency_range_dl: stated downlink frequency range text (e.g., "617–652 MHz")
      - duplex_mode: stated duplex mode ("FDD" expected for n71)
      - channel_bandwidth_mhz: stated channel bandwidth text (e.g., "20 MHz")
      - carriers: list of carriers explicitly associated with n71 (should include T-Mobile if stated)
      - urls: list of reference URLs for n71
    - latency_spec: stated latency target or compliance text (e.g., "1–10 ms", "under 10 ms"), else null
    - urls: any additional Zone 2 references (deduplicate if overlapping)

    For Zone 3 (High-Density mmWave Hotspot), extract:
    - bands: list of 5G NR band numbers (e.g., ["n260"])
    - frequency_range: stated operating frequency range text (e.g., "37–40 GHz")
    - duplex_mode: stated duplex mode ("TDD" expected)
    - channel_bandwidth_mhz: stated channel bandwidth (e.g., "400 MHz", ">= 400 MHz")
    - carriers: list of US carriers explicitly associated with this mmWave band (at least two if stated)
    - fr2_compliance: text indicating FR2 support if present (e.g., "FR2", "FR2 compliant", "Supported"), else null
    - urls: list of all reference URLs cited for Zone 3

    Notes:
    - Only extract what is explicitly in the answer. Do not infer missing values.
    - For any missing item, return null or empty list as appropriate.
    - For URL extraction, include only valid URLs that are explicitly present in the answer (plain URL or in markdown).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_urls(*url_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str):
                u = u.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(bool((u or "").strip()) for u in urls)


# --------------------------------------------------------------------------- #
# Zone verifications                                                          #
# --------------------------------------------------------------------------- #
async def verify_zone1(evaluator: Evaluator, parent, z1: Optional[Zone1Spec]) -> None:
    node = evaluator.add_parallel(
        id="Zone_1_Urban_Core",
        desc="Zone 1: Urban high-capacity core requirements (C-band, max bandwidth, duplex mode, SA, carriers, references).",
        parent=parent,
        critical=False
    )

    urls = _merge_urls(z1.urls if z1 else [])

    # Reference URLs presence (critical gating)
    ref_node = evaluator.add_custom_node(
        result=_has_urls(urls),
        id="z1_reference_urls",
        desc="Zone 1: At least one reference URL is provided",
        parent=node,
        critical=True
    )

    # Band Number (n77)
    band_leaf = evaluator.add_leaf(
        id="z1_band_number",
        desc="Specifies the correct 5G NR band number(s) for the C-band zone (n77).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="5G NR band n77 corresponds to the C-band used for mid-band 5G in the United States.",
        node=band_leaf,
        sources=urls,
        additional_instruction="Accept mentions such as 'C-band 3.7 GHz' mapping to n77. The page should clearly indicate n77 for C-band in US context."
    )

    # Operating Frequency Range (n77: 3300–4200 MHz)
    freq_leaf = evaluator.add_leaf(
        id="z1_operating_frequency_range",
        desc="Specifies the correct 3GPP operating frequency range for n77 (3300–4200 MHz).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="5G NR band n77 has an operating frequency range of approximately 3300–4200 MHz (3.3–4.2 GHz).",
        node=freq_leaf,
        sources=urls,
        additional_instruction="Minor notation differences are acceptable (e.g., 3.3–4.2 GHz)."
    )

    # Duplex Mode (TDD)
    duplex_leaf = evaluator.add_leaf(
        id="z1_duplex_mode",
        desc="Specifies the appropriate duplex mode for the chosen C-band (n77 uses TDD).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="5G NR band n77 uses TDD (Time Division Duplex).",
        node=duplex_leaf,
        sources=urls,
        additional_instruction="Confirm that n77 is categorized as a TDD band."
    )

    # Channel Bandwidth (n77: 100 MHz in FR1)
    bw_leaf = evaluator.add_leaf(
        id="z1_channel_bandwidth",
        desc="Provides the maximum channel bandwidth supported by n77 per 3GPP (100 MHz).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In FR1, the maximum channel bandwidth for 5G NR band n77 is 100 MHz.",
        node=bw_leaf,
        sources=urls,
        additional_instruction="It's acceptable if the source states 'up to 100 MHz' for n77 in FR1."
    )

    # Standalone Architecture Support (SA)
    sa_leaf = evaluator.add_leaf(
        id="z1_sa_support",
        desc="Confirms support for 5G standalone (SA) architecture.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The referenced documentation indicates support for 5G Standalone (SA) architecture for the Zone 1 solution/band configuration.",
        node=sa_leaf,
        sources=urls,
        additional_instruction="Support can be indicated on vendor equipment pages or standards/overview pages explicitly mentioning SA capability."
    )

    # Carrier Aggregation Support (SHOULD)
    ca_leaf = evaluator.add_leaf(
        id="z1_ca_support",
        desc="Confirms carrier aggregation capability.",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="Carrier Aggregation (CA) capability is supported for the Zone 1 solution/band configuration.",
        node=ca_leaf,
        sources=urls,
        additional_instruction="CA may be referenced generally for the band or specifically for the equipment; either is acceptable."
    )

    # Carrier Deployments (must include Verizon, AT&T, and T-Mobile)
    carriers_leaf = evaluator.add_leaf(
        id="z1_carrier_deployments",
        desc="Lists which major US carriers deploy the identified band for Zone 1 (Verizon, AT&T, and T-Mobile).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In the United States, Verizon, AT&T, and T-Mobile deploy C-band 5G using NR band n77.",
        node=carriers_leaf,
        sources=urls,
        additional_instruction="A single authoritative page listing all three is ideal; otherwise, any page that clearly states that all three use n77 should be considered supporting."
    )


async def verify_zone2(evaluator: Evaluator, parent, z2: Optional[Zone2Spec]) -> None:
    node = evaluator.add_parallel(
        id="Zone_2_Suburban_Coverage",
        desc="Zone 2: Suburban balanced coverage requirements (n41 + n71, duplex modes, bandwidth, latency, carriers, references).",
        parent=parent,
        critical=False
    )

    urls_union = _merge_urls(
        z2.urls if z2 else [],
        z2.mid_band_n41.urls if (z2 and z2.mid_band_n41) else [],
        z2.low_band_n71.urls if (z2 and z2.low_band_n71) else []
    )

    # Reference URLs presence (critical gating)
    ref_node = evaluator.add_custom_node(
        result=_has_urls(urls_union),
        id="z2_reference_urls",
        desc="Zone 2: At least one reference URL is provided",
        parent=node,
        critical=True
    )

    # ---- Mid-band n41 ----
    n41_node = evaluator.add_parallel(
        id="z2_midband_n41",
        desc="Zone 2 Mid-band (n41) verification",
        parent=node,
        critical=True
    )

    n41_urls = _merge_urls(z2.mid_band_n41.urls if (z2 and z2.mid_band_n41) else [], urls_union)

    n41_band = evaluator.add_leaf(
        id="z2_n41_band_number",
        desc="Identifies the mid-band as n41.",
        parent=n41_node,
        critical=True
    )
    await evaluator.verify(
        claim="5G NR band n41 corresponds to the 2.5 GHz mid-band.",
        node=n41_band,
        sources=n41_urls,
        additional_instruction="The source should map 2.5 GHz (around 2500–2600 MHz) to band n41."
    )

    n41_freq = evaluator.add_leaf(
        id="z2_n41_operating_frequency_range",
        desc="Specifies the n41 operating frequency range (2496–2690 MHz).",
        parent=n41_node,
        critical=True
    )
    await evaluator.verify(
        claim="5G NR band n41 has an operating frequency range of approximately 2496–2690 MHz.",
        node=n41_freq,
        sources=n41_urls,
        additional_instruction="Minor notation differences are acceptable."
    )

    n41_duplex = evaluator.add_leaf(
        id="z2_n41_duplex_mode",
        desc="Specifies the appropriate duplex mode for n41 (TDD).",
        parent=n41_node,
        critical=True
    )
    await evaluator.verify(
        claim="5G NR band n41 uses TDD (Time Division Duplex).",
        node=n41_duplex,
        sources=n41_urls,
        additional_instruction="Confirm n41 is categorized as a TDD band."
    )

    n41_bw = evaluator.add_leaf(
        id="z2_n41_channel_bandwidth",
        desc="Specifies a mid-band channel bandwidth meeting the minimum requirement (>= 80 MHz).",
        parent=n41_node,
        critical=True
    )
    await evaluator.verify(
        claim="5G NR band n41 supports channel bandwidths of at least 80 MHz (commonly up to 100 MHz).",
        node=n41_bw,
        sources=n41_urls,
        additional_instruction="Any statement indicating >=80 MHz, or 'up to 100 MHz' should be considered compliant."
    )

    n41_carriers = evaluator.add_leaf(
        id="z2_n41_carrier_deployments",
        desc="Lists which major US carriers deploy n41 (must include T-Mobile).",
        parent=n41_node,
        critical=True
    )
    await evaluator.verify(
        claim="T-Mobile deploys 2.5 GHz mid-band 5G using NR band n41 in the United States.",
        node=n41_carriers,
        sources=n41_urls,
        additional_instruction="The page should explicitly connect T-Mobile's 2.5 GHz holdings (from Sprint) to band n41."
    )

    # ---- Low-band n71 ----
    n71_node = evaluator.add_parallel(
        id="z2_lowband_n71",
        desc="Zone 2 Low-band (n71) verification",
        parent=node,
        critical=True
    )

    n71_urls = _merge_urls(z2.low_band_n71.urls if (z2 and z2.low_band_n71) else [], urls_union)

    n71_band = evaluator.add_leaf(
        id="z2_n71_band_number",
        desc="Identifies the low-band as n71.",
        parent=n71_node,
        critical=True
    )
    await evaluator.verify(
        claim="5G NR band n71 corresponds to 600 MHz low-band spectrum.",
        node=n71_band,
        sources=n71_urls,
        additional_instruction="The source should clearly identify n71 as the 600 MHz NR band."
    )

    n71_freq = evaluator.add_leaf(
        id="z2_n71_operating_frequency_range",
        desc="Specifies the n71 operating frequency ranges (UL 663–698 MHz and DL 617–652 MHz).",
        parent=n71_node,
        critical=True
    )
    await evaluator.verify(
        claim="5G NR band n71 frequency ranges are approximately UL 663–698 MHz and DL 617–652 MHz.",
        node=n71_freq,
        sources=n71_urls,
        additional_instruction="Accept formats listing separate UL/DL ranges for n71."
    )

    n71_duplex = evaluator.add_leaf(
        id="z2_n71_duplex_mode",
        desc="Specifies the appropriate duplex mode for n71 (FDD).",
        parent=n71_node,
        critical=True
    )
    await evaluator.verify(
        claim="5G NR band n71 uses FDD (Frequency Division Duplex).",
        node=n71_duplex,
        sources=n71_urls,
        additional_instruction="Confirm n71 is categorized as an FDD band."
    )

    n71_bw = evaluator.add_leaf(
        id="z2_n71_channel_bandwidth",
        desc="Specifies a channel bandwidth for n71 in MHz (consistent with constraints).",
        parent=n71_node,
        critical=True
    )
    await evaluator.verify(
        claim="5G NR band n71 supports channel bandwidths up to around 20 MHz.",
        node=n71_bw,
        sources=n71_urls,
        additional_instruction="Any source indicating typical NR n71 bandwidths (e.g., 5/10/15/20 MHz) should be considered valid."
    )

    n71_carriers = evaluator.add_leaf(
        id="z2_n71_carrier_deployments",
        desc="Lists which major US carriers deploy n71 (must include T-Mobile).",
        parent=n71_node,
        critical=True
    )
    await evaluator.verify(
        claim="T-Mobile deploys 600 MHz 5G using NR band n71 in the United States.",
        node=n71_carriers,
        sources=n71_urls,
        additional_instruction="The page should explicitly tie T-Mobile's 600 MHz to band n71."
    )

    # Latency compliance within 5G targets
    latency_leaf = evaluator.add_leaf(
        id="z2_latency_compliance",
        desc="Confirms latency is within 5G target specifications (about 1–10 ms).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="5G target user-plane latency is on the order of 1–10 ms; the Zone 2 plan is aligned with these 5G latency targets.",
        node=latency_leaf,
        sources=urls_union,
        additional_instruction="Standards/overview sources (3GPP/ITU/vendor whitepapers) indicating 5G latency targets are acceptable."
    )


async def verify_zone3(evaluator: Evaluator, parent, z3: Optional[Zone3Spec]) -> None:
    node = evaluator.add_parallel(
        id="Zone_3_mmWave_Hotspot",
        desc="Zone 3: High-density mmWave hotspot requirements (37–40 GHz, bandwidth >=400 MHz, >=2 carriers, TDD, FR2 SHOULD, references).",
        parent=parent,
        critical=False
    )

    urls = _merge_urls(z3.urls if z3 else [])

    # Reference URLs presence (critical gating)
    ref_node = evaluator.add_custom_node(
        result=_has_urls(urls),
        id="z3_reference_urls",
        desc="Zone 3: At least one reference URL is provided",
        parent=node,
        critical=True
    )

    # Band Number (n260)
    band_leaf = evaluator.add_leaf(
        id="z3_band_number",
        desc="Specifies the correct 5G NR band number(s) matching 37–40 GHz (n260).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="5G NR band n260 corresponds to the 37–40 GHz mmWave range.",
        node=band_leaf,
        sources=urls,
        additional_instruction="The page should explicitly map n260 to 37–40 GHz."
    )

    # Operating Frequency Range (37–40 GHz)
    freq_leaf = evaluator.add_leaf(
        id="z3_operating_frequency_range",
        desc="Specifies the correct operating frequency range for the selected mmWave band (37–40 GHz).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The operating frequency range for NR band n260 is approximately 37–40 GHz.",
        node=freq_leaf,
        sources=urls,
        additional_instruction="Minor notation differences are acceptable."
    )

    # Duplex Mode (TDD)
    duplex_leaf = evaluator.add_leaf(
        id="z3_duplex_mode",
        desc="Specifies TDD duplex mode for the mmWave band.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="5G NR band n260 uses TDD (Time Division Duplex).",
        node=duplex_leaf,
        sources=urls,
        additional_instruction="Confirm n260 is categorized as a TDD band."
    )

    # Channel Bandwidth (>= 400 MHz)
    bw_leaf = evaluator.add_leaf(
        id="z3_channel_bandwidth",
        desc="Specifies channel bandwidth meeting the minimum requirement (>= 400 MHz).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="5G NR band n260 supports at least 400 MHz channel bandwidth (e.g., 400 or 800 MHz).",
        node=bw_leaf,
        sources=urls,
        additional_instruction="Any authoritative indication of ≥400 MHz for n260 suffices."
    )

    # Carrier Deployments (>= 2 major US carriers)
    carriers_leaf = evaluator.add_leaf(
        id="z3_carrier_deployments",
        desc="Lists at least two major US carriers that deploy the identified mmWave band.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="At least two major US carriers (for example, Verizon and AT&T, or T-Mobile) deploy the mmWave 37–40 GHz NR band n260 in the United States.",
        node=carriers_leaf,
        sources=urls,
        additional_instruction="Any page clearly indicating real-world n260 deployments by ≥2 of Verizon/AT&T/T-Mobile qualifies."
    )

    # FR2 Compliance (SHOULD)
    fr2_leaf = evaluator.add_leaf(
        id="z3_fr2_compliance",
        desc="Confirms support for FR2 specifications.",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="NR band n260 is within the FR2 frequency range.",
        node=fr2_leaf,
        sources=urls,
        additional_instruction="A straightforward mapping of n260 to FR2 is sufficient."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 5G Multi-Zone Deployment task using the obj_task_eval framework.
    """

    # Initialize evaluator (root: parallel, non-critical to allow partial credit across zones)
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

    # Record ground truth info for reference
    evaluator.add_ground_truth(GROUND_TRUTH_REFERENCE, gt_type="canonical_3gpp_reference")

    # Extract structured specs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_specs(),
        template_class=SpecExtraction,
        extraction_name="spec_extraction"
    )

    # Build verification tree for each zone
    await verify_zone1(evaluator, root, extracted.zone1)
    await verify_zone2(evaluator, root, extracted.zone2)
    await verify_zone3(evaluator, root, extracted.zone3)

    # Return evaluation summary
    return evaluator.get_summary()