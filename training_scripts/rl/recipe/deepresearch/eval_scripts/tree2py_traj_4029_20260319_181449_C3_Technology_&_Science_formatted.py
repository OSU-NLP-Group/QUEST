import asyncio
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "5g_urban_deployment_spec"
TASK_DESCRIPTION = (
    "A telecommunications company plans to deploy cellular network coverage in a 2 square kilometer downtown urban area. "
    "The deployment must achieve a maximum latency of 10 milliseconds and maintain a minimum signal strength of -85 dBm throughout the coverage area. "
    "Determine the complete technical deployment specification, including: 1. The network technology generation (e.g., LTE, 5G) and justification, "
    "2. The cell deployment type (macro cell or small cell) and spectrum band range, 3. The coverage radius per cell in meters, "
    "4. The coverage area per individual cell in square kilometers, 5. The total number of base stations required, "
    "6. The resulting base station density (base stations per square kilometer), 7. The compliant antenna height range in meters, "
    "8. Verification that the proposed deployment achieves the required signal strength and latency. "
    "Provide all numerical calculations and reference the technical specifications that support your determinations."
)

TOTAL_AREA_KM2 = 2.0
TARGET_LATENCY_MS_MAX = 10.0
TARGET_SIGNAL_DBM_MIN = -85.0
TARGET_HIGH_BAND_LO_GHZ = 24.0
TARGET_HIGH_BAND_HI_GHZ = 40.0
REFERENCE_APPROX_RADIUS_M = 100.0


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DeploymentExtraction(BaseModel):
    # Network tech
    technology_generation: Optional[str] = None
    technology_justification_urls: List[str] = Field(default_factory=list)

    # Deployment choices
    cell_type: Optional[str] = None
    spectrum_band_range: Optional[str] = None
    spectrum_support_urls: List[str] = Field(default_factory=list)

    # Coverage parameters
    coverage_radius_m: Optional[str] = None
    coverage_radius_support_urls: List[str] = Field(default_factory=list)
    coverage_area_per_cell_km2: Optional[str] = None

    # Stationing
    base_station_count: Optional[str] = None
    base_station_density_per_km2: Optional[str] = None
    density_reference_urls: List[str] = Field(default_factory=list)

    # Antenna
    antenna_height_min_m: Optional[str] = None
    antenna_height_max_m: Optional[str] = None
    antenna_height_support_urls: List[str] = Field(default_factory=list)

    # Performance
    performance_support_urls: List[str] = Field(default_factory=list)
    signal_strength_requirement_dbm: Optional[str] = None
    latency_requirement_ms: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_deployment_spec() -> str:
    return """
Extract the complete deployment specification details as explicitly stated in the answer. Return the following JSON fields:

1) technology_generation: The selected network generation (e.g., "5G", "LTE").
2) technology_justification_urls: An array of URLs that the answer cites to justify the technology choice (especially latency-related evidence).
3) cell_type: The selected cell deployment type (e.g., "small cell", "macro cell").
4) spectrum_band_range: The chosen spectrum band or range text exactly as in the answer (e.g., "28 GHz", "24–39 GHz", "n258 (26 GHz)").
5) spectrum_support_urls: An array of URLs that support the spectrum choice (e.g., mmWave suitability for low-latency urban 5G).
6) coverage_radius_m: The per-cell coverage radius, as written in the answer (include units if present; do not invent).
7) coverage_radius_support_urls: URLs that support or justify the selected coverage radius (e.g., typical mmWave cell range statements).
8) coverage_area_per_cell_km2: The per-cell coverage area in square kilometers as stated in the answer (include exact formatting; do not convert).
9) base_station_count: The total number of base stations required as stated.
10) base_station_density_per_km2: The station density (base stations per square kilometer) as stated.
11) density_reference_urls: URLs that the answer cites for the "typical urban 5G density" comparison or standard range.
12) antenna_height_min_m: The minimum antenna height in meters as stated (include units if present; do not convert).
13) antenna_height_max_m: The maximum antenna height in meters as stated.
14) antenna_height_support_urls: URLs that support antenna height compliance (e.g., ITU-R P.1410).
15) performance_support_urls: URLs that support the performance claims (signal strength and/or latency).
16) signal_strength_requirement_dbm: The minimum signal strength requirement as stated in the answer (e.g., "-85 dBm").
17) latency_requirement_ms: The maximum latency requirement as stated in the answer (e.g., "10 ms").

Rules:
- Extract only what appears explicitly in the answer. If any item is missing, set it to null (for strings) or [] (for URL lists).
- For URL arrays, include only valid HTTP/HTTPS URLs that appear in the answer (plain or markdown).
- Do not infer or calculate values that are not present in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper parsing and math utilities                                           #
# --------------------------------------------------------------------------- #
def parse_first_float(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    m = re.search(r'[-+]?\d*\.?\d+', value.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def find_numbers_with_units(s: str) -> List[Tuple[float, Optional[str]]]:
    """
    Return list of (number, unit_str_or_None). Recognizes GHz/MHz if present.
    """
    if not s:
        return []
    pattern = re.compile(r'([-+]?\d*\.?\d+)\s*(GHz|ghz|MHZ|Mhz|MHz)?')
    out: List[Tuple[float, Optional[str]]] = []
    for m in pattern.finditer(s.replace(",", "")):
        num = m.group(1)
        unit = m.group(2)
        try:
            out.append((float(num), unit))
        except Exception:
            continue
    return out


def parse_band_range_ghz(band_str: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Attempt to parse a GHz range (min,max) from a free-text band string.
    Heuristics:
    - If units are 'MHz', convert to GHz by dividing by 1000.
    - If no units, assume GHz.
    - If only one number present, min==max.
    - If numbers > 1000, assume MHz and convert to GHz.
    """
    if not band_str:
        return None, None
    pairs = find_numbers_with_units(band_str)
    ghz_vals: List[float] = []
    for val, unit in pairs:
        if unit and unit.lower() == "mhz":
            ghz_vals.append(val / 1000.0)
        else:
            # No unit or GHz -> assume GHz; also reduce numbers > 1000 as MHz heuristic
            if val > 1000:
                ghz_vals.append(val / 1000.0)
            else:
                ghz_vals.append(val)
    if not ghz_vals:
        return None, None
    if len(ghz_vals) == 1:
        return ghz_vals[0], ghz_vals[0]
    return min(ghz_vals), max(ghz_vals)


def area_from_radius_km2(radius_m: float) -> float:
    return math.pi * (radius_m ** 2) / 1_000_000.0


def approx_equal(a: float, b: float, rel_tol: float = 0.15, abs_tol: float = 0.0) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b), 1e-12))


def combine_sources(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for url in lst or []:
            if isinstance(url, str):
                u = url.strip()
                if u and u not in seen:
                    seen.add(u)
                    out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_network_technology_determination(evaluator: Evaluator, parent, spec: DeploymentExtraction) -> None:
    nt_node = evaluator.add_sequential(
        id="Network_Technology_Determination",
        desc="Determination of appropriate network technology based on requirements",
        parent=parent,
        critical=True
    )

    # Technology_Generation (split into multiple leaf checks)
    tg_node = evaluator.add_sequential(
        id="Technology_Generation",
        desc="Network generation identified as 5G based on latency requirement under 10ms, with justification that 5G achieves latency as low as 1ms while LTE provides 50-100ms",
        parent=nt_node,
        critical=True
    )

    # 1) The answer selects 5G
    tech_is_5g_leaf = evaluator.add_leaf(
        id="Tech_Is_5G",
        desc="Answer explicitly selects 5G as the network generation",
        parent=tg_node,
        critical=True
    )
    tech_claim = f"The answer selects 5G (5G NR) as the network technology generation. Extracted value: {spec.technology_generation or 'null'}."
    await evaluator.verify(
        claim=tech_claim,
        node=tech_is_5g_leaf,
        additional_instruction="Judge this purely by checking the ANSWER text. Accept '5G', '5G NR', or equivalent phrasing as a positive match."
    )

    # 2) Sources provided for latency justification
    tech_sources = combine_sources(spec.technology_justification_urls, spec.performance_support_urls)
    evaluator.add_custom_node(
        result=bool(tech_sources),
        id="Technology_Justification_Sources_Provided",
        desc="Latency justification sources provided for 5G vs LTE",
        parent=tg_node,
        critical=True
    )

    # 3) Latency justification supported by sources
    latency_just_leaf = evaluator.add_leaf(
        id="Technology_Latency_Justified",
        desc="5G latency as low as ~1 ms and LTE ~50–100 ms supported by sources; thus <10 ms requirement is met",
        parent=tg_node,
        critical=True
    )
    latency_just_claim = (
        "5G wireless networks can achieve end-to-end latency as low as around 1 millisecond, while LTE networks typically "
        "have latency on the order of about 50–100 milliseconds. Therefore, selecting 5G satisfies a maximum latency "
        "requirement under 10 milliseconds."
    )
    await evaluator.verify(
        claim=latency_just_claim,
        node=latency_just_leaf,
        sources=tech_sources,
        additional_instruction="Look for quantitative latency figures on the cited pages. Minor differences in exact figures are acceptable if they convey the same conclusion."
    )

    # Cell_Type_And_Spectrum
    cts_node = evaluator.add_parallel(
        id="Cell_Type_And_Spectrum",
        desc="Cell deployment type and spectrum band determination for urban 5G",
        parent=nt_node,
        critical=True
    )

    # Cell_Type_Selection
    cell_type_leaf = evaluator.add_leaf(
        id="Cell_Type_Selection",
        desc="Cell type identified as small cell appropriate for downtown urban 5G deployment",
        parent=cts_node,
        critical=True
    )
    cell_type_claim = f"The deployment uses small cells (not macro cells). Extracted cell type: {spec.cell_type or 'null'}."
    await evaluator.verify(
        claim=cell_type_claim,
        node=cell_type_leaf,
        additional_instruction="Judge this purely by checking the ANSWER text for 'small cell' (or equivalent)."
    )

    # Spectrum_Band_Selection (split into numeric check + source support)
    sbs_node = evaluator.add_sequential(
        id="Spectrum_Band_Selection",
        desc="Spectrum band identified as high-band (24-40 GHz range) for low-latency urban 5G deployment",
        parent=cts_node,
        critical=True
    )

    min_ghz, max_ghz = parse_band_range_ghz(spec.spectrum_band_range or "")
    in_high_band = (
        min_ghz is not None and max_ghz is not None and
        TARGET_HIGH_BAND_LO_GHZ <= min_ghz <= TARGET_HIGH_BAND_HI_GHZ and
        TARGET_HIGH_BAND_LO_GHZ <= max_ghz <= TARGET_HIGH_BAND_HI_GHZ
    )
    evaluator.add_custom_node(
        result=in_high_band,
        id="Spectrum_Band_In_High_Band",
        desc=f"Spectrum band '{spec.spectrum_band_range or 'null'}' falls within 24–40 GHz mmWave range",
        parent=sbs_node,
        critical=True
    )

    spectrum_sources = combine_sources(spec.spectrum_support_urls)
    evaluator.add_custom_node(
        result=bool(spectrum_sources),
        id="Spectrum_Sources_Provided",
        desc="Sources provided supporting high-band/mmWave spectrum choice",
        parent=sbs_node,
        critical=True
    )

    spectrum_supported_leaf = evaluator.add_leaf(
        id="Spectrum_Band_Supported",
        desc="High-band (24–40 GHz) choice supported by cited sources",
        parent=sbs_node,
        critical=True
    )
    spectrum_claim = (
        f"The proposed spectrum band for the deployment is in the 24–40 GHz high-band (mmWave) range "
        f"({spec.spectrum_band_range or 'unspecified'}), appropriate for low-latency urban 5G small-cell deployments."
    )
    await evaluator.verify(
        claim=spectrum_claim,
        node=spectrum_supported_leaf,
        sources=spectrum_sources,
        additional_instruction="Verify the cited pages indicate the band lies in (or is recognized as) mmWave/high-band around 24–40 GHz and is suitable for low-latency urban small-cell 5G."
    )


async def build_coverage_analysis(evaluator: Evaluator, parent, spec: DeploymentExtraction) -> None:
    cov_node = evaluator.add_sequential(
        id="Coverage_Analysis",
        desc="Coverage parameters and base station requirements calculation",
        parent=parent,
        critical=True
    )

    # Coverage_Radius_Determination
    rad_node = evaluator.add_sequential(
        id="Coverage_Radius_Determination",
        desc="Coverage radius per cell determined based on 5G high-band specifications, approximately 100 meters",
        parent=cov_node,
        critical=True
    )

    radius_val = parse_first_float(spec.coverage_radius_m)
    approx_100_ok = radius_val is not None and (70.0 <= radius_val <= 150.0)
    evaluator.add_custom_node(
        result=approx_100_ok,
        id="Radius_Approx_100m",
        desc=f"Per-cell coverage radius is approximately 100 m (stated: {spec.coverage_radius_m or 'null'})",
        parent=rad_node,
        critical=True
    )

    radius_sources = combine_sources(spec.coverage_radius_support_urls, spec.spectrum_support_urls)
    evaluator.add_custom_node(
        result=bool(radius_sources),
        id="Radius_Sources_Provided",
        desc="Sources provided supporting mmWave small-cell radius scale",
        parent=rad_node,
        critical=True
    )

    radius_supported_leaf = evaluator.add_leaf(
        id="Radius_Sources_Verified",
        desc="Typical 5G high-band small-cell radius around ~100 m supported by cited sources",
        parent=rad_node,
        critical=True
    )
    radius_supported_claim = "In urban 5G high-band/mmWave deployments, a typical small-cell coverage radius is on the order of ~100 meters (roughly 100–200 m)."
    await evaluator.verify(
        claim=radius_supported_claim,
        node=radius_supported_leaf,
        sources=radius_sources,
        additional_instruction="Look for explicit mmWave/small-cell range statements or examples in urban contexts near ~100 m."
    )

    # Coverage_Area_Calculation
    area_node = evaluator.add_sequential(
        id="Coverage_Area_Calculation",
        desc="Coverage area per individual cell calculated using πr², yielding approximately 0.0314 km² for 100m radius",
        parent=cov_node,
        critical=True
    )
    stated_area = parse_first_float(spec.coverage_area_per_cell_km2)
    computed_area = area_from_radius_km2(radius_val) if radius_val is not None else None
    area_ok = (
        stated_area is not None and computed_area is not None and
        approx_equal(stated_area, computed_area, rel_tol=0.15, abs_tol=0.005)
    )
    evaluator.add_custom_node(
        result=area_ok,
        id="Area_Equals_PiR2",
        desc=f"Per-cell coverage area matches πr² using the stated radius (stated={spec.coverage_area_per_cell_km2 or 'null'}, computed≈{computed_area:.6f} km²)" if computed_area is not None else "Per-cell coverage area πr² check (insufficient data)",
        parent=area_node,
        critical=True
    )

    # Base_Station_Count
    bsc_node = evaluator.add_sequential(
        id="Base_Station_Count",
        desc="Total number of base stations determined by dividing total area by coverage per cell",
        parent=cov_node,
        critical=True
    )
    # Derivation
    count_float = parse_first_float(spec.base_station_count)
    count_int = int(round(count_float)) if count_float is not None else None

    effective_area = None
    if stated_area and stated_area > 0:
        effective_area = stated_area
    elif computed_area and computed_area > 0:
        effective_area = computed_area

    expected_cells = math.ceil(TOTAL_AREA_KM2 / effective_area) if effective_area else None
    deriv_ok = (expected_cells is not None) and (count_int is not None) and (count_int == expected_cells)
    evaluator.add_custom_node(
        result=deriv_ok,
        id="Station_Count_Derivation",
        desc=f"Base stations computed from 2.0 km² / per-cell area equals stated count (expected={expected_cells}, stated={spec.base_station_count or 'null'})",
        parent=bsc_node,
        critical=True
    )

    # Density_Calculation (split: numeric correctness + reference support)
    dens_node = evaluator.add_sequential(
        id="Density_Calculation",
        desc="Base station density calculated in BS/km² and compared against urban 5G standard range of 40-50 BS/km²",
        parent=bsc_node,
        critical=True
    )

    density_val = parse_first_float(spec.base_station_density_per_km2)
    expected_density = None
    if count_int is not None:
        expected_density = count_int / TOTAL_AREA_KM2
    elif expected_cells is not None:
        expected_density = expected_cells / TOTAL_AREA_KM2

    density_ok = (
        density_val is not None and expected_density is not None and
        approx_equal(density_val, expected_density, rel_tol=0.1, abs_tol=0.5)
    )
    evaluator.add_custom_node(
        result=density_ok,
        id="Density_Correct",
        desc=f"Base station density equals count / total area (expected≈{expected_density}, stated={spec.base_station_density_per_km2 or 'null'})",
        parent=dens_node,
        critical=True
    )

    density_sources = combine_sources(spec.density_reference_urls)
    evaluator.add_custom_node(
        result=bool(density_sources),
        id="Density_Standard_Sources_Provided",
        desc="Sources provided for typical urban 5G density range reference",
        parent=dens_node,
        critical=True
    )

    density_supported_leaf = evaluator.add_leaf(
        id="Density_Standard_Supported",
        desc="Urban 5G small-cell typical density range around 40–50 BS/km² is supported by cited sources",
        parent=dens_node,
        critical=True
    )
    density_supported_claim = (
        "Authoritative references indicate that typical urban 5G small‑cell site densities can be on the order of roughly "
        "40–50 base stations per square kilometer (allowing minor variation)."
    )
    await evaluator.verify(
        claim=density_supported_claim,
        node=density_supported_leaf,
        sources=density_sources,
        additional_instruction="Accept reasonable equivalence (e.g., 'tens per km²' with concrete examples). Prefer explicit numeric ranges when available."
    )


async def build_compliance_and_performance(evaluator: Evaluator, parent, spec: DeploymentExtraction) -> None:
    cap_node = evaluator.add_parallel(
        id="Compliance_And_Performance_Verification",
        desc="Regulatory compliance and performance requirement validation",
        parent=parent,
        critical=True
    )

    # Antenna height compliance
    ant_node = evaluator.add_sequential(
        id="Antenna_Height_Compliance",
        desc="Antenna height range specified in compliance with ITU-R P.1410 (15–60 m), urban typically 25–35 m",
        parent=cap_node,
        critical=True
    )
    min_h = parse_first_float(spec.antenna_height_min_m)
    max_h = parse_first_float(spec.antenna_height_max_m)
    height_ok = (min_h is not None and max_h is not None and 15.0 <= min_h < max_h <= 60.0)
    evaluator.add_custom_node(
        result=height_ok,
        id="Antenna_Range_Within_15_60",
        desc=f"Antenna height range within 15–60 m (stated: min={spec.antenna_height_min_m or 'null'}, max={spec.antenna_height_max_m or 'null'})",
        parent=ant_node,
        critical=True
    )

    ant_sources = combine_sources(spec.antenna_height_support_urls)
    evaluator.add_custom_node(
        result=bool(ant_sources),
        id="Antenna_Sources_Provided",
        desc="Sources provided supporting ITU‑R P.1410 antenna height compliance",
        parent=ant_node,
        critical=True
    )

    ant_supported_leaf = evaluator.add_leaf(
        id="Antenna_Compliance_Supported",
        desc="Antenna height range complies with ITU‑R P.1410 (15–60 m), typical 25–35 m for urban, supported by sources",
        parent=ant_node,
        critical=True
    )
    ant_claim = (
        "According to ITU‑R P.1410 (and related guidance), base‑station antenna heights in urban cellular planning are "
        "typically within an approximate 15–60 meter range, with urban deployments often around 25–35 meters. The proposed "
        "antenna height range complies with these recommendations."
    )
    await evaluator.verify(
        claim=ant_claim,
        node=ant_supported_leaf,
        sources=ant_sources,
        additional_instruction="Check that the cited document(s) mention antenna heights in this range for urban cellular (planning or modeling assumptions)."
    )

    # Performance verification: signal strength + latency
    perf_node = evaluator.add_parallel(
        id="Performance_Verification",
        desc="Verification that deployment achieves required signal strength and latency",
        parent=cap_node,
        critical=True
    )

    # Signal strength
    sig_node = evaluator.add_sequential(
        id="Signal_Strength_Verification",
        desc="Verification that minimum signal strength of -85 dBm is achievable with specified deployment",
        parent=perf_node,
        critical=True
    )
    perf_sources = combine_sources(spec.performance_support_urls, spec.technology_justification_urls)
    evaluator.add_custom_node(
        result=bool(perf_sources),
        id="Signal_Sources_Provided",
        desc="Sources provided supporting -85 dBm achievability",
        parent=sig_node,
        critical=True
    )
    ss_val = parse_first_float(spec.signal_strength_requirement_dbm) or TARGET_SIGNAL_DBM_MIN
    radius_val = parse_first_float(spec.coverage_radius_m)
    radius_txt = f"~{radius_val:.0f} m" if radius_val is not None else "~100 m"
    sig_leaf = evaluator.add_leaf(
        id="Signal_Strength_Achievable",
        desc="Proposed deployment achieves ≥ -85 dBm across coverage area, supported by sources",
        parent=sig_node,
        critical=True
    )
    sig_claim = (
        f"In urban 5G high-band small-cell deployments (radius {radius_txt}), achieving a minimum received signal strength "
        f"of about {ss_val:.0f} dBm (or better) across the coverage area is feasible; −85 dBm is generally considered a "
        f"'good' cellular signal level."
    )
    await evaluator.verify(
        claim=sig_claim,
        node=sig_leaf,
        sources=perf_sources,
        additional_instruction="Look for link budget examples, mmWave/small-cell performance, or references stating that around −85 dBm is achievable/considered good in cellular contexts."
    )

    # Latency
    lat_node = evaluator.add_sequential(
        id="Latency_Verification",
        desc="Verification that maximum latency under 10ms is achievable with 5G deployment",
        parent=perf_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(perf_sources),
        id="Latency_Sources_Provided",
        desc="Sources provided supporting <10 ms latency for 5G",
        parent=lat_node,
        critical=True
    )
    latency_req = parse_first_float(spec.latency_requirement_ms) or TARGET_LATENCY_MS_MAX
    lat_leaf = evaluator.add_leaf(
        id="Latency_Under_10ms",
        desc="5G can achieve <10 ms latency (as low as ~1 ms), supported by sources",
        parent=lat_node,
        critical=True
    )
    lat_claim = (
        f"5G networks can achieve end‑to‑end latency under {latency_req:.0f} milliseconds (with URLLC profiles as low as ~1 ms), "
        f"meeting the specified latency requirement."
    )
    await evaluator.verify(
        claim=lat_claim,
        node=lat_leaf,
        sources=perf_sources,
        additional_instruction="Look for explicit latency figures for 5G (e.g., 'as low as 1 ms', 'below 10 ms')."
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
    # Initialize evaluator (root is a container; add a critical child as the true task root)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # top container
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

    # Add the critical task node mirroring the rubric's top-level
    task_root = evaluator.add_sequential(
        id="Deployment_Specification",
        desc="Complete 5G deployment specification for 2 km² urban area meeting latency and signal strength requirements",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extracted_spec = await evaluator.extract(
        prompt=prompt_extract_deployment_spec(),
        template_class=DeploymentExtraction,
        extraction_name="deployment_spec_extraction"
    )

    # Record ground-truth constraints for reference (not used for scoring directly)
    evaluator.add_ground_truth({
        "total_area_km2": TOTAL_AREA_KM2,
        "required_latency_ms_max": TARGET_LATENCY_MS_MAX,
        "required_signal_strength_dbm_min": TARGET_SIGNAL_DBM_MIN,
        "target_high_band_ghz_range": [TARGET_HIGH_BAND_LO_GHZ, TARGET_HIGH_BAND_HI_GHZ],
        "reference_approx_radius_m": REFERENCE_APPROX_RADIUS_M
    })

    # Build subtrees according to rubric
    await build_network_technology_determination(evaluator, task_root, extracted_spec)
    await build_coverage_analysis(evaluator, task_root, extracted_spec)
    await build_compliance_and_performance(evaluator, task_root, extracted_spec)

    # Return the structured evaluation summary
    return evaluator.get_summary()