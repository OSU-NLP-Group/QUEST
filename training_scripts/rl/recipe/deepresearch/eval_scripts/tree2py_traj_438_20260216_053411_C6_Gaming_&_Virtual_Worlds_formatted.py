import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "college_esports_programs_identification"
TASK_DESCRIPTION = (
    "Identify three college esports programs in the United States that meet comprehensive institutional, "
    "facility, technical, accessibility, and competitive program requirements, and provide supporting URLs."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProgramFacility(BaseModel):
    facility_name: Optional[str] = None
    facility_urls: List[str] = Field(default_factory=list)
    station_count: Optional[str] = None
    station_count_urls: List[str] = Field(default_factory=list)


class HardwareSpec(BaseModel):
    cpu: Optional[str] = None
    cpu_urls: List[str] = Field(default_factory=list)
    ram: Optional[str] = None
    ram_urls: List[str] = Field(default_factory=list)
    gpu: Optional[str] = None
    gpu_urls: List[str] = Field(default_factory=list)
    storage: Optional[str] = None
    storage_urls: List[str] = Field(default_factory=list)


class MonitorSpec(BaseModel):
    refresh_rate: Optional[str] = None
    resolution: Optional[str] = None
    monitor_urls: List[str] = Field(default_factory=list)


class InternetSpec(BaseModel):
    upload_speed: Optional[str] = None
    dedicated_conn: Optional[bool] = None
    internet_urls: List[str] = Field(default_factory=list)


class AccessibilityInfo(BaseModel):
    feature_1: Optional[str] = None
    feature_1_urls: List[str] = Field(default_factory=list)
    feature_2: Optional[str] = None
    feature_2_urls: List[str] = Field(default_factory=list)


class CompetitionInfo(BaseModel):
    game_1: Optional[str] = None
    game_1_urls: List[str] = Field(default_factory=list)
    game_2: Optional[str] = None
    game_2_urls: List[str] = Field(default_factory=list)


class ProgramItem(BaseModel):
    institution_name: Optional[str] = None
    state: Optional[str] = None
    nace_urls: List[str] = Field(default_factory=list)
    location_urls: List[str] = Field(default_factory=list)
    facility: ProgramFacility = ProgramFacility()
    hardware: HardwareSpec = HardwareSpec()
    monitors: MonitorSpec = MonitorSpec()
    internet: InternetSpec = InternetSpec()
    accessibility: AccessibilityInfo = AccessibilityInfo()
    competition: CompetitionInfo = CompetitionInfo()
    other_urls: List[str] = Field(default_factory=list)


class ProgramsExtraction(BaseModel):
    programs: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return (
        "Extract up to three college esports programs (only the first three if more are present) from the answer, "
        "capturing all required fields and supporting URLs. Return an array 'programs' of objects with the following fields:\n"
        "1) institution_name: The institution’s name\n"
        "2) state: The U.S. state where the institution is located (string; do not infer; use the answer text)\n"
        "3) nace_urls: Array of URLs that specifically confirm NACE membership for the institution\n"
        "4) location_urls: Array of URLs that document the institution’s location/state\n"
        "5) facility: { facility_name, facility_urls, station_count, station_count_urls }\n"
        "   - facility_name: Named esports arena/facility on campus\n"
        "   - facility_urls: URLs describing the dedicated esports facility (not a general lab)\n"
        "   - station_count: Number of gaming stations (string as mentioned)\n"
        "   - station_count_urls: URLs supporting the number of stations\n"
        "6) hardware: { cpu, cpu_urls, ram, ram_urls, gpu, gpu_urls, storage, storage_urls }\n"
        "   - cpu: Text describing CPU (e.g., model, core count); do not infer beyond the answer\n"
        "   - cpu_urls: URLs supporting CPU specifications\n"
        "   - ram: Text describing RAM amount (e.g., '16GB')\n"
        "   - ram_urls: URLs supporting RAM specifications\n"
        "   - gpu: Text describing GPU (e.g., model)\n"
        "   - gpu_urls: URLs supporting GPU specifications\n"
        "   - storage: Text describing storage type (e.g., 'SSD')\n"
        "   - storage_urls: URLs supporting storage type\n"
        "7) monitors: { refresh_rate, resolution, monitor_urls }\n"
        "   - refresh_rate: Text describing monitor refresh rate (e.g., '144Hz')\n"
        "   - resolution: Text describing monitor resolution (e.g., '1920x1080')\n"
        "   - monitor_urls: URLs supporting monitor specifications\n"
        "8) internet: { upload_speed, dedicated_conn, internet_urls }\n"
        "   - upload_speed: Text describing upload speed (e.g., '10 Mbps')\n"
        "   - dedicated_conn: Boolean if the answer explicitly states a dedicated high-speed connection (true/false/null)\n"
        "   - internet_urls: URLs supporting internet connectivity details\n"
        "9) accessibility: { feature_1, feature_1_urls, feature_2, feature_2_urls }\n"
        "   - feature_1/feature_2: Text describing each accessibility feature (e.g., 'ADA-compliant access', 'ergonomic chairs', 'controller support', 'customizable displays')\n"
        "   - feature_1_urls/feature_2_urls: URLs supporting each feature\n"
        "10) competition: { game_1, game_1_urls, game_2, game_2_urls }\n"
        "   - game_1/game_2: Esports titles actively competed in (e.g., 'League of Legends', 'Valorant')\n"
        "   - game_1_urls/game_2_urls: URLs showing competition/participation in these titles\n"
        "11) other_urls: Any additional relevant URLs provided in the answer.\n\n"
        "Rules:\n"
        "- Extract only what appears in the answer verbatim; do not invent or infer.\n"
        "- URLs may be plain or markdown; extract the actual URL.\n"
        "- If any item is missing, return null for that field or an empty array for URLs.\n"
        "- Keep numbers as strings to avoid mis-parsing; do not normalize values.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def nonempty_urls(*url_lists: List[str]) -> List[str]:
    """Return the first non-empty URL list among the provided lists; otherwise empty list."""
    for lst in url_lists:
        if lst and len(lst) > 0:
            return lst
    return []


def safe_facility_name(p: ProgramItem) -> str:
    return p.facility.facility_name or "the esports facility"


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_institutional_eligibility(evaluator: Evaluator, parent, p: ProgramItem, idx: int):
    node = evaluator.add_sequential(
        id=f"program_{idx}_institutional_eligibility",
        desc="Verify the program's institutional credentials and location",
        parent=parent,
        critical=True
    )

    # NACE Membership group
    nace_group = evaluator.add_parallel(
        id=f"program_{idx}_nace_membership",
        desc="Confirm the institution is a current NACE member",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(p.nace_urls),
        id=f"program_{idx}_nace_reference_url",
        desc="Provide URL confirming NACE membership status",
        parent=nace_group,
        critical=True
    )
    nace_leaf = evaluator.add_leaf(
        id=f"program_{idx}_nace_member_leaf",
        desc="Institution is a current NACE member",
        parent=nace_group,
        critical=True
    )
    claim = f"The institution '{p.institution_name or 'Unknown'}' is a current member of the National Association of Collegiate Esports (NACE)."
    await evaluator.verify(
        claim=claim,
        node=nace_leaf,
        sources=p.nace_urls,
        additional_instruction=(
            "Check the provided URLs for explicit evidence of NACE membership (e.g., official NACE member list or an institution page stating NACE membership). "
            "If the page is irrelevant or lacks explicit membership evidence, judge as not supported."
        )
    )

    # State Location group
    loc_group = evaluator.add_parallel(
        id=f"program_{idx}_state_location",
        desc="Document the state where the institution is located",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(p.location_urls),
        id=f"program_{idx}_location_reference_url",
        desc="Provide URL documenting the institution's location",
        parent=loc_group,
        critical=True
    )
    state_leaf = evaluator.add_leaf(
        id=f"program_{idx}_state_leaf",
        desc="Institution location (state) is correctly documented",
        parent=loc_group,
        critical=True
    )
    state_txt = p.state or "Unknown"
    claim = f"The institution '{p.institution_name or 'Unknown'}' is located in the U.S. state of {state_txt}."
    await evaluator.verify(
        claim=claim,
        node=state_leaf,
        sources=p.location_urls,
        additional_instruction=(
            "Focus on verifying the U.S. state location. City+state from official pages (or trusted directories) is acceptable. "
            "Minor formatting differences should be allowed."
        )
    )


async def build_facility_infrastructure(evaluator: Evaluator, parent, p: ProgramItem, idx: int):
    node = evaluator.add_sequential(
        id=f"program_{idx}_facility_infrastructure",
        desc="Verify the program has adequate dedicated esports facility infrastructure",
        parent=parent,
        critical=True
    )

    # Dedicated Esports Arena group
    arena_group = evaluator.add_parallel(
        id=f"program_{idx}_dedicated_esports_arena",
        desc="Confirm dedicated, named esports arena/facility on campus",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(p.facility.facility_urls),
        id=f"program_{idx}_arena_reference_url",
        desc="Provide URL describing the dedicated esports facility",
        parent=arena_group,
        critical=True
    )
    arena_leaf = evaluator.add_leaf(
        id=f"program_{idx}_arena_leaf",
        desc="Dedicated named esports facility exists on campus",
        parent=arena_group,
        critical=True
    )
    fname = safe_facility_name(p)
    claim = (
        f"The institution '{p.institution_name or 'Unknown'}' has a dedicated, named esports facility on campus called {fname}."
    )
    await evaluator.verify(
        claim=claim,
        node=arena_leaf,
        sources=p.facility.facility_urls,
        additional_instruction=(
            "Confirm the facility is dedicated to esports (e.g., 'Esports Arena', 'Esports Lab'), not a generic computer lab/shared space. "
            "Explicit naming on the page is required."
        )
    )

    # Minimum Gaming Stations group
    stations_group = evaluator.add_parallel(
        id=f"program_{idx}_minimum_gaming_stations",
        desc="Verify the facility has at least 15 gaming stations",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(p.facility.station_count_urls or p.facility.facility_urls),
        id=f"program_{idx}_station_count_reference_url",
        desc="Provide URL documenting the number of gaming stations",
        parent=stations_group,
        critical=True
    )
    stations_leaf = evaluator.add_leaf(
        id=f"program_{idx}_stations_leaf",
        desc="Facility has at least 15 gaming stations",
        parent=stations_group,
        critical=True
    )
    station_sources = nonempty_urls(p.facility.station_count_urls, p.facility.facility_urls)
    claim = f"The esports facility {fname} has at least 15 gaming stations available for player use."
    await evaluator.verify(
        claim=claim,
        node=stations_leaf,
        sources=station_sources,
        additional_instruction=(
            "Look for explicit counts like '15 PCs' or '15+ stations'. Synonyms such as gaming PCs/computers/stations are acceptable. "
            "If the count is ambiguous or missing, judge as not supported."
        )
    )


async def build_technical_specifications(evaluator: Evaluator, parent, p: ProgramItem, idx: int):
    node = evaluator.add_parallel(
        id=f"program_{idx}_technical_specifications",
        desc="Verify the gaming equipment meets competitive standards",
        parent=parent,
        critical=True
    )

    # Gaming PC Hardware group
    hw_group = evaluator.add_parallel(
        id=f"program_{idx}_gaming_pc_hardware",
        desc="Verify PC hardware specifications meet minimum requirements",
        parent=node,
        critical=True
    )

    # CPU
    cpu_group = evaluator.add_parallel(
        id=f"program_{idx}_cpu_specification",
        desc="Verify CPU has minimum 6 cores",
        parent=hw_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(p.hardware.cpu_urls or p.facility.facility_urls),
        id=f"program_{idx}_cpu_reference_url",
        desc="Provide URL documenting CPU specifications",
        parent=cpu_group,
        critical=True
    )
    cpu_leaf = evaluator.add_leaf(
        id=f"program_{idx}_cpu_leaf",
        desc="CPU has at least 6 cores",
        parent=cpu_group,
        critical=True
    )
    cpu_sources = nonempty_urls(p.hardware.cpu_urls, p.facility.facility_urls)
    await evaluator.verify(
        claim="The gaming PCs used by the program have CPUs with at least 6 physical cores.",
        node=cpu_leaf,
        sources=cpu_sources,
        additional_instruction=(
            "Verify via the provided pages that CPU models have ≥6 cores. Accept known models meeting or exceeding this threshold."
        )
    )

    # RAM
    ram_group = evaluator.add_parallel(
        id=f"program_{idx}_ram_specification",
        desc="Verify RAM is minimum 16GB",
        parent=hw_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(p.hardware.ram_urls or p.facility.facility_urls),
        id=f"program_{idx}_ram_reference_url",
        desc="Provide URL documenting RAM specifications",
        parent=ram_group,
        critical=True
    )
    ram_leaf = evaluator.add_leaf(
        id=f"program_{idx}_ram_leaf",
        desc="RAM is at least 16GB",
        parent=ram_group,
        critical=True
    )
    ram_sources = nonempty_urls(p.hardware.ram_urls, p.facility.facility_urls)
    await evaluator.verify(
        claim="The gaming PCs used by the program have at least 16GB of RAM.",
        node=ram_leaf,
        sources=ram_sources,
        additional_instruction=(
            "Look for explicit RAM amounts. If the page shows ≥16GB RAM, count as supported; otherwise not supported."
        )
    )

    # GPU
    gpu_group = evaluator.add_parallel(
        id=f"program_{idx}_gpu_specification",
        desc="Verify GPU meets or exceeds GTX 1060 / RX 580 equivalent",
        parent=hw_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(p.hardware.gpu_urls or p.facility.facility_urls),
        id=f"program_{idx}_gpu_reference_url",
        desc="Provide URL documenting GPU specifications",
        parent=gpu_group,
        critical=True
    )
    gpu_leaf = evaluator.add_leaf(
        id=f"program_{idx}_gpu_leaf",
        desc="GPU meets/exceeds GTX 1060 or RX 580 equivalent",
        parent=gpu_group,
        critical=True
    )
    gpu_sources = nonempty_urls(p.hardware.gpu_urls, p.facility.facility_urls)
    await evaluator.verify(
        claim=(
            "The gaming PCs used by the program have dedicated graphics equal to or better than NVIDIA GTX 1060 "
            "or AMD RX 580 (or modern equivalents)."
        ),
        node=gpu_leaf,
        sources=gpu_sources,
        additional_instruction=(
            "Confirm the GPU model listed meets or exceeds GTX 1060/RX 580 performance class. If clearly superior (e.g., RTX series), count as supported."
        )
    )

    # Storage
    storage_group = evaluator.add_parallel(
        id=f"program_{idx}_storage_type",
        desc="Verify storage is SSD type",
        parent=hw_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(p.hardware.storage_urls or p.facility.facility_urls),
        id=f"program_{idx}_storage_reference_url",
        desc="Provide URL documenting storage type",
        parent=storage_group,
        critical=True
    )
    storage_leaf = evaluator.add_leaf(
        id=f"program_{idx}_storage_leaf",
        desc="Storage is SSD",
        parent=storage_group,
        critical=True
    )
    storage_sources = nonempty_urls(p.hardware.storage_urls, p.facility.facility_urls)
    await evaluator.verify(
        claim="The gaming PCs used by the program use SSD storage.",
        node=storage_leaf,
        sources=storage_sources,
        additional_instruction=(
            "Look for explicit mention of SSD storage (NVMe/SATA). If only HDD is mentioned, not supported."
        )
    )

    # Monitor Specifications group
    mon_group = evaluator.add_parallel(
        id=f"program_{idx}_monitor_specifications",
        desc="Verify monitor specifications meet competitive gaming standards",
        parent=node,
        critical=True
    )

    # Refresh Rate
    rr_group = evaluator.add_parallel(
        id=f"program_{idx}_refresh_rate",
        desc="Verify monitors have minimum 144Hz refresh rate",
        parent=mon_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(p.monitors.monitor_urls),
        id=f"program_{idx}_refresh_rate_reference_url",
        desc="Provide URL documenting monitor refresh rate",
        parent=rr_group,
        critical=True
    )
    rr_leaf = evaluator.add_leaf(
        id=f"program_{idx}_refresh_rate_leaf",
        desc="Monitors support at least 144Hz",
        parent=rr_group,
        critical=True
    )
    await evaluator.verify(
        claim="The gaming monitors used by the program support a refresh rate of at least 144Hz.",
        node=rr_leaf,
        sources=p.monitors.monitor_urls,
        additional_instruction=(
            "Confirm via the provided pages the monitors are 144Hz or higher. If lower or not stated, not supported."
        )
    )

    # Resolution
    res_group = evaluator.add_parallel(
        id=f"program_{idx}_screen_resolution",
        desc="Verify monitors have minimum 1080p resolution",
        parent=mon_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(p.monitors.monitor_urls),
        id=f"program_{idx}_resolution_reference_url",
        desc="Provide URL documenting monitor resolution",
        parent=res_group,
        critical=True
    )
    res_leaf = evaluator.add_leaf(
        id=f"program_{idx}_resolution_leaf",
        desc="Monitors support at least 1080p (1920×1080)",
        parent=res_group,
        critical=True
    )
    await evaluator.verify(
        claim="The gaming monitors used by the program support a resolution of at least 1920×1080 (1080p).",
        node=res_leaf,
        sources=p.monitors.monitor_urls,
        additional_instruction=(
            "Confirm via the provided pages the monitors support 1080p or higher (e.g., 1440p/4K)."
        )
    )

    # Internet Connectivity group
    net_group = evaluator.add_parallel(
        id=f"program_{idx}_internet_connectivity",
        desc="Verify facility has minimum 10 Mbps upload speed dedicated internet",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(p.internet.internet_urls),
        id=f"program_{idx}_internet_reference_url",
        desc="Provide URL documenting internet connectivity specifications",
        parent=net_group,
        critical=True
    )
    net_leaf = evaluator.add_leaf(
        id=f"program_{idx}_internet_leaf",
        desc="Facility has dedicated high-speed internet with ≥10 Mbps upload",
        parent=net_group,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The esports facility has dedicated high-speed internet connectivity with at least 10 Mbps upload speed, "
            "suitable for streaming and online competition."
        ),
        node=net_leaf,
        sources=p.internet.internet_urls,
        additional_instruction=(
            "Confirm upload speed and dedicated high-speed connectivity. If only download is provided or upload <10 Mbps, not supported."
        )
    )


async def build_accessibility_compliance(evaluator: Evaluator, parent, p: ProgramItem, idx: int):
    node = evaluator.add_parallel(
        id=f"program_{idx}_accessibility_compliance",
        desc="Verify the facility demonstrates compliance with at least two accessibility features",
        parent=parent,
        critical=True
    )

    # Feature 1
    f1_group = evaluator.add_parallel(
        id=f"program_{idx}_accessibility_feature_1",
        desc="Document the first accessibility feature",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(p.accessibility.feature_1_urls),
        id=f"program_{idx}_accessibility_feature_1_reference_url",
        desc="Provide URL documenting the first accessibility feature",
        parent=f1_group,
        critical=True
    )
    f1_leaf = evaluator.add_leaf(
        id=f"program_{idx}_accessibility_feature_1_leaf",
        desc="First accessibility feature is implemented",
        parent=f1_group,
        critical=True
    )
    claim = f"The esports facility implements the accessibility feature: {p.accessibility.feature_1 or 'Unknown feature'}."
    await evaluator.verify(
        claim=claim,
        node=f1_leaf,
        sources=p.accessibility.feature_1_urls,
        additional_instruction=(
            "Accept features aligned with established guidelines (e.g., ADA-compliant access, ergonomic gaming furniture, multiple controller input support, customizable display settings)."
        )
    )

    # Feature 2
    f2_group = evaluator.add_parallel(
        id=f"program_{idx}_accessibility_feature_2",
        desc="Document the second accessibility feature",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(p.accessibility.feature_2_urls),
        id=f"program_{idx}_accessibility_feature_2_reference_url",
        desc="Provide URL documenting the second accessibility feature",
        parent=f2_group,
        critical=True
    )
    f2_leaf = evaluator.add_leaf(
        id=f"program_{idx}_accessibility_feature_2_leaf",
        desc="Second accessibility feature is implemented",
        parent=f2_group,
        critical=True
    )
    claim = f"The esports facility implements the accessibility feature: {p.accessibility.feature_2 or 'Unknown feature'}."
    await evaluator.verify(
        claim=claim,
        node=f2_leaf,
        sources=p.accessibility.feature_2_urls,
        additional_instruction=(
            "Accept features aligned with established guidelines (e.g., ADA-compliant access, ergonomic gaming furniture, multiple controller input support, customizable display settings)."
        )
    )


async def build_competitive_program(evaluator: Evaluator, parent, p: ProgramItem, idx: int):
    node = evaluator.add_parallel(
        id=f"program_{idx}_competitive_gaming_program",
        desc="Verify the program actively competes in at least two esports titles",
        parent=parent,
        critical=True
    )

    # Game 1
    g1_group = evaluator.add_parallel(
        id=f"program_{idx}_first_competitive_game",
        desc="Document the first esports title the program competes in",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(p.competition.game_1_urls),
        id=f"program_{idx}_game_1_reference_url",
        desc="Provide URL documenting competition in the first game title",
        parent=g1_group,
        critical=True
    )
    g1_leaf = evaluator.add_leaf(
        id=f"program_{idx}_game_1_leaf",
        desc="Program competes in the first esports title",
        parent=g1_group,
        critical=True
    )
    claim = f"The program actively competes in {p.competition.game_1 or 'Unknown title'}."
    await evaluator.verify(
        claim=claim,
        node=g1_leaf,
        sources=p.competition.game_1_urls,
        additional_instruction=(
            "Look for match schedules, results, rosters, or league pages indicating participation in this title."
        )
    )

    # Game 2
    g2_group = evaluator.add_parallel(
        id=f"program_{idx}_second_competitive_game",
        desc="Document the second esports title the program competes in",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(p.competition.game_2_urls),
        id=f"program_{idx}_game_2_reference_url",
        desc="Provide URL documenting competition in the second game title",
        parent=g2_group,
        critical=True
    )
    g2_leaf = evaluator.add_leaf(
        id=f"program_{idx}_game_2_leaf",
        desc="Program competes in the second esports title",
        parent=g2_group,
        critical=True
    )
    claim = f"The program actively competes in {p.competition.game_2 or 'Unknown title'}."
    await evaluator.verify(
        claim=claim,
        node=g2_leaf,
        sources=p.competition.game_2_urls,
        additional_instruction=(
            "Look for match schedules, results, rosters, or league pages indicating participation in this title."
        )
    )


async def verify_program(evaluator: Evaluator, root_parent, p: ProgramItem, idx: int):
    prog_node = evaluator.add_parallel(
        id=f"Program_{idx + 1}",
        desc=f"Program #{idx + 1} verification",
        parent=root_parent,
        critical=False
    )

    # Institutional eligibility
    await build_institutional_eligibility(evaluator, prog_node, p, idx)
    # Facility infrastructure
    await build_facility_infrastructure(evaluator, prog_node, p, idx)
    # Technical specifications
    await build_technical_specifications(evaluator, prog_node, p, idx)
    # Accessibility compliance
    await build_accessibility_compliance(evaluator, prog_node, p, idx)
    # Competitive program
    await build_competitive_program(evaluator, prog_node, p, idx)


def build_geographic_diversity_node(evaluator: Evaluator, parent, programs: List[ProgramItem]):
    geo_node = evaluator.add_parallel(
        id="Geographic_Diversity_Verification",
        desc="Verify that the three identified programs are located in three different U.S. states",
        parent=parent,
        critical=True
    )
    states = [p.state for p in programs[:3] if p and p.state]
    unique_states = set(s.strip().lower() for s in states if isinstance(s, str))
    evaluator.add_custom_node(
        result=(len(unique_states) == 3),
        id="Different_States_Confirmation",
        desc="Confirm all three programs are in different states based on documented locations",
        parent=geo_node,
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
    Evaluate an answer for the college esports programs identification task.
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

    # Extract program information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction"
    )

    # Prepare exactly 3 programs
    programs = list(extraction.programs[:3])
    while len(programs) < 3:
        programs.append(ProgramItem())

    # Build verification subtrees for each program
    for i in range(3):
        await verify_program(evaluator, root, programs[i], i)

    # Geographic diversity verification (critical)
    build_geographic_diversity_node(evaluator, root, programs)

    # Add custom info summarizing requirements checked
    evaluator.add_custom_info(
        info={
            "institutional_requirements": [
                "NACE membership (with supporting URLs)",
                "State location documented (with supporting URLs)",
                "Programs must be in three different states (critical)"
            ],
            "facility_requirements": [
                "Dedicated named esports facility (with supporting URLs)",
                "Minimum 15 gaming stations (with supporting URLs)"
            ],
            "technical_specifications": [
                "CPU ≥ 6 cores",
                "RAM ≥ 16GB",
                "GPU ≥ GTX 1060 / RX 580 equivalent",
                "Storage: SSD",
                "Monitors: ≥144Hz",
                "Monitors: ≥1080p resolution",
                "Internet: ≥10 Mbps upload, dedicated high-speed connectivity"
            ],
            "accessibility_requirements": [
                "At least two specific accessibility features with URLs"
            ],
            "competitive_program_requirements": [
                "At least two esports titles with URLs"
            ]
        },
        info_type="requirements_checked"
    )

    # Return structured summary
    return evaluator.get_summary()