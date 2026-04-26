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
TASK_ID = "lic_leed_cornell_year_chain"
TASK_DESCRIPTION = (
    "In Long Island City, Queens, New York City, there is a residential building that holds the distinction of "
    "being the first multi-family residential building in the world to achieve LEED v4 BD+C: New Construction "
    "Platinum certification. This building was designed by an architecture firm that was founded in 1994. The "
    "founding partner of this architecture firm previously worked at Kohn Pedersen Fox Associates before establishing "
    "his own practice. This founding partner earned a Bachelor of Architecture degree from Cornell University. "
    "In what year was the architecture program at Cornell University first established?"
)

EXPECTED_CORNELL_YEAR = "1871"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BuildingInfo(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    certification: Optional[str] = None
    first_multifamily_claim: Optional[str] = None
    developer_name: Optional[str] = None
    developer_founded_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FirmInfo(BaseModel):
    name: Optional[str] = None
    founded_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FounderInfo(BaseModel):
    name: Optional[str] = None
    previous_employer: Optional[str] = None
    education: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CornellProgramInfo(BaseModel):
    year_established: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ChainExtraction(BaseModel):
    building: Optional[BuildingInfo] = None
    firm: Optional[FirmInfo] = None
    founder: Optional[FounderInfo] = None
    cornell_program: Optional[CornellProgramInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_chain() -> str:
    return (
        "Extract the entity chain described in the answer and the supporting sources (URLs). "
        "Return a JSON object with these top-level keys: building, firm, founder, cornell_program. "
        "Populate each with the following fields as available from the answer:\n"
        "- building: {\n"
        "    name: the building's name,\n"
        "    location: the stated location text for the building,\n"
        "    certification: the stated certification text (e.g., 'LEED v4 BD+C: New Construction Platinum'),\n"
        "    first_multifamily_claim: the wording used to claim it is the first multi-family to achieve the certification,\n"
        "    developer_name: the developer/real estate organization name,\n"
        "    developer_founded_year: the stated year the developer was founded,\n"
        "    sources: all URLs in the answer that support the building-related claims\n"
        "  }\n"
        "- firm: {\n"
        "    name: the architecture firm credited with designing the building,\n"
        "    founded_year: the year the firm was founded,\n"
        "    sources: all URLs in the answer that support the firm's details or its role designing the building\n"
        "  }\n"
        "- founder: {\n"
        "    name: the founding partner of the architecture firm,\n"
        "    previous_employer: the prior employer (e.g., 'Kohn Pedersen Fox Associates' or 'KPF'),\n"
        "    education: the relevant degree and institution text (e.g., 'B.Arch from Cornell University'),\n"
        "    sources: all URLs in the answer that support the founder details\n"
        "  }\n"
        "- cornell_program: {\n"
        "    year_established: the year the architecture program at Cornell University was first established,\n"
        "    sources: all URLs in the answer that support the Cornell program year\n"
        "  }\n"
        "Rules:\n"
        "1) Extract only what is explicitly stated in the answer; do not invent details.\n"
        "2) Include only valid URLs present in the answer for 'sources'.\n"
        "3) If any field is missing, return null for that field. If no sources are mentioned for a section, return an empty array.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(*list_groups: Optional[List[str]]) -> List[str]:
    urls: List[str] = []
    for group in list_groups:
        if group:
            for u in group:
                if isinstance(u, str) and u.strip():
                    urls.append(u.strip())
    return urls


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_building(
    evaluator: Evaluator,
    parent_node,
    chain: ChainExtraction,
) -> None:
    building = chain.building or BuildingInfo()
    bnode = evaluator.add_parallel(
        id="identify_qualifying_building",
        desc="Identify the residential building described in the constraints (the target building).",
        parent=parent_node,
        critical=True,
    )

    # Existence gate (critical)
    evaluator.add_custom_node(
        result=bool(building.name) and len(building.sources) > 0,
        id="building_identified_with_sources",
        desc="Target building is identified with at least one source.",
        parent=bnode,
        critical=True,
    )

    # 1) Location
    loc_leaf = evaluator.add_leaf(
        id="building_in_long_island_city",
        desc="The building is located in Long Island City, Queens, New York City.",
        parent=bnode,
        critical=True,
    )
    claim_loc = (
        f"The building '{building.name}' is located in Long Island City, Queens, New York City."
        if building.name else
        "The target building is located in Long Island City, Queens, New York City."
    )
    await evaluator.verify(
        claim=claim_loc,
        node=loc_leaf,
        sources=building.sources,
        additional_instruction="Allow 'LIC' as an abbreviation for Long Island City. It is a neighborhood in the borough of Queens in New York City.",
    )

    # 2) LEED v4 BD+C: New Construction Platinum
    leed_leaf = evaluator.add_leaf(
        id="building_leed_v4_platinum",
        desc="The building has LEED v4 BD+C: New Construction Platinum certification.",
        parent=bnode,
        critical=True,
    )
    claim_leed = (
        f"The building '{building.name}' achieved LEED v4 BD+C: New Construction Platinum certification."
        if building.name else
        "The target building achieved LEED v4 BD+C: New Construction Platinum certification."
    )
    await evaluator.verify(
        claim=claim_leed,
        node=leed_leaf,
        sources=building.sources,
        additional_instruction="Confirm the specific LEED version (v4), rating system (BD+C: New Construction), and level (Platinum). Minor wording variants like 'LEED v4 BD+C NC Platinum' are acceptable.",
    )

    # 3) First multi-family to achieve this
    first_leaf = evaluator.add_leaf(
        id="building_is_first_multifamily_to_achieve_this",
        desc="The building is documented as the first multi-family residential building in the world to achieve LEED v4 BD+C: New Construction Platinum.",
        parent=bnode,
        critical=True,
    )
    claim_first = (
        f"The building '{building.name}' is the first multi-family residential building in the world to achieve LEED v4 BD+C: New Construction Platinum."
        if building.name else
        "The target building is the first multi-family residential building in the world to achieve LEED v4 BD+C: New Construction Platinum."
    )
    await evaluator.verify(
        claim=claim_first,
        node=first_leaf,
        sources=building.sources,
        additional_instruction="Look for explicit 'first' language. Accept close variants like 'first large-scale residential building' if it clearly refers to multi-family at building scale.",
    )

    # 4) Developer founded in 1915
    dev_leaf = evaluator.add_leaf(
        id="building_developer_founded_1915",
        desc="The building was developed by a real estate organization founded in 1915.",
        parent=bnode,
        critical=True,
    )
    dev_name = building.developer_name or "the developer"
    claim_dev = (
        f"The building '{building.name}' was developed by {dev_name}, which was founded in 1915."
        if building.name else
        f"The target building was developed by {dev_name}, which was founded in 1915."
    )
    await evaluator.verify(
        claim=claim_dev,
        node=dev_leaf,
        sources=building.sources,
        additional_instruction="Verify both that the developer is correctly associated with the building and that the developer's founding year is 1915.",
    )


async def verify_firm(
    evaluator: Evaluator,
    parent_node,
    chain: ChainExtraction,
) -> None:
    building = chain.building or BuildingInfo()
    firm = chain.firm or FirmInfo()

    fnode = evaluator.add_parallel(
        id="identify_architecture_firm",
        desc="Identify the architecture firm that designed the qualifying building.",
        parent=parent_node,
        critical=True,
    )

    # Existence gate (critical)
    evaluator.add_custom_node(
        result=bool(firm.name) and (len(firm.sources) > 0 or len((building.sources or [])) > 0),
        id="firm_identified_with_sources",
        desc="Architecture firm is identified with supporting sources.",
        parent=fnode,
        critical=True,
    )

    # 1) Firm designed the building
    designed_leaf = evaluator.add_leaf(
        id="firm_designed_building",
        desc="The firm is documented as the designer of the qualifying building.",
        parent=fnode,
        critical=True,
    )
    combined_urls = combine_sources(firm.sources, building.sources)
    claim_designed = (
        f"The architecture firm '{firm.name}' is documented as the design architect (or architect of record) of the building '{building.name}'."
        if firm.name and building.name else
        "The architecture firm is documented as the designer of the qualifying building."
    )
    await evaluator.verify(
        claim=claim_designed,
        node=designed_leaf,
        sources=combined_urls,
        additional_instruction="Accept roles such as 'architect of record' or 'design architect' as evidence the firm designed the building.",
    )

    # 2) Firm founded in 1994
    founded_leaf = evaluator.add_leaf(
        id="firm_founded_1994",
        desc="The architecture firm was founded in 1994.",
        parent=fnode,
        critical=True,
    )
    claim_founded = (
        f"The architecture firm '{firm.name}' was founded in 1994."
        if firm.name else
        "The architecture firm was founded in 1994."
    )
    await evaluator.verify(
        claim=claim_founded,
        node=founded_leaf,
        sources=firm.sources,
        additional_instruction="Check firm overview pages, official bios, or reputable sources that list the firm's founding year.",
    )


async def verify_founder(
    evaluator: Evaluator,
    parent_node,
    chain: ChainExtraction,
) -> None:
    founder = chain.founder or FounderInfo()

    node = evaluator.add_parallel(
        id="identify_founding_partner",
        desc="Identify the founding partner of the architecture firm specified in the constraints.",
        parent=parent_node,
        critical=True,
    )

    # Existence gate (critical)
    evaluator.add_custom_node(
        result=bool(founder.name) and len(founder.sources) > 0,
        id="founder_identified_with_sources",
        desc="Founding partner is identified with supporting sources.",
        parent=node,
        critical=True,
    )

    # 1) Founder worked at KPF
    kpf_leaf = evaluator.add_leaf(
        id="founder_worked_at_kpf",
        desc="The founding partner previously worked at Kohn Pedersen Fox Associates (KPF) before establishing their own practice.",
        parent=node,
        critical=True,
    )
    claim_kpf = (
        f"The founding partner '{founder.name}' previously worked at Kohn Pedersen Fox Associates (KPF) before establishing his own practice."
        if founder.name else
        "The founding partner previously worked at Kohn Pedersen Fox Associates (KPF) before establishing his own practice."
    )
    await evaluator.verify(
        claim=claim_kpf,
        node=kpf_leaf,
        sources=founder.sources,
        additional_instruction="Look for wording like 'worked at KPF', 'former KPF principal/partner', or similar.",
    )

    # 2) Founder has Cornell B.Arch
    barch_leaf = evaluator.add_leaf(
        id="founder_has_cornell_barch",
        desc="The founding partner earned a Bachelor of Architecture (B.Arch) degree from Cornell University.",
        parent=node,
        critical=True,
    )
    claim_barch = (
        f"The founding partner '{founder.name}' earned a Bachelor of Architecture (B.Arch) degree from Cornell University."
        if founder.name else
        "The founding partner earned a Bachelor of Architecture (B.Arch) degree from Cornell University."
    )
    await evaluator.verify(
        claim=claim_barch,
        node=barch_leaf,
        sources=founder.sources,
        additional_instruction="Accept phrasing variants like 'BArch' or 'Bachelor of Architecture'.",
    )


async def verify_cornell_year(
    evaluator: Evaluator,
    parent_node,
    chain: ChainExtraction,
) -> None:
    cornell = chain.cornell_program or CornellProgramInfo()

    node = evaluator.add_parallel(
        id="answer_cornell_architecture_program_year",
        desc="Provide the year Cornell University’s architecture program was first established.",
        parent=parent_node,
        critical=True,
    )

    # Optional existence gate to ensure we have something to verify
    evaluator.add_custom_node(
        result=bool(cornell.year_established),
        id="cornell_year_provided",
        desc="A year was provided for Cornell University's architecture program establishment.",
        parent=node,
        critical=True,
    )

    year_leaf = evaluator.add_leaf(
        id="year_is_1871",
        desc="The year provided is 1871 (as specified in the constraints).",
        parent=node,
        critical=True,
    )

    claim_year = "The architecture program at Cornell University was first established in 1871."
    await evaluator.verify(
        claim=claim_year,
        node=year_leaf,
        sources=cornell.sources,
        additional_instruction="Confirm 'first established' or 'founded' year for the architecture program at Cornell University. Accept authoritative Cornell or academic sources.",
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Determine the year Cornell University’s architecture program was first established, following the entity chain specified in the prompt/constraints.",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract chain information from the answer
    chain = await evaluator.extract(
        prompt=prompt_extract_chain(),
        template_class=ChainExtraction,
        extraction_name="entity_chain_extraction",
    )

    # Add ground truth information (for final check reference)
    evaluator.add_ground_truth({
        "expected_cornell_architecture_program_year": EXPECTED_CORNELL_YEAR,
        "chain_requirements": [
            "Building is in Long Island City, Queens, NYC",
            "Building achieved LEED v4 BD+C: New Construction Platinum",
            "Building is the first multi-family to achieve it",
            "Developer founded in 1915",
            "Architecture firm designed the building; founded in 1994",
            "Founding partner previously worked at KPF",
            "Founding partner has Cornell B.Arch",
        ],
    })

    # Build verification subtrees in sequence
    await verify_building(evaluator, root, chain)
    await verify_firm(evaluator, root, chain)
    await verify_founder(evaluator, root, chain)
    await verify_cornell_year(evaluator, root, chain)

    return evaluator.get_summary()