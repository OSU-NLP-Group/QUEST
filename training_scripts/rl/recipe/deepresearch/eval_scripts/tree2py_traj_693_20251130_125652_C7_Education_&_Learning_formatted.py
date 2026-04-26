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
TASK_ID = "state_athletics_football_2024"
TASK_DESCRIPTION = """
A high school sports administration consultant is preparing a comparative analysis of state athletic association football playoff structures to present to a state education board considering reforms to their own playoff system. The consultant needs to gather specific structural information about several state systems from the 2024 season.

Provide the following information:

1. NFHS (National Federation): How many total member associations does the NFHS have?

2. IHSA (Illinois High School Association): For the 2024 football season:
   - How many football classes does IHSA use for playoff classification?
   - What is the total number of teams that qualify for the playoffs across all classes?
   - What enrollment multiplier does IHSA apply to non-boundaried schools?
   - How many teams qualify per class in the playoffs?

3. OHSAA (Ohio High School Athletic Association): For the 2024 football season:
   - How many football divisions does OHSAA have?
   - How many teams qualified per region in the playoffs?
   - How many regions exist within each division?
   - What is the minimum enrollment threshold (adjusted enrollment) for Division I classification?

4. LHSAA (Louisiana High School Athletic Association): For the 2024-2026 cycle:
   - How many enrollment-based classifications (classes, such as 1A, 2A, etc.) does LHSAA use for football?
   - How many total playoff divisions does LHSAA create from these classifications?
   - How many of these playoff divisions are designated as "Select" divisions?
   - How many of these playoff divisions are designated as "Non-Select" divisions?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NFHSInfo(BaseModel):
    total_member_associations: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class IHSAInfo(BaseModel):
    class_count: Optional[str] = None
    total_playoff_qualifiers: Optional[str] = None
    non_boundaried_multiplier: Optional[str] = None
    teams_per_class: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class OHSAAInfo(BaseModel):
    division_count: Optional[str] = None
    qualifiers_per_region: Optional[str] = None
    regions_per_division: Optional[str] = None
    div1_min_enrollment: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LHSAAInfo(BaseModel):
    enrollment_classifications_count: Optional[str] = None
    total_playoff_divisions: Optional[str] = None
    select_divisions_count: Optional[str] = None
    non_select_divisions_count: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AssociationsExtraction(BaseModel):
    nfhs: Optional[NFHSInfo] = None
    ihsa: Optional[IHSAInfo] = None
    ohsaa: Optional[OHSAAInfo] = None
    lhsaa: Optional[LHSAAInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract the specific football/playoff structural facts explicitly stated in the answer text, along with any URLs cited for each association.
    Return the data using the following JSON fields (use strings for numbers as they appear in the answer; if missing, use null; for sources, only include actual URLs that appear in the answer):

    - nfhs:
        - total_member_associations
        - sources  (all URLs specifically used for the NFHS membership statement)
    - ihsa (2024 season):
        - class_count
        - total_playoff_qualifiers
        - non_boundaried_multiplier
        - teams_per_class
        - sources  (all URLs specifically used for the IHSA facts)
    - ohsaa (2024 season):
        - division_count
        - qualifiers_per_region
        - regions_per_division
        - div1_min_enrollment
        - sources  (all URLs specifically used for the OHSAA facts)
    - lhsaa (2024–2026 cycle):
        - enrollment_classifications_count
        - total_playoff_divisions
        - select_divisions_count
        - non_select_divisions_count
        - sources  (all URLs specifically used for the LHSAA facts)

    Important:
    - Do not infer or invent numbers or URLs; only extract what appears in the answer.
    - For URLs, handle plain links or markdown links and include the actual URL strings.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _src_list_or_none(lst: Optional[List[str]]) -> Optional[List[str]]:
    if not lst:
        return None
    # Filter obviously malformed entries
    cleaned = [s for s in lst if isinstance(s, str) and len(s.strip()) > 0]
    return cleaned if cleaned else None


def _expected_values() -> Dict[str, Any]:
    return {
        "NFHS_Total_Member_Associations": 51,

        "IHSA_Class_Count": 8,
        "IHSA_Total_Playoff_Qualifiers": 256,
        "IHSA_NonBoundaried_Enrollment_Multiplier": "1.65",
        "IHSA_Teams_Per_Class": 32,

        "OHSAA_Division_Count": 7,
        "OHSAA_Qualifiers_Per_Region": 16,
        "OHSAA_Regions_Per_Division": 4,
        "OHSAA_Division_I_Min_Enrollment": 592,

        "LHSAA_Enrollment_Classifications": 5,
        "LHSAA_Total_Playoff_Divisions": 8,
        "LHSAA_Select_Divisions": 4,
        "LHSAA_NonSelect_Divisions": 4,
    }


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_nfhs_nodes(evaluator: Evaluator, parent, sources: Optional[List[str]]) -> None:
    nfhs_node = evaluator.add_parallel(
        id="NFHS",
        desc="NFHS membership total",
        parent=parent,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="NFHS_Total_Member_Associations_Equals_51",
        desc="States that NFHS has exactly 51 member associations (50 states + DC).",
        parent=nfhs_node,
        critical=True,
    )
    claim = "The National Federation of State High School Associations (NFHS) has exactly 51 member associations (50 states plus the District of Columbia)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_src_list_or_none(sources),
        additional_instruction="Confirm the official NFHS membership count. Prefer official NFHS pages or authoritative references; accept minor phrasing variants, but the number must be 51."
    )


async def build_ihsa_nodes(evaluator: Evaluator, parent, sources: Optional[List[str]]) -> None:
    ihsa_node = evaluator.add_parallel(
        id="IHSA_2024",
        desc="IHSA 2024 football playoff structure facts",
        parent=parent,
        critical=True,
    )

    leaves = []
    claims_and_instructions = []

    leaf = evaluator.add_leaf(
        id="IHSA_Class_Count_Equals_8",
        desc="States that IHSA uses exactly 8 football classes for playoff classification (2024 season).",
        parent=ihsa_node,
        critical=True,
    )
    leaves.append(leaf)
    claims_and_instructions.append((
        "For the 2024 season, the Illinois High School Association (IHSA) uses exactly 8 football classes for playoff classification.",
        "Verify the 2024 IHSA football classification count. Use IHSA official pages or documents when available."
    ))

    leaf = evaluator.add_leaf(
        id="IHSA_Total_Playoff_Qualifiers_Equals_256",
        desc="States that exactly 256 teams qualify for IHSA playoffs across all classes (2024 season).",
        parent=ihsa_node,
        critical=True,
    )
    leaves.append(leaf)
    claims_and_instructions.append((
        "For the 2024 season, a total of 256 teams qualify for the IHSA football playoffs across all classes.",
        "Confirm the total playoff field size across all classes in 2024 (8 classes × 32 each = 256); prefer official IHSA sources."
    ))

    leaf = evaluator.add_leaf(
        id="IHSA_NonBoundaried_Enrollment_Multiplier_Equals_1_65",
        desc="States that IHSA applies a 1.65 enrollment multiplier to non-boundaried schools.",
        parent=ihsa_node,
        critical=True,
    )
    leaves.append(leaf)
    claims_and_instructions.append((
        "IHSA applies a 1.65 enrollment multiplier to non-boundaried schools.",
        "Check IHSA policies/regulations regarding multipliers applied to non-boundaried (e.g., private) schools. The value must be 1.65."
    ))

    leaf = evaluator.add_leaf(
        id="IHSA_Teams_Per_Class_Equals_32",
        desc="States that 32 teams qualify per class in IHSA playoffs.",
        parent=ihsa_node,
        critical=True,
    )
    leaves.append(leaf)
    claims_and_instructions.append((
        "In the IHSA football playoffs, 32 teams qualify in each class.",
        "Confirm per-class qualifiers for IHSA football playoffs. The number must be 32."
    ))

    batch = []
    srcs = _src_list_or_none(sources)
    for (claim, add_ins), node in zip(claims_and_instructions, leaves):
        batch.append((claim, srcs, node, add_ins))
    await evaluator.batch_verify(batch)


async def build_ohsaa_nodes(evaluator: Evaluator, parent, sources: Optional[List[str]]) -> None:
    ohsaa_node = evaluator.add_parallel(
        id="OHSAA_2024",
        desc="OHSAA 2024 football playoff structure facts",
        parent=parent,
        critical=True,
    )

    leaves = []
    claims_and_instructions = []

    leaf = evaluator.add_leaf(
        id="OHSAA_Division_Count_Equals_7",
        desc="States that OHSAA has exactly 7 football divisions (Division I–VII) in 2024.",
        parent=ohsaa_node,
        critical=True,
    )
    leaves.append(leaf)
    claims_and_instructions.append((
        "For 2024, the Ohio High School Athletic Association (OHSAA) has exactly 7 football divisions (Division I through Division VII).",
        "Confirm the number of football divisions for OHSAA in the 2024 season using official OHSAA sources."
    ))

    leaf = evaluator.add_leaf(
        id="OHSAA_Qualifiers_Per_Region_Equals_16",
        desc="States that 16 teams qualified per region in the 2024 OHSAA football playoffs.",
        parent=ohsaa_node,
        critical=True,
    )
    leaves.append(leaf)
    claims_and_instructions.append((
        "In the 2024 OHSAA football playoffs, 16 teams qualified per region.",
        "Verify the per‑region playoff qualifiers count for 2024 OHSAA football. The number must be 16."
    ))

    leaf = evaluator.add_leaf(
        id="OHSAA_Regions_Per_Division_Equals_4",
        desc="States that there are 4 regions within each OHSAA football division.",
        parent=ohsaa_node,
        critical=True,
    )
    leaves.append(leaf)
    claims_and_instructions.append((
        "Each OHSAA football division has 4 regions.",
        "Confirm the number of regions per division for OHSAA football (should be 4)."
    ))

    leaf = evaluator.add_leaf(
        id="OHSAA_Division_I_Min_Enrollment_Equals_592",
        desc="States that the minimum adjusted enrollment threshold for OHSAA Division I is 592 students.",
        parent=ohsaa_node,
        critical=True,
    )
    leaves.append(leaf)
    claims_and_instructions.append((
        "The minimum adjusted enrollment threshold for OHSAA Division I classification is 592 students.",
        "Check the OHSAA divisional enrollment guidelines/regulations for 2024 and confirm the Division I minimum adjusted enrollment equals 592."
    ))

    batch = []
    srcs = _src_list_or_none(sources)
    for (claim, add_ins), node in zip(claims_and_instructions, leaves):
        batch.append((claim, srcs, node, add_ins))
    await evaluator.batch_verify(batch)


async def build_lhsaa_nodes(evaluator: Evaluator, parent, sources: Optional[List[str]]) -> None:
    lhsaa_node = evaluator.add_parallel(
        id="LHSAA_2024_2026_Cycle",
        desc="LHSAA 2024–2026 cycle football classification/division facts",
        parent=parent,
        critical=True,
    )

    leaves = []
    claims_and_instructions = []

    leaf = evaluator.add_leaf(
        id="LHSAA_Enrollment_Classifications_Equals_5",
        desc="States that LHSAA uses exactly 5 enrollment-based football classifications (Class 1A–5A) for the 2024–2026 cycle.",
        parent=lhsaa_node,
        critical=True,
    )
    leaves.append(leaf)
    claims_and_instructions.append((
        "For the 2024–2026 cycle, the LHSAA uses exactly 5 enrollment-based football classifications (Class 1A through 5A).",
        "Verify the LHSAA classification structure for football in the 2024–2026 cycle."
    ))

    leaf = evaluator.add_leaf(
        id="LHSAA_Total_Playoff_Divisions_Equals_8",
        desc="States that LHSAA creates 8 total playoff divisions from these classifications.",
        parent=lhsaa_node,
        critical=True,
    )
    leaves.append(leaf)
    claims_and_instructions.append((
        "From these classifications, the LHSAA creates 8 total football playoff divisions.",
        "Confirm how many total playoff divisions are formed from the classifications in the 2024–2026 cycle."
    ))

    leaf = evaluator.add_leaf(
        id="LHSAA_Select_Divisions_Equals_4",
        desc="States that LHSAA has 4 Select playoff divisions.",
        parent=lhsaa_node,
        critical=True,
    )
    leaves.append(leaf)
    claims_and_instructions.append((
        "Of these LHSAA football playoff divisions, 4 are designated Select divisions.",
        "Verify the count of Select divisions in the 2024–2026 LHSAA football structure."
    ))

    leaf = evaluator.add_leaf(
        id="LHSAA_NonSelect_Divisions_Equals_4",
        desc="States that LHSAA has 4 Non-Select playoff divisions.",
        parent=lhsaa_node,
        critical=True,
    )
    leaves.append(leaf)
    claims_and_instructions.append((
        "Of these LHSAA football playoff divisions, 4 are designated Non-Select divisions.",
        "Verify the count of Non‑Select divisions in the 2024–2026 LHSAA football structure."
    ))

    batch = []
    srcs = _src_list_or_none(sources)
    for (claim, add_ins), node in zip(claims_and_instructions, leaves):
        batch.append((claim, srcs, node, add_ins))
    await evaluator.batch_verify(batch)


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
    Evaluate an answer for the State Athletic Associations Football Information task.
    """
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

    # Create the rubric root under framework root
    rubric_root = evaluator.add_parallel(
        id="State_Athletic_Associations_Football_Information",
        desc="Provide the specified NFHS/IHSA/OHSAA/LHSAA football/playoff structure facts as required by the prompt and constraints.",
        parent=root,
        critical=True,
    )

    # Extract structured claims and sources from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AssociationsExtraction,
        extraction_name="extracted_facts",
    )

    # Ground truth reference values (for record)
    evaluator.add_ground_truth(
        {
            "expected_values": _expected_values(),
            "notes": "These are the target facts to verify against cited sources where available.",
        },
        gt_type="ground_truth_values",
    )

    # Build subtrees and verify claims
    nfhs_sources = extracted.nfhs.sources if extracted.nfhs else []
    ihsa_sources = extracted.ihsa.sources if extracted.ihsa else []
    ohsaa_sources = extracted.ohsaa.sources if extracted.ohsaa else []
    lhsaa_sources = extracted.lhsaa.sources if extracted.lhsaa else []

    await asyncio.gather(
        build_nfhs_nodes(evaluator, rubric_root, nfhs_sources),
        build_ihsa_nodes(evaluator, rubric_root, ihsa_sources),
        build_ohsaa_nodes(evaluator, rubric_root, ohsaa_sources),
        build_lhsaa_nodes(evaluator, rubric_root, lhsaa_sources),
    )

    return evaluator.get_summary()