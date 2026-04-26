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
TASK_ID = "2024_nif_highest_yield"
TASK_DESCRIPTION = (
    "In 2024, the National Ignition Facility (NIF) conducted multiple fusion ignition experiments with varying levels "
    "of success. Identify the specific experiment that achieved the highest fusion energy yield during calendar year 2024.\n\n"
    "For this experiment, provide:\n"
    "1. The exact date (month, day, and year) when the experiment was conducted\n"
    "2. The laser input energy delivered to the target (in megajoules)\n"
    "3. The fusion energy yield produced (in megajoules)\n\n"
    "Additionally, provide the following contextual information about NIF:\n"
    "4. The city and state where the National Ignition Facility is located\n"
    "5. The full name of the current NIF Director\n\n"
    "Include reference URLs from official NIF or Lawrence Livermore National Laboratory sources to support your findings."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ExperimentSelection(BaseModel):
    identifier: Optional[str] = None
    date: Optional[str] = None

    id_source_urls: List[str] = Field(default_factory=list)
    date_source_urls: List[str] = Field(default_factory=list)
    nif_source_urls: List[str] = Field(default_factory=list)
    ignition_source_urls: List[str] = Field(default_factory=list)
    highest_yield_source_urls: List[str] = Field(default_factory=list)


class EnergyMetrics(BaseModel):
    laser_input_energy_mj: Optional[str] = None
    laser_input_source_urls: List[str] = Field(default_factory=list)

    fusion_yield_mj: Optional[str] = None
    fusion_yield_source_urls: List[str] = Field(default_factory=list)


class NIFContext(BaseModel):
    city: Optional[str] = None
    city_source_urls: List[str] = Field(default_factory=list)

    state: Optional[str] = None
    state_source_urls: List[str] = Field(default_factory=list)

    director_full_name: Optional[str] = None
    director_source_urls: List[str] = Field(default_factory=list)


class NIF2024HighestYieldExtraction(BaseModel):
    experiment: Optional[ExperimentSelection] = None
    energy: Optional[EnergyMetrics] = None
    context: Optional[NIFContext] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_nif_2024_highest() -> str:
    return """
    From the provided answer text, extract a structured summary about the single NIF experiment that the answer identifies as having the highest fusion energy yield in calendar year 2024, along with required energy metrics and NIF facility context. Extract only what is explicitly present in the answer. Do not infer or invent any information.

    REQUIRED STRUCTURE AND FIELDS:

    experiment:
      - identifier: A specific identifier or uniquely identifying description for the experiment (e.g., a NIF shot identifier and/or a unique description).
      - date: The exact date of the experiment (as written in the answer; any readable format is acceptable).
      - id_source_urls: URLs cited in the answer that support the unique identification of this specific experiment (prefer official NIF/LLNL).
      - date_source_urls: URLs cited in the answer that support the exact experiment date (prefer official NIF/LLNL).
      - nif_source_urls: URLs cited in the answer that support that the experiment was conducted at the National Ignition Facility (prefer official NIF/LLNL).
      - ignition_source_urls: URLs cited in the answer that support that this is a fusion ignition experiment (prefer official NIF/LLNL).
      - highest_yield_source_urls: URLs cited in the answer that support that this experiment had the highest fusion energy yield among NIF fusion ignition experiments conducted in calendar year 2024 (prefer official NIF/LLNL).

    energy:
      - laser_input_energy_mj: The laser input energy delivered to the target (as a string; keep units or SI symbol as the answer shows, typically in MJ).
      - laser_input_source_urls: URLs cited in the answer that support the laser input energy value (prefer official NIF/LLNL).
      - fusion_yield_mj: The fusion energy yield produced (as a string; keep units or SI symbol as the answer shows, typically in MJ).
      - fusion_yield_source_urls: URLs cited in the answer that support the fusion yield value (prefer official NIF/LLNL).

    context:
      - city: The city where the National Ignition Facility is located (as stated in the answer).
      - city_source_urls: URLs cited in the answer that support the city (prefer official NIF/LLNL).
      - state: The U.S. state where the National Ignition Facility is located (as stated in the answer).
      - state_source_urls: URLs cited in the answer that support the state (prefer official NIF/LLNL).
      - director_full_name: The full name of the current NIF Director (as stated in the answer).
      - director_source_urls: URLs cited in the answer that support the current director information (prefer official NIF/LLNL).

    IMPORTANT INSTRUCTIONS:
    - Only extract URLs that are explicitly present in the answer. If none are provided for a field, return an empty list for the corresponding URLs.
    - Do not attempt to create or infer URLs. If the answer references a source without a URL, do not add one.
    - Keep numbers as free-form strings (e.g., '2.05 MJ' or '2 MJ') exactly as presented in the answer; do not perform unit conversions.
    - If any field is missing from the answer, set it to null (for strings) or an empty list (for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def safe_str(value: Optional[str], fallback: str) -> str:
    if value is None:
        return fallback
    s = value.strip()
    return s if s else fallback


def union_urls(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not isinstance(url, str):
                continue
            u = url.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_selection_checks(
    evaluator: Evaluator,
    parent_node,
    extraction: NIF2024HighestYieldExtraction
) -> None:
    """
    Build and verify 'Experiment_Selection_and_Validity' checks (critical, parallel).
    """
    node = evaluator.add_parallel(
        id="Experiment_Selection_and_Validity",
        desc="Correctly identify the experiment and verify it satisfies all selection constraints (2024, at NIF, ignition, highest yield among 2024).",
        parent=parent_node,
        critical=True,
    )

    exp = extraction.experiment or ExperimentSelection()

    # Aggregate sources to increase robustness (any official NIF/LLNL URL among these can support the claim)
    selection_sources = union_urls(
        exp.id_source_urls,
        exp.date_source_urls,
        exp.nif_source_urls,
        exp.ignition_source_urls,
        exp.highest_yield_source_urls,
    )

    # 1) Unique identification
    n1 = evaluator.add_leaf(
        id="Experiment_Unique_Identification_With_Official_Citation",
        desc="Provide a specific identifier/description that uniquely identifies the experiment (e.g., shot identifier and/or uniquely identifying description), supported by an official NIF or LLNL URL.",
        parent=node,
        critical=True,
    )
    ident_disp = safe_str(exp.identifier, "the identified 2024 NIF experiment")
    c1 = (
        f"The official source(s) explicitly identify a single specific NIF experiment in 2024, "
        f"described as {ident_disp} (e.g., via a shot number and/or a uniquely identifying description)."
    )
    i1 = (
        "Only accept support from official NIF or LLNL webpages (e.g., lasers.llnl.gov, llnl.gov). "
        "The page should clearly identify a single specific experiment (not just a general program)."
    )

    # 2) Exact date in 2024
    n2 = evaluator.add_leaf(
        id="Experiment_Exact_Date_In_2024_With_Official_Citation",
        desc="Provide the exact experiment date (month, day, year) and it must fall in calendar year 2024, supported by an official NIF or LLNL URL.",
        parent=node,
        critical=True,
    )
    date_disp = safe_str(exp.date, "an exact date in 2024")
    c2 = (
        f"The experiment occurred on {date_disp}, and that date falls within calendar year 2024."
    )
    i2 = (
        "Verify the exact date on the official NIF/LLNL page. The formatting may vary (e.g., 'December 11, 2024' "
        "or '2024-12-11'), but the year must be 2024."
    )

    # 3) Conducted at NIF
    n3 = evaluator.add_leaf(
        id="Experiment_Conducted_At_NIF_With_Official_Citation",
        desc="Provide evidence the experiment was conducted at the National Ignition Facility (NIF), supported by an official NIF or LLNL URL.",
        parent=node,
        critical=True,
    )
    c3 = (
        "The experiment was conducted at the National Ignition Facility (NIF)."
    )
    i3 = (
        "Confirm the page explicitly ties this specific experiment to NIF (not a different facility). "
        "Only accept official NIF/LLNL sources."
    )

    # 4) Is a fusion ignition experiment
    n4 = evaluator.add_leaf(
        id="Experiment_Is_Fusion_Ignition_With_Official_Citation",
        desc="Provide evidence the experiment is a fusion ignition experiment, supported by an official NIF or LLNL URL.",
        parent=node,
        critical=True,
    )
    c4 = (
        "This experiment is a fusion ignition experiment (i.e., officially described by NIF/LLNL as 'ignition')."
    )
    i4 = (
        "Look for explicit terms like 'ignition', 'achieved ignition', or official definitions consistent with NIF/LLNL usage. "
        "Only accept official NIF/LLNL sources."
    )

    # 5) Highest fusion yield in 2024 justification
    n5 = evaluator.add_leaf(
        id="Highest_Fusion_Yield_In_2024_Justification_With_Official_Citation",
        desc="Provide evidence/justification that this experiment has the highest fusion energy yield among NIF fusion ignition experiments conducted in 2024, supported by official NIF or LLNL URL(s).",
        parent=node,
        critical=True,
    )
    c5 = (
        "This experiment achieved the highest fusion energy yield among NIF fusion ignition experiments conducted during calendar year 2024."
    )
    i5 = (
        "The support should come from official NIF/LLNL sources. The justification should clearly indicate that, "
        "within 2024, no other NIF ignition shot produced a higher fusion energy yield."
    )

    claims_and_sources = [
        (c1, selection_sources, n1, i1),
        (c2, union_urls(exp.date_source_urls, selection_sources), n2, i2),
        (c3, union_urls(exp.nif_source_urls, selection_sources), n3, i3),
        (c4, union_urls(exp.ignition_source_urls, selection_sources), n4, i4),
        (c5, union_urls(exp.highest_yield_source_urls, selection_sources), n5, i5),
    ]

    await evaluator.batch_verify(claims_and_sources)


async def build_energy_checks(
    evaluator: Evaluator,
    parent_node,
    extraction: NIF2024HighestYieldExtraction
) -> None:
    """
    Build and verify 'Experiment_Energy_Metrics' checks (critical, parallel).
    """
    node = evaluator.add_parallel(
        id="Experiment_Energy_Metrics",
        desc="Report the required energy metrics for the identified experiment, with units and official citations.",
        parent=parent_node,
        critical=True,
    )

    eng = extraction.energy or EnergyMetrics()

    # Laser input energy
    n1 = evaluator.add_leaf(
        id="Laser_Input_Energy_MJ_With_Official_Citation",
        desc="Provide the laser input energy delivered to the target, specified in megajoules (MJ), supported by an official NIF or LLNL URL.",
        parent=node,
        critical=True,
    )
    laser_disp = safe_str(eng.laser_input_energy_mj, "a stated laser input energy in MJ")
    c1 = f"The laser input energy delivered to the target for this experiment was {laser_disp} (in megajoules, MJ)."
    i1 = (
        "Confirm that the official NIF/LLNL source states the laser input energy (in MJ). "
        "Allow minor rounding differences or formatting variations (e.g., '2.05 MJ' vs '2.0 MJ')."
    )

    # Fusion energy yield
    n2 = evaluator.add_leaf(
        id="Fusion_Energy_Yield_MJ_With_Official_Citation",
        desc="Provide the fusion energy yield produced, specified in megajoules (MJ), supported by an official NIF or LLNL URL.",
        parent=node,
        critical=True,
    )
    yield_disp = safe_str(eng.fusion_yield_mj, "a stated fusion energy yield in MJ")
    c2 = f"The fusion energy yield produced by this experiment was {yield_disp} (in megajoules, MJ)."
    i2 = (
        "Confirm that the official NIF/LLNL source states the fusion energy yield (in MJ). "
        "Allow minor rounding or formatting differences."
    )

    claims_and_sources = [
        (c1, list(eng.laser_input_source_urls or []), n1, i1),
        (c2, list(eng.fusion_yield_source_urls or []), n2, i2),
    ]

    await evaluator.batch_verify(claims_and_sources)


async def build_context_checks(
    evaluator: Evaluator,
    parent_node,
    extraction: NIF2024HighestYieldExtraction
) -> None:
    """
    Build and verify 'NIF_Facility_Context' checks (critical, parallel).
    """
    node = evaluator.add_parallel(
        id="NIF_Facility_Context",
        desc="Provide required contextual information about NIF (location and current director), with official citations.",
        parent=parent_node,
        critical=True,
    )

    ctx = extraction.context or NIFContext()

    # City
    n1 = evaluator.add_leaf(
        id="NIF_Location_City_With_Official_Citation",
        desc="Provide the city where NIF is located, supported by an official NIF or LLNL URL.",
        parent=node,
        critical=True,
    )
    city_disp = safe_str(ctx.city, "the correct city for NIF")
    c1 = f"The National Ignition Facility (NIF) is located in {city_disp}."
    i1 = (
        "Confirm the city of NIF on an official NIF/LLNL page. The page may present 'Livermore' or 'Livermore, California'. "
        "Focus on city content; do not infer beyond the page."
    )

    # State
    n2 = evaluator.add_leaf(
        id="NIF_Location_State_With_Official_Citation",
        desc="Provide the state where NIF is located, supported by an official NIF or LLNL URL.",
        parent=node,
        critical=True,
    )
    state_disp = safe_str(ctx.state, "the correct U.S. state for NIF")
    c2 = f"The National Ignition Facility (NIF) is located in the state of {state_disp}."
    i2 = (
        "Confirm the U.S. state for NIF on an official NIF/LLNL page (e.g., California). "
        "Do not rely on external knowledge; use the page content."
    )

    # Current NIF Director
    n3 = evaluator.add_leaf(
        id="Current_NIF_Director_Full_Name_With_Official_Citation",
        desc="Provide the full name of the current NIF Director, supported by an official NIF or LLNL URL.",
        parent=node,
        critical=True,
    )
    director_disp = safe_str(ctx.director_full_name, "the current NIF Director's full name")
    c3 = f"The current Director of the National Ignition Facility (NIF) is {director_disp}."
    i3 = (
        "Verify on an official NIF/LLNL page that this person is the current NIF Director. "
        "Allow minor name formatting variants (e.g., middle initials)."
    )

    claims_and_sources = [
        (c1, list(ctx.city_source_urls or []), n1, i1),
        (c2, list(ctx.state_source_urls or []), n2, i2),
        (c3, list(ctx.director_source_urls or []), n3, i3),
    ]

    await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the '2024 NIF Highest Fusion Yield Experiment' task and return a structured result summary.
    """
    # Initialize evaluator and root
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_nif_2024_highest(),
        template_class=NIF2024HighestYieldExtraction,
        extraction_name="nif_2024_highest_extraction",
    )

    # Build top-level critical node mirroring the rubric root (so children must all pass)
    top_node = evaluator.add_parallel(
        id="2024_NIF_Highest_Yield_Experiment",
        desc="Identify the specific NIF fusion ignition experiment in calendar year 2024 with the highest fusion energy yield, and provide required experiment details plus NIF context, each supported by official NIF/LLNL URLs.",
        parent=root,
        critical=True,
    )

    # Build and verify subtrees (all critical)
    await build_selection_checks(evaluator, top_node, extraction)
    await build_energy_checks(evaluator, top_node, extraction)
    await build_context_checks(evaluator, top_node, extraction)

    # Return the structured evaluation summary
    return evaluator.get_summary()