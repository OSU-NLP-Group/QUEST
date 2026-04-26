import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "doe_qis_centers_5"
TASK_DESCRIPTION = (
    "Identify all five U.S. Department of Energy (DOE) National Quantum Information Science (QIS) Research Centers. "
    "For each center, provide the following information: "
    "(1) Official Center Name: The full official name or acronym of the center, "
    "(2) Lead Institution: The primary national laboratory or organization leading the center, "
    "(3) Location: The city and state where the lead institution is located, "
    "(4) Director: The name of the center director, and "
    "(5) Reference URL: A link to an official page (from DOE, the lead institution, or the center's website) that "
    "confirms the center's designation and details."
)

# Known DOE NQISRCs (for info only; not enforced in verification)
GROUND_TRUTH_INFO = {
    "expected_centers": [
        {"name": "Q-NEXT", "lead_institution": "Argonne National Laboratory"},
        {"name": "Co-design Center for Quantum Advantage (C2QA)", "lead_institution": "Brookhaven National Laboratory"},
        {"name": "Quantum Science Center (QSC)", "lead_institution": "Oak Ridge National Laboratory"},
        {"name": "Superconducting Quantum Materials and Systems Center (SQMS)", "lead_institution": "Fermilab"},
        {"name": "Quantum Systems Accelerator (QSA)", "lead_institution": "Lawrence Berkeley National Laboratory"},
    ],
    "note": "This ground truth is recorded for reference only. All verification must rely on the URLs provided in the answer."
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CenterEntry(BaseModel):
    official_center_name_or_acronym: Optional[str] = None
    lead_institution: Optional[str] = None
    location_city_state: Optional[str] = None
    director_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class CentersExtraction(BaseModel):
    centers: List[CenterEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_centers() -> str:
    return (
        "Extract up to five entries that the answer claims are DOE National Quantum Information Science (QIS) "
        "Research Centers. For each entry, extract the following fields exactly as written in the answer:\n"
        "1. official_center_name_or_acronym: The official center name or acronym as provided (e.g., 'Q-NEXT', "
        "'Co-design Center for Quantum Advantage (C2QA)', 'Quantum Systems Accelerator (QSA)', etc.).\n"
        "2. lead_institution: The primary national laboratory or organization claimed to lead the center (e.g., "
        "'Argonne National Laboratory', 'Fermilab').\n"
        "3. location_city_state: The city and state where the lead institution is located (e.g., 'Berkeley, California', "
        "'Argonne, Illinois').\n"
        "4. director_name: The name of the center director (e.g., 'David Awschalom').\n"
        "5. reference_urls: A list of all URLs in the answer that are presented as references for this center. "
        "These should be official pages if the answer provides them (DOE sites, national lab sites, or the center's own site). "
        "Extract only explicit URLs present in the answer; do not invent or infer new URLs.\n\n"
        "Return a JSON object with a single field 'centers', which is an array of up to five objects of the form above. "
        "If a field is missing for a center, set it to null (or [] for reference_urls). "
        "If the answer provides more than five centers, include only the first five. If fewer, include only those provided."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    # 1 -> 1st, 2 -> 2nd, ...
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def nonempty(s: Optional[str]) -> str:
    return s if s is not None else ""


def official_url_guidance() -> str:
    # Guidance text for verifying official sources; used in multiple claims
    return (
        "A URL is considered 'official' if it is from: "
        "(a) a DOE domain (e.g., energy.gov, science.osti.gov), "
        "(b) the lead institution's official domain (e.g., anl.gov, bnl.gov, ornl.gov, lbl.gov, fnal.gov), or "
        "(c) the center's own official website (e.g., q-next.org, qsa.lbl.gov, qsc.ornl.gov, c2qa.bnl.gov, sqms.fnal.gov). "
        "Use reasonable judgment from the webpage branding and content (headers, footers, logos) to decide if the page is official. "
        "Treat the claim as supported if at least one provided URL qualifies."
    )


# --------------------------------------------------------------------------- #
# Verification for one center                                                 #
# --------------------------------------------------------------------------- #
async def verify_one_center(
    evaluator: Evaluator,
    parent_node,
    center: CenterEntry,
    index: int,
) -> None:
    idx = index + 1
    center_node = evaluator.add_parallel(
        id=f"Center_{idx}",
        desc=f"Evaluate the {ordinal(idx)} provided center entry.",
        parent=parent_node,
        critical=False  # Allow partial credit across centers
    )

    name = nonempty(center.official_center_name_or_acronym).strip()
    lead = nonempty(center.lead_institution).strip()
    loc = nonempty(center.location_city_state).strip()
    director = nonempty(center.director_name).strip()
    urls = center.reference_urls or []

    # Leaf 1: Is official DOE QIS center
    leaf_is_official = evaluator.add_leaf(
        id=f"center_{idx}_Is_official_DOE_QIS_center",
        desc="The center is one of the five officially designated DOE National QIS Research Centers (per an official source).",
        parent=center_node,
        critical=True
    )
    claim_is_official = (
        f"This webpage confirms that the center '{name}' is a U.S. Department of Energy National Quantum Information "
        f"Science Research Center (NQISRC). It is sufficient if the page explicitly states it is a DOE National QIS "
        f"Research Center; it does not need to say 'one of five'."
    )

    # Leaf 2: Official center name or acronym
    leaf_official_name = evaluator.add_leaf(
        id=f"center_{idx}_Official_center_name_or_acronym",
        desc="Provides the official center name and/or acronym exactly as designated by DOE.",
        parent=center_node,
        critical=True
    )
    claim_official_name = (
        f"This webpage presents the center's official name or commonly used acronym as '{name}', "
        f"referring to the same center."
    )

    # Leaf 3: Lead institution
    leaf_lead = evaluator.add_leaf(
        id=f"center_{idx}_Lead_institution",
        desc="Correctly identifies the lead institution (primary national laboratory or organization leading the center).",
        parent=center_node,
        critical=True
    )
    claim_lead = (
        f"This webpage confirms that the lead institution for the center '{name}' is '{lead}'. "
        f"Accept phrases like 'led by', 'lead institution', 'hosted by', 'managed by', or similar."
    )

    # Leaf 4: Location city/state
    leaf_location = evaluator.add_leaf(
        id=f"center_{idx}_Location_city_state",
        desc="Provides the city and state where the lead institution is located.",
        parent=center_node,
        critical=True
    )
    claim_location = (
        f"This webpage confirms that the lead institution '{lead}' is located in '{loc}'. "
        f"Minor variants (e.g., 'CA' vs 'California') should be accepted as matches."
    )

    # Leaf 5: Director name
    leaf_director = evaluator.add_leaf(
        id=f"center_{idx}_Director_name",
        desc="Provides the name of the center director.",
        parent=center_node,
        critical=True
    )
    claim_director = (
        f"This webpage confirms that the director (or center director) of '{name}' is '{director}'. "
        f"Accept synonyms such as 'Center Director' or 'Director'. If the page states an 'Executive Director' who is clearly "
        f"the center's primary director, that is acceptable."
    )

    # Leaf 6: Reference URL official
    leaf_reference_official = evaluator.add_leaf(
        id=f"center_{idx}_Reference_URL_official",
        desc="Provides a valid reference URL to an official page (DOE, lead institution, or center website) confirming the center’s designation/details.",
        parent=center_node,
        critical=True
    )
    claim_reference_official = (
        f"This webpage is an official page by DOE, the lead institution, or the center itself, "
        f"and it confirms the center's designation or key details for '{name}'."
    )

    # Batch verify all six leaves in parallel for this center
    claims_and_sources = [
        (claim_is_official, urls, leaf_is_official,
         "Verify that the page explicitly indicates the center is a DOE National Quantum Information Science "
         "Research Center (NQISRC). Prefer explicit wording like 'DOE National Quantum Information Science Research Center' "
         "or 'National QIS Research Center'. "
         "Reject if the page is unrelated or does not support the designation."),
        (claim_official_name, urls, leaf_official_name,
         "Allow minor formatting differences (capitalization, hyphenation). "
         "The page should clearly identify the center using the provided name or acronym."),
        (claim_lead, urls, leaf_lead,
         "Look for phrases like 'led by', 'lead institution', 'hosted by', or 'managed by'. "
         "The identified institution must match or clearly refer to the same organization."),
        (claim_location, urls, leaf_location,
         "Check that the page (or the official site it belongs to) supports the city and state of the lead institution. "
         "Accept minor variants such as 'CA' vs 'California'."),
        (claim_director, urls, leaf_director,
         "Confirm that the page identifies the person as the Director of the center. "
         "Titles like 'Center Director' are acceptable. If the page lists 'Executive Director' clearly in the context of the "
         "center's leadership, accept it as equivalent."),
        (claim_reference_official, urls, leaf_reference_official,
         official_url_guidance() + " The page should also confirm the center's designation or core details "
         "(e.g., name, lead, or director)."),
    ]

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
    model: str = "o4-mini",
) -> Dict:
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

    # Add a main node to mirror the rubric's top-level container
    main_node = evaluator.add_parallel(
        id="DOE_National_QIS_Research_Centers",
        desc="Response identifies five DOE National Quantum Information Science (QIS) Research Centers and, for each one, provides the required fields with official-source verification.",
        parent=root,
        critical=False  # Set to non-critical to allow partial credit across centers (adjusted from rubric to satisfy framework constraints)
    )

    # Extract centers from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_centers(),
        template_class=CentersExtraction,
        extraction_name="centers_extraction",
    )

    # Record ground truth info (for reference only)
    evaluator.add_ground_truth(GROUND_TRUTH_INFO, gt_type="reference_centers")

    # Prepare exactly five centers (pad with empty entries if fewer)
    centers = list(extracted.centers[:5])
    while len(centers) < 5:
        centers.append(CenterEntry())

    # Verify each of the five centers
    for idx, center in enumerate(centers):
        await verify_one_center(evaluator, main_node, center, idx)

    # Optionally record a quick summary of extracted names for convenience
    evaluator.add_custom_info(
        info={"extracted_center_names": [c.official_center_name_or_acronym for c in centers]},
        info_type="custom",
        info_name="extraction_summary"
    )

    return evaluator.get_summary()