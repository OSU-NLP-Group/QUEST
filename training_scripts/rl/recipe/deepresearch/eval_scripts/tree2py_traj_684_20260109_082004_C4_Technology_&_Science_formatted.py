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
TASK_ID = "battery_facility_name_ms"
TASK_DESCRIPTION = (
    "A battery cell manufacturing facility for commercial vehicles (heavy and medium duty trucks) was planned as a joint venture "
    "between exactly three companies: Accelera by Cummins, Daimler Truck, and PACCAR. This facility is located in the state of "
    "Mississippi on a 500-acre site. The building itself is 2 million square feet in size, with a planned annual manufacturing "
    "capacity of 21 gigawatt-hours (GWh). Construction on this facility began in July 2024. The facility was originally planned "
    "to start production in 2027, but this timeline was later delayed to 2028. What is the name of this battery cell manufacturing facility?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FacilityExtraction(BaseModel):
    """
    Extract a single proper facility name and all explicit source URLs cited in the answer.
    """
    facility_name: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facility() -> str:
    return """
    Your task is to extract from the answer:
    1) facility_name: The single explicit proper name of the battery cell manufacturing facility that the answer claims as the final answer.
       - This should be a proper name (e.g., "[Name] Battery Cell Plant", "[Name] Battery Factory", "[Name] Gigafactory", etc.), not a generic description.
       - If multiple names are mentioned, pick the single primary/official facility name the answer presents as the answer.
       - Do NOT return company names (e.g., Cummins, Daimler Truck, PACCAR) as the facility name. The facility name should be the plant/facility's proper name.
       - If the answer fails to provide a clear, single proper facility name, return null.

    2) source_urls: An array of all explicit URLs cited in the answer text that are intended to support the facts about this facility.
       - Include URLs in any reasonable format (plain link or markdown).
       - Deduplicate obvious duplicates.
       - Only include valid-looking URLs. If missing protocol, prepend http://
       - If no URLs are provided in the answer, return an empty array.

    Return JSON with keys: facility_name (string or null), source_urls (array of strings).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _mk_claims_for_constraints(facility_name: Optional[str]) -> List[Dict[str, str]]:
    """
    Build the set of claim + additional_instruction pairs for all constraints.
    We keep each check as a single, atomic verification leaf.
    """
    # Use a neutral referent when name is missing; sequential gating will skip these anyway if name is not provided.
    subject = f"the facility named '{facility_name}'" if facility_name else "the facility"

    claims = [
        {
            "id": "Facility_Type",
            "desc": "The facility must be a battery cell manufacturing plant for commercial vehicles (heavy and medium duty trucks)",
            "claim": f"{subject} is a battery cell manufacturing plant intended for commercial vehicles (heavy- and medium-duty trucks).",
            "add_ins": (
                "Confirm that this facility manufactures battery cells (not just packs or modules) and is dedicated to commercial vehicles. "
                "Accept wording like 'commercial vehicles', 'heavy-duty trucks', 'medium-duty trucks', and similar synonyms."
            ),
        },
        {
            "id": "Location_State",
            "desc": "The facility must be located in the state of Mississippi",
            "claim": f"{subject} is located in the U.S. state of Mississippi.",
            "add_ins": (
                "Look for 'Mississippi' (or 'MS') as the state for the facility location. "
                "Mentions of specific Mississippi counties/cities are fine as long as the state is explicitly Mississippi."
            ),
        },
        {
            "id": "Joint_Venture_Partners",
            "desc": "The facility must be a joint venture between exactly three companies: Accelera by Cummins, Daimler Truck, and PACCAR",
            "claim": (
                f"{subject} is a joint venture between exactly three companies: "
                "Accelera by Cummins, Daimler Truck, and PACCAR."
            ),
            "add_ins": (
                "Verify that there are exactly these three partners and no additional JV partners. "
                "Treat 'Accelera by Cummins' as acceptable even if phrased as 'Accelera (a business segment of Cummins)' or similar. "
                "Treat 'Daimler Truck' as acceptable even if phrased as 'Daimler Truck North America (DTNA)' or 'Daimler Truck AG'. "
                "Treat 'PACCAR' as acceptable even if phrased as 'PACCAR Inc'."
            ),
        },
        {
            "id": "Annual_Capacity",
            "desc": "The facility's planned annual manufacturing capacity must be 21 gigawatt-hours (GWh)",
            "claim": f"{subject} has a planned annual manufacturing capacity of about 21 GWh.",
            "add_ins": (
                "Accept approximate phrasing such as 'approximately 21 GWh' or 'around 21 GWh'. "
                "The value should clearly refer to annual cell manufacturing capacity."
            ),
        },
        {
            "id": "Building_Size",
            "desc": "The facility's building must be 2 million square feet in size",
            "claim": f"The building area of {subject} is about 2 million square feet.",
            "add_ins": (
                "Accept approximate phrasing such as 'approximately 2 million square feet', 'over 2 million sq ft', or '2,000,000 square feet'."
            ),
        },
        {
            "id": "Construction_Start",
            "desc": "Construction on the facility must have begun in July 2024",
            "claim": f"Construction on {subject} began in July 2024.",
            "add_ins": (
                "Phrases like 'construction began', 'construction started', or 'groundbreaking' in July 2024 should be treated as satisfying this condition."
            ),
        },
        {
            "id": "Site_Size",
            "desc": "The facility must be located on a 500-acre site",
            "claim": f"{subject} is sited on approximately 500 acres.",
            "add_ins": (
                "Accept approximate phrasing like 'about 500 acres' or 'approx. 500-acre site'."
            ),
        },
    ]

    # Split the production timeline into two atomic leaves to avoid multi-source cross-page coupling
    claims.extend([
        {
            "id": "Production_Originally_2027",
            "desc": "Original plan: production start year was 2027",
            "claim": f"The original plan for {subject} scheduled production start in 2027.",
            "add_ins": (
                "Look for earlier announcements or planning documents stating 2027 as the initial production start target. "
                "This check only concerns the original/initial plan."
            ),
        },
        {
            "id": "Production_Delayed_2028",
            "desc": "Updated plan: production start later delayed to 2028",
            "claim": f"{subject}'s production start timeline was subsequently updated/delayed to 2028.",
            "add_ins": (
                "Look for later announcements or updates that changed the start of production to 2028. "
                "This check only concerns the later update to 2028."
            ),
        },
    ])

    return claims


# --------------------------------------------------------------------------- #
# Tree building and verification                                              #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root_node,
    extracted: FacilityExtraction,
) -> None:
    """
    Build the verification tree based on the rubric and run the corresponding checks.
    """
    # Create the main sequential node representing the whole question (critical)
    q_node = evaluator.add_sequential(
        id="Facility_Name_Question",
        desc="Answer identifies the name of the battery cell manufacturing facility described in the prompt",
        parent=root_node,
        critical=True,
    )

    # 1) Facility name provided (single, explicit) — treat as existence/format gate (critical)
    name_ok = bool(extracted.facility_name and isinstance(extracted.facility_name, str) and extracted.facility_name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id="Facility_Name_Provided",
        desc="Response provides a single, explicit facility name (proper name) as the answer",
        parent=q_node,
        critical=True,
    )

    # 2) Constraints bundle (critical, parallel)
    constraints_node = evaluator.add_parallel(
        id="Facility_Matches_All_Stated_Constraints",
        desc="The named facility satisfies all constraints stated in the prompt",
        parent=q_node,
        critical=True,
    )

    # 2.a) Create the eight atomic constraint leaves (split production timeline into two atomic checks)
    # We will place all leaves directly under the constraints node; each is critical.
    claims = _mk_claims_for_constraints(extracted.facility_name)
    leaves: List = []
    for c in claims:
        leaf = evaluator.add_leaf(
            id=c["id"],
            desc=c["desc"],
            parent=constraints_node,
            critical=True,
        )
        leaves.append((c, leaf))

    # 2.b) Run verifications (prefer evidence from URLs; falls back to simple verify when no URLs are provided)
    url_sources: List[str] = extracted.source_urls or []
    batch: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []
    for c, leaf_node in leaves:
        batch.append((
            c["claim"],
            url_sources if len(url_sources) > 1 else (url_sources[0] if len(url_sources) == 1 else None),
            leaf_node,
            c["add_ins"],
        ))

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
    Evaluate an answer for the Mississippi battery cell facility naming task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root wrapper; real logic lives under a critical sequential child
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

    # Extract the facility name and the cited source URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_facility(),
        template_class=FacilityExtraction,
        extraction_name="facility_extraction",
    )

    # Optional ground truth info (what we will check for)
    evaluator.add_ground_truth({
        "must_have": {
            "state": "Mississippi",
            "site_size": "500 acres (approx.)",
            "building_area": "2 million square feet (approx.)",
            "capacity": "21 GWh (approx.)",
            "construction_start": "July 2024",
            "production_timeline": {
                "original": 2027,
                "updated": 2028
            },
            "joint_venture_partners": [
                "Accelera by Cummins",
                "Daimler Truck",
                "PACCAR"
            ],
            "facility_type": "Battery cell manufacturing for commercial vehicles (heavy- and medium-duty trucks)"
        }
    }, gt_type="expected_constraints_summary")

    # Build the tree and run all verifications
    await build_and_verify_tree(evaluator, root, extracted)

    # Return standardized evaluation summary
    return evaluator.get_summary()