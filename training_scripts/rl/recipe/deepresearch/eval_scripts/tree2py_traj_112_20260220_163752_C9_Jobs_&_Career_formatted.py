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
TASK_ID = "tx_superintendents_feb2026"
TASK_DESCRIPTION = (
    "For each of the following four large Texas school districts—Houston ISD, Cypress-Fairbanks ISD, Northside ISD, and North East ISD—"
    "identify the current superintendent as of February 2026 and provide the following information: "
    "(1) The superintendent's full legal name, "
    "(2) The superintendent's highest earned degree (which must be a Master's degree or higher to meet Texas Education Agency requirements), "
    "(3) Evidence that the superintendent holds or has held a Texas Principal certificate or equivalent administrative certification "
    "(as required by TEA for superintendent certification), and "
    "(4) The superintendent's base salary as reported in the Texas Education Agency superintendent salary database for the 2024-2025 school year. "
    "For each piece of information, provide a reference URL from an official or authoritative source (such as the district's official website, "
    "the TEA database at data.texas.gov, or other publicly available official documentation)."
)

DISTRICTS = [
    {"key": "houston_isd", "display": "Houston ISD"},
    {"key": "cypress_fairbanks_isd", "display": "Cypress-Fairbanks ISD"},
    {"key": "northside_isd", "display": "Northside ISD"},
    {"key": "north_east_isd", "display": "North East ISD"},
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DistrictInfo(BaseModel):
    full_name: Optional[str] = None
    identity_urls: List[str] = Field(default_factory=list)

    degree: Optional[str] = None  # Highest earned degree (string)
    degree_urls: List[str] = Field(default_factory=list)

    certification_evidence: Optional[str] = None  # Free text evidence/summary
    certification_urls: List[str] = Field(default_factory=list)

    base_salary_2024_25: Optional[str] = None  # Keep as string to allow symbols/ranges
    salary_urls: List[str] = Field(default_factory=list)


class FourDistrictsExtraction(BaseModel):
    houston_isd: Optional[DistrictInfo] = None
    cypress_fairbanks_isd: Optional[DistrictInfo] = None
    northside_isd: Optional[DistrictInfo] = None
    north_east_isd: Optional[DistrictInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_four_districts() -> str:
    return """
Extract the superintendent information for the following four Texas school districts as explicitly presented in the answer text:
- Houston ISD
- Cypress-Fairbanks ISD
- Northside ISD
- North East ISD

For each district, extract the following fields:
1) full_name: The superintendent's full legal name exactly as written in the answer.
2) identity_urls: An array of URL(s) to official district webpage(s) that confirm this person is the current superintendent. Extract only URLs explicitly present.
3) degree: The superintendent's highest earned degree as stated (e.g., "Ed.D. in Educational Leadership", "M.Ed.", "Ph.D.", etc.). Use the exact text from the answer when possible.
4) degree_urls: An array of URL(s) providing evidence of the degree. Extract only URLs explicitly present.
5) certification_evidence: A short snippet from the answer indicating the superintendent holds or has held a Texas Principal certificate or equivalent administrative certification (e.g., "Principal EC-12", "Mid-Management Administrator").
6) certification_urls: An array of URL(s) providing evidence of the certification. Extract only URLs explicitly present.
7) base_salary_2024_25: The base salary for the superintendent as reported in the TEA superintendent salary database for the 2024-2025 school year, exactly as written in the answer (e.g., "$345,000", "345000", etc.).
8) salary_urls: An array of URL(s) to the TEA database page(s) (e.g., on data.texas.gov) or other official documentation where this salary is shown. Extract only URLs explicitly present.

Return a JSON object with exactly four top-level keys:
- "houston_isd"
- "cypress_fairbanks_isd"
- "northside_isd"
- "north_east_isd"

Each value should be an object with the fields described above. If any field is missing in the answer, set it to null (for strings) or [] (for URL lists).
Do not infer or invent any URLs or values that are not explicitly present in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper: Build safe strings                                                  #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s if s is not None else ""


def _urls(lst: Optional[List[str]]) -> List[str]:
    return lst if lst else []


# --------------------------------------------------------------------------- #
# Verification for a single district                                          #
# --------------------------------------------------------------------------- #
async def verify_district(
    evaluator: Evaluator,
    parent_node,
    district_key: str,
    district_display: str,
    info: Optional[DistrictInfo],
) -> None:
    # Create district node (Sequential as per rubric)
    district_node = evaluator.add_sequential(
        id=district_key,
        desc=f"{district_display} superintendent information and qualification verification",
        parent=parent_node,
        critical=False,  # Allow partial at root level across districts
    )

    # If info is None, create placeholder to avoid attribute errors; nodes will fail appropriately
    info = info or DistrictInfo()

    # ------------------------- Superintendent Identity ------------------------- #
    identity_node = evaluator.add_parallel(
        id=f"{district_key}_identity",
        desc=f"Identification of current {district_display} superintendent",
        parent=district_node,
        critical=True,  # Critical block
    )

    # Full name provided (existence check)
    evaluator.add_custom_node(
        result=bool(info.full_name and info.full_name.strip()),
        id=f"{district_key}_full_name_provided",
        desc="Full legal name of superintendent provided",
        parent=identity_node,
        critical=True,
    )

    # Identity reference URL(s) verify
    identity_leaf = evaluator.add_leaf(
        id=f"{district_key}_identity_reference_url",
        desc="Reference URL confirming superintendent identity from official district source",
        parent=identity_node,
        critical=True,
    )
    identity_claim_name = _safe(info.full_name).strip()
    identity_claim = (
        f"The cited official district page(s) confirm that {identity_claim_name} is the current superintendent of "
        f"{district_display} as of February 2026."
    )
    await evaluator.verify(
        claim=identity_claim,
        node=identity_leaf,
        sources=_urls(info.identity_urls),
        additional_instruction=(
            "Check the provided official district webpage(s). Accept clear statements like 'Superintendent' or "
            "'Superintendent of Schools'. Minor wording variations are acceptable. If the page clearly indicates the "
            "person is the current superintendent, consider it supported. If the page explicitly labels an 'Interim Superintendent', "
            "treat it as the current superintendent only if the answer likewise indicates 'Interim'."
        ),
    )

    # ---------------------- Superintendent Qualifications ---------------------- #
    qual_node = evaluator.add_parallel(
        id=f"{district_key}_qualifications",
        desc="Verification of required qualifications per TEA regulations",
        parent=district_node,
        critical=False,  # Non-critical block within the district
    )

    # -- Educational Degree --
    degree_node = evaluator.add_parallel(
        id=f"{district_key}_degree",
        desc="Educational degree verification",
        parent=qual_node,
        critical=True,  # Critical sub-block
    )

    # Degree meets TEA requirement (Master's or higher) - verify against cited degree URLs
    degree_meets_leaf = evaluator.add_leaf(
        id=f"{district_key}_degree_meets_requirement",
        desc="Superintendent holds Master's degree or higher as required by TEA",
        parent=degree_node,
        critical=True,
    )
    degree_text = _safe(info.degree).strip()
    degree_claim = (
        f"According to the cited page(s), the highest earned degree for {identity_claim_name} is '{degree_text}', "
        "which is at least a Master's degree."
    )
    await evaluator.verify(
        claim=degree_claim,
        node=degree_meets_leaf,
        sources=_urls(info.degree_urls),
        additional_instruction=(
            "Verify that the page(s) indicate a Master's degree or a higher degree (e.g., M.Ed., M.A., M.S., MBA, "
            "Ed.M., MEng, J.D., M.P.A., Ph.D., Ed.D., Doctorate). Reasonable abbreviations are acceptable. "
            "If multiple degrees are listed, consider the highest degree."
        ),
    )

    # Degree reference URL presence (existence check for sources)
    evaluator.add_custom_node(
        result=bool(info.degree_urls and len(info.degree_urls) > 0),
        id=f"{district_key}_degree_reference_url",
        desc="Reference URL providing evidence of educational degree",
        parent=degree_node,
        critical=True,
    )

    # -- Principal Certification --
    cert_node = evaluator.add_parallel(
        id=f"{district_key}_certification",
        desc="Principal certification verification",
        parent=qual_node,
        critical=True,  # Critical sub-block
    )

    cert_leaf = evaluator.add_leaf(
        id=f"{district_key}_certification_evidence",
        desc="Evidence of Texas Principal certificate or equivalent administrative certification",
        parent=cert_node,
        critical=True,
    )
    cert_claim = (
        f"The cited page(s) provide evidence that {identity_claim_name} holds or has held a Texas Principal certificate "
        "or an equivalent administrative certification required for superintendent certification in Texas."
    )
    await evaluator.verify(
        claim=cert_claim,
        node=cert_leaf,
        sources=_urls(info.certification_urls),
        additional_instruction=(
            "Accept evidence such as 'Texas Principal Certificate', 'Principal EC-12', 'Mid-Management Administrator', "
            "'Administrator' where applicable under historical certification frameworks, or other official Texas admin "
            "certification indicating eligibility for superintendent certification. The page must clearly support this."
        ),
    )

    evaluator.add_custom_node(
        result=bool(info.certification_urls and len(info.certification_urls) > 0),
        id=f"{district_key}_certification_reference_url",
        desc="Reference URL providing evidence of certification",
        parent=cert_node,
        critical=True,
    )

    # -- Salary Information --
    salary_node = evaluator.add_parallel(
        id=f"{district_key}_salary",
        desc="Salary data verification from TEA database",
        parent=qual_node,
        critical=True,  # Critical sub-block
    )

    salary_leaf = evaluator.add_leaf(
        id=f"{district_key}_base_salary_reported",
        desc="Base salary amount from 2024-25 TEA superintendent salary database provided",
        parent=salary_node,
        critical=True,
    )
    salary_value = _safe(info.base_salary_2024_25).strip()
    salary_claim = (
        f"The Texas Education Agency superintendent salary database for the 2024-25 school year reports a base salary "
        f"of '{salary_value}' for the {district_display} superintendent {identity_claim_name}."
    )
    await evaluator.verify(
        claim=salary_claim,
        node=salary_leaf,
        sources=_urls(info.salary_urls),
        additional_instruction=(
            "Verify the base salary on the cited TEA or official salary documentation (e.g., data.texas.gov). "
            "Allow for formatting variations such as commas or currency symbols. The school year must be 2024-2025 "
            "or the corresponding FY 2025 notation."
        ),
    )

    evaluator.add_custom_node(
        result=bool(info.salary_urls and len(info.salary_urls) > 0),
        id=f"{district_key}_salary_reference_url",
        desc="Reference URL to TEA database or official salary documentation",
        parent=salary_node,
        critical=True,
    )


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the four large Texas districts' superintendent analysis.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is parallel across districts
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

    # Extract structured information for all four districts
    extracted = await evaluator.extract(
        prompt=prompt_extract_four_districts(),
        template_class=FourDistrictsExtraction,
        extraction_name="four_districts_extraction",
    )

    # Create top-level node grouping (optional; root is already parallel, but we add a descriptive node)
    top_node = evaluator.add_parallel(
        id="Four_Large_Texas_Districts_Superintendent_Analysis",
        desc="Comprehensive analysis of current superintendents in four major Texas school districts",
        parent=root,
        critical=False,
    )

    # Map extracted info by key for convenience
    info_map: Dict[str, Optional[DistrictInfo]] = {
        "houston_isd": extracted.houston_isd,
        "cypress_fairbanks_isd": extracted.cypress_fairbanks_isd,
        "northside_isd": extracted.northside_isd,
        "north_east_isd": extracted.north_east_isd,
    }

    # Build verification trees for each district
    for d in DISTRICTS:
        key = d["key"]
        display = d["display"]
        await verify_district(
            evaluator=evaluator,
            parent_node=top_node,
            district_key=key,
            district_display=display,
            info=info_map.get(key),
        )

    return evaluator.get_summary()