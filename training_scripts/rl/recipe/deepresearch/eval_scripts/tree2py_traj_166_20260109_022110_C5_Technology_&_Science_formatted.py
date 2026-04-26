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
TASK_ID = "asian_solid_state_battery_2024"
TASK_DESCRIPTION = """
Which Asian battery manufacturer announced in 2024 that their all-solid-state battery technology has achieved a volumetric energy density of at least 800 Wh/L and has set a mass production target of 2027 or earlier? The company must have progressed beyond pure research by establishing pilot production capabilities or delivering sample units, and the technology must utilize solid electrolyte materials. Provide the company's full name and confirm its headquarters location in Asia.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CompanyExtraction(BaseModel):
    company_name: Optional[str] = None
    headquarters_location: Optional[str] = None  # e.g., "Tokyo, Japan"
    headquarters_country: Optional[str] = None   # e.g., "Japan"
    company_type: Optional[str] = None           # e.g., "battery manufacturer" or "automotive company"
    manufacturing_capabilities: Optional[str] = None  # any mention of manufacturing capability
    identity_sources: List[str] = Field(default_factory=list)  # URLs supporting identity/headquarters/type info


class AnnouncementExtraction(BaseModel):
    announcement_year: Optional[str] = None  # year like "2024"
    volumetric_energy_density_text: Optional[str] = None  # e.g., "800 Wh/L", "≥800 Wh/L"
    volumetric_energy_density_value: Optional[str] = None  # numeric string if available, else null
    volumetric_energy_density_unit: Optional[str] = None   # e.g., "Wh/L"
    mass_production_target_year: Optional[str] = None      # e.g., "2027", "2026"
    pilot_or_samples_evidence: Optional[str] = None        # text indicating pilot line or sample deliveries
    uses_solid_electrolyte: Optional[bool] = None          # true/false if explicitly stated
    electrolyte_materials: Optional[str] = None            # e.g., "sulfide solid electrolyte"
    spec_sources: List[str] = Field(default_factory=list)  # URLs supporting announcement/specifications


class SolidStateEvaluationExtraction(BaseModel):
    company: Optional[CompanyExtraction] = None
    announcement: Optional[AnnouncementExtraction] = None
    all_sources: List[str] = Field(default_factory=list)   # any general sources mentioned in the answer


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_company_and_announcement() -> str:
    return """
    Extract the main company and the key announcement details from the answer. Focus on a single, clearly identified company (if multiple are mentioned, select the primary/first one that fits the task).
    Return a JSON object with the following structure:

    {
      "company": {
        "company_name": string or null,
        "headquarters_location": string or null,
        "headquarters_country": string or null,
        "company_type": string or null,
        "manufacturing_capabilities": string or null,
        "identity_sources": array of URLs (can be empty, only include URLs explicitly present in the answer)
      },
      "announcement": {
        "announcement_year": string or null,                      // e.g., "2024"
        "volumetric_energy_density_text": string or null,         // e.g., "800 Wh/L" or "≥800 Wh/L"
        "volumetric_energy_density_value": string or null,        // numeric portion if present, e.g., "800"
        "volumetric_energy_density_unit": string or null,         // expected "Wh/L" or similar
        "mass_production_target_year": string or null,            // e.g., "2027", "2026"
        "pilot_or_samples_evidence": string or null,              // evidence text indicating pilot line or sample units delivery
        "uses_solid_electrolyte": boolean or null,                // true if explicitly stated solid electrolyte is used
        "electrolyte_materials": string or null,                  // e.g., "sulfide solid electrolyte"
        "spec_sources": array of URLs (can be empty, include URLs in the answer that support announcement/specs)
      },
      "all_sources": array of URLs (optional union of all cited URLs in the answer, can be empty)
    }

    Rules:
    - Extract EXACTLY what is present in the answer; do not invent values.
    - For URLs, only include those explicitly present in the answer (including markdown links).
    - If something is not mentioned, use null for that field; arrays can be empty if no URLs are present.
    - Prefer the primary company if multiple are listed; choose the one tied to the 2024 solid-state battery announcement with specs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def collect_sources(ext: SolidStateEvaluationExtraction) -> List[str]:
    """Collect a union of all sources extracted."""
    union: List[str] = []
    if ext:
        if ext.company and ext.company.identity_sources:
            union.extend(ext.company.identity_sources)
        if ext.announcement and ext.announcement.spec_sources:
            union.extend(ext.announcement.spec_sources)
        if ext.all_sources:
            union.extend(ext.all_sources)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for url in union:
        if url and url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_company_identification(
    evaluator: Evaluator,
    parent_node,
    ext: SolidStateEvaluationExtraction,
) -> None:
    """Build and verify the Company Identification subtree."""
    company = ext.company or CompanyExtraction()
    all_urls = collect_sources(ext)
    id_urls = company.identity_sources if company.identity_sources else all_urls

    comp_node = evaluator.add_parallel(
        id="Company_Identification",
        desc="Verify the respondent identified a specific company and provided required identity/location details.",
        parent=parent_node,
        critical=True
    )

    # Company_Full_Name (existence check)
    has_name = bool(company.company_name and company.company_name.strip())
    evaluator.add_custom_node(
        result=has_name,
        id="Company_Full_Name",
        desc="Provides the company's full legal or commonly recognized full name.",
        parent=comp_node,
        critical=True
    )

    # Headquarters_In_Asia (verification)
    hq_leaf = evaluator.add_leaf(
        id="Headquarters_In_Asia",
        desc="Confirms the company's headquarters is in Asia (Japan, South Korea, or China).",
        parent=comp_node,
        critical=True
    )
    hq_loc = company.headquarters_location or company.headquarters_country or ""
    comp_name_safe = company.company_name or "the company"

    await evaluator.verify(
        claim=f"The headquarters of {comp_name_safe} is located in {hq_loc}, which is in Asia (specifically Japan, South Korea, or China).",
        node=hq_leaf,
        sources=id_urls,
        additional_instruction=(
            "Accept synonyms and formal names: 'Republic of Korea' == 'South Korea', 'PRC' == 'China'. "
            "Use the source(s) to determine headquarters. If multiple locations are shown, "
            "use the primary corporate headquarters. If unclear or contradicted, judge as not supported."
        ),
    )

    # Eligible_Company_Type (verification)
    type_leaf = evaluator.add_leaf(
        id="Eligible_Company_Type",
        desc="Company is a major battery manufacturer or automotive company with established manufacturing capabilities.",
        parent=comp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{comp_name_safe} is a major battery manufacturer or automotive company with established manufacturing capabilities.",
        node=type_leaf,
        sources=id_urls,
        additional_instruction=(
            "Look for explicit evidence that the company manufactures batteries or automobiles at scale "
            "(e.g., factories, production lines, OEM status). Marketing-only or R&D-only entities without "
            "manufacturing capabilities should not pass."
        ),
    )

    # Public_2024_Disclosure (verification)
    pub_leaf = evaluator.add_leaf(
        id="Public_2024_Disclosure",
        desc="Relevant announcement/specifications about the solid-state battery were publicly disclosed or updated in 2024.",
        parent=comp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In 2024, {comp_name_safe} publicly disclosed or updated relevant specifications/announcements about its all-solid-state battery technology.",
        node=pub_leaf,
        sources=(ext.announcement.spec_sources if ext.announcement and ext.announcement.spec_sources else all_urls),
        additional_instruction=(
            "Verify that the source clearly indicates a public disclosure or update in the year 2024 "
            "(press release date, news article date, web page update date, etc.). If the evidence shows a different year, fail."
        ),
    )


async def build_technology_constraints(
    evaluator: Evaluator,
    parent_node,
    ext: SolidStateEvaluationExtraction,
) -> None:
    """Build and verify the Technology Constraints subtree."""
    ann = ext.announcement or AnnouncementExtraction()
    all_urls = collect_sources(ext)
    spec_urls = ann.spec_sources if ann.spec_sources else all_urls
    comp_name_safe = (ext.company.company_name if ext.company else None) or "the company"

    tech_node = evaluator.add_parallel(
        id="Technology_Constraints",
        desc="Verify the announced technology meets the technical, maturity, and timeline constraints.",
        parent=parent_node,
        critical=True
    )

    # Volumetric_Energy_Density >= 800 Wh/L
    density_leaf = evaluator.add_leaf(
        id="Volumetric_Energy_Density",
        desc="Announced or demonstrated volumetric energy density is at least 800 Wh/L.",
        parent=tech_node,
        critical=True
    )
    announced_text = ann.volumetric_energy_density_text or ""
    await evaluator.verify(
        claim=(
            f"{comp_name_safe}'s all-solid-state battery announcement states a volumetric energy density of {announced_text}, "
            f"which is at least 800 Wh/L."
            if announced_text else
            f"{comp_name_safe}'s all-solid-state battery announcement states a volumetric energy density of at least 800 Wh/L."
        ),
        node=density_leaf,
        sources=spec_urls,
        additional_instruction=(
            "Look for phrases like '800 Wh/L', '≥800 Wh/L', 'at least 800 Wh/L'. "
            "If the source shows a lower figure or a different metric (e.g., gravimetric Wh/kg only) without volumetric Wh/L, fail."
        ),
    )

    # Mass_Production_Target <= 2027
    mass_leaf = evaluator.add_leaf(
        id="Mass_Production_Target",
        desc="Announced mass production target year is 2027 or earlier.",
        parent=tech_node,
        critical=True
    )
    target_year = ann.mass_production_target_year or "2027"
    await evaluator.verify(
        claim=f"The company's announcement sets mass production target by {target_year}, which is 2027 or earlier.",
        node=mass_leaf,
        sources=spec_urls,
        additional_instruction=(
            "Confirm the mass production target year from the source. If the stated target is after 2027 or only 'post-2027', fail. "
            "Phrases like 'by 2027', 'in 2026', 'target 2027' are acceptable. Vague long-term targets without a year should fail."
        ),
    )

    # Beyond_Pure_Research (pilot capability and/or samples delivered)
    maturity_leaf = evaluator.add_leaf(
        id="Beyond_Pure_Research",
        desc="Shows progress beyond pure research via a pilot production capability and/or delivery of sample units.",
        parent=tech_node,
        critical=True
    )
    evidence_text = ann.pilot_or_samples_evidence or ""
    await evaluator.verify(
        claim=(
            f"{comp_name_safe} has progressed beyond pure research for its all-solid-state battery technology via "
            f"a pilot production capability and/or delivery of sample units. {evidence_text}"
        ),
        node=maturity_leaf,
        sources=spec_urls,
        additional_instruction=(
            "Look for explicit mentions of 'pilot line', 'pilot production', 'sample cells delivered', 'evaluation samples', "
            "or similar. Mere lab prototypes, research publications, or future plans without pilot/sample evidence should fail."
        ),
    )

    # Solid_Electrolyte_Used
    solid_leaf = evaluator.add_leaf(
        id="Solid_Electrolyte_Used",
        desc="Technology uses a solid electrolyte material (not liquid or gel electrolyte).",
        parent=tech_node,
        critical=True
    )
    electrolyte_text = ann.electrolyte_materials or ""
    uses_solid = ann.uses_solid_electrolyte
    claim_text = (
        f"The company's all-solid-state battery technology uses a solid electrolyte material. {electrolyte_text}"
        if electrolyte_text else
        "The company's all-solid-state battery technology uses a solid electrolyte material."
    )
    await evaluator.verify(
        claim=claim_text,
        node=solid_leaf,
        sources=spec_urls,
        additional_instruction=(
            "Confirm the electrolyte is solid (e.g., sulfide, oxide, polymer solid). If the source indicates liquid or gel electrolyte, fail. "
            "Mentions of 'all-solid-state battery' typically imply solid electrolyte; confirm explicitly if possible."
        ),
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
) -> Dict[str, Any]:
    """
    Evaluate the answer for the Asian all-solid-state battery manufacturer task.
    Builds a sequential root tree: first identify company details, then verify technology constraints.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extraction
    ext = await evaluator.extract(
        prompt=prompt_extract_company_and_announcement(),
        template_class=SolidStateEvaluationExtraction,
        extraction_name="solid_state_company_announcement"
    )

    # Build verification tree according to rubric
    await build_company_identification(evaluator, root, ext)
    await build_technology_constraints(evaluator, root, ext)

    # Return summary
    return evaluator.get_summary()