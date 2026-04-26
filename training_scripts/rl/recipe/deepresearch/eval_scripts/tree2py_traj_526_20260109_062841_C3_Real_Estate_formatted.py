import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "leed_sf_ceo_education"
TASK_DESCRIPTION = """
Identify a LEED Platinum certified office building located in San Francisco, California. Once identified, determine the primary real estate development company that developed this building. Then, identify the current Chief Executive Officer (CEO) of that development company. Finally, provide detailed information about the CEO's educational background, including: (1) the name of the university where the CEO obtained their undergraduate degree, the specific degree earned, and the field of study; and (2) the name of the business school where the CEO obtained their MBA degree and the years of attendance.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BuildingInfo(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None  # e.g., "San Francisco, California"
    leed_certification: Optional[str] = None  # e.g., "LEED Platinum"
    building_type: Optional[str] = None  # e.g., "office", "commercial", "mixed-use (office)"
    sources: List[str] = Field(default_factory=list)


class CompanyInfo(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None  # e.g., "developer", "owner", "primary developer", "primary owner"
    sources: List[str] = Field(default_factory=list)


class CEOInfo(BaseModel):
    name: Optional[str] = None
    position_title: Optional[str] = None  # e.g., "Chief Executive Officer", "CEO"
    sources: List[str] = Field(default_factory=list)


class UndergradEducation(BaseModel):
    institution: Optional[str] = None
    degree_type: Optional[str] = None  # e.g., "B.S.", "B.A.", "Bachelor of Science"
    field_of_study: Optional[str] = None  # e.g., "Economics"
    sources: List[str] = Field(default_factory=list)


class MBAEducation(BaseModel):
    business_school: Optional[str] = None  # e.g., "Harvard Business School"
    years_attended: Optional[str] = None  # e.g., "2003–2005" or "Class of 2005"
    sources: List[str] = Field(default_factory=list)


class ResearchExtraction(BaseModel):
    building: Optional[BuildingInfo] = None
    company: Optional[CompanyInfo] = None
    ceo: Optional[CEOInfo] = None
    undergraduate: Optional[UndergradEducation] = None
    mba: Optional[MBAEducation] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_research() -> str:
    return """
    Extract structured information from the answer for the following items. Extract only what is explicitly in the answer. If an item is not present, set it to null; if sources are not provided, return an empty list. Ensure URLs are valid and taken directly from the answer (plain URLs or markdown links).

    1) building:
       - name: The building's name.
       - location: The city/state string for the building's location (e.g., "San Francisco, California").
       - leed_certification: The building's LEED certification level as stated (e.g., "LEED Platinum").
       - building_type: The building type (e.g., "office", "commercial", "office tower", "mixed-use (office)").
       - sources: All URLs cited in the answer that support the building's details (certification, location, type).

    2) company:
       - name: The primary real estate development company that developed or currently owns the identified building.
       - role: A short label like "developer" or "owner" as described in the answer.
       - sources: All URLs cited that support the company’s relationship to the building.

    3) ceo:
       - name: The full name of the current Chief Executive Officer of the identified company.
       - position_title: The position title as explicitly stated (e.g., "Chief Executive Officer", "CEO").
       - sources: All URLs cited that support the CEO identity or current status.

    4) undergraduate:
       - institution: University name for the CEO's undergraduate degree.
       - degree_type: Specific degree type (e.g., "B.S.", "B.A.", "Bachelor of Science").
       - field_of_study: Major or field of study (e.g., "Economics", "Engineering").
       - sources: URLs cited that support the undergraduate information.

    5) mba:
       - business_school: Name of the business school where the CEO obtained the MBA.
       - years_attended: Years of attendance, as presented (e.g., "2003–2005", "Class of 2005").
       - sources: URLs cited that support the MBA information.

    Notes:
    - Return a single JSON with keys: building, company, ceo, undergraduate, mba.
    - Do not invent information; only extract what is clearly present in the answer.
    - For URL fields, include only valid URLs explicitly present in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def merge_sources(*lists: Optional[List[str]]) -> List[str]:
    """Merge multiple URL lists and deduplicate while preserving order."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if url and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_building(evaluator: Evaluator, parent_node, data: ResearchExtraction) -> None:
    b = data.building or BuildingInfo()
    building_node = evaluator.add_parallel(
        id="LEED_Platinum_Building_San_Francisco",
        desc="Identify a LEED Platinum certified office/commercial building in San Francisco, California.",
        parent=parent_node,
        critical=True
    )

    # Existence & sources gate
    building_has_min = bool((b.name or "").strip()) and bool(b.sources)
    evaluator.add_custom_node(
        result=building_has_min,
        id="Building_Identified_With_Sources",
        desc="A specific building is identified and at least one supporting source URL is provided.",
        parent=building_node,
        critical=True
    )

    # LEED Platinum verification
    leaf_leed = evaluator.add_leaf(
        id="Building_LEED_Certification_Level",
        desc="The building must have LEED Platinum certification (not Gold, Silver, or basic Certified).",
        parent=building_node,
        critical=True
    )
    leed_claim = f"The building '{b.name or ''}' has achieved LEED Platinum certification."
    await evaluator.verify(
        claim=leed_claim,
        node=leaf_leed,
        sources=b.sources,
        additional_instruction="Confirm the page explicitly indicates LEED Platinum (any rating system variant, e.g., BD+C, EBOM, Core & Shell). Do not accept Gold, Silver, or 'Certified'."
    )

    # Location verification
    leaf_loc = evaluator.add_leaf(
        id="Building_Location",
        desc="The building must be located in San Francisco, California.",
        parent=building_node,
        critical=True
    )
    loc_claim = f"The building '{b.name or ''}' is located in San Francisco, California."
    await evaluator.verify(
        claim=loc_claim,
        node=leaf_loc,
        sources=b.sources,
        additional_instruction="Accept 'San Francisco, CA' or clear evidence the building's address is in San Francisco, California."
    )

    # Building type verification
    leaf_type = evaluator.add_leaf(
        id="Building_Type",
        desc="The building must be an office or commercial building.",
        parent=building_node,
        critical=True
    )
    type_claim = f"The building '{b.name or ''}' is an office or commercial building."
    await evaluator.verify(
        claim=type_claim,
        node=leaf_type,
        sources=b.sources,
        additional_instruction="Accept 'office', 'office tower', 'commercial', or mixed-use buildings that clearly include office/commercial primary use."
    )


async def verify_company(evaluator: Evaluator, parent_node, data: ResearchExtraction) -> None:
    b = data.building or BuildingInfo()
    c = data.company or CompanyInfo()

    company_node = evaluator.add_parallel(
        id="Developer_Owner_Company",
        desc="Identify the primary real estate development company that developed or currently owns the identified building.",
        parent=parent_node,
        critical=True
    )

    # Existence & sources gate
    comp_has_min = bool((c.name or "").strip()) and bool(merge_sources(c.sources, b.sources))
    evaluator.add_custom_node(
        result=comp_has_min,
        id="Company_Exists_With_Sources",
        desc="A company is identified and at least one supporting source URL is provided for its role with the building.",
        parent=company_node,
        critical=True
    )

    # Company identification leaf
    leaf_company = evaluator.add_leaf(
        id="Company_Identification",
        desc="The company must be the primary developer or current owner of the identified LEED Platinum building.",
        parent=company_node,
        critical=True
    )
    company_claim = f"The company '{c.name or ''}' is the primary developer or current owner of the building '{b.name or ''}'."
    await evaluator.verify(
        claim=company_claim,
        node=leaf_company,
        sources=merge_sources(c.sources, b.sources),
        additional_instruction="Verify that the sources explicitly indicate the company developed the building or currently owns it. Accept synonyms like 'developer', 'owner', 'co-developer' if clearly primary."
    )


async def verify_ceo(evaluator: Evaluator, parent_node, data: ResearchExtraction) -> None:
    c = data.company or CompanyInfo()
    ceo = data.ceo or CEOInfo()

    ceo_node = evaluator.add_parallel(
        id="Current_CEO_Identification",
        desc="Identify the current Chief Executive Officer (CEO) of the identified company.",
        parent=parent_node,
        critical=True
    )

    # Existence & sources gate
    ceo_has_min = bool((ceo.name or "").strip()) and bool(merge_sources(ceo.sources, c.sources))
    evaluator.add_custom_node(
        result=ceo_has_min,
        id="CEO_Exists_With_Sources",
        desc="A CEO name is provided and at least one supporting source URL confirms their role.",
        parent=ceo_node,
        critical=True
    )

    # CEO name verification
    leaf_ceo_name = evaluator.add_leaf(
        id="CEO_Name",
        desc="Provide the full name of the current CEO of the identified company.",
        parent=ceo_node,
        critical=True
    )
    ceo_name_claim = f"The current CEO of {c.name or ''} is {ceo.name or ''}."
    await evaluator.verify(
        claim=ceo_name_claim,
        node=leaf_ceo_name,
        sources=merge_sources(ceo.sources, c.sources),
        additional_instruction="Verify the page(s) explicitly state the person as CEO/Chief Executive Officer of the company."
    )

    # CEO current status verification
    leaf_ceo_status = evaluator.add_leaf(
        id="CEO_Current_Status",
        desc="Confirm that the identified person currently holds the CEO position at the time of answering.",
        parent=ceo_node,
        critical=True
    )
    ceo_status_claim = f"As of now, {ceo.name or ''} currently holds the CEO position at {c.name or ''}."
    await evaluator.verify(
        claim=ceo_status_claim,
        node=leaf_ceo_status,
        sources=merge_sources(ceo.sources, c.sources),
        additional_instruction="Confirm the sources reflect present status (e.g., 'CEO', 'Chief Executive Officer') rather than 'former'. Accept recent press/leadership pages indicating current role."
    )


async def verify_education(evaluator: Evaluator, parent_node, data: ResearchExtraction) -> None:
    ceo = data.ceo or CEOInfo()
    ug = data.undergraduate or UndergradEducation()
    mba = data.mba or MBAEducation()

    edu_node = evaluator.add_parallel(
        id="CEO_Educational_Background",
        desc="Provide detailed information about the CEO's educational credentials including undergraduate and MBA details.",
        parent=parent_node,
        critical=True
    )

    # Undergraduate subtree
    ug_node = evaluator.add_parallel(
        id="Undergraduate_Degree",
        desc="Details of the CEO's undergraduate education.",
        parent=edu_node,
        critical=True
    )
    ug_has_min = bool((ug.institution or "").strip()) and bool((ug.degree_type or "").strip()) and bool((ug.field_of_study or "").strip()) and bool(ug.sources)
    evaluator.add_custom_node(
        result=ug_has_min,
        id="Undergraduate_Info_Provided",
        desc="Undergraduate institution, degree type, field of study, and at least one source URL are provided.",
        parent=ug_node,
        critical=True
    )
    # Institution leaf
    leaf_ug_inst = evaluator.add_leaf(
        id="Undergraduate_Institution",
        desc="Name of the university where the CEO obtained their undergraduate degree.",
        parent=ug_node,
        critical=True
    )
    ug_inst_claim = f"The CEO {ceo.name or ''} obtained their undergraduate degree at {ug.institution or ''}."
    await evaluator.verify(
        claim=ug_inst_claim,
        node=leaf_ug_inst,
        sources=ug.sources,
        additional_instruction="Confirm the source explicitly mentions the undergraduate university for the CEO."
    )

    # Degree type leaf
    leaf_ug_deg = evaluator.add_leaf(
        id="Undergraduate_Degree_Type",
        desc="The specific undergraduate degree earned (e.g., B.S., B.A., etc.).",
        parent=ug_node,
        critical=True
    )
    ug_deg_claim = f"The CEO {ceo.name or ''} earned an undergraduate degree of type '{ug.degree_type or ''}'."
    await evaluator.verify(
        claim=ug_deg_claim,
        node=leaf_ug_deg,
        sources=ug.sources,
        additional_instruction="Accept reasonable variants/abbreviations (e.g., B.S., BSc, Bachelor of Science; B.A., BA, Bachelor of Arts)."
    )

    # Field of study leaf
    leaf_ug_field = evaluator.add_leaf(
        id="Undergraduate_Field_of_Study",
        desc="The major or field of study for the undergraduate degree.",
        parent=ug_node,
        critical=True
    )
    ug_field_claim = f"The CEO {ceo.name or ''}'s undergraduate field of study was '{ug.field_of_study or ''}'."
    await evaluator.verify(
        claim=ug_field_claim,
        node=leaf_ug_field,
        sources=ug.sources,
        additional_instruction="Confirm the source states the undergraduate major/field for the CEO."
    )

    # MBA subtree
    mba_node = evaluator.add_parallel(
        id="MBA_Degree",
        desc="Details of the CEO's MBA education.",
        parent=edu_node,
        critical=True
    )
    mba_has_min = bool((mba.business_school or "").strip()) and bool((mba.years_attended or "").strip()) and bool(mba.sources)
    evaluator.add_custom_node(
        result=mba_has_min,
        id="MBA_Info_Provided",
        desc="MBA business school, years attended, and at least one source URL are provided.",
        parent=mba_node,
        critical=True
    )

    # MBA institution leaf
    leaf_mba_inst = evaluator.add_leaf(
        id="MBA_Institution",
        desc="Name of the business school where the CEO obtained their MBA.",
        parent=mba_node,
        critical=True
    )
    mba_inst_claim = f"The CEO {ceo.name or ''} obtained an MBA at {mba.business_school or ''}."
    await evaluator.verify(
        claim=mba_inst_claim,
        node=leaf_mba_inst,
        sources=mba.sources,
        additional_instruction="Confirm the source explicitly names the business school awarding the MBA."
    )

    # MBA years attended leaf
    leaf_mba_years = evaluator.add_leaf(
        id="MBA_Years_Attended",
        desc="The years during which the CEO attended the MBA program.",
        parent=mba_node,
        critical=True
    )
    mba_years_claim = f"The CEO {ceo.name or ''} attended the MBA program at {mba.business_school or ''} during {mba.years_attended or ''}."
    await evaluator.verify(
        claim=mba_years_claim,
        node=leaf_mba_years,
        sources=mba.sources,
        additional_instruction="Accept ranges (e.g., 2003–2005), class years, or explicit attendance years as long as the source clearly supports them."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the LEED SF building, developer, CEO, and education task.
    """
    # Initialize evaluator
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
        default_model=model
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_research(),
        template_class=ResearchExtraction,
        extraction_name="research_extraction"
    )

    # Build the verification tree according to rubric
    top_seq = evaluator.add_sequential(
        id="Building_and_Leadership_Research",
        desc="Sequential task requiring identification of a LEED Platinum office/commercial building in San Francisco, its primary developer/owner company, the company's current CEO, and the CEO's educational credentials.",
        parent=root,
        critical=True
    )

    # 1) Building verification
    await verify_building(evaluator, top_seq, extracted)

    # 2) Company verification
    await verify_company(evaluator, top_seq, extracted)

    # 3) CEO verification
    await verify_ceo(evaluator, top_seq, extracted)

    # 4) Education verification
    await verify_education(evaluator, top_seq, extracted)

    # Return standardized summary
    return evaluator.get_summary()