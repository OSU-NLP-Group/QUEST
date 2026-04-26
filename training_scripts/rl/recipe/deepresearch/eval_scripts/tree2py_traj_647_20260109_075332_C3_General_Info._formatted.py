import asyncio
import logging
from datetime import datetime
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tallest_building_research_chain"
TASK_DESCRIPTION = (
    "Starting with the world's tallest completed building, identify its lead architect. Then, find the "
    "architecture firm that this architect founded after working on that building, and identify one of the founding "
    "partners of this new firm (other than the lead architect). For this founding partner, determine which university "
    "they attended for their undergraduate architecture degree, and provide the year that university was founded. "
    "For each step in this research chain, provide relevant reference URLs that document your findings."
)
CURRENT_DATE = datetime.utcnow().date().isoformat()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Step1Tallest(BaseModel):
    building_name: Optional[str] = None
    building_height: Optional[str] = None
    tallest_status_urls: List[str] = Field(default_factory=list)
    height_urls: List[str] = Field(default_factory=list)


class Step2Architect(BaseModel):
    lead_architect_name: Optional[str] = None
    official_role_urls: List[str] = Field(default_factory=list)


class Step3Firm(BaseModel):
    firm_name: Optional[str] = None
    firm_founding_date: Optional[str] = None
    architect_founder_urls: List[str] = Field(default_factory=list)
    founded_after_building_urls: List[str] = Field(default_factory=list)


class Step4PartnerEducation(BaseModel):
    partner_name: Optional[str] = None
    partner_founder_urls: List[str] = Field(default_factory=list)
    undergrad_degree_is_architecture: Optional[str] = None
    undergrad_university_name: Optional[str] = None
    education_urls: List[str] = Field(default_factory=list)


class Step5University(BaseModel):
    university_founding_year: Optional[str] = None
    founding_year_urls: List[str] = Field(default_factory=list)


class ResearchChainExtraction(BaseModel):
    step1: Optional[Step1Tallest] = None
    step2: Optional[Step2Architect] = None
    step3: Optional[Step3Firm] = None
    step4: Optional[Step4PartnerEducation] = None
    step5: Optional[Step5University] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_research_chain() -> str:
    return f"""
Extract the research chain information exactly as presented in the answer. Return a single JSON object matching these fields:

step1:
  building_name: The name of the world's tallest completed building mentioned in the answer.
  building_height: The documented height string exactly as written (include units).
  tallest_status_urls: A list of URL(s) that explicitly support the claim that this building is the world's tallest completed building (as of the current date).
  height_urls: A list of URL(s) that document/verify the building height value provided.

step2:
  lead_architect_name: The full name of the lead/design architect (primary designer) for the building.
  official_role_urls: A list of URL(s) from official sources (e.g., building owner/developer, credited design firm's official project page, official publication by a directly responsible organization) that credit this person as the lead/design architect or primary designer.

step3:
  firm_name: The name of an architecture firm founded by the lead architect (after working on the building).
  firm_founding_date: The firm's founding date or year as provided in the answer (string, keep original format, e.g., "2018" or "October 2018").
  architect_founder_urls: A list of URL(s) that document the lead architect as a founder of the firm.
  founded_after_building_urls: A list of URL(s) that support the temporal ordering that the firm was founded after the architect worked on the building (ideally a page that states this directly).

step4:
  partner_name: The full name of a founding partner of the firm (other than the lead architect).
  partner_founder_urls: A list of URL(s) that document this person as a founding partner/founder of the firm.
  undergrad_degree_is_architecture: The statement describing that the partner's undergraduate degree is in architecture (keep as a short string, e.g., "B.Arch." or "Bachelor of Architecture" or "BA in Architecture"); if unspecified, return null.
  undergrad_university_name: The name of the university where the partner earned their undergraduate architecture degree.
  education_urls: A list of URL(s) that document the partner's undergraduate architecture degree and the university attended.

step5:
  university_founding_year: The year (string) that the undergrad university was founded (as provided in the answer).
  founding_year_urls: A list of URL(s) that document this university founding year.

Rules:
- Only extract information explicitly present in the answer.
- For all URL list fields, include only actual URLs that appear in the answer (plain or markdown link). If a field lacks URLs, return an empty list (not null).
- If a non-URL field is missing, return null for that field.
- Do not invent or infer details not present in the answer.
- Keep all strings exactly as presented in the answer (do not normalize units or numbers).

Current date for context: {CURRENT_DATE}.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s) and str(s).strip() != ""


def _clean_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


def _merge_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_step_1(
    evaluator: Evaluator,
    parent,
    data: ResearchChainExtraction,
) -> None:
    step = data.step1 or Step1Tallest()
    step_node = evaluator.add_parallel(
        id="Step_1_Tallest_Completed_Building",
        desc="Identify the world's tallest completed building (as of current date) and its documented height.",
        parent=parent,
        critical=True,
    )

    building_name = step.building_name or ""
    building_height = step.building_height or ""
    status_urls = _clean_urls(step.tallest_status_urls)
    height_urls = _clean_urls(step.height_urls)

    # URL provided checks (critical preconditions to avoid fallback to simple verify)
    status_url_provided = evaluator.add_custom_node(
        result=len(status_urls) > 0,
        id="Tallest_Completed_Status_URL_Provided",
        desc="At least one status reference URL is provided to support 'world's tallest completed building'.",
        parent=step_node,
        critical=True,
    )
    height_url_provided = evaluator.add_custom_node(
        result=len(height_urls) > 0,
        id="Height_URL_Provided",
        desc="At least one height reference URL is provided.",
        parent=step_node,
        critical=True,
    )

    # Building_Name
    name_node = evaluator.add_leaf(
        id="Building_Name",
        desc="Provide the name of the world's tallest completed building (as of current date).",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page indicates that the world's tallest completed building is named '{building_name}'.",
        node=name_node,
        sources=status_urls,
        additional_instruction=(
            "Verify that the page explicitly or unambiguously identifies the building by this name. "
            "Minor formatting variations (e.g., presence/absence of diacritics or article words) are acceptable."
        ),
        extra_prerequisites=[status_url_provided],
    )

    # Building_Height
    height_node = evaluator.add_leaf(
        id="Building_Height",
        desc="Provide the building's documented height (with units as stated in the source).",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The building's height is '{building_height}'.",
        node=height_node,
        sources=height_urls,
        additional_instruction=(
            "Confirm that the page lists this height value. Allow minor formatting variations, "
            "but the numeric value and units should match the provided string in essence."
        ),
        extra_prerequisites=[height_url_provided],
    )

    # Tallest_Completed_Status_Reference
    status_ref_node = evaluator.add_leaf(
        id="Tallest_Completed_Status_Reference",
        desc="Provide a reference URL that explicitly supports that this building is the world's tallest completed building as of the current date.",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page explicitly states that '{building_name}' is the world's tallest completed building.",
        node=status_ref_node,
        sources=status_urls,
        additional_instruction=(
            "Favor statements that clearly indicate 'world's tallest' and that it is completed. "
            "If the page indicates it 'was once' the tallest or uses an outdated context that is no longer true, "
            "treat it as not supported."
        ),
        extra_prerequisites=[status_url_provided],
    )

    # Height_Reference
    height_ref_node = evaluator.add_leaf(
        id="Height_Reference",
        desc="Provide a reference URL that documents/verifies the building height value given.",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page documents the building height as '{building_height}'.",
        node=height_ref_node,
        sources=height_urls,
        additional_instruction=(
            "The page should clearly include the given height information. Minor variations in punctuation or unit formatting are acceptable."
        ),
        extra_prerequisites=[height_url_provided],
    )


async def build_step_2(
    evaluator: Evaluator,
    parent,
    data: ResearchChainExtraction,
) -> None:
    step1 = data.step1 or Step1Tallest()
    step2 = data.step2 or Step2Architect()
    step_node = evaluator.add_parallel(
        id="Step_2_Lead_Architect",
        desc="Identify the lead/design architect (primary designer) of the tallest completed building.",
        parent=parent,
        critical=True,
    )

    building_name = step1.building_name or "the building"
    lead_architect = step2.lead_architect_name or ""
    official_urls = _clean_urls(step2.official_role_urls)

    official_url_provided = evaluator.add_custom_node(
        result=len(official_urls) > 0,
        id="Lead_Architect_Official_Role_URL_Provided",
        desc="At least one official-role reference URL is provided.",
        parent=step_node,
        critical=True,
    )

    # Lead_Architect_Name
    lead_name_node = evaluator.add_leaf(
        id="Lead_Architect_Name",
        desc="Provide the full name of the lead/design architect.",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page credits '{lead_architect}' as the lead/design architect (primary designer) of {building_name}.",
        node=lead_name_node,
        sources=official_urls,
        additional_instruction=(
            "The page should clearly attribute the role of lead or design architect (primary designer) to this person."
        ),
        extra_prerequisites=[official_url_provided],
    )

    # Lead_Architect_Official_Role_Reference
    official_ref_node = evaluator.add_leaf(
        id="Lead_Architect_Official_Role_Reference",
        desc="Provide a reference URL from an official source ... documenting this person as the lead/design architect.",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"This page is an official source (e.g., the building's official site/owner/developer or the credited design firm's "
            f"official project page, or equivalent official publication) and it credits '{lead_architect}' as the lead/design architect "
            f"for {building_name}."
        ),
        node=official_ref_node,
        sources=official_urls,
        additional_instruction=(
            "Be strict about 'official source'. Accept only pages owned by the building owner/developer, the credited design firm, "
            "or a directly responsible organization for the project. General media or third-party wikis are not official."
        ),
        extra_prerequisites=[official_url_provided],
    )


async def build_step_3(
    evaluator: Evaluator,
    parent,
    data: ResearchChainExtraction,
) -> None:
    step1 = data.step1 or Step1Tallest()
    step2 = data.step2 or Step2Architect()
    step3 = data.step3 or Step3Firm()
    step_node = evaluator.add_parallel(
        id="Step_3_Firm_Founded_By_Architect_After_Building_Work",
        desc="Identify an architecture firm founded by the lead architect after working on the tallest building, including founding date and documentation.",
        parent=parent,
        critical=True,
    )

    building_name = step1.building_name or "the building"
    lead_architect = step2.lead_architect_name or "the lead architect"
    firm_name = step3.firm_name or ""
    founding_date = step3.firm_founding_date or ""
    founder_urls = _clean_urls(step3.architect_founder_urls)
    after_urls = _clean_urls(step3.founded_after_building_urls)
    official_role_urls = _clean_urls(step2.official_role_urls)

    founder_url_provided = evaluator.add_custom_node(
        result=len(founder_urls) > 0,
        id="Architect_Founder_URL_Provided",
        desc="At least one URL is provided that documents the architect as a founder of the firm.",
        parent=step_node,
        critical=True,
    )
    after_url_provided = evaluator.add_custom_node(
        result=len(after_urls) > 0,
        id="Founded_After_Work_URL_Provided",
        desc="At least one URL is provided supporting that the firm was founded after work on the building.",
        parent=step_node,
        critical=True,
    )

    # Firm_Name
    firm_name_node = evaluator.add_leaf(
        id="Firm_Name",
        desc="Provide the name of the architecture firm founded by the lead architect.",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The architecture firm founded by {lead_architect} is named '{firm_name}'.",
        node=firm_name_node,
        sources=founder_urls,
        additional_instruction="The page should clearly associate the firm's name with being founded by the named architect.",
        extra_prerequisites=[founder_url_provided],
    )

    # Firm_Founding_Date
    firm_founding_date_node = evaluator.add_leaf(
        id="Firm_Founding_Date",
        desc="Provide the founding date (or founding year) of the firm.",
        parent=step_node,
        critical=True,
    )
    combined_for_date = _merge_urls(after_urls, founder_urls)
    await evaluator.verify(
        claim=f"The firm '{firm_name}' was founded on/in '{founding_date}'.",
        node=firm_founding_date_node,
        sources=combined_for_date,
        additional_instruction=(
            "Verify that the page(s) provide the firm's founding date or founding year matching the given string. "
            "Minor formatting differences are acceptable."
        ),
        extra_prerequisites=[after_url_provided],
    )

    # Architect_Founder_Reference
    founder_ref_node = evaluator.add_leaf(
        id="Architect_Founder_Reference",
        desc="Provide a reference URL documenting that the lead architect is a founder of the firm.",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page documents that {lead_architect} founded or co-founded the firm '{firm_name}'.",
        node=founder_ref_node,
        sources=founder_urls,
        additional_instruction="The page must explicitly indicate the architect's founder or co-founder status.",
        extra_prerequisites=[founder_url_provided],
    )

    # Founded_After_Work_On_Building_Reference
    after_ref_node = evaluator.add_leaf(
        id="Founded_After_Work_On_Building_Reference",
        desc="Provide a reference URL that supports the claim that the firm was founded after the architect worked on the tallest building project.",
        parent=step_node,
        critical=True,
    )
    # Try with the 'after_urls' first; if one page explicitly states the sequence, that's sufficient
    urls_for_after = _merge_urls(after_urls, official_role_urls)
    await evaluator.verify(
        claim=f"This page shows that the firm '{firm_name}' was founded after {lead_architect}'s work on {building_name}.",
        node=after_ref_node,
        sources=urls_for_after,
        additional_instruction=(
            "Prefer a page that explicitly states the temporal order (e.g., 'after completing X, [architect] founded Y'). "
            "If the page establishes this directly, it's supported."
        ),
        extra_prerequisites=[after_url_provided],
    )


async def build_step_4(
    evaluator: Evaluator,
    parent,
    data: ResearchChainExtraction,
) -> None:
    step2 = data.step2 or Step2Architect()
    step3 = data.step3 or Step3Firm()
    step4 = data.step4 or Step4PartnerEducation()

    step_node = evaluator.add_parallel(
        id="Step_4_Other_Founding_Partner_And_Undergrad_Education",
        desc="Identify one founding partner of the firm other than the lead architect, and determine their undergraduate architecture-degree university.",
        parent=parent,
        critical=True,
    )

    lead_architect = step2.lead_architect_name or ""
    firm_name = step3.firm_name or "the firm"
    partner_name = step4.partner_name or ""
    partner_urls = _clean_urls(step4.partner_founder_urls)
    education_urls = _clean_urls(step4.education_urls)
    undergrad_arch = step4.undergrad_degree_is_architecture or ""
    undergrad_univ = step4.undergrad_university_name or ""

    partner_url_provided = evaluator.add_custom_node(
        result=len(partner_urls) > 0,
        id="Partner_Is_Founding_Partner_URL_Provided",
        desc="At least one URL is provided documenting partner as founding partner.",
        parent=step_node,
        critical=True,
    )
    education_url_provided = evaluator.add_custom_node(
        result=len(education_urls) > 0,
        id="Education_URL_Provided",
        desc="At least one education reference URL is provided.",
        parent=step_node,
        critical=True,
    )

    # Partner_Name
    partner_name_node = evaluator.add_leaf(
        id="Partner_Name",
        desc="Provide the full name of one founding partner of the firm.",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page identifies '{partner_name}' as associated with the firm '{firm_name}'.",
        node=partner_name_node,
        sources=partner_urls,
        additional_instruction="The page should clearly show the partner's name in the firm founder/founding-partner context.",
        extra_prerequisites=[partner_url_provided],
    )

    # Partner_Is_Founding_Partner_Reference
    partner_founder_ref_node = evaluator.add_leaf(
        id="Partner_Is_Founding_Partner_Reference",
        desc="Provide a reference URL documenting that this person is a founding partner/founder of the firm.",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page documents that '{partner_name}' is a founding partner/founder of '{firm_name}'.",
        node=partner_founder_ref_node,
        sources=partner_urls,
        additional_instruction="The page must explicitly show founding partner/founder status.",
        extra_prerequisites=[partner_url_provided],
    )

    # Partner_Is_Not_Lead_Architect
    not_same_node = evaluator.add_leaf(
        id="Partner_Is_Not_Lead_Architect",
        desc="Ensure the selected founding partner is not the same person as the lead architect identified in Step 2.",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{partner_name}' and '{lead_architect}' are different individuals (not the same person).",
        node=not_same_node,
        additional_instruction=(
            "Return Correct only if the two names refer to different people. "
            "Minor variations like middle initials should be considered the same person; "
            "different given/family names indicate different people."
        ),
    )

    # Undergrad_Degree_Is_Architecture
    degree_arch_node = evaluator.add_leaf(
        id="Undergrad_Degree_Is_Architecture",
        desc="State the partner's undergraduate degree is in architecture (or explicitly architecture-designated undergraduate credential).",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"This page shows that '{partner_name}' holds an undergraduate degree in architecture "
            f"(e.g., 'Bachelor of Architecture', 'B.Arch.', or clearly architecture-designated), described as '{undergrad_arch}'."
        ),
        node=degree_arch_node,
        sources=education_urls,
        additional_instruction=(
            "Accept common variants like 'Bachelor of Architecture', 'B.Arch.', 'BA in Architecture', or localized equivalents. "
            "Degree should clearly be an undergraduate credential in architecture."
        ),
        extra_prerequisites=[education_url_provided],
    )

    # Undergrad_University_Name
    undergrad_univ_node = evaluator.add_leaf(
        id="Undergrad_University_Name",
        desc="Provide the name of the university where the partner earned their undergraduate architecture degree.",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page shows that '{partner_name}' completed an undergraduate architecture degree at '{undergrad_univ}'.",
        node=undergrad_univ_node,
        sources=education_urls,
        additional_instruction="The page should clearly associate the undergraduate architecture degree with the specified university.",
        extra_prerequisites=[education_url_provided],
    )

    # Education_Reference
    edu_ref_node = evaluator.add_leaf(
        id="Education_Reference",
        desc="Provide a reference URL documenting the partner's undergraduate architecture degree and the university attended.",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"This page documents both: (1) '{partner_name}' has an undergraduate architecture degree, and "
            f"(2) it was earned at '{undergrad_univ}'."
        ),
        node=edu_ref_node,
        sources=education_urls,
        additional_instruction="Reject if the page does not cover both the degree and the university.",
        extra_prerequisites=[education_url_provided],
    )


async def build_step_5(
    evaluator: Evaluator,
    parent,
    data: ResearchChainExtraction,
) -> None:
    step4 = data.step4 or Step4PartnerEducation()
    step5 = data.step5 or Step5University()

    step_node = evaluator.add_parallel(
        id="Step_5_University_Founding_Year",
        desc="Provide the founding year of the partner's undergraduate university, with documentation.",
        parent=parent,
        critical=True,
    )

    undergrad_univ = step4.undergrad_university_name or "the university"
    founding_year = step5.university_founding_year or ""
    year_urls = _clean_urls(step5.founding_year_urls)

    year_url_provided = evaluator.add_custom_node(
        result=len(year_urls) > 0,
        id="University_Founding_Year_URL_Provided",
        desc="At least one university founding-year reference URL is provided.",
        parent=step_node,
        critical=True,
    )

    # University_Founding_Year
    year_node = evaluator.add_leaf(
        id="University_Founding_Year",
        desc="Provide the year the university was founded.",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page indicates that '{undergrad_univ}' was founded in '{founding_year}'.",
        node=year_node,
        sources=year_urls,
        additional_instruction=(
            "Accept if the page clearly provides the founding year that matches the provided value. "
            "Minor formatting variations acceptable (e.g., mentioning exact date that contains the year)."
        ),
        extra_prerequisites=[year_url_provided],
    )

    # University_Founding_Year_Reference
    year_ref_node = evaluator.add_leaf(
        id="University_Founding_Year_Reference",
        desc="Provide a reference URL documenting the university's founding year.",
        parent=step_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page provides the founding year for '{undergrad_univ}'.",
        node=year_ref_node,
        sources=year_urls,
        additional_instruction="The page should unambiguously provide the founding year information.",
        extra_prerequisites=[year_url_provided],
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the research chain task using the Mind2Web2 evaluation framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Entire chain is sequential
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
        prompt=prompt_extract_research_chain(),
        template_class=ResearchChainExtraction,
        extraction_name="research_chain_extraction",
    )

    # Build verification steps (sequential chain)
    await build_step_1(evaluator, root, extraction)
    await build_step_2(evaluator, root, extraction)
    await build_step_3(evaluator, root, extraction)
    await build_step_4(evaluator, root, extraction)
    await build_step_5(evaluator, root, extraction)

    # Return summary
    return evaluator.get_summary()