import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "tx_superintendent_education_constraints"
TASK_DESCRIPTION = (
    "Identify the full name of the current superintendent of a public school district in Texas who satisfies all of the following educational background requirements:\n\n"
    "1. Holds a doctoral degree from a university that was founded in 1927\n"
    "2. Holds a master's degree from an institution that is both designated as a Historically Black College or University (HBCU) and designated as an 1890 land-grant institution under the Second Morrill Act of 1890\n"
    "3. Holds a bachelor's degree from an HBCU that was founded in the same year (1927) as the university where they obtained their doctoral degree\n"
    "4. All three degrees (bachelor's, master's, and doctorate) were obtained from three different institutions"
)


class DegreeInfo(BaseModel):
    degree_type: Optional[str] = None
    institution: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SuperintendentExtraction(BaseModel):
    full_name: Optional[str] = None
    district_name: Optional[str] = None
    position_sources: List[str] = Field(default_factory=list)

    doctoral: Optional[DegreeInfo] = None
    masters: Optional[DegreeInfo] = None
    bachelors: Optional[DegreeInfo] = None


def prompt_extract_superintendent() -> str:
    return (
        "From the provided answer, extract the following structured information about the identified individual and their education. "
        "Return JSON with exactly these fields.\n\n"
        "Fields:\n"
        "- full_name: The person's full name as claimed in the answer.\n"
        "- district_name: The name of the Texas public school district the person currently serves as superintendent.\n"
        "- position_sources: All URLs cited that support or are associated with the claim about current superintendent role in the Texas district. "
        "Include district official pages, press releases, biographies, or credible news articles. If none are provided, return an empty list.\n"
        "- doctoral: An object with fields:\n"
        "  - degree_type: The doctoral degree type as stated (e.g., Ph.D., Ed.D.). If unspecified, use 'doctorate' if the answer clearly states a doctoral degree.\n"
        "  - institution: The name of the university where the doctoral degree was obtained.\n"
        "  - sources: All URLs cited related to the doctoral degree and/or the institution, including pages that may state founding year or the person's degree. If none, return an empty list.\n"
        "- masters: An object with fields:\n"
        "  - degree_type: The master's degree type (e.g., M.Ed., M.S.). If unspecified but clearly a master's degree, use 'master's'.\n"
        "  - institution: The name of the institution where the master's degree was obtained.\n"
        "  - sources: All URLs cited related to the master's degree and/or the institution, including pages that may state HBCU designation or 1890 land‑grant status. If none, return an empty list.\n"
        "- bachelors: An object with fields:\n"
        "  - degree_type: The bachelor's degree type (e.g., B.S., B.A.). If unspecified but clearly a bachelor's degree, use 'bachelor's'.\n"
        "  - institution: The name of the institution where the bachelor's degree was obtained.\n"
        "  - sources: All URLs cited related to the bachelor's degree and/or the institution, including pages that may state founding year or HBCU status. If none, return an empty list.\n\n"
        "Important:\n"
        "• Extract only what is explicitly stated in the answer. Do not invent institutions, degree types, or URLs.\n"
        "• URLs can appear as raw links or markdown links; extract the actual URLs.\n"
        "• If a field is missing in the answer, set it to null (for strings) or empty list (for sources).\n"
    )


async def verify_current_position(evaluator: Evaluator, parent_node, data: SuperintendentExtraction) -> None:
    node = evaluator.add_parallel(
        id="Current_Position",
        desc="The individual currently serves as superintendent of a public school district in Texas",
        parent=parent_node,
        critical=True
    )

    exists = bool(data.full_name and data.district_name and data.position_sources and len(data.position_sources) > 0)
    evaluator.add_custom_node(
        result=exists,
        id="Current_Position_Data_Provided",
        desc="Name, district, and at least one position source URL are provided in the answer",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Current_Position_Verified",
        desc="Verify the individual is currently superintendent of a Texas public school district",
        parent=node,
        critical=True
    )

    full_name = data.full_name or ""
    district = data.district_name or ""
    claim = (
        f"{full_name} is currently the superintendent of {district}, and {district} is a public school district located in Texas."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.position_sources,
        additional_instruction=(
            "Use the provided URLs to confirm both: (1) the person is the current superintendent of the named district "
            "and (2) that district is a Texas public school district (e.g., ISD/CISD in TX). "
            "Prefer official district pages, current leadership listings, or recent credible news."
        ),
    )


async def verify_doctoral_degree(evaluator: Evaluator, parent_node, data: SuperintendentExtraction) -> None:
    node = evaluator.add_parallel(
        id="Doctoral_Degree",
        desc="The individual holds a doctoral degree from a university founded in 1927",
        parent=parent_node,
        critical=True
    )
    doc = data.doctoral or DegreeInfo()

    exists = bool(doc.institution and doc.sources and len(doc.sources) > 0)
    evaluator.add_custom_node(
        result=exists,
        id="Doctoral_Info_Provided",
        desc="Doctoral institution and at least one source URL are provided",
        parent=node,
        critical=True
    )

    founded_leaf = evaluator.add_leaf(
        id="Doctorate_Institution_Founded_1927",
        desc="The doctoral degree-granting institution was founded in 1927",
        parent=node,
        critical=True
    )
    claim_found = f"The institution {doc.institution or ''} was founded in 1927."
    await evaluator.verify(
        claim=claim_found,
        node=founded_leaf,
        sources=doc.sources,
        additional_instruction="Confirm the institution's founding year is 1927 using the cited sources (e.g., official history page or Wikipedia).",
    )

    degree_leaf = evaluator.add_leaf(
        id="Doctorate_Degree_Obtained",
        desc="The individual obtained a doctoral degree from this institution",
        parent=node,
        critical=True
    )
    full_name = data.full_name or ""
    degree_claim = f"{full_name} obtained a doctoral degree from {doc.institution or ''}."
    await evaluator.verify(
        claim=degree_claim,
        node=degree_leaf,
        sources=doc.sources,
        additional_instruction="Verify that the person earned a doctoral degree (e.g., Ph.D., Ed.D.) from the specified institution.",
    )


async def verify_masters_degree(evaluator: Evaluator, parent_node, data: SuperintendentExtraction) -> None:
    node = evaluator.add_parallel(
        id="Masters_Degree",
        desc="The individual holds a master's degree from an institution that is both an HBCU and an 1890 land-grant institution",
        parent=parent_node,
        critical=True
    )
    mas = data.masters or DegreeInfo()

    exists = bool(mas.institution and mas.sources and len(mas.sources) > 0)
    evaluator.add_custom_node(
        result=exists,
        id="Masters_Info_Provided",
        desc="Master's institution and at least one source URL are provided",
        parent=node,
        critical=True
    )

    hbcu_leaf = evaluator.add_leaf(
        id="Masters_Institution_HBCU_Status",
        desc="The master's institution is designated as an HBCU",
        parent=node,
        critical=True
    )
    hbcu_claim = f"{mas.institution or ''} is designated as a Historically Black College or University (HBCU)."
    await evaluator.verify(
        claim=hbcu_claim,
        node=hbcu_leaf,
        sources=mas.sources,
        additional_instruction="Confirm HBCU designation via authoritative sources (e.g., U.S. Dept. of Education list, Wikipedia, or the institution stating HBCU status).",
    )

    land_leaf = evaluator.add_leaf(
        id="Masters_Institution_Land_Grant_Status",
        desc="The master's institution is designated as an 1890 land-grant institution",
        parent=node,
        critical=True
    )
    land_claim = f"{mas.institution or ''} is designated as an 1890 land-grant institution under the Second Morrill Act of 1890."
    await evaluator.verify(
        claim=land_claim,
        node=land_leaf,
        sources=mas.sources,
        additional_instruction="Confirm 1890 land-grant designation using authoritative sources (e.g., USDA/NIFA 1890 list or credible references).",
    )

    degree_leaf = evaluator.add_leaf(
        id="Masters_Degree_Obtained",
        desc="The individual obtained a master's degree from this institution",
        parent=node,
        critical=True
    )
    full_name = data.full_name or ""
    m_claim = f"{full_name} obtained a master's degree from {mas.institution or ''}."
    await evaluator.verify(
        claim=m_claim,
        node=degree_leaf,
        sources=mas.sources,
        additional_instruction="Verify the person earned a master's degree from the specified institution.",
    )


async def verify_bachelors_degree(evaluator: Evaluator, parent_node, data: SuperintendentExtraction) -> None:
    node = evaluator.add_parallel(
        id="Bachelors_Degree",
        desc="The individual holds a bachelor's degree from an HBCU founded in 1927",
        parent=parent_node,
        critical=True
    )
    bac = data.bachelors or DegreeInfo()

    exists = bool(bac.institution and bac.sources and len(bac.sources) > 0)
    evaluator.add_custom_node(
        result=exists,
        id="Bachelors_Info_Provided",
        desc="Bachelor's institution and at least one source URL are provided",
        parent=node,
        critical=True
    )

    hbcu_leaf = evaluator.add_leaf(
        id="Bachelors_Institution_HBCU_Status",
        desc="The bachelor's institution is designated as an HBCU",
        parent=node,
        critical=True
    )
    hbcu_claim = f"{bac.institution or ''} is designated as a Historically Black College or University (HBCU)."
    await evaluator.verify(
        claim=hbcu_claim,
        node=hbcu_leaf,
        sources=bac.sources,
        additional_instruction="Confirm HBCU designation via authoritative sources (e.g., U.S. Dept. of Education list, Wikipedia, or the institution stating HBCU status).",
    )

    founded_leaf = evaluator.add_leaf(
        id="Bachelors_Institution_Founded_1927",
        desc="The bachelor's institution was founded in 1927",
        parent=node,
        critical=True
    )
    founded_claim = f"The institution {bac.institution or ''} was founded in 1927."
    await evaluator.verify(
        claim=founded_claim,
        node=founded_leaf,
        sources=bac.sources,
        additional_instruction="Confirm the institution's founding year is 1927 using the cited sources (e.g., official history page or Wikipedia).",
    )

    degree_leaf = evaluator.add_leaf(
        id="Bachelors_Degree_Obtained",
        desc="The individual obtained a bachelor's degree from this institution",
        parent=node,
        critical=True
    )
    full_name = data.full_name or ""
    b_claim = f"{full_name} obtained a bachelor's degree from {bac.institution or ''}."
    await evaluator.verify(
        claim=b_claim,
        node=degree_leaf,
        sources=bac.sources,
        additional_instruction="Verify the person earned a bachelor's degree from the specified institution.",
    )


async def verify_institutional_diversity(evaluator: Evaluator, parent_node, data: SuperintendentExtraction) -> None:
    doc_inst = (data.doctoral.institution if data.doctoral else None) or ""
    mas_inst = (data.masters.institution if data.masters else None) or ""
    bac_inst = (data.bachelors.institution if data.bachelors else None) or ""

    all_present = bool(doc_inst and mas_inst and bac_inst)
    all_distinct = all_present and (len({doc_inst.strip(), mas_inst.strip(), bac_inst.strip()}) == 3)

    evaluator.add_custom_node(
        result=all_distinct,
        id="Institutional_Diversity",
        desc="All three degrees were obtained from three different institutions",
        parent=parent_node,
        critical=True
    )


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
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    extracted = await evaluator.extract(
        prompt=prompt_extract_superintendent(),
        template_class=SuperintendentExtraction,
        extraction_name="extracted_superintendent_info",
    )

    evaluator.add_custom_info(
        info={
            "requirements": {
                "doctorate_institution_founded_year": 1927,
                "masters_institution_hbcu": True,
                "masters_institution_1890_land_grant": True,
                "bachelors_institution_hbcu": True,
                "bachelors_institution_founded_year": 1927,
                "institutions_all_distinct": True,
                "current_superintendent_in_texas": True,
            }
        },
        info_type="constraints",
        info_name="task_constraints"
    )

    main = evaluator.add_sequential(
        id="Root",
        desc="Identify the superintendent who satisfies all specified educational background constraints",
        parent=root,
        critical=True
    )

    await verify_current_position(evaluator, main, extracted)

    edu = evaluator.add_sequential(
        id="Educational_Background_Verification",
        desc="Verify all educational credentials meet the specified requirements",
        parent=main,
        critical=True
    )

    await verify_doctoral_degree(evaluator, edu, extracted)
    await verify_masters_degree(evaluator, edu, extracted)
    await verify_bachelors_degree(evaluator, edu, extracted)
    await verify_institutional_diversity(evaluator, edu, extracted)

    return evaluator.get_summary()