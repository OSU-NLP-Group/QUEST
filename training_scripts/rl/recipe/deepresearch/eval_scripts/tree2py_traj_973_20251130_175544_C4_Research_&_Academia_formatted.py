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
TASK_ID = "nsf_ai_2023_institute"
TASK_DESCRIPTION = (
    "Identify an NSF AI Research Institute that satisfies all of the following criteria: "
    "(1) The institute was established in 2023 as part of the third cohort of NSF AI Research Institutes announced in May 2023. "
    "(2) The lead institution of the institute is located in a state on the East Coast of the United States (specifically, a state that borders the Atlantic Ocean). "
    "(3) The institute operates as a multi-institution partnership involving at least three distinct universities or research institutions as formal partners. "
    "(4) The institute receives funding from at least two distinct federal agencies or federal funding sources. "
    "(5) The institute's primary research mission includes trustworthiness, governance, ethics, or societal implications of AI as a central theme of its work. "
    "(6) The institute has administrative offices or headquarters located in more than one geographic location, indicating a distributed leadership structure. "
    "(7) The institute has a publicly accessible dedicated website that provides comprehensive information about its mission, partner institutions, and contact details. "
    "Provide the name of the institute, the lead institution, and a reference URL to the institute's official website."
)

EAST_COAST_STATES = [
    "Maine", "New Hampshire", "Massachusetts", "Rhode Island", "Connecticut",
    "New York", "New Jersey", "Delaware", "Maryland", "Virginia",
    "North Carolina", "South Carolina", "Georgia", "Florida"
]


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class InstituteCore(BaseModel):
    institute_name: Optional[str] = None
    lead_institution: Optional[str] = None
    website_url: Optional[str] = None


class URLList(BaseModel):
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_institute_core() -> str:
    return """
    From the answer, extract the following fields for the NSF AI Research Institute the answer identifies:
    - institute_name: The full official name of the NSF AI Research Institute.
    - lead_institution: The lead institution (university or research organization) for the institute.
    - website_url: The official dedicated website URL for the institute (not a press release or news article).
      If multiple URLs are present, choose the one that is the dedicated institute site (home page or about page).
    If any field is missing from the answer, set it to null.
    """


def prompt_extract_all_urls() -> str:
    return """
    Extract all URLs appearing in the answer, regardless of their purpose.
    Return them as a list under the field 'urls'.
    Include the official institute website if present, along with any press releases, partner pages, or other references.
    Remove duplicates if they appear. If no URLs are present, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper Utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for u in urls:
        if not u:
            continue
        v = u.strip()
        if not v:
            continue
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


# --------------------------------------------------------------------------- #
# Verification subtree builder                                                #
# --------------------------------------------------------------------------- #
async def verify_institute(
    evaluator: Evaluator,
    parent_node,
    core: InstituteCore,
    all_urls: URLList,
) -> None:
    # Create top-level non-critical parallel aggregator matching rubric "Institute_Identification"
    inst_node = evaluator.add_parallel(
        id="Institute_Identification",
        desc="The identified NSF AI Research Institute satisfies all specified criteria and the answer provides all required information",
        parent=parent_node,
        critical=False,
    )

    # Presence checks (critical)
    evaluator.add_custom_node(
        result=bool(core.institute_name and core.institute_name.strip()),
        id="Institute_Name_Provided",
        desc="The answer provides the name of the institute",
        parent=inst_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(core.lead_institution and core.lead_institution.strip()),
        id="Lead_Institution_Provided",
        desc="The answer provides the name of the lead institution",
        parent=inst_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(core.website_url and core.website_url.strip()),
        id="Reference_URL_Provided",
        desc="The answer provides a reference URL to the institute's official website",
        parent=inst_node,
        critical=True
    )

    # Prepare sources: official website plus any other URLs in the answer
    merged_sources: List[str] = _dedup_urls([core.website_url] + all_urls.urls)

    # 1) Establishment year and cohort (critical)
    est_leaf = evaluator.add_leaf(
        id="Establishment_Year",
        desc="The institute was established in 2023 as part of the third cohort of NSF AI Research Institutes",
        parent=inst_node,
        critical=True
    )
    est_claim = (
        f"The institute '{core.institute_name or 'the identified institute'}' "
        f"was established in 2023 as part of the third cohort of NSF AI Research Institutes announced in May 2023."
    )
    await evaluator.verify(
        claim=est_claim,
        node=est_leaf,
        sources=merged_sources,
        additional_instruction=(
            "Verify BOTH aspects: (a) the institute is part of the 'third cohort' of NSF AI Research Institutes, "
            "and (b) this cohort was announced in May 2023 (i.e., the institute is a 2023 cohort member). "
            "Accept phrasing like 'announced in May 2023' or 'launched in 2023'. If the source doesn't support both, mark as not supported."
        ),
    )

    # 2) East Coast lead institution (critical)
    east_leaf = evaluator.add_leaf(
        id="East_Coast_Location",
        desc="The lead institution of the institute is located in a state on the East Coast of the United States (a state bordering the Atlantic Ocean)",
        parent=inst_node,
        critical=True
    )
    east_claim = (
        f"The lead institution '{core.lead_institution or 'the identified lead institution'}' "
        f"is located in a U.S. state that borders the Atlantic Ocean (i.e., an East Coast state)."
    )
    await evaluator.verify(
        claim=east_claim,
        node=east_leaf,
        sources=merged_sources,
        additional_instruction=(
            "Focus on whether the lead institution's state borders the Atlantic Ocean. "
            f"East Coast states include: {', '.join(EAST_COAST_STATES)}. "
            "Look for location/address pages, 'About' pages, or official descriptions indicating the state. "
            "Minor variants in naming are acceptable (e.g., 'UMD' for 'University of Maryland')."
        ),
    )

    # 3) Multi-institution partnership (>=3 partners) (critical)
    partners_leaf = evaluator.add_leaf(
        id="Multi_Institution_Partnership",
        desc="The institute is a partnership involving at least three distinct universities or research institutions as formal partners",
        parent=inst_node,
        critical=True
    )
    partners_claim = (
        f"The institute '{core.institute_name or 'the identified institute'}' "
        "is a multi-institution partnership that involves at least three distinct universities or research institutions as formal partners."
    )
    await evaluator.verify(
        claim=partners_claim,
        node=partners_leaf,
        sources=merged_sources,
        additional_instruction=(
            "Check the institute website or NSF pages for a list of partner institutions. "
            "Count distinct universities/research institutions. Accept labels like 'partners', 'core institutions', "
            "'member institutions', or 'collaborators' if they clearly denote formal partnership. "
            "At least three distinct named institutions must be present."
        ),
    )

    # 4) Dual federal funding (critical)
    funding_leaf = evaluator.add_leaf(
        id="Dual_Federal_Funding",
        desc="The institute receives funding from at least two distinct federal agencies or federal funding sources",
        parent=inst_node,
        critical=True
    )
    funding_claim = (
        f"The institute '{core.institute_name or 'the identified institute'}' "
        "receives funding from at least two distinct U.S. federal agencies or federal funding sources."
    )
    await evaluator.verify(
        claim=funding_claim,
        node=funding_leaf,
        sources=merged_sources,
        additional_instruction=(
            "Look for statements indicating funding from multiple federal agencies (e.g., NSF, USDA NIFA, DOE, DOT, DHS, NIH, NASA, DoD, etc.). "
            "Phrasing like 'co-funded by' or 'supported by' counts. The evidence must clearly indicate at least two distinct federal sources."
        ),
    )

    # 5) Trustworthiness/governance/ethics/societal implications focus (critical)
    trust_leaf = evaluator.add_leaf(
        id="Trustworthiness_Research_Focus",
        desc="The institute's primary research mission includes trustworthiness, governance, ethics, or societal implications of AI as a central theme",
        parent=inst_node,
        critical=True
    )
    trust_claim = (
        f"The institute '{core.institute_name or 'the identified institute'}' "
        "has a primary research mission in AI that centrally includes trustworthiness, governance, ethics, or societal/socio-technical implications."
    )
    await evaluator.verify(
        claim=trust_claim,
        node=trust_leaf,
        sources=merged_sources,
        additional_instruction=(
            "Check mission statements or research themes. Accept synonyms such as 'responsible AI', 'AI safety', "
            "'fairness', 'accountability', 'governance', 'ethics', 'social impacts', or 'socio-technical' as central themes. "
            "A passing judgment requires that such themes are clearly central, not merely incidental."
        ),
    )

    # 6) Distributed administrative offices/headquarters (critical)
    distributed_leaf = evaluator.add_leaf(
        id="Distributed_Administrative_Structure",
        desc="The institute has administrative offices or headquarters located in more than one geographic location",
        parent=inst_node,
        critical=True
    )
    distributed_claim = (
        f"The institute '{core.institute_name or 'the identified institute'}' "
        "has administrative offices or headquarters in more than one location, indicating a distributed leadership/administrative structure."
    )
    await evaluator.verify(
        claim=distributed_claim,
        node=distributed_leaf,
        sources=merged_sources,
        additional_instruction=(
            "Look for terms like 'co-headquartered', 'administrative offices', 'administrative home', 'co-located leadership', "
            "or explicit mentions of multiple HQ/administrative sites. Multiple campuses where leadership/admin functions reside qualifies."
        ),
    )

    # 7) Public website documentation (critical) – check on the official website specifically
    website_leaf = evaluator.add_leaf(
        id="Public_Website_Documentation",
        desc="The institute has a publicly accessible dedicated website providing information about its mission, partners, and contact details",
        parent=inst_node,
        critical=True
    )
    website_claim = (
        f"The official website for '{core.institute_name or 'the identified institute'}' "
        "is publicly accessible and provides comprehensive information including (1) the mission, (2) partner institutions, and (3) contact details."
    )
    await evaluator.verify(
        claim=website_claim,
        node=website_leaf,
        sources=core.website_url if core.website_url else None,
        additional_instruction=(
            "Confirm the site is the institute's dedicated website. Verify the presence of: "
            "(a) a mission/about page or clear mission statement, "
            "(b) a section listing partner institutions (or equivalent), and "
            "(c) contact information (e.g., email, contact form, or address)."
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
    model: str = "o4-mini",
) -> Dict:
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

    # Extract core fields and all URLs from the answer
    core = await evaluator.extract(
        prompt=prompt_extract_institute_core(),
        template_class=InstituteCore,
        extraction_name="institute_core",
    )

    url_list = await evaluator.extract(
        prompt=prompt_extract_all_urls(),
        template_class=URLList,
        extraction_name="all_urls",
    )

    # Optional: record East Coast states as custom info for transparency
    evaluator.add_custom_info(
        {"east_coast_states": EAST_COAST_STATES},
        info_type="reference_info",
        info_name="geo_reference"
    )

    # Build verification nodes and run checks
    await verify_institute(evaluator, root, core, url_list)

    # Return structured evaluation summary
    return evaluator.get_summary()