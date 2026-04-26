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
TASK_ID = "animal_orgs_excellence"
TASK_DESCRIPTION = """
Identify four US-based animal welfare organizations that demonstrate excellence in animal care and public accountability. For each organization, provide the following information:

1. Organization Name: The full legal name of the organization.

2. Charity Navigator Rating: The organization must have a Charity Navigator score of 95% or higher. Provide the specific percentage score and a link to the Charity Navigator profile page.

3. Headquarters Location: Specify the US state where the organization is headquartered. All four organizations must be based in different states. Provide a reference URL confirming the location.

4. Accreditation Status: Each organization must hold at least one recognized accreditation from either:
   - Global Federation of Animal Sanctuaries (GFAS)
   - Association of Zoos and Aquariums (AZA)
   - Charity Navigator 4-star rating

   Specify the accreditation type and provide a reference URL confirming the accreditation.

5. Mission Focus: Describe the organization's primary mission as it relates to direct animal care, rescue, or sanctuary services. The mission must focus on hands-on animal welfare (not solely advocacy or research). Provide a reference URL from the organization's official website or Charity Navigator profile.

6. Public Engagement: Identify at least one form of public engagement the organization offers (such as tours, volunteer programs, or educational programs). Provide a reference URL confirming these opportunities.

7. Animal Category: Specify the primary category of animals served by the organization (e.g., farm animals, wildlife, equines, companion animals, marine animals). All four organizations must serve different primary animal categories. Provide a reference URL confirming the types of animals served.

All reference URLs must be from official sources (the organization's website, Charity Navigator, GFAS, or AZA).
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OrganizationItem(BaseModel):
    name: Optional[str] = None

    cn_score: Optional[str] = None
    cn_profile_url: Optional[str] = None

    hq_state: Optional[str] = None
    hq_location_urls: List[str] = Field(default_factory=list)

    accreditation_type: Optional[str] = None  # one of: GFAS, AZA, Charity Navigator 4-star rating (free-form allowed)
    accreditation_urls: List[str] = Field(default_factory=list)

    mission_description: Optional[str] = None
    mission_urls: List[str] = Field(default_factory=list)

    engagement_type: Optional[str] = None
    engagement_urls: List[str] = Field(default_factory=list)

    animal_category: Optional[str] = None
    animal_urls: List[str] = Field(default_factory=list)


class OrganizationsExtraction(BaseModel):
    organizations: List[OrganizationItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_organizations() -> str:
    return """
    Extract up to FOUR organizations from the answer that the author proposed for this task. For each organization, return the following fields exactly:

    - name: Full legal organization name (string).
    - cn_score: The numeric or percentage Charity Navigator overall score mentioned in the answer (string; keep as it appears, e.g., "97%", "99/100", "99.0").
    - cn_profile_url: The URL to the organization's Charity Navigator profile (string or null).
    - hq_state: The US state where the organization is headquartered (string; abbreviations or full names accepted).
    - hq_location_urls: An array of URLs that the answer cites for the headquarters location (empty array if none).
    - accreditation_type: A stated accreditation type among (GFAS, AZA, or Charity Navigator 4-star rating). Keep the exact phrasing used in the answer (string or null).
    - accreditation_urls: An array of URLs cited to confirm accreditation (empty array if none).
    - mission_description: A concise description of the organization's mission from the answer, focusing on direct care, rescue, rehab, or sanctuary services (string or null).
    - mission_urls: An array of URLs cited to confirm the mission (empty array if none). Prefer official org website or Charity Navigator.
    - engagement_type: One public engagement type mentioned (e.g., "tours", "volunteer programs", "educational programs") (string or null).
    - engagement_urls: An array of URLs cited to confirm public engagement (empty array if none).
    - animal_category: The primary category of animals served (e.g., farm animals, wildlife, equines, companion animals, marine animals) (string or null).
    - animal_urls: An array of URLs cited to confirm the animal category (empty array if none).

    IMPORTANT:
    - Only extract information explicitly present in the provided answer. Do not invent missing values—use null or [].
    - Extract URLs exactly as they appear (plain or inside markdown links).
    - If the answer contains more than four organizations, extract the first four only. If fewer, extract those available.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _clean_list(values: List[Optional[str]]) -> List[str]:
    return [v.strip() for v in values if v and isinstance(v, str) and v.strip()]


def _no_duplicates_nonempty(values: List[Optional[str]]) -> bool:
    cleaned = [v.strip().lower() for v in values if v and isinstance(v, str) and v.strip()]
    return len(cleaned) == len(set(cleaned))


def _first_url_or_none(urls: List[str]) -> Optional[str]:
    return urls[0] if urls else None


# --------------------------------------------------------------------------- #
# Verification logic per-organization                                         #
# --------------------------------------------------------------------------- #
async def verify_single_org(evaluator: Evaluator, root, org: OrganizationItem, idx: int) -> None:
    """
    Build verification subtree and run checks for a single organization.
    Mirrors the rubric's structure closely while adding minimal gating for missing references.
    """
    org_idx = idx + 1
    org_prefix = f"Org{org_idx}_"
    org_node = evaluator.add_parallel(
        id=f"Organization_{org_idx}",
        desc=f"{['First','Second','Third','Fourth'][idx]} identified animal welfare organization meeting all specified criteria",
        parent=root,
        critical=False
    )

    # 1) Identification (critical leaf: existence of name)
    evaluator.add_custom_node(
        result=bool(org.name and org.name.strip()),
        id=f"{org_prefix}Identification",
        desc="Provide the full legal name of the organization",
        parent=org_node,
        critical=True
    )

    # 2) Charity Navigator Rating (critical group)
    cn_group = evaluator.add_parallel(
        id=f"{org_prefix}Charity_Navigator_Rating",
        desc="Verify the organization's Charity Navigator rating and score",
        parent=org_node,
        critical=True
    )

    # Gating: reference provided and appears to be Charity Navigator
    cn_ref_present = bool(org.cn_profile_url and org.cn_profile_url.strip())
    evaluator.add_custom_node(
        result=cn_ref_present,
        id=f"{org_prefix}CN_Ref_Provided",
        desc="Charity Navigator profile URL is provided",
        parent=cn_group,
        critical=True
    )

    # 2a) CN Score >= 95%
    cn_score_leaf = evaluator.add_leaf(
        id=f"{org_prefix}CN_Score",
        desc="The organization must have a Charity Navigator score of 95% or higher",
        parent=cn_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The organization '{org.name or ''}' has an overall Charity Navigator score of at least 95 out of 100 (95% or higher).",
        node=cn_score_leaf,
        sources=org.cn_profile_url,
        additional_instruction="Use ONLY the Charity Navigator profile page. Accept if overall/combined/total score is 95 or higher (out of 100). Minor formatting differences like '95/100' vs '95%' are acceptable."
    )

    # 2b) CN Reference authenticity/content
    cn_ref_leaf = evaluator.add_leaf(
        id=f"{org_prefix}CN_Reference",
        desc="Provide reference URL from Charity Navigator showing the rating",
        parent=cn_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"This webpage is the Charity Navigator profile for '{org.name or ''}' and shows the organization's overall rating/score.",
        node=cn_ref_leaf,
        sources=org.cn_profile_url,
        additional_instruction="Judge correctness only if the page is clearly on charitynavigator.org and displays an overall rating/score section."
    )

    # 3) Geographic Location (critical group)
    geo_group = evaluator.add_parallel(
        id=f"{org_prefix}Geographic_Location",
        desc="Verify the organization's headquarters location",
        parent=org_node,
        critical=True
    )

    # Gating: at least one location URL
    has_loc_ref = len(_clean_list(org.hq_location_urls)) > 0
    evaluator.add_custom_node(
        result=has_loc_ref,
        id=f"{org_prefix}Location_Ref_Provided",
        desc="Provide reference URL confirming the headquarters location (at least one URL provided)",
        parent=geo_group,
        critical=True
    )

    # 3a) State verification (with evidence)
    state_leaf = evaluator.add_leaf(
        id=f"{org_prefix}State",
        desc="Provide the US state where the organization is headquartered",
        parent=geo_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The organization '{org.name or ''}' is headquartered in the U.S. state '{org.hq_state or ''}'.",
        node=state_leaf,
        sources=org.hq_location_urls,
        additional_instruction="Verify the U.S. state (accept either full state name or USPS two-letter abbreviation). Use only official pages (org site or Charity Navigator page with address)."
    )

    # 3b) Location reference authenticity (official source)
    loc_ref_leaf = evaluator.add_leaf(
        id=f"{org_prefix}Location_Reference",
        desc="Provide reference URL confirming the headquarters location",
        parent=geo_group,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an official source (the organization's own website or Charity Navigator) confirming the organization's headquarters location.",
        node=loc_ref_leaf,
        sources=org.hq_location_urls,
        additional_instruction="Accept only if the domain is the organization's website (brand-matching) or charitynavigator.org, and the page shows location/address details."
    )

    # 4) Accreditation Status (critical group)
    accred_group = evaluator.add_parallel(
        id=f"{org_prefix}Accreditation_Status",
        desc="Verify the organization holds at least one recognized accreditation",
        parent=org_node,
        critical=True
    )

    has_accred_ref = len(_clean_list(org.accreditation_urls)) > 0
    evaluator.add_custom_node(
        result=has_accred_ref,
        id=f"{org_prefix}Accreditation_Ref_Provided",
        desc="Provide reference URL confirming the accreditation status (at least one URL provided)",
        parent=accred_group,
        critical=True
    )

    # 4a) Accreditation type content (GFAS/AZA/CN 4-star)
    accred_type_leaf = evaluator.add_leaf(
        id=f"{org_prefix}Accreditation_Type",
        desc="Specify the type of accreditation (GFAS, AZA, or Charity Navigator 4-star)",
        parent=accred_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The organization '{org.name or ''}' holds the accreditation '{org.accreditation_type or ''}', which must be one of: GFAS accreditation, AZA accreditation, or a Charity Navigator 4-star rating.",
        node=accred_type_leaf,
        sources=org.accreditation_urls,
        additional_instruction="Verify that the page explicitly indicates GFAS-accredited, AZA-accredited, or a 4-star rating by Charity Navigator for this organization."
    )

    # 4b) Accreditation reference authenticity
    accred_ref_leaf = evaluator.add_leaf(
        id=f"{org_prefix}Accreditation_Reference",
        desc="Provide reference URL confirming the accreditation status",
        parent=accred_group,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an official source (GFAS, AZA, Charity Navigator, or the organization's website) that confirms the accreditation status.",
        node=accred_ref_leaf,
        sources=org.accreditation_urls,
        additional_instruction="Accept if the URL domain is clearly gfas.org, aza.org, charitynavigator.org, or the organization's official website; and the page states accreditation."
    )

    # 5) Mission Focus (critical group)
    mission_group = evaluator.add_parallel(
        id=f"{org_prefix}Mission_Focus",
        desc="Verify the organization's mission focuses on direct animal care, rescue, or sanctuary services",
        parent=org_node,
        critical=True
    )

    has_mission_ref = len(_clean_list(org.mission_urls)) > 0
    evaluator.add_custom_node(
        result=has_mission_ref,
        id=f"{org_prefix}Mission_Ref_Provided",
        desc="Provide mission reference URL(s) (at least one provided)",
        parent=mission_group,
        critical=True
    )

    # 5a) Mission description and direct-care emphasis
    mission_desc_leaf = evaluator.add_leaf(
        id=f"{org_prefix}Mission_Description",
        desc="Describe the organization's primary mission related to direct animal care",
        parent=mission_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"According to the source, the organization's primary mission involves: {org.mission_description or ''}. The mission clearly emphasizes hands-on animal welfare (e.g., direct care, rescue, rehabilitation, or sanctuary services), not solely advocacy or research.",
        node=mission_desc_leaf,
        sources=org.mission_urls,
        additional_instruction="Look for 'mission', 'about', or similar sections. Ensure the language indicates direct care/rescue/sanctuary activities as a core focus."
    )

    # 5b) Mission reference authenticity
    mission_ref_leaf = evaluator.add_leaf(
        id=f"{org_prefix}Mission_Reference",
        desc="Provide reference URL from the organization's official website or Charity Navigator profile describing their mission",
        parent=mission_group,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an official source (the organization's website or Charity Navigator) that describes the mission.",
        node=mission_ref_leaf,
        sources=org.mission_urls,
        additional_instruction="Accept only if the page is on the organization's official website (matching brand/domain) or charitynavigator.org."
    )

    # 6) Public Engagement (critical group)
    engage_group = evaluator.add_parallel(
        id=f"{org_prefix}Public_Engagement",
        desc="Verify the organization offers at least one form of public engagement",
        parent=org_node,
        critical=True
    )

    has_engage_ref = len(_clean_list(org.engagement_urls)) > 0
    evaluator.add_custom_node(
        result=has_engage_ref,
        id=f"{org_prefix}Engagement_Ref_Provided",
        desc="Provide public engagement reference URL(s) (at least one provided)",
        parent=engage_group,
        critical=True
    )

    # 6a) Engagement type content
    engage_type_leaf = evaluator.add_leaf(
        id=f"{org_prefix}Engagement_Type",
        desc="Specify the type of public engagement offered (tours, volunteer programs, or educational programs)",
        parent=engage_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"According to the source, the organization offers public engagement opportunities such as '{org.engagement_type or ''}' (e.g., tours, volunteer programs, or educational programs).",
        node=engage_type_leaf,
        sources=org.engagement_urls,
        additional_instruction="Look for explicit mentions of tours, volunteering, internships, education, events, or similar programs offered to the public."
    )

    # 6b) Engagement reference authenticity
    engage_ref_leaf = evaluator.add_leaf(
        id=f"{org_prefix}Engagement_Reference",
        desc="Provide reference URL confirming the public engagement opportunities",
        parent=engage_group,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an official source (the organization's website or Charity Navigator) confirming public engagement opportunities.",
        node=engage_ref_leaf,
        sources=org.engagement_urls,
        additional_instruction="Accept only if the page is on the organization's official website (matching brand/domain) or charitynavigator.org."
    )

    # 7) Animal Category (critical group)
    animal_group = evaluator.add_parallel(
        id=f"{org_prefix}Animal_Category",
        desc="Identify the primary category of animals served by the organization",
        parent=org_node,
        critical=True
    )

    has_animal_ref = len(_clean_list(org.animal_urls)) > 0
    evaluator.add_custom_node(
        result=has_animal_ref,
        id=f"{org_prefix}Animal_Ref_Provided",
        desc="Provide reference URL confirming the types of animals served (at least one provided)",
        parent=animal_group,
        critical=True
    )

    # 7a) Animal category content
    animal_type_leaf = evaluator.add_leaf(
        id=f"{org_prefix}Animal_Type",
        desc="Specify the animal category (e.g., farm animals, wildlife, equines, companion animals, marine animals)",
        parent=animal_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"According to the source, the organization's primary animal focus/category is '{org.animal_category or ''}'.",
        node=animal_type_leaf,
        sources=org.animal_urls,
        additional_instruction="Allow reasonable synonyms: e.g., 'equine'/'horses', 'companion animals'/'dogs and cats', 'marine animals'/'marine mammals', 'farm animals'/'farmed animals'."
    )

    # 7b) Animal reference authenticity
    animal_ref_leaf = evaluator.add_leaf(
        id=f"{org_prefix}Animal_Reference",
        desc="Provide reference URL confirming the types of animals served",
        parent=animal_group,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an official source (the organization's website or Charity Navigator, GFAS, or AZA) confirming the types of animals served.",
        node=animal_ref_leaf,
        sources=org.animal_urls,
        additional_instruction="Accept only if the domain is the organization's website (brand-matching) or charitynavigator.org, gfas.org, or aza.org."
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
) -> Dict:
    """
    Evaluate an answer for the US-based animal welfare organizations task.
    """
    # Initialize evaluator (root is parallel as in rubric)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_organizations(),
        template_class=OrganizationsExtraction,
        extraction_name="organizations_extraction"
    )

    # Normalize to 4 organizations (truncate or pad)
    orgs: List[OrganizationItem] = list(extracted.organizations[:4])
    while len(orgs) < 4:
        orgs.append(OrganizationItem())

    # Record informational custom guidance
    evaluator.add_custom_info(
        info={
            "recognized_accreditations": [
                "GFAS (Global Federation of Animal Sanctuaries) accreditation",
                "AZA (Association of Zoos and Aquariums) accreditation",
                "Charity Navigator 4-star rating"
            ],
            "official_sources_policy": "All reference URLs must be from the organization's website, Charity Navigator, GFAS, or AZA.",
        },
        info_type="policy",
        info_name="evaluation_policy"
    )

    # Build per-organization verification subtrees
    for i, org in enumerate(orgs):
        await verify_single_org(evaluator, root, org, i)

    # Global distinctness checks (non-critical to allow partial credit)
    states = [o.hq_state for o in orgs]
    categories = [o.animal_category for o in orgs]

    evaluator.add_custom_node(
        result=_no_duplicates_nonempty(states),
        id="Distinct_States",
        desc="All organizations are headquartered in distinct US states (among the provided non-empty states).",
        parent=root,
        critical=False
    )

    evaluator.add_custom_node(
        result=_no_duplicates_nonempty(categories),
        id="Distinct_Animal_Categories",
        desc="All organizations have distinct primary animal categories (among the provided non-empty categories).",
        parent=root,
        critical=False
    )

    # Return structured summary
    return evaluator.get_summary()