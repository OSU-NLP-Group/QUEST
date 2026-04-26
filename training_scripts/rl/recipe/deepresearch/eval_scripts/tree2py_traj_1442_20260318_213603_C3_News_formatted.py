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
TASK_ID = "senator_montana_criteria_2026"
TASK_DESCRIPTION = """
Identify the full name and provide the birth date (month, day, and year) of the U.S. Senator who meets ALL of the following criteria: (1) Currently serving Montana in the U.S. Senate as of March 2026; (2) Is a member of the Republican Party; (3) Was sworn into office in January 2025; (4) Graduated from the U.S. Naval Academy in 2008, having attended from June 2004 to May 2008; (5) Was the first Midshipman to participate in the U.S. Army Special Operations exchange program; (6) Was the first Midshipman to graduate from Army Ranger School; (7) Served as a Navy SEAL officer; (8) Received a Bronze Star Medal with 'V' device for valor; (9) Received a Purple Heart Medal; (10) Founded an aerial firefighting company named Bridger Aerospace in 2014, headquartered in Belgrade, Montana; (11) Secured provisions in the FY 2026 National Defense Authorization Act and announced these achievements on October 10, 2025, with the provisions focusing on modernizing the defense acquisition system. Provide the senator's full name and exact birth date (including month, day, and year) along with their birthplace.
"""

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class CriterionEvidence(BaseModel):
    urls: List[str] = Field(default_factory=list)


class CriteriaSources(BaseModel):
    current_serving: CriterionEvidence = Field(default_factory=CriterionEvidence)
    party: CriterionEvidence = Field(default_factory=CriterionEvidence)
    sworn_in_jan_2025: CriterionEvidence = Field(default_factory=CriterionEvidence)
    usna_2008: CriterionEvidence = Field(default_factory=CriterionEvidence)
    first_midshipman_sof_exchange: CriterionEvidence = Field(default_factory=CriterionEvidence)
    first_midshipman_ranger: CriterionEvidence = Field(default_factory=CriterionEvidence)
    navy_seal_officer: CriterionEvidence = Field(default_factory=CriterionEvidence)
    bronze_star_v: CriterionEvidence = Field(default_factory=CriterionEvidence)
    purple_heart: CriterionEvidence = Field(default_factory=CriterionEvidence)
    founded_bridger_2014_belgrade: CriterionEvidence = Field(default_factory=CriterionEvidence)
    fy2026_ndaa_announcement_2025_10_10: CriterionEvidence = Field(default_factory=CriterionEvidence)
    full_name: CriterionEvidence = Field(default_factory=CriterionEvidence)
    birth_date: CriterionEvidence = Field(default_factory=CriterionEvidence)
    birthplace: CriterionEvidence = Field(default_factory=CriterionEvidence)


class SenatorExtraction(BaseModel):
    senator_name: Optional[str] = None
    birth_date: Optional[str] = None  # Prefer "Month Day, Year" as in the answer
    birth_place: Optional[str] = None
    all_urls: List[str] = Field(default_factory=list)
    sources: CriteriaSources = Field(default_factory=CriteriaSources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_senator() -> str:
    return """
    From the answer, extract:
    - senator_name: The full name of the identified U.S. Senator.
    - birth_date: The senator’s full birth date as provided in the answer (month, day, year). If not present, return null.
    - birth_place: The senator’s birthplace as provided in the answer. If not present, return null.
    - all_urls: A flat list of every URL explicitly present anywhere in the answer (plain URLs or in markdown).
    - sources: For each specific criterion below, extract the subset of URLs from the answer that the answer associates with that claim. If the answer does not clearly associate URLs to that claim, return an empty list for that item.

    Structure the "sources" object with these fields (each holds "urls": []):
      - current_serving
      - party
      - sworn_in_jan_2025
      - usna_2008
      - first_midshipman_sof_exchange
      - first_midshipman_ranger
      - navy_seal_officer
      - bronze_star_v
      - purple_heart
      - founded_bridger_2014_belgrade
      - fy2026_ndaa_announcement_2025_10_10
      - full_name
      - birth_date
      - birthplace

    Important:
    - Do not invent any URLs. Only extract URLs explicitly present in the answer text.
    - If any field is missing in the answer, set it to null (or an empty list for urls arrays).
    - Keep text exactly as in the answer; do not normalize names, dates, or places.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for x in items:
        if x is None:
            continue
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def _choose_urls(preferred: List[str], fallback_all: List[str]) -> Optional[List[str]]:
    """Prefer per-criterion URLs; otherwise use all URLs; if still empty, return None."""
    preferred = [u for u in preferred if isinstance(u, str) and u.strip()]
    fallback_all = [u for u in fallback_all if isinstance(u, str) and u.strip()]
    urls = preferred if len(preferred) > 0 else fallback_all
    urls = _dedupe_preserve_order(urls)
    return urls if len(urls) > 0 else None


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_and_verify_criteria(
    evaluator: Evaluator,
    parent_node,
    ex: SenatorExtraction,
):
    """
    Build the 'Meets_All_Question_Criteria' branch and run URL-grounded verifications.
    """
    n = ex.senator_name or ""
    all_urls = _dedupe_preserve_order(ex.all_urls or [])

    crit_node = evaluator.add_parallel(
        id="Meets_All_Question_Criteria",
        desc="The identified senator satisfies every criterion listed in the question.",
        parent=parent_node,
        critical=True,
    )

    # Prepare leaf nodes
    leaf_nodes = []

    # 1) Currently serving MT as of Mar 2026
    node_1 = evaluator.add_leaf(
        id="Currently_Serving_Montana_As_Of_March_2026",
        desc="The senator is currently serving as a U.S. Senator representing Montana as of March 2026.",
        parent=crit_node,
        critical=True,
    )
    claim_1 = f"As of March 2026, {n} is serving as a United States Senator representing Montana."
    urls_1 = _choose_urls(ex.sources.current_serving.urls, all_urls)
    add_1 = (
        "Confirm that the page shows the person is a U.S. Senator from Montana. "
        "If the page shows a term that began in January 2025 (standard 6-year term), "
        "that implies serving as of March 2026."
    )
    leaf_nodes.append((claim_1, urls_1, node_1, add_1))

    # 2) Republican Party
    node_2 = evaluator.add_leaf(
        id="Republican_Party",
        desc="The senator is a member of the Republican Party.",
        parent=crit_node,
        critical=True,
    )
    claim_2 = f"{n} is a member of the Republican Party."
    urls_2 = _choose_urls(ex.sources.party.urls, all_urls)
    add_2 = "Verify the page explicitly states Republican affiliation (R)."
    leaf_nodes.append((claim_2, urls_2, node_2, add_2))

    # 3) Sworn in January 2025
    node_3 = evaluator.add_leaf(
        id="Sworn_In_January_2025",
        desc="The senator was sworn into office in January 2025.",
        parent=crit_node,
        critical=True,
    )
    claim_3 = f"{n} was sworn into the U.S. Senate in January 2025."
    urls_3 = _choose_urls(ex.sources.sworn_in_jan_2025.urls, all_urls)
    add_3 = "Accept exact dates like January 3, 2025 or phrasing 'sworn in January 2025'."
    leaf_nodes.append((claim_3, urls_3, node_3, add_3))

    # 4) USNA 2008 graduation; attendance June 2004–May 2008
    node_4 = evaluator.add_leaf(
        id="USNA_Graduation_And_Attendance",
        desc="The senator attended the U.S. Naval Academy from June 2004 to May 2008 and graduated in 2008.",
        parent=crit_node,
        critical=True,
    )
    claim_4 = f"{n} graduated from the U.S. Naval Academy in 2008 and attended from June 2004 to May 2008."
    urls_4 = _choose_urls(ex.sources.usna_2008.urls, all_urls)
    add_4 = (
        "Look for explicit mentions of USNA graduation year (2008). "
        "Attendance months may be stated directly; if only years (2004–2008) are provided but consistent with June 2004–May 2008, consider this acceptable."
    )
    leaf_nodes.append((claim_4, urls_4, node_4, add_4))

    # 5) First Midshipman to participate in U.S. Army SOF exchange
    node_5 = evaluator.add_leaf(
        id="First_Midshipman_SOF_Exchange",
        desc="The senator was the first Midshipman to participate in the U.S. Army Special Operations exchange program.",
        parent=crit_node,
        critical=True,
    )
    claim_5 = f"{n} was the first Midshipman to participate in the U.S. Army Special Operations exchange program."
    urls_5 = _choose_urls(ex.sources.first_midshipman_sof_exchange.urls, all_urls)
    add_5 = "The source should explicitly state 'first Midshipman' in the Army Special Operations exchange context."
    leaf_nodes.append((claim_5, urls_5, node_5, add_5))

    # 6) First Midshipman to graduate Army Ranger School
    node_6 = evaluator.add_leaf(
        id="First_Midshipman_Ranger_School_Graduate",
        desc="The senator was the first Midshipman to graduate from Army Ranger School.",
        parent=crit_node,
        critical=True,
    )
    claim_6 = f"{n} was the first Midshipman to graduate from Army Ranger School."
    urls_6 = _choose_urls(ex.sources.first_midshipman_ranger.urls, all_urls)
    add_6 = "The source should make clear he was the first Midshipman to graduate from Army Ranger School."
    leaf_nodes.append((claim_6, urls_6, node_6, add_6))

    # 7) Served as a Navy SEAL officer
    node_7 = evaluator.add_leaf(
        id="Served_As_Navy_SEAL_Officer",
        desc="The senator served as a Navy SEAL officer.",
        parent=crit_node,
        critical=True,
    )
    claim_7 = f"{n} served as a Navy SEAL officer."
    urls_7 = _choose_urls(ex.sources.navy_seal_officer.urls, all_urls)
    add_7 = "Look for explicit mention that he served as a Navy SEAL and held officer rank."
    leaf_nodes.append((claim_7, urls_7, node_7, add_7))

    # 8) Bronze Star with "V" device
    node_8 = evaluator.add_leaf(
        id="Bronze_Star_With_V_Device",
        desc="The senator received a Bronze Star Medal with a 'V' device for valor.",
        parent=crit_node,
        critical=True,
    )
    claim_8 = f"{n} received a Bronze Star Medal with a 'V' device for valor."
    urls_8 = _choose_urls(ex.sources.bronze_star_v.urls, all_urls)
    add_8 = "The page should clearly indicate Bronze Star with 'V' (valor) device."
    leaf_nodes.append((claim_8, urls_8, node_8, add_8))

    # 9) Purple Heart Medal
    node_9 = evaluator.add_leaf(
        id="Purple_Heart_Medal",
        desc="The senator received a Purple Heart Medal.",
        parent=crit_node,
        critical=True,
    )
    claim_9 = f"{n} received a Purple Heart Medal."
    urls_9 = _choose_urls(ex.sources.purple_heart.urls, all_urls)
    add_9 = "The page should explicitly state that he received a Purple Heart."
    leaf_nodes.append((claim_9, urls_9, node_9, add_9))

    # 10) Founded Bridger Aerospace in 2014; HQ Belgrade, MT
    node_10 = evaluator.add_leaf(
        id="Founded_Bridger_Aerospace_With_Specified_Details",
        desc="The senator founded an aerial firefighting company named Bridger Aerospace in 2014, headquartered in Belgrade, Montana.",
        parent=crit_node,
        critical=True,
    )
    claim_10 = (
        f"{n} founded an aerial firefighting company named Bridger Aerospace in 2014, headquartered in Belgrade, Montana."
    )
    urls_10 = _choose_urls(ex.sources.founded_bridger_2014_belgrade.urls, all_urls)
    add_10 = "Verify both the founding in 2014 and the headquarters in Belgrade, Montana."
    leaf_nodes.append((claim_10, urls_10, node_10, add_10))

    # 11) FY2026 NDAA provisions; announced Oct 10, 2025; focus on modernizing defense acquisition
    node_11 = evaluator.add_leaf(
        id="FY2026_NDAA_Provisions_And_Announcement",
        desc="The senator secured provisions in the FY 2026 NDAA, announced these on Oct 10, 2025, focusing on modernizing the defense acquisition system.",
        parent=crit_node,
        critical=True,
    )
    claim_11 = (
        f"On October 10, 2025, {n} announced that he secured provisions in the FY 2026 National Defense Authorization Act "
        f"that focus on modernizing the defense acquisition system."
    )
    urls_11 = _choose_urls(ex.sources.fy2026_ndaa_announcement_2025_10_10.urls, all_urls)
    add_11 = "Look for a press release or credible report dated October 10, 2025 describing FY2026 NDAA provisions on acquisition modernization."
    leaf_nodes.append((claim_11, urls_11, node_11, add_11))

    # Execute verifications in parallel
    await evaluator.batch_verify(leaf_nodes)


async def build_and_verify_identity(
    evaluator: Evaluator,
    parent_node,
    ex: SenatorExtraction,
):
    """
    Build the 'Provides_Requested_Identity_Details_Correctly' branch and run URL-grounded verifications.
    """
    n = ex.senator_name or ""
    bdate = ex.birth_date or ""
    bplace = ex.birth_place or ""
    all_urls = _dedupe_preserve_order(ex.all_urls or [])

    id_node = evaluator.add_parallel(
        id="Provides_Requested_Identity_Details_Correctly",
        desc="The answer provides the requested identity information for the identified senator, and the details are correct for that senator.",
        parent=parent_node,
        critical=True,
    )

    leaf_nodes = []

    # Full name
    name_node = evaluator.add_leaf(
        id="Provide_Full_Name_Correct",
        desc="The answer provides the senator's full name (correct for the identified senator).",
        parent=id_node,
        critical=True,
    )
    claim_name = f"The senator's full name is {n}."
    urls_name = _choose_urls(ex.sources.full_name.urls, all_urls)
    add_name = "Verify that the page clearly identifies the person's full name as provided."
    leaf_nodes.append((claim_name, urls_name, name_node, add_name))

    # Birth date (MDY)
    bdate_node = evaluator.add_leaf(
        id="Provide_Birth_Date_MDY_Correct",
        desc="The answer provides the senator's birth date with month, day, and year (correct for the identified senator).",
        parent=id_node,
        critical=True,
    )
    claim_bdate = f"{n} was born on {bdate}."
    urls_bdate = _choose_urls(ex.sources.birth_date.urls, all_urls)
    add_bdate = "Confirm the exact month-day-year birth date for the same person."
    leaf_nodes.append((claim_bdate, urls_bdate, bdate_node, add_bdate))

    # Birthplace
    bplace_node = evaluator.add_leaf(
        id="Provide_Birthplace_Correct",
        desc="The answer provides the senator's birthplace (correct for the identified senator).",
        parent=id_node,
        critical=True,
    )
    claim_bplace = f"{n} was born in {bplace}."
    urls_bplace = _choose_urls(ex.sources.birthplace.urls, all_urls)
    add_bplace = "Confirm the birthplace location listed is correct for the same person."
    leaf_nodes.append((claim_bplace, urls_bplace, bplace_node, add_bplace))

    # Execute verifications in parallel
    await evaluator.batch_verify(leaf_nodes)


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
    Evaluate an answer for the Montana Senator identification and identity details task.
    """
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_senator(),
        template_class=SenatorExtraction,
        extraction_name="senator_extraction",
    )

    # Build top-level task node (critical, sequential)
    task_node = evaluator.add_sequential(
        id="Senator_Identification_Task",
        desc="Identify the U.S. Senator from Montana who meets all specified criteria and provide the requested identity details.",
        parent=root,
        critical=True,
    )

    # First branch: Meet all criteria
    await build_and_verify_criteria(evaluator, task_node, extracted)

    # Second branch: Provide requested identity details correctly
    await build_and_verify_identity(evaluator, task_node, extracted)

    return evaluator.get_summary()