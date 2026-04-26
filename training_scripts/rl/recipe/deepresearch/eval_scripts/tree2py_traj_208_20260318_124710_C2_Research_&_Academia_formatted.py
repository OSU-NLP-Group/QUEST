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
TASK_ID = "identify_erau_inspiration4_nasa_isdc_person"
TASK_DESCRIPTION = (
    "An individual graduated from Embry-Riddle Aeronautical University in 2011 with a bachelor's degree in Aeronautics "
    "and later received an honorary doctoral degree from the same institution in 2024. This person commanded Inspiration4, "
    "the world's first all-civilian mission to space, which raised over $240 million for St. Jude Children's Research Hospital. "
    "The individual was confirmed as NASA's 15th Administrator by the U.S. Senate on December 17, 2025. Identify this individual "
    "by providing their full name. Additionally, verify whether this person is listed as a scheduled speaker for the International "
    "Space Development Conference (ISDC) 2026, which takes place June 4-7, 2026, in McLean, Virginia, according to the official "
    "ISDC 2026 website. Provide supporting URL references for: (1) the individual's educational background at Embry-Riddle, "
    "(2) their role in the Inspiration4 mission and fundraising for St. Jude, (3) their confirmation as NASA Administrator, "
    "and (4) their speaking status at ISDC 2026."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class URLBundle(BaseModel):
    urls: List[str] = Field(default_factory=list)


class MainExtraction(BaseModel):
    full_name: Optional[str] = None
    education_urls: List[str] = Field(default_factory=list)
    inspiration4_urls: List[str] = Field(default_factory=list)
    nasa_admin_urls: List[str] = Field(default_factory=list)
    isdc_2026_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return """
    Extract the following fields from the provided answer:

    1) full_name: The individual's full name (the person described in the task).
    2) education_urls: All URLs cited to support the Embry-Riddle educational background (2011 bachelor's in Aeronautics and 2024 honorary doctorate).
    3) inspiration4_urls: All URLs cited to support that the individual commanded Inspiration4 (the world's first all-civilian mission to space/orbit) and that the mission raised over $240 million for St. Jude Children's Research Hospital.
    4) nasa_admin_urls: All URLs cited to support that the U.S. Senate confirmed the individual as NASA's 15th Administrator on December 17, 2025.
    5) isdc_2026_urls: All URLs cited (prefer official ISDC 2026 website pages) to support ISDC 2026 details (dates/location) and the individual's speaking status for ISDC 2026. The official site typically uses the domain 'isdc.nss.org/2026'. Include any speakers, schedule, program, or past speakers pages that the answer references.

    IMPORTANT:
    - Only extract URLs explicitly present in the answer text. Do not invent or infer URLs.
    - Return null for full_name if not provided in the answer. Return empty arrays for any URL lists that are not present.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_official_isdc_2026_url(urls: List[str]) -> bool:
    for u in urls or []:
        lu = (u or "").lower()
        if "isdc.nss.org/2026" in lu:
            return True
    return False


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def add_and_verify_education(
    evaluator: Evaluator,
    parent,
    name: Optional[str],
    urls: List[str],
):
    node = evaluator.add_parallel(
        id="education_verification",
        desc="Verify the individual's Embry-Riddle educational background per constraints and provide supporting URL(s).",
        parent=parent,
        critical=True,
    )

    # URL existence (critical)
    _ = evaluator.add_custom_node(
        result=(len(urls) > 0),
        id="url_education_embryriddle",
        desc="Provide supporting URL reference(s) for the individual's Embry-Riddle educational background.",
        parent=node,
        critical=True,
    )

    # 2011 Bachelor's in Aeronautics
    leaf_bach = evaluator.add_leaf(
        id="erau_2011_bachelors_aeronautics",
        desc="Verify the individual graduated from Embry-Riddle Aeronautical University in 2011 with a bachelor's degree in Aeronautics.",
        parent=node,
        critical=True,
    )
    claim_bach = (
        f"{name} graduated from Embry-Riddle Aeronautical University in 2011 with a bachelor's degree in Aeronautics "
        "(acceptable variants include 'Bachelor of Science in Aeronautics' or 'Bachelor of Professional Aeronautics', "
        "including Embry‑Riddle Worldwide/online)."
    )

    # 2024 Honorary doctorate
    leaf_hon = evaluator.add_leaf(
        id="erau_2024_honorary_doctorate",
        desc="Verify the individual received an honorary doctoral degree from Embry-Riddle Aeronautical University in 2024.",
        parent=node,
        critical=True,
    )
    claim_hon = (
        f"In 2024, Embry‑Riddle Aeronautical University awarded {name} an honorary doctoral degree "
        "(acceptable phrasings: 'honorary doctorate', 'honoris causa', 'honorary Doctor of Science', etc.)."
    )

    await evaluator.batch_verify(
        [
            (
                claim_bach,
                urls,
                leaf_bach,
                "Use only the provided URLs. Allow reasonable wording variants (e.g., Bachelor's in Aeronautics vs. Bachelor of Professional Aeronautics). The key points are: Embry‑Riddle, year 2011, bachelor's-level Aeronautics degree.",
            ),
            (
                claim_hon,
                urls,
                leaf_hon,
                "Use only the provided URLs. Confirm that Embry‑Riddle granted an honorary doctoral/doctorate-level degree in the year 2024 to this individual.",
            ),
        ]
    )


async def add_and_verify_inspiration4(
    evaluator: Evaluator,
    parent,
    name: Optional[str],
    urls: List[str],
):
    node = evaluator.add_parallel(
        id="inspiration4_verification",
        desc="Verify the individual's role in Inspiration4 and the fundraising result for St. Jude, and provide supporting URL(s).",
        parent=parent,
        critical=True,
    )

    # URL existence (critical)
    _ = evaluator.add_custom_node(
        result=(len(urls) > 0),
        id="url_inspiration4_and_fundraising",
        desc="Provide supporting URL reference(s) for the individual's role in Inspiration4 and the fundraising for St. Jude.",
        parent=node,
        critical=True,
    )

    # Commanded Inspiration4
    leaf_cmd = evaluator.add_leaf(
        id="commanded_inspiration4",
        desc="Verify the individual commanded Inspiration4, described as the world's first all-civilian mission to space.",
        parent=node,
        critical=True,
    )
    claim_cmd = (
        f"{name} served as the mission commander of Inspiration4, which is described as the world's first all‑civilian "
        "mission to space (or to orbit)."
    )

    # Raised over $240M for St. Jude
    leaf_funds = evaluator.add_leaf(
        id="raised_over_240m_for_stjude",
        desc="Verify Inspiration4 raised over $240 million for St. Jude Children's Research Hospital.",
        parent=node,
        critical=True,
    )
    claim_funds = (
        "The Inspiration4 mission raised over $240 million for St. Jude Children's Research Hospital."
    )

    await evaluator.batch_verify(
        [
            (
                claim_cmd,
                urls,
                leaf_cmd,
                "Confirm the person was Inspiration4's commander and that the mission is characterized as the first all‑civilian crewed orbital/space mission. Allow minor wording variations.",
            ),
            (
                claim_funds,
                urls,
                leaf_funds,
                "Confirm the total raised exceeded $240 million; allow ranges like '$240+ million' or '$243 million', etc.",
            ),
        ]
    )


async def add_and_verify_nasa_admin(
    evaluator: Evaluator,
    parent,
    name: Optional[str],
    urls: List[str],
):
    node = evaluator.add_parallel(
        id="nasa_administrator_verification",
        desc="Verify the individual's U.S. Senate confirmation as NASA Administrator as specified, and provide supporting URL(s).",
        parent=parent,
        critical=True,
    )

    # URL existence (critical)
    _ = evaluator.add_custom_node(
        result=(len(urls) > 0),
        id="url_nasa_confirmation",
        desc="Provide supporting URL reference(s) for the individual's confirmation as NASA Administrator.",
        parent=node,
        critical=True,
    )

    # Confirmation as NASA's 15th Administrator on Dec 17, 2025
    leaf_confirm = evaluator.add_leaf(
        id="confirmed_nasa_15th_admin_dec_17_2025",
        desc="Verify the individual was confirmed as NASA's 15th Administrator by the U.S. Senate on December 17, 2025.",
        parent=node,
        critical=True,
    )
    claim_confirm = (
        f"On December 17, 2025, the U.S. Senate confirmed {name} as NASA's 15th Administrator."
    )

    await evaluator.verify(
        claim=claim_confirm,
        node=leaf_confirm,
        sources=urls,
        additional_instruction="Focus on the exact date (December 17, 2025), the confirmation action by the U.S. Senate, and the ordinal '15th Administrator'. Allow minor textual variants."
    )


async def add_and_verify_isdc_2026(
    evaluator: Evaluator,
    parent,
    name: Optional[str],
    urls: List[str],
):
    node = evaluator.add_parallel(
        id="isdc_2026_verification",
        desc="Verify ISDC 2026 details and the individual's speaking status using the official ISDC 2026 website, and provide supporting URL(s) for speaking status.",
        parent=parent,
        critical=True,
    )

    # URL existence from official 2026 site (critical)
    _ = evaluator.add_custom_node(
        result=(len(urls) > 0 and has_official_isdc_2026_url(urls)),
        id="url_isdc_2026_official_site_speaking_status",
        desc="Provide supporting URL reference(s) from the official ISDC 2026 website for the individual's speaking status.",
        parent=node,
        critical=True,
    )

    # ISDC 2026 dates and location
    leaf_dates_loc = evaluator.add_leaf(
        id="isdc_2026_dates_and_location",
        desc="Verify ISDC 2026 takes place June 4–7, 2026, in McLean, Virginia.",
        parent=node,
        critical=True,
    )
    claim_dates_loc = (
        "The International Space Development Conference (ISDC) 2026 takes place June 4–7, 2026, in McLean, Virginia."
    )

    # Listed as past speaker (on official 2026 website)
    leaf_past = evaluator.add_leaf(
        id="isdc_listed_as_past_speaker",
        desc="Verify (per the official ISDC 2026 website) that the individual is listed as a past speaker.",
        parent=node,
        critical=True,
    )
    claim_past = (
        f"According to the official ISDC 2026 website, {name} is listed as a past speaker."
    )

    # Not currently scheduled for 2026 program (on official 2026 website)
    leaf_not_sched = evaluator.add_leaf(
        id="isdc_not_currently_scheduled_for_2026",
        desc="Verify (per the official ISDC 2026 website) that the individual is not currently scheduled for the upcoming 2026 program.",
        parent=node,
        critical=True,
    )
    claim_not_sched = (
        f"According to the official ISDC 2026 website, {name} is not currently listed as a scheduled speaker for ISDC 2026."
    )

    await evaluator.batch_verify(
        [
            (
                claim_dates_loc,
                urls,
                leaf_dates_loc,
                "Use only pages from the official ISDC 2026 website where possible. Confirm both the dates (June 4–7, 2026) and the location (McLean, Virginia).",
            ),
            (
                claim_past,
                urls,
                leaf_past,
                "Use only official ISDC 2026 pages (e.g., 'Past Speakers' or equivalent) to see if the individual appears as a past speaker. Allow reasonable name variants.",
            ),
            (
                claim_not_sched,
                urls,
                leaf_not_sched,
                "Use only official ISDC 2026 speaker/program/schedule pages. Verify that the individual is not listed among current 2026 scheduled speakers at the time reflected by the provided pages.",
            ),
        ]
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=MainExtraction,
        extraction_name="main_extraction",
    )

    # Build overall critical sequential node (as per rubric)
    overall = evaluator.add_sequential(
        id="overall_task_completion",
        desc="Identify the individual and provide all required verifications with supporting URLs across the four requested categories.",
        parent=root,
        critical=True,
    )

    # 1) Individual name provided (critical leaf)
    _ = evaluator.add_custom_node(
        result=(extraction.full_name is not None and str(extraction.full_name).strip() != ""),
        id="individual_name_provided",
        desc="Provide the individual's full name.",
        parent=overall,
        critical=True,
    )

    # 2) Required verifications and citations (critical parallel group)
    req = evaluator.add_parallel(
        id="required_verifications_and_citations",
        desc="Provide all required verification statements and the required supporting URL references (four categories).",
        parent=overall,
        critical=True,
    )

    # Education verification (critical group)
    await add_and_verify_education(
        evaluator,
        req,
        extraction.full_name,
        extraction.education_urls or [],
    )

    # Inspiration4 verification (critical group)
    await add_and_verify_inspiration4(
        evaluator,
        req,
        extraction.full_name,
        extraction.inspiration4_urls or [],
    )

    # NASA Administrator verification (critical group)
    await add_and_verify_nasa_admin(
        evaluator,
        req,
        extraction.full_name,
        extraction.nasa_admin_urls or [],
    )

    # ISDC 2026 verification (critical group)
    await add_and_verify_isdc_2026(
        evaluator,
        req,
        extraction.full_name,
        extraction.isdc_2026_urls or [],
    )

    # Add helpful custom info for debugging
    evaluator.add_custom_info(
        info={
            "extracted_full_name": extraction.full_name,
            "education_urls": extraction.education_urls,
            "inspiration4_urls": extraction.inspiration4_urls,
            "nasa_admin_urls": extraction.nasa_admin_urls,
            "isdc_2026_urls": extraction.isdc_2026_urls,
            "isdc_official_2026_url_detected": has_official_isdc_2026_url(extraction.isdc_2026_urls or []),
        },
        info_type="debug_extraction_overview",
    )

    return evaluator.get_summary()