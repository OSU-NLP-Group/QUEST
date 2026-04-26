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
TASK_ID = "dan_driscoll_complete_profile"
TASK_DESCRIPTION = (
    "Dan Driscoll, who serves as the secretary of a U.S. military department, unexpectedly emerged in late November 2025 "
    "as a central figure in the Trump administration's diplomatic efforts to negotiate peace between Russia and Ukraine. "
    "Provide a comprehensive profile of Dan Driscoll that includes:\n\n"
    "1. Official Position: His specific Cabinet-level secretary title, the exact age at which he was confirmed by the U.S. Senate for this position, "
    "the month and year of his confirmation, and what age-related historical distinction he holds for this position\n\n"
    "2. Educational Background: The names of both his undergraduate institution and his law school\n\n"
    "3. Ukraine Peace Negotiation Activities: Documentation of (a) his visit to Ukraine's capital city where he met with the Ukrainian president, "
    "(b) his participation in multilateral discussions held in Geneva, Switzerland, alongside other senior U.S. administration officials, and "
    "(c) his meetings with Russian officials held in Abu Dhabi, United Arab Emirates\n\n"
    "Each factual claim in your response must be supported by reference URLs from reliable, verifiable sources."
)

# Expected canonical claims (used to phrase verification statements)
EXPECTED = {
    "secretary_title": "Secretary of the Army",
    "confirmation_age": "38",
    "confirmation_month_year": "February 2025",
    "age_related_distinction": "youngest-ever Secretary of the Army",
    "undergraduate_institution": "University of North Carolina at Chapel Hill",
    "law_school": "Yale Law School",
    "kyiv_claim": "Dan Driscoll visited Kyiv, Ukraine, and met with Ukrainian President Volodymyr Zelenskyy.",
    "geneva_claim": "Dan Driscoll participated in multilateral discussions in Geneva, Switzerland, alongside Secretary of State Marco Rubio, Steve Witkoff, and Jared Kushner.",
    "abu_dhabi_claim": "Dan Driscoll met with Russian officials in Abu Dhabi, United Arab Emirates.",
}

STRICT_SOURCE_POLICY = (
    "Important: The claim must be supported directly by the provided URL(s). "
    "If no URL is provided, or the URL does not explicitly support the claim, judge the claim as NOT SUPPORTED."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DanDriscollProfileExtraction(BaseModel):
    # Official position and confirmation details
    secretary_title: Optional[str] = None
    secretary_title_sources: List[str] = Field(default_factory=list)

    confirmation_age: Optional[str] = None
    confirmation_age_sources: List[str] = Field(default_factory=list)

    confirmation_month_year: Optional[str] = None
    confirmation_month_year_sources: List[str] = Field(default_factory=list)

    age_related_distinction: Optional[str] = None
    age_related_distinction_sources: List[str] = Field(default_factory=list)

    # Education
    undergraduate_institution: Optional[str] = None
    undergraduate_sources: List[str] = Field(default_factory=list)

    law_school: Optional[str] = None
    law_school_sources: List[str] = Field(default_factory=list)

    # Ukraine peace negotiation activities
    kyiv_visit_and_zelensky_meeting: Optional[str] = None
    kyiv_sources: List[str] = Field(default_factory=list)

    geneva_discussions_with_named_us_officials: Optional[str] = None
    geneva_sources: List[str] = Field(default_factory=list)

    abu_dhabi_talks_with_russian_officials: Optional[str] = None
    abu_dhabi_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_profile() -> str:
    return """
Extract the following information about Dan Driscoll STRICTLY from the provided answer text. Do not invent anything.
For EVERY claim, also extract the URL(s) explicitly cited in the answer that support that specific claim. Accept URLs in plain or markdown format; return them as absolute URLs (include http/https). If the answer does not provide a URL for a claim, return an empty list for that claim's sources.

Fields to extract (use null for missing strings; use [] for missing URL lists):

Official Position and Confirmation:
- secretary_title: The exact secretary title as stated in the answer (e.g., "Secretary of the Army", "United States Army Secretary").
- secretary_title_sources: URL(s) cited that support his title.

- confirmation_age: The age stated for his U.S. Senate confirmation (e.g., "38").
- confirmation_age_sources: URL(s) cited that support the confirmation age.

- confirmation_month_year: The stated month and year of Senate confirmation (e.g., "February 2025").
- confirmation_month_year_sources: URL(s) cited that support the month and year.

- age_related_distinction: The stated age-related historical distinction (e.g., "youngest-ever Army Secretary").
- age_related_distinction_sources: URL(s) cited that support that distinction.

Education:
- undergraduate_institution: The stated undergraduate institution (e.g., "University of North Carolina at Chapel Hill", "UNC-Chapel Hill").
- undergraduate_sources: URL(s) cited that support the undergraduate institution.

- law_school: The stated law school (e.g., "Yale Law School").
- law_school_sources: URL(s) cited that support the law school.

Ukraine Peace Negotiation Activities:
- kyiv_visit_and_zelensky_meeting: A short phrase/summary (from the answer) stating he visited Kyiv and met with President Volodymyr Zelenskyy; null if not stated.
- kyiv_sources: URL(s) cited that support the Kyiv visit + Zelenskyy meeting.

- geneva_discussions_with_named_us_officials: A short phrase/summary (from the answer) stating he participated in multilateral discussions in Geneva alongside Secretary of State Marco Rubio, Steve Witkoff, and Jared Kushner; null if not stated.
- geneva_sources: URL(s) cited that support the Geneva discussions with those named officials.

- abu_dhabi_talks_with_russian_officials: A short phrase/summary (from the answer) stating he met with Russian officials in Abu Dhabi; null if not stated.
- abu_dhabi_sources: URL(s) cited that support the Abu Dhabi meetings with Russian officials.

Return a single JSON object with exactly these fields, filling missing strings as null and missing URL lists as [].
Make sure every URL comes directly from the answer text.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_urls(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    return any(isinstance(u, str) and u.strip() for u in urls)


def _urls_or_empty(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_official_position_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: DanDriscollProfileExtraction,
):
    """
    Build and verify 'Official_Position' parallel node with four critical leaf checks.
    """
    official_node = evaluator.add_parallel(
        id="Official_Position",
        desc="Provides the required official position and confirmation/distinction details.",
        parent=parent_node,
        critical=True,
    )

    # Secretary_Title
    sec_title_node = evaluator.add_leaf(
        id="Secretary_Title",
        desc="States his specific secretary title (Secretary of the Army / United States Army Secretary).",
        parent=official_node,
        critical=True,
    )
    sec_title_claim = "Dan Driscoll serves as the United States Secretary of the Army (Army Secretary)."
    await evaluator.verify(
        claim=sec_title_claim,
        node=sec_title_node,
        sources=_urls_or_empty(extracted.secretary_title_sources),
        additional_instruction=(
            "Use the provided webpage(s) to determine whether Dan Driscoll holds the office of 'Secretary of the Army'. "
            "Equivalent phrases like 'United States Secretary of the Army' or 'Army Secretary' should be treated as the same title. "
            + STRICT_SOURCE_POLICY
        ),
    )

    # Confirmation_Age
    conf_age_node = evaluator.add_leaf(
        id="Confirmation_Age",
        desc="States the exact age at which he was confirmed by the U.S. Senate (38).",
        parent=official_node,
        critical=True,
    )
    conf_age_claim = "Dan Driscoll was confirmed by the U.S. Senate at age 38."
    await evaluator.verify(
        claim=conf_age_claim,
        node=conf_age_node,
        sources=_urls_or_empty(extracted.confirmation_age_sources),
        additional_instruction=(
            "Verify the age at the time of confirmation is 38. "
            "The page may state age explicitly or provide birthdate and confirmation date that imply 38; either is acceptable if clearly supported. "
            + STRICT_SOURCE_POLICY
        ),
    )

    # Confirmation_Month_Year
    conf_month_year_node = evaluator.add_leaf(
        id="Confirmation_Month_Year",
        desc="States the month and year of his Senate confirmation (February 2025).",
        parent=official_node,
        critical=True,
    )
    conf_month_year_claim = "Dan Driscoll was confirmed by the U.S. Senate in February 2025."
    await evaluator.verify(
        claim=conf_month_year_claim,
        node=conf_month_year_node,
        sources=_urls_or_empty(extracted.confirmation_month_year_sources),
        additional_instruction=(
            "Confirm that the month and year of Senate confirmation are 'February 2025'. Minor stylistic variants like 'Feb. 2025' are acceptable. "
            + STRICT_SOURCE_POLICY
        ),
    )

    # Age_Related_Distinction
    age_dist_node = evaluator.add_leaf(
        id="Age_Related_Distinction",
        desc="States the age-related historical distinction for the position (youngest-ever Army Secretary).",
        parent=official_node,
        critical=True,
    )
    age_dist_claim = "Dan Driscoll is the youngest-ever United States Secretary of the Army."
    await evaluator.verify(
        claim=age_dist_claim,
        node=age_dist_node,
        sources=_urls_or_empty(extracted.age_related_distinction_sources),
        additional_instruction=(
            "The page must explicitly indicate that he is the 'youngest-ever' (or 'youngest in history') Secretary of the Army. "
            "Phrases like 'one of the youngest' are NOT sufficient. "
            + STRICT_SOURCE_POLICY
        ),
    )


async def build_education_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: DanDriscollProfileExtraction,
):
    """
    Build and verify 'Educational_Background' parallel node with two critical leaf checks.
    """
    edu_node = evaluator.add_parallel(
        id="Educational_Background",
        desc="Provides the required undergraduate institution and law school.",
        parent=parent_node,
        critical=True,
    )

    # Undergraduate_Institution
    ug_node = evaluator.add_leaf(
        id="Undergraduate_Institution",
        desc="Identifies the University of North Carolina at Chapel Hill as his undergraduate institution.",
        parent=edu_node,
        critical=True,
    )
    ug_claim = (
        "Dan Driscoll's undergraduate institution is the University of North Carolina at Chapel Hill "
        "(UNC-Chapel Hill / UNC Chapel Hill)."
    )
    await evaluator.verify(
        claim=ug_claim,
        node=ug_node,
        sources=_urls_or_empty(extracted.undergraduate_sources),
        additional_instruction=(
            "Treat 'UNC-Chapel Hill' and 'UNC Chapel Hill' as equivalent to 'University of North Carolina at Chapel Hill'. "
            + STRICT_SOURCE_POLICY
        ),
    )

    # Law_School
    law_node = evaluator.add_leaf(
        id="Law_School",
        desc="Identifies Yale Law School as his law school.",
        parent=edu_node,
        critical=True,
    )
    law_claim = "Dan Driscoll attended or graduated from Yale Law School."
    await evaluator.verify(
        claim=law_claim,
        node=law_node,
        sources=_urls_or_empty(extracted.law_school_sources),
        additional_instruction=STRICT_SOURCE_POLICY,
    )


async def build_negotiation_activity_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: DanDriscollProfileExtraction,
):
    """
    Build and verify 'Ukraine_Peace_Negotiation_Activities' parallel node with three critical leaf checks.
    """
    act_node = evaluator.add_parallel(
        id="Ukraine_Peace_Negotiation_Activities",
        desc="Documents the specified negotiation-related activities (Kyiv visit + Zelenskyy meeting, Geneva discussions with named officials, Abu Dhabi talks with Russian officials).",
        parent=parent_node,
        critical=True,
    )

    # Kyiv_Visit_And_Zelenskyy_Meeting
    kyiv_node = evaluator.add_leaf(
        id="Kyiv_Visit_And_Zelenskyy_Meeting",
        desc="Documents his visit to Kyiv, Ukraine, where he met with Ukrainian President Volodymyr Zelenskyy.",
        parent=act_node,
        critical=True,
    )
    kyiv_claim = EXPECTED["kyiv_claim"]
    await evaluator.verify(
        claim=kyiv_claim,
        node=kyiv_node,
        sources=_urls_or_empty(extracted.kyiv_sources),
        additional_instruction=(
            "The page must indicate both elements: (1) a visit to Kyiv, and (2) a meeting with President Volodymyr Zelenskyy "
            "(allow 'Zelensky'/'Zelenskyy' variants). " + STRICT_SOURCE_POLICY
        ),
    )

    # Geneva_Discussions_With_Named_US_Officials
    geneva_node = evaluator.add_leaf(
        id="Geneva_Discussions_With_Named_US_Officials",
        desc="Documents his participation in multilateral discussions in Geneva, Switzerland, alongside Secretary of State Marco Rubio, Steve Witkoff, and Jared Kushner.",
        parent=act_node,
        critical=True,
    )
    geneva_claim = EXPECTED["geneva_claim"]
    await evaluator.verify(
        claim=geneva_claim,
        node=geneva_node,
        sources=_urls_or_empty(extracted.geneva_sources),
        additional_instruction=(
            "The page must indicate multilateral discussions in Geneva involving Dan Driscoll together with ALL of the following: "
            "Secretary of State Marco Rubio, Steve Witkoff, and Jared Kushner. All three names must be present alongside Driscoll. "
            + STRICT_SOURCE_POLICY
        ),
    )

    # Abu_Dhabi_Talks_With_Russian_Officials
    abu_node = evaluator.add_leaf(
        id="Abu_Dhabi_Talks_With_Russian_Officials",
        desc="Documents his meetings/talks with Russian officials in Abu Dhabi, United Arab Emirates.",
        parent=act_node,
        critical=True,
    )
    abu_claim = EXPECTED["abu_dhabi_claim"]
    await evaluator.verify(
        claim=abu_claim,
        node=abu_node,
        sources=_urls_or_empty(extracted.abu_dhabi_sources),
        additional_instruction=(
            "The page must describe Dan Driscoll meeting or holding talks with Russian officials in Abu Dhabi, UAE. "
            + STRICT_SOURCE_POLICY
        ),
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
    Evaluate an answer for the Dan Driscoll comprehensive profile task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as parallel aggregator
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
    extracted: DanDriscollProfileExtraction = await evaluator.extract(
        prompt=prompt_extract_profile(),
        template_class=DanDriscollProfileExtraction,
        extraction_name="dan_driscoll_profile_extraction",
    )

    # Add expected "ground truth-like" targets (for transparency in summary only)
    evaluator.add_ground_truth(
        {
            "expected_title": EXPECTED["secretary_title"],
            "expected_confirmation_age": EXPECTED["confirmation_age"],
            "expected_confirmation_month_year": EXPECTED["confirmation_month_year"],
            "expected_age_distinction": EXPECTED["age_related_distinction"],
            "expected_undergraduate_institution": EXPECTED["undergraduate_institution"],
            "expected_law_school": EXPECTED["law_school"],
            "expected_activity_claims": {
                "kyiv_visit_and_zelensky_meeting": EXPECTED["kyiv_claim"],
                "geneva_discussions_with_named_us_officials": EXPECTED["geneva_claim"],
                "abu_dhabi_talks_with_russian_officials": EXPECTED["abu_dhabi_claim"],
            },
        },
        gt_type="expected_targets",
    )

    # Build the rubric tree: Top-level critical node
    top_node = evaluator.add_parallel(
        id="Dan_Driscoll_Complete_Profile",
        desc="Answer provides the required comprehensive profile (position/confirmation details, education, specified Ukraine peace negotiation activities) and supplies supporting reference URLs.",
        parent=root,
        critical=True,
    )

    # Build subtrees
    await build_official_position_checks(evaluator, top_node, extracted)
    await build_education_checks(evaluator, top_node, extracted)
    await build_negotiation_activity_checks(evaluator, top_node, extracted)

    # Citations_Provided (single critical leaf ensuring each required component has URL(s))
    # This node enforces that every required claim is accompanied by at least one reference URL in the answer.
    citations_ok = all(
        [
            _nonempty_urls(extracted.secretary_title_sources),
            _nonempty_urls(extracted.confirmation_age_sources),
            _nonempty_urls(extracted.confirmation_month_year_sources),
            _nonempty_urls(extracted.age_related_distinction_sources),
            _nonempty_urls(extracted.undergraduate_sources),
            _nonempty_urls(extracted.law_school_sources),
            _nonempty_urls(extracted.kyiv_sources),
            _nonempty_urls(extracted.geneva_sources),
            _nonempty_urls(extracted.abu_dhabi_sources),
        ]
    )
    evaluator.add_custom_node(
        result=citations_ok,
        id="Citations_Provided",
        desc="Provides reference URLs to reliable, verifiable sources supporting each required factual claim/component in the answer.",
        parent=top_node,
        critical=True,
    )

    # Return the evaluation summary
    return evaluator.get_summary()