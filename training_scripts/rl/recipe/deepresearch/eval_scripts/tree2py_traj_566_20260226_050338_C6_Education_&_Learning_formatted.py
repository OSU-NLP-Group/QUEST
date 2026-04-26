import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "us_coaching_2025_2026"
TASK_DESCRIPTION = (
    "In the 2025-2026 academic year, several educational institutions in the United States experienced significant "
    "coaching transitions and organizational milestones. Identify three specific institutions that meet all of the following criteria:\n\n"
    "Institution 1: A Big Ten Conference university that:\n"
    "- Hired a new head football coach in December 2025\n"
    "- The new head coach previously served as head coach at Northwestern University for exactly 17 years\n"
    "- During his tenure at Northwestern, he won exactly 2 Big Ten West Division titles\n"
    "- The same university also hired an offensive coordinator in January 2026\n"
    "- This offensive coordinator served as Alabama's offensive coordinator in 2024\n"
    "- This offensive coordinator played quarterback at the University of Michigan and earned letters in 2008 and 2009\n\n"
    "Institution 2: A high school in Georgia that:\n"
    "- Is located in Loganville, in Gwinnett County\n"
    "- Won a GHSA state football championship in 2024\n"
    "- This 2024 championship was the school's 4th total state championship in football history\n"
    "- Promoted its defensive coordinator to head coach in January 2026\n\n"
    "Institution 3: A school district in Texas that:\n"
    "- Is located in Plano\n"
    "- Contains exactly 77 schools in total\n"
    "- Has exactly 3 senior high schools\n\n"
    "For each institution, provide the institution's full name and reference URLs that verify the key facts stated above."
)


# --------------------------- Data Models ------------------------------------ #
class Institution1Extraction(BaseModel):
    full_name: Optional[str] = None
    head_coach_name: Optional[str] = None
    head_coach_hire_date: Optional[str] = None
    oc_name: Optional[str] = None
    oc_hire_date: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class Institution2Extraction(BaseModel):
    full_name: Optional[str] = None
    defensive_coordinator_name: Optional[str] = None
    promotion_date: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class Institution3Extraction(BaseModel):
    full_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class InstitutionsExtraction(BaseModel):
    institution1: Optional[Institution1Extraction] = None
    institution2: Optional[Institution2Extraction] = None
    institution3: Optional[Institution3Extraction] = None


# --------------------------- Extraction Prompt ------------------------------ #
def prompt_extract_institutions() -> str:
    return """
Extract, from the provided answer text, the information for three institutions as structured fields.

For Institution 1 (Big Ten university), extract:
- full_name: the university's full official name.
- head_coach_name: the name of the newly hired head football coach (if provided).
- head_coach_hire_date: the date string of the head coach hire as written in the answer (e.g., 'December 2025').
- oc_name: the name of the offensive coordinator hired (if provided).
- oc_hire_date: the date string of the OC hire as written in the answer (e.g., 'January 2026').
- reference_urls: all URLs the answer cites to support Institution 1’s facts (membership, hires, tenure, titles, OC background). Include every relevant URL explicitly present in the answer.

For Institution 2 (Georgia high school), extract:
- full_name: the high school's full official name.
- defensive_coordinator_name: the DC’s name if mentioned.
- promotion_date: the date string of the DC promotion as written (e.g., 'January 2026').
- reference_urls: all URLs the answer cites to support location, 2024 GHSA title, fourth total title, and DC promotion.

For Institution 3 (Texas school district), extract:
- full_name: the district's full official name.
- reference_urls: all URLs the answer cites to support location (Plano), total schools count, and number of senior high schools.

General URL extraction rules:
- Only include valid, explicit URLs contained in the answer (plain or in markdown).
- Do not invent or infer URLs.
- If no URLs are provided for a section, return an empty list for reference_urls.

Return a single JSON object with fields institution1, institution2, and institution3 holding the respective objects. Use null for any missing field.
    """


# --------------------------- Helper ----------------------------------------- #
def _safe_name(name: Optional[str], fallback: str) -> str:
    if name and str(name).strip():
        return name.strip()
    return fallback


# --------------------------- Verification Builders -------------------------- #
async def verify_institution_1(evaluator: Evaluator, parent, info: Institution1Extraction) -> None:
    node = evaluator.add_parallel(
        id="Institution1_BigTenUniversity",
        desc="Institution 1: A Big Ten Conference university meeting all specified head coach and offensive coordinator criteria.",
        parent=parent,
        critical=False
    )

    # Existence checks
    evaluator.add_custom_node(
        result=bool(info and info.full_name and info.full_name.strip()),
        id="inst1_full_name_provided",
        desc="Provides the institution's full name.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info and info.reference_urls and len(info.reference_urls) > 0),
        id="inst1_reference_urls_provided",
        desc="Provides reference URLs that collectively verify all key facts listed for Institution 1.",
        parent=node,
        critical=True
    )

    uni_name = _safe_name(info.full_name, "the university")
    head_coach = _safe_name(info.head_coach_name, "the new head coach")
    oc_name = _safe_name(info.oc_name, "the offensive coordinator")
    urls = info.reference_urls if info and info.reference_urls else []

    # Create leaf nodes
    big_ten_node = evaluator.add_leaf(
        id="inst1_is_big_ten",
        desc="The institution is a Big Ten Conference university.",
        parent=node,
        critical=True
    )
    hc_dec2025_node = evaluator.add_leaf(
        id="inst1_hc_dec2025",
        desc="Hired a new head football coach in December 2025.",
        parent=node,
        critical=True
    )
    hc_nu_17yrs_node = evaluator.add_leaf(
        id="inst1_hc_nu_17_years",
        desc="The new head coach previously served as head coach at Northwestern University for exactly 17 years.",
        parent=node,
        critical=True
    )
    hc_nu_west2_node = evaluator.add_leaf(
        id="inst1_hc_nu_2_west_titles",
        desc="During his tenure at Northwestern, the coach won exactly 2 Big Ten West Division titles.",
        parent=node,
        critical=True
    )
    oc_jan2026_node = evaluator.add_leaf(
        id="inst1_oc_jan2026",
        desc="Hired an offensive coordinator in January 2026.",
        parent=node,
        critical=True
    )
    oc_bama_2024_node = evaluator.add_leaf(
        id="inst1_oc_bama_2024",
        desc="That offensive coordinator served as Alabama's offensive coordinator in 2024.",
        parent=node,
        critical=True
    )
    oc_um_letters_node = evaluator.add_leaf(
        id="inst1_oc_um_letters_2008_2009",
        desc="That offensive coordinator played quarterback at the University of Michigan and earned letters in 2008 and 2009.",
        parent=node,
        critical=True
    )

    claims_and_sources = [
        (
            f"{uni_name} is a member of the Big Ten Conference.",
            urls,
            big_ten_node,
            "Verify that the institution is a Big Ten member. Allow synonyms like 'B1G'. Prefer official athletic sites, Big Ten pages, or credible news."
        ),
        (
            f"In December 2025, {uni_name} hired {head_coach} as its head football coach.",
            urls,
            hc_dec2025_node,
            "Confirm that the hiring/announcement occurred in December 2025. Accept phrasing like 'announced in Dec. 2025' or 'agreed in Dec. 2025'."
        ),
        (
            f"{head_coach} previously served as Northwestern University's head football coach for exactly 17 years.",
            urls,
            hc_nu_17yrs_node,
            "Check biographical or news sources confirming 17 years (or '17 seasons') as Northwestern's head coach."
        ),
        (
            f"While at Northwestern, {head_coach} won exactly 2 Big Ten West Division titles.",
            urls,
            hc_nu_west2_node,
            "Verify that the number of Big Ten West Division titles is exactly 2. Accept 'two' as equivalent to '2'."
        ),
        (
            f"In January 2026, {uni_name} hired {oc_name} as offensive coordinator.",
            urls,
            oc_jan2026_node,
            "Verify an official or reputable report indicating an OC hire in January 2026 for this institution."
        ),
        (
            f"{oc_name} served as Alabama's offensive coordinator in 2024.",
            urls,
            oc_bama_2024_node,
            "Confirm that this coach served as Alabama's offensive coordinator (or co-offensive coordinator) in 2024."
        ),
        (
            f"{oc_name} played quarterback at the University of Michigan and earned letters in 2008 and 2009.",
            urls,
            oc_um_letters_node,
            "Verify Michigan player biography or official records indicating QB position and letters (varsity letters) in 2008 and 2009."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


async def verify_institution_2(evaluator: Evaluator, parent, info: Institution2Extraction) -> None:
    node = evaluator.add_parallel(
        id="Institution2_GeorgiaHighSchool",
        desc="Institution 2: A Georgia high school meeting all stated location, championship, and coaching-promotion criteria.",
        parent=parent,
        critical=False
    )

    # Existence checks
    evaluator.add_custom_node(
        result=bool(info and info.full_name and info.full_name.strip()),
        id="inst2_full_name_provided",
        desc="Provides the institution's full name.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info and info.reference_urls and len(info.reference_urls) > 0),
        id="inst2_reference_urls_provided",
        desc="Provides reference URLs that collectively verify all key facts listed for Institution 2.",
        parent=node,
        critical=True
    )

    school_name = _safe_name(info.full_name, "the high school")
    dc_name = _safe_name(info.defensive_coordinator_name, "its defensive coordinator")
    urls = info.reference_urls if info and info.reference_urls else []

    loc_node = evaluator.add_leaf(
        id="inst2_loc_loganville_gwinnett",
        desc="The high school is located in Loganville, in Gwinnett County.",
        parent=node,
        critical=True
    )
    ghsa_2024_node = evaluator.add_leaf(
        id="inst2_ghsa_2024_title",
        desc="Won a GHSA state football championship in 2024.",
        parent=node,
        critical=True
    )
    fourth_total_node = evaluator.add_leaf(
        id="inst2_fourth_total_title",
        desc="The 2024 championship was the school's 4th total state football championship.",
        parent=node,
        critical=True
    )
    dc_promoted_node = evaluator.add_leaf(
        id="inst2_dc_promoted_jan2026",
        desc="Promoted its defensive coordinator to head coach in January 2026.",
        parent=node,
        critical=True
    )

    claims_and_sources = [
        (
            f"{school_name} is located in Loganville, in Gwinnett County, Georgia.",
            urls,
            loc_node,
            "Confirm the school's city is Loganville and the county is Gwinnett in Georgia. Official school pages or district profiles preferred."
        ),
        (
            f"In 2024, {school_name} won a GHSA state football championship.",
            urls,
            ghsa_2024_node,
            "Verify via GHSA records, school announcements, or credible news that the school won the 2024 state football title."
        ),
        (
            f"The 2024 state football championship was {school_name}'s 4th in school history.",
            urls,
            fourth_total_node,
            "Confirm the 2024 title marked the school's fourth overall state football championship. Accept 'fourth' as '4th'."
        ),
        (
            f"In January 2026, {school_name} promoted {dc_name} to head coach.",
            urls,
            dc_promoted_node,
            "Verify an official announcement or reputable report indicating the defensive coordinator was promoted to head coach in January 2026."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


async def verify_institution_3(evaluator: Evaluator, parent, info: Institution3Extraction) -> None:
    node = evaluator.add_parallel(
        id="Institution3_TexasSchoolDistrict",
        desc="Institution 3: A Texas school district meeting all stated location and school-count criteria.",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(info and info.full_name and info.full_name.strip()),
        id="inst3_full_name_provided",
        desc="Provides the institution's full name.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info and info.reference_urls and len(info.reference_urls) > 0),
        id="inst3_reference_urls_provided",
        desc="Provides reference URLs that collectively verify all key facts listed for Institution 3.",
        parent=node,
        critical=True
    )

    district_name = _safe_name(info.full_name, "the school district")
    urls = info.reference_urls if info and info.reference_urls else []

    loc_plano_node = evaluator.add_leaf(
        id="inst3_loc_plano",
        desc="The school district is located in Plano.",
        parent=node,
        critical=True
    )
    total_77_node = evaluator.add_leaf(
        id="inst3_total_77_schools",
        desc="Contains exactly 77 schools in total.",
        parent=node,
        critical=True
    )
    senior3_node = evaluator.add_leaf(
        id="inst3_three_senior_hs",
        desc="Has exactly 3 senior high schools.",
        parent=node,
        critical=True
    )

    claims_and_sources = [
        (
            f"{district_name} is located in Plano, Texas.",
            urls,
            loc_plano_node,
            "Verify the district headquarters or primary location is Plano, Texas. Official district 'About' pages preferred."
        ),
        (
            f"{district_name} contains exactly 77 schools in total.",
            urls,
            total_77_node,
            "Confirm the official total number of schools is 77 (accept exact phrasing or an official count list)."
        ),
        (
            f"{district_name} has exactly 3 senior high schools.",
            urls,
            senior3_node,
            "Verify that the number of 'senior high schools' in the district is exactly three."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


# --------------------------- Main Entry ------------------------------------- #
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
        strategy=AggregationStrategy.PARALLEL,
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_institutions(),
        template_class=InstitutionsExtraction,
        extraction_name="institutions_extraction"
    )

    inst1 = extracted.institution1 or Institution1Extraction()
    inst2 = extracted.institution2 or Institution2Extraction()
    inst3 = extracted.institution3 or Institution3Extraction()

    await verify_institution_1(evaluator, root, inst1)
    await verify_institution_2(evaluator, root, inst2)
    await verify_institution_3(evaluator, root, inst3)

    return evaluator.get_summary()