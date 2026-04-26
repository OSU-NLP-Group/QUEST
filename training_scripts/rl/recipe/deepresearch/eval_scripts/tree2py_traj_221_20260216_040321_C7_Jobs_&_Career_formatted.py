import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "tx_sped_comp_2025_2026"
TASK_DESCRIPTION = (
    "You are a special education teacher with a Master's degree considering employment in a large Texas public school district for the 2025-2026 school year. "
    "Research and identify one such district that is actively recruiting special education teachers, then provide the following comprehensive compensation details: "
    "the district's name and location; base salary for first-year teachers (0 years of experience); regular special education teacher stipend amount; specialized "
    "special education stipend amount for programs such as behavioral support, life skills, or adaptive curriculum; annual Master's degree stipend; annual Doctorate "
    "degree stipend; sign-on bonus amount for newly hired special education teachers; sign-on bonus amount for educational diagnosticians; sign-on bonus amount for "
    "speech-language pathologists; base salary amounts for years 2, 3, 4, and 5 of teaching experience; Texas teacher retirement system contribution percentage; "
    "and official district URLs for both the current teacher salary schedule and the stipend/supplemental pay schedule."
)


# --------------------------------------------------------------------------- #
# Extraction Models                                                            #
# --------------------------------------------------------------------------- #
class CompensationExtraction(BaseModel):
    # District identity
    district_name: Optional[str] = None
    district_location: Optional[str] = None  # e.g., "Houston, TX", "Texas", "Austin, Texas"

    # Core salary and stipend values
    base_salary_year_0: Optional[str] = None
    salary_year_2: Optional[str] = None
    salary_year_3: Optional[str] = None
    salary_year_4: Optional[str] = None
    salary_year_5: Optional[str] = None

    special_ed_stipend_regular: Optional[str] = None
    special_ed_stipend_specialized: Optional[str] = None

    masters_stipend: Optional[str] = None
    doctorate_stipend: Optional[str] = None

    sign_on_bonus_special_ed: Optional[str] = None
    sign_on_bonus_diagnostician: Optional[str] = None
    sign_on_bonus_slp: Optional[str] = None

    retirement_contribution: Optional[str] = None  # Keep as string (e.g., "8%", "8.25%")

    # URLs (evidence)
    salary_schedule_url: Optional[str] = None
    stipend_schedule_url: Optional[str] = None
    hiring_urls: List[str] = Field(default_factory=list)         # careers/jobs pages supporting active recruitment
    benefits_urls: List[str] = Field(default_factory=list)       # benefits or TRS info page URLs
    district_homepage_url: Optional[str] = None                  # official homepage if provided
    extra_official_urls: List[str] = Field(default_factory=list) # any other official district URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                            #
# --------------------------------------------------------------------------- #
def prompt_extract_compensation() -> str:
    return """
Extract the following information exactly as explicitly stated in the provided answer text. Do not infer or add information not present in the answer. Return null for missing fields.

Required fields:
- district_name: Official district name (e.g., "Houston Independent School District" or "Dallas ISD")
- district_location: City and state if available; otherwise any location text as provided in the answer (e.g., "Houston, TX", "Texas")

Core compensation values (record exactly as written, including $ and % if present):
- base_salary_year_0: Base salary for first-year teachers (0 years of experience)
- salary_year_2: Base salary for year 2 (1 year of completed experience)
- salary_year_3: Base salary for year 3 (2 years of completed experience)
- salary_year_4: Base salary for year 4 (3 years of completed experience)
- salary_year_5: Base salary for year 5 (4 years of completed experience)
- special_ed_stipend_regular: Regular special education teacher stipend amount
- special_ed_stipend_specialized: Specialized special education stipend for programs like behavior support, life skills, adaptive curriculum, etc.
- masters_stipend: Annual stipend for holding a Master's degree
- doctorate_stipend: Annual stipend for holding a Doctorate degree
- sign_on_bonus_special_ed: Sign-on bonus amount for newly hired special education teachers
- sign_on_bonus_diagnostician: Sign-on bonus amount for educational diagnosticians
- sign_on_bonus_slp: Sign-on bonus amount for speech-language pathologists
- retirement_contribution: The Texas Teacher Retirement System (TRS) contribution percentage as stated in the answer (include % sign if present)

Official URLs (extract only if explicitly present in the answer; must be full URLs):
- salary_schedule_url: Official district URL for the current teacher salary schedule
- stipend_schedule_url: Official district URL for the current stipend or supplemental pay schedule
- hiring_urls: List of any official district job postings or careers pages that support active recruitment of special education teachers for 2025-2026
- benefits_urls: List of any official district benefits or TRS-related pages cited to support the TRS contribution percentage
- district_homepage_url: Official district home page URL, if present
- extra_official_urls: Any other official district URLs included in the answer that may contain relevant info

Return a single JSON object with these fields.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                             #
# --------------------------------------------------------------------------- #
def _merge_sources(*args: Any) -> List[str]:
    """Merge and deduplicate string or list[str] sources; ignore Nones and empty strings."""
    urls: List[str] = []
    for item in args:
        if not item:
            continue
        if isinstance(item, str):
            s = item.strip()
            if s:
                urls.append(s)
        elif isinstance(item, list):
            for u in item:
                if isinstance(u, str) and u.strip():
                    urls.append(u.strip())
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _safe_value(v: Optional[str], placeholder: str = "<not provided>") -> str:
    return v if (v is not None and str(v).strip() != "") else placeholder


# --------------------------------------------------------------------------- #
# Verification logic                                                           #
# --------------------------------------------------------------------------- #
async def verify_compensation(evaluator: Evaluator, parent_node, info: CompensationExtraction) -> None:
    """
    Build and execute verification leaves according to the rubric.
    Root node is parallel; each leaf corresponds to a specific check.
    """

    # Common source buckets
    all_official_urls = _merge_sources(
        info.salary_schedule_url,
        info.stipend_schedule_url,
        info.hiring_urls,
        info.benefits_urls,
        info.district_homepage_url,
        info.extra_official_urls
    )
    salary_sources = _merge_sources(info.salary_schedule_url, info.extra_official_urls)
    stipend_sources = _merge_sources(info.stipend_schedule_url, info.extra_official_urls)
    hiring_sources = _merge_sources(info.hiring_urls, info.district_homepage_url, info.extra_official_urls)
    retirement_sources = _merge_sources(info.benefits_urls, info.district_homepage_url, info.extra_official_urls)

    # ----------------------- District name ---------------------------------
    node = evaluator.add_leaf(
        id="district_name",
        desc="Provides the accurate name of a large Texas public school district",
        parent=parent_node,
        critical=True
    )
    claim = f"The official district name is '{_safe_value(info.district_name)}'."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=all_official_urls,
        additional_instruction="Verify that the cited official district pages display the same district name (allow common abbreviations like ISD vs. Independent School District)."
    )

    # ----------------------- District location (Texas) ----------------------
    node = evaluator.add_leaf(
        id="district_location",
        desc="Confirms the district is located in Texas",
        parent=parent_node,
        critical=True
    )
    loc_val = _safe_value(info.district_location)
    claim = f"The district is located in Texas (the location provided in the answer is '{loc_val}')."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=all_official_urls,
        additional_instruction="Check the official pages for indications like 'TX', 'Texas', or city names in Texas. Accept if the evidence clearly shows the district is in Texas."
    )

    # ----------------------- District status (large + recruiting) -----------
    node = evaluator.add_leaf(
        id="district_status",
        desc="Verifies the district is a large public school district and is actively recruiting special education teachers for 2025-2026",
        parent=parent_node,
        critical=True
    )
    claim = (
        "This is a large public school district in Texas and it is actively recruiting special education teachers "
        "for the 2025-2026 school year."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=hiring_sources,
        additional_instruction=(
            "Use the careers/jobs pages or official recruiting materials. "
            "To satisfy 'actively recruiting', look for current postings for Special Education teachers, "
            "or language like '2025-2026', '25-26', or 'school year 2025-2026'. "
            "To satisfy 'large', accept explicit statements like 'one of the largest', enrollment counts in the tens of thousands, "
            "or other clear indicators of large district scale on official pages."
        )
    )

    # ----------------------- Base salary (Year 0) ---------------------------
    node = evaluator.add_leaf(
        id="base_salary_year_0",
        desc="Provides accurate base salary for first-year teachers with 0 years of experience",
        parent=parent_node,
        critical=True
    )
    claim = f"The base salary for a first-year teacher (0 years of experience) is {_safe_value(info.base_salary_year_0)}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=salary_sources,
        additional_instruction=(
            "Check the teacher salary schedule (compensation plan). Accept synonyms like 'Step 0', '0 Years', 'new teacher'. "
            "Match the claimed amount to the base teacher scale."
        )
    )

    # ----------------------- Regular SPED stipend --------------------------
    node = evaluator.add_leaf(
        id="special_ed_stipend_regular",
        desc="Provides accurate regular special education teacher stipend amount",
        parent=parent_node,
        critical=True
    )
    claim = f"The regular special education teacher stipend is {_safe_value(info.special_ed_stipend_regular)} per year."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=stipend_sources,
        additional_instruction=(
            "Check the stipend or supplemental pay schedule for 'Special Education stipend' (SPED). "
            "If there are multiple SPED categories, this refers to the standard/general SPED stipend."
        )
    )

    # ----------------------- Specialized SPED stipend -----------------------
    node = evaluator.add_leaf(
        id="special_ed_stipend_specialized",
        desc="Provides accurate specialized special education stipend amount for programs like behavioral support or life skills",
        parent=parent_node,
        critical=True
    )
    claim = (
        f"The specialized special education stipend (e.g., behavior support, life skills, adaptive/structured program) is "
        f"{_safe_value(info.special_ed_stipend_specialized)} per year."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=stipend_sources,
        additional_instruction=(
            "Look for categories like 'Life Skills', 'Behavior Unit', 'Adaptive Curriculum', 'Structured Learning', etc., "
            "and verify the claimed amount matches any such specialized SPED stipend."
        )
    )

    # ----------------------- Master's stipend --------------------------------
    node = evaluator.add_leaf(
        id="masters_stipend",
        desc="Provides accurate annual Master's degree stipend amount",
        parent=parent_node,
        critical=True
    )
    claim = f"The annual Master's degree stipend is {_safe_value(info.masters_stipend)}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=stipend_sources,
        additional_instruction="Verify the stipend/supplemental pay schedule lists a Master's degree stipend matching the claimed amount."
    )

    # ----------------------- Doctorate stipend -------------------------------
    node = evaluator.add_leaf(
        id="doctorate_stipend",
        desc="Provides accurate annual Doctorate degree stipend amount",
        parent=parent_node,
        critical=True
    )
    claim = f"The annual Doctorate degree stipend is {_safe_value(info.doctorate_stipend)}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=stipend_sources,
        additional_instruction="Verify the stipend/supplemental pay schedule lists a Doctorate/Doctoral degree stipend matching the claimed amount."
    )

    # ----------------------- Sign-on bonus SPED teachers ---------------------
    node = evaluator.add_leaf(
        id="sign_on_bonus_special_ed",
        desc="Provides accurate sign-on bonus amount for newly hired special education teachers",
        parent=parent_node,
        critical=True
    )
    claim = f"The sign-on bonus for newly hired special education teachers is {_safe_value(info.sign_on_bonus_special_ed)}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=_merge_sources(stipend_sources, hiring_sources),
        additional_instruction=(
            "Look for 'sign-on bonus', 'hiring/recruitment incentive', or equivalent on stipend schedules or recruiting pages."
        )
    )

    # ----------------------- Sign-on bonus Diagnostician ---------------------
    node = evaluator.add_leaf(
        id="sign_on_bonus_diagnostician",
        desc="Provides accurate sign-on bonus amount for educational diagnosticians",
        parent=parent_node,
        critical=True
    )
    claim = f"The sign-on bonus for educational diagnosticians is {_safe_value(info.sign_on_bonus_diagnostician)}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=_merge_sources(stipend_sources, hiring_sources),
        additional_instruction="Verify a sign-on/recruitment bonus for Educational Diagnostician with the claimed amount."
    )

    # ----------------------- Sign-on bonus SLP -------------------------------
    node = evaluator.add_leaf(
        id="sign_on_bonus_slp",
        desc="Provides accurate sign-on bonus amount for speech-language pathologists",
        parent=parent_node,
        critical=True
    )
    claim = f"The sign-on bonus for speech-language pathologists is {_safe_value(info.sign_on_bonus_slp)}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=_merge_sources(stipend_sources, hiring_sources),
        additional_instruction="Verify a sign-on/recruitment bonus for Speech-Language Pathologists (SLP) with the claimed amount."
    )

    # ----------------------- Salary Year 2 -----------------------------------
    node = evaluator.add_leaf(
        id="salary_year_2",
        desc="Provides accurate base salary for year 2 (1 year of completed experience)",
        parent=parent_node,
        critical=False
    )
    claim = f"The base salary for year 2 (1 year of completed experience) is {_safe_value(info.salary_year_2)}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=salary_sources,
        additional_instruction="On the teacher salary schedule, this corresponds to Step 1 or '1 year'."
    )

    # ----------------------- Salary Year 3 -----------------------------------
    node = evaluator.add_leaf(
        id="salary_year_3",
        desc="Provides accurate base salary for year 3 (2 years of completed experience)",
        parent=parent_node,
        critical=False
    )
    claim = f"The base salary for year 3 (2 years of completed experience) is {_safe_value(info.salary_year_3)}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=salary_sources,
        additional_instruction="On the teacher salary schedule, this corresponds to Step 2 or '2 years'."
    )

    # ----------------------- Salary Year 4 -----------------------------------
    node = evaluator.add_leaf(
        id="salary_year_4",
        desc="Provides accurate base salary for year 4 (3 years of completed experience)",
        parent=parent_node,
        critical=False
    )
    claim = f"The base salary for year 4 (3 years of completed experience) is {_safe_value(info.salary_year_4)}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=salary_sources,
        additional_instruction="On the teacher salary schedule, this corresponds to Step 3 or '3 years'."
    )

    # ----------------------- Salary Year 5 -----------------------------------
    node = evaluator.add_leaf(
        id="salary_year_5",
        desc="Provides accurate base salary for year 5 (4 years of completed experience)",
        parent=parent_node,
        critical=False
    )
    claim = f"The base salary for year 5 (4 years of completed experience) is {_safe_value(info.salary_year_5)}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=salary_sources,
        additional_instruction="On the teacher salary schedule, this corresponds to Step 4 or '4 years'."
    )

    # ----------------------- TRS retirement contribution ---------------------
    node = evaluator.add_leaf(
        id="retirement_contribution",
        desc="Provides accurate Texas teacher retirement system contribution percentage",
        parent=parent_node,
        critical=True
    )
    claim = f"The Texas Teacher Retirement System contribution percentage is {_safe_value(info.retirement_contribution)}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=retirement_sources,
        additional_instruction=(
            "Check the district benefits page or TRS-related official info cited in the answer. "
            "Accept if the page clearly states the same TRS member or employer contribution percentage as claimed."
        )
    )

    # ----------------------- Salary schedule URL validity --------------------
    node = evaluator.add_leaf(
        id="salary_schedule_url",
        desc="Provides valid official district URL for the current teacher salary schedule that can be accessed and verified",
        parent=parent_node,
        critical=True
    )
    claim = (
        "This URL is an official district webpage (or PDF) that contains the current teacher salary schedule for the "
        "2025-2026 school year (or the current compensation plan covering 2025-2026)."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=info.salary_schedule_url,
        additional_instruction=(
            "Verify that the page belongs to the district and clearly contains 'teacher salary schedule', 'salary schedule', or an equivalent compensation table. "
            "Accept '2025-2026', '2025-26', or clear indication that it is the current schedule for the relevant school year."
        )
    )

    # ----------------------- Stipend schedule URL validity -------------------
    node = evaluator.add_leaf(
        id="stipend_schedule_url",
        desc="Provides valid official district URL for the current stipend/supplemental pay schedule that can be accessed and verified",
        parent=parent_node,
        critical=True
    )
    claim = (
        "This URL is an official district webpage (or PDF) that contains the current stipend or supplemental pay schedule "
        "for the 2025-2026 school year."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=info.stipend_schedule_url,
        additional_instruction=(
            "Verify that the page belongs to the district and clearly lists stipends/supplemental pay (including SPED, degree stipends, sign-on bonuses if applicable). "
            "Accept '2025-2026', '2025-26', or clear indication that it is the current schedule."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                  #
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
    Entry point for evaluating an agent's answer against the rubric using the Mind2Web2 framework.
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
        default_model=model
    )

    # Extraction
    comp_info: CompensationExtraction = await evaluator.extract(
        prompt=prompt_extract_compensation(),
        template_class=CompensationExtraction,
        extraction_name="compensation_extraction"
    )

    # Build and run verification leaves
    await verify_compensation(evaluator, root, comp_info)

    # Return evaluation summary
    return evaluator.get_summary()