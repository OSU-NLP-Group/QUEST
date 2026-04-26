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
TASK_ID = "higher_ed_admin_jobs_2026"
TASK_DESCRIPTION = """
You are researching career pathways in higher education administration. Identify four current job postings (as of February 2026) from accredited U.S. colleges or universities, with one position from each of the following four hierarchical levels:

1. Entry-Level Position: A position with a title indicating entry-level status (e.g., Coordinator, Advisor, Assistant, Specialist) that requires a Bachelor's degree and 2-5 years of relevant experience.

2. Mid-Level Director Position: A position with "Director" in the title (not Assistant/Associate Director, and not VP/Dean level) that requires a Master's degree in a relevant field and 5-8 years of progressive experience in higher education.

3. Senior-Level Position: A position titled Associate Dean, Assistant Dean, Associate Vice President, or Assistant Vice President that requires a Master's degree minimum (with doctorate preferred or required) and 8-10 years of progressive leadership experience in higher education.

4. Executive-Level Position: A position titled Dean (not Associate/Assistant Dean), Vice President (not Associate/Assistant VP), or Provost that requires a doctorate degree and a minimum of 10 years of progressive administrative and leadership experience in higher education.

For each position, provide:
- The institution name and a URL confirming it is an accredited U.S. college or university
- The complete position title
- The functional area (e.g., Student Affairs, Academic Affairs, Career Services, Enrollment Management, etc.)
- A direct URL to the job posting from HigherEdJobs, Chronicle of Higher Education, Inside Higher Ed, or the institution's official career site
- The stated education requirement with a URL reference from the job posting
- The stated experience requirement with a URL reference from the job posting
- If disclosed, the salary or salary range

All job postings must be current (dated in February 2026 or explicitly accepting applications). Each position must be from a different institution and represent a different functional area of higher education administration.
"""

CURRENT_WINDOW = "February 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PositionData(BaseModel):
    institution_name: Optional[str] = None
    institution_ref_url: Optional[str] = None  # accreditation or official site URL
    position_title: Optional[str] = None
    functional_area: Optional[str] = None
    job_posting_url: Optional[str] = None
    education_requirement_text: Optional[str] = None
    education_requirement_url: Optional[str] = None
    experience_requirement_text: Optional[str] = None
    experience_requirement_url: Optional[str] = None
    salary_text: Optional[str] = None
    posting_date_text: Optional[str] = None


class AllPositionsExtraction(BaseModel):
    entry: Optional[PositionData] = None
    mid: Optional[PositionData] = None
    senior: Optional[PositionData] = None
    executive: Optional[PositionData] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    Extract structured details for four higher education administrative job postings from the answer text.
    There must be exactly one position for each level: entry, mid, senior, executive.

    For each level, extract the following fields:
    - institution_name: The name of the hiring accredited U.S. college/university
    - institution_ref_url: A URL that is either the institution’s official website (.edu or official domain) OR an accreditation confirmation page (e.g., regional accreditors or U.S. Dept of Education database)
    - position_title: The complete job title as written
    - functional_area: The functional area (e.g., Student Affairs, Academic Affairs, Career Services, Enrollment Management, Registrar, Athletics/Compliance, etc.)
    - job_posting_url: Direct URL to the actual job posting (HigherEdJobs, Chronicle of Higher Education, Inside Higher Ed, OR the institution’s official careers site)
    - education_requirement_text: The stated educational requirement text, exactly as presented in the answer
    - education_requirement_url: A URL from the job posting that shows the education requirement (often same as job_posting_url)
    - experience_requirement_text: The stated experience requirement text, exactly as presented in the answer
    - experience_requirement_url: A URL from the job posting that shows the experience requirement (often same as job_posting_url)
    - salary_text: If disclosed, the salary or salary range as written; otherwise null
    - posting_date_text: The posting date text or explicit note that applications are currently being accepted; otherwise null

    Return a JSON object with fields: entry, mid, senior, executive; each is an object with the above fields.
    If any field is missing in the answer, set it to null.
    Do not fabricate URLs; only include URLs explicitly present in the answer text.
    Ensure that the job_posting_url is a direct link to the posting (not a generic site homepage).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip().lower()
    return u.startswith("http://") or u.startswith("https://")


def _contains_any(url: str, substrings: List[str]) -> bool:
    try:
        low = url.lower()
    except Exception:
        return False
    return any(s in low for s in substrings)


def _normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return "".join(ch for ch in name.lower().strip() if ch.isalnum() or ch.isspace())


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_institution_verification(
    evaluator: Evaluator,
    parent,
    prefix: str,
    data: PositionData,
) -> Dict[str, Any]:
    """
    Build institution verification sequential group:
    - URL reference presence/validity (custom)
    - Institution exists / matches page evidence (verify by URL)
    """
    node = evaluator.add_sequential(
        id=f"{prefix}_institution_verification",
        desc="The hiring institution is a verifiable accredited U.S. college or university",
        parent=parent,
        critical=True,
    )

    # URL reference presence/validity
    accred_domains = ["hlcommission.org", "sacscoc.org", "msche.org", "wscuc.org", "nwccu.org", "neche.org", "ed.gov", "ope.ed.gov"]
    url_valid = _is_valid_url(data.institution_ref_url) and (
        _contains_any(data.institution_ref_url or "", accred_domains) or _contains_any(data.institution_ref_url or "", [".edu"])
    )
    url_ref_leaf = evaluator.add_custom_node(
        result=url_valid,
        id=f"{prefix}_institution_url_reference",
        desc="A valid URL reference confirming the institution's accreditation or official website is provided",
        parent=node,
        critical=True,
    )

    # Institution exists - verify by URL, depends on URL presence
    inst_exists_leaf = evaluator.add_leaf(
        id=f"{prefix}_institution_exists",
        desc="The institution name corresponds to an actual accredited higher education institution",
        parent=node,
        critical=True,
    )
    claim_exists = (
        f"This webpage is the official site or an accreditation listing for '{data.institution_name}', "
        f"which is an accredited U.S. college or university."
    )
    await evaluator.verify(
        claim=claim_exists,
        node=inst_exists_leaf,
        sources=data.institution_ref_url,
        additional_instruction=(
            "Accept as valid if the URL is the institution's official .edu site or an accreditation source "
            "(e.g., regional accreditors, U.S. Dept of Education database). "
            "Confirm that the institution name on the page matches or clearly refers to the provided institution."
        ),
    )

    return {"node": node, "url_ref_leaf": url_ref_leaf, "inst_exists_leaf": inst_exists_leaf}


async def build_job_posting_verification(
    evaluator: Evaluator,
    parent,
    prefix: str,
    data: PositionData,
) -> Dict[str, Any]:
    """
    Build job posting verification sequential group:
    - URL reference is allowed source / official careers (custom presence+domain check)
    - Posting current in February 2026 or accepting applications (verify by URL)
    """
    node = evaluator.add_sequential(
        id=f"{prefix}_job_posting_verification",
        desc="The position is from a current job posting (as of February 2026) on a verified higher education job board or university career site",
        parent=parent,
        critical=True,
    )

    # Posting URL - presence and allowed source (.edu or recognized boards)
    allowed_board_substrings = ["higheredjobs.com", "insidehighered.com", "jobs.chronicle.com", "chronicle.com", ".edu"]
    posting_url_ok = _is_valid_url(data.job_posting_url) and _contains_any(data.job_posting_url or "", allowed_board_substrings)
    posting_url_leaf = evaluator.add_custom_node(
        result=posting_url_ok,
        id=f"{prefix}_posting_url_reference",
        desc="A direct URL to the actual job posting is provided from HigherEdJobs, Chronicle of Higher Education, Inside Higher Ed, or the institution's official career site",
        parent=node,
        critical=True,
    )

    # Posting current (Feb 2026 or explicitly accepting applications)
    posting_current_leaf = evaluator.add_leaf(
        id=f"{prefix}_posting_current",
        desc="The job posting is dated within February 2026 or explicitly stated as currently accepting applications",
        parent=node,
        critical=True,
    )
    claim_current = (
        "This job posting is dated within February 2026 or clearly indicates that applications are currently being accepted "
        "(e.g., 'Currently accepting applications', 'Open until filled', or a listed posting/update date in Feb 2026)."
    )
    await evaluator.verify(
        claim=claim_current,
        node=posting_current_leaf,
        sources=data.job_posting_url,
        additional_instruction=(
            "Check the posting page for a visible date within February 2026 or explicit language indicating current application acceptance. "
            "If only a 'posted' or 'updated' date is shown, it must fall within February 2026."
        ),
    )

    return {"node": node, "posting_url_leaf": posting_url_leaf, "posting_current_leaf": posting_current_leaf}


async def build_title_and_area_verification(
    evaluator: Evaluator,
    parent,
    prefix: str,
    data: PositionData,
    level: str,
    prereq_nodes: Optional[List] = None,
) -> Dict[str, Any]:
    """
    Build parallel group for title level appropriateness and functional area.
    """
    node = evaluator.add_parallel(
        id=f"{prefix}_position_title",
        desc=f"The position title accurately reflects a {level} role in higher education",
        parent=parent,
        critical=True,
    )

    # Title level appropriate
    title_leaf = evaluator.add_leaf(
        id=f"{prefix}_title_level_appropriate",
        desc={
            "entry": "The title indicates an entry-level position (e.g., Coordinator, Advisor, Assistant, Specialist) rather than director or executive level",
            "mid": "The title explicitly includes 'Director' (not Assistant/Associate Director, and not Vice President or Dean level)",
            "senior": "The title includes 'Associate Dean', 'Assistant Dean', 'Associate Vice President', or 'Assistant Vice President'",
            "executive": "The title includes 'Dean' (not Associate/Assistant Dean), 'Vice President' (not Associate/Assistant VP), or 'Provost'",
        }[level],
        parent=node,
        critical=True,
    )

    if level == "entry":
        claim_title = (
            f"The position title on this posting is '{data.position_title}'. "
            f"It indicates an entry-level administrative role (e.g., Coordinator, Advisor, Assistant, Specialist) "
            f"and does not indicate Director, Dean, Vice President, or Provost."
        )
    elif level == "mid":
        claim_title = (
            f"The position title on this posting is '{data.position_title}'. "
            f"It includes the word 'Director' and is not 'Assistant Director' or 'Associate Director', "
            f"and it is not a Vice President or Dean-level role."
        )
    elif level == "senior":
        claim_title = (
            f"The position title on this posting is '{data.position_title}'. "
            f"It is one of: Associate Dean, Assistant Dean, Associate Vice President, or Assistant Vice President (or clear equivalent)."
        )
    else:  # executive
        claim_title = (
            f"The position title on this posting is '{data.position_title}'. "
            f"It is one of: Dean (not Associate/Assistant Dean), Vice President (not Associate/Assistant Vice President), or Provost."
        )

    await evaluator.verify(
        claim=claim_title,
        node=title_leaf,
        sources=data.job_posting_url,
        additional_instruction="Use the job posting page to confirm the exact position title and its level.",
        extra_prerequisites=prereq_nodes or [],
    )

    # Functional area
    area_leaf = evaluator.add_leaf(
        id=f"{prefix}_title_functional_area",
        desc="The title identifies a recognized functional area of higher education administration",
        parent=node,
        critical=True,
    )
    claim_area = (
        f"The job posting indicates that this position belongs to the functional area '{data.functional_area}', "
        f"such as Academic Affairs, Student Affairs, Enrollment Management, Career Services, Registrar, or similar."
    )
    await evaluator.verify(
        claim=claim_area,
        node=area_leaf,
        sources=data.job_posting_url,
        additional_instruction=(
            "Look for signals in the posting (department, division, context, responsibilities) that indicate the functional area. "
            "Allow reasonable synonyms or closely related sub-areas."
        ),
        extra_prerequisites=prereq_nodes or [],
    )

    return {"node": node, "title_leaf": title_leaf, "area_leaf": area_leaf}


async def build_education_verification(
    evaluator: Evaluator,
    parent,
    prefix: str,
    data: PositionData,
    level: str,
) -> Dict[str, Any]:
    """
    Build education requirement sequential group. Order: URL presence (custom) -> requirement verification (by URL).
    """
    node = evaluator.add_sequential(
        id=f"{prefix}_education_requirement",
        desc=f"The stated education requirement meets {level}-level standards",
        parent=parent,
        critical=True,
    )

    # URL presence
    edu_url_ok = _is_valid_url(data.education_requirement_url)
    edu_url_leaf = evaluator.add_custom_node(
        result=edu_url_ok,
        id=f"{prefix}_education_url_reference",
        desc="A URL reference from the job posting confirming the education requirement is provided",
        parent=node,
        critical=True,
    )

    # Requirement verification
    edu_leaf = evaluator.add_leaf(
        id=f"{prefix}_education_minimum",
        desc={
            "entry": "The position requires a minimum of a Bachelor's degree",
            "mid": "The position requires a Master's degree in a relevant field (Higher Education Administration, Student Affairs, Counseling, or related field)",
            "senior": "The position requires a Master's degree at minimum, with doctorate preferred or required",
            "executive": "The position requires a doctorate degree (Ph.D., Ed.D., or terminal degree in relevant field)",
        }[level],
        parent=node,
        critical=True,
    )

    if level == "entry":
        claim = "The posting states that the minimum education requirement is a Bachelor's degree."
    elif level == "mid":
        claim = "The posting states that a Master's degree in a relevant field is required."
    elif level == "senior":
        claim = "The posting states that at least a Master's degree is required, and a doctorate is preferred or required."
    else:  # executive
        claim = "The posting states that a doctorate degree (e.g., Ph.D., Ed.D., or terminal degree) is required."

    await evaluator.verify(
        claim=claim,
        node=edu_leaf,
        sources=data.education_requirement_url,
        additional_instruction="Verify the minimum degree requirement as stated on the posting page.",
    )

    return {"node": node, "edu_url_leaf": edu_url_leaf, "edu_leaf": edu_leaf}


async def build_experience_verification(
    evaluator: Evaluator,
    parent,
    prefix: str,
    data: PositionData,
    level: str,
) -> Dict[str, Any]:
    """
    Build experience requirement sequential group. Order: URL presence (custom) -> requirement verification (by URL).
    """
    node = evaluator.add_sequential(
        id=f"{prefix}_experience_requirement",
        desc=f"The stated experience requirement aligns with {level}-level expectations",
        parent=parent,
        critical=True,
    )

    # URL presence
    exp_url_ok = _is_valid_url(data.experience_requirement_url)
    exp_url_leaf = evaluator.add_custom_node(
        result=exp_url_ok,
        id=f"{prefix}_experience_url_reference",
        desc="A URL reference from the job posting confirming the experience requirement is provided",
        parent=node,
        critical=True,
    )

    # Requirement verification
    exp_leaf = evaluator.add_leaf(
        id=f"{prefix}_experience_range",
        desc={
            "entry": "The position requires between 2 and 5 years of relevant experience, or indicates entry-level status",
            "mid": "The position requires between 5 and 8 years of progressive or relevant experience in higher education",
            "senior": "The position requires between 8 and 10 years of progressive leadership experience in higher education",
            "executive": "The position requires a minimum of 10 years of progressive administrative and leadership experience in higher education",
        }[level],
        parent=node,
        critical=True,
    )

    if level == "entry":
        claim = "The posting requires between 2 and 5 years of relevant experience, or indicates the position is entry-level."
    elif level == "mid":
        claim = "The posting requires between 5 and 8 years of progressive experience in higher education."
    elif level == "senior":
        claim = "The posting requires between 8 and 10 years of progressive leadership experience in higher education."
    else:  # executive
        claim = "The posting requires a minimum of 10 years of progressive administrative and leadership experience in higher education."

    await evaluator.verify(
        claim=claim,
        node=exp_leaf,
        sources=data.experience_requirement_url,
        additional_instruction="Check the required experience as stated in the job posting.",
    )

    return {"node": node, "exp_url_leaf": exp_url_leaf, "exp_leaf": exp_leaf}


async def build_entry_salary_verification(
    evaluator: Evaluator,
    parent,
    prefix: str,
    data: PositionData,
) -> Dict[str, Any]:
    """
    Build entry-level salary information (non-critical, parallel).
    """
    node = evaluator.add_parallel(
        id=f"{prefix}_salary_information",
        desc="Salary information is provided and falls within the entry-level range",
        parent=parent,
        critical=False,
    )

    disclosed = bool(data.salary_text and data.salary_text.strip())
    disclosed_leaf = evaluator.add_custom_node(
        result=disclosed,
        id=f"{prefix}_salary_disclosed",
        desc="The job posting includes salary information (range or specific amount)",
        parent=node,
        critical=False,
    )

    range_leaf = evaluator.add_leaf(
        id=f"{prefix}_salary_range_appropriate",
        desc="If disclosed, the salary falls within the typical entry-level range ($36,000-$50,000)",
        parent=node,
        critical=False,
    )
    claim = "The disclosed salary falls within $36,000 to $50,000 (or an hourly equivalent consistent with this range)."
    await evaluator.verify(
        claim=claim,
        node=range_leaf,
        sources=data.job_posting_url,
        additional_instruction=(
            "If salary is hourly, convert approximately using 2,080 hours/year and allow reasonable rounding. "
            "If no salary is disclosed on the posting, this should not pass."
        ),
        extra_prerequisites=[disclosed_leaf],
    )

    return {"node": node, "disclosed_leaf": disclosed_leaf, "range_leaf": range_leaf}


async def build_mid_supervisory_verification(
    evaluator: Evaluator,
    parent,
    prefix: str,
    data: PositionData,
) -> Dict[str, Any]:
    """
    Build mid-level supervisory expectations (non-critical, parallel).
    """
    node = evaluator.add_parallel(
        id=f"{prefix}_supervisory_requirement",
        desc="The position includes supervisory or management responsibilities typical of director-level roles",
        parent=parent,
        critical=False,
    )

    stated_leaf = evaluator.add_leaf(
        id=f"{prefix}_supervisory_stated",
        desc="The job posting explicitly mentions supervisory, management, or leadership responsibilities",
        parent=node,
        critical=False,
    )
    claim_stated = "The posting explicitly mentions supervisory, management, or leadership responsibilities."
    await evaluator.verify(
        claim=claim_stated,
        node=stated_leaf,
        sources=data.job_posting_url,
        additional_instruction="Look for language like 'supervise', 'manage', 'lead', 'directs staff', or similar.",
    )

    prior_exp_leaf = evaluator.add_leaf(
        id=f"{prefix}_supervisory_experience",
        desc="The position requires prior supervisory experience (typically 2-4 years)",
        parent=node,
        critical=False,
    )
    claim_prior = "The posting requires prior supervisory experience (e.g., 2–4 years or similar)."
    await evaluator.verify(
        claim=claim_prior,
        node=prior_exp_leaf,
        sources=data.job_posting_url,
        additional_instruction="Check minimum qualifications or preferred qualifications sections for supervisory experience requirements.",
    )

    return {"node": node, "stated_leaf": stated_leaf, "prior_exp_leaf": prior_exp_leaf}


async def build_senior_leadership_verification(
    evaluator: Evaluator,
    parent,
    prefix: str,
    data: PositionData,
) -> Dict[str, Any]:
    """
    Build senior-level leadership emphasis (non-critical, parallel).
    """
    node = evaluator.add_parallel(
        id=f"{prefix}_leadership_requirement",
        desc="The position emphasizes progressive leadership and administrative experience",
        parent=parent,
        critical=False,
    )

    prog_leaf = evaluator.add_leaf(
        id=f"{prefix}_progressive_leadership",
        desc="The job posting explicitly requires 'progressive' or 'progressively responsible' leadership experience",
        parent=node,
        critical=False,
    )
    claim_prog = "The posting explicitly requires 'progressive' or 'progressively responsible' leadership experience."
    await evaluator.verify(
        claim=claim_prog,
        node=prog_leaf,
        sources=data.job_posting_url,
        additional_instruction="Look for phrases like 'progressively responsible leadership experience'.",
    )

    scope_leaf = evaluator.add_leaf(
        id=f"{prefix}_administrative_scope",
        desc="The position includes oversight of departments, programs, or significant institutional functions",
        parent=node,
        critical=False,
    )
    claim_scope = "The position includes oversight of departments, programs, or significant institutional functions."
    await evaluator.verify(
        claim=claim_scope,
        node=scope_leaf,
        sources=data.job_posting_url,
        additional_instruction="Look for language such as 'oversees', 'directs', 'manages' departments, programs, or major functions.",
    )

    return {"node": node, "prog_leaf": prog_leaf, "scope_leaf": scope_leaf}


async def build_exec_leadership_scope(
    evaluator: Evaluator,
    parent,
    prefix: str,
    data: PositionData,
) -> Dict[str, Any]:
    """
    Build executive-level leadership scope (non-critical, parallel), including salary range check.
    """
    node = evaluator.add_parallel(
        id=f"{prefix}_leadership_scope",
        desc="The position demonstrates executive-level leadership scope and responsibilities",
        parent=parent,
        critical=False,
    )

    strategic_leaf = evaluator.add_leaf(
        id=f"{prefix}_strategic_leadership",
        desc="The job posting emphasizes strategic planning, institutional leadership, or vision-setting responsibilities",
        parent=node,
        critical=False,
    )
    claim_strat = "The posting emphasizes strategic planning, institutional leadership, or vision-setting responsibilities."
    await evaluator.verify(
        claim=claim_strat,
        node=strategic_leaf,
        sources=data.job_posting_url,
        additional_instruction="Look for 'strategic planning', 'institutional leadership', 'vision', or similar executive language.",
    )

    oversight_leaf = evaluator.add_leaf(
        id=f"{prefix}_organizational_oversight",
        desc="The position includes oversight of multiple departments, divisions, or significant institutional operations",
        parent=node,
        critical=False,
    )
    claim_over = "The position includes oversight of multiple departments, divisions, or significant institutional operations."
    await evaluator.verify(
        claim=claim_over,
        node=oversight_leaf,
        sources=data.job_posting_url,
        additional_instruction="Look for broad organizational oversight typical of executive roles.",
    )

    salary_leaf = evaluator.add_leaf(
        id=f"{prefix}_salary_range",
        desc="If disclosed, the salary falls within executive-level range ($140,000-$225,000+)",
        parent=node,
        critical=False,
    )
    claim_salary = "If salary is disclosed on the posting, it falls within $140,000 to $225,000 or higher."
    await evaluator.verify(
        claim=claim_salary,
        node=salary_leaf,
        sources=data.job_posting_url,
        additional_instruction="If salary is not disclosed, consider this not satisfied. If disclosed, check whether it meets or exceeds the range.",
    )

    return {"node": node, "strategic_leaf": strategic_leaf, "oversight_leaf": oversight_leaf, "salary_leaf": salary_leaf}


# --------------------------------------------------------------------------- #
# Position-specific verifiers                                                 #
# --------------------------------------------------------------------------- #
async def verify_entry_position(evaluator: Evaluator, parent, data: PositionData) -> None:
    pos_node = evaluator.add_parallel(
        id="entry_level_position",
        desc="Identify one entry-level higher education administrative position meeting all specified criteria",
        parent=parent,
        critical=False,
    )

    # Institution verification
    inst = await build_institution_verification(evaluator, pos_node, "entry", data)

    # Job posting verification
    post = await build_job_posting_verification(evaluator, pos_node, "entry", data)

    # Title and functional area (gate on posting URL check)
    await build_title_and_area_verification(
        evaluator, pos_node, "entry", data, "entry", prereq_nodes=[post["posting_url_leaf"]]
    )

    # Education
    await build_education_verification(evaluator, pos_node, "entry", data, "entry")

    # Experience
    await build_experience_verification(evaluator, pos_node, "entry", data, "entry")

    # Salary (non-critical)
    await build_entry_salary_verification(evaluator, pos_node, "entry", data)


async def verify_mid_position(evaluator: Evaluator, parent, data: PositionData) -> None:
    pos_node = evaluator.add_parallel(
        id="mid_level_director_position",
        desc="Identify one mid-level director position in higher education administration meeting all specified criteria",
        parent=parent,
        critical=False,
    )

    inst = await build_institution_verification(evaluator, pos_node, "mid", data)
    post = await build_job_posting_verification(evaluator, pos_node, "mid", data)

    await build_title_and_area_verification(
        evaluator, pos_node, "mid", data, "mid", prereq_nodes=[post["posting_url_leaf"]]
    )
    await build_education_verification(evaluator, pos_node, "mid", data, "mid")
    await build_experience_verification(evaluator, pos_node, "mid", data, "mid")
    await build_mid_supervisory_verification(evaluator, pos_node, "mid", data)


async def verify_senior_position(evaluator: Evaluator, parent, data: PositionData) -> None:
    pos_node = evaluator.add_parallel(
        id="senior_level_position",
        desc="Identify one senior-level administrative position (Associate/Assistant Dean or Associate/Assistant Vice President) meeting all specified criteria",
        parent=parent,
        critical=False,
    )

    inst = await build_institution_verification(evaluator, pos_node, "senior", data)
    post = await build_job_posting_verification(evaluator, pos_node, "senior", data)

    await build_title_and_area_verification(
        evaluator, pos_node, "senior", data, "senior", prereq_nodes=[post["posting_url_leaf"]]
    )
    await build_education_verification(evaluator, pos_node, "senior", data, "senior")
    await build_experience_verification(evaluator, pos_node, "senior", data, "senior")
    await build_senior_leadership_verification(evaluator, pos_node, "senior", data)


async def verify_exec_position(evaluator: Evaluator, parent, data: PositionData) -> None:
    pos_node = evaluator.add_parallel(
        id="executive_level_position",
        desc="Identify one executive-level administrative position (Dean, Vice President, or Provost) meeting all specified criteria",
        parent=parent,
        critical=False,
    )

    inst = await build_institution_verification(evaluator, pos_node, "exec", data)
    post = await build_job_posting_verification(evaluator, pos_node, "exec", data)

    await build_title_and_area_verification(
        evaluator, pos_node, "exec", data, "executive", prereq_nodes=[post["posting_url_leaf"]]
    )
    await build_education_verification(evaluator, pos_node, "exec", data, "executive")
    await build_experience_verification(evaluator, pos_node, "exec", data, "executive")
    await build_exec_leadership_scope(evaluator, pos_node, "exec", data)


# --------------------------------------------------------------------------- #
# Diversity checks                                                            #
# --------------------------------------------------------------------------- #
def add_institution_diversity_check(evaluator: Evaluator, parent, positions: List[PositionData]) -> None:
    node = evaluator.add_parallel(
        id="institution_diversity",
        desc="All four positions are from different institutions",
        parent=parent,
        critical=True,
    )

    names = [_normalize_name(p.institution_name) for p in positions]
    all_present = all(bool(n) for n in names)
    unique_insts = len(set(names)) == 4 if all_present else False

    evaluator.add_custom_node(
        result=unique_insts,
        id="institutions_all_different",
        desc="The entry-level, mid-level, senior-level, and executive-level positions are each from a different accredited U.S. college or university (no institution appears more than once)",
        parent=node,
        critical=True,
    )


def add_functional_area_diversity_check(evaluator: Evaluator, parent, positions: List[PositionData]) -> None:
    node = evaluator.add_parallel(
        id="functional_area_diversity",
        desc="All four positions represent different functional areas of higher education administration",
        parent=parent,
        critical=True,
    )

    areas = [_normalize_name(p.functional_area) for p in positions]
    all_present = all(bool(a) for a in areas)
    unique_areas = len(set(areas)) == 4 if all_present else False

    evaluator.add_custom_node(
        result=unique_areas,
        id="functional_areas_all_different",
        desc="The entry-level, mid-level, senior-level, and executive-level positions each represent a different functional area (e.g., Academic Affairs, Student Affairs, Career Services, Enrollment Management, Registrar Services, Athletics/Compliance) with no functional area appearing more than once",
        parent=node,
        critical=True,
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
    Evaluate an answer for the higher education admin jobs task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Root parallel aggregation
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
    # Ensure root is non-critical to allow non-critical children (diversity nodes will be critical)
    root.critical = False

    # Extract structured positions from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=AllPositionsExtraction,
        extraction_name="positions_extraction",
    )

    # Normalize missing positions as empty to avoid attribute errors
    entry = extracted.entry or PositionData()
    mid = extracted.mid or PositionData()
    senior = extracted.senior or PositionData()
    executive = extracted.executive or PositionData()

    # Add optional custom info (for debugging)
    evaluator.add_custom_info(
        {
            "current_window": CURRENT_WINDOW,
            "allowed_boards": ["HigherEdJobs", "Chronicle of Higher Education", "Inside Higher Ed", "Institution Career Site"],
        },
        info_type="configuration",
        info_name="evaluation_context",
    )

    # Build verification subtrees for each level
    await verify_entry_position(evaluator, root, entry)
    await verify_mid_position(evaluator, root, mid)
    await verify_senior_position(evaluator, root, senior)
    await verify_exec_position(evaluator, root, executive)

    # Global diversity checks (critical)
    positions_list = [entry, mid, senior, executive]
    add_institution_diversity_check(evaluator, root, positions_list)
    add_functional_area_diversity_check(evaluator, root, positions_list)

    # Return structured evaluation summary
    return evaluator.get_summary()