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
TASK_ID = "us_edu_institutions_4"
TASK_DESCRIPTION = """
Identify four educational institutions in the United States that meet all the following criteria:

Institution 1 - A Public University:
- Founded in 1889 through a land bequest
- The bequest included exactly 814 acres
- This university is the alma mater (bachelor's and master's degrees) of a person who became superintendent of the 42nd largest school district in the nation in 2012

Institution 2 - A Private University:
- Established a cooperative education program in 1909 that began with exactly 8 students and 4 employers
- The program started in the College of Engineering
- Initially accredited by the New England Commission of Higher Education (NECHE) in 1940
- Currently maintains "Member" status with NECHE

Institution 3 - A Private University:
- Has exactly 14 residential colleges
- The first seven residential colleges opened on September 25, 1933
- Announced a test-flexible admissions policy in February 2024
- The new policy applies to applicants for fall 2025 entry and accepts four types of standardized tests: SAT, ACT, AP, and IB

Institution 4 - A Public High School:
- Located in Massachusetts
- Total enrollment in 2024 was 2,177 students
- The Class of 2024 senior class consisted of exactly 512 students
- The Class of 2023 mean SAT scores were: EBRW 643, Math 660, Total 1303

For each institution, provide:
1. The full official name of the institution
2. The city and state where it is located
3. Reference URL(s) that verify the key criteria
"""

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class Institution1(BaseModel):
    official_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    type: Optional[str] = None  # e.g., "public university"
    founded_year: Optional[str] = None
    founding_basis: Optional[str] = None  # e.g., "land bequest"
    bequest_acres: Optional[str] = None   # e.g., "814 acres"
    alumni_superintendent_person: Optional[str] = None
    alumni_degrees_summary: Optional[str] = None  # e.g., "bachelor's and master's"
    superintendent_district_name: Optional[str] = None
    superintendent_district_rank: Optional[str] = None  # e.g., "42nd largest"
    superintendent_year: Optional[str] = None  # e.g., "2012"
    sources: List[str] = Field(default_factory=list)


class Institution2(BaseModel):
    official_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    type: Optional[str] = None  # e.g., "private university"
    co_op_established_year: Optional[str] = None  # e.g., "1909"
    co_op_initial_students: Optional[str] = None  # e.g., "8"
    co_op_initial_employers: Optional[str] = None  # e.g., "4"
    co_op_origin_college: Optional[str] = None  # e.g., "College of Engineering"
    neche_initial_accredit_year: Optional[str] = None  # e.g., "1940"
    neche_current_status: Optional[str] = None  # e.g., "Member"
    sources: List[str] = Field(default_factory=list)


class Institution3(BaseModel):
    official_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    type: Optional[str] = None  # e.g., "private university"
    residential_colleges_count: Optional[str] = None  # e.g., "14"
    first_seven_opened_date: Optional[str] = None  # e.g., "September 25, 1933"
    test_flexible_announcement_month_year: Optional[str] = None  # e.g., "February 2024"
    test_flexible_applicable_entry_term: Optional[str] = None  # e.g., "fall 2025"
    test_flexible_accepted_tests: List[str] = Field(default_factory=list)  # e.g., ["SAT","ACT","AP","IB"]
    sources: List[str] = Field(default_factory=list)


class Institution4(BaseModel):
    official_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    type: Optional[str] = None  # e.g., "public high school"
    enrollment_2024: Optional[str] = None  # e.g., "2,177"
    senior_class_2024: Optional[str] = None  # e.g., "512"
    sat_2023_ebrw: Optional[str] = None  # e.g., "643"
    sat_2023_math: Optional[str] = None  # e.g., "660"
    sat_2023_total: Optional[str] = None  # e.g., "1303"
    sources: List[str] = Field(default_factory=list)


class InstitutionsExtraction(BaseModel):
    institution_1: Optional[Institution1] = None
    institution_2: Optional[Institution2] = None
    institution_3: Optional[Institution3] = None
    institution_4: Optional[Institution4] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_institutions() -> str:
    return """
    Extract structured information for four institutions (as presented in the answer). For each institution, return the fields exactly as they appear in the answer; if a field is missing, return null (or an empty list for list fields). Also extract all reference URLs explicitly listed in the answer for that institution (do not invent URLs).

    institution_1 (Public University):
      - official_name (string)
      - city (string)
      - state (string)
      - type (string; as stated, e.g., "public university")
      - founded_year (string)
      - founding_basis (string, e.g., "land bequest")
      - bequest_acres (string, e.g., "814 acres" or "814")
      - alumni_superintendent_person (string; the person's full name if given)
      - alumni_degrees_summary (string; e.g., "bachelor's and master's")
      - superintendent_district_name (string, if provided)
      - superintendent_district_rank (string; e.g., "42nd largest", if provided)
      - superintendent_year (string; e.g., "2012", if provided)
      - sources (array of URLs). Include every URL cited for Institution 1.

    institution_2 (Private University):
      - official_name
      - city
      - state
      - type
      - co_op_established_year (string)
      - co_op_initial_students (string)
      - co_op_initial_employers (string)
      - co_op_origin_college (string)
      - neche_initial_accredit_year (string)
      - neche_current_status (string; e.g., "Member")
      - sources (array of URLs). Include every URL cited for Institution 2.

    institution_3 (Private University):
      - official_name
      - city
      - state
      - type
      - residential_colleges_count (string; e.g., "14")
      - first_seven_opened_date (string; e.g., "September 25, 1933")
      - test_flexible_announcement_month_year (string; e.g., "February 2024")
      - test_flexible_applicable_entry_term (string; e.g., "fall 2025")
      - test_flexible_accepted_tests (array of strings; e.g., ["SAT","ACT","AP","IB"])
      - sources (array of URLs). Include every URL cited for Institution 3.

    institution_4 (Public High School in Massachusetts):
      - official_name
      - city
      - state
      - type
      - enrollment_2024 (string; e.g., "2,177" or "2177")
      - senior_class_2024 (string; e.g., "512")
      - sat_2023_ebrw (string; e.g., "643")
      - sat_2023_math (string; e.g., "660")
      - sat_2023_total (string; e.g., "1303")
      - sources (array of URLs). Include every URL cited for Institution 4.

    IMPORTANT FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer. Accept plain URLs or URLs inside markdown links. Ensure they have a valid protocol (http/https).
    - Do not deduplicate aggressively; include all provided URLs for each institution.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _city_state_str(city: Optional[str], state: Optional[str]) -> str:
    if city and state:
        return f"{city}, {state}"
    if state:
        return state
    if city:
        return city
    return ""


def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0


# --------------------------------------------------------------------------- #
# Verification builders per institution                                       #
# --------------------------------------------------------------------------- #
async def verify_institution_1(evaluator: Evaluator, parent_node, inst: Optional[Institution1]) -> None:
    inst = inst or Institution1()
    name = inst.official_name or "the institution"
    loc = _city_state_str(inst.city, inst.state)
    urls = inst.sources or []

    # Institution 1 node
    node_inst = evaluator.add_parallel(
        id="Institution_1",
        desc="Institution 1 (public university) meets all specified criteria and required outputs are provided",
        parent=parent_node,
        critical=False
    )

    # Identity Output (critical)
    node_identity = evaluator.add_parallel(
        id="Institution_1_Identity_Output",
        desc="Provide required identity information for Institution 1",
        parent=node_inst,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(inst.official_name and inst.official_name.strip()),
        id="Institution_1_Official_Name",
        desc="Full official institution name is provided",
        parent=node_identity,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(inst.city and inst.state and inst.city.strip() and inst.state.strip()),
        id="Institution_1_City_State",
        desc="City and state are provided",
        parent=node_identity,
        critical=True
    )

    # References (critical: must have URLs)
    evaluator.add_custom_node(
        result=_has_sources(urls),
        id="Institution_1_References",
        desc="Provide reference URL(s) that verify Institution 1's key criteria",
        parent=node_inst,
        critical=True
    )

    # Eligibility Criteria (critical)
    node_elig = evaluator.add_parallel(
        id="Institution_1_Eligibility_Criteria",
        desc="Institution 1 satisfies the question’s eligibility constraints",
        parent=node_inst,
        critical=True
    )

    # Build leaves
    leaves_and_claims = []

    # Public university in US
    n1 = evaluator.add_leaf(
        id="Institution_1_Public_University_In_US",
        desc="Institution 1 is a public university located in the United States",
        parent=node_elig,
        critical=True
    )
    claim = f"{name} is a public university in the United States."
    if loc:
        claim += f" It is located in {loc}."
    add_ins = "Confirm that the institution is a public university (e.g., public, state, or land‑grant) within the United States."
    leaves_and_claims.append((claim, urls, n1, add_ins))

    # Founded in 1889
    n2 = evaluator.add_leaf(
        id="Institution_1_Founded_1889",
        desc="Institution 1 was founded in 1889",
        parent=node_elig,
        critical=True
    )
    claim = f"{name} was founded in 1889."
    add_ins = "Verify the founding year is 1889. Minor phrasing differences are acceptable as long as the year matches."
    leaves_and_claims.append((claim, urls, n2, add_ins))

    # Founded through a land bequest
    n3 = evaluator.add_leaf(
        id="Institution_1_Founded_Through_Land_Bequest",
        desc="Institution 1 was founded through a land bequest",
        parent=node_elig,
        critical=True
    )
    claim = f"{name} was founded through a land bequest."
    add_ins = "Look for explicit mention that the founding was via a land bequest (donation of land)."
    leaves_and_claims.append((claim, urls, n3, add_ins))

    # Bequest included exactly 814 acres
    n4 = evaluator.add_leaf(
        id="Institution_1_Bequest_814_Acres",
        desc="The founding land bequest included exactly 814 acres",
        parent=node_elig,
        critical=True
    )
    claim = f"The founding land bequest for {name} included exactly 814 acres."
    add_ins = "Treat '814 acres' and '814‑acre' as equivalent. Numeric formatting (with or without comma) should be considered equivalent."
    leaves_and_claims.append((claim, urls, n4, add_ins))

    # Alumni superintendent in 2012 claim
    n5 = evaluator.add_leaf(
        id="Institution_1_Alumni_Superintendent_2012",
        desc="There exists a person who earned both a bachelor's and a master's degree from Institution 1 and who became superintendent of the 42nd-largest school district in the nation in 2012",
        parent=node_elig,
        critical=True
    )
    if inst.alumni_superintendent_person:
        person_part = f"{inst.alumni_superintendent_person} "
    else:
        person_part = "A person "
    claim = (
        f"{person_part}earned both a bachelor's and a master's degree from {name} "
        f"and became superintendent of the 42nd-largest school district in the United States in 2012."
    )
    if inst.superintendent_district_name:
        claim += f" The district referenced is {inst.superintendent_district_name}."
    add_ins = (
        "Verify that the cited person holds both bachelor's and master's degrees from the institution "
        "and was appointed superintendent in 2012 of the nation's 42nd‑largest school district. "
        "Allow '42nd largest' with or without hyphen; focus on explicit evidence from the provided pages."
    )
    leaves_and_claims.append((claim, urls, n5, add_ins))

    # Execute batch verification for eligibility leaves
    await evaluator.batch_verify(leaves_and_claims)


async def verify_institution_2(evaluator: Evaluator, parent_node, inst: Optional[Institution2]) -> None:
    inst = inst or Institution2()
    name = inst.official_name or "the institution"
    loc = _city_state_str(inst.city, inst.state)
    urls = inst.sources or []

    node_inst = evaluator.add_parallel(
        id="Institution_2",
        desc="Institution 2 (private university) meets all co-op + NECHE criteria and required outputs are provided",
        parent=parent_node,
        critical=False
    )

    # Identity Output (critical)
    node_identity = evaluator.add_parallel(
        id="Institution_2_Identity_Output",
        desc="Provide required identity information for Institution 2",
        parent=node_inst,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(inst.official_name and inst.official_name.strip()),
        id="Institution_2_Official_Name",
        desc="Full official institution name is provided",
        parent=node_identity,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(inst.city and inst.state and inst.city.strip() and inst.state.strip()),
        id="Institution_2_City_State",
        desc="City and state are provided",
        parent=node_identity,
        critical=True
    )

    # References (critical)
    evaluator.add_custom_node(
        result=_has_sources(urls),
        id="Institution_2_References",
        desc="Provide reference URL(s) that verify Institution 2's key criteria",
        parent=node_inst,
        critical=True
    )

    # Eligibility (critical)
    node_elig = evaluator.add_parallel(
        id="Institution_2_Eligibility_Criteria",
        desc="Institution 2 satisfies the question’s eligibility constraints",
        parent=node_inst,
        critical=True
    )

    leaves_and_claims = []

    # Private university in US
    n1 = evaluator.add_leaf(
        id="Institution_2_Private_University_In_US",
        desc="Institution 2 is a private university located in the United States",
        parent=node_elig,
        critical=True
    )
    claim = f"{name} is a private university in the United States."
    if loc:
        claim += f" It is located in {loc}."
    add_ins = "Confirm that the institution is private and in the US."
    leaves_and_claims.append((claim, urls, n1, add_ins))

    # Co-op established in 1909
    n2 = evaluator.add_leaf(
        id="Institution_2_Coop_Established_1909",
        desc="Institution 2 established a cooperative education program in 1909",
        parent=node_elig,
        critical=True
    )
    claim = f"In 1909, {name} established a cooperative education (co‑op) program."
    add_ins = "Look for explicit mention of 1909 as the year the co‑op program was established."
    leaves_and_claims.append((claim, urls, n2, add_ins))

    # Co-op started with 8 students and 4 employers
    n3 = evaluator.add_leaf(
        id="Institution_2_Coop_Started_With_8_Students_4_Employers",
        desc="The cooperative education program began with exactly 8 students and exactly 4 employers",
        parent=node_elig,
        critical=True
    )
    claim = "The cooperative education program began with exactly 8 students and 4 employers."
    add_ins = "Confirm both numbers (8 students and 4 employers). Allow numeric formatting variation (e.g., 'eight' vs '8') only if clearly the same counts."
    leaves_and_claims.append((claim, urls, n3, add_ins))

    # Co-op started in College of Engineering
    n4 = evaluator.add_leaf(
        id="Institution_2_Coop_Started_In_College_Of_Engineering",
        desc="The cooperative education program started in the College of Engineering",
        parent=node_elig,
        critical=True
    )
    claim = "The cooperative education program started in the College of Engineering."
    add_ins = "Look for explicit attribution to the College of Engineering as the origin of the co‑op program."
    leaves_and_claims.append((claim, urls, n4, add_ins))

    # NECHE initially accredited in 1940
    n5 = evaluator.add_leaf(
        id="Institution_2_NECHE_Initially_Accredited_1940",
        desc="Institution 2 was initially accredited by NECHE in 1940",
        parent=node_elig,
        critical=True
    )
    claim = f"{name} was initially accredited by the New England Commission of Higher Education (NECHE) in 1940."
    add_ins = (
        "NECHE has continuity with the New England Association of Schools and Colleges (NEASC) Commission on Institutions of Higher Education. "
        "If the older name (NEASC/CIHE) is used for 1940, treat it as NECHE lineage."
    )
    leaves_and_claims.append((claim, urls, n5, add_ins))

    # NECHE current Member status
    n6 = evaluator.add_leaf(
        id="Institution_2_NECHE_Current_Member_Status",
        desc="Institution 2 currently maintains \"Member\" status with NECHE",
        parent=node_elig,
        critical=True
    )
    claim = f"{name} currently maintains 'Member' status with NECHE."
    add_ins = "Check the NECHE (or equivalent official) directory/profile page for the institution. Status should explicitly be 'Member'."
    leaves_and_claims.append((claim, urls, n6, add_ins))

    await evaluator.batch_verify(leaves_and_claims)


async def verify_institution_3(evaluator: Evaluator, parent_node, inst: Optional[Institution3]) -> None:
    inst = inst or Institution3()
    name = inst.official_name or "the institution"
    loc = _city_state_str(inst.city, inst.state)
    urls = inst.sources or []

    node_inst = evaluator.add_parallel(
        id="Institution_3",
        desc="Institution 3 (private university) meets all residential-college + admissions-policy criteria and required outputs are provided",
        parent=parent_node,
        critical=False
    )

    # Identity Output (critical)
    node_identity = evaluator.add_parallel(
        id="Institution_3_Identity_Output",
        desc="Provide required identity information for Institution 3",
        parent=node_inst,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(inst.official_name and inst.official_name.strip()),
        id="Institution_3_Official_Name",
        desc="Full official institution name is provided",
        parent=node_identity,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(inst.city and inst.state and inst.city.strip() and inst.state.strip()),
        id="Institution_3_City_State",
        desc="City and state are provided",
        parent=node_identity,
        critical=True
    )

    # References (critical)
    evaluator.add_custom_node(
        result=_has_sources(urls),
        id="Institution_3_References",
        desc="Provide reference URL(s) that verify Institution 3's key criteria",
        parent=node_inst,
        critical=True
    )

    # Eligibility (critical)
    node_elig = evaluator.add_parallel(
        id="Institution_3_Eligibility_Criteria",
        desc="Institution 3 satisfies the question’s eligibility constraints",
        parent=node_inst,
        critical=True
    )

    leaves_and_claims = []

    # Private university in US
    n1 = evaluator.add_leaf(
        id="Institution_3_Private_University_In_US",
        desc="Institution 3 is a private university located in the United States",
        parent=node_elig,
        critical=True
    )
    claim = f"{name} is a private university in the United States."
    if loc:
        claim += f" It is located in {loc}."
    add_ins = "Confirm that the institution is private and in the US."
    leaves_and_claims.append((claim, urls, n1, add_ins))

    # Exactly 14 residential colleges
    n2 = evaluator.add_leaf(
        id="Institution_3_Residential_Colleges_Exactly_14",
        desc="Institution 3 has exactly 14 residential colleges",
        parent=node_elig,
        critical=True
    )
    claim = f"{name} has exactly 14 residential colleges."
    add_ins = "Verify the total count is 14 (allow 'fourteen')."
    leaves_and_claims.append((claim, urls, n2, add_ins))

    # First seven opened on 1933-09-25
    n3 = evaluator.add_leaf(
        id="Institution_3_First_Seven_Opened_1933_09_25",
        desc="The first seven residential colleges opened on September 25, 1933",
        parent=node_elig,
        critical=True
    )
    claim = "The first seven residential colleges opened on September 25, 1933."
    add_ins = "Look for explicit historical date indicating the opening of the first seven residential colleges."
    leaves_and_claims.append((claim, urls, n3, add_ins))

    # Test-flexible announced Feb 2024
    n4 = evaluator.add_leaf(
        id="Institution_3_Test_Flexible_Announced_Feb_2024",
        desc="Institution 3 announced a test-flexible admissions policy in February 2024",
        parent=node_elig,
        critical=True
    )
    claim = f"In February 2024, {name} announced a test‑flexible admissions policy."
    add_ins = "Look for an official announcement or policy page with month/year (February 2024)."
    leaves_and_claims.append((claim, urls, n4, add_ins))

    # Test-flexible applies to fall 2025 entry
    n5 = evaluator.add_leaf(
        id="Institution_3_Test_Flexible_Applies_Fall_2025",
        desc="The test-flexible policy applies to applicants for fall 2025 entry",
        parent=node_elig,
        critical=True
    )
    claim = "The new test‑flexible admissions policy applies to applicants seeking entry in fall 2025."
    add_ins = "Verify applicability timeline explicitly mentions fall 2025 entrants."
    leaves_and_claims.append((claim, urls, n5, add_ins))

    # Test-flexible accepts exactly SAT, ACT, AP, IB
    n6 = evaluator.add_leaf(
        id="Institution_3_Test_Flexible_Accepts_SAT_ACT_AP_IB",
        desc="The test-flexible policy accepts exactly four types of standardized tests: SAT, ACT, AP, and IB",
        parent=node_elig,
        critical=True
    )
    claim = "Under the test‑flexible policy, applicants may submit standardized test results from exactly these four options: SAT, ACT, AP, and IB."
    add_ins = (
        "Confirm that the accepted standardized tests are SAT, ACT, AP, and IB. "
        "Treat equivalent phrasing (e.g., 'International Baccalaureate' for IB) as a match. "
        "The set should be limited to these four."
    )
    leaves_and_claims.append((claim, urls, n6, add_ins))

    await evaluator.batch_verify(leaves_and_claims)


async def verify_institution_4(evaluator: Evaluator, parent_node, inst: Optional[Institution4]) -> None:
    inst = inst or Institution4()
    name = inst.official_name or "the institution"
    urls = inst.sources or []
    loc = _city_state_str(inst.city, inst.state)

    node_inst = evaluator.add_parallel(
        id="Institution_4",
        desc="Institution 4 (Massachusetts public high school) meets all enrollment + SAT-score criteria and required outputs are provided",
        parent=parent_node,
        critical=False
    )

    # Identity Output (critical)
    node_identity = evaluator.add_parallel(
        id="Institution_4_Identity_Output",
        desc="Provide required identity information for Institution 4",
        parent=node_inst,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(inst.official_name and inst.official_name.strip()),
        id="Institution_4_Official_Name",
        desc="Full official institution name is provided",
        parent=node_identity,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(inst.city and inst.state and inst.city.strip() and inst.state.strip()),
        id="Institution_4_City_State",
        desc="City and state are provided",
        parent=node_identity,
        critical=True
    )

    # References (critical)
    evaluator.add_custom_node(
        result=_has_sources(urls),
        id="Institution_4_References",
        desc="Provide reference URL(s) that verify Institution 4's key criteria",
        parent=node_inst,
        critical=True
    )

    # Eligibility (critical)
    node_elig = evaluator.add_parallel(
        id="Institution_4_Eligibility_Criteria",
        desc="Institution 4 satisfies the question’s eligibility constraints",
        parent=node_inst,
        critical=True
    )

    leaves_and_claims = []

    # Public high school in Massachusetts
    n1 = evaluator.add_leaf(
        id="Institution_4_Public_High_School_In_MA",
        desc="Institution 4 is a public high school located in Massachusetts",
        parent=node_elig,
        critical=True
    )
    claim = f"{name} is a public high school located in Massachusetts."
    if loc and "Massachusetts" not in loc:
        claim += f" It is located in {loc}."
    add_ins = "Confirm both 'public high school' and that it is in the state of Massachusetts."
    leaves_and_claims.append((claim, urls, n1, add_ins))

    # Enrollment 2024 exactly 2,177
    n2 = evaluator.add_leaf(
        id="Institution_4_Enrollment_2024_2177",
        desc="Total enrollment in 2024 was exactly 2,177 students",
        parent=node_elig,
        critical=True
    )
    claim = "Total enrollment in 2024 was exactly 2,177 students."
    add_ins = "Accept numeric formatting variations (e.g., '2,177' vs '2177') as equivalent."
    leaves_and_claims.append((claim, urls, n2, add_ins))

    # Senior class 2024 exactly 512
    n3 = evaluator.add_leaf(
        id="Institution_4_Senior_Class_2024_512",
        desc="Class of 2024 senior class consisted of exactly 512 students",
        parent=node_elig,
        critical=True
    )
    claim = "The Class of 2024 senior class consisted of exactly 512 students."
    add_ins = "Look for explicit counts for the Class of 2024 seniors."
    leaves_and_claims.append((claim, urls, n3, add_ins))

    # SAT scores 2023 exact set
    n4 = evaluator.add_leaf(
        id="Institution_4_SAT_Scores_2023_Exact_Set",
        desc="Class of 2023 mean SAT scores are exactly: EBRW 643, Math 660, Total 1303",
        parent=node_elig,
        critical=True
    )
    claim = "For the Class of 2023, the mean SAT scores were exactly: EBRW 643, Math 660, Total 1303."
    add_ins = "Verify the three-part SAT mean breakdown for Class of 2023 matches exactly. Minor formatting differences are acceptable."
    leaves_and_claims.append((claim, urls, n4, add_ins))

    await evaluator.batch_verify(leaves_and_claims)


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
    Evaluate the answer for the 'us_edu_institutions_4' task and return a structured summary.
    """
    # Initialize evaluator (root is non-critical by framework design)
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

    # Optional additional grouping node (non-critical to allow partial credit)
    overall = evaluator.add_parallel(
        id="Root",
        desc="Identify four US educational institutions meeting the specified per-institution criteria, and provide required identity info and reference URL(s) for each",
        parent=root,
        critical=False
    )

    # Extract structured information once
    extracted = await evaluator.extract(
        prompt=prompt_extract_institutions(),
        template_class=InstitutionsExtraction,
        extraction_name="institutions_extraction"
    )

    # Verify each institution block
    await verify_institution_1(evaluator, overall, extracted.institution_1 if extracted else None)
    await verify_institution_2(evaluator, overall, extracted.institution_2 if extracted else None)
    await verify_institution_3(evaluator, overall, extracted.institution_3 if extracted else None)
    await verify_institution_4(evaluator, overall, extracted.institution_4 if extracted else None)

    # Return evaluation summary
    return evaluator.get_summary()