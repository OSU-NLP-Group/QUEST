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
TASK_ID = "tx_isd_graduation_requirements_ctx"
TASK_DESCRIPTION = (
    "A family is relocating to Central Texas and needs to find a public school district for their high school-aged child. "
    "They have specific requirements for graduation standards and district characteristics. Identify one Texas public "
    "independent school district (ISD) that meets ALL of the following criteria:\n\n"
    "1. The district must be located in one of these Central Texas counties: Travis, Williamson, Hays, or Bastrop\n"
    "2. The district must serve high school students (in grades 9-12 or in a K-12 configuration)\n"
    "3. The district must use the Texas Foundation High School Program (FHSP) for high school graduation\n"
    "4. The district must require students to pass all 5 STAAR End-of-Course (EOC) assessments for graduation: Algebra I, English I, English II, Biology, and U.S. History\n"
    "5. The district must require a minimum of 22 credits for graduation under the Foundation Program\n"
    "6. The district must require 4 credits in each of the four core subject areas: English, Mathematics, Science, and Social Studies\n"
    "7. The district must have publicly available information about graduation requirements and STAAR testing on its official website\n"
    "8. The district should have a total student enrollment of at least 20,000 students\n"
    "9. The district should offer at least 3 endorsement options for high school students\n\n"
    "Provide the name of the district, the county (or counties) in which it is located, and verify that it meets each of the above criteria."
)

ALLOWED_COUNTIES = {"travis", "williamson", "hays", "bastrop"}
MIN_ENROLLMENT = 20_000
EOC_LIST = ["Algebra I", "English I", "English II", "Biology", "U.S. History"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DistrictBasics(BaseModel):
    district_name: Optional[str] = None
    counties: List[str] = Field(default_factory=list)
    serves_grades: Optional[str] = None  # e.g., "9-12", "K-12", "PreK-12"
    total_enrollment: Optional[str] = None  # Keep as string to be lenient (e.g., "≈ 25,000")


class ProgramRequirements(BaseModel):
    graduation_program: Optional[str] = None  # Expect references to "Foundation High School Program" or "FHSP"
    eoc_requirements_text: Optional[str] = None  # Text that states EOC passage requirement
    min_credits: Optional[str] = None  # e.g., "22"
    core_credits_policy: Optional[str] = None  # e.g., "4/4/4/4 for ELA/Math/Science/Social Studies"
    endorsement_options: List[str] = Field(default_factory=list)  # e.g., ["STEM", "Business & Industry", ...]


class Sources(BaseModel):
    # Category-specific URLs if provided in the answer (optional but helpful)
    graduation_urls: List[str] = Field(default_factory=list)
    staar_urls: List[str] = Field(default_factory=list)
    enrollment_urls: List[str] = Field(default_factory=list)
    location_urls: List[str] = Field(default_factory=list)
    endorsement_urls: List[str] = Field(default_factory=list)
    official_urls: List[str] = Field(default_factory=list)  # URLs on the district's official domain
    all_urls: List[str] = Field(default_factory=list)  # All URLs cited for this district


class DistrictExtraction(BaseModel):
    basics: Optional[DistrictBasics] = None
    requirements: Optional[ProgramRequirements] = None
    sources: Sources = Field(default_factory=Sources)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_district() -> str:
    return """
    From the answer, extract structured information about exactly one Texas public independent school district (ISD) that the answer claims meets the listed criteria.

    You must strictly follow these rules:
    - Extract only what is explicitly present in the answer text.
    - Do not invent or infer any information not present in the answer.
    - For URLs, extract only explicit URLs shown in the answer. Include both web pages and PDFs if they are linked.
    - Try to categorize URLs if the answer implies which page supports which claim. If uncertain, include them in all_urls.

    Extract the following JSON object:

    {
      "basics": {
        "district_name": string or null,
        "counties": [list of counties mentioned (e.g., "Travis County", "Williamson")],
        "serves_grades": string or null,  // examples: "9-12", "K-12", "PreK-12"
        "total_enrollment": string or null  // number as written in answer, allow commas or words like "approximately"
      },
      "requirements": {
        "graduation_program": string or null,  // e.g., "Foundation High School Program", "FHSP"
        "eoc_requirements_text": string or null, // the sentence(s) that state the STAAR EOC passage requirement
        "min_credits": string or null, // e.g., "22"
        "core_credits_policy": string or null, // any text that states required 4 credits in ELA/Math/Science/Social Studies
        "endorsement_options": [list of endorsement names mentioned] // e.g., STEM, Business & Industry, Public Services, Arts & Humanities, Multidisciplinary Studies
      },
      "sources": {
        "graduation_urls": [URLs that discuss graduation requirements or FHSP],
        "staar_urls": [URLs that discuss STAAR testing or EOC requirements],
        "enrollment_urls": [URLs that discuss district enrollment size],
        "location_urls": [URLs that show district location or counties served],
        "endorsement_urls": [URLs that list available endorsements],
        "official_urls": [URLs on the district's official website/domain],
        "all_urls": [every URL in the answer, de-duplicated]
      }
    }

    URL extraction requirements:
    - Extract only valid, complete URLs explicitly present in the answer.
    - If a URL is missing "http:// or https://", prepend "http://".
    - Deduplicate URLs across all fields. 'all_urls' should contain every distinct URL you found.
    - 'official_urls' should include any URLs on the district's official domain (e.g., *.isd.org, *.k12.tx.us, *.net). Do your best based on the text; if unclear, leave empty.

    If any field is not explicitly stated in the answer, return null (or empty list).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def unique_merge(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst or []:
            if not url:
                continue
            u = url.strip()
            if u and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def normalize_county_name(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip().lower()
    for token in ["county", "co.", "co", "texas", ",", ".", " "]:
        s = s.replace(token, " ")
    return " ".join(s.split())


def pick_allowed_county(counties: List[str]) -> Optional[str]:
    for c in counties or []:
        base = normalize_county_name(c)
        if base in ALLOWED_COUNTIES:
            return base.title()
    return None


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, root_node, extracted: DistrictExtraction) -> None:
    basics = extracted.basics or DistrictBasics()
    reqs = extracted.requirements or ProgramRequirements()
    src = extracted.sources or Sources()

    district_name = basics.district_name or ""
    counties_list = basics.counties or []
    allowed_county_in_answer = pick_allowed_county(counties_list)

    # Compose frequently used source groups
    urls_all = unique_merge(src.all_urls)
    urls_official = unique_merge(src.official_urls)
    urls_grad = unique_merge(src.graduation_urls, urls_official, urls_all)
    urls_staar = unique_merge(src.staar_urls, urls_official, urls_all)
    urls_enroll = unique_merge(src.enrollment_urls, urls_official, urls_all)
    urls_location = unique_merge(src.location_urls, urls_official, urls_all)
    urls_endorse = unique_merge(src.endorsement_urls, urls_grad, urls_official, urls_all)

    # 1) District Identification (Critical) - existence check
    evaluator.add_custom_node(
        result=bool(district_name.strip()),
        id="District_Identification",
        desc="Provides the specific name of a Texas public independent school district (ISD)",
        parent=root_node,
        critical=True
    )

    # 2) Central Texas Location (Critical) - verify with URLs
    loc_leaf = evaluator.add_leaf(
        id="Central_Texas_Location",
        desc="The district is located in Travis, Williamson, Hays, or Bastrop county in Central Texas",
        parent=root_node,
        critical=True
    )
    if allowed_county_in_answer:
        county_phrase = f"{allowed_county_in_answer} County"
    else:
        county_phrase = "one of Travis, Williamson, Hays, or Bastrop County"

    claim_loc = (
        f"The official or authoritative webpage shows that the school district '{district_name}' is located in or serves communities within {county_phrase} in Central Texas."
    )
    await evaluator.verify(
        claim=claim_loc,
        node=loc_leaf,
        sources=urls_location,
        additional_instruction=(
            "Accept evidence if the page explicitly states the county served, or if the district name clearly indicates the county "
            "(e.g., 'Hays CISD' indicates Hays County), or if campus addresses/catchment maps imply the county. "
            "Prefer official district pages; if absent, other authoritative Texas education pages are acceptable."
        )
    )

    # 3) High School Service (Critical)
    hs_leaf = evaluator.add_leaf(
        id="High_School_Service",
        desc="The district serves high school students in grades 9-12 or K-12 configuration",
        parent=root_node,
        critical=True
    )
    claim_hs = (
        f"The official webpage shows that '{district_name}' operates high schools and serves students in grades 9–12 (i.e., high school). "
        "Any campus directory listing high schools or a statement like 'grades 9–12' is sufficient."
    )
    await evaluator.verify(
        claim=claim_hs,
        node=hs_leaf,
        sources=urls_official or urls_all,
        additional_instruction=(
            "Look for explicit mentions of 'High School', 'grades 9–12', or campus directory pages listing high schools. "
            "If the district is K–12, that also satisfies serving high school students."
        )
    )

    # 4) Foundation Program (Critical)
    fhsp_leaf = evaluator.add_leaf(
        id="Foundation_Program",
        desc="The district uses the Texas Foundation High School Program (FHSP) for high school graduation",
        parent=root_node,
        critical=True
    )
    claim_fhsp = (
        "This official district page states that high school graduation requirements follow the Texas Foundation High School Program "
        "(FHSP), sometimes described as the 'Foundation Program' or 'Foundation with Endorsement(s)'."
    )
    await evaluator.verify(
        claim=claim_fhsp,
        node=fhsp_leaf,
        sources=urls_grad,
        additional_instruction=(
            "Accept synonyms such as 'Foundation High School Program', 'FHSP', 'Foundation Program', or 'Foundation with Endorsements (FHSP-E)'. "
            "The statement must clearly tie graduation requirements to the state FHSP."
        )
    )

    # 5) STAAR EOC Requirements (Critical)
    eoc_leaf = evaluator.add_leaf(
        id="STAAR_EOC_Requirements",
        desc="The district requires students to pass all 5 STAAR End-of-Course assessments for graduation: Algebra I, English I, English II, Biology, and U.S. History",
        parent=root_node,
        critical=True
    )
    claim_eoc = (
        "This page shows that the district requires students to pass all five STAAR EOC exams—Algebra I, English I, English II, Biology, and U.S. History—for graduation."
    )
    await evaluator.verify(
        claim=claim_eoc,
        node=eoc_leaf,
        sources=urls_staar or urls_grad,
        additional_instruction=(
            "The page should list all five EOCs or clearly say 'all required EOC assessments' and elsewhere enumerate these five. "
            "All five subjects must be present; partial lists are insufficient."
        )
    )

    # 6) Credit Minimum (Critical)
    credits_leaf = evaluator.add_leaf(
        id="Credit_Minimum",
        desc="The district requires a minimum of 22 credits for graduation under the Foundation Program",
        parent=root_node,
        critical=True
    )
    claim_credits = (
        "This page states that the minimum number of credits required for graduation under the Foundation Program is 22 credits."
    )
    await evaluator.verify(
        claim=claim_credits,
        node=credits_leaf,
        sources=urls_grad,
        additional_instruction=(
            "Look for explicit references to '22 credits' or 'twenty-two (22) credits' under the Foundation plan."
        )
    )

    # 7) Core Credits Policy (Critical)
    core_leaf = evaluator.add_leaf(
        id="Core_Credits",
        desc="The district requires 4 credits in each of the four core subject areas: English, Mathematics, Science, and Social Studies",
        parent=root_node,
        critical=True
    )
    claim_core = (
        "This page shows that graduation requires 4 credits each in English (ELA), Mathematics, Science, and Social Studies."
    )
    await evaluator.verify(
        claim=claim_core,
        node=core_leaf,
        sources=urls_grad,
        additional_instruction=(
            "The statement must indicate 4 credits in each of the four listed core areas, not just totals. "
            "Use reasonable synonym tolerance: 'English Language Arts' for English; 'Math' for Mathematics; "
            "'Social Studies' may include U.S. History, Government, Economics, World Geography/History, etc."
        )
    )

    # 8) Public Information (Critical) - broken into two concrete leaves under one critical parent
    public_info_parent = evaluator.add_parallel(
        id="Public_Information",
        desc="The district has publicly available information about graduation requirements and STAAR testing on its official website",
        parent=root_node,
        critical=True
    )
    # 8a) Graduation info exists
    pub_grad_leaf = evaluator.add_leaf(
        id="Public_Info_Graduation",
        desc="Official district website publicly posts high school graduation requirements",
        parent=public_info_parent,
        critical=True
    )
    claim_pub_grad = (
        "This is an official district webpage that publicly provides the district's high school graduation requirements."
    )
    await evaluator.verify(
        claim=claim_pub_grad,
        node=pub_grad_leaf,
        sources=urls_grad or urls_official or urls_all,
        additional_instruction=(
            "Confirm the page is on the district's official domain and contains graduation requirement details (e.g., credit counts, FHSP). "
            "PDFs hosted on the district domain (e.g., course guides) count as official pages."
        )
    )
    # 8b) STAAR info exists
    pub_staar_leaf = evaluator.add_leaf(
        id="Public_Info_STAAR",
        desc="Official district website publicly provides information about STAAR testing (including EOC)",
        parent=public_info_parent,
        critical=True
    )
    claim_pub_staar = (
        "This is an official district webpage that publicly provides information about STAAR testing, including End-of-Course (EOC) assessments."
    )
    await evaluator.verify(
        claim=claim_pub_staar,
        node=pub_staar_leaf,
        sources=urls_staar or urls_official or urls_all,
        additional_instruction=(
            "Confirm the page is on the district's official domain and mentions STAAR and/or EOC assessments. "
            "Testing calendars, assessment overviews, or counseling/testing pages qualify."
        )
    )

    # 9) Enrollment Size (Non-Critical)
    enroll_leaf = evaluator.add_leaf(
        id="Enrollment_Size",
        desc="The district has a total student enrollment of at least 20,000 students",
        parent=root_node,
        critical=False
    )
    claim_enroll = (
        f"This page shows that the total district enrollment is at least {MIN_ENROLLMENT} students."
    )
    await evaluator.verify(
        claim=claim_enroll,
        node=enroll_leaf,
        sources=urls_enroll or urls_official or urls_all,
        additional_instruction=(
            "Accept reasonable rounding or approximate figures as long as they are ≥ 20,000. "
            "If multiple years or counts are shown, use the most recent overall K–12 total enrollment. "
            "If the total is presented as the sum of campuses, that is acceptable if it clearly exceeds the threshold."
        )
    )

    # 10) Endorsement Options (Non-Critical)
    endorse_leaf = evaluator.add_leaf(
        id="Endorsement_Options",
        desc="The district offers at least 3 endorsement options for high school students",
        parent=root_node,
        critical=False
    )
    claim_endorse = (
        "This page shows that the district offers at least three endorsement options for high school students, "
        "such as STEM, Business & Industry, Public Services, Arts & Humanities, and Multidisciplinary Studies."
    )
    await evaluator.verify(
        claim=claim_endorse,
        node=endorse_leaf,
        sources=urls_endorse or urls_grad or urls_official or urls_all,
        additional_instruction=(
            "Count distinct endorsements listed. Accept synonyms/variants if clearly the state-defined categories "
            "(STEM; Business & Industry; Public Services; Arts & Humanities; Multidisciplinary Studies)."
        )
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
    Evaluate an answer for the Central Texas ISD graduation requirements task.
    """
    # Initialize evaluator with a parallel root to mirror rubric's top-level aggregation
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

    # Extract structured district info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_district(),
        template_class=DistrictExtraction,
        extraction_name="district_extraction"
    )

    # Add helpful context info to the summary
    evaluator.add_custom_info(
        info={
            "allowed_counties": sorted(list(ALLOWED_COUNTIES)),
            "min_enrollment_threshold": MIN_ENROLLMENT,
            "eoc_list": EOC_LIST
        },
        info_type="constraints",
        info_name="evaluation_constraints"
    )

    # Build verification nodes and run checks
    await build_and_verify(evaluator, root, extracted)

    return evaluator.get_summary()