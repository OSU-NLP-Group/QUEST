import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "midwest_universities_constrained_selection"
TASK_DESCRIPTION = """
I am researching higher education options in the Midwest region and need to identify four public universities that meet specific criteria for athletic programs, housing policies, transfer student requirements, and application procedures.

Please identify four public universities located in Midwest states (Illinois, Indiana, Iowa, Kansas, Michigan, Minnesota, Missouri, Nebraska, North Dakota, Ohio, South Dakota, or Wisconsin) that meet ALL of the following requirements:

1. Public Institution Status: The university must be a public (state-supported) institution.
2. Regional Accreditation: The university must hold regional accreditation from one of the six recognized U.S. regional accrediting agencies.
3. NCAA Division I Athletics: The university must compete in NCAA Division I intercollegiate athletics.
4. Conference Membership: The university must be a member of either the Big Ten Conference or the Mid-American Conference (MAC).
5. Freshman Housing Requirement: The university must have an on-campus residency requirement for first-year undergraduate students.
6. Transfer Student GPA Requirements: The university must have a stated minimum GPA requirement for transfer student admission.
7. Transfer Credit Hour Thresholds: The university must specify credit hour thresholds that affect transfer admission requirements (e.g., different requirements based on number of transfer credits).
8. Application Deadlines: The university must have a published application deadline for fall 2026 freshman admission.
9. Enrollment Size: The university must have a total enrollment of at least 10,000 students.
10. Business Program: The university must offer an undergraduate business administration or business management degree program.
11. Transfer Agreements: The university must have documented articulation or transfer agreements with at least one community college.

For each of the four universities, please provide:
- University name
- Main campus physical address
- Regional accrediting agency
- Athletic conference (Big Ten or MAC)
- Description of freshman housing policy
- Transfer GPA requirement
- Transfer credit hour threshold policy
- Application deadline for fall 2026
- Total enrollment figure
- Confirmation of undergraduate business program
- Description of community college transfer agreement partnerships
- Reference URLs for each piece of information
"""

MIDWEST_STATES = [
    "Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota",
    "Missouri", "Nebraska", "North Dakota", "Ohio", "South Dakota", "Wisconsin"
]

RECOGNIZED_REGIONAL_ACCREDITORS = [
    "Higher Learning Commission",
    "Middle States Commission on Higher Education",
    "New England Commission of Higher Education",
    "Northwest Commission on Colleges and Universities",
    "Southern Association of Colleges and Schools Commission on Colleges",
    "WASC Senior College and University Commission"
]

CONFERENCE_WHITELIST = ["Big Ten Conference", "Mid-American Conference", "MAC", "Big Ten", "B1G"]


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class FieldTextURLs(BaseModel):
    """A generic field value paired with one or more source URLs from the answer."""
    text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class UniversityItem(BaseModel):
    """All fields expected per university item in the answer."""
    name: Optional[str] = None

    # Core requirements
    public_status: FieldTextURLs = Field(default_factory=FieldTextURLs)
    location: FieldTextURLs = Field(default_factory=FieldTextURLs)  # Should contain state name in text
    address: FieldTextURLs = Field(default_factory=FieldTextURLs)

    accreditation: FieldTextURLs = Field(default_factory=FieldTextURLs)

    ncaa_division: FieldTextURLs = Field(default_factory=FieldTextURLs)       # Expect "Division I" or equivalent
    conference: FieldTextURLs = Field(default_factory=FieldTextURLs)          # Expect "Big Ten Conference" or "Mid-American Conference"

    freshman_housing: FieldTextURLs = Field(default_factory=FieldTextURLs)    # Policy text
    transfer_min_gpa: FieldTextURLs = Field(default_factory=FieldTextURLs)    # e.g., "2.5 GPA"
    transfer_credit_policy: FieldTextURLs = Field(default_factory=FieldTextURLs)  # thresholds description
    application_deadline_fall_2026: FieldTextURLs = Field(default_factory=FieldTextURLs)

    total_enrollment: FieldTextURLs = Field(default_factory=FieldTextURLs)    # numeric or phrase

    business_program: FieldTextURLs = Field(default_factory=FieldTextURLs)    # BA/BBA/BSBA in Business/Management
    transfer_agreements: FieldTextURLs = Field(default_factory=FieldTextURLs) # agreements with CCs


class UniversitiesExtraction(BaseModel):
    """Top-level extraction result: up to 4 universities."""
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    states_str = ", ".join(MIDWEST_STATES)
    accreditors_str = "; ".join(RECOGNIZED_REGIONAL_ACCREDITORS)
    return f"""
Extract up to four universities from the answer, structuring each as an object with the exact fields below.
For ANY field where the answer provides URLs/sources, extract the concrete URLs (plain or from markdown). If a field has no URLs in the answer, set its 'urls' to an empty list.

Return JSON with a top-level key "universities": an array of at most 4 university objects. Each university object MUST have:
- name: string | null
- public_status: {{ text: string | null, urls: string[] }}
- location: {{ text: string | null, urls: string[] }}  // Put the U.S. state (spelled out). If city/state appears in the answer, set the state's name here.
- address: {{ text: string | null, urls: string[] }}   // Main campus physical address (street + city + state if available)
- accreditation: {{ text: string | null, urls: string[] }} // Regional accreditor name (e.g., Higher Learning Commission) and its proof URL(s)
- ncaa_division: {{ text: string | null, urls: string[] }} // e.g., "NCAA Division I" and evidence URLs
- conference: {{ text: string | null, urls: string[] }}     // e.g., "Big Ten Conference" or "Mid-American Conference (MAC)" and evidence URLs
- freshman_housing: {{ text: string | null, urls: string[] }} // Description of first-year on-campus housing requirement + policy URL(s)
- transfer_min_gpa: {{ text: string | null, urls: string[] }} // Stated minimum GPA for transfer admission + URL(s)
- transfer_credit_policy: {{ text: string | null, urls: string[] }} // Thresholds affecting transfer admission (e.g., different criteria above 24/30 credits) + URL(s)
- application_deadline_fall_2026: {{ text: string | null, urls: string[] }} // The published deadline text for Fall 2026 freshman admission + URL(s)
- total_enrollment: {{ text: string | null, urls: string[] }} // Total enrollment figure + URL(s)
- business_program: {{ text: string | null, urls: string[] }} // Confirmation/description of undergrad business administration/management program + URL(s)
- transfer_agreements: {{ text: string | null, urls: string[] }} // Description/confirmation of community-college articulation/transfer agreements + URL(s)

Rules & guidance:
- If the answer lists more than 4 universities, include only the first 4.
- For location.text: Extract the U.S. state spelled out (e.g., "Ohio") if available; otherwise, use the best available clue from the answer.
- For conference, restrict to “Big Ten Conference” or “Mid-American Conference (MAC)” (or obvious variants like “Big Ten”, “B1G”, “MAC”).
- For accreditation, only extract a recognized U.S. regional accreditor; expected examples include:
  {accreditors_str}
- Only extract URLs explicitly present in the answer text.
- Do not invent or infer any URLs.
- If a value is missing in the answer, set its text to null (and urls to []).

Midwest states to consider valid: {states_str}.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_urls(*url_lists: List[str]) -> List[str]:
    """Combine and de-duplicate multiple URL lists, preserving order."""
    seen = set()
    out: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if isinstance(u, str):
                u2 = u.strip()
                if u2 and u2 not in seen:
                    seen.add(u2)
                    out.append(u2)
    return out


def _has_urls(urls: List[str]) -> bool:
    return any(isinstance(u, str) and u.strip().startswith(("http://", "https://")) for u in (urls or []))


# --------------------------------------------------------------------------- #
# Verification per-university                                                 #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    idx: int
) -> None:
    """
    Build verification sub-tree and run checks for one university item.
    All mandatory criteria are implemented as critical checks (URL presence + content verification).
    Note: To maintain strict gating ("must meet ALL requirements"), we mark every requirement here as critical.
    """
    uni_name = uni.name or f"University #{idx}"

    # Parent node for this university (non-critical at root level to allow partial credit across universities)
    uni_node = evaluator.add_parallel(
        id=f"University_{idx}",
        desc=f"University {idx}: {uni_name} meets all mandatory criteria",
        parent=parent_node,
        critical=False
    )

    # 1) Public Institution Status ------------------------------------------------
    grp_public = evaluator.add_parallel(
        id=f"U{idx}_Public_Status",
        desc="The university must be a public (state-supported) institution",
        parent=uni_node,
        critical=True
    )
    # URL presence
    evaluator.add_custom_node(
        result=_has_urls(uni.public_status.urls),
        id=f"U{idx}_Public_Status_URL",
        desc="URL reference confirming public institution status is provided",
        parent=grp_public,
        critical=True
    )
    # Verification
    leaf_public = evaluator.add_leaf(
        id=f"U{idx}_Public_Status_Verify",
        desc="Confirmation that the university is a public state-supported institution",
        parent=grp_public,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} is a public, state-supported university (not a private institution).",
        node=leaf_public,
        sources=uni.public_status.urls,
        additional_instruction="Rely on official 'About' pages, state designations, or accreditor profiles. Phrases like 'public research university' qualify."
    )

    # 2) Geographic Location (Midwest state) ------------------------------------
    grp_geo = evaluator.add_parallel(
        id=f"U{idx}_Geographic_Location",
        desc="The university is located in an eligible Midwest state",
        parent=uni_node,
        critical=True
    )
    geo_urls = _combine_urls(uni.location.urls, uni.address.urls)
    evaluator.add_custom_node(
        result=_has_urls(geo_urls),
        id=f"U{idx}_Geo_URL",
        desc="URL reference for location/state is provided",
        parent=grp_geo,
        critical=True
    )
    state_txt = (uni.location.text or "").strip()
    leaf_geo = evaluator.add_leaf(
        id=f"U{idx}_Geo_Verify",
        desc="University is located in a listed Midwest state (IL, IN, IA, KS, MI, MN, MO, NE, ND, OH, SD, or WI)",
        parent=grp_geo,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"{uni_name} is located in {state_txt if state_txt else 'a listed Midwest state'}; "
            f"valid Midwest states include: {', '.join(MIDWEST_STATES)}."
        ),
        node=leaf_geo,
        sources=geo_urls,
        additional_instruction="Confirm the state of the main campus is among the provided Midwest list. Use contact/address or 'About' pages."
    )

    # 3) Regional Accreditation ---------------------------------------------------
    grp_accr = evaluator.add_parallel(
        id=f"U{idx}_Regional_Accreditation",
        desc="The university holds recognized U.S. regional accreditation",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.accreditation.urls),
        id=f"U{idx}_Accreditation_URL",
        desc="URL reference confirming the university's regional accreditation status",
        parent=grp_accr,
        critical=True
    )
    leaf_accr = evaluator.add_leaf(
        id=f"U{idx}_Accreditation_Verification",
        desc="University is accredited by one of the six recognized U.S. regional accreditors",
        parent=grp_accr,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"{uni_name} is regionally accredited by {uni.accreditation.text}. "
            f"This agency is one of the six recognized U.S. regional accreditors."
        ),
        node=leaf_accr,
        sources=uni.accreditation.urls,
        additional_instruction=(
            "Accept only if the accreditor matches one of: "
            + "; ".join(RECOGNIZED_REGIONAL_ACCREDITORS)
            + ". Verify that the page explicitly confirms institutional accreditation."
        )
    )

    # 4) NCAA Division I ----------------------------------------------------------
    grp_div = evaluator.add_parallel(
        id=f"U{idx}_NCAA_Division",
        desc="The university competes in NCAA Division I athletics",
        parent=uni_node,
        critical=True
    )
    div_urls = _combine_urls(uni.ncaa_division.urls, uni.conference.urls)
    evaluator.add_custom_node(
        result=_has_urls(div_urls),
        id=f"U{idx}_Division_URL",
        desc="URL reference confirming NCAA Division I membership",
        parent=grp_div,
        critical=True
    )
    leaf_div = evaluator.add_leaf(
        id=f"U{idx}_Division_Status",
        desc="Confirmation that the university fields NCAA Division I athletic programs",
        parent=grp_div,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} competes in NCAA Division I intercollegiate athletics.",
        node=leaf_div,
        sources=div_urls,
        additional_instruction="Look for NCAA, conference, athletics, or official profiles explicitly stating 'Division I'."
    )

    # 5) Conference Membership (Big Ten or MAC) ----------------------------------
    grp_conf = evaluator.add_parallel(
        id=f"U{idx}_Conference_Membership",
        desc="The university is a member of either the Big Ten or the Mid-American Conference",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.conference.urls),
        id=f"U{idx}_Conference_URL",
        desc="URL reference confirming conference membership",
        parent=grp_conf,
        critical=True
    )
    conf_txt = (uni.conference.text or "").strip()
    leaf_conf = evaluator.add_leaf(
        id=f"U{idx}_Conference_Identification",
        desc="Identification of which qualifying conference the university belongs to (Big Ten or MAC)",
        parent=grp_conf,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"{uni_name} is a member of the {conf_txt if conf_txt else 'Big Ten or Mid-American Conference'}; "
            "membership must be Big Ten (B1G) or Mid-American Conference (MAC)."
        ),
        node=leaf_conf,
        sources=uni.conference.urls,
        additional_instruction="Reject any conference other than Big Ten or MAC. Verify membership via conference or school athletics pages."
    )

    # 6) Freshman Housing Requirement --------------------------------------------
    grp_housing = evaluator.add_parallel(
        id=f"U{idx}_Freshman_Housing",
        desc="On-campus residency requirement exists for first-year undergraduates",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.freshman_housing.urls),
        id=f"U{idx}_Housing_URL",
        desc="URL reference to the university's first-year housing policy",
        parent=grp_housing,
        critical=True
    )
    leaf_housing = evaluator.add_leaf(
        id=f"U{idx}_Housing_Policy",
        desc="Verification of the specific freshman housing policy requiring on-campus residence",
        parent=grp_housing,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"{uni_name} requires first-year undergraduate students to live on campus "
            "(allowing typical exemptions such as age, veteran status, proximity, etc.)."
        ),
        node=leaf_housing,
        sources=uni.freshman_housing.urls,
        additional_instruction="The page must clearly indicate an on-campus residency requirement for first-year students (with usual exemptions allowed)."
    )

    # 7) Transfer Student GPA Requirements ---------------------------------------
    grp_gpa = evaluator.add_parallel(
        id=f"U{idx}_Transfer_GPA",
        desc="The university states a minimum transfer GPA requirement",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.transfer_min_gpa.urls),
        id=f"U{idx}_GPA_URL",
        desc="URL reference to the university's transfer GPA requirements",
        parent=grp_gpa,
        critical=True
    )
    gpa_txt = (uni.transfer_min_gpa.text or "").strip()
    leaf_gpa = evaluator.add_leaf(
        id=f"U{idx}_GPA_Requirement",
        desc="Identification of the specific minimum GPA threshold for transfer admission",
        parent=grp_gpa,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} publishes a minimum GPA requirement for transfer admission: {gpa_txt}.",
        node=leaf_gpa,
        sources=uni.transfer_min_gpa.urls,
        additional_instruction="Confirm that a minimum GPA threshold is stated (e.g., 2.0, 2.5, 3.0). If multiple thresholds exist by credits/college, the extracted value should be visible on the cited page."
    )

    # 8) Transfer Credit Hour Thresholds -----------------------------------------
    grp_credit = evaluator.add_parallel(
        id=f"U{idx}_Transfer_Credit_Threshold",
        desc="Transfer credit hour thresholds affect admission requirements",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.transfer_credit_policy.urls),
        id=f"U{idx}_Credit_URL",
        desc="URL reference to transfer credit-hour threshold policies",
        parent=grp_credit,
        critical=True
    )
    credit_txt = (uni.transfer_credit_policy.text or "").strip()
    leaf_credit = evaluator.add_leaf(
        id=f"U{idx}_Credit_Hour_Policy",
        desc="Description of how the number of transfer credits affects admission criteria",
        parent=grp_credit,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"{uni_name} specifies credit-hour thresholds that change transfer admission requirements. "
            f"Policy evidence: {credit_txt}"
        ),
        node=leaf_credit,
        sources=uni.transfer_credit_policy.urls,
        additional_instruction="Look for thresholds such as 12/24/30/60 credits determining which criteria/transcripts/test scores are required."
    )

    # 9) Application Deadline for Fall 2026 --------------------------------------
    grp_deadline = evaluator.add_parallel(
        id=f"U{idx}_Application_Deadline",
        desc="Published application deadline for Fall 2026 freshman admission",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.application_deadline_fall_2026.urls),
        id=f"U{idx}_Deadline_URL",
        desc="URL reference to application deadlines for Fall 2026 admission",
        parent=grp_deadline,
        critical=True
    )
    deadline_txt = (uni.application_deadline_fall_2026.text or "").strip()
    leaf_deadline = evaluator.add_leaf(
        id=f"U{idx}_Deadline_Date",
        desc="Specific application deadline date (EA/ED/Regular) for Fall 2026 is published",
        parent=grp_deadline,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} publishes an application deadline for Fall 2026 freshman admission: {deadline_txt}.",
        node=leaf_deadline,
        sources=uni.application_deadline_fall_2026.urls,
        additional_instruction="Accept priority/early/regular deadlines as long as they are clearly associated with Fall 2026."
    )

    # 10) Enrollment Size >= 10,000 ----------------------------------------------
    grp_enroll = evaluator.add_parallel(
        id=f"U{idx}_Enrollment_Size",
        desc="Total enrollment is at least 10,000 students",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.total_enrollment.urls),
        id=f"U{idx}_Enrollment_URL",
        desc="URL reference to enrollment statistics",
        parent=grp_enroll,
        critical=True
    )
    leaf_enroll = evaluator.add_leaf(
        id=f"U{idx}_Enrollment_Number",
        desc="Verification that total enrollment meets or exceeds 10,000 students",
        parent=grp_enroll,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name}'s total student enrollment is at least 10,000.",
        node=leaf_enroll,
        sources=uni.total_enrollment.urls,
        additional_instruction="Use official Fact Books, About pages, or IPEDS/official stats. Total = undergrad + grad."
    )

    # 11) Undergraduate Business Program -----------------------------------------
    grp_biz = evaluator.add_parallel(
        id=f"U{idx}_Business_Program",
        desc="University offers an undergraduate business administration/management degree",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.business_program.urls),
        id=f"U{idx}_Program_URL",
        desc="URL reference to business program information",
        parent=grp_biz,
        critical=True
    )
    biz_txt = (uni.business_program.text or "").strip()
    leaf_biz = evaluator.add_leaf(
        id=f"U{idx}_Program_Offering",
        desc="Confirmation that the university offers undergraduate business degree programs",
        parent=grp_biz,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"{uni_name} offers an undergraduate business program (e.g., Business Administration/Management): {biz_txt}"
        ),
        node=leaf_biz,
        sources=uni.business_program.urls,
        additional_instruction="Accept BA/BBA/BSBA in Business Administration/Management or closely named majors under an accredited business school/college."
    )

    # 12) Transfer Agreements with Community Colleges ----------------------------
    grp_agree = evaluator.add_parallel(
        id=f"U{idx}_Transfer_Agreements",
        desc="Documented articulation/transfer agreements with at least one community college",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.transfer_agreements.urls),
        id=f"U{idx}_Agreement_URL",
        desc="URL reference to community college transfer partnerships or articulation agreements",
        parent=grp_agree,
        critical=True
    )
    agree_txt = (uni.transfer_agreements.text or "").strip()
    leaf_agree = evaluator.add_leaf(
        id=f"U{idx}_Agreement_Existence",
        desc="Verification that formal transfer agreements exist with community college partners",
        parent=grp_agree,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"{uni_name} has formal articulation or transfer agreements with at least one community college. "
            f"Example/description: {agree_txt}"
        ),
        node=leaf_agree,
        sources=uni.transfer_agreements.urls,
        additional_instruction="Evidence may include transfer partnership listings, 2+2 agreements, pathway pages, or MOUs with named community colleges."
    )

    # 13) Main Campus Physical Address (treated as critical here to avoid soft-child scoring issues)
    # Note: Though the rubric marks address as non-critical info, we mark it critical within the university node
    # to keep the university node free of non-critical children (ensures strict gating semantics for mandatory checks).
    addr_urls = uni.address.urls
    leaf_addr = evaluator.add_leaf(
        id=f"U{idx}_Physical_Address",
        desc="Main campus physical address is provided and supported by sources",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The main campus physical address for {uni_name} is: {uni.address.text or ''}",
        node=leaf_addr,
        sources=addr_urls,
        additional_instruction="Match the primary campus postal address (street/city/state). Minor formatting differences are acceptable."
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
    Evaluate an answer for the Midwest public universities task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Universities evaluated independently
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

    # Extract structured university information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Normalize to exactly 4 items (pad with empty if fewer; take first 4 if more)
    universities: List[UniversityItem] = list(extracted.universities or [])
    universities = universities[:4]
    while len(universities) < 4:
        universities.append(UniversityItem())

    # Record GT/reference info helpful to judge context
    evaluator.add_ground_truth({
        "midwest_states": MIDWEST_STATES,
        "recognized_regional_accreditors": RECOGNIZED_REGIONAL_ACCREDITORS,
        "allowed_conferences": CONFERENCE_WHITELIST
    }, gt_type="constraints_reference")

    # Build verification tree for each university
    tasks = []
    for i, uni in enumerate(universities, start=1):
        tasks.append(verify_university(evaluator, root, uni, i))
    # Run sequentially to keep logs ordered; could also use asyncio.gather for concurrency
    for t in tasks:
        await t

    # Return structured evaluation summary
    return evaluator.get_summary()