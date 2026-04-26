import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "transfer_friendly_public_universities"
TASK_DESCRIPTION = """
Identify four public universities in the United States that meet all of the following criteria:

1. The university must be regionally accredited by one of the seven regional accrediting agencies recognized by the Council for Higher Education Accreditation (CHEA).
2. The university must be a public institution (state-funded and part of a state university system).
3. The university must offer fully online bachelor's degree programs (100% online, not hybrid or primarily on-campus).
4. The university must have a published transfer credit policy that explicitly states it accepts at least 60 semester credits (or equivalent quarter credits) from community colleges toward a bachelor's degree.

For each university, provide:
- The full name of the university
- The specific regional accrediting agency that accredits the university
- A direct URL from the accrediting agency's official directory or website that confirms the university's current accreditation status
- A URL to the university's official transfer credit policy page
- A URL to the university's official online bachelor's degree programs page
"""

# Recognized regional accreditors (CHEA-recognized, higher-ed institutional)
CHEA_REGIONALLY_RECOGNIZED_AGENCIES = [
    "Middle States Commission on Higher Education",
    "MSCHE",
    "New England Commission of Higher Education",
    "NECHE",
    "Higher Learning Commission",
    "HLC",
    "Southern Association of Colleges and Schools Commission on Colleges",
    "SACSCOC",
    "WASC Senior College and University Commission",
    "WSCUC",
    "Northwest Commission on Colleges and Universities",
    "NWCCU",
    "Accrediting Commission for Community and Junior Colleges",
    "ACCJC"
]

# Official directory domains/patterns for the recognized accreditors
ACCREDITOR_OFFICIAL_DOMAINS = [
    "msche.org",         # MSCHE
    "neche.org",         # NECHE
    "hlcommission.org",  # HLC
    "sacscoc.org",       # SACSCOC
    "wscuc.org",         # WSCUC
    "nwccu.org",         # NWCCU
    "accjc.org",         # ACCJC
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    university_name: Optional[str] = None
    accrediting_agency: Optional[str] = None
    accreditation_directory_url: Optional[str] = None
    transfer_policy_url: Optional[str] = None
    online_bachelors_url: Optional[str] = None


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    You must extract up to the first four universities described in the answer that purport to meet all requested criteria.
    For each university, extract exactly the following fields:

    - university_name: The full official name of the university (string).
    - accrediting_agency: The name of the regional accrediting agency claimed in the answer (string). Use the name as written in the answer.
    - accreditation_directory_url: A direct URL to the accrediting agency’s official institution directory/listing page that confirms the institution’s current accreditation status (string URL). Examples of valid domains include: msche.org, neche.org, hlcommission.org, sacscoc.org, wscuc.org, nwccu.org, accjc.org. Do NOT use Wikipedia, third-party lists, or the institution’s own website here.
    - transfer_policy_url: A URL on the university’s official .edu domain that states the official transfer credit policy (string URL). Prefer a policy or catalog page that explicitly states minimum/maximum transferable credits from community colleges.
    - online_bachelors_url: A URL on the university’s official .edu domain that lists fully online bachelor’s degree programs (string URL). It must be an official page, not a third-party aggregator.

    Additional instructions:
    - Extract only URLs explicitly present in the answer. Do not invent or infer URLs.
    - If the answer lists more than four universities, keep only the first four.
    - If any field is missing for a university, set it to null.
    - Preserve the exact text for names and agencies as written in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"][n] if 0 <= n < 6 else f"#{n+1}"


def build_accreditation_additional_instruction(agency: Optional[str]) -> str:
    agencies_list = ", ".join(CHEA_REGIONALLY_RECOGNIZED_AGENCIES)
    domain_list = ", ".join(ACCREDITOR_OFFICIAL_DOMAINS)
    return (
        "Use only the provided source webpage(s) to judge. If no URL is provided or the URL is invalid/inaccessible, "
        "you must judge this claim as Incorrect.\n\n"
        f"The accrediting agency stated in the answer is: '{agency or 'UNKNOWN'}'. "
        "Confirm two things from the accreditor's official directory page:\n"
        "1) The page explicitly lists the institution and indicates that it is currently 'Accredited' (or equivalent active institutional status).\n"
        "2) The named agency is one of the CHEA-recognized U.S. regional institutional accreditors. The allowed agency names/abbreviations are:\n"
        f"- {agencies_list}\n\n"
        "Notes:\n"
        "- Ignore programmatic/specialized accreditations; this task is about institutional accreditation only.\n"
        "- If the page shows 'Formerly accredited', 'Withdrawn', 'Terminated', or similar, treat as not meeting the requirement.\n"
        f"- The accreditor directory is typically hosted on one of these official domains: {domain_list}."
    )


def build_public_status_additional_instruction(university: Optional[str]) -> str:
    return (
        "Use only the provided source webpage(s) to judge. If no URL is provided or the URL is invalid/inaccessible, "
        "you must judge this claim as Incorrect.\n\n"
        f"From the accreditor directory page, determine whether {university or 'the institution'} is a PUBLIC institution. "
        "Look for explicit fields like 'Control: Public', 'Type: Public', 'Sector: Public, 4-year', or other clear labels indicating the institution is public/state.\n"
        "Do not rely on your own knowledge; rely on the evidence on the page."
    )


def build_online_bachelors_additional_instruction(university: Optional[str]) -> str:
    return (
        "Use only the provided source webpage(s) to judge. If no URL is provided or the URL is invalid/inaccessible, "
        "you must judge this claim as Incorrect.\n\n"
        f"Verify that the page is an official {university or 'university'} webpage and that it lists fully online (100% online) bachelor's degree programs. "
        "Accept phrases like 'online', 'fully online', '100% online', 'entirely online'. "
        "Do NOT count hybrid, campus-based with occasional online courses, or 'online completion' programs that mandate in-person requirements. "
        "It is sufficient if at least one bachelor's program is clearly marked as fully online."
    )


def build_transfer_policy_additional_instruction(university: Optional[str]) -> str:
    return (
        "Use only the provided source webpage(s) to judge. If no URL is provided or the URL is invalid/inaccessible, "
        "you must judge this claim as Incorrect.\n\n"
        f"Confirm that the official {university or 'university'} transfer credit policy explicitly allows at least 60 semester credits "
        "(or 90 quarter credits) from community colleges toward a bachelor's degree. Equivalent phrasings like 'up to 64 semester hours', "
        "'junior standing with 60 credits', or 'associate degree (60 credits) accepted' are acceptable. "
        "Reject policies that allow fewer than 60 semester credits or that only allow 2-year credits toward non-bachelor awards."
    )


def build_accred_url_reference_instruction(agency: Optional[str], university: Optional[str]) -> str:
    domain_list = ", ".join(ACCREDITOR_OFFICIAL_DOMAINS)
    return (
        "If no URL is provided or invalid, judge as Incorrect.\n"
        "Confirm that this URL is the OFFICIAL directory/listing page of the regional accrediting agency, and that it lists the specified institution.\n"
        f"- Institution name: {university or 'UNKNOWN'}\n"
        f"- Accrediting agency: {agency or 'UNKNOWN'}\n"
        f"Allowed official domains include: {domain_list}.\n"
        "Reject third-party sites, Wikipedia, or the institution's own site for this item."
    )


def build_transfer_url_reference_instruction(university: Optional[str]) -> str:
    return (
        "If no URL is provided or invalid, judge as Incorrect.\n"
        "Confirm that this page is on the institution's official domain (typically .edu) and that it specifically presents the official transfer credit policy "
        f"for {university or 'the institution'}. General marketing pages are insufficient; the page should state policy or official catalog rules."
    )


def build_online_url_reference_instruction(university: Optional[str]) -> str:
    return (
        "If no URL is provided or invalid, judge as Incorrect.\n"
        "Confirm that this page is on the institution's official domain (typically .edu) and that it specifically lists ONLINE bachelor's degree programs. "
        "Reject third-party aggregators or general admissions pages that don't actually list online bachelor's programs."
    )


# --------------------------------------------------------------------------- #
# Verification per university                                                 #
# --------------------------------------------------------------------------- #
async def verify_one_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    index: int,
) -> None:
    """
    Build verification subtree and run checks for a single university.
    The top university node is non-critical to allow partial credit across universities.
    Each criterion leaf under it is critical as per the rubric.
    """
    uni_title = f"{ordinal(index)} qualifying public university with complete information"
    uni_node = evaluator.add_parallel(
        id=f"university_{index+1}",
        desc=uni_title,
        parent=parent_node,
        critical=False
    )

    # Prepare leaf nodes
    # 1) Regional Accreditation
    accred_leaf = evaluator.add_leaf(
        id=f"university_{index+1}_regional_accreditation",
        desc="University is accredited by a CHEA-recognized U.S. regional accrediting agency",
        parent=uni_node,
        critical=True
    )

    # 2) Public Institution Status
    public_leaf = evaluator.add_leaf(
        id=f"university_{index+1}_public_status",
        desc="University is a public institution (state-funded, part of a state university system)",
        parent=uni_node,
        critical=True
    )

    # 3) Online Bachelor's Programs (fully online)
    online_leaf = evaluator.add_leaf(
        id=f"university_{index+1}_online_bachelors",
        desc="University offers fully online bachelor's degree programs (100% online, not hybrid)",
        parent=uni_node,
        critical=True
    )

    # 4) Transfer Credit Acceptance (>= 60 semester credits)
    transfer_leaf = evaluator.add_leaf(
        id=f"university_{index+1}_transfer_60",
        desc="Published policy accepts at least 60 semester credits (or 90 quarter credits) from community colleges",
        parent=uni_node,
        critical=True
    )

    # 5) Accreditation Reference URL validity/suitability
    accred_url_leaf = evaluator.add_leaf(
        id=f"university_{index+1}_accred_url_valid",
        desc="Accrediting agency official directory URL confirms accreditation status",
        parent=uni_node,
        critical=True
    )

    # 6) Transfer Policy Reference URL validity/suitability
    transfer_url_leaf = evaluator.add_leaf(
        id=f"university_{index+1}_transfer_url_valid",
        desc="Official university transfer credit policy page URL is provided and appropriate",
        parent=uni_node,
        critical=True
    )

    # 7) Online Programs Reference URL validity/suitability
    online_url_leaf = evaluator.add_leaf(
        id=f"university_{index+1}_online_url_valid",
        desc="Official page for online bachelor's degree programs URL is provided and appropriate",
        parent=uni_node,
        critical=True
    )

    # Build claims and sources
    university_name = uni.university_name or ""
    agency = uni.accrediting_agency or ""

    claims_and_sources = [
        (
            f"According to the accrediting agency's official directory page, {university_name} is institutionally accredited by {agency}.",
            uni.accreditation_directory_url,
            accred_leaf,
            build_accreditation_additional_instruction(agency),
        ),
        (
            f"The accrediting agency directory page explicitly indicates that {university_name} is a PUBLIC institution (e.g., 'Control: Public', 'Type: Public', or similar).",
            uni.accreditation_directory_url,
            public_leaf,
            build_public_status_additional_instruction(university_name),
        ),
        (
            f"This official university webpage lists fully online (100% online) bachelor's degree programs offered by {university_name}.",
            uni.online_bachelors_url,
            online_leaf,
            build_online_bachelors_additional_instruction(university_name),
        ),
        (
            f"The official transfer policy states that {university_name} accepts at least 60 semester credits (or 90 quarter credits) from community colleges toward a bachelor's degree.",
            uni.transfer_policy_url,
            transfer_leaf,
            build_transfer_policy_additional_instruction(university_name),
        ),
        (
            f"This URL is the official directory page of the accrediting agency and it confirms current accreditation for {university_name}.",
            uni.accreditation_directory_url,
            accred_url_leaf,
            build_accred_url_reference_instruction(agency, university_name),
        ),
        (
            f"This URL is {university_name}'s official transfer credit policy page (policy/catalog) on the institution's domain.",
            uni.transfer_policy_url,
            transfer_url_leaf,
            build_transfer_url_reference_instruction(university_name),
        ),
        (
            f"This URL is an official {university_name} page that lists ONLINE bachelor's degree programs.",
            uni.online_bachelors_url,
            online_url_leaf,
            build_online_url_reference_instruction(university_name),
        ),
    ]

    # Run all verifications in parallel to avoid cross-sibling precondition interference
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the 'transfer_friendly_public_universities' task.
    """
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

    # Extract structured universities info from the agent's answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Record recognized agencies as custom info for transparency
    evaluator.add_custom_info(
        {
            "recognized_regional_agencies_examples": CHEA_REGIONALLY_RECOGNIZED_AGENCIES,
            "official_accreditor_domains": ACCREDITOR_OFFICIAL_DOMAINS
        },
        info_type="reference_lists",
        info_name="chea_regional_reference"
    )

    # We need to verify exactly 4 universities; if fewer extracted, pad with blanks
    unis: List[UniversityItem] = list(extracted.universities[:4])
    while len(unis) < 4:
        unis.append(UniversityItem())

    # Build subtrees and verify each university in parallel
    tasks = []
    for i, uni in enumerate(unis[:4]):
        tasks.append(verify_one_university(evaluator, root, uni, i))

    await asyncio.gather(*tasks)

    # Return standardized summary
    return evaluator.get_summary()