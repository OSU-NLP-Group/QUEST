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
TASK_ID = "us_uni_corp_partnerships_4"
TASK_DESCRIPTION = """
Identify four (4) U.S. universities that currently operate employer partnership or corporate partner programs through their career centers with the following characteristics:

1. The program must have at least three (3) distinct membership tiers or levels (e.g., Bronze/Silver/Gold, or similarly named tiers)

2. At least one tier must explicitly offer priority registration or early access to career fairs as a documented benefit

3. The program must have publicly available pricing or fee information for each tier (this may be displayed directly on the website, in a linked PDF document, or available through a specified contact method listed on the page)

4. The program must provide at least two (2) types of exclusive benefits beyond basic career fair registration access, such as:
   - Expedited support on job posting platforms (e.g., Handshake)
   - Organization logo placement on career center materials or website
   - Targeted email campaigns to students
   - Social media posts featuring the employer
   - Resume books or candidate screening services
   - Exclusive networking events or receptions
   - Complimentary registrations for additional fairs

For each university, provide:
- University name
- URL to the employer partnership program page on the university's career center website
- Names of the three or more partnership tiers
- Brief description of how priority registration is offered
- Brief description of at least two exclusive benefits beyond fair access
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    name: Optional[str] = None
    program_url: Optional[str] = None
    tier_names: List[str] = Field(default_factory=list)
    priority_registration_desc: Optional[str] = None
    exclusive_benefits: List[str] = Field(default_factory=list)
    pricing_info: List[str] = Field(default_factory=list)
    additional_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to four (4) university employer partnership or corporate partner program entries exactly as stated in the answer text.

    For each university, extract the following fields if present; if a field is missing in the answer, return null or an empty list for that field:
    - name: The university name as written in the answer
    - program_url: The URL to the employer partnership/corporate partner program page (or official linked document) on the university career center site
    - tier_names: An array of the names of the partnership membership tiers/levels as listed in the answer (e.g., ["Bronze", "Silver", "Gold"]). If fewer than three names are present, include what is present.
    - priority_registration_desc: A brief snippet from the answer that describes how priority registration or early access to career fairs is offered (if mentioned)
    - exclusive_benefits: An array of at least two benefits beyond basic career fair access as they are worded in the answer (e.g., "logo on website", "targeted email to students"); if fewer than two are present in the answer, include what is present
    - pricing_info: An array of any pricing/fee information statements from the answer (e.g., "Gold $5,000", "contact careers@x.edu for tier pricing"). Include each tier's fee if the answer provided them. If the answer states that pricing is available via a specified contact method, include that statement here.
    - additional_urls: Any other URLs the answer explicitly cites for this university (e.g., linked PDFs, fair schedule pages). Only include URLs that actually appear in the answer.

    Rules:
    - Do not invent or infer any information not present in the answer.
    - Normalize any URLs: if a URL is missing a protocol, prepend http://
    - Preserve the tier names and benefit names as close to the answer text as possible.
    - Return a JSON object with a 'universities' array of up to 4 UniversityItem objects, in the same order as they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def gather_sources(item: UniversityItem) -> List[str]:
    seen = set()
    urls: List[str] = []
    if item.program_url and item.program_url.strip():
        u = item.program_url.strip()
        if not u.startswith("http"):
            u = "http://" + u
        if u not in seen:
            seen.add(u)
            urls.append(u)
    for u in item.additional_urls or []:
        if not u:
            continue
        uu = u.strip()
        if not uu:
            continue
        if not uu.startswith("http"):
            uu = "http://" + uu
        if uu not in seen:
            seen.add(uu)
            urls.append(uu)
    return urls


def _distinct_claim(curr_name: Optional[str], prev_names: List[str]) -> str:
    prev = ", ".join([f"'{n}'" for n in prev_names if n])
    curr = curr_name or "UNKNOWN"
    return f"The institution '{curr}' is a different institution than {prev}. Different campuses within the same university system should be considered different institutions for this task."


# --------------------------------------------------------------------------- #
# Verification per university                                                 #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    root,
    item: UniversityItem,
    idx: int,
    prev_items: List[UniversityItem],
) -> None:
    uni_num = idx + 1
    uni_node = evaluator.add_parallel(
        id=f"university_{uni_num}",
        desc=f"University #{uni_num} evaluation",
        parent=root,
        critical=False
    )

    # Distinctness checks for universities #2, #3, #4
    prev_names = [pi.name or "" for pi in prev_items]
    if idx == 1:
        node = evaluator.add_leaf(
            id="u2_distinct_from_u1",
            desc="University #2 is not the same institution as University #1 (no duplicates)",
            parent=uni_node,
            critical=True
        )
        await evaluator.verify(
            claim=_distinct_claim(item.name, prev_names[:1]),
            node=node,
            additional_instruction="Use common sense. Minor name variants for the same institution should count as the same school; different campuses in a multi-campus system count as distinct institutions."
        )
    elif idx == 2:
        node = evaluator.add_leaf(
            id="u3_distinct_from_u1_u2",
            desc="University #3 is not the same institution as University #1 or University #2 (no duplicates)",
            parent=uni_node,
            critical=True
        )
        await evaluator.verify(
            claim=_distinct_claim(item.name, prev_names[:2]),
            node=node,
            additional_instruction="Use common sense. Minor name variants for the same institution should count as the same school; different campuses in a multi-campus system count as distinct institutions."
        )
    elif idx == 3:
        node = evaluator.add_leaf(
            id="u4_distinct_from_u1_u2_u3",
            desc="University #4 is not the same institution as University #1, #2, or #3 (no duplicates)",
            parent=uni_node,
            critical=True
        )
        await evaluator.verify(
            claim=_distinct_claim(item.name, prev_names[:3]),
            node=node,
            additional_instruction="Use common sense. Minor name variants for the same institution should count as the same school; different campuses in a multi-campus system count as distinct institutions."
        )

    # Existence: University name provided
    evaluator.add_custom_node(
        result=bool(item.name and item.name.strip()),
        id=f"u{uni_num}_university_name_provided",
        desc="University name is provided",
        parent=uni_node,
        critical=True
    )

    # Existence: Program page URL provided
    evaluator.add_custom_node(
        result=bool(item.program_url and item.program_url.strip()),
        id=f"u{uni_num}_program_page_url_provided",
        desc="A URL is provided to the employer partnership/corporate partner program page (or official linked document) on the university career center site",
        parent=uni_node,
        critical=True
    )

    sources = gather_sources(item)

    # US university (verify with sources)
    node = evaluator.add_leaf(
        id=f"u{uni_num}_us_university",
        desc="University is located in the United States",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The university '{item.name or 'UNKNOWN'}' is a U.S. institution (located in the United States).",
        node=node,
        sources=sources,
        additional_instruction="Use the provided page(s) to infer location (e.g., state, address, .edu domain context). If the page clearly indicates a U.S. location or the known institution is U.S.-based, mark supported."
    )

    # Formalized career center employer partnership/corporate partner program
    node = evaluator.add_leaf(
        id=f"u{uni_num}_formalized_career_center_program",
        desc="The referenced page/document indicates a formalized employer partnership/corporate partner program for career services (not an informal/one-off offering)",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page (or linked official document) describes a formalized, ongoing employer partnership or corporate partner program operated by the university's career center, with named membership tiers/levels and defined benefits.",
        node=node,
        sources=sources,
        additional_instruction="Look for language such as 'Employer Partner Program', 'Corporate Partners', sponsorship tiers, membership levels, or a benefits matrix that indicates a standing multi-tier program."
    )

    # Currently active for 2024–2025 or 2025–2026
    node = evaluator.add_leaf(
        id=f"u{uni_num}_currently_active_year_window",
        desc="Program is documented as currently active for the 2024–2025 or 2025–2026 academic year",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program is documented as active for either the 2024–2025 or the 2025–2026 academic year.",
        node=node,
        sources=sources,
        additional_instruction="Accept explicit references such as '2024–2025', '2025–2026', 'AY 2024-25', 'AY 2025-26', 'Fall 2024', 'Spring 2025', 'Fall 2025', or 'Spring 2026' that clearly apply to the employer partner program details. If the page (or a linked official PDF) is explicitly labeled with one of those academic years for this program, consider supported."
    )

    # Fairs scheduled for Fall and/or Spring
    node = evaluator.add_leaf(
        id=f"u{uni_num}_fall_or_spring_career_fairs",
        desc="Documentation indicates career fairs are scheduled for Fall and/or Spring semester",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim="The documentation indicates that the university schedules career fairs in the Fall and/or Spring semesters (e.g., 'Fall Career Fair', 'Spring Career Fair').",
        node=node,
        sources=sources,
        additional_instruction="It is sufficient if the program page or an official linked page mentions 'Fall Career Fair' and/or 'Spring Career Fair' or similar seasonal naming that clearly maps to academic semesters."
    )

    # At least 3 tiers (verify on page)
    node = evaluator.add_leaf(
        id=f"u{uni_num}_tier_count",
        desc="Program offers at least three (3) distinct membership tiers/levels",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim="The employer partnership program offers at least three distinct membership tiers or levels.",
        node=node,
        sources=sources,
        additional_instruction="Look for three or more distinct tier names/levels (e.g., Bronze/Silver/Gold, Blue/Gold/Platinum, etc.)."
    )

    # Tier names reported in the response (check from answer only)
    evaluator.add_custom_node(
        result=bool(item.tier_names and len([t for t in item.tier_names if t and t.strip()]) >= 3),
        id=f"u{uni_num}_tier_names_reported",
        desc="Response reports the names of the three or more partnership tiers/levels",
        parent=uni_node,
        critical=True
    )

    # Benefits documented by tier (verify on page)
    node = evaluator.add_leaf(
        id=f"u{uni_num}_tier_benefits_documented",
        desc="Benefits are documented for each tier/level in official career center materials",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim="The official materials document benefits for each tier/level (e.g., a benefits matrix or tier-specific lists).",
        node=node,
        sources=sources,
        additional_instruction="Evidence should show per-tier benefit delineation, such as a table with checks per tier or bullet lists separated by tier headings."
    )

    # Priority registration or early access benefit (verify on page)
    node = evaluator.add_leaf(
        id=f"u{uni_num}_priority_registration_benefit",
        desc="At least one tier explicitly offers priority registration or early access to career fairs, and the response briefly describes how it is offered",
        parent=uni_node,
        critical=True
    )
    pr_detail = item.priority_registration_desc or "Priority or early access to career fairs"
    await evaluator.verify(
        claim=f"At least one membership tier explicitly offers priority registration or early access to career fairs. The answer notes: \"{pr_detail}\".",
        node=node,
        sources=sources,
        additional_instruction="Accept terms like 'priority registration', 'early access', 'first-access window', or similar, clearly tied to career fairs."
    )

    # Pricing per tier publicly available (verify on page or linked doc/contact method)
    node = evaluator.add_leaf(
        id=f"u{uni_num}_pricing_per_tier",
        desc="Pricing/fee information is publicly available for each tier (shown on the page or in a linked official document, or available via a specified contact method listed on the page)",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim="Pricing or fee information for each membership tier is publicly available on this page, in a linked official document (e.g., PDF), or the page specifies a contact method for obtaining per-tier pricing.",
        node=node,
        sources=sources,
        additional_instruction="Accept explicit per-tier dollar amounts, a linked official PDF/guide with per-tier pricing, or a clearly stated contact method on the page to obtain per-tier pricing. Generic 'contact us' without clear relation to per-tier pricing should not pass."
    )

    # At least two exclusive benefits beyond fair access (verify on page)
    node = evaluator.add_leaf(
        id=f"u{uni_num}_exclusive_benefits_beyond_fair_access",
        desc="Program provides at least two (2) exclusive benefits beyond basic career fair registration access, and the response briefly describes at least two of these benefits",
        parent=uni_node,
        critical=True
    )
    benefits_preview = ", ".join(item.exclusive_benefits[:3]) if item.exclusive_benefits else "N/A"
    await evaluator.verify(
        claim="The program provides at least two exclusive benefits beyond basic career fair registration access.",
        node=node,
        sources=sources,
        additional_instruction=f"Examples that qualify include: expedited job posting/Handshake support, logo placement, targeted email campaigns, social media posts, resume books/candidate screening, exclusive networking receptions, or complimentary registrations. The answer mentioned: {benefits_preview}. Verify that at least two such exclusive benefits are indeed offered."
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

    # Extract up to 4 universities from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extracted"
    )

    universities: List[UniversityItem] = list(extracted.universities or [])

    # Pad or trim to exactly 4 entries for evaluation
    if len(universities) > 4:
        universities = universities[:4]
    while len(universities) < 4:
        universities.append(UniversityItem())

    # Build tree and verify each university
    prev_items: List[UniversityItem] = []
    for idx, uni in enumerate(universities):
        await verify_university(evaluator, root, uni, idx, prev_items)
        prev_items.append(uni)

    return evaluator.get_summary()