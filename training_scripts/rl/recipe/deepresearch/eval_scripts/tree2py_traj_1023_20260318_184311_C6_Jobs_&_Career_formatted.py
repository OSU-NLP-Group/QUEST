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
TASK_ID = "career_services_positions_three_cases"
TASK_DESCRIPTION = """
You are helping a higher education consulting firm compile examples of career services positions that demonstrate different professional standards and program scales. Identify three currently posted or recently posted (within the past 12 months) career services positions at U.S. universities that collectively meet the following criteria:

Position 1: A career counselor or career advisor position that:
- Is located at a university in the United States
- Requires or strongly prefers a master's degree
- Lists GCDF (Global Career Development Facilitator) certification as a required or preferred qualification

Position 2: A director-level career services position (Director, Associate/Assistant Director, or Executive Director) that:
- Is located at a university in the United States
- Explicitly requires (not just prefers) a master's degree
- Makes reference to NACE (National Association of Colleges and Employers) professional standards or competencies in the job description or institutional career services materials

Position 3: An internship coordinator or internship program manager position that:
- Is located at a university in the United States
- Requires a minimum of a bachelor's degree
- Manages or coordinates an internship program that serves at least 500 students or facilitates at least 500 internship placements annually

Additional Requirement:
The three positions must collectively represent at least three different U.S. states.

For each position, provide:
- The specific job title
- The university name and state
- A description of how it meets the specified criteria
- A valid reference URL (job posting, university career services page, or institutional document)
"""

CURRENT_DATE_ISO = "2026-03-22"


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
US_STATE_MAP = {
    # Full name -> 2-letter code
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "washington, dc": "DC", "washington dc": "DC", "dc": "DC",
}

US_STATE_CODES = set(US_STATE_MAP.values()) | {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"
}


def normalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip()
    if not s:
        return None
    s_lower = s.lower()
    # Already a two-letter code
    if len(s) == 2 and s.upper() in US_STATE_CODES:
        return s.upper()
    # Try map
    if s_lower in US_STATE_MAP:
        return US_STATE_MAP[s_lower]
    # Try remove punctuation
    s_lower2 = s_lower.replace(".", "").replace(",", "").strip()
    if s_lower2 in US_STATE_MAP:
        return US_STATE_MAP[s_lower2]
    if len(s) == 2 and s.isalpha():
        return s.upper()
    return s  # fallback as-is


def filter_valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    seen = set()
    for u in urls:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if not uu:
            continue
        # Basic validity
        if uu.startswith("http://") or uu.startswith("https://"):
            if uu not in seen:
                seen.add(uu)
                out.append(uu)
    return out


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    title: Optional[str] = None
    university: Optional[str] = None
    state: Optional[str] = None
    urls: List[str] = Field(default_factory=list)

    # Helpful textual fields extracted from the answer (strings preferred per guidance)
    degree_requirement_summary: Optional[str] = None   # e.g., "Master's degree required" or "Master's preferred"
    role_summary: Optional[str] = None                 # brief description of role nature
    certification_keywords: List[str] = Field(default_factory=list)  # e.g., ["GCDF", "NACE"]
    program_scale_claim: Optional[str] = None          # e.g., "Program serves 800 students annually"
    posting_date_text: Optional[str] = None            # e.g., "Posted Jan 15, 2026" / "Updated Sep 2025"
    notes: Optional[str] = None


class PositionsExtraction(BaseModel):
    position1: Optional[PositionItem] = None
    position2: Optional[PositionItem] = None
    position3: Optional[PositionItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return f"""
You will extract structured details about three positions described in the answer. Extract ONLY what is explicitly present in the answer text.

For each of the three positions (position1, position2, position3), extract:
- title: The exact job title
- university: The university name
- state: The U.S. state for the job's location (prefer the 2-letter postal code if provided; otherwise use the full state name; if unknown, return null)
- urls: Array of ALL URLs the answer cites for that position (job posting, university career services, or institutional document)
- degree_requirement_summary: The exact phrasing in the answer about minimum degree level and whether it is required or preferred (do not infer)
- role_summary: Short phrase from the answer summarizing the role nature (e.g., career counselor/advisor; director-level; internship coordinator/manager)
- certification_keywords: Array of any certification or standards acronyms/titles mentioned for the position (e.g., "GCDF", "Global Career Development Facilitator", "NACE")
- program_scale_claim: If the answer states the internship program size (e.g., "serves 500+ students"), copy that phrase here; otherwise null
- posting_date_text: Any posting/updated/closing date or recency indicator mentioned in the answer (verbatim), else null
- notes: Any other clarifying note from the answer (optional)

Return a JSON object with three top-level fields: position1, position2, position3. If any field is missing in the answer, set it to null or an empty list accordingly.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_position_1(evaluator: Evaluator, parent_node, p: Optional[PositionItem]) -> None:
    # Container node for Position 1
    p1_node = evaluator.add_parallel(
        id="Position_1",
        desc="Career counselor position requiring GCDF certification",
        parent=parent_node,
        critical=False
    )

    # Normalize data
    urls = filter_valid_urls(p.urls if p else [])
    state_norm = normalize_state(p.state if p else None)
    uni = (p.university or "").strip() if p else ""
    title = (p.title or "").strip() if p else ""

    # Reference URL presence (critical gating)
    p1_url_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id="P1_Reference_URL",
        desc="Valid reference URL provided for position 1",
        parent=p1_node,
        critical=True
    )

    # US Location at a university
    p1_loc = evaluator.add_leaf(
        id="P1_US_Location",
        desc="Position is at a university in the United States",
        parent=p1_node,
        critical=True
    )
    claim_loc = (
        f"This page is a job posting for a position at a U.S. university. "
        f"The employer is a university and the position is located in the United States"
        + (f", specifically in the state of {state_norm}." if state_norm else ".")
    )
    await evaluator.verify(
        claim=claim_loc,
        node=p1_loc,
        sources=urls,
        additional_instruction="Accept if the page clearly indicates a U.S. university employer (.edu domain, campus info) and a U.S. location (city-state). If a specific state was provided in the answer, treat it as the expected state."
    )

    # Role is Career Counselor/Advisor
    p1_role = evaluator.add_leaf(
        id="P1_Career_Services_Role",
        desc="Position is a career counselor or career advisor role",
        parent=p1_node,
        critical=True
    )
    claim_role = (
        "The job is a career services counseling/advising role (e.g., Career Counselor, Career Advisor) "
        "that provides career guidance/advising to students or alumni."
    )
    await evaluator.verify(
        claim=claim_role,
        node=p1_role,
        sources=urls,
        additional_instruction="Look for title keywords like 'Career Counselor' or 'Career Advisor' and responsibilities describing career counseling/advising."
    )

    # Master's degree required or strongly preferred
    p1_masters = evaluator.add_leaf(
        id="P1_Masters_Degree",
        desc="Position requires or strongly prefers a master's degree",
        parent=p1_node,
        critical=True
    )
    claim_masters = "The posting states that a master's degree is required or strongly preferred."
    await evaluator.verify(
        claim=claim_masters,
        node=p1_masters,
        sources=urls,
        additional_instruction="Confirm explicit language indicating 'master’s degree required' or 'master’s degree preferred/strongly preferred'."
    )

    # GCDF certification required or preferred
    p1_gcdf = evaluator.add_leaf(
        id="P1_GCDF_Certification",
        desc="Position lists GCDF certification as required or preferred qualification",
        parent=p1_node,
        critical=True
    )
    claim_gcdf = "The posting lists 'GCDF' (Global Career Development Facilitator) certification as a required or preferred qualification."
    await evaluator.verify(
        claim=claim_gcdf,
        node=p1_gcdf,
        sources=urls,
        additional_instruction="Accept if 'GCDF' or 'Global Career Development Facilitator' is mentioned among required or preferred qualifications."
    )

    # Recency within past 12 months
    p1_recent = evaluator.add_leaf(
        id="P1_Recency",
        desc="Position was posted currently or within the past 12 months",
        parent=p1_node,
        critical=True
    )
    claim_recent = f"This posting is from within the past 12 months relative to {CURRENT_DATE_ISO}."
    await evaluator.verify(
        claim=claim_recent,
        node=p1_recent,
        sources=urls,
        additional_instruction="Use posted/updated/closing/review-begins dates. If multiple dates appear, use the most recent relevant date. If indeterminable, mark as not supported."
    )


async def verify_position_2(evaluator: Evaluator, parent_node, p: Optional[PositionItem]) -> None:
    # Container node for Position 2
    p2_node = evaluator.add_parallel(
        id="Position_2",
        desc="Director-level position with NACE standards adherence",
        parent=parent_node,
        critical=False
    )

    urls = filter_valid_urls(p.urls if p else [])
    state_norm = normalize_state(p.state if p else None)
    uni = (p.university or "").strip() if p else ""
    title = (p.title or "").strip() if p else ""

    # Reference URL presence (critical gating)
    p2_url_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id="P2_Reference_URL",
        desc="Valid reference URL provided for position 2",
        parent=p2_node,
        critical=True
    )

    # US Location at a university
    p2_loc = evaluator.add_leaf(
        id="P2_US_Location",
        desc="Position is at a university in the United States",
        parent=p2_node,
        critical=True
    )
    claim_loc = (
        f"This page is a job posting for a position at a U.S. university. "
        f"The employer is a university and the position is located in the United States"
        + (f", specifically in the state of {state_norm}." if state_norm else ".")
    )
    await evaluator.verify(
        claim=claim_loc,
        node=p2_loc,
        sources=urls,
        additional_instruction="Accept if the page clearly indicates a U.S. university employer (.edu domain, campus info) and a U.S. location (city-state). If a specific state was provided in the answer, treat it as the expected state."
    )

    # Director-level role
    p2_dir = evaluator.add_leaf(
        id="P2_Director_Level",
        desc="Position is at director level or above (Director, Associate/Assistant Director, or Executive Director)",
        parent=p2_node,
        critical=True
    )
    claim_dir = (
        "The job is a director-level career services role (Director, Associate Director, Assistant Director, or Executive Director)."
    )
    await evaluator.verify(
        claim=claim_dir,
        node=p2_dir,
        sources=urls,
        additional_instruction="Check title and description for director-level seniority; acceptable variants include 'Associate Director', 'Assistant Director', or 'Executive Director'."
    )

    # Master's explicitly required (not just preferred)
    p2_master_req = evaluator.add_leaf(
        id="P2_Masters_Required",
        desc="Position explicitly requires (not just prefers) a master's degree",
        parent=p2_node,
        critical=True
    )
    claim_mreq = "The posting explicitly requires a master's degree (i.e., it is a requirement, not merely preferred)."
    await evaluator.verify(
        claim=claim_mreq,
        node=p2_master_req,
        sources=urls,
        additional_instruction="Look for explicit 'required' language tied to a master's degree; 'preferred' alone is insufficient."
    )

    # NACE reference present (posting or institutional materials)
    p2_nace = evaluator.add_leaf(
        id="P2_NACE_Reference",
        desc="Position or institutional materials reference NACE professional standards or competencies",
        parent=p2_node,
        critical=True
    )
    claim_nace = (
        "The job posting or the institution's linked career services materials explicitly reference NACE "
        "(National Association of Colleges and Employers) standards or competencies."
    )
    await evaluator.verify(
        claim=claim_nace,
        node=p2_nace,
        sources=urls,
        additional_instruction="Accept mentions of 'NACE' or 'National Association of Colleges and Employers', including references to NACE competencies or standards on the job page or directly linked institutional materials."
    )

    # Recency within past 12 months
    p2_recent = evaluator.add_leaf(
        id="P2_Recency",
        desc="Position was posted currently or within the past 12 months",
        parent=p2_node,
        critical=True
    )
    claim_recent = f"This posting is from within the past 12 months relative to {CURRENT_DATE_ISO}."
    await evaluator.verify(
        claim=claim_recent,
        node=p2_recent,
        sources=urls,
        additional_instruction="Use posted/updated/closing/review-begins dates. If multiple dates appear, use the most recent relevant date. If indeterminable, mark as not supported."
    )


async def verify_position_3(evaluator: Evaluator, parent_node, p: Optional[PositionItem]) -> None:
    # Container node for Position 3
    p3_node = evaluator.add_parallel(
        id="Position_3",
        desc="Internship coordinator for program serving 500+ students",
        parent=parent_node,
        critical=False
    )

    urls = filter_valid_urls(p.urls if p else [])
    state_norm = normalize_state(p.state if p else None)
    uni = (p.university or "").strip() if p else ""
    title = (p.title or "").strip() if p else ""

    # Reference URL presence (critical gating)
    p3_url_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id="P3_Reference_URL",
        desc="Valid reference URL provided for position 3",
        parent=p3_node,
        critical=True
    )

    # US Location at a university
    p3_loc = evaluator.add_leaf(
        id="P3_US_Location",
        desc="Position is at a university in the United States",
        parent=p3_node,
        critical=True
    )
    claim_loc = (
        f"This page is a job posting for a position at a U.S. university. "
        f"The employer is a university and the position is located in the United States"
        + (f", specifically in the state of {state_norm}." if state_norm else ".")
    )
    await evaluator.verify(
        claim=claim_loc,
        node=p3_loc,
        sources=urls,
        additional_instruction="Accept if the page clearly indicates a U.S. university employer (.edu domain, campus info) and a U.S. location (city-state). If a specific state was provided in the answer, treat it as the expected state."
    )

    # Internship coordinator/manager role
    p3_role = evaluator.add_leaf(
        id="P3_Internship_Role",
        desc="Position is an internship coordinator or internship program manager",
        parent=p3_node,
        critical=True
    )
    claim_role = (
        "The job is an internship coordination/management role (e.g., Internship Coordinator, Internship Program Manager), "
        "responsible for coordinating or managing internship programs."
    )
    await evaluator.verify(
        claim=claim_role,
        node=p3_role,
        sources=urls,
        additional_instruction="Look for internship-focused responsibilities; accept synonyms like 'Experiential Learning Coordinator' if duties clearly center on internships."
    )

    # Requires at least a bachelor's degree
    p3_bach = evaluator.add_leaf(
        id="P3_Bachelors_Minimum",
        desc="Position requires a minimum of a bachelor's degree",
        parent=p3_node,
        critical=True
    )
    claim_bach = "The posting requires at least a bachelor's degree (B.A./B.S. or equivalent) as a minimum qualification."
    await evaluator.verify(
        claim=claim_bach,
        node=p3_bach,
        sources=urls,
        additional_instruction="Confirm explicit 'required' language tied to a bachelor's or higher degree; 'preferred' alone is insufficient."
    )

    # Program scale: serves >= 500 students or >= 500 placements annually
    p3_scale = evaluator.add_leaf(
        id="P3_Program_Scale",
        desc="Position manages or coordinates an internship program serving at least 500 students or facilitating at least 500 placements annually",
        parent=p3_node,
        critical=True
    )
    claim_scale = (
        "The job posting or linked institutional program materials state that the internship program serves at least 500 students "
        "or facilitates at least 500 internship placements annually."
    )
    await evaluator.verify(
        claim=claim_scale,
        node=p3_scale,
        sources=urls,
        additional_instruction="Accept phrasing like '500+', 'at least 500', '≥ 500', or specific counts >= 500. Evidence may appear on a linked program page rather than the posting."
    )

    # Recency within past 12 months
    p3_recent = evaluator.add_leaf(
        id="P3_Recency",
        desc="Position was posted currently or within the past 12 months",
        parent=p3_node,
        critical=True
    )
    claim_recent = f"This posting is from within the past 12 months relative to {CURRENT_DATE_ISO}."
    await evaluator.verify(
        claim=claim_recent,
        node=p3_recent,
        sources=urls,
        additional_instruction="Use posted/updated/closing/review-begins dates. If multiple dates appear, use the most recent relevant date. If indeterminable, mark as not supported."
    )


def collect_unique_states(extracted: PositionsExtraction) -> List[str]:
    states = []
    for p in [extracted.position1, extracted.position2, extracted.position3]:
        if p and p.state:
            s = normalize_state(p.state)
            if s:
                states.append(s)
    # Keep order but unique
    seen = set()
    out = []
    for s in states:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


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
    Evaluate an answer for the career services positions task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    # IMPORTANT: Root set to non-critical to satisfy framework constraint (critical parents must have all-critical children)
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

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction"
    )

    # Build verification subtrees
    await verify_position_1(evaluator, root, extracted.position1)
    await verify_position_2(evaluator, root, extracted.position2)
    await verify_position_3(evaluator, root, extracted.position3)

    # Additional Requirement: Geographic distribution across at least 3 different U.S. states (Critical)
    unique_states = collect_unique_states(extracted)
    geo_ok = len(unique_states) >= 3
    evaluator.add_custom_info(
        info={"unique_states": unique_states, "count": len(unique_states)},
        info_type="geographic_distribution",
        info_name="geo_distribution_check"
    )

    evaluator.add_custom_node(
        result=geo_ok,
        id="Geographic_Distribution",
        desc="Positions represent at least three different U.S. states",
        parent=root,
        critical=True
    )

    # Return summary
    return evaluator.get_summary()