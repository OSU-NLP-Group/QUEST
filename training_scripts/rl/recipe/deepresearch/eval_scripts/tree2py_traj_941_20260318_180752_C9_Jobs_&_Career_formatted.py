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
TASK_ID = "educational_leaders_2022_2025"
TASK_DESCRIPTION = """
Identify three educational leaders who satisfy all of the following criteria:

1. The leader holds a position as either (a) a university president or vice-chancellor, or (b) a public K-12 school district superintendent.

2. The leader was officially appointed to their current position with an appointment announcement made between January 1, 2022 and December 31, 2025.

3. The leader holds a doctoral degree (PhD, EdD, or equivalent).

4. The leader received a named professional award or recognition from a national or state-level professional organization in either 2024 or 2025.

5. For university presidents: the leader must have previously served as a provost, vice president, or senior academic administrator at a university immediately before the current appointment. For school superintendents: the leader must have previously served as a superintendent of a different public school district immediately before the current appointment.

6. The appointment includes a specified start date.

7. The name of the previous institution or school district where the leader worked immediately before the current appointment must be explicitly identifiable from official sources.

For each of the three educational leaders you identify, provide:
- Full name
- Current position title and institution/district name
- Date of appointment announcement (month and year minimum)
- Start date for the new position
- Doctoral degree held and field
- Previous position title and institution/district name
- Name of the award or recognition received in 2024 or 2025
- Name of the organization that granted the award
- Reference URL(s) supporting the above information
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LeaderItem(BaseModel):
    # Identity and current role
    name: Optional[str] = None
    position_title: Optional[str] = None
    institution_name: Optional[str] = None
    # institution_type: "university" | "public_k12_district" | "other"
    institution_type: Optional[str] = None
    # position_category: "university_president" | "vice_chancellor" | "k12_superintendent" | "other"
    position_category: Optional[str] = None

    # Appointment information
    appointment_announcement_date: Optional[str] = None  # e.g., "March 2024"
    start_date: Optional[str] = None  # e.g., "July 1, 2024"

    # Education (doctoral)
    doctoral_degree: Optional[str] = None  # e.g., "PhD in Chemistry"
    doctoral_field: Optional[str] = None

    # Previous role
    prev_position_title: Optional[str] = None
    prev_institution_name: Optional[str] = None
    prev_institution_type: Optional[str] = None  # university | public_k12_district | other

    # Award
    award_name: Optional[str] = None
    award_org: Optional[str] = None
    award_year: Optional[str] = None  # Keep string for flexibility

    # Source URLs
    appointment_urls: List[str] = Field(default_factory=list)
    degree_urls: List[str] = Field(default_factory=list)
    award_urls: List[str] = Field(default_factory=list)
    prev_role_urls: List[str] = Field(default_factory=list)
    institution_urls: List[str] = Field(default_factory=list)
    # Optional general references if the answer provides them
    other_urls: List[str] = Field(default_factory=list)


class LeadersExtraction(BaseModel):
    leaders: List[LeaderItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_leaders() -> str:
    return """
    Extract up to three educational leaders mentioned in the answer that match the task criteria.
    For each leader, extract the following fields exactly as written in the answer:
      - name
      - position_title (e.g., "President", "Vice-Chancellor", "Superintendent")
      - institution_name (university or school district)
      - institution_type: one of ["university", "public_k12_district", "other"] (infer from the answer if possible)
      - position_category: one of ["university_president", "vice_chancellor", "k12_superintendent", "other"] (infer from the answer)
      - appointment_announcement_date (month and year minimum, as written, e.g., "March 2024")
      - start_date (as stated, e.g., "July 1, 2024", "effective August 2023", etc.)
      - doctoral_degree (e.g., "PhD in Biology", "EdD in Educational Leadership")
      - doctoral_field (if explicitly available; otherwise null)
      - prev_position_title (immediately before current appointment)
      - prev_institution_name (immediately before current appointment)
      - prev_institution_type: one of ["university", "public_k12_district", "other"] if available
      - award_name (a named professional award/recognition received in 2024 or 2025)
      - award_org (the organization that granted the award)
      - award_year (as written, prefer 4-digit year string like "2024" or "2025")
      - appointment_urls: list of URLs that confirm the appointment announcement and start date (ideally official sources such as .edu domains or district websites)
      - degree_urls: list of URLs that confirm the doctoral degree (official CV/biography pages or institutional pages preferred)
      - award_urls: list of URLs that confirm the award/recognition and year
      - prev_role_urls: list of URLs that confirm the immediately-previous role/institution
      - institution_urls: list of URLs that confirm the current institution type, if available
      - other_urls: list of any additional relevant reference URLs cited in the answer

    Rules:
      - Only include the first three leaders that the answer attempts to provide; if more are present, keep only the first three in their original order.
      - If any item is missing, set the corresponding field to null or an empty list for URLs.
      - Do NOT invent URLs; only extract URLs explicitly present (plain or markdown link).
    Return a JSON object with a single key "leaders" that is an array of leader objects.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _first_second_third(idx: int) -> str:
    return ["First", "Second", "Third"][idx] if 0 <= idx < 3 else f"Leader {idx+1}"


def _combine_sources(*url_lists: List[str]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for urls in url_lists:
        for u in urls or []:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


def _leader_kind_from_extraction(item: LeaderItem) -> str:
    """
    Return 'university' if university president/vice-chancellor;
    'k12' if K-12 superintendent; otherwise 'unknown'.
    """
    cat = (item.position_category or "").lower()
    title = (item.position_title or "").lower()
    inst_type = (item.institution_type or "").lower()

    if "superintendent" in title or cat == "k12_superintendent" or inst_type == "public_k12_district":
        return "k12"
    if any(k in title for k in ["president", "chancellor", "vice-chancellor"]) or \
            cat in ("university_president", "vice_chancellor") or inst_type == "university":
        return "university"
    return "unknown"


# --------------------------------------------------------------------------- #
# Verification logic per leader                                               #
# --------------------------------------------------------------------------- #
async def verify_single_leader(evaluator: Evaluator, parent, item: LeaderItem, idx: int) -> None:
    """
    Build the full verification subtree for one leader following the rubric.
    All verification leaves are binary; missing URLs explicitly fail relevant leaves.
    """
    lid = idx + 1
    leader_node = evaluator.add_parallel(
        id=f"Leader_{lid}",
        desc=f"{_first_second_third(idx)} educational leader meeting all criteria",
        parent=parent,
        critical=False  # Parent allows partial credit across leaders
    )

    # GROUP 1: Appointment Information (critical)
    appt_group = evaluator.add_parallel(
        id=f"L{lid}_Appointment_Information",
        desc=f"Appointment details for Leader {lid}",
        parent=leader_node,
        critical=True
    )

    # 1.A Position Verification (critical)
    pos_ver_group = evaluator.add_parallel(
        id=f"L{lid}_Position_Verification",
        desc="Verify position type and appointment timing",
        parent=appt_group,
        critical=True
    )

    # Lx_Position_Type (leaf)
    l_pos_type = evaluator.add_leaf(
        id=f"L{lid}_Position_Type",
        desc="Verify position is university president/vice-chancellor OR school superintendent",
        parent=pos_ver_group,
        critical=True
    )
    pos_sources = _combine_sources(item.appointment_urls, item.institution_urls, item.other_urls)
    pos_claim = (
        f"{item.name or 'The leader'} holds the position '{item.position_title or ''}' at "
        f"'{item.institution_name or ''}', which must qualify as either: "
        f"(a) a university president/vice-chancellor (or equivalent such as 'Chancellor'), "
        f"or (b) a public K-12 school district superintendent."
    )
    if pos_sources:
        await evaluator.verify(
            claim=pos_claim,
            node=l_pos_type,
            sources=pos_sources,
            additional_instruction=(
                "Judge based on the role and the institution: "
                "- 'President', 'Chancellor', or 'Vice-Chancellor' at a university qualifies; "
                "- 'Superintendent' at a public K-12 school district qualifies. "
                "Use the page content and domain context (e.g., .edu or district sites) to decide."
            )
        )
    else:
        l_pos_type.score = 0.0
        l_pos_type.status = "failed"

    # Lx_Appointment_Announcement_Date (leaf)
    l_appt_window = evaluator.add_leaf(
        id=f"L{lid}_Appointment_Announcement_Date",
        desc="Verify appointment was announced between January 1, 2022 and December 31, 2025",
        parent=pos_ver_group,
        critical=True
    )
    if item.appointment_urls:
        await evaluator.verify(
            claim=(
                f"The official appointment announcement for {item.name or 'the leader'}'s role at "
                f"{item.institution_name or 'the institution'} was made between January 1, 2022 and "
                f"December 31, 2025 (inclusive)."
            ),
            node=l_appt_window,
            sources=item.appointment_urls,
            additional_instruction=(
                "Use the press release or official announcement's published/posted date as the "
                "announcement date. If multiple dates appear, use the article/publication date. "
                "Pass only if the date clearly falls within 2022-01-01 to 2025-12-31."
            )
        )
    else:
        l_appt_window.score = 0.0
        l_appt_window.status = "failed"

    # 1.B Start Date Verification (critical)
    start_group = evaluator.add_parallel(
        id=f"L{lid}_Start_Date_Verification",
        desc="Verify start date is specified and documented",
        parent=appt_group,
        critical=True
    )

    # Lx_Start_Date_Specified (leaf)
    l_start_spec = evaluator.add_leaf(
        id=f"L{lid}_Start_Date_Specified",
        desc="Verify start date is specified",
        parent=start_group,
        critical=True
    )
    if item.appointment_urls:
        await evaluator.verify(
            claim=(
                f"The appointment announcement or official source specifies a start date for "
                f"{item.name or 'the leader'}'s new position at {item.institution_name or 'the institution'}."
            ),
            node=l_start_spec,
            sources=item.appointment_urls,
            additional_instruction=(
                "Look for phrases like 'starts', 'effective', 'beginning', or a stated date/month-year. "
                "Pass only if a start date is explicitly indicated."
            )
        )
    else:
        l_start_spec.score = 0.0
        l_start_spec.status = "failed"

    # Lx_Appointment_URL (leaf)
    l_appt_url = evaluator.add_leaf(
        id=f"L{lid}_Appointment_URL",
        desc="Verify appointment information with official source URL",
        parent=start_group,
        critical=True
    )
    if item.appointment_urls:
        await evaluator.verify(
            claim=(
                f"At least one of the provided appointment URLs is an official or authoritative source "
                f"(e.g., university .edu site, the school district's official website, or an official press release) "
                f"that confirms {item.name or 'the leader'}'s appointment."
            ),
            node=l_appt_url,
            sources=item.appointment_urls,
            additional_instruction=(
                "Prefer official domains (e.g., .edu, .k12., district.gov). Pass if the page confirms the appointment."
            )
        )
    else:
        l_appt_url.score = 0.0
        l_appt_url.status = "failed"

    # GROUP 2: Educational Background (critical)
    edu_group = evaluator.add_parallel(
        id=f"L{lid}_Educational_Background",
        desc=f"Educational qualifications for Leader {lid}",
        parent=leader_node,
        critical=True
    )

    degree_group = evaluator.add_parallel(
        id=f"L{lid}_Degree_Verification",
        desc="Verify doctoral degree and documentation",
        parent=edu_group,
        critical=True
    )

    # Lx_Doctoral_Degree (leaf)
    l_doctoral = evaluator.add_leaf(
        id=f"L{lid}_Doctoral_Degree",
        desc="Verify leader holds a doctoral degree (PhD, EdD, or equivalent)",
        parent=degree_group,
        critical=True
    )
    degree_sources = _combine_sources(item.degree_urls, item.appointment_urls, item.other_urls)
    doc_claim = (
        f"{item.name or 'The leader'} holds a doctoral degree (PhD, EdD, or equivalent)"
        + (f", specifically '{item.doctoral_degree}'." if item.doctoral_degree else ".")
    )
    if degree_sources:
        await evaluator.verify(
            claim=doc_claim,
            node=l_doctoral,
            sources=degree_sources,
            additional_instruction=(
                "Accept doctoral variants such as PhD, EdD, DPhil, ScD, DrPH, or equivalent. "
                "The source must clearly indicate a doctoral degree."
            )
        )
    else:
        l_doctoral.score = 0.0
        l_doctoral.status = "failed"

    # Lx_Education_URL (leaf)
    l_edu_url = evaluator.add_leaf(
        id=f"L{lid}_Education_URL",
        desc="Verify educational background with source URL",
        parent=degree_group,
        critical=True
    )
    if degree_sources:
        await evaluator.verify(
            claim=(
                f"At least one provided URL confirms the doctoral degree information for {item.name or 'the leader'}."
            ),
            node=l_edu_url,
            sources=degree_sources,
            additional_instruction="Prefer official biographies, CVs, or institutional pages that state the doctoral degree."
        )
    else:
        l_edu_url.score = 0.0
        l_edu_url.status = "failed"

    # GROUP 3: Recognition in 2024-2025 (critical)
    award_group = evaluator.add_parallel(
        id=f"L{lid}_Recognition_2024_2025",
        desc="Professional award or recognition in 2024-2025",
        parent=leader_node,
        critical=True
    )

    award_details = evaluator.add_parallel(
        id=f"L{lid}_Award_Details",
        desc="Verify award name and timing",
        parent=award_group,
        critical=True
    )

    # Lx_Award_Name (leaf)
    l_award_name = evaluator.add_leaf(
        id=f"L{lid}_Award_Name",
        desc="Verify specific name of award or recognition received",
        parent=award_details,
        critical=True
    )
    if item.award_urls:
        await evaluator.verify(
            claim=f"{item.name or 'The leader'} received the award/recognition '{item.award_name or ''}'.",
            node=l_award_name,
            sources=item.award_urls,
            additional_instruction=(
                "Pass only if the page clearly names a specific professional award/recognition corresponding to the extracted name."
            )
        )
    else:
        l_award_name.score = 0.0
        l_award_name.status = "failed"

    # Lx_Award_Year (leaf)
    l_award_year = evaluator.add_leaf(
        id=f"L{lid}_Award_Year",
        desc="Verify award was received in 2024 or 2025",
        parent=award_details,
        critical=True
    )
    if item.award_urls:
        await evaluator.verify(
            claim=(
                f"The award/recognition for {item.name or 'the leader'} was received in 2024 or 2025."
            ),
            node=l_award_year,
            sources=item.award_urls,
            additional_instruction=(
                "Use the award page or announcement date/award year listed. "
                "Pass only if the evidence supports 2024 or 2025 as the award year."
            )
        )
    else:
        l_award_year.score = 0.0
        l_award_year.status = "failed"

    # Award source verification (critical)
    award_src_group = evaluator.add_parallel(
        id=f"L{lid}_Award_Source_Verification",
        desc="Verify award source and documentation",
        parent=award_group,
        critical=True
    )

    # Lx_Granting_Organization (leaf)
    l_award_org = evaluator.add_leaf(
        id=f"L{lid}_Granting_Organization",
        desc="Verify award was from a national or state-level professional organization",
        parent=award_src_group,
        critical=True
    )
    if item.award_urls:
        await evaluator.verify(
            claim=(
                f"The granting organization '{item.award_org or ''}' is a national- or state-level professional organization."
            ),
            node=l_award_org,
            sources=item.award_urls,
            additional_instruction=(
                "Check the organization's scope using the award page (and linked 'About' if available). "
                "Pass if it is clearly national (multi-state) or state-level (official state professional association). "
                "Do not pass for local/city-only or internal institutional awards."
            )
        )
    else:
        l_award_org.score = 0.0
        l_award_org.status = "failed"

    # Lx_Award_URL (leaf)
    l_award_url = evaluator.add_leaf(
        id=f"L{lid}_Award_URL",
        desc="Verify award information with source URL",
        parent=award_src_group,
        critical=True
    )
    if item.award_urls:
        await evaluator.verify(
            claim=(
                f"At least one provided URL confirms the award's name and that {item.name or 'the leader'} received it."
            ),
            node=l_award_url,
            sources=item.award_urls,
            additional_instruction="Prefer the professional organization's own page or official press releases."
        )
    else:
        l_award_url.score = 0.0
        l_award_url.status = "failed"

    # GROUP 4: Previous Position (critical)
    prev_group = evaluator.add_parallel(
        id=f"L{lid}_Previous_Position",
        desc="Previous leadership role before current appointment",
        parent=leader_node,
        critical=True
    )

    prev_role_ver = evaluator.add_parallel(
        id=f"L{lid}_Previous_Role_Verification",
        desc="Verify previous position details and qualifications",
        parent=prev_group,
        critical=True
    )

    # Lx_Previous_Role_Title (leaf)
    l_prev_title = evaluator.add_leaf(
        id=f"L{lid}_Previous_Role_Title",
        desc="Verify previous position title (provost/VP for universities, or superintendent for K-12)",
        parent=prev_role_ver,
        critical=True
    )
    leader_kind = _leader_kind_from_extraction(item)
    prev_sources = _combine_sources(item.prev_role_urls, item.appointment_urls, item.other_urls)
    if prev_sources:
        if leader_kind == "university":
            claim_prev = (
                f"Immediately before this appointment, {item.name or 'the leader'} served as a provost, "
                f"vice president, or senior academic administrator at a university "
                f"(e.g., 'Provost', 'Executive Vice President and Provost', 'Senior Vice President for Academic Affairs')."
            )
        elif leader_kind == "k12":
            claim_prev = (
                f"Immediately before this appointment, {item.name or 'the leader'} served as a superintendent "
                f"of a different public school district."
            )
        else:
            claim_prev = (
                f"Immediately before this appointment, {item.name or 'the leader'} held a qualifying prior role "
                f"that matches the requirement based on whether this is a university or K-12 district role."
            )

        await evaluator.verify(
            claim=claim_prev,
            node=l_prev_title,
            sources=prev_sources,
            additional_instruction=(
                "Look for phrases like 'most recently', 'immediately prior to', or similar. "
                "For university leaders, the prior role must be provost/VP/senior academic admin at a university. "
                "For K-12 superintendents, the prior role must be superintendent of a different public school district. "
                f"The current institution/district is '{item.institution_name or ''}' for comparison."
            )
        )
    else:
        l_prev_title.score = 0.0
        l_prev_title.status = "failed"

    # Lx_Previous_Institution_Name (leaf)
    l_prev_name = evaluator.add_leaf(
        id=f"L{lid}_Previous_Institution_Name",
        desc="Verify name of previous institution or district",
        parent=prev_role_ver,
        critical=True
    )
    if prev_sources:
        await evaluator.verify(
            claim=(
                f"The immediately previous institution/district for {item.name or 'the leader'} was "
                f"'{item.prev_institution_name or ''}'."
            ),
            node=l_prev_name,
            sources=prev_sources,
            additional_instruction="Pass only if the previous institution/district name is explicitly identifiable on the page."
        )
    else:
        l_prev_name.score = 0.0
        l_prev_name.status = "failed"

    # Previous Position Documentation (critical)
    prev_doc_group = evaluator.add_parallel(
        id=f"L{lid}_Previous_Position_Documentation",
        desc="Verify previous position documentation",
        parent=prev_group,
        critical=True
    )

    # Lx_Previous_Position_URL (leaf)
    l_prev_url = evaluator.add_leaf(
        id=f"L{lid}_Previous_Position_URL",
        desc="Verify previous position information with source URL",
        parent=prev_doc_group,
        critical=True
    )
    if prev_sources:
        await evaluator.verify(
            claim=(
                f"At least one provided URL confirms the immediately-previous role and institution for "
                f"{item.name or 'the leader'}."
            ),
            node=l_prev_url,
            sources=prev_sources,
            additional_instruction="The evidence should clearly tie the leader to the previous role and organization."
        )
    else:
        l_prev_url.score = 0.0
        l_prev_url.status = "failed"

    # GROUP 5: Current Institution Characteristics (critical)
    inst_group = evaluator.add_parallel(
        id=f"L{lid}_Current_Institution_Characteristics",
        desc="Characteristics of current appointing institution",
        parent=leader_node,
        critical=True
    )

    inst_ver = evaluator.add_parallel(
        id=f"L{lid}_Institution_Verification",
        desc="Verify institution type and documentation",
        parent=inst_group,
        critical=True
    )

    # Lx_Institution_Type (leaf)
    l_inst_type = evaluator.add_leaf(
        id=f"L{lid}_Institution_Type",
        desc="Verify institution type (university OR public K-12 district)",
        parent=inst_ver,
        critical=True
    )
    inst_sources = _combine_sources(item.institution_urls, item.appointment_urls, item.other_urls)
    if inst_sources:
        await evaluator.verify(
            claim=(
                f"{item.institution_name or 'The institution'} is either a university or a public K-12 school district."
            ),
            node=l_inst_type,
            sources=inst_sources,
            additional_instruction=(
                "Pass if the institution is clearly a higher-education university/college (degree-granting) "
                "or a public K-12 school district (e.g., USD/ISD). Use official site context."
            )
        )
    else:
        l_inst_type.score = 0.0
        l_inst_type.status = "failed"

    # Lx_Institution_Verification_URL (leaf)
    l_inst_url = evaluator.add_leaf(
        id=f"L{lid}_Institution_Verification_URL",
        desc="Verify institution characteristics with source URL",
        parent=inst_ver,
        critical=True
    )
    if inst_sources:
        await evaluator.verify(
            claim=(
                f"At least one provided URL confirms the institution type for {item.institution_name or 'the institution'}."
            ),
            node=l_inst_url,
            sources=inst_sources,
            additional_instruction="Prefer the institution's official website or authoritative listings."
        )
    else:
        l_inst_url.score = 0.0
        l_inst_url.status = "failed"


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
    Evaluate an answer for the educational leaders 2022–2025 task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Leaders evaluated independently
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
    # Note: Root is intentionally non-critical to allow partial credit across leaders
    # due to framework constraint that critical parents must have only critical children.

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_leaders(),
        template_class=LeadersExtraction,
        extraction_name="leaders_extraction",
    )

    # Normalize to exactly 3 leaders (pad with empty if fewer)
    leaders: List[LeaderItem] = list(extracted.leaders or [])
    while len(leaders) < 3:
        leaders.append(LeaderItem())
    leaders = leaders[:3]

    # Build verification tree for each leader
    for i, leader in enumerate(leaders[:3]):
        await verify_single_leader(evaluator, root, leader, i)

    # Return structured summary
    return evaluator.get_summary()