import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "officials_2026_events"
TASK_DESCRIPTION = """
In early 2026, several major political transitions and cultural events occurred that drew international attention. Identify four specific government and cultural officials who played key roles in these events by providing the following information for each:

Official A: The Venezuelan government official who was sworn in as acting president on January 5, 2026, following the US military operation in Venezuela. Provide: (1) their full name, (2) the official title/position they held immediately before becoming acting president, (3) the exact date they were sworn in, and (4) their familial relationship to the person who serves as President of Venezuela's National Assembly.

Official B: The person who serves as President of Venezuela's National Assembly and was sworn in for a new term on January 5, 2026. Provide: (1) their full name, (2) their official title, (3) the exact date of their swearing-in for the 2026 term, and (4) the term period (start year and end year) for which they were sworn in.

Official C: The person who was appointed as the new CEO/Executive Director of the Kennedy Center for the Performing Arts in March 2026, replacing Richard Grenell. Provide: (1) their full name, (2) their new official title at the Kennedy Center, (3) the position they held at the Kennedy Center immediately before this appointment, and (4) the date when President Trump announced this appointment.

Official D: The person who physically accepted the 2025 Nobel Peace Prize on behalf of Venezuelan opposition leader María Corina Machado at the award ceremony. Provide: (1) their full name, (2) their relationship to María Corina Machado, (3) the exact date of the Nobel Peace Prize ceremony, and (4) the city where the ceremony took place.

For each official, provide valid reference URLs from web search results that support your answers.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OfficialAExtraction(BaseModel):
    full_name: Optional[str] = None
    name_sources: List[str] = Field(default_factory=list)

    previous_title: Optional[str] = None
    previous_title_sources: List[str] = Field(default_factory=list)

    swearing_in_date: Optional[str] = None
    swearing_in_sources: List[str] = Field(default_factory=list)

    family_relationship_to_assembly_president: Optional[str] = None
    relationship_sources: List[str] = Field(default_factory=list)


class OfficialBExtraction(BaseModel):
    full_name: Optional[str] = None
    name_sources: List[str] = Field(default_factory=list)

    title: Optional[str] = None
    title_sources: List[str] = Field(default_factory=list)

    swearing_in_date: Optional[str] = None
    swearing_in_sources: List[str] = Field(default_factory=list)

    term_period: Optional[str] = None  # e.g., "2026–2027", "2026-2028"
    term_sources: List[str] = Field(default_factory=list)


class OfficialCExtraction(BaseModel):
    full_name: Optional[str] = None
    name_sources: List[str] = Field(default_factory=list)

    new_title: Optional[str] = None
    new_title_sources: List[str] = Field(default_factory=list)

    previous_role: Optional[str] = None
    previous_role_sources: List[str] = Field(default_factory=list)

    announcement_date: Optional[str] = None
    announcement_sources: List[str] = Field(default_factory=list)


class OfficialDExtraction(BaseModel):
    full_name: Optional[str] = None
    name_sources: List[str] = Field(default_factory=list)

    relationship_to_machado: Optional[str] = None
    relationship_sources: List[str] = Field(default_factory=list)

    ceremony_date: Optional[str] = None
    ceremony_city: Optional[str] = None
    ceremony_sources: List[str] = Field(default_factory=list)


class OfficialsExtraction(BaseModel):
    official_a: Optional[OfficialAExtraction] = None
    official_b: Optional[OfficialBExtraction] = None
    official_c: Optional[OfficialCExtraction] = None
    official_d: Optional[OfficialDExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_officials() -> str:
    return """
    Extract structured details for four officials referenced in the answer. Return a single JSON object with keys:
      - official_a
      - official_b
      - official_c
      - official_d

    For each official, extract exactly the following fields. Use null for missing values, and arrays for URL lists. Only include URLs that are explicitly present in the answer text.

    official_a (Venezuela acting president sworn on Jan 5, 2026):
      - full_name: string
      - name_sources: array of URLs that explicitly confirm this person's identity in this context
      - previous_title: string — the official title held immediately before becoming acting president
      - previous_title_sources: array of URLs that explicitly confirm that previous title
      - swearing_in_date: string — the exact date they were sworn in
      - swearing_in_sources: array of URLs that explicitly confirm the swearing-in date
      - family_relationship_to_assembly_president: string — the familial relationship to the person serving as President of Venezuela's National Assembly (e.g., "son-in-law", "daughter", "spouse", etc.)
      - relationship_sources: array of URLs that explicitly confirm that family relationship

    official_b (President of Venezuela's National Assembly sworn in for a new term on Jan 5, 2026):
      - full_name: string
      - name_sources: array of URLs confirming this identity/role
      - title: string — the official title
      - title_sources: array of URLs confirming the title
      - swearing_in_date: string — the exact swearing-in date for the 2026 term
      - swearing_in_sources: array of URLs confirming the swearing-in date
      - term_period: string — a range stating the start year and end year (e.g., "2026-2027")
      - term_sources: array of URLs confirming the term period

    official_c (new Kennedy Center CEO/Executive Director appointed March 2026, replacing Richard Grenell):
      - full_name: string
      - name_sources: array of URLs confirming the identity/appointment
      - new_title: string — the new official title at the Kennedy Center
      - new_title_sources: array of URLs confirming the new title
      - previous_role: string — the role at the Kennedy Center immediately before this appointment
      - previous_role_sources: array of URLs confirming the previous role
      - announcement_date: string — the date when President Trump announced this appointment
      - announcement_sources: array of URLs confirming the announcement date

    official_d (person who physically accepted the 2025 Nobel Peace Prize on behalf of María Corina Machado):
      - full_name: string
      - name_sources: array of URLs confirming that this person physically accepted the prize on her behalf
      - relationship_to_machado: string — how the person is related to María Corina Machado
      - relationship_sources: array of URLs confirming that relationship
      - ceremony_date: string — the exact date of the Nobel Peace Prize ceremony
      - ceremony_city: string — the city where the ceremony took place
      - ceremony_sources: array of URLs confirming the ceremony date and city

    Rules:
    - Extract only what is explicitly written in the answer. Do not invent or infer missing URLs or values.
    - For any field without any source URLs mentioned in the answer, return an empty array for that field's URL list.
    - Preserve the exact text as it appears in the answer for names, titles, roles, dates, and relationships.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


def _date_is_jan_5_2026(date_str: Optional[str]) -> bool:
    if not _nonempty(date_str):
        return False
    s = date_str.strip().lower()

    # ISO-like patterns: 2026-01-05, 2026/01/05, 2026.01.05
    if re.search(r"\b2026[-/.]0?1[-/.]0?5\b", s):
        return True

    # Day-first or month-first numeric with separators: 05/01/2026 or 01/05/2026
    if re.search(r"\b0?5[-/.]0?1[-/.]2026\b", s):
        return True
    if re.search(r"\b0?1[-/.]0?5[-/.]2026\b", s):
        return True

    # English textual forms
    if re.search(r"\b(january|jan)\s+0?5,\s*2026\b", s):
        return True
    if re.search(r"\b0?5\s+(january|jan)\s+2026\b", s):
        return True
    if re.search(r"\b(january|jan)\s+0?5\s*,?\s*2026\b", s):
        return True

    # Spanish textual forms
    if re.search(r"\b0?5\s+de\s+enero\s+de\s+2026\b", s):
        return True
    if re.search(r"\benero\s+0?5(\s+de)?\s*2026\b", s):
        return True

    return False


def _has_two_years_range(term_str: Optional[str]) -> bool:
    if not _nonempty(term_str):
        return False
    years = re.findall(r"(19|20)\d{2}", term_str)
    # Accept if at least two (could be same repeated, ensure distinct where possible)
    return len(set(years)) >= 2


async def _create_url_leaf_and_verify(
    evaluator: Evaluator,
    *,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: Optional[List[str]],
    critical: bool = True,
    additional_instruction: str = "None"
):
    srcs = _safe_urls(sources)
    if not srcs:
        # No URLs => fail this URL-grounded leaf immediately (do not attempt simple verification)
        evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=parent,
            critical=critical,
            score=0.0,
            status="failed",
        )
        return

    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_official_a(evaluator: Evaluator, parent_node, a: Optional[OfficialAExtraction], b: Optional[OfficialBExtraction]):
    group = evaluator.add_parallel(
        id="VenezuelaActingPresident",
        desc="Complete information about the Venezuelan acting president sworn in January 2026",
        parent=parent_node,
        critical=False
    )

    full_name = a.full_name if a else None
    previous_title = a.previous_title if a else None
    swearing_date = a.swearing_in_date if a else None
    relationship = a.family_relationship_to_assembly_president if a else None
    assembly_name = b.full_name if (b and _nonempty(b.full_name)) else "the President of Venezuela's National Assembly"

    # Identity
    identity = evaluator.add_parallel(
        id="ActingPres_Identity",
        desc="Verified identity of the acting president",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(full_name),
        id="ActingPres_FullName",
        desc="Provides the full name of the Venezuelan official who was sworn in as acting president on January 5, 2026",
        parent=identity,
        critical=True
    )
    await _create_url_leaf_and_verify(
        evaluator,
        parent=identity,
        node_id="ActingPres_NameURL",
        desc="Provides a valid URL reference confirming the acting president's name",
        claim=f"This source confirms that {full_name} served as Venezuela's acting/interim president in January 2026.",
        sources=(a.name_sources if a else []),
        critical=True,
        additional_instruction="Accept synonyms like 'acting', 'interim', or Spanish 'presidente encargado'. Focus on confirming identity in this context."
    )

    # Previous position
    prev = evaluator.add_parallel(
        id="ActingPres_PreviousPosition",
        desc="Position held before becoming acting president",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(previous_title),
        id="ActingPres_Title",
        desc="Provides the official title/position this person held immediately before becoming acting president",
        parent=prev,
        critical=True
    )
    await _create_url_leaf_and_verify(
        evaluator,
        parent=prev,
        node_id="ActingPres_TitleURL",
        desc="Provides a valid URL reference confirming the previous title/position",
        claim=f"Immediately before being sworn in as acting president, {full_name} served as {previous_title}.",
        sources=(a.previous_title_sources if a else []),
        critical=True,
        additional_instruction="Confirm the exact role title held immediately prior to the swearing-in."
    )

    # Swearing-in
    swear = evaluator.add_parallel(
        id="ActingPres_SwearingIn",
        desc="Details of swearing-in ceremony",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_date_is_jan_5_2026(swearing_date),
        id="ActingPres_Date",
        desc="Provides the exact date of swearing-in, which must be January 5, 2026",
        parent=swear,
        critical=True
    )
    await _create_url_leaf_and_verify(
        evaluator,
        parent=swear,
        node_id="ActingPres_DateURL",
        desc="Provides a valid URL reference confirming the swearing-in date of January 5, 2026",
        claim=f"{full_name} was sworn in as acting president on January 5, 2026.",
        sources=(a.swearing_in_sources if a else []),
        critical=True,
        additional_instruction="The source must explicitly indicate the swearing-in took place on 5 January 2026 (accept 'January 5, 2026' or Spanish '5 de enero de 2026')."
    )

    # Family connection to Assembly President
    fam = evaluator.add_parallel(
        id="ActingPres_FamilyConnection",
        desc="Family relationship to National Assembly President",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(relationship),
        id="ActingPres_Relationship",
        desc="Identifies the familial relationship between the acting president and the person who serves as President of Venezuela's National Assembly",
        parent=fam,
        critical=True
    )
    await _create_url_leaf_and_verify(
        evaluator,
        parent=fam,
        node_id="ActingPres_RelationshipURL",
        desc="Provides a valid URL reference confirming the familial relationship",
        claim=f"{full_name} is the {relationship} of {assembly_name}.",
        sources=(a.relationship_sources if a else []),
        critical=True,
        additional_instruction="Confirm familial relationship (e.g., spouse, sibling, parent/child, in-law). Minor variations in phrasing are acceptable."
    )


async def verify_official_b(evaluator: Evaluator, parent_node, b: Optional[OfficialBExtraction]):
    group = evaluator.add_parallel(
        id="VenezuelaNationalAssemblyPresident",
        desc="Complete information about Venezuela's National Assembly President",
        parent=parent_node,
        critical=False
    )

    full_name = b.full_name if b else None
    title = b.title if b else None
    date = b.swearing_in_date if b else None
    term = b.term_period if b else None

    # Identity
    identity = evaluator.add_parallel(
        id="Assembly_Identity",
        desc="Verified identity of the National Assembly President",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(full_name),
        id="Assembly_FullName",
        desc="Provides the full name of the person who serves as President of Venezuela's National Assembly",
        parent=identity,
        critical=True
    )
    await _create_url_leaf_and_verify(
        evaluator,
        parent=identity,
        node_id="Assembly_NameURL",
        desc="Provides a valid URL reference confirming the National Assembly President's name",
        claim=f"{full_name} serves as the President of Venezuela's National Assembly.",
        sources=(b.name_sources if b else []),
        critical=True,
        additional_instruction="Allow minor variants in the official body's name (e.g., 'National Assembly of Venezuela')."
    )

    # Position/title
    pos = evaluator.add_parallel(
        id="Assembly_Position",
        desc="Official position details",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(title),
        id="Assembly_Title",
        desc="Provides the official title of this position",
        parent=pos,
        critical=True
    )
    await _create_url_leaf_and_verify(
        evaluator,
        parent=pos,
        node_id="Assembly_TitleURL",
        desc="Provides a valid URL reference confirming the official title",
        claim=f"The official title for {full_name} in this role is '{title}'.",
        sources=(b.title_sources if b else []),
        critical=True,
        additional_instruction="The source should state the position title held by the person (e.g., President of the National Assembly)."
    )

    # Swearing-in and term
    swear = evaluator.add_parallel(
        id="Assembly_SwearingIn",
        desc="Details of swearing-in for 2026 term",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_date_is_jan_5_2026(date),
        id="Assembly_Date",
        desc="Provides the exact date of swearing-in for the 2026 term, which must be January 5, 2026",
        parent=swear,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_two_years_range(term),
        id="Assembly_Term",
        desc="Provides the term period (start year and end year) for which this person was sworn in",
        parent=swear,
        critical=True
    )

    combined_sources = _safe_urls((b.swearing_in_sources if b else [])) + _safe_urls((b.term_sources if b else []))
    await _create_url_leaf_and_verify(
        evaluator,
        parent=swear,
        node_id="Assembly_SwearingURL",
        desc="Provides a valid URL reference confirming the swearing-in date and term period",
        claim=f"{full_name} was sworn in on January 5, 2026, for a term covering {term}.",
        sources=combined_sources,
        critical=True,
        additional_instruction="Verify both the swearing-in date (5 January 2026) and the stated term years."
    )


async def verify_official_c(evaluator: Evaluator, parent_node, c: Optional[OfficialCExtraction]):
    group = evaluator.add_parallel(
        id="KennedyCenterCEO",
        desc="Complete information about the new Kennedy Center CEO appointed March 2026",
        parent=parent_node,
        critical=False
    )

    full_name = c.full_name if c else None
    new_title = c.new_title if c else None
    previous_role = c.previous_role if c else None
    announcement_date = c.announcement_date if c else None

    # Identity
    identity = evaluator.add_parallel(
        id="KC_Identity",
        desc="Verified identity of the new CEO",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(full_name),
        id="KC_FullName",
        desc="Provides the full name of the person who was appointed as the new CEO/Executive Director of the Kennedy Center in March 2026",
        parent=identity,
        critical=True
    )
    await _create_url_leaf_and_verify(
        evaluator,
        parent=identity,
        node_id="KC_NameURL",
        desc="Provides a valid URL reference confirming the new CEO's name",
        claim=f"This source confirms that {full_name} was appointed the new CEO/Executive Director of the Kennedy Center in March 2026.",
        sources=(c.name_sources if c else []),
        critical=True,
        additional_instruction="Look for language indicating appointment as CEO/Executive Director, replacing Richard Grenell."
    )

    # New position
    newpos = evaluator.add_parallel(
        id="KC_NewPosition",
        desc="Details of the new CEO position",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(new_title),
        id="KC_Title",
        desc="Provides the new official title at the Kennedy Center",
        parent=newpos,
        critical=True
    )
    await _create_url_leaf_and_verify(
        evaluator,
        parent=newpos,
        node_id="KC_TitleURL",
        desc="Provides a valid URL reference confirming the new official title",
        claim=f"{full_name}'s new official title at the Kennedy Center is '{new_title}'.",
        sources=(c.new_title_sources if c else []),
        critical=True,
        additional_instruction="The source should clearly state the new official title (e.g., CEO, Executive Director)."
    )

    # Previous role
    prev = evaluator.add_parallel(
        id="KC_PreviousRole",
        desc="Position held before CEO appointment",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(previous_role),
        id="KC_FormerTitle",
        desc="Provides the position this person held at the Kennedy Center immediately before the CEO appointment",
        parent=prev,
        critical=True
    )
    await _create_url_leaf_and_verify(
        evaluator,
        parent=prev,
        node_id="KC_FormerTitleURL",
        desc="Provides a valid URL reference confirming the previous position at the Kennedy Center",
        claim=f"Immediately prior to this appointment, {full_name} served as {previous_role} at the Kennedy Center.",
        sources=(c.previous_role_sources if c else []),
        critical=True,
        additional_instruction="Confirm the role was at the Kennedy Center and immediately prior to appointment."
    )

    # Appointment announcement
    appt = evaluator.add_parallel(
        id="KC_Appointment",
        desc="Details of the appointment announcement",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(announcement_date),
        id="KC_AnnouncementDate",
        desc="Provides the date when President Trump announced this appointment",
        parent=appt,
        critical=True
    )
    await _create_url_leaf_and_verify(
        evaluator,
        parent=appt,
        node_id="KC_AppointmentURL",
        desc="Provides a valid URL reference confirming the appointment announcement date",
        claim=f"President Trump announced {full_name}'s Kennedy Center appointment on {announcement_date}.",
        sources=(c.announcement_sources if c else []),
        critical=True,
        additional_instruction="Prefer official announcements or reliable news citing the exact announcement date."
    )


async def verify_official_d(evaluator: Evaluator, parent_node, d: Optional[OfficialDExtraction]):
    group = evaluator.add_parallel(
        id="NobelPrizeRepresentative",
        desc="Complete information about the person who accepted the Nobel Prize on behalf of María Corina Machado",
        parent=parent_node,
        critical=False
    )

    full_name = d.full_name if d else None
    relationship = d.relationship_to_machado if d else None
    ceremony_date = d.ceremony_date if d else None
    ceremony_city = d.ceremony_city if d else None

    # Identity
    identity = evaluator.add_parallel(
        id="Nobel_Identity",
        desc="Verified identity of the prize acceptor",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(full_name),
        id="Nobel_FullName",
        desc="Provides the full name of the person who physically accepted the 2025 Nobel Peace Prize on behalf of María Corina Machado",
        parent=identity,
        critical=True
    )
    await _create_url_leaf_and_verify(
        evaluator,
        parent=identity,
        node_id="Nobel_NameURL",
        desc="Provides a valid URL reference confirming the acceptor's name",
        claim=f"{full_name} physically accepted the 2025 Nobel Peace Prize on behalf of María Corina Machado.",
        sources=(d.name_sources if d else []),
        critical=True,
        additional_instruction="The source should explicitly indicate that this person accepted the prize at the ceremony on Machado's behalf."
    )

    # Relationship
    rel = evaluator.add_parallel(
        id="Nobel_RelationshipToLaureate",
        desc="Relationship to the Nobel Prize winner",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(relationship),
        id="Nobel_Relationship",
        desc="Identifies the relationship of the acceptor to María Corina Machado",
        parent=rel,
        critical=True
    )
    await _create_url_leaf_and_verify(
        evaluator,
        parent=rel,
        node_id="Nobel_RelationshipURL",
        desc="Provides a valid URL reference confirming the relationship to María Corina Machado",
        claim=f"{full_name} is {relationship} of María Corina Machado.",
        sources=(d.relationship_sources if d else []),
        critical=True,
        additional_instruction="Confirm the familial or personal relationship (e.g., relative, spouse, representative)."
    )

    # Ceremony details
    cer = evaluator.add_parallel(
        id="Nobel_CeremonyDetails",
        desc="Details of the Nobel Prize ceremony attendance",
        parent=group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(ceremony_date),
        id="Nobel_Date",
        desc="Provides the exact date of the Nobel Peace Prize ceremony",
        parent=cer,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(ceremony_city),
        id="Nobel_Location",
        desc="Provides the city where the Nobel Peace Prize ceremony took place",
        parent=cer,
        critical=True
    )
    await _create_url_leaf_and_verify(
        evaluator,
        parent=cer,
        node_id="Nobel_CeremonyURL",
        desc="Provides a valid URL reference confirming the ceremony date and location",
        claim=f"The 2025 Nobel Peace Prize ceremony took place on {ceremony_date} in {ceremony_city}.",
        sources=(d.ceremony_sources if d else []),
        critical=True,
        additional_instruction="Accept 'Oslo City Hall' as location detail for Oslo. The page must mention both date and city."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    # Initialize evaluator with a parallel root as in rubric
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

    # Extract all officials info
    extracted = await evaluator.extract(
        prompt=prompt_extract_officials(),
        template_class=OfficialsExtraction,
        extraction_name="officials_extraction",
    )

    # Build the verification tree according to rubric (top-level parallel)
    # Official A
    await verify_official_a(
        evaluator,
        root,
        extracted.official_a if extracted else None,
        extracted.official_b if extracted else None
    )

    # Official B
    await verify_official_b(
        evaluator,
        root,
        extracted.official_b if extracted else None
    )

    # Official C
    await verify_official_c(
        evaluator,
        root,
        extracted.official_c if extracted else None
    )

    # Official D
    await verify_official_d(
        evaluator,
        root,
        extracted.official_d if extracted else None
    )

    # Return summary
    return evaluator.get_summary()