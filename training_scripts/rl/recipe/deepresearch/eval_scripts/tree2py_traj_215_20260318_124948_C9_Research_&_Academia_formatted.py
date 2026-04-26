import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "aura_members_2026_institutions"
TASK_DESCRIPTION = """
Identify four universities that are current members of the Association of Universities for Research in Astronomy (AURA) and that each satisfy all of the following criteria:

1. The institution must be listed as a current AURA member, with documented information about the year it joined AURA and the name of its current AURA member representative.

2. The institution must have an active astronomy or astrophysics department or program that offers at least an undergraduate degree in astronomy or astrophysics. The specific name of the department and the confirmation that an undergraduate program exists must be documentable.

3. The institution must have documented access to at least one major telescope facility (such as Keck Observatory, Gemini Observatory, VLT, Magellan, or similar professional research telescopes). The name of the facility and evidence of the institution's access must be verifiable.

4. The institution must have documented involvement in astronomical activities occurring in the year 2026. This could include participation in astronomical conferences held in 2026 (such as AAS meetings), involvement in space missions with 2026 milestones, participation in observational campaigns related to 2026 astronomical events, or hosting 2026 astronomy-related events or programs.

For each of the four institutions, provide:
- The institution name
- AURA membership year and current member representative name
- Astronomy department/program name
- Confirmation of undergraduate astronomy degree program
- Name of at least one major telescope facility the institution has access to
- Description of the institution's 2026 astronomical activity involvement
- Reference URLs supporting each piece of information
"""


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class InstitutionItem(BaseModel):
    # Institution basics
    name: Optional[str] = None

    # AURA membership
    aura_join_year: Optional[str] = None
    aura_representative: Optional[str] = None
    aura_urls: List[str] = Field(default_factory=list)

    # Astronomy program
    program_department_name: Optional[str] = None
    program_undergrad_exists: Optional[bool] = None  # if stated explicitly
    program_graduate_status: Optional[str] = None  # e.g., "PhD offered" or "No graduate program", optional
    program_urls: List[str] = Field(default_factory=list)

    # Telescope access
    telescope_facility_name: Optional[str] = None
    telescope_access_type: Optional[str] = None  # e.g., partnership, consortium, time-allocation, etc.
    telescope_urls: List[str] = Field(default_factory=list)

    # 2026 involvement
    involvement_2026_description: Optional[str] = None
    involvement_2026_type: Optional[str] = None  # e.g., conference participation, mission milestone, event hosting
    involvement_2026_urls: List[str] = Field(default_factory=list)


class InstitutionsExtraction(BaseModel):
    institutions: List[InstitutionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_institutions() -> str:
    return """
    Extract up to four (4) institutions described in the answer that claim to meet the AURA-member requirements.
    For each institution, return the following fields (use null for missing fields and [] for missing URL lists):

    - name: The institution name as stated.
    - aura_join_year: The year the institution joined AURA, as a 4-digit year string (if present).
    - aura_representative: The CURRENT AURA member representative's name for the institution (if stated).
    - aura_urls: A list of URLs that the answer cites to support AURA membership details (prefer AURA.org or official pages).

    - program_department_name: The official astronomy/astrophysics department/program name (exact phrasing if possible).
    - program_undergrad_exists: true if the answer explicitly claims an undergraduate degree (major/minor/BS/BA) in astronomy or astrophysics exists; otherwise false or null if unstated.
    - program_graduate_status: A brief phrase describing graduate program status if the answer mentions it (e.g., "PhD offered", "MS program", or "no graduate program"); otherwise null.
    - program_urls: A list of URLs that the answer cites to support program/department and undergraduate offering information (prefer official university pages).

    - telescope_facility_name: The name of at least one major professional telescope facility the institution has access to (e.g., Keck, Gemini, VLT, Magellan, etc.).
    - telescope_access_type: Brief description of how access is obtained (e.g., partnership, consortium, time-allocation, membership), if mentioned; otherwise null.
    - telescope_urls: A list of URLs that the answer cites to support facility access (prefer official consortium/observatory/university pages).

    - involvement_2026_description: A brief description of the institution's astronomical activity in the year 2026 (e.g., AAS 2026 participation, a 2026 mission milestone, a 2026 event).
    - involvement_2026_type: The nature of involvement (e.g., "conference presentation", "mission partner", "event host"), if stated; otherwise null.
    - involvement_2026_urls: A list of URLs that the answer cites to support the 2026 involvement claim (must clearly reference activities in 2026).

    Return the result as:
    {
      "institutions": [ { ... up to 4 items ... } ]
    }

    Rules:
    - Extract only information explicitly present in the answer text.
    - For URL lists, include only valid full URLs (http/https), exactly as they appear in the answer text (plain or markdown).
    - Do not invent URLs. If no URL is provided in the answer for a field, use an empty list for that URL list.
    - If an institution provides multiple URLs for the same aspect, include all of them.
    - If more than four institutions are described, only include the first four mentioned.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls)


def _first_n_institutions(extracted: InstitutionsExtraction, n: int = 4) -> List[InstitutionItem]:
    items = list(extracted.institutions or [])
    if len(items) >= n:
        return items[:n]
    # pad with empty placeholders
    while len(items) < n:
        items.append(InstitutionItem())
    return items


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_institution(
    evaluator: Evaluator,
    parent_node,
    inst: InstitutionItem,
    idx_one_based: int,
):
    # Institution container (non-critical; allows partial credit across institutions)
    inst_node = evaluator.add_parallel(
        id=f"institution_{idx_one_based}",
        desc=f"#{idx_one_based} AURA member institution meeting all requirements",
        parent=parent_node,
        critical=False,
    )

    inst_name = inst.name or f"Institution #{idx_one_based}"

    # ---------------------- AURA membership (critical group) ----------------------
    aura_node = evaluator.add_parallel(
        id=f"inst{idx_one_based}_aura_membership",
        desc="AURA membership details provided: confirmed member status, joining year, and current representative name",
        parent=inst_node,
        critical=True,  # All children under here must be critical to respect consistency
    )

    # Presence of membership reference URL(s) - critical existence gate
    evaluator.add_custom_node(
        result=_has_urls(inst.aura_urls),
        id=f"inst{idx_one_based}_aura_reference_url",
        desc="Reference URL confirming AURA membership information",
        parent=aura_node,
        critical=True,
    )

    # Confirm institution is a current AURA member
    if inst.name and _has_urls(inst.aura_urls):
        m_node = evaluator.add_leaf(
            id=f"inst{idx_one_based}_aura_member_confirmed",
            desc="Institution is confirmed as current AURA member",
            parent=aura_node,
            critical=True,
        )
        claim = f"{inst.name} is listed as a CURRENT member institution of the Association of Universities for Research in Astronomy (AURA)."
        await evaluator.verify(
            claim=claim,
            node=m_node,
            sources=inst.aura_urls,
            additional_instruction="Verify that the page explicitly indicates the institution is a current AURA member (not former). Prefer AURA.org membership lists or official pages.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"inst{idx_one_based}_aura_member_confirmed",
            desc="Institution is confirmed as current AURA member",
            parent=aura_node,
            critical=True,
        )

    # Joining year
    if inst.name and inst.aura_join_year and _has_urls(inst.aura_urls):
        y_node = evaluator.add_leaf(
            id=f"inst{idx_one_based}_aura_joining_year",
            desc="Year institution joined AURA is provided",
            parent=aura_node,
            critical=True,
        )
        claim = f"{inst.name} joined AURA in {inst.aura_join_year}."
        await evaluator.verify(
            claim=claim,
            node=y_node,
            sources=inst.aura_urls,
            additional_instruction="Confirm the membership 'since' or joining year on the cited page(s). Allow minor wording variants, but the year must match.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"inst{idx_one_based}_aura_joining_year",
            desc="Year institution joined AURA is provided",
            parent=aura_node,
            critical=True,
        )

    # Representative name
    if inst.name and inst.aura_representative and _has_urls(inst.aura_urls):
        r_node = evaluator.add_leaf(
            id=f"inst{idx_one_based}_aura_representative",
            desc="Current AURA member representative name is provided",
            parent=aura_node,
            critical=True,
        )
        claim = f"The current AURA member representative for {inst.name} is {inst.aura_representative}."
        await evaluator.verify(
            claim=claim,
            node=r_node,
            sources=inst.aura_urls,
            additional_instruction="Confirm the name of the CURRENT AURA representative for the institution; allow minor spelling/format variants.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"inst{idx_one_based}_aura_representative",
            desc="Current AURA member representative name is provided",
            parent=aura_node,
            critical=True,
        )

    # ---------------------- Astronomy program (critical group) ----------------------
    # To avoid mixing critical and soft children (which changes gating semantics), we keep only critical checks here.
    prog_node = evaluator.add_parallel(
        id=f"inst{idx_one_based}_astronomy_program",
        desc="Astronomy program details provided: department name and undergraduate program confirmation",
        parent=inst_node,
        critical=True,
    )

    # Program reference URLs (existence)
    evaluator.add_custom_node(
        result=_has_urls(inst.program_urls),
        id=f"inst{idx_one_based}_program_reference_url",
        desc="Reference URL confirming astronomy program details",
        parent=prog_node,
        critical=True,
    )

    # Department name verification
    if inst.name and inst.program_department_name and _has_urls(inst.program_urls):
        d_node = evaluator.add_leaf(
            id=f"inst{idx_one_based}_department_name",
            desc="Official astronomy/astrophysics department name is provided",
            parent=prog_node,
            critical=True,
        )
        claim = f"The official astronomy/astrophysics department or program at {inst.name} is named '{inst.program_department_name}'."
        await evaluator.verify(
            claim=claim,
            node=d_node,
            sources=inst.program_urls,
            additional_instruction="Verify that the cited page shows the department/program name (allow minor variants like 'Department of Astronomy and Astrophysics').",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"inst{idx_one_based}_department_name",
            desc="Official astronomy/astrophysics department name is provided",
            parent=prog_node,
            critical=True,
        )

    # Undergraduate astronomy program existence
    if _has_urls(inst.program_urls) and inst.name:
        u_node = evaluator.add_leaf(
            id=f"inst{idx_one_based}_undergraduate_program",
            desc="Undergraduate astronomy degree program existence is confirmed",
            parent=prog_node,
            critical=True,
        )
        claim = f"{inst.name} offers at least one undergraduate degree (major/minor/BA/BS) in astronomy or astrophysics."
        await evaluator.verify(
            claim=claim,
            node=u_node,
            sources=inst.program_urls,
            additional_instruction="Confirm that the cited page indicates an undergraduate degree (major/minor/BA/BS) in astronomy or astrophysics. Accept closely related names if clearly astronomy-focused.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"inst{idx_one_based}_undergraduate_program",
            desc="Undergraduate astronomy degree program existence is confirmed",
            parent=prog_node,
            critical=True,
        )

    # ---------------------- Telescope access (critical group) ----------------------
    tel_node = evaluator.add_parallel(
        id=f"inst{idx_one_based}_telescope_access",
        desc="Telescope facility access documented: facility name provided",
        parent=inst_node,
        critical=True,
    )

    # Facility reference URL(s) presence
    evaluator.add_custom_node(
        result=_has_urls(inst.telescope_urls),
        id=f"inst{idx_one_based}_facility_reference_url",
        desc="Reference URL confirming telescope facility access",
        parent=tel_node,
        critical=True,
    )

    # Facility name + access verification
    if inst.name and inst.telescope_facility_name and _has_urls(inst.telescope_urls):
        f_node = evaluator.add_leaf(
            id=f"inst{idx_one_based}_facility_name",
            desc="Name of at least one major telescope facility is provided",
            parent=tel_node,
            critical=True,
        )
        claim = f"{inst.name} has access to the {inst.telescope_facility_name} telescope facility (e.g., through membership, partnership, consortium, or time-allocation)."
        await evaluator.verify(
            claim=claim,
            node=f_node,
            sources=inst.telescope_urls,
            additional_instruction="Confirm that the cited page(s) indicate the institution has access to the named professional telescope facility.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"inst{idx_one_based}_facility_name",
            desc="Name of at least one major telescope facility is provided",
            parent=tel_node,
            critical=True,
        )

    # ---------------------- 2026 involvement (critical group) ----------------------
    inv_node = evaluator.add_parallel(
        id=f"inst{idx_one_based}_2026_involvement",
        desc="2026 astronomical activity involvement documented",
        parent=inst_node,
        critical=True,
    )

    # 2026 involvement reference URL(s) presence
    evaluator.add_custom_node(
        result=_has_urls(inst.involvement_2026_urls),
        id=f"inst{idx_one_based}_2026_reference_url",
        desc="Reference URL confirming 2026 involvement",
        parent=inv_node,
        critical=True,
    )

    # 2026 activity described and verified
    if inst.name and inst.involvement_2026_description and _has_urls(inst.involvement_2026_urls):
        a_node = evaluator.add_leaf(
            id=f"inst{idx_one_based}_2026_activity_described",
            desc="Specific 2026 astronomical activity is described",
            parent=inv_node,
            critical=True,
        )
        claim = f"In 2026, {inst.name} was involved in the following astronomical activity: {inst.involvement_2026_description}"
        await evaluator.verify(
            claim=claim,
            node=a_node,
            sources=inst.involvement_2026_urls,
            additional_instruction="Verify that the cited page(s) explicitly show activity in the year 2026 involving the institution (e.g., conference participation, mission milestone, hosted event). The year 2026 must be clear.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"inst{idx_one_based}_2026_activity_described",
            desc="Specific 2026 astronomical activity is described",
            parent=inv_node,
            critical=True,
        )

    # ---------------------- Non-critical extras (kept separate) ----------------------
    # Place the 'soft' items outside critical groups to avoid gate-then-average interference.
    extras_node = evaluator.add_parallel(
        id=f"inst{idx_one_based}_non_critical_extras",
        desc="Non-critical supplemental details (graduate program, access type, involvement type)",
        parent=inst_node,
        critical=False,
    )

    # Graduate program documented (presence only)
    evaluator.add_custom_node(
        result=bool(inst.program_graduate_status and inst.program_graduate_status.strip()),
        id=f"inst{idx_one_based}_graduate_program",
        desc="Graduate astronomy program status is documented",
        parent=extras_node,
        critical=False,
    )

    # Telescope access type documented (presence only)
    evaluator.add_custom_node(
        result=bool(inst.telescope_access_type and inst.telescope_access_type.strip()),
        id=f"inst{idx_one_based}_access_type",
        desc="Type of telescope access arrangement is described",
        parent=extras_node,
        critical=False,
    )

    # 2026 involvement type documented (presence only)
    evaluator.add_custom_node(
        result=bool(inst.involvement_2026_type and inst.involvement_2026_type.strip()),
        id=f"inst{idx_one_based}_involvement_type",
        desc="Nature of 2026 involvement is specified",
        parent=extras_node,
        critical=False,
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Institutions evaluated independently
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

    # NOTE: Make root non-critical to allow partial credit across institutions (JSON's root critical would violate
    #       the framework constraint that a critical parent cannot have non-critical children).
    root.critical = False

    # Extract institutions from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_institutions(),
        template_class=InstitutionsExtraction,
        extraction_name="institutions_extraction",
    )

    # Normalize to exactly four institutions (pad if fewer, trim if more)
    institutions = _first_n_institutions(extracted, n=4)

    # Build verification subtrees for each institution
    for i, inst in enumerate(institutions, start=1):
        await verify_institution(evaluator, root, inst, i)

    # Return structured evaluation summary
    return evaluator.get_summary()