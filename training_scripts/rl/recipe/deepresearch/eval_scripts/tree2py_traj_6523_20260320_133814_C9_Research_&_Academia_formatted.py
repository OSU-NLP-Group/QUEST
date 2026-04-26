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
TASK_ID = "gmt_mirror_university"
TASK_DESCRIPTION = """
Identify the US research university that operates the Richard F. Caris Mirror Laboratory, which is responsible for fabricating the primary mirrors for the Giant Magellan Telescope, AND is a founding member of the Giant Magellan Telescope consortium. Provide the complete official name of the university, along with verification that it meets both of these primary criteria.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityExtraction(BaseModel):
    # Core identification
    university_name: Optional[str] = None
    mirror_lab_name: Optional[str] = None

    # Source URLs cited in the answer (explicit URLs only)
    gmt_member_urls: List[str] = Field(default_factory=list)         # For founding membership verification (prefer official GMT/GMTO)
    mirror_lab_urls: List[str] = Field(default_factory=list)         # For mirror lab name/affiliation/GMT fabrication
    location_urls: List[str] = Field(default_factory=list)           # For US location verification

    # Optional/ancillary support URLs
    capabilities_urls: List[str] = Field(default_factory=list)       # For "largest mirrors" and lightweight capabilities
    exclusivity_urls: List[str] = Field(default_factory=list)        # For exclusive GMT mirror fabrication claim
    aura_urls: List[str] = Field(default_factory=list)               # For AURA membership
    steward_urls: List[str] = Field(default_factory=list)            # For Steward Observatory affiliation
    telescope_urls: List[str] = Field(default_factory=list)          # For telescope access claims
    research_urls: List[str] = Field(default_factory=list)           # For research program claims


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university() -> str:
    return """
    Extract the information the answer provides about the target university and its supporting sources.

    Required fields:
    - university_name: The complete official name of the identified US university (as stated in the answer).
    - mirror_lab_name: The name of the mirror facility cited in the answer (e.g., "Richard F. Caris Mirror Laboratory", "Caris Mirror Lab"). If not explicitly stated, return null.

    URL fields (extract only explicit, valid URLs; deduplicate; keep as provided in the answer):
    - gmt_member_urls: All URLs cited that directly support the claim that the university is a founding member of the Giant Magellan Telescope consortium (prefer official Giant Magellan Telescope / GMTO sources when present).
    - mirror_lab_urls: All URLs cited that support mirror lab details (facility name, affiliation with the university, and that it fabricates GMT primary mirrors).
    - location_urls: All URLs cited that support that the university is located in the United States (official pages or Wikipedia acceptable).

    Optional (ancillary) URL fields if present in the answer; otherwise, return empty lists:
    - capabilities_urls: URLs supporting claims about making the world's largest or lightweight telescope mirrors.
    - exclusivity_urls: URLs supporting exclusive production of GMT primary mirror segments.
    - aura_urls: URLs supporting AURA membership.
    - steward_urls: URLs supporting that Steward Observatory is operated by the university.
    - telescope_urls: URLs supporting institutional access to major ground-based optical telescopes or multiple observatory partnerships.
    - research_urls: URLs supporting that the university conducts active astronomical research across multiple areas.

    Important:
    - Extract only from the provided answer text; do not invent or infer.
    - If a required field is missing, set it to null (for strings) or [] (for lists).
    - Accept URLs in plain text or markdown; output the actual URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def choose_sources(primary: List[str], fallback: List[str]) -> List[str]:
    """Return primary if non-empty; otherwise return fallback."""
    return primary if primary else fallback


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def add_university_identified_tree(
    evaluator: Evaluator,
    parent_node,
    data: UniversityExtraction,
) -> None:
    """
    Build the verification subtree under "University_Identified" according to the rubric.
    Essential criteria (founding membership, mirror lab operations, US location) are critical.
    Optional criteria are non‑critical.
    """
    # University_Identified main node (parallel aggregation, non-critical to allow soft extras)
    uni_node = evaluator.add_parallel(
        id="University_Identified",
        desc="The correct US university is identified by name",
        parent=parent_node,
        critical=False
    )

    # Name presence check (critical gate for downstream verification)
    name_present = evaluator.add_custom_node(
        result=bool(data.university_name and data.university_name.strip()),
        id="University_Name_Provided",
        desc="University name is provided in the answer",
        parent=uni_node,
        critical=True
    )

    # --------------------- GMT Founding Membership (CRITICAL) ---------------------
    await add_gmt_founding_membership(
        evaluator=evaluator,
        parent=uni_node,
        university_name=data.university_name or "",
        urls=data.gmt_member_urls,
        prereq=[name_present]
    )

    # --------------------- Mirror Lab Operation (CRITICAL) -----------------------
    await add_mirror_lab_operation(
        evaluator=evaluator,
        parent=uni_node,
        university_name=data.university_name or "",
        mirror_lab_name=data.mirror_lab_name or "Richard F. Caris Mirror Laboratory",
        urls=data.mirror_lab_urls,
        prereq=[name_present]
    )

    # --------------------- US Location (CRITICAL) --------------------------------
    await add_us_location(
        evaluator=evaluator,
        parent=uni_node,
        university_name=data.university_name or "",
        urls=data.location_urls,
        prereq=[name_present]
    )

    # --------------------- Optional: Largest/Lightweight Capability --------------
    await add_mirror_lab_largest_capability(
        evaluator=evaluator,
        parent=uni_node,
        mirror_lab_name=data.mirror_lab_name or "Richard F. Caris Mirror Laboratory",
        urls=choose_sources(data.capabilities_urls, data.mirror_lab_urls),
        prereq=[name_present]
    )

    # --------------------- Optional: Exclusive GMT Fabricator --------------------
    await add_exclusive_gmt_fabricator(
        evaluator=evaluator,
        parent=uni_node,
        mirror_lab_name=data.mirror_lab_name or "Richard F. Caris Mirror Laboratory",
        urls=choose_sources(data.exclusivity_urls, data.mirror_lab_urls),
        prereq=[name_present]
    )

    # --------------------- Optional: AURA Membership -----------------------------
    await add_aura_membership(
        evaluator=evaluator,
        parent=uni_node,
        university_name=data.university_name or "",
        urls=data.aura_urls,
        prereq=[name_present]
    )

    # --------------------- Optional: Steward Observatory -------------------------
    await add_steward_observatory(
        evaluator=evaluator,
        parent=uni_node,
        university_name=data.university_name or "",
        urls=choose_sources(data.steward_urls, data.mirror_lab_urls),
        prereq=[name_present]
    )

    # --------------------- Optional: Telescope Access ----------------------------
    await add_telescope_access(
        evaluator=evaluator,
        parent=uni_node,
        university_name=data.university_name or "",
        urls=data.telescope_urls,
        prereq=[name_present]
    )

    # --------------------- Optional: Research Programs ---------------------------
    await add_active_research_programs(
        evaluator=evaluator,
        parent=uni_node,
        university_name=data.university_name or "",
        urls=data.research_urls,
        prereq=[name_present]
    )


# ----- Essential Criterion: GMT Founding Membership ---------------------------- #
async def add_gmt_founding_membership(
    evaluator: Evaluator,
    parent,
    university_name: str,
    urls: List[str],
    prereq: Optional[List] = None
) -> None:
    node = evaluator.add_parallel(
        id="GMT_Founding_Membership",
        desc="University is a founding member of the Giant Magellan Telescope consortium",
        parent=parent,
        critical=True
    )

    # Existence of supporting URL (critical)
    url_exists = evaluator.add_custom_node(
        result=bool(urls),
        id="GMT_Membership_URL",
        desc="URL reference for GMT founding member status",
        parent=node,
        critical=True
    )

    # Founding member status verification (critical)
    founding_leaf = evaluator.add_leaf(
        id="GMT_Founding_Status",
        desc="University is explicitly listed as a founding member in GMT official sources",
        parent=node,
        critical=True
    )
    claim = f"{university_name} is a founding member (founding institution/partner) of the Giant Magellan Telescope (GMT) consortium."
    await evaluator.verify(
        claim=claim,
        node=founding_leaf,
        sources=urls,
        additional_instruction="Verify on official GMT/GMTO or clearly authoritative consortium pages. Accept synonyms like 'founding partner' or 'founding institution'.",
        extra_prerequisites=(prereq or []) + [url_exists]
    )


# ----- Essential Criterion: Mirror Lab Operation -------------------------------- #
async def add_mirror_lab_operation(
    evaluator: Evaluator,
    parent,
    university_name: str,
    mirror_lab_name: str,
    urls: List[str],
    prereq: Optional[List] = None
) -> None:
    node = evaluator.add_parallel(
        id="Mirror_Lab_Operation",
        desc="University operates the Richard F. Caris Mirror Laboratory",
        parent=parent,
        critical=True
    )

    # Existence of mirror lab-related URLs (critical)
    ml_url_exists = evaluator.add_custom_node(
        result=bool(urls),
        id="Mirror_Lab_URL",
        desc="URL reference for mirror lab operations",
        parent=node,
        critical=True
    )

    # Facility name verification (critical)
    name_leaf = evaluator.add_leaf(
        id="Mirror_Lab_Name",
        desc="Facility is named 'Richard F. Caris Mirror Laboratory' or 'Caris Mirror Lab'",
        parent=node,
        critical=True
    )
    name_claim = (
        f"The mirror facility's official name corresponds to 'Richard F. Caris Mirror Laboratory' "
        f"(also referred to as 'Caris Mirror Lab'). The name stated in the answer is '{mirror_lab_name}'."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=urls,
        additional_instruction="Allow reasonable variations in formatting or abbreviations. Confirm the facility name on an authoritative site (e.g., the lab or university).",
        extra_prerequisites=(prereq or []) + [ml_url_exists]
    )

    # University affiliation (critical)
    aff_leaf = evaluator.add_leaf(
        id="Mirror_Lab_University_Affiliation",
        desc="Mirror lab is operated by or located at the identified university",
        parent=node,
        critical=True
    )
    aff_claim = f"The {mirror_lab_name} is operated by or located at {university_name} (e.g., under Steward Observatory or the university)."
    await evaluator.verify(
        claim=aff_claim,
        node=aff_leaf,
        sources=urls,
        additional_instruction="Accept phrasing like 'operated by', 'part of', 'at', or 'within' the university (including Steward Observatory affiliation).",
        extra_prerequisites=(prereq or []) + [ml_url_exists]
    )

    # GMT primary mirror fabrication (critical)
    gmt_fab_leaf = evaluator.add_leaf(
        id="GMT_Mirror_Fabrication",
        desc="Mirror lab fabricates the primary mirrors for GMT",
        parent=node,
        critical=True
    )
    gmt_fab_claim = f"The {mirror_lab_name} fabricates the primary mirror segments for the Giant Magellan Telescope (GMT)."
    await evaluator.verify(
        claim=gmt_fab_claim,
        node=gmt_fab_leaf,
        sources=urls,
        additional_instruction="Look for language like 'fabricating', 'casting', 'polishing' the GMT primary mirror segments at the lab.",
        extra_prerequisites=(prereq or []) + [ml_url_exists]
    )


# ----- Essential Criterion: US Location ----------------------------------------- #
async def add_us_location(
    evaluator: Evaluator,
    parent,
    university_name: str,
    urls: List[str],
    prereq: Optional[List] = None
) -> None:
    node = evaluator.add_parallel(
        id="US_Location",
        desc="University is located in the United States",
        parent=parent,
        critical=True
    )

    # Existence of location URL (critical)
    loc_url_exists = evaluator.add_custom_node(
        result=bool(urls),
        id="Location_URL",
        desc="URL reference for university location",
        parent=node,
        critical=True
    )

    # State/US location verification (critical)
    state_leaf = evaluator.add_leaf(
        id="State_Location",
        desc="University is in a US state",
        parent=node,
        critical=True
    )
    state_claim = f"{university_name} is a university located in the United States (i.e., within a U.S. state)."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=urls,
        additional_instruction="Accept official university pages or reputable sources (e.g., Wikipedia) clearly stating the university is in the United States.",
        extra_prerequisites=(prereq or []) + [loc_url_exists]
    )


# ----- Optional: Mirror Lab Largest/Lightweight Capability --------------------- #
async def add_mirror_lab_largest_capability(
    evaluator: Evaluator,
    parent,
    mirror_lab_name: str,
    urls: List[str],
    prereq: Optional[List] = None
) -> None:
    node = evaluator.add_parallel(
        id="Mirror_Lab_Largest_Capability",
        desc="Mirror lab makes the largest lightweight telescope mirrors in the world",
        parent=parent,
        critical=False
    )

    # Optional: capabilities URL exists (non-critical)
    cap_url_exists = evaluator.add_custom_node(
        result=bool(urls),
        id="Capabilities_URL",
        desc="URL reference for mirror lab capabilities",
        parent=node,
        critical=False
    )

    largest_leaf = evaluator.add_leaf(
        id="Largest_Mirrors_Verified",
        desc="Mirror lab is documented as making the world's largest telescope mirrors",
        parent=node,
        critical=False
    )
    largest_claim = f"The {mirror_lab_name} is documented as making some of the world's largest telescope mirrors (e.g., 8.4-meter class)."
    await evaluator.verify(
        claim=largest_claim,
        node=largest_leaf,
        sources=urls,
        additional_instruction="Look for explicit statements about 'largest' mirrors or specific very large diameters (e.g., 8.4 m).",
        extra_prerequisites=(prereq or []) + [cap_url_exists]
    )

    light_leaf = evaluator.add_leaf(
        id="Lightweight_Technology",
        desc="Specializes in lightweight mirror design",
        parent=node,
        critical=False
    )
    light_claim = f"The {mirror_lab_name} specializes in lightweight, honeycomb or similar advanced lightweight mirror technology."
    await evaluator.verify(
        claim=light_claim,
        node=light_leaf,
        sources=urls,
        additional_instruction="Look for 'lightweight', 'honeycomb', 'cellular' mirror structure phrasing.",
        extra_prerequisites=(prereq or []) + [cap_url_exists]
    )


# ----- Optional: Exclusive GMT Fabricator -------------------------------------- #
async def add_exclusive_gmt_fabricator(
    evaluator: Evaluator,
    parent,
    mirror_lab_name: str,
    urls: List[str],
    prereq: Optional[List] = None
) -> None:
    node = evaluator.add_parallel(
        id="Exclusive_GMT_Fabricator",
        desc="Facility is the exclusive producer of GMT primary mirror segments",
        parent=parent,
        critical=False
    )

    excl_leaf = evaluator.add_leaf(
        id="Unique_GMT_Production",
        desc="Only this facility fabricates GMT primary mirrors",
        parent=node,
        critical=False
    )
    excl_claim = f"The {mirror_lab_name} is the exclusive (only) facility producing the GMT primary mirror segments."
    await evaluator.verify(
        claim=excl_claim,
        node=excl_leaf,
        sources=urls,
        additional_instruction="Look for phrasing like 'only at' or 'exclusively produced' at the lab for the GMT primary segments.",
        extra_prerequisites=(prereq or [])
    )

    excl_url_exists = evaluator.add_custom_node(
        result=bool(urls),
        id="Exclusivity_URL",
        desc="URL reference for exclusive fabrication role",
        parent=node,
        critical=False
    )


# ----- Optional: AURA Membership ----------------------------------------------- #
async def add_aura_membership(
    evaluator: Evaluator,
    parent,
    university_name: str,
    urls: List[str],
    prereq: Optional[List] = None
) -> None:
    node = evaluator.add_parallel(
        id="AURA_Membership",
        desc="University is a member institution of AURA",
        parent=parent,
        critical=False
    )

    aura_leaf = evaluator.add_leaf(
        id="AURA_Member_Listed",
        desc="University appears in AURA member institutions list",
        parent=node,
        critical=False
    )
    aura_claim = f"{university_name} appears in the list of AURA member institutions."
    await evaluator.verify(
        claim=aura_claim,
        node=aura_leaf,
        sources=urls,
        additional_instruction="Verify on an AURA membership page or other authoritative listing.",
        extra_prerequisites=(prereq or [])
    )

    aura_url_exists = evaluator.add_custom_node(
        result=bool(urls),
        id="AURA_URL",
        desc="URL reference for AURA membership",
        parent=node,
        critical=False
    )


# ----- Optional: Steward Observatory ------------------------------------------- #
async def add_steward_observatory(
    evaluator: Evaluator,
    parent,
    university_name: str,
    urls: List[str],
    prereq: Optional[List] = None
) -> None:
    node = evaluator.add_parallel(
        id="Steward_Observatory",
        desc="University operates Steward Observatory",
        parent=parent,
        critical=False
    )

    st_leaf = evaluator.add_leaf(
        id="Steward_Affiliation",
        desc="Steward Observatory is operated by the university",
        parent=node,
        critical=False
    )
    st_claim = f"Steward Observatory is operated by {university_name}."
    await evaluator.verify(
        claim=st_claim,
        node=st_leaf,
        sources=urls,
        additional_instruction="Accept equivalent phrasing such as 'operated by', 'part of', or 'within' the university.",
        extra_prerequisites=(prereq or [])
    )

    st_url_exists = evaluator.add_custom_node(
        result=bool(urls),
        id="Steward_URL",
        desc="URL reference for Steward Observatory",
        parent=node,
        critical=False
    )


# ----- Optional: Telescope Access ---------------------------------------------- #
async def add_telescope_access(
    evaluator: Evaluator,
    parent,
    university_name: str,
    urls: List[str],
    prereq: Optional[List] = None
) -> None:
    node = evaluator.add_parallel(
        id="Telescope_Access",
        desc="University has institutional access to world-class ground-based optical telescopes",
        parent=parent,
        critical=False
    )

    maj_leaf = evaluator.add_leaf(
        id="Major_Telescope_Access",
        desc="Has access to major optical telescope facilities",
        parent=node,
        critical=False
    )
    maj_claim = f"{university_name} has institutional access to major, world-class ground-based optical telescopes."
    await evaluator.verify(
        claim=maj_claim,
        node=maj_leaf,
        sources=urls,
        additional_instruction="Look for access/partnerships/ownership in large facilities (e.g., Magellan, MMT, LBT).",
        extra_prerequisites=(prereq or [])
    )

    multi_leaf = evaluator.add_leaf(
        id="Multiple_Observatory_Partnerships",
        desc="Partnerships with multiple observatories documented",
        parent=node,
        critical=False
    )
    multi_claim = f"{university_name} participates in multiple observatory partnerships or telescope collaborations."
    await evaluator.verify(
        claim=multi_claim,
        node=multi_leaf,
        sources=urls,
        additional_instruction="Evidence might include membership or observing time agreements across multiple facilities.",
        extra_prerequisites=(prereq or [])
    )

    tel_url_exists = evaluator.add_custom_node(
        result=bool(urls),
        id="Telescope_URL",
        desc="URL reference for telescope access",
        parent=node,
        critical=False
    )


# ----- Optional: Active Research Programs -------------------------------------- #
async def add_active_research_programs(
    evaluator: Evaluator,
    parent,
    university_name: str,
    urls: List[str],
    prereq: Optional[List] = None
) -> None:
    node = evaluator.add_parallel(
        id="Active_Research_Programs",
        desc="University conducts active astronomical research programs across multiple areas",
        parent=parent,
        critical=False
    )

    multi_area_leaf = evaluator.add_leaf(
        id="Multiple_Research_Areas",
        desc="Conducts research in multiple areas of astronomy",
        parent=node,
        critical=False
    )
    multi_area_claim = f"{university_name} conducts astronomical research across multiple topical areas."
    await evaluator.verify(
        claim=multi_area_claim,
        node=multi_area_leaf,
        sources=urls,
        additional_instruction="Look for listings of research groups, centers, or themes indicating breadth.",
        extra_prerequisites=(prereq or [])
    )

    doc_leaf = evaluator.add_leaf(
        id="Research_Documentation",
        desc="Research programs are documented and active",
        parent=node,
        critical=False
    )
    doc_claim = f"{university_name} maintains active and documented astronomy/astrophysics research programs (e.g., recent publications, active labs/groups)."
    await evaluator.verify(
        claim=doc_claim,
        node=doc_leaf,
        sources=urls,
        additional_instruction="Evidence can include active group pages, recent news, or publications.",
        extra_prerequisites=(prereq or [])
    )

    res_url_exists = evaluator.add_custom_node(
        result=bool(urls),
        id="Research_URL",
        desc="URL reference for research programs",
        parent=node,
        critical=False
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
    """
    Evaluate an answer for the university operating the Richard F. Caris Mirror Laboratory and
    being a founding member of the GMT consortium.
    """
    # 1) Initialize evaluator with a sequential root (overall task)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # 2) Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_university(),
        template_class=UniversityExtraction,
        extraction_name="university_extraction"
    )

    # 3) Build verification tree under root
    # Root is sequential (but only one child group here)
    await add_university_identified_tree(evaluator, root, extracted)

    # 4) Return summary
    return evaluator.get_summary()