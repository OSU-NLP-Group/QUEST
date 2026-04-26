import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "exec_cod_respawn_fatality_2025"
TASK_DESCRIPTION = (
    "Identify the video game industry executive who co-founded a game development studio in 2002 that released "
    "the first Call of Duty game in October 2003 for PC. This same person later co-founded another studio in 2010 "
    "that was subsequently acquired by Electronic Arts in 2017, and which released Titanfall in March 2014 and "
    "Apex Legends in February 2019. This executive tragically died in a car accident on December 21, 2025, on a "
    "scenic highway in Southern California's San Gabriel Mountains. Provide the following information: "
    "(1) The person's full name, (2) Their birth date (in format: Month Day, Year), (3) The name of the highway where "
    "the fatal accident occurred, and (4) The approximate elevation (in feet) of the accident location, which was near "
    "a set of tunnels on this highway."
)

# Constraints from rubric
EXPECTED_BIRTH_DATE = "October 1, 1970"
EXPECTED_HIGHWAY_KEYWORDS = [
    "angeles crest highway",
    "california state route 2",
    "ca-2",
    "sr 2",
    "state route 2",
]
EXPECTED_ELEVATION_FT_TARGET = 6100
EXPECTED_ELEVATION_TOLERANCE_FT = 300  # Accept approx ±300 ft
EXPECTED_TITANFALL_RELEASE_CLAIM = "March 11, 2014"
EXPECTED_APEX_RELEASE_CLAIM = "February 4, 2019"
EXPECTED_COD_PC_RELEASE_CLAIM = "October 29, 2003"
EXPECTED_RESPAWN_ACQUIRED_YEAR = "2017"
EXPECTED_RESPAWN_COFUND_YEAR = "2010"
EXPECTED_STUDIO_COFUND_YEAR = "2002"
EXPECTED_DEATH_DATE = "December 21, 2025"
EXPECTED_AGE_AT_DEATH = 55


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ExecAnswerExtraction(BaseModel):
    # Requested fields
    full_name: Optional[str] = None
    birth_date: Optional[str] = None
    accident_highway: Optional[str] = None
    accident_elevation_ft: Optional[str] = None

    # Background/Context fields for verification
    death_date: Optional[str] = None

    # Sources for requested fields
    birth_date_sources: List[str] = Field(default_factory=list)
    accident_highway_sources: List[str] = Field(default_factory=list)
    accident_elevation_sources: List[str] = Field(default_factory=list)
    death_date_sources: List[str] = Field(default_factory=list)

    # Background constraint sources
    cofound_2002_sources: List[str] = Field(default_factory=list)
    first_cod_pc_2003_sources: List[str] = Field(default_factory=list)
    cofound_respawn_2010_sources: List[str] = Field(default_factory=list)
    respawn_acq_ea_2017_sources: List[str] = Field(default_factory=list)
    titanfall_2014_sources: List[str] = Field(default_factory=list)
    apex_2019_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_exec_info() -> str:
    return """
Extract the following from the answer text, returning a single JSON object with the specified fields. Follow these rules strictly:
- Return strings exactly as stated in the answer for the fields.
- For all URL fields, extract only the URLs explicitly present in the answer text. If none are provided, use an empty array.
- Do not invent any information that is missing.

Fields to extract:
1) full_name: The executive’s full name (string).
2) birth_date: The person's birth date, as presented in the answer (string, e.g., "October 1, 1970").
3) accident_highway: The highway name where the fatal accident occurred, as presented (string, e.g., "Angeles Crest Highway" or "California State Route 2").
4) accident_elevation_ft: The approximate elevation in feet near the tunnels where the accident occurred, as presented (string; extract the value as written, e.g., "6,100 feet" or "~6100 ft").
5) death_date: The death date as presented in the answer (string, e.g., "December 21, 2025").

Also extract URL sources cited in the answer for each fact (if any):
- birth_date_sources: URLs supporting the birth date (array of strings).
- accident_highway_sources: URLs supporting the accident highway identification (array of strings).
- accident_elevation_sources: URLs supporting the elevation near the tunnels (array of strings).
- death_date_sources: URLs supporting the death date (array of strings).

For the background constraints used to identify the person, extract URLs from the answer that specifically support each statement (if any), preserving duplicates only once:
- cofound_2002_sources: URLs supporting that the person co-founded a game development studio in 2002 with Jason West and Grant Collier (typically Infinity Ward).
- first_cod_pc_2003_sources: URLs supporting that that studio released the first Call of Duty game for PC on October 29, 2003.
- cofound_respawn_2010_sources: URLs supporting that the person co-founded Respawn Entertainment in 2010 with Jason West.
- respawn_acq_ea_2017_sources: URLs supporting that Respawn Entertainment was acquired by Electronic Arts in 2017.
- titanfall_2014_sources: URLs supporting that Respawn Entertainment released Titanfall on March 11, 2014 (North America).
- apex_2019_sources: URLs supporting that Respawn Entertainment released Apex Legends on February 4, 2019).

If the answer does not provide a value for a string field, set it to null.
If no sources are provided for a given sources array, return an empty array for that field.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm_str(s: Optional[str]) -> str:
    return (s or "").strip()


def _lower_ascii(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        u = (u or "").strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _parse_mmddyyyy_full_month(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        # Format like "October 1, 1970"
        return datetime.strptime(date_str.strip(), "%B %d, %Y")
    except Exception:
        return None


def _extract_numeric_value(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Find groups of digits possibly with commas, pick the first reasonable match
    # e.g., "6,100 ft" -> 6100
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d{3,5})", text.replace("\u2009", ""))
    if not m:
        return None
    digits = m.group(1).replace(",", "")
    try:
        return float(digits)
    except Exception:
        return None


def _approx_equal(val: float, target: float, tol: float) -> bool:
    return abs(val - target) <= tol


def _highway_matches_expected(name: Optional[str]) -> bool:
    s = _lower_ascii(name)
    if not s:
        return False
    # Accept presence of any of the synonyms/variants
    return any(k in s for k in EXPECTED_HIGHWAY_KEYWORDS)


def _compute_age_at_death(birth_date: Optional[str], death_date: Optional[str]) -> Optional[int]:
    b = _parse_mmddyyyy_full_month(birth_date)
    d = _parse_mmddyyyy_full_month(death_date)
    if not b or not d:
        return None
    # Compute age in years at death (standard approach)
    years = d.year - b.year
    if (d.month, d.day) < (b.month, b.day):
        years -= 1
    return years


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def _verify_fact_with_sources(
    evaluator: Evaluator,
    parent,
    node_id_prefix: str,
    claim_desc: str,
    claim_text: str,
    sources: List[str],
    add_ins: str,
    critical: bool = True,
) -> None:
    """
    Create a sequential sub-tree for a single fact:
      1) sources_exist (critical custom)
      2) supported_by_sources (critical leaf verified via URLs)
    """
    seq_node = evaluator.add_sequential(
        id=f"{node_id_prefix}_seq",
        desc=claim_desc,
        parent=parent,
        critical=critical
    )

    # Gate: sources present
    sources_exist = len(_dedup_urls(sources)) > 0
    evaluator.add_custom_node(
        result=sources_exist,
        id=f"{node_id_prefix}_sources_exist",
        desc=f"Sources provided for: {claim_desc}",
        parent=seq_node,
        critical=True
    )

    # Verify by provided sources (skipped automatically if previous critical fails due to sequential strategy)
    sup_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_supported",
        desc=f"Claim supported by cited sources: {claim_desc}",
        parent=seq_node,
        critical=True
    )

    await evaluator.verify(
        claim=claim_text,
        node=sup_leaf,
        sources=_dedup_urls(sources),
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Build background constraints verification                                   #
# --------------------------------------------------------------------------- #
async def verify_background_constraints(
    evaluator: Evaluator,
    parent,
    data: ExecAnswerExtraction
) -> None:
    """
    Verify all critical background constraints in parallel under a critical node.
    """
    bg_node = evaluator.add_parallel(
        id="Background_Constraints_Satisfied",
        desc="The named person satisfies each stated background constraint used to identify them.",
        parent=parent,
        critical=True
    )

    person = _norm_str(data.full_name) or "the person"

    # Co-founded a studio in 2002 with Jason West and Grant Collier (Infinity Ward)
    await _verify_fact_with_sources(
        evaluator=evaluator,
        parent=bg_node,
        node_id_prefix="cofound_2002",
        claim_desc="Co-founded a studio in 2002 with Jason West and Grant Collier",
        claim_text=f"In {EXPECTED_STUDIO_COFUND_YEAR}, {person} co-founded Infinity Ward with Jason West and Grant Collier.",
        sources=data.cofound_2002_sources,
        add_ins="The webpage should explicitly connect the person with co-founding a game studio in 2002 with Jason West and Grant Collier (typically Infinity Ward). Allow minor textual variations.",
        critical=True
    )

    # That studio released the first Call of Duty for PC on October 29, 2003
    await _verify_fact_with_sources(
        evaluator=evaluator,
        parent=bg_node,
        node_id_prefix="first_cod_pc_2003",
        claim_desc="2003 PC release of the first Call of Duty",
        claim_text=f"Infinity Ward released the first Call of Duty for PC on {EXPECTED_COD_PC_RELEASE_CLAIM}.",
        sources=data.first_cod_pc_2003_sources,
        add_ins="Confirm that the studio that released the first Call of Duty for PC was Infinity Ward and the date was October 29, 2003. Small formatting differences are fine.",
        critical=True
    )

    # Co-founded Respawn in 2010 with Jason West
    await _verify_fact_with_sources(
        evaluator=evaluator,
        parent=bg_node,
        node_id_prefix="cofound_respawn_2010",
        claim_desc="Co-founded Respawn Entertainment in 2010 with Jason West",
        claim_text=f"In {EXPECTED_RESPAWN_COFUND_YEAR}, {person} co-founded Respawn Entertainment with Jason West.",
        sources=data.cofound_respawn_2010_sources,
        add_ins="Confirm that the person co-founded Respawn Entertainment in 2010 with Jason West.",
        critical=True
    )

    # Respawn acquired by EA in 2017
    await _verify_fact_with_sources(
        evaluator=evaluator,
        parent=bg_node,
        node_id_prefix="respawn_acq_ea_2017",
        claim_desc="Respawn Entertainment acquired by EA in 2017",
        claim_text="Respawn Entertainment was acquired by Electronic Arts in 2017.",
        sources=data.respawn_acq_ea_2017_sources,
        add_ins="The page should state or clearly imply EA acquired Respawn in 2017 (month/day variations acceptable).",
        critical=True
    )

    # Titanfall released March 11, 2014 (NA)
    await _verify_fact_with_sources(
        evaluator=evaluator,
        parent=bg_node,
        node_id_prefix="titanfall_2014",
        claim_desc="Respawn released Titanfall on March 11, 2014 (NA)",
        claim_text=f"Respawn Entertainment released Titanfall on {EXPECTED_TITANFALL_RELEASE_CLAIM} in North America.",
        sources=data.titanfall_2014_sources,
        add_ins="Confirm Titanfall's North America release date as March 11, 2014.",
        critical=True
    )

    # Apex Legends released February 4, 2019
    await _verify_fact_with_sources(
        evaluator=evaluator,
        parent=bg_node,
        node_id_prefix="apex_2019",
        claim_desc="Respawn released Apex Legends on February 4, 2019",
        claim_text=f"Respawn Entertainment released Apex Legends on {EXPECTED_APEX_RELEASE_CLAIM}.",
        sources=data.apex_2019_sources,
        add_ins="Confirm the release date of Apex Legends as February 4, 2019.",
        critical=True
    )

    # Died in a car accident on December 21, 2025
    await _verify_fact_with_sources(
        evaluator=evaluator,
        parent=bg_node,
        node_id_prefix="death_2025",
        claim_desc="Died in a car accident on December 21, 2025",
        claim_text=f"{person} died in a car accident on {EXPECTED_DEATH_DATE}.",
        sources=data.death_date_sources if data.death_date_sources else data.death_date_sources,
        add_ins="The page should clearly state the death date as December 21, 2025, with mention of a car accident.",
        critical=True
    )

    # Age 55 at time of death (computed check)
    age_val = _compute_age_at_death(data.birth_date, data.death_date)
    evaluator.add_custom_node(
        result=(age_val == EXPECTED_AGE_AT_DEATH),
        id="age_55_at_death",
        desc=f"Person was {EXPECTED_AGE_AT_DEATH} years old at the time of death (computed from birth and death dates)",
        parent=bg_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Verify requested fields                                                     #
# --------------------------------------------------------------------------- #
async def verify_requested_fields(
    evaluator: Evaluator,
    parent,
    data: ExecAnswerExtraction
) -> None:
    """
    Verify the three requested items under 'Provide_Requested_Fields' (critical parallel):
      - Birth date value and format (and source)
      - Accident highway name (and source)
      - Accident elevation in feet approx 6100 (and source)
    """
    req_node = evaluator.add_parallel(
        id="Provide_Requested_Fields",
        desc="Provides the four requested pieces of information (birth date, accident highway, and elevation details) in the required form.",
        parent=parent,
        critical=True
    )

    # Birth date checks (format + equals expected + supported by sources)
    bd_main = evaluator.add_sequential(
        id="Birth_Date_Value_And_Format",
        desc="Birth date provided in 'Month Day, Year' format and matches the constraint (October 1, 1970)",
        parent=req_node,
        critical=True
    )

    # Format check (custom)
    bd_format_ok = _parse_mmddyyyy_full_month(data.birth_date) is not None
    evaluator.add_custom_node(
        result=bd_format_ok,
        id="birth_date_format_ok",
        desc="Birth date is in 'Month Day, Year' format",
        parent=bd_main,
        critical=True
    )

    # Value equality check (custom)
    bd_equal_ok = _norm_str(data.birth_date).strip() == EXPECTED_BIRTH_DATE
    evaluator.add_custom_node(
        result=bd_equal_ok,
        id="birth_date_value_match",
        desc=f"Birth date equals '{EXPECTED_BIRTH_DATE}'",
        parent=bd_main,
        critical=True
    )

    # Source support for birth date
    bd_leaf = evaluator.add_leaf(
        id="birth_date_source_supported",
        desc="Birth date is supported by cited sources",
        parent=bd_main,
        critical=True
    )
    await evaluator.verify(
        claim=f"The person's birth date is {EXPECTED_BIRTH_DATE}.",
        node=bd_leaf,
        sources=_dedup_urls(data.birth_date_sources),
        additional_instruction="The provided page(s) should explicitly state the person's birth date. Minor formatting variations are acceptable, but the date must match October 1, 1970."
    )

    # Accident highway name checks (value + supported by sources)
    hwy_main = evaluator.add_sequential(
        id="Accident_Highway_Name",
        desc="Identifies the correct accident highway (Angeles Crest Highway / California State Route 2), in the San Gabriel Mountains",
        parent=req_node,
        critical=True
    )

    # Value match against allowed variants (custom)
    hwy_value_ok = _highway_matches_expected(data.accident_highway)
    evaluator.add_custom_node(
        result=hwy_value_ok,
        id="accident_highway_value_match",
        desc="Accident highway matches one of: 'Angeles Crest Highway', 'California State Route 2' (CA-2/SR 2)",
        parent=hwy_main,
        critical=True
    )

    # Source support for highway claim
    hwy_leaf = evaluator.add_leaf(
        id="accident_highway_source_supported",
        desc="Accident highway identification supported by cited sources",
        parent=hwy_main,
        critical=True
    )
    await evaluator.verify(
        claim="The fatal accident occurred on Angeles Crest Highway (California State Route 2) in the San Gabriel Mountains.",
        node=hwy_leaf,
        sources=_dedup_urls(data.accident_highway_sources if data.accident_highway_sources else data.death_date_sources),
        additional_instruction="The page should explicitly name Angeles Crest Highway or California State Route 2 (CA-2/SR 2) as the location, within the San Gabriel Mountains."
    )

    # Accident elevation checks (approx value + supported by sources)
    elev_main = evaluator.add_sequential(
        id="Accident_Elevation_In_Feet",
        desc="Approximate elevation (~6,100 feet) of accident location near tunnels is provided and supported",
        parent=req_node,
        critical=True
    )

    elev_val = _extract_numeric_value(data.accident_elevation_ft)
    elev_ok = (elev_val is not None) and _approx_equal(elev_val, EXPECTED_ELEVATION_FT_TARGET, EXPECTED_ELEVATION_TOLERANCE_FT)
    evaluator.add_custom_node(
        result=elev_ok,
        id="accident_elevation_value_approx",
        desc=f"Accident elevation is approximately {EXPECTED_ELEVATION_FT_TARGET} feet (±{EXPECTED_ELEVATION_TOLERANCE_FT} ft allowed)",
        parent=elev_main,
        critical=True
    )

    elev_leaf = evaluator.add_leaf(
        id="accident_elevation_source_supported",
        desc="Elevation near tunnels (~6,100 ft) on this highway is supported by cited sources",
        parent=elev_main,
        critical=True
    )
    await evaluator.verify(
        claim=f"The accident location near a set of tunnels on Angeles Crest Highway is at an approximate elevation around {EXPECTED_ELEVATION_FT_TARGET} feet.",
        node=elev_leaf,
        sources=_dedup_urls(data.accident_elevation_sources if data.accident_elevation_sources else data.accident_highway_sources),
        additional_instruction="The page should support that near the relevant tunnels on Angeles Crest Highway (CA-2), the elevation is roughly around 6,100 ft (reasonable approximations accepted)."
    )


# --------------------------------------------------------------------------- #
# Identify executive node                                                     #
# --------------------------------------------------------------------------- #
async def verify_identify_executive(
    evaluator: Evaluator,
    parent,
    data: ExecAnswerExtraction
) -> None:
    """
    Build the 'Identify_Executive' critical sequential node:
      - Full name provided
      - Background_Constraints_Satisfied (parallel critical; handled by helper)
    """
    id_node = evaluator.add_sequential(
        id="Identify_Executive",
        desc="The response provides a specific executive (by full name) who satisfies the described background constraints.",
        parent=parent,
        critical=True
    )

    # Full name provided (critical)
    full_name_ok = bool(_norm_str(data.full_name))
    evaluator.add_custom_node(
        result=full_name_ok,
        id="Full_Name_Provided",
        desc="Provides the person’s full name.",
        parent=id_node,
        critical=True
    )

    # Background constraints (parallel critical)
    await verify_background_constraints(evaluator, id_node, data)


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
    Evaluate an answer for the executive identification task.
    """
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

    # Extract structured info from the answer
    extracted: ExecAnswerExtraction = await evaluator.extract(
        prompt=prompt_extract_exec_info(),
        template_class=ExecAnswerExtraction,
        extraction_name="extracted_exec_info"
    )

    # Top-level critical node reflecting the rubric "Complete_Answer"
    complete_node = evaluator.add_sequential(
        id="Complete_Answer",
        desc="Answer identifies the correct executive described and provides all requested attributes, consistent with the stated constraints.",
        parent=root,
        critical=True
    )

    # Identify the executive and verify background
    await verify_identify_executive(evaluator, complete_node, extracted)

    # Verify requested fields (birth date, highway, elevation)
    await verify_requested_fields(evaluator, complete_node, extracted)

    # Return evaluator summary
    return evaluator.get_summary()