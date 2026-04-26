import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "midwest_immunization_depts"
TASK_DESCRIPTION = (
    "Following the January 2026 update to the CDC's childhood immunization schedule, which reduced universal vaccine "
    "recommendations from 17 to 11 diseases, identify four county health departments in Ohio, Indiana, or Michigan "
    "that operate immunization clinics providing vaccines for children. For each, provide county/state, clinic address, "
    "official website URL, phone, at least two vaccines from the 2026 universal/routine category, clinic hours, and "
    "walk-in availability. Each entry must be supported by an official county health department or credible government "
    "health source URL."
)

ALLOWED_STATES = {"ohio", "indiana", "michigan", "oh", "in", "mi"}

# Universal vaccine categories and common synonyms/aliases
VAX_SYNONYMS = {
    "diphtheria": ["diphtheria", "dtap", "td", "tdap"],
    "tetanus": ["tetanus", "dtap", "td", "tdap"],
    "pertussis": ["pertussis", "whooping cough", "dtap", "tdap"],
    "hib": ["hib", "haemophilus influenzae type b", "haemophilus influenzae b"],
    "pneumococcal conjugate": ["pneumococcal", "pcv", "pcv13", "pcv15", "pcv20", "pneumococcal conjugate"],
    "polio": ["polio", "ipv", "ipol"],
    "measles": ["measles", "mmr"],
    "mumps": ["mumps", "mmr"],
    "rubella": ["rubella", "mmr"],
    "hpv": ["hpv", "human papillomavirus", "gardasil"],
    "varicella": ["varicella", "chickenpox"],
}

# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
class HealthDeptItem(BaseModel):
    county: Optional[str] = None
    state: Optional[str] = None  # Can be full name or USPS abbreviation
    address: Optional[str] = None  # Full physical clinic address
    website_url: Optional[str] = None  # Official health department website
    phone: Optional[str] = None
    vaccines: List[str] = Field(default_factory=list)  # As stated in the answer
    hours: Optional[str] = None  # Free-text hours, include days/times if present
    walk_in: Optional[str] = None  # e.g., "walk-in", "appointments only", "both", or descriptive
    reference_urls: List[str] = Field(default_factory=list)  # Official or credible government sources


class HealthDeptExtraction(BaseModel):
    items: List[HealthDeptItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_health_depts() -> str:
    """
    Instruct the extractor to parse up to 4 county health department entries with required fields.
    """
    return """
You will extract up to four county health departments in Ohio, Indiana, or Michigan that (according to the provided answer)
operate pediatric immunization clinics. Strictly extract what is explicitly present in the answer.

For each health department, extract the following fields:
- county: The county name (e.g., "Wayne")
- state: The state, using either the full name "Ohio/Indiana/Michigan" or standard postal abbreviations "OH/IN/MI"
- address: The complete physical clinic address as written (include street, city, state, ZIP if present)
- website_url: The official health department website URL (if provided)
- phone: A contact phone number for the clinic or main department
- vaccines: A list of vaccine names that the answer claims this clinic offers; include at least two that clearly come from the 2026 universal/routine set when available.
  These 11 categories are: diphtheria, tetanus, pertussis (whooping cough), Haemophilus influenzae type b (Hib),
  pneumococcal conjugate (PCV), polio, measles, mumps, rubella, human papillomavirus (HPV), varicella (chickenpox).
  Accept common combination names and abbreviations, e.g., DTaP / Tdap (counts toward diphtheria/tetanus/pertussis)
  and MMR (counts toward measles/mumps/rubella), PCV (pneumococcal conjugate), IPV (polio).
- hours: The clinic operational hours (include specific days and times if present). Return a concise text snippet exactly as in the answer.
- walk_in: The walk-in availability status (e.g., "walk-ins accepted", "appointments only", "both", or a short text as stated)
- reference_urls: 1–3 URLs cited in the answer that support this clinic’s immunization services or details. These should be
  official county/state/municipal health department pages or credible .gov sources. If the answer provides URLs in markdown,
  extract the actual links. If none are present for an item, return an empty list.

Return a JSON object with a single field:
- items: an array of up to four objects, each with the specified fields.

Do not fabricate information; if any field is missing in the answer, set it to null or an empty list as appropriate.
Only extract URLs explicitly present in the answer text (including markdown links).
"""


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def is_universal_vaccine(name: str) -> bool:
    if not name:
        return False
    s = name.strip().lower()
    for syns in VAX_SYNONYMS.values():
        for token in syns:
            if token in s:
                return True
    return False


def select_two_universal(vaccines: List[str]) -> List[str]:
    """Return up to two vaccine strings (as written in the answer) that map to the 11-category universal set."""
    seen_lower = set()
    picked: List[str] = []
    for v in vaccines or []:
        if not v:
            continue
        if is_universal_vaccine(v):
            key = v.strip().lower()
            if key not in seen_lower:
                seen_lower.add(key)
                picked.append(v.strip())
                if len(picked) == 2:
                    break
    return picked


def normalize_state_name(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    val = s.strip().lower()
    if val in {"oh"}:
        return "Ohio"
    if val in {"in"}:
        return "Indiana"
    if val in {"mi"}:
        return "Michigan"
    if val in {"ohio", "indiana", "michigan"}:
        return val.capitalize()
    return s


def get_sources_for_item(item: HealthDeptItem) -> Optional[List[str] | str]:
    """Prefer the provided reference URLs; fall back to the official website if no references."""
    if item.reference_urls:
        return item.reference_urls
    return item.website_url if item.website_url else None


def ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(n, f"#{n}")


def categorize_walk_in_status(text: Optional[str]) -> str:
    if not text:
        return "unknown"
    s = text.strip().lower()
    if "walk-in" in s or "walk in" in s or "walkins" in s:
        if "appointment" in s:
            # phrases like "walk-ins accepted; appointments preferred" -> treat as both
            return "both"
        return "walk-in"
    if "appointment only" in s or "appointments only" in s or "by appointment only" in s:
        return "appointment only"
    if "appointment" in s:
        # generic "by appointment" without negating walk-ins -> lean to appointment only
        return "appointment only"
    return "unknown"


# -----------------------------------------------------------------------------
# Verification builder for each department
# -----------------------------------------------------------------------------
async def verify_health_dept(
    evaluator: Evaluator,
    parent_node,
    item: HealthDeptItem,
    index_1based: int,
) -> None:
    dept_node = evaluator.add_parallel(
        id=f"Health_Department_{index_1based}",
        desc=f"{ordinal(index_1based)} county health department meeting all specified criteria",
        parent=parent_node,
        critical=False,
    )

    # ----- Location group (critical) -----
    loc_node = evaluator.add_parallel(
        id=f"HD{index_1based}_Location",
        desc=f"Geographic location information for the {ordinal(index_1based).lower()} health department",
        parent=dept_node,
        critical=True,
    )

    # County leaf
    if item.county and item.county.strip():
        county_leaf = evaluator.add_leaf(
            id=f"HD{index_1based}_County",
            desc="The county name is provided",
            parent=loc_node,
            critical=True,
        )
        county_claim = (
            f"This webpage provides immunization clinic/service information for {item.county.strip()} County "
            f"or the {item.county.strip()} County Health Department."
        )
        await evaluator.verify(
            claim=county_claim,
            node=county_leaf,
            sources=get_sources_for_item(item),
            additional_instruction="Accept if the page clearly indicates the named county in page title, header, breadcrumb, or body. Minor formatting variations are acceptable.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"HD{index_1based}_County",
            desc="The county name is provided",
            parent=loc_node,
            critical=True,
        )

    # State leaf (logic check that state ∈ {OH/IN/MI})
    if item.state and item.state.strip():
        state_leaf = evaluator.add_leaf(
            id=f"HD{index_1based}_State",
            desc="The state is one of Ohio, Indiana, or Michigan",
            parent=loc_node,
            critical=True,
        )
        state_claim = (
            f"The stated state value '{item.state.strip()}' denotes one of: Ohio (OH), Indiana (IN), or Michigan (MI). "
            f"Treat standard USPS abbreviations as valid."
        )
        await evaluator.verify(
            claim=state_claim,
            node=state_leaf,
            # Logic-only verification; no URL required
            additional_instruction="Judge purely by the provided value; do not use external knowledge beyond mapping OH→Ohio, IN→Indiana, MI→Michigan.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"HD{index_1based}_State",
            desc="The state is one of Ohio, Indiana, or Michigan",
            parent=loc_node,
            critical=True,
        )

    # ----- Contact group (critical) -----
    contact_node = evaluator.add_parallel(
        id=f"HD{index_1based}_Contact",
        desc=f"Contact and location details for the {ordinal(index_1based).lower()} health department",
        parent=dept_node,
        critical=True,
    )

    # Address leaf
    if item.address and item.address.strip():
        addr_leaf = evaluator.add_leaf(
            id=f"HD{index_1based}_Address",
            desc="A complete physical address for the immunization clinic is provided",
            parent=contact_node,
            critical=True,
        )
        addr_claim = f"The immunization clinic physical address is: {item.address.strip()}."
        await evaluator.verify(
            claim=addr_claim,
            node=addr_leaf,
            sources=get_sources_for_item(item),
            additional_instruction="Verify that the page lists this same location/address (allow minor punctuation/format variations and ZIP+4 vs ZIP).",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"HD{index_1based}_Address",
            desc="A complete physical address for the immunization clinic is provided",
            parent=contact_node,
            critical=True,
        )

    # Phone leaf
    if item.phone and item.phone.strip():
        phone_leaf = evaluator.add_leaf(
            id=f"HD{index_1based}_Phone",
            desc="A contact phone number is provided",
            parent=contact_node,
            critical=True,
        )
        phone_claim = f"The contact phone number for the clinic or department is {item.phone.strip()}."
        await evaluator.verify(
            claim=phone_claim,
            node=phone_leaf,
            sources=get_sources_for_item(item),
            additional_instruction="Phone formatting (parentheses, spaces, hyphens) may vary; treat digit-equivalent numbers as matches.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"HD{index_1based}_Phone",
            desc="A contact phone number is provided",
            parent=contact_node,
            critical=True,
        )

    # Website URL presence leaf
    evaluator.add_custom_node(
        result=bool(item.website_url and item.website_url.strip()),
        id=f"HD{index_1based}_Website",
        desc="The official health department website URL is provided",
        parent=contact_node,
        critical=True,
    )

    # ----- Services group (critical) -----
    services_node = evaluator.add_parallel(
        id=f"HD{index_1based}_Services",
        desc=f"Service information for the {ordinal(index_1based).lower()} health department",
        parent=dept_node,
        critical=True,
    )

    picked_vax = select_two_universal(item.vaccines)

    # Vaccine1 leaf
    if len(picked_vax) >= 1:
        v1_leaf = evaluator.add_leaf(
            id=f"HD{index_1based}_Vaccine1",
            desc="At least one vaccine from the 2026 universal/routine category (diphtheria, tetanus, pertussis, Hib, pneumococcal, polio, measles, mumps, rubella, HPV, or varicella) is listed as offered",
            parent=services_node,
            critical=True,
        )
        v1_claim = f"The clinic offers the vaccine '{picked_vax[0]}'."
        await evaluator.verify(
            claim=v1_claim,
            node=v1_leaf,
            sources=get_sources_for_item(item),
            additional_instruction=(
                "Treat common combination names/abbreviations as valid universal vaccines: "
                "DTaP/Tdap (diphtheria, tetanus, pertussis), MMR (measles, mumps, rubella), "
                "PCV (pneumococcal conjugate), IPV (polio), Hib (Haemophilus influenzae type b), "
                "HPV (human papillomavirus), Varicella (chickenpox)."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"HD{index_1based}_Vaccine1",
            desc="At least one vaccine from the 2026 universal/routine category (diphtheria, tetanus, pertussis, Hib, pneumococcal, polio, measles, mumps, rubella, HPV, or varicella) is listed as offered",
            parent=services_node,
            critical=True,
        )

    # Vaccine2 leaf
    if len(picked_vax) >= 2:
        v2_leaf = evaluator.add_leaf(
            id=f"HD{index_1based}_Vaccine2",
            desc="A second vaccine from the 2026 universal/routine category is listed as offered",
            parent=services_node,
            critical=True,
        )
        v2_claim = f"The clinic offers the vaccine '{picked_vax[1]}'."
        await evaluator.verify(
            claim=v2_claim,
            node=v2_leaf,
            sources=get_sources_for_item(item),
            additional_instruction=(
                "Treat common combination names/abbreviations as valid universal vaccines: "
                "DTaP/Tdap (diphtheria, tetanus, pertussis), MMR (measles, mumps, rubella), "
                "PCV (pneumococcal conjugate), IPV (polio), Hib (Haemophilus influenzae type b), "
                "HPV (human papillomavirus), Varicella (chickenpox). The second vaccine must be distinct from the first."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"HD{index_1based}_Vaccine2",
            desc="A second vaccine from the 2026 universal/routine category is listed as offered",
            parent=services_node,
            critical=True,
        )

    # Hours leaf
    if item.hours and item.hours.strip():
        hours_leaf = evaluator.add_leaf(
            id=f"HD{index_1based}_Hours",
            desc="Clinic operational hours (specific days and times) are provided",
            parent=services_node,
            critical=True,
        )
        hours_claim = f"The clinic operational hours include: {item.hours.strip()}."
        await evaluator.verify(
            claim=hours_claim,
            node=hours_leaf,
            sources=get_sources_for_item(item),
            additional_instruction="Allow minor rephrasings/formatting; confirm that the page states equivalent days and times.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"HD{index_1based}_Hours",
            desc="Clinic operational hours (specific days and times) are provided",
            parent=services_node,
            critical=True,
        )

    # Walk-in leaf
    if item.walk_in and item.walk_in.strip():
        walk_leaf = evaluator.add_leaf(
            id=f"HD{index_1based}_WalkIn",
            desc="Information about walk-in availability (yes/no or specific walk-in hours) is provided",
            parent=services_node,
            critical=True,
        )
        category = categorize_walk_in_status(item.walk_in)
        if category == "walk-in":
            walk_claim = "The clinic accepts walk-ins for immunizations (no appointment required)."
        elif category == "appointment only":
            walk_claim = "The clinic requires appointments for immunizations (no walk-ins)."
        elif category == "both":
            walk_claim = "The clinic accepts walk-ins and also offers appointments for immunizations."
        else:
            walk_claim = f"Walk-in availability is described as: {item.walk_in.strip()}."
        await evaluator.verify(
            claim=walk_claim,
            node=walk_leaf,
            sources=get_sources_for_item(item),
            additional_instruction="Accept semantically equivalent phrasing like 'walk-ins welcome', 'no walk-ins', 'by appointment only', or 'walk-ins on specific days'.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"HD{index_1based}_WalkIn",
            desc="Information about walk-in availability (yes/no or specific walk-in hours) is provided",
            parent=services_node,
            critical=True,
        )

    # ----- Reference (critical leaf) -----
    # Must have at least one official/credible government health page supporting the entry
    if item.reference_urls:
        ref_leaf = evaluator.add_leaf(
            id=f"HD{index_1based}_Reference",
            desc=f"URL reference supporting the information about the {ordinal(index_1based).lower()} health department",
            parent=dept_node,
            critical=True,
        )
        ref_claim = (
            "This webpage is an official county/state/municipal public health department page, or a credible .gov health source, "
            "and it provides information about immunization clinic services for children (pediatric)."
        )
        await evaluator.verify(
            claim=ref_claim,
            node=ref_leaf,
            sources=item.reference_urls,
            additional_instruction="Accept county/state health department domains (including subdomains) and .gov sites. PDFs hosted on these sites are acceptable.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"HD{index_1based}_Reference",
            desc=f"URL reference supporting the information about the {ordinal(index_1based).lower()} health department",
            parent=dept_node,
            critical=True,
        )


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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

    # Extract structured items
    extraction: HealthDeptExtraction = await evaluator.extract(
        prompt=prompt_extract_health_depts(),
        template_class=HealthDeptExtraction,
        extraction_name="health_departments",
    )

    # Prepare exactly four entries (pad with empty if fewer; take first four if more)
    items: List[HealthDeptItem] = list(extraction.items or [])
    items = items[:4]
    while len(items) < 4:
        items.append(HealthDeptItem())

    # Build verification tree for each department (parallel under root)
    for idx, item in enumerate(items, start=1):
        await verify_health_dept(evaluator, root, item, idx)

    return evaluator.get_summary()