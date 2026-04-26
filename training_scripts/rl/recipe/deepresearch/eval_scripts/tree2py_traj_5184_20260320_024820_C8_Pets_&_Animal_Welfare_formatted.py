import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "emergency_vet_phx"
TASK_DESCRIPTION = (
    "I am relocating to Phoenix, Arizona with my pets and need to identify reliable emergency veterinary facilities "
    "in case of urgent situations. Please identify four distinct 24-hour emergency veterinary facilities located in the "
    "Phoenix metropolitan area (including Phoenix, Scottsdale, Glendale, Peoria, Tempe, Mesa, Chandler, or Gilbert). "
    "For each facility, provide: 1) The facility name, 2) A reference URL (from the facility's official website or a "
    "reliable veterinary directory), 3) Confirmation that the facility operates 24 hours a day, 7 days a week, 365 days "
    "a year, 4) The complete physical address showing it is located within the Phoenix metro area, 5) A publicly available "
    "emergency phone number, 6) Confirmation that the facility offers on-site emergency surgical services, and 7) "
    "Confirmation that the facility provides diagnostic services (such as x-rays, ultrasounds, or laboratory testing) "
    "on-site. Each of the four facilities must be distinct organizations (not different locations of the same chain or "
    "hospital system)."
)

PHX_METRO_CITIES = [
    "Phoenix", "Scottsdale", "Glendale", "Peoria", "Tempe", "Mesa", "Chandler", "Gilbert"
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FacilityItem(BaseModel):
    name: Optional[str] = None
    reference_url: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    operations_24_7_text: Optional[str] = None
    emergency_surgical_services_text: Optional[str] = None
    diagnostic_services_text: Optional[str] = None


class FacilitiesExtraction(BaseModel):
    facilities: List[FacilityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return """
    Extract all emergency veterinary facilities mentioned in the answer. For each facility, extract:
    - name: Facility or hospital name exactly as written in the answer.
    - reference_url: A URL from the facility's official website or a reliable veterinary directory that the answer cites for that facility. If multiple URLs are present, select the most official or most directly relevant one. If none is provided, set null.
    - address: The full physical address as provided in the answer (street, city, state, ZIP) if present; otherwise null.
    - phone: The publicly available phone number as provided in the answer (keep formatting) if present; otherwise null.
    - operations_24_7_text: Any quoted or paraphrased text from the answer indicating 24/7/365 emergency operations, if present; otherwise null.
    - emergency_surgical_services_text: Any quoted or paraphrased text from the answer indicating on-site emergency surgical services, if present; otherwise null.
    - diagnostic_services_text: Any quoted or paraphrased text from the answer indicating on-site diagnostic services (e.g., x-rays, ultrasound, in-house lab), if present; otherwise null.

    Return a JSON object with a single field:
    {
      "facilities": [ { ... }, { ... }, ... ]
    }

    Notes:
    - Do not invent data; only extract what is explicitly present in the answer.
    - For reference_url, extract the actual URL string that appears in the answer. If the answer uses markdown links, return the URL target.
    - The answer may contain more than four facilities; include all that appear. Missing items should be null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _valid_url(u: Optional[str]) -> bool:
    return _non_empty(u) and (u.strip().startswith("http://") or u.strip().startswith("https://"))


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_facility(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    facility: FacilityItem,
    idx: int,
) -> Dict[str, VerificationNode]:
    """
    Build the verification subtree for a single facility and run all checks.

    Returns a dict with references to key leaf nodes to support distinctness prerequisites.
    """
    num = idx + 1
    facility_node = evaluator.add_parallel(
        id=f"Facility_{num}",
        desc=f"{['First','Second','Third','Fourth'][idx]} emergency veterinary facility meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # Existence checks (critical)
    name_present = _non_empty(facility.name)
    name_node = evaluator.add_custom_node(
        result=name_present,
        id=f"Facility_Name_{num}",
        desc="A facility name is provided",
        parent=facility_node,
        critical=True
    )

    url_present = _valid_url(facility.reference_url)
    url_node = evaluator.add_custom_node(
        result=url_present,
        id=f"Reference_URL_{num}",
        desc="A valid reference URL from the facility's official website or a reliable veterinary directory confirming the facility's existence and services",
        parent=facility_node,
        critical=True
    )

    # 24/7 operations verification (critical)
    op_leaf = evaluator.add_leaf(
        id=f"24_7_Operations_{num}",
        desc="The facility explicitly states it provides 24-hour emergency veterinary services, open 7 days a week, 365 days a year",
        parent=facility_node,
        critical=True
    )
    claim_24_7 = (
        f"The facility named '{facility.name or 'the facility'}' provides 24-hour emergency veterinary services and is open 24/7 "
        f"(24 hours a day, 7 days a week, year-round)."
    )
    await evaluator.verify(
        claim=claim_24_7,
        node=op_leaf,
        sources=facility.reference_url if url_present else None,
        additional_instruction=(
            "Check the webpage for phrases like '24/7', '24 hours', 'open 24 hours', 'always open', "
            "'365 days a year', or 'every day including holidays'. If the page clearly indicates 24/7 emergency "
            "availability, consider this supported even if '365' is not explicitly written."
        )
    )

    # Phoenix metro location verification (critical)
    loc_leaf = evaluator.add_leaf(
        id=f"Phoenix_Metro_Location_{num}",
        desc="The facility has a physical address located within Phoenix, Arizona or the greater Phoenix metro area (including Scottsdale, Glendale, Peoria, Tempe, Mesa, Chandler, Gilbert)",
        parent=facility_node,
        critical=True
    )
    addr_text = f" The address is '{facility.address}'." if _non_empty(facility.address) else ""
    claim_loc = (
        f"The facility '{facility.name or 'the facility'}' has a physical address located in the Phoenix metropolitan area "
        f"(i.e., in Phoenix, Scottsdale, Glendale, Peoria, Tempe, Mesa, Chandler, or Gilbert).{addr_text}"
    )
    await evaluator.verify(
        claim=claim_loc,
        node=loc_leaf,
        sources=facility.reference_url if url_present else None,
        additional_instruction=(
            "Verify that the webpage lists a full street address whose city is one of: Phoenix, Scottsdale, Glendale, "
            "Peoria, Tempe, Mesa, Chandler, Gilbert (all in Arizona). Minor formatting differences in the address are acceptable."
        )
    )

    # Emergency phone number verification (critical)
    phone_leaf = evaluator.add_leaf(
        id=f"Emergency_Phone_Number_{num}",
        desc="The facility provides a publicly available phone number for emergency contact",
        parent=facility_node,
        critical=True
    )
    if _non_empty(facility.phone):
        claim_phone = (
            f"The webpage lists the emergency contact phone number {facility.phone} for the facility "
            f"(format variations are acceptable)."
        )
        add_ins_phone = (
            "Confirm a publicly listed phone number on the page. Prefer numbers explicitly marked for 'emergency' or "
            "'call us' for urgent care, but if only a main line is provided and is used for emergencies, that is acceptable."
        )
    else:
        claim_phone = (
            "The webpage lists a publicly available phone number to call for emergency veterinary care."
        )
        add_ins_phone = (
            "Confirm that at least one phone number is present on the page for contacting the facility for emergency or urgent care."
        )
    await evaluator.verify(
        claim=claim_phone,
        node=phone_leaf,
        sources=facility.reference_url if url_present else None,
        additional_instruction=add_ins_phone
    )

    # Emergency surgical services verification (critical)
    surg_leaf = evaluator.add_leaf(
        id=f"Emergency_Surgical_Services_{num}",
        desc="The facility explicitly states it offers on-site emergency surgical services",
        parent=facility_node,
        critical=True
    )
    claim_surg = (
        f"The facility '{facility.name or 'the facility'}' offers on-site emergency surgical services "
        f"(e.g., emergency surgery, surgical suite, surgeons available)."
    )
    await evaluator.verify(
        claim=claim_surg,
        node=surg_leaf,
        sources=facility.reference_url if url_present else None,
        additional_instruction=(
            "Look for wording such as 'emergency surgery', 'on-site surgery', 'in-house surgical suite', "
            "'24/7 surgeons', 'board-certified surgeon available'. It must be provided by this facility on-site "
            "(not merely via referral elsewhere)."
        )
    )

    # Diagnostic services verification (critical)
    diag_leaf = evaluator.add_leaf(
        id=f"Diagnostic_Services_{num}",
        desc="The facility explicitly states it provides diagnostic services such as x-rays, ultrasounds, or laboratory testing on-site",
        parent=facility_node,
        critical=True
    )
    claim_diag = (
        f"The facility '{facility.name or 'the facility'}' provides on-site diagnostic services such as "
        f"x-rays/radiography, ultrasound, and/or in-house laboratory testing."
    )
    await evaluator.verify(
        claim=claim_diag,
        node=diag_leaf,
        sources=facility.reference_url if url_present else None,
        additional_instruction=(
            "Accept synonyms like 'digital radiography', 'x-ray', 'ultrasonography', 'point-of-care ultrasound', "
            "'in-house lab', 'on-site laboratory', 'diagnostic imaging'. The services must be available on-site at this facility."
        )
    )

    return {
        "facility_node": facility_node,
        "name_node": name_node,
        "url_node": url_node,
    }


async def verify_distinctness(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    facilities: List[FacilityItem],
    prereq_nodes: List[Dict[str, VerificationNode]],
) -> None:
    """
    Build the 'Facilities_Are_Distinct' critical node and verify pairwise distinctness
    across the four facilities (not different locations of the same chain/hospital system).
    """
    distinct_parent = evaluator.add_parallel(
        id="Facilities_Are_Distinct",
        desc="All four identified facilities must be distinct organizations (not different locations of the same chain or hospital system)",
        parent=parent_node,
        critical=True
    )

    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    for (i, j) in pairs:
        leaf = evaluator.add_leaf(
            id=f"Facilities_Distinct_{i+1}_{j+1}",
            desc=f"Facility #{i+1} and Facility #{j+1} are distinct organizations (not different locations of the same chain or hospital system)",
            parent=distinct_parent,
            critical=True
        )

        name_i = facilities[i].name or f"Facility #{i+1}"
        name_j = facilities[j].name or f"Facility #{j+1}"
        claim = (
            f"'{name_i}' and '{name_j}' are distinct veterinary organizations and are not merely different locations "
            f"or brand names of the same chain/hospital system."
        )

        srcs: List[str] = []
        if _valid_url(facilities[i].reference_url):
            srcs.append(facilities[i].reference_url.strip())
        if _valid_url(facilities[j].reference_url):
            srcs.append(facilities[j].reference_url.strip())

        # Make verification depend on the existence of names and URLs for both facilities
        extra_pre = [
            prereq_nodes[i]["name_node"],
            prereq_nodes[i]["url_node"],
            prereq_nodes[j]["name_node"],
            prereq_nodes[j]["url_node"],
        ]

        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=srcs if srcs else None,
            additional_instruction=(
                "Use the provided webpages to determine organizational identity. If the two facilities are part of the "
                "same corporate brand or hospital system (e.g., VCA, BluePearl, NVA, Ethos, Pathway, or any other common brand), "
                "then they are NOT distinct organizations. Distinct organizations typically have different brand families, "
                "owners, or governance, not just different street addresses under the same brand."
            ),
            extra_prerequisites=extra_pre
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
    Evaluate an answer for the Phoenix emergency veterinary facilities task.
    """
    # Initialize evaluator with a parallel root (critical distinctness + 4 facilities in parallel)
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

    # Record helpful context
    evaluator.add_custom_info(
        info={"phoenix_metro_cities": PHX_METRO_CITIES},
        info_type="context",
        info_name="allowed_cities"
    )

    # Extract facilities from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extracted"
    )

    # Prepare exactly four facilities (pad if fewer)
    facilities: List[FacilityItem] = list(extracted.facilities[:4])
    while len(facilities) < 4:
        facilities.append(FacilityItem())

    # Build verification subtrees for four facilities
    built_nodes: List[Dict[str, VerificationNode]] = []
    for i in range(4):
        nodes = await build_and_verify_facility(evaluator, root, facilities[i], i)
        built_nodes.append(nodes)

    # Distinctness verification (critical)
    await verify_distinctness(evaluator, root, facilities, built_nodes)

    # Return structured summary
    return evaluator.get_summary()