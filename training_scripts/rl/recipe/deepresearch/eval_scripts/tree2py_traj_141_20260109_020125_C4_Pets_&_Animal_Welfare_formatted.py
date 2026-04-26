import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "vt_gfas_sanctuary"
TASK_DESCRIPTION = (
    "Identify a Vermont-based animal welfare organization that is a 501(c)(3) nonprofit, operates as a farm animal sanctuary, "
    "received GFAS (Global Federation of Animal Sanctuaries) accreditation in December 2025, maintains the minimum board size required "
    "by Vermont state law for nonprofit corporations, and holds currently valid accreditation status. Provide the organization's name."
)


class OrganizationExtraction(BaseModel):
    organization_name: Optional[str] = None
    homepage_url: Optional[str] = None
    gfas_url: Optional[str] = None
    board_page_url: Optional[str] = None
    other_source_urls: List[str] = Field(default_factory=list)
    location_state: Optional[str] = None
    org_type: Optional[str] = None
    is_501c3: Optional[str] = None
    gfas_accreditation_date: Optional[str] = None
    accreditation_status: Optional[str] = None
    board_member_count: Optional[str] = None
    board_member_names: List[str] = Field(default_factory=list)


def prompt_extract_organization_info() -> str:
    return (
        "Extract the single organization identified in the answer that is claimed to meet all constraints. Return:\n"
        "1. organization_name: The organization's name.\n"
        "2. homepage_url: The organization's main website URL (if provided).\n"
        "3. gfas_url: The specific GFAS accreditation page URL for the organization (if provided).\n"
        "4. board_page_url: A URL to the page listing board or governance (if provided).\n"
        "5. other_source_urls: All other URLs cited in the answer relevant to verifying location, nonprofit status, sanctuary type, accreditation, or board size.\n"
        "6. location_state: The state explicitly claimed for the organization's location (e.g., 'Vermont', 'VT').\n"
        "7. org_type: The type of organization claimed (e.g., 'farm animal sanctuary').\n"
        "8. is_501c3: Whether the answer claims 501(c)(3) status; extract the phrase as it appears (e.g., '501(c)(3)').\n"
        "9. gfas_accreditation_date: The accreditation date claimed in the answer for GFAS (e.g., 'December 2025').\n"
        "10. accreditation_status: Claimed current accreditation status (e.g., 'accredited', 'currently accredited').\n"
        "11. board_member_count: The count claimed for board members (if any; extract as a string).\n"
        "12. board_member_names: List of board member names if the answer provides them.\n\n"
        "Rules:\n"
        "- Extract only what is explicitly present in the answer.\n"
        "- For URLs, include full URLs; if protocol missing, prepend http://.\n"
        "- If a field is missing, return null; for lists, return empty lists.\n"
    )


def _unique_non_empty(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for u in urls:
        if not u:
            continue
        s = u.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


async def build_and_verify(evaluator: Evaluator, extracted: OrganizationExtraction) -> None:
    root_node = evaluator.add_parallel(
        id="Organization_Identification",
        desc="Correctly identify the Vermont-based farm animal sanctuary nonprofit that satisfies all stated legal and GFAS accreditation constraints",
        parent=evaluator.root,
        critical=True,
    )

    org_name = extracted.organization_name or "the identified organization"

    common_sources = _unique_non_empty(
        [
            extracted.homepage_url,
            extracted.board_page_url,
        ]
        + (extracted.other_source_urls or [])
    )

    # Vermont Location
    vt_node = evaluator.add_leaf(
        id="Vermont_Location",
        desc="The identified organization is physically located in Vermont",
        parent=root_node,
        critical=True,
    )
    vt_claim = f"The organization {org_name} is physically located in Vermont (VT)."
    vt_sources = common_sources
    await evaluator.verify(
        claim=vt_claim,
        node=vt_node,
        sources=vt_sources,
        additional_instruction=(
            "Confirm the organization's physical location is in the U.S. state of Vermont (VT). "
            "Look for an address or explicit statement on the organization's website (e.g., About/Contact pages) "
            "or other credible sources provided. Accept 'VT' as Vermont."
        ),
    )

    # 501(c)(3) Nonprofit Status
    nonprofit_node = evaluator.add_leaf(
        id="501c3_Nonprofit_Status",
        desc="The identified organization is a 501(c)(3) tax-exempt nonprofit organization",
        parent=root_node,
        critical=True,
    )
    nonprofit_claim = f"The organization {org_name} is a 501(c)(3) tax-exempt nonprofit organization."
    nonprofit_sources = common_sources
    await evaluator.verify(
        claim=nonprofit_claim,
        node=nonprofit_node,
        sources=nonprofit_sources,
        additional_instruction=(
            "Verify the presence of '501(c)(3)' or equivalent phrasing, or clear tax-exempt status indicators (e.g., EIN listed, IRS language) "
            "on the organization's site or the provided sources. Minor formatting variations like '501c3' should be acceptable."
        ),
    )

    # Farm Animal Sanctuary Type
    sanctuary_node = evaluator.add_leaf(
        id="Farm_Animal_Sanctuary_Type",
        desc="The identified organization operates as a farm animal sanctuary (not a companion animal shelter, wildlife rehabilitation center, or veterinary clinic)",
        parent=root_node,
        critical=True,
    )
    sanctuary_claim = (
        f"The organization {org_name} operates specifically as a farm animal sanctuary (serving farmed animals), "
        "not primarily as a companion animal shelter, wildlife rehabilitation center, or veterinary clinic."
    )
    sanctuary_sources = common_sources
    await evaluator.verify(
        claim=sanctuary_claim,
        node=sanctuary_node,
        sources=sanctuary_sources,
        additional_instruction=(
            "Use the organization's mission, program descriptions, and animal population info from the provided sources to confirm it is a farm animal sanctuary. "
            "Accept synonyms like 'farm sanctuary' or 'sanctuary for farmed animals'. If the organization's primary focus is companion animals or wildlife rehab, mark incorrect."
        ),
    )

    # GFAS Accreditation Received December 2025
    gfas_date_node = evaluator.add_leaf(
        id="GFAS_Accreditation_Received_Dec_2025",
        desc="The identified organization received GFAS (Global Federation of Animal Sanctuaries) accreditation in December 2025",
        parent=root_node,
        critical=True,
    )
    gfas_date_claim = f"The organization {org_name} received GFAS accreditation in December 2025."
    gfas_date_sources = _unique_non_empty([extracted.gfas_url] + (extracted.other_source_urls or []))
    await evaluator.verify(
        claim=gfas_date_claim,
        node=gfas_date_node,
        sources=gfas_date_sources,
        additional_instruction=(
            "Check the GFAS page or credible accreditation announcement among the provided sources to confirm the accreditation month/year is December 2025. "
            "Accept formatting like 'Dec 2025', 'December 2025', or a date string within that month/year."
        ),
    )

    # Current Accreditation Status
    gfas_current_node = evaluator.add_leaf(
        id="Current_Accreditation_Status",
        desc="The identified organization holds currently valid GFAS accreditation status (i.e., accreditation has not expired/been revoked; within the stated validity period if applicable)",
        parent=root_node,
        critical=True,
    )
    gfas_current_claim = f"The organization {org_name} currently holds a valid GFAS accreditation status."
    gfas_current_sources = _unique_non_empty([extracted.gfas_url] + (extracted.other_source_urls or []))
    await evaluator.verify(
        claim=gfas_current_claim,
        node=gfas_current_node,
        sources=gfas_current_sources,
        additional_instruction=(
            "Using the GFAS page or provided sources, verify that accreditation is shown as current/not expired or revoked as of the present evaluation date. "
            "If a validity window is displayed, ensure today's date falls within it. Today's date is 2026-01-11."
        ),
    )

    # Minimum Board Size (>= 3)
    board_node = evaluator.add_leaf(
        id="Minimum_Board_Size",
        desc="The identified organization maintains at least the minimum board size required by Vermont state law for nonprofit corporations (minimum of 3 board members per the stated constraint)",
        parent=root_node,
        critical=True,
    )
    board_claim = (
        f"The organization {org_name} maintains at least three (3) board members (meeting Vermont nonprofit corporate law minimum)."
    )
    board_sources = _unique_non_empty([extracted.board_page_url, extracted.homepage_url] + (extracted.other_source_urls or []))
    await evaluator.verify(
        claim=board_claim,
        node=board_node,
        sources=board_sources,
        additional_instruction=(
            "Check the organization's Board/Governance page or equivalent provided sources. "
            "Pass if there are at least three named board members/directors/trustees shown. "
            "If the page lists a count (>=3) or clearly displays at least three names, mark correct."
        ),
    )


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
    evaluator.initialize(
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_organization_info(),
        template_class=OrganizationExtraction,
        extraction_name="organization_extraction",
    )

    evaluator.add_custom_info(
        info={"evaluation_date": "2026-01-11", "constraints": {
            "state": "Vermont",
            "nonprofit_status": "501(c)(3)",
            "type": "Farm animal sanctuary",
            "gfas_accreditation_month_year": "December 2025",
            "current_accreditation": True,
            "vt_board_minimum": 3
        }},
        info_type="meta",
        info_name="evaluation_constraints"
    )

    await build_and_verify(evaluator, extracted)

    return evaluator.get_summary()