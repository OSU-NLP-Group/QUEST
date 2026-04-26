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
TASK_ID = "us_nonprofit_makerspaces"
TASK_DESCRIPTION = """I am interested in joining a community makerspace to pursue DIY projects and hobbies. I would like to identify three different nonprofit makerspaces in the United States that meet all of the following requirements:

1. The makerspace must be explicitly identified as a nonprofit organization or 501(c)(3) entity on their website or official materials.

2. The monthly membership fee must be less than $100 and publicly listed on their website.

3. The makerspace must offer 24/7 access (24-hour, around-the-clock access) to members.

4. The makerspace must provide access to at least three distinct categories of equipment from the following list: woodworking tools, metalworking tools, 3D printing equipment, laser cutting/engraving equipment, electronics equipment, textiles/sewing equipment, or CNC equipment.

5. The makerspace must have a complete physical street address (including street address, city, state, and ZIP code) located in the United States.

For each of the three makerspaces, please provide:
- The name of the makerspace
- Confirmation of nonprofit status with a reference URL
- The monthly membership cost with a reference URL
- Confirmation of 24/7 access with a reference URL
- A list of at least three equipment categories available with a reference URL
- The complete physical street address with a reference URL
"""

ALLOWED_EQUIPMENT_CATEGORIES = [
    "woodworking",
    "metalworking",
    "3d printing",
    "laser cutting/engraving",
    "electronics",
    "textiles/sewing",
    "cnc"
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MakerspaceEntry(BaseModel):
    name: Optional[str] = None

    nonprofit_statement: Optional[str] = None
    nonprofit_urls: List[str] = Field(default_factory=list)

    membership_monthly_fee: Optional[str] = None
    membership_urls: List[str] = Field(default_factory=list)

    access_247_statement: Optional[str] = None
    access_247_urls: List[str] = Field(default_factory=list)

    equipment_categories: List[str] = Field(default_factory=list)
    equipment_urls: List[str] = Field(default_factory=list)

    address_full: Optional[str] = None
    address_urls: List[str] = Field(default_factory=list)


class MakerspacesExtraction(BaseModel):
    makerspaces: List[MakerspaceEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_makerspaces() -> str:
    return """
    Extract up to three makerspaces mentioned in the answer along with structured fields needed to verify the requirements.
    For each makerspace, return an object with the following fields:

    1. name: The name of the makerspace as stated in the answer.
    2. nonprofit_statement: The sentence or phrase from the answer asserting nonprofit status (e.g., "501(c)(3) nonprofit").
    3. nonprofit_urls: All URLs in the answer that are used to confirm nonprofit status (official site pages, IRS listing, etc.). Only include URLs explicitly present in the answer.
    4. membership_monthly_fee: The monthly membership fee value mentioned for the makerspace (e.g., "$85 per month", "USD 90/mo"). If the answer gives multiple tiers, choose the standard individual/adult monthly price if available.
    5. membership_urls: All URLs in the answer that show pricing information or membership rates.
    6. access_247_statement: The phrase asserting 24/7 or 24-hour access from the answer (e.g., "members have 24/7 access").
    7. access_247_urls: All URLs in the answer that confirm 24/7 access policy.
    8. equipment_categories: A list of equipment categories the answer claims are available at the makerspace. Only use categories from this allowed set (case-insensitive, allow synonyms): woodworking, metalworking, 3D printing, laser cutting/engraving, electronics, textiles/sewing, CNC.
    9. equipment_urls: All URLs in the answer that show equipment/tools availability.
    10. address_full: The complete physical street address as stated in the answer (must include street, city, state, ZIP).
    11. address_urls: All URLs in the answer that show the physical address.

    Return a JSON object with a 'makerspaces' array of up to three objects (in the same order as they appear in the answer).
    If a field is missing for a makerspace, set it to null (for strings) or [] (for lists).
    Do not invent URLs or details not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def idx_label(idx: int) -> str:
    return ["First", "Second", "Third"][idx] if 0 <= idx < 3 else f"Makerspace_{idx+1}"


def safe_list(vals: Optional[List[str]]) -> List[str]:
    return vals if isinstance(vals, list) else []


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_makerspace(
    evaluator: Evaluator,
    parent_node,
    entry: MakerspaceEntry,
    idx: int,
) -> None:
    """
    Build verification sub-tree and run verifications for a single makerspace.
    """
    label = idx_label(idx)

    # Create parent node for this makerspace (non-critical, parallel aggregation)
    ms_node = evaluator.add_parallel(
        id=f"{label}_Makerspace",
        desc=f"{label} qualifying nonprofit makerspace with all required details",
        parent=parent_node,
        critical=False
    )

    name_for_claim = entry.name or f"{label} makerspace"

    # ---------------- Organization Credentials ----------------
    org_node = evaluator.add_parallel(
        id=f"makerspace_{idx}_organization_credentials",
        desc="Makerspace must be explicitly identified as a nonprofit organization or 501(c)(3) entity",
        parent=ms_node,
        critical=True
    )

    nonprofit_urls = safe_list(entry.nonprofit_urls)

    evaluator.add_custom_node(
        result=(len(nonprofit_urls) > 0),
        id=f"makerspace_{idx}_nonprofit_reference_url",
        desc="URL reference confirming nonprofit status",
        parent=org_node,
        critical=True
    )

    nonprofit_verified_leaf = evaluator.add_leaf(
        id=f"makerspace_{idx}_nonprofit_status_verified",
        desc="Direct statement of nonprofit status found on official website or organizational materials",
        parent=org_node,
        critical=True
    )
    nonprofit_claim = (
        f"The organization '{name_for_claim}' is explicitly described on the cited page(s) as a nonprofit "
        f"organization (e.g., 501(c)(3) nonprofit)."
    )
    await evaluator.verify(
        claim=nonprofit_claim,
        node=nonprofit_verified_leaf,
        sources=nonprofit_urls,
        additional_instruction=(
            "Check for explicit language like 'nonprofit', 'non-profit', '501(c)(3)', '501c3', or similar. "
            "The confirmation should be on official materials (e.g., the organization's own website or official listings)."
        ),
    )

    # ---------------- Membership Information ----------------
    membership_node = evaluator.add_parallel(
        id=f"makerspace_{idx}_membership_information",
        desc="Monthly membership fee must be publicly listed and less than $100",
        parent=ms_node,
        critical=True
    )

    membership_urls = safe_list(entry.membership_urls)

    evaluator.add_custom_node(
        result=(len(membership_urls) > 0),
        id=f"makerspace_{idx}_pricing_reference_url",
        desc="URL reference showing membership pricing information",
        parent=membership_node,
        critical=True
    )

    fee_listed_leaf = evaluator.add_leaf(
        id=f"makerspace_{idx}_monthly_fee_listed",
        desc="Monthly membership cost is publicly stated on the website",
        parent=membership_node,
        critical=True
    )
    fee_listed_claim = (
        f"The cited page(s) for '{name_for_claim}' explicitly list a monthly membership fee (a recurring per-month price)."
    )
    await evaluator.verify(
        claim=fee_listed_claim,
        node=fee_listed_leaf,
        sources=membership_urls,
        additional_instruction=(
            "Confirm that pricing is clearly labeled as monthly (e.g., '$85/month', 'USD 90 per month', 'Monthly membership'). "
            "Day passes, annual-only pricing, or one-time fees alone do not count."
        ),
    )

    fee_str = entry.membership_monthly_fee or "a monthly fee"
    fee_under_100_leaf = evaluator.add_leaf(
        id=f"makerspace_{idx}_fee_under_100",
        desc="Monthly membership cost is less than $100",
        parent=membership_node,
        critical=True
    )
    fee_under_100_claim = (
        f"The monthly membership fee for '{name_for_claim}' is {fee_str}, which is less than $100."
    )
    await evaluator.verify(
        claim=fee_under_100_claim,
        node=fee_under_100_leaf,
        sources=membership_urls,
        additional_instruction=(
            "Check the monthly price shown. If multiple monthly tiers exist, consider the standard individual/adult monthly tier. "
            "Trial offers or time-limited discounts should not be counted unless clearly presented as the normal monthly rate."
        ),
    )

    # ---------------- Access Privileges ----------------
    access_node = evaluator.add_parallel(
        id=f"makerspace_{idx}_access_privileges",
        desc="Makerspace must offer 24/7 member access",
        parent=ms_node,
        critical=True
    )

    access_urls = safe_list(entry.access_247_urls)

    evaluator.add_custom_node(
        result=(len(access_urls) > 0),
        id=f"makerspace_{idx}_access_reference_url",
        desc="URL reference confirming 24/7 access policy",
        parent=access_node,
        critical=True
    )

    access_leaf = evaluator.add_leaf(
        id=f"makerspace_{idx}_247_access_confirmed",
        desc="Explicit statement of 24/7 or 24-hour member access found",
        parent=access_node,
        critical=True
    )
    access_claim = (
        f"The makerspace '{name_for_claim}' offers 24/7 (around-the-clock) access to members."
    )
    await evaluator.verify(
        claim=access_claim,
        node=access_leaf,
        sources=access_urls,
        additional_instruction=(
            "Look for phrases such as '24/7 access', '24 hours a day', 'around-the-clock access', "
            "'key fob 24/7', or similar. It must clearly apply to members."
        ),
    )

    # ---------------- Equipment Inventory ----------------
    equipment_node = evaluator.add_parallel(
        id=f"makerspace_{idx}_equipment_inventory",
        desc="Makerspace must provide at least three distinct equipment categories",
        parent=ms_node,
        critical=True
    )

    equipment_urls = safe_list(entry.equipment_urls)
    evaluator.add_custom_node(
        result=(len(equipment_urls) > 0),
        id=f"makerspace_{idx}_equipment_reference_url",
        desc="URL reference showing equipment or tools available",
        parent=equipment_node,
        critical=True
    )

    categories_listed = entry.equipment_categories or []
    equipment_leaf = evaluator.add_leaf(
        id=f"makerspace_{idx}_min_three_categories",
        desc="At least three distinct equipment categories from the specified list are available (woodworking, metalworking, 3D printing, laser cutting, electronics, textiles/sewing, CNC)",
        parent=equipment_node,
        critical=True
    )
    equipment_claim = (
        f"On the cited page(s), the makerspace '{name_for_claim}' offers at least three distinct equipment categories among: "
        f"woodworking, metalworking, 3D printing, laser cutting/engraving, electronics, textiles/sewing, CNC. "
        f"The answer-listed categories are: {categories_listed}."
    )
    await evaluator.verify(
        claim=equipment_claim,
        node=equipment_leaf,
        sources=equipment_urls,
        additional_instruction=(
            "Confirm from the page(s) that at least three distinct categories from the allowed set are available. "
            "Allow common synonyms and shop names: 'woodshop' -> woodworking; 'metal shop' -> metalworking; "
            "'laser cutter/engraver' -> laser cutting/engraving; '3D printers' -> 3D printing; "
            "'electronics lab' -> electronics; 'sewing/textiles/fiber arts' -> textiles/sewing; "
            "'CNC router/mill' -> CNC."
        ),
    )

    # ---------------- Location Details ----------------
    location_node = evaluator.add_parallel(
        id=f"makerspace_{idx}_location_details",
        desc="Complete physical street address in the United States",
        parent=ms_node,
        critical=True
    )

    address_urls = safe_list(entry.address_urls)
    evaluator.add_custom_node(
        result=(len(address_urls) > 0),
        id=f"makerspace_{idx}_address_reference_url",
        desc="URL reference showing physical address",
        parent=location_node,
        critical=True
    )

    addr_example = entry.address_full or "a complete street address"
    address_complete_leaf = evaluator.add_leaf(
        id=f"makerspace_{idx}_complete_address_provided",
        desc="Full street address including street, city, state, and ZIP code is provided",
        parent=location_node,
        critical=True
    )
    address_complete_claim = (
        f"The cited page(s) show a complete street address for '{name_for_claim}', including street, city, state, and ZIP code "
        f"(example extracted: '{addr_example}')."
    )
    await evaluator.verify(
        claim=address_complete_claim,
        node=address_complete_leaf,
        sources=address_urls,
        additional_instruction=(
            "Verify that the address includes all components: street/address line with number, city, state (full name or USPS abbreviation), "
            "and ZIP code (5-digit or ZIP+4)."
        ),
    )

    address_us_leaf = evaluator.add_leaf(
        id=f"makerspace_{idx}_us_location_verified",
        desc="Address is located in the United States",
        parent=location_node,
        critical=True
    )
    address_us_claim = (
        f"The address for '{name_for_claim}' is located in the United States."
    )
    await evaluator.verify(
        claim=address_us_claim,
        node=address_us_leaf,
        sources=address_urls,
        additional_instruction=(
            "Confirm that the address is in the USA. Evidence may include a US state name/abbreviation, 'United States', or a valid US ZIP code pattern."
        ),
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
    """
    Evaluate an answer for the nonprofit makerspaces task.
    """
    # Initialize evaluator with parallel root (three makerspaces evaluated independently)
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

    # Record requirements as ground truth context
    evaluator.add_ground_truth({
        "requirements": {
            "nonprofit": "Explicit nonprofit or 501(c)(3) on official materials",
            "membership_fee": "Monthly fee publicly listed and < $100",
            "access": "24/7 member access",
            "equipment": f"At least three categories among {ALLOWED_EQUIPMENT_CATEGORIES}",
            "address": "Complete US street address (street, city, state, ZIP)"
        }
    })

    # Extract makerspaces from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_makerspaces(),
        template_class=MakerspacesExtraction,
        extraction_name="makerspaces_extraction",
    )

    # Prepare exactly three entries (pad with empty entries if needed)
    entries: List[MakerspaceEntry] = list(extracted.makerspaces[:3])
    while len(entries) < 3:
        entries.append(MakerspaceEntry())

    # Build subtrees for each makerspace (parallel)
    for i in range(3):
        await verify_single_makerspace(evaluator, root, entries[i], i)

    # Add custom info about allowed equipment categories
    evaluator.add_custom_info(
        {"allowed_equipment_categories": ALLOWED_EQUIPMENT_CATEGORIES},
        info_type="config",
        info_name="equipment_category_policy"
    )

    # Return structured summary
    return evaluator.get_summary()