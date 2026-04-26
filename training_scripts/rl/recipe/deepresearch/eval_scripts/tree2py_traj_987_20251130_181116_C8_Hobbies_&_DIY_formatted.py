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
TASK_ID = "atl_makerspaces"
TASK_DESCRIPTION = (
    "You are interested in pursuing woodworking as a hobby and need to find a suitable community makerspace in Atlanta, Georgia. "
    "To help you make an informed decision, identify three community makerspaces located in the Atlanta metropolitan area that offer both "
    "woodworking facilities and 24/7 member access. For each of the three makerspaces, provide the following information: "
    "(1) Name of the makerspace, (2) Specific street address, (3) Monthly membership cost, (4) One-time application or setup fees (if any), "
    "(5) Access method (e.g., key fob, keycard, code), (6) Safety training requirements for woodworking equipment, (7) Age restrictions (if any), "
    "(8) Guest policy (whether guests/visitors are allowed), (9) Class discount benefits for members, (10) Storage options available to members, "
    "(11) Onboarding or joining process description, (12) Contact information (email, phone, or website), and (13) Reference URL for verification. "
    "Present your findings in a clear, organized format that allows for easy comparison between the three makerspaces."
)

# Ground-truth constraints (recorded for transparency; used to guide verification claims)
GROUND_TRUTH_CONSTRAINTS = {
    "MASS Collective": {
        "name": "MASS Collective",
        "address": "364 Nelson Street SW, Atlanta, GA 30313",
        "metro": "Atlanta metropolitan area",
        "woodworking": True,
        "access_24_7": True,
        "access_method": "access fob",
        "monthly_cost": "180/month or $360 for 3 months",  # phrased to allow textual variants
        "class_discount": "10% discount on classes",
        "storage": "personal member storage",
    },
    "Decatur Makers": {
        "name": "Decatur Makers",
        "address": "605 W. Ponce de Leon Ave., Decatur, GA 30030",
        "metro": "Atlanta metropolitan area",
        "woodworking": True,
        "access_24_7": True,  # 24/7/365
        "access_method": "key fob",
        "monthly_cost": "$35/month (individual membership)",
        "one_time_fee": "$60 one-time application fee (individual membership)",
        "safety_training": "woodshop plus other listed trainings as applicable",
        "age_restriction": "kids 11+ can join under Student Membership",
        "background_check": "required for members 18+",
        "guest_policy": "guests allowed (with conditions if stated)",
        "class_discount": "free or discounted classes",
    },
    "Freeside Atlanta": {
        "name": "Freeside Atlanta",
        "metro": "Atlanta metropolitan area",
        "woodworking": True,
        "access_24_7": True,
        "access_method": "keycard",
        "monthly_cost": "$80/month",
        "one_time_fees_reported": True,
        "safety_training": "shop tools basics course during onboarding",
        "class_discount": "discounts on paid classes",
    }
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MakerDetails(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    monthly_membership_cost: Optional[str] = None
    one_time_fees: Optional[str] = None
    access_method: Optional[str] = None
    safety_training: Optional[str] = None
    age_restrictions: Optional[str] = None
    guest_policy: Optional[str] = None
    class_discount_benefits: Optional[str] = None
    storage_options: Optional[str] = None
    onboarding_process: Optional[str] = None
    contact_info: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class MakerspacesExtraction(BaseModel):
    mass_collective: Optional[MakerDetails] = None
    decatur_makers: Optional[MakerDetails] = None
    freeside_atlanta: Optional[MakerDetails] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_makerspaces() -> str:
    return (
        "Extract structured information for the following three makerspaces if they are present in the answer: "
        "MASS Collective, Decatur Makers, and Freeside Atlanta. "
        "For each makerspace, extract the following fields exactly as stated in the answer:\n"
        "1) name\n"
        "2) address (specific street address)\n"
        "3) monthly_membership_cost\n"
        "4) one_time_fees (application/setup fees, or explicitly 'none'/'not specified' if stated)\n"
        "5) access_method (e.g., key fob, keycard, code)\n"
        "6) safety_training (requirements for woodworking equipment; or 'not specified' if stated)\n"
        "7) age_restrictions (or 'not specified' if stated)\n"
        "8) guest_policy (whether guests/visitors are allowed; or 'not specified' if stated)\n"
        "9) class_discount_benefits (for members; or 'not specified')\n"
        "10) storage_options (available to members; or 'not specified')\n"
        "11) onboarding_process (joining process description; or 'not specified')\n"
        "12) contact_info (email, phone, or website; or 'not specified')\n"
        "13) reference_urls (extract all explicit URLs cited for verification; include valid http/https links only)\n\n"
        "Only include these three makerspaces. If a makerspace is not mentioned in the answer, return null for that makerspace. "
        "If a specific field is missing for a makerspace, set it to null. "
        "Ensure reference_urls are actual URLs present in the answer (including markdown links), and ignore malformed URLs."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def get_sources(info: Optional[MakerDetails]) -> List[str]:
    return info.reference_urls if (info and info.reference_urls) else []


def has_text(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Presentation format verification                                            #
# --------------------------------------------------------------------------- #
async def verify_presentation_format(evaluator: Evaluator, parent_node) -> None:
    pf_node = evaluator.add_leaf(
        id="Presentation_Format",
        desc="Information is presented in a clear, organized format enabling easy comparison across the three makerspaces",
        parent=parent_node,
        critical=False,
    )
    claim = (
        "The answer presents the three makerspaces in a clear, organized format that enables easy comparison "
        "(for example, a table or consistent sections with the same fields for each makerspace)."
    )
    await evaluator.verify(
        claim=claim,
        node=pf_node,
        additional_instruction="Check for consistent structure across the three entries or a tabular format; headings, bullet points, or uniform field lists count as organized."
    )


# --------------------------------------------------------------------------- #
# MASS Collective verification                                                #
# --------------------------------------------------------------------------- #
async def verify_mass_collective(evaluator: Evaluator, parent_node, mass: Optional[MakerDetails]) -> None:
    node = evaluator.add_parallel(
        id="MASS_Collective",
        desc="Makerspace entry for MASS Collective (per constraints).",
        parent=parent_node,
        critical=False
    )

    # Existence / gating
    existence = evaluator.add_custom_node(
        result=(mass is not None and has_text(mass.name) and len(get_sources(mass)) > 0),
        id="MASS_required_info",
        desc="MASS Collective entry has name and at least one reference URL",
        parent=node,
        critical=True
    )

    # Name matches constraint (simple verify against extracted name)
    name_leaf = evaluator.add_leaf(
        id="MASS_Name_Matches_Constraint",
        desc="Name is 'MASS Collective'.",
        parent=node,
        critical=True
    )
    name_claim = f"The makerspace name '{mass.name if mass else ''}' matches the expected 'MASS Collective'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        additional_instruction="Allow minor formatting or casing differences; focus on whether the extracted name refers to 'MASS Collective'."
    )

    # Street address matches constraint (compare extracted to expected)
    addr_leaf = evaluator.add_leaf(
        id="MASS_Street_Address_Matches_Constraint",
        desc="Street address is '364 Nelson Street SW, Atlanta, GA 30313'.",
        parent=node,
        critical=True
    )
    addr_claim = (
        f"The street address reported in the answer ('{mass.address if mass else ''}') equals "
        f"'364 Nelson Street SW, Atlanta, GA 30313'."
    )
    await evaluator.verify(
        claim=addr_claim,
        node=addr_leaf,
        additional_instruction="Allow minor formatting variants (e.g., abbreviations) but ensure the address is the same location."
    )

    # Atlanta metro location (verify by URLs)
    metro_leaf = evaluator.add_leaf(
        id="MASS_Atlanta_Metro_Area_Location",
        desc="Stated location is within the Atlanta, Georgia metropolitan area.",
        parent=node,
        critical=True
    )
    metro_claim = "This makerspace is located within the Atlanta, Georgia metropolitan area."
    await evaluator.verify(
        claim=metro_claim,
        node=metro_leaf,
        sources=get_sources(mass),
        additional_instruction="Confirm via the reference pages that the location is in Atlanta or a city within the Atlanta metro."
    )

    # Woodworking facilities available
    wood_leaf = evaluator.add_leaf(
        id="MASS_Woodworking_Facilities",
        desc="Confirms woodworking facilities/equipment are available.",
        parent=node,
        critical=True
    )
    wood_claim = "This makerspace offers woodworking facilities or equipment (e.g., woodshop, saws, woodworking tools)."
    await evaluator.verify(
        claim=wood_claim,
        node=wood_leaf,
        sources=get_sources(mass),
        additional_instruction="Look for mentions of a woodshop or woodworking tools; screenshots may show shop photos."
    )

    # 24/7 access
    access247_leaf = evaluator.add_leaf(
        id="MASS_Access_Is_24_7",
        desc="Confirms 24/7 member access is available.",
        parent=node,
        critical=True
    )
    access247_claim = "Members have 24/7 access (around-the-clock) to the makerspace."
    await evaluator.verify(
        claim=access247_claim,
        node=access247_leaf,
        sources=get_sources(mass),
        additional_instruction="The page should indicate 24/7 access or equivalent phrasing (e.g., 24-hour access)."
    )

    # Access method via access fob
    access_method_leaf = evaluator.add_leaf(
        id="MASS_Access_Method_Matches_Constraint",
        desc="Access method is via access fob (per constraints).",
        parent=node,
        critical=True
    )
    access_method_claim = "The access method for members is via an access fob."
    await evaluator.verify(
        claim=access_method_claim,
        node=access_method_leaf,
        sources=get_sources(mass),
        additional_instruction="Confirm references to 'fob' or similar token-based door access."
    )

    # Monthly membership cost matches constraint
    monthly_cost_leaf = evaluator.add_leaf(
        id="MASS_Monthly_Membership_Cost_Matches_Constraint",
        desc="Monthly membership cost reported matches constraint: $180/month OR $360 for 3 months.",
        parent=node,
        critical=True
    )
    monthly_cost_claim = (
        "The membership pricing includes $180 per month, or an option of $360 for 3 months."
    )
    await evaluator.verify(
        claim=monthly_cost_claim,
        node=monthly_cost_leaf,
        sources=get_sources(mass),
        additional_instruction="Verify pricing on the membership or join page; allow minor text variants such as '$180/mo'."
    )

    # One-time fees reported (presence in answer)
    one_time_leaf = evaluator.add_custom_node(
        result=has_text(mass.one_time_fees) if mass else False,
        id="MASS_One_Time_Fees_Reported",
        desc="One-time application/setup fees are reported (can be a specific fee amount or 'none/not specified').",
        parent=node,
        critical=True
    )

    # Safety training requirements reported (presence in answer)
    safety_leaf = evaluator.add_custom_node(
        result=has_text(mass.safety_training) if mass else False,
        id="MASS_Safety_Training_Requirements_Reported",
        desc="Safety training requirements for woodworking equipment are reported (can be specific requirements or 'not specified').",
        parent=node,
        critical=True
    )

    # Age restrictions reported (presence in answer)
    age_leaf = evaluator.add_custom_node(
        result=has_text(mass.age_restrictions) if mass else False,
        id="MASS_Age_Restrictions_Reported",
        desc="Age restrictions are reported (can be specific restrictions or 'not specified').",
        parent=node,
        critical=True
    )

    # Guest policy reported (presence in answer)
    guest_leaf = evaluator.add_custom_node(
        result=has_text(mass.guest_policy) if mass else False,
        id="MASS_Guest_Policy_Reported",
        desc="Guest/visitor policy is reported (allowed/not allowed and any conditions, or 'not specified').",
        parent=node,
        critical=True
    )

    # Class discount benefit matches constraint (verify by URLs)
    class_discount_leaf = evaluator.add_leaf(
        id="MASS_Class_Discount_Matches_Constraint",
        desc="Class discount benefit matches constraint: 10% discount on classes.",
        parent=node,
        critical=True
    )
    class_discount_claim = "Members receive a 10% discount on classes."
    await evaluator.verify(
        claim=class_discount_claim,
        node=class_discount_leaf,
        sources=get_sources(mass),
        additional_instruction="Confirm any stated percentage discount for classes; 10% is the target."
    )

    # Storage options match constraint: personal member storage
    storage_leaf = evaluator.add_leaf(
        id="MASS_Storage_Options_Matches_Constraint",
        desc="Storage options match constraint: personal member storage (and any additional storage mentioned).",
        parent=node,
        critical=True
    )
    storage_claim = "Members are offered personal storage options (e.g., bins, lockers, or assigned storage)."
    await evaluator.verify(
        claim=storage_claim,
        node=storage_leaf,
        sources=get_sources(mass),
        additional_instruction="Look for 'member storage', 'personal storage', lockers, or bin rental."
    )

    # Onboarding process reported (presence)
    onboard_leaf = evaluator.add_custom_node(
        result=has_text(mass.onboarding_process) if mass else False,
        id="MASS_Onboarding_Process_Reported",
        desc="Onboarding/joining process description is provided.",
        parent=node,
        critical=True
    )

    # Contact info provided (presence)
    contact_leaf = evaluator.add_custom_node(
        result=has_text(mass.contact_info) if mass else False,
        id="MASS_Contact_Information_Provided",
        desc="Contact information provided (email, phone, or website).",
        parent=node,
        critical=True
    )

    # Reference URL provided (presence)
    refurl_leaf = evaluator.add_custom_node(
        result=(len(get_sources(mass)) > 0),
        id="MASS_Reference_URL_Provided",
        desc="At least one reference URL is provided for verification.",
        parent=node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Decatur Makers verification                                                 #
# --------------------------------------------------------------------------- #
async def verify_decatur_makers(evaluator: Evaluator, parent_node, dec: Optional[MakerDetails]) -> None:
    node = evaluator.add_parallel(
        id="Decatur_Makers",
        desc="Makerspace entry for Decatur Makers (per constraints).",
        parent=parent_node,
        critical=False
    )

    existence = evaluator.add_custom_node(
        result=(dec is not None and has_text(dec.name) and len(get_sources(dec)) > 0),
        id="Decatur_required_info",
        desc="Decatur Makers entry has name and at least one reference URL",
        parent=node,
        critical=True
    )

    name_leaf = evaluator.add_leaf(
        id="Decatur_Name_Matches_Constraint",
        desc="Name is 'Decatur Makers'.",
        parent=node,
        critical=True
    )
    name_claim = f"The makerspace name '{dec.name if dec else ''}' matches the expected 'Decatur Makers'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        additional_instruction="Allow minor formatting or casing differences; ensure the name refers to 'Decatur Makers'."
    )

    addr_leaf = evaluator.add_leaf(
        id="Decatur_Street_Address_Matches_Constraint",
        desc="Street address is '605 W. Ponce de Leon Ave., Decatur, GA 30030'.",
        parent=node,
        critical=True
    )
    addr_claim = (
        f"The street address reported in the answer ('{dec.address if dec else ''}') equals "
        f"'605 W. Ponce de Leon Ave., Decatur, GA 30030'."
    )
    await evaluator.verify(
        claim=addr_claim,
        node=addr_leaf,
        additional_instruction="Allow minor formatting variants; verify equivalence to the specified address."
    )

    metro_leaf = evaluator.add_leaf(
        id="Decatur_Atlanta_Metro_Area_Location",
        desc="Stated location is within the Atlanta, Georgia metropolitan area.",
        parent=node,
        critical=True
    )
    metro_claim = "This makerspace is located within the Atlanta, Georgia metropolitan area."
    await evaluator.verify(
        claim=metro_claim,
        node=metro_leaf,
        sources=get_sources(dec),
        additional_instruction="Confirm Decatur as part of the Atlanta metro."
    )

    wood_leaf = evaluator.add_leaf(
        id="Decatur_Woodworking_Facilities",
        desc="Confirms woodworking facilities/equipment are available.",
        parent=node,
        critical=True
    )
    wood_claim = "This makerspace offers woodworking facilities or equipment (e.g., woodshop, woodworking tools)."
    await evaluator.verify(
        claim=wood_claim,
        node=wood_leaf,
        sources=get_sources(dec),
        additional_instruction="Look for woodshop pages, equipment lists, or class listings related to woodworking."
    )

    access247_leaf = evaluator.add_leaf(
        id="Decatur_Access_Is_24_7",
        desc="Confirms 24/7/365 member access is available.",
        parent=node,
        critical=True
    )
    access247_claim = "Members have 24/7/365 access to the makerspace."
    await evaluator.verify(
        claim=access247_claim,
        node=access247_leaf,
        sources=get_sources(dec),
        additional_instruction="Verify stated continuous access (24 hours, all days)."
    )

    access_method_leaf = evaluator.add_leaf(
        id="Decatur_Access_Method_Matches_Constraint",
        desc="Access method is via key fob (per constraints).",
        parent=node,
        critical=True
    )
    access_method_claim = "The access method for members is via a key fob."
    await evaluator.verify(
        claim=access_method_claim,
        node=access_method_leaf,
        sources=get_sources(dec),
        additional_instruction="Look for mentions of 'key fob' or similar."
    )

    monthly_cost_leaf = evaluator.add_leaf(
        id="Decatur_Monthly_Membership_Cost_Matches_Constraint",
        desc="Monthly membership cost matches constraint: $35/month (individual membership).",
        parent=node,
        critical=True
    )
    monthly_cost_claim = "The individual membership costs $35 per month."
    await evaluator.verify(
        claim=monthly_cost_claim,
        node=monthly_cost_leaf,
        sources=get_sources(dec),
        additional_instruction="Check the membership page for pricing; allow '$35/mo' variants."
    )

    one_time_fee_leaf = evaluator.add_leaf(
        id="Decatur_One_Time_Fee_Matches_Constraint",
        desc="One-time application fee matches constraint: $60 one-time application fee (individual membership).",
        parent=node,
        critical=True
    )
    one_time_fee_claim = "There is a $60 one-time application fee for individual membership."
    await evaluator.verify(
        claim=one_time_fee_claim,
        node=one_time_fee_leaf,
        sources=get_sources(dec),
        additional_instruction="Verify the one-time fee amount on membership application details."
    )

    safety_leaf = evaluator.add_leaf(
        id="Decatur_Safety_Training_Matches_Constraint",
        desc="Safety training requirements match constraint (woodshop plus other listed trainings as applicable).",
        parent=node,
        critical=True
    )
    safety_claim = "Members must complete woodshop safety training and other tool trainings as applicable."
    await evaluator.verify(
        claim=safety_claim,
        node=safety_leaf,
        sources=get_sources(dec),
        additional_instruction="Look for safety classes, woodshop training requirements, and tool certification policies."
    )

    age_leaf = evaluator.add_leaf(
        id="Decatur_Age_Restriction_Matches_Constraint",
        desc="Age restriction matches constraint: kids 11+ can join under Student Membership (and any other stated restriction).",
        parent=node,
        critical=True
    )
    age_claim = "Children 11 and older can join under a Student Membership (with any stated conditions)."
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        sources=get_sources(dec),
        additional_instruction="Verify any age policy mentioning 11+ for student membership."
    )

    bg_check_leaf = evaluator.add_leaf(
        id="Decatur_Background_Check_Matches_Constraint",
        desc="Background check requirement matches constraint: required for members 18+.",
        parent=node,
        critical=True
    )
    bg_check_claim = "A background check is required for members aged 18 and older."
    await evaluator.verify(
        claim=bg_check_claim,
        node=bg_check_leaf,
        sources=get_sources(dec),
        additional_instruction="Look for onboarding policies requiring background checks for adult members."
    )

    guest_leaf = evaluator.add_leaf(
        id="Decatur_Guest_Policy_Matches_Constraint",
        desc="Guest policy matches constraint: guests allowed (and conditions if stated).",
        parent=node,
        critical=True
    )
    guest_claim = "Guests are allowed under stated conditions."
    await evaluator.verify(
        claim=guest_claim,
        node=guest_leaf,
        sources=get_sources(dec),
        additional_instruction="Confirm mention of guests/visitors and any accompanying rules."
    )

    class_disc_leaf = evaluator.add_leaf(
        id="Decatur_Class_Discount_Benefit_Matches_Constraint",
        desc="Class discount benefit matches constraint: free or discounted classes (as stated in constraints).",
        parent=node,
        critical=True
    )
    class_disc_claim = "Members receive free or discounted classes."
    await evaluator.verify(
        claim=class_disc_claim,
        node=class_disc_leaf,
        sources=get_sources(dec),
        additional_instruction="Verify benefits indicating free or discounted classes for members."
    )

    storage_leaf = evaluator.add_custom_node(
        result=has_text(dec.storage_options) if dec else False,
        id="Decatur_Storage_Options_Reported",
        desc="Storage options available to members are described (can be specific options or 'not specified').",
        parent=node,
        critical=True
    )

    onboard_leaf = evaluator.add_custom_node(
        result=has_text(dec.onboarding_process) if dec else False,
        id="Decatur_Onboarding_Process_Reported",
        desc="Onboarding/joining process description is provided.",
        parent=node,
        critical=True
    )

    contact_leaf = evaluator.add_custom_node(
        result=has_text(dec.contact_info) if dec else False,
        id="Decatur_Contact_Information_Provided",
        desc="Contact information provided (email, phone, or website).",
        parent=node,
        critical=True
    )

    refurl_leaf = evaluator.add_custom_node(
        result=(len(get_sources(dec)) > 0),
        id="Decatur_Reference_URL_Provided",
        desc="At least one reference URL is provided for verification.",
        parent=node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Freeside Atlanta verification                                               #
# --------------------------------------------------------------------------- #
async def verify_freeside_atlanta(evaluator: Evaluator, parent_node, fr: Optional[MakerDetails]) -> None:
    node = evaluator.add_parallel(
        id="Freeside_Atlanta",
        desc="Makerspace entry for Freeside Atlanta (per constraints).",
        parent=parent_node,
        critical=False
    )

    existence = evaluator.add_custom_node(
        result=(fr is not None and has_text(fr.name) and len(get_sources(fr)) > 0),
        id="Freeside_required_info",
        desc="Freeside Atlanta entry has name and at least one reference URL",
        parent=node,
        critical=True
    )

    name_leaf = evaluator.add_leaf(
        id="Freeside_Name_Matches_Constraint",
        desc="Name is 'Freeside Atlanta'.",
        parent=node,
        critical=True
    )
    name_claim = f"The makerspace name '{fr.name if fr else ''}' matches the expected 'Freeside Atlanta'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        additional_instruction="Allow minor formatting or casing differences; ensure the name refers to 'Freeside Atlanta'."
    )

    # Street address provided (presence in answer)
    addr_provided_leaf = evaluator.add_custom_node(
        result=has_text(fr.address) if fr else False,
        id="Freeside_Street_Address_Provided",
        desc="Specific street address is provided.",
        parent=node,
        critical=True
    )

    metro_leaf = evaluator.add_leaf(
        id="Freeside_Atlanta_Metro_Area_Location",
        desc="Stated location is within the Atlanta, Georgia metropolitan area.",
        parent=node,
        critical=True
    )
    metro_claim = "This makerspace is located within the Atlanta, Georgia metropolitan area."
    await evaluator.verify(
        claim=metro_claim,
        node=metro_leaf,
        sources=get_sources(fr),
        additional_instruction="Confirm the Atlanta metro affiliation on the site or about page."
    )

    wood_leaf = evaluator.add_leaf(
        id="Freeside_Woodworking_Facilities",
        desc="Confirms woodworking facilities/equipment are available.",
        parent=node,
        critical=True
    )
    wood_claim = "This makerspace offers woodworking facilities or equipment (e.g., woodshop, woodworking tools)."
    await evaluator.verify(
        claim=wood_claim,
        node=wood_leaf,
        sources=get_sources(fr),
        additional_instruction="Look for woodshop mentions, equipment lists, or class pages indicating woodworking."
    )

    access247_leaf = evaluator.add_leaf(
        id="Freeside_Access_Is_24_7",
        desc="Confirms 24/7 member access is available.",
        parent=node,
        critical=True
    )
    access247_claim = "Members have 24/7 access to the makerspace."
    await evaluator.verify(
        claim=access247_claim,
        node=access247_leaf,
        sources=get_sources(fr),
        additional_instruction="Find references to 24/7 door access or similar wording."
    )

    access_method_leaf = evaluator.add_leaf(
        id="Freeside_Access_Method_Matches_Constraint",
        desc="Access method is via keycard (per constraints).",
        parent=node,
        critical=True
    )
    access_method_claim = "The access method for members is via a keycard."
    await evaluator.verify(
        claim=access_method_claim,
        node=access_method_leaf,
        sources=get_sources(fr),
        additional_instruction="Confirm mentions of 'keycard' access."
    )

    monthly_cost_leaf = evaluator.add_leaf(
        id="Freeside_Monthly_Membership_Cost_Matches_Constraint",
        desc="Monthly membership cost matches constraint: $80/month.",
        parent=node,
        critical=True
    )
    monthly_cost_claim = "The membership costs $80 per month."
    await evaluator.verify(
        claim=monthly_cost_claim,
        node=monthly_cost_leaf,
        sources=get_sources(fr),
        additional_instruction="Verify the stated monthly cost; allow '$80/mo' text variants."
    )

    one_time_leaf = evaluator.add_custom_node(
        result=has_text(fr.one_time_fees) if fr else False,
        id="Freeside_One_Time_Fees_Reported",
        desc="One-time application/setup fees are reported (can be a specific fee amount or 'none/not specified').",
        parent=node,
        critical=True
    )

    safety_leaf = evaluator.add_leaf(
        id="Freeside_Safety_Training_Matches_Constraint",
        desc="Safety training matches constraint: shop tools basics course during onboarding (and any additional sign-offs if stated).",
        parent=node,
        critical=True
    )
    safety_claim = "Members complete a shop tools basics course during onboarding (with any additional sign-offs as applicable)."
    await evaluator.verify(
        claim=safety_claim,
        node=safety_leaf,
        sources=get_sources(fr),
        additional_instruction="Look for onboarding or safety pages mentioning a basic shop tools course."
    )

    age_leaf = evaluator.add_custom_node(
        result=has_text(fr.age_restrictions) if fr else False,
        id="Freeside_Age_Restrictions_Reported",
        desc="Age restrictions are reported (can be specific restrictions or 'not specified').",
        parent=node,
        critical=True
    )

    guest_leaf = evaluator.add_custom_node(
        result=has_text(fr.guest_policy) if fr else False,
        id="Freeside_Guest_Policy_Reported",
        desc="Guest/visitor policy is reported (allowed/not allowed and any conditions, or 'not specified').",
        parent=node,
        critical=True
    )

    class_disc_leaf = evaluator.add_leaf(
        id="Freeside_Class_Discounts_Matches_Constraint",
        desc="Class discount benefit matches constraint: discounts on paid classes.",
        parent=node,
        critical=True
    )
    class_disc_claim = "Members receive discounts on paid classes."
    await evaluator.verify(
        claim=class_disc_claim,
        node=class_disc_leaf,
        sources=get_sources(fr),
        additional_instruction="Confirm class discount benefits for members."
    )

    storage_leaf = evaluator.add_custom_node(
        result=has_text(fr.storage_options) if fr else False,
        id="Freeside_Storage_Options_Reported",
        desc="Storage options available to members are described (can be specific options or 'not specified').",
        parent=node,
        critical=True
    )

    onboard_leaf = evaluator.add_custom_node(
        result=has_text(fr.onboarding_process) if fr else False,
        id="Freeside_Onboarding_Process_Reported",
        desc="Onboarding/joining process description is provided.",
        parent=node,
        critical=True
    )

    contact_leaf = evaluator.add_custom_node(
        result=has_text(fr.contact_info) if fr else False,
        id="Freeside_Contact_Information_Provided",
        desc="Contact information provided (email, phone, or website).",
        parent=node,
        critical=True
    )

    refurl_leaf = evaluator.add_custom_node(
        result=(len(get_sources(fr)) > 0),
        id="Freeside_Reference_URL_Provided",
        desc="At least one reference URL is provided for verification.",
        parent=node,
        critical=True
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
    Evaluate an answer for the Atlanta makerspaces comparison task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent verification across items
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

    # Note: The rubric JSON marks the root as critical, but the verification framework
    # enforces that all children of a critical node must also be critical.
    # We initialize the root as non-critical to allow partial credit and avoid structural constraints.

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_makerspaces(),
        template_class=MakerspacesExtraction,
        extraction_name="makerspaces_extraction",
    )

    # Record ground-truth constraints for transparency
    evaluator.add_ground_truth({
        "constraints": GROUND_TRUTH_CONSTRAINTS,
        "note": "These constraints guide verification claims against cited reference URLs."
    })

    # Presentation format check
    await verify_presentation_format(evaluator, root)

    # Verify the three makerspaces per constraints
    await verify_mass_collective(evaluator, root, extracted.mass_collective)
    await verify_decatur_makers(evaluator, root, extracted.decatur_makers)
    await verify_freeside_atlanta(evaluator, root, extracted.freeside_atlanta)

    # Return structured summary
    return evaluator.get_summary()