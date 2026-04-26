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
TASK_ID = "spring_2026_family_vacation_compliance"
TASK_DESCRIPTION = """A Maryland family of four is planning a spring 2026 multi-destination vacation and needs to ensure full compliance with all current travel regulations and booking policies.

Family Composition:
- Parent A (age 42) - holds Maryland driver's license without REAL ID star
- Parent B (age 40) - holds valid U.S. passport
- Teenager (age 16)
- Child (age 9)

Planned Itinerary:
- March 20, 2026: Domestic flight from Baltimore/Washington International (BWI) Airport to Fort Lauderdale, Florida
- March 21, 2026, 4:00 PM: Embarkation on Celebrity Cruises Caribbean cruise from Fort Lauderdale (7-day cruise)
- March 28, 2026: Cruise conclusion and disembarkation
- April 1-4, 2026: Four-day visit to Dollywood theme park in Pigeon Forge, Tennessee, with resort hotel accommodation
- April 5, 2026: Return flight to Baltimore via United Airlines (economy class) with checked luggage

Create a comprehensive travel compliance checklist that documents:
1. Specific identification requirements for each family member to clear TSA security at BWI on March 20, including any applicable fees or advance verification procedures effective February 2026
2. Passport validity requirements and any additional documentation needed for each family member for the Celebrity Cruise
3. The embarkation deadline (latest arrival time) at the Fort Lauderdale cruise terminal for their 4:00 PM departure
4. The validity period for their 4-day Dollywood tickets when the first use date is April 1, 2026
5. Advance booking requirements to qualify for discounts on Dollywood resort hotel reservations
6. Checked baggage dimension and weight restrictions for United Airlines economy passengers on their April 5 return flight

All requirements must be supported with specific reference URLs from official airline, cruise line, TSA, or theme park sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TSASection(BaseModel):
    adult_id_policy_text: Optional[str] = None
    parent_a_alt_id_program_name: Optional[str] = None
    parent_a_alt_id_fee: Optional[str] = None
    parent_a_alt_id_validity_period: Optional[str] = None
    parent_a_alt_id_effective_date: Optional[str] = None
    parent_b_passport_statement: Optional[str] = None
    minors_id_policy_statement: Optional[str] = None
    tsa_urls: List[str] = Field(default_factory=list)


class CelebritySection(BaseModel):
    passport_validity_statement: Optional[str] = None
    cabin_age_policy_statement: Optional[str] = None
    minor_consent_statement: Optional[str] = None
    embarkation_deadline_statement: Optional[str] = None
    celebrity_urls: List[str] = Field(default_factory=list)


class DollywoodSection(BaseModel):
    four_day_ticket_validity_statement: Optional[str] = None
    resort_discount_booking_requirement_statement: Optional[str] = None
    dollywood_urls: List[str] = Field(default_factory=list)


class UnitedSection(BaseModel):
    checked_bag_dimensions_statement: Optional[str] = None
    checked_bag_weight_statement: Optional[str] = None
    united_urls: List[str] = Field(default_factory=list)


class FamilyTravelChecklistExtraction(BaseModel):
    tsa: Optional[TSASection] = None
    celebrity: Optional[CelebritySection] = None
    dollywood: Optional[DollywoodSection] = None
    united: Optional[UnitedSection] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_checklist_sections() -> str:
    return """
    Extract the family's travel compliance checklist content and the official reference URLs cited in the answer, organized into four sections (TSA, Celebrity Cruises, Dollywood, United Airlines). Return JSON strictly following this schema:

    {
      "tsa": {
        "adult_id_policy_text": string or null,
        "parent_a_alt_id_program_name": string or null,
        "parent_a_alt_id_fee": string or null,
        "parent_a_alt_id_validity_period": string or null,
        "parent_a_alt_id_effective_date": string or null,
        "parent_b_passport_statement": string or null,
        "minors_id_policy_statement": string or null,
        "tsa_urls": [string, ...]
      },
      "celebrity": {
        "passport_validity_statement": string or null,
        "cabin_age_policy_statement": string or null,
        "minor_consent_statement": string or null,
        "embarkation_deadline_statement": string or null,
        "celebrity_urls": [string, ...]
      },
      "dollywood": {
        "four_day_ticket_validity_statement": string or null,
        "resort_discount_booking_requirement_statement": string or null,
        "dollywood_urls": [string, ...]
      },
      "united": {
        "checked_bag_dimensions_statement": string or null,
        "checked_bag_weight_statement": string or null,
        "united_urls": [string, ...]
      }
    }

    Instructions:
    - For each "..._statement" field, extract the exact statement from the answer that corresponds to the requested rule/policy. If the answer does not include such a statement, set it to null.
    - For TSA Parent A (non-REAL ID), if the answer mentions any alternate identity program (e.g., ConfirmID) and specifies a fee, validity period, or effective date, extract those exact values into the corresponding fields; otherwise return null for any missing fields.
    - For URL arrays (tsa_urls, celebrity_urls, dollywood_urls, united_urls):
      * Extract ONLY URLs that are explicitly present in the answer text.
      * Include official sources only (TSA: *.tsa.gov; Celebrity Cruises: *.celebritycruises.com; Dollywood: *.dollywood.com; United Airlines: *.united.com). Ignore non-official sources.
      * If no official URLs are present for a section, return an empty list for that section.
    - Do not infer or fabricate any URLs or statements not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(x: Optional[List[str]]) -> List[str]:
    return x if isinstance(x, list) else []


def _has_domain(urls: List[str], keyword: str) -> bool:
    if not urls:
        return False
    kw = keyword.lower()
    for u in urls:
        if isinstance(u, str) and kw in u.lower():
            return True
    return False


def _merge_sources(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            if u and isinstance(u, str) and u not in merged:
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Claim builders                                                               #
# --------------------------------------------------------------------------- #
def claim_adult_real_id_requirement() -> str:
    return ("TSA policy for domestic air travel in 2026 requires that travelers aged 18 and older present a "
            "REAL ID–compliant driver's license or another TSA-acceptable identification document to clear security.")


def claim_parent_a_path(tsa: TSASection) -> str:
    base = ("If a traveler aged 18 or older does not have a REAL ID–compliant driver's license, they must present "
            "another TSA-acceptable ID such as a valid U.S. passport to clear TSA screening for a domestic flight.")
    parts = []
    if tsa.parent_a_alt_id_program_name:
        parts.append(f"TSA also offers an alternate identity verification program called '{tsa.parent_a_alt_id_program_name}'.")
    if tsa.parent_a_alt_id_fee:
        parts.append(f"The program fee is {tsa.parent_a_alt_id_fee}.")
    if tsa.parent_a_alt_id_validity_period:
        parts.append(f"The validity period is {tsa.parent_a_alt_id_validity_period}.")
    if tsa.parent_a_alt_id_effective_date:
        parts.append(f"The stated effective date is {tsa.parent_a_alt_id_effective_date}.")
    if parts:
        return base + " " + " ".join(parts)
    return base


def claim_parent_b_passport_ok() -> str:
    return ("A valid U.S. passport is an acceptable form of identification for TSA screening for domestic flights.")


def claim_minors_id_policy() -> str:
    # Allows airline policy to support birth-certificate sufficiency while TSA policy confirms no ID required
    return ("For domestic flights within the United States, travelers under 18 who are traveling with an adult companion "
            "are not required by TSA to present identification. Airlines may accept a birth certificate as sufficient "
            "documentation if any age or identity verification is requested.")


def claim_celebrity_passport_validity() -> str:
    return ("Celebrity Cruises requires that guest passports be valid for at least six months after the cruise ends.")


def claim_celebrity_cabin_age21() -> str:
    return ("Celebrity Cruises policy requires at least one guest age 21 or older in each stateroom.")


def claim_celebrity_minor_consent() -> str:
    return ("When minors (17 and under) sail with their parent(s) or legal guardian(s), a notarized consent form is not required by Celebrity Cruises; "
            "consent documentation is required only when a minor travels without their legal guardian.")


def claim_celebrity_embark_deadline() -> str:
    # 90 minutes before 4:00 PM => 2:30 PM latest
    return ("Celebrity Cruises requires that check-in/boarding be completed no later than 90 minutes before the scheduled departure time. "
            "For a 4:00 PM departure, the latest arrival/check-in time is 2:30 PM.")


def claim_dollywood_ticket_validity(dolly: DollywoodSection) -> str:
    if dolly.four_day_ticket_validity_statement:
        # Use the user's stated policy text, verifying it against Dollywood's page
        return dolly.four_day_ticket_validity_statement
    # Fallback generic phrasing to be checked against official page description
    return ("A 4-day Dollywood ticket is valid for four days within the stated validity window from the first use date, "
            "as described by Dollywood’s official ticket policy.")


def claim_dollywood_resort_discount(dolly: DollywoodSection) -> str:
    if dolly.resort_discount_booking_requirement_statement:
        return dolly.resort_discount_booking_requirement_statement
    return ("Dollywood’s resort discount offers require reservations to be made in advance according to the lead time "
            "and terms specified in Dollywood’s official resort booking policy.")


def claim_united_bag_dimensions() -> str:
    return ("United Airlines’ standard checked baggage size limit for economy passengers is a maximum of 62 linear inches "
            "(length + width + height).")


def claim_united_bag_weight() -> str:
    return ("United Airlines’ standard checked baggage weight limit for economy passengers is a maximum of 50 pounds (23 kg).")


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
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
    # 1) Initialize evaluator
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

    # 2) Extract structured checklist + URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_checklist_sections(),
        template_class=FamilyTravelChecklistExtraction,
        extraction_name="checklist_extraction",
    )

    tsa = extracted.tsa or TSASection()
    celebrity = extracted.celebrity or CelebritySection()
    dolly = extracted.dollywood or DollywoodSection()
    united = extracted.united or UnitedSection()

    tsa_urls = _safe_list(tsa.tsa_urls)
    celebrity_urls = _safe_list(celebrity.celebrity_urls)
    dolly_urls = _safe_list(dolly.dollywood_urls)
    united_urls = _safe_list(united.united_urls)

    # 3) Build verification nodes

    # Top-level documentation node (critical)
    vacation_node = evaluator.add_parallel(
        id="vacation_compliance_documentation",
        desc="Complete and accurate travel compliance checklist for the family's spring 2026 itinerary, covering all 6 requested elements.",
        parent=root,
        critical=True
    )

    # ---- TSA identification requirements (critical) ----
    tsa_node = evaluator.add_parallel(
        id="tsa_bwi_identification_mar20_2026",
        desc="TSA checkpoint identification requirements at BWI on March 20, 2026, for each family member, including any applicable fees/advance verification procedures effective Feb 2026.",
        parent=vacation_node,
        critical=True
    )

    # Adult REAL ID / Acceptable ID requirement
    tsa_adult_leaf = evaluator.add_leaf(
        id="adult_real_id_or_acceptable_id_requirement",
        desc="States that travelers age 18+ must present a REAL ID-compliant license or other TSA-acceptable ID to board domestic flights (per the stated effective date/policy).",
        parent=tsa_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_adult_real_id_requirement(),
        node=tsa_adult_leaf,
        sources=tsa_urls if tsa_urls else None,
        additional_instruction="Verify this against an official TSA page listing acceptable IDs and REAL ID enforcement for domestic flights in 2026."
    )

    # Parent A path (non-REAL-ID) including alternate verification program details if stated
    tsa_parent_a_leaf = evaluator.add_leaf(
        id="parent_a_id_path_without_real_id_including_confirmid",
        desc="States a compliant path for Parent A (age 42) with a non-REAL-ID license: present another acceptable ID (e.g., passport) OR use TSA ConfirmID/alternate identity verification if lacking acceptable ID, including the fee and validity period from the constraints ($45 for a 10-day travel period, starting Feb 1, 2026).",
        parent=tsa_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_parent_a_path(tsa),
        node=tsa_parent_a_leaf,
        sources=tsa_urls if tsa_urls else None,
        additional_instruction="Check whether the official TSA source(s) support the alternate acceptable ID path and, if the answer claims a named program, fee amount, validity period, or effective date, look for those exact specifics."
    )

    # Parent B passport acceptable for TSA
    tsa_parent_b_leaf = evaluator.add_leaf(
        id="parent_b_passport_acceptable_for_tsa",
        desc="States that Parent B’s valid U.S. passport is acceptable identification for TSA screening for domestic flights.",
        parent=tsa_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_parent_b_passport_ok(),
        node=tsa_parent_b_leaf,
        sources=tsa_urls if tsa_urls else None,
        additional_instruction="Confirm that a valid U.S. passport is listed as an acceptable ID on official TSA pages."
    )

    # Minors under 18 (teen and child)
    tsa_minors_leaf = evaluator.add_leaf(
        id="minors_under_18_tsa_id_policy_applies_to_both_children",
        desc="States that both minors (age 16 and age 9) do not need photo ID for domestic flights, and that a birth certificate is sufficient per the stated policy.",
        parent=tsa_node,
        critical=True
    )
    minors_sources = _merge_sources(tsa_urls, united_urls) if (tsa_urls or united_urls) else None
    await evaluator.verify(
        claim=claim_minors_id_policy(),
        node=tsa_minors_leaf,
        sources=minors_sources,
        additional_instruction="This may be supported by TSA policy (no ID required for minors traveling with an adult) and/or airline policy (birth certificates accepted). Accept support from TSA or the operating airline’s official page."
    )

    # ---- Celebrity Cruises documentation and timing (critical) ----
    celeb_node = evaluator.add_parallel(
        id="celebrity_cruise_documentation_and_timing",
        desc="Celebrity Cruises passport/documentation requirements and the embarkation deadline for the March 21, 2026 4:00 PM sailing from Fort Lauderdale.",
        parent=vacation_node,
        critical=True
    )

    celeb_passport_leaf = evaluator.add_leaf(
        id="passport_validity_rule_6_months",
        desc="States Celebrity Cruises passport validity requirement: passports valid at least 6 months after the cruise ends.",
        parent=celeb_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_celebrity_passport_validity(),
        node=celeb_passport_leaf,
        sources=celebrity_urls if celebrity_urls else None,
        additional_instruction="Verify on Celebrity Cruises’ official documentation/FAQ pages that passports must have at least six months validity beyond the end of the cruise."
    )

    celeb_cabin_age_leaf = evaluator.add_leaf(
        id="cabin_age_21_requirement",
        desc="States and applies the policy that at least one person age 21 or older is required in each cabin (verifying the family satisfies it).",
        parent=celeb_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_celebrity_cabin_age21(),
        node=celeb_cabin_age_leaf,
        sources=celebrity_urls if celebrity_urls else None,
        additional_instruction="Check Celebrity’s policy regarding minimum age requirements per stateroom."
    )

    celeb_minor_consent_leaf = evaluator.add_leaf(
        id="minor_consent_form_applicability",
        desc="Addresses whether minors (17 and under) need a notarized consent form and correctly applies it based on whether they are traveling without a legal guardian.",
        parent=celeb_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_celebrity_minor_consent(),
        node=celeb_minor_consent_leaf,
        sources=celebrity_urls if celebrity_urls else None,
        additional_instruction="Verify Celebrity’s documentation about consent requirements for minors traveling with or without their legal guardians."
    )

    celeb_embark_leaf = evaluator.add_leaf(
        id="embarkation_deadline_latest_arrival_time",
        desc="Provides the embarkation deadline (latest arrival/check-in time) for a 4:00 PM departure consistent with the stated 90-minute cutoff policy (i.e., 90 minutes prior).",
        parent=celeb_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_celebrity_embark_deadline(),
        node=celeb_embark_leaf,
        sources=celebrity_urls if celebrity_urls else None,
        additional_instruction="Verify that Celebrity specifies a 90-minute pre-departure cutoff for check-in/boarding and compute the latest arrival for a 4:00 PM departure."
    )

    # ---- Dollywood (critical) ----
    dolly_node = evaluator.add_parallel(
        id="dollywood_tickets_and_resort_booking",
        desc="Dollywood ticket validity and Dollywood resort hotel discount booking requirements for the April 1–4, 2026 visit.",
        parent=vacation_node,
        critical=True
    )

    dolly_ticket_leaf = evaluator.add_leaf(
        id="dollywood_4day_ticket_validity_from_apr1",
        desc="States the validity period/window for 4-day Dollywood tickets when first used April 1, 2026 (either the number-of-days window or the window end date, per policy).",
        parent=dolly_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_dollywood_ticket_validity(dolly),
        node=dolly_ticket_leaf,
        sources=dolly_urls if dolly_urls else None,
        additional_instruction="Verify the validity window for a 4-day ticket from Dollywood’s official ticket policy page (e.g., number of consecutive days from first use)."
    )

    dolly_resort_leaf = evaluator.add_leaf(
        id="dollywood_resort_discount_advance_booking",
        desc="States the advance booking requirement to qualify for Dollywood resort hotel discounts (including required lead time and discount terms, per policy).",
        parent=dolly_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_dollywood_resort_discount(dolly),
        node=dolly_resort_leaf,
        sources=dolly_urls if dolly_urls else None,
        additional_instruction="Verify any advance booking lead-time and other conditions for Dollywood resort discounts on Dollywood’s official site."
    )

    # ---- United Airlines checked baggage (critical) ----
    ua_node = evaluator.add_parallel(
        id="united_checked_baggage_apr5_return",
        desc="United Airlines economy checked-baggage restrictions for the April 5, 2026 return flight.",
        parent=vacation_node,
        critical=True
    )

    ua_dim_leaf = evaluator.add_leaf(
        id="united_checked_bag_dimensions",
        desc="States United’s checked bag size limit for economy passengers: maximum 62 linear inches (L+W+H).",
        parent=ua_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_united_bag_dimensions(),
        node=ua_dim_leaf,
        sources=united_urls if united_urls else None,
        additional_instruction="Verify this on United’s official baggage policy page for standard economy checked bags."
    )

    ua_weight_leaf = evaluator.add_leaf(
        id="united_checked_bag_weight",
        desc="States United’s checked bag weight limit for economy passengers: maximum 50 pounds.",
        parent=ua_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_united_bag_weight(),
        node=ua_weight_leaf,
        sources=united_urls if united_urls else None,
        additional_instruction="Verify this on United’s official baggage policy page for standard economy checked bags."
    )

    # ---- Official reference URLs (critical) ----
    refs_node = evaluator.add_parallel(
        id="official_reference_urls",
        desc="All stated requirements are supported with specific reference URL(s) from official sources in the allowed categories (TSA, cruise line, theme park, airline).",
        parent=vacation_node,
        critical=True
    )

    tsa_refs_exist = _has_domain(tsa_urls, "tsa.gov")
    celeb_refs_exist = _has_domain(celebrity_urls, "celebritycruises.com")
    dolly_refs_exist = _has_domain(dolly_urls, "dollywood.com")
    ua_refs_exist = _has_domain(united_urls, "united.com")

    evaluator.add_custom_node(
        result=tsa_refs_exist,
        id="tsa_official_urls_present",
        desc="Provides at least one official TSA URL supporting the TSA identification/ConfirmID-related claims used in the checklist.",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=celeb_refs_exist,
        id="celebrity_official_urls_present",
        desc="Provides at least one official Celebrity Cruises URL supporting the cruise documentation and embarkation-deadline claims used in the checklist.",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=dolly_refs_exist,
        id="dollywood_official_urls_present",
        desc="Provides at least one official Dollywood/theme-park (or Dollywood resort) URL supporting the ticket-validity and resort-discount booking claims used in the checklist.",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=ua_refs_exist,
        id="united_official_urls_present",
        desc="Provides at least one official United Airlines URL supporting the checked-baggage restriction claims used in the checklist.",
        parent=refs_node,
        critical=True
    )

    # 4) Return summary
    return evaluator.get_summary()