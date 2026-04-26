import asyncio
import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tsa_maldives_prep_2026"
TASK_DESCRIPTION = """
I am a U.S. citizen planning to travel to the Maldives for a 25-day vacation. My departure date is March 15, 2026, and I will return on April 9, 2026. I want to enroll in TSA PreCheck before my trip to use expedited security screening at U.S. airports. What documents and preparations do I need to complete for both the TSA PreCheck enrollment and for entering the Maldives? Please provide: (1) All required documents for TSA PreCheck enrollment, (2) The enrollment fee I need to pay, (3) How much time I should allow for processing before my March 15 departure, (4) All entry requirements for the Maldives, including specific passport validity requirements based on my travel dates, and (5) What proof of travel and accommodation I need for Maldives entry. Please specify the types of documents accepted and any specific validity or authenticity requirements.
"""

DEPARTURE_DATE = date(2026, 3, 15)
ARRIVAL_DATE = date(2026, 3, 15)  # Arrival to Maldives for this itinerary
# Minimum valid-through date for passport: at least 1 month beyond arrival
MIN_VALID_THROUGH_DATE = ARRIVAL_DATE + timedelta(days=31)  # Approximate "1 month"
MIN_VALID_THROUGH_STR = MIN_VALID_THROUGH_DATE.strftime("%B %d, %Y")


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TSAInfo(BaseModel):
    # Identity document
    identity_required: Optional[bool] = None
    identity_docs: List[str] = Field(default_factory=list)
    identity_validity_notes: List[str] = Field(default_factory=list)

    # Citizenship document
    citizenship_required: Optional[bool] = None
    citizenship_docs: List[str] = Field(default_factory=list)
    birth_certificate_requirements: List[str] = Field(default_factory=list)

    # Fingerprinting
    fingerprinting_required: Optional[bool] = None

    # Fee and coverage
    fee_amount: Optional[str] = None
    fee_coverage_years: Optional[str] = None

    # Processing time and recommendation
    processing_typical: Optional[str] = None
    processing_max: Optional[str] = None
    processing_recommended_timeframe: Optional[str] = None
    processing_recommended_date: Optional[str] = None

    # Sources the answer cites for TSA (if any)
    source_urls: List[str] = Field(default_factory=list)


class MaldivesInfo(BaseModel):
    # Visa and passport validity
    visa_on_arrival_free_30_days: Optional[bool] = None
    passport_validity_rule_text: Optional[str] = None
    applied_min_valid_date_text: Optional[str] = None

    # Proofs of travel and accommodation
    onward_return_proof_text: Optional[str] = None
    accommodation_proof_text: Optional[str] = None

    # Sufficient funds
    sufficient_funds_text: Optional[str] = None

    # Sources the answer cites for Maldives (if any)
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_tsa_info() -> str:
    return """
    Extract from the answer all TSA PreCheck enrollment details as structured fields.

    Return a JSON object with these fields:
    - identity_required: boolean; whether the answer explicitly says an acceptable identity document is required
    - identity_docs: array of strings; list each identity document option the answer mentions (normalize names when possible):
        examples: "U.S. passport book", "U.S. passport card", "Enhanced Tribal Card (ETC)", "Free and Secure Trade (FAST) card", "state-issued photo ID", "driver's license"
    - identity_validity_notes: array of strings; validity phrases the answer mentions for identity docs, e.g., "unexpired", "valid"
    - citizenship_required: boolean; whether the answer explicitly says proof of U.S. citizenship is required
    - citizenship_docs: array of strings; list each citizenship document option the answer mentions:
        expected canonical options include: "U.S. passport", "U.S. birth certificate", "Certificate of Naturalization (N-550/N-570)", "Certificate of Citizenship (N-560/N-561)"
    - birth_certificate_requirements: array of strings; if birth-certificate is mentioned, list authenticity/format requirements stated in the answer,
        examples: "long-form state or territory issued", "certified/sealed", "original or certified copy", "must say 'Birth Certificate'"
    - fingerprinting_required: boolean; whether the answer explicitly requires fingerprinting in-person at an enrollment center
    - fee_amount: string; the fee stated (e.g., "$85")
    - fee_coverage_years: string; the coverage term stated (e.g., "5 years")
    - processing_typical: string; typical processing time text (e.g., "3–5 days")
    - processing_max: string; maximum processing time text (e.g., "up to 60 days")
    - processing_recommended_timeframe: string; recommended latest-start timeframe before the March 15, 2026 departure (e.g., "at least 60 days before", "two months before", "by mid-January 2026")
    - processing_recommended_date: string; if a concrete recommended latest-start date is given (e.g., "January 15, 2026"), return it; otherwise null
    - source_urls: array of strings; any URLs the answer cites specifically for TSA PreCheck information

    If any item is not in the answer, set it to null or an empty list, as appropriate.
    """


def prompt_extract_maldives_info() -> str:
    return """
    Extract from the answer all Maldives entry requirements and proofs as structured fields.

    Return a JSON object with these fields:
    - visa_on_arrival_free_30_days: boolean; whether the answer says Maldives grants a free 30-day tourist visa on arrival and no pre-approval is required
    - passport_validity_rule_text: string; the passport validity rule text stated (e.g., "at least 1 month beyond arrival")
    - applied_min_valid_date_text: string; the answer's application of the rule to the stated arrival (March 15, 2026), e.g., "through at least April 15, 2026" or "at least one month/30+ days beyond arrival"
    - onward_return_proof_text: string; what the answer lists as acceptable proof of onward/return travel (e.g., "return flight booking confirmation", "itinerary")
    - accommodation_proof_text: string; what the answer lists as acceptable proof of confirmed accommodation (e.g., "hotel/resort/guesthouse booking confirmation")
    - sufficient_funds_text: string; the funds requirement text including the recommended amount (e.g., "at least US$100 + US$50 per day")
    - source_urls: array of strings; any URLs the answer cites for Maldives entry requirements

    If any item is not in the answer, set it to null. If multiple acceptable proofs are listed, include them in the text.
    """


# --------------------------------------------------------------------------- #
# Verification helper functions                                               #
# --------------------------------------------------------------------------- #
async def verify_tsa_precheck(evaluator: Evaluator, parent_node, tsa: TSAInfo) -> None:
    # TSA PreCheck parent node (critical parallel)
    tsa_node = evaluator.add_parallel(
        id="tsa_precheck_enrollment",
        desc="Provide TSA PreCheck enrollment documents/steps, fee, and timing guidance before the March 15, 2026 departure",
        parent=parent_node,
        critical=True,
    )

    # TSA required documents group (critical parallel)
    tsa_docs_node = evaluator.add_parallel(
        id="tsa_required_documents",
        desc="Provide all required TSA PreCheck enrollment documents, including accepted types and validity/authenticity requirements per constraints",
        parent=tsa_node,
        critical=True,
    )

    # Identity document leaf
    identity_leaf = evaluator.add_leaf(
        id="tsa_identity_document",
        desc=("States that an acceptable identity document is required and lists the constraint-specified acceptable "
              "options (unexpired U.S. passport book/card, unexpired Enhanced Tribal Card (ETC), unexpired Free and "
              "Secure Trade (FAST) card, or valid state-issued photo ID), including any stated validity condition "
              "(e.g., unexpired where specified)"),
        parent=tsa_docs_node,
        critical=True,
    )
    identity_claim = (
        "The answer explicitly states that an acceptable identity document is required for TSA PreCheck enrollment "
        "and it lists acceptable options and validity conditions (e.g., unexpired) that include the set: "
        "U.S. passport (book or card), Enhanced Tribal Card (ETC), Free and Secure Trade (FAST) card, and a valid "
        "state‑issued photo ID."
    )
    await evaluator.verify(
        claim=identity_claim,
        node=identity_leaf,
        sources=None,
        additional_instruction=(
            "Judge only based on the answer text. The answer must enumerate acceptable identity options and validity. "
            "Minimum acceptable set: 'U.S. passport (book or card)' with 'unexpired', 'Enhanced Tribal Card (ETC)' "
            "with 'unexpired', 'Free and Secure Trade (FAST) card' with 'unexpired', and 'state‑issued photo ID'. "
            f"The answer listed: {tsa.identity_docs}. Validity notes mentioned: {tsa.identity_validity_notes}."
        ),
    )

    # Citizenship document leaf
    citizenship_leaf = evaluator.add_leaf(
        id="tsa_citizenship_document",
        desc=("States that proof of U.S. citizenship is required and lists the constraint-specified acceptable options "
              "(U.S. passport, U.S. birth certificate, Certificate of Naturalization N-550/N-570, or Certificate of "
              "Citizenship N-560/N-561); if the birth-certificate option is mentioned, includes the constraint-specified "
              "authenticity/format requirements (long-form state/territory-issued; certified/sealed; original or certified copy; "
              "must say 'Birth Certificate')"),
        parent=tsa_docs_node,
        critical=True,
    )
    citizenship_claim = (
        "The answer explicitly states that proof of U.S. citizenship is required and it lists acceptable options "
        "including U.S. passport, U.S. birth certificate, Certificate of Naturalization (N‑550/N‑570), and "
        "Certificate of Citizenship (N‑560/N‑561). If 'birth certificate' is included, the answer also states "
        "authenticity/format requirements such as: long‑form state/territory‑issued, certified/sealed, original or "
        "certified copy, and must say 'Birth Certificate'."
    )
    await evaluator.verify(
        claim=citizenship_claim,
        node=citizenship_leaf,
        sources=None,
        additional_instruction=(
            "Judge only based on the answer text. Confirm presence of the required set of acceptable citizenship proofs "
            "and, when 'birth certificate' appears, confirm authenticity details. "
            f"Answer listed citizenship docs: {tsa.citizenship_docs}. Birth certificate requirements mentioned: {tsa.birth_certificate_requirements}."
        ),
    )

    # Fingerprinting leaf
    fingerprint_leaf = evaluator.add_leaf(
        id="tsa_fingerprinting",
        desc="States fingerprinting (in-person at an enrollment center) is required as part of TSA PreCheck enrollment",
        parent=tsa_node,
        critical=True,
    )
    fingerprint_claim = (
        "The answer explicitly states that fingerprinting, in person at an enrollment center, is required as part of TSA PreCheck enrollment."
    )
    await evaluator.verify(
        claim=fingerprint_claim,
        node=fingerprint_leaf,
        sources=None,
        additional_instruction="Judge only based on the answer text; confirm clear mention of in‑person fingerprinting requirement.",
    )

    # Fee leaf
    fee_leaf = evaluator.add_leaf(
        id="tsa_fee",
        desc="States the TSA PreCheck new enrollment fee is $85 and that it covers 5 years (per constraints)",
        parent=tsa_node,
        critical=True,
    )
    fee_claim = "The answer states that the TSA PreCheck new enrollment fee is $85 and that it covers 5 years."
    await evaluator.verify(
        claim=fee_claim,
        node=fee_leaf,
        sources=None,
        additional_instruction=(
            "Judge only based on the answer text. The answer must explicitly say '$85' and '5 years' for coverage."
            f" The answer's extracted fee: {tsa.fee_amount}; coverage term: {tsa.fee_coverage_years}."
        ),
    )

    # Processing time node (split into two checks under a critical parent for clarity)
    processing_node = evaluator.add_parallel(
        id="tsa_processing_time",
        desc="Provides processing-time guidance consistent with constraints and a recommended latest-start timeframe before March 15, 2026",
        parent=tsa_node,
        critical=True,
    )

    # Processing facts leaf
    processing_facts_leaf = evaluator.add_leaf(
        id="tsa_processing_time_facts",
        desc="States processing is typically 3–5 days and can take up to 60 days",
        parent=processing_node,
        critical=True,
    )
    processing_facts_claim = "The answer states that TSA PreCheck processing is typically 3–5 days and can take up to 60 days."
    await evaluator.verify(
        claim=processing_facts_claim,
        node=processing_facts_leaf,
        sources=None,
        additional_instruction=(
            "Judge only based on the answer text. Require both phrases: 'typically 3–5 days' and 'can take up to 60 days'. "
            f"Extracted typical: {tsa.processing_typical}; max: {tsa.processing_max}."
        ),
    )

    # Processing recommendation leaf
    processing_reco_leaf = evaluator.add_leaf(
        id="tsa_processing_time_recommendation",
        desc=("States a recommended latest-start timeframe before March 15, 2026 that accounts for the up-to-60-day possibility"),
        parent=processing_node,
        critical=True,
    )
    processing_reco_claim = (
        "The answer provides a recommended latest-start timeframe before the March 15, 2026 departure that accounts for the "
        "possibility of up to 60 days processing (e.g., at least 60 days before departure, two months before, or a date no later than mid‑January 2026)."
    )
    await evaluator.verify(
        claim=processing_reco_claim,
        node=processing_reco_leaf,
        sources=None,
        additional_instruction=(
            f"Judge only based on the answer text. Accept forms like 'start at least 60 days before', 'two months before', or a concrete date no later than January 14, 2026. "
            f"Extracted recommendation: timeframe='{tsa.processing_recommended_timeframe}', date='{tsa.processing_recommended_date}'. Departure={DEPARTURE_DATE.strftime('%B %d, %Y')}."
        ),
    )


async def verify_maldives_requirements(evaluator: Evaluator, parent_node, mdv: MaldivesInfo) -> None:
    # Maldives entry requirements parent node (critical parallel)
    mdv_node = evaluator.add_parallel(
        id="maldives_entry_requirements",
        desc="Provide all Maldives entry requirements relevant to the trip, including passport validity rules and required proofs per constraints",
        parent=parent_node,
        critical=True,
    )

    # Visa on arrival leaf
    visa_leaf = evaluator.add_leaf(
        id="maldives_visa_on_arrival",
        desc="States that the Maldives grants a free 30-day tourist visa on arrival (no pre-approval required) per constraints",
        parent=mdv_node,
        critical=True,
    )
    visa_claim = "The answer states that the Maldives grants a free 30-day tourist visa on arrival and no pre-approval is required."
    await evaluator.verify(
        claim=visa_claim,
        node=visa_leaf,
        sources=None,
        additional_instruction="Judge only based on the answer text. Confirm 'free 30‑day visa on arrival' and 'no pre‑approval' are both stated explicitly.",
    )

    # Passport validity group
    passport_node = evaluator.add_parallel(
        id="maldives_passport_validity",
        desc="States the passport validity rule and applies it to the stated arrival date",
        parent=mdv_node,
        critical=True,
    )

    # Passport validity rule leaf
    passport_rule_leaf = evaluator.add_leaf(
        id="maldives_passport_validity_rule",
        desc="States the passport validity rule per constraints (at least 1 month beyond date of arrival)",
        parent=passport_node,
        critical=True,
    )
    passport_rule_claim = "The answer states that a passport must be valid at least 1 month beyond the date of arrival in the Maldives."
    await evaluator.verify(
        claim=passport_rule_claim,
        node=passport_rule_leaf,
        sources=None,
        additional_instruction=f"Judge only based on the answer text. Extracted rule: {mdv.passport_validity_rule_text}. Require explicit 'at least 1 month beyond arrival'.",
    )

    # Passport validity applied leaf
    passport_applied_leaf = evaluator.add_leaf(
        id="maldives_passport_validity_applied",
        desc=("Applies the passport validity rule to the arrival date (March 15, 2026) by giving the implied minimum "
              f"passport-valid-through timing/date (e.g., through at least {MIN_VALID_THROUGH_STR}, or equivalently at least one month/30+ days beyond arrival)"),
        parent=passport_node,
        critical=True,
    )
    passport_applied_claim = (
        f"The answer applies the passport validity rule to the stated arrival (March 15, 2026) by giving the implied minimum valid‑through date, "
        f"such as 'through at least {MIN_VALID_THROUGH_STR}' or an equivalent phrasing ('at least one month/30+ days beyond arrival')."
    )
    await evaluator.verify(
        claim=passport_applied_claim,
        node=passport_applied_leaf,
        sources=None,
        additional_instruction=(
            f"Judge only based on the answer text. Look for an explicit application to March 15, 2026 (e.g., 'valid through at least {MIN_VALID_THROUGH_STR}'). "
            f"Extracted applied text: {mdv.applied_min_valid_date_text}."
        ),
    )

    # Onward/return ticket leaf
    onward_leaf = evaluator.add_leaf(
        id="maldives_onward_return_ticket",
        desc=("States that proof of onward or return travel is required for Maldives entry (per constraints) and specifies what document types "
              "can serve as proof at a general level (e.g., a return/onward flight booking confirmation or itinerary)"),
        parent=mdv_node,
        critical=True,
    )
    onward_claim = (
        "The answer states that proof of onward or return travel is required for Maldives entry and it specifies acceptable proof like a flight booking confirmation or itinerary."
    )
    await evaluator.verify(
        claim=onward_claim,
        node=onward_leaf,
        sources=None,
        additional_instruction=(
            f"Judge only based on the answer text. Require explicit mention of proof of onward/return travel and examples of acceptable proof documents. "
            f"Extracted: {mdv.onward_return_proof_text}."
        ),
    )

    # Accommodation booking leaf
    accom_leaf = evaluator.add_leaf(
        id="maldives_accommodation_booking",
        desc=("States that confirmed accommodation booking is required for Maldives entry (per constraints) and specifies what document types "
              "can serve as proof at a general level (e.g., hotel/resort/guesthouse booking confirmation)"),
        parent=mdv_node,
        critical=True,
    )
    accom_claim = (
        "The answer states that confirmed accommodation booking is required for Maldives entry and it specifies acceptable proof like a hotel/resort/guesthouse booking confirmation."
    )
    await evaluator.verify(
        claim=accom_claim,
        node=accom_leaf,
        sources=None,
        additional_instruction=(
            f"Judge only based on the answer text. Require explicit mention of confirmed accommodation and examples of acceptable proof. "
            f"Extracted: {mdv.accommodation_proof_text}."
        ),
    )

    # Sufficient funds leaf
    funds_leaf = evaluator.add_leaf(
        id="maldives_sufficient_funds",
        desc=("States that travelers must demonstrate sufficient funds to cover their stay and includes the constraint-provided recommended amount "
              "(at least US$100 + US$50 per day)"),
        parent=mdv_node,
        critical=True,
    )
    funds_claim = "The answer states that travelers must demonstrate sufficient funds and includes the recommended amount: at least US$100 plus US$50 per day."
    await evaluator.verify(
        claim=funds_claim,
        node=funds_leaf,
        sources=None,
        additional_instruction=(
            f"Judge only based on the answer text. Require both 'sufficient funds' and the specific recommended amount 'US$100 + US$50/day'. "
            f"Extracted: {mdv.sufficient_funds_text}."
        ),
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
    Evaluate the answer for TSA PreCheck enrollment and Maldives entry requirements completeness and correctness per constraints.
    """
    # Initialize evaluator (root is non-critical by design; add a critical child as the actual root of rubric)
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

    # Create rubric root node (critical parallel)
    rubric_root = evaluator.add_parallel(
        id="travel_preparation",
        desc="Answer covers TSA PreCheck enrollment requirements and Maldives entry requirements for the stated trip",
        parent=root,
        critical=True,
    )

    # Extract TSA and Maldives info from the answer
    tsa_info = await evaluator.extract(
        prompt=prompt_extract_tsa_info(),
        template_class=TSAInfo,
        extraction_name="tsa_info_extraction",
    )
    mdv_info = await evaluator.extract(
        prompt=prompt_extract_maldives_info(),
        template_class=MaldivesInfo,
        extraction_name="maldives_info_extraction",
    )

    # Add helpful ground-truth context (dates) to summary (not used for verification directly)
    evaluator.add_ground_truth({
        "departure_date": DEPARTURE_DATE.strftime("%B %d, %Y"),
        "arrival_date": ARRIVAL_DATE.strftime("%B %d, %Y"),
        "passport_min_valid_through": MIN_VALID_THROUGH_STR,
        "tsa_processing_expectation": {"typical": "3–5 days", "max": "up to 60 days"},
        "tsa_fee_and_coverage": {"fee": "$85", "coverage_years": "5 years"},
        "maldives_visa_on_arrival": "Free 30-day tourist visa, no pre-approval",
        "maldives_proofs_required": ["onward/return travel", "confirmed accommodation", "sufficient funds"]
    })

    # Build TSA subtree verifications
    await verify_tsa_precheck(evaluator, rubric_root, tsa_info)

    # Build Maldives subtree verifications
    await verify_maldives_requirements(evaluator, rubric_root, mdv_info)

    # Return standardized summary
    return evaluator.get_summary()