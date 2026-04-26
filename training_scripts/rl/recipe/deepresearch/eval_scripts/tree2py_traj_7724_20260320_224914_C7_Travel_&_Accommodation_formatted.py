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
TASK_ID = "lga_to_nairobi_turkish_pretravel"
TASK_DESCRIPTION = """
A US citizen is planning to fly from LaGuardia Airport (LGA) in New York to Nairobi, Kenya using Turkish Airlines in Economy Class, and needs to park their car at the airport for the duration of the trip. Provide comprehensive pre-travel planning information including: 
(1) Which terminal at LaGuardia Airport does Turkish Airlines operate from? 
(2) What travel authorization is required for US citizens to enter Kenya? 
(3) What is the standard processing time and fee for this authorization? 
(4) What are the passport validity and blank page requirements for Kenya entry? 
(5) What documents are needed to apply for Kenya's entry authorization? 
(6) What customs form must be completed upon arrival in Kenya? 
(7) What is the cost range for on-site parking at Turkish Airlines' LaGuardia terminal for a 24-hour period? 
(8) How many checked bags are allowed in Turkish Airlines Economy Class on international flights under the piece concept? 
(9) What is the maximum weight per checked bag in Turkish Airlines Economy Class on piece concept routes? 
(10) Where is the baggage claim area located at LaGuardia terminals? 
(11) Through which city do Turkish Airlines flights from New York to Nairobi connect? 
Provide all answers with supporting reference URLs.
"""

# Optional: record expected facts for reference (not used as gating checks)
EXPECTED_INFO = {
    "turkish_airlines_terminal": "Terminal B (LGA)",
    "kenya_authorization": "Electronic Travel Authorization (eTA)",
    "eta_processing_time": "3 working (business) days",
    "eta_cost": "USD $30 (base government fee)",
    "passport_validity": "At least 6 months beyond arrival/entry date",
    "passport_blank_pages": "At least one blank page",
    "eta_documents_minimum": ["valid passport", "selfie/passport-type photo", "contact details (email and phone)"],
    "customs_form": "Passenger Declaration Form F88",
    "terminal_b_parking_24h_range": "$39–$80 per 24 hours",
    "checked_bags_allowance_piece": "2 checked bags (Economy, international piece concept)",
    "baggage_weight_limit_piece": "23 kg (50 lb) per bag",
    "baggage_claim_location": "Arrivals level",
    "connection_city": "Istanbul Airport (IST)",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PreTravelExtraction(BaseModel):
    # 1) LGA terminal for Turkish Airlines
    terminal: Optional[str] = None
    terminal_urls: List[str] = Field(default_factory=list)

    # 2) Kenya entry authorization for US citizens
    kenya_authorization: Optional[str] = None
    kenya_authorization_urls: List[str] = Field(default_factory=list)

    # 3) eTA processing time
    eta_processing_time: Optional[str] = None
    eta_processing_time_urls: List[str] = Field(default_factory=list)

    # 4) eTA cost / fee
    eta_cost: Optional[str] = None
    eta_cost_urls: List[str] = Field(default_factory=list)

    # 5) Passport validity requirement
    passport_validity: Optional[str] = None
    passport_validity_urls: List[str] = Field(default_factory=list)

    # 6) Passport blank pages
    passport_blank_pages: Optional[str] = None
    passport_blank_pages_urls: List[str] = Field(default_factory=list)

    # 7) eTA documents required
    eta_documents: List[str] = Field(default_factory=list)
    eta_documents_urls: List[str] = Field(default_factory=list)

    # 8) Customs form on arrival
    customs_form: Optional[str] = None
    customs_form_urls: List[str] = Field(default_factory=list)

    # 9) Terminal B parking cost per 24 hours (range)
    parking_cost_24h_range: Optional[str] = None
    parking_cost_urls: List[str] = Field(default_factory=list)

    # 10) Checked bags allowance (Economy, international piece concept)
    checked_bags_allowance: Optional[str] = None
    checked_bags_urls: List[str] = Field(default_factory=list)

    # 11) Max weight per checked bag (piece concept)
    baggage_weight_limit: Optional[str] = None
    baggage_weight_urls: List[str] = Field(default_factory=list)

    # 12) Baggage claim location at LGA
    baggage_claim_location: Optional[str] = None
    baggage_claim_urls: List[str] = Field(default_factory=list)

    # 13) Connection city for JFK/LGA->NBO on Turkish Airlines
    connection_city: Optional[str] = None
    connection_city_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pretravel_info() -> str:
    return """
    Extract the specific pre-travel planning information as directly stated in the answer. 
    Return concise values and the supporting URL list for each item as they appear in the answer text.

    Required fields (return null if missing; return [] for missing URL lists):
    1) terminal: Short terminal name at LGA for Turkish Airlines (e.g., "Terminal B")
       terminal_urls: URLs cited that explicitly support the terminal assignment.
    2) kenya_authorization: The named travel authorization US citizens need to enter Kenya (e.g., "Electronic Travel Authorization (eTA)")
       kenya_authorization_urls: Supporting URLs.
    3) eta_processing_time: The standard processing time (e.g., "3 working days")
       eta_processing_time_urls: Supporting URLs.
    4) eta_cost: The standard base government fee for the authorization (e.g., "USD $30")
       eta_cost_urls: Supporting URLs.
    5) passport_validity: The validity requirement (e.g., "at least 6 months beyond arrival")
       passport_validity_urls: Supporting URLs.
    6) passport_blank_pages: The blank page requirement (e.g., "at least one blank page", "1 blank page")
       passport_blank_pages_urls: Supporting URLs.
    7) eta_documents: A list of required application documents as stated (e.g., ["valid passport", "selfie/passport-type photo", "contact details (email and phone)"])
       eta_documents_urls: Supporting URLs.
    8) customs_form: The named customs/passenger declaration form required on arrival in Kenya (e.g., "Passenger Declaration Form F88")
       customs_form_urls: Supporting URLs.
    9) parking_cost_24h_range: The 24-hour on-site parking cost range for LGA Terminal B, as stated (e.g., "$39–$80")
       parking_cost_urls: Supporting URLs.
    10) checked_bags_allowance: Stated number of checked bags in Economy for Turkish Airlines under the international piece concept (e.g., "2 checked bags")
        checked_bags_urls: Supporting URLs.
    11) baggage_weight_limit: Stated max weight per checked bag on piece concept routes (e.g., "23 kg (50 lb)")
        baggage_weight_urls: Supporting URLs.
    12) baggage_claim_location: Where baggage claim is located at LGA terminals (e.g., "Arrivals level")
        baggage_claim_urls: Supporting URLs.
    13) connection_city: The connection city for Turkish Airlines flights from New York to Nairobi (e.g., "Istanbul Airport (IST)")
        connection_city_urls: Supporting URLs.

    Rules:
    - Extract ONLY what is explicitly stated in the answer.
    - For each *_urls field, include only URLs present in the answer; include duplicates if repeated.
    - Keep value fields succinct (avoid extra commentary).
    - If a value is a range, preserve it succinctly (e.g., "$39–$80").
    - If units are provided (kg/lb, USD), keep them in the value string.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_value(val: Optional[str] | List[str]) -> bool:
    if val is None:
        return False
    if isinstance(val, str):
        return val.strip() != ""
    if isinstance(val, list):
        return len(val) > 0
    return False


def _norm_list_str(items: List[str]) -> str:
    return ", ".join([s.strip() for s in items if s and s.strip()]) if items else ""


# --------------------------------------------------------------------------- #
# Build per-item verification                                                 #
# --------------------------------------------------------------------------- #
async def add_item_verification(
    evaluator: Evaluator,
    root,
    *,
    item_id: str,
    parent_desc: str,
    value_present: bool,
    urls_present: bool,
    claim: str,
    sources: List[str],
    additional_instruction: str,
) -> None:
    """
    For each rubric leaf, we create a small sub-tree:
    - A parent parallel node to group checks for this item (non-critical; allows partial credit overall).
    - A critical existence check (value + at least one URL).
    - A critical verification leaf grounded by the provided URLs (skipped automatically if existence fails).
    """
    parent_node = evaluator.add_parallel(
        id=item_id,
        desc=parent_desc,
        parent=root,
        critical=False
    )

    # Existence (value + URL) must both be present
    exists_node = evaluator.add_custom_node(
        result=(value_present and urls_present),
        id=f"{item_id}_exists",
        desc=f"{parent_desc} — value and source URL(s) provided in the answer",
        parent=parent_node,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id=f"{item_id}_supported",
        desc=parent_desc,
        parent=parent_node,
        critical=True
    )

    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=sources,
        additional_instruction=additional_instruction
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
    Evaluate an answer for the LaGuardia -> Nairobi (Turkish Airlines) pre-travel planning task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Each requirement is independent; allow partial credit
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

    # Add GT info for context in the summary (non-gating)
    evaluator.add_ground_truth(EXPECTED_INFO, gt_type="expected_info_reference")

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_pretravel_info(),
        template_class=PreTravelExtraction,
        extraction_name="pretravel_extraction"
    )

    # Build all item verifications
    tasks = []

    # 1) Turkish Airlines terminal at LGA
    term_val = extracted.terminal or ""
    tasks.append(
        add_item_verification(
            evaluator, root,
            item_id="turkish_airlines_terminal",
            parent_desc="Correctly identify that Turkish Airlines operates from Terminal B at LaGuardia Airport",
            value_present=_has_value(extracted.terminal),
            urls_present=_has_value(extracted.terminal_urls),
            claim=f"Turkish Airlines operates from {term_val} at LaGuardia Airport (LGA).",
            sources=extracted.terminal_urls,
            additional_instruction="Verify the airline-terminal assignment on the provided page(s). Accept minor formatting variants like 'Terminal B' vs 'Terminal B (LGA)'. The claim is correct only if the page explicitly indicates Turkish Airlines at Terminal B."
        )
    )

    # 2) Kenya entry authorization for US citizens
    auth_val = extracted.kenya_authorization or ""
    tasks.append(
        add_item_verification(
            evaluator, root,
            item_id="kenya_eta_requirement",
            parent_desc="State that US citizens require an Electronic Travel Authorization (eTA) to enter Kenya",
            value_present=_has_value(extracted.kenya_authorization),
            urls_present=_has_value(extracted.kenya_authorization_urls),
            claim=f"US citizens require {auth_val} to enter Kenya.",
            sources=extracted.kenya_authorization_urls,
            additional_instruction="Confirm that the page explicitly states US citizens must obtain an Electronic Travel Authorization (eTA). Accept abbreviations like 'eTA' or full 'Electronic Travel Authorization'."
        )
    )

    # 3) eTA processing time
    proc_val = extracted.eta_processing_time or ""
    tasks.append(
        add_item_verification(
            evaluator, root,
            item_id="eta_processing_time",
            parent_desc="Provide the standard eTA processing time of 3 working days",
            value_present=_has_value(extracted.eta_processing_time),
            urls_present=_has_value(extracted.eta_processing_time_urls),
            claim=f"The standard processing time for Kenya's eTA is {proc_val}.",
            sources=extracted.eta_processing_time_urls,
            additional_instruction="Check for phrasing like '3 working days' or '3 business days'. Allow small variants such as 'up to 3 working days'."
        )
    )

    # 4) eTA cost
    cost_val = extracted.eta_cost or ""
    tasks.append(
        add_item_verification(
            evaluator, root,
            item_id="eta_cost",
            parent_desc="State the standard eTA fee of USD $30",
            value_present=_has_value(extracted.eta_cost),
            urls_present=_has_value(extracted.eta_cost_urls),
            claim=f"The standard government fee for Kenya's eTA is {cost_val}.",
            sources=extracted.eta_cost_urls,
            additional_instruction="Verify the base government fee (exclude service/processing fees). Accept '$30', 'USD 30', or 'US$ 30' as equivalent."
        )
    )

    # 5) Passport validity
    validity_val = extracted.passport_validity or ""
    tasks.append(
        add_item_verification(
            evaluator, root,
            item_id="passport_validity",
            parent_desc="Specify that passport must be valid for at least 6 months beyond the planned arrival date in Kenya",
            value_present=_has_value(extracted.passport_validity),
            urls_present=_has_value(extracted.passport_validity_urls),
            claim=f"Kenya requires that a passport be valid {validity_val} for entry.",
            sources=extracted.passport_validity_urls,
            additional_instruction="Confirm the rule is at least 6 months validity from the date of entry/arrival. Accept synonyms (e.g., 'six months')."
        )
    )

    # 6) Passport blank pages
    pages_val = extracted.passport_blank_pages or ""
    tasks.append(
        add_item_verification(
            evaluator, root,
            item_id="passport_blank_pages",
            parent_desc="State that at least one blank page is required in the passport",
            value_present=_has_value(extracted.passport_blank_pages),
            urls_present=_has_value(extracted.passport_blank_pages_urls),
            claim=f"At least {pages_val} blank page is required in the passport for Kenyan entry.",
            sources=extracted.passport_blank_pages_urls,
            additional_instruction="Confirm a minimum of one blank page is required. Accept small textual variants ('1 blank page', 'at least one blank page')."
        )
    )

    # 7) eTA documents
    docs_val_list = extracted.eta_documents or []
    docs_val = _norm_list_str(docs_val_list)
    tasks.append(
        add_item_verification(
            evaluator, root,
            item_id="eta_documents",
            parent_desc="List the required eTA application documents: valid passport, selfie or passport-type photo, and contact details (email and phone number)",
            value_present=_has_value(extracted.eta_documents),
            urls_present=_has_value(extracted.eta_documents_urls),
            claim=f"The required documents to apply for Kenya's entry authorization include: {docs_val}.",
            sources=extracted.eta_documents_urls,
            additional_instruction=(
                "Verify the required application materials for Kenya's eTA/electronic authorization. "
                "Consider synonyms: 'selfie' ~ 'passport photo' or 'recent color photograph'; 'contact details' should include email and phone. "
                "The list should cover at least: a valid passport, a selfie/passport-type photo, and contact details (email and phone)."
            )
        )
    )

    # 8) Customs/passenger declaration form
    customs_val = extracted.customs_form or ""
    tasks.append(
        add_item_verification(
            evaluator, root,
            item_id="customs_form",
            parent_desc="Mention that all passengers arriving in Kenya must complete Passenger Declaration Form F88",
            value_present=_has_value(extracted.customs_form),
            urls_present=_has_value(extracted.customs_form_urls),
            claim=f"Passengers arriving in Kenya must complete {customs_val}.",
            sources=extracted.customs_form_urls,
            additional_instruction="Confirm that the page indicates a mandatory passenger/customs declaration on arrival, often referred to as Passenger Declaration Form F88 (accept 'Form F88' or equivalent official naming)."
        )
    )

    # 9) Parking cost (24h range) at Terminal B
    park_val = extracted.parking_cost_24h_range or ""
    tasks.append(
        add_item_verification(
            evaluator, root,
            item_id="parking_cost",
            parent_desc="Provide the Terminal B on-site parking cost range of $39 to $80 per 24-hour period",
            value_present=_has_value(extracted.parking_cost_24h_range),
            urls_present=_has_value(extracted.parking_cost_urls),
            claim=f"The on-site parking cost for a 24-hour period at LaGuardia Terminal B is {park_val}.",
            sources=extracted.parking_cost_urls,
            additional_instruction="Verify the 24-hour rate(s) for Terminal B on-site parking. Interpret ranges like '$39–$80' as inclusive. Accept equivalent formats (per day/daily max)."
        )
    )

    # 10) Checked bags allowance under piece concept
    bags_val = extracted.checked_bags_allowance or ""
    tasks.append(
        add_item_verification(
            evaluator, root,
            item_id="checked_bags_allowance",
            parent_desc="State that Turkish Airlines Economy Class allows 2 checked bags on international piece concept flights",
            value_present=_has_value(extracted.checked_bags_allowance),
            urls_present=_has_value(extracted.checked_bags_urls),
            claim=f"Under the international piece concept, Turkish Airlines Economy Class allows {bags_val} on applicable routes.",
            sources=extracted.checked_bags_urls,
            additional_instruction="Check Turkish Airlines baggage rules for piece concept routes (e.g., transatlantic). The expected allowance is 2 checked bags in Economy; accept phrasing like '2 pieces'."
        )
    )

    # 11) Max weight per checked bag (piece concept)
    weight_val = extracted.baggage_weight_limit or ""
    tasks.append(
        add_item_verification(
            evaluator, root,
            item_id="baggage_weight_limit",
            parent_desc="Specify that Turkish Airlines Economy Class maximum weight per checked bag is 23kg (50 lbs) on piece concept routes",
            value_present=_has_value(extracted.baggage_weight_limit),
            urls_present=_has_value(extracted.baggage_weight_urls),
            claim=f"The maximum weight per checked bag on Turkish Airlines Economy Class piece concept routes is {weight_val}.",
            sources=extracted.baggage_weight_urls,
            additional_instruction="Confirm the per-bag limit is 23 kg (50 lb). Accept minor text variations (e.g., '23kg', '23 kilograms', '50 lbs')."
        )
    )

    # 12) Baggage claim location at LGA
    claim_loc_val = extracted.baggage_claim_location or ""
    tasks.append(
        add_item_verification(
            evaluator, root,
            item_id="baggage_claim_location",
            parent_desc="Indicate that baggage claim is located on the arrivals level at LaGuardia terminals",
            value_present=_has_value(extracted.baggage_claim_location),
            urls_present=_has_value(extracted.baggage_claim_urls),
            claim=f"At LaGuardia terminals, baggage claim is located on the {claim_loc_val}.",
            sources=extracted.baggage_claim_urls,
            additional_instruction="Verify terminal maps or guides stating baggage claim is on the Arrivals level. Accept capitalization variants."
        )
    )

    # 13) Connection city for New York -> Nairobi on Turkish Airlines
    conn_val = extracted.connection_city or ""
    tasks.append(
        add_item_verification(
            evaluator, root,
            item_id="connection_city",
            parent_desc="Identify that Turkish Airlines flights from New York to Nairobi connect through Istanbul Airport",
            value_present=_has_value(extracted.connection_city),
            urls_present=_has_value(extracted.connection_city_urls),
            claim=f"Turkish Airlines flights from New York to Nairobi connect through {conn_val}.",
            sources=extracted.connection_city_urls,
            additional_instruction="Confirm the typical routing shows a connection via Istanbul (IST). Accept 'Istanbul' or 'Istanbul Airport (IST)'."
        )
    )

    # Execute all per-item subtasks
    await asyncio.gather(*tasks)

    # Return final structured summary
    return evaluator.get_summary()