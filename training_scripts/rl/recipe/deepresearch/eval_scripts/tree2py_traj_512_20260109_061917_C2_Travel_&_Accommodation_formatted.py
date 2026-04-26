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
TASK_ID = "aaa_four_diamond_amtrak_lounge"
TASK_DESCRIPTION = (
    "Identify one hotel in the United States that currently holds an AAA Four Diamond rating and is located in a city "
    "that has an Amtrak staffed Metropolitan Lounge. For the selected hotel, provide the hotel's complete name and full "
    "street address, along with a reference URL from an official AAA source that verifies its Four Diamond rating. "
    "Additionally, provide the Single Visit Pass price for the Amtrak Metropolitan Lounge in that city, along with a "
    "reference URL confirming the lounge's existence in that city."
)

# Allowed cities with staffed Amtrak Metropolitan Lounges and expected Single Visit Pass price
CANONICAL_CITIES = [
    "boston",
    "chicago",
    "los angeles",
    "new york",
    "philadelphia",
    "portland",
    "washington d.c."
]

EXPECTED_LOUNGE_PRICE = {
    "boston": 35,
    "chicago": 35,
    "los angeles": 35,
    "new york": 50,        # Moynihan Train Hall (NYC)
    "philadelphia": 35,
    "portland": 35,        # Portland, OR
    "washington d.c.": 35,
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SelectedHotelAndLounge(BaseModel):
    # Hotel info
    hotel_name: Optional[str] = None
    hotel_address: Optional[str] = None
    hotel_city: Optional[str] = None
    # AAA verification
    aaa_verification_url: Optional[str] = None
    # Lounge info
    lounge_url: Optional[str] = None
    single_visit_pass_price: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_selected_items() -> str:
    return """
Extract exactly one selected hotel and the related lounge information as provided in the answer.

Required fields:
- hotel_name: The hotel's complete official name, exactly as written in the answer.
- hotel_address: The hotel's full street address as given (include street number, street name, city, state, ZIP if shown).
- hotel_city: The U.S. city where the hotel is located (for example, extract "Chicago" from "Chicago, IL").
- aaa_verification_url: A URL from an official AAA source referenced in the answer that verifies the hotel's AAA Four Diamond rating. This should typically be a URL on an AAA-owned domain (e.g., *.aaa.com). If multiple are present, pick the most direct.
- lounge_url: A URL referenced in the answer that confirms the existence of an Amtrak "Metropolitan Lounge" in the same city as the hotel.
- single_visit_pass_price: The stated Single Visit Pass price for that city's Metropolitan Lounge exactly as mentioned in the answer (for example, "$35" or "$50"). If a numeric value is shown without a currency symbol, extract it as-is.

Extraction rules:
- Only extract what is explicitly present in the answer. Do not invent any fields.
- If any required item is missing from the answer, return null for that field.
- If there are multiple hotels or lounges mentioned, extract only the one the answer proposes as the final choice.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_city(raw: Optional[str], address: Optional[str]) -> Optional[str]:
    if not raw and not address:
        return None

    def clean(s: str) -> str:
        return ' '.join(s.lower().strip().replace("’", "'").replace(".", "").split())

    city_src = clean(raw) if raw else ""
    addr_src = clean(address) if address else ""

    # Common aliases
    if city_src in ("nyc", "new york city", "new york, ny"):
        city_src = "new york"
    if city_src in ("la", "l a", "los angeles ca"):
        city_src = "los angeles"
    if city_src in ("washington dc", "washington d c", "washington, dc", "washington, d c"):
        city_src = "washington dc"
    if city_src in ("washington",):
        # Try to disambiguate via address tokens
        if " dc" in addr_src or " d c" in addr_src or "district of columbia" in addr_src:
            city_src = "washington dc"
        else:
            # Ambiguous "Washington" (state vs DC) — treat as not matching lounge list unless address hints DC
            return None
    if city_src in ("portland or", "portland, or"):
        city_src = "portland"
    if "portland" in city_src:
        # Avoid misclassifying Portland, ME
        if "maine" in addr_src or " me " in f" {addr_src} ":
            return None
        city_src = "portland"

    # Map to canonical
    if city_src in ("boston",):
        return "boston"
    if city_src in ("chicago",):
        return "chicago"
    if city_src in ("los angeles",):
        return "los angeles"
    if city_src in ("new york",):
        return "new york"
    if city_src in ("philadelphia",):
        return "philadelphia"
    if city_src in ("portland",):
        return "portland"
    if city_src in ("washington dc", "washington d c"):
        return "washington d.c."

    return None


def _parse_price_to_int(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    s = raw.strip().lower()
    # keep only digits
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def _has_street_number(address: Optional[str]) -> bool:
    if not address:
        return False
    return any(ch.isdigit() for ch in address)


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: SelectedHotelAndLounge) -> None:
    # Root node (Sequential as specified)
    root = evaluator.find_node("root")

    # Add global ground truth/meta info for debugging
    evaluator.add_ground_truth({
        "allowed_cities": CANONICAL_CITIES,
        "expected_lounge_price": EXPECTED_LOUNGE_PRICE
    }, gt_type="constraints")

    # Compute normalized city and expected price
    norm_city = _normalize_city(extracted.hotel_city, extracted.hotel_address)
    expected_price_val = EXPECTED_LOUNGE_PRICE.get(norm_city) if norm_city else None
    provided_price_val = _parse_price_to_int(extracted.single_visit_pass_price)

    # 1) Hotel_Meets_Both_Criteria (parallel, critical)
    meets_node = evaluator.add_parallel(
        id="hotel_meets_both_criteria",
        desc="The selected hotel must satisfy both the city location requirement and the AAA Four Diamond rating requirement",
        parent=root,
        critical=True
    )

    # 1.a) City_Criterion (leaf) - check that hotel is in allowed lounge cities
    city_ok = norm_city in CANONICAL_CITIES
    evaluator.add_custom_node(
        result=city_ok,
        id="city_criterion",
        desc="The hotel is located in one of the cities with an Amtrak staffed Metropolitan Lounge (Boston, Chicago, Los Angeles, New York, Philadelphia, Portland, Washington D.C.)",
        parent=meets_node,
        critical=True
    )

    # 1.b) Rating_Criterion (leaf) - verify AAA Four Diamond using the AAA URL
    rating_leaf = evaluator.add_leaf(
        id="rating_criterion",
        desc="The hotel holds a current AAA Four Diamond rating",
        parent=meets_node,
        critical=True
    )
    # Ensure URL present; if missing, fail this leaf immediately
    if not extracted.aaa_verification_url:
        rating_leaf.score = 0.0
        rating_leaf.status = "failed"
    else:
        hotel_for_claim = extracted.hotel_name or "the property"
        city_for_claim = extracted.hotel_city or (norm_city or "the listed city")
        claim_rating = (
            f"The official AAA page confirms that '{hotel_for_claim}' in {city_for_claim} "
            f"has an AAA Four Diamond rating (current)."
        )
        await evaluator.verify(
            claim=claim_rating,
            node=rating_leaf,
            sources=extracted.aaa_verification_url,
            additional_instruction=(
                "Only support if this webpage is an official AAA source (e.g., a domain that includes 'aaa.com' "
                "or a clear AAA organization site) and it explicitly indicates the property has a Four Diamond rating. "
                "Allow minor name/casing variations and city formatting."
            ),
        )

    # 2) Complete_Information_Package (parallel, critical)
    info_pkg_node = evaluator.add_parallel(
        id="complete_information_package",
        desc="Provide complete information about both the selected hotel and the city's Metropolitan Lounge",
        parent=root,
        critical=True
    )

    # 2.a) Hotel_Information (parallel, critical)
    hotel_info_node = evaluator.add_parallel(
        id="hotel_information",
        desc="Provide complete information about the selected hotel including name, address, and AAA verification",
        parent=info_pkg_node,
        critical=True
    )

    # 2.a.i) Hotel_Name_And_Address (leaf) - ensure name and full street address provided
    name_addr_ok = bool(extracted.hotel_name and extracted.hotel_name.strip()) and _has_street_number(extracted.hotel_address)
    evaluator.add_custom_node(
        result=name_addr_ok,
        id="hotel_name_and_address",
        desc="Hotel's full name and complete street address are provided",
        parent=hotel_info_node,
        critical=True
    )

    # 2.a.ii) AAA_Rating_Verification_URL (leaf) - verify via AAA URL again as the provided verification
    aaa_verify_leaf = evaluator.add_leaf(
        id="aaa_rating_verification_url",
        desc="A reference URL from an official AAA source verifies the hotel's Four Diamond rating",
        parent=hotel_info_node,
        critical=True
    )
    if not extracted.aaa_verification_url:
        aaa_verify_leaf.score = 0.0
        aaa_verify_leaf.status = "failed"
    else:
        hotel_for_claim = extracted.hotel_name or "the property"
        city_for_claim = extracted.hotel_city or (norm_city or "the listed city")
        claim_aaa_url = (
            f"This page is an official AAA source and confirms that '{hotel_for_claim}' in {city_for_claim} "
            f"has an AAA Four Diamond rating."
        )
        await evaluator.verify(
            claim=claim_aaa_url,
            node=aaa_verify_leaf,
            sources=extracted.aaa_verification_url,
            additional_instruction=(
                "Support only if the page is clearly an official AAA website (e.g., *.aaa.com or other official AAA "
                "club sites) and explicitly shows Four Diamond rating for the named property (allow minor variations)."
            ),
        )

    # 2.b) Lounge_Information (parallel, critical)
    lounge_info_node = evaluator.add_parallel(
        id="lounge_information",
        desc="Provide complete information about the Metropolitan Lounge including pass price and verification",
        parent=info_pkg_node,
        critical=True
    )

    # 2.b.i) Single_Visit_Pass_Price (leaf) - check correctness of price by the specified mapping
    price_correct = (expected_price_val is not None) and (provided_price_val == expected_price_val)
    evaluator.add_custom_node(
        result=price_correct,
        id="single_visit_pass_price",
        desc="Single Visit Pass price is correct for the city's Metropolitan Lounge ($35 for Boston, Chicago, Los Angeles, Philadelphia, Portland, Washington D.C.; $50 for New York Moynihan Train Hall)",
        parent=lounge_info_node,
        critical=True
    )

    # 2.b.ii) Lounge_Verification_URL (leaf) - verify that the lounge exists in the hotel's city
    lounge_verify_leaf = evaluator.add_leaf(
        id="lounge_verification_url",
        desc="A reference URL confirms the existence of the Amtrak Metropolitan Lounge in the hotel's city",
        parent=lounge_info_node,
        critical=True
    )
    if not extracted.lounge_url or not norm_city:
        lounge_verify_leaf.score = 0.0
        lounge_verify_leaf.status = "failed"
    else:
        city_for_claim = norm_city
        # Capitalize city for readability in claim
        city_readable = city_for_claim.title() if city_for_claim != "washington d.c." else "Washington D.C."
        claim_lounge = f"There is an Amtrak Metropolitan Lounge located in {city_readable}."
        await evaluator.verify(
            claim=claim_lounge,
            node=lounge_verify_leaf,
            sources=extracted.lounge_url,
            additional_instruction=(
                "Support only if the page explicitly mentions an Amtrak 'Metropolitan Lounge' in the specified city. "
                "For New York, references to the 'Moynihan Train Hall' Metropolitan Lounge are acceptable. "
                "Allow reasonable naming variations and check both text and screenshot."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the AAA Four Diamond hotel + Amtrak Metropolitan Lounge task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # As specified by the rubric
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

    # Extract structured data from the answer
    extracted: SelectedHotelAndLounge = await evaluator.extract(
        prompt=prompt_extract_selected_items(),
        template_class=SelectedHotelAndLounge,
        extraction_name="selected_hotel_and_lounge"
    )

    # Build and verify tree according to rubric
    await build_and_verify_tree(evaluator, extracted)

    # Return standardized summary
    return evaluator.get_summary()