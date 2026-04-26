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
TASK_ID = "apple_airpods_pro3_hr_ppg_eval"
TASK_DESCRIPTION = (
    "Identify the Apple wireless earbuds model that was announced in September 2025 and includes a heart rate sensing "
    "feature implemented via a custom photoplethysmography (PPG) sensor using infrared light pulsed at 256 times per second. "
    "For this identified model, provide: (1) The exact model name and announcement date, "
    "(2) Verification that the heart rate sensor uses infrared light pulsed at exactly 256 times per second, "
    "(3) The battery life specification (up to how many hours of listening time on a single charge with Active Noise Cancellation enabled), "
    "(4) The water resistance rating (IP rating under IEC standard 60529), "
    "(5) The starting price in U.S. dollars, "
    "(6) The date when in-store availability began, "
    "(7) The name of the major U.S. mobile carrier that experienced a significant network outage in January 2026, "
    "the specific dates of that outage, and the cause of the outage as reported by the carrier. "
    "For each piece of information, provide a reference URL from a credible source (apple.com for product specifications, "
    "or established news outlets for the network outage information)."
)

# Expected values used for simple sanity checks and to phrase claims
EXPECTED_MODEL_NAME = "Apple AirPods Pro 3"
EXPECTED_ANNOUNCEMENT_DATE = "September 9, 2025"
EXPECTED_PPG_PULSE_RATE = "256 times per second"
EXPECTED_BATTERY_ANC_HOURS = "8 hours"
EXPECTED_IP_RATING = "IP57"
EXPECTED_IP_STANDARD = "IEC 60529"
EXPECTED_STARTING_PRICE_USD = "$249"
EXPECTED_IN_STORE_AVAILABILITY = "September 19, 2025"
EXPECTED_OUTAGE_CARRIER = "Verizon"
EXPECTED_OUTAGE_DATES_TEXT = "January 14–15, 2026"
EXPECTED_OUTAGE_CAUSE = "software issue"
EXPECTED_DDT_REPORTS = "about 2.3 million"
EXPECTED_PRIOR_DISRUPTION_DATE = "September 30, 2024"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProductModel(BaseModel):
    model_name: Optional[str] = None
    announcement_date: Optional[str] = None
    model_urls: List[str] = Field(default_factory=list)


class SensorDetails(BaseModel):
    # raw phrasing as written in the answer (optional)
    description: Optional[str] = None
    # optional binary/self-reported flags (may be missing)
    mentions_custom_ppg: Optional[bool] = None
    mentions_infrared: Optional[bool] = None
    pulse_frequency_text: Optional[str] = None
    sensor_urls: List[str] = Field(default_factory=list)  # Prefer apple.com URLs


class BatteryDetails(BaseModel):
    anc_listening_time_single_charge: Optional[str] = None
    battery_urls: List[str] = Field(default_factory=list)  # Prefer apple.com URLs


class WaterDetails(BaseModel):
    ip_rating_str: Optional[str] = None  # e.g., "IP57"
    standard_text: Optional[str] = None  # e.g., "IEC 60529"
    water_urls: List[str] = Field(default_factory=list)  # Prefer apple.com URLs


class PriceDetails(BaseModel):
    starting_price_usd: Optional[str] = None  # e.g., "$249"
    price_urls: List[str] = Field(default_factory=list)  # Prefer apple.com URLs


class AvailabilityDetails(BaseModel):
    in_store_availability_date: Optional[str] = None  # e.g., "September 19, 2025"
    availability_urls: List[str] = Field(default_factory=list)  # Prefer apple.com URLs


class OutageDetails(BaseModel):
    carrier_name: Optional[str] = None
    outage_dates_text: Optional[str] = None  # e.g., "January 14–15, 2026"
    cause_summary: Optional[str] = None      # e.g., "software issue"
    downdetector_reports_text: Optional[str] = None  # e.g., "about 2.3 million"
    prior_disruption_date: Optional[str] = None      # e.g., "September 30, 2024"
    prior_disruption_desc: Optional[str] = None      # short phrase
    outage_urls: List[str] = Field(default_factory=list)  # Credible news URLs


class FullExtraction(BaseModel):
    product: Optional[ProductModel] = None
    sensor: Optional[SensorDetails] = None
    battery: Optional[BatteryDetails] = None
    water: Optional[WaterDetails] = None
    price: Optional[PriceDetails] = None
    availability: Optional[AvailabilityDetails] = None
    outage: Optional[OutageDetails] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract the following fields exactly as claimed in the answer. If a field is not explicitly provided in the answer, return null or an empty list as appropriate. Extract only the URLs that are explicitly present in the answer.

Product (earbuds):
- product.model_name: the exact model name as written in the answer (e.g., "Apple AirPods Pro 3" or "AirPods Pro (3rd generation)")
- product.announcement_date: the announcement date as written (e.g., "September 9, 2025")
- product.model_urls: a list of URLs cited for the model identity and/or announcement date (prefer Apple URLs)

Heart-rate sensor details:
- sensor.description: the sentence or phrase describing the heart-rate sensor, if present
- sensor.mentions_custom_ppg: true/false if explicitly stated that it uses a custom photoplethysmography (PPG) sensor; null if not clearly stated
- sensor.mentions_infrared: true/false if explicitly stated that it uses infrared light; null if not clearly stated
- sensor.pulse_frequency_text: the pulse frequency text as written (e.g., "256 times per second", "256 Hz")
- sensor.sensor_urls: a list of URLs cited for the heart-rate sensor implementation (strongly prefer apple.com)

Battery life (ANC enabled):
- battery.anc_listening_time_single_charge: the exact phrase for listening time on a single charge with Active Noise Cancellation enabled (e.g., "up to 8 hours")
- battery.battery_urls: a list of URLs cited for the battery life (prefer apple.com)

Water resistance:
- water.ip_rating_str: the IP rating text (e.g., "IP57")
- water.standard_text: the standard reference text if present (e.g., "IEC 60529")
- water.water_urls: a list of URLs cited for water resistance (prefer apple.com)

Price:
- price.starting_price_usd: the stated U.S. starting price as written (e.g., "$249" or "249 USD")
- price.price_urls: a list of URLs cited for price (prefer apple.com)

In-store availability:
- availability.in_store_availability_date: the in-store availability date as written (e.g., "September 19, 2025")
- availability.availability_urls: a list of URLs cited for availability (prefer apple.com)

Carrier outage (January 2026):
- outage.carrier_name: the carrier's name as stated (e.g., "Verizon")
- outage.outage_dates_text: the outage date range as written (e.g., "January 14–15, 2026")
- outage.cause_summary: the cause of the outage as reported by the carrier (e.g., "software issue")
- outage.downdetector_reports_text: the approximate Downdetector report count if provided (e.g., "about 2.3 million")
- outage.prior_disruption_date: the prior disruption date as written (e.g., "September 30, 2024")
- outage.prior_disruption_desc: short description for the prior disruption if provided
- outage.outage_urls: a list of credible news URLs cited for the outage facts (carrier, dates, cause, and other outage details)

Rules:
- Only extract what the answer explicitly states.
- For URL fields, return every URL mentioned for that topic. Keep full URLs (include http/https).
- If a URL is missing or not present, leave the list empty.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def _filter_domain(urls: List[str], keyword: str) -> List[str]:
    kw = keyword.lower()
    return [u for u in urls if kw in u.lower()]


# --------------------------------------------------------------------------- #
# Subtree verifications                                                       #
# --------------------------------------------------------------------------- #
async def build_model_identity_and_announcement(evaluator: Evaluator, parent, product: Optional[ProductModel]) -> None:
    node = evaluator.add_parallel(
        id="model_identity_and_announcement",
        desc="Correctly identify the product model and announcement date per constraints, with citation",
        parent=parent,
        critical=True
    )

    model_name = (product.model_name or "").strip() if product else ""
    ann_date = (product.announcement_date or "").strip() if product else ""
    model_urls = _safe_urls(product.model_urls if product else None)
    apple_urls = _filter_domain(model_urls, "apple.com")

    # Leaf: model_is_airpods_pro_3 (simple equivalence check)
    leaf_model = evaluator.add_leaf(
        id="model_is_airpods_pro_3",
        desc="Model is explicitly identified as Apple AirPods Pro 3",
        parent=node,
        critical=True
    )
    claim_model = (
        f"The provided model name '{model_name}' is equivalent to '{EXPECTED_MODEL_NAME}'. "
        "Treat variants like 'AirPods Pro (3rd generation)' or inclusion/omission of 'Apple' brand as equivalent "
        "if they clearly refer to the same product generation."
    )
    await evaluator.verify(
        claim=claim_model,
        node=leaf_model,
        additional_instruction="This is a name matching check allowing minor formatting variants and synonyms."
    )

    # Leaf: announcement_date_is_2025_09_09 (simple equivalence check with formatting tolerance)
    leaf_ann = evaluator.add_leaf(
        id="announcement_date_is_2025_09_09",
        desc="Announcement date is September 9, 2025",
        parent=node,
        critical=True
    )
    claim_ann = (
        f"The announcement date '{ann_date}' is the same as '{EXPECTED_ANNOUNCEMENT_DATE}', "
        "allowing common date formatting variants (e.g., 'Sep 9, 2025')."
    )
    await evaluator.verify(
        claim=claim_ann,
        node=leaf_ann,
        additional_instruction="Be lenient to common date formatting differences while ensuring the same calendar date."
    )

    # Leaf: citation_model_and_announcement (Apple source must support both identity and announcement date)
    leaf_cite = evaluator.add_leaf(
        id="citation_model_and_announcement",
        desc="Provides an apple.com URL supporting the model identity and announcement date",
        parent=node,
        critical=True
    )
    claim_cite = (
        f"An official Apple webpage confirms that the earbuds model is '{EXPECTED_MODEL_NAME}' "
        f"and that it was announced on {EXPECTED_ANNOUNCEMENT_DATE}."
    )
    await evaluator.verify(
        claim=claim_cite,
        node=leaf_cite,
        sources=apple_urls,
        additional_instruction="Use only Apple (apple.com) domain pages to confirm both the exact model identity and the announcement date. "
                              "If no valid Apple URL is provided, this should fail."
    )


async def build_sensor_specs(evaluator: Evaluator, parent, sensor: Optional[SensorDetails]) -> None:
    node = evaluator.add_parallel(
        id="heart_rate_sensor_specs",
        desc="Provide and verify the constrained heart-rate sensor implementation details, with citation",
        parent=parent,
        critical=True
    )

    sensor_urls = _safe_urls(sensor.sensor_urls if sensor else None)
    apple_urls = _filter_domain(sensor_urls, "apple.com")

    # uses_custom_ppg_sensor
    leaf_ppg = evaluator.add_leaf(
        id="uses_custom_ppg_sensor",
        desc="States heart-rate sensing uses a custom photoplethysmography (PPG) sensor",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The earbuds perform heart-rate sensing using a custom photoplethysmography (PPG) sensor.",
        node=leaf_ppg,
        sources=apple_urls,
        additional_instruction="Confirm explicitly that the PPG sensor is 'custom' and is used for heart-rate sensing."
    )

    # uses_infrared_light
    leaf_ir = evaluator.add_leaf(
        id="uses_infrared_light",
        desc="States the sensor uses infrared light",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The heart-rate sensing mechanism uses infrared light.",
        node=leaf_ir,
        sources=apple_urls,
        additional_instruction="Look for explicit mention of infrared light for the sensor modality."
    )

    # pulsed_at_256_per_second
    leaf_256 = evaluator.add_leaf(
        id="pulsed_at_256_per_second",
        desc="States the infrared light is pulsed at exactly 256 times per second",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The infrared light for the heart-rate sensor is pulsed at exactly {EXPECTED_PPG_PULSE_RATE} (256 Hz).",
        node=leaf_256,
        sources=apple_urls,
        additional_instruction="The page must explicitly indicate 256 pulses per second (256 Hz) for a correct pass."
    )

    # citation_sensor_details (combined)
    leaf_sensor_cite = evaluator.add_leaf(
        id="citation_sensor_details",
        desc="Provides an apple.com URL supporting the PPG + infrared + 256 pulses/sec details",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "An official Apple webpage explicitly describes that the earbuds use a custom PPG heart-rate sensor that "
            "uses infrared light pulsed at 256 times per second."
        ),
        node=leaf_sensor_cite,
        sources=apple_urls,
        additional_instruction="Use Apple (apple.com) sources only. The single page can mention these together or across Apple pages; "
                              "as long as at least one provided Apple URL supports the combined statement unambiguously, pass."
    )


async def build_battery_life(evaluator: Evaluator, parent, battery: Optional[BatteryDetails]) -> None:
    node = evaluator.add_parallel(
        id="battery_life_with_anc",
        desc="Provide the constrained battery life with ANC enabled, with citation",
        parent=parent,
        critical=True
    )

    anc_text = (battery.anc_listening_time_single_charge or "").strip() if battery else ""
    battery_urls = _safe_urls(battery.battery_urls if battery else None)
    apple_urls = _filter_domain(battery_urls, "apple.com")

    # battery_life_is_up_to_8_hours_anc_single_charge (simple equivalence check)
    leaf_batt = evaluator.add_leaf(
        id="battery_life_is_up_to_8_hours_anc_single_charge",
        desc="States battery life is up to 8 hours listening time on a single charge with Active Noise Cancellation enabled",
        parent=node,
        critical=True
    )
    claim_batt = (
        f"The stated ANC-enabled single-charge listening time '{anc_text}' is equivalent to 'up to {EXPECTED_BATTERY_ANC_HOURS}'. "
        "Minor wording variations like 'up to eight hours' should be considered equivalent."
    )
    await evaluator.verify(
        claim=claim_batt,
        node=leaf_batt,
        additional_instruction="This is a semantic equivalence check for the stated battery life phrasing."
    )

    # citation_battery_life (Apple source)
    leaf_batt_cite = evaluator.add_leaf(
        id="citation_battery_life",
        desc="Provides an apple.com URL supporting the stated ANC battery life",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"An official Apple webpage confirms that the earbuds provide up to {EXPECTED_BATTERY_ANC_HOURS} of listening time "
              f"on a single charge with Active Noise Cancellation enabled.",
        node=leaf_batt_cite,
        sources=apple_urls,
        additional_instruction="Use Apple (apple.com) sources only."
    )


async def build_water_resistance(evaluator: Evaluator, parent, water: Optional[WaterDetails]) -> None:
    node = evaluator.add_parallel(
        id="water_resistance_rating",
        desc="Provide the constrained IP rating under IEC 60529, with citation",
        parent=parent,
        critical=True
    )

    ip_rating = (water.ip_rating_str or "").strip() if water else ""
    std_text = (water.standard_text or "").strip() if water else ""
    water_urls = _safe_urls(water.water_urls if water else None)
    apple_urls = _filter_domain(water_urls, "apple.com")

    # ip_rating_is_ip57_under_iec_60529 (simple equivalence check)
    leaf_ip = evaluator.add_leaf(
        id="ip_rating_is_ip57_under_iec_60529",
        desc="States an IP57 rating under IEC standard 60529",
        parent=node,
        critical=True
    )
    claim_ip = (
        f"The stated water resistance '{ip_rating}' with standard reference '{std_text}' "
        f"is equivalent to '{EXPECTED_IP_RATING}' under '{EXPECTED_IP_STANDARD}'. "
        "Allow minor variation in how the standard is referenced (e.g., 'IEC standard 60529')."
    )
    await evaluator.verify(
        claim=claim_ip,
        node=leaf_ip,
        additional_instruction="This is an equivalence check for the IP rating and the IEC 60529 reference."
    )

    # citation_ip_rating (Apple source)
    leaf_ip_cite = evaluator.add_leaf(
        id="citation_ip_rating",
        desc="Provides an apple.com URL supporting the IP57 rating under IEC 60529",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"An official Apple webpage confirms the earbuds have an {EXPECTED_IP_RATING} rating under IEC 60529.",
        node=leaf_ip_cite,
        sources=apple_urls,
        additional_instruction="Use Apple (apple.com) sources only."
    )


async def build_starting_price(evaluator: Evaluator, parent, price: Optional[PriceDetails]) -> None:
    node = evaluator.add_parallel(
        id="starting_price_usd",
        desc="Provide the constrained U.S. starting price, with citation",
        parent=parent,
        critical=True
    )

    price_text = (price.starting_price_usd or "").strip() if price else ""
    price_urls = _safe_urls(price.price_urls if price else None)
    apple_urls = _filter_domain(price_urls, "apple.com")

    # starting_price_is_249_usd (simple equivalence)
    leaf_price = evaluator.add_leaf(
        id="starting_price_is_249_usd",
        desc="States the starting price in the U.S. is $249",
        parent=node,
        critical=True
    )
    claim_price = (
        f"The stated starting price '{price_text}' is equivalent to '{EXPECTED_STARTING_PRICE_USD}' for the U.S. market, "
        "allowing formatting variants like '249 USD'."
    )
    await evaluator.verify(
        claim=claim_price,
        node=leaf_price,
        additional_instruction="Check semantic equivalence for the price format."
    )

    # citation_price (Apple source)
    leaf_price_cite = evaluator.add_leaf(
        id="citation_price",
        desc="Provides an apple.com URL supporting the $249 starting price",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"An official Apple webpage confirms the U.S. starting price is {EXPECTED_STARTING_PRICE_USD}.",
        node=leaf_price_cite,
        sources=apple_urls,
        additional_instruction="Use Apple (apple.com) sources only."
    )


async def build_in_store_availability(evaluator: Evaluator, parent, availability: Optional[AvailabilityDetails]) -> None:
    node = evaluator.add_parallel(
        id="in_store_availability",
        desc="Provide the constrained in-store availability start date, with citation",
        parent=parent,
        critical=True
    )

    avail_date = (availability.in_store_availability_date or "").strip() if availability else ""
    availability_urls = _safe_urls(availability.availability_urls if availability else None)
    apple_urls = _filter_domain(availability_urls, "apple.com")

    # in_store_availability_is_2025_09_19 (simple equivalence; day-of-week lenient)
    leaf_avail = evaluator.add_leaf(
        id="in_store_availability_is_2025_09_19",
        desc="States in-store availability began on Friday, September 19, 2025",
        parent=node,
        critical=True
    )
    claim_avail = (
        f"The in-store availability date '{avail_date}' is equivalent to '{EXPECTED_IN_STORE_AVAILABILITY}'. "
        "Allow presence/absence of the weekday name (e.g., 'Friday') and minor formatting variants."
    )
    await evaluator.verify(
        claim=claim_avail,
        node=leaf_avail,
        additional_instruction="Match the calendar date even if the weekday is omitted or phrased differently."
    )

    # citation_availability (Apple source)
    leaf_avail_cite = evaluator.add_leaf(
        id="citation_availability",
        desc="Provides an apple.com URL supporting the in-store availability date",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"An official Apple webpage confirms in-store availability began on {EXPECTED_IN_STORE_AVAILABILITY}.",
        node=leaf_avail_cite,
        sources=apple_urls,
        additional_instruction="Use Apple (apple.com) sources only."
    )


async def build_outage_details(evaluator: Evaluator, parent, outage: Optional[OutageDetails]) -> None:
    node = evaluator.add_parallel(
        id="carrier_outage_details",
        desc="Provide the constrained January 2026 outage details (and other constrained outage-related facts), with citations",
        parent=parent,
        critical=True
    )

    outage_urls = _safe_urls(outage.outage_urls if outage else None)

    # carrier_is_verizon
    leaf_carrier = evaluator.add_leaf(
        id="carrier_is_verizon",
        desc="Identifies the carrier as Verizon",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The major U.S. mobile carrier that had a significant outage in January 2026 was {EXPECTED_OUTAGE_CARRIER}.",
        node=leaf_carrier,
        sources=outage_urls,
        additional_instruction="Verify using credible news sources."
    )

    # outage_dates_are_2026_01_14_to_2026_01_15
    leaf_dates = evaluator.add_leaf(
        id="outage_dates_are_2026_01_14_to_2026_01_15",
        desc="States the outage dates were January 14–15, 2026",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The outage occurred on {EXPECTED_OUTAGE_DATES_TEXT} (January 14 and 15, 2026).",
        node=leaf_dates,
        sources=outage_urls,
        additional_instruction="Allow common dash variations in the date range."
    )

    # cause_is_software_issue_per_verizon
    leaf_cause = evaluator.add_leaf(
        id="cause_is_software_issue_per_verizon",
        desc="States the cause was a software issue, explicitly attributed to Verizon's report/statement",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"According to Verizon's statement, the cause of the January 2026 outage was a {EXPECTED_OUTAGE_CAUSE}.",
        node=leaf_cause,
        sources=outage_urls,
        additional_instruction="The support must explicitly attribute the cause to Verizon's own statement or report."
    )

    # downdetector_reports_about_2_3_million
    leaf_dd = evaluator.add_leaf(
        id="downdetector_reports_about_2_3_million",
        desc="States Downdetector received approximately 2.3 million outage reports during the event",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Downdetector received approximately 2.3 million outage reports during the January 2026 event.",
        node=leaf_dd,
        sources=outage_urls,
        additional_instruction="Look for explicit figures around 2.3 million reported outages on Downdetector."
    )

    # prior_disruption_2024_09_30
    leaf_prior = evaluator.add_leaf(
        id="prior_disruption_2024_09_30",
        desc="States Verizon experienced a prior disruption across several major cities on September 30, 2024",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Verizon experienced a prior disruption across several major cities on {EXPECTED_PRIOR_DISRUPTION_DATE}.",
        node=leaf_prior,
        sources=outage_urls,
        additional_instruction="A credible news source should mention the September 30, 2024 disruption event."
    )

    # citation_outage_details (combined)
    leaf_outage_cite = evaluator.add_leaf(
        id="citation_outage_details",
        desc="Provides at least one credible news URL supporting the carrier, dates, and carrier-reported cause (and URLs supporting the additional constrained outage facts if not covered by the same source)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"A credible news article confirms that the January 2026 outage was for {EXPECTED_OUTAGE_CARRIER}, "
            f"occurred on {EXPECTED_OUTAGE_DATES_TEXT}, and that Verizon attributed the cause to a {EXPECTED_OUTAGE_CAUSE}."
        ),
        node=leaf_outage_cite,
        sources=outage_urls,
        additional_instruction="At least one provided credible news URL should support the carrier, the date range, and the carrier-attributed cause."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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

    # Extract structured info from the answer
    extracted: FullExtraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=FullExtraction,
        extraction_name="extracted_answer_fields"
    )

    # Add expected info as ground truth context (for transparency in summary)
    evaluator.add_ground_truth({
        "expected": {
            "model_name": EXPECTED_MODEL_NAME,
            "announcement_date": EXPECTED_ANNOUNCEMENT_DATE,
            "sensor": {
                "modality": "infrared",
                "ppg": "custom PPG",
                "pulse_rate": EXPECTED_PPG_PULSE_RATE
            },
            "battery_anc_single_charge": f"up to {EXPECTED_BATTERY_ANC_HOURS}",
            "ip_rating": EXPECTED_IP_RATING,
            "ip_standard": EXPECTED_IP_STANDARD,
            "starting_price_usd": EXPECTED_STARTING_PRICE_USD,
            "in_store_availability": EXPECTED_IN_STORE_AVAILABILITY,
            "outage": {
                "carrier": EXPECTED_OUTAGE_CARRIER,
                "dates": EXPECTED_OUTAGE_DATES_TEXT,
                "cause": EXPECTED_OUTAGE_CAUSE,
                "downdetector_reports": EXPECTED_DDT_REPORTS,
                "prior_disruption_date": EXPECTED_PRIOR_DISRUPTION_DATE
            }
        }
    })

    # Create a critical parallel task root under the evaluator root
    task_root = evaluator.add_parallel(
        id="task_root_critical",
        desc="Identify the required Apple earbuds model and provide all constrained product + outage details, each supported by appropriate citations",
        parent=root,
        critical=True
    )

    # Build all rubric subtrees (all critical)
    await build_model_identity_and_announcement(evaluator, task_root, extracted.product)
    await build_sensor_specs(evaluator, task_root, extracted.sensor)
    await build_battery_life(evaluator, task_root, extracted.battery)
    await build_water_resistance(evaluator, task_root, extracted.water)
    await build_starting_price(evaluator, task_root, extracted.price)
    await build_in_store_availability(evaluator, task_root, extracted.availability)
    await build_outage_details(evaluator, task_root, extracted.outage)

    # Return summary
    return evaluator.get_summary()