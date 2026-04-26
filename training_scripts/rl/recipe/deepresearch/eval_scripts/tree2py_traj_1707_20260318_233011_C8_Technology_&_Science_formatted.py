import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wireless_audio_products_collection"
TASK_DESCRIPTION = """
Find 4 different wireless audio products (headphones or earbuds) currently available for purchase, with each product fulfilling a specific use case and meeting all the technical requirements listed below. Provide the product name, manufacturer, and a link to the official product page or authorized retailer page for each.

Product 1: Premium Travel Over-Ear Headphone
- Battery life of at least 30 hours on a single charge with Active Noise Cancellation (ANC) enabled
- Must have Active Noise Cancellation capability
- Must be an over-ear (circumaural) closed-back headphone design

Product 2: Workout True Wireless Earbuds
- Water/sweat resistance rating of IPX5 or higher
- Must have ear hooks, wing tips, or other secure-fit design features specifically for sports/fitness
- Single-charge battery life of at least 6 hours

Product 3: Audiophile Hi-Res Wireless Earbuds
- Must support LDAC or aptX Adaptive codec for hi-res wireless audio
- Driver size of 10mm or larger
- Frequency response upper limit of at least 40kHz (hi-res audio range)

Product 4: Office Multipoint Wireless Earbuds
- Multipoint Bluetooth connectivity supporting simultaneous connection to 2 or more devices
- Transparency, Ambient Sound, or Hear-Through mode for environmental awareness
- Multiple microphones (3 or more) or dedicated call quality enhancement features
- Single-charge battery life of at least 8 hours

For each product, provide:
1. Product name and manufacturer
2. A direct link to the official product page from the manufacturer's website OR a product page from a major authorized retailer (such as Amazon, Best Buy, B&H Photo, etc.)
3. Brief confirmation that each requirement is met (with specific values where applicable, such as "battery life: 32 hours with ANC on")
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PremiumTravelOverEar(BaseModel):
    name: Optional[str] = None
    manufacturer: Optional[str] = None
    url: Optional[str] = None
    battery_life_with_anc: Optional[str] = None  # e.g., "32 hours", "40h (ANC on)"
    anc_feature: Optional[str] = None  # e.g., "Active Noise Cancellation"
    form_factor: Optional[str] = None  # e.g., "Over-ear, closed-back"


class WorkoutEarbuds(BaseModel):
    name: Optional[str] = None
    manufacturer: Optional[str] = None
    url: Optional[str] = None
    water_resistance: Optional[str] = None  # e.g., "IPX5", "IP57"
    secure_fit_features: Optional[str] = None  # e.g., "ear hooks", "wing tips"
    battery_life: Optional[str] = None  # single-charge, e.g., "8 hours"


class AudiophileHiResEarbuds(BaseModel):
    name: Optional[str] = None
    manufacturer: Optional[str] = None
    url: Optional[str] = None
    hires_codec: Optional[str] = None  # e.g., "LDAC" or "aptX Adaptive"
    driver_size: Optional[str] = None  # e.g., "11 mm"
    freq_response_upper: Optional[str] = None  # e.g., "40 kHz", "48kHz"


class OfficeMultipointEarbuds(BaseModel):
    name: Optional[str] = None
    manufacturer: Optional[str] = None
    url: Optional[str] = None
    multipoint: Optional[str] = None  # e.g., "Multipoint (2 devices)"
    transparency_mode: Optional[str] = None  # e.g., "Transparency", "Ambient"
    call_quality_features: Optional[str] = None  # e.g., "3 mics", "AI noise reduction"
    battery_life: Optional[str] = None  # single-charge, e.g., "8 hours"


class ProductsExtraction(BaseModel):
    premium_travel_overear: Optional[PremiumTravelOverEar] = None
    workout_earbuds: Optional[WorkoutEarbuds] = None
    audiophile_hires_earbuds: Optional[AudiophileHiResEarbuds] = None
    office_multipoint_earbuds: Optional[OfficeMultipointEarbuds] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_products() -> str:
    return """
    Extract structured information for exactly four products mentioned in the answer, one for each required use case.

    For each product, extract the following fields from the answer text exactly as written (do not infer or invent):
    1) premium_travel_overear:
       - name: product name (string)
       - manufacturer: brand/manufacturer (string)
       - url: direct product page URL (manufacturer site or major authorized retailer)
       - battery_life_with_anc: the single-charge battery life with ANC enabled as stated (string, e.g., "32 hours with ANC on")
       - anc_feature: a phrase indicating ANC is present (string, e.g., "Active Noise Cancellation")
       - form_factor: wording indicating "over-ear closed-back" (string, e.g., "over-ear, closed-back")
    2) workout_earbuds:
       - name, manufacturer, url
       - water_resistance: IP rating as stated (string, e.g., "IPX5", "IP57")
       - secure_fit_features: the sport-fit design features stated (string, e.g., "ear hooks", "wing tips")
       - battery_life: the single-charge battery life as stated (string, e.g., "8 hours")
    3) audiophile_hires_earbuds:
       - name, manufacturer, url
       - hires_codec: the hi-res codec claimed (must be "LDAC" or "aptX Adaptive" if present)
       - driver_size: driver size as stated (string, e.g., "10 mm", "11mm")
       - freq_response_upper: the stated upper frequency response limit (string, e.g., "40 kHz")
    4) office_multipoint_earbuds:
       - name, manufacturer, url
       - multipoint: wording indicating simultaneous connection to 2+ devices (string)
       - transparency_mode: wording indicating transparency/ambient/hear-through mode (string)
       - call_quality_features: wording indicating multiple mics (e.g., "3 mics") or dedicated call-quality features (e.g., ENC, beamforming, AI noise reduction)
       - battery_life: the single-charge battery life as stated (string, e.g., "8 hours")

    Rules:
    - Extract only what is explicitly present in the answer text. If a field is not provided, set it to null.
    - Do not infer numeric thresholds; just capture the exact phrase/value in the answer.
    - Extract URLs exactly as shown (markdown links should be resolved to their raw URLs).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
AUTHORIZED_RETAILER_HINT = (
    "Major authorized retailers include Amazon, Best Buy, B&H Photo, Crutchfield, Adorama, Walmart, Target, Newegg, and similar. "
    "Manufacturer official domains include brand sites like sony.com, bose.com, sennheiser-hearing.com, apple.com, samsung.com, jbl.com, "
    "anker.com/soundcore.com, technics.com, shure.com, jabra.com, earfun.com, cambridgeaudio.com, etc."
)


def norm_name(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower()
    s = s.replace("™", "").replace("®", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def product_tuple_list(extracted: ProductsExtraction) -> List[tuple]:
    return [
        ("PTH", extracted.premium_travel_overear),
        ("WE", extracted.workout_earbuds),
        ("AHE", extracted.audiophile_hires_earbuds),
        ("OME", extracted.office_multipoint_earbuds),
    ]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_premium_travel_headphone(evaluator: Evaluator, parent_node, data: Optional[PremiumTravelOverEar]) -> None:
    node = evaluator.add_parallel(
        id="Premium_Travel_Headphone",
        desc="An over-ear wireless headphone optimized for travel with premium noise cancellation and extended battery life",
        parent=parent_node,
        critical=False
    )

    # Leaf: Product URL validity (critical)
    url_leaf = evaluator.add_leaf(
        id="PTH_Product_URL",
        desc="Valid product page URL from manufacturer or authorized retailer",
        parent=node,
        critical=True
    )
    url_claim = f"This page is the official manufacturer product page or a major authorized retailer product page for the product '{(data.name if data else '')}' by '{(data.manufacturer if data else '')}'."
    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=(data.url if data else None),
        additional_instruction=(
            "Verify the page is clearly a product detail page for the exact model (title/branding/specs match). "
            f"It should be on the manufacturer's official site OR a major authorized retailer. {AUTHORIZED_RETAILER_HINT} "
            "Reject aggregator, review blogs, random marketplaces with no official authorization, or generic category pages."
        ),
    )

    # Leaf: Battery life with ANC >= 30h (critical)
    battery_leaf = evaluator.add_leaf(
        id="PTH_Battery_Life",
        desc="Battery life of at least 30 hours on a single charge with Active Noise Cancellation enabled",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This product provides at least 30 hours of battery life on a single charge with ANC enabled (ANC on).",
        node=battery_leaf,
        sources=(data.url if data else None),
        additional_instruction=(
            "Look for explicit battery life with ANC enabled. If only ANC-off is stated and ANC-on is lower than 30 hours, this should fail. "
            "Accept phrasing like 'with noise cancelling on/active' or similar. If multiple modes are listed, consider the ANC-on figure."
        ),
        extra_prerequisites=[url_leaf],
    )

    # Leaf: ANC capability present (critical)
    anc_leaf = evaluator.add_leaf(
        id="PTH_ANC_Capability",
        desc="Active Noise Cancellation (ANC) feature present and functional",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This product supports Active Noise Cancellation (ANC).",
        node=anc_leaf,
        sources=(data.url if data else None),
        additional_instruction="Confirm explicit mention of Active Noise Cancellation (ANC). Variants like 'noise cancelling' are acceptable.",
        extra_prerequisites=[url_leaf],
    )

    # Leaf: Over-ear closed-back design (critical)
    form_leaf = evaluator.add_leaf(
        id="PTH_Form_Factor",
        desc="Over-ear (circumaural) closed-back headphone design",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This product is an over-ear (circumaural), closed-back headphone.",
        node=form_leaf,
        sources=(data.url if data else None),
        additional_instruction="Accept synonyms like 'over-ear'/'around-ear' and 'closed-back'/'closed design'. Reject on-ear, open-back, or in-ear.",
        extra_prerequisites=[url_leaf],
    )


async def verify_workout_earbuds(evaluator: Evaluator, parent_node, data: Optional[WorkoutEarbuds]) -> None:
    node = evaluator.add_parallel(
        id="Workout_Earbuds",
        desc="True wireless earbuds designed for sports and fitness activities with sweat resistance",
        parent=parent_node,
        critical=False
    )

    url_leaf = evaluator.add_leaf(
        id="WE_Product_URL",
        desc="Valid product page URL from manufacturer or authorized retailer",
        parent=node,
        critical=True
    )
    url_claim = f"This page is the official manufacturer product page or a major authorized retailer product page for the product '{(data.name if data else '')}' by '{(data.manufacturer if data else '')}'."
    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=(data.url if data else None),
        additional_instruction=(
            "Confirm it's a genuine product detail page (manufacturer site or major authorized retailer). "
            f"{AUTHORIZED_RETAILER_HINT}"
        ),
    )

    water_leaf = evaluator.add_leaf(
        id="WE_Water_Resistance",
        desc="Water/sweat resistance rating of IPX5 or higher",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This product has a water/sweat resistance rating of IPX5 or higher.",
        node=water_leaf,
        sources=(data.url if data else None),
        additional_instruction="Accept IPX5, IP55, IPX6, IPX7, IPX8, etc. Reject IPX4 or below for this requirement.",
        extra_prerequisites=[url_leaf],
    )

    fit_leaf = evaluator.add_leaf(
        id="WE_Secure_Fit",
        desc="Ear hooks, wing tips, or other secure-fit design features for sports",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This product includes secure-fit features for sports, such as ear hooks, wing tips, ear fins, or stabilizers.",
        node=fit_leaf,
        sources=(data.url if data else None),
        additional_instruction="Look for explicit mentions of ear hooks, wing tips, stabilizing fins, or similar sport-focused retention features.",
        extra_prerequisites=[url_leaf],
    )

    battery_leaf = evaluator.add_leaf(
        id="WE_Battery_Life",
        desc="Single-charge battery life of at least 6 hours",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This product provides at least 6 hours of battery life on a single charge (earbuds only).",
        node=battery_leaf,
        sources=(data.url if data else None),
        additional_instruction="Check the single-charge (earbuds) figure. Do not count total with case. If multiple modes, the normal mode must be >= 6 hours.",
        extra_prerequisites=[url_leaf],
    )


async def verify_audiophile_hires_earbuds(evaluator: Evaluator, parent_node, data: Optional[AudiophileHiResEarbuds]) -> None:
    node = evaluator.add_parallel(
        id="Audiophile_HiRes_Earbuds",
        desc="Premium wireless earbuds with hi-res audio codec support for superior sound quality",
        parent=parent_node,
        critical=False
    )

    url_leaf = evaluator.add_leaf(
        id="AHE_Product_URL",
        desc="Valid product page URL from manufacturer or authorized retailer",
        parent=node,
        critical=True
    )
    url_claim = f"This page is the official manufacturer product page or a major authorized retailer product page for the product '{(data.name if data else '')}' by '{(data.manufacturer if data else '')}'."
    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=(data.url if data else None),
        additional_instruction=(
            "Confirm it's a genuine product detail page (manufacturer site or major authorized retailer). "
            f"{AUTHORIZED_RETAILER_HINT}"
        ),
    )

    codec_leaf = evaluator.add_leaf(
        id="AHE_Codec_Support",
        desc="Support for LDAC or aptX Adaptive codec for hi-res wireless audio",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This product supports LDAC or aptX Adaptive codec for hi-res wireless audio.",
        node=codec_leaf,
        sources=(data.url if data else None),
        additional_instruction="Pass only if LDAC or aptX Adaptive is explicitly listed. Other codecs alone (e.g., AAC, SBC, aptX, aptX HD) are insufficient.",
        extra_prerequisites=[url_leaf],
    )

    driver_leaf = evaluator.add_leaf(
        id="AHE_Driver_Size",
        desc="Driver size of 10mm or larger",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This product uses a driver that is at least 10 mm in size.",
        node=driver_leaf,
        sources=(data.url if data else None),
        additional_instruction="Look for driver size in mm (e.g., 10 mm, 11 mm). If multiple drivers, at least one primary dynamic driver should be >= 10 mm.",
        extra_prerequisites=[url_leaf],
    )

    freq_leaf = evaluator.add_leaf(
        id="AHE_Frequency_Response",
        desc="Frequency response upper limit of at least 40kHz (hi-res audio range)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This product's stated frequency response upper limit is at least 40 kHz.",
        node=freq_leaf,
        sources=(data.url if data else None),
        additional_instruction="Accept 40 kHz or higher (e.g., 40kHz, 42 kHz). If only 20 kHz is listed, fail.",
        extra_prerequisites=[url_leaf],
    )


async def verify_office_multipoint_earbuds(evaluator: Evaluator, parent_node, data: Optional[OfficeMultipointEarbuds]) -> None:
    node = evaluator.add_parallel(
        id="Office_Multipoint_Earbuds",
        desc="Wireless earbuds with multipoint connectivity for seamless switching between work devices",
        parent=parent_node,
        critical=False
    )

    url_leaf = evaluator.add_leaf(
        id="OME_Product_URL",
        desc="Valid product page URL from manufacturer or authorized retailer",
        parent=node,
        critical=True
    )
    url_claim = f"This page is the official manufacturer product page or a major authorized retailer product page for the product '{(data.name if data else '')}' by '{(data.manufacturer if data else '')}'."
    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=(data.url if data else None),
        additional_instruction=(
            "Confirm it's a genuine product detail page (manufacturer site or major authorized retailer). "
            f"{AUTHORIZED_RETAILER_HINT}"
        ),
    )

    multipoint_leaf = evaluator.add_leaf(
        id="OME_Multipoint",
        desc="Multipoint Bluetooth connectivity supporting simultaneous connection to 2 or more devices",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This product supports multipoint Bluetooth connectivity (simultaneous connection to two or more devices).",
        node=multipoint_leaf,
        sources=(data.url if data else None),
        additional_instruction="Look for 'multipoint', 'connect two devices', or similar explicit wording. Fast switching without true simultaneous connections should not count.",
        extra_prerequisites=[url_leaf],
    )

    transparency_leaf = evaluator.add_leaf(
        id="OME_Transparency_Mode",
        desc="Transparency, Ambient Sound, or Hear-Through mode for environmental awareness",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This product supports a mode for environmental awareness such as Transparency, Ambient Sound, or Hear-Through.",
        node=transparency_leaf,
        sources=(data.url if data else None),
        additional_instruction="Accept terms like 'Transparency', 'Ambient Sound', 'Hear-Through', or equivalent.",
        extra_prerequisites=[url_leaf],
    )

    callq_leaf = evaluator.add_leaf(
        id="OME_Call_Quality",
        desc="Multiple microphones (3+) or dedicated call quality enhancement features",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This product provides either three or more microphones OR dedicated call-quality enhancement features (e.g., beamforming mics, ENC, AI noise reduction).",
        node=callq_leaf,
        sources=(data.url if data else None),
        additional_instruction="Pass if specs mention '3 mics' (or more) OR named call-quality features beyond basic mic (e.g., beamforming, ENC, AI NR, Clear Voice).",
        extra_prerequisites=[url_leaf],
    )

    battery_leaf = evaluator.add_leaf(
        id="OME_Battery_Life",
        desc="Single-charge battery life of at least 8 hours",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This product provides at least 8 hours of battery life on a single charge (earbuds only).",
        node=battery_leaf,
        sources=(data.url if data else None),
        additional_instruction="Check the single-charge (earbuds) figure; do not count total with case.",
        extra_prerequisites=[url_leaf],
    )


async def verify_all_products_wireless(evaluator: Evaluator, parent_node, extracted: ProductsExtraction, url_prereq_map: Dict[str, Any]) -> None:
    """
    Build a critical parallel node that verifies each product is wireless (Bluetooth).
    url_prereq_map maps product key to its URL leaf node to enforce preconditions.
    """
    wireless_node = evaluator.add_parallel(
        id="All_Products_Wireless",
        desc="All products must be wireless (Bluetooth) headphones or earbuds",
        parent=parent_node,
        critical=True
    )

    for key, pdata in [
        ("PTH", extracted.premium_travel_overear),
        ("WE", extracted.workout_earbuds),
        ("AHE", extracted.audiophile_hires_earbuds),
        ("OME", extracted.office_multipoint_earbuds),
    ]:
        leaf = evaluator.add_leaf(
            id=f"All_Products_Wireless_{key}",
            desc=f"{key} product is wireless (Bluetooth)",
            parent=wireless_node,
            critical=True
        )
        url_leaf = url_prereq_map.get(key)
        await evaluator.verify(
            claim=f"The product '{(pdata.name if pdata else '')}' is a wireless Bluetooth headphone/earbuds.",
            node=leaf,
            sources=(pdata.url if pdata else None),
            additional_instruction="Look for 'Bluetooth', 'wireless', or 'true wireless' on the product page. Reject wired-only models.",
            extra_prerequisites=([url_leaf] if url_leaf else None),
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
    Evaluate an answer for the Wireless Audio Products Collection task.
    """
    # Initialize evaluator
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

    # Extraction
    extracted: ProductsExtraction = await evaluator.extract(
        prompt=prompt_extract_products(),
        template_class=ProductsExtraction,
        extraction_name="products_extraction"
    )

    # Build main collection node under root (parallel)
    collection_node = evaluator.add_parallel(
        id="Wireless_Audio_Products_Collection",
        desc="Find 4 different wireless audio products (headphones or earbuds) that meet specific technical requirements for different use cases",
        parent=root,
        critical=False
    )

    # Per-product verifications
    # Keep a map of URL leaves to use as preconditions for cross checks
    url_prereq_map: Dict[str, Any] = {}

    # Premium Travel Over-Ear
    await verify_premium_travel_headphone(evaluator, collection_node, extracted.premium_travel_overear)
    url_prereq_map["PTH"] = evaluator.find_node("PTH_Product_URL")

    # Workout Earbuds
    await verify_workout_earbuds(evaluator, collection_node, extracted.workout_earbuds)
    url_prereq_map["WE"] = evaluator.find_node("WE_Product_URL")

    # Audiophile Hi-Res Earbuds
    await verify_audiophile_hires_earbuds(evaluator, collection_node, extracted.audiophile_hires_earbuds)
    url_prereq_map["AHE"] = evaluator.find_node("AHE_Product_URL")

    # Office Multipoint Earbuds
    await verify_office_multipoint_earbuds(evaluator, collection_node, extracted.office_multipoint_earbuds)
    url_prereq_map["OME"] = evaluator.find_node("OME_Product_URL")

    # Critical check: All products different models
    names = [
        norm_name(extracted.premium_travel_overear.name if extracted.premium_travel_overear else None),
        norm_name(extracted.workout_earbuds.name if extracted.workout_earbuds else None),
        norm_name(extracted.audiophile_hires_earbuds.name if extracted.audiophile_hires_earbuds else None),
        norm_name(extracted.office_multipoint_earbuds.name if extracted.office_multipoint_earbuds else None),
    ]
    unique_nonempty = set([n for n in names if n])
    all_different = (len([n for n in names if n]) == 4) and (len(unique_nonempty) == 4)

    evaluator.add_custom_node(
        result=all_different,
        id="All_Products_Different",
        desc="All 4 products must be different models from each other (no duplicate products)",
        parent=collection_node,
        critical=True
    )

    # Critical check: All products wireless — verify per product with sources
    await verify_all_products_wireless(evaluator, collection_node, extracted, url_prereq_map)

    # Return final structured evaluation summary
    return evaluator.get_summary()