import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "smartphones_6000mah_silicon_carbon_cband_us_2024_2025"
TASK_DESCRIPTION = """Identify at least three smartphones that meet ALL of the following technical and market requirements:

1. Battery capacity of at least 6000mAh
2. Silicon-carbon battery technology
3. Wireless charging support at a minimum of 15W power output
4. Support for 5G C-band (n77 band) for mid-band 5G connectivity
5. Official availability for purchase in the United States market
6. Compatibility with at least one major US carrier (AT&T, T-Mobile, or Verizon)
7. Released or made available during 2024 or 2025

For each smartphone you identify, provide:
- The exact model name and manufacturer
- Confirmation of all seven requirements listed above
- A reference URL that verifies the phone's specifications
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class PhoneEntry(BaseModel):
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    battery_capacity: Optional[str] = None
    battery_technology: Optional[str] = None
    wireless_charging_power: Optional[str] = None
    bands: List[str] = Field(default_factory=list)
    us_availability: Optional[str] = None
    carriers: List[str] = Field(default_factory=list)
    release_year: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class PhonesExtraction(BaseModel):
    phones: List[PhoneEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_phones() -> str:
    return """
    Extract up to all smartphones listed in the answer that the author claims meet the specified requirements.
    For each smartphone, extract the following fields exactly as stated in the answer text:
    - manufacturer: The brand or manufacturer name (e.g., "Samsung", "Motorola"). If missing, return null.
    - model: The exact model name (e.g., "Galaxy M55", "Edge Plus 2024"). If missing, return null.
    - battery_capacity: The stated battery capacity text (e.g., "6000 mAh", "6,000mAh"). If not stated, return null.
    - battery_technology: The stated battery technology text (e.g., "silicon-carbon", "Si-C anode"). If not stated, return null.
    - wireless_charging_power: The stated wireless charging power text (e.g., "15W", "Qi2 15W"). If not stated, return null.
    - bands: A list of all 5G band identifiers mentioned for this model (e.g., ["n77","n78"]). If not listed, return an empty list.
    - us_availability: The provided statement or phrase indicating US availability (e.g., "available in the US", "sold on official US store"). If not stated, return null.
    - carriers: A list of carriers stated to be compatible (e.g., ["Verizon","AT&T"]). If not listed, return an empty list.
    - release_year: The year of release or US availability mentioned (e.g., "2024", "2025"). If not stated, return null.
    - reference_urls: A list of all URLs in the answer that are intended to support/verify the phone's specifications or availability. Include URLs in plain form or markdown links. If none, return an empty list.

    Return a JSON object with a `phones` array of PhoneEntry objects in the order they appear in the answer.
    Do not invent any information not explicitly present in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _fmt_phone_name(phone: PhoneEntry, fallback: str) -> str:
    if phone.manufacturer and phone.model:
        return f"{phone.manufacturer} {phone.model}"
    if phone.model:
        return phone.model
    if phone.manufacturer:
        return f"{phone.manufacturer} (unspecified model)"
    return fallback


def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    clean: List[str] = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            # best effort: prepend http:// if missing, as per framework guidance
            u = "http://" + u
        if u not in seen:
            seen.add(u)
            clean.append(u)
    return clean


# --------------------------------------------------------------------------- #
# Verification for a single phone                                             #
# --------------------------------------------------------------------------- #
async def verify_single_phone(
    evaluator: Evaluator,
    parent_node,
    phone: PhoneEntry,
    index: int,
) -> None:
    """
    Build and verify the rubric subtree for one smartphone.
    The phone node aggregates 8 critical checks in parallel. We first validate reference URLs,
    then make all other checks depend on the reference node to enforce source-grounding.
    """
    phone_node = evaluator.add_parallel(
        id=f"phone_{index}",
        desc=(
            "First qualifying smartphone with all required specifications" if index == 1 else
            "Second qualifying smartphone with all required specifications" if index == 2 else
            "Third qualifying smartphone with all required specifications"
        ),
        parent=parent_node,
        critical=False,
    )

    model_disp = _fmt_phone_name(phone, fallback=f"Phone #{index}")
    sources = _dedupe_urls(phone.reference_urls or [])

    # 1) Reference URL validity/specs page
    reference_leaf = evaluator.add_leaf(
        id=f"phone_{index}_reference",
        desc="Provide valid reference URL confirming the phone's specifications",
        parent=phone_node,
        critical=True,
    )

    if len(sources) == 0:
        # Explicitly fail when no sources are provided to avoid non-evidence simple verification
        reference_leaf.score = 0.0
        reference_leaf.status = "failed"
    else:
        ref_claim = (
            f"This webpage explicitly lists the technical specifications for the {model_disp} smartphone "
            f"(e.g., official manufacturer page, official press release, major retailer/store page, or a well-known spec database)."
        )
        await evaluator.verify(
            claim=ref_claim,
            node=reference_leaf,
            sources=sources,
            additional_instruction=(
                "Accept pages that clearly present technical specifications such as battery capacity, wireless charging wattage, and network bands. "
                "Ignore general news/blog posts if they do not list specs. The page does not have to list all specs but must be a bona fide specs page."
            ),
        )

    # Helper: all other checks should rely on the reference being valid
    prereq = [reference_leaf]

    # 2) Battery capacity >= 6000 mAh
    cap_leaf = evaluator.add_leaf(
        id=f"phone_{index}_battery_capacity",
        desc="Battery capacity is at least 6000mAh",
        parent=phone_node,
        critical=True,
    )
    cap_claim = f"The {model_disp} has a battery capacity of at least 6000 mAh."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=sources,
        additional_instruction=(
            "Look for battery capacity on the specs page. Accept typical/rated values expressed as 6000 mAh or higher, "
            "including formats like '6,000 mAh' or '≥6000 mAh'."
        ),
        extra_prerequisites=prereq,
    )

    # 3) Silicon-carbon battery technology
    tech_leaf = evaluator.add_leaf(
        id=f"phone_{index}_battery_technology",
        desc="Uses silicon-carbon battery technology",
        parent=phone_node,
        critical=True,
    )
    tech_claim = (
        f"The {model_disp} uses silicon-carbon battery technology (e.g., silicon-carbon anode, Si–C, or silicon-carbon composite)."
    )
    await evaluator.verify(
        claim=tech_claim,
        node=tech_leaf,
        sources=sources,
        additional_instruction=(
            "Accept synonyms/phrases: 'silicon‑carbon', 'Si‑C', 'silicon carbon anode', 'silicon‑carbon composite'. "
            "If the page mentions only standard graphite or other chemistries without silicon‑carbon, this should fail."
        ),
        extra_prerequisites=prereq,
    )

    # 4) Wireless charging >= 15 W
    wlc_leaf = evaluator.add_leaf(
        id=f"phone_{index}_wireless_charging",
        desc="Supports wireless charging at minimum 15W power output",
        parent=phone_node,
        critical=True,
    )
    wlc_claim = f"The {model_disp} supports wireless charging of at least 15 W."
    await evaluator.verify(
        claim=wlc_claim,
        node=wlc_leaf,
        sources=sources,
        additional_instruction=(
            "Check the specs for wireless charging wattage. Accept terms like 'Qi', 'Qi2', 'MagSafe' if the power is 15W or higher. "
            "If only lower wattage (e.g., 10W) is listed, this should fail."
        ),
        extra_prerequisites=prereq,
    )

    # 5) 5G C-band (n77)
    cband_leaf = evaluator.add_leaf(
        id=f"phone_{index}_5g_cband",
        desc="Supports 5G C-band (n77 band) for mid-band 5G connectivity",
        parent=phone_node,
        critical=True,
    )
    cband_claim = f"The {model_disp} supports 5G band n77 (C‑band)."
    await evaluator.verify(
        claim=cband_claim,
        node=cband_leaf,
        sources=sources,
        additional_instruction=(
            "Look for 'n77' in the 5G bands list. Also accept explicit references to 'C‑band' or 3.7–3.98 GHz spectrum associated with n77. "
            "Do not confuse with n78 unless n77 is also listed."
        ),
        extra_prerequisites=prereq,
    )

    # 6) Official US availability
    usa_leaf = evaluator.add_leaf(
        id=f"phone_{index}_us_availability",
        desc="Officially available for purchase in the United States market",
        parent=phone_node,
        critical=True,
    )
    usa_claim = f"The {model_disp} is officially available for purchase in the United States market."
    await evaluator.verify(
        claim=usa_claim,
        node=usa_leaf,
        sources=sources,
        additional_instruction=(
            "Accept evidence such as a US manufacturer site listing, US press release, carrier store page, or major authorized US retailer (e.g., Best Buy, Amazon US) "
            "offering the device officially. 'International import only' without official US availability should fail."
        ),
        extra_prerequisites=prereq,
    )

    # 7) Compatibility with at least one major US carrier
    carrier_leaf = evaluator.add_leaf(
        id=f"phone_{index}_carrier_compatibility",
        desc="Compatible with at least one major US carrier (AT&T, T-Mobile, or Verizon)",
        parent=phone_node,
        critical=True,
    )
    carrier_claim = (
        f"The {model_disp} is compatible with at least one major US carrier: AT&T, T‑Mobile, or Verizon."
    )
    carriers_list_txt = ", ".join(phone.carriers) if phone.carriers else "none specified in the answer"
    await evaluator.verify(
        claim=carrier_claim,
        node=carrier_leaf,
        sources=sources,
        additional_instruction=(
            "Accept explicit statements such as 'works with Verizon', 'compatible with AT&T', 'T‑Mobile certified', or listing by that carrier's store. "
            "Indirect inferences solely from band listings are weaker; prefer explicit carrier compatibility statements. "
            f"In the answer, the claimed compatible carriers were: {carriers_list_txt}. Prefer verifying those if present."
        ),
        extra_prerequisites=prereq,
    )

    # 8) Release/availability during 2024 or 2025
    rel_leaf = evaluator.add_leaf(
        id=f"phone_{index}_release_timeline",
        desc="Released or made available during 2024 or 2025",
        parent=phone_node,
        critical=True,
    )
    rel_claim = f"The {model_disp} was released or made available during 2024 or 2025."
    await evaluator.verify(
        claim=rel_claim,
        node=rel_leaf,
        sources=sources,
        additional_instruction=(
            "Use the announcement, launch, or first-availability date shown on the source(s). "
            "US market availability date within 2024/2025 is acceptable even if global announcement was earlier."
        ),
        extra_prerequisites=prereq,
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
    Evaluate an agent's answer for the smartphone requirements task.
    """
    # Initialize evaluator/root
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

    # Extract structured information
    extracted: PhonesExtraction = await evaluator.extract(
        prompt=prompt_extract_phones(),
        template_class=PhonesExtraction,
        extraction_name="phones_extraction",
    )

    # Select first three phones; pad if fewer
    phones = list(extracted.phones[:3])
    while len(phones) < 3:
        phones.append(PhoneEntry())

    # Build three parallel phone subtrees
    await verify_single_phone(evaluator, root, phones[0], index=1)
    await verify_single_phone(evaluator, root, phones[1], index=2)
    await verify_single_phone(evaluator, root, phones[2], index=3)

    # Return evaluation summary
    return evaluator.get_summary()