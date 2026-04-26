import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "android_qi2_first_device"
TASK_DESCRIPTION = (
    "Identify the Android smartphone that meets ALL of the following criteria: "
    "(1) It is Qi2 certified by the Wireless Power Consortium (WPC); "
    "(2) It was the first Android smartphone to officially launch with Qi2 support; "
    "(3) It was announced on July 18, 2024 and released on July 19, 2024; "
    "(4) It is powered by a Qualcomm Snapdragon 7s Gen 2 processor; "
    "(5) It has a 4600 mAh battery capacity; "
    "(6) It supports 15W Qi2 magnetic wireless charging; "
    "(7) It supports 33W wired charging; "
    "(8) Its manufacturer is a WPC member in good standing (a prerequisite for Qi2 certification). "
    "Provide the complete device name and manufacturer name."
)


# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class PhoneExtraction(BaseModel):
    device_name: Optional[str] = None
    manufacturer_name: Optional[str] = None

    announcement_date: Optional[str] = None
    release_date: Optional[str] = None
    processor: Optional[str] = None
    battery_capacity: Optional[str] = None
    wireless_charging: Optional[str] = None  # e.g., "15W Qi2 magnetic wireless charging"
    wired_charging: Optional[str] = None     # e.g., "33W wired"
    qi2_support: Optional[str] = None        # e.g., "Qi2 support/certified"

    # URL sources (answer may provide one pool of links or per-claim links)
    urls: List[str] = Field(default_factory=list)  # general/all sources mentioned
    wpc_cert_urls: List[str] = Field(default_factory=list)
    first_android_qi2_urls: List[str] = Field(default_factory=list)
    release_urls: List[str] = Field(default_factory=list)
    processor_urls: List[str] = Field(default_factory=list)
    battery_urls: List[str] = Field(default_factory=list)
    wireless_urls: List[str] = Field(default_factory=list)
    wired_urls: List[str] = Field(default_factory=list)
    membership_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_phone() -> str:
    return """
    Extract the single Android smartphone that the answer claims meets ALL the specified criteria.
    You must extract:
    - device_name: the complete device model name (e.g., "Brand Model X")
    - manufacturer_name: the company/brand name responsible for the device (e.g., "Brand Inc." or "Nothing Technology Ltd.")
    - announcement_date: the device announcement date string as written (e.g., "July 18, 2024")
    - release_date: the device release/on-sale/launch date string as written (e.g., "July 19, 2024")
    - processor: the SoC as written (e.g., "Qualcomm Snapdragon 7s Gen 2")
    - battery_capacity: the battery capacity as written (e.g., "4600 mAh", accept typical/rated values as stated)
    - wireless_charging: the wireless charging spec as written (e.g., "15W Qi2 magnetic wireless charging")
    - wired_charging: the wired charging spec as written (e.g., "33W wired")
    - qi2_support: any explicit text in the answer that the device supports Qi2 or is Qi2 certified

    Also extract URL sources mentioned in the answer:
    - urls: all URLs mentioned in the answer (include everything, deduplicate not necessary)
    - wpc_cert_urls: URLs that specifically point to WPC Qi certification listings for this device, if any
    - first_android_qi2_urls: URLs that claim it is the first Android smartphone to officially launch with Qi2, if any
    - release_urls: URLs that support the announcement and release dates, if any
    - processor_urls: URLs that support the processor spec, if any
    - battery_urls: URLs that support the battery capacity, if any
    - wireless_urls: URLs that support the 15W Qi2 magnetic wireless charging spec, if any
    - wired_urls: URLs that support the 33W wired charging spec, if any
    - membership_urls: URLs that support that the manufacturer is a WPC member in good standing (e.g., WPC member directory), if any

    Rules:
    - Extract only what is explicitly present in the answer. Do not invent values.
    - For URLs, include all that are presented, in plain or markdown form. If protocol is missing, prepend http://
    - If multiple devices are mentioned, choose the primary device that the answer concludes meets all criteria.
    - If any field is not present in the answer, set it to null or an empty array as appropriate.
    """


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def _merge_dedup_urls(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for u in lst or []:
            if not isinstance(u, str):
                continue
            u = u.strip()
            if not u:
                continue
            # Normalize simple malformed URLs: if missing protocol, add http://
            if not (u.startswith("http://") or u.startswith("https://")):
                u = "http://" + u
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


async def _verify_with_sources_or_fail(
    evaluator: Evaluator,
    *,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: List[str],
    additional_instruction: str,
    critical: bool = True,
) -> None:
    node = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    if sources:
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=sources,
            additional_instruction=additional_instruction
        )
    else:
        # Enforce source-grounding: fail the node if no sources provided
        node.score = 0.0
        node.status = "failed"
        evaluator.add_custom_info(
            info={"node_id": node_id, "reason": "No sources provided in answer for this verification."},
            info_type="missing_sources",
            info_name=f"missing_sources_{node_id}"
        )


# --------------------------------------------------------------------------- #
# Tree construction and verification                                          #
# --------------------------------------------------------------------------- #
async def _build_and_verify_smartphone_tree(
    evaluator: Evaluator,
    root,
    extracted: PhoneExtraction,
) -> None:
    # Create the main critical node mirroring the rubric root
    smartphone_node = evaluator.add_parallel(
        id="Smartphone_Identification",
        desc="Identify the Android smartphone that was the first to launch with Qi2 certification and meets all specified technical requirements",
        parent=root,
        critical=True
    )

    # Prepare helpful strings for claims
    device = extracted.device_name or "the device"
    manufacturer = extracted.manufacturer_name or "the manufacturer"

    # Sources for each check (fallback to general urls)
    all_urls = _merge_dedup_urls(extracted.urls)

    # 1) Qi2 Certification by WPC
    qi2_cert_sources = _merge_dedup_urls(extracted.wpc_cert_urls, all_urls)
    await _verify_with_sources_or_fail(
        evaluator,
        parent=smartphone_node,
        node_id="Qi2_Certification",
        desc="The device is Qi2 certified by the Wireless Power Consortium",
        claim=f"{device} is Qi2 certified by the Wireless Power Consortium (WPC), appearing in the WPC Qi Certification database as a Qi v2.0 (Qi2) certified device.",
        sources=qi2_cert_sources,
        additional_instruction="Look for the device in the official WPC Qi Certification database or WPC announcements indicating Qi2 (Qi v2.0) certification."
    )

    # 2) First Android smartphone to launch with Qi2 support
    first_android_sources = _merge_dedup_urls(extracted.first_android_qi2_urls, all_urls)
    await _verify_with_sources_or_fail(
        evaluator,
        parent=smartphone_node,
        node_id="First_Android_Qi2",
        desc="The device is the first Android smartphone to officially launch with Qi2 support",
        claim=f"{device} was the first Android smartphone to officially launch with Qi2 support.",
        sources=first_android_sources,
        additional_instruction="Check reputable sources (press releases, WPC, major tech publications) that explicitly state it is the first Android phone to launch with Qi2. Distinguish 'launch with Qi2' from 'later received Qi2 via update'."
    )

    # 3) Announced July 18, 2024; Released July 19, 2024
    release_sources = _merge_dedup_urls(extracted.release_urls, all_urls)
    await _verify_with_sources_or_fail(
        evaluator,
        parent=smartphone_node,
        node_id="Release_Date",
        desc="The device was announced on July 18, 2024 and released on July 19, 2024",
        claim=f"{device} was announced on July 18, 2024 and released (went on sale/available) on July 19, 2024.",
        sources=release_sources,
        additional_instruction="Confirm both dates. Accept phrasing like 'announcement' for July 18, 2024 and 'release/on sale/availability' for July 19, 2024. Minor timezone phrasing is acceptable if the dates match."
    )

    # 4) Processor: Snapdragon 7s Gen 2
    proc_sources = _merge_dedup_urls(extracted.processor_urls, all_urls)
    await _verify_with_sources_or_fail(
        evaluator,
        parent=smartphone_node,
        node_id="Processor_Specification",
        desc="The device is powered by Qualcomm Snapdragon 7s Gen 2 processor",
        claim=f"{device} is powered by the Qualcomm Snapdragon 7s Gen 2 processor.",
        sources=proc_sources,
        additional_instruction="Verify device specifications on official product pages or trusted spec databases explicitly naming 'Qualcomm Snapdragon 7s Gen 2'."
    )

    # 5) Battery capacity: 4600 mAh
    batt_sources = _merge_dedup_urls(extracted.battery_urls, all_urls)
    await _verify_with_sources_or_fail(
        evaluator,
        parent=smartphone_node,
        node_id="Battery_Capacity",
        desc="The device has a 4600 mAh battery capacity",
        claim=f"{device} has a 4600 mAh battery capacity.",
        sources=batt_sources,
        additional_instruction="Accept typical vs. rated capacity if the site states 4600 mAh. Minor formatting differences (e.g., '4,600 mAh') should be treated as equivalent."
    )

    # 6) Wireless charging: 15W Qi2 magnetic
    wireless_sources = _merge_dedup_urls(extracted.wireless_urls, all_urls)
    await _verify_with_sources_or_fail(
        evaluator,
        parent=smartphone_node,
        node_id="Wireless_Charging_Power",
        desc="The device supports 15W Qi2 magnetic wireless charging",
        claim=f"{device} supports 15W Qi2 magnetic wireless charging.",
        sources=wireless_sources,
        additional_instruction="Look for 'Qi2', 'Qi v2.0', 'magnetic', and '15W' on spec sheets or official pages. Accept 'up to 15W' phrasing."
    )

    # 7) Wired charging: 33W
    wired_sources = _merge_dedup_urls(extracted.wired_urls, all_urls)
    await _verify_with_sources_or_fail(
        evaluator,
        parent=smartphone_node,
        node_id="Wired_Charging_Power",
        desc="The device supports 33W wired charging",
        claim=f"{device} supports 33W wired charging.",
        sources=wired_sources,
        additional_instruction="Verify spec pages for '33W wired' or equivalent terms like '33W fast charging'."
    )

    # 8) Manufacturer is a WPC member in good standing
    membership_sources = _merge_dedup_urls(extracted.membership_urls, extracted.wpc_cert_urls, all_urls)
    await _verify_with_sources_or_fail(
        evaluator,
        parent=smartphone_node,
        node_id="WPC_Membership_Verification",
        desc="The device manufacturer is a WPC member in good standing, which is a prerequisite for Qi2 certification",
        claim=f"{manufacturer} is a Wireless Power Consortium (WPC) member in good standing.",
        sources=membership_sources,
        additional_instruction="Check the WPC Member Directory or official WPC materials. Allow reasonable name variants or parent-company names if clearly referring to the same entity."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # The rubric root is parallel
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_phone(),
        template_class=PhoneExtraction,
        extraction_name="phone_extraction"
    )

    # Build tree and verify
    await _build_and_verify_smartphone_tree(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()