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
TASK_ID = "3gpp_min_release_for_factory_5g_2024"
TASK_DESCRIPTION = (
    "A telecommunications company is deploying a private 5G Standalone (SA) network for an advanced "
    "manufacturing facility in the United States in 2024. The deployment must meet the following technical "
    "requirements: Support indoor positioning accuracy of 20 centimeters or better for autonomous guided vehicles (AGVs); "
    "Provide Ultra-Reliable Low-Latency Communication (URLLC) with end-to-end latency of 1 millisecond or less and 99.999% reliability "
    "for robotic control systems; Enable deployment of cost-effective Reduced Capability (RedCap) IoT sensors throughout the facility; "
    "Support enhanced sidelink communication for direct device-to-device communication between manufacturing equipment; "
    "Implement Integrated Access and Backhaul (IAB) to reduce fiber backhaul costs; Provide enhanced beam management for massive MIMO "
    "deployments using 64T64R antenna configurations; Support multi-SIM devices for redundancy in critical applications; "
    "Utilize the 3.5 GHz CBRS spectrum band (n78); Implement advanced network slicing to separate eMBB, URLLC, and mMTC traffic; "
    "Offer enhanced power saving features for battery-operated IoT devices; Support carrier aggregation for increased capacity; "
    "Enable voice services through Voice over New Radio (VoNR). What is the minimum 3GPP Release specification that supports all of these "
    "requirements? Provide the release number (15, 16, or 17) and reference URLs from 3GPP or authoritative technical sources that "
    "confirm each major technical capability is supported in that release."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ReleaseEvidence(BaseModel):
    """
    Extract the stated minimum release (only 15, 16, or 17) and all capability-specific supporting URLs.
    All URLs must be explicitly present in the answer text; if missing, leave the list empty.
    """
    release: Optional[str] = None

    positioning_urls: List[str] = Field(default_factory=list)
    urllc_urls: List[str] = Field(default_factory=list)
    redcap_urls: List[str] = Field(default_factory=list)
    sidelink_urls: List[str] = Field(default_factory=list)
    iab_urls: List[str] = Field(default_factory=list)
    beam_mgmt_urls: List[str] = Field(default_factory=list)
    multi_sim_urls: List[str] = Field(default_factory=list)
    n78_urls: List[str] = Field(default_factory=list)
    slicing_urls: List[str] = Field(default_factory=list)
    power_saving_urls: List[str] = Field(default_factory=list)
    ca_urls: List[str] = Field(default_factory=list)
    vonr_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_release_and_urls() -> str:
    return """
    Your task is to extract the single, explicit minimum 3GPP Release number and capability-specific supporting URLs
    cited in the answer.

    1) release: Extract the one (and only one) minimum 3GPP Release number explicitly stated in the answer as the final result.
       - Valid values must be exactly one of: "15", "16", or "17".
       - If the answer lists multiple releases (e.g., "Release 16 or 17", or gives a range) OR does not clearly state a single final release,
         set `release` to null.

    2) For each capability below, extract all URLs that the answer cites as evidence specifically tying that capability to the chosen release.
       - Extract only actual URLs explicitly present in the answer text (plain URLs or markdown links).
       - Do not fabricate or infer URLs.
       - If no applicable URL is present for a capability, return an empty list for that capability.

       Capabilities and their output fields:
       - positioning_urls: URLs that claim indoor positioning accuracy of ~20 cm or better is supported in the chosen release.
       - urllc_urls: URLs that claim 5G URLLC with ~1 ms E2E latency and 99.999% reliability is supported in the chosen release.
       - redcap_urls: URLs that claim Reduced Capability (NR-Light/RedCap) device support is included in the chosen release.
       - sidelink_urls: URLs that claim enhanced sidelink (NR sidelink, D2D) is supported in the chosen release.
       - iab_urls: URLs that claim Integrated Access and Backhaul (IAB) is supported in the chosen release.
       - beam_mgmt_urls: URLs that claim enhanced beam management for massive MIMO (e.g., 64T64R) is supported in the chosen release.
       - multi_sim_urls: URLs that claim multi-SIM / multi-USIM support for 5G is supported in the chosen release.
       - n78_urls: URLs that claim operation in NR band n78 (3300–3800 MHz) is supported in the chosen release.
       - slicing_urls: URLs that claim advanced network slicing (eMBB/URLLC/mMTC separation) is supported in the chosen release.
       - power_saving_urls: URLs that claim enhanced power saving features for battery-operated IoT are supported in the chosen release.
       - ca_urls: URLs that claim NR carrier aggregation is supported in the chosen release.
       - vonr_urls: URLs that claim Voice over New Radio (VoNR) in 5G SA is supported in the chosen release.

    Return a single JSON object exactly matching the expected schema.
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def normalize_release(extracted: Optional[str]) -> Optional[str]:
    """
    Normalize the extracted release to "15", "16", or "17" if possible; otherwise return None.
    Accepts strings like "Release 17", "Rel-17", "R17", "17".
    """
    if not extracted:
        return None
    s = extracted.strip().lower()
    for key in ("15", "16", "17"):
        if key in s:
            return key
    return None


def _capability_map(extracted: ReleaseEvidence) -> Dict[str, Dict[str, Any]]:
    """
    Build a mapping from capability node IDs to their label, URLs and tailored instructions.
    """
    return {
        "positioning_20cm": {
            "label": "indoor positioning accuracy of about 20 centimeters or better (decimeter-level NR positioning)",
            "urls": extracted.positioning_urls,
            "instruction": (
                "Confirm that the page ties 5G NR positioning with ~20 cm (decimeter-level) accuracy to 3GPP Release {rel}. "
                "Accept synonyms like 'decimeter-level' or '20 cm'. The page should explicitly mention Release {rel} (e.g., 'Release {rel}', 'Rel-{rel}', 'R{rel}'), "
                "or cite a 3GPP TS/TR known to be part of Release {rel}."
            ),
        },
        "urllc_latency_and_reliability": {
            "label": "URLLC supporting ~1 ms end-to-end latency and 99.999% reliability targets",
            "urls": extracted.urllc_urls,
            "instruction": (
                "Verify that the page ties NR URLLC targets (~1 ms latency and ~99.999% reliability) to 3GPP Release {rel}. "
                "Allow approximate phrasing (e.g., '≈1 ms', '1 ms-level', 'five-nines'). The page should explicitly mention Release {rel} or an R{rel} TS/TR."
            ),
        },
        "redcap": {
            "label": "Reduced Capability (NR-Light / RedCap) device support",
            "urls": extracted.redcap_urls,
            "instruction": (
                "Verify that the page explicitly states that RedCap (NR-Light) is included in 3GPP Release {rel}. "
                "Look for exact release references like 'Release {rel}', 'Rel-{rel}', or 'R{rel}'."
            ),
        },
        "enhanced_sidelink": {
            "label": "enhanced 5G NR sidelink (direct device-to-device) communication",
            "urls": extracted.sidelink_urls,
            "instruction": (
                "Verify that the page ties NR sidelink (and its enhancements) to 3GPP Release {rel}."
            ),
        },
        "iab": {
            "label": "Integrated Access and Backhaul (IAB)",
            "urls": extracted.iab_urls,
            "instruction": (
                "Verify that the page ties Integrated Access and Backhaul (IAB) to 3GPP Release {rel}."
            ),
        },
        "beam_mgmt_massive_mimo_64t64r": {
            "label": "enhanced beam management for massive MIMO deployments (e.g., 64T64R)",
            "urls": extracted.beam_mgmt_urls,
            "instruction": (
                "Verify that the page connects enhanced NR beam management / beamforming (e.g., 64T64R massive MIMO) to 3GPP Release {rel}."
            ),
        },
        "multi_sim": {
            "label": "multi-SIM / multi-USIM support for redundancy in 5G",
            "urls": extracted.multi_sim_urls,
            "instruction": (
                "Verify that the page ties multi-SIM / multi-USIM functionality for 5G NR/5GS to 3GPP Release {rel}."
            ),
        },
        "spectrum_n78_cbrs": {
            "label": "operation in NR band n78 (3300–3800 MHz)",
            "urls": extracted.n78_urls,
            "instruction": (
                "Verify that the page shows band n78 (3300–3800 MHz) as part of 3GPP Release {rel} support (e.g., via TS 38.101/38.104 for Release {rel}). "
                "There must be a clear tie to Release {rel} (e.g., explicit 'Release {rel}' or a Release-{rel} spec document)."
            ),
        },
        "network_slicing": {
            "label": "advanced network slicing (separate eMBB, URLLC, mMTC)",
            "urls": extracted.slicing_urls,
            "instruction": (
                "Verify that the page ties advanced 5G network slicing (supporting eMBB, URLLC, mMTC separation) to 3GPP Release {rel}."
            ),
        },
        "power_saving": {
            "label": "enhanced power saving features for battery-operated IoT devices",
            "urls": extracted.power_saving_urls,
            "instruction": (
                "Verify that the page ties enhanced NR/5GS power-saving features for IoT to 3GPP Release {rel} "
                "(e.g., RRC Inactive improvements, eDRX/PSM applicability for NR or specific Release {rel} IoT optimizations)."
            ),
        },
        "carrier_aggregation": {
            "label": "NR carrier aggregation",
            "urls": extracted.ca_urls,
            "instruction": (
                "Verify that the page ties NR Carrier Aggregation to 3GPP Release {rel}."
            ),
        },
        "vonr": {
            "label": "Voice over New Radio (VoNR) in 5G Standalone",
            "urls": extracted.vonr_urls,
            "instruction": (
                "Verify that the page ties VoNR (voice over NR) in 5G SA to 3GPP Release {rel}."
            ),
        },
    }


def _authority_instruction_suffix() -> str:
    """
    Shared criteria about acceptable sources and explicit release tie requirement.
    """
    return (
        "Sources should be from 3GPP or authoritative technical sources (e.g., 3gpp.org, ETSI/3GPP specs TS/TR, "
        "recognized operator/vendor technical whitepapers or standards summaries). "
        "Reject sources that do not mention the release number or do not clearly tie the capability to the stated release. "
        "Do not accept generic marketing statements without explicit Release linkage."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_release_nodes(
    evaluator: Evaluator,
    parent_node,
    extracted: ReleaseEvidence,
) -> None:
    """
    Add and verify:
    - release_choice: exactly one release (15/16/17) clearly stated.
    - release_is_minimum: the chosen release is the minimum among {15,16,17} that supports all listed capabilities.
    """
    release_norm = normalize_release(extracted.release)

    # Leaf: release_choice
    release_choice_node = evaluator.add_leaf(
        id="release_choice",
        desc="States exactly one 3GPP Release number and it is one of {15, 16, 17}.",
        parent=parent_node,
        critical=True,
    )
    release_claim = (
        f"Within the answer, exactly one 3GPP Release is clearly declared as the final minimum release, and that number is '{release_norm}'. "
        "The answer does not present multiple possible releases (e.g., '16 or 17') or ranges, and the stated value is one of 15, 16, or 17."
    )
    await evaluator.verify(
        claim=release_claim,
        node=release_choice_node,
        additional_instruction=(
            "Check the final stated result in the answer body. Ignore incidental mentions inside citations or quoted text. "
            "If the answer lists multiple releases or is ambiguous, mark as Incorrect."
        ),
    )

    # Leaf: release_is_minimum
    release_min_node = evaluator.add_leaf(
        id="release_is_minimum",
        desc="The stated release is in fact the minimum among {15, 16, 17} that supports all listed required capabilities.",
        parent=parent_node,
        critical=True,
    )

    # Use a reasoning-based verification (no URLs) that leverages the verified capability evidence and common release chronology.
    # This node is blocked automatically by critical sibling failures (capability checks) due to evaluator's auto preconditions.
    cap_list = [
        "20 cm indoor positioning",
        "URLLC (~1 ms, 99.999% reliability)",
        "RedCap",
        "enhanced NR sidelink",
        "IAB",
        "enhanced beam management for 64T64R massive MIMO",
        "multi-SIM/multi-USIM",
        "NR band n78 operation",
        "advanced network slicing (eMBB/URLLC/mMTC)",
        "enhanced IoT power saving",
        "NR carrier aggregation",
        "VoNR (SA)",
    ]
    min_claim = (
        f"Given the complete set of required capabilities {cap_list}, the chosen release '{release_norm}' is the minimum among {{15,16,17}} "
        "that supports all of them. Answer Correct only if no smaller release in {15,16,17} can satisfy the full set simultaneously."
    )
    await evaluator.verify(
        claim=min_claim,
        node=release_min_node,
        additional_instruction=(
            "Use established 3GPP release chronology when necessary. For example, Reduced Capability (NR-Light/RedCap) is widely known to be a Release 17 feature; "
            "IAB and NR sidelink were specified in Release 16; many other features originated or matured across 15/16/17. "
            "If the chosen release is 17, it is sufficient to recognize that at least one required capability (e.g., RedCap) is only in 17, "
            "which means 15 or 16 cannot satisfy the full set. If the chosen release is 16 or 15, verify that all listed capabilities are indeed present in that release; "
            "if any required capability is only in a later release, then the claim is Incorrect."
        ),
    )


async def verify_capabilities_with_citations(
    evaluator: Evaluator,
    parent_node,
    extracted: ReleaseEvidence,
) -> None:
    """
    Build the 'capability_support_and_citations' parallel critical node.
    For each capability, add a single critical leaf node and verify with provided URLs.
    If the answer provides no URL for a capability, mark that leaf as failed (enforcing source-grounding).
    """
    release_norm = normalize_release(extracted.release) or "unknown"

    caps_parent = evaluator.add_parallel(
        id="capability_support_and_citations",
        desc="For each required capability, the answer claims support in the chosen release and provides at least one supporting URL.",
        parent=parent_node,
        critical=True,
    )

    cap_map = _capability_map(extracted)
    claims_and_sources: List[tuple[str, List[str], Any, Optional[str]]] = []

    # Create leaves and prepare batch verifications
    for cap_id, info in cap_map.items():
        node = evaluator.add_leaf(
            id=cap_id,
            desc=info["label"] + f" is supported in the chosen release (Release {release_norm}), with at least one supporting URL.",
            parent=caps_parent,
            critical=True,
        )

        urls: List[str] = info["urls"] or []
        # Enforce source-grounding: if no URL is provided, fail immediately without LLM verification
        if len(urls) == 0 or all((u or "").strip() == "" for u in urls):
            node.score = 0.0
            node.status = "failed"
            continue

        claim = (
            f"The cited page explicitly ties {info['label']} to 3GPP Release {release_norm} for 5G NR/5GS (Standalone applicable). "
            "The page should contain an explicit mention of the release (e.g., 'Release {rel}', 'Rel-{rel}', 'R{rel}') or reference a 3GPP TS/TR belonging to that release."
        ).replace("{rel}", str(release_norm))

        add_ins = (
            info["instruction"].replace("{rel}", str(release_norm)) + " " + _authority_instruction_suffix()
        )

        claims_and_sources.append((claim, urls, node, add_ins))

    # Run verifications in parallel for all leaves that had URLs
    if claims_and_sources:
        await evaluator.batch_verify(claims_and_sources)


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
    Entry point to evaluate an answer for the 'minimum 3GPP Release' task.
    """
    # Initialize evaluator (root aggregation is parallel per rubric)
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

    # Extract stated release and capability-specific URLs
    extracted = await evaluator.extract(
        prompt=prompt_extract_release_and_urls(),
        template_class=ReleaseEvidence,
        extraction_name="release_and_capability_urls",
    )

    # Build and verify according to rubric
    # 1) release_choice and release_is_minimum (both critical)
    await verify_release_nodes(evaluator, root, extracted)

    # 2) capability_support_and_citations subtree (all children critical)
    await verify_capabilities_with_citations(evaluator, root, extracted)

    # Optionally add quick summary info for debugging/traceability
    evaluator.add_custom_info(
        info={
            "extracted_release": normalize_release(extracted.release),
            "url_counts": {
                "positioning": len(extracted.positioning_urls),
                "urllc": len(extracted.urllc_urls),
                "redcap": len(extracted.redcap_urls),
                "sidelink": len(extracted.sidelink_urls),
                "iab": len(extracted.iab_urls),
                "beam_mgmt": len(extracted.beam_mgmt_urls),
                "multi_sim": len(extracted.multi_sim_urls),
                "n78": len(extracted.n78_urls),
                "slicing": len(extracted.slicing_urls),
                "power_saving": len(extracted.power_saving_urls),
                "ca": len(extracted.ca_urls),
                "vonr": len(extracted.vonr_urls),
            },
        },
        info_type="extraction_stats",
        info_name="extraction_summary",
    )

    # Return structured summary
    return evaluator.get_summary()