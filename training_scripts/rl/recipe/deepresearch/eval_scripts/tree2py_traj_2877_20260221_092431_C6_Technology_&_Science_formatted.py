import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "chicago_cell_backup_2026"
TASK_DESCRIPTION = (
    "Following the January 2026 Verizon outage that affected thousands of US customers, you are a "
    "network administrator in Chicago, Illinois tasked with establishing backup cellular connectivity plans "
    "for critical business operations. Identify three major alternative carriers from AT&T, T-Mobile, or UScellular "
    "that provide service in the Chicago metropolitan area. For each carrier, document: (1) Verification that they "
    "provide 4G LTE coverage in Chicago meeting FCC minimum standards (5 Mbps download / 1 Mbps upload), "
    "(2) The complete official iPhone SOS mode troubleshooting procedure including all required device settings, "
    "timing parameters, and configuration steps, (3) The carrier settings update verification process for iPhone devices, "
    "and (4) The method or tool for checking network outage status in real-time. Provide official reference sources for all technical information."
)

ALLOWED_CARRIERS = {"AT&T", "T-Mobile", "UScellular"}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CoverageVerification(BaseModel):
    service_source_urls: List[str] = Field(default_factory=list)   # Official carrier service/coverage pages
    fcc_source_urls: List[str] = Field(default_factory=list)       # FCC standardized coverage data sources (e.g., broadbandmap.fcc.gov)


class SOSProcedure(BaseModel):
    steps: List[str] = Field(default_factory=list)
    includes_airplane_mode_15s: Optional[bool] = None
    includes_device_restart: Optional[bool] = None
    includes_ios_update_verification: Optional[bool] = None
    includes_carrier_settings_check_about: Optional[bool] = None
    includes_cellular_data_enabled: Optional[bool] = None
    includes_lte_enabled: Optional[bool] = None
    apple_support_urls: List[str] = Field(default_factory=list)


class CarrierSettingsUpdate(BaseModel):
    process_text: Optional[str] = None
    specifies_wifi_requirement: Optional[bool] = None
    official_urls: List[str] = Field(default_factory=list)


class OutageStatus(BaseModel):
    method_text: Optional[str] = None
    official_urls: List[str] = Field(default_factory=list)


class CarrierDoc(BaseModel):
    name: Optional[str] = None
    coverage: CoverageVerification = Field(default_factory=CoverageVerification)
    sos: SOSProcedure = Field(default_factory=SOSProcedure)
    carrier_settings: CarrierSettingsUpdate = Field(default_factory=CarrierSettingsUpdate)
    outage: OutageStatus = Field(default_factory=OutageStatus)


class PlanExtraction(BaseModel):
    selected_carriers: List[CarrierDoc] = Field(default_factory=list)
    all_carriers_mentioned_allowed_set: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return (
        "Extract the backup cellular connectivity plan details from the answer. Only consider carriers from the allowed set: "
        "{AT&T, T-Mobile, UScellular}. Identify up to the first three carriers the answer uses as the plan items.\n\n"
        "Return a JSON object with fields:\n"
        "1) selected_carriers: array of up to three carrier objects, in the order presented in the answer. For each carrier object:\n"
        "   - name: the carrier name (must be exactly 'AT&T', 'T-Mobile', or 'UScellular' if present). If ambiguous or not in allowed set, set to null.\n"
        "   - coverage:\n"
        "       • service_source_urls: all official carrier URLs cited that support service or coverage in Chicago (e.g., coverage map, service availability pages). Extract only URLs actually present in the answer.\n"
        "       • fcc_source_urls: all FCC standardized coverage data URLs cited (e.g., FCC National Broadband Map pages, fcc.gov pages). Extract only URLs present in the answer.\n"
        "   - sos:\n"
        "       • steps: list of the SOS troubleshooting steps described in the answer.\n"
        "       • includes_airplane_mode_15s: true if the answer explicitly instructs toggling Airplane Mode and waiting at least 15 seconds before turning it off; otherwise false or null.\n"
        "       • includes_device_restart: true if device restart is included.\n"
        "       • includes_ios_update_verification: true if verifying iOS/iPadOS software update is included.\n"
        "       • includes_carrier_settings_check_about: true if checking for carrier settings update via Settings > General > About is included.\n"
        "       • includes_cellular_data_enabled: true if verifying Cellular Data is enabled is included.\n"
        "       • includes_lte_enabled: true if verifying LTE is enabled (Settings > Cellular > Cellular Data Options) is included.\n"
        "       • apple_support_urls: all Apple Support URLs cited for the SOS troubleshooting procedure.\n"
        "   - carrier_settings:\n"
        "       • process_text: the text description the answer provides for verifying or updating iPhone carrier settings.\n"
        "       • specifies_wifi_requirement: true if the answer states a Wi‑Fi connection is required for carrier settings update verification; otherwise false or null.\n"
        "       • official_urls: all official URLs cited (Apple Support or carrier-official) describing the carrier settings update verification process.\n"
        "   - outage:\n"
        "       • method_text: the described method/tool in the answer to check real-time network outage/status for that carrier.\n"
        "       • official_urls: all official carrier URLs cited that provide the real-time outage/status tool or page.\n"
        "2) all_carriers_mentioned_allowed_set: list of all allowed carrier names mentioned anywhere in the answer (subset of {AT&T, T-Mobile, UScellular}).\n\n"
        "Rules:\n"
        "- Extract only information explicitly present in the answer; do not infer.\n"
        "- URLs may appear as plain or markdown links; extract the actual URL strings.\n"
        "- If any field is missing, set to null or an empty list as appropriate.\n"
        "- If the answer mentions more than three allowed carriers, include only the first three as selected_carriers.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(name: Optional[str]) -> str:
    return name if (name and name.strip()) else "Unknown carrier"


def _exists_nonempty_str(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _exists_nonempty_list(lst: Optional[List[str]]) -> bool:
    return bool(lst and len(lst) > 0)


# --------------------------------------------------------------------------- #
# Verification per carrier                                                    #
# --------------------------------------------------------------------------- #
async def verify_carrier(
    evaluator: Evaluator,
    parent_node,
    carrier: CarrierDoc,
    idx: int,
) -> None:
    """
    Build verification sub-tree for one carrier entry according to the rubric.
    All children under this carrier node are critical (the rubric requires complete documentation).
    """
    carrier_name = _safe_name(carrier.name)

    # Carrier node (critical, parallel aggregation)
    carrier_node = evaluator.add_parallel(
        id=f"carrier_{idx + 1}",
        desc=f"Carrier {idx + 1} documentation (as listed in the answer): {carrier_name}",
        parent=parent_node,
        critical=True,
    )

    # 1) Chicago service claim stated in the answer (simple verification against the answer text)
    chicago_claim_node = evaluator.add_leaf(
        id=f"carrier_{idx + 1}_chicago_service_claim",
        desc="States the carrier provides service in the Chicago metropolitan area.",
        parent=carrier_node,
        critical=True,
    )
    claim_service = (
        f"The answer explicitly states that {carrier_name} provides service or coverage in the Chicago metropolitan area."
    )
    await evaluator.verify(
        claim=claim_service,
        node=chicago_claim_node,
        additional_instruction=(
            "Verify by reading the answer text. Accept if the answer clearly asserts service or coverage for Chicago "
            f"for {carrier_name}. If the carrier name is missing or ambiguous, mark incorrect."
        ),
    )

    # 2) Chicago service official source URL is provided (existence check)
    service_source_exists = _exists_nonempty_list(carrier.coverage.service_source_urls)
    evaluator.add_custom_node(
        result=service_source_exists,
        id=f"carrier_{idx + 1}_chicago_service_official_source",
        desc="Provides an official source URL supporting the Chicago-area service/coverage claim.",
        parent=carrier_node,
        critical=True,
    )

    # 3) Coverage meets FCC minimum (verify against FCC standardized sources)
    coverage_min_node = evaluator.add_leaf(
        id=f"carrier_{idx + 1}_coverage_meets_fcc_minimum",
        desc="Verifies 4G LTE coverage in Chicago meets FCC minimum standard (5 Mbps download / 1 Mbps upload).",
        parent=carrier_node,
        critical=True,
    )
    claim_fcc_min = (
        f"The provided FCC standardized coverage source(s) show that {carrier_name}'s 4G LTE coverage in Chicago "
        "meets or exceeds the FCC minimum mobile broadband standard of 5 Mbps download and 1 Mbps upload."
    )
    await evaluator.verify(
        claim=claim_fcc_min,
        node=coverage_min_node,
        sources=carrier.coverage.fcc_source_urls,
        additional_instruction=(
            "Use the FCC National Broadband Map or other FCC standardized datasets. "
            "Look for the mobility/coverage layers and any legend/category indicating coverage at >=5 Mbps down and >=1 Mbps up. "
            "The page should be relevant to the Chicago area; if not clearly Chicago-relevant or speed-threshold relevant, mark not supported."
        ),
    )

    # 4) Coverage uses FCC standardized data (verify at least one FCC standardized page)
    fcc_data_node = evaluator.add_leaf(
        id=f"carrier_{idx + 1}_coverage_uses_fcc_standardized_data",
        desc="References FCC-standardized coverage data/source used for coverage verification.",
        parent=carrier_node,
        critical=True,
    )
    claim_fcc_source = (
        "This page is an official FCC standardized coverage data source relevant to mobile broadband coverage in Chicago "
        "(e.g., FCC National Broadband Map or a page on fcc.gov)."
    )
    await evaluator.verify(
        claim=claim_fcc_source,
        node=fcc_data_node,
        sources=carrier.coverage.fcc_source_urls,
        additional_instruction=(
            "Confirm the page is an FCC standardized data source (domain such as fcc.gov or broadbandmap.fcc.gov) "
            "and pertains to mobile broadband coverage relevant to Chicago."
        ),
    )

    # 5) SOS troubleshooting procedure with required elements and Apple Support reference
    sos_node = evaluator.add_parallel(
        id=f"carrier_{idx + 1}_sos_troubleshooting_procedure",
        desc="Provides the iPhone SOS mode troubleshooting procedure and includes all required elements and an Apple Support reference.",
        parent=carrier_node,
        critical=True,
    )

    # Required element checks (existence in the answer, each critical)
    evaluator.add_custom_node(
        result=bool(carrier.sos.includes_airplane_mode_15s),
        id=f"carrier_{idx + 1}_sos_airplane_mode_15s",
        desc="SOS troubleshooting includes toggling Airplane Mode and waiting at least 15 seconds before turning it off.",
        parent=sos_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(carrier.sos.includes_device_restart),
        id=f"carrier_{idx + 1}_sos_device_restart",
        desc="SOS troubleshooting includes a device restart step.",
        parent=sos_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(carrier.sos.includes_ios_update_verification),
        id=f"carrier_{idx + 1}_sos_ios_update_verification",
        desc="SOS troubleshooting includes iOS/iPadOS software update verification.",
        parent=sos_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(carrier.sos.includes_carrier_settings_check_about),
        id=f"carrier_{idx + 1}_sos_carrier_settings_check_about",
        desc="SOS troubleshooting includes checking for carrier settings update via Settings > General > About.",
        parent=sos_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(carrier.sos.includes_cellular_data_enabled),
        id=f"carrier_{idx + 1}_sos_cellular_data_enabled",
        desc="SOS troubleshooting includes verification that Cellular Data is enabled.",
        parent=sos_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(carrier.sos.includes_lte_enabled),
        id=f"carrier_{idx + 1}_sos_lte_enabled",
        desc="SOS troubleshooting includes verification that LTE is enabled (Settings > Cellular > Cellular Data Options).",
        parent=sos_node,
        critical=True,
    )

    # Apple Support source verification
    sos_source_node = evaluator.add_leaf(
        id=f"carrier_{idx + 1}_sos_troubleshooting_apple_source",
        desc="Cites an official Apple Support source URL for the SOS troubleshooting procedure.",
        parent=sos_node,
        critical=True,
    )
    claim_apple_sos = (
        "This page is an official Apple Support article that covers iPhone 'SOS' or 'SOS Only' status and "
        "provides troubleshooting guidance."
    )
    await evaluator.verify(
        claim=claim_apple_sos,
        node=sos_source_node,
        sources=carrier.sos.apple_support_urls,
        additional_instruction=(
            "Confirm the domain is support.apple.com (localized variants permitted) and the content is about SOS/SOS Only "
            "status troubleshooting for iPhone."
        ),
    )

    # 6) Carrier settings update process described (check via answer text)
    settings_process_node = evaluator.add_leaf(
        id=f"carrier_{idx + 1}_carrier_settings_update_process_described",
        desc="Describes the carrier settings update verification process for iPhone devices.",
        parent=carrier_node,
        critical=True,
    )
    claim_settings_process = (
        "The answer describes how to verify or update iPhone carrier settings, including navigating to Settings > General > About "
        "to check for a carrier settings update prompt."
    )
    await evaluator.verify(
        claim=claim_settings_process,
        node=settings_process_node,
        additional_instruction=(
            "Verify by reading the answer text only. Look for the explicit path 'Settings > General > About' and an instruction "
            "to accept or check for a carrier settings update."
        ),
    )

    # 7) Carrier settings Wi‑Fi requirement specified (existence check in answer)
    wifi_req = bool(carrier.carrier_settings.specifies_wifi_requirement)
    evaluator.add_custom_node(
        result=wifi_req,
        id=f"carrier_{idx + 1}_carrier_settings_update_wifi_requirement",
        desc="Carrier settings update verification process specifies the Wi‑Fi connection requirement.",
        parent=carrier_node,
        critical=True,
    )

    # 8) Carrier settings update official source verification
    settings_source_node = evaluator.add_leaf(
        id=f"carrier_{idx + 1}_carrier_settings_update_official_source",
        desc="Provides an official source URL (Apple Support or carrier-official) supporting the carrier settings update verification process.",
        parent=carrier_node,
        critical=True,
    )
    claim_settings_source = (
        "This page is an official source (Apple Support or an official carrier page) that documents how to update or verify "
        "iPhone carrier settings."
    )
    await evaluator.verify(
        claim=claim_settings_source,
        node=settings_source_node,
        sources=carrier.carrier_settings.official_urls,
        additional_instruction=(
            "Accept support.apple.com, att.com/support, t-mobile.com/support or help pages, and uscellular.com official pages "
            "that explain carrier settings updates."
        ),
    )

    # 9) Outage status method specified (existence check in answer)
    outage_method_exists = _exists_nonempty_str(carrier.outage.method_text)
    evaluator.add_custom_node(
        result=outage_method_exists,
        id=f"carrier_{idx + 1}_outage_status_method",
        desc="Specifies a method/tool to check real-time network outage status for the carrier.",
        parent=carrier_node,
        critical=True,
    )

    # 10) Outage status official source verification
    outage_source_node = evaluator.add_leaf(
        id=f"carrier_{idx + 1}_outage_status_official_source",
        desc="Provides an official carrier source URL for the real-time outage/status method/tool.",
        parent=carrier_node,
        critical=True,
    )
    claim_outage_source = (
        f"This page is {carrier_name}'s official live network status/outage page or tool."
    )
    await evaluator.verify(
        claim=claim_outage_source,
        node=outage_source_node,
        sources=carrier.outage.official_urls,
        additional_instruction=(
            "Confirm this is an official carrier-hosted network status or outage map/tool. "
            "Accept att.com, t-mobile.com, uscellular.com domains or relevant subdomains hosting outage or network status pages."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate the answer for the Chicago backup cellular connectivity plan following the January 2026 Verizon outage.
    """
    # Initialize evaluator with a sequential root (carrier selection first, then per-carrier documentation)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Record allowed carriers in summary
    evaluator.add_ground_truth({"allowed_carriers": sorted(list(ALLOWED_CARRIERS))})

    # Extract structured plan details from the answer
    plan: PlanExtraction = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction",
    )

    # -------------------- Carrier selection checks (critical) ----------------- #
    selection_node = evaluator.add_parallel(
        id="carrier_selection",
        desc="Select exactly three distinct carriers from the allowed set.",
        parent=root,
        critical=True,
    )

    # Compute selection info
    selected_names = [c.name for c in plan.selected_carriers if _exists_nonempty_str(c.name)]
    exactly_three = (len(plan.selected_carriers) == 3) and (len(selected_names) == 3)
    from_allowed = all(name in ALLOWED_CARRIERS for name in selected_names) if selected_names else False
    distinct = len(set(selected_names)) == len(selected_names) == 3

    evaluator.add_custom_node(
        result=exactly_three,
        id="exactly_three_carriers_provided",
        desc="Answer identifies exactly three carriers (no fewer, no more).",
        parent=selection_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=from_allowed,
        id="carriers_from_allowed_set",
        desc="All identified carriers are from {AT&T, T-Mobile, UScellular}.",
        parent=selection_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=distinct,
        id="carriers_are_distinct",
        desc="The three identified carriers are all different (no duplicates).",
        parent=selection_node,
        critical=True,
    )

    # -------------------- Per-carrier documentation (critical) ---------------- #
    per_carrier_node = evaluator.add_parallel(
        id="per_carrier_documentation",
        desc="For each of the three carriers, provide all required documentation and official references.",
        parent=root,
        critical=True,
    )

    # Prepare exactly three carrier entries (pad with empty CarrierDoc if fewer extracted)
    carriers_for_verification: List[CarrierDoc] = list(plan.selected_carriers[:3])
    while len(carriers_for_verification) < 3:
        carriers_for_verification.append(CarrierDoc())

    # Build per-carrier verification subtrees
    for idx, carrier in enumerate(carriers_for_verification):
        await verify_carrier(evaluator, per_carrier_node, carrier, idx)

    # Final summary
    return evaluator.get_summary()