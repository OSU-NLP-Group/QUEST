import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gaming_console_specs"
TASK_DESCRIPTION = (
    "Which current-generation gaming console features exactly 825GB of built-in solid-state storage with 667GB available "
    "to users, includes 16GB GDDR6 memory, and supports M.2 SSD expansion using a PCIe Gen4x4 interface with storage "
    "capacity ranging from 250GB to 8TB? Additionally, provide the minimum connection speed required for its Remote Play "
    "feature, the complete port configuration, and the minimum upload speed required for streaming gameplay from this "
    "console at 1080p resolution and 60fps."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConsoleSpecsExtraction(BaseModel):
    """
    Extracted console identification and specification values as presented in the agent's answer,
    along with the specific URLs cited for each specification.
    """
    console_name: Optional[str] = None

    built_in_storage: Optional[str] = None
    built_in_storage_sources: List[str] = Field(default_factory=list)

    user_available_storage: Optional[str] = None
    user_available_storage_sources: List[str] = Field(default_factory=list)

    memory_spec: Optional[str] = None
    memory_sources: List[str] = Field(default_factory=list)

    expansion_interface: Optional[str] = None
    expansion_interface_sources: List[str] = Field(default_factory=list)

    expansion_capacity_range: Optional[str] = None
    expansion_capacity_sources: List[str] = Field(default_factory=list)

    remote_play_speed: Optional[str] = None
    remote_play_sources: List[str] = Field(default_factory=list)

    port_configuration: Optional[str] = None
    port_config_sources: List[str] = Field(default_factory=list)

    streaming_upload_requirement: Optional[str] = None
    streaming_upload_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_console_specs() -> str:
    return (
        "From the provided answer, extract the identified console name and the specific values the answer claims for each "
        "of the following specifications, as well as the exact URLs the answer cites for each spec. Return null for any "
        "value that is not explicitly present in the answer and return an empty array for sources when no URLs are given.\n"
        "\n"
        "Required fields to extract:\n"
        "- console_name: The name/model of the console being discussed.\n"
        "- built_in_storage: The stated built-in solid-state storage capacity (e.g., \"825GB\").\n"
        "- built_in_storage_sources: All URLs the answer cites that support the built-in storage claim.\n"
        "- user_available_storage: The stated user-available storage capacity (e.g., \"667GB\").\n"
        "- user_available_storage_sources: All URLs the answer cites for user-available storage.\n"
        "- memory_spec: The stated memory spec (e.g., \"16GB GDDR6\").\n"
        "- memory_sources: All URLs the answer cites for memory.\n"
        "- expansion_interface: The stated expansion interface (e.g., \"M.2 SSD with PCIe Gen4x4\").\n"
        "- expansion_interface_sources: All URLs supporting the expansion interface.\n"
        "- expansion_capacity_range: The stated supported M.2 capacity range (e.g., \"250GB to 8TB\").\n"
        "- expansion_capacity_sources: All URLs for the capacity range.\n"
        "- remote_play_speed: The stated minimum connection speed required for Remote Play (e.g., \"5 Mbps for upload and download\").\n"
        "- remote_play_sources: All URLs supporting the Remote Play minimum speed.\n"
        "- port_configuration: A normalized string listing port counts/types (e.g., \"1 x HDMI 2.1, 1 x Ethernet, 2 x USB-C, 2 x USB-A\").\n"
        "- port_config_sources: All URLs supporting the port configuration.\n"
        "- streaming_upload_requirement: Minimum upload speed required for streaming gameplay at 1080p 60fps (e.g., \"12 Mbps\").\n"
        "- streaming_upload_sources: All URLs supporting the streaming upload requirement.\n"
        "\n"
        "IMPORTANT:\n"
        "• Extract only information explicitly present in the answer. Do not invent any values.\n"
        "• Extract only valid URLs that appear in the answer text (including markdown links). If a source is mentioned without a URL, do not add a URL.\n"
        "• Preserve units and formatting in the extracted values exactly as they appear in the answer."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _spec_additional_instruction(spec_key: str) -> str:
    """
    Provide targeted instructions to the LLM verifier per spec to help robust matching and avoid nitpicking on formatting.
    """
    mapping = {
        "Built_In_Storage": (
            "Verify that the referenced webpage explicitly states the console has 825GB of built-in solid-state storage. "
            "Allow minor formatting variations (e.g., \"825 GB\", \"825GB SSD\"). Ensure this refers to the internal storage."
        ),
        "User_Available_Storage": (
            "Verify that the webpage explicitly indicates approximately 667GB (or stated as 667GB) user-available storage. "
            "Minor rounding differences are acceptable only if the text clearly states ~667GB available to users."
        ),
        "Memory_Specifications": (
            "Confirm the console includes 16GB GDDR6 memory. Accept synonyms like \"GDDR6 RAM 16GB\"."
        ),
        "Expansion_Interface": (
            "Confirm the console supports M.2 SSD expansion using a PCIe Gen4 x4 interface. "
            "Accept equivalent phrasing (e.g., \"PCIe 4.0 x4\", \"Gen 4 x4\", \"NVMe M.2 PCIe 4.0 x4\")."
        ),
        "Expansion_Capacity_Range": (
            "Confirm the supported M.2 SSD capacity range includes both the minimum of 250GB and the maximum of 8TB. "
            "Phrasing like \"from 250GB up to 8TB\" or \"250GB–8TB\" should be accepted."
        ),
        "Remote_Play_Speed": (
            "Confirm the Remote Play minimum connection speed requirement is 5 Mbps for both upload and download. "
            "Accept wording like \"at least 5 Mbps upstream and downstream\"."
        ),
        "Port_Configuration": (
            "Confirm the console port configuration includes exactly 1 x HDMI 2.1, 1 x Ethernet (LAN), 2 x USB-C, and 2 x USB-A. "
            "You must verify the counts and types match exactly; minor name variants like \"LAN\" for Ethernet are acceptable."
        ),
        "Streaming_Upload_Requirement": (
            "Confirm that streaming gameplay at 1080p resolution and 60fps from this console requires a minimum upload speed of 12 Mbps. "
            "Accept phrasing like \"at least 12 Mbps upload for 1080p60 streaming\"."
        ),
    }
    return mapping.get(spec_key, "Verify the claim against the webpage evidence, allowing minor formatting variations.")


def _build_claim_text(spec_key: str) -> str:
    """
    Build the exact claim text corresponding to each rubric leaf.
    """
    claims = {
        "Built_In_Storage": "This console has exactly 825GB of built-in solid-state storage.",
        "User_Available_Storage": "This console provides 667GB of storage available to users for game installation.",
        "Memory_Specifications": "This console includes 16GB of GDDR6 memory.",
        "Expansion_Interface": "This console supports M.2 SSD expansion using a PCIe Gen4x4 interface.",
        "Expansion_Capacity_Range": "The M.2 SSD expansion supports storage capacity between 250GB and 8TB.",
        "Remote_Play_Speed": "The console's Remote Play feature requires a minimum 5 Mbps connection speed for both upload and download.",
        "Port_Configuration": "The console includes 1 x HDMI 2.1, 1 x Ethernet, 2 x USB-C, and 2 x USB-A ports.",
        "Streaming_Upload_Requirement": "Streaming gameplay at 1080p resolution and 60fps from this console requires a minimum upload speed of 12 Mbps.",
    }
    return claims[spec_key]


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_and_verify_console_specs(
    evaluator: Evaluator,
    root: VerificationNode,
    extracted: ConsoleSpecsExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """
    # Create top-level critical parallel node
    specs_node = evaluator.add_parallel(
        id="Gaming_Console_Specifications",
        desc="Evaluate whether the identified gaming console meets all specified technical requirements",
        parent=root,
        critical=True,
    )

    # Prepare spec definitions aligned with rubric leaves
    spec_defs = [
        ("Built_In_Storage", "The console has 825GB of built-in solid-state storage", extracted.built_in_storage_sources),
        ("User_Available_Storage", "The console provides 667GB of storage available to users for game installation", extracted.user_available_storage_sources),
        ("Memory_Specifications", "The console includes 16GB GDDR6 memory", extracted.memory_sources),
        ("Expansion_Interface", "The console supports M.2 SSD expansion with PCIe Gen4x4 interface", extracted.expansion_interface_sources),
        ("Expansion_Capacity_Range", "The M.2 SSD expansion supports storage capacity between 250GB and 8TB", extracted.expansion_capacity_sources),
        ("Remote_Play_Speed", "The console's Remote Play feature requires minimum 5 Mbps connection speed for both upload and download", extracted.remote_play_sources),
        ("Port_Configuration", "The console includes 1 x HDMI 2.1, 1 x Ethernet, 2 x USB-C, and 2 x USB-A ports", extracted.port_config_sources),
        ("Streaming_Upload_Requirement", "For streaming gameplay at 1080p 60fps from this console, a minimum upload speed of 12 Mbps is required", extracted.streaming_upload_sources),
    ]

    # Collect verifications for batch execution (only when sources exist)
    batch_items: List[Tuple[str, List[str] | str | None, VerificationNode, Optional[str]]] = []

    for spec_id, spec_desc, sources in spec_defs:
        # If there are no sources provided in the answer, treat it as failure for this critical leaf.
        if not sources:
            evaluator.add_custom_node(
                result=False,
                id=spec_id,
                desc=spec_desc,
                parent=specs_node,
                critical=True,
            )
            continue

        # Create leaf node and schedule verification with URLs
        leaf = evaluator.add_leaf(
            id=spec_id,
            desc=spec_desc,
            parent=specs_node,
            critical=True,
        )
        claim_text = _build_claim_text(spec_id)
        add_ins = _spec_additional_instruction(spec_id)

        batch_items.append((claim_text, sources, leaf, add_ins))

    # Run all URL-based verifications in parallel
    if batch_items:
        await evaluator.batch_verify(batch_items)

    # Record some custom info for transparency
    evaluator.add_custom_info(
        info={
            "console_name_extracted": extracted.console_name,
            "built_in_storage": extracted.built_in_storage,
            "user_available_storage": extracted.user_available_storage,
            "memory_spec": extracted.memory_spec,
            "expansion_interface": extracted.expansion_interface,
            "expansion_capacity_range": extracted.expansion_capacity_range,
            "remote_play_speed": extracted.remote_play_speed,
            "port_configuration": extracted.port_configuration,
            "streaming_upload_requirement": extracted.streaming_upload_requirement,
            "policy_note": "Each critical spec must be supported by at least one cited URL; missing sources cause an immediate failure on that spec."
        },
        info_type="extraction_summary",
        info_name="extracted_console_specs_overview"
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
    Entry point to evaluate the agent's answer for the gaming console specifications task.
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_console_specs(),
        template_class=ConsoleSpecsExtraction,
        extraction_name="console_specs_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_console_specs(evaluator, root, extracted)

    # Final summary
    return evaluator.get_summary()