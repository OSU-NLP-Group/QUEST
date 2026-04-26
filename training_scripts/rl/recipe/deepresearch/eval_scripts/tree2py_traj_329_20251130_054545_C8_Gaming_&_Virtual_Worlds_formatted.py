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
TASK_ID = "console_selection_2024"
TASK_DESCRIPTION = """
A gaming enthusiast in California is looking to purchase a new current-generation gaming console in November 2024 that meets the following comprehensive requirements for an optimal gaming setup:

The console must have at least 2TB of built-in SSD storage to accommodate multiple modern AAA games. It must deliver at least 16 TFLOPs of GPU compute performance for high-end graphics processing. The console's retail price in the United States must not exceed $750.

The console must be compatible with a VR (virtual reality) headset that is currently available for purchase at a price below $400. It must support 4K gaming at up to 120 frames per second (fps), and must support hardware-based ray tracing for enhanced visual effects.

The annual subscription cost for online multiplayer gaming on this console must not exceed $80. The console must be currently available for purchase in the United States (released by November 30, 2024).

Additionally, the console must support HDR (High Dynamic Range) for enhanced visual quality, must support Variable Refresh Rate (VRR) technology, and must support HDMI 2.1 connectivity for high-performance gaming features. Finally, the console must support backward compatibility with games from the previous generation of the same console family.

Which specific console model currently available in the United States meets all of these requirements? Provide the console name, its key specifications (storage capacity and GPU TFLOPs), current US retail price, a compatible VR headset with its price, the name and annual cost of the required online multiplayer subscription service, release date, and URL references supporting each specification.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ConsoleSelection(BaseModel):
    # Identification
    console_name: Optional[str] = None

    # Storage
    storage_capacity: Optional[str] = None
    storage_urls: List[str] = Field(default_factory=list)

    # GPU TFLOPs
    gpu_tflops: Optional[str] = None
    gpu_urls: List[str] = Field(default_factory=list)

    # Price (US retail)
    us_retail_price: Optional[str] = None
    price_urls: List[str] = Field(default_factory=list)

    # VR headset compatibility, price, availability
    vr_headset_name: Optional[str] = None
    vr_headset_price: Optional[str] = None
    vr_price_urls: List[str] = Field(default_factory=list)
    vr_available_statement: Optional[str] = None
    vr_availability_urls: List[str] = Field(default_factory=list)
    vr_compatibility_statement: Optional[str] = None
    vr_compatibility_urls: List[str] = Field(default_factory=list)

    # 4K/120 fps support
    support_4k120_statement: Optional[str] = None
    support_4k120_urls: List[str] = Field(default_factory=list)

    # Ray tracing support
    ray_tracing_statement: Optional[str] = None
    ray_tracing_urls: List[str] = Field(default_factory=list)

    # Online subscription
    subscription_name: Optional[str] = None
    subscription_annual_cost: Optional[str] = None
    subscription_urls: List[str] = Field(default_factory=list)

    # Release status and US availability
    release_date: Optional[str] = None
    release_date_urls: List[str] = Field(default_factory=list)
    us_availability_statement: Optional[str] = None
    us_availability_urls: List[str] = Field(default_factory=list)

    # HDR / VRR / HDMI 2.1
    hdr_statement: Optional[str] = None
    hdr_urls: List[str] = Field(default_factory=list)
    vrr_statement: Optional[str] = None
    vrr_urls: List[str] = Field(default_factory=list)
    hdmi_2_1_statement: Optional[str] = None
    hdmi_urls: List[str] = Field(default_factory=list)

    # Backward compatibility
    backcompat_statement: Optional[str] = None
    backcompat_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_console_selection() -> str:
    return """
Extract the specific console model and all requested fields as presented in the answer text. Return each field exactly as stated in the answer when applicable. For all URL fields, extract the actual URLs explicitly present in the answer (including within markdown links). Do not invent or infer URLs.

Required fields (return null if a field is not stated in the answer):
1) console_name: The exact specific console model name.

2) storage_capacity: The built-in (internal) SSD storage capacity as stated (e.g., "2TB", "2 TB SSD", "2000 GB SSD").
   storage_urls: URLs that support the stated built-in SSD capacity.

3) gpu_tflops: The GPU compute performance figure as stated (e.g., "16 TFLOPs", "10.28 TFLOPs").
   gpu_urls: URLs that support the stated GPU TFLOPs.

4) us_retail_price: The current US retail price (e.g., "$699", "USD 699").
   price_urls: URLs that support the stated US retail price.

5) vr_headset_name: The compatible VR headset model name.
   vr_headset_price: The VR headset price as stated (e.g., "$299", "USD 299").
   vr_price_urls: URLs that support the VR headset price.
   vr_available_statement: A short phrase from the answer indicating the VR headset is currently available for purchase (e.g., "currently available", "in stock"). If not stated, return null.
   vr_availability_urls: URLs that support that the VR headset is currently purchasable.
   vr_compatibility_statement: A short phrase from the answer indicating compatibility between the console and the VR headset (e.g., "compatible with").
   vr_compatibility_urls: URLs that support compatibility between the console and the VR headset.

6) support_4k120_statement: A phrase from the answer indicating 4K up to 120 fps support (e.g., "4K 120fps").
   support_4k120_urls: URLs that support 4K/120fps support.

7) ray_tracing_statement: A phrase indicating hardware-based ray tracing support.
   ray_tracing_urls: URLs that support ray tracing support.

8) subscription_name: The online multiplayer subscription service name.
   subscription_annual_cost: The annual cost as stated in the answer (e.g., "$79.99/year"). Ensure it is an annual figure.
   subscription_urls: URLs that support the subscription pricing/annual cost.

9) release_date: The release date as stated in the answer (e.g., "November 10, 2020", "October 2024").
   release_date_urls: URLs that support the release date.
   us_availability_statement: A phrase indicating the console is currently available for purchase in the United States.
   us_availability_urls: URLs that support current US availability.

10) hdr_statement: A phrase indicating HDR support.
    hdr_urls: URLs that support HDR.

11) vrr_statement: A phrase indicating VRR support.
    vrr_urls: URLs that support VRR.

12) hdmi_2_1_statement: A phrase indicating HDMI 2.1 support.
    hdmi_urls: URLs that support HDMI 2.1.

13) backcompat_statement: A phrase indicating backward compatibility with previous generation games of the same console family.
    backcompat_urls: URLs that support backward compatibility.

Return a single JSON object with these fields. For all URL lists, include every valid URL explicitly present in the answer for that field. If no URL is provided in the answer for a field, return an empty array for that field.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


async def _simple_check(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    add_ins: str,
    critical: bool = True,
):
    node = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction=add_ins,
    )
    return node


async def _url_support_check(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    urls: Optional[List[str]],
    add_ins: str,
    critical: bool = True,
):
    node = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    # Enforce that URLs must be present; if none, hard-fail this leaf
    if not urls or len(urls) == 0:
        node.score = 0.0
        node.status = "failed"
        return node

    await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,  # Use multi-URL verification when multiple are present; single URL when only one is present
        additional_instruction=add_ins,
    )
    return node


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def verify_console_requirements(evaluator: Evaluator, root, data: ConsoleSelection):
    # Top-level critical node
    console_node = evaluator.add_parallel(
        id="console_identification",
        desc="Identify a current-generation gaming console model currently available for purchase in the United States that meets all listed requirements, and provide the requested fields with supporting URLs.",
        parent=root,
        critical=True,
    )

    # 1) Console name provided
    evaluator.add_custom_node(
        result=_non_empty(data.console_name),
        id="console_name_provided",
        desc="Provide the specific console model name.",
        parent=console_node,
        critical=True,
    )

    # 2) Storage requirement: ≥ 2TB built-in SSD
    storage_node = evaluator.add_parallel(
        id="storage_capacity_requirement",
        desc="Console has at least 2TB of built-in SSD storage.",
        parent=console_node,
        critical=True,
    )

    storage_value_str = data.storage_capacity or ""
    await _simple_check(
        evaluator,
        storage_node,
        "storage_value_check",
        "Answer states the console's built-in SSD storage capacity and it is ≥ 2TB.",
        claim=f"According to the answer, the console has built-in SSD storage capacity '{storage_value_str}', and this capacity is at least 2 TB (terabytes).",
        add_ins=(
            "Judge based on the answer text only. Consider '2 TB', '2TB', '2000 GB', or larger as meeting ≥ 2 TB. "
            "If the answer does not clearly state a capacity or it is below 2 TB, mark Incorrect."
        ),
        critical=True,
    )

    await _url_support_check(
        evaluator,
        storage_node,
        "storage_url_check",
        "Provide URL reference(s) supporting the built-in SSD storage capacity claim.",
        claim=f"The console's built-in internal SSD storage capacity is {storage_value_str}.",
        urls=data.storage_urls,
        add_ins=(
            "Verify that the provided page(s) explicitly list the internal/built-in storage capacity and that it matches or is equivalent to the stated value."
        ),
        critical=True,
    )

    # 3) GPU TFLOPs requirement: ≥ 16 TFLOPs
    gpu_node = evaluator.add_parallel(
        id="gpu_performance_requirement",
        desc="Console delivers at least 16 TFLOPs of GPU compute performance.",
        parent=console_node,
        critical=True,
    )
    gpu_str = data.gpu_tflops or ""
    await _simple_check(
        evaluator,
        gpu_node,
        "gpu_value_check",
        "Answer states the console's GPU compute performance and it is ≥ 16 TFLOPs.",
        claim=f"According to the answer, the console's GPU compute performance is '{gpu_str}', and this is at least 16 TFLOPs.",
        add_ins=(
            "Judge against the answer text. Treat 16.0 TFLOPs as meeting the requirement; anything less than 16 TFLOPs fails. "
            "If the answer omits the GPU TFLOPs or states a lower figure, mark Incorrect."
        ),
        critical=True,
    )
    await _url_support_check(
        evaluator,
        gpu_node,
        "gpu_url_check",
        "Provide URL reference(s) supporting the GPU TFLOPs claim.",
        claim=f"The console's GPU compute performance is {gpu_str}.",
        urls=data.gpu_urls,
        add_ins="Verify that the page(s) explicitly mention the GPU compute performance matching the stated figure.",
        critical=True,
    )

    # 4) Price requirement: ≤ $750
    price_node = evaluator.add_parallel(
        id="price_requirement",
        desc="Console US retail price does not exceed $750.",
        parent=console_node,
        critical=True,
    )
    price_str = data.us_retail_price or ""
    await _simple_check(
        evaluator,
        price_node,
        "price_value_check",
        "Answer states a current US retail price and it is ≤ $750.",
        claim=f"The stated current US retail price '{price_str}' is less than or equal to $750 (USD).",
        add_ins=(
            "Judge based on the answer's stated price only. Ignore tax and shipping. "
            "If a price range is given, use the lowest figure to judge; if the price is not clearly ≤ $750, mark Incorrect."
        ),
        critical=True,
    )
    await _url_support_check(
        evaluator,
        price_node,
        "price_url_check",
        "Provide URL reference(s) supporting the stated US retail price.",
        claim=f"The console's current US retail price is {price_str}.",
        urls=data.price_urls,
        add_ins="Verify that the URL(s) reflect US pricing for the console model (MSRP or current retail).",
        critical=True,
    )

    # 5) VR compatibility requirement
    vr_node = evaluator.add_parallel(
        id="vr_compatibility_requirement",
        desc="Console is compatible with a VR headset currently available for purchase with price below $400.",
        parent=console_node,
        critical=True,
    )
    vr_name = data.vr_headset_name or ""
    evaluator.add_custom_node(
        result=_non_empty(data.vr_headset_name),
        id="vr_headset_name_provided",
        desc="Provide the compatible VR headset model name.",
        parent=vr_node,
        critical=True,
    )

    vr_price_str = data.vr_headset_price or ""
    await _simple_check(
        evaluator,
        vr_node,
        "vr_headset_price_check",
        "Answer states the VR headset price and it is < $400.",
        claim=f"The stated price for the VR headset '{vr_name}' is '{vr_price_str}', which is less than $400 (USD).",
        add_ins=(
            "Judge based on the answer's price text only. If the value is $400 or higher, or not clearly below $400, mark Incorrect."
        ),
        critical=True,
    )

    await _simple_check(
        evaluator,
        vr_node,
        "vr_headset_availability_check",
        "Answer indicates the VR headset is currently available for purchase.",
        claim=(
            "According to the answer text, the VR headset is currently available for purchase (e.g., in stock, available now)."
        ),
        add_ins="Check the answer text for an explicit statement of present availability. If absent or ambiguous, mark Incorrect.",
        critical=True,
    )

    await _simple_check(
        evaluator,
        vr_node,
        "vr_compatibility_check",
        "Answer indicates the VR headset is compatible with the identified console.",
        claim=f"According to the answer text, the VR headset '{vr_name}' is compatible with the console '{data.console_name or ''}'.",
        add_ins="Check that the answer explicitly states or clearly implies compatibility between the named console and VR headset.",
        critical=True,
    )

    await _url_support_check(
        evaluator,
        vr_node,
        "vr_price_url_check",
        "Provide URL reference(s) supporting the VR headset price claim.",
        claim=f"The price of the VR headset '{vr_name}' is {vr_price_str}.",
        urls=data.vr_price_urls,
        add_ins="Verify that the URL(s) show the listed price for the stated VR headset model.",
        critical=True,
    )

    await _url_support_check(
        evaluator,
        vr_node,
        "vr_availability_url_check",
        "Provide URL reference(s) supporting that the VR headset is currently available for purchase.",
        claim=f"The VR headset '{vr_name}' is currently available for purchase.",
        urls=data.vr_availability_urls,
        add_ins="Verify that the page indicates present purchase availability (e.g., add to cart, in stock, available now).",
        critical=True,
    )

    await _url_support_check(
        evaluator,
        vr_node,
        "vr_compatibility_url_check",
        "Provide URL reference(s) supporting VR headset compatibility with the console.",
        claim=f"The VR headset '{vr_name}' is compatible with the console '{data.console_name or ''}'.",
        urls=data.vr_compatibility_urls,
        add_ins="Verify that the page explicitly states compatibility between the named console and headset.",
        critical=True,
    )

    # 6) 4K @ 120 fps
    fourk_node = evaluator.add_parallel(
        id="4k_120fps_requirement",
        desc="Console supports 4K gaming at up to 120 fps.",
        parent=console_node,
        critical=True,
    )
    await _simple_check(
        evaluator,
        fourk_node,
        "4k120_value_check",
        "Answer states the console supports 4K output at up to 120 fps.",
        claim="According to the answer text, the console supports 4K gaming at up to 120 frames per second.",
        add_ins="If the answer doesn't clearly state 4K at up to 120 fps, mark Incorrect.",
        critical=True,
    )
    await _url_support_check(
        evaluator,
        fourk_node,
        "4k120_url_check",
        "Provide URL reference(s) supporting 4K/120fps support.",
        claim="The console supports 4K gaming at up to 120 frames per second.",
        urls=data.support_4k120_urls,
        add_ins="Verify that the specification page(s) mention 4K at 120 fps support.",
        critical=True,
    )

    # 7) Ray tracing (hardware-based)
    rt_node = evaluator.add_parallel(
        id="ray_tracing_requirement",
        desc="Console supports hardware-based ray tracing.",
        parent=console_node,
        critical=True,
    )
    await _simple_check(
        evaluator,
        rt_node,
        "ray_tracing_value_check",
        "Answer states the console supports hardware-based ray tracing.",
        claim="According to the answer text, the console supports hardware-based ray tracing.",
        add_ins="If the answer doesn't clearly state hardware ray tracing support, mark Incorrect.",
        critical=True,
    )
    await _url_support_check(
        evaluator,
        rt_node,
        "ray_tracing_url_check",
        "Provide URL reference(s) supporting ray tracing support.",
        claim="The console supports hardware-based ray tracing.",
        urls=data.ray_tracing_urls,
        add_ins="Verify that the specification or official page mentions hardware ray tracing.",
        critical=True,
    )

    # 8) Online subscription: ≤ $80/year
    sub_node = evaluator.add_parallel(
        id="online_subscription_requirement",
        desc="Annual online multiplayer subscription cost does not exceed $80, and subscription name/cost are provided.",
        parent=console_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(data.subscription_name),
        id="subscription_name_provided",
        desc="Provide the name of the online multiplayer subscription service.",
        parent=sub_node,
        critical=True,
    )
    sub_cost_str = data.subscription_annual_cost or ""
    await _simple_check(
        evaluator,
        sub_node,
        "subscription_annual_cost_check",
        "Answer states an annual subscription cost and it is ≤ $80.",
        claim=f"The annual cost for the online multiplayer subscription '{data.subscription_name or ''}' is '{sub_cost_str}', which is less than or equal to $80 (USD).",
        add_ins="Judge based on the answer's annual figure only. If monthly pricing is provided instead of annual, or the annual cost exceeds $80, mark Incorrect.",
        critical=True,
    )
    await _url_support_check(
        evaluator,
        sub_node,
        "subscription_url_check",
        "Provide URL reference(s) supporting the subscription pricing/annual cost.",
        claim=f"The annual cost of the online multiplayer subscription '{data.subscription_name or ''}' is {sub_cost_str} in the United States.",
        urls=data.subscription_urls,
        add_ins="Verify that the URL(s) provide an annual price consistent with the stated amount.",
        critical=True,
    )

    # 9) Release status and US availability
    rel_node = evaluator.add_parallel(
        id="release_status_requirement",
        desc="Console is released by November 30, 2024 and currently available for purchase in the United States; release date is provided.",
        parent=console_node,
        critical=True,
    )
    rel_date_str = data.release_date or ""
    await _simple_check(
        evaluator,
        rel_node,
        "release_date_provided_and_check",
        "Answer provides a release date and it is on or before November 30, 2024.",
        claim=f"The stated release date '{rel_date_str}' is on or before November 30, 2024.",
        add_ins=(
            "Judge based on the answer text. Interpret month/year formats reasonably (e.g., 'November 2024' counts as on or before Nov 30, 2024). "
            "If no release date is provided or it's after Nov 30, 2024, mark Incorrect."
        ),
        critical=True,
    )
    await _simple_check(
        evaluator,
        rel_node,
        "us_availability_check",
        "Answer indicates the console is currently available for purchase in the United States.",
        claim="According to the answer text, the console is currently available for purchase in the United States.",
        add_ins="If the answer does not explicitly state current US availability, mark Incorrect.",
        critical=True,
    )
    await _url_support_check(
        evaluator,
        rel_node,
        "release_date_url_check",
        "Provide URL reference(s) supporting the stated release date.",
        claim=f"The console's release date is {rel_date_str}.",
        urls=data.release_date_urls,
        add_ins="Verify that the page(s) indicate the stated release date for the console model.",
        critical=True,
    )
    await _url_support_check(
        evaluator,
        rel_node,
        "us_availability_url_check",
        "Provide URL reference(s) supporting that the console is currently available for purchase in the United States.",
        claim="The console is currently available for purchase in the United States.",
        urls=data.us_availability_urls,
        add_ins="Verify that the page(s) indicate current US retail availability (e.g., purchasable on US retailer or official US site).",
        critical=True,
    )

    # 10) HDR
    hdr_node = evaluator.add_parallel(
        id="hdr_support_requirement",
        desc="Console supports HDR.",
        parent=console_node,
        critical=True,
    )
    await _simple_check(
        evaluator,
        hdr_node,
        "hdr_value_check",
        "Answer states the console supports HDR output.",
        claim="According to the answer text, the console supports HDR (High Dynamic Range) output.",
        add_ins="If the answer does not clearly state HDR support, mark Incorrect.",
        critical=True,
    )
    await _url_support_check(
        evaluator,
        hdr_node,
        "hdr_url_check",
        "Provide URL reference(s) supporting HDR support.",
        claim="The console supports HDR (High Dynamic Range).",
        urls=data.hdr_urls,
        add_ins="Verify that the linked page explicitly mentions HDR support.",
        critical=True,
    )

    # 11) VRR
    vrr_node = evaluator.add_parallel(
        id="vrr_support_requirement",
        desc="Console supports Variable Refresh Rate (VRR).",
        parent=console_node,
        critical=True,
    )
    await _simple_check(
        evaluator,
        vrr_node,
        "vrr_value_check",
        "Answer states the console supports VRR.",
        claim="According to the answer text, the console supports Variable Refresh Rate (VRR).",
        add_ins="If the answer does not clearly state VRR support, mark Incorrect.",
        critical=True,
    )
    await _url_support_check(
        evaluator,
        vrr_node,
        "vrr_url_check",
        "Provide URL reference(s) supporting VRR support.",
        claim="The console supports Variable Refresh Rate (VRR).",
        urls=data.vrr_urls,
        add_ins="Verify that the linked page explicitly mentions VRR support.",
        critical=True,
    )

    # 12) HDMI 2.1
    hdmi_node = evaluator.add_parallel(
        id="hdmi_2_1_requirement",
        desc="Console supports HDMI 2.1 connectivity.",
        parent=console_node,
        critical=True,
    )
    await _simple_check(
        evaluator,
        hdmi_node,
        "hdmi_value_check",
        "Answer states the console supports HDMI 2.1.",
        claim="According to the answer text, the console supports HDMI 2.1 connectivity.",
        add_ins="If the answer does not clearly state HDMI 2.1 support, mark Incorrect.",
        critical=True,
    )
    await _url_support_check(
        evaluator,
        hdmi_node,
        "hdmi_url_check",
        "Provide URL reference(s) supporting HDMI 2.1 support.",
        claim="The console supports HDMI 2.1 connectivity.",
        urls=data.hdmi_urls,
        add_ins="Verify that the page(s) explicitly mention HDMI 2.1.",
        critical=True,
    )

    # 13) Backward compatibility
    backcompat_node = evaluator.add_parallel(
        id="backward_compatibility_requirement",
        desc="Console supports backward compatibility with games from the previous generation of the same console family.",
        parent=console_node,
        critical=True,
    )
    await _simple_check(
        evaluator,
        backcompat_node,
        "backcompat_value_check",
        "Answer states the console supports backward compatibility with the previous generation in the same family.",
        claim="According to the answer text, the console supports backward compatibility with games from the previous generation of the same console family.",
        add_ins="If the answer does not clearly state backward compatibility with the prior generation, mark Incorrect.",
        critical=True,
    )
    await _url_support_check(
        evaluator,
        backcompat_node,
        "backcompat_url_check",
        "Provide URL reference(s) supporting the backward compatibility claim.",
        claim="The console supports backward compatibility with games from the previous generation of the same console family.",
        urls=data.backcompat_urls,
        add_ins="Verify that the page(s) explicitly mention backward compatibility with previous generation titles.",
        critical=True,
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
    Evaluate an answer for the console selection task (November 2024 constraints).
    """
    # Initialize evaluator with a parallel root (allows modular additions if needed)
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

    # Extract structured info from the answer
    extracted: ConsoleSelection = await evaluator.extract(
        prompt=prompt_extract_console_selection(),
        template_class=ConsoleSelection,
        extraction_name="console_selection_extraction",
    )

    # Build verification tree and run checks
    await verify_console_requirements(evaluator, root, extracted)

    # Return summary with tree and scores
    return evaluator.get_summary()