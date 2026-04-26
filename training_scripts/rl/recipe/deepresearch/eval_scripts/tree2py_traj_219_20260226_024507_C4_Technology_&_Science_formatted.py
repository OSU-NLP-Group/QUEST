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
TASK_ID = "dual_display_rtx5090_ces2026"
TASK_DESCRIPTION = (
    "In 2026, one gaming laptop manufacturer unveiled a unique dual-display laptop featuring the NVIDIA GeForce RTX 5090 "
    "Laptop GPU at CES 2026. This laptop stands out as having two full-size OLED display panels, each with high refresh rates "
    "and high resolution.\n\nIdentify this specific laptop model and provide the following information:\n"
    "1. The manufacturer name and complete model name\n"
    "2. Confirmation that it features the NVIDIA GeForce RTX 5090 Laptop GPU\n"
    "3. Confirmation that it has dual OLED display panels\n"
    "4. The refresh rate of the displays (must be at least 120Hz)\n"
    "5. The resolution of the displays (must be 3K or higher)\n"
    "6. Confirmation of Thunderbolt 4 or Thunderbolt 5 connectivity\n"
    "7. The specified TGP (Total Graphics Power) value for the RTX 5090 GPU\n"
    "8. Confirmation that it was announced at CES 2026\n"
    "9. A reference URL from an official manufacturer source or credible technology publication"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LaptopExtraction(BaseModel):
    """Information about the identified dual-display gaming laptop from the answer."""
    manufacturer: Optional[str] = None
    model_name: Optional[str] = None  # Complete model name as given in the answer
    gpu_name: Optional[str] = None  # As mentioned in the answer (e.g., 'NVIDIA GeForce RTX 5090 Laptop GPU')
    dual_oled_statement: Optional[str] = None  # Any phrase confirming dual OLED panels
    refresh_rate: Optional[str] = None  # e.g., '120Hz', '240 Hz', 'up to 240Hz'
    resolution: Optional[str] = None  # e.g., '3K', '2880x1800', '3840×2160', '3.2K'
    thunderbolt_statement: Optional[str] = None  # e.g., 'Thunderbolt 5', 'Thunderbolt 4'
    tgp: Optional[str] = None  # e.g., '175W', '155 W', 'up to 175 W'
    announcement_event: Optional[str] = None  # e.g., 'CES 2026', 'Consumer Electronics Show 2026'
    reference_urls: List[str] = Field(default_factory=list)  # all URLs cited in the answer for this laptop


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_laptop_info() -> str:
    return """
    You must extract details for the ONE specific laptop in the answer that matches ALL of these criteria:
    – A gaming laptop with TWO full-size OLED display panels (dual OLED displays).
    – Uses the NVIDIA GeForce RTX 5090 Laptop GPU.
    – Was announced at CES 2026.

    If the answer mentions multiple laptops, select the one that best matches all of the above and extract fields for that single model only.

    Extract the following fields from the answer text exactly as written:
    1) manufacturer: The manufacturer's name (e.g., "ASUS", "Razer", "Lenovo").
    2) model_name: The complete model name as presented (include series/submodel if present).
    3) gpu_name: The GPU name string as stated in the answer (e.g., "NVIDIA GeForce RTX 5090 Laptop GPU").
    4) dual_oled_statement: A short phrase or sentence from the answer confirming dual OLED displays (if present).
    5) refresh_rate: The refresh rate mention relevant to the displays (e.g., "120Hz", "240 Hz", "up to 240Hz"). If multiple values are present, prefer the one that applies to the OLED panels.
    6) resolution: The resolution mentioned for the displays (e.g., "3K", "2880x1800", "3200×2000", "4K"). Prefer the resolution describing each OLED panel. If multiple, pick the one most clearly describing the main panels.
    7) thunderbolt_statement: Any mention of Thunderbolt 4 or Thunderbolt 5 connectivity (e.g., "Thunderbolt 5", "TB5", "Thunderbolt 4").
    8) tgp: A specific TGP value for the RTX 5090 Laptop GPU (e.g., "175W", "up to 175 W", "Max TGP 175W"). Include units and qualifiers if present.
    9) announcement_event: The event where it was announced, if mentioned (e.g., "CES 2026", "Consumer Electronics Show 2026").
    10) reference_urls: An array of all URLs explicitly present in the answer that refer to this laptop (manufacturer pages or credible tech publications). Extract actual URLs only (plain URLs or markdown links), deduplicate exact duplicates.

    Rules:
    – Do NOT invent values that are not in the answer.
    – If a field is missing in the answer, set it to null (for strings) or an empty array (for reference_urls).
    – For URLs, include only valid ones; ensure they start with http:// or https://. If a URL lacks protocol, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_valid_http_url(url: Optional[str]) -> bool:
    if not url:
        return False
    url = url.strip()
    return url.startswith("http://") or url.startswith("https://")


def _valid_urls(urls: List[str]) -> List[str]:
    return [u.strip() for u in urls if _is_valid_http_url(u)]


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: LaptopExtraction) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    All checks are placed under a single critical parallel aggregator so that any failure fails the task.
    """
    # Critical aggregator node: any failed child fails the whole group
    crit_root = evaluator.add_parallel(
        id="all_criteria",
        desc="All critical criteria for dual-display RTX 5090 CES 2026 laptop identification must be satisfied",
        parent=evaluator.root,
        critical=True
    )

    # Normalize sources list
    sources_list = _valid_urls(extracted.reference_urls)

    # ------------------------------------------------------------------ #
    # Reference URL checks (group): existence + credibility              #
    # ------------------------------------------------------------------ #
    ref_group = evaluator.add_parallel(
        id="reference_url_group",
        desc="Reference URL(s) availability and credibility",
        parent=crit_root,
        critical=True
    )

    # Existence check: At least one valid reference URL provided in the answer
    has_ref_url = len(sources_list) > 0
    ref_provided_node = evaluator.add_custom_node(
        result=has_ref_url,
        id="reference_url_provided",
        desc="At least one valid reference URL is provided in the answer",
        parent=ref_group,
        critical=True
    )

    # Credibility check: Official manufacturer page OR credible tech publication
    ref_credible_node = evaluator.add_leaf(
        id="reference_url",
        desc="A valid reference URL from an official manufacturer source or credible technology publication is provided",
        parent=ref_group,
        critical=True
    )
    cred_claim = (
        "This page is either an official manufacturer webpage for the product (domain owned by the manufacturer) "
        "or an article by a widely recognized credible technology publication."
    )
    cred_add_ins = (
        "Judge credibility by domain and site identity. Manufacturer examples: asus.com, lenovo.com, razer.com, msi.com, acer.com, hp.com, dell.com. "
        "Credible tech publications include outlets such as The Verge, Engadget, Tom's Hardware, AnandTech, Notebookcheck, PC Gamer, "
        "TechRadar, CNET, Ars Technica, Digital Trends, PCWorld, IGN, T3, etc. Marketing aggregators, small blogs, or generic retailers alone are not sufficient."
    )
    await evaluator.verify(
        claim=cred_claim,
        node=ref_credible_node,
        sources=sources_list,
        additional_instruction=cred_add_ins,
        extra_prerequisites=[ref_provided_node]
    )

    # ------------------------------------------------------------------ #
    # Manufacturer + Model (group): presence + page confirmation         #
    # ------------------------------------------------------------------ #
    model_group = evaluator.add_parallel(
        id="manufacturer_model_group",
        desc="Manufacturer and complete model name are provided and correct",
        parent=crit_root,
        critical=True
    )

    # Existence of manufacturer and model in the answer
    has_model_info = bool((extracted.manufacturer or "").strip()) and bool((extracted.model_name or "").strip())
    model_exist_node = evaluator.add_custom_node(
        result=has_model_info,
        id="manufacturer_model_provided",
        desc="Manufacturer and complete model name are provided in the answer",
        parent=model_group,
        critical=True
    )

    # Verify that the referenced page(s) clearly name the product as Manufacturer + Model
    manu_model_leaf = evaluator.add_leaf(
        id="manufacturer_and_model",
        desc="The correct manufacturer name and complete laptop model name featuring dual OLED displays with RTX 5090 announced at CES 2026 is provided",
        parent=model_group,
        critical=True
    )
    man = extracted.manufacturer or ""
    mdl = extracted.model_name or ""
    manu_model_claim = (
        f"This page describes a laptop model from {man.strip()} with the model name '{mdl.strip()}'."
        if man or mdl else
        "This page clearly states the manufacturer and complete model name of the dual-display gaming laptop."
    )
    manu_model_add_ins = (
        "Allow minor naming variations, punctuation, and capitalization differences. "
        "If a family/series and submodel are used together (e.g., 'ROG Zephyrus Duo 16 (2026)'), it should still count as the correct complete name."
    )
    await evaluator.verify(
        claim=manu_model_claim,
        node=manu_model_leaf,
        sources=sources_list,
        additional_instruction=manu_model_add_ins,
        extra_prerequisites=[ref_provided_node, model_exist_node]
    )

    # ------------------------------------------------------------------ #
    # RTX 5090 GPU                                                       #
    # ------------------------------------------------------------------ #
    rtx_leaf = evaluator.add_leaf(
        id="rtx_5090_gpu",
        desc="The laptop is confirmed to feature NVIDIA GeForce RTX 5090 Laptop GPU",
        parent=crit_root,
        critical=True
    )
    rtx_claim = "This laptop features the NVIDIA GeForce RTX 5090 Laptop GPU."
    await evaluator.verify(
        claim=rtx_claim,
        node=rtx_leaf,
        sources=sources_list,
        additional_instruction="Look for explicit mentions like 'GeForce RTX 5090 Laptop GPU', 'NVIDIA RTX 5090 (Laptop)'. Desktop 5090 alone is not acceptable.",
        extra_prerequisites=[ref_provided_node]
    )

    # ------------------------------------------------------------------ #
    # Dual OLED displays                                                 #
    # ------------------------------------------------------------------ #
    dual_oled_leaf = evaluator.add_leaf(
        id="dual_oled_displays",
        desc="The laptop features dual OLED display panels",
        parent=crit_root,
        critical=True
    )
    dual_oled_claim = "This laptop has two full-size OLED display panels (dual OLED displays)."
    dual_oled_ins = (
        "Accept phrasings such as 'dual OLED', 'two OLED displays', 'two full-size OLED panels'. "
        "A small secondary strip or a narrow touch bar does not count; both panels should be full-size displays."
    )
    await evaluator.verify(
        claim=dual_oled_claim,
        node=dual_oled_leaf,
        sources=sources_list,
        additional_instruction=dual_oled_ins,
        extra_prerequisites=[ref_provided_node]
    )

    # ------------------------------------------------------------------ #
    # Refresh rate (>= 120Hz)                                            #
    # ------------------------------------------------------------------ #
    refresh_leaf = evaluator.add_leaf(
        id="refresh_rate",
        desc="Each OLED display has a refresh rate of at least 120Hz",
        parent=crit_root,
        critical=True
    )
    refresh_claim = "Each of the two OLED displays supports at least a 120Hz refresh rate."
    refresh_ins = (
        "Check that both OLED panels are 120Hz or higher. 'Up to 120Hz/240Hz' counts if it applies to the OLED panels. "
        "If the two panels have different refresh rates, the lower must still be >= 120Hz."
    )
    await evaluator.verify(
        claim=refresh_claim,
        node=refresh_leaf,
        sources=sources_list,
        additional_instruction=refresh_ins,
        extra_prerequisites=[ref_provided_node, dual_oled_leaf]
    )

    # ------------------------------------------------------------------ #
    # Resolution (>= 3K)                                                 #
    # ------------------------------------------------------------------ #
    res_leaf = evaluator.add_leaf(
        id="display_resolution",
        desc="The displays have 3K resolution or higher",
        parent=crit_root,
        critical=True
    )
    res_claim = "Each of the two OLED displays has a resolution that is 3K or higher."
    res_ins = (
        "Treat '3K' as ~2880 horizontal pixels or greater. Accept examples like 2880×1800, 3.2K, 3200×2000, 3840×2160 (4K). "
        "If the two panels have different resolutions, the lower must still be at least 3K."
    )
    await evaluator.verify(
        claim=res_claim,
        node=res_leaf,
        sources=sources_list,
        additional_instruction=res_ins,
        extra_prerequisites=[ref_provided_node, dual_oled_leaf]
    )

    # ------------------------------------------------------------------ #
    # Thunderbolt 4 or 5                                                 #
    # ------------------------------------------------------------------ #
    tb_leaf = evaluator.add_leaf(
        id="thunderbolt_connectivity",
        desc="The laptop includes Thunderbolt 4 or Thunderbolt 5 connectivity",
        parent=crit_root,
        critical=True
    )
    tb_claim = "The laptop includes Thunderbolt 4 or Thunderbolt 5 connectivity."
    tb_ins = (
        "Accept phrases like 'Thunderbolt 5', 'TB5', 'Thunderbolt 4', or ports specified as Thunderbolt 4/5 over USB-C. "
        "Generic USB-C alone without Thunderbolt mention does not satisfy this."
    )
    await evaluator.verify(
        claim=tb_claim,
        node=tb_leaf,
        sources=sources_list,
        additional_instruction=tb_ins,
        extra_prerequisites=[ref_provided_node]
    )

    # ------------------------------------------------------------------ #
    # TGP value (presence and verification)                              #
    # ------------------------------------------------------------------ #
    tgp_group = evaluator.add_parallel(
        id="tgp_group",
        desc="TGP value presence and verification",
        parent=crit_root,
        critical=True
    )
    has_tgp = bool((extracted.tgp or "").strip())
    tgp_exist_node = evaluator.add_custom_node(
        result=has_tgp,
        id="tgp_value_provided",
        desc="A specific TGP (Total Graphics Power) value for the RTX 5090 GPU is provided in the answer",
        parent=tgp_group,
        critical=True
    )

    tgp_leaf = evaluator.add_leaf(
        id="tgp_value",
        desc="A specific TGP (Total Graphics Power) value for the RTX 5090 GPU is provided",
        parent=tgp_group,
        critical=True
    )
    if has_tgp:
        tgp_claim = (
            f"This page specifies a TGP (Total Graphics Power) value for the NVIDIA GeForce RTX 5090 Laptop GPU used in this laptop, "
            f"specifically '{(extracted.tgp or '').strip()}'. Minor formatting variations are acceptable."
        )
        tgp_ins = (
            "Look for 'TGP', 'Max TGP', 'Total Graphics Power', or phrasing like 'up to 175W'. "
            "Accept presence of a numeric wattage (e.g., 155W, 175W) explicitly tied to the RTX 5090 Laptop GPU in this laptop."
        )
    else:
        tgp_claim = (
            "This page specifies a numeric TGP (Total Graphics Power) value for the NVIDIA GeForce RTX 5090 Laptop GPU used in this laptop."
        )
        tgp_ins = (
            "Look for 'TGP', 'Max TGP', 'Total Graphics Power', or phrasing like 'up to XXXW'. "
            "There must be an explicit numeric wattage value tied to the RTX 5090 Laptop GPU."
        )
    await evaluator.verify(
        claim=tgp_claim,
        node=tgp_leaf,
        sources=sources_list,
        additional_instruction=tgp_ins,
        extra_prerequisites=[ref_provided_node, tgp_exist_node]
    )

    # ------------------------------------------------------------------ #
    # CES 2026 announcement                                              #
    # ------------------------------------------------------------------ #
    ces_leaf = evaluator.add_leaf(
        id="ces_2026_announcement",
        desc="The laptop was announced at CES 2026",
        parent=crit_root,
        critical=True
    )
    ces_claim = "This product was announced at CES 2026 (Consumer Electronics Show 2026)."
    ces_ins = (
        "Accept variants such as 'announced at CES 2026', 'revealed during CES 2026', "
        "'debuted at the Consumer Electronics Show 2026', or similar clear phrasing."
    )
    await evaluator.verify(
        claim=ces_claim,
        node=ces_leaf,
        sources=sources_list,
        additional_instruction=ces_ins,
        extra_prerequisites=[ref_provided_node]
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
    Evaluate an answer for the dual-display RTX 5090 CES 2026 laptop identification task.
    """
    # Initialize evaluator with a parallel root; we will add a critical aggregator under it
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract laptop info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_laptop_info(),
        template_class=LaptopExtraction,
        extraction_name="laptop_info",
    )

    # Build verification tree and run verifications
    await build_and_verify_tree(evaluator, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()