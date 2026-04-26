import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "console_2025_requirements"
TASK_DESCRIPTION = (
    "Identify a gaming console that was released in 2025, features a built-in screen measuring between 7.5 and 8.5 inches "
    "diagonally, supports a native screen resolution of at least 1920x1080 pixels, has at least 12GB of RAM, includes internal "
    "storage of at least 256GB, supports expandable storage via memory cards, offers backward compatibility with games from its "
    "predecessor console, and has an official tech specs page published on the manufacturer's website."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ConsoleSpecsExtraction(BaseModel):
    console_name: Optional[str] = None
    manufacturer: Optional[str] = None
    release_date: Optional[str] = None
    screen_size_inches: Optional[str] = None
    resolution: Optional[str] = None
    ram: Optional[str] = None
    internal_storage: Optional[str] = None
    expandable_storage: Optional[str] = None
    backward_compatibility: Optional[str] = None
    predecessor_console: Optional[str] = None
    official_specs_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_console_specs() -> str:
    return """
    Extract information about a single gaming console mentioned in the answer that the answer claims meets the specified requirements.
    If multiple consoles are mentioned, extract the first one that the answer uses as the candidate (or the first console mentioned).

    Return a JSON object with the following fields (strings unless noted):
    - console_name: The console's official name as stated in the answer.
    - manufacturer: The manufacturer's name (e.g., Nintendo, Sony, Microsoft, Valve).
    - release_date: The stated public release/launch date (or date range/month/year) for the console.
    - screen_size_inches: The stated built-in screen diagonal size (e.g., "8.0 inches", "8-inch").
    - resolution: The stated native resolution of the built-in screen (e.g., "1920x1080", "1080p").
    - ram: The stated memory amount (e.g., "12GB", "16 GB LPDDR5").
    - internal_storage: The stated internal storage capacity (e.g., "256 GB", "512GB SSD").
    - expandable_storage: The stated expandable storage support description (e.g., "microSD/SDXC slot", "supports memory cards").
    - backward_compatibility: A description of backward compatibility (e.g., "plays Nintendo Switch games", "BC with PS4 titles").
    - predecessor_console: The name of the predecessor console if explicitly stated (e.g., "Nintendo Switch", "PlayStation 4").
    - official_specs_url: The URL for the official technical specifications page on the manufacturer's own website (NOT a third-party site). If none is explicitly provided in the answer, set to null.
    - additional_urls: An array of any other URLs explicitly provided in the answer for this console (press releases, product pages, support pages, etc.). Only include valid URLs that appear in the answer.

    Rules:
    - Do NOT invent or infer any URLs; only extract those explicitly present in the answer text.
    - Do NOT normalize values beyond what is in the answer; keep the exact strings (including units) as written.
    - If any requested field is not mentioned in the answer, set it to null (or empty list for additional_urls).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_nonempty(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if u and isinstance(u, str):
            s = u.strip()
            if s and s not in seen:
                out.append(s)
                seen.add(s)
    return out


def _base_domain(netloc: str) -> str:
    host = netloc.split(":")[0].lower()
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])  # naive base (e.g., sony.com, nintendo.com)
    return host


def _extract_base_domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        netloc = urlparse(url).netloc
        return _base_domain(netloc)
    except Exception:
        return None


def _filter_official_urls(candidates: List[str], official_url: Optional[str]) -> List[str]:
    """
    Keep only URLs that appear to be on the same base domain as the official URL.
    If no official URL is provided, return empty list (we'll require official evidence).
    """
    base = _extract_base_domain(official_url)
    if not base:
        return []
    out = []
    for u in candidates:
        try:
            nl = urlparse(u).netloc
            if _base_domain(nl).endswith(base):
                out.append(u)
        except Exception:
            pass
    return _unique_nonempty(out)


def _official_sources(specs: ConsoleSpecsExtraction) -> List[str]:
    """
    Build a list of official sources: the official specs URL (if present) plus any additional URLs
    that share the same base domain.
    """
    primary = _unique_nonempty([specs.official_specs_url])
    official_extras = _filter_official_urls(specs.additional_urls or [], specs.official_specs_url)
    return _unique_nonempty(primary + official_extras)


def _ins_require_official(specs: ConsoleSpecsExtraction, extra_rules: str = "") -> str:
    """
    Build a strict additional instruction requiring official manufacturer evidence.
    """
    base = _extract_base_domain(specs.official_specs_url) or "the manufacturer's official domain"
    preface = (
        f"Only consider the claim supported if it is explicitly stated on an official manufacturer webpage. "
        f"Prioritize the official technical specifications page. A valid official page should be hosted on '{base}'. "
        f"If no valid official URL is provided or the page content does not clearly state the claim, mark it as NOT supported."
    )
    if extra_rules:
        return preface + " " + extra_rules
    return preface


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_requirements(
    evaluator: Evaluator,
    parent_node,
    specs: ConsoleSpecsExtraction,
) -> None:
    """
    Build the verification subtree according to the rubric and execute verifications.
    """
    node_all = evaluator.add_parallel(
        id="Console_Meeting_All_Requirements",
        desc="The identified console meets all specified technical and release requirements",
        parent=parent_node,
        critical=True,
    )

    # Prepare nodes (all critical leaves under a critical parent)
    n_release = evaluator.add_leaf(
        id="Release_Date_2025",
        desc="The console was released between January 1, 2025 and December 31, 2025 (inclusive)",
        parent=node_all,
        critical=True,
    )
    n_screen = evaluator.add_leaf(
        id="Screen_Size_7.5_to_8.5_Inches",
        desc="The console has a built-in screen with diagonal size between 7.5 and 8.5 inches",
        parent=node_all,
        critical=True,
    )
    n_resolution = evaluator.add_leaf(
        id="Resolution_1080p_or_Higher",
        desc="The console supports native screen resolution of at least 1920x1080 pixels",
        parent=node_all,
        critical=True,
    )
    n_ram = evaluator.add_leaf(
        id="RAM_12GB_or_More",
        desc="The console has at least 12GB of RAM",
        parent=node_all,
        critical=True,
    )
    n_storage = evaluator.add_leaf(
        id="Internal_Storage_256GB_or_More",
        desc="The console has internal storage of at least 256GB",
        parent=node_all,
        critical=True,
    )
    n_expand = evaluator.add_leaf(
        id="Expandable_Storage_Support",
        desc="The console supports expandable storage via memory cards",
        parent=node_all,
        critical=True,
    )
    n_bc = evaluator.add_leaf(
        id="Backward_Compatibility",
        desc="The console supports backward compatibility with games from its predecessor console",
        parent=node_all,
        critical=True,
    )
    n_specs = evaluator.add_leaf(
        id="Official_Tech_Specs_Page",
        desc="The console has an official tech specs page on the manufacturer's website",
        parent=node_all,
        critical=True,
    )

    # Build claims
    cname = specs.console_name or "the console"
    predecessor = specs.predecessor_console or "its predecessor console"

    claim_release = f"{cname} was released in calendar year 2025 (between January 1, 2025 and December 31, 2025 inclusive)."
    claim_screen = f"{cname} includes a built-in screen measuring between 7.5 inches and 8.5 inches diagonally."
    claim_resolution = f"The built-in screen of {cname} natively supports at least 1920×1080 (1080p) resolution."
    claim_ram = f"{cname} has at least 12 GB of RAM (system memory)."
    claim_storage = f"{cname} includes internal storage of at least 256 GB."
    claim_expand = f"{cname} supports expandable storage via memory cards (such as microSD/SDHC/SDXC)."
    claim_bc = f"{cname} is backward compatible with games from {predecessor}."
    claim_specs = f"This webpage is the official technical specifications page for {cname} hosted on the manufacturer's website."

    # Sources: require official manufacturer evidence whenever possible
    official_sources = _official_sources(specs)
    sources_common = official_sources if official_sources else None  # If empty, force strict instruction to fail

    # Additional instructions per check
    ins_release = _ins_require_official(
        specs,
        "Treat 'release' as the public commercial launch. If multiple regions/dates exist, at least one official release date in 2025 counts."
    )
    ins_screen = _ins_require_official(
        specs,
        "Focus on the built-in handheld screen diagonal. Accept 7.5–8.5 inches inclusive. Ignore external displays."
    )
    ins_resolution = _ins_require_official(
        specs,
        "The threshold is native resolution of the built-in screen. Do NOT use docked/external display or upscaled output to satisfy this."
    )
    ins_ram = _ins_require_official(
        specs,
        "Accept synonyms such as 'memory', 'system memory', 'LPDDR' etc. Value must be >= 12 GB."
    )
    ins_storage = _ins_require_official(
        specs,
        "Count only internal storage (e.g., eMMC/SSD) included with the device; exclude removable/memory cards."
    )
    ins_expand = _ins_require_official(
        specs,
        "Look for explicit mention of memory card support (e.g., microSD/SDXC). USB storage alone does not count."
    )
    ins_bc = _ins_require_official(
        specs,
        "Backwards compatibility should mean the console can play prior generation games (natively or officially supported). "
        "Cloud streaming alone does not count as backward compatibility."
    )

    # Official tech specs page requires the specific URL; if it's missing, the instruction enforces a fail
    ins_specs = _ins_require_official(
        specs,
        "This must be a 'Specifications' or 'Tech Specs' style page on the official manufacturer domain. "
        "Mark as NOT supported if the URL is missing or is not on the official domain."
    )
    specs_url_only = specs.official_specs_url if (specs.official_specs_url and isinstance(specs.official_specs_url, str)) else None

    # Prepare batch verifications
    claims_and_sources = [
        (claim_release, sources_common, n_release, ins_release),
        (claim_screen, sources_common, n_screen, ins_screen),
        (claim_resolution, sources_common, n_resolution, ins_resolution),
        (claim_ram, sources_common, n_ram, ins_ram),
        (claim_storage, sources_common, n_storage, ins_storage),
        (claim_expand, sources_common, n_expand, ins_expand),
        (claim_bc, sources_common, n_bc, ins_bc),
        (claim_specs, specs_url_only, n_specs, ins_specs),
    ]

    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the 2025 console requirements task.
    """
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

    # Extract structured console specs from the answer
    specs = await evaluator.extract(
        prompt=prompt_extract_console_specs(),
        template_class=ConsoleSpecsExtraction,
        extraction_name="console_specs_extraction",
    )

    # Record some helper info for transparency
    evaluator.add_custom_info(
        info={
            "console_name": specs.console_name,
            "manufacturer": specs.manufacturer,
            "official_specs_url": specs.official_specs_url,
            "additional_urls_count": len(specs.additional_urls or []),
            "official_sources_used": _official_sources(specs),
        },
        info_type="diagnostics",
        info_name="extraction_diagnostics",
    )

    # Build verification tree and run checks
    await build_and_verify_requirements(evaluator, root, specs)

    return evaluator.get_summary()