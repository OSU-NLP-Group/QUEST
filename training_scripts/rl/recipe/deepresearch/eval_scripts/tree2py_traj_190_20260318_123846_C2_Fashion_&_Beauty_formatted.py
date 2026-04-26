import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "actress_constraints_1996_2014_2024_2026_rhode"
TASK_DESCRIPTION = """
Identify the actress who was born in 1996, had her first television appearance in a show that premiered in 2014, made her Broadway debut in 2024, and announced her first beauty brand partnership in March 2026 with Rhode, a skincare and cosmetics brand founded by Hailey Bieber.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BioInfo(BaseModel):
    birth_year: Optional[str] = None
    bio_sources: List[str] = Field(default_factory=list)


class FirstTVInfo(BaseModel):
    show_title: Optional[str] = None
    premiered_year: Optional[str] = None
    claim_first_tv: Optional[str] = None  # e.g., "first TV appearance", "television debut", etc.
    tv_sources: List[str] = Field(default_factory=list)


class BroadwayInfo(BaseModel):
    debut_year: Optional[str] = None
    production_title: Optional[str] = None
    role: Optional[str] = None
    broadway_sources: List[str] = Field(default_factory=list)


class BeautyPartnershipInfo(BaseModel):
    announced_date_text: Optional[str] = None  # e.g., "March 2026"
    month: Optional[str] = None               # e.g., "March"
    year: Optional[str] = None                # e.g., "2026"
    brand_name: Optional[str] = None          # e.g., "Rhode"
    brand_founder: Optional[str] = None       # e.g., "Hailey Bieber"
    is_first_beauty_campaign: Optional[str] = None  # "yes" / "no" / or phrasing from the answer
    partnership_sources: List[str] = Field(default_factory=list)


class ActressExtraction(BaseModel):
    actress_name: Optional[str] = None
    alt_names: List[str] = Field(default_factory=list)

    bio: Optional[BioInfo] = None
    first_tv: Optional[FirstTVInfo] = None
    broadway: Optional[BroadwayInfo] = None
    beauty: Optional[BeautyPartnershipInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_actress_info() -> str:
    return """
    From the answer, extract the identified actress and the supporting details for the following constraints.
    Extract exactly what the answer states; do not infer or add missing information. Include URLs explicitly cited in the answer.

    Fields to extract:
    - actress_name: The explicitly named actress (string). If no single clear actress is identified, return null.
    - alt_names: All alternate spellings or stage names for the same actress mentioned in the answer (array of strings). If none, return [].

    - bio:
        - birth_year: The year of birth stated in the answer for the actress (string, e.g., "1996"). If not stated, return null.
        - bio_sources: All URLs provided that support the birth year claim. If none, return [].

    - first_tv:
        - show_title: The title of the show the answer claims was the actress's first TV appearance (string). If not given, return null.
        - premiered_year: The year the answer claims that show premiered (string). If not given, return null.
        - claim_first_tv: The exact phrase used in the answer indicating it was the first television appearance (string). If not present, return null.
        - tv_sources: All URLs provided that support the first TV appearance and/or premiere year. If none, return [].

    - broadway:
        - debut_year: The year of the actress's Broadway debut as claimed in the answer (string). If missing, return null.
        - production_title: The production title for the Broadway debut if mentioned (string). If missing, return null.
        - role: The role played if mentioned (string). If missing, return null.
        - broadway_sources: All URLs provided that support the Broadway debut claim. If none, return [].

    - beauty:
        - announced_date_text: The announced date text for the beauty partnership (e.g., "March 2026") exactly as written in the answer (string). If missing, return null.
        - month: The month component if specified (e.g., "March") (string or null).
        - year: The year component if specified (e.g., "2026") (string or null).
        - brand_name: The beauty brand’s name (e.g., "Rhode") (string or null).
        - brand_founder: The brand founder’s name if provided (e.g., "Hailey Bieber") (string or null).
        - is_first_beauty_campaign: The answer’s explicit assertion that this was the actress’s "first" beauty brand campaign/partnership (string; store the phrase like "first beauty campaign" or "debut beauty campaign" if present; else null).
        - partnership_sources: All URLs provided that support the beauty partnership details. If none, return [].

    SPECIAL URL RULES:
    - Only include URLs that actually appear in the answer (including markdown links).
    - If the answer references a source without a URL (e.g., "according to Wikipedia") and provides no URL, do not invent one; return [] for that sources field.

    Return a single JSON object with the schema above.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _safe_sources(urls: Optional[List[str]]) -> Optional[List[str]]:
    if urls and isinstance(urls, list):
        cleaned = [u for u in urls if _nonempty_str(u)]
        return cleaned if cleaned else None
    return None


def require_url_support_instruction(extra: str = "") -> str:
    base = (
        "Only mark the claim as Correct if it is explicitly supported by the provided URL source(s). "
        "If no valid URL is provided, or the URL(s) do not clearly support the claim, mark it as Incorrect. "
        "Do not rely on your own knowledge; rely on the cited webpage content."
    )
    if extra:
        return base + " " + extra
    return base


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: ActressExtraction) -> None:
    # Top-level critical sequential node (the rubric root)
    root_crit = evaluator.add_sequential(
        id="actress_identification",
        desc="Identify the actress and verify she satisfies all specified career and partnership constraints",
        parent=evaluator.root,
        critical=True,
    )

    # 1) Identity provided (critical)
    actress_name = extracted.actress_name or ""
    identity_exists = _nonempty_str(actress_name)
    evaluator.add_custom_node(
        result=identity_exists,
        id="provides_actress_identity",
        desc="Response explicitly names/identifies a specific actress",
        parent=root_crit,
        critical=True,
    )

    # 2) Constraint verification (critical, parallel)
    constraints = evaluator.add_parallel(
        id="constraint_verification",
        desc="Verify the identified actress meets all stated constraints",
        parent=root_crit,
        critical=True,
    )

    # 2.a) Birth year 1996 (critical)
    birth_leaf = evaluator.add_leaf(
        id="birth_year_1996",
        desc="The actress was born in 1996",
        parent=constraints,
        critical=True,
    )
    birth_sources = _safe_sources(extracted.bio.bio_sources) if extracted.bio else None
    birth_claim = f"{actress_name} was born in 1996."
    await evaluator.verify(
        claim=birth_claim,
        node=birth_leaf,
        sources=birth_sources,
        additional_instruction=require_url_support_instruction(
            "Look for explicit birth year statements on authoritative biography pages."
        ),
    )

    # 2.b) First TV appearance in a show that premiered in 2014 (critical)
    tv_leaf = evaluator.add_leaf(
        id="first_tv_appearance_show_premiered_2014",
        desc="The actress's first television appearance was in a show that premiered in 2014",
        parent=constraints,
        critical=True,
    )
    tv_sources = _safe_sources(extracted.first_tv.tv_sources) if extracted.first_tv else None
    show_title = extracted.first_tv.show_title if extracted.first_tv else None
    tv_claim = (
        f"{actress_name}'s first television appearance was in the show '{show_title}', which premiered in 2014."
        if _nonempty_str(show_title)
        else f"{actress_name}'s first television appearance was in a show that premiered in 2014."
    )
    await evaluator.verify(
        claim=tv_claim,
        node=tv_leaf,
        sources=tv_sources,
        additional_instruction=require_url_support_instruction(
            "To pass, the evidence must support BOTH that it was the actress's first TV appearance and that the show premiered in 2014."
        ),
    )

    # 2.c) Broadway debut in 2024 (critical)
    bway_leaf = evaluator.add_leaf(
        id="broadway_debut_2024",
        desc="The actress made her Broadway debut in 2024",
        parent=constraints,
        critical=True,
    )
    bway_sources = _safe_sources(extracted.broadway.broadway_sources) if extracted.broadway else None
    bway_claim = f"{actress_name} made her Broadway debut in 2024."
    await evaluator.verify(
        claim=bway_claim,
        node=bway_leaf,
        sources=bway_sources,
        additional_instruction=require_url_support_instruction(
            "Accept phrasing like 'Broadway debut' or 'first appearance on Broadway' occurring in 2024."
        ),
    )

    # 2.d) Beauty partnership verification (critical, parallel)
    beauty_node = evaluator.add_parallel(
        id="beauty_partnership_verification",
        desc="Verify the beauty brand partnership details",
        parent=constraints,
        critical=True,
    )

    beauty_sources = _safe_sources(extracted.beauty.partnership_sources) if extracted.beauty else None
    month = (extracted.beauty.month or "").strip() if extracted.beauty else ""
    year = (extracted.beauty.year or "").strip() if extracted.beauty else ""
    brand = (extracted.beauty.brand_name or "").strip() if extracted.beauty else ""
    founder = (extracted.beauty.brand_founder or "").strip() if extracted.beauty else ""

    # 2.d.i) First beauty partnership announced March 2026 (critical)
    beauty_announce_leaf = evaluator.add_leaf(
        id="first_beauty_partnership_announced_march_2026",
        desc="The actress announced her first beauty brand partnership in March 2026",
        parent=beauty_node,
        critical=True,
    )
    announce_claim = f"In March 2026, {actress_name} announced her first beauty brand partnership."
    await evaluator.verify(
        claim=announce_claim,
        node=beauty_announce_leaf,
        sources=beauty_sources,
        additional_instruction=require_url_support_instruction(
            "The source should clearly mention March 2026 and that it was framed as her 'first' beauty partnership."
        ),
    )

    # 2.d.ii) It was the actress's first beauty brand campaign (critical)
    first_campaign_leaf = evaluator.add_leaf(
        id="first_beauty_brand_campaign",
        desc="This partnership was the actress's first beauty brand campaign",
        parent=beauty_node,
        critical=True,
    )
    first_campaign_claim = f"The Rhode partnership was {actress_name}'s first-ever beauty brand campaign."
    await evaluator.verify(
        claim=first_campaign_claim,
        node=first_campaign_leaf,
        sources=beauty_sources,
        additional_instruction=require_url_support_instruction(
            "Look for phrases like 'first beauty campaign', 'debut beauty campaign', or equivalent."
        ),
    )

    # 2.d.iii) Brand is Rhode, founded by Hailey Bieber (critical)
    brand_leaf = evaluator.add_leaf(
        id="brand_is_rhode_founded_by_hailey_bieber",
        desc="The beauty brand is Rhode, a skincare and cosmetics brand founded by Hailey Bieber",
        parent=beauty_node,
        critical=True,
    )
    brand_claim = (
        f"The partnership brand is Rhode, a skincare and cosmetics brand founded by Hailey Bieber."
    )
    await evaluator.verify(
        claim=brand_claim,
        node=brand_leaf,
        sources=beauty_sources,
        additional_instruction=require_url_support_instruction(
            "The page(s) should confirm that the brand is Rhode and that it was founded by Hailey Bieber."
        ),
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    # Initialize evaluator (framework root is always non-critical)
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Overall flow is sequential at the very top
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
        prompt=prompt_extract_actress_info(),
        template_class=ActressExtraction,
        extraction_name="actress_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()