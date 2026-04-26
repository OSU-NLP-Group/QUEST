import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bna_tsa_precheck_msc_closed_loop"
TASK_DESCRIPTION = (
    "A U.S. citizen is planning to take a closed-loop cruise with MSC Cruises to the Caribbean (departing from and "
    "returning to the same U.S. port). Before their trip, they want to enroll in TSA PreCheck at Nashville International "
    "Airport (BNA). They do not currently have a passport book or passport card. Provide the following information: "
    "(1) The specific location of the TSA PreCheck enrollment office within BNA airport, including the terminal, "
    "floor/level designation, and position within that level. (2) The minimum set of two documents they need to bring "
    "that will satisfy both TSA PreCheck List B enrollment requirements and MSC Cruises' boarding requirements for "
    "closed-loop cruises. For each document, specify the document type and any important qualifications or restrictions "
    "(such as whether certified copies are acceptable, or whether hospital certificates are acceptable)."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LocationInfo(BaseModel):
    terminal: Optional[str] = None
    level: Optional[str] = None
    position: Optional[str] = None  # e.g., "north side of the terminal"
    source_urls: List[str] = Field(default_factory=list)


class DocumentItem(BaseModel):
    category: Optional[str] = None  # expected values: "photo_id" or "citizenship"
    document_type: Optional[str] = None
    qualifiers: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class TravelPrepExtraction(BaseModel):
    location: Optional[LocationInfo] = None
    documents: List[DocumentItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_travel_prep_info() -> str:
    return """
    Extract the requested information exactly as stated in the answer.

    Part A: TSA PreCheck enrollment office location at Nashville International Airport (BNA)
    - terminal: The terminal name the answer gives for the TSA PreCheck enrollment office (e.g., "Main Terminal").
    - level: The floor/level designation the answer gives (e.g., "Ground Transportation Level (Level 1)").
    - position: The position within the level the answer gives (e.g., "north side of the terminal").
    - source_urls: All URLs cited in the answer that specifically support the location (airport page, TSA/IdentoGO/BNA page, etc.).

    Part B: Minimum document set proposed by the answer (for a U.S. citizen without a passport) that satisfies BOTH
            (1) TSA PreCheck enrollment (the List B path: acceptable photo ID + proof of citizenship), and
            (2) MSC Cruises' boarding requirements for closed-loop U.S. cruises (government-issued photo ID + birth certificate).
    Extract each document the answer proposes (at least two) as items with:
    - category: "photo_id" if it's a government-issued photo ID (e.g., driver's license or state ID);
                "citizenship" if it's a proof of citizenship (e.g., birth certificate).
    - document_type: The document type as written in the answer (e.g., "REAL ID-compliant driver's license", "U.S. birth certificate").
    - qualifiers: A list of key qualifications/restrictions exactly as the answer states them (e.g., "unexpired", "government-issued",
                  "REAL ID", "original or certified copy", "hospital certificates not accepted", "baptismal papers not accepted").
    - source_urls: All URLs cited in the answer that support the acceptability/requirements of this document (TSA/Universal Enroll pages,
                   MSC Cruises travel documents pages, etc.).

    Output JSON should follow this schema:
    {
      "location": {
        "terminal": ...,
        "level": ...,
        "position": ...,
        "source_urls": [...]
      },
      "documents": [
        {
          "category": ...,
          "document_type": ...,
          "qualifiers": [...],
          "source_urls": [...]
        },
        ...
      ]
    }

    Important:
    - Extract only what is explicitly present in the answer.
    - If a field is not mentioned in the answer, return null for that field (or an empty list for URLs/qualifiers).
    - Do not invent URLs. Only include URLs explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _lc(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _filter_urls(urls: List[str], include_keywords: List[str]) -> List[str]:
    if not urls:
        return []
    result = []
    for u in urls:
        lu = (u or "").lower()
        if any(k in lu for k in include_keywords):
            result.append(u)
    return result


def _find_document_by_category_or_keywords(
    docs: List[DocumentItem],
    preferred_category: str,
    keyword_any: List[str],
) -> Optional[DocumentItem]:
    # Prefer exact category match
    for d in docs:
        if _lc(d.category) == preferred_category:
            return d
    # Fallback by keywords in document_type
    for d in docs:
        dtype = _lc(d.document_type)
        if any(k in dtype for k in keyword_any):
            return d
    return None


async def _verify_equivalence(evaluator: Evaluator, a: str, b: str, context: str, add_ins: str = "None") -> bool:
    """
    Use simple verification to decide if 'a' is equivalent to 'b' given the context.
    """
    claim = f"In the context of {context}, the phrase '{a}' is equivalent to '{b}'."
    return await evaluator.verify(
        claim=claim,
        node=None,
        additional_instruction=add_ins
    )


async def _verify_with_sources(
    evaluator: Evaluator,
    claim: str,
    urls: List[str],
    add_ins: str = "None"
) -> bool:
    """
    Verify a claim using provided URLs; returns False if no URLs are available.
    """
    if not urls:
        return False
    return await evaluator.verify(
        claim=claim,
        node=None,
        sources=urls,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_location(
    evaluator: Evaluator,
    parent_node,
    loc: Optional[LocationInfo]
) -> None:
    # Prepare defaults
    terminal_val = loc.terminal if loc else None
    level_val = loc.level if loc else None
    position_val = loc.position if loc else None
    loc_urls = loc.source_urls if loc else []

    # Terminal specified leaf
    term_present = bool(_lc(terminal_val))
    eq_term = False
    support_term = False
    if term_present:
        # Check equivalence to expected wording
        eq_term = await _verify_equivalence(
            evaluator,
            a=terminal_val,
            b="Main Terminal",
            context="describing the TSA PreCheck enrollment office location at BNA",
            add_ins="Allow common synonyms; minor formatting differences are fine."
        )
        # Check web support
        support_term = await _verify_with_sources(
            evaluator,
            claim="The TSA PreCheck enrollment office at Nashville International Airport (BNA) is located in the Main Terminal.",
            urls=loc_urls,
            add_ins="Rely on the airport/operator page(s) or the enrollment partner page(s) that explicitly describe the location within the BNA terminal."
        )
    term_result = term_present and eq_term and support_term
    evaluator.add_custom_node(
        result=term_result,
        id="terminal_specified",
        desc="Specify that the office is located in the Main Terminal",
        parent=parent_node,
        critical=True
    )

    # Level specified leaf
    level_present = bool(_lc(level_val))
    eq_level = False
    support_level = False
    if level_present:
        eq_level = await _verify_equivalence(
            evaluator,
            a=level_val,
            b="Ground Transportation Level (Level 1)",
            context="BNA terminal level labeling for the TSA PreCheck enrollment office",
            add_ins="Treat 'Level 1', 'Ground Transportation Level', and 'Ground Transportation (Level 1)' as equivalent."
        )
        support_level = await _verify_with_sources(
            evaluator,
            claim="The TSA PreCheck enrollment office at BNA is on the Ground Transportation Level (Level 1).",
            urls=loc_urls,
            add_ins="The page should explicitly indicate the Ground Transportation Level or Level 1 for the enrollment office."
        )
    level_result = level_present and eq_level and support_level
    evaluator.add_custom_node(
        result=level_result,
        id="level_specified",
        desc="Specify that the office is on the Ground Transportation Level (Level 1)",
        parent=parent_node,
        critical=True
    )

    # Position specified leaf
    pos_present = bool(_lc(position_val))
    eq_pos = False
    support_pos = False
    if pos_present:
        eq_pos = await _verify_equivalence(
            evaluator,
            a=position_val,
            b="north side of the terminal",
            context="describing the office's position on that level within BNA",
            add_ins="Accept equivalent phrasings like 'north side' or 'on the north side of the terminal'."
        )
        support_pos = await _verify_with_sources(
            evaluator,
            claim="The TSA PreCheck enrollment office at BNA is on the north side of the terminal.",
            urls=loc_urls,
            add_ins="The location description on the referenced page(s) should clearly indicate 'north side' (or equivalent) for the office position."
        )
    pos_result = pos_present and eq_pos and support_pos
    evaluator.add_custom_node(
        result=pos_result,
        id="position_specified",
        desc="Specify that the office is on the north side of the terminal",
        parent=parent_node,
        critical=True
    )


async def verify_documents(
    evaluator: Evaluator,
    parent_node,
    docs: List[DocumentItem]
) -> None:
    # Identify photo ID doc (driver's license/state ID)
    photo_item = _find_document_by_category_or_keywords(
        docs,
        preferred_category="photo_id",
        keyword_any=["driver", "license", "state id", "photo id", "identification", "real id"]
    )
    # Identify citizenship doc (birth certificate)
    citizen_item = _find_document_by_category_or_keywords(
        docs,
        preferred_category="citizenship",
        keyword_any=["birth certificate", "certificate of birth"]
    )

    # Helper domain filters for verification specificity
    def tsa_urls(urls: List[str]) -> List[str]:
        return _filter_urls(urls, ["universalenroll.dhs.gov", "tsa.gov", "identogo.com"])

    def msc_urls(urls: List[str]) -> List[str]:
        return _filter_urls(urls, ["msccruisesusa.com", "msccruises.com"])

    # ---------------- Photo Identification leaf ----------------
    photo_present = photo_item is not None and bool(_lc(photo_item.document_type))
    sem_photo = False
    tsa_support_photo = False
    msc_support_photo = False

    if photo_present:
        qualifiers_text = "; ".join(photo_item.qualifiers) if photo_item.qualifiers else ""
        sem_photo = await evaluator.verify(
            claim=(
                f"The described document matches a valid, unexpired, government-issued REAL ID-compliant driver's license "
                f"or state-issued photo ID. Document: '{photo_item.document_type}'. Qualifiers: '{qualifiers_text}'."
            ),
            node=None,
            additional_instruction=(
                "Judge based on the description provided in the answer text. Consider phrasing variations; "
                "ensure the description clearly implies: government-issued, photo ID, unexpired, and REAL ID-compliant (or equivalent)."
            )
        )

        tsa_support_photo = await _verify_with_sources(
            evaluator,
            claim=(
                "For TSA PreCheck enrollment documentation (using the path that requires multiple documents), a valid "
                "driver's license or state-issued photo ID is an acceptable identity document (the ID must be government-issued and unexpired; "
                "REAL ID-compliant IDs are acceptable)."
            ),
            urls=tsa_urls(photo_item.source_urls),
            add_ins="Look for TSA/Universal Enroll/IdentoGO documentation that lists acceptable identity documents for TSA PreCheck enrollment."
        )

        msc_support_photo = await _verify_with_sources(
            evaluator,
            claim=(
                "For U.S. citizens on a closed-loop cruise with MSC Cruises (departing and returning to the same U.S. port), "
                "a government-issued photo ID is accepted when presented with an original or certified copy of a U.S. birth certificate."
            ),
            urls=msc_urls(photo_item.source_urls),
            add_ins="Look for MSC Cruises official documentation that explicitly allows government-issued photo ID with a birth certificate for closed-loop cruises."
        )

    photo_result = photo_present and sem_photo and tsa_support_photo and msc_support_photo
    evaluator.add_custom_node(
        result=photo_result,
        id="photo_identification",
        desc="Valid REAL ID-compliant driver's license or state-issued photo ID (unexpired, government-issued)",
        parent=parent_node,
        critical=True
    )

    # ---------------- Citizenship Proof leaf ----------------
    citizen_present = citizen_item is not None and bool(_lc(citizen_item.document_type))
    sem_citizen = False
    tsa_support_citizen = False
    msc_support_citizen = False

    if citizen_present:
        qualifiers_text = "; ".join(citizen_item.qualifiers) if citizen_item.qualifiers else ""
        # Semantics: ensure the answer itself indicates "original or certified copy" and excludes "hospital certificate/baptismal papers"
        sem_citizen = await evaluator.verify(
            claim=(
                f"The described document is an original or certified copy of a U.S. birth certificate, and the answer explicitly notes "
                f"that hospital certificates or baptismal papers are not acceptable. Document: '{citizen_item.document_type}'. "
                f"Qualifiers: '{qualifiers_text}'."
            ),
            node=None,
            additional_instruction=(
                "Judge solely based on the answer's wording. Accept common phrasings for 'original or certified copy'. "
                "The answer should explicitly mention that hospital certificates or baptismal papers are not accepted."
            )
        )

        tsa_support_citizen = await _verify_with_sources(
            evaluator,
            claim=(
                "For TSA PreCheck enrollment documentation (the path that uses multiple documents), an original or certified copy of a "
                "U.S. birth certificate is accepted as proof of citizenship when presented with an acceptable photo ID."
            ),
            urls=tsa_urls(citizen_item.source_urls),
            add_ins="Look for TSA/Universal Enroll/IdentoGO pages listing acceptable proof of citizenship for TSA PreCheck enrollment."
        )

        msc_support_citizen = await _verify_with_sources(
            evaluator,
            claim=(
                "For U.S. citizens on a closed-loop cruise with MSC Cruises, an original or certified copy of a U.S. birth certificate is accepted, "
                "and hospital certificates or baptismal papers are not accepted."
            ),
            urls=msc_urls(citizen_item.source_urls),
            add_ins="Look for MSC Cruises official documentation that explicitly states acceptable birth certificates and rejects hospital/baptismal certificates."
        )

    citizen_result = citizen_present and sem_citizen and tsa_support_citizen and msc_support_citizen
    evaluator.add_custom_node(
        result=citizen_result,
        id="citizenship_proof",
        desc="Original or certified copy of U.S. birth certificate (not hospital certificate or baptismal paper)",
        parent=parent_node,
        critical=True
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
    """
    Evaluate an answer for the BNA TSA PreCheck location and MSC closed-loop cruise documentation task.
    """
    # Initialize evaluator (root node is non-critical by framework default; children will be critical)
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
        default_model=model
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_travel_prep_info(),
        template_class=TravelPrepExtraction,
        extraction_name="travel_prep_extraction"
    )

    # Add lightweight ground truth expectations (phrases we expect to see for location)
    evaluator.add_ground_truth({
        "expected_location": {
            "terminal": "Main Terminal",
            "level": "Ground Transportation Level (Level 1)",
            "position": "north side of the terminal"
        },
        "expected_documents": [
            "Valid REAL ID-compliant, unexpired, government-issued driver's license or state-issued photo ID",
            "Original or certified copy of a U.S. birth certificate; hospital/baptismal certificates not accepted"
        ]
    }, gt_type="expected_requirements")

    # Build the rubric tree according to JSON
    # 1) TSA PreCheck office location at BNA (critical)
    loc_node = evaluator.add_parallel(
        id="tsa_precheck_office_location",
        desc="TSA PreCheck enrollment office location at Nashville International Airport (BNA)",
        parent=root,
        critical=True
    )
    await verify_location(evaluator, loc_node, extracted.location)

    # 2) Documents that satisfy both TSA PreCheck List B and MSC closed-loop cruise requirements (critical)
    docs_node = evaluator.add_parallel(
        id="document_requirements",
        desc="Minimum document set that satisfies both TSA PreCheck List B and MSC closed-loop cruise requirements",
        parent=root,
        critical=True
    )
    await verify_documents(evaluator, docs_node, extracted.documents or [])

    # Return the evaluation summary
    return evaluator.get_summary()