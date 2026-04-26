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
TASK_ID = "first_autopen_signing"
TASK_DESCRIPTION = """
In the history of U.S. presidential document signing, there was a first occasion when a president used an autopen (automated signature device) to sign a bill into law, rather than just using it for routine correspondence. Research and provide comprehensive information about this historic first use of an autopen for signing legislation, including: (1) which U.S. president was the first to use an autopen to sign a bill into law, (2) the name of the specific bill that was signed, (3) the exact date this occurred, (4) the circumstances and location that necessitated using an autopen rather than a manual signature, (5) the Justice Department legal opinion that established the legal basis for this practice, including the specific date that opinion was issued and the presidential administration under which it was issued.
"""

# Optional ground truth info for reference in the evaluation summary
GROUND_TRUTH = {
    "first_president": "Barack Obama",
    "first_bill": "PATRIOT Sunsets Extension Act of 2011 (S. 990; Public Law 112-14)",
    "signing_date_us": "May 26, 2011 (U.S. time)",
    "signing_time_france": "Approximately 5:45 a.m. local time in France on May 27, 2011",
    "location_circumstances": "Deauville, France during the G8 summit; overseas, provisions of the Patriot Act were about to expire",
    "olc_opinion_title": "Authority of the President to Sign a Bill by Directing the Affixing of His Signature by Autopen",
    "olc_opinion_date": "July 7, 2005",
    "olc_opinion_administration": "George W. Bush",
    "olc_validity_condition": "President must review/approve the bill and direct/authorize affixing his signature (e.g., via autopen)"
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AutopenEventExtraction(BaseModel):
    # Core facts extracted from the answer
    president: Optional[str] = None  # First US president to use autopen to sign a bill into law
    bill_name: Optional[str] = None  # Official act name, e.g., "PATRIOT Sunsets Extension Act of 2011"
    signing_date_us: Optional[str] = None  # e.g., "May 26, 2011"
    signing_date_france: Optional[str] = None  # e.g., "May 27, 2011"
    signing_time_france: Optional[str] = None  # e.g., "5:45 a.m."
    location: Optional[str] = None  # e.g., "Deauville, France"
    circumstances: Optional[str] = None  # narrative explanation (overseas; imminent expiration of provisions)

    # Sources for each fact (URLs explicitly cited in the answer)
    sources_president: List[str] = Field(default_factory=list)
    sources_bill: List[str] = Field(default_factory=list)
    sources_datetime: List[str] = Field(default_factory=list)
    sources_circumstances: List[str] = Field(default_factory=list)

    # OLC legal basis fields
    olc_opinion_title: Optional[str] = None
    olc_opinion_date: Optional[str] = None  # expected "July 7, 2005"
    olc_opinion_administration: Optional[str] = None  # expected "George W. Bush"
    olc_validity_condition: Optional[str] = None  # e.g., president must direct/authorize affixing signature
    sources_olc: List[str] = Field(default_factory=list)

    # All URLs found anywhere in the answer (for fallback verification)
    all_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_autopen_event() -> str:
    return """
    Extract structured information from the answer about the first use of an autopen to sign a bill into law.

    Required fields (return null for missing text fields; return [] for missing URL lists):
    1. president: The first U.S. president to use an autopen to sign a bill into law (not routine correspondence).
    2. bill_name: The official name of the bill/act signed via autopen. If multiple variants are given (e.g., "PATRIOT Sunsets Extension Act of 2011", "S. 990", "Public Law 112-14"), extract the main/official name as stated in the answer.
    3. signing_date_us: The U.S. date of signing if provided (e.g., "May 26, 2011").
    4. signing_date_france: The local date in France if provided (e.g., "May 27, 2011").
    5. signing_time_france: The local time in France if provided (e.g., "5:45 a.m.").
    6. location: The location/city/country at the time of authorization/signing (e.g., "Deauville, France").
    7. circumstances: A concise summary of why autopen was used (e.g., "overseas at G8; Patriot Act provisions were about to expire").
    8. sources_president: All URLs specifically cited in the answer that support the identity of the president.
    9. sources_bill: All URLs specifically cited in the answer that support the identity/name of the bill.
    10. sources_datetime: All URLs specifically cited in the answer that support the date/time details (U.S. date OR France local date/time).
    11. sources_circumstances: All URLs specifically cited in the answer that support the location/circumstances (France/G8; imminent expiration).
    12. olc_opinion_title: The title of the DOJ/OLC opinion that provides the legal basis for autopen signing of bills (as stated in the answer).
    13. olc_opinion_date: The issuance date of the OLC opinion (expected July 7, 2005 if correct).
    14. olc_opinion_administration: The presidential administration under which the opinion was issued (expected George W. Bush).
    15. olc_validity_condition: Summarize the key validity condition from the opinion (e.g., the president must direct/authorize affixing his signature).
    16. sources_olc: All URLs specifically cited in the answer that support the OLC opinion, its date, its administration, and the validity condition.
    17. all_sources: A list of all URLs appearing anywhere in the answer (including markdown links).

    Extraction rules for URLs:
    - Extract only real URLs explicitly present in the answer (plain or markdown links).
    - If a URL is missing a protocol, prepend http://.
    - Deduplicate URLs if repeated.

    If any required text information is not mentioned, set it to null.
    If any required sources field is not mentioned, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def pick_sources(primary: List[str], fallback: List[str]) -> List[str]:
    """Return primary source list if non-empty, otherwise return fallback list."""
    if primary and len(primary) > 0:
        return primary
    return fallback


def safe_text(x: Optional[str]) -> str:
    """Return a safe string for None values."""
    return x or ""


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root_node,
    info: AutopenEventExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and perform verifications.
    We create a critical top-level node under the evaluator's root to honor the rubric's critical root requirement.
    """
    # Top critical aggregation node (to simulate a critical root)
    critical_root = evaluator.add_parallel(
        id="critical_root",
        desc="Answer provides historically correct details about the first autopen signing of legislation",
        parent=root_node,
        critical=True
    )

    # ------------------------ first_president ------------------------ #
    # Group node to host existence check (sequential to enforce order)
    pres_group = evaluator.add_sequential(
        id="first_president_main",
        desc="Answer correctly identifies the first U.S. president to use an autopen to sign a bill into law",
        parent=critical_root,
        critical=True
    )
    # Existence check
    evaluator.add_custom_node(
        result=bool(info.president and info.president.strip()),
        id="first_president_exists",
        desc="President name is provided in the answer",
        parent=pres_group,
        critical=True
    )
    # Verification leaf
    pres_leaf = evaluator.add_leaf(
        id="first_president",
        desc="Answer correctly identifies the first U.S. president to use an autopen to sign a bill into law (not routine correspondence)",
        parent=pres_group,
        critical=True
    )
    pres_claim = f"The first U.S. president to use an autopen to sign a bill into law was {safe_text(info.president)}."
    await evaluator.verify(
        claim=pres_claim,
        node=pres_leaf,
        sources=pick_sources(info.sources_president, info.all_sources),
        additional_instruction="This is specifically about the first use of autopen for signing an act of Congress into law. Earlier uses for routine correspondence do NOT count."
    )

    # ------------------------ first_bill ----------------------------- #
    bill_group = evaluator.add_sequential(
        id="first_bill_main",
        desc="Answer correctly identifies the first bill signed into law via autopen",
        parent=critical_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.bill_name and info.bill_name.strip()),
        id="first_bill_exists",
        desc="Bill/act name is provided in the answer",
        parent=bill_group,
        critical=True
    )
    bill_leaf = evaluator.add_leaf(
        id="first_bill",
        desc="Answer correctly identifies the first bill signed into law via autopen (unique act name)",
        parent=bill_group,
        critical=True
    )
    bill_claim = f"The first bill signed into law via autopen was {safe_text(info.bill_name)}."
    await evaluator.verify(
        claim=bill_claim,
        node=bill_leaf,
        sources=pick_sources(info.sources_bill, info.all_sources),
        additional_instruction="Allow common variants and identifiers (e.g., official title, S. number, or Public Law number) that uniquely refer to the same act."
    )

    # ------------------------ signing_datetime ----------------------- #
    datetime_group = evaluator.add_sequential(
        id="signing_datetime_main",
        desc="Answer gives the correct signing date/time details",
        parent=critical_root,
        critical=True
    )
    # Existence: at least one of US date OR France local date/time should be present
    has_dt = bool((info.signing_date_us and info.signing_date_us.strip()) or
                  (info.signing_date_france and info.signing_date_france.strip()) or
                  (info.signing_time_france and info.signing_time_france.strip()))
    evaluator.add_custom_node(
        result=has_dt,
        id="signing_datetime_exists",
        desc="Answer provides at least one of: U.S. date OR France local date/time",
        parent=datetime_group,
        critical=True
    )
    dt_leaf = evaluator.add_leaf(
        id="signing_datetime",
        desc="Answer gives the correct signing date (accept May 26, 2011 U.S. time OR about 5:45 a.m. France time on May 27, 2011)",
        parent=datetime_group,
        critical=True
    )
    # Build a flexible claim using what the answer provided, while instructing acceptance criteria
    parts = []
    if info.signing_date_us:
        parts.append(f"{info.signing_date_us} (U.S. time)")
    if info.signing_date_france or info.signing_time_france:
        local_desc = "France local time"
        if info.signing_date_france and info.signing_time_france:
            parts.append(f"{info.signing_time_france} {info.signing_date_france} ({local_desc})")
        elif info.signing_date_france:
            parts.append(f"{info.signing_date_france} ({local_desc})")
        elif info.signing_time_france:
            parts.append(f"{info.signing_time_france} ({local_desc})")
    if parts:
        dt_claim = f"The autopen signing occurred on {', OR '.join(parts)}."
    else:
        # Fallback string to avoid empty claim text
        dt_claim = "The autopen signing date/time matches accepted historical records."
    await evaluator.verify(
        claim=dt_claim,
        node=dt_leaf,
        sources=pick_sources(info.sources_datetime, info.all_sources),
        additional_instruction="Pass if sources support either variant: (a) May 26, 2011 U.S. time OR (b) approximately 5:45 a.m. local time on May 27, 2011 in France."
    )

    # ------------------------ circumstances_location ----------------- #
    circ_node = evaluator.add_parallel(
        id="circumstances_location",
        desc="Answer correctly describes the location and circumstances necessitating autopen use",
        parent=critical_root,
        critical=True
    )
    # Optional existence precheck at the group level
    evaluator.add_custom_node(
        result=bool((info.location and info.location.strip()) or (info.circumstances and info.circumstances.strip())),
        id="circumstances_location_exists",
        desc="Answer provides text for location/circumstances",
        parent=circ_node,
        critical=True
    )
    # Sub-leaf: location (France at G8 summit)
    loc_leaf = evaluator.add_leaf(
        id="location_france_g8",
        desc="States that the president was in France for the G8 summit at the time of authorization/signing",
        parent=circ_node,
        critical=True
    )
    # Use extracted location info in the claim text for fidelity to the answer
    loc_claim = f"At the time of authorization/signing, the president was in {safe_text(info.location)}, attending the G8 summit in France."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=pick_sources(info.sources_circumstances, info.all_sources),
        additional_instruction="Support can mention 'France' and the 'G8 summit' (e.g., Deauville, France during G8). Minor wording variations acceptable."
    )
    # Sub-leaf: necessity (overseas and imminent expiration)
    need_leaf = evaluator.add_leaf(
        id="necessity_overseas_and_expiration",
        desc="States that autopen was used because the president was overseas and relevant Patriot Act provisions were about to expire",
        parent=circ_node,
        critical=True
    )
    need_claim = f"Autopen was used because the president was overseas and {safe_text(info.circumstances)} included imminent expiration of key Patriot Act provisions."
    await evaluator.verify(
        claim=need_claim,
        node=need_leaf,
        sources=pick_sources(info.sources_circumstances, info.all_sources),
        additional_instruction="The rationale should include both being overseas and that several Patriot Act provisions were about to expire, necessitating immediate signature."
    )

    # ------------------------ legal_basis_olc ------------------------- #
    olc_node = evaluator.add_parallel(
        id="legal_basis_olc",
        desc="Answer correctly identifies the DOJ/OLC opinion providing the legal basis for autopen signing of bills, with required metadata",
        parent=critical_root,
        critical=True
    )
    # Group existence precheck for OLC fields
    evaluator.add_custom_node(
        result=bool((info.olc_opinion_title and info.olc_opinion_title.strip()) or
                    (info.olc_opinion_date and info.olc_opinion_date.strip()) or
                    (info.olc_opinion_administration and info.olc_opinion_administration.strip()) or
                    (info.olc_validity_condition and info.olc_validity_condition.strip())),
        id="olc_info_exists",
        desc="Answer provides at least one OLC opinion detail (title/date/administration/validity condition)",
        parent=olc_node,
        critical=True
    )

    # Leaf: identifies the 2005 OLC opinion that permits directing a signature to be affixed (autopen)
    olc_ident_leaf = evaluator.add_leaf(
        id="olc_opinion_identified",
        desc="References the 2005 DOJ/OLC opinion establishing legality of signing by directing a signature to be affixed (e.g., by autopen)",
        parent=olc_node,
        critical=True
    )
    olc_ident_claim = (
        f"The DOJ Office of Legal Counsel issued an opinion in 2005 (titled '{safe_text(info.olc_opinion_title)}' or similar) "
        f"stating that the President may sign a bill into law by directing that his signature be affixed by an autopen or comparable device."
    )
    await evaluator.verify(
        claim=olc_ident_claim,
        node=olc_ident_leaf,
        sources=pick_sources(info.sources_olc, info.all_sources),
        additional_instruction="Minor variations in the opinion title are acceptable as long as the substance (directing affixing of signature via autopen) is clearly supported."
    )

    # Leaf: OLC opinion issuance date
    olc_date_leaf = evaluator.add_leaf(
        id="olc_opinion_date",
        desc="Provides the correct issuance date of the DOJ/OLC opinion (July 7, 2005)",
        parent=olc_node,
        critical=True
    )
    olc_date_claim = f"The DOJ/OLC opinion was issued on {safe_text(info.olc_opinion_date)}."
    await evaluator.verify(
        claim=olc_date_claim,
        node=olc_date_leaf,
        sources=pick_sources(info.sources_olc, info.all_sources),
        additional_instruction="This date should match July 7, 2005."
    )

    # Leaf: OLC opinion administration
    olc_admin_leaf = evaluator.add_leaf(
        id="olc_opinion_administration",
        desc="Identifies that the opinion was issued under President George W. Bush's administration",
        parent=olc_node,
        critical=True
    )
    olc_admin_claim = f"The opinion was issued under the {safe_text(info.olc_opinion_administration)} administration."
    await evaluator.verify(
        claim=olc_admin_claim,
        node=olc_admin_leaf,
        sources=pick_sources(info.sources_olc, info.all_sources),
        additional_instruction="This should correspond to the George W. Bush administration."
    )

    # Leaf: OLC validity condition (presidential direction/authorization)
    olc_valid_leaf = evaluator.add_leaf(
        id="olc_requires_presidential_direction",
        desc="States the legal-validity condition: the president must direct/authorize the affixing of the signature for it to be legally valid",
        parent=olc_node,
        critical=True
    )
    olc_valid_claim = (
        f"The OLC opinion explains that for autopen signing to be legally valid, the President must review/approve the bill and "
        f"direct or authorize the affixing of his signature (e.g., via autopen): {safe_text(info.olc_validity_condition)}."
    )
    await evaluator.verify(
        claim=olc_valid_claim,
        node=olc_valid_leaf,
        sources=pick_sources(info.sources_olc, info.all_sources),
        additional_instruction="Support should state the requirement for presidential direction/authorization of affixing the signature."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the first autopen signing task and return a structured summary.
    """
    # 1) Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root node (non-critical by framework); we'll add a critical child
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

    # 2) Extract structured information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_autopen_event(),
        template_class=AutopenEventExtraction,
        extraction_name="autopen_event_info"
    )

    # 3) Add ground truth (for reference only, not used to score)
    evaluator.add_ground_truth({
        "expected": GROUND_TRUTH,
        "notes": "Ground truth provided for reference in the summary; verification uses answer-provided sources where available."
    })

    # 4) Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, extracted_info)

    # 5) Return standard summary
    return evaluator.get_summary()