import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "travel_id_guidance_dollywood_ca_mdl"
TASK_DESCRIPTION = (
    "A traveler from California is planning to visit Dollywood theme park in Pigeon Forge, Tennessee, and wants to use "
    "their California mobile driver's license (digital ID) stored in Apple Wallet for airport security. Identify the nearest major "
    "commercial airport to Dollywood and determine whether they can use their California digital ID at that airport's TSA checkpoint. "
    "Additionally, provide information about alternative identification options available as of February 2026 for travelers who may "
    "not have their REAL ID-compliant physical identification."
)


# =========================
# Extraction Models
# =========================
class AirportSection(BaseModel):
    airport_name: Optional[str] = None
    airport_iata: Optional[str] = None
    airport_sources: List[str] = Field(default_factory=list)


class DigitalIDSection(BaseModel):
    ca_participation_statement: Optional[str] = None
    ca_participation_sources: List[str] = Field(default_factory=list)

    mdl_real_id_basis_statement: Optional[str] = None
    mdl_real_id_basis_sources: List[str] = Field(default_factory=list)

    airport_acceptance_statement: Optional[str] = None
    acceptance_sources: List[str] = Field(default_factory=list)

    checkpoint_variability_statement: Optional[str] = None
    checkpoint_sources: List[str] = Field(default_factory=list)

    clear_conclusion: Optional[str] = None
    conclusion_sources: List[str] = Field(default_factory=list)


class AlternativesSection(BaseModel):
    real_id_enforcement_statement: Optional[str] = None
    real_id_sources: List[str] = Field(default_factory=list)

    physical_alternatives_list: List[str] = Field(default_factory=list)
    alternatives_sources: List[str] = Field(default_factory=list)

    tsa_confirmid_statement: Optional[str] = None
    tsa_confirmid_sources: List[str] = Field(default_factory=list)


class TravelIDExtraction(BaseModel):
    nearest_airport: Optional[AirportSection] = None
    digital_id: Optional[DigitalIDSection] = None
    alternatives: Optional[AlternativesSection] = None


# =========================
# Extraction Prompt
# =========================
def prompt_extract_travel_id() -> str:
    return """
    You will extract structured information from the answer related to:
    (1) the nearest major commercial airport to Dollywood/Pigeon Forge, TN,
    (2) the usability of a California mobile driver's license (mDL) in Apple Wallet at that airport's TSA checkpoint,
    (3) alternative identification options as of February 2026 for travelers who do not have a REAL ID-compliant physical ID.

    IMPORTANT URL RULES:
    - Only include URLs explicitly present in the answer text. Do not invent URLs.
    - If no URLs are provided for a field, return an empty array for that sources field.
    - Accept URLs in plain or markdown link format (extract the actual URL).

    Return a JSON with the following structure:

    {
      "nearest_airport": {
        "airport_name": string or null,
        "airport_iata": string or null,
        "airport_sources": [url, ...]
      },
      "digital_id": {
        "ca_participation_statement": string or null,
        "ca_participation_sources": [url, ...],

        "mdl_real_id_basis_statement": string or null,
        "mdl_real_id_basis_sources": [url, ...],

        "airport_acceptance_statement": string or null,
        "acceptance_sources": [url, ...],

        "checkpoint_variability_statement": string or null,
        "checkpoint_sources": [url, ...],

        "clear_conclusion": string or null,
        "conclusion_sources": [url, ...]
      },
      "alternatives": {
        "real_id_enforcement_statement": string or null,
        "real_id_sources": [url, ...],

        "physical_alternatives_list": [string, ...],
        "alternatives_sources": [url, ...],

        "tsa_confirmid_statement": string or null,
        "tsa_confirmid_sources": [url, ...]
      }
    }

    EXTRACTION GUIDANCE:
    - nearest_airport.airport_name: The airport the answer claims is the nearest major commercial airport to Dollywood/Pigeon Forge.
    - nearest_airport.airport_iata: If an IATA code (e.g., TYS) is given, extract it; else null.
    - nearest_airport.airport_sources: All URLs cited in the answer to support that airport claim.

    - digital_id.ca_participation_statement: The answer’s statement about whether California is a participating state/territory for TSA-accepted digital IDs (mDL).
    - digital_id.ca_participation_sources: URLs provided to support the California participation statement.

    - digital_id.mdl_real_id_basis_statement: The answer’s statement that to use mDL as TSA Digital ID, it must be based on a REAL ID-compliant or Enhanced license/ID.
    - digital_id.mdl_real_id_basis_sources: URLs supporting that requirement.

    - digital_id.airport_acceptance_statement: The answer’s explicit determination of whether the identified airport’s TSA checkpoint(s) accept TSA Digital ID/mDL.
    - digital_id.acceptance_sources: URLs supporting the airport acceptance determination.

    - digital_id.checkpoint_variability_statement: Any note that acceptance can be checkpoint/terminal specific or may require checking specific TSA/airport guidance.
    - digital_id.checkpoint_sources: URLs supporting that variability note.

    - digital_id.clear_conclusion: The final explicit conclusion advising the traveler whether they can use their California Apple Wallet mDL at that airport’s checkpoint(s).
    - digital_id.conclusion_sources: Any URLs cited together with this conclusion (if any).

    - alternatives.real_id_enforcement_statement: A statement recognizing the REAL ID enforcement requirement beginning May 7, 2025 for domestic flights.
    - alternatives.real_id_sources: URLs supporting the REAL ID enforcement date/requirement.

    - alternatives.physical_alternatives_list: A list of one or more acceptable physical ID alternatives the answer mentions (e.g., passport).
    - alternatives.alternatives_sources: URLs supporting acceptability of those physical alternatives.

    - alternatives.tsa_confirmid_statement: The answer’s statement about TSA ConfirmID (as of Feb 1, 2026) including the $45 fee and 10-day travel period.
    - alternatives.tsa_confirmid_sources: URLs supporting the ConfirmID details.

    If some items are not mentioned in the answer, set the string fields to null and lists to empty.
    """


# =========================
# Helper Functions
# =========================
def _infer_acceptance_bool(statement: Optional[str]) -> Optional[bool]:
    if not statement:
        return None
    s = statement.lower()
    negatives = [
        "not accept", "not accepted", "does not accept", "don't accept", "doesn't accept",
        "not available", "no longer accept", "isn't accepted", "is not accepted",
        "cannot use", "can't use", "unavailable"
    ]
    if any(neg in s for neg in negatives):
        return False
    positives = ["accept", "accepted", "supports", "supported", "available", "participating"]
    if any(pos in s for pos in positives):
        return True
    return None


def _fmt_airport(airport: AirportSection) -> str:
    if airport and airport.airport_name:
        if airport.airport_iata:
            return f"{airport.airport_name} ({airport.airport_iata})"
        return airport.airport_name
    return "the identified airport"


# =========================
# Verification Builders
# =========================
async def verify_airport_and_digital_id(evaluator: Evaluator, parent_node, extracted: TravelIDExtraction) -> None:
    # Critical sequential block: identify airport, then digital ID usability
    seq_node = evaluator.add_sequential(
        id="Airport_and_Digital_ID_Assessment",
        desc="Identify the nearest airport, then evaluate CA mDL usability at that airport's TSA checkpoint.",
        parent=parent_node,
        critical=True
    )

    # 1) Nearest Airport (critical)
    nearest = extracted.nearest_airport or AirportSection()
    nearest_node = evaluator.add_parallel(
        id="Nearest_Airport",
        desc="Identify the nearest major commercial airport to Dollywood/Pigeon Forge, TN.",
        parent=seq_node,
        critical=True
    )

    airport_leaf = evaluator.add_leaf(
        id="Nearest_Airport_Named",
        desc="Names the nearest major commercial airport serving the Dollywood/Pigeon Forge area.",
        parent=nearest_node,
        critical=True
    )

    airport_str = _fmt_airport(nearest)
    claim_nearest = f"The nearest major commercial airport to Dollywood in Pigeon Forge, Tennessee is {airport_str}."
    await evaluator.verify(
        claim=claim_nearest,
        node=airport_leaf,
        sources=nearest.airport_sources,
        additional_instruction=(
            "Assess whether the provided sources clearly indicate that this is the nearest major commercial airport "
            "to Dollywood/Pigeon Forge (i.e., an airport with regular commercial airline service). "
            "Allow reasonable wording variants like 'closest' or 'primary airport for the region'."
        ),
    )

    # 2) Digital ID Usability at that Airport (critical)
    # Note: To satisfy the framework's constraint that all children of a critical node must also be critical,
    # we set all leaves under this node as critical.
    did = extracted.digital_id or DigitalIDSection()
    did_node = evaluator.add_parallel(
        id="Digital_ID_Usability_At_That_Airport",
        desc="Determine whether the traveler can use a California mDL in Apple Wallet at the identified airport's TSA checkpoint.",
        parent=seq_node,
        critical=True
    )

    ca_part_leaf = evaluator.add_leaf(
        id="CA_mDL_TSA_Participation",
        desc="States that California is a participating state/territory for TSA-accepted digital IDs (mDL).",
        parent=did_node,
        critical=True
    )
    ca_part_claim = did.ca_participation_statement or "California is a participating state for TSA-accepted Digital IDs (mobile driver's license)."

    realid_req_leaf = evaluator.add_leaf(
        id="mDL_REAL_ID_Basis_Requirement",
        desc="States that the mobile driver's license must be based on a REAL ID-compliant license/ID (or Enhanced DL/ID) to be eligible for TSA Digital ID acceptance.",
        parent=did_node,
        critical=True
    )
    realid_req_claim = did.mdl_real_id_basis_statement or (
        "To be accepted as a TSA Digital ID, a mobile driver's license must be based on a REAL ID-compliant or Enhanced driver's license/ID."
    )

    airport_acc_leaf = evaluator.add_leaf(
        id="Airport_mDL_Acceptance_Determination",
        desc="States whether the identified airport's TSA checkpoint(s) accept TSA Digital ID/mDL.",
        parent=did_node,
        critical=True
    )
    acc_bool = _infer_acceptance_bool(did.airport_acceptance_statement)
    airport_name_for_claim = _fmt_airport(nearest)
    if acc_bool is True:
        airport_acc_claim = f"The TSA checkpoint(s) at {airport_name_for_claim} accept TSA Digital ID/mobile driver's license."
    elif acc_bool is False:
        airport_acc_claim = f"The TSA checkpoint(s) at {airport_name_for_claim} do not accept TSA Digital ID/mobile driver's license."
    else:
        # Fall back to whatever statement the answer provided, if any
        airport_acc_claim = did.airport_acceptance_statement or (
            f"Determine whether the TSA checkpoint(s) at {airport_name_for_claim} accept TSA Digital ID/mobile driver's license."
        )

    # Although originally labeled non-critical in the rubric, the framework requires critical consistency.
    # We therefore mark this as critical under the critical parent.
    checkpoint_note_leaf = evaluator.add_leaf(
        id="Checkpoint_Variability_Note",
        desc="Notes that digital ID acceptance can be terminal/checkpoint-specific and may require checking TSA/airport guidance for the specific checkpoint.",
        parent=did_node,
        critical=True
    )
    checkpoint_note_claim = did.checkpoint_variability_statement or (
        "Digital ID acceptance can vary by terminal or checkpoint and travelers should check TSA or airport guidance for the specific checkpoint."
    )

    conclusion_leaf = evaluator.add_leaf(
        id="Clear_Conclusion_For_Traveler",
        desc="Gives an explicit, unambiguous conclusion on whether the traveler can use the CA Apple Wallet digital ID at that airport's TSA checkpoint, consistent with the acceptance determination stated.",
        parent=did_node,
        critical=True
    )
    # Simple verify focusing on clarity/consistency within the answer text
    conclusion_claim = (
        f"The answer provides a clear, explicit conclusion about whether a California Apple Wallet mobile driver's license can be used at "
        f"{airport_name_for_claim}'s TSA checkpoint(s), and that conclusion is consistent with the acceptance determination stated in the answer."
    )

    # Batch verify the four URL-grounded leaves (CA participation, REAL ID basis, airport acceptance, checkpoint note)
    await evaluator.batch_verify([
        (
            ca_part_claim,
            did.ca_participation_sources,
            ca_part_leaf,
            "Verify whether TSA (or authoritative sources) lists California as a participating Digital ID (mDL) state/territory."
        ),
        (
            realid_req_claim,
            did.mdl_real_id_basis_sources,
            realid_req_leaf,
            "Verify that TSA Digital ID acceptance requires the underlying mDL to be based on a REAL ID-compliant or Enhanced license/ID."
        ),
        (
            airport_acc_claim,
            did.acceptance_sources,
            airport_acc_leaf,
            f"Verify whether {airport_name_for_claim} appears in TSA/airport lists of participating airports or otherwise indicates Digital ID (mDL) acceptance. "
            "If the source lists participating airports, check if the airport is present; if not, consider it not accepted."
        ),
        (
            checkpoint_note_claim,
            did.checkpoint_sources,
            checkpoint_note_leaf,
            "Verify that acceptance can vary by terminal/checkpoint and that travelers should confirm the specific checkpoint's guidance."
        ),
    ])

    # Verify the clarity/consistency conclusion using simple verification (no URLs required)
    await evaluator.verify(
        claim=conclusion_claim,
        node=conclusion_leaf,
        sources=None,
        additional_instruction=(
            "Judge only from the provided answer: Is there a clear yes/no style conclusion about using the CA Apple Wallet mDL at the identified airport, and is it consistent with "
            "the acceptance determination stated earlier in the answer? Allow concise wording variants."
        ),
    )


async def verify_alternatives(evaluator: Evaluator, parent_node, extracted: TravelIDExtraction) -> None:
    # Critical parallel block for alternatives
    alt = extracted.alternatives or AlternativesSection()
    alt_node = evaluator.add_parallel(
        id="Alternative_Identification_Options_As_Of_Feb_2026",
        desc="Provide alternative identification options as of Feb 2026 for travelers who do not have REAL ID-compliant physical identification available.",
        parent=parent_node,
        critical=True
    )

    # REAL ID enforcement requirement (critical)
    realid_leaf = evaluator.add_leaf(
        id="REAL_ID_Enforcement_Context",
        desc="Mentions the REAL ID enforcement requirement beginning May 7, 2025 for domestic flights (need REAL ID-compliant license or other acceptable ID).",
        parent=alt_node,
        critical=True
    )
    realid_claim = alt.real_id_enforcement_statement or (
        "REAL ID enforcement for domestic flights began on May 7, 2025, requiring a REAL ID-compliant driver's license/ID or another acceptable ID to board."
    )

    # Acceptable physical alternative(s) (critical)
    alt_physical_leaf = evaluator.add_leaf(
        id="Acceptable_Physical_Alternatives",
        desc="Mentions at least one acceptable alternative physical ID option (e.g., passport or other TSA-acceptable identification) for boarding domestic flights.",
        parent=alt_node,
        critical=True
    )
    alt_item = alt.physical_alternatives_list[0] if alt.physical_alternatives_list else "a valid U.S. passport"
    alt_physical_claim = (
        f"The following is an acceptable physical identification for boarding domestic flights: {alt_item}."
    )

    # TSA ConfirmID option (critical)
    confirmid_leaf = evaluator.add_leaf(
        id="TSA_ConfirmID_Option",
        desc="Mentions the TSA ConfirmID option starting Feb 1, 2026, including the $45 fee and 10-day travel period.",
        parent=alt_node,
        critical=True
    )
    confirmid_claim = alt.tsa_confirmid_statement or (
        "TSA ConfirmID is available starting February 1, 2026, costs $45, and covers travel for a 10-day period."
    )

    # Batch verify three leaves with URLs
    await evaluator.batch_verify([
        (
            realid_claim,
            alt.real_id_sources,
            realid_leaf,
            "Verify the REAL ID enforcement start date (May 7, 2025) and requirement for domestic flights."
        ),
        (
            alt_physical_claim,
            alt.alternatives_sources,
            alt_physical_leaf,
            "Verify that the mentioned item is listed by TSA as an acceptable physical ID for domestic air travel."
        ),
        (
            confirmid_claim,
            alt.tsa_confirmid_sources,
            confirmid_leaf,
            "Verify the ConfirmID availability starting Feb 1, 2026, the $45 fee, and that it covers a 10-day travel period."
        ),
    ])


# =========================
# Main Evaluation Entry
# =========================
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

    # 1) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_travel_id(),
        template_class=TravelIDExtraction,
        extraction_name="travel_id_guidance_extraction",
    )

    # 2) Build critical top-level node representing the rubric root
    top_node = evaluator.add_parallel(
        id="Travel_ID_Guidance",
        desc="Answer all parts: (1) nearest major commercial airport to Dollywood, (2) whether a CA Apple Wallet mDL can be used at that airport's TSA checkpoint, and (3) alternative ID options as of Feb 2026 for travelers lacking REAL ID-compliant physical ID.",
        parent=root,
        critical=True,
    )

    # 3) Subtrees
    await verify_airport_and_digital_id(evaluator, top_node, extracted)
    await verify_alternatives(evaluator, top_node, extracted)

    # 4) Return evaluation summary
    return evaluator.get_summary()