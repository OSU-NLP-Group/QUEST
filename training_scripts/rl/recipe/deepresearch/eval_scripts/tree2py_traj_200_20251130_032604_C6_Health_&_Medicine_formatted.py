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
TASK_ID = "newsweek_weight_loss_2025_two_clinics"
TASK_DESCRIPTION = (
    "Identify two weight loss clinics that appear on the Newsweek \"America's Best Weight Loss Clinics & Centers 2025\" "
    "ranking: one located in Texas and one located in Florida. Both clinics must offer gastric bypass bariatric surgery "
    "as part of their treatment services.\n\n"
    "For each of the two clinics, provide the following information:\n"
    "1. The official facility name as it appears on the Newsweek ranking\n"
    "2. The city where the facility is located\n"
    "3. The specific bariatric surgery procedures offered by the facility (which must include gastric bypass)\n"
    "4. A reference URL to verify the information (either the Newsweek ranking page at "
    "https://rankings.newsweek.com/americas-best-weight-loss-clinics-centers-2025 or the facility's official website)\n\n"
    "All information must be verifiable through the official Newsweek 2025 ranking or the facilities' official websites."
)

NEWSWEEK_RANKING_URL = "https://rankings.newsweek.com/americas-best-weight-loss-clinics-centers-2025"

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class Clinic(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    procedures: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class ClinicsExtraction(BaseModel):
    texas: Optional[Clinic] = None
    florida: Optional[Clinic] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_clinics() -> str:
    return (
        "Extract information for two qualifying weight loss clinics referenced in the answer: one located in Texas "
        "and one located in Florida. Use only the information explicitly present in the answer text.\n\n"
        "For each state (Texas and Florida), extract the following fields as a JSON object:\n"
        "- name: The official facility name as stated in the answer (ideally matching the Newsweek listing name). If not present, set null.\n"
        "- city: The city where the facility is located. If not present, set null.\n"
        "- state: The state as written in the answer (e.g., 'Texas', 'TX', 'Florida', 'FL'). If not present, set null.\n"
        "- procedures: A list of bariatric surgery procedures the facility offers as stated in the answer. "
        "Return an array of strings. The answer may mention terms like 'gastric bypass', 'Roux-en-Y gastric bypass', 'RYGB', "
        "'sleeve gastrectomy', 'duodenal switch', etc. If not present, return an empty array.\n"
        "- reference_urls: A list of URLs provided in the answer that can verify the information for this facility. "
        "These may include the Newsweek ranking page or the facility's official website. If none are provided, return an empty array.\n\n"
        "Notes:\n"
        "1) If the answer lists multiple clinics per state, return the first clinic mentioned for each state.\n"
        "2) Do not invent data; if a field is missing in the answer, return null or an empty array accordingly.\n"
        "3) Extract only valid URLs that are explicitly present in the answer (plain or markdown links)."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(clinic: Optional[Clinic]) -> List[str]:
    urls: List[str] = []
    if NEWSWEEK_RANKING_URL:
        urls.append(NEWSWEEK_RANKING_URL)
    if clinic and clinic.reference_urls:
        for u in clinic.reference_urls:
            if u and u not in urls:
                urls.append(u)
    return urls


# --------------------------------------------------------------------------- #
# Verification sub-tree for a clinic                                          #
# --------------------------------------------------------------------------- #
async def verify_clinic(
    evaluator: Evaluator,
    parent_node,
    clinic: Optional[Clinic],
    state_target: str,
    prefix: str
) -> None:
    """
    Build verification nodes for one clinic (Texas or Florida) and run verifications.
    """
    # Container for the clinic in the given state
    facility_node = evaluator.add_parallel(
        id=f"{prefix}_facility",
        desc=f"Identification of a qualifying weight loss clinic located in {state_target}",
        parent=parent_node,
        critical=False
    )

    # Basic facility information (critical)
    basic_info_node = evaluator.add_parallel(
        id=f"{prefix}_basic_info",
        desc="Basic facility identification information",
        parent=facility_node,
        critical=True
    )

    # name existence
    name_exists = bool(clinic and clinic.name and clinic.name.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id=f"{prefix}_name",
        desc="The official facility name is provided",
        parent=basic_info_node,
        critical=True
    )

    # newsweek listing membership
    newsweek_listing_node = evaluator.add_leaf(
        id=f"{prefix}_newsweek_listing",
        desc="The facility appears on the official Newsweek America's Best Weight Loss Clinics & Centers 2025 ranking",
        parent=basic_info_node,
        critical=True
    )
    fac_name = clinic.name if clinic and clinic.name else ""
    listing_claim = (
        f"The Newsweek 'America's Best Weight Loss Clinics & Centers 2025' ranking page lists a facility named '{fac_name}'."
    )
    await evaluator.verify(
        claim=listing_claim,
        node=newsweek_listing_node,
        sources=NEWSWEEK_RANKING_URL,
        additional_instruction=(
            "Search the ranking page for the facility name. Allow minor variants (e.g., punctuation, LLC/Inc suffixes, "
            "health system naming differences). If the page is irrelevant or inaccessible, mark as not supported."
        )
    )

    # state verification
    state_check_node = evaluator.add_leaf(
        id=f"{prefix}_state",
        desc=f"The facility is located in the state of {state_target}",
        parent=basic_info_node,
        critical=True
    )
    state_claim = (
        f"The facility named '{fac_name}' is located in {state_target} (abbreviations like TX/FL are acceptable)."
    )
    await evaluator.verify(
        claim=state_claim,
        node=state_check_node,
        sources=_combine_sources(clinic),
        additional_instruction=(
            "Verify the facility's state as shown on the Newsweek ranking page or the facility's official website. "
            "Treat state abbreviations (TX, FL) as equivalent to full names (Texas, Florida)."
        )
    )

    # city existence (only ensure it is specified in the answer extraction)
    city_exists = bool(clinic and clinic.city and clinic.city.strip())
    evaluator.add_custom_node(
        result=city_exists,
        id=f"{prefix}_city",
        desc="The city where the facility is located is specified",
        parent=basic_info_node,
        critical=True
    )

    # Treatment offerings (critical)
    treatment_node = evaluator.add_parallel(
        id=f"{prefix}_treatment",
        desc="Verification of bariatric surgery treatment offerings",
        parent=facility_node,
        critical=True
    )

    # bariatric_surgery offered
    bariatric_offered_node = evaluator.add_leaf(
        id=f"{prefix}_bariatric_surgery",
        desc="The facility offers bariatric surgery procedures",
        parent=treatment_node,
        critical=True
    )
    bariatric_claim = (
        f"The facility named '{fac_name}' offers bariatric surgery procedures (e.g., gastric bypass, sleeve gastrectomy, duodenal switch)."
    )
    await evaluator.verify(
        claim=bariatric_claim,
        node=bariatric_offered_node,
        sources=_combine_sources(clinic),
        additional_instruction=(
            "Confirm via the Newsweek page or the official website that the facility provides bariatric surgery services. "
            "It's acceptable if the page lists 'bariatric surgery' generally without enumerating all procedures."
        )
    )

    # gastric bypass offered
    gastric_bypass_node = evaluator.add_leaf(
        id=f"{prefix}_gastric_bypass",
        desc="Gastric bypass surgery is among the bariatric procedures offered",
        parent=treatment_node,
        critical=True
    )
    gb_claim = (
        f"The facility named '{fac_name}' offers gastric bypass surgery (including synonymous phrasing such as Roux-en-Y gastric bypass, RYGB, or RNY)."
    )
    await evaluator.verify(
        claim=gb_claim,
        node=gastric_bypass_node,
        sources=_combine_sources(clinic),
        additional_instruction=(
            "Explicitly check for 'gastric bypass' terms or common synonyms like 'Roux-en-Y gastric bypass', 'RYGB', or 'RNY' on the sources."
        )
    )

    # treatment documentation of listed procedures
    procedures_list = clinic.procedures if clinic else []
    procedures_text = ", ".join(procedures_list) if procedures_list else ""
    treatment_doc_node = evaluator.add_leaf(
        id=f"{prefix}_treatment_documentation",
        desc="The treatment types are documented through verifiable sources (Newsweek listing or facility's official information)",
        parent=treatment_node,
        critical=True
    )
    doc_claim = (
        f"The sources explicitly state or strongly imply that the following procedures are offered: {procedures_text}."
        if procedures_text
        else "The sources explicitly state or strongly imply bariatric procedures offered by the facility."
    )
    await evaluator.verify(
        claim=doc_claim,
        node=treatment_doc_node,
        sources=_combine_sources(clinic),
        additional_instruction=(
            "Match the procedures listed in the answer to the wording on the sources. Allow minor naming variants "
            "and synonyms for bariatric procedures. If the answer lists procedures but they are not supported by sources, mark as not supported."
        )
    )

    # Reference URL container (critical) with existence + validity checks
    reference_node = evaluator.add_parallel(
        id=f"{prefix}_reference",
        desc="A reference URL is provided that links to a valid source (Newsweek ranking page or facility website)",
        parent=facility_node,
        critical=True
    )

    # reference provided
    ref_provided = bool(clinic and clinic.reference_urls and len(clinic.reference_urls) > 0 and clinic.reference_urls[0].strip())
    evaluator.add_custom_node(
        result=ref_provided,
        id=f"{prefix}_reference_provided",
        desc="A reference URL is provided",
        parent=reference_node,
        critical=True
    )

    # reference validity
    reference_valid_node = evaluator.add_leaf(
        id=f"{prefix}_reference_valid",
        desc="Provided reference URL(s) is a valid source (Newsweek ranking page or official facility website)",
        parent=reference_node,
        critical=True
    )
    ref_urls = clinic.reference_urls if clinic and clinic.reference_urls else []
    valid_claim = (
        f"Each provided reference URL is either the official Newsweek 2025 ranking page or the official website of the facility named '{fac_name}'."
        if len(ref_urls) > 1 else
        f"The provided reference URL is either the official Newsweek 2025 ranking page or the official website of the facility named '{fac_name}'."
    )
    # Verify against multiple URLs (if multiple) so any valid one suffices
    sources_for_validity = ref_urls if ref_urls else [NEWSWEEK_RANKING_URL]
    await evaluator.verify(
        claim=valid_claim,
        node=reference_valid_node,
        sources=sources_for_validity,
        additional_instruction=(
            "Treat the URL as valid if it is exactly the Newsweek ranking page for 2025 or appears to be the facility's official website "
            "(e.g., domain name aligns with the facility brand, site includes contact/about pages, and clearly identifies the facility)."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Newsweek 2025 weight loss clinics task.
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
        default_model=model
    )

    # Extract clinic info for Texas and Florida
    extracted = await evaluator.extract(
        prompt=prompt_extract_clinics(),
        template_class=ClinicsExtraction,
        extraction_name="clinics_extraction"
    )

    # Build and verify Texas facility sub-tree
    await verify_clinic(
        evaluator=evaluator,
        parent_node=root,
        clinic=extracted.texas,
        state_target="Texas",
        prefix="texas"
    )

    # Build and verify Florida facility sub-tree
    await verify_clinic(
        evaluator=evaluator,
        parent_node=root,
        clinic=extracted.florida,
        state_target="Florida",
        prefix="florida"
    )

    # Return structured evaluation summary
    return evaluator.get_summary()