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
TASK_ID = "neuralink_prime_phoenix_institution"
TASK_DESCRIPTION = (
    "In January 2024, the first human patient received a brain-computer interface implant as part of Neuralink's "
    "PRIME Study at a neurological institute in Phoenix, Arizona. Identify the full name of this medical institution "
    "and provide the following information about it: (1) The complete street address of the institution, "
    "(2) Its Newsweek 2025 national ranking for neurosurgery, "
    "(3) How many consecutive years it has held that national ranking, "
    "(4) The number of brain and spine surgeries it performed in the past fiscal year, "
    "(5) The number of neurosurgery-dedicated operating rooms it has, "
    "(6) Its Doximity ranking for the neurosurgery residency program by reputation."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ValueWithSources(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class InstitutionExtraction(BaseModel):
    institution_name: ValueWithSources = Field(default_factory=ValueWithSources)
    street_address: ValueWithSources = Field(default_factory=ValueWithSources)
    newsweek_2025_neurosurgery_ranking: ValueWithSources = Field(default_factory=ValueWithSources)
    consecutive_years_at_ranking: ValueWithSources = Field(default_factory=ValueWithSources)
    annual_brain_spine_surgery_count_past_fy: ValueWithSources = Field(default_factory=ValueWithSources)
    neurosurgery_dedicated_or_count: ValueWithSources = Field(default_factory=ValueWithSources)
    doximity_neurosurgery_residency_reputation_ranking: ValueWithSources = Field(default_factory=ValueWithSources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_institution_info() -> str:
    return """
    Extract, exactly as stated in the answer, the following information about the neurological institute in Phoenix, Arizona
    where the first human Neuralink PRIME Study implant surgery occurred in January 2024. For each item, also extract all URLs
    that the answer explicitly cites as sources specifically supporting that item.

    You must return a JSON object with the following fields (each field has a nested JSON object with keys 'value' and 'sources'):

    1) institution_name:
       - value: The full official name of the medical/neurological institution (not the hospital system unless they are the same).
       - sources: All URLs the answer cites that support the institution identification and its role in hosting the first human Neuralink implant in Jan 2024.

    2) street_address:
       - value: The complete street address of the institution (street number and street name, city, state, ZIP).
       - sources: All URLs the answer cites that support this address.

    3) newsweek_2025_neurosurgery_ranking:
       - value: The institution’s Newsweek 2025 national ranking for neurosurgery (return exactly as phrased, e.g., '#1', 'No. 2', '5th', 'Top 5', etc.).
       - sources: All URLs the answer cites that support this ranking.

    4) consecutive_years_at_ranking:
       - value: How many consecutive years the institution has held that Newsweek neurosurgery national ranking (exact wording or number as stated).
       - sources: All URLs the answer cites that support this "consecutive years" statement.

    5) annual_brain_spine_surgery_count_past_fy:
       - value: The number of brain and spine surgeries performed in the past fiscal year (exact count or phrasing as stated).
       - sources: All URLs the answer cites that support this count.

    6) neurosurgery_dedicated_or_count:
       - value: The number of neurosurgery-dedicated operating rooms (exact number or phrasing as stated).
       - sources: All URLs the answer cites that support this OR count.

    7) doximity_neurosurgery_residency_reputation_ranking:
       - value: The Doximity by-reputation ranking of the institution’s neurosurgery residency program (exact phrasing or ordinal as stated).
       - sources: All URLs the answer cites that support this ranking. If the answer cites an institutional page quoting Doximity, include that URL.

    IMPORTANT RULES:
    - Extract only what is explicitly present in the answer. Do not infer or fabricate.
    - For each 'sources' list, include only URLs explicitly present in the answer text. If no URL is present for an item, return an empty list for that item.
    - For addresses and numeric values, keep the exact formatting as written (e.g., thousands separators, 'approximately', 'about', etc.).
    - If any requested value is missing from the answer, set its 'value' to null and 'sources' to an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for url in lst or []:
            if isinstance(url, str):
                u = url.strip()
                if u and u not in seen:
                    out.append(u)
                    seen.add(u)
    return out


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: InstitutionExtraction) -> None:
    """
    Build the verification tree following the rubric and run verifications.
    """
    # Parent node (critical, parallel aggregation)
    info_node = evaluator.add_parallel(
        id="Medical_Institution_Information",
        desc="Verify that the answer correctly identifies the institution and correctly provides all requested attributes about it",
        parent=evaluator.root,
        critical=True
    )

    # 1) Institution Name
    name_leaf = evaluator.add_leaf(
        id="Institution_Name",
        desc="Correctly identifies the full name of the neurological institute in Phoenix, Arizona that hosted the first human Neuralink PRIME Study implant surgery in January 2024",
        parent=info_node,
        critical=True
    )
    name_val = extracted.institution_name.value or ""
    name_sources = _combine_sources(extracted.institution_name.sources)
    name_claim = (
        f"The first human Neuralink PRIME Study implant surgery in January 2024 took place at '{name_val}', "
        f"a neurological institute located in Phoenix, Arizona."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=name_sources,
        additional_instruction=(
            "Verify that the cited page(s) clearly state that the first human Neuralink PRIME Study implant in January 2024 "
            "occurred at the named neurological institute and that it is located in Phoenix, Arizona. "
            "Allow minor name formatting differences (e.g., punctuation, 'Institute' vs. 'Neurological Institute') as long as it refers to the same organization."
        )
    )

    # 2) Street Address
    addr_leaf = evaluator.add_leaf(
        id="Street_Address",
        desc="Correctly provides the institution’s complete street address",
        parent=info_node,
        critical=True
    )
    address_val = extracted.street_address.value or ""
    address_sources = _combine_sources(extracted.street_address.sources, name_sources)
    addr_claim = f"The complete street address of '{name_val}' is '{address_val}'."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_leaf,
        sources=address_sources,
        additional_instruction=(
            "Check that the page(s) provide a complete postal address including street number, street name, city, state, and ZIP code. "
            "Accept minor formatting differences (e.g., abbreviations like 'Rd' vs 'Road')."
        )
    )

    # 3) Newsweek 2025 national ranking for neurosurgery
    newsweek_leaf = evaluator.add_leaf(
        id="Newsweek_National_Ranking",
        desc="Correctly provides the institution’s Newsweek 2025 national ranking for neurosurgery",
        parent=info_node,
        critical=True
    )
    nw_val = extracted.newsweek_2025_neurosurgery_ranking.value or ""
    nw_sources = _combine_sources(extracted.newsweek_2025_neurosurgery_ranking.sources, name_sources)
    nw_claim = (
        f"According to Newsweek's 2025 U.S. national ranking for neurosurgery (or Neurology & Neurosurgery specialized hospitals), "
        f"'{name_val}' is ranked '{nw_val}'."
    )
    await evaluator.verify(
        claim=nw_claim,
        node=newsweek_leaf,
        sources=nw_sources,
        additional_instruction=(
            "Focus on Newsweek 2025 rankings pertinent to neurosurgery (or Neurology & Neurosurgery specialized hospitals if that's the labeling used). "
            "Confirm the stated rank; accept small stylistic variants (e.g., '#1' vs 'No. 1' vs '1st')."
        )
    )

    # 4) Consecutive years holding that ranking
    years_leaf = evaluator.add_leaf(
        id="Consecutive_Years_At_Ranking",
        desc="Correctly provides how many consecutive years the institution has held that Newsweek national ranking",
        parent=info_node,
        critical=True
    )
    years_val = extracted.consecutive_years_at_ranking.value or ""
    years_sources = _combine_sources(extracted.consecutive_years_at_ranking.sources, nw_sources)
    years_claim = (
        f"'{name_val}' has held its Newsweek U.S. national neurosurgery ranking position for '{years_val}' consecutive years."
    )
    await evaluator.verify(
        claim=years_claim,
        node=years_leaf,
        sources=years_sources,
        additional_instruction=(
            "Confirm that the page(s) explicitly mention the number of consecutive years the institution has maintained its Newsweek neurosurgery ranking. "
            "If the page states 'for X years in a row' or similar, that counts."
        )
    )

    # 5) Annual brain and spine surgeries in past fiscal year
    surgery_leaf = evaluator.add_leaf(
        id="Annual_Surgery_Count",
        desc="Correctly provides the number of brain and spine surgeries the institution performed in the past fiscal year",
        parent=info_node,
        critical=True
    )
    surgery_val = extracted.annual_brain_spine_surgery_count_past_fy.value or ""
    surgery_sources = _combine_sources(extracted.annual_brain_spine_surgery_count_past_fy.sources, name_sources)
    surgery_claim = (
        f"In the past fiscal year, '{name_val}' performed '{surgery_val}' brain and spine surgeries."
    )
    await evaluator.verify(
        claim=surgery_claim,
        node=surgery_leaf,
        sources=surgery_sources,
        additional_instruction=(
            "Verify that the page(s) explicitly state the number of brain and spine surgeries performed in the most recent fiscal year. "
            "Accept phrasing like 'approximately' or comma separators if used."
        )
    )

    # 6) Neurosurgery-dedicated operating rooms count
    or_leaf = evaluator.add_leaf(
        id="Operating_Room_Count",
        desc="Correctly provides the number of neurosurgery-dedicated operating rooms the institution has",
        parent=info_node,
        critical=True
    )
    or_val = extracted.neurosurgery_dedicated_or_count.value or ""
    or_sources = _combine_sources(extracted.neurosurgery_dedicated_or_count.sources, name_sources)
    or_claim = (
        f"'{name_val}' has '{or_val}' neurosurgery-dedicated operating rooms."
    )
    await evaluator.verify(
        claim=or_claim,
        node=or_leaf,
        sources=or_sources,
        additional_instruction=(
            "Confirm that the page(s) specify a count of operating rooms dedicated specifically to neurosurgery. "
            "Do not confuse with total hospital ORs; it must be neurosurgery-dedicated."
        )
    )

    # 7) Doximity neurosurgery residency program by-reputation ranking
    dox_leaf = evaluator.add_leaf(
        id="Residency_Program_Ranking",
        desc="Correctly provides the Doximity by-reputation ranking of the institution’s neurosurgery residency program",
        parent=info_node,
        critical=True
    )
    dox_val = extracted.doximity_neurosurgery_residency_reputation_ranking.value or ""
    dox_sources = _combine_sources(extracted.doximity_neurosurgery_residency_reputation_ranking.sources, name_sources)
    dox_claim = (
        f"According to Doximity's by-reputation ranking, the neurosurgery residency program at '{name_val}' is ranked '{dox_val}'."
    )
    await evaluator.verify(
        claim=dox_claim,
        node=dox_leaf,
        sources=dox_sources,
        additional_instruction=(
            "Confirm the Doximity reputation-based ranking for the neurosurgery residency program. "
            "If Doximity is inaccessible, an institutional page explicitly quoting Doximity is acceptable."
        )
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Neuralink PRIME Study Phoenix institution and its attributes task.
    """
    # Initialize evaluator
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
        default_model=model
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_institution_info(),
        template_class=InstitutionExtraction,
        extraction_name="institution_extraction"
    )

    # Build tree and verify according to rubric
    await build_and_verify_tree(evaluator, extracted)

    # Return final summary
    return evaluator.get_summary()