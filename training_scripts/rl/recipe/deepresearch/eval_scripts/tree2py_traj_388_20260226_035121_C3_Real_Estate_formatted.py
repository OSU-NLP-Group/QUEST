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
TASK_ID = "developer_ss_athenia"
TASK_DESCRIPTION = (
    "Identify the real estate developer who became the youngest certified public accountant (CPA) in the state of Texas "
    "in 1938 at the age of 24, served in the U.S. Navy during World War II with the initial rank of ensign, and married on "
    "August 15, 1942. Then identify the developer's spouse by providing their full name including maiden name. The spouse must "
    "have survived the sinking of the SS Athenia on September 3, 1939, which was the first British ship to be sunk by a Nazi "
    "U-boat during World War II. Provide the names of both the developer and the spouse, along with supporting URL references "
    "that verify: (1) the developer's CPA achievement at age 24 in 1938, (2) the spouse's full name including maiden name, "
    "and (3) the spouse's survival of the SS Athenia sinking."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DeveloperInfo(BaseModel):
    name: Optional[str] = None
    name_sources: List[str] = Field(default_factory=list)

    cpa_year: Optional[str] = None
    cpa_age_at_certification: Optional[str] = None
    cpa_youngest_note: Optional[str] = None
    cpa_sources: List[str] = Field(default_factory=list)

    navy_initial_rank: Optional[str] = None
    navy_sources: List[str] = Field(default_factory=list)

    marriage_date: Optional[str] = None
    marriage_sources: List[str] = Field(default_factory=list)


class SpouseInfo(BaseModel):
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    maiden_name: Optional[str] = None
    full_name_sources: List[str] = Field(default_factory=list)

    survival_sources: List[str] = Field(default_factory=list)


class CoreExtraction(BaseModel):
    developer: Optional[DeveloperInfo] = None
    spouse: Optional[SpouseInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_core() -> str:
    return """
    From the provided answer, extract the following structured information about the developer and the spouse. 
    Extract ONLY what is explicitly stated in the answer. If a field is not present, set it to null or an empty list as appropriate.

    For the developer:
    - name: The full name of the real estate developer.
    - name_sources: A list of URLs explicitly provided in the answer that document the developer's identity (biographies, official pages, credible profiles).
    - cpa_year: The year the developer became a certified public accountant (CPA).
    - cpa_age_at_certification: The age at which they became a CPA (as written in the answer).
    - cpa_youngest_note: Any text indicating the developer was the youngest CPA in Texas (as written in the answer).
    - cpa_sources: A list of URLs explicitly provided that support the CPA claim (year 1938, age 24, youngest in Texas).
    - navy_initial_rank: The initial rank given for the developer’s service in the U.S. Navy (e.g., "ensign") as stated in the answer.
    - navy_sources: A list of URLs explicitly provided that support the Navy service and rank.
    - marriage_date: The marriage date provided in the answer (keep as text exactly as shown).
    - marriage_sources: A list of URLs explicitly provided that support the marriage date.

    For the spouse:
    - full_name: The spouse’s full name including maiden name if present in the answer.
    - first_name: The first name of the spouse (as written).
    - maiden_name: The spouse’s maiden name (as written).
    - full_name_sources: A list of URLs explicitly provided that document the spouse’s full name (including maiden name) as the spouse of the developer.
    - survival_sources: A list of URLs explicitly provided that document the spouse’s survival of the SS Athenia sinking.

    SPECIAL RULES FOR URL FIELDS:
    - Extract only actual URLs that appear in the answer (plain URLs or URLs inside markdown links). Do not invent or infer URLs.
    - Return an empty list when no URLs are provided in the answer.
    - If a URL is missing a protocol, prepend http://.

    Return a JSON object with fields:
    {
      "developer": { ... },
      "spouse": { ... }
    }
    """


# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #
def _list_or_empty(maybe_list: Optional[List[str]]) -> List[str]:
    return maybe_list if isinstance(maybe_list, list) else []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_developer_checks(evaluator: Evaluator, parent_node, info: Optional[DeveloperInfo]) -> None:
    """
    Build the 'Developer_Identification' subtree with critical checks as per rubric.
    """
    dev = info or DeveloperInfo()
    dev_name = dev.name or "the developer"

    # Developer_Identification (critical, parallel)
    dev_root = evaluator.add_parallel(
        id="Developer_Identification",
        desc="Identify the real estate developer who became the youngest CPA in Texas in 1938 at age 24, served in the U.S. Navy during WWII, and married on August 15, 1942",
        parent=parent_node,
        critical=True,
    )

    # Developer_Name (critical, parallel)
    dev_name_node = evaluator.add_parallel(
        id="Developer_Name",
        desc="Provide the developer's name",
        parent=dev_root,
        critical=True,
    )

    # Name_URL_Reference leaf
    name_ref_leaf = evaluator.add_leaf(
        id="Name_URL_Reference",
        desc="Provide a URL reference that documents the developer's identity",
        parent=dev_name_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page identifies the real estate developer as {dev_name}. It is about the same individual referenced in the answer.",
        node=name_ref_leaf,
        sources=_list_or_empty(dev.name_sources),
        additional_instruction="Verify that the page is about this real estate developer (biography/profile/obituary/official page). Minor variations in full name are acceptable as long as it refers to the same person."
    )

    # CPA_Achievement (critical, parallel; single leaf for URL)
    cpa_node = evaluator.add_parallel(
        id="CPA_Achievement",
        desc="Verify that the developer became a certified public accountant in Texas in 1938 at age 24 and was the youngest CPA in Texas at that time",
        parent=dev_root,
        critical=True,
    )

    # Optional: ensure sources provided (task explicitly requires this reference)
    cpa_sources_present = evaluator.add_custom_node(
        result=len(_list_or_empty(dev.cpa_sources)) > 0,
        id="CPA_Sources_Provided",
        desc="CPA achievement source(s) are provided",
        parent=cpa_node,
        critical=True,
    )

    cpa_leaf = evaluator.add_leaf(
        id="CPA_URL_Reference",
        desc="Provide a URL reference that documents the developer's CPA achievement, age, and year",
        parent=cpa_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page states that {dev_name} became a certified public accountant in Texas in 1938 at the age of 24, and that he was the youngest CPA in Texas at that time.",
        node=cpa_leaf,
        sources=_list_or_empty(dev.cpa_sources),
        additional_instruction="All three elements should be supported: (1) CPA in Texas, (2) in 1938, (3) at age 24, and note that he was the youngest in the state at that time. Minor wording differences are fine."
    )

    # Navy_Service (critical, parallel; single leaf for URL)
    navy_node = evaluator.add_parallel(
        id="Navy_Service",
        desc="Verify that the developer served in the U.S. Navy during World War II with the initial rank of ensign",
        parent=dev_root,
        critical=True,
    )
    navy_leaf = evaluator.add_leaf(
        id="Navy_URL_Reference",
        desc="Provide a URL reference that documents the developer's Navy service and rank",
        parent=navy_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page states that {dev_name} served in the U.S. Navy during World War II with the initial rank of ensign.",
        node=navy_leaf,
        sources=_list_or_empty(dev.navy_sources),
        additional_instruction="Look for phrasing that the person joined or served as an 'ensign'. Variants like 'Ensign' capitalization, or mentions of WWII context are acceptable."
    )

    # Marriage_Date (critical, parallel; single leaf for URL)
    marriage_node = evaluator.add_parallel(
        id="Marriage_Date",
        desc="Verify that the developer married on August 15, 1942",
        parent=dev_root,
        critical=True,
    )
    marriage_leaf = evaluator.add_leaf(
        id="Marriage_URL_Reference",
        desc="Provide a URL reference that documents the marriage date",
        parent=marriage_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page states that {dev_name} married on August 15, 1942.",
        node=marriage_leaf,
        sources=_list_or_empty(dev.marriage_sources),
        additional_instruction="Allow date formats like 'August 15, 1942', '15 August 1942', or numeric formats that clearly correspond to the same date."
    )


async def build_spouse_checks(evaluator: Evaluator, parent_node, dev: Optional[DeveloperInfo], info: Optional[SpouseInfo]) -> None:
    """
    Build the 'Spouse_Identification' subtree with critical checks as per rubric.
    """
    spouse = info or SpouseInfo()
    dev_name = (dev.name if dev and dev.name else "the developer")
    spouse_full = spouse.full_name or "the spouse"

    # Spouse_Identification (critical, parallel)
    spouse_root = evaluator.add_parallel(
        id="Spouse_Identification",
        desc="Identify the spouse who married the developer on August 15, 1942 and survived the SS Athenia sinking on September 3, 1939",
        parent=parent_node,
        critical=True,
    )

    # Spouse_Full_Name (critical, parallel)
    full_name_node = evaluator.add_parallel(
        id="Spouse_Full_Name",
        desc="Provide the spouse's full name including maiden name",
        parent=spouse_root,
        critical=True,
    )

    # First_Name leaf (critical)
    first_name_leaf = evaluator.add_leaf(
        id="First_Name",
        desc="Verify the spouse's first name is Margaret",
        parent=full_name_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The spouse's first name is Margaret.",
        node=first_name_leaf,
        sources=_list_or_empty(spouse.full_name_sources),
        additional_instruction=f"Verify that the spouse of {dev_name} has first name 'Margaret'. Pages may mention 'Peggy' as a nickname; if it is clearly for Margaret, consider it acceptable."
    )

    # Maiden_Name leaf (critical)
    maiden_name_leaf = evaluator.add_leaf(
        id="Maiden_Name",
        desc="Verify the spouse's maiden name is Doggett",
        parent=full_name_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The spouse's maiden name is Doggett.",
        node=maiden_name_leaf,
        sources=_list_or_empty(spouse.full_name_sources),
        additional_instruction=f"Verify that {spouse_full} (spouse of {dev_name}) has maiden name 'Doggett'. Minor spelling/case variations are acceptable only if clearly the same surname."
    )

    # Ensure full-name sources are provided (task explicitly requires this)
    full_name_sources_present = evaluator.add_custom_node(
        result=len(_list_or_empty(spouse.full_name_sources)) > 0,
        id="Full_Name_Sources_Provided",
        desc="Spouse full name (including maiden name) source(s) are provided",
        parent=full_name_node,
        critical=True,
    )

    # Full_Name_URL_Reference leaf (critical)
    full_name_ref_leaf = evaluator.add_leaf(
        id="Full_Name_URL_Reference",
        desc="Provide a URL reference that documents the spouse's full name including maiden name",
        parent=full_name_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page identifies the spouse of {dev_name} as {spouse_full}, and shows the maiden name 'Doggett'.",
        node=full_name_ref_leaf,
        sources=_list_or_empty(spouse.full_name_sources),
        additional_instruction="It is acceptable if the page shows the full married name while also explicitly stating the maiden name 'Doggett'."
    )

    # Ship_Survival_Event (critical, sequential)
    survival_seq = evaluator.add_sequential(
        id="Ship_Survival_Event",
        desc="Verify that the spouse survived the sinking of SS Athenia on September 3, 1939",
        parent=spouse_root,
        critical=True,
    )

    # Ship_Identification (critical, parallel)
    ship_ident_node = evaluator.add_parallel(
        id="Ship_Identification",
        desc="Verify that the ship was named SS Athenia",
        parent=survival_seq,
        critical=True,
    )

    # Leaf to explicitly check ship name is SS Athenia
    ship_name_leaf = evaluator.add_leaf(
        id="Ship_Name_Is_Athenia",
        desc="Confirm the ship name SS Athenia is correctly identified",
        parent=ship_ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The ship involved in the incident was named SS Athenia.",
        node=ship_name_leaf,
        sources=_list_or_empty(spouse.survival_sources),
        additional_instruction="The page should clearly reference the ship 'SS Athenia'."
    )

    # Event_Historical_Details (critical, parallel)
    event_hist_node = evaluator.add_parallel(
        id="Event_Historical_Details",
        desc="Verify that SS Athenia was the first British ship sunk by a Nazi U-boat during World War II and that the sinking occurred on September 3, 1939",
        parent=ship_ident_node,
        critical=True,
    )

    # Ensure survival sources are provided (task explicitly requires this)
    survival_sources_present = evaluator.add_custom_node(
        result=len(_list_or_empty(spouse.survival_sources)) > 0,
        id="Survival_Sources_Provided",
        desc="Survival event source(s) are provided",
        parent=event_hist_node,
        critical=True,
    )

    # Historical details leaf (critical)
    hist_details_leaf = evaluator.add_leaf(
        id="Event_History_Details_Check",
        desc="Verify SS Athenia was the first British ship sunk by a Nazi U-boat during WWII and that the sinking occurred on September 3, 1939",
        parent=event_hist_node,
        critical=True,
    )
    await evaluator.verify(
        claim="SS Athenia was the first British ship sunk by a Nazi U-boat during World War II, and the sinking occurred on September 3, 1939.",
        node=hist_details_leaf,
        sources=_list_or_empty(spouse.survival_sources),
        additional_instruction="Wording can vary; both facts must be supported by the provided source(s): (1) first British ship sunk by a Nazi U-boat in WWII; (2) date of sinking is 3 September 1939."
    )

    # Survival_URL_Reference leaf (critical)
    survival_leaf = evaluator.add_leaf(
        id="Survival_URL_Reference",
        desc="Provide a URL reference that documents the spouse's survival of the SS Athenia sinking",
        parent=event_hist_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page states that {spouse_full} survived the sinking of the SS Athenia.",
        node=survival_leaf,
        sources=_list_or_empty(spouse.survival_sources),
        additional_instruction=f"If the page refers to the spouse by married name or maiden name, ensure it is the same person identified as the spouse of {dev_name}. It must state or clearly imply survival of the sinking."
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

    # Extract core info once
    extracted = await evaluator.extract(
        prompt=prompt_extract_core(),
        template_class=CoreExtraction,
        extraction_name="core_extraction",
    )

    # Build the top-level node per rubric
    complete_node = evaluator.add_parallel(
        id="Complete_Answer",
        desc="Provide the name of the real estate developer and the full name of their spouse (including maiden name)",
        parent=root,
        critical=True,
    )

    # Developer subtree
    await build_developer_checks(
        evaluator=evaluator,
        parent_node=complete_node,
        info=extracted.developer if extracted else None
    )

    # Spouse subtree
    await build_spouse_checks(
        evaluator=evaluator,
        parent_node=complete_node,
        dev=(extracted.developer if extracted else None),
        info=(extracted.spouse if extracted else None),
    )

    # Return summary
    return evaluator.get_summary()