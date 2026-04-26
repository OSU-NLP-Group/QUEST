import asyncio
import logging
from typing import Optional, Dict, Any

from pydantic import BaseModel

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dc_ng_shooting_prosecutor_2025"
TASK_DESCRIPTION = (
    "On November 26, 2025, two members of the West Virginia National Guard were shot near Farragut West Metro Station "
    "in Washington, DC. One victim, Army Specialist Sarah Beckstrom, aged 20 from Summersville, West Virginia, died "
    "from her injuries on November 28, 2025. Identify the federal prosecutor responsible for handling this criminal case "
    "and provide the following information: (1) the prosecutor's full name, (2) their current official title, and (3) "
    "the complete physical address of their office. For each piece of information, include a supporting URL from an "
    "official U.S. government source."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProsecutorInfo(BaseModel):
    """
    Structured information about the identified federal prosecutor, as extracted from the answer.
    All fields should come directly from the answer text.
    """
    full_name: Optional[str] = None
    official_title: Optional[str] = None
    office_address: Optional[str] = None

    identity_source_url: Optional[str] = None
    title_source_url: Optional[str] = None
    address_source_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_prosecutor_info() -> str:
    return (
        "Extract the federal prosecutor details provided in the answer. Return the following fields:\n"
        "1) full_name: The prosecutor’s complete legal name.\n"
        "2) official_title: Their current official title (e.g., 'United States Attorney for the District of Columbia', "
        "'Assistant United States Attorney', etc.).\n"
        "3) office_address: The complete physical address of the prosecutor’s office, including street, city, state, and ZIP code.\n"
        "4) identity_source_url: A URL from an official U.S. government source (e.g., justice.gov or another .gov domain) that confirms the prosecutor’s identity.\n"
        "5) title_source_url: A URL from an official U.S. government source confirming the prosecutor’s current official title.\n"
        "6) address_source_url: A URL from an official U.S. government source confirming the office’s physical address.\n\n"
        "RULES:\n"
        "- Extract only what is explicitly present in the answer; do not invent any information.\n"
        "- For URLs, extract the actual URL strings as written in the answer (plain or markdown link). Do not infer URLs.\n"
        "- Prefer URLs from official U.S. government domains (e.g., *.gov, justice.gov, usdoj.gov, *.justice.gov). If multiple URLs are provided for a field, extract the first one mentioned.\n"
        "- If any field is missing in the answer, set it to null."
    )


# --------------------------------------------------------------------------- #
# Helper functions for verification instructions                              #
# --------------------------------------------------------------------------- #
def instruction_official_gov_source() -> str:
    return (
        "Conclude 'supported' only if the provided page is an official U.S. government source (e.g., a .gov domain such "
        "as justice.gov, usdoj.gov, *.justice.gov, or other federal .gov sites). Do not accept private news websites, "
        "blogs, social media, or non-government domains."
    )


def instruction_name_match(name: Optional[str]) -> str:
    nm = name or ""
    return (
        f"Verify that the webpage explicitly names the prosecutor as '{nm}' (allow reasonable variants like middle "
        "initials, capitalization, diacritics). The page must clearly identify the same person."
    )


def instruction_title_match(title: Optional[str]) -> str:
    tt = title or ""
    return (
        f"Verify that the webpage explicitly lists the person’s current official title as '{tt}', allowing minor "
        "formatting variations (e.g., 'Assistant U.S. Attorney' vs 'Assistant United States Attorney')."
    )


def instruction_address_match(address: Optional[str]) -> str:
    ad = address or ""
    return (
        f"Verify that the webpage lists the complete physical office address '{ad}'. Minor formatting differences are "
        "acceptable, but street, city, state, and ZIP must correspond."
    )


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, info: ProsecutorInfo) -> None:
    """
    Construct the verification tree as per the rubric and perform the verifications.
    """
    # Top-level critical sequential node representing the complete investigation
    investigation_node = evaluator.add_sequential(
        id="DC_National_Guard_Shooting_Prosecutor_Investigation",
        desc=("Complete investigation identifying and verifying information about the federal prosecutor handling the "
              "November 26, 2025 shooting of two National Guard members near Farragut West Metro Station, Washington DC"),
        parent=evaluator.root,
        critical=True
    )

    # Case prosecutor identification (critical sequential) – serves as an umbrella step
    case_ident_node = evaluator.add_sequential(
        id="Case_Prosecutor_Identification",
        desc="Identify the specific federal prosecutor responsible for prosecuting this case",
        parent=investigation_node,
        critical=True
    )

    # Prosecutor information details (critical parallel)
    details_node = evaluator.add_parallel(
        id="Prosecutor_Information_Details",
        desc="Provide comprehensive verified information about the identified prosecutor",
        parent=case_ident_node,
        critical=True
    )

    # ----------------------- Prosecutor Identity Group ---------------------- #
    identity_group = evaluator.add_parallel(
        id="Prosecutor_Identity",
        desc="Provide the prosecutor's complete legal name with supporting URL",
        parent=details_node,
        critical=True
    )

    # 1) Identity Source URL – verify government source nature first (gating)
    identity_source_leaf = evaluator.add_leaf(
        id="Identity_Source_URL",
        desc=("A valid URL from an official government source (e.g., justice.gov) confirming the prosecutor's identity "
              "is provided"),
        parent=identity_group,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an official U.S. government source.",
        node=identity_source_leaf,
        sources=info.identity_source_url,
        additional_instruction=instruction_official_gov_source()
    )

    # 2) Full Name – verify name is supported by the identity source page
    full_name_leaf = evaluator.add_leaf(
        id="Full_Name",
        desc="The prosecutor's full name is correctly provided",
        parent=identity_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The prosecutor's full name is '{info.full_name}'.",
        node=full_name_leaf,
        sources=info.identity_source_url,
        additional_instruction=instruction_name_match(info.full_name)
    )

    # -------------------- Current Position Title Group ---------------------- #
    title_group = evaluator.add_parallel(
        id="Current_Position_Title",
        desc="Provide the prosecutor's current official title with supporting URL",
        parent=details_node,
        critical=True
    )

    # 3) Title Source URL – verify government source nature first (gating)
    title_source_leaf = evaluator.add_leaf(
        id="Title_Source_URL",
        desc="A valid URL from an official government source confirming the prosecutor's title is provided",
        parent=title_group,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an official U.S. government source.",
        node=title_source_leaf,
        sources=info.title_source_url,
        additional_instruction=instruction_official_gov_source()
    )

    # 4) Official Title – verify title is supported by the title source page
    official_title_leaf = evaluator.add_leaf(
        id="Official_Title",
        desc="The prosecutor's official title is correctly provided",
        parent=title_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The prosecutor's current official title is '{info.official_title}'.",
        node=official_title_leaf,
        sources=info.title_source_url,
        additional_instruction=instruction_title_match(info.official_title)
    )

    # -------------------- Office Physical Location Group -------------------- #
    address_group = evaluator.add_parallel(
        id="Office_Physical_Location",
        desc="Provide the physical address of the prosecutor's office with supporting URL",
        parent=details_node,
        critical=True
    )

    # 5) Address Source URL – verify government source nature first (gating)
    address_source_leaf = evaluator.add_leaf(
        id="Address_Source_URL",
        desc="A valid URL from an official government source confirming the office address is provided",
        parent=address_group,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an official U.S. government source.",
        node=address_source_leaf,
        sources=info.address_source_url,
        additional_instruction=instruction_official_gov_source()
    )

    # 6) Physical Address – verify address is supported by the address source page
    physical_address_leaf = evaluator.add_leaf(
        id="Physical_Address",
        desc="The complete physical address of the prosecutor's office is correctly provided",
        parent=address_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The prosecutor's office physical address is '{info.office_address}'.",
        node=physical_address_leaf,
        sources=info.address_source_url,
        additional_instruction=instruction_address_match(info.office_address)
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Entry point to evaluate an agent's answer for the DC National Guard shooting prosecutor identification task.
    Returns a structured summary including the verification tree and final score.
    """
    # Initialize evaluator with a sequential root to reflect end-to-end dependency
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured prosecutor info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_prosecutor_info(),
        template_class=ProsecutorInfo,
        extraction_name="prosecutor_info"
    )

    # Record the case meta as custom info for context
    evaluator.add_custom_info(
        info={
            "incident_date": "2025-11-26",
            "victim_death_date": "2025-11-28",
            "location": "Farragut West Metro Station, Washington, DC",
            "requirement": "Identify federal prosecutor handling the case and verify name, title, office address with official government URLs."
        },
        info_type="case_context",
        info_name="dc_ng_shooting_case_context"
    )

    # Build the verification tree and run checks
    await build_verification_tree(evaluator, extracted_info)

    # Return standardized evaluation summary
    return evaluator.get_summary()