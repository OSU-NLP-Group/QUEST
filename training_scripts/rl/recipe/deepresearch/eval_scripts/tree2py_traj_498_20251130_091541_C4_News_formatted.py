import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "az_senator_2025_identification"
TASK_DESCRIPTION = (
    "Who is the U.S. Senator currently representing Arizona who assumed office on January 3, 2025, "
    "was born in Chicago, Illinois in 1979, graduated from Harvard University in 2004, served in the U.S. Marine Corps "
    "with deployment to Iraq in 2005 as part of the 3rd Battalion, 25th Marines, previously served in the Arizona House "
    "of Representatives from 2010 to 2014, and is the first Latino to represent Arizona in the United States Senate?"
)

# Optional ground-truth information (for reporting only; not enforced in verification)
GROUND_TRUTH_INFO = {
    "expected_senator_name": "Ruben Gallego",
    "assumed_office_date": "January 3, 2025",
    "state": "Arizona",
    "birth_place": "Chicago, Illinois",
    "birth_year": "1979",
    "university": "Harvard University",
    "graduation_year": "2004",
    "military_service": "United States Marine Corps",
    "deployment_year": "2005",
    "unit": "3rd Battalion, 25th Marines (3/25)",
    "az_house_service": "2010–2014",
    "first_latino_senator_from_arizona": True,
}

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class SenatorBasicExtraction(BaseModel):
    """Basic identification and sources extracted from the answer text."""
    name: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class SenatorDetailsExtraction(BaseModel):
    """Details extracted from the answer text (strings preferred to allow flexibility)."""
    state_represented: Optional[str] = None
    assumed_office_date: Optional[str] = None
    birth_place: Optional[str] = None
    birth_year: Optional[str] = None
    university: Optional[str] = None
    graduation_year: Optional[str] = None
    marine_corps_service: Optional[str] = None
    iraq_deployment_year: Optional[str] = None
    military_unit: Optional[str] = None
    az_house_service_years: Optional[str] = None
    first_latino_senator_az: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_basic() -> str:
    return (
        "Extract the individual's name (the senator identified in the answer) and all explicit source URLs mentioned "
        "anywhere in the answer. Return JSON with:\n"
        "1) name: the senator's full name (string), or null if not provided;\n"
        "2) source_urls: array of all URLs explicitly present in the answer (including plain URLs or markdown links).\n"
        "Follow URL extraction rules strictly: only extract URLs explicitly present in the answer, and include a valid "
        "protocol (http:// or https://). If none are present, return an empty array."
    )


def prompt_extract_details() -> str:
    return (
        "Extract any details the answer explicitly states about the identified senator. Return JSON fields as strings "
        "or null if missing:\n"
        "- state_represented\n"
        "- assumed_office_date\n"
        "- birth_place\n"
        "- birth_year\n"
        "- university\n"
        "- graduation_year\n"
        "- marine_corps_service\n"
        "- iraq_deployment_year\n"
        "- military_unit\n"
        "- az_house_service_years\n"
        "- first_latino_senator_az\n"
        "Do not invent information. Extract exactly what appears in the answer text."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def subject_name(extracted: SenatorBasicExtraction) -> str:
    """Return the subject name or a neutral placeholder if not provided."""
    if extracted and extracted.name and extracted.name.strip():
        return extracted.name.strip()
    return "the identified individual"


def sources_or_none(urls: List[str]) -> Optional[List[str]]:
    """Return the list of URLs if non-empty, otherwise None to route simple verification."""
    return urls if urls else None


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_senator_identification(
    evaluator: Evaluator,
    root_node,
    basic: SenatorBasicExtraction,
    details: SenatorDetailsExtraction,
) -> None:
    """
    Build the critical parallel node 'Senator_Identification' and attach all critical leaf verifications.
    """
    ident_node = evaluator.add_parallel(
        id="Senator_Identification",
        desc="Correct identification of the U.S. Senator meeting all specified biographical and career criteria",
        parent=root_node,
        critical=True,
    )

    subject = subject_name(basic)
    srcs = sources_or_none(basic.source_urls)

    # 1) Current_Senate_Service_Arizona
    n1 = evaluator.add_leaf(
        id="Current_Senate_Service_Arizona",
        desc="The individual currently serves as a U.S. Senator representing Arizona",
        parent=ident_node,
        critical=True,
    )
    claim1 = f"{subject} currently serves as a U.S. Senator representing the state of Arizona."
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=srcs,
        additional_instruction=(
            "Confirm that the person is a current U.S. Senator for Arizona. Accept equivalent phrasings such as "
            "'Senator from Arizona' or 'represents Arizona in the U.S. Senate'. Evidence should clearly indicate "
            "current officeholding status."
        ),
    )

    # 2) Senate_Assumption_Date
    n2 = evaluator.add_leaf(
        id="Senate_Assumption_Date",
        desc="The individual assumed office in the U.S. Senate on January 3, 2025",
        parent=ident_node,
        critical=True,
    )
    claim2 = f"{subject} assumed office in the United States Senate on January 3, 2025."
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=srcs,
        additional_instruction=(
            "Look for 'Assumed office' or swearing-in date on official or biographical sources. Minor wording "
            "variations are fine, but the date must be January 3, 2025."
        ),
    )

    # 3) Birth_Location_Chicago
    n3 = evaluator.add_leaf(
        id="Birth_Location_Chicago",
        desc="The individual was born in Chicago, Illinois",
        parent=ident_node,
        critical=True,
    )
    claim3 = f"{subject} was born in Chicago, Illinois."
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=srcs,
        additional_instruction="Accept 'Chicago, IL' as equivalent. The birthplace must be clearly indicated.",
    )

    # 4) Birth_Year_1979
    n4 = evaluator.add_leaf(
        id="Birth_Year_1979",
        desc="The individual was born in 1979",
        parent=ident_node,
        critical=True,
    )
    claim4 = f"{subject} was born in 1979."
    await evaluator.verify(
        claim=claim4,
        node=n4,
        sources=srcs,
        additional_instruction=(
            "Confirm the year of birth is 1979. Accept evidence that shows a full birthdate in 1979."
        ),
    )

    # 5) Harvard_Graduation_2004
    n5 = evaluator.add_leaf(
        id="Harvard_Graduation_2004",
        desc="The individual graduated from Harvard University in 2004",
        parent=ident_node,
        critical=True,
    )
    claim5 = f"{subject} graduated from Harvard University in 2004."
    await evaluator.verify(
        claim=claim5,
        node=n5,
        sources=srcs,
        additional_instruction=(
            "Accept 'Harvard College' or 'Harvard University' as equivalent if graduation is in 2004."
        ),
    )

    # 6) Marine_Corps_Service
    n6 = evaluator.add_leaf(
        id="Marine_Corps_Service",
        desc="The individual served in the U.S. Marine Corps",
        parent=ident_node,
        critical=True,
    )
    claim6 = f"{subject} served in the United States Marine Corps."
    await evaluator.verify(
        claim=claim6,
        node=n6,
        sources=srcs,
        additional_instruction=(
            "Service in the USMC (active or reserve) counts. Accept 'United States Marine Corps Reserve' as valid."
        ),
    )

    # 7) Iraq_Deployment_2005
    n7 = evaluator.add_leaf(
        id="Iraq_Deployment_2005",
        desc="The individual was deployed to Iraq in 2005",
        parent=ident_node,
        critical=True,
    )
    claim7 = f"{subject} was deployed to Iraq in 2005."
    await evaluator.verify(
        claim=claim7,
        node=n7,
        sources=srcs,
        additional_instruction=(
            "Evidence should indicate deployment to Iraq in 2005. Accept phrasing that includes the year 2005 "
            "and deployment to Iraq, even if ranges like 2005–2006 are mentioned."
        ),
    )

    # 8) Military_Unit_3_25
    n8 = evaluator.add_leaf(
        id="Military_Unit_3_25",
        desc="The individual served with the 3rd Battalion, 25th Marines",
        parent=ident_node,
        critical=True,
    )
    claim8 = f"{subject} served with the 3rd Battalion, 25th Marines."
    await evaluator.verify(
        claim=claim8,
        node=n8,
        sources=srcs,
        additional_instruction=(
            "Accept equivalent unit naming such as '3/25', '3rd Battalion, 25th Marine Regiment', or '3rd Battalion, "
            "25th Marines'."
        ),
    )

    # 9) Arizona_House_Service
    n9 = evaluator.add_leaf(
        id="Arizona_House_Service",
        desc="The individual served in the Arizona House of Representatives from 2010 to 2014",
        parent=ident_node,
        critical=True,
    )
    claim9 = f"{subject} served in the Arizona House of Representatives from 2010 to 2014."
    await evaluator.verify(
        claim=claim9,
        node=n9,
        sources=srcs,
        additional_instruction=(
            "Confirm service in the Arizona House with the inclusive years 2010 through 2014. Minor phrasing variations "
            "are acceptable if the timespan matches."
        ),
    )

    # 10) First_Latino_Senator_Arizona
    n10 = evaluator.add_leaf(
        id="First_Latino_Senator_Arizona",
        desc="The individual is the first Latino to represent Arizona in the U.S. Senate",
        parent=ident_node,
        critical=True,
    )
    claim10 = f"{subject} is the first Latino to represent Arizona in the United States Senate."
    await evaluator.verify(
        claim=claim10,
        node=n10,
        sources=srcs,
        additional_instruction=(
            "Confirm the historical 'first Latino' milestone for Arizona's U.S. Senate representation. Accept 'Hispanic' "
            "as equivalent to 'Latino' in credible reporting."
        ),
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
    """
    Evaluate the agent's answer for the Arizona Senator identification task.
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

    # Extract basic info and details from the answer
    basic_info = await evaluator.extract(
        prompt=prompt_extract_basic(),
        template_class=SenatorBasicExtraction,
        extraction_name="senator_basic",
    )

    details_info = await evaluator.extract(
        prompt=prompt_extract_details(),
        template_class=SenatorDetailsExtraction,
        extraction_name="senator_details",
    )

    # Record ground truth for reference in summary
    evaluator.add_ground_truth(GROUND_TRUTH_INFO, gt_type="ground_truth")

    # Build verification tree and run checks
    await build_and_verify_senator_identification(evaluator, root, basic_info, details_info)

    # Return standardized summary with verification tree
    return evaluator.get_summary()