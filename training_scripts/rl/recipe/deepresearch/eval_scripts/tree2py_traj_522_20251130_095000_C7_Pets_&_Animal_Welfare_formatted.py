import asyncio
import logging
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "texas_dog_training_care_facility"
TASK_DESCRIPTION = (
    "Identify a dog training and care facility located in Texas that meets ALL of the following 12 requirements: "
    "(1) Offers AKC Canine Good Citizen (CGC) certification program, "
    "(2) Offers therapy dog preparation or training program, "
    "(3) Has at least one CCPDT (Certification Council for Professional Dog Trainers) certified trainer on staff, "
    "(4) Offers puppy training classes for dogs under 6 months of age, "
    "(5) Offers advanced obedience training classes beyond basic commands, "
    "(6) Offers private one-on-one training sessions, "
    "(7) Has an indoor training facility or building, "
    "(8) Offers overnight dog boarding services, "
    "(9) Offers dog daycare services, "
    "(10) Requires proof of current vaccinations including rabies and DHPP for all dogs, "
    "(11) Operates at least 6 days per week, "
    "(12) Accepts dogs of all breeds without breed-specific restrictions. "
    "Provide the facility name, city location in Texas, and reference URL for verification."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacilityExtraction(BaseModel):
    facility_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Expect "TX" or "Texas" if provided
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_facility_info() -> str:
    return (
        "From the answer, extract the following fields about the identified facility:\n"
        "1. facility_name: The facility's official name.\n"
        "2. city: The city where the facility is located.\n"
        "3. state: The state (use 'TX' or 'Texas' if provided).\n"
        "4. reference_urls: All URLs cited for verification. Include website pages, policy pages, services pages, "
        "   hours pages, or any other referenced URLs. Return only actual URLs explicitly present in the answer.\n"
        "Return a single JSON object with these fields. If any are missing, set them to null (or [] for reference_urls)."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_sources(urls: List[str]) -> List[str]:
    """Ensure we have a list of non-empty URLs."""
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def _safe_name(name: Optional[str]) -> str:
    return name.strip() if name else ""


def _safe_city(city: Optional[str]) -> str:
    return city.strip() if city else ""


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def verify_texas_facility(
    evaluator: Evaluator,
    parent_node,
    info: FacilityExtraction,
) -> None:
    """
    Build the critical parallel node for all requirements and perform verifications.
    """
    name = _safe_name(info.facility_name)
    city = _safe_city(info.city)
    sources = _normalize_sources(info.reference_urls)

    # Create main critical node to gate all requirements
    facility_node = evaluator.add_parallel(
        id="TexasDogTrainingCareFacility",
        desc="Identify a dog training and care facility located in Texas and provide required identifying info, such that it meets all listed service and policy requirements.",
        parent=parent_node,
        critical=True,
    )

    # 1) Existence checks (custom critical nodes)
    evaluator.add_custom_node(
        result=bool(name),
        id="Facility_Name_Provided",
        desc="Answer provides the facility name.",
        parent=facility_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(city),
        id="City_Provided",
        desc="Answer provides a city location.",
        parent=facility_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(sources),
        id="Reference_URL_Provided",
        desc="Answer provides at least one reference URL for verification.",
        parent=facility_node,
        critical=True,
    )

    # 2) Verify the city location is in Texas (leaf)
    city_tx_leaf = evaluator.add_leaf(
        id="City_Location_In_Texas_Provided",
        desc="Answer provides a city location that is in Texas (TX).",
        parent=facility_node,
        critical=True,
    )
    city_claim = (
        f"The facility '{name}' is located in {city}, Texas (TX). "
        f"If the page shows the address in {city}, TX or 'Texas', consider it supported."
    )
    await evaluator.verify(
        claim=city_claim,
        node=city_tx_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the facility's address or location is in Texas. Accept 'TX', 'Texas', or address lines "
            "showing the city with 'TX'. If an address for the facility is shown in Texas, pass."
        ),
    )

    # 3) Service & policy requirements (all critical leaves under the facility node)
    # Prepare all leaves & claims
    leaves_and_jobs: List[tuple[str, List[str], Any, str]] = []

    # CGC Program
    cgc_leaf = evaluator.add_leaf(
        id="CGC_Program",
        desc="Facility offers AKC Canine Good Citizen (CGC) certification program.",
        parent=facility_node,
        critical=True,
    )
    cgc_claim = (
        f"The facility '{name}' offers the AKC Canine Good Citizen (CGC) program or CGC testing/certification."
    )
    cgc_instr = (
        "Look for 'AKC Canine Good Citizen', 'CGC', or 'CGC testing/certification'. Synonyms include 'AKC CGC', "
        "'Canine Good Citizen'."
    )
    leaves_and_jobs.append((cgc_claim, sources, cgc_leaf, cgc_instr))

    # Therapy Dog Training
    therapy_leaf = evaluator.add_leaf(
        id="Therapy_Dog_Training",
        desc="Facility offers therapy dog preparation or training program.",
        parent=facility_node,
        critical=True,
    )
    therapy_claim = f"The facility '{name}' offers therapy dog preparation or training."
    therapy_instr = (
        "Look for 'therapy dog', 'animal-assisted therapy', 'therapy team training', or preparation for therapy dog evaluations."
    )
    leaves_and_jobs.append((therapy_claim, sources, therapy_leaf, therapy_instr))

    # CCPDT Certified Trainer
    ccpdt_leaf = evaluator.add_leaf(
        id="CCPDT_Certified_Trainer",
        desc="Facility has at least one CCPDT certified trainer on staff.",
        parent=facility_node,
        critical=True,
    )
    ccpdt_claim = (
        f"The facility '{name}' has at least one trainer holding a CCPDT certification (e.g., CPDT-KA, CPDT-KSA, CBCC-KA)."
    )
    ccpdt_instr = (
        "Look for 'CCPDT', 'CPDT-KA', 'CPDT-KSA', 'CBCC-KA', or explicit mention of Certification Council for Professional Dog Trainers."
    )
    leaves_and_jobs.append((ccpdt_claim, sources, ccpdt_leaf, ccpdt_instr))

    # Puppy Training Under 6 Months
    puppy_leaf = evaluator.add_leaf(
        id="Puppy_Training_Under_6_Months",
        desc="Facility offers puppy training classes for dogs under 6 months of age.",
        parent=facility_node,
        critical=True,
    )
    puppy_claim = f"The facility '{name}' offers puppy training classes suitable for dogs under 6 months of age."
    puppy_instr = (
        "Look for 'puppy class', 'puppy kindergarten', age ranges like '8 weeks to 6 months', or explicit mention of under 6 months."
    )
    leaves_and_jobs.append((puppy_claim, sources, puppy_leaf, puppy_instr))

    # Advanced Obedience Beyond Basic
    advanced_leaf = evaluator.add_leaf(
        id="Advanced_Obedience_Beyond_Basic",
        desc="Facility offers advanced obedience training classes beyond basic commands.",
        parent=facility_node,
        critical=True,
    )
    advanced_claim = f"The facility '{name}' offers advanced obedience classes beyond basic commands."
    advanced_instr = (
        "Look for 'Advanced Obedience', 'Level 2/3', 'Intermediate/Advanced', 'off-leash training', "
        "or language indicating beyond basic commands."
    )
    leaves_and_jobs.append((advanced_claim, sources, advanced_leaf, advanced_instr))

    # Private One-on-One Training
    private_leaf = evaluator.add_leaf(
        id="Private_One_on_One_Training",
        desc="Facility offers private one-on-one training sessions.",
        parent=facility_node,
        critical=True,
    )
    private_claim = f"The facility '{name}' offers private one-on-one training sessions."
    private_instr = "Look for 'private lessons', 'one-on-one training', or 'individual training sessions'."
    leaves_and_jobs.append((private_claim, sources, private_leaf, private_instr))

    # Indoor Climate-Controlled Training Facility
    indoor_leaf = evaluator.add_leaf(
        id="Indoor_Climate_Controlled_Training_Facility",
        desc="Facility has an indoor training facility/building with climate control for year-round training.",
        parent=facility_node,
        critical=True,
    )
    indoor_claim = (
        f"The facility '{name}' has an indoor, climate-controlled training building suitable for year-round training."
    )
    indoor_instr = (
        "Look for 'indoor training', 'climate-controlled', 'air-conditioned', 'heated', 'indoor facility/building', "
        "or similar wording implying indoor climate control."
    )
    leaves_and_jobs.append((indoor_claim, sources, indoor_leaf, indoor_instr))

    # Overnight Boarding
    boarding_leaf = evaluator.add_leaf(
        id="Overnight_Boarding",
        desc="Facility offers overnight dog boarding services.",
        parent=facility_node,
        critical=True,
    )
    boarding_claim = f"The facility '{name}' offers overnight dog boarding services."
    boarding_instr = "Look for 'boarding', 'overnight boarding', 'lodging', or kennel services."
    leaves_and_jobs.append((boarding_claim, sources, boarding_leaf, boarding_instr))

    # Dog Daycare
    daycare_leaf = evaluator.add_leaf(
        id="Dog_Daycare",
        desc="Facility offers dog daycare services.",
        parent=facility_node,
        critical=True,
    )
    daycare_claim = f"The facility '{name}' offers dog daycare services."
    daycare_instr = "Look for 'daycare', 'dog day care', or 'day camp' services."
    leaves_and_jobs.append((daycare_claim, sources, daycare_leaf, daycare_instr))

    # Vaccination Proof: Rabies and DHPP
    vacc_leaf = evaluator.add_leaf(
        id="Vaccination_Proof_Rabies_and_DHPP",
        desc="Facility requires proof of current vaccinations including rabies and DHPP for all dogs.",
        parent=facility_node,
        critical=True,
    )
    vacc_claim = (
        f"The facility '{name}' requires proof of current vaccinations including rabies and DHPP (or equivalent such as DA2PP/DAPP/DHLPP) for all dogs."
    )
    vacc_instr = (
        "Look for vaccination requirements including 'rabies' and a distemper-parvo combination (e.g., DHPP, DA2PP, DAPP, DHLPP). "
        "Explicit requirement for both is needed."
    )
    leaves_and_jobs.append((vacc_claim, sources, vacc_leaf, vacc_instr))

    # Operates at least 6 days per week
    days_leaf = evaluator.add_leaf(
        id="Operates_At_Least_6_Days_Per_Week",
        desc="Facility operates at least 6 days per week.",
        parent=facility_node,
        critical=True,
    )
    days_claim = f"The facility '{name}' operates at least 6 days per week."
    days_instr = (
        "Check posted hours/schedule; pass if open for at least 6 different days within a typical week (e.g., closed only 1 day). "
        "Consider weekly hours page or policy/FAQ pages."
    )
    leaves_and_jobs.append((days_claim, sources, days_leaf, days_instr))

    # Accepts all breeds (no BSL restrictions)
    breeds_leaf = evaluator.add_leaf(
        id="Accepts_All_Breeds_No_BSL",
        desc="Facility accepts dogs of all breeds without breed-specific restrictions.",
        parent=facility_node,
        critical=True,
    )
    breeds_claim = f"The facility '{name}' accepts dogs of all breeds without breed-specific restrictions."
    breeds_instr = (
        "Look for 'all breeds welcome', 'no breed restrictions', 'we accept all breeds', or explicit statements indicating "
        "no breed-specific bans. If the page lists prohibited breeds, fail."
    )
    leaves_and_jobs.append((breeds_claim, sources, breeds_leaf, breeds_instr))

    # Execute all verifications in parallel
    await evaluator.batch_verify(leaves_and_jobs)


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
    Evaluate an answer for the Texas dog training and care facility task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation at root
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

    # Extraction: facility basic info and URLs
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_facility_info(),
        template_class=FacilityExtraction,
        extraction_name="facility_basic_info",
    )

    # Optional: record requirements in summary for clarity
    evaluator.add_custom_info(
        info={
            "requirements": [
                "AKC CGC program",
                "Therapy dog training",
                "CCPDT certified trainer",
                "Puppy training under 6 months",
                "Advanced obedience beyond basic",
                "Private one-on-one training",
                "Indoor climate-controlled training facility",
                "Overnight boarding",
                "Dog daycare",
                "Vaccination proof: rabies and DHPP",
                "Operates at least 6 days per week",
                "Accepts all breeds (no BSL restrictions)",
            ]
        },
        info_type="requirements",
        info_name="facility_requirements",
    )

    # Build verification tree and run checks
    await verify_texas_facility(evaluator, root, extracted_info)

    return evaluator.get_summary()