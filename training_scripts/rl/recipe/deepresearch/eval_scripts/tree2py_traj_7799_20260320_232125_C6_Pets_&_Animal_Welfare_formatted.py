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
TASK_ID = "akc_hawaii_transport_2025_bis"
TASK_DESCRIPTION = """
You are planning to transport the Best in Show winner from the 2025 AKC National Championship dog show from the mainland United States to Hawaii. Please provide the following information:
1. Dog Identification: What is the name and breed of the 2025 AKC National Championship Best in Show winner? Provide a reference URL confirming this information from an official source.
2. Breed Classification: What AKC group does this breed belong to (Sporting, Hound, Working, Terrier, Toy, Non-Sporting, or Herding)? Also provide the AKC breed standard height range for this breed. Include a reference URL confirming the group classification.
3. Hawaii Import Eligibility: Based on the dog's date of birth, determine the dog's current age (as of March 20, 2026). Is this dog old enough to have completed all necessary preparations for Hawaii import under the Direct Airport Release program? Consider that dogs need approximately 6 months from birth to complete the two rabies vaccinations, FAVN test, and required waiting periods. Provide your eligibility conclusion and reasoning.
4. Vaccination Requirements: What are the specific rabies vaccination and testing requirements for importing a dog to Hawaii under the Direct Airport Release program? Include: (a) the number of rabies vaccinations required, (b) whether an OIE-FAVN test is required, (c) the waiting periods required after vaccination and testing, and (d) a reference URL from the Hawaii Department of Agriculture confirming these requirements.
5. Documentation Timeline: Within how many days before arrival must the health certificate be issued for Hawaii import? By what deadline must complete pre-arrival documentation be submitted to qualify for the lower Direct Airport Release fee of $185 instead of $244? What documents must be submitted? Provide a reference URL from the Hawaii Department of Agriculture confirming these timing requirements.
6. GFAS Sanctuaries: Identify two GFAS-accredited animal sanctuaries located in Hawaii. For each sanctuary, provide: (a) the sanctuary name, (b) the location (city/area), (c) the type of animals housed there, and (d) a reference URL confirming the sanctuary's GFAS accreditation status.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WinnerExtraction(BaseModel):
    name: Optional[str] = None
    breed: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class BreedClassificationExtraction(BaseModel):
    breed: Optional[str] = None
    akc_group: Optional[str] = None
    height_range: Optional[str] = None
    group_source_urls: List[str] = Field(default_factory=list)


class EligibilityExtraction(BaseModel):
    dob: Optional[str] = None
    dob_source_urls: List[str] = Field(default_factory=list)
    age_as_of_2026_03_20: Optional[str] = None
    eligibility_conclusion: Optional[str] = None  # e.g., "Eligible" / "Not eligible" with brief reasoning in text


class VaccinationRequirementsExtraction(BaseModel):
    num_rabies_vaccinations: Optional[str] = None  # Expect "2" or "two"
    favn_required_text: Optional[str] = None       # e.g., "Yes" or description indicating required
    wait_after_favn_days: Optional[str] = None     # Expect "30 days" or "at least 30 days"
    wait_after_last_rabies_days: Optional[str] = None  # Expect "30 days" or "at least 30 days"
    hdoa_urls: List[str] = Field(default_factory=list)


class DocumentationTimelineExtraction(BaseModel):
    health_certificate_window_days: Optional[str] = None  # Expect "14 days" or "within 14 days"
    health_certificate_issuer: Optional[str] = None       # Expect "licensed veterinarian"
    lower_fee_deadline_days: Optional[str] = None         # Expect "10 days"
    required_documents: List[str] = Field(default_factory=list)  # Expect list including 2 rabies certs, FAVN result, AQS-279
    hdoa_timing_urls: List[str] = Field(default_factory=list)


class SanctuaryInfo(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None  # city/area; must be in Hawaii
    animal_types: Optional[str] = None
    gfas_url: Optional[str] = None


class GFASanctuariesExtraction(BaseModel):
    sanctuaries: List[SanctuaryInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_winner() -> str:
    return """
    From the answer, extract the Best in Show winner details specifically for the 2025 AKC National Championship (not Westminster, not the National Dog Show).
    Return:
    - name: the winner dog's name as stated in the answer.
    - breed: the winner dog's breed as stated in the answer.
    - source_urls: a list of one or more official/reference URLs cited in the answer that confirm the winner's name and breed for the 2025 AKC National Championship.
    If any field is missing in the answer, return null for that field (or an empty list for URLs).
    """


def prompt_extract_breed_classification() -> str:
    return """
    From the answer, extract the AKC breed classification details for the winner's breed:
    - breed: the breed name used in the answer for the winner.
    - akc_group: one of {Sporting, Hound, Working, Terrier, Toy, Non-Sporting, Herding} as stated in the answer.
    - height_range: the AKC breed standard height range as stated in the answer (include units, e.g., inches; include sex qualifiers if the answer presents them as a combined range string).
    - group_source_urls: one or more URLs cited in the answer that confirm the AKC group classification (ideally the official AKC page).
    If any field is missing in the answer, set it to null (or [] for URLs).
    """


def prompt_extract_eligibility() -> str:
    return """
    From the answer, extract the dog’s documented date of birth and the import eligibility reasoning:
    - dob: the dog's date of birth as stated in the answer (verbatim).
    - dob_source_urls: URLs cited in the answer that document the dog's date of birth.
    - age_as_of_2026_03_20: the age of the dog as of March 20, 2026, as explicitly calculated or stated in the answer (e.g., '1 year 8 months', '20 months', etc.).
    - eligibility_conclusion: the answer’s conclusion about whether the dog is old enough to have completed the Direct Airport Release preparations (include the yes/no sense and reasoning as a brief sentence from the answer).
    If something is not provided in the answer, return null (or [] for URLs).
    """


def prompt_extract_vaccination_requirements() -> str:
    return """
    From the answer, extract the Hawaii Direct Airport Release (DAR) rabies vaccination and testing requirements:
    - num_rabies_vaccinations: the number of rabies vaccinations required (as written in the answer, e.g., '2' or 'two').
    - favn_required_text: whether an OIE-FAVN rabies antibody test is required (use the phrasing in the answer, e.g., 'Yes, OIE-FAVN required').
    - wait_after_favn_days: the minimum waiting period after a successful FAVN test before arrival (as stated in the answer, e.g., '≥ 30 days', 'at least 30 days', or '30 days').
    - wait_after_last_rabies_days: the minimum waiting period after the most recent rabies vaccine before arrival (as stated, e.g., '≥ 30 days' or '30 days').
    - hdoa_urls: one or more Hawaii Department of Agriculture URLs cited in the answer that confirm these requirements.
    If a field is missing, set it to null (or [] for URLs).
    """


def prompt_extract_documentation_timeline() -> str:
    return """
    From the answer, extract the Direct Airport Release documentation timing and fee details for Hawaii import:
    - health_certificate_window_days: the stated window for when the health certificate must be issued before arrival (e.g., 'within 14 days').
    - health_certificate_issuer: who must issue the health certificate (e.g., 'licensed veterinarian').
    - lower_fee_deadline_days: the submission deadline to qualify for the lower $185 Direct Airport Release fee (e.g., 'at least 10 days before arrival').
    - required_documents: a list of the required pre-arrival documents (e.g., 'two rabies vaccination certificates', 'FAVN test result', 'completed AQS-279 form').
    - hdoa_timing_urls: one or more Hawaii Department of Agriculture URLs cited in the answer that confirm the timing and fee rules.
    If missing, set to null (or [] for URLs).
    """


def prompt_extract_gfas_sanctuaries() -> str:
    return """
    From the answer, extract two GFAS-accredited animal sanctuaries located in Hawaii. For each sanctuary, return:
    - name: sanctuary name as in the answer
    - location: the city/area in Hawaii as in the answer
    - animal_types: the type of animals housed there as in the answer
    - gfas_url: a URL cited in the answer that confirms GFAS accreditation status (ideally on gfas.org or gfasanctuaries.org)
    Return an array 'sanctuaries' of up to two such objects in the order they appear in the answer. If fewer than two are present, include the ones available; leave missing fields as null.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _first_n(lst: List[Any], n: int) -> List[Any]:
    return lst[:n] if lst else []


def _urls_or_none(urls: List[str]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and u.strip()] if urls else []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_dog_show_winner(
    evaluator: Evaluator,
    parent,
    winner: WinnerExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Dog_Show_Winner_Identification",
        desc="Correctly identify the 2025 AKC National Championship Best in Show winner (not other shows).",
        parent=parent,
        critical=True,
    )

    # Winner_Source_URL (existence as a binary custom leaf)
    srcs = _urls_or_none(winner.source_urls)
    evaluator.add_custom_node(
        result=bool(srcs),
        id="Winner_Source_URL",
        desc="Provide an official-source URL confirming the winner name and breed.",
        parent=node,
        critical=True,
    )

    # Correct_Show
    leaf_correct_show = evaluator.add_leaf(
        id="Correct_Show",
        desc="Winner identification is explicitly for the 2025 AKC National Championship (not Westminster or National Dog Show).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This page shows the Best in Show winner for the 2025 AKC National Championship (American Kennel Club), not the Westminster Kennel Club Dog Show and not the National Dog Show.",
        node=leaf_correct_show,
        sources=srcs,
        additional_instruction="Confirm the event is 'AKC National Championship' for year 2025 and that it's the Best in Show result.",
    )

    # Winner_Name
    leaf_name = evaluator.add_leaf(
        id="Winner_Name",
        desc="Provide the name of the Best in Show winner.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The 2025 AKC National Championship Best in Show winner is named '{winner.name}'.",
        node=leaf_name,
        sources=srcs,
        additional_instruction="Allow minor formatting or registered name variations if they clearly refer to the same dog.",
    )

    # Winner_Breed
    leaf_breed = evaluator.add_leaf(
        id="Winner_Breed",
        desc="Provide the breed of the Best in Show winner.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The winner's breed is '{winner.breed}'.",
        node=leaf_breed,
        sources=srcs,
        additional_instruction="Verify the page explicitly states the breed of the Best in Show winner.",
    )


async def verify_breed_classification(
    evaluator: Evaluator,
    parent,
    breed_info: BreedClassificationExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Breed_Classification",
        desc="Provide AKC group classification and AKC breed standard height range for the winner's breed, with citation for group classification.",
        parent=parent,
        critical=True,
    )

    srcs = _urls_or_none(breed_info.group_source_urls)

    # Group_Classification_URL (existence check as custom)
    evaluator.add_custom_node(
        result=bool(srcs),
        id="Group_Classification_URL",
        desc="Provide a reference URL confirming the breed's AKC group classification.",
        parent=node,
        critical=True,
    )

    # AKC_Group
    leaf_group = evaluator.add_leaf(
        id="AKC_Group",
        desc="State which of the seven AKC groups the breed belongs to (Sporting, Hound, Working, Terrier, Toy, Non-Sporting, Herding).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The breed '{breed_info.breed}' belongs to the AKC '{breed_info.akc_group}' Group.",
        node=leaf_group,
        sources=srcs,
        additional_instruction="Confirm this on an AKC page or equivalent authoritative AKC source.",
    )

    # Breed_Height_Standard (require presence + support)
    # Existence check to ensure height was provided
    evaluator.add_custom_node(
        result=bool(breed_info.height_range and breed_info.height_range.strip()),
        id="Breed_Height_Standard_Provided",
        desc="AKC breed standard height range is provided in the answer.",
        parent=node,
        critical=True,
    )
    leaf_height = evaluator.add_leaf(
        id="Breed_Height_Standard",
        desc="Provide the AKC breed standard height range for the breed.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The AKC breed standard height range for '{breed_info.breed}' is '{breed_info.height_range}'.",
        node=leaf_height,
        sources=srcs,
        additional_instruction="Allow minor formatting differences; if sex-specific values are given, accept the combined range as equivalent.",
    )


async def verify_hawaii_import_eligibility(
    evaluator: Evaluator,
    parent,
    elig: EligibilityExtraction,
) -> None:
    node = evaluator.add_sequential(
        id="Hawaii_Import_Eligibility",
        desc="Compute age as of 2026-03-20 and conclude if old enough for DAR (~6 months needed).",
        parent=parent,
        critical=True,
    )

    dob_srcs = _urls_or_none(elig.dob_source_urls)

    # Step 1: Documented DOB (verify with URLs)
    leaf_dob = evaluator.add_leaf(
        id="Documented_Date_of_Birth",
        desc="Provide the dog's documented date of birth (or equivalent documented DOB evidence) used for the age calculation.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The dog's date of birth is '{elig.dob}'.",
        node=leaf_dob,
        sources=dob_srcs,
        additional_instruction="Confirm the DOB (or equivalent official documented birthdate) on the provided source page(s).",
    )

    # Step 2: Age as of 2026-03-20 (simple logical verification)
    leaf_age = evaluator.add_leaf(
        id="Current_Age_As_Of_2026_03_20",
        desc="Calculate the dog's age as of March 20, 2026 based on the documented date of birth.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Given the date of birth '{elig.dob}', the dog's age as of March 20, 2026 is correctly stated as '{elig.age_as_of_2026_03_20}' in the answer.",
        node=leaf_age,
        additional_instruction="Independently compute the age from the DOB and check whether the provided age statement is correct. Allow reasonable rounding (e.g., months).",
    )

    # Step 3: Eligibility conclusion with reasoning (~6 months needed)
    leaf_elig = evaluator.add_leaf(
        id="Age_Eligibility_Conclusion_With_Reasoning",
        desc="Conclude whether the dog is old enough given ~6 months are needed to complete two rabies vaccinations, FAVN test, and waiting periods; include reasoning.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Given approximately 6 months are needed from birth to complete the two rabies vaccinations, FAVN test, and required waiting periods for Direct Airport Release, a dog with DOB '{elig.dob}' is correctly concluded in the answer as: {elig.eligibility_conclusion}.",
        node=leaf_elig,
        additional_instruction="Judge whether the conclusion is logically consistent with the computed age as of 2026-03-20 (>= 6 months => eligible to have completed preparations).",
    )


async def verify_vaccination_requirements(
    evaluator: Evaluator,
    parent,
    vac: VaccinationRequirementsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Vaccination_Requirements",
        desc="State Hawaii Direct Airport Release rabies vaccination/testing requirements and waiting periods, with Hawaii DOA citation.",
        parent=parent,
        critical=True,
    )

    srcs = _urls_or_none(vac.hdoa_urls)

    # HDOA URL evidence leaf
    leaf_url = evaluator.add_leaf(
        id="HDOA_Vaccination_Requirements_URL",
        desc="Provide a Hawaii Department of Agriculture URL confirming the vaccination/testing requirements and waiting periods.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This page from the Hawaii Department of Agriculture (Animal Quarantine Branch) describes the rabies vaccination and OIE-FAVN testing requirements for Direct Airport Release.",
        node=leaf_url,
        sources=srcs,
        additional_instruction="The page should be an HDOA source (hawaii.gov domain or official HDOA site) that outlines DAR requirements.",
    )

    # Two rabies vaccinations
    leaf_two = evaluator.add_leaf(
        id="Two_Rabies_Vaccinations",
        desc="State that two rabies vaccinations (with certificates) are required.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Under Hawaii's Direct Airport Release program, two rabies vaccinations (with certificates) are required.",
        node=leaf_two,
        sources=srcs,
        additional_instruction="Verify the number is two (2).",
    )

    # FAVN required
    leaf_favn = evaluator.add_leaf(
        id="FAVN_Test_Required",
        desc="State that a passing OIE-FAVN rabies antibody test is required.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="A passing OIE-FAVN rabies antibody test is required for Hawaii Direct Airport Release.",
        node=leaf_favn,
        sources=srcs,
        additional_instruction="Confirm that a passing FAVN titer is required prior to arrival.",
    )

    # Waiting period after FAVN
    leaf_wait_favn = evaluator.add_leaf(
        id="Waiting_Period_After_FAVN",
        desc="State the minimum 30-day waiting period after a successful FAVN test before arrival in Hawaii.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="There is a minimum 30-day waiting period after a successful OIE-FAVN test result before the animal can arrive in Hawaii under DAR.",
        node=leaf_wait_favn,
        sources=srcs,
        additional_instruction="Confirm at least 30 days after the FAVN date (or result date) is required before arrival.",
    )

    # Waiting period after last rabies vaccine
    leaf_wait_vax = evaluator.add_leaf(
        id="Waiting_Period_After_Last_Rabies_Vax",
        desc="State the minimum 30-day waiting period after the most recent rabies vaccination before arrival in Hawaii.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="There is a minimum 30-day waiting period after the most recent rabies vaccination before arrival in Hawaii under DAR.",
        node=leaf_wait_vax,
        sources=srcs,
        additional_instruction="Confirm the minimum 30 days after the latest rabies shot prior to entry.",
    )


async def verify_documentation_timeline(
    evaluator: Evaluator,
    parent,
    doc: DocumentationTimelineExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Documentation_Timeline",
        desc="Provide required timing windows, fee deadline, and required pre-arrival documents for Direct Airport Release, with Hawaii DOA citation.",
        parent=parent,
        critical=True,
    )

    srcs = _urls_or_none(doc.hdoa_timing_urls)

    # HDOA timing URL evidence
    leaf_url = evaluator.add_leaf(
        id="HDOA_Documentation_Timing_URL",
        desc="Provide a Hawaii Department of Agriculture URL confirming the timing requirements (health certificate window and documentation submission deadline/fee rule).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This Hawaii Department of Agriculture page states the health certificate issuance window and the pre-arrival documentation submission deadline/fee rule for Direct Airport Release.",
        node=leaf_url,
        sources=srcs,
        additional_instruction="The page should clearly discuss timing and fees for DAR.",
    )

    # Health certificate window
    leaf_hc_window = evaluator.add_leaf(
        id="Health_Certificate_Window",
        desc="State that the health certificate must be issued within 14 days prior to arrival in Hawaii.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The health certificate must be issued within 14 days prior to arrival in Hawaii.",
        node=leaf_hc_window,
        sources=srcs,
        additional_instruction="Confirm 'within 14 days' (or equivalent exact phrasing) on the HDOA page.",
    )

    # Health certificate issuer
    leaf_hc_issuer = evaluator.add_leaf(
        id="Health_Certificate_Issuer",
        desc="State that the health certificate must be issued by a licensed veterinarian.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The health certificate must be issued by a licensed veterinarian.",
        node=leaf_hc_issuer,
        sources=srcs,
        additional_instruction="Confirm the issuer requirement on the HDOA page.",
    )

    # Lower fee deadline
    leaf_fee_deadline = evaluator.add_leaf(
        id="Lower_Fee_Submission_Deadline",
        desc="State that complete pre-arrival documentation must be submitted at least 10 days before arrival to qualify for the $185 fee instead of $244.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Complete pre-arrival documentation must be received at least 10 days before arrival to qualify for the $185 Direct Airport Release fee; otherwise, the fee is $244.",
        node=leaf_fee_deadline,
        sources=srcs,
        additional_instruction="Verify both the 10-day deadline and the $185 vs $244 fee amounts.",
    )

    # Required documents
    leaf_docs = evaluator.add_leaf(
        id="Required_Pre_Arrival_Documents",
        desc="List the required documents to be submitted (two rabies vaccination certificates, FAVN test results, completed AQS-279 form).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Required pre-arrival documents include: two rabies vaccination certificates, a passing FAVN test result, and a completed AQS-279 Dog & Cat Import Form (with microchip identification).",
        node=leaf_docs,
        sources=srcs,
        additional_instruction="Minor wording variations are acceptable; the key items must appear.",
    )


async def _verify_one_sanctuary(
    evaluator: Evaluator,
    parent,
    idx: int,
    sanctuary: SanctuaryInfo,
) -> None:
    node = evaluator.add_parallel(
        id=f"Sanctuary_{idx+1}",
        desc=f"{'First' if idx==0 else 'Second'} GFAS-accredited sanctuary in Hawaii with required details.",
        parent=parent,
        critical=True,  # Make children of a critical parent also critical to satisfy framework constraint
    )

    src = sanctuary.gfas_url or ""
    srcs = [src] if src else []

    # Sanctuary URL existence check to gate subsequent URL-based verifications
    evaluator.add_custom_node(
        result=bool(srcs),
        id=f"Sanctuary_{idx+1}_GFASEvidence_URL_Provided",
        desc="GFAS accreditation evidence URL is provided.",
        parent=node,
        critical=True,
    )

    # Name
    leaf_name = evaluator.add_leaf(
        id=f"Sanctuary_{idx+1}_Name",
        desc="Provide the sanctuary name.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The GFAS page lists the sanctuary named '{sanctuary.name}'.",
        node=leaf_name,
        sources=srcs,
        additional_instruction="Verify the sanctuary name on the GFAS page (or accreditation listing).",
    )

    # Location in Hawaii
    leaf_loc = evaluator.add_leaf(
        id=f"Sanctuary_{idx+1}_Location_In_Hawaii",
        desc="Provide the location (city/area) and it must be in Hawaii.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The sanctuary is located in Hawaii (location noted as '{sanctuary.location}').",
        node=leaf_loc,
        sources=srcs,
        additional_instruction="Confirm the location is in the U.S. state of Hawaii.",
    )

    # Animal types
    leaf_animals = evaluator.add_leaf(
        id=f"Sanctuary_{idx+1}_Animal_Types",
        desc="Describe the type of animals housed there.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The sanctuary houses: {sanctuary.animal_types}.",
        node=leaf_animals,
        sources=srcs,
        additional_instruction="Confirm on the GFAS page (or linked official page) the general types of animals at the sanctuary.",
    )

    # GFAS accreditation evidence
    leaf_gfas = evaluator.add_leaf(
        id=f"Sanctuary_{idx+1}_GFASEvidence_URL",
        desc="Provide a URL confirming GFAS accreditation status.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This page confirms that the sanctuary is GFAS-accredited (listed or recognized by GFAS).",
        node=leaf_gfas,
        sources=srcs,
        additional_instruction="Prefer an official GFAS listing or accreditation page for the sanctuary.",
    )


async def verify_gfas_sanctuaries(
    evaluator: Evaluator,
    parent,
    gfas: GFASanctuariesExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="GFAS_Sanctuary_Identification",
        desc="Identify two GFAS-accredited sanctuaries located in Hawaii; for each provide name, Hawaii location, animal types, and GFAS accreditation URL.",
        parent=parent,
        critical=True,
    )

    sancts = _first_n(gfas.sanctuaries, 2)
    # Pad with empty entries if fewer than 2 provided
    while len(sancts) < 2:
        sancts.append(SanctuaryInfo())

    for i, info in enumerate(sancts[:2]):
        await _verify_one_sanctuary(evaluator, node, i, info)


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

    # Create a top-level critical node to aggregate all required parts
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Provide all requested information for transporting the 2025 AKC National Championship Best in Show winner to Hawaii and identify two GFAS-accredited sanctuaries in Hawaii.",
        parent=root,
        critical=True,
    )

    # Run extractions (can be parallelized)
    winner_extraction_task = evaluator.extract(
        prompt=prompt_extract_winner(),
        template_class=WinnerExtraction,
        extraction_name="winner_info",
    )
    breed_extraction_task = evaluator.extract(
        prompt=prompt_extract_breed_classification(),
        template_class=BreedClassificationExtraction,
        extraction_name="breed_classification",
    )
    eligibility_extraction_task = evaluator.extract(
        prompt=prompt_extract_eligibility(),
        template_class=EligibilityExtraction,
        extraction_name="eligibility",
    )
    vacc_extraction_task = evaluator.extract(
        prompt=prompt_extract_vaccination_requirements(),
        template_class=VaccinationRequirementsExtraction,
        extraction_name="vaccination_requirements",
    )
    doc_timeline_extraction_task = evaluator.extract(
        prompt=prompt_extract_documentation_timeline(),
        template_class=DocumentationTimelineExtraction,
        extraction_name="documentation_timeline",
    )
    gfas_extraction_task = evaluator.extract(
        prompt=prompt_extract_gfas_sanctuaries(),
        template_class=GFASanctuariesExtraction,
        extraction_name="gfas_sanctuaries",
    )

    (
        winner_info,
        breed_info,
        elig_info,
        vacc_info,
        doc_info,
        gfas_info,
    ) = await asyncio.gather(
        winner_extraction_task,
        breed_extraction_task,
        eligibility_extraction_task,
        vacc_extraction_task,
        doc_timeline_extraction_task,
        gfas_extraction_task,
    )

    # Build verification tree
    await verify_dog_show_winner(evaluator, task_node, winner_info)
    await verify_breed_classification(evaluator, task_node, breed_info)
    await verify_hawaii_import_eligibility(evaluator, task_node, elig_info)
    await verify_vaccination_requirements(evaluator, task_node, vacc_info)
    await verify_documentation_timeline(evaluator, task_node, doc_info)
    await verify_gfas_sanctuaries(evaluator, task_node, gfas_info)

    return evaluator.get_summary()