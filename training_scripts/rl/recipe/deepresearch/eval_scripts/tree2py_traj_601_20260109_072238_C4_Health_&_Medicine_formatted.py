import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "fl_hospital_multi_cert"
TASK_DESCRIPTION = (
    "Identify a hospital in Florida that simultaneously holds all of the following certifications and designations: "
    "(1) Level I Adult Trauma Center verification from the American College of Surgeons (ACS), "
    "(2) Comprehensive Stroke Center certification from Joint Commission, DNV, or American Heart Association, "
    "(3) National Cancer Institute (NCI) designation as a Cancer Center or Comprehensive Cancer Center, "
    "(4) American Burn Association (ABA) Burn Center verification, and "
    "(5) American College of Cardiology (ACC) Chest Pain Center accreditation. "
    "Provide the hospital's full name and location (city)."
)


class HospitalCertExtraction(BaseModel):
    hospital_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    trauma_urls: List[str] = Field(default_factory=list)  # ACS Level I Adult Trauma verification sources
    stroke_urls: List[str] = Field(default_factory=list)  # Joint Commission, DNV, or AHA Comprehensive Stroke Center sources
    nci_urls: List[str] = Field(default_factory=list)     # NCI designation sources
    burn_urls: List[str] = Field(default_factory=list)    # ABA Verified Burn Center sources
    acc_urls: List[str] = Field(default_factory=list)     # ACC Chest Pain Center accreditation sources


def prompt_extract_hospital_certs() -> str:
    return """
    Extract the single hospital identified in the answer that is claimed to meet ALL required certifications/designations and provide the requested location information.

    Return the following fields:
    - hospital_name: The full official name of the hospital (not the system name unless the answer explicitly uses the system as the hospital).
    - city: The city where the hospital is located.
    - state: The state where the hospital is located (expect 'Florida' or 'FL').
    - trauma_urls: All URLs cited in the answer that specifically support ACS verification as a Level I Adult Trauma Center for this hospital.
    - stroke_urls: All URLs cited in the answer that specifically support that the hospital is a Comprehensive Stroke Center certified by one of: The Joint Commission (TJC), DNV Healthcare (DNV), or the American Heart Association (AHA).
    - nci_urls: All URLs cited in the answer that specifically support that the hospital (or the cancer center housed at/affiliated with the hospital) is designated by the National Cancer Institute (NCI) as a Cancer Center or Comprehensive Cancer Center.
    - burn_urls: All URLs cited in the answer that specifically support ABA Burn Center verification for this hospital.
    - acc_urls: All URLs cited in the answer that specifically support ACC Chest Pain Center accreditation for this hospital.

    SPECIAL RULES:
    - Extract only URLs explicitly present in the answer (including markdown links); do not invent or infer URLs.
    - If multiple hospitals are mentioned, pick the one the answer claims meets ALL the requirements; if unclear, choose the first hospital mentioned.
    - If a field is missing, set it to null; if a URL list for a category is missing, return an empty array for that list.
    - Do not mix sources across hospitals; only include URLs that the answer associates with the chosen hospital.
    """


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        nu = u.strip()
        if nu and nu not in seen:
            seen.add(nu)
            result.append(nu)
    return result


def _combine_all_urls(ex: HospitalCertExtraction) -> List[str]:
    return _dedup_urls(
        (ex.trauma_urls or [])
        + (ex.stroke_urls or [])
        + (ex.nci_urls or [])
        + (ex.burn_urls or [])
        + (ex.acc_urls or [])
    )


async def _verify_hospital(evaluator: Evaluator, root_node, ex: HospitalCertExtraction) -> None:
    hospital_node = evaluator.add_parallel(
        id="Hospital",
        desc="A single hospital in Florida is identified that meets all required certifications and provides all requested information",
        parent=root_node,
        critical=True
    )

    name_ok = bool(ex.hospital_name and ex.hospital_name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id="Hospital_Name_Provided",
        desc="The answer provides the full name of the hospital",
        parent=hospital_node,
        critical=True
    )

    city_ok = bool(ex.city and ex.city.strip())
    evaluator.add_custom_node(
        result=city_ok,
        id="City_Location_Provided",
        desc="The answer provides the city where the hospital is located",
        parent=hospital_node,
        critical=True
    )

    # Florida location verification as a leaf (using any provided sources across categories)
    florida_leaf = evaluator.add_leaf(
        id="Florida_Location",
        desc="The hospital is located in the state of Florida",
        parent=hospital_node,
        critical=True
    )
    fl_claim = f"The hospital '{ex.hospital_name or 'the hospital'}' is located in the state of Florida."
    all_urls = _combine_all_urls(ex)
    await evaluator.verify(
        claim=fl_claim,
        node=florida_leaf,
        sources=all_urls if all_urls else None,
        additional_instruction="Confirm evidence that the hospital is in Florida. Accept explicit state 'Florida' or abbreviation 'FL' on official pages (hospital, accrediting body) or widely accepted references. If the pages indicate a location outside Florida, the claim is not supported."
    )

    # Certification verifications
    # 1) ACS Level I Adult Trauma Center
    trauma_leaf = evaluator.add_leaf(
        id="Level_I_Trauma_Center",
        desc="The hospital is verified as a Level I Adult Trauma Center by the American College of Surgeons (ACS)",
        parent=hospital_node,
        critical=True
    )
    trauma_claim = f"The hospital '{ex.hospital_name or 'the hospital'}' is verified by the American College of Surgeons (ACS) as a Level I Adult Trauma Center."
    await evaluator.verify(
        claim=trauma_claim,
        node=trauma_leaf,
        sources=ex.trauma_urls if ex.trauma_urls else None,
        additional_instruction=(
            "Look for ACS verification specifically stating 'Level I' and 'Adult'. "
            "State designation alone is insufficient if ACS verification is not present. "
            "If the evidence shows Pediatric-only, Level II/III, or no ACS verification, the claim fails."
        )
    )

    # 2) Comprehensive Stroke Center (Joint Commission, DNV, or AHA)
    stroke_leaf = evaluator.add_leaf(
        id="Comprehensive_Stroke_Center",
        desc="The hospital holds Comprehensive Stroke Center certification from Joint Commission, DNV, or American Heart Association",
        parent=hospital_node,
        critical=True
    )
    stroke_claim = (
        f"The hospital '{ex.hospital_name or 'the hospital'}' holds Comprehensive Stroke Center certification "
        f"from either The Joint Commission (TJC), DNV Healthcare (DNV), or the American Heart Association (AHA)."
    )
    await evaluator.verify(
        claim=stroke_claim,
        node=stroke_leaf,
        sources=ex.stroke_urls if ex.stroke_urls else None,
        additional_instruction=(
            "Confirm Comprehensive Stroke Center (CSC) certification from one of: TJC, DNV, or AHA. "
            "Equivalent wording 'Comprehensive Stroke Center' or 'CSC' is acceptable. "
            "Lower tiers (e.g., Primary Stroke Center, Thrombectomy-Capable) are insufficient."
        )
    )

    # 3) NCI designation
    nci_leaf = evaluator.add_leaf(
        id="NCI_Designated_Cancer_Center",
        desc="The hospital is designated by the National Cancer Institute (NCI) as a Cancer Center or Comprehensive Cancer Center",
        parent=hospital_node,
        critical=True
    )
    nci_claim = (
        f"The hospital '{ex.hospital_name or 'the hospital'}' (or the cancer center housed at/affiliated with it) "
        f"is designated by the National Cancer Institute (NCI) as a Cancer Center or Comprehensive Cancer Center."
    )
    await evaluator.verify(
        claim=nci_claim,
        node=nci_leaf,
        sources=ex.nci_urls if ex.nci_urls else None,
        additional_instruction=(
            "Prefer NCI's official site listing designated centers. "
            "Accept if the hospital is the host/home campus of a named NCI-designated Cancer Center. "
            "Membership in networks or participation in trials without formal NCI 'Designated Cancer Center' status is insufficient."
        )
    )

    # 4) ABA Verified Burn Center
    burn_leaf = evaluator.add_leaf(
        id="ABA_Verified_Burn_Center",
        desc="The hospital is verified as a Burn Center by the American Burn Association (ABA)",
        parent=hospital_node,
        critical=True
    )
    burn_claim = f"The hospital '{ex.hospital_name or 'the hospital'}' is verified by the American Burn Association (ABA) as a Burn Center."
    await evaluator.verify(
        claim=burn_claim,
        node=burn_leaf,
        sources=ex.burn_urls if ex.burn_urls else None,
        additional_instruction=(
            "Look for 'Verified Burn Center' status by the ABA (often jointly with ACS). "
            "If the page indicates no verification or different status, the claim fails."
        )
    )

    # 5) ACC Chest Pain Center accreditation
    acc_leaf = evaluator.add_leaf(
        id="ACC_Chest_Pain_Accreditation",
        desc="The hospital holds Chest Pain Center accreditation from the American College of Cardiology (ACC)",
        parent=hospital_node,
        critical=True
    )
    acc_claim = f"The hospital '{ex.hospital_name or 'the hospital'}' holds Chest Pain Center accreditation from the American College of Cardiology (ACC)."
    await evaluator.verify(
        claim=acc_claim,
        node=acc_leaf,
        sources=ex.acc_urls if ex.acc_urls else None,
        additional_instruction=(
            "Confirm 'Chest Pain Center Accreditation' from ACC on ACC's site or credible hospital/press pages explicitly referencing ACC. "
            "If only similar-sounding programs without ACC accreditation, the claim fails."
        )
    )


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
        default_model=model
    )

    extracted = await evaluator.extract(
        prompt=prompt_extract_hospital_certs(),
        template_class=HospitalCertExtraction,
        extraction_name="hospital_cert_extraction"
    )

    # Optional: record some aggregate info for debugging
    evaluator.add_custom_info(
        info={
            "hospital_name": extracted.hospital_name,
            "city": extracted.city,
            "state": extracted.state,
            "counts": {
                "trauma_urls": len(extracted.trauma_urls),
                "stroke_urls": len(extracted.stroke_urls),
                "nci_urls": len(extracted.nci_urls),
                "burn_urls": len(extracted.burn_urls),
                "acc_urls": len(extracted.acc_urls),
                "all_urls_combined": len(_combine_all_urls(extracted)),
            }
        },
        info_type="extraction_summary",
        info_name="extraction_summary"
    )

    await _verify_hospital(evaluator, root, extracted)

    return evaluator.get_summary()