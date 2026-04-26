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
TASK_ID = "nsf_gen4_erc_refrigerant_2024"
TASK_DESCRIPTION = (
    "In August 2024, the National Science Foundation announced four new Gen-4 Engineering Research Centers, "
    "each receiving substantial multi-year funding. One of these centers specifically focuses on developing "
    "sustainable refrigerant technology to address climate change and reduce global warming from cooling systems. "
    "Identify this center by providing: (1) its acronym, (2) its official full name, and (3) the lead university."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CenterInfo(BaseModel):
    """Information for the identified Gen-4 ERC focused on sustainable refrigerant technology."""
    acronym: Optional[str] = None
    full_name: Optional[str] = None
    lead_university: Optional[str] = None
    # Official reference URLs explicitly mentioned in the answer, e.g., NSF announcement or center's official site
    official_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_center_info() -> str:
    return (
        "From the provided answer, extract the information for the NSF Gen-4 Engineering Research Center "
        "that focuses on developing sustainable refrigerant technology (climate-friendly cooling) announced in August 2024.\n\n"
        "Return a JSON object with these fields:\n"
        "1) acronym: The center's acronym as stated in the answer.\n"
        "2) full_name: The center's official full name as stated in the answer.\n"
        "3) lead_university: The lead university (lead institution) for the center.\n"
        "4) official_urls: An array of all official reference URLs explicitly provided in the answer that support this center's identity, "
        "   such as links to nsf.gov announcements, the center's official website or official university pages. "
        "   Extract only actual URLs shown in the answer (including markdown links); do not invent or infer URLs.\n\n"
        "If any field is not present in the answer, set it to null (or an empty array for official_urls). "
        "If the answer mentions multiple centers, select the one explicitly tied to sustainable refrigerant technology / "
        "climate-friendly cooling systems."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def center_display_name(info: CenterInfo) -> str:
    """Prefer full name, fallback to acronym, else a generic label."""
    if info.full_name and info.full_name.strip():
        return info.full_name.strip()
    if info.acronym and info.acronym.strip():
        return info.acronym.strip()
    return "the identified center"


def build_urls_instruction(urls: List[str], purpose_hint: str) -> str:
    """
    Build an additional instruction string that clarifies URL support requirements
    and guides the verifier on how to use the provided URLs.
    """
    if urls:
        listed = "\n".join(urls)
        return (
            f"The answer provided {len(urls)} URL source(s) to support this claim:\n{listed}\n"
            f"Use these URLs (prefer official sources like nsf.gov or the center's official website) to verify the claim. "
            f"If the provided URLs do not support the claim, mark it as not supported. {purpose_hint}"
        )
    else:
        return (
            "No official reference URLs were provided in the answer for this claim. "
            "Because the rubric requires support from an official reference URL, treat this claim as not supported "
            "if there is no URL evidence. "
            f"{purpose_hint}"
        )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_center(evaluator: Evaluator, parent_node, info: CenterInfo) -> None:
    """
    Build the verification tree according to the rubric and run the checks.
    """
    # Top-level identification node (critical, sequential)
    erc_node = evaluator.add_sequential(
        id="ERC_Identification",
        desc="Identify the NSF Gen-4 Engineering Research Center (announced Aug 21, 2024) focused on sustainable refrigerant technology by providing its acronym, official full name, and lead university, with official reference URL support.",
        parent=parent_node,
        critical=True,
    )

    # =========================
    # 1) Eligibility Constraints (critical, parallel)
    # =========================
    constraints_node = evaluator.add_parallel(
        id="Center_Eligibility_Matches_Constraints",
        desc="The identified center satisfies the program/timing and topical-focus constraints from the question/constraints, supported by official references.",
        parent=erc_node,
        critical=True,
    )

    name_for_claims = center_display_name(info)
    urls = info.official_urls

    # 1.a Announced Aug 21, 2024, one of four Gen-4 ERCs
    aug21_leaf = evaluator.add_leaf(
        id="Gen4_ERC_Announced_Aug21_2024_With_Official_Ref",
        desc="The center is one of the four NSF Gen-4 Engineering Research Centers announced on Aug 21, 2024, supported by an official reference URL.",
        parent=constraints_node,
        critical=True,
    )
    aug21_claim = (
        f"The center '{name_for_claims}' is one of the four NSF Gen-4 Engineering Research Centers announced on August 21, 2024."
    )
    await evaluator.verify(
        claim=aug21_claim,
        node=aug21_leaf,
        sources=urls,
        additional_instruction=build_urls_instruction(
            urls,
            "Confirm explicitly via the provided official announcement or center page that this center belongs to the Gen-4 cohort announced on Aug 21, 2024."
        ),
    )

    # 1.b Focus: sustainable refrigerant technology / climate-friendly cooling systems
    focus_leaf = evaluator.add_leaf(
        id="Focus_Sustainable_Refrigerant_Technology_With_Official_Ref",
        desc="The center specifically focuses on sustainable refrigerant technology / climate-friendly cooling systems, supported by an official reference URL.",
        parent=constraints_node,
        critical=True,
    )
    focus_claim = (
        f"The center '{name_for_claims}' focuses on developing sustainable refrigerant technology to enable climate-friendly cooling systems."
    )
    await evaluator.verify(
        claim=focus_claim,
        node=focus_leaf,
        sources=urls,
        additional_instruction=build_urls_instruction(
            urls,
            "Look for phrasing such as sustainable refrigerants, climate-friendly cooling, low-global-warming potential refrigerants, etc., in official sources."
        ),
    )

    # 1.c Addresses both HFC emission reduction and warming impacts from refrigeration/cooling
    hfc_warming_leaf = evaluator.add_leaf(
        id="Addresses_HFC_And_Warming_With_Official_Ref",
        desc="The center addresses BOTH reduction of HFC emissions and reduction of global-warming impacts from refrigeration/cooling systems, supported by an official reference URL.",
        parent=constraints_node,
        critical=True,
    )
    hfc_warming_claim = (
        f"The center '{name_for_claims}' addresses both the reduction of hydrofluorocarbon (HFC) emissions and the reduction of global-warming impacts from refrigeration/cooling systems."
    )
    await evaluator.verify(
        claim=hfc_warming_claim,
        node=hfc_warming_leaf,
        sources=urls,
        additional_instruction=build_urls_instruction(
            urls,
            "Verify that the official source explicitly links the center’s mission to reducing HFC emissions and lowering global-warming impacts from cooling systems."
        ),
    )

    # =========================
    # 2) Acronym (critical, sequential)
    # =========================
    acronym_node = evaluator.add_sequential(
        id="Center_Acronym",
        desc="Provide the center's acronym and support it with a verifiable official reference URL.",
        parent=erc_node,
        critical=True,
    )

    # Existence check for acronym
    evaluator.add_custom_node(
        result=bool(info.acronym and info.acronym.strip()),
        id="Acronym_Provided",
        desc="An acronym for the identified center is provided.",
        parent=acronym_node,
        critical=True,
    )

    acronym_support_leaf = evaluator.add_leaf(
        id="Acronym_Supported_By_Official_Reference",
        desc="A verifiable official reference URL supports the provided acronym.",
        parent=acronym_node,
        critical=True,
    )
    acronym_claim = f"The center's acronym is '{(info.acronym or '').strip()}'."  # empty string handled by existence gate
    await evaluator.verify(
        claim=acronym_claim,
        node=acronym_support_leaf,
        sources=urls,
        additional_instruction=build_urls_instruction(
            urls,
            "The acronym should appear on official pages (NSF or the center’s official website). Minor formatting differences are acceptable."
        ),
    )

    # =========================
    # 3) Official full name (critical, sequential)
    # =========================
    fullname_node = evaluator.add_sequential(
        id="Center_Official_Full_Name",
        desc="Provide the center's official full name and support it with a verifiable official reference URL.",
        parent=erc_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.full_name and info.full_name.strip()),
        id="Full_Name_Provided",
        desc="The official full name of the identified center is provided.",
        parent=fullname_node,
        critical=True,
    )

    fullname_support_leaf = evaluator.add_leaf(
        id="Full_Name_Supported_By_Official_Reference",
        desc="A verifiable official reference URL supports the provided official full name.",
        parent=fullname_node,
        critical=True,
    )
    fullname_claim = f"The center's official full name is '{(info.full_name or '').strip()}'."  # gated by existence
    await evaluator.verify(
        claim=fullname_claim,
        node=fullname_support_leaf,
        sources=urls,
        additional_instruction=build_urls_instruction(
            urls,
            "Confirm the exact official full name on an NSF page or the center’s official website. Allow minor punctuation or capitalization differences."
        ),
    )

    # =========================
    # 4) Lead university (critical, sequential)
    # =========================
    lead_node = evaluator.add_sequential(
        id="Lead_University",
        desc="Identify the lead university (lead institution) and support it with a verifiable official reference URL.",
        parent=erc_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.lead_university and info.lead_university.strip()),
        id="Lead_Institution_Provided",
        desc="A lead university/lead institution for the identified center is provided.",
        parent=lead_node,
        critical=True,
    )

    lead_support_leaf = evaluator.add_leaf(
        id="Lead_Institution_Supported_By_Official_Reference",
        desc="A verifiable official reference URL supports the identified lead university/lead institution.",
        parent=lead_node,
        critical=True,
    )
    lead_claim = (
        f"The lead university (lead institution) of the center '{name_for_claims}' is '{(info.lead_university or '').strip()}'."
    )
    await evaluator.verify(
        claim=lead_claim,
        node=lead_support_leaf,
        sources=urls,
        additional_instruction=build_urls_instruction(
            urls,
            "Look for explicit 'lead institution' or 'lead university' statements on official sources. Synonyms like 'lead organization' are acceptable if clearly referring to the lead institution."
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
    Evaluate the answer for identifying the NSF Gen-4 ERC focused on sustainable refrigerants (Aug 2024).
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root can be parallel; actual rubric root is added as a critical sequential node
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

    # Extract structured center info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_center_info(),
        template_class=CenterInfo,
        extraction_name="erc_center_info",
    )

    # Add custom info for debugging/traceability
    evaluator.add_custom_info(
        info={
            "extracted_acronym": extracted_info.acronym,
            "extracted_full_name": extracted_info.full_name,
            "extracted_lead_university": extracted_info.lead_university,
            "official_urls": extracted_info.official_urls,
            "num_official_urls": len(extracted_info.official_urls),
        },
        info_type="extraction_summary",
        info_name="center_extraction_summary",
    )

    # Build verification tree and run checks
    await verify_center(evaluator, root, extracted_info)

    # Return structured result with verification tree and aggregation
    return evaluator.get_summary()