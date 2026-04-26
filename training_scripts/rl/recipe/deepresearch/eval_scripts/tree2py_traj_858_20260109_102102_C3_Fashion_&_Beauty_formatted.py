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
TASK_ID = "designer_medicine_1953_brand_1975_compasso_2014"
TASK_DESCRIPTION = (
    "Who is the Italian fashion designer who studied medicine at a university before entering the fashion industry, "
    "left their medical studies in 1953, founded their eponymous fashion brand in 1975, and was awarded the "
    "Compasso d'Oro Award in 2014 for lifetime achievement in revolutionizing ready-to-wear fashion?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DesignerExtraction(BaseModel):
    name: Optional[str] = None
    nationality: Optional[str] = None
    studied_medicine_university: Optional[str] = None
    left_med_year: Optional[str] = None
    eponymous_brand_name: Optional[str] = None
    foundation_year: Optional[str] = None
    compasso_doro_year: Optional[str] = None
    compasso_doro_context: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_designer() -> str:
    return """
    Extract the single fashion designer the answer proposes. Capture biographical and award details if present, and all URLs cited.

    Return a JSON object with:
    - name: Full name of the designer proposed as the answer.
    - nationality: The nationality explicitly stated (e.g., "Italian") if present; otherwise null.
    - studied_medicine_university: The university where the designer studied medicine (if stated); otherwise null.
    - left_med_year: The year the designer left/dropped/quit medical studies (as a 4-digit year string) if stated; otherwise null.
    - eponymous_brand_name: The name of the designer’s eponymous brand (e.g., "Giorgio Armani", "Giorgio Armani S.p.A.") if stated; otherwise null.
    - foundation_year: The year the brand was founded (as a 4-digit year string) if stated; otherwise null.
    - compasso_doro_year: The year of Compasso d'Oro award if the answer mentions it; otherwise null.
    - compasso_doro_context: Any description/context about the award (e.g., "lifetime achievement", "alla carriera", "revolutionizing ready-to-wear") if present; otherwise null.
    - sources: All URLs explicitly present in the answer (including markdown links). If none, return an empty list.

    If multiple designers are mentioned, choose the one clearly proposed as the answer for "who is ... ?".
    Do not invent any value that is not stated. Use exact strings from the answer.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: DesignerExtraction) -> None:
    # Create the top-level critical node mirroring the rubric root
    root_criteria = evaluator.add_parallel(
        id="Designer_Identification_And_Constraint_Checks",
        desc="Verify the answer identifies a single designer and that the designer satisfies all stated biographical and award constraints.",
        parent=evaluator.root,
        critical=True,
    )

    # Sources used for evidence-based checks (may be empty; framework will fallback to simple verify)
    sources = extracted.sources or []

    # 1) Provides_Designer_Identity (existence check)
    name_present = extracted.name is not None and extracted.name.strip() != ""
    evaluator.add_custom_node(
        result=name_present,
        id="Provides_Designer_Identity",
        desc="Response clearly names/identifies the fashion designer being proposed as the answer.",
        parent=root_criteria,
        critical=True,
    )

    # Prepare reusable strings
    name = extracted.name or "the proposed designer"

    # 2) Italian_Nationality
    italian_node = evaluator.add_leaf(
        id="Italian_Nationality",
        desc="The designer must be Italian by nationality.",
        parent=root_criteria,
        critical=True,
    )
    italian_claim = f"The designer {name} is Italian by nationality."
    await evaluator.verify(
        claim=italian_claim,
        node=italian_node,
        sources=sources,
        additional_instruction=(
            "Verify nationality from the provided webpages if any. Accept wordings like 'Italian fashion designer' "
            "or 'born in Italy' indicating Italian nationality. Minor phrasing variations are acceptable."
        ),
    )

    # 3) Studied_Medicine_Before_Fashion
    studied_node = evaluator.add_leaf(
        id="Studied_Medicine_Before_Fashion",
        desc="The designer must have studied medicine at a university before pursuing a career in fashion.",
        parent=root_criteria,
        critical=True,
    )
    if extracted.studied_medicine_university:
        studied_claim = (
            f"Before entering the fashion industry, {name} studied medicine at {extracted.studied_medicine_university}."
        )
    else:
        studied_claim = f"Before entering the fashion industry, {name} studied medicine at a university."
    await evaluator.verify(
        claim=studied_claim,
        node=studied_node,
        sources=sources,
        additional_instruction=(
            "Confirm that the person studied medicine at a university prior to starting a fashion career. "
            "Accept synonyms like 'medical school' or 'medical studies'."
        ),
    )

    # 4) Left_Medical_Studies_1953
    left_1953_node = evaluator.add_leaf(
        id="Left_Medical_Studies_1953",
        desc="The designer must have left medical studies in 1953.",
        parent=root_criteria,
        critical=True,
    )
    left_claim = f"{name} left their medical studies in 1953."
    await evaluator.verify(
        claim=left_claim,
        node=left_1953_node,
        sources=sources,
        additional_instruction=(
            "Look for explicit mention that medical studies were abandoned/left/stopped in 1953. "
            "Accept equivalent phrasings such as 'dropped medicine in 1953' or 'left in 1953 to pursue another path'."
        ),
    )

    # 5) Founded_Eponymous_Brand_1975
    brand_node = evaluator.add_leaf(
        id="Founded_Eponymous_Brand_1975",
        desc="The designer must have founded their eponymous fashion brand in 1975.",
        parent=root_criteria,
        critical=True,
    )
    if extracted.eponymous_brand_name:
        brand_claim = (
            f"In 1975, {name} founded an eponymous fashion brand named '{extracted.eponymous_brand_name}'. "
            f"The brand is eponymous (it bears the designer's name)."
        )
    else:
        brand_claim = f"In 1975, {name} founded their eponymous fashion brand."
    await evaluator.verify(
        claim=brand_claim,
        node=brand_node,
        sources=sources,
        additional_instruction=(
            "Verify both (1) the founding year is 1975 and (2) the brand is eponymous (i.e., bears the designer's name). "
            "Accept formal corporate variants (e.g., 'S.p.A.', 'Ltd.') as long as the name reflects the designer's name."
        ),
    )

    # 6) Compasso_dOro_2014_Lifetime_Achievement_Detail
    compasso_node = evaluator.add_leaf(
        id="Compasso_dOro_2014_Lifetime_Achievement_Detail",
        desc=(
            "The designer must have received the Compasso d'Oro Award in 2014, and it must be specifically for lifetime "
            "achievement (as described in the question/constraints, e.g., lifetime achievement in fashion / "
            "revolutionizing ready-to-wear fashion)."
        ),
        parent=root_criteria,
        critical=True,
    )
    compasso_claim = (
        f"In 2014, {name} received the Compasso d'Oro (ADI Compasso d'Oro) lifetime achievement/career award, "
        f"recognizing their contributions to ready-to-wear fashion."
    )
    await evaluator.verify(
        claim=compasso_claim,
        node=compasso_node,
        sources=sources,
        additional_instruction=(
            "Confirm the award year is 2014 and that it was a lifetime achievement/career award (e.g., 'alla carriera'). "
            "Specific phrasing like 'revolutionizing ready-to-wear' can be expressed with equivalent wording such as "
            "'contributions to ready-to-wear' or 'significant influence on prêt-à-porter'."
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall evaluation can be parallel; the critical gating is handled in the child node
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_designer(),
        template_class=DesignerExtraction,
        extraction_name="designer_extraction",
    )

    # Optionally record normalized constants as custom info
    evaluator.add_custom_info(
        info={
            "required_years": {"left_medical_studies": 1953, "brand_founded": 1975, "compasso_doro": 2014},
            "note": "All core checks are critical under a single parallel node."
        },
        info_type="meta",
        info_name="evaluation_requirements",
    )

    # Build verification nodes and run checks
    await build_verification_tree(evaluator, extracted)

    # Return final summary
    return evaluator.get_summary()