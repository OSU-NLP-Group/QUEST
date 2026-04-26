import asyncio
import logging
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "turkey_preparation_guide"
TASK_DESCRIPTION = """
Provide a complete preparation guide for brining and carving a Thanksgiving turkey. Your guide must include: the correct brine ratio (salt to water), brining duration and temperature requirements, post-brining preparation steps, the resting time needed after roasting, the proper carving sequence, and the recommended slicing technique.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TurkeyGuideExtraction(BaseModel):
    """Complete extraction of turkey preparation guide"""
    # Brining information
    salt_amount: Optional[str] = None
    water_amount: Optional[str] = None
    salt_type: Optional[str] = None
    brining_duration: Optional[str] = None
    refrigeration_requirement: Optional[str] = None
    submersion_requirement: Optional[str] = None
    
    # Post-brining preparation
    post_brine_steps: Optional[str] = None
    
    # Resting and carving
    resting_time: Optional[str] = None
    carving_sequence: Optional[str] = None
    slicing_technique: Optional[str] = None
    knife_recommendation: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_turkey_guide() -> str:
    return """
    Extract all information about turkey preparation from the answer. Include:
    
    BRINING INFORMATION:
    - salt_amount: The amount of salt for the brine (e.g., "1 cup")
    - water_amount: The amount of water for the brine (e.g., "4 quarts" or "1 gallon")
    - salt_type: The type of salt recommended (e.g., "kosher salt", "canning salt", or warnings about table salt)
    - brining_duration: The recommended brining time (e.g., "12-24 hours")
    - refrigeration_requirement: Any mention of keeping the turkey refrigerated/cold during brining
    - submersion_requirement: Any mention of keeping the turkey fully submerged in brine
    
    POST-BRINING PREPARATION:
    - post_brine_steps: Steps to take after brining (e.g., rinsing with water, patting dry)
    
    RESTING AND CARVING:
    - resting_time: How long the turkey should rest after roasting before carving
    - carving_sequence: The order/steps for carving the turkey
    - slicing_technique: How to slice the meat (e.g., against the grain)
    - knife_recommendation: Any recommendations about using a sharp knife
    
    Extract exactly what is stated in the answer. If any information is not mentioned, return null for that field.
    Preserve the original phrasing as much as possible.
    """


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict:
    """
    Evaluate an answer for the turkey preparation guide task.
    """
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

    # Extract all turkey preparation information
    extracted_guide = await evaluator.extract(
        prompt=prompt_extract_turkey_guide(),
        template_class=TurkeyGuideExtraction,
        extraction_name="turkey_preparation_guide",
    )

    # Add ground truth information
    evaluator.add_ground_truth({
        "expected_brine_ratio": "1 cup kosher salt per 4 quarts (1 gallon) water",
        "expected_duration": "12-24 hours (16-24 hours acceptable)",
        "expected_requirements": [
            "Refrigeration during brining",
            "Use kosher/canning salt, not table salt at same volume",
            "Full submersion in brine",
            "Rinse and pat dry after brining",
            "Rest 30+ minutes after roasting",
            "Carve in sequence: legs/thighs → breasts → wings → separate & debone",
            "Slice against grain",
            "Use sharp knife"
        ]
    })

    # --------------------------------------------------------------------------- #
    # Verification: Brine Ratio                                                   #
    # --------------------------------------------------------------------------- #
    brine_ratio_node = evaluator.add_leaf(
        id="brine_ratio",
        desc="Specifies the brine ratio as approximately 1 cup kosher salt per 4 quarts (1 gallon) of water",
        parent=root,
        critical=True,
    )
    
    claim = f"The answer specifies a brine ratio of approximately 1 cup of salt per 4 quarts (or 1 gallon) of water. The extracted salt amount is '{extracted_guide.salt_amount}' and water amount is '{extracted_guide.water_amount}'."
    await evaluator.verify(
        claim=claim,
        node=brine_ratio_node,
        additional_instruction="Verify that the ratio is approximately 1 cup salt to 4 quarts (1 gallon) water. Allow for reasonable variations like '1 cup per gallon', 'about 1 cup salt to 1 gallon water', or similar expressions that convey the same ratio. The ratio should be approximately correct even if not stated in exact terms."
    )

    # --------------------------------------------------------------------------- #
    # Verification: Brining Duration                                              #
    # --------------------------------------------------------------------------- #
    brining_duration_node = evaluator.add_leaf(
        id="brining_duration",
        desc="Specifies brining duration as 12–24 hours (with 16–24 hours recommended/acceptable)",
        parent=root,
        critical=True,
    )
    
    claim = f"The answer specifies a brining duration in the range of 12–24 hours (with 16–24 hours also being acceptable). The extracted duration is '{extracted_guide.brining_duration}'."
    await evaluator.verify(
        claim=claim,
        node=brining_duration_node,
        additional_instruction="Verify that the duration falls within or overlaps with the 12-24 hour range. Acceptable ranges include '12-24 hours', '16-24 hours', 'overnight (about 12-18 hours)', or similar time frames. The exact format may vary but should indicate a duration in this general range."
    )

    # --------------------------------------------------------------------------- #
    # Verification: Refrigeration Requirement                                     #
    # --------------------------------------------------------------------------- #
    refrigeration_node = evaluator.add_leaf(
        id="refrigeration_requirement",
        desc="States the turkey must be kept refrigerated during the entire brining period (temperature/food safety requirement)",
        parent=root,
        critical=True,
    )
    
    claim = f"The answer states that the turkey must be kept refrigerated during the brining period for food safety. The extracted refrigeration information is '{extracted_guide.refrigeration_requirement}'."
    await evaluator.verify(
        claim=claim,
        node=refrigeration_node,
        additional_instruction="Verify that the answer mentions keeping the turkey refrigerated, cold, or in the fridge during brining. This could be stated in various ways like 'keep refrigerated', 'store in refrigerator', 'keep cold', 'refrigerate while brining', etc. This is a critical food safety requirement."
    )

    # --------------------------------------------------------------------------- #
    # Verification: Salt Type Requirement                                         #
    # --------------------------------------------------------------------------- #
    salt_type_node = evaluator.add_leaf(
        id="salt_type_requirement",
        desc="Specifies using kosher salt or canning salt; not fine table salt at the same volume",
        parent=root,
        critical=True,
    )
    
    claim = f"The answer specifies using kosher salt or canning salt (or similar coarse salt), and warns against using fine table salt at the same volume measurement. The extracted salt type information is '{extracted_guide.salt_type}'."
    await evaluator.verify(
        claim=claim,
        node=salt_type_node,
        additional_instruction="Verify that the answer recommends kosher salt or canning salt (or similar coarse salt types). It should either explicitly state to use these types OR warn that table salt should not be used at the same volume/cup measurement because it's finer/denser. Look for mentions like 'kosher salt', 'coarse salt', 'canning salt', 'pickling salt', or cautions about 'table salt' being too concentrated."
    )

    # --------------------------------------------------------------------------- #
    # Verification: Submersion Requirement                                        #
    # --------------------------------------------------------------------------- #
    submersion_node = evaluator.add_leaf(
        id="submersion_requirement",
        desc="States the turkey must be fully submerged in the brine solution",
        parent=root,
        critical=True,
    )
    
    claim = f"The answer states that the turkey must be fully submerged or completely covered in the brine solution. The extracted submersion information is '{extracted_guide.submersion_requirement}'."
    await evaluator.verify(
        claim=claim,
        node=submersion_node,
        additional_instruction="Verify that the answer mentions the turkey should be fully submerged, completely covered, or entirely immersed in the brine. This could be phrased as 'fully submerged', 'completely covered', 'ensure turkey is covered by brine', 'submerge completely', etc."
    )

    # --------------------------------------------------------------------------- #
    # Verification: Post-Brine Preparation                                        #
    # --------------------------------------------------------------------------- #
    post_brine_node = evaluator.add_leaf(
        id="post_brine_preparation",
        desc="After brining: rinse turkey with cool water and pat dry before cooking",
        parent=root,
        critical=True,
    )
    
    claim = f"The answer describes post-brining preparation steps that include both rinsing the turkey with cool/cold water AND patting it dry before cooking. The extracted post-brine steps are '{extracted_guide.post_brine_steps}'."
    await evaluator.verify(
        claim=claim,
        node=post_brine_node,
        additional_instruction="Verify that the answer mentions BOTH rinsing (with water, could be cool/cold water) AND patting dry after brining. Both steps should be present. Look for phrases like 'rinse', 'rinse off', 'rinse with water' combined with 'pat dry', 'dry with paper towels', 'dry thoroughly', etc."
    )

    # --------------------------------------------------------------------------- #
    # Verification: Resting Time                                                  #
    # --------------------------------------------------------------------------- #
    resting_time_node = evaluator.add_leaf(
        id="resting_time",
        desc="Specifies the turkey must rest at least 30 minutes after roasting before carving",
        parent=root,
        critical=True,
    )
    
    claim = f"The answer specifies that the turkey should rest for at least 30 minutes after roasting and before carving. The extracted resting time is '{extracted_guide.resting_time}'."
    await evaluator.verify(
        claim=claim,
        node=resting_time_node,
        additional_instruction="Verify that the resting time is at least 30 minutes. Acceptable formats include '30 minutes', '30-45 minutes', 'at least 30 minutes', 'minimum 30 minutes', 'rest for 30 min', etc. Times of 30 minutes or longer are acceptable."
    )

    # --------------------------------------------------------------------------- #
    # Verification: Carving Sequence                                              #
    # --------------------------------------------------------------------------- #
    carving_sequence_node = evaluator.add_leaf(
        id="carving_sequence",
        desc="Describes the proper carving order: remove legs/thighs first, then breasts, then wings, then separate drumsticks from thighs, then debone thighs, and finally slice the meat",
        parent=root,
        critical=True,
    )
    
    claim = f"The answer describes a comprehensive carving sequence that follows a logical order: starting with legs/thighs, then breasts, then wings, with additional steps for separating drumsticks from thighs, deboning thighs, and finally slicing the meat. The extracted carving sequence is '{extracted_guide.carving_sequence}'."
    await evaluator.verify(
        claim=claim,
        node=carving_sequence_node,
        additional_instruction="Verify that the answer provides a detailed carving sequence covering the major steps in a logical order. The sequence should include: (1) removing legs/thighs first, (2) then breasts, (3) then wings, (4) separating drumsticks from thighs, (5) deboning thighs, and (6) slicing the meat. The exact wording may vary but the general order and key steps should be present. Not all steps need to be explicitly stated, but the main sequence (legs → breasts → wings, with separation/deboning mentioned) should be clear."
    )

    # --------------------------------------------------------------------------- #
    # Verification: Slicing Technique                                             #
    # --------------------------------------------------------------------------- #
    slicing_technique_node = evaluator.add_leaf(
        id="slicing_technique",
        desc="Recommends slicing against the grain for tenderness",
        parent=root,
        critical=True,
    )
    
    claim = f"The answer recommends slicing the turkey meat against the grain for better tenderness. The extracted slicing technique is '{extracted_guide.slicing_technique}'."
    await evaluator.verify(
        claim=claim,
        node=slicing_technique_node,
        additional_instruction="Verify that the answer mentions slicing 'against the grain' or using similar terminology that indicates this technique (e.g., 'slice perpendicular to the grain', 'cut across the grain', 'slice crosswise to the fibers'). This technique is important for achieving tender slices."
    )

    # --------------------------------------------------------------------------- #
    # Verification: Sharp Knife Recommendation                                    #
    # --------------------------------------------------------------------------- #
    sharp_knife_node = evaluator.add_leaf(
        id="sharp_knife_recommendation",
        desc="Recommends using a sharp knife to help keep the skin intact",
        parent=root,
        critical=True,
    )
    
    claim = f"The answer recommends using a sharp knife for carving, which helps keep the skin intact and makes cleaner cuts. The extracted knife recommendation is '{extracted_guide.knife_recommendation}'."
    await evaluator.verify(
        claim=claim,
        node=sharp_knife_node,
        additional_instruction="Verify that the answer mentions using a sharp knife. This could be stated as 'use a sharp knife', 'sharp carving knife', 'keep knife sharp', or similar phrases. The benefit (keeping skin intact, clean cuts) may or may not be explicitly mentioned, but the recommendation to use a sharp knife should be present."
    )

    # Return structured result
    return evaluator.get_summary()