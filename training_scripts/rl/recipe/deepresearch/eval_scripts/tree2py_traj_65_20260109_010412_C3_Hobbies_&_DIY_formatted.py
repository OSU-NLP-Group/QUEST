import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "shaker_side_table_specs"
TASK_DESCRIPTION = (
    "You are planning to build a traditional Shaker-style side table using only hand tools and authentic joinery methods. "
    "Your design calls for a table with 3/4-inch thick apron stock connected to the legs using mortise and tenon joints. "
    "You will use a bevel-down bench plane for final smoothing of all surfaces.\n\n"
    "Following traditional Shaker woodworking practices and accepted joinery rules, provide the following specifications:\n\n"
    "1. Wood Species: Name one traditional wood species historically used by Shakers that would be appropriate for this table.\n\n"
    "2. Mortise and Tenon Dimensions: Based on the 3/4-inch thick apron stock and applying the standard proportioning rule, calculate:\n"
    "   - The correct tenon thickness (width of mortise)\n"
    "   - The minimum tenon length\n\n"
    "3. Hand Plane Blade Angles: Specify the two blade bevel angles required for your smoothing plane:\n"
    "   - Primary bevel angle\n"
    "   - Secondary (honing/micro) bevel angle\n\n"
    "4. Leg Taper Specifications: Define the characteristic taper for Shaker table legs:\n"
    "   - Where the taper should start (distance from top of leg)\n"
    "   - How much the leg dimension should reduce by the bottom\n\n"
    "For each specification, provide a reference URL that supports your answer."
)

ALLOWED_SHAKER_SPECIES = [
    "cherry", "maple", "pine", "birch", "walnut", "hickory", "butternut", "beech", "oak", "poplar"
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class WoodSpeciesSpec(BaseModel):
    species: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class MortiseTenonSpec(BaseModel):
    tenon_thickness: Optional[str] = None
    minimum_tenon_length: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PlaneAnglesSpec(BaseModel):
    primary_bevel: Optional[str] = None
    secondary_bevel: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class LegTaperSpec(BaseModel):
    taper_start_distance: Optional[str] = None
    taper_reduction: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ShakerTableSpecs(BaseModel):
    wood_species: Optional[WoodSpeciesSpec] = None
    mortise_and_tenon_dimensions: Optional[MortiseTenonSpec] = None
    hand_plane_blade_angles: Optional[PlaneAnglesSpec] = None
    leg_taper_specification: Optional[LegTaperSpec] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_table_specs() -> str:
    return """
Extract the requested Shaker side table specifications from the answer, preserving the exact wording used by the answer wherever possible. Return a JSON object with the following structure:

{
  "wood_species": {
    "species": string | null,
    "urls": string[]    // all reference URLs explicitly provided in the answer that support the wood species choice
  },
  "mortise_and_tenon_dimensions": {
    "tenon_thickness": string | null,       // e.g., "1/4 inch", "0.25 in", "6 mm"
    "minimum_tenon_length": string | null,  // e.g., "1-1/4 inch", "1.25 in", "32 mm"
    "urls": string[]                        // URLs supporting the sizing rule(s)
  },
  "hand_plane_blade_angles": {
    "primary_bevel": string | null,         // e.g., "25°"
    "secondary_bevel": string | null,       // e.g., "30° micro-bevel"
    "urls": string[]                        // URLs supporting these angles for a bevel-down smoothing plane
  },
  "leg_taper_specification": {
    "taper_start_distance": string | null,  // e.g., "5 inches from the top"
    "taper_reduction": string | null,       // e.g., "to about half the thickness at the bottom"
    "urls": string[]                        // URLs supporting the taper start and reduction
  }
}

Rules:
- Extract only what the answer explicitly states.
- For any missing field, use null. For URL arrays, use [] if none are provided.
- For URLs, extract only valid URLs that appear in the answer (including markdown links). Do not invent URLs.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_wood_species(
    evaluator: Evaluator,
    root_node,
    specs: ShakerTableSpecs
) -> None:
    node = evaluator.add_parallel(
        id="wood_species",
        desc="Identify a traditional Shaker wood species suitable for the table",
        parent=root_node,
        critical=True
    )

    species = (specs.wood_species.species if specs.wood_species else None) or ""
    species_urls = (specs.wood_species.urls if specs.wood_species else []) or []

    # species_verification
    species_leaf = evaluator.add_leaf(
        id="species_verification",
        desc="Selected wood species must be historically used by Shakers (cherry, maple, pine, birch, walnut, hickory, butternut, beech, oak, or poplar)",
        parent=node,
        critical=True
    )
    species_claim = (
        f"The selected wood species '{species}' is historically used by Shakers and is among typical choices "
        f"such as {', '.join(ALLOWED_SHAKER_SPECIES)}. Allow close variants like black cherry, hard/sugar maple, "
        f"white pine, or tulip poplar to count as those base species."
    )
    await evaluator.verify(
        claim=species_claim,
        node=species_leaf,
        additional_instruction="Judge based on the answer content and common-sense equivalences (e.g., black cherry == cherry)."
    )

    # reference_url
    ref_leaf = evaluator.add_leaf(
        id="reference_url",
        desc="Provide a reference URL supporting the wood species selection",
        parent=node,
        critical=True
    )
    if not species_urls:
        ref_leaf.score = 0.0
        ref_leaf.status = "failed"
    else:
        ref_claim = (
            f"At least one of these pages explicitly states that Shaker furniture or the Shakers historically used "
            f"{species} (or an accepted synonym of it) as a material."
        )
        await evaluator.verify(
            claim=ref_claim,
            node=ref_leaf,
            sources=species_urls,
            additional_instruction="Accept phrasing like 'commonly used in Shaker furniture' or 'typical Shaker wood'."
        )


async def verify_mortise_and_tenon(
    evaluator: Evaluator,
    root_node,
    specs: ShakerTableSpecs
) -> None:
    node = evaluator.add_parallel(
        id="mortise_and_tenon_dimensions",
        desc="Provide mortise-and-tenon sizing for 3/4-inch apron stock (tenon thickness and minimum tenon length)",
        parent=root_node,
        critical=True
    )

    tenon_thickness = (specs.mortise_and_tenon_dimensions.tenon_thickness if specs.mortise_and_tenon_dimensions else None) or ""
    tenon_length = (specs.mortise_and_tenon_dimensions.minimum_tenon_length if specs.mortise_and_tenon_dimensions else None) or ""
    mt_urls = (specs.mortise_and_tenon_dimensions.urls if specs.mortise_and_tenon_dimensions else []) or []

    # tenon_thickness (≈ one-third of 3/4" => ~1/4")
    thickness_leaf = evaluator.add_leaf(
        id="tenon_thickness",
        desc="Tenon thickness must follow the one-third rule for 3/4-inch stock (≈ 1/4 inch)",
        parent=node,
        critical=True
    )
    thickness_claim = (
        f"The answer specifies a tenon thickness of '{tenon_thickness}'. Verify it matches the one-third rule for 3/4-inch stock, "
        f"which is approximately 1/4 inch (0.25 in, about 6–7 mm)."
    )
    await evaluator.verify(
        claim=thickness_claim,
        node=thickness_leaf,
        additional_instruction="Accept minor rounding and equivalent fractional/metric expressions that are near 1/4 inch."
    )

    # tenon_length (≥ 5× thickness; for ~1/4\" thickness, ≥ 1-1/4\")
    length_leaf = evaluator.add_leaf(
        id="tenon_length",
        desc="Minimum tenon length must follow the five-times rule (≥ 5× tenon thickness), which for this case is ≈ 1-1/4 inches",
        parent=node,
        critical=True
    )
    length_claim = (
        f"The answer specifies a minimum tenon length of '{tenon_length}'. Verify it is at least five times the tenon thickness. "
        f"If the tenon thickness is about 1/4 inch, then the minimum length should be around 1-1/4 inch (≈ 32 mm) or greater."
    )
    await evaluator.verify(
        claim=length_claim,
        node=length_leaf,
        additional_instruction="If the provided length is clearly ≥ 1-1/4 inch, consider it complying with the 5× rule when thickness is ~1/4 inch."
    )

    # mortise_tenon_reference
    mt_ref_leaf = evaluator.add_leaf(
        id="mortise_tenon_reference",
        desc="Provide a reference URL supporting the mortise-and-tenon sizing rule(s)",
        parent=node,
        critical=True
    )
    if not mt_urls:
        mt_ref_leaf.score = 0.0
        mt_ref_leaf.status = "failed"
    else:
        mt_ref_claim = (
            "At least one of these sources supports standard mortise-and-tenon proportioning: "
            "the 'one-third thickness' rule (for 3/4-inch stock this is about 1/4 inch) and/or a recommended minimum tenon length "
            "of about five times the tenon thickness."
        )
        await evaluator.verify(
            claim=mt_ref_claim,
            node=mt_ref_leaf,
            sources=mt_urls,
            additional_instruction="Support can be explicit for either rule; accepting ranges that include 5× is okay."
        )


async def verify_plane_angles(
    evaluator: Evaluator,
    root_node,
    specs: ShakerTableSpecs
) -> None:
    node = evaluator.add_parallel(
        id="hand_plane_blade_angles",
        desc="Specify blade bevel angles for a bevel-down bench plane used for smoothing",
        parent=root_node,
        critical=True
    )

    primary = (specs.hand_plane_blade_angles.primary_bevel if specs.hand_plane_blade_angles else None) or ""
    secondary = (specs.hand_plane_blade_angles.secondary_bevel if specs.hand_plane_blade_angles else None) or ""
    plane_urls = (specs.hand_plane_blade_angles.urls if specs.hand_plane_blade_angles else []) or []

    # primary_bevel
    primary_leaf = evaluator.add_leaf(
        id="primary_bevel",
        desc="Primary bevel angle must be 25 degrees",
        parent=node,
        critical=True
    )
    primary_claim = f"The primary bevel angle specified in the answer is '{primary}'. Verify that it is 25 degrees."
    await evaluator.verify(
        claim=primary_claim,
        node=primary_leaf,
        additional_instruction="Allow minor formatting variants like 25°, 25 deg, or 'twenty-five degrees'."
    )

    # secondary_bevel
    secondary_leaf = evaluator.add_leaf(
        id="secondary_bevel",
        desc="Secondary (honing/micro) bevel angle must be approximately 30 degrees",
        parent=node,
        critical=True
    )
    secondary_claim = (
        f"The secondary (honing/micro) bevel angle specified in the answer is '{secondary}'. "
        f"Verify that it is approximately 30 degrees (e.g., 29–31°)."
    )
    await evaluator.verify(
        claim=secondary_claim,
        node=secondary_leaf,
        additional_instruction="Small variations around 30° are acceptable as 'approximately 30 degrees'."
    )

    # bevel_reference
    bevel_ref_leaf = evaluator.add_leaf(
        id="bevel_reference",
        desc="Provide a reference URL supporting the specified bevel angles",
        parent=node,
        critical=True
    )
    if not plane_urls:
        bevel_ref_leaf.score = 0.0
        bevel_ref_leaf.status = "failed"
    else:
        bevel_ref_claim = (
            "At least one of these pages recommends for a bevel-down bench plane used for smoothing: a 25° primary bevel and "
            "a ~30° secondary (micro) bevel."
        )
        await evaluator.verify(
            claim=bevel_ref_claim,
            node=bevel_ref_leaf,
            sources=plane_urls,
            additional_instruction="The page should clearly indicate typical sharpening guidance: 25° primary and ~30° micro-bevel."
        )


async def verify_leg_taper(
    evaluator: Evaluator,
    root_node,
    specs: ShakerTableSpecs
) -> None:
    node = evaluator.add_parallel(
        id="leg_taper_specification",
        desc="Define characteristic Shaker leg taper start location and amount of taper",
        parent=root_node,
        critical=True
    )

    taper_start = (specs.leg_taper_specification.taper_start_distance if specs.leg_taper_specification else None) or ""
    taper_reduction = (specs.leg_taper_specification.taper_reduction if specs.leg_taper_specification else None) or ""
    taper_urls = (specs.leg_taper_specification.urls if specs.leg_taper_specification else []) or []

    # taper_start_point
    start_leaf = evaluator.add_leaf(
        id="taper_start_point",
        desc="Leg taper must start 4–6 inches from the top of the leg",
        parent=node,
        critical=True
    )
    start_claim = (
        f"The leg taper start location specified in the answer is '{taper_start}'. "
        f"Verify that the taper starts between 4 and 6 inches from the top of the leg."
    )
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        additional_instruction="If a range or approximate value is provided that falls within 4–6 inches, consider it acceptable."
    )

    # taper_reduction
    reduction_leaf = evaluator.add_leaf(
        id="taper_reduction",
        desc="Leg must taper to approximately half the original thickness/dimension at the bottom",
        parent=node,
        critical=True
    )
    reduction_claim = (
        f"The leg taper reduction specified in the answer is '{taper_reduction}'. "
        f"Verify that the bottom dimension is approximately half of the original thickness."
    )
    await evaluator.verify(
        claim=reduction_claim,
        node=reduction_leaf,
        additional_instruction="Accept phrasings like 'about half', '~50%', or numeric examples consistent with half."
    )

    # taper_reference
    taper_ref_leaf = evaluator.add_leaf(
        id="taper_reference",
        desc="Provide a reference URL supporting the Shaker leg taper specification(s)",
        parent=node,
        critical=True
    )
    if not taper_urls:
        taper_ref_leaf.score = 0.0
        taper_ref_leaf.status = "failed"
    else:
        taper_ref_claim = (
            "At least one of these sources describes a Shaker-style table leg taper that begins roughly 4–6 inches from the top "
            "and results in approximately half thickness at the bottom (often tapered on two inside faces)."
        )
        await evaluator.verify(
            claim=taper_ref_claim,
            node=taper_ref_leaf,
            sources=taper_urls,
            additional_instruction="Close variants are acceptable if they effectively communicate the same start location and reduction."
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
    # Make root critical as per rubric (must set before adding children)
    root.critical = True

    # Extraction
    extracted_specs = await evaluator.extract(
        prompt=prompt_extract_table_specs(),
        template_class=ShakerTableSpecs,
        extraction_name="shaker_table_specs"
    )

    # Build verification tree according to rubric
    await verify_wood_species(evaluator, root, extracted_specs)
    await verify_mortise_and_tenon(evaluator, root, extracted_specs)
    await verify_plane_angles(evaluator, root, extracted_specs)
    await verify_leg_taper(evaluator, root, extracted_specs)

    return evaluator.get_summary()