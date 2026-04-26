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
TASK_ID = "bookshelf_specs"
TASK_DESCRIPTION = """
You are planning to build a DIY bookshelf for your home library that will hold a collection of heavy hardcover books. To ensure the bookshelf is structurally sound, aesthetically pleasing, and safe to construct, you need to determine the complete material and construction specifications. Your specifications must address the following requirements: (1) Material Selection: What type of wood material (plywood type and grade) should be used for the visible surfaces to achieve furniture-quality appearance? What minimum thickness is required for shelves that will support books? What is the acceptable moisture content range for wood used in indoor furniture? (2) Dimensional Standards: What is the standard depth range for bookshelves to accommodate most books? What is the recommended vertical spacing between shelves to accommodate various book sizes? What is the maximum unsupported span for shelves made from 3/4-inch plywood when they will be supporting books? (3) Safety Equipment: What essential personal protective equipment (at minimum: eye protection, and ideally hearing and respiratory protection) must be used during the construction process? For each specification, provide the specific values or ranges, and include URL references from reputable woodworking sources to support your specifications.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class MaterialProperties(BaseModel):
    plywood_type: Optional[str] = None
    plywood_type_urls: List[str] = Field(default_factory=list)
    plywood_grade: Optional[str] = None
    plywood_grade_urls: List[str] = Field(default_factory=list)
    shelf_thickness: Optional[str] = None
    thickness_urls: List[str] = Field(default_factory=list)


class MoistureSpec(BaseModel):
    moisture_content: Optional[str] = None
    moisture_urls: List[str] = Field(default_factory=list)


class DimensionalStandards(BaseModel):
    shelf_depth: Optional[str] = None
    depth_urls: List[str] = Field(default_factory=list)
    shelf_spacing: Optional[str] = None
    spacing_urls: List[str] = Field(default_factory=list)
    max_span: Optional[str] = None
    span_urls: List[str] = Field(default_factory=list)


class SafetyEquipment(BaseModel):
    eye_protection: Optional[str] = None
    eye_urls: List[str] = Field(default_factory=list)
    hearing_protection: Optional[str] = None
    respiratory_protection: Optional[str] = None


class BookshelfSpecsExtraction(BaseModel):
    material: Optional[MaterialProperties] = MaterialProperties()
    moisture: Optional[MoistureSpec] = MoistureSpec()
    dimensions: Optional[DimensionalStandards] = DimensionalStandards()
    safety: Optional[SafetyEquipment] = SafetyEquipment()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_bookshelf_specs() -> str:
    return """
    Extract a complete specification set for the bookshelf from the answer. Return a JSON object with the following structure and fields:

    {
      "material": {
        "plywood_type": string | null,                // e.g., "hardwood plywood (birch/maple/oak) for visible faces"
        "plywood_type_urls": string[]                 // URLs cited for the plywood type/visible faces recommendation
        "plywood_grade": string | null,               // e.g., "A-grade" or "AA"
        "plywood_grade_urls": string[]                // URLs cited for the plywood grade recommendation
        "shelf_thickness": string | null,             // e.g., "3/4 inch (19 mm) minimum"
        "thickness_urls": string[]                    // URLs cited for shelf thickness recommendation
      },
      "moisture": {
        "moisture_content": string | null,            // e.g., "6–8% for indoor furniture"
        "moisture_urls": string[]                     // URLs cited for the moisture content recommendation
      },
      "dimensions": {
        "shelf_depth": string | null,                 // e.g., "10–12 inches"
        "depth_urls": string[],                       // URLs cited for shelf depth
        "shelf_spacing": string | null,               // e.g., "9–13 inches"
        "spacing_urls": string[],                     // URLs cited for shelf spacing
        "max_span": string | null,                    // e.g., "≤ 32 inches for 3/4-inch plywood with books"
        "span_urls": string[]                         // URLs cited for maximum unsupported span
      },
      "safety": {
        "eye_protection": string | null,              // e.g., "safety glasses or goggles"
        "eye_urls": string[],                         // URLs cited for safety/PPE (eye protection)
        "hearing_protection": string | null,          // e.g., "hearing protection (earmuffs or earplugs)"
        "respiratory_protection": string | null       // e.g., "dust mask or respirator"
      }
    }

    Rules:
    - Only extract information explicitly present in the answer text.
    - For each '*_urls' field, extract every URL cited in the answer that supports that particular specification. Return an empty array if none are provided.
    - Keep textual values as they appear, allowing ranges like "6–8%" or "10–12 inches".
    - Do not fabricate or infer URLs. Include full URLs (prepend http:// if missing).
    """


# --------------------------------------------------------------------------- #
# Helper: add a reference-supported verification pair                         #
# --------------------------------------------------------------------------- #
async def add_reference_support_check(
    evaluator: Evaluator,
    parent_node,
    base_id: str,
    existence_desc: str,
    support_desc: str,
    ref_urls: List[str],
    claim: str,
    add_ins: str,
    critical: bool = True,
):
    # Existence check for references (critical to gate the subsequent support check)
    evaluator.add_custom_node(
        result=bool(ref_urls),
        id=f"{base_id}_reference_exists",
        desc=existence_desc,
        parent=parent_node,
        critical=critical
    )

    # Support check by URLs
    support_node = evaluator.add_leaf(
        id=f"{base_id}_reference_support",
        desc=support_desc,
        parent=parent_node,
        critical=critical
    )

    if ref_urls:
        await evaluator.verify(
            claim=claim,
            node=support_node,
            sources=ref_urls,
            additional_instruction=add_ins
        )
    else:
        # No references -> fail this support leaf
        support_node.score = 0.0
        support_node.status = "failed"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_material_selection(evaluator: Evaluator, root):
    """
    Build and verify the Material Selection subtree.
    """
    specs = await evaluator.extract(
        prompt=prompt_extract_bookshelf_specs(),
        template_class=BookshelfSpecsExtraction,
        extraction_name="bookshelf_specs"
    )

    # Material Selection (critical)
    material_node = evaluator.add_parallel(
        id="Material_Selection",
        desc="Appropriate wood material and grade selection for furniture construction",
        parent=root,
        critical=True
    )

    # Material Properties (critical)
    mat_props_node = evaluator.add_parallel(
        id="Material_Properties",
        desc="Physical and quality properties of the wood material",
        parent=material_node,
        critical=True
    )

    # Plywood Type (critical)
    ply_type_node = evaluator.add_parallel(
        id="Plywood_Type_Specification",
        desc="Specification includes hardwood plywood (birch, maple, or oak) for visible surfaces rather than softwood",
        parent=mat_props_node,
        critical=True
    )

    # Leaf: Specification stated in the answer
    ply_type_answer_leaf = evaluator.add_leaf(
        id="Plywood_Type_In_Answer",
        desc="Answer specifies hardwood plywood for visible faces (e.g., birch/maple/oak), not softwood",
        parent=ply_type_node,
        critical=True
    )
    await evaluator.verify(
        claim="The specification for visible surfaces uses hardwood plywood (e.g., birch, maple, or oak) rather than softwood.",
        node=ply_type_answer_leaf,
        additional_instruction="Judge solely from the answer text. Accept equivalent phrasings like 'hardwood veneer plywood', 'cabinet-grade hardwood plywood', or explicit hardwood species (birch, maple, oak)."
    )

    # Leaf(s): Reference existence + support
    ply_type_urls = (specs.material.plywood_type_urls if specs and specs.material else []) or []
    await add_reference_support_check(
        evaluator=evaluator,
        parent_node=ply_type_node,
        base_id="Plywood_Type_Reference",
        existence_desc="Provides URL reference(s) supporting the plywood type specification",
        support_desc="Cited URL(s) support using hardwood plywood for visible furniture/cabinet surfaces",
        ref_urls=ply_type_urls,
        claim="Hardwood plywood (such as birch, maple, or oak) is recommended for furniture-quality visible surfaces (as opposed to softwood plywood).",
        add_ins="Verify the page recommends hardwood plywood for cabinets/furniture or visible faces. Accept synonymous terms like 'cabinet-grade hardwood plywood' or species-specific recommendations.",
        critical=True
    )

    # Plywood Grade (critical)
    ply_grade_node = evaluator.add_parallel(
        id="Plywood_Grade_Specification",
        desc="Specification includes A-grade or A-A grade plywood for furniture-quality surfaces",
        parent=mat_props_node,
        critical=True
    )

    grade_answer_leaf = evaluator.add_leaf(
        id="Plywood_Grade_In_Answer",
        desc="Answer specifies A-grade or AA-grade plywood for furniture-quality visible surfaces",
        parent=ply_grade_node,
        critical=True
    )
    await evaluator.verify(
        claim="The specification calls for A-grade or AA-grade plywood for furniture-quality visible surfaces.",
        node=grade_answer_leaf,
        additional_instruction="Check the answer text for A, AA, A-A, or 'cabinet/furniture grade' explicitly tied to plywood grade."
    )

    grade_urls = (specs.material.plywood_grade_urls if specs and specs.material else []) or []
    await add_reference_support_check(
        evaluator=evaluator,
        parent_node=ply_grade_node,
        base_id="Plywood_Grade_Reference",
        existence_desc="Provides URL reference(s) supporting the plywood grade specification",
        support_desc="Cited URL(s) support using A/AA-grade plywood for furniture-quality visible surfaces",
        ref_urls=grade_urls,
        claim="A-grade or AA-grade plywood is appropriate for furniture-quality visible surfaces.",
        add_ins="Verify the page describes plywood grading and indicates that A or AA grades are suitable for furniture/cabinetry visible faces.",
        critical=True
    )

    # Thickness (critical)
    thickness_node = evaluator.add_parallel(
        id="Thickness_Specification",
        desc="Specification includes 3/4 inch (or thicker) plywood for shelves supporting books",
        parent=mat_props_node,
        critical=True
    )

    thickness_answer_leaf = evaluator.add_leaf(
        id="Shelf_Thickness_In_Answer",
        desc="Answer specifies shelves are at least 3/4 inch (19 mm) thick",
        parent=thickness_node,
        critical=True
    )
    await evaluator.verify(
        claim="Shelves that will support books are specified to be at least 3/4 inch (19 mm) thick.",
        node=thickness_answer_leaf,
        additional_instruction="Check the answer text; values greater than 3/4 inch are acceptable as meeting or exceeding the minimum."
    )

    thickness_urls = (specs.material.thickness_urls if specs and specs.material else []) or []
    await add_reference_support_check(
        evaluator=evaluator,
        parent_node=thickness_node,
        base_id="Thickness_Reference",
        existence_desc="Provides URL reference(s) supporting the shelf thickness specification",
        support_desc="Cited URL(s) support that shelves for books should be 3/4 inch thick or thicker",
        ref_urls=thickness_urls,
        claim="Shelves intended to support books should be at least 3/4 inch (19 mm) thick.",
        add_ins="Verify the page recommends ~3/4 inch thickness (or thicker) for book-bearing shelves.",
        critical=True
    )

    # Moisture Content (critical, sibling under Material_Selection)
    moisture_node = evaluator.add_parallel(
        id="Moisture_Content_Specification",
        desc="Specification includes wood moisture content between 6% and 8% for indoor furniture",
        parent=material_node,
        critical=True
    )

    moisture_answer_leaf = evaluator.add_leaf(
        id="Moisture_Content_In_Answer",
        desc="Answer specifies wood moisture content 6–8% for indoor furniture",
        parent=moisture_node,
        critical=True
    )
    await evaluator.verify(
        claim="The wood moisture content for indoor furniture is specified between 6% and 8%.",
        node=moisture_answer_leaf,
        additional_instruction="Judge from the answer text; allow equivalent formatting like '6 to 8 percent' or 'MC 6–8%'."
    )

    moisture_urls = (specs.moisture.moisture_urls if specs and specs.moisture else []) or []
    await add_reference_support_check(
        evaluator=evaluator,
        parent_node=moisture_node,
        base_id="Moisture_Content_Reference",
        existence_desc="Provides URL reference(s) supporting the moisture content specification",
        support_desc="Cited URL(s) support indoor furniture wood moisture content of about 6–8%",
        ref_urls=moisture_urls,
        claim="For indoor furniture, wood moisture content of approximately 6–8% is recommended.",
        add_ins="Accept reputable woodworking sources stating ~6–8% for indoor furniture MC; minor variations (e.g., 6–9%) are acceptable if they include or closely align with 6–8%.",
        critical=True
    )


async def verify_dimensional_standards(evaluator: Evaluator, root):
    """
    Build and verify the Dimensional Standards subtree.
    Note: Parent is critical; to satisfy framework constraints, we set child nodes critical as well.
    """
    # Reuse the previously extracted structure from evaluator records
    # Find the last extraction entry of type 'bookshelf_specs'
    # But we already executed extraction in verify_material_selection and recorded it.
    # For simplicity and consistency, re-extract here (idempotent and recorded).
    specs = await evaluator.extract(
        prompt=prompt_extract_bookshelf_specs(),
        template_class=BookshelfSpecsExtraction,
        extraction_name="bookshelf_specs_dims"
    )

    dims_node = evaluator.add_parallel(
        id="Dimensional_Standards",
        desc="Shelf dimensions that meet standard bookshelf requirements",
        parent=root,
        critical=True
    )

    # Shelf Depth (critical)
    depth_node = evaluator.add_parallel(
        id="Shelf_Depth_Standard",
        desc="Shelf depth specification between 10 and 12 inches to accommodate standard books",
        parent=dims_node,
        critical=True
    )

    depth_answer_leaf = evaluator.add_leaf(
        id="Shelf_Depth_In_Answer",
        desc="Answer specifies shelf depth between 10 and 12 inches",
        parent=depth_node,
        critical=True
    )
    await evaluator.verify(
        claim="The shelf depth is specified between 10 and 12 inches.",
        node=depth_answer_leaf,
        additional_instruction="Judge from the answer text; accept equivalent phrasing like 'about 11 inches' or '10–12 in'."
    )

    depth_urls = (specs.dimensions.depth_urls if specs and specs.dimensions else []) or []
    await add_reference_support_check(
        evaluator=evaluator,
        parent_node=depth_node,
        base_id="Depth_Reference",
        existence_desc="Provides URL reference(s) supporting the shelf depth standard",
        support_desc="Cited URL(s) support typical bookshelf depth around 10–12 inches",
        ref_urls=depth_urls,
        claim="Typical bookshelf shelf depth is around 10–12 inches.",
        add_ins="Verify the page indicates standard/typical bookshelf depths near 10–12 inches.",
        critical=True
    )

    # Shelf Spacing (JSON marks non-critical, but parent is critical; set to critical=True to satisfy constraint)
    spacing_node = evaluator.add_parallel(
        id="Shelf_Spacing_Standard",
        desc="Vertical spacing between shelves specified as 9 to 13 inches to accommodate various book sizes",
        parent=dims_node,
        critical=True
    )

    spacing_answer_leaf = evaluator.add_leaf(
        id="Shelf_Spacing_In_Answer",
        desc="Answer specifies vertical shelf spacing between 9 and 13 inches",
        parent=spacing_node,
        critical=True
    )
    await evaluator.verify(
        claim="The recommended vertical spacing between shelves is 9 to 13 inches.",
        node=spacing_answer_leaf,
        additional_instruction="Judge from the answer text; accept ranges overlapping 9–13 inches if clearly presented as recommended spacing for books."
    )

    spacing_urls = (specs.dimensions.spacing_urls if specs and specs.dimensions else []) or []
    await add_reference_support_check(
        evaluator=evaluator,
        parent_node=spacing_node,
        base_id="Spacing_Reference",
        existence_desc="Provides URL reference(s) supporting the shelf spacing standard",
        support_desc="Cited URL(s) support shelf spacing recommendations around 9–13 inches",
        ref_urls=spacing_urls,
        claim="Recommended vertical shelf spacing for books is about 9–13 inches.",
        add_ins="Verify the page recommends spacing within or near 9–13 inches for typical book storage.",
        critical=True
    )

    # Maximum Span (critical)
    span_node = evaluator.add_parallel(
        id="Maximum_Span_Specification",
        desc="Maximum unsupported shelf span specified as 32 inches or less for 3/4 inch plywood supporting books",
        parent=dims_node,
        critical=True
    )

    span_answer_leaf = evaluator.add_leaf(
        id="Max_Span_In_Answer",
        desc="Answer specifies maximum unsupported span ≤ 32 inches for 3/4-inch plywood with books",
        parent=span_node,
        critical=True
    )
    await evaluator.verify(
        claim="The maximum unsupported span for 3/4-inch plywood shelves carrying books is specified as 32 inches or less.",
        node=span_answer_leaf,
        additional_instruction="Judge from the answer text; accept stricter limits (e.g., 30 in) as meeting the '≤ 32 in' requirement."
    )

    span_urls = (specs.dimensions.span_urls if specs and specs.dimensions else []) or []
    await add_reference_support_check(
        evaluator=evaluator,
        parent_node=span_node,
        base_id="Span_Reference",
        existence_desc="Provides URL reference(s) supporting the maximum span specification",
        support_desc="Cited URL(s) support ≤ 32 inch max unsupported span for 3/4-inch plywood bookshelves",
        ref_urls=span_urls,
        claim="For 3/4-inch plywood shelves loaded with books, the maximum unsupported span should be about 32 inches or less.",
        add_ins="Verify the page indicates a typical maximum span near or below 32 inches for 3/4 in plywood supporting books (without a center support).",
        critical=True
    )


async def verify_safety_equipment(evaluator: Evaluator, root):
    """
    Build and verify the Safety Equipment subtree.
    """
    specs = await evaluator.extract(
        prompt=prompt_extract_bookshelf_specs(),
        template_class=BookshelfSpecsExtraction,
        extraction_name="bookshelf_specs_safety"
    )

    safety_node = evaluator.add_parallel(
        id="Safety_Equipment_Requirements",
        desc="Essential safety equipment for woodworking operations",
        parent=root,
        critical=False
    )

    # Eye Protection (critical under Safety per rubric)
    eye_node = evaluator.add_parallel(
        id="Eye_Protection",
        desc="Specification includes safety glasses or goggles for eye protection",
        parent=safety_node,
        critical=True
    )

    eye_in_answer = evaluator.add_leaf(
        id="Eye_Protection_In_Answer",
        desc="Answer specifies safety glasses or goggles for eye protection",
        parent=eye_node,
        critical=True
    )
    await evaluator.verify(
        claim="The specification requires eye protection, such as safety glasses or goggles.",
        node=eye_in_answer,
        additional_instruction="Judge from the answer text; accept equivalent phrasings indicating mandatory eye protection."
    )

    eye_urls = (specs.safety.eye_urls if specs and specs.safety else []) or []
    await add_reference_support_check(
        evaluator=evaluator,
        parent_node=eye_node,
        base_id="Safety_Reference",
        existence_desc="Provides URL reference(s) supporting safety equipment requirements (eye protection)",
        support_desc="Cited URL(s) support the requirement to wear eye protection (safety glasses/goggles) during woodworking",
        ref_urls=eye_urls,
        claim="Woodworking safety guidance requires wearing eye protection such as safety glasses or goggles.",
        add_ins="Verify the page is a reputable safety/woodworking source and clearly requires or strongly recommends eye protection.",
        critical=True
    )

    # Hearing Protection (non-critical leaf)
    hearing_leaf = evaluator.add_leaf(
        id="Hearing_Protection",
        desc="Specification includes hearing protection (earmuffs or earplugs)",
        parent=safety_node,
        critical=False
    )
    await evaluator.verify(
        claim="The specification includes hearing protection, such as earmuffs or earplugs.",
        node=hearing_leaf,
        additional_instruction="Judge from the answer text; accept equivalent phrasing that indicates hearing protection is recommended/required."
    )

    # Respiratory Protection (non-critical leaf)
    resp_leaf = evaluator.add_leaf(
        id="Respiratory_Protection",
        desc="Specification includes dust mask or respirator for respiratory protection",
        parent=safety_node,
        critical=False
    )
    await evaluator.verify(
        claim="The specification includes respiratory protection, such as a dust mask or respirator.",
        node=resp_leaf,
        additional_instruction="Judge from the answer text; accept equivalent phrasing for respirators or dust masks."
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
    """
    Evaluate an answer for the DIY bookshelf construction specifications task.
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

    # Optional: Document expected targets as "ground truth" context (for interpretability only)
    evaluator.add_ground_truth({
        "expected_specs": {
            "material": {
                "plywood_type": "Hardwood plywood (e.g., birch/maple/oak) for visible faces",
                "plywood_grade": "A or AA grade for furniture-quality surfaces",
                "shelf_thickness_min": "≥ 3/4 inch (19 mm)"
            },
            "moisture": {
                "indoor_furniture_mc": "Approximately 6–8%"
            },
            "dimensions": {
                "shelf_depth": "10–12 inches",
                "shelf_spacing": "9–13 inches",
                "max_span_3_4_plywood_books": "≤ 32 inches"
            },
            "safety": {
                "eye": "Safety glasses/goggles (required)",
                "hearing": "Earmuffs/earplugs (recommended)",
                "respiratory": "Dust mask/respirator (recommended)"
            }
        }
    })

    # Build subtrees
    await verify_material_selection(evaluator, root)
    await verify_dimensional_standards(evaluator, root)
    await verify_safety_equipment(evaluator, root)

    return evaluator.get_summary()