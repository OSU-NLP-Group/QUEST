import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task metadata                                                               #
# --------------------------------------------------------------------------- #
TASK_ID = "drawer_through_dovetail_tutorial"
TASK_DESCRIPTION = (
    "Find a comprehensive woodworking tutorial that teaches how to construct drawers using through dovetail joints and "
    "meets all of the following traditional furniture-making standards:\n\n"
    "1. The tutorial must specifically cover the through dovetail technique for drawer construction (not half-blind or other dovetail variations)\n\n"
    "2. Technical specifications must include:\n"
    "   - A dovetail angle ratio of 1:7 or an equivalent angle specification of approximately 7-8 degrees\n"
    "   - Guidance on pin dimensions, recommending widths in the range of 1/4 inch to 3/8 inch\n\n"
    "3. The tutorial must recommend using appropriate hardwood species for the drawer sides, such as oak, maple, birch, "
    "or other hardwoods with similar strength characteristics (Janka hardness rating above 1,200 lbf)\n\n"
    "4. The tutorial must be from a recognized authority in traditional woodworking, such as:\n"
    "   - Fine Woodworking magazine\n"
    "   - A book or video series by an established master woodworker (e.g., Paul Sellers, Frank Klausz, or equivalent)\n"
    "   - Another established authoritative woodworking publication or educational source\n\n"
    "Provide the tutorial title, author/source, and URL (if available online), along with specific evidence showing where "
    "each of the four requirements above is satisfied in the tutorial."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EvidenceItem(BaseModel):
    quote_or_excerpt: Optional[str] = None
    location: Optional[str] = None  # e.g., timestamp, page/section, URL anchor, heading


class TutorialMetadata(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None  # publisher/site/series
    tutorial_url: Optional[str] = None
    support_urls: List[str] = Field(default_factory=list)  # additional URLs cited in the answer (if any)


class TechnicalSpecs(BaseModel):
    dovetail_angle_spec: Optional[str] = None  # e.g., "1:7", "7 degrees", "8°", etc.
    pin_width_spec: Optional[str] = None       # e.g., "1/4 to 3/8 inch", or similar phrasing


class MaterialsRecommendation(BaseModel):
    hardwood_species: List[str] = Field(default_factory=list)  # e.g., ["oak", "maple", "birch"]
    note: Optional[str] = None  # any textual note the answer provides


class EvidenceBundle(BaseModel):
    technique: Optional[EvidenceItem] = None
    angle: Optional[EvidenceItem] = None
    pin_width: Optional[EvidenceItem] = None
    hardwood: Optional[EvidenceItem] = None
    authority: Optional[EvidenceItem] = None


class TutorialExtraction(BaseModel):
    metadata: Optional[TutorialMetadata] = None
    technique_focus: Optional[str] = None  # short phrase: e.g., "through dovetail drawers"
    recognized_authority_name: Optional[str] = None
    technical_specs: Optional[TechnicalSpecs] = None
    materials: Optional[MaterialsRecommendation] = None
    evidence: Optional[EvidenceBundle] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tutorial_info() -> str:
    return """
    From the answer, extract the following structured information about a single woodworking tutorial that teaches how to construct drawers using through dovetail joints.

    Return a JSON object with the following structure:
    {
      "metadata": {
        "title": string|null,
        "author": string|null,
        "source": string|null,
        "tutorial_url": string|null,
        "support_urls": string[]    // any additional URLs cited in the answer that relate to the tutorial or its authority
      },
      "technique_focus": string|null,  // the technique focus stated in the answer, e.g., "through dovetail drawers"
      "recognized_authority_name": string|null,  // e.g., "Fine Woodworking", "Paul Sellers", "Frank Klausz", etc., if stated
      "technical_specs": {
        "dovetail_angle_spec": string|null,  // the exact phrasing about angle or ratio (e.g., "1:7", "7 degrees", "8°")
        "pin_width_spec": string|null        // the exact phrasing about pin width (e.g., "1/4 to 3/8 inch")
      },
      "materials": {
        "hardwood_species": string[],  // list of hardwood species recommended for drawer sides, if any (e.g., ["oak","maple","birch"])
        "note": string|null            // any extra notes captured in the answer about material strength or hardness
      },
      "evidence": {
        "technique": {"quote_or_excerpt": string|null, "location": string|null},
        "angle": {"quote_or_excerpt": string|null, "location": string|null},
        "pin_width": {"quote_or_excerpt": string|null, "location": string|null},
        "hardwood": {"quote_or_excerpt": string|null, "location": string|null},
        "authority": {"quote_or_excerpt": string|null, "location": string|null}
      }
    }

    Rules:
    - Only extract what is explicitly present in the answer.
    - For any field that is not present, set it to null (or empty array for lists).
    - For URLs, return full URLs if present; if none are present, return null (or empty array).
    - Evidence.location can be a timestamp (e.g., 03:21), a page/section reference, a heading, or a URL anchor/fragment as stated in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _gather_sources(meta: Optional[TutorialMetadata]) -> List[str]:
    if not meta:
        return []
    urls = []
    if _non_empty(meta.tutorial_url):
        urls.append(meta.tutorial_url.strip())
    if meta.support_urls:
        urls.extend([u.strip() for u in meta.support_urls if _non_empty(u)])
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


# --------------------------------------------------------------------------- #
# Build verification tree and run checks                                      #
# --------------------------------------------------------------------------- #
async def _build_and_verify(evaluator: Evaluator, extracted: TutorialExtraction) -> None:
    # Root node (sequential, critical)
    root = evaluator.root

    # 1) Metadata node (parallel, child of root, but since root is critical, children must be critical)
    meta_node = evaluator.add_parallel(
        id="Provide_Tutorial_Metadata",
        desc="Provide the tutorial title, author/source, and URL (if available online).",
        parent=root,
        critical=True
    )

    metadata = extracted.metadata or TutorialMetadata()

    # 1.a Title provided
    evaluator.add_custom_node(
        result=_non_empty(metadata.title),
        id="Title_Provided",
        desc="Tutorial title is provided.",
        parent=meta_node,
        critical=True
    )

    # 1.b Author or Source provided
    evaluator.add_custom_node(
        result=_non_empty(metadata.author) or _non_empty(metadata.source),
        id="Author_or_Source_Provided",
        desc="Author and/or source/publisher is provided.",
        parent=meta_node,
        critical=True
    )

    # 1.c URL provided (adjusted to critical due to framework constraints with critical parent)
    evaluator.add_custom_node(
        result=_non_empty(metadata.tutorial_url),
        id="URL_Provided",
        desc="URL is provided for the tutorial.",
        parent=meta_node,
        critical=True
    )

    # 2) Validate all constraints with evidence (parallel, critical)
    validate_node = evaluator.add_parallel(
        id="Validate_All_Constraints_With_Evidence",
        desc="Demonstrate with citations that the tutorial satisfies each stated requirement.",
        parent=root,
        critical=True
    )

    sources = _gather_sources(metadata)

    # 2.a Through dovetail for drawers (not other variants)
    node_technique = evaluator.add_leaf(
        id="Through_Dovetail_For_Drawers_Not_Other_Variants",
        desc="Evidence shows the tutorial covers through dovetail joints specifically for drawer construction (not half-blind/sliding) as the taught technique.",
        parent=validate_node,
        critical=True
    )
    claim_technique = (
        "This tutorial teaches the construction of drawers using through dovetail joints as the primary technique "
        "(not half-blind or sliding dovetails). The drawer construction context is explicit."
    )
    await evaluator.verify(
        claim=claim_technique,
        node=node_technique,
        sources=sources,
        additional_instruction=(
            "Look for terms like 'through dovetail' alongside 'drawer'. It's okay if other variants are mentioned for comparison, "
            "but the main technique taught for the drawers must be through dovetails."
        )
    )

    # 2.b Angle spec explicit (1:7 or approx 7–8 degrees)
    node_angle = evaluator.add_leaf(
        id="Angle_Spec_Explicit",
        desc="Evidence shows an explicit dovetail angle specification of 1:7 or approximately 7–8 degrees.",
        parent=validate_node,
        critical=True
    )
    angle_hint = extracted.technical_specs.dovetail_angle_spec if (extracted.technical_specs and _non_empty(extracted.technical_specs.dovetail_angle_spec)) else None
    extra_angle = "For reference from the answer: " + angle_hint if angle_hint else "No specific phrasing was extracted from the answer."
    claim_angle = (
        "The tutorial explicitly specifies a dovetail angle suitable for through dovetails on drawers, and either states a 1:7 slope "
        "(1 in 7) or an equivalent angle around 7–8 degrees."
    )
    await evaluator.verify(
        claim=claim_angle,
        node=node_angle,
        sources=sources,
        additional_instruction=(
            "Accept '1:7', '1 in 7', or explicit degrees like '7°' or '8°' (or '7-8 degrees'). "
            "Reject if only 1:6 or 1:8 is given without including 1:7 or 7–8°. "
            + extra_angle
        )
    )

    # 2.c Pin width spec explicit (1/4 inch to 3/8 inch)
    node_pin = evaluator.add_leaf(
        id="Pin_Width_Spec_Explicit",
        desc="Evidence shows explicit guidance on pin widths in the range 1/4 inch to 3/8 inch.",
        parent=validate_node,
        critical=True
    )
    pin_hint = extracted.technical_specs.pin_width_spec if (extracted.technical_specs and _non_empty(extracted.technical_specs.pin_width_spec)) else None
    extra_pin = "For reference from the answer: " + pin_hint if pin_hint else "No specific phrasing was extracted from the answer."
    claim_pin = (
        "The tutorial provides explicit guidance that dovetail pin widths should be approximately 1/4 inch to 3/8 inch "
        "(or equivalent metric values around 6–10 mm)."
    )
    await evaluator.verify(
        claim=claim_pin,
        node=node_pin,
        sources=sources,
        additional_instruction=(
            "Look for explicit numeric guidance within that range. Accept phrasing like 'about 1/4 to 3/8 in.' or similar. "
            + extra_pin
        )
    )

    # 2.d Hardwood recommendation explicit with strength threshold
    node_hardwood = evaluator.add_leaf(
        id="Hardwood_Recommendation_Explicit_With_Strength_Threshold",
        desc="Evidence shows recommendation of appropriate hardwood species for drawer sides (e.g., oak/maple/birch or equivalent) aligned with Janka hardness above 1,200 lbf.",
        parent=validate_node,
        critical=True
    )
    species_list = (extracted.materials.hardwood_species if extracted.materials else []) or []
    species_hint = ", ".join(species_list) if species_list else "none extracted"
    claim_hardwood = (
        "For the drawer sides, the tutorial recommends appropriate hardwood species such as oak, maple, birch, beech, ash, or similar. "
        "These are woods typically above 1,200 lbf on the Janka hardness scale."
    )
    await evaluator.verify(
        claim=claim_hardwood,
        node=node_hardwood,
        sources=sources,
        additional_instruction=(
            "It is sufficient if the tutorial explicitly recommends common hardwoods like oak, maple, or birch for drawer sides. "
            "The page does not need to state the Janka number explicitly; listing those species is enough to satisfy the strength criterion. "
            f"Species mentioned by the answer: {species_hint}."
        )
    )

    # 2.e Recognized authority source
    node_authority = evaluator.add_leaf(
        id="Recognized_Authority_Source",
        desc="Evidence shows the tutorial is from a recognized authority (e.g., Fine Woodworking, Paul Sellers, Frank Klausz, or equivalent).",
        parent=validate_node,
        critical=True
    )
    authority_hint = extracted.recognized_authority_name if _non_empty(extracted.recognized_authority_name) else "not explicitly named in the answer"
    claim_authority = (
        "This tutorial is published by a recognized authority in traditional woodworking, such as Fine Woodworking magazine, "
        "Paul Sellers, Frank Klausz, or an established, reputable woodworking publication or educational source."
    )
    await evaluator.verify(
        claim=claim_authority,
        node=node_authority,
        sources=sources,
        additional_instruction=(
            "Accept if the page clearly indicates a reputable brand/author (e.g., finewoodworking.com, Paul Sellers, Frank Klausz, "
            "Lost Art Press, Popular Woodworking, The Woodwright's Shop, Lie-Nielsen/Veritas educational content). "
            f"Authority per answer: {authority_hint}."
        )
    )

    # 2.f Comprehensive instruction
    node_comprehensive = evaluator.add_leaf(
        id="Comprehensive_Instruction",
        desc="Evidence shows the tutorial is comprehensive (complete technique, not a brief mention).",
        parent=validate_node,
        critical=True
    )
    claim_comprehensive = (
        "The tutorial is comprehensive and teaches the complete technique for constructing drawers with through dovetail joints, "
        "including layout/marking, sawing, chopping/paring, fitting, and assembly (step-by-step or equivalent detailed coverage)."
    )
    await evaluator.verify(
        claim=claim_comprehensive,
        node=node_comprehensive,
        sources=sources,
        additional_instruction=(
            "Look for multi-step instructions, headings, sequence of steps, or a multi-part video/article series specifically covering "
            "through dovetail drawer construction from start to finish."
        )
    )

    # 2.g Evidence locations for each requirement (check based on the answer content)
    # We implement this as a custom node requiring that the answer provided specific evidence pointers
    # for technique, angle, pin width, hardwood, and authority (quotes, timestamps, sections, etc.).
    ev = extracted.evidence or EvidenceBundle()
    has_tech_evidence = bool(ev.technique and (_non_empty(ev.technique.quote_or_excerpt) or _non_empty(ev.technique.location)))
    has_angle_evidence = bool(ev.angle and (_non_empty(ev.angle.quote_or_excerpt) or _non_empty(ev.angle.location)))
    has_pin_evidence = bool(ev.pin_width and (_non_empty(ev.pin_width.quote_or_excerpt) or _non_empty(ev.pin_width.location)))
    has_hardwood_evidence = bool(ev.hardwood and (_non_empty(ev.hardwood.quote_or_excerpt) or _non_empty(ev.hardwood.location)))
    has_authority_evidence = bool(ev.authority and (_non_empty(ev.authority.quote_or_excerpt) or _non_empty(ev.authority.location)))

    evaluator.add_custom_node(
        result=(has_tech_evidence and has_angle_evidence and has_pin_evidence and has_hardwood_evidence and has_authority_evidence),
        id="Evidence_Locations_For_Each_Requirement",
        desc="Answer provides specific evidence locations (quote/timestamp/page/section) for technique, angle, pin width, hardwood, and authority.",
        parent=validate_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 'through dovetail drawer tutorial' task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # metadata first, then validation
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_tutorial_info(),
        template_class=TutorialExtraction,
        extraction_name="tutorial_extraction"
    )

    # Optionally record custom info summary about sources
    meta = extracted.metadata or TutorialMetadata()
    evaluator.add_custom_info(
        info={
            "title": meta.title,
            "author": meta.author,
            "source": meta.source,
            "tutorial_url": meta.tutorial_url,
            "support_urls_count": len(meta.support_urls or []),
        },
        info_type="extraction_summary",
        info_name="tutorial_metadata_summary"
    )

    # Build verification tree and run verifications
    await _build_and_verify(evaluator, extracted)

    # Return standardized summary
    return evaluator.get_summary()