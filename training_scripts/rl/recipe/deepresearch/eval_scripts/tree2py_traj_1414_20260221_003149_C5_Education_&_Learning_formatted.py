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
TASK_ID = "educational_pathway_research"
TASK_DESCRIPTION = (
    "Identify three community colleges located in Alabama or Texas that offer OSHA 30-Hour Construction certification training. "
    "For each community college, provide the following information: the name of the community college, the state where it is located "
    "(Alabama or Texas), confirmation that the program specifically offers OSHA 30-Hour Construction certification (not General Industry), "
    "whether the program provides an official OSHA DOL card upon completion, and a direct link to the community college's webpage describing "
    "their OSHA 30-Hour Construction training program. Additionally, identify two universities located in Texas that have bachelor's degree "
    "programs in Construction Management, Construction Science, or Construction Science & Management that are accredited by ACCE (American "
    "Council for Construction Education) and accept transfer students from community colleges. For each university, provide: the name of the "
    "university, the specific name of the Construction Management or Construction Science bachelor's degree program, confirmation that the "
    "program is accredited by ACCE, confirmation that the university accepts transfer students from community colleges, and a direct link to "
    "the university's webpage describing their ACCE-accredited Construction Management or Construction Science program."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CollegeItem(BaseModel):
    college_name: Optional[str] = None
    state: Optional[str] = None  # Expect "Alabama" or "Texas" (or AL/TX abbreviations)
    program_url: Optional[str] = None
    offers_osha_30_construction: Optional[str] = None  # "yes"/"no"/"unknown" as stated in answer
    provides_dol_card: Optional[str] = None            # "yes"/"no"/"unknown" as stated in answer


class CollegesExtraction(BaseModel):
    colleges: List[CollegeItem] = Field(default_factory=list)


class UniversityItem(BaseModel):
    university_name: Optional[str] = None
    program_name: Optional[str] = None  # e.g., "BS in Construction Management"
    program_url: Optional[str] = None
    acce_accredited: Optional[str] = None     # "yes"/"no"/"unknown" as stated in answer
    accepts_transfers: Optional[str] = None   # "yes"/"no"/"unknown" as stated in answer
    state: Optional[str] = None               # Expect "Texas" if provided


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_colleges() -> str:
    return (
        "Extract up to three community colleges mentioned in the answer that offer OSHA 30-Hour Construction certification training. "
        "For each community college, extract the following fields strictly from the answer text:\n"
        "- college_name: the name of the community college\n"
        "- state: the state indicated in the answer for the college (prefer full name like 'Alabama' or 'Texas'; abbreviations 'AL'/'TX' are acceptable)\n"
        "- program_url: the direct URL to the college's webpage that describes their OSHA 30-Hour Construction training program; if multiple URLs are given, choose the most direct program page URL; if none, return null\n"
        "- offers_osha_30_construction: whether the answer explicitly states that the program is OSHA 30-Hour Construction (not General Industry); return 'yes', 'no', or 'unknown'\n"
        "- provides_dol_card: whether the answer mentions that an official OSHA DOL card is provided upon completion; return 'yes', 'no', or 'unknown'\n\n"
        "Return a JSON object: { colleges: [ {college_name, state, program_url, offers_osha_30_construction, provides_dol_card}, ... ] }.\n"
        "If the answer lists more than three eligible colleges, include only the first three. If fewer are mentioned, include all that are present.\n"
        "Apply the URL extraction rules: extract actual URLs present in the answer; if missing protocol, prepend http://; do not invent URLs."
    )


def prompt_extract_universities() -> str:
    return (
        "Extract up to two Texas universities mentioned in the answer that have bachelor's programs in Construction Management, "
        "Construction Science, or Construction Science & Management, are ACCE-accredited, and accept transfer students from community colleges. "
        "For each university, extract the following fields strictly from the answer text:\n"
        "- university_name: the name of the university\n"
        "- program_name: the specific name of the bachelor's program (e.g., 'BS in Construction Management')\n"
        "- program_url: the direct URL to the university's webpage describing the program (prefer a page that mentions ACCE accreditation if provided); if multiple URLs are given, choose the most direct program page URL; if none, return null\n"
        "- acce_accredited: whether the answer explicitly states the program is ACCE-accredited; return 'yes', 'no', or 'unknown'\n"
        "- accepts_transfers: whether the answer states the university accepts transfer students from community colleges; return 'yes', 'no', or 'unknown'\n"
        "- state: the state indicated for the university if provided; if missing, return null\n\n"
        "Return a JSON object: { universities: [ {university_name, program_name, program_url, acce_accredited, accepts_transfers, state}, ... ] }.\n"
        "If the answer lists more than two eligible universities, include only the first two. If fewer are mentioned, include all that are present.\n"
        "Apply the URL extraction rules: extract actual URLs present in the answer; if missing protocol, prepend http://; do not invent URLs."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_community_college(
    evaluator: Evaluator,
    parent_node,
    item: CollegeItem,
    index: int,
) -> None:
    """
    Build and verify the subtree for a single community college.
    """
    cc_node = evaluator.add_parallel(
        id=f"Community_College_{index + 1}",
        desc=f"{['First','Second','Third'][index]} community college in Alabama or Texas offering OSHA 30-Hour Construction certification",
        parent=parent_node,
        critical=False,
    )

    # College URL Reference (critical)
    url_desc = "A valid URL reference to the community college's OSHA training program page"
    url_node = evaluator.add_leaf(
        id=f"cc_{index}_url_reference",
        desc=url_desc,
        parent=cc_node,
        critical=True,
    )

    # If we have a program_url, verify by URL that it is indeed the program page. Otherwise, verify presence of a URL in the answer.
    if item.program_url:
        claim = "This URL is a community college webpage that describes its OSHA 30-Hour Construction training program."
        await evaluator.verify(
            claim=claim,
            node=url_node,
            sources=item.program_url,
            additional_instruction="Confirm the page belongs to a community college and specifically describes OSHA 30-Hour Construction training (not General Industry)."
        )
    else:
        claim = "The answer includes a valid direct URL to the community college's OSHA 30-Hour Construction training program page."
        await evaluator.verify(
            claim=claim,
            node=url_node,
            sources=None,
            additional_instruction="Check the answer text for any direct URL pointing to the program page; accept markdown links or plain URLs. If no such URL is present, mark as Incorrect."
        )

    # OSHA 30-Hour Construction Offered (critical)
    osha30_node = evaluator.add_leaf(
        id=f"cc_{index}_osha30_construction_offered",
        desc="The community college offers OSHA 30-Hour Construction certification (not General Industry)",
        parent=cc_node,
        critical=True,
    )
    claim = "This program page explicitly offers OSHA 30-Hour Construction training (Construction Industry), not the General Industry version."
    await evaluator.verify(
        claim=claim,
        node=osha30_node,
        sources=item.program_url,
        additional_instruction="Look for phrases like 'OSHA 30-Hour Construction', 'Construction Industry', 'OSHA 30 Construction'. Do NOT accept 'OSHA 30-Hour General Industry'.",
        extra_prerequisites=[url_node]
    )

    # State Location (critical)
    state_node = evaluator.add_leaf(
        id=f"cc_{index}_state_location",
        desc="The community college is located in Alabama or Texas",
        parent=cc_node,
        critical=True,
    )
    claim = "This community college is located in Alabama or Texas."
    await evaluator.verify(
        claim=claim,
        node=state_node,
        sources=item.program_url,
        additional_instruction="Verify the college's location from the program page (address, campus info, or footer/contact sections). Accept 'AL' or 'TX' and city names in those states.",
        extra_prerequisites=[url_node]
    )

    # DOL Card Provided (critical)
    dol_node = evaluator.add_leaf(
        id=f"cc_{index}_dol_card_provided",
        desc="The program provides an official OSHA DOL card upon completion",
        parent=cc_node,
        critical=True,
    )
    claim = "This OSHA 30-Hour Construction training program provides an official OSHA Department of Labor (DOL) card upon completion."
    await evaluator.verify(
        claim=claim,
        node=dol_node,
        sources=item.program_url,
        additional_instruction="Look for mentions of 'OSHA DOL card', 'Department of Labor card', 'OSHA 30 card issued upon completion'.",
        extra_prerequisites=[url_node]
    )


async def verify_texas_university(
    evaluator: Evaluator,
    parent_node,
    item: UniversityItem,
    index: int,
) -> None:
    """
    Build and verify the subtree for a single Texas university's program.
    """
    uni_node = evaluator.add_parallel(
        id=f"Texas_University_{index + 1}",
        desc=f"{['First','Second'][index]} Texas university with ACCE-accredited Construction Management program accepting transfers",
        parent=parent_node,
        critical=False,
    )

    # University URL Reference (critical)
    url_desc = "A valid URL reference to the university's Construction Management program page"
    url_node = evaluator.add_leaf(
        id=f"uni_{index}_url_reference",
        desc=url_desc,
        parent=uni_node,
        critical=True,
    )

    if item.program_url:
        claim = "This URL is the university's program page describing the Construction Management/Construction Science bachelor's program."
        await evaluator.verify(
            claim=claim,
            node=url_node,
            sources=item.program_url,
            additional_instruction="Confirm the page belongs to the university and specifically describes the bachelor's program in Construction Management, Construction Science, or Construction Science & Management."
        )
    else:
        claim = "The answer includes a valid direct URL to the university's program page for Construction Management/Construction Science."
        await evaluator.verify(
            claim=claim,
            node=url_node,
            sources=None,
            additional_instruction="Check the answer text for any direct program page URL; accept markdown links or plain URLs. If no such URL is present, mark as Incorrect."
        )

    # State Location Texas (critical)
    state_node = evaluator.add_leaf(
        id=f"uni_{index}_state_location_texas",
        desc="The university is located in Texas",
        parent=uni_node,
        critical=True,
    )
    claim = "This university is located in Texas."
    await evaluator.verify(
        claim=claim,
        node=state_node,
        sources=item.program_url,
        additional_instruction="Verify location from the program page or footer. Accept evidence such as 'TX', city in Texas, or 'Texas' stated.",
        extra_prerequisites=[url_node]
    )

    # ACCE Accreditation (critical)
    acce_node = evaluator.add_leaf(
        id=f"uni_{index}_acce_accreditation",
        desc="The university's Construction Management or Construction Science program is accredited by ACCE (American Council for Construction Education)",
        parent=uni_node,
        critical=True,
    )
    claim = "This program is accredited by ACCE (American Council for Construction Education)."
    await evaluator.verify(
        claim=claim,
        node=acce_node,
        sources=item.program_url,
        additional_instruction="Look for explicit mention of 'ACCE accredited' or ACCE accreditation details on the program page.",
        extra_prerequisites=[url_node]
    )

    # Program Type (critical)
    prog_node = evaluator.add_leaf(
        id=f"uni_{index}_program_type",
        desc="The program is a bachelor's degree in Construction Management, Construction Science, or Construction Science & Management",
        parent=uni_node,
        critical=True,
    )
    if item.program_name:
        claim = f"The page describes the bachelor's program '{item.program_name}', which is in Construction Management/Construction Science."
    else:
        claim = "This page describes a bachelor's program in Construction Management, Construction Science, or Construction Science & Management."
    await evaluator.verify(
        claim=claim,
        node=prog_node,
        sources=item.program_url,
        additional_instruction="Confirm it is a bachelor's degree. Accept synonyms like 'BS in Construction Management' or 'B.S. Construction Science'.",
        extra_prerequisites=[url_node]
    )

    # Transfer Acceptance (critical)
    transfer_node = evaluator.add_leaf(
        id=f"uni_{index}_transfer_acceptance",
        desc="The university accepts transfer students from community colleges",
        parent=uni_node,
        critical=True,
    )
    claim = "This university accepts transfer students from community colleges."
    await evaluator.verify(
        claim=claim,
        node=transfer_node,
        sources=item.program_url,
        additional_instruction="Look for 'transfer students', 'transfer from community colleges', '2-year colleges', or transfer admissions guidance. Evidence on the program page or clearly linked content is acceptable if visible in the page text.",
        extra_prerequisites=[url_node]
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
) -> Dict[str, Any]:
    """
    Evaluate the answer for the educational pathway research task using Mind2Web2.
    """
    # Initialize evaluator and root
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

    # Create the top-level rubric node
    pathway_node = evaluator.add_parallel(
        id="Educational_Pathway_Research",
        desc="Research task to identify community colleges offering OSHA 30-Hour Construction certification and Texas universities with ACCE-accredited Construction Management programs that accept transfers",
        parent=root,
        critical=False,
    )

    # Extract structured information in parallel
    colleges_task = evaluator.extract(
        prompt=prompt_extract_colleges(),
        template_class=CollegesExtraction,
        extraction_name="colleges_extraction",
    )
    universities_task = evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )
    colleges_extracted, universities_extracted = await asyncio.gather(colleges_task, universities_task)

    # Prepare items: limit to first 3 colleges and first 2 universities; pad if fewer
    cc_items: List[CollegeItem] = (colleges_extracted.colleges or [])[:3]
    while len(cc_items) < 3:
        cc_items.append(CollegeItem())

    uni_items: List[UniversityItem] = (universities_extracted.universities or [])[:2]
    while len(uni_items) < 2:
        uni_items.append(UniversityItem())

    # Build college verification subtrees
    for i, cc in enumerate(cc_items):
        await verify_community_college(evaluator, pathway_node, cc, i)

    # Build university verification subtrees
    for j, uni in enumerate(uni_items):
        await verify_texas_university(evaluator, pathway_node, uni, j)

    # Return the evaluator's summary
    return evaluator.get_summary()