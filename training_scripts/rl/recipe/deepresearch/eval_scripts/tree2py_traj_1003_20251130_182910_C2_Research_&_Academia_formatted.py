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
TASK_ID = "mars_tribo_purdue_labspace"
TASK_DESCRIPTION = (
    "A research paper titled 'Detection of triboelectric discharges during dust events on Mars' was published in "
    "Nature on November 26, 2025, with DOI 10.1038/s41586-025-09736-y. This paper presents evidence of atmospheric "
    "electrical activity on Mars detected by NASA's Perseverance rover. Identify the last author of this paper and "
    "determine their departmental affiliation at Purdue University. Then, calculate the total square footage dedicated "
    "to research laboratories in that department."
)

# Ground truth paper attributes to verify against sources
EXPECTED_TITLE = "Detection of triboelectric discharges during dust events on Mars"
EXPECTED_JOURNAL = "Nature"
EXPECTED_PUBLICATION_DATE = "November 26, 2025"
EXPECTED_DOI = "10.1038/s41586-025-09736-y"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PaperInfo(BaseModel):
    title: Optional[str] = None
    journal: Optional[str] = None
    publication_date: Optional[str] = None
    doi: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class LastAuthorInfo(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AffiliationInfo(BaseModel):
    university: Optional[str] = None
    department: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    department_urls: List[str] = Field(default_factory=list)


class DepartmentCalc(BaseModel):
    department_name: Optional[str] = None
    total_department_sqft: Optional[str] = None
    research_lab_fraction: Optional[str] = None  # Accept strings like "35%" or "0.35"
    calculated_lab_sqft: Optional[str] = None    # The final reported number in square feet
    sources: List[str] = Field(default_factory=list)


class MarsPaperAnswerExtraction(BaseModel):
    paper: Optional[PaperInfo] = None
    last_author: Optional[LastAuthorInfo] = None
    affiliation: Optional[AffiliationInfo] = None
    department_calc: Optional[DepartmentCalc] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return (
        "Extract the following structured information from the answer. Return null for any field that the answer does "
        "not explicitly provide. Do not invent information.\n\n"
        "1) paper: Details for the Nature paper in question.\n"
        "   - title: exact paper title as stated in the answer\n"
        "   - journal: journal name as stated (e.g., Nature)\n"
        "   - publication_date: publication date string as stated (e.g., 'November 26, 2025')\n"
        "   - doi: DOI string as stated (e.g., '10.1038/s41586-025-09736-y')\n"
        "   - urls: list of all URLs the answer cites for the paper (Nature page, DOI page, etc.)\n\n"
        "2) last_author: The last (final-listed) author for the paper.\n"
        "   - name: the last author's full name as stated in the answer\n"
        "   - sources: list of URLs the answer cites for supporting the author identification (paper page, DOI page, author page)\n\n"
        "3) affiliation: The last author's Purdue University affiliation.\n"
        "   - university: university name string as stated for the last author (should be 'Purdue University' if claimed)\n"
        "   - department: the Purdue department name as stated (e.g., 'Department of X' or 'School of Y')\n"
        "   - sources: list of URLs the answer cites to support the Purdue affiliation (paper affiliation list, Purdue profile, etc.)\n"
        "   - department_urls: list of URLs the answer cites that are specifically Purdue department pages, if any\n\n"
        "4) department_calc: The calculation of total research-lab square footage for that department.\n"
        "   - department_name: the department used for the calculation, as stated in the answer\n"
        "   - total_department_sqft: the total departmental square footage used in the calculation, as a string (keep commas or units if present in the answer)\n"
        "   - research_lab_fraction: the fraction of department space that is research labs, as stated (e.g., '35%' or '0.35')\n"
        "   - calculated_lab_sqft: the final computed research-lab square footage reported in the answer (in square feet), as a string\n"
        "   - sources: list of URLs the answer cites for the total departmental square footage and the research-lab fraction\n\n"
        "Return a JSON object with keys: paper, last_author, affiliation, department_calc, following the schema provided."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*sources_lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in sources_lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str):
                s = u.strip()
                if s and s not in merged:
                    merged.append(s)
    return merged


def _non_empty_str(s: Optional[str]) -> bool:
    return s is not None and isinstance(s, str) and s.strip() != ""


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_paper_identification(evaluator: Evaluator, parent, extracted: MarsPaperAnswerExtraction) -> None:
    node = evaluator.add_parallel(
        id="paper_identification",
        desc="Uses the correct target paper as specified in the prompt (matches the given title, journal, publication date, and DOI)",
        parent=parent,
        critical=True,
    )

    paper = extracted.paper or PaperInfo()
    paper_urls = paper.urls or []

    # Gate: at least one cited paper URL is present
    evaluator.add_custom_node(
        result=len(paper_urls) > 0,
        id="paper_urls_provided",
        desc="At least one cited URL for the paper is provided in the answer",
        parent=node,
        critical=True
    )

    # Title match
    title_leaf = evaluator.add_leaf(
        id="paper_title_match",
        desc=f"Paper title matches '{EXPECTED_TITLE}'",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The paper at the cited source(s) has the title '{EXPECTED_TITLE}'.",
        node=title_leaf,
        sources=paper_urls,
        additional_instruction="Treat minor case or punctuation differences as matches; ensure this is the Nature article with the exact same meaning/title."
    )

    # Journal match
    journal_leaf = evaluator.add_leaf(
        id="paper_journal_match",
        desc=f"Paper journal matches '{EXPECTED_JOURNAL}'",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The journal of the cited paper is '{EXPECTED_JOURNAL}'.",
        node=journal_leaf,
        sources=paper_urls,
        additional_instruction="Confirm that the article is published in Nature."
    )

    # Publication date match
    date_leaf = evaluator.add_leaf(
        id="paper_date_match",
        desc=f"Paper publication date matches '{EXPECTED_PUBLICATION_DATE}'",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publication date of the cited paper is {EXPECTED_PUBLICATION_DATE}.",
        node=date_leaf,
        sources=paper_urls,
        additional_instruction="Accept reasonable formatting variants like '26 November 2025'."
    )

    # DOI match
    doi_leaf = evaluator.add_leaf(
        id="paper_doi_match",
        desc=f"Paper DOI matches '{EXPECTED_DOI}'",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The DOI of the cited paper is {EXPECTED_DOI}.",
        node=doi_leaf,
        sources=paper_urls,
        additional_instruction="Confirm the DOI exactly; minor formatting like 'https://doi.org/DOI' is acceptable."
    )


async def verify_last_author(evaluator: Evaluator, parent, extracted: MarsPaperAnswerExtraction) -> None:
    node = evaluator.add_parallel(
        id="last_author",
        desc="Correctly identifies the last (final-listed) author of the specified paper",
        parent=parent,
        critical=True
    )

    paper = extracted.paper or PaperInfo()
    la = extracted.last_author or LastAuthorInfo()
    combined_sources = _merge_sources(paper.urls, la.sources)

    # Gate: name provided
    evaluator.add_custom_node(
        result=_non_empty_str(la.name),
        id="last_author_provided",
        desc="Last author name is provided in the answer",
        parent=node,
        critical=True
    )

    # Verify last author using paper sources
    last_author_leaf = evaluator.add_leaf(
        id="last_author_supported",
        desc="The last (final-listed) author is correctly identified",
        parent=node,
        critical=True
    )
    la_name = la.name or ""
    await evaluator.verify(
        claim=f"The last (final-listed) author of the paper is {la_name}.",
        node=last_author_leaf,
        sources=combined_sources,
        additional_instruction="Check the authors list on the paper page; verify that this person appears as the final name in the author list of the Nature article."
    )


async def verify_purdue_affiliation(evaluator: Evaluator, parent, extracted: MarsPaperAnswerExtraction) -> None:
    node = evaluator.add_parallel(
        id="purdue_departmental_affiliation",
        desc="Provides the last author’s departmental affiliation at Purdue University",
        parent=parent,
        critical=True
    )

    paper = extracted.paper or PaperInfo()
    la = extracted.last_author or LastAuthorInfo()
    aff = extracted.affiliation or AffiliationInfo()

    combined_sources = _merge_sources(aff.sources, aff.department_urls, la.sources, paper.urls)

    # Gate: at least one source for affiliation
    evaluator.add_custom_node(
        result=len(combined_sources) > 0,
        id="affiliation_sources_provided",
        desc="Cited sources for Purdue affiliation/department are provided",
        parent=node,
        critical=True
    )

    # Gate: department provided
    evaluator.add_custom_node(
        result=_non_empty_str(aff.department),
        id="department_provided",
        desc="Department name at Purdue is provided in the answer",
        parent=node,
        critical=True
    )

    # Affiliation includes Purdue University
    include_purdue_leaf = evaluator.add_leaf(
        id="affiliation_includes_purdue",
        desc="States that the last author is affiliated with Purdue University",
        parent=node,
        critical=True
    )
    la_name = la.name or "the last author"
    await evaluator.verify(
        claim=f"{la_name} is affiliated with Purdue University.",
        node=include_purdue_leaf,
        sources=combined_sources,
        additional_instruction="Look for affiliations on the article page or Purdue pages; accept 'Purdue University', 'Purdue U.', or 'Purdue University West Lafayette' as indicating Purdue."
    )

    # Department named
    dept_leaf = evaluator.add_leaf(
        id="department_named",
        desc="Names the Purdue University department the last author is affiliated with",
        parent=node,
        critical=True
    )
    dept_name = aff.department or ""
    await evaluator.verify(
        claim=f"The Purdue University department for {la_name} is '{dept_name}'.",
        node=dept_leaf,
        sources=combined_sources,
        additional_instruction="Accept reasonable department naming variants (e.g., 'School of X' vs 'Department of X') as equivalent if they refer to the same Purdue academic unit."
    )


async def verify_research_sqft(evaluator: Evaluator, parent, extracted: MarsPaperAnswerExtraction) -> None:
    node = evaluator.add_parallel(
        id="research_lab_square_footage",
        desc="Calculates the total square footage dedicated to research laboratories in that department, consistent with the department’s total space and research-lab share (research-lab sqft = total departmental sqft × research-lab fraction) and reports the result in square feet",
        parent=parent,
        critical=True
    )

    aff = extracted.affiliation or AffiliationInfo()
    calc = extracted.department_calc or DepartmentCalc()

    dept_name = calc.department_name or (aff.department if _non_empty_str(aff.department) else "the department")
    total_sqft = calc.total_department_sqft or ""
    fraction = calc.research_lab_fraction or ""
    result_sqft = calc.calculated_lab_sqft or ""
    calc_sources = calc.sources or []

    # Gates: inputs and sources provided
    evaluator.add_custom_node(
        result=_non_empty_str(total_sqft) and _non_empty_str(fraction) and _non_empty_str(result_sqft),
        id="calc_inputs_provided",
        desc="Total departmental sqft, research-lab fraction, and calculated research-lab sqft are all provided",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(calc_sources) > 0,
        id="calc_sources_provided",
        desc="Cited sources for total sqft and research-lab fraction are provided",
        parent=node,
        critical=True
    )

    # Inputs supported by sources
    inputs_supported_leaf = evaluator.add_leaf(
        id="calc_supported_by_sources",
        desc="The total sqft and research-lab fraction are supported by the cited sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"According to the cited sources, the total departmental space for '{dept_name}' at Purdue University is {total_sqft} (in square feet) and the research-lab fraction is {fraction}.",
        node=inputs_supported_leaf,
        sources=calc_sources,
        additional_instruction="Confirm both figures appear on the cited pages; allow minor formatting differences like commas in numbers or percent signs."
    )

    # Math correctness (logic check using the answer as context)
    calc_math_leaf = evaluator.add_leaf(
        id="calc_math_correct",
        desc="The reported research-lab square footage equals total departmental sqft × research-lab fraction",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Given total departmental space {total_sqft} square feet and research-lab fraction {fraction}, the calculated research-lab square footage of {result_sqft} square feet is correct (i.e., equals total × fraction, allowing reasonable rounding).",
        node=calc_math_leaf,
        additional_instruction="Interpret values like '35%' as 0.35. Treat commas in numbers as formatting. Accept small rounding differences."
    )

    # Unit correctness (square feet)
    unit_leaf = evaluator.add_leaf(
        id="unit_is_sqft",
        desc="The final reported result is expressed in square feet",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The final reported research-lab total is expressed in square feet (sq ft).",
        node=unit_leaf,
        additional_instruction="Check the answer text; if units are implicit but sources are clearly in square feet, consider it acceptable."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for:
    - Identifying the last author of the specified Nature paper
    - Providing their Purdue departmental affiliation
    - Calculating the total research-lab square footage for that department
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Record ground truth for the target paper
    evaluator.add_ground_truth(
        {
            "expected_title": EXPECTED_TITLE,
            "expected_journal": EXPECTED_JOURNAL,
            "expected_publication_date": EXPECTED_PUBLICATION_DATE,
            "expected_doi": EXPECTED_DOI
        },
        gt_type="paper_ground_truth"
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=MarsPaperAnswerExtraction,
        extraction_name="extracted_info"
    )

    # Build and verify tree according to rubric steps (sequential at root)
    await verify_paper_identification(evaluator, root, extracted)
    await verify_last_author(evaluator, root, extracted)
    await verify_purdue_affiliation(evaluator, root, extracted)
    await verify_research_sqft(evaluator, root, extracted)

    return evaluator.get_summary()