import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "stanford_genz_empathy_whr_2025"
TASK_DESCRIPTION = (
    "A researcher at Stanford University published research in March 2025 discussing why social connection is "
    "particularly difficult for Generation Z. This same researcher directs a laboratory at Stanford and co-authored "
    "a longitudinal study published in October 2025 that introduced the concept of an 'empathy perception gap' as a "
    "barrier to social connection among young adults. The researcher also contributed a chapter to the World Happiness "
    "Report 2025, co-authored with Rui Pei, focusing on how social connections improve the happiness of young adults. "
    "Identify this researcher and provide the following information: (1) The researcher's full name, (2) The exact name "
    "of the laboratory they direct at Stanford University, (3) Rui Pei's institutional affiliation as listed in the "
    "World Happiness Report 2025 chapter. For each piece of information, provide supporting reference URLs from credible sources."
)


# ----------------------------- Data Models --------------------------------- #
class RequestedOutputs(BaseModel):
    researcher_full_name: Optional[str] = None
    researcher_name_urls: List[str] = Field(default_factory=list)

    lab_exact_name: Optional[str] = None
    lab_name_urls: List[str] = Field(default_factory=list)
    lab_directorship_urls: List[str] = Field(default_factory=list)

    rui_pei_affiliation: Optional[str] = None
    rui_pei_whr_urls: List[str] = Field(default_factory=list)


class ConstraintSources(BaseModel):
    stanford_affiliation_2025_urls: List[str] = Field(default_factory=list)

    march_2025_gz_urls: List[str] = Field(default_factory=list)

    oct_2025_study_urls: List[str] = Field(default_factory=list)

    stanford_official_channel_2025_urls: List[str] = Field(default_factory=list)

    whr_2025_chapter_urls: List[str] = Field(default_factory=list)


# --------------------------- Extraction Prompts ---------------------------- #
def prompt_extract_requested_outputs() -> str:
    return (
        "Extract the three requested outputs and the supporting reference URLs explicitly present in the answer.\n"
        "Required outputs:\n"
        "1) researcher_full_name: The full name of the Stanford-affiliated researcher described.\n"
        "   researcher_name_urls: A list of credible source URLs that confirm the person's identity (e.g., official university pages, official report/publisher pages, peer‑reviewed journal sites, DOI/PubMed records). Extract only URLs explicitly present in the answer.\n"
        "2) lab_exact_name: The exact name of the laboratory/research group the researcher directs at Stanford University.\n"
        "   lab_name_urls: A list of credible source URLs that support the exact lab name (prefer official Stanford/lab pages or equivalent authoritative institutional pages).\n"
        "   lab_directorship_urls: A list of credible source URLs that explicitly state the researcher directs/leads that lab at Stanford.\n"
        "3) rui_pei_affiliation: Rui Pei's institutional affiliation as listed in the World Happiness Report 2025 chapter.\n"
        "   rui_pei_whr_urls: A list of credible source URLs that host the WHR 2025 chapter or official publisher page showing Rui Pei's affiliation.\n\n"
        "GENERAL RULES:\n"
        "- Extract only information explicitly present in the answer text. Do not invent or infer.\n"
        "- For each URL field, extract valid fully-qualified URLs found in the answer (plain or markdown). If a field is missing, return null or an empty list accordingly.\n"
        "- Do not include duplicate URLs.\n"
    )


def prompt_extract_constraint_sources() -> str:
    return (
        "Extract URLs explicitly present in the answer that support the scenario constraints. Organize them into fields:\n"
        "1) stanford_affiliation_2025_urls: URLs supporting that the identified researcher is affiliated with Stanford University as of 2025.\n"
        "2) march_2025_gz_urls: URLs for a March 2025 publication by/featuring the researcher that discusses why social connection is particularly difficult for Generation Z.\n"
        "3) oct_2025_study_urls: URLs for the October 2025 longitudinal study co-authored by the researcher that introduced an 'empathy perception gap' as a barrier to social connection among young adults. Use the same list for timing, longitudinal nature, and concept verification.\n"
        "4) stanford_official_channel_2025_urls: URLs showing that in 2025 Stanford University's official news/report channels featured the researcher's work.\n"
        "5) whr_2025_chapter_urls: URLs confirming the researcher contributed a chapter to the World Happiness Report 2025 (prefer official WHR site or official publisher host pages).\n\n"
        "GENERAL RULES:\n"
        "- Extract only URLs that are explicitly present in the answer text. Do not invent or infer.\n"
        "- Return empty lists if a category is not mentioned.\n"
        "- Use fully-qualified URLs. Remove duplicates.\n"
    )


# ------------------------ Verification Subroutines ------------------------- #
async def build_requested_outputs_subtree(
    evaluator: Evaluator,
    parent_node,
    outputs: RequestedOutputs,
) -> None:
    requested_node = evaluator.add_parallel(
        id="Requested_Outputs_With_Citations",
        desc="Provide the three requested outputs; each output must include at least one supporting reference URL from a credible source.",
        parent=parent_node,
        critical=True,
    )

    # Researcher Full Name
    name_node = evaluator.add_parallel(
        id="Researcher_Full_Name",
        desc="Provide the researcher's full name with credible supporting citation(s).",
        parent=requested_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(outputs.researcher_full_name and outputs.researcher_full_name.strip()),
        id="Name_Content",
        desc="The researcher's full name is provided.",
        parent=name_node,
        critical=True,
    )
    name_ref_leaf = evaluator.add_leaf(
        id="Name_Credible_Reference_URL",
        desc="At least one supporting reference URL from a credible source type is provided for the researcher's identity.",
        parent=name_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The reference page(s) confirm the identity of the researcher named '{outputs.researcher_full_name}'.",
        node=name_ref_leaf,
        sources=outputs.researcher_name_urls,
        additional_instruction=(
            "Verify that at least one URL is from a credible source (e.g., official university domain like stanford.edu, "
            "official report/publisher site, major peer‑reviewed journal site, DOI/PubMed record) and that the page confirms "
            "the person's full name. Allow minor variations in name formatting (middle initials, casing)."
        ),
    )

    # Lab Exact Name
    lab_node = evaluator.add_parallel(
        id="Lab_Exact_Name",
        desc="Provide the exact name of the Stanford lab/research group the researcher directs, with credible supporting citation(s).",
        parent=requested_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(outputs.lab_exact_name and outputs.lab_exact_name.strip()),
        id="Lab_Name_Content",
        desc="The exact lab/research group name is provided.",
        parent=lab_node,
        critical=True,
    )
    lab_name_leaf = evaluator.add_leaf(
        id="Lab_Name_Credible_Reference_URL",
        desc="At least one credible reference URL is provided supporting the lab/research group name.",
        parent=lab_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The exact laboratory/research group name is '{outputs.lab_exact_name}'.",
        node=lab_name_leaf,
        sources=outputs.lab_name_urls,
        additional_instruction=(
            "Confirm that the page explicitly shows the lab's official name as provided. Prefer official Stanford/lab pages "
            "or equivalent authoritative institutional pages. Allow minor punctuation/casing variations but not synonyms."
        ),
    )
    lab_directorship_leaf = evaluator.add_leaf(
        id="Lab_Directorship_Credible_Reference_URL",
        desc="At least one credible reference URL is provided supporting that the researcher directs/leads the lab at Stanford.",
        parent=lab_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page states that '{outputs.researcher_full_name}' directs/leads the '{outputs.lab_exact_name}' at Stanford University.",
        node=lab_directorship_leaf,
        sources=outputs.lab_directorship_urls,
        additional_instruction=(
            "Verify explicit leadership/directorship language (e.g., 'director', 'leads', 'heads') and ensure the page is a credible source, "
            "preferably an official Stanford domain or lab page."
        ),
    )

    # Rui Pei Affiliation
    rui_node = evaluator.add_parallel(
        id="Rui_Pei_Institutional_Affiliation",
        desc="Provide Rui Pei's institutional affiliation as listed in the World Happiness Report 2025 chapter, with credible supporting citation(s).",
        parent=requested_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(outputs.rui_pei_affiliation and outputs.rui_pei_affiliation.strip()),
        id="Rui_Pei_Affiliation_Content",
        desc="Rui Pei's institutional affiliation (as listed in the WHR 2025 chapter) is provided.",
        parent=rui_node,
        critical=True,
    )
    rui_aff_leaf = evaluator.add_leaf(
        id="Rui_Pei_Affiliation_Credible_WHR_URL",
        desc="At least one credible reference URL contains/hosts the World Happiness Report 2025 chapter where Rui Pei's affiliation is listed.",
        parent=rui_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"On the World Happiness Report 2025 chapter page, Rui Pei's affiliation is listed as '{outputs.rui_pei_affiliation}'.",
        node=rui_aff_leaf,
        sources=outputs.rui_pei_whr_urls,
        additional_instruction=(
            "Use official WHR site or official publisher/host. Confirm that the chapter visibly lists Rui Pei's affiliation exactly or "
            "with minor formatting differences."
        ),
    )


async def build_constraints_subtree(
    evaluator: Evaluator,
    parent_node,
    outputs: RequestedOutputs,
    constraints: ConstraintSources,
) -> None:
    constraints_node = evaluator.add_parallel(
        id="Constraint_Satisfaction",
        desc="The identified researcher satisfies the explicit scenario constraints in the prompt.",
        parent=parent_node,
        critical=True,
    )

    # Stanford affiliation as of 2025
    stanford_aff_leaf = evaluator.add_leaf(
        id="Stanford_Affiliation_2025",
        desc="The researcher is affiliated with Stanford University as of 2025.",
        parent=constraints_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"As of 2025, '{outputs.researcher_full_name}' is affiliated with Stanford University.",
        node=stanford_aff_leaf,
        sources=constraints.stanford_affiliation_2025_urls,
        additional_instruction=(
            "Confirm affiliation on a credible page (preferably stanford.edu) with evidence tied to 2025 (page date, publication date, "
            "or context clearly indicating 2025)."
        ),
    )

    # March 2025 Gen Z social connection difficulty
    march_leaf = evaluator.add_leaf(
        id="March_2025_GenZ_Social_Connection_Difficulty",
        desc="There is a March 2025 publication by/featuring the researcher discussing Gen Z social connection difficulty.",
        parent=constraints_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"There is a March 2025 publication by or featuring '{outputs.researcher_full_name}' that discusses why social connection "
            "is particularly difficult for Generation Z."
        ),
        node=march_leaf,
        sources=constraints.march_2025_gz_urls,
        additional_instruction=(
            "Verify both the March 2025 timing and the Gen Z social connection difficulty topic. The page should clearly tie the publication "
            "to the researcher (authored by or prominently featuring). Prefer credible/official channels."
        ),
    )

    # October 2025 longitudinal study introducing 'empathy perception gap'
    oct_node = evaluator.add_parallel(
        id="October_2025_Longitudinal_Empathy_Perception_Gap",
        desc="The researcher co-authored an October 2025 longitudinal study introducing an 'empathy perception gap' barrier.",
        parent=constraints_node,
        critical=True,
    )

    oct_timing_leaf = evaluator.add_leaf(
        id="October_2025_Timing",
        desc="The relevant publication is identified as being from October 2025.",
        parent=oct_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The publication date is October 2025.",
        node=oct_timing_leaf,
        sources=constraints.oct_2025_study_urls,
        additional_instruction="Confirm that the page explicitly shows October 2025 as the publication date.",
    )

    longitudinal_leaf = evaluator.add_leaf(
        id="Longitudinal_Study",
        desc="The October 2025 publication is described as a longitudinal study.",
        parent=oct_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The publication is a longitudinal study.",
        node=longitudinal_leaf,
        sources=constraints.oct_2025_study_urls,
        additional_instruction=(
            "Confirm that the page describes the methodological nature as 'longitudinal study' or equivalent terminology."
        ),
    )

    empathy_gap_leaf = evaluator.add_leaf(
        id="Empathy_Perception_Gap_Introduced",
        desc="The October 2025 publication introduced an 'empathy perception gap' as a barrier to social connection among young adults.",
        parent=oct_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The publication introduces the concept of an 'empathy perception gap' as a barrier to social connection among young adults.",
        node=empathy_gap_leaf,
        sources=constraints.oct_2025_study_urls,
        additional_instruction=(
            "Confirm explicit mention of 'empathy perception gap' being introduced (or first articulated) in the study and its role as a barrier "
            "to young adults' social connection. Also ensure the named researcher is a co-author."
        ),
    )

    # Stanford official channel feature in 2025
    official_leaf = evaluator.add_leaf(
        id="Stanford_Official_Channel_Feature_2025",
        desc="The researcher's work was featured in Stanford University official news/report channels in 2025.",
        parent=constraints_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"In 2025, Stanford University's official channels featured work by '{outputs.researcher_full_name}'.",
        node=official_leaf,
        sources=constraints.stanford_official_channel_2025_urls,
        additional_instruction=(
            "Confirm that the URL is an official Stanford domain (e.g., news.stanford.edu, stanford.edu) and the feature occurred in 2025."
        ),
    )

    # World Happiness Report 2025 chapter contribution details
    whr_node = evaluator.add_parallel(
        id="World_Happiness_Report_2025_Chapter",
        desc="The researcher contributed a WHR 2025 chapter meeting the co-author/topic constraints.",
        parent=constraints_node,
        critical=True,
    )

    whr_contrib_leaf = evaluator.add_leaf(
        id="WHR_2025_Contribution",
        desc="The researcher contributed a chapter to the World Happiness Report 2025.",
        parent=whr_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{outputs.researcher_full_name}' contributed a chapter to the World Happiness Report 2025.",
        node=whr_contrib_leaf,
        sources=constraints.whr_2025_chapter_urls,
        additional_instruction=(
            "Confirm that the page lists the chapter among WHR 2025 contributions authored or co-authored by the researcher."
        ),
    )

    whr_focus_leaf = evaluator.add_leaf(
        id="WHR_Chapter_Focus",
        desc="The chapter focuses on social connections and young adults (how social connections improve young adults' happiness).",
        parent=whr_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The chapter focuses on social connections and young adults, specifically how social connections improve young adults' happiness.",
        node=whr_focus_leaf,
        sources=constraints.whr_2025_chapter_urls,
        additional_instruction=(
            "Confirm explicit topical focus aligning with social connections and young adults' happiness."
        ),
    )

    whr_coauthor_leaf = evaluator.add_leaf(
        id="WHR_Coauthor_Rui_Pei",
        desc="The chapter is co-authored with Rui Pei.",
        parent=whr_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The chapter is co-authored by Rui Pei.",
        node=whr_coauthor_leaf,
        sources=constraints.whr_2025_chapter_urls,
        additional_instruction=(
            "Confirm that Rui Pei is listed as co-author for this WHR 2025 chapter."
        ),
    )


# ----------------------------- Main Evaluator ------------------------------ #
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

    # Extract requested outputs and constraint sources in parallel
    outputs_task = evaluator.extract(
        prompt=prompt_extract_requested_outputs(),
        template_class=RequestedOutputs,
        extraction_name="requested_outputs",
    )
    constraints_task = evaluator.extract(
        prompt=prompt_extract_constraint_sources(),
        template_class=ConstraintSources,
        extraction_name="constraint_sources",
    )
    outputs, constraints = await asyncio.gather(outputs_task, constraints_task)

    # Build critical top-level node representing the rubric root
    ri_root = evaluator.add_parallel(
        id="Researcher_Identification",
        desc="Identify the correct Stanford-affiliated researcher described and provide requested fields with citations, satisfying constraints.",
        parent=root,
        critical=True,
    )

    # Requested outputs subtree
    await build_requested_outputs_subtree(evaluator, ri_root, outputs)

    # Constraints subtree
    await build_constraints_subtree(evaluator, ri_root, outputs, constraints)

    return evaluator.get_summary()