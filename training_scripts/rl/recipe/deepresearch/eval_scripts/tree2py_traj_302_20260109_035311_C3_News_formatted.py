import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "cnn_journalist_identification"
TASK_DESCRIPTION = """
Identify the journalist who served as CNN bureau chief in Manila, Philippines from 1987 to 1995, then subsequently served as CNN bureau chief in Jakarta, Indonesia from 1995 to 2005, co-founded an online news platform in the Philippines in 2012, and received the Nobel Peace Prize in 2021 for efforts to safeguard freedom of expression as a precondition for democracy and lasting peace.
"""


class JournalistExtraction(BaseModel):
    name: Optional[str] = None

    manila_years: Optional[str] = None
    sources_manila: List[str] = Field(default_factory=list)

    jakarta_years: Optional[str] = None
    sources_jakarta: List[str] = Field(default_factory=list)

    platform_name: Optional[str] = None
    platform_year: Optional[str] = None
    sources_platform: List[str] = Field(default_factory=list)

    nobel_year: Optional[str] = None
    nobel_motivation: Optional[str] = None
    sources_nobel: List[str] = Field(default_factory=list)


def prompt_extract_journalist_info() -> str:
    return """
    Extract the identity and any cited sources from the answer relevant to the following checks. Return exactly the fields requested.

    Required fields:
    - name: The full name of the journalist identified in the answer.
    - manila_years: The years given in the answer for the Manila CNN bureau chief role (e.g., "1987–1995"), if explicitly stated; else null.
    - sources_manila: Array of URLs in the answer that support the Manila CNN bureau chief role (only extract actual URLs present).
    - jakarta_years: The years given in the answer for the Jakarta CNN bureau chief role (e.g., "1995–2005"), if explicitly stated; else null.
    - sources_jakarta: Array of URLs in the answer that support the Jakarta CNN bureau chief role (only extract actual URLs present).
    - platform_name: The name of the online news platform in the Philippines the journalist co-founded (e.g., "Rappler"), if mentioned.
    - platform_year: The year provided in the answer for that founding (e.g., "2012"), if mentioned; else null.
    - sources_platform: Array of URLs in the answer that support the online news platform founding (only extract actual URLs present).
    - nobel_year: The year provided in the answer for the Nobel Peace Prize (e.g., "2021"), if mentioned; else null.
    - nobel_motivation: The motivation/citation text or summary if the answer includes it; else null.
    - sources_nobel: Array of URLs in the answer that support the Nobel Peace Prize award and its motivation (only extract actual URLs present).

    URL extraction rules:
    - Extract only URLs explicitly present in the answer (plain URLs or within markdown links).
    - Do not invent or infer URLs.
    - Include full URLs; if a protocol is missing, prepend http://.

    If a field is not present in the answer, return null (or empty array for URL lists).
    """


async def build_verification_tree(evaluator: Evaluator, extracted: JournalistExtraction) -> None:
    root = evaluator.find_node("root")

    # Top-level critical node aggregating all criteria
    main_node = evaluator.add_parallel(
        id="Journalist_Identification",
        desc="Response identifies a specific journalist who meets all stated career and award constraints",
        parent=root,
        critical=True
    )

    # 1) Journalist identity provided
    identity_exists = bool(extracted.name and extracted.name.strip())
    evaluator.add_custom_node(
        result=identity_exists,
        id="Provides_Journalist_Identity",
        desc="Response provides the journalist's name/identity (a specific person)",
        parent=main_node,
        critical=True
    )

    # 2) CNN Bureau Chief career (sequential: Manila then Jakarta)
    career_node = evaluator.add_sequential(
        id="CNN_Bureau_Chief_Career",
        desc="Journalist served as CNN bureau chief in Manila (1987–1995) and then Jakarta (1995–2005) in that chronological order",
        parent=main_node,
        critical=True
    )

    # 2.a) Manila role
    manila_leaf = evaluator.add_leaf(
        id="Manila_Bureau_Chief_Role",
        desc="Journalist served as CNN bureau chief in Manila, Philippines from 1987 to 1995",
        parent=career_node,
        critical=True
    )
    manila_name = extracted.name or "the identified journalist"
    manila_claim = f"{manila_name} served as CNN bureau chief in Manila, Philippines from 1987 to 1995."
    await evaluator.verify(
        claim=manila_claim,
        node=manila_leaf,
        sources=extracted.sources_manila if extracted.sources_manila else None,
        additional_instruction=(
            "Verify that the page explicitly supports that this person was CNN's bureau chief (or equivalent phrasing: "
            "head of the CNN bureau, CNN bureau head, led CNN's bureau) in Manila, Philippines, with tenure covering "
            "1987 through 1995. Minor wording variations are acceptable, but both the role and the timeframe must be supported."
        )
    )

    # 2.b) Jakarta role (sequentially dependent on Manila)
    jakarta_leaf = evaluator.add_leaf(
        id="Jakarta_Bureau_Chief_Role",
        desc="After Manila, journalist served as CNN bureau chief in Jakarta, Indonesia from 1995 to 2005",
        parent=career_node,
        critical=True
    )
    jakarta_name = extracted.name or "the identified journalist"
    jakarta_claim = f"{jakarta_name} served as CNN bureau chief in Jakarta, Indonesia from 1995 to 2005."
    await evaluator.verify(
        claim=jakarta_claim,
        node=jakarta_leaf,
        sources=extracted.sources_jakarta if extracted.sources_jakarta else None,
        additional_instruction=(
            "Verify that the page explicitly supports that this person was CNN's bureau chief (or equivalent phrasing) "
            "in Jakarta, Indonesia, with tenure covering 1995 through 2005. Minor wording variations are acceptable, "
            "but both the role and the timeframe must be supported."
        )
    )

    # 3) Online news organization founding (2012, Philippines)
    platform_leaf = evaluator.add_leaf(
        id="Online_News_Organization_Founding",
        desc="Journalist co-founded an online news platform/organization in the Philippines in 2012",
        parent=main_node,
        critical=True
    )
    platform_name = extracted.platform_name or "an online news platform"
    platform_name_for_claim = platform_name if platform_name else "an online news platform"
    platform_name_snippet = f" '{platform_name}'" if extracted.platform_name else ""
    platform_claim = f"{(extracted.name or 'The identified journalist')} co-founded{platform_name_snippet} in the Philippines in 2012."
    await evaluator.verify(
        claim=platform_claim,
        node=platform_leaf,
        sources=extracted.sources_platform if extracted.sources_platform else None,
        additional_instruction=(
            "Verify that the page indicates the person co-founded an online news platform/organization in the Philippines in 2012. "
            "If the platform's name (e.g., Rappler) is present, it should match. Accept wording variants such as 'co-founded', "
            "'founding editor among co-founders', or 'co-established', but year 2012 and Philippine context must be supported."
        )
    )

    # 4) Nobel Peace Prize recognition (parallel: award year and motivation)
    nobel_node = evaluator.add_parallel(
        id="Nobel_Peace_Prize_Recognition",
        desc="Journalist received the Nobel Peace Prize in 2021 with the stated motivation",
        parent=main_node,
        critical=True
    )

    nobel_year_leaf = evaluator.add_leaf(
        id="Journalist_Received_Nobel_Peace_Prize_In_2021",
        desc="Journalist received (was awarded) the Nobel Peace Prize in 2021",
        parent=nobel_node,
        critical=True
    )
    nobel_name = extracted.name or "the identified journalist"
    nobel_year_claim = f"{nobel_name} received the Nobel Peace Prize in 2021."
    await evaluator.verify(
        claim=nobel_year_claim,
        node=nobel_year_leaf,
        sources=extracted.sources_nobel if extracted.sources_nobel else None,
        additional_instruction=(
            "Verify that the page states the person was awarded the Nobel Peace Prize in 2021. "
            "Shared awards are acceptable (e.g., co-laureates)."
        )
    )

    nobel_motivation_leaf = evaluator.add_leaf(
        id="Prize_Motivation_Verification",
        desc="The Nobel Peace Prize motivation/citation for the journalist specifically cites efforts to safeguard freedom of expression as a precondition for democracy and lasting peace",
        parent=nobel_node,
        critical=True
    )
    nobel_motivation_claim = (
        f"The 2021 Nobel Peace Prize citation for {nobel_name} states efforts to safeguard freedom of expression, "
        "which is a precondition for democracy and lasting peace."
    )
    await evaluator.verify(
        claim=nobel_motivation_claim,
        node=nobel_motivation_leaf,
        sources=extracted.sources_nobel if extracted.sources_nobel else None,
        additional_instruction=(
            "Verify the official wording or a faithful paraphrase of the Nobel Peace Prize motivation that explicitly mentions "
            "efforts to safeguard freedom of expression as a precondition for democracy and lasting peace."
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
    evaluator.initialize(
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
        prompt=prompt_extract_journalist_info(),
        template_class=JournalistExtraction,
        extraction_name="journalist_extraction"
    )

    evaluator.add_ground_truth({
        "hint_expected_identity": "Maria Ressa",
        "expected_roles": {
            "Manila": "CNN bureau chief, 1987–1995",
            "Jakarta": "CNN bureau chief, 1995–2005"
        },
        "expected_platform": {"name": "Rappler", "country": "Philippines", "year": "2012"},
        "expected_nobel": {"year": "2021", "motivation_keyphrase": "safeguard freedom of expression"}
    })

    await build_verification_tree(evaluator, extracted)

    return evaluator.get_summary()