import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "grammy_flomachine_museum_chain"
TASK_DESCRIPTION = (
    "A music producer won the Grammy Award for Producer of the Year, Non-Classical in 2023. "
    "One of the albums that contributed to this Grammy win was by the band Florence and the Machine and was released on May 13, 2022. "
    "This album was released by a record label that is owned by a parent company. The parent company's corporate headquarters are located in a city in the Netherlands. "
    "That city is in a province which has a capital city. In that capital city, there is a famous art museum named after a Dutch Golden Age painter, and the museum houses the world's largest collection of that painter's works. "
    "What is the name of this art museum?"
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ProducerExtraction(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AlbumExtraction(BaseModel):
    title: Optional[str] = None
    artist: Optional[str] = None
    release_date: Optional[str] = None
    label: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    grammy_contribution_sources: List[str] = Field(default_factory=list)


class LabelChainExtraction(BaseModel):
    label: Optional[str] = None
    parent_company: Optional[str] = None
    parent_hq_city: Optional[str] = None
    parent_hq_country: Optional[str] = None
    city_province: Optional[str] = None
    province_capital: Optional[str] = None
    sources_label: List[str] = Field(default_factory=list)
    sources_parent: List[str] = Field(default_factory=list)
    sources_hq: List[str] = Field(default_factory=list)
    sources_geo: List[str] = Field(default_factory=list)


class MuseumExtraction(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    painter: Optional[str] = None
    largest_collection_claim: Optional[str] = None  # free text claim as stated in the answer
    sources: List[str] = Field(default_factory=list)


class FinalAnswerExtraction(BaseModel):
    final_museum_name: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_producer() -> str:
    return """
    From the answer, extract:
    - name: the person the answer identifies as the winner of the 2023 Grammy Award for Producer of the Year, Non-Classical.
    - sources: all URLs explicitly cited to support that this person won that Grammy (accept Grammys.com, reputable news orgs, Wikipedia, etc. but only URLs actually present in the answer).

    Return exactly these fields. If missing, return null or an empty list where applicable.
    """


def prompt_extract_album() -> str:
    return """
    From the answer, extract the Florence + the Machine album that is tied to the producer's 2023 Grammy win. Provide:
    - title: the album title
    - artist: the band/artist name as written in the answer
    - release_date: the release date text as given in the answer
    - label: the record label stated for this album (as given in the answer)
    - sources: URLs explicitly cited for the album facts (title/artist/release date/label)
    - grammy_contribution_sources: URLs explicitly cited to support that this album contributed to the producer's 2023 Grammy win

    Only use URLs that actually appear in the answer. If any field is missing, set it to null or an empty list.
    """


def prompt_extract_label_chain() -> str:
    return """
    From the answer, extract the label → parent company → HQ city → province → provincial capital chain. Provide:
    - label: the album's label (as referenced in the chain)
    - parent_company: the owner/parent of that label
    - parent_hq_city: the stated corporate headquarters city of the parent company
    - parent_hq_country: the stated country of that HQ city
    - city_province: the province in which that HQ city lies
    - province_capital: the capital city of that province

    Also extract any URLs used in the answer that specifically support each piece of this chain:
    - sources_label: URLs supporting the album's label information (e.g., that the album was released by that label)
    - sources_parent: URLs supporting that the label is owned by the parent company
    - sources_hq: URLs supporting the parent company's HQ location (city, country)
    - sources_geo: URLs supporting the city→province relationship and the province's capital

    Use only URLs that actually appear in the answer. If a field or URL list is missing, set it to null or [] as appropriate.
    """


def prompt_extract_museum() -> str:
    return """
    From the answer, extract the final museum details in Haarlem:
    - name: the museum's name as stated
    - city: the city where the museum is located
    - painter: the Dutch Golden Age painter after whom the museum is named (if stated)
    - largest_collection_claim: the statement or short phrase (if any) that the museum houses the world's largest collection of that painter's works
    - sources: all URLs the answer cites for this museum and its properties

    Use only information present in the answer text and only URLs actually present in the answer. If missing, set fields to null or [].
    """


def prompt_extract_final_answer() -> str:
    return """
    Extract the final museum name that the answer presents as the ultimate answer to the question. 
    - final_museum_name: a single string that is the museum's name as the final answer.

    If the answer explicitly marks a final answer (e.g., 'Answer:' or the last bolded name), use that. 
    Otherwise, use the museum name that the answer most clearly concludes with. If missing, set to null.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*maybe_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for lst in maybe_lists:
        if not lst:
            continue
        for url in lst:
            if isinstance(url, str) and url.strip() and url not in seen:
                seen.add(url)
                ordered.append(url)
    return ordered


def _safe(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_producer(
    evaluator: Evaluator,
    parent_node,
    prod: ProducerExtraction,
) -> None:
    # Aggregator for producer identification (critical)
    producer_node = evaluator.add_parallel(
        id="producer_identification",
        desc="Identifies the music producer who won the Grammy Award for Producer of the Year, Non-Classical in 2023.",
        parent=parent_node,
        critical=True,
    )

    # Existence and sourcing check (critical)
    evaluator.add_custom_node(
        result=bool(prod.name and prod.name.strip()) and bool(prod.sources),
        id="producer_name_and_sources_provided",
        desc="Producer name and supporting source URLs are provided in the answer.",
        parent=producer_node,
        critical=True,
    )

    # Verify producer actually won the 2023 Grammy (critical)
    leaf = evaluator.add_leaf(
        id="producer_won_2023_grammy",
        desc="The identified producer won the 2023 Grammy Award for Producer of the Year, Non-Classical.",
        parent=producer_node,
        critical=True,
    )
    claim = (
        f"{_safe(prod.name)} won the Grammy Award for Producer of the Year, Non-Classical in 2023."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=prod.sources,
        additional_instruction="Verify this specific category and year. Accept stylistic variants like 'Producer of the Year (Non-Classical)'.",
    )


async def verify_album(
    evaluator: Evaluator,
    parent_node,
    album: AlbumExtraction,
    prod: ProducerExtraction,
) -> None:
    album_node = evaluator.add_parallel(
        id="album_identification",
        desc="Identifies the Florence + the Machine album that meets all stated constraints.",
        parent=parent_node,
        critical=True,
    )

    # 1) Album is by Florence + the Machine (critical)
    leaf_band = evaluator.add_leaf(
        id="album_is_by_florence_and_the_machine",
        desc="The identified album is by Florence and the Machine.",
        parent=album_node,
        critical=True,
    )
    claim_band = (
        f"The album '{_safe(album.title)}' is by Florence + the Machine (also stylized as 'Florence and the Machine')."
    )
    await evaluator.verify(
        claim=claim_band,
        node=leaf_band,
        sources=album.sources,
        additional_instruction="Allow stylistic variations: '+' vs 'and', case-insensitive matching. Verify using album sources.",
    )

    # 2) Release date is May 13, 2022 (critical)
    leaf_date = evaluator.add_leaf(
        id="album_release_date_may_13_2022",
        desc="The identified album was released on May 13, 2022.",
        parent=album_node,
        critical=True,
    )
    claim_date = f"The album '{_safe(album.title)}' was released on May 13, 2022."
    await evaluator.verify(
        claim=claim_date,
        node=leaf_date,
        sources=album.sources,
        additional_instruction="Verify the stated release date. Accept regional formatting like '13 May 2022'.",
    )

    # 3) Album contributed to the producer's 2023 Grammy win (critical)
    leaf_contrib = evaluator.add_leaf(
        id="album_is_cited_as_contributing_to_grammy_win",
        desc="The response provides verifiable support that this album contributed to the producer's 2023 Grammy win (e.g., credible citation or official Grammy-related documentation).",
        parent=album_node,
        critical=True,
    )
    contrib_sources = _combine_sources(album.grammy_contribution_sources, prod.sources, album.sources)
    claim_contrib = (
        f"The album '{_safe(album.title)}' by Florence + the Machine was among the credited works contributing to {_safe(prod.name)}'s 2023 Grammy Award for Producer of the Year, Non-Classical."
    )
    await evaluator.verify(
        claim=claim_contrib,
        node=leaf_contrib,
        sources=contrib_sources,
        additional_instruction="Look for Grammys.com pages or reputable sources explicitly listing credited works for the 2023 Producer of the Year (Non-Classical).",
    )


async def verify_label_chain(
    evaluator: Evaluator,
    parent_node,
    album: AlbumExtraction,
    chain: LabelChainExtraction,
) -> None:
    chain_node = evaluator.add_sequential(
        id="label_parent_hq_geo_chain",
        desc="Verifies the required label → parent company → HQ city → province → provincial capital chain using the explicit constraint values.",
        parent=parent_node,
        critical=True,
    )

    # Album released by Polydor Records
    leaf_label = evaluator.add_leaf(
        id="album_label_is_polydor_records",
        desc="The album was released by Polydor Records.",
        parent=chain_node,
        critical=True,
    )
    label_sources = _combine_sources(album.sources, chain.sources_label)
    claim_label = f"The album '{_safe(album.title)}' was released by Polydor Records."
    await evaluator.verify(
        claim=claim_label,
        node=leaf_label,
        sources=label_sources,
        additional_instruction="Accept if Polydor Records is one of the labels of release (e.g., region-specific releases).",
    )

    # Polydor is owned by Universal Music Group
    leaf_owner = evaluator.add_leaf(
        id="polydor_owned_by_umg",
        desc="Polydor Records is owned by Universal Music Group.",
        parent=chain_node,
        critical=True,
    )
    owner_sources = _combine_sources(chain.sources_parent)
    claim_owner = "Polydor Records is owned by Universal Music Group (UMG)."
    await evaluator.verify(
        claim=claim_owner,
        node=leaf_owner,
        sources=owner_sources,
        additional_instruction="Verify label ownership; accept reputable sources like company pages, Wikipedia, industry publications.",
    )

    # UMG HQ in Hilversum, Netherlands
    leaf_hq = evaluator.add_leaf(
        id="umg_hq_in_hilversum",
        desc="Universal Music Group's corporate headquarters are located in Hilversum, Netherlands.",
        parent=chain_node,
        critical=True,
    )
    hq_sources = _combine_sources(chain.sources_hq)
    claim_hq = "Universal Music Group's corporate headquarters are located in Hilversum, Netherlands."
    await evaluator.verify(
        claim=claim_hq,
        node=leaf_hq,
        sources=hq_sources,
        additional_instruction="Prefer Universal Music Group N.V. corporate information and reputable references. Distinguish corporate HQ in the Netherlands from U.S. operational offices.",
    )

    # Hilversum in North Holland
    leaf_city_province = evaluator.add_leaf(
        id="hilversum_in_north_holland",
        desc="Hilversum is located in North Holland province.",
        parent=chain_node,
        critical=True,
    )
    city_province_sources = _combine_sources(chain.sources_geo)
    claim_city_province = "Hilversum is located in the province of North Holland in the Netherlands."
    await evaluator.verify(
        claim=claim_city_province,
        node=leaf_city_province,
        sources=city_province_sources,
        additional_instruction="Use authoritative geographic sources (e.g., Wikipedia, official municipal/provincial pages).",
    )

    # North Holland capital is Haarlem
    leaf_capital = evaluator.add_leaf(
        id="north_holland_capital_is_haarlem",
        desc="The capital of North Holland province is Haarlem.",
        parent=chain_node,
        critical=True,
    )
    claim_capital = "The capital city of North Holland province is Haarlem."
    await evaluator.verify(
        claim=claim_capital,
        node=leaf_capital,
        sources=city_province_sources,
        additional_instruction="Verify provincial capital via authoritative sources.",
    )


async def verify_museum(
    evaluator: Evaluator,
    parent_node,
    museum: MuseumExtraction,
    final_ans: FinalAnswerExtraction,
) -> None:
    museum_node = evaluator.add_parallel(
        id="museum_answer",
        desc="Verifies the final museum in Haarlem satisfies the stated museum constraints and that the response outputs its name.",
        parent=parent_node,
        critical=True,
    )

    # Museum meets constraints in Haarlem
    leaf_constraints = evaluator.add_leaf(
        id="museum_in_haarlem_and_qualifies",
        desc="The named museum is located in Haarlem, is named after a Dutch Golden Age painter, and houses the world's largest collection of that painter's works.",
        parent=museum_node,
        critical=True,
    )
    claim_constraints = (
        f"The {_safe(museum.name)} is located in Haarlem; it is named after Dutch Golden Age painter {_safe(museum.painter)}; "
        "and the museum houses the world's largest collection of that painter's works."
    )
    await evaluator.verify(
        claim=claim_constraints,
        node=leaf_constraints,
        sources=museum.sources,
        additional_instruction="Allow synonymous phrasing like 'holds the largest collection'. Verify all three conditions: location in Haarlem, named after a Dutch Golden Age painter, and largest collection claim.",
    )

    # Final answer states the museum's name
    leaf_final = evaluator.add_leaf(
        id="final_answer_is_museum_name",
        desc="The response clearly states the museum’s name as the final answer.",
        parent=museum_node,
        critical=True,
    )
    claim_final = (
        f"The strings '{_safe(final_ans.final_museum_name)}' and '{_safe(museum.name)}' refer to the same museum."
    )
    await evaluator.verify(
        claim=claim_final,
        node=leaf_final,
        sources=None,
        additional_instruction="Perform a direct name-equivalence check allowing minor variations (punctuation, casing).",
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Grammy → Florence + the Machine album → NL geo chain → museum task.
    Returns the evaluation summary produced by the obj_task_eval Evaluator.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # create a non-critical root; add a critical sequential child as the true root
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

    # Create the critical sequential task-chain root under the global root
    task_chain = evaluator.add_sequential(
        id="task_chain",
        desc="Verify the full reasoning chain from producer → album → label/owner/HQ/city/province/capital → museum.",
        parent=root,
        critical=True,
    )

    # Parallelize extractions
    producer_extraction_task = evaluator.extract(
        prompt=prompt_extract_producer(),
        template_class=ProducerExtraction,
        extraction_name="producer_extraction",
    )
    album_extraction_task = evaluator.extract(
        prompt=prompt_extract_album(),
        template_class=AlbumExtraction,
        extraction_name="album_extraction",
    )
    label_chain_extraction_task = evaluator.extract(
        prompt=prompt_extract_label_chain(),
        template_class=LabelChainExtraction,
        extraction_name="label_chain_extraction",
    )
    museum_extraction_task = evaluator.extract(
        prompt=prompt_extract_museum(),
        template_class=MuseumExtraction,
        extraction_name="museum_extraction",
    )
    final_answer_extraction_task = evaluator.extract(
        prompt=prompt_extract_final_answer(),
        template_class=FinalAnswerExtraction,
        extraction_name="final_answer_extraction",
    )

    (
        producer_info,
        album_info,
        label_chain_info,
        museum_info,
        final_answer_info,
    ) = await asyncio.gather(
        producer_extraction_task,
        album_extraction_task,
        label_chain_extraction_task,
        museum_extraction_task,
        final_answer_extraction_task,
    )

    # Build and run verifications following the rubric tree
    await verify_producer(evaluator, task_chain, producer_info)
    await verify_album(evaluator, task_chain, album_info, producer_info)
    await verify_label_chain(evaluator, task_chain, album_info, label_chain_info)
    await verify_museum(evaluator, task_chain, museum_info, final_answer_info)

    return evaluator.get_summary()