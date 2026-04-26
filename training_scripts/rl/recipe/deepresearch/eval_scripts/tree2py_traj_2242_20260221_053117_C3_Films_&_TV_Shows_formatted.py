import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "documentary_identification_quincy"
TASK_DESCRIPTION = (
    "Identify a documentary film that meets all of the following criteria:\n"
    "1. Released on Netflix on September 21, 2018.\n"
    "2. Won the Grammy Award for Best Music Film at the 2019 Grammy Awards.\n"
    "3. About a music producer and composer born on March 14, 1933, in Chicago, Illinois.\n"
    "4. Subject received the Grammy Legend Award in 1992.\n"
    "5. Subject produced Michael Jackson's album 'Thriller' (1982), recognized as the best-selling album of all time.\n"
    "6. One co-director was born in Wollongong, Australia, and attended William Paterson University on a full scholarship.\n"
    "7. The same Australian-born co-director previously directed another documentary shortlisted for the Academy Award for Best Documentary Feature in 2015.\n"
    "8. That prior documentary (shortlisted in 2015) was about a jazz trumpeter born on December 14, 1920.\n"
    "9. One of the documentary's co-directors is the daughter of the subject.\n"
    "Provide the title of this documentary film."
)


# ----------------------------- Data Models ---------------------------------- #
class DocumentaryExtraction(BaseModel):
    film_title: Optional[str] = None
    subject_name: Optional[str] = None
    codirectors: List[str] = Field(default_factory=list)

    australian_codirector_name: Optional[str] = None
    prior_doc_title: Optional[str] = None
    prior_doc_trumpeter_name: Optional[str] = None

    general_film_urls: List[str] = Field(default_factory=list)

    sources_netflix_release: List[str] = Field(default_factory=list)
    sources_grammy_best_music_film_2019: List[str] = Field(default_factory=list)
    sources_subject_birth_profile: List[str] = Field(default_factory=list)
    sources_subject_legend_award_1992: List[str] = Field(default_factory=list)
    sources_subject_thriller_best_selling: List[str] = Field(default_factory=list)
    sources_codirector_wollongong_wpu: List[str] = Field(default_factory=list)
    sources_codirector_prior_doc_shortlisted_2015: List[str] = Field(default_factory=list)
    sources_prior_doc_jazz_trumpeter_born_1920_12_14: List[str] = Field(default_factory=list)
    sources_codirector_is_subjects_daughter: List[str] = Field(default_factory=list)


# --------------------------- Extraction Prompt ------------------------------ #
def prompt_extract_documentary_info() -> str:
    return (
        "Extract structured information about the documentary film identified in the answer and categorize any cited URLs to verify each criterion.\n"
        "Return a JSON object with the following fields:\n"
        "- film_title: The exact title of the documentary film provided in the answer.\n"
        "- subject_name: The name of the documentary's main subject, if mentioned.\n"
        "- codirectors: List the names of co-directors mentioned, if any.\n"
        "- australian_codirector_name: If the answer mentions which co-director was born in Wollongong, Australia, return their name; otherwise null.\n"
        "- prior_doc_title: The title of the Australian co-director's earlier documentary (shortlisted in 2015), if mentioned.\n"
        "- prior_doc_trumpeter_name: The name of the jazz trumpeter featured in that prior documentary, if mentioned.\n"
        "- general_film_urls: Any general URLs for the film (e.g., Netflix page, film website, IMDb, Wikipedia, press releases) presented in the answer.\n"
        "For each criterion, categorize URLs explicitly cited in the answer into the following lists. If the answer provides only general film URLs that plausibly support multiple criteria, include those URLs in multiple lists.\n"
        "- sources_netflix_release: URLs that support the Netflix release date being September 21, 2018.\n"
        "- sources_grammy_best_music_film_2019: URLs that support the film winning the Grammy Award for Best Music Film in 2019.\n"
        "- sources_subject_birth_profile: URLs that support the subject being a music producer and composer born on March 14, 1933 in Chicago, Illinois.\n"
        "- sources_subject_legend_award_1992: URLs that support the subject receiving the Grammy Legend Award in 1992.\n"
        "- sources_subject_thriller_best_selling: URLs that support the subject producing Michael Jackson's album 'Thriller' (1982) and that the album is recognized as the best-selling album of all time.\n"
        "- sources_codirector_wollongong_wpu: URLs that support that one co-director was born in Wollongong, Australia, and attended William Paterson University on a full scholarship.\n"
        "- sources_codirector_prior_doc_shortlisted_2015: URLs that support that the same Australian-born co-director previously directed a documentary shortlisted for the Academy Award for Best Documentary Feature in 2015.\n"
        "- sources_prior_doc_jazz_trumpeter_born_1920_12_14: URLs that support that the prior documentary (shortlisted in 2015) was about a jazz trumpeter born on December 14, 1920.\n"
        "- sources_codirector_is_subjects_daughter: URLs that support that one co-director is the daughter of the documentary's subject.\n"
        "Special rules:\n"
        "1) Only include URLs explicitly present in the answer. Do not invent URLs.\n"
        "2) If a single URL plausibly supports multiple criteria, include it in multiple lists.\n"
        "3) If the answer does not provide a URL for a criterion, return an empty list for that criterion.\n"
        "4) Accept URLs in plain text or markdown link format; extract the actual URL string.\n"
    )


# ---------------------------- Helper Utilities ------------------------------ #
def _title_phrase(extracted: DocumentaryExtraction) -> str:
    return f"titled '{extracted.film_title}'" if extracted.film_title else "the identified documentary film"

def _subject_phrase(extracted: DocumentaryExtraction) -> str:
    return extracted.subject_name if extracted.subject_name else "the documentary's subject"

def _australian_codirector_phrase(extracted: DocumentaryExtraction) -> str:
    return extracted.australian_codirector_name if extracted.australian_codirector_name else "the Australian-born co-director"

def _use_sources(primary: List[str], fallback: List[str]) -> List[str]:
    # Prefer primary sources; if empty, use fallback general film URLs
    if primary and len(primary) > 0:
        return primary
    return fallback or []


# -------------------------- Verification Subtrees --------------------------- #
async def verify_netflix_release(evaluator: Evaluator, parent, extracted: DocumentaryExtraction) -> None:
    node = evaluator.add_sequential(
        id="Netflix_Release",
        desc="Documentary was released on Netflix on September 21, 2018.",
        parent=parent,
        critical=True,
    )
    sources = _use_sources(extracted.sources_netflix_release, extracted.general_film_urls)
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id="Netflix_Release_sources_exist",
        desc="Sources provided for Netflix release date verification",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Netflix_Release_supported",
        desc="Claim supported by cited sources: Netflix release on September 21, 2018",
        parent=node,
        critical=True,
    )
    claim = f"The documentary {_title_phrase(extracted)} was released on Netflix on September 21, 2018."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm the Netflix release date is September 21, 2018. Accept minor date formatting variations."
    )


async def verify_grammy_best_music_film_2019(evaluator: Evaluator, parent, extracted: DocumentaryExtraction) -> None:
    node = evaluator.add_sequential(
        id="Grammy_Best_Music_Film_2019",
        desc="Documentary won the Grammy Award for Best Music Film at the 2019 Grammy Awards.",
        parent=parent,
        critical=True,
    )
    sources = _use_sources(extracted.sources_grammy_best_music_film_2019, extracted.general_film_urls)
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id="Grammy_Best_Music_Film_2019_sources_exist",
        desc="Sources provided for 2019 Grammy Best Music Film verification",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Grammy_Best_Music_Film_2019_supported",
        desc="Claim supported by cited sources: won Grammy Best Music Film (2019)",
        parent=node,
        critical=True,
    )
    claim = f"The documentary {_title_phrase(extracted)} won the Grammy Award for Best Music Film at the 2019 Grammy Awards."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Look for Grammy records, press releases, or reliable sources confirming Best Music Film winner at the 2019 Grammys."
    )


async def verify_subject_profile(evaluator: Evaluator, parent, extracted: DocumentaryExtraction) -> None:
    node = evaluator.add_sequential(
        id="Subject_Profile",
        desc="Documentary is about a music producer and composer born on March 14, 1933, in Chicago, Illinois.",
        parent=parent,
        critical=True,
    )
    sources = _use_sources(extracted.sources_subject_birth_profile, extracted.general_film_urls)
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id="Subject_Profile_sources_exist",
        desc="Sources provided for subject profile verification",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Subject_Profile_supported",
        desc="Claim supported by cited sources: subject is a music producer/composer born March 14, 1933 in Chicago",
        parent=node,
        critical=True,
    )
    subject = _subject_phrase(extracted)
    claim = f"The documentary is about {subject}, a music producer and composer who was born on March 14, 1933, in Chicago, Illinois."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Verify both profession (producer and composer) and birth details (March 14, 1933; Chicago, Illinois)."
    )


async def verify_subject_legend_award_1992(evaluator: Evaluator, parent, extracted: DocumentaryExtraction) -> None:
    node = evaluator.add_sequential(
        id="Subject_Grammy_Legend_1992",
        desc="Documentary's subject received the Grammy Legend Award in 1992.",
        parent=parent,
        critical=True,
    )
    sources = _use_sources(extracted.sources_subject_legend_award_1992, extracted.general_film_urls)
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id="Subject_Grammy_Legend_1992_sources_exist",
        desc="Sources provided for Grammy Legend Award (1992) verification",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Subject_Grammy_Legend_1992_supported",
        desc="Claim supported by cited sources: subject received Grammy Legend Award in 1992",
        parent=node,
        critical=True,
    )
    subject = _subject_phrase(extracted)
    claim = f"{subject} received the Grammy Legend Award in 1992."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm the subject's Grammy Legend Award year is 1992."
    )


async def verify_subject_thriller_best_selling(evaluator: Evaluator, parent, extracted: DocumentaryExtraction) -> None:
    node = evaluator.add_sequential(
        id="Subject_Produced_Thriller_Constraint",
        desc="Subject produced Michael Jackson's 'Thriller' (1982), recognized as the best-selling album of all time.",
        parent=parent,
        critical=True,
    )
    sources = _use_sources(extracted.sources_subject_thriller_best_selling, extracted.general_film_urls)
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id="Subject_Produced_Thriller_sources_exist",
        desc="Sources provided for Thriller production and best-selling status verification",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Subject_Produced_Thriller_supported",
        desc="Claim supported by cited sources: subject produced Thriller (1982), best-selling album of all time",
        parent=node,
        critical=True,
    )
    subject = _subject_phrase(extracted)
    claim = (
        f"{subject} produced Michael Jackson's album 'Thriller', which was released in 1982 and is recognized as the best-selling album of all time."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Verify both parts: production credit for 'Thriller' (1982) and recognition as best-selling album of all time; accept widely cited authoritative sources."
    )


async def verify_codirector_wollongong_wpu(evaluator: Evaluator, parent, extracted: DocumentaryExtraction) -> None:
    node = evaluator.add_sequential(
        id="CoDirector_Wollongong_And_WPU_Scholarship",
        desc="A co-director was born in Wollongong, Australia, and attended William Paterson University on a full scholarship.",
        parent=parent,
        critical=True,
    )
    sources = _use_sources(extracted.sources_codirector_wollongong_wpu, extracted.general_film_urls)
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id="CoDirector_Wollongong_WPU_sources_exist",
        desc="Sources provided for co-director Wollongong birth and WPU scholarship verification",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="CoDirector_Wollongong_WPU_supported",
        desc="Claim supported by cited sources: co-director born in Wollongong and attended WPU on full scholarship",
        parent=node,
        critical=True,
    )
    codir = _australian_codirector_phrase(extracted)
    claim = f"One of the documentary's co-directors, {codir}, was born in Wollongong, Australia, and attended William Paterson University on a full scholarship."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm both birthplace (Wollongong, Australia) and attendance at William Paterson University on a full scholarship."
    )


async def verify_codirector_prior_doc_shortlisted_2015(evaluator: Evaluator, parent, extracted: DocumentaryExtraction) -> None:
    node = evaluator.add_sequential(
        id="CoDirector_Prior_Doc_Shortlisted_2015",
        desc="Same Australian-born co-director previously directed a documentary shortlisted for the 2015 Oscar for Best Documentary Feature.",
        parent=parent,
        critical=True,
    )
    sources = _use_sources(extracted.sources_codirector_prior_doc_shortlisted_2015, extracted.general_film_urls)
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id="CoDirector_Prior_Doc_Shortlisted_2015_sources_exist",
        desc="Sources provided for co-director prior documentary shortlist (2015) verification",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="CoDirector_Prior_Doc_Shortlisted_2015_supported",
        desc="Claim supported by cited sources: prior documentary shortlisted for 2015 Best Documentary Feature",
        parent=node,
        critical=True,
    )
    codir = _australian_codirector_phrase(extracted)
    claim = f"The same Australian-born co-director, {codir}, previously directed another documentary that was shortlisted for the Academy Award for Best Documentary Feature in 2015."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Look for Oscar shortlist lists or reliable media confirming the co-director's previous documentary was shortlisted in 2015."
    )


async def verify_prior_doc_about_trumpeter_born_1920_12_14(evaluator: Evaluator, parent, extracted: DocumentaryExtraction) -> None:
    node = evaluator.add_sequential(
        id="Prior_Doc_About_Jazz_Trumpeter_Born_1920_12_14",
        desc="That prior documentary was about a jazz trumpeter born on December 14, 1920.",
        parent=parent,
        critical=True,
    )
    sources = _use_sources(extracted.sources_prior_doc_jazz_trumpeter_born_1920_12_14, extracted.general_film_urls)
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id="Prior_Doc_Trumpeter_sources_exist",
        desc="Sources provided for prior documentary trumpeter birth date verification",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Prior_Doc_Trumpeter_supported",
        desc="Claim supported by cited sources: prior documentary subject born December 14, 1920",
        parent=node,
        critical=True,
    )
    prior_doc = extracted.prior_doc_title if extracted.prior_doc_title else "the prior documentary"
    trumpeter = extracted.prior_doc_trumpeter_name if extracted.prior_doc_trumpeter_name else "the jazz trumpeter featured in the prior documentary"
    claim = f"{prior_doc} was about {trumpeter}, who was born on December 14, 1920."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm the prior documentary's subject is a jazz trumpeter with birth date December 14, 1920."
    )


async def verify_codirector_is_subjects_daughter(evaluator: Evaluator, parent, extracted: DocumentaryExtraction) -> None:
    node = evaluator.add_sequential(
        id="CoDirector_Is_Subjects_Daughter",
        desc="One of the documentary's co-directors is the daughter of the subject.",
        parent=parent,
        critical=True,
    )
    sources = _use_sources(extracted.sources_codirector_is_subjects_daughter, extracted.general_film_urls)
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id="CoDirector_Is_Subjects_Daughter_sources_exist",
        desc="Sources provided for co-director being the subject's daughter verification",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="CoDirector_Is_Subjects_Daughter_supported",
        desc="Claim supported by cited sources: a co-director is the subject's daughter",
        parent=node,
        critical=True,
    )
    subject = _subject_phrase(extracted)
    claim = f"One of the documentary's co-directors is the daughter of {subject}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Verify familial relationship: co-director is explicitly identified as the subject's daughter."
    )


# --------------------------- Main Evaluation -------------------------------- #
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_documentary_info(),
        template_class=DocumentaryExtraction,
        extraction_name="documentary_extraction",
    )

    # Optional ground-truth info for summary (not used in verification)
    evaluator.add_ground_truth({
        "expected_title": "Quincy",
        "expected_subject": "Quincy Jones",
        "expected_codirectors": ["Rashida Jones", "Alan Hicks"],
        "expected_prior_doc": "Keep On Keepin' On",
        "expected_trumpeter": "Clark Terry",
        "expected_trumpeter_birth": "December 14, 1920",
        "key_milestones": [
            "Netflix release on September 21, 2018",
            "Grammy Best Music Film (2019)",
            "Subject: music producer/composer born March 14, 1933 in Chicago",
            "Grammy Legend Award (1992)",
            "Produced 'Thriller' (1982), best-selling album of all time",
            "Co-director born in Wollongong; attended WPU on full scholarship",
            "Prior doc shortlisted for 2015 Oscar",
            "Prior doc about a trumpeter born Dec 14, 1920",
            "Co-director is subject's daughter"
        ]
    })

    doc_root = evaluator.add_parallel(
        id="Documentary_Film_Identification",
        desc="Provide the title of the documentary film that satisfies all stated conditions.",
        parent=root,
        critical=True,
    )

    # Build and verify each criterion subtree
    await verify_netflix_release(evaluator, doc_root, extracted)
    await verify_grammy_best_music_film_2019(evaluator, doc_root, extracted)
    await verify_subject_profile(evaluator, doc_root, extracted)
    await verify_subject_legend_award_1992(evaluator, doc_root, extracted)
    await verify_subject_thriller_best_selling(evaluator, doc_root, extracted)
    await verify_codirector_wollongong_wpu(evaluator, doc_root, extracted)
    await verify_codirector_prior_doc_shortlisted_2015(evaluator, doc_root, extracted)
    await verify_prior_doc_about_trumpeter_born_1920_12_14(evaluator, doc_root, extracted)
    await verify_codirector_is_subjects_daughter(evaluator, doc_root, extracted)

    return evaluator.get_summary()