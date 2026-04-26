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
TASK_ID = "publisher_identification_task"
TASK_DESCRIPTION = (
    "A British translator who graduated from the University of Cambridge began learning a specific East Asian "
    "language after completing their degree, starting sometime between 2009 and 2010. This translator went on to "
    "win the 2016 International Booker Prize for translating a novel from that language into English. The winning "
    "translation was published in English by Portobello Books, an independent publisher founded in 2005. The novel "
    "tells the story of a woman whose decision to stop eating meat leads to profound and unexpected consequences for "
    "her life and relationships. In 2015, this same translator founded a non-profit publishing house dedicated to "
    "publishing contemporary literature from Asia, with a particular mission to bring works into English that might "
    "not otherwise be translated. What is the name of the non-profit publisher founded by this translator?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PublisherIdentificationExtraction(BaseModel):
    # Translator info
    translator_name: Optional[str] = None
    translator_name_urls: List[str] = Field(default_factory=list)

    nationality: Optional[str] = None
    cambridge_grad_text: Optional[str] = None
    nationality_education_urls: List[str] = Field(default_factory=list)

    korean_learning_start_text: Optional[str] = None
    korean_learning_start_year: Optional[str] = None
    korean_learning_urls: List[str] = Field(default_factory=list)

    # Prize info
    prize_name: Optional[str] = None
    prize_year: Optional[str] = None
    prize_urls: List[str] = Field(default_factory=list)

    # Book info
    book_title: Optional[str] = None
    book_author: Optional[str] = None
    book_identification_urls: List[str] = Field(default_factory=list)

    book_original_language: Optional[str] = None
    book_subject_summary: Optional[str] = None
    original_work_urls: List[str] = Field(default_factory=list)

    # English publication details
    english_publisher: Optional[str] = None
    portobello_founded_year: Optional[str] = None
    english_publication_urls: List[str] = Field(default_factory=list)

    # Founded publisher details (final target)
    publisher_name: Optional[str] = None
    publisher_founded_year: Optional[str] = None
    founding_details_urls: List[str] = Field(default_factory=list)

    non_profit_urls: List[str] = Field(default_factory=list)
    focus_mission_urls: List[str] = Field(default_factory=list)
    final_publisher_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return """
    Extract structured information from the answer that matches the following fields. Return null for any field not present.
    Also extract URLs explicitly mentioned in the answer that support each claim.

    Fields to extract:
    - translator_name: The full name of the translator.
    - translator_name_urls: A list of URLs that confirm the translator's identity (as the translator of the prize-winning work).
    - nationality: The nationality of the translator (e.g., "British").
    - cambridge_grad_text: Any phrase or sentence that indicates the translator graduated from the University of Cambridge.
    - nationality_education_urls: URLs confirming British nationality and Cambridge education.

    - korean_learning_start_text: A phrase indicating when the translator began learning Korean (e.g., "between 2009 and 2010" or "in 2009/2010 after graduating").
    - korean_learning_start_year: The year or range (e.g., "2009", "2010", or "2009-2010").
    - korean_learning_urls: URLs confirming when the translator began learning Korean.

    - prize_name: The name of the prize (accept variants like "International Booker Prize" or "Man Booker International Prize").
    - prize_year: The year the translation won (expected "2016").
    - prize_urls: URLs confirming the 2016 prize win for this translation.

    - book_title: The title of the prize-winning original work (expected "The Vegetarian").
    - book_author: The author of the original work (expected "Han Kang").
    - book_identification_urls: URLs confirming the book title and author.

    - book_original_language: The original language of the work (expected "Korean").
    - book_subject_summary: A brief summary confirming the book is about a woman who stops eating meat leading to consequences.
    - original_work_urls: URLs confirming the original language and the subject matter.

    - english_publisher: The English-language publisher of the translation (expected "Portobello Books").
    - portobello_founded_year: The year Portobello Books was founded (expected "2005").
    - english_publication_urls: URLs confirming the English publisher and its founding year.

    - publisher_name: The name of the non-profit publishing house founded by the translator (expected "Tilted Axis Press").
    - publisher_founded_year: The year the publisher was founded (expected "2015").
    - founding_details_urls: URLs confirming the 2015 founding and the translator as the founder.

    - non_profit_urls: URLs confirming the publisher operates as a non-profit organization (accept "not-for-profit").
    - focus_mission_urls: URLs confirming the publisher's focus on contemporary Asian literature and mission to bring works into English that might not otherwise be translated.
    - final_publisher_urls: URLs that confirm the publisher name along with the above details.

    Important:
    - Only extract URLs explicitly present in the answer. If none are provided for a field, return an empty list.
    - Keep all fields as strings (even for years). Do not invent information.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


def _safe(s: Optional[str], fallback: str = "") -> str:
    return (s or fallback).strip()


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_prize_winning_translation(
    evaluator: Evaluator,
    root: Any,
    data: PublisherIdentificationExtraction,
) -> None:
    # Parent node: Prize_Winning_Translation_Verification (critical, parallel)
    prize_node = evaluator.add_parallel(
        id="Prize_Winning_Translation_Verification",
        desc="Verify the details of the translation that won the 2016 International Booker Prize",
        parent=root,
        critical=True,
    )

    # 1) Prize Year and Category
    prize_year_cat = evaluator.add_parallel(
        id="Prize_Year_And_Category",
        desc="Verify the prize year and category",
        parent=prize_node,
        critical=True,
    )

    won_leaf = evaluator.add_leaf(
        id="Won_2016_International_Booker",
        desc="The translation won the International Booker Prize in 2016",
        parent=prize_year_cat,
        critical=True,
    )
    # Build claim (robust to missing pieces)
    claim_won = (
        "The English translation of the work (commonly identified as 'The Vegetarian') won the International Booker Prize in 2016. "
        "Naming variants like 'Man Booker International Prize 2016' refer to the same award year."
    )
    await evaluator.verify(
        claim=claim_won,
        node=won_leaf,
        sources=data.prize_urls,
        additional_instruction=(
            "Treat 'International Booker Prize' and 'Man Booker International Prize' in 2016 as the same award. "
            "Confirm that the 2016 award recognized this translation, typically associated with Han Kang's 'The Vegetarian' "
            "and translator Deborah Smith."
        ),
    )

    # Prize Reference existence (critical)
    evaluator.add_custom_node(
        result=_has_urls(data.prize_urls),
        id="Prize_Reference",
        desc="Provide URL confirming the 2016 International Booker Prize win",
        parent=prize_year_cat,
        critical=True,
    )

    # 2) Original Work Details
    original_details = evaluator.add_parallel(
        id="Original_Work_Details",
        desc="Verify details about the original work",
        parent=prize_node,
        critical=True,
    )

    lang_leaf = evaluator.add_leaf(
        id="Original_Language_Korean",
        desc="The original work was written in Korean",
        parent=original_details,
        critical=True,
    )
    claim_lang = "The original work was written in Korean."
    await evaluator.verify(
        claim=claim_lang,
        node=lang_leaf,
        sources=data.original_work_urls,
        additional_instruction=(
            "Verify that the book (The Vegetarian by Han Kang) is originally a Korean-language work."
        ),
    )

    subject_leaf = evaluator.add_leaf(
        id="Book_Subject_Matter",
        desc="The book concerns a woman whose decision to stop eating meat has unexpected consequences",
        parent=original_details,
        critical=True,
    )
    claim_subject = (
        "The book tells the story of a woman who decides to stop eating meat, leading to profound and unexpected "
        "consequences for her life and relationships."
    )
    await evaluator.verify(
        claim=claim_subject,
        node=subject_leaf,
        sources=data.original_work_urls,
        additional_instruction="Confirm via synopsis/description that this is the central premise.",
    )

    evaluator.add_custom_node(
        result=_has_urls(data.original_work_urls),
        id="Original_Work_Reference",
        desc="Provide URL confirming the original language and subject matter",
        parent=original_details,
        critical=True,
    )

    # 3) English Publication Details
    english_pub = evaluator.add_parallel(
        id="English_Publication_Details",
        desc="Verify English publication details",
        parent=prize_node,
        critical=True,
    )

    pub_by_portobello = evaluator.add_leaf(
        id="Published_By_Portobello",
        desc="The English translation was published by Portobello Books",
        parent=english_pub,
        critical=True,
    )
    claim_portobello = "The English translation of the work was published by Portobello Books."
    await evaluator.verify(
        claim=claim_portobello,
        node=pub_by_portobello,
        sources=data.english_publication_urls,
        additional_instruction=(
            "Confirm from publisher pages or credible sources that Portobello Books published the English translation."
        ),
    )

    founded_2005 = evaluator.add_leaf(
        id="Portobello_Founded_2005",
        desc="Portobello Books is an independent publisher founded in 2005",
        parent=english_pub,
        critical=True,
    )
    claim_pb_2005 = "Portobello Books was founded in 2005 and operates as an independent publisher."
    await evaluator.verify(
        claim=claim_pb_2005,
        node=founded_2005,
        sources=data.english_publication_urls,
        additional_instruction=(
            "Confirm the founding year 2005 for Portobello Books using publisher history or reputable references."
        ),
    )

    evaluator.add_custom_node(
        result=_has_urls(data.english_publication_urls),
        id="English_Publication_Reference",
        desc="Provide URL confirming Portobello Books as publisher and its founding in 2005",
        parent=english_pub,
        critical=True,
    )

    # 4) Book Identification
    book_ident = evaluator.add_parallel(
        id="Book_Identification",
        desc="Provide the title and author of the prize-winning book",
        parent=prize_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(_safe(data.book_author)),
        id="Author_Name_Provided",
        desc="The author's name is provided",
        parent=book_ident,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(_safe(data.book_title)),
        id="Book_Title_Provided",
        desc="The book's title is provided",
        parent=book_ident,
        critical=True,
    )

    book_ref_leaf = evaluator.add_leaf(
        id="Book_Identification_Reference",
        desc="Provide URL confirming the book title and author",
        parent=book_ident,
        critical=True,
    )
    sources_for_book = data.book_identification_urls if _has_urls(data.book_identification_urls) else data.original_work_urls
    claim_book_ident = f"The prize-winning book is '{_safe(data.book_title, 'The Vegetarian')}' by {_safe(data.book_author, 'Han Kang')}."
    await evaluator.verify(
        claim=claim_book_ident,
        node=book_ref_leaf,
        sources=sources_for_book,
        additional_instruction="Confirm the exact pairing of title and author for the prize-winning work.",
    )


async def verify_translator_background(
    evaluator: Evaluator,
    root: Any,
    data: PublisherIdentificationExtraction,
) -> None:
    # Parent: Translator_Identification_And_Background
    translator_bg = evaluator.add_parallel(
        id="Translator_Identification_And_Background",
        desc="Identify and verify background information about the translator of the prize-winning work",
        parent=root,
        critical=True,
    )

    # 1) Nationality & Education
    nat_edu = evaluator.add_parallel(
        id="Translator_Nationality_And_Education",
        desc="Verify the translator's nationality and educational background",
        parent=translator_bg,
        critical=True,
    )

    british_leaf = evaluator.add_leaf(
        id="British_Nationality",
        desc="The translator is British",
        parent=nat_edu,
        critical=True,
    )
    claim_british = f"The translator {_safe(data.translator_name, 'the translator')} is British."
    await evaluator.verify(
        claim=claim_british,
        node=british_leaf,
        sources=data.nationality_education_urls,
        additional_instruction="Confirm nationality via reputable sources (biographies, interviews, publisher profiles).",
    )

    cam_leaf = evaluator.add_leaf(
        id="Cambridge_Graduate",
        desc="The translator graduated from the University of Cambridge",
        parent=nat_edu,
        critical=True,
    )
    claim_cam = f"The translator {_safe(data.translator_name, 'the translator')} graduated from the University of Cambridge."
    await evaluator.verify(
        claim=claim_cam,
        node=cam_leaf,
        sources=data.nationality_education_urls,
        additional_instruction="Accept synonyms such as 'Cambridge University' referring to the University of Cambridge.",
    )

    evaluator.add_custom_node(
        result=_has_urls(data.nationality_education_urls),
        id="Nationality_Education_Reference",
        desc="Provide URL confirming British nationality and Cambridge education",
        parent=nat_edu,
        critical=True,
    )

    # 2) Korean Learning History
    korean_hist = evaluator.add_parallel(
        id="Korean_Learning_History",
        desc="Verify when the translator began learning Korean",
        parent=translator_bg,
        critical=True,
    )

    timeline_leaf = evaluator.add_leaf(
        id="Korean_Learning_Timeline",
        desc="The translator began learning Korean after graduating from Cambridge, between 2009 and 2010",
        parent=korean_hist,
        critical=True,
    )
    claim_timeline = (
        f"The translator {_safe(data.translator_name, 'the translator')} began learning Korean after finishing their degree "
        "at Cambridge, sometime between 2009 and 2010."
    )
    await evaluator.verify(
        claim=claim_timeline,
        node=timeline_leaf,
        sources=data.korean_learning_urls,
        additional_instruction=(
            "Allow reasonable wording such as '2009', '2010', or 'between 2009 and 2010', and explicitly that this occurred after graduating from Cambridge."
        ),
    )

    evaluator.add_custom_node(
        result=_has_urls(data.korean_learning_urls),
        id="Learning_Timeline_Reference",
        desc="Provide URL confirming when the translator began learning Korean",
        parent=korean_hist,
        critical=True,
    )

    # 3) Translator Name Identification
    name_ident = evaluator.add_parallel(
        id="Translator_Name_Identification",
        desc="Provide the full name of the translator",
        parent=translator_bg,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(_safe(data.translator_name)),
        id="Translator_Name_Provided",
        desc="The translator's full name is provided",
        parent=name_ident,
        critical=True,
    )

    name_ref_leaf = evaluator.add_leaf(
        id="Translator_Name_Reference",
        desc="Provide URL confirming the translator's identity",
        parent=name_ident,
        critical=True,
    )
    claim_name = (
        f"The translator of the English version that won in 2016 is {_safe(data.translator_name, 'the translator')}."
    )
    await evaluator.verify(
        claim=claim_name,
        node=name_ref_leaf,
        sources=data.translator_name_urls,
        additional_instruction="Confirm that this person is credited as the translator of the prize-winning English translation.",
    )


async def verify_founded_publisher(
    evaluator: Evaluator,
    root: Any,
    data: PublisherIdentificationExtraction,
) -> None:
    # Parent: Founded_Publisher_Verification
    founded_pub = evaluator.add_parallel(
        id="Founded_Publisher_Verification",
        desc="Verify the details of the publishing house founded by the translator",
        parent=root,
        critical=True,
    )

    # 1) Founding Details
    founding = evaluator.add_parallel(
        id="Founding_Details",
        desc="Verify founding year and founder",
        parent=founded_pub,
        critical=True,
    )

    founded_2015_leaf = evaluator.add_leaf(
        id="Founded_In_2015",
        desc="The publisher was founded in 2015",
        parent=founding,
        critical=True,
    )
    claim_2015 = f"The publisher '{_safe(data.publisher_name, 'the publisher')}' was founded in 2015."
    await evaluator.verify(
        claim=claim_2015,
        node=founded_2015_leaf,
        sources=data.founding_details_urls,
        additional_instruction="Confirm the founding year as 2015 using official sources or reputable coverage.",
    )

    founder_ident_leaf = evaluator.add_leaf(
        id="Founded_By_Identified_Translator",
        desc="The publisher was founded by the translator identified in the previous step",
        parent=founding,
        critical=True,
    )
    claim_founder = f"The publisher '{_safe(data.publisher_name, 'the publisher')}' was founded by {_safe(data.translator_name, 'the translator')}."
    await evaluator.verify(
        claim=claim_founder,
        node=founder_ident_leaf,
        sources=data.founding_details_urls,
        additional_instruction="Confirm the founder matches the identified translator.",
    )

    evaluator.add_custom_node(
        result=_has_urls(data.founding_details_urls),
        id="Founding_Details_Reference",
        desc="Provide URL confirming the 2015 founding and the translator as founder",
        parent=founding,
        critical=True,
    )

    # 2) Organizational Structure
    org_struct = evaluator.add_parallel(
        id="Organizational_Structure",
        desc="Verify the publisher's organizational status",
        parent=founded_pub,
        critical=True,
    )

    nonprofit_leaf = evaluator.add_leaf(
        id="Non_Profit_Status",
        desc="The publisher operates as a non-profit organization",
        parent=org_struct,
        critical=True,
    )
    claim_nonprofit = f"The publisher '{_safe(data.publisher_name, 'the publisher')}' operates as a non-profit (not-for-profit) organization."
    await evaluator.verify(
        claim=claim_nonprofit,
        node=nonprofit_leaf,
        sources=data.non_profit_urls,
        additional_instruction="Accept 'non-profit' and 'not-for-profit' as equivalent status descriptions.",
    )

    evaluator.add_custom_node(
        result=_has_urls(data.non_profit_urls),
        id="Non_Profit_Reference",
        desc="Provide URL confirming non-profit status",
        parent=org_struct,
        critical=True,
    )

    # 3) Publishing Focus & Mission
    focus_mission = evaluator.add_parallel(
        id="Publishing_Focus_And_Mission",
        desc="Verify the publisher's focus and mission",
        parent=founded_pub,
        critical=True,
    )

    focus_leaf = evaluator.add_leaf(
        id="Focus_On_Asian_Literature",
        desc="The publisher specializes in contemporary Asian literature",
        parent=focus_mission,
        critical=True,
    )
    claim_focus = f"The publisher '{_safe(data.publisher_name, 'the publisher')}' specializes in contemporary Asian literature."
    await evaluator.verify(
        claim=claim_focus,
        node=focus_leaf,
        sources=data.focus_mission_urls,
        additional_instruction="Confirm specialization explicitly; accept wording like 'contemporary writing from Asia'.",
    )

    mission_leaf = evaluator.add_leaf(
        id="Translation_Mission",
        desc="The publisher's mission includes bringing works into English that might not otherwise be translated",
        parent=focus_mission,
        critical=True,
    )
    claim_mission = (
        f"The publisher '{_safe(data.publisher_name, 'the publisher')}' aims to bring works into English that might not otherwise be translated."
    )
    await evaluator.verify(
        claim=claim_mission,
        node=mission_leaf,
        sources=data.focus_mission_urls,
        additional_instruction="Confirm that the mission explicitly prioritizes translation into English of works unlikely to be translated otherwise.",
    )

    evaluator.add_custom_node(
        result=_has_urls(data.focus_mission_urls),
        id="Focus_Mission_Reference",
        desc="Provide URL confirming the Asian literature focus and translation mission",
        parent=focus_mission,
        critical=True,
    )


async def verify_final_publisher_name(
    evaluator: Evaluator,
    root: Any,
    data: PublisherIdentificationExtraction,
) -> None:
    # Parent: Publisher_Name_Final_Answer
    final_node = evaluator.add_parallel(
        id="Publisher_Name_Final_Answer",
        desc="Provide the name of the non-profit publisher",
        parent=root,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(_safe(data.publisher_name)),
        id="Publisher_Name_Provided",
        desc="The complete name of the non-profit publisher is provided",
        parent=final_node,
        critical=True,
    )

    final_ref_leaf = evaluator.add_leaf(
        id="Final_Verification_Reference",
        desc="Provide URL that confirms all details about the publisher match the specified criteria",
        parent=final_node,
        critical=True,
    )
    claim_final = (
        f"The non-profit publisher founded by {_safe(data.translator_name, 'the translator')} in 2015, "
        f"focused on contemporary Asian literature and bringing works into English that might not otherwise be translated, "
        f"is named '{_safe(data.publisher_name, 'Tilted Axis Press')}'."
    )
    await evaluator.verify(
        claim=claim_final,
        node=final_ref_leaf,
        sources=data.final_publisher_urls,
        additional_instruction=(
            "Confirm the publisher name and that it matches the described founding year (2015), non-profit status, "
            "focus on contemporary Asian literature, and mission to bring works into English that might not otherwise be translated."
        ),
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
) -> Dict:
    # Initialize evaluator with Sequential root
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
        default_model=model,
    )

    # Extract structured information from the answer
    extraction: PublisherIdentificationExtraction = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=PublisherIdentificationExtraction,
        extraction_name="publisher_identification_extraction",
    )

    # Build the verification tree according to rubric (in sequence)
    await verify_prize_winning_translation(evaluator, root, extraction)
    await verify_translator_background(evaluator, root, extraction)
    await verify_founded_publisher(evaluator, root, extraction)
    await verify_final_publisher_name(evaluator, root, extraction)

    # Optional: record a hint ground truth name (for reference; not used in scoring)
    evaluator.add_ground_truth({"expected_publisher_name": "Tilted Axis Press"})

    # Return summary
    return evaluator.get_summary()