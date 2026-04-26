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
TASK_ID = "nba_fiction_2024_winner_info"
TASK_DESCRIPTION = """
I'm researching the 2024 National Book Award for Fiction winner. Please provide me with the following information:

1. What is the title of the novel that won the 2024 National Book Award for Fiction?
2. Who published this novel, and in what year was it published?
3. This novel also won another major prestigious literary award in 2025. Which award was it?
4. Where was the author of this winning novel born (include both city and state)?
5. At which university did this author complete their undergraduate degree?
6. The author had another novel that was shortlisted for the Booker Prize in 2022. What was the title of that novel?
7. Please provide the name and book title of at least one other author who was a finalist (but did not win) in the 2024 National Book Award for Fiction category.

Please include reference URLs for all information provided.
""".strip()


# --------------------------------------------------------------------------- #
# Data model for extraction                                                   #
# --------------------------------------------------------------------------- #
class NBAFiction2024InfoExtraction(BaseModel):
    # Core winner/author facts
    winning_novel_title: Optional[str] = None
    publisher: Optional[str] = None
    publication_year: Optional[str] = None
    award_2025: Optional[str] = None  # Name of the other major award (won in 2025)
    author_name: Optional[str] = None
    birthplace_city: Optional[str] = None
    birthplace_state: Optional[str] = None
    undergraduate_university: Optional[str] = None
    booker_2022_shortlisted_title: Optional[str] = None

    # Finalist example (non-winning)
    finalist_author_name: Optional[str] = None
    finalist_book_title: Optional[str] = None

    # URLs for each claim
    urls_winning_title: List[str] = Field(default_factory=list)
    urls_publisher: List[str] = Field(default_factory=list)
    urls_publication_year: List[str] = Field(default_factory=list)
    urls_award_2025: List[str] = Field(default_factory=list)
    urls_birthplace: List[str] = Field(default_factory=list)
    urls_undergrad_university: List[str] = Field(default_factory=list)
    urls_booker_2022: List[str] = Field(default_factory=list)
    urls_finalist_example: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_nba2024_info() -> str:
    return """
    Extract the requested information about the 2024 National Book Award for Fiction winner from the provided answer text. Return exactly the fields defined below.

    REQUIRED FIELDS (strings; return null if missing):
    - winning_novel_title: The title of the novel that won the 2024 National Book Award for Fiction.
    - publisher: The publisher (or imprint; e.g., Knopf, Farrar Straus and Giroux, Penguin Press) of the winning novel.
    - publication_year: The year the winning novel was published (e.g., "2024").
    - award_2025: The name of another major prestigious literary award the winning novel won in 2025 (e.g., Pulitzer Prize for Fiction, Women's Prize for Fiction, Booker Prize).
    - author_name: The full name of the author of the winning novel (if provided).
    - birthplace_city: The author's birthplace city (or base/town/installation if that is how it is commonly stated).
    - birthplace_state: The author's birthplace state (for U.S. contexts; use the U.S. state if applicable).
    - undergraduate_university: The university where the author completed their undergraduate degree.
    - booker_2022_shortlisted_title: The title of the author's novel that was shortlisted for the Booker Prize in 2022.
    - finalist_author_name: The name of at least one other author who was a finalist (but did not win) in the 2024 National Book Award for Fiction category.
    - finalist_book_title: That finalist's nominated book title.

    URL FIELDS (arrays of strings; include only URLs explicitly present in the answer):
    - urls_winning_title: URLs that support the winning title claim.
    - urls_publisher: URLs that support the publisher claim.
    - urls_publication_year: URLs that support the publication year claim.
    - urls_award_2025: URLs that support the 2025 additional award claim.
    - urls_birthplace: URLs that support the author's birthplace (city and state) claim.
    - urls_undergrad_university: URLs that support the undergraduate university claim.
    - urls_booker_2022: URLs that support the 2022 Booker-shortlisted novel title claim.
    - urls_finalist_example: URLs that support the non-winning finalist author+book claim.

    RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer text (plain URLs or URLs inside Markdown links).
    - If a URL is missing an explicit scheme, prepend http://.
    - Do not invent or infer URLs.
    - If there are no URLs for a given claim, return an empty list for that URL field.

    Return a single JSON object strictly matching the NBAFiction2024InfoExtraction schema.
    """.strip()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _has_text(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _fmt_author_prefix(author_name: Optional[str]) -> str:
    return author_name.strip() if _has_text(author_name) else "the author of the 2024 National Book Award for Fiction winning novel"


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
    # 1) Initialize evaluator and root
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

    # 2) Extract structured information from the answer
    data = await evaluator.extract(
        prompt=prompt_extract_nba2024_info(),
        template_class=NBAFiction2024InfoExtraction,
        extraction_name="nba_2024_fiction_extraction"
    )

    # 3) Build main critical node representing the entire requested info
    main_node = evaluator.add_parallel(
        id="2024_National_Book_Award_Fiction_Winner_Information",
        desc="Provide all requested information about the 2024 National Book Award for Fiction winner, the author, and at least one non-winning finalist, with supporting reference URLs.",
        parent=root,
        critical=True
    )

    # 4) References node (create early to host URL presence checks used as prerequisites)
    references_node = evaluator.add_parallel(
        id="References",
        desc="Provides supporting reference URLs for each requested claim and ensures they support the claims and come from reliable sources.",
        parent=main_node,
        critical=True
    )

    # 4.1) URL presence checks (critical, act as prerequisites for factual verifications)
    url_nodes = {}

    url_nodes["winning_title"] = evaluator.add_custom_node(
        result=bool(data.urls_winning_title),
        id="URL_For_Winning_Novel_Title",
        desc="Includes at least one reference URL supporting the winning novel title claim.",
        parent=references_node,
        critical=True
    )

    url_nodes["publisher"] = evaluator.add_custom_node(
        result=bool(data.urls_publisher),
        id="URL_For_Publisher",
        desc="Includes at least one reference URL supporting the publisher claim.",
        parent=references_node,
        critical=True
    )

    url_nodes["publication_year"] = evaluator.add_custom_node(
        result=bool(data.urls_publication_year),
        id="URL_For_Publication_Year",
        desc="Includes at least one reference URL supporting the publication year claim.",
        parent=references_node,
        critical=True
    )

    url_nodes["award_2025"] = evaluator.add_custom_node(
        result=bool(data.urls_award_2025),
        id="URL_For_2025_Award",
        desc="Includes at least one reference URL supporting the 2025 award claim.",
        parent=references_node,
        critical=True
    )

    url_nodes["birthplace"] = evaluator.add_custom_node(
        result=bool(data.urls_birthplace),
        id="URL_For_Author_Birthplace",
        desc="Includes at least one reference URL supporting the author's birthplace (city and state) claim.",
        parent=references_node,
        critical=True
    )

    url_nodes["undergrad"] = evaluator.add_custom_node(
        result=bool(data.urls_undergrad_university),
        id="URL_For_Undergraduate_University",
        desc="Includes at least one reference URL supporting the undergraduate university claim.",
        parent=references_node,
        critical=True
    )

    url_nodes["booker_2022"] = evaluator.add_custom_node(
        result=bool(data.urls_booker_2022),
        id="URL_For_2022_Booker_Shortlisted_Novel",
        desc="Includes at least one reference URL supporting the 2022 Booker-shortlisted novel title claim.",
        parent=references_node,
        critical=True
    )

    url_nodes["finalist"] = evaluator.add_custom_node(
        result=bool(data.urls_finalist_example),
        id="URL_For_Non_Winning_Finalist_Example",
        desc="Includes at least one reference URL supporting the non-winning finalist author+book claim.",
        parent=references_node,
        critical=True
    )

    # 5) Winning novel title
    title_provided = evaluator.add_custom_node(
        result=_has_text(data.winning_novel_title),
        id="Winning_Novel_Title_Provided",
        desc="Winning novel title is provided in the answer.",
        parent=main_node,
        critical=True
    )
    winning_title_leaf = evaluator.add_leaf(
        id="Winning_Novel_Title",
        desc="States the title of the novel that won the 2024 National Book Award for Fiction.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The title of the novel that won the 2024 National Book Award for Fiction is '{data.winning_novel_title or ''}'.",
        node=winning_title_leaf,
        sources=data.urls_winning_title,
        additional_instruction="Verify on the cited page(s) that the stated title is the winner of the 2024 National Book Award for Fiction. Allow for minor punctuation/casing variations.",
        extra_prerequisites=[title_provided, url_nodes["winning_title"]]
    )

    # 6) Publication details (publisher + publication year)
    pub_details_node = evaluator.add_parallel(
        id="Winning_Novel_Publication_Details",
        desc="Provides the publisher and publication year for the winning novel.",
        parent=main_node,
        critical=True
    )

    # 6.1) Publisher
    publisher_provided = evaluator.add_custom_node(
        result=_has_text(data.publisher),
        id="Publisher_Provided",
        desc="Publisher value is provided in the answer.",
        parent=pub_details_node,
        critical=True
    )
    publisher_leaf = evaluator.add_leaf(
        id="Publisher",
        desc="Identifies the publisher of the winning novel.",
        parent=pub_details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publisher of the winning novel is '{data.publisher or ''}'. Imprints or parent publishers are acceptable if they reasonably correspond.",
        node=publisher_leaf,
        sources=data.urls_publisher,
        additional_instruction="Accept publisher imprints or parent company names if the page clearly indicates the relationship (e.g., Knopf/PRH, FSG/Macmillan). Match the specific book/edition when possible.",
        extra_prerequisites=[publisher_provided, url_nodes["publisher"]]
    )

    # 6.2) Publication year
    pubyear_provided = evaluator.add_custom_node(
        result=_has_text(data.publication_year),
        id="Publication_Year_Provided",
        desc="Publication year is provided in the answer.",
        parent=pub_details_node,
        critical=True
    )
    publication_year_leaf = evaluator.add_leaf(
        id="Publication_Year",
        desc="States the publication year of the winning novel.",
        parent=pub_details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The winning novel was published in {data.publication_year or ''}.",
        node=publication_year_leaf,
        sources=data.urls_publication_year,
        additional_instruction="Confirm the publication year for the book (original publication year or the commonly cited publication year for the winning edition). Minor discrepancies (e.g., US vs. UK release) should be judged reasonably.",
        extra_prerequisites=[pubyear_provided, url_nodes["publication_year"]]
    )

    # 7) Additional major award in 2025
    award_2025_provided = evaluator.add_custom_node(
        result=_has_text(data.award_2025),
        id="Additional_Major_Award_2025_Provided",
        desc="The 2025 additional major award is provided in the answer.",
        parent=main_node,
        critical=True
    )
    award_2025_leaf = evaluator.add_leaf(
        id="Additional_Major_Award_2025",
        desc="Names the other major prestigious literary award that the winning novel won in 2025.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The winning novel also won the '{data.award_2025 or ''}' in 2025.",
        node=award_2025_leaf,
        sources=data.urls_award_2025,
        additional_instruction="Verify that the cited page explicitly states the novel won this named award in 2025 (not just shortlisted or longlisted).",
        extra_prerequisites=[award_2025_provided, url_nodes["award_2025"]]
    )

    # 8) Author birthplace (city, state)
    birthplace_provided = evaluator.add_custom_node(
        result=_has_text(data.birthplace_city) and _has_text(data.birthplace_state),
        id="Author_Birthplace_Provided",
        desc="Author birthplace (city and state) is provided in the answer.",
        parent=main_node,
        critical=True
    )
    birthplace_leaf = evaluator.add_leaf(
        id="Author_Birthplace",
        desc="Gives the author's birthplace including both city (or installation/town as applicable) and state.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_fmt_author_prefix(data.author_name)} was born in {data.birthplace_city or ''}, {data.birthplace_state or ''}.",
        node=birthplace_leaf,
        sources=data.urls_birthplace,
        additional_instruction="Confirm that the page explicitly states the author's birthplace including both city (or equivalent) and state. Allow for reasonable variants (e.g., military installation names).",
        extra_prerequisites=[birthplace_provided, url_nodes["birthplace"]]
    )

    # 9) Author undergraduate university
    undergrad_provided = evaluator.add_custom_node(
        result=_has_text(data.undergraduate_university),
        id="Author_Undergraduate_University_Provided",
        desc="Undergraduate university is provided in the answer.",
        parent=main_node,
        critical=True
    )
    undergrad_leaf = evaluator.add_leaf(
        id="Author_Undergraduate_University",
        desc="Identifies the university where the author completed their undergraduate degree.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_fmt_author_prefix(data.author_name)} completed their undergraduate degree at {data.undergraduate_university or ''}.",
        node=undergrad_leaf,
        sources=data.urls_undergrad_university,
        additional_instruction="Confirm that the page indicates the author's undergraduate institution (e.g., alumni bio, university page, reputable profile).",
        extra_prerequisites=[undergrad_provided, url_nodes["undergrad"]]
    )

    # 10) Author's 2022 Booker shortlisted novel
    booker22_provided = evaluator.add_custom_node(
        result=_has_text(data.booker_2022_shortlisted_title),
        id="Author_Booker_Shortlisted_2022_Novel_Provided",
        desc="The title of the author's 2022 Booker-shortlisted novel is provided in the answer.",
        parent=main_node,
        critical=True
    )
    booker22_leaf = evaluator.add_leaf(
        id="Author_Booker_Shortlisted_2022_Novel",
        desc="Provides the title of the author's novel that was shortlisted for the Booker Prize in 2022.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The author's novel shortlisted for the 2022 Booker Prize is '{data.booker_2022_shortlisted_title or ''}'.",
        node=booker22_leaf,
        sources=data.urls_booker_2022,
        additional_instruction="Confirm that the page shows the specified novel on the 2022 Booker Prize shortlist (not just longlist).",
        extra_prerequisites=[booker22_provided, url_nodes["booker_2022"]]
    )

    # 11) Non-winning finalist example (author + book) for 2024 NBA Fiction
    finalist_node = evaluator.add_parallel(
        id="Non_Winning_Finalist_Example",
        desc="Provides at least one other 2024 National Book Award for Fiction finalist (not the winner), including author name and nominated book title.",
        parent=main_node,
        critical=True
    )

    finalist_author_provided = evaluator.add_custom_node(
        result=_has_text(data.finalist_author_name),
        id="Finalist_Author_Name",
        desc="Provides the name of a finalist other than the winner.",
        parent=finalist_node,
        critical=True
    )
    finalist_book_provided = evaluator.add_custom_node(
        result=_has_text(data.finalist_book_title),
        id="Finalist_Book_Title",
        desc="Provides the nominated book title for that non-winning finalist.",
        parent=finalist_node,
        critical=True
    )
    finalist_pair_leaf = evaluator.add_leaf(
        id="Finalist_Is_2024_Fiction_Finalist_And_Not_Winner",
        desc="The provided author+book pair is verifiably a 2024 National Book Award for Fiction finalist and is not the winner.",
        parent=finalist_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The pair '{data.finalist_author_name or ''}' and '{data.finalist_book_title or ''}' was a 2024 National Book Award for Fiction finalist and did not win.",
        node=finalist_pair_leaf,
        sources=data.urls_finalist_example,
        additional_instruction="Verify that the cited page lists this author+book as a 2024 National Book Award for Fiction finalist. It is sufficient that they are clearly labeled as 'Finalist' rather than 'Winner'.",
        extra_prerequisites=[finalist_author_provided, finalist_book_provided, url_nodes["finalist"]]
    )

    # 12) References: URLs support claims (implemented as a critical parallel group of custom checks)
    urls_support_group = evaluator.add_parallel(
        id="URLs_Support_Claims",
        desc="The provided URLs contain information that directly supports the corresponding claims.",
        parent=references_node,
        critical=True
    )

    # These checks reflect whether the earlier URL-grounded verifications passed
    evaluator.add_custom_node(
        result=(winning_title_leaf.status == "passed"),
        id="URLs_support_winning_title",
        desc="URLs support the winning title claim.",
        parent=urls_support_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=(publisher_leaf.status == "passed"),
        id="URLs_support_publisher",
        desc="URLs support the publisher claim.",
        parent=urls_support_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=(publication_year_leaf.status == "passed"),
        id="URLs_support_publication_year",
        desc="URLs support the publication year claim.",
        parent=urls_support_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=(award_2025_leaf.status == "passed"),
        id="URLs_support_award_2025",
        desc="URLs support the 2025 award claim.",
        parent=urls_support_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=(birthplace_leaf.status == "passed"),
        id="URLs_support_birthplace",
        desc="URLs support the author birthplace claim.",
        parent=urls_support_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=(undergrad_leaf.status == "passed"),
        id="URLs_support_undergrad",
        desc="URLs support the undergraduate university claim.",
        parent=urls_support_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=(booker22_leaf.status == "passed"),
        id="URLs_support_booker_2022",
        desc="URLs support the 2022 Booker-shortlisted novel claim.",
        parent=urls_support_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=(finalist_pair_leaf.status == "passed"),
        id="URLs_support_finalist_example",
        desc="URLs support the non-winning finalist author+book claim.",
        parent=urls_support_group,
        critical=True
    )

    # 13) References: Source reliability checks (at least one authoritative/reputable URL per claim)
    reliability_group = evaluator.add_parallel(
        id="Source_Reliability_Meets_Stated_Constraint",
        desc="For each claim, at least one supporting URL is from an authoritative/reputable source relevant to that claim.",
        parent=references_node,
        critical=True
    )

    def reliability_instruction(topic_hint: str) -> str:
        return (
            "Judge whether the URL is an authoritative or reputable source to verify the specified topic. "
            "Authoritative examples include: the National Book Foundation (nationalbook.org) for winners/finalists; "
            "official award sites; the Booker Prize site (thebookerprizes.com) for Booker info; "
            "publisher/imprint or catalog pages (e.g., PRH/Knopf, FSG/Macmillan, HarperCollins); "
            "major reputable news outlets (NYT, NPR, The Guardian, AP); university/official alumni profiles; "
            "or widely used bibliographic references (Library of Congress, WorldCat). "
            "Avoid personal blogs or unverified sources. "
            f"Topic to assess: {topic_hint}."
        )

    # Create reliability leaves per topic
    reliability_checks = [
        ("Reliability_Winning_Title", "Authoritative source for verifying the 2024 National Book Award for Fiction winning title.", data.urls_winning_title, "2024 National Book Award for Fiction winning title"),
        ("Reliability_Publisher", "Authoritative source for verifying the publisher of the winning novel.", data.urls_publisher, "publisher of the winning novel"),
        ("Reliability_Publication_Year", "Authoritative source for verifying the publication year of the winning novel.", data.urls_publication_year, "publication year of the winning novel"),
        ("Reliability_Award_2025", "Authoritative source for verifying the additional 2025 award.", data.urls_award_2025, "the novel's additional major award in 2025"),
        ("Reliability_Birthplace", "Authoritative source for verifying the author's birthplace (city, state).", data.urls_birthplace, "author's birthplace (city and state)"),
        ("Reliability_Undergrad", "Authoritative source for verifying the author's undergraduate university.", data.urls_undergrad_university, "author's undergraduate university"),
        ("Reliability_Booker_2022", "Authoritative source for verifying the 2022 Booker-shortlisted novel.", data.urls_booker_2022, "author's 2022 Booker-shortlisted novel title"),
        ("Reliability_Finalist", "Authoritative source for verifying the 2024 NBA Fiction finalist author+book pair.", data.urls_finalist_example, "2024 NBA Fiction finalist author+book pair"),
    ]

    # Add reliability verifications (gated by URL presence)
    reliability_leaf_nodes = []
    for rid, rdesc, rurls, topic in reliability_checks:
        leaf = evaluator.add_leaf(
            id=rid,
            desc=rdesc,
            parent=reliability_group,
            critical=True
        )
        reliability_leaf_nodes.append(leaf)
        await evaluator.verify(
            claim="This webpage is a reputable or authoritative source for the specified topic.",
            node=leaf,
            sources=rurls,
            additional_instruction=reliability_instruction(topic),
            extra_prerequisites=[url_nodes.get({
                "Reliability_Winning_Title": "winning_title",
                "Reliability_Publisher": "publisher",
                "Reliability_Publication_Year": "publication_year",
                "Reliability_Award_2025": "award_2025",
                "Reliability_Birthplace": "birthplace",
                "Reliability_Undergrad": "undergrad",
                "Reliability_Booker_2022": "booker_2022",
                "Reliability_Finalist": "finalist",
            }[rid])]
        )

    # 14) Optional: record some custom summary info
    evaluator.add_custom_info(
        info={
            "extracted_author": data.author_name,
            "url_counts": {
                "winning_title": len(data.urls_winning_title),
                "publisher": len(data.urls_publisher),
                "publication_year": len(data.urls_publication_year),
                "award_2025": len(data.urls_award_2025),
                "birthplace": len(data.urls_birthplace),
                "undergrad_university": len(data.urls_undergrad_university),
                "booker_2022": len(data.urls_booker_2022),
                "finalist_example": len(data.urls_finalist_example),
            }
        },
        info_type="extraction_summary",
        info_name="extraction_summary"
    )

    # 15) Return evaluator summary
    return evaluator.get_summary()