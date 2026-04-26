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
TASK_ID = "cf_2024_first_novel_prize_author"
TASK_DESCRIPTION = """
I'm looking for information about a debut novelist who won a major literary award in 2024. This author satisfies all of the following criteria:

- Won The Center for Fiction's 2024 First Novel Prize, which awards $15,000 to the winner
- Is a veteran who served in the Iraq War
- Holds an MFA in prose from the University of Notre Dame
- Earned a PhD in English from the University of Pennsylvania
- Is originally from Philadelphia
- Has worked as an EMS (Emergency Medical Services) worker
- Published their debut novel in 2024 with Grand Central Publishing
- Grand Central Publishing was formerly known as Warner Books (founded in 1970)
- Grand Central Publishing is an imprint of Hachette Book Group (founded March 31, 2006)
- Is an American citizen or permanent resident (required for prize eligibility)

Please identify this author, provide the title of their winning debut novel, and include any information about their other published works. Provide a reference URL that verifies the award announcement.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BackgroundInfo(BaseModel):
    # MFA
    mfa_text: Optional[str] = None
    mfa_institution: Optional[str] = None
    mfa_field: Optional[str] = None
    mfa_sources: List[str] = Field(default_factory=list)

    # PhD
    phd_text: Optional[str] = None
    phd_institution: Optional[str] = None
    phd_field: Optional[str] = None
    phd_sources: List[str] = Field(default_factory=list)

    # Military service
    military_service_text: Optional[str] = None
    military_sources: List[str] = Field(default_factory=list)

    # Origin
    origin_city: Optional[str] = None
    origin_sources: List[str] = Field(default_factory=list)

    # EMS professional background
    ems_experience_text: Optional[str] = None
    ems_sources: List[str] = Field(default_factory=list)

    # Citizenship / residency
    citizenship_text: Optional[str] = None  # e.g., "American citizen", "U.S. permanent resident"
    citizenship_sources: List[str] = Field(default_factory=list)


class NovelInfo(BaseModel):
    publication_date: Optional[str] = None  # Any format as stated in the answer
    novel_sources: List[str] = Field(default_factory=list)  # sources verifying title, debut status, pub date, publisher
    debut_sources: List[str] = Field(default_factory=list)  # if separate sources for debut status are given


class PublisherInfo(BaseModel):
    publisher_name: Optional[str] = None  # e.g., "Grand Central Publishing"
    publisher_sources: List[str] = Field(default_factory=list)  # pages that state publisher of the book
    publisher_history_sources: List[str] = Field(default_factory=list)  # for Warner Books (1970) history
    publisher_parent_sources: List[str] = Field(default_factory=list)  # for Hachette Book Group parent (founded 2006-03-31)


class OtherWorkItem(BaseModel):
    title: Optional[str] = None
    year: Optional[str] = None
    publisher: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AuthorExtraction(BaseModel):
    author_name: Optional[str] = None
    novel_title: Optional[str] = None
    award_urls: List[str] = Field(default_factory=list)

    background: Optional[BackgroundInfo] = None
    novel: Optional[NovelInfo] = None
    publisher: Optional[PublisherInfo] = None

    other_works: List[OtherWorkItem] = Field(default_factory=list)
    explicitly_no_other_works: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_author_info() -> str:
    return """
Extract the requested structured information from the answer text. Return exactly the fields required by the JSON schema. Follow the URL extraction special rules.

Required fields to extract:

1) author_name: The identified debut novelist's full name as presented.

2) novel_title: The title of the winning debut novel.

3) award_urls: All URLs explicitly cited that verify the award announcement or details about The Center for Fiction's 2024 First Novel Prize (including press releases, official pages, reputable news coverage). Return every URL the answer provided for this.

4) background: An object with the following fields and sources to substantiate each claim when available:
   - mfa_text: The exact phrasing about the author's MFA (e.g., "MFA in prose from the University of Notre Dame")
   - mfa_institution: The institution name (e.g., "University of Notre Dame")
   - mfa_field: The stated field (e.g., "prose", "creative writing (prose)")
   - mfa_sources: URLs that support the MFA claim
   - phd_text: The exact phrasing about the author's PhD (e.g., "PhD in English from the University of Pennsylvania")
   - phd_institution: The institution name (e.g., "University of Pennsylvania")
   - phd_field: The stated field (e.g., "English")
   - phd_sources: URLs that support the PhD claim
   - military_service_text: Phrasing indicating the author is a veteran who served in the Iraq War
   - military_sources: URLs that support the Iraq War service claim
   - origin_city: The stated city of origin (e.g., "Philadelphia")
   - origin_sources: URLs supporting the origin
   - ems_experience_text: Phrasing indicating EMS work experience
   - ems_sources: URLs supporting EMS background
   - citizenship_text: Phrasing indicating the author is an American citizen or a U.S. permanent resident
   - citizenship_sources: URLs supporting citizenship or residency status

5) novel: An object with:
   - publication_date: The publication date of the debut novel as stated (any format; do not normalize)
   - novel_sources: URLs that support the novel title, publication date, publisher, and/or award linkage
   - debut_sources: URLs that explicitly support that this is the author's debut novel (if separately provided)

6) publisher: An object with:
   - publisher_name: The stated publisher of the debut novel (e.g., "Grand Central Publishing")
   - publisher_sources: URLs that show the book's publisher (e.g., publisher's book page, catalog, retailer page)
   - publisher_history_sources: URLs supporting that Grand Central Publishing was formerly known as Warner Books and that Warner Books was founded in 1970
   - publisher_parent_sources: URLs supporting that Grand Central Publishing is an imprint of Hachette Book Group and that Hachette Book Group was founded on March 31, 2006

7) other_works: A list of other published works by the author (excluding the debut novel). For each item, extract:
   - title
   - year (if mentioned)
   - publisher (if mentioned)
   - sources: URLs that support this work
   If the answer states there are none, leave this list empty.

8) explicitly_no_other_works: true if the answer explicitly says the author has no other published works or none are known; false otherwise (or null if not stated).

Important:
- Only extract URLs that appear explicitly in the answer text. Include both plain URLs and markdown-style links (extract the actual URL).
- If any item or URL is missing in the answer, set it to null or an empty list as appropriate.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return lst if isinstance(lst, list) else []


def _combine_lists(*lists: Optional[List[str]]) -> List[str]:
    combo: List[str] = []
    for l in lists:
        if l:
            combo.extend([u for u in l if isinstance(u, str) and u.strip() != ""])
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in combo:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_author_name(evaluator: Evaluator, parent, extraction: AuthorExtraction) -> None:
    # Existence check (critical within core requirements)
    name_exists = isinstance(extraction.author_name, str) and extraction.author_name.strip() != ""
    evaluator.add_custom_node(
        result=name_exists,
        id="Author_Name_Provided",
        desc="Provide the author's name/identity",
        parent=parent,
        critical=True
    )


async def verify_author_background(evaluator: Evaluator, parent, extraction: AuthorExtraction) -> None:
    # Parent node for background (critical group)
    bg_node = evaluator.add_parallel(
        id="Author_Background",
        desc="Verify the author's educational, professional, military, and origin background per constraints",
        parent=parent,
        critical=True
    )

    bg = extraction.background or BackgroundInfo()

    # Educational - MFA
    mfa_leaf = evaluator.add_leaf(
        id="Educational_Background_MFA",
        desc="The author has an MFA in prose from the University of Notre Dame",
        parent=bg_node,
        critical=True
    )
    mfa_claim = (
        f"The author {extraction.author_name} has an MFA (or M.F.A.) in prose (or creative writing—prose) "
        f"from the University of Notre Dame."
    )
    await evaluator.verify(
        claim=mfa_claim,
        node=mfa_leaf,
        sources=_safe_list(bg.mfa_sources),
        additional_instruction="Allow minor wording variations for degree naming (e.g., MFA/M.F.A.) and field phrasing like 'creative writing (prose)'."
    )

    # Educational - PhD
    phd_leaf = evaluator.add_leaf(
        id="Educational_Background_PhD",
        desc="The author has a PhD in English from the University of Pennsylvania",
        parent=bg_node,
        critical=True
    )
    phd_claim = (
        f"The author {extraction.author_name} has a PhD (or Ph.D.) in English from the University of Pennsylvania."
    )
    await evaluator.verify(
        claim=phd_claim,
        node=phd_leaf,
        sources=_safe_list(bg.phd_sources),
        additional_instruction="Allow PhD/Ph.D. spelling variants; verify the field 'English' and institution."
    )

    # Military Service - Iraq War
    mil_leaf = evaluator.add_leaf(
        id="Military_Service",
        desc="The author is a veteran who served in the Iraq War",
        parent=bg_node,
        critical=True
    )
    mil_claim = f"The author {extraction.author_name} is a veteran who served in the Iraq War."
    await evaluator.verify(
        claim=mil_claim,
        node=mil_leaf,
        sources=_safe_list(bg.military_sources),
        additional_instruction="Look for explicit confirmation of Iraq War service; allow bios or interviews."
    )

    # Origin - Philadelphia
    origin_leaf = evaluator.add_leaf(
        id="Geographic_Origin",
        desc="The author is originally from Philadelphia",
        parent=bg_node,
        critical=True
    )
    origin_claim = f"The author {extraction.author_name} is originally from Philadelphia."
    await evaluator.verify(
        claim=origin_claim,
        node=origin_leaf,
        sources=_safe_list(bg.origin_sources),
        additional_instruction="Accept reasonable phrasings like 'born in Philadelphia', 'from Philadelphia', 'Philadelphia native'."
    )

    # Professional - EMS work
    ems_leaf = evaluator.add_leaf(
        id="Professional_Background",
        desc="The author has worked as an EMS (Emergency Medical Services) worker",
        parent=bg_node,
        critical=True
    )
    ems_claim = f"The author {extraction.author_name} has worked as an EMS (Emergency Medical Services) worker."
    await evaluator.verify(
        claim=ems_claim,
        node=ems_leaf,
        sources=_safe_list(bg.ems_sources),
        additional_instruction="Accept related phrasing like EMT/paramedic if clearly within EMS."
    )


async def verify_award(evaluator: Evaluator, parent, extraction: AuthorExtraction) -> None:
    award_node = evaluator.add_parallel(
        id="Award_Verification",
        desc="Verify award win, prize amount, eligibility requirement, and provide verifying URL",
        parent=parent,
        critical=True
    )

    # Reference URL existence (critical gating)
    has_award_url = len(_safe_list(extraction.award_urls)) > 0
    evaluator.add_custom_node(
        result=has_award_url,
        id="Reference_URL",
        desc="Provide a reference URL that verifies the award announcement",
        parent=award_node,
        critical=True
    )

    # Award Win
    award_win_leaf = evaluator.add_leaf(
        id="Award_Win",
        desc="The author won The Center for Fiction's 2024 First Novel Prize",
        parent=award_node,
        critical=True
    )
    award_claim = (
        f"{extraction.author_name} won The Center for Fiction's 2024 First Novel Prize."
    )
    await evaluator.verify(
        claim=award_claim,
        node=award_win_leaf,
        sources=_safe_list(extraction.award_urls),
        additional_instruction="Verify the winner and year 2024 explicitly from the award announcement or reputable coverage."
    )

    # Prize Amount
    prize_leaf = evaluator.add_leaf(
        id="Prize_Amount",
        desc="The First Novel Prize winner receives $15,000",
        parent=award_node,
        critical=True
    )
    prize_claim = "The Center for Fiction's First Novel Prize winner receives $15,000."
    await evaluator.verify(
        claim=prize_claim,
        node=prize_leaf,
        sources=_safe_list(extraction.award_urls),
        additional_instruction="Confirm that the prize amount for the First Novel Prize is $15,000 for the 2024 cycle."
    )

    # Eligibility - citizenship/residency (author)
    bg = extraction.background or BackgroundInfo()
    elig_leaf = evaluator.add_leaf(
        id="Eligibility_Status",
        desc="The author is an American citizen or permanent resident (required for prize eligibility)",
        parent=award_node,
        critical=True
    )
    elig_claim = f"The author {extraction.author_name} is an American citizen or a U.S. permanent resident."
    await evaluator.verify(
        claim=elig_claim,
        node=elig_leaf,
        sources=_safe_list(bg.citizenship_sources),
        additional_instruction="Look for explicit statements (e.g., bio, interview, publisher page) indicating U.S. citizenship or permanent residency."
    )


async def verify_novel_and_publisher(evaluator: Evaluator, parent, extraction: AuthorExtraction) -> None:
    np_node = evaluator.add_parallel(
        id="Novel_And_Publisher_Details",
        desc="Verify the winning debut novel details and the publisher details per constraints",
        parent=parent,
        critical=True
    )

    novel_title = extraction.novel_title or ""
    novel = extraction.novel or NovelInfo()
    publisher = extraction.publisher or PublisherInfo()

    # Sources for novel facts (combine novel sources and award URLs)
    novel_fact_urls = _combine_lists(novel.novel_sources, extraction.award_urls)

    # Novel Title
    title_leaf = evaluator.add_leaf(
        id="Novel_Title",
        desc="Provide the title of the winning debut novel",
        parent=np_node,
        critical=True
    )
    title_claim = f"The author's winning debut novel is titled '{novel_title}'."
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=novel_fact_urls,
        additional_instruction="Verify that the stated title matches the winner's debut novel title."
    )

    # Debut Status
    debut_leaf = evaluator.add_leaf(
        id="Debut_Status",
        desc="The author is a first-time novelist and the identified novel is their debut novel",
        parent=np_node,
        critical=True
    )
    debut_claim = f"'{novel_title}' is the debut novel of {extraction.author_name}."
    debut_urls = _combine_lists(novel.debut_sources, novel_fact_urls)
    await evaluator.verify(
        claim=debut_claim,
        node=debut_leaf,
        sources=debut_urls,
        additional_instruction="Look for phrases like 'debut novel' or 'first novel' associated with the identified book."
    )

    # Publication Date Within 2024
    pub_leaf = evaluator.add_leaf(
        id="Publication_Date_Within_2024",
        desc="The debut novel was published between January 1 and December 31, 2024",
        parent=np_node,
        critical=True
    )
    pub_claim = f"The novel '{novel_title}' was published in 2024 (between January 1 and December 31, 2024)."
    await evaluator.verify(
        claim=pub_claim,
        node=pub_leaf,
        sources=_safe_list(novel.novel_sources),
        additional_instruction="Confirm the publication year is 2024; allow minor date format variations."
    )

    # Publisher Information (nested critical group)
    pubinfo_node = evaluator.add_parallel(
        id="Publisher_Information",
        desc="Verify the publisher details and corporate history per constraints",
        parent=np_node,
        critical=True
    )

    # Publisher Name
    pubname_leaf = evaluator.add_leaf(
        id="Publisher_Name",
        desc="The debut novel was published by Grand Central Publishing",
        parent=pubinfo_node,
        critical=True
    )
    pubname_claim = f"The novel '{novel_title}' was published by Grand Central Publishing."
    await evaluator.verify(
        claim=pubname_claim,
        node=pubname_leaf,
        sources=_combine_lists(novel_fact_urls, publisher.publisher_sources),
        additional_instruction="Verify that the book's publisher is listed as Grand Central Publishing on credible pages (publisher page preferred)."
    )

    # Publisher History
    pubhist_leaf = evaluator.add_leaf(
        id="Publisher_History",
        desc="Grand Central Publishing was formerly known as Warner Books, which was founded in 1970",
        parent=pubinfo_node,
        critical=True
    )
    pubhist_claim = "Grand Central Publishing was formerly known as Warner Books, which was founded in 1970."
    await evaluator.verify(
        claim=pubhist_claim,
        node=pubhist_leaf,
        sources=_safe_list(publisher.publisher_history_sources),
        additional_instruction="Confirm both parts: (1) Grand Central Publishing's former name was Warner Books; (2) Warner Books was founded in 1970."
    )

    # Publisher Parent
    pubparent_leaf = evaluator.add_leaf(
        id="Publisher_Parent",
        desc="Grand Central Publishing is an imprint of/part of Hachette Book Group, which was founded on March 31, 2006",
        parent=pubinfo_node,
        critical=True
    )
    pubparent_claim = "Grand Central Publishing is an imprint of Hachette Book Group, which was founded on March 31, 2006."
    await evaluator.verify(
        claim=pubparent_claim,
        node=pubparent_leaf,
        sources=_safe_list(publisher.publisher_parent_sources),
        additional_instruction="Verify both: (1) imprint relationship to Hachette Book Group; (2) Hachette Book Group founding date March 31, 2006."
    )


async def verify_other_works(evaluator: Evaluator, parent, extraction: AuthorExtraction) -> None:
    # Non-critical group
    ow_node = evaluator.add_parallel(
        id="Other_Published_Works",
        desc="Include information about the author's other published works (or explicitly state none are known), excluding the debut novel",
        parent=parent,
        critical=False
    )

    info_present = bool(extraction.explicitly_no_other_works) or (len(extraction.other_works) > 0)
    evaluator.add_custom_node(
        result=info_present,
        id="Other_Works_Info_Present",
        desc="The answer includes info about other works or explicitly states none are known",
        parent=ow_node,
        critical=False
    )

    # Verify up to first 3 other works (non-critical leaf checks)
    max_works = min(3, len(extraction.other_works))
    for i in range(max_works):
        item = extraction.other_works[i]
        if not item or not item.title:
            # Add a failed check for transparency
            evaluator.add_custom_node(
                result=False,
                id=f"Other_Work_{i}_has_title",
                desc=f"Other work #{i + 1} has a title provided",
                parent=ow_node,
                critical=False
            )
            continue

        leaf = evaluator.add_leaf(
            id=f"Other_Work_{i}_Supported",
            desc=f"Other published work #{i + 1} is accurately cited with sources",
            parent=ow_node,
            critical=False
        )
        other_claim = f"{extraction.author_name} has also published a work titled '{item.title}'."
        await evaluator.verify(
            claim=other_claim,
            node=leaf,
            sources=_safe_list(item.sources),
            additional_instruction="Verify that the cited work is by the same author; minor title formatting variations are acceptable."
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
    Evaluate an answer for the 2024 Center for Fiction First Novel Prize author identification task.
    Note: The framework disallows a critical node having non-critical children. Therefore, we group all
    critical checks under a 'core_requirements' critical node and keep optional other-works checks separate.
    """
    # 1) Initialize evaluator (root is non-critical, parallel aggregation)
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
        default_model=model
    )

    # 2) Extraction
    extraction: AuthorExtraction = await evaluator.extract(
        prompt=prompt_extract_author_info(),
        template_class=AuthorExtraction,
        extraction_name="parsed_author_info"
    )

    # 3) Build verification tree
    # Core requirements (all children critical)
    core = evaluator.add_parallel(
        id="Author_Identification",
        desc="Identify the debut novelist and provide required supporting details per the question and constraints",
        parent=root,
        critical=True
    )

    # 3.1 Author name (existence)
    await verify_author_name(evaluator, core, extraction)

    # 3.2 Background checks
    await verify_author_background(evaluator, core, extraction)

    # 3.3 Award verification (win, prize, eligibility, reference URL)
    await verify_award(evaluator, core, extraction)

    # 3.4 Novel + Publisher constraints
    await verify_novel_and_publisher(evaluator, core, extraction)

    # 3.5 Other published works (non-critical, separate group)
    await verify_other_works(evaluator, root, extraction)

    # 4) Return evaluation summary
    return evaluator.get_summary()