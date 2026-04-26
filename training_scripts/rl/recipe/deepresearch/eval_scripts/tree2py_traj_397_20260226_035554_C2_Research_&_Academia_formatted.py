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
TASK_ID = "genie3_announcement"
TASK_DESCRIPTION = (
    "In August 2025, Google DeepMind announced a new world model called Genie 3 through an official blog post. "
    "Identify the official announcement blog post by locating its publication date and complete title. Then, "
    "identify the two authors who wrote this announcement. Provide the blog post URL as a reference, confirm "
    "the exact publication date (month, day, and year), state the complete blog post title, and list both authors' "
    "full names as they appear in the announcement."
)

EXPECTED_DATE = "August 5, 2025"
EXPECTED_TITLE = "Genie 3: A new frontier for world models"
EXPECTED_AUTHORS = ["Jack Parker-Holder", "Shlomi Fruchter"]

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Genie3Announcement(BaseModel):
    blog_url: Optional[str] = None
    blog_title: Optional[str] = None
    publication_date: Optional[str] = None
    authors: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_genie3_info() -> str:
    return """
    Extract the details of the official Google DeepMind announcement for Genie 3 as explicitly presented in the answer text.

    You must extract exactly what the answer states (do not infer):
    - blog_url: The single URL the answer cites as the official Google DeepMind blog post announcing Genie 3.
                 If multiple URLs appear, choose the one that is the official Google DeepMind blog page
                 (prefer domains like deepmind.google or deepmind.com/blog). If none is provided, set to null.
    - blog_title: The complete title of the blog post exactly as written in the answer. If missing, set to null.
    - publication_date: The publication date of the blog post (month, day, year) exactly as written in the answer. If missing, set to null.
    - authors: A list of the authors' full names exactly as written in the answer, in the order they appear. If none are written, return an empty list.

    Do not fabricate any fields. Only extract what is explicitly present in the answer text.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_genie3_announcement(evaluator: Evaluator, parent_node, extracted: Genie3Announcement) -> None:
    # Top-level critical research node
    research_node = evaluator.add_parallel(
        id="Genie_3_Announcement_Research",
        desc="Complete research task on Genie 3 announcement and authors",
        parent=parent_node,
        critical=True
    )

    # 1) Announcement Details (critical, parallel)
    details_node = evaluator.add_parallel(
        id="Announcement_Details",
        desc="Verify announcement blog post details",
        parent=research_node,
        critical=True
    )

    # 1.a) Blog URL reference (critical leaf)
    blog_url_leaf = evaluator.add_leaf(
        id="Blog_URL_Reference",
        desc="Provide reference URL to the official Google DeepMind blog post",
        parent=details_node,
        critical=True
    )

    if extracted.blog_url and extracted.blog_url.strip():
        claim = (
            "This URL is the official Google DeepMind announcement blog post for Genie 3."
        )
        await evaluator.verify(
            claim=claim,
            node=blog_url_leaf,
            sources=extracted.blog_url,
            additional_instruction=(
                "Confirm that the page is an official Google DeepMind blog post (e.g., domain like deepmind.google "
                "or deepmind.com/blog) and that it announces Genie 3."
            ),
        )
    else:
        # Verify (using only the answer text as context) that a proper official DeepMind blog URL is provided.
        claim = "The answer provides an official Google DeepMind blog URL to the Genie 3 announcement."
        await evaluator.verify(
            claim=claim,
            node=blog_url_leaf,
            sources=None,
            additional_instruction=(
                "Check the answer text only. Determine if it contains a URL pointing to the official Google DeepMind "
                "blog post announcing Genie 3 (domain like deepmind.google or deepmind.com/blog). Mark incorrect if not provided."
            ),
        )

    # 1.b) Publication Date (critical leaf)
    pub_date_leaf = evaluator.add_leaf(
        id="Publication_Date",
        desc="The announcement must be published on August 5, 2025",
        parent=details_node,
        critical=True
    )
    if extracted.publication_date and extracted.publication_date.strip():
        pub_claim = (
            f"The blog post was published on {extracted.publication_date}, and that date is August 5, 2025."
        )
    else:
        pub_claim = "The blog post was published on August 5, 2025."
    await evaluator.verify(
        claim=pub_claim,
        node=pub_date_leaf,
        sources=extracted.blog_url if extracted.blog_url else None,
        extra_prerequisites=[blog_url_leaf],
        additional_instruction=(
            "Verify the exact publication date shown on the page. Accept reasonable formatting variants like "
            "'Aug 5, 2025' or '5 August 2025' as the same calendar date."
        ),
    )

    # 1.c) Blog Title (critical leaf)
    title_leaf = evaluator.add_leaf(
        id="Blog_Title",
        desc="The blog post title must be 'Genie 3: A new frontier for world models'",
        parent=details_node,
        critical=True
    )
    if extracted.blog_title and extracted.blog_title.strip():
        title_claim = (
            f"The blog post title is exactly '{extracted.blog_title}', and it equals 'Genie 3: A new frontier for world models'."
        )
    else:
        title_claim = "The blog post title is 'Genie 3: A new frontier for world models'."
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=extracted.blog_url if extracted.blog_url else None,
        extra_prerequisites=[blog_url_leaf],
        additional_instruction=(
            "Verify the page's displayed title text. Allow minor typographical variations only if clearly equivalent, "
            "but prefer exact match."
        ),
    )

    # 2) Author Identification (critical, parallel)
    authors_node = evaluator.add_parallel(
        id="Author_Identification",
        desc="Identify and verify the authors of the announcement",
        parent=research_node,
        critical=True
    )

    # 2.a) First Author: Jack Parker-Holder
    first_author_leaf = evaluator.add_leaf(
        id="First_Author",
        desc="Jack Parker-Holder must be listed as an author",
        parent=authors_node,
        critical=True
    )
    await evaluator.verify(
        claim="The blog post lists 'Jack Parker-Holder' as an author.",
        node=first_author_leaf,
        sources=extracted.blog_url if extracted.blog_url else None,
        extra_prerequisites=[blog_url_leaf],
        additional_instruction=(
            "Check the author byline or author section on the blog post page. Allow minor punctuation or hyphenation "
            "variations but ensure the name clearly matches 'Jack Parker-Holder'."
        ),
    )

    # 2.b) Second Author: Shlomi Fruchter
    second_author_leaf = evaluator.add_leaf(
        id="Second_Author",
        desc="Shlomi Fruchter must be listed as an author",
        parent=authors_node,
        critical=True
    )
    await evaluator.verify(
        claim="The blog post lists 'Shlomi Fruchter' as an author.",
        node=second_author_leaf,
        sources=extracted.blog_url if extracted.blog_url else None,
        extra_prerequisites=[blog_url_leaf],
        additional_instruction=(
            "Check the author byline or author section on the blog post page. Allow minor punctuation variations but "
            "ensure the name clearly matches 'Shlomi Fruchter'."
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
    # Initialize evaluator
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

    # Record expected ground truth facts for transparency
    evaluator.add_ground_truth(
        {
            "expected_publication_date": EXPECTED_DATE,
            "expected_title": EXPECTED_TITLE,
            "expected_authors": EXPECTED_AUTHORS,
            "note": "These are the expected canonical details of the official Google DeepMind Genie 3 announcement.",
        },
        gt_type="ground_truth",
    )

    # Extract information from the provided answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_genie3_info(),
        template_class=Genie3Announcement,
        extraction_name="genie3_announcement_extraction",
    )

    # Build verification tree and run checks
    await verify_genie3_announcement(evaluator, root, extracted_info)

    # Return structured evaluation summary
    return evaluator.get_summary()