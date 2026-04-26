import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "golf_course_hosting_verification"
TASK_DESCRIPTION = """
A California golf course is scheduled to host the 131st US Open Championship from June 12-15, 2031. This same course hosted a US Open Championship in 1948 that was won by Ben Hogan, marking the first US Open ever held in California. The course also hosted PGA Championships in 1983 and 1995. Looking toward the near future, this course will host the US Women's Open in 2026 and will serve as the venue for both the men's and women's golf competitions at the 2028 Olympic Games. The course is the long-time host of an annual PGA Tour event known as the Genesis Invitational (formerly the Los Angeles Open), which it has hosted since 1929. The course opened in 1927 and was designed by George C. Thomas Jr. with assistance from William P. Bell. It is located in Pacific Palisades, California. What is the name of this golf course, and provide reference URLs that verify: (1) the 2031 US Open hosting announcement with specific dates, (2) the 1948 US Open championship details including the winner, (3) the 2028 Olympic Games golf hosting, (4) the PGA Championship hosting in both 1983 and 1995, (5) that it currently hosts the Genesis Invitational, and (6) the course's opening year, designers, and location?
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class CourseExtraction(BaseModel):
    course_name: Optional[str] = None

    urls_2031_us_open: List[str] = Field(default_factory=list)
    urls_1948_us_open: List[str] = Field(default_factory=list)
    urls_pga_championships: List[str] = Field(default_factory=list)
    urls_womens_open_2026: List[str] = Field(default_factory=list)
    urls_olympics_2028: List[str] = Field(default_factory=list)
    urls_genesis_invitational: List[str] = Field(default_factory=list)
    urls_course_history: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_course_info() -> str:
    return """
    Your goal is to extract from the answer:
    1) The golf course name
    2) The explicit URL sources that the answer provides for each verification area below.
    
    You must follow these strict rules:
    - Extract only URLs that are explicitly present in the answer.
    - Do not invent or infer URLs.
    - Return full URLs. If a URL is missing a protocol, prepend http://
    - If the answer doesn’t provide any URL for a category, return an empty list for that category.
    
    Required fields to extract:
    - course_name: the name of the golf course stated in the answer.
    - urls_2031_us_open: URLs supporting that the 2031 U.S. Open will be held at this course and the specific dates June 12–15, 2031 (and that it is the 131st U.S. Open).
    - urls_1948_us_open: URLs supporting that the 1948 U.S. Open was held at this course, was won by Ben Hogan, and was the first U.S. Open ever held in California.
    - urls_pga_championships: URLs supporting that this course hosted PGA Championships in 1983 and 1995.
    - urls_womens_open_2026: URLs supporting that this course will host the 2026 U.S. Women’s Open.
    - urls_olympics_2028: URLs supporting that this course will host BOTH men’s and women’s golf competitions at the Los Angeles 2028 Olympics (LA28).
    - urls_genesis_invitational: URLs supporting that this course currently hosts the Genesis Invitational (formerly the Los Angeles Open) and that it first hosted this tournament in 1929.
    - urls_course_history: URLs supporting the course’s opening year (1927), primary designer (George C. Thomas Jr.), assistant designer (William P. Bell), and its location (Pacific Palisades, California).
    
    If a category isn’t supported by any URL in the answer, return an empty array for that category.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def _union_sources(ex: CourseExtraction, fields: List[str]) -> List[str]:
    all_urls: List[str] = []
    for f in fields:
        all_urls.extend(getattr(ex, f, []) or [])
    return _dedup_urls(all_urls)


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_top_level_course_nodes(evaluator: Evaluator, parent, ex: CourseExtraction):
    """
    Build top-level identification node and course-name checks.
    """
    golf_node = evaluator.add_parallel(
        id="Golf_Course_Identification",
        desc="Identify a California golf course that meets all specified historical and future event hosting criteria",
        parent=parent,
        critical=True
    )

    # Optional but useful: ensure the answer provides a course name
    evaluator.add_custom_node(
        result=_nonempty(ex.course_name),
        id="Course_Name_Provided",
        desc="A course name is provided in the answer",
        parent=golf_node,
        critical=True
    )

    # Optionally verify that at least one provided URL across categories supports the stated course name
    # Use union of all sources
    all_sources = _union_sources(
        ex,
        [
            "urls_2031_us_open",
            "urls_1948_us_open",
            "urls_pga_championships",
            "urls_womens_open_2026",
            "urls_olympics_2028",
            "urls_genesis_invitational",
            "urls_course_history",
        ],
    )
    name_supported_node = evaluator.add_leaf(
        id="Course_Name_Supported_By_Sources",
        desc="The stated course name is supported by at least one cited source",
        parent=golf_node,
        critical=True
    )
    claim = f"The name of the golf course is '{ex.course_name}'." if _nonempty(ex.course_name) else "The name of the golf course is stated."
    await evaluator.verify(
        claim=claim,
        node=name_supported_node,
        sources=all_sources,
        additional_instruction="Confirm that at least one provided source page explicitly names this golf course."
    )

    return golf_node


async def build_past_major_championships_nodes(evaluator: Evaluator, parent, ex: CourseExtraction):
    """
    Past_Major_Championships subtree:
      - US_Open_1948 (with winner and 'first in California' checks + URL existence)
      - PGA_Championships (1983 and 1995 + URL existence)
    """
    past_node = evaluator.add_parallel(
        id="Past_Major_Championships",
        desc="Verify the course has hosted specific major championships in its history",
        parent=parent,
        critical=True
    )

    # 1948 U.S. Open group
    us_open_1948_node = evaluator.add_parallel(
        id="US_Open_1948",
        desc="Verify the course hosted the 1948 US Open Championship with specific details",
        parent=past_node,
        critical=True
    )

    # Existence of URLs for 1948 facts
    evaluator.add_custom_node(
        result=len(ex.urls_1948_us_open) > 0,
        id="US_Open_1948_Reference_URL",
        desc="Provide a reference URL confirming the 1948 US Open details",
        parent=us_open_1948_node,
        critical=True
    )

    # Winner: Ben Hogan
    winner_node = evaluator.add_leaf(
        id="Winner_Ben_Hogan",
        desc="The 1948 US Open was won by Ben Hogan",
        parent=us_open_1948_node,
        critical=True
    )
    claim_winner = (
        f"Ben Hogan won the 1948 U.S. Open Championship held at {_safe_course(ex.course_name)}."
        if _nonempty(ex.course_name) else
        "Ben Hogan won the 1948 U.S. Open Championship."
    )
    await evaluator.verify(
        claim=claim_winner,
        node=winner_node,
        sources=ex.urls_1948_us_open,
        additional_instruction="The source must explicitly state that Ben Hogan won the 1948 U.S. Open; ideally it also indicates the host course."
    )

    # First U.S. Open in California
    first_ca_node = evaluator.add_leaf(
        id="First_California_US_Open",
        desc="This was the first US Open held in California",
        parent=us_open_1948_node,
        critical=True
    )
    claim_first = (
        f"The 1948 U.S. Open at {_safe_course(ex.course_name)} was the first U.S. Open ever held in California."
        if _nonempty(ex.course_name) else
        "The 1948 U.S. Open was the first U.S. Open ever held in California."
    )
    await evaluator.verify(
        claim=claim_first,
        node=first_ca_node,
        sources=ex.urls_1948_us_open,
        additional_instruction="Look for wording that the 1948 U.S. Open was the first time the U.S. Open was held in California."
    )

    # PGA Championships group
    pga_node = evaluator.add_parallel(
        id="PGA_Championships",
        desc="Verify the course hosted PGA Championships in 1983 and 1995",
        parent=past_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(ex.urls_pga_championships) > 0,
        id="PGA_Championships_Reference_URL",
        desc="Provide a reference URL confirming the PGA Championship hosting",
        parent=pga_node,
        critical=True
    )

    # 1983 PGA Championship
    pga_1983 = evaluator.add_leaf(
        id="PGA_1983",
        desc="The course hosted the 1983 PGA Championship",
        parent=pga_node,
        critical=True
    )
    claim_83 = (
        f"{_safe_course(ex.course_name)} hosted the 1983 PGA Championship."
        if _nonempty(ex.course_name) else
        "The course hosted the 1983 PGA Championship."
    )
    # 1995 PGA Championship
    pga_1995 = evaluator.add_leaf(
        id="PGA_1995",
        desc="The course hosted the 1995 PGA Championship",
        parent=pga_node,
        critical=True
    )
    claim_95 = (
        f"{_safe_course(ex.course_name)} hosted the 1995 PGA Championship."
        if _nonempty(ex.course_name) else
        "The course hosted the 1995 PGA Championship."
    )

    await evaluator.batch_verify(
        [
            (
                claim_83,
                ex.urls_pga_championships,
                pga_1983,
                "Confirm the course is listed as the site of the 1983 PGA Championship."
            ),
            (
                claim_95,
                ex.urls_pga_championships,
                pga_1995,
                "Confirm the course is listed as the site of the 1995 PGA Championship."
            ),
        ]
    )

    return past_node


async def build_future_events_nodes(evaluator: Evaluator, parent, ex: CourseExtraction):
    """
    Future_Major_Events subtree:
      - US_Womens_Open_2026 (scheduled + URL existence)
      - Olympics_2028 (both men's & women's, LA28, + URL existence)
      - US_Open_2031 (dates & 131st + URL existence)
    """
    future_node = evaluator.add_parallel(
        id="Future_Major_Events",
        desc="Verify the course is scheduled to host specific future major championships and events",
        parent=parent,
        critical=True
    )

    # 2026 U.S. Women's Open
    wopen_node = evaluator.add_parallel(
        id="US_Womens_Open_2026",
        desc="Verify the course is scheduled to host the 2026 US Women's Open",
        parent=future_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(ex.urls_womens_open_2026) > 0,
        id="US_Womens_Open_Reference_URL",
        desc="Provide a reference URL confirming the 2026 US Women's Open scheduling",
        parent=wopen_node,
        critical=True
    )
    wopen_leaf = evaluator.add_leaf(
        id="US_Womens_Open_Scheduled",
        desc="The course is scheduled to host the 2026 US Women's Open",
        parent=wopen_node,
        critical=True
    )
    claim_wopen = (
        f"{_safe_course(ex.course_name)} is scheduled to host the 2026 U.S. Women's Open."
        if _nonempty(ex.course_name) else
        "The course is scheduled to host the 2026 U.S. Women's Open."
    )
    await evaluator.verify(
        claim=claim_wopen,
        node=wopen_leaf,
        sources=ex.urls_womens_open_2026,
        additional_instruction="The source should explicitly list the course as a host site for the 2026 U.S. Women's Open."
    )

    # 2028 Olympics (LA28)
    olympics_node = evaluator.add_parallel(
        id="Olympics_2028",
        desc="Verify the course is scheduled to host golf competitions for the 2028 Olympic Games",
        parent=future_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(ex.urls_olympics_2028) > 0,
        id="Olympics_Reference_URL",
        desc="Provide a reference URL confirming the 2028 Olympic golf hosting",
        parent=olympics_node,
        critical=True
    )

    both_leaf = evaluator.add_leaf(
        id="Both_Mens_Womens",
        desc="The course will host both men's and women's Olympic golf competitions",
        parent=olympics_node,
        critical=True
    )
    claim_both = (
        f"{_safe_course(ex.course_name)} will host both the men's and women's golf competitions for the 2028 Olympics."
        if _nonempty(ex.course_name) else
        "The course will host both the men's and women's golf competitions for the 2028 Olympics."
    )

    la28_leaf = evaluator.add_leaf(
        id="Olympics_2028_Location",
        desc="The Olympics are the Los Angeles 2028 Games",
        parent=olympics_node,
        critical=True
    )
    claim_la28 = "These are the Los Angeles 2028 Olympic Games (LA28)."

    await evaluator.batch_verify(
        [
            (
                claim_both,
                ex.urls_olympics_2028,
                both_leaf,
                "Verify the page states that both men's and women's golf competitions will be hosted at the course."
            ),
            (
                claim_la28,
                ex.urls_olympics_2028,
                la28_leaf,
                "Verify that the Olympics referred to are LA28 (Los Angeles 2028)."
            ),
        ]
    )

    # 2031 U.S. Open
    open_2031_node = evaluator.add_parallel(
        id="US_Open_2031",
        desc="Verify the course is scheduled to host the 2031 US Open Championship",
        parent=future_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(ex.urls_2031_us_open) > 0,
        id="US_Open_2031_Reference_URL",
        desc="Provide a reference URL confirming the 2031 US Open scheduling and dates",
        parent=open_2031_node,
        critical=True
    )

    # Date check: June 12–15, 2031
    date_leaf = evaluator.add_leaf(
        id="Dates_June_12_15_2031",
        desc="The 2031 US Open is scheduled for June 12-15, 2031",
        parent=open_2031_node,
        critical=True
    )
    claim_dates = (
        f"The 2031 U.S. Open at {_safe_course(ex.course_name)} is scheduled for June 12-15, 2031."
        if _nonempty(ex.course_name) else
        "The 2031 U.S. Open is scheduled for June 12-15, 2031."
    )

    # 131st championship ordinal check
    ordinal_leaf = evaluator.add_leaf(
        id="131st_Championship",
        desc="This will be the 131st US Open Championship",
        parent=open_2031_node,
        critical=True
    )
    claim_ordinal = "The 2031 U.S. Open will be the 131st U.S. Open Championship."

    await evaluator.batch_verify(
        [
            (
                claim_dates,
                ex.urls_2031_us_open,
                date_leaf,
                "Verify that the announcement explicitly lists the dates June 12–15, 2031. Allow minor punctuation variants (e.g., en-dash)."
            ),
            (
                claim_ordinal,
                ex.urls_2031_us_open,
                ordinal_leaf,
                "Verify that the announcement states it will be the 131st U.S. Open."
            ),
        ]
    )

    return future_node


async def build_annual_pga_event_nodes(evaluator: Evaluator, parent, ex: CourseExtraction):
    """
    Annual_PGA_Tour_Event subtree:
      - Genesis Invitational 'current host', 'formerly known as LA Open', 'first hosted 1929' + URL existence
    """
    annual_node = evaluator.add_parallel(
        id="Annual_PGA_Tour_Event",
        desc="Verify the course hosts the Genesis Invitational as its annual PGA Tour event",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(ex.urls_genesis_invitational) > 0,
        id="Genesis_Invitational_Reference_URL",
        desc="Provide a reference URL confirming the Genesis Invitational is held at this course",
        parent=annual_node,
        critical=True
    )

    current_leaf = evaluator.add_leaf(
        id="Genesis_Invitational_Current",
        desc="The course currently hosts the Genesis Invitational",
        parent=annual_node,
        critical=True
    )
    claim_current = (
        f"{_safe_course(ex.course_name)} currently hosts the Genesis Invitational on the PGA Tour."
        if _nonempty(ex.course_name) else
        "The course currently hosts the Genesis Invitational on the PGA Tour."
    )

    former_leaf = evaluator.add_leaf(
        id="Formerly_LA_Open",
        desc="The event was formerly known as the Los Angeles Open",
        parent=annual_node,
        critical=True
    )
    claim_former = "The Genesis Invitational was formerly known as the Los Angeles Open."

    first_1929_leaf = evaluator.add_leaf(
        id="First_Hosted_1929",
        desc="The course first hosted this tournament in 1929",
        parent=annual_node,
        critical=True
    )
    claim_1929 = (
        f"{_safe_course(ex.course_name)} first hosted this PGA Tour event in 1929."
        if _nonempty(ex.course_name) else
        "The course first hosted this PGA Tour event in 1929."
    )

    await evaluator.batch_verify(
        [
            (claim_current, ex.urls_genesis_invitational, current_leaf, "Verify the course is the current host venue of the Genesis Invitational."),
            (claim_former, ex.urls_genesis_invitational, former_leaf, "Verify that the tournament was formerly called the Los Angeles Open."),
            (claim_1929, ex.urls_genesis_invitational, first_1929_leaf, "Verify that the first year the course hosted this event was 1929."),
        ]
    )

    return annual_node


async def build_course_history_nodes(evaluator: Evaluator, parent, ex: CourseExtraction):
    """
    Course_History_and_Design subtree:
      - Opening year (1927)
      - Designers (Thomas primary, Bell assistant)
      - Location (Pacific Palisades, CA)
      + URL existence
    """
    history_node = evaluator.add_parallel(
        id="Course_History_and_Design",
        desc="Verify the course's historical background, design, and specifications",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(ex.urls_course_history) > 0,
        id="Course_History_Reference_URL",
        desc="Provide a reference URL confirming the course's design, opening year, and location",
        parent=history_node,
        critical=True
    )

    # Opening year 1927
    open_leaf = evaluator.add_leaf(
        id="Opening_Year_1927",
        desc="The course opened in 1927",
        parent=history_node,
        critical=True
    )
    claim_open = "The course opened in 1927."

    # Designers
    thomas_leaf = evaluator.add_leaf(
        id="Primary_Designer_Thomas",
        desc="George C. Thomas Jr. was the primary designer",
        parent=history_node,
        critical=True
    )
    claim_thomas = "George C. Thomas Jr. was the primary designer of the course."

    bell_leaf = evaluator.add_leaf(
        id="Assistant_Designer_Bell",
        desc="William P. Bell assisted with the design",
        parent=history_node,
        critical=True
    )
    claim_bell = "William P. Bell assisted with the design of the course."

    # Location
    location_leaf = evaluator.add_leaf(
        id="Location_Pacific_Palisades",
        desc="The course is located in Pacific Palisades, California",
        parent=history_node,
        critical=True
    )
    claim_loc = "The course is located in Pacific Palisades, California."

    await evaluator.batch_verify(
        [
            (claim_open, ex.urls_course_history, open_leaf, "Verify the page states the opening year as 1927."),
            (claim_thomas, ex.urls_course_history, thomas_leaf, "Verify the page credits George C. Thomas Jr. as primary designer."),
            (claim_bell, ex.urls_course_history, bell_leaf, "Verify the page credits William P. Bell as assisting with design."),
            (claim_loc, ex.urls_course_history, location_leaf, "Verify the page lists Pacific Palisades, California as the course location."),
        ]
    )

    return history_node


def _safe_course(name: Optional[str]) -> str:
    return name if _nonempty(name) else "the course"


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
    Evaluate an answer for the golf course identification and evidence task.
    """
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
        default_model=model
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_course_info(),
        template_class=CourseExtraction,
        extraction_name="course_info"
    )

    # Build tree according to rubric
    golf_node = await build_top_level_course_nodes(evaluator, root, extracted)

    # Children of Golf_Course_Identification (all critical, parallel)
    await build_past_major_championships_nodes(evaluator, golf_node, extracted)
    await build_future_events_nodes(evaluator, golf_node, extracted)
    await build_annual_pga_event_nodes(evaluator, golf_node, extracted)
    await build_course_history_nodes(evaluator, golf_node, extracted)

    # Return summary
    return evaluator.get_summary()