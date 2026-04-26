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
TASK_ID = "ai_coding_assistants_2024_2025"
TASK_DESCRIPTION = """
A software development organization is building their AI-assisted development toolkit for 2024-2025 and needs to select 4 different AI coding assistant tools, each serving a distinct role in their workflow. Identify the 4 tools that meet the following requirements:

Tool 1 - Editor Assistant for Inline Coding:
- Must provide inline code suggestions and auto-completion within the IDE
- Must integrate with either Visual Studio Code or JetBrains IDEs
- Must be able to generate complete functions from comments or descriptions
- Must support multiple programming languages
- Must cost $20 per month or less for individual users

Tool 2 - Repository-Level Agent for Refactoring:
- Must be capable of performing multi-file edits and refactoring across an entire codebase
- Must provide project-level or repository-level context awareness (not just file-level)
- Must be able to run commands or tests as part of its workflow
- Must show proposed changes as diffs before applying them
- Must have an editor or IDE integration mechanism
- Must have been available for use in 2024 or later

Tool 3 - Security and Pre-Merge Review Tool:
- Must detect security vulnerabilities in code
- Must be able to identify specific vulnerability types such as Cross-Site Scripting (XSS), injection flaws, or unsafe data flows
- Must operate at the pre-merge or pull request review stage
- Must integrate with pull request or CI/CD workflows
- Must provide structured reports or findings for identified issues
- Must be capable of tracing data flow paths from input sources to execution points

Tool 4 - Local Execution Tool for Privacy-Conscious Development:
- Must support running LLM models locally on the developer's machine
- Must provide privacy and data security benefits through local execution
- Must be able to function without requiring constant internet connectivity for code generation
- Must offer a free tier with meaningful functionality (not just a trial period)
- Must provide code snippet saving, sharing, or management features beyond basic code generation

For each tool, provide:
1. The tool name
2. A description of how it meets the specified requirements
3. Reference URLs that document the tool's capabilities and pricing/availability
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Tool1Extraction(BaseModel):
    name: Optional[str] = None
    inline_suggestions: Optional[str] = None
    ide_integration: Optional[str] = None
    function_generation: Optional[str] = None
    multi_language: Optional[str] = None
    pricing_model: Optional[str] = None
    capability_urls: List[str] = Field(default_factory=list)
    pricing_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


class Tool2Extraction(BaseModel):
    name: Optional[str] = None
    multi_file: Optional[str] = None
    codebase_context: Optional[str] = None
    command_execution: Optional[str] = None
    editor_integration: Optional[str] = None
    diff_display: Optional[str] = None
    availability: Optional[str] = None
    capability_urls: List[str] = Field(default_factory=list)
    usage_urls: List[str] = Field(default_factory=list)
    integration_urls: List[str] = Field(default_factory=list)
    availability_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


class Tool3Extraction(BaseModel):
    name: Optional[str] = None
    vulnerability_detection: Optional[str] = None
    vulnerability_types: Optional[str] = None
    review_timing: Optional[str] = None
    pr_integration: Optional[str] = None
    reporting: Optional[str] = None
    data_flow_analysis: Optional[str] = None
    security_urls: List[str] = Field(default_factory=list)
    integration_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


class Tool4Extraction(BaseModel):
    name: Optional[str] = None
    local_execution: Optional[str] = None
    privacy_benefit: Optional[str] = None
    offline_capability: Optional[str] = None
    cost_model: Optional[str] = None
    free_tier: Optional[str] = None
    code_management: Optional[str] = None
    local_urls: List[str] = Field(default_factory=list)
    cost_urls: List[str] = Field(default_factory=list)
    code_mgmt_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


class ToolsExtraction(BaseModel):
    tool1: Optional[Tool1Extraction] = None
    tool2: Optional[Tool2Extraction] = None
    tool3: Optional[Tool3Extraction] = None
    tool4: Optional[Tool4Extraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tools() -> str:
    return """
    Extract structured information for exactly four AI coding assistant tools mentioned in the answer, mapped to the following roles:

    tool1 (Editor Assistant for Inline Coding):
      - name
      - inline_suggestions: short evidence phrase from the answer (if present)
      - ide_integration: short evidence phrase from the answer (if present)
      - function_generation: short evidence phrase from the answer (if present)
      - multi_language: short evidence phrase from the answer (if present)
      - pricing_model: short evidence phrase from the answer (if present)
      - capability_urls: array of URLs that document inline/IDE capability, function gen, multi-language
      - pricing_urls: array of URLs that document pricing and plans
      - other_urls: any additional relevant URLs for tool1

    tool2 (Repository-Level Agent for Refactoring):
      - name
      - multi_file: short evidence phrase about multi-file edits/refactoring
      - codebase_context: short evidence phrase about project/repository-level context awareness
      - command_execution: short evidence phrase about running commands/tests
      - editor_integration: short evidence phrase about editor/IDE integration
      - diff_display: short evidence phrase about showing diffs before applying changes
      - availability: short evidence phrase indicating availability in 2024 or later
      - capability_urls: URLs documenting repository-level or multi-file capabilities
      - usage_urls: URLs showing examples/docs of multi-file editing features or workflows
      - integration_urls: URLs documenting editor/IDE or developer workflow integration
      - availability_urls: URLs documenting availability timeframe (e.g., release notes, docs updated in 2024+)
      - other_urls: any additional relevant URLs for tool2

    tool3 (Security and Pre-Merge Review Tool):
      - name
      - vulnerability_detection: short evidence phrase about detecting vulnerabilities
      - vulnerability_types: short evidence phrase mentioning types like XSS, injection, unsafe data flows
      - review_timing: short phrase indicating pre-merge/PR stage
      - pr_integration: short phrase indicating PR/CI/CD integration
      - reporting: short phrase indicating structured reports/findings
      - data_flow_analysis: short phrase about data flow tracing from sources to sinks
      - security_urls: URLs documenting security scanning/detection features
      - integration_urls: URLs documenting PR or CI/CD integration and reporting
      - other_urls: any additional relevant URLs for tool3

    tool4 (Local Execution Tool for Privacy-Conscious Development):
      - name
      - local_execution: short evidence phrase about running LLMs locally
      - privacy_benefit: short phrase about privacy and data security via local execution
      - offline_capability: short phrase about functioning without constant internet
      - cost_model: short phrase about cost structure
      - free_tier: short phrase indicating a free tier with meaningful functionality
      - code_management: short phrase about code snippet saving/sharing/management features
      - local_urls: URLs documenting local execution/offline/privacy features
      - cost_urls: URLs documenting pricing or free tier
      - code_mgmt_urls: URLs documenting code snippet management features
      - other_urls: any additional relevant URLs for tool4

    URL extraction rules:
    - Extract only URLs explicitly present in the answer.
    - If a URL fits multiple categories, include it in all relevant arrays.
    - Ensure all URLs are full and valid (prepend http:// if protocol is missing).
    - If a field is not present in the answer, set it to null (or empty array for URL lists).

    Return a JSON object with fields: tool1, tool2, tool3, tool4, each following the schemas above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def unify_urls(*url_lists: List[List[str]]) -> List[str]:
    """Merge multiple URL lists into a de-duplicated list while preserving order."""
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            u_clean = u.strip()
            if not u_clean:
                continue
            if u_clean not in seen:
                seen.add(u_clean)
                merged.append(u_clean)
    return merged


async def verify_or_fail_if_no_urls(
    evaluator: Evaluator,
    node,
    claim: str,
    urls: List[str],
    add_ins: str,
) -> None:
    """Enforce source-grounding: if no URLs, mark as failed; otherwise verify with URLs."""
    if urls and len(urls) > 0:
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction=add_ins,
        )
    else:
        node.score = 0.0
        node.status = "failed"


# --------------------------------------------------------------------------- #
# Verification builders per tool                                              #
# --------------------------------------------------------------------------- #
async def verify_tool_1(evaluator: Evaluator, parent) -> None:
    info: Tool1Extraction = evaluator.extractor and evaluator._extraction_results and evaluator._extraction_results[-1]["result"].get("tool1")  # type: ignore
    # To be robust across different evaluator ordering, fetch from latest recorded extraction in function args instead.
    # We'll pass the parsed object directly instead of relying on this hack.
    pass


async def build_tool_1(
    evaluator: Evaluator,
    parent,
    t: Optional[Tool1Extraction],
) -> None:
    name = (t.name or "the tool").strip() if t else "the tool"
    tool_node = evaluator.add_parallel(
        id="tool_1_editor_assistant",
        desc="Identify an editor assistant tool for inline code generation and IDE integration",
        parent=parent,
        critical=False  # Adjusted to allow partial credit at top level
    )

    # Capabilities group
    caps_node = evaluator.add_parallel(
        id="tool_1_capabilities",
        desc="Verify the tool provides editor assistant capabilities",
        parent=tool_node,
        critical=True
    )

    # URLs for capabilities
    cap_urls = unify_urls(
        t.capability_urls if t else [],
        t.other_urls if t else []
    )

    # Inline suggestions
    n_inline = evaluator.add_leaf(
        id="tool_1_inline_suggestions",
        desc="The tool provides inline code suggestions and auto-completion",
        parent=caps_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_inline,
        claim=f"The tool {name} provides inline code suggestions and auto-completion within the IDE (in-editor ghost text or similar).",
        urls=cap_urls,
        add_ins="Accept synonyms like 'inline suggestions', 'ghost text', or 'autocomplete'. Focus on IDE in-editor completions."
    )

    # IDE integration
    n_ide = evaluator.add_leaf(
        id="tool_1_ide_integration",
        desc="The tool integrates with Visual Studio Code or JetBrains IDEs",
        parent=caps_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_ide,
        claim=f"The tool {name} integrates with either Visual Studio Code or JetBrains IDEs (via extension/plugin).",
        urls=cap_urls,
        add_ins="Accept explicit mentions of VS Code or JetBrains (IntelliJ, PyCharm, etc.)."
    )

    # Function generation
    n_fn = evaluator.add_leaf(
        id="tool_1_function_generation",
        desc="The tool can generate complete functions from comments or descriptions",
        parent=caps_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_fn,
        claim=f"The tool {name} can generate complete functions from comments, docstrings, or natural language descriptions.",
        urls=cap_urls,
        add_ins="Look for features like 'generate function from comment' or 'NL to code' within the IDE."
    )

    # Pricing group (sequential)
    pricing_node = evaluator.add_sequential(
        id="tool_1_pricing",
        desc="Verify the tool's pricing model",
        parent=tool_node,
        critical=True
    )

    price_urls = unify_urls(
        t.pricing_urls if t else [],
        t.other_urls if t else []
    )

    # Pricing model exists
    n_price_exists = evaluator.add_leaf(
        id="tool_1_pricing_model_exists",
        desc="The tool has a documented pricing model",
        parent=pricing_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_price_exists,
        claim=f"The tool {name} has a documented pricing model or plans page (including 'Free' if applicable).",
        urls=price_urls,
        add_ins="Any explicit pricing page or plan table qualifies; documentation should be official or authoritative."
    )

    # Pricing amount constraint
    n_price_amount = evaluator.add_leaf(
        id="tool_1_pricing_amount",
        desc="The tool costs $20 per month or less for individual users",
        parent=pricing_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_price_amount,
        claim=f"The tool {name} offers an individual plan priced at $20/month or less. 'Free' or open-source also satisfies this.",
        urls=price_urls,
        add_ins="Check for an individual plan price. If multiple currencies or billing terms exist, confirm monthly equivalent is <= $20."
    )

    # Multi-language support
    n_multi_lang = evaluator.add_leaf(
        id="tool_1_multi_language",
        desc="The tool supports multiple programming languages",
        parent=tool_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_multi_lang,
        claim=f"The tool {name} supports multiple programming languages.",
        urls=cap_urls,
        add_ins="Look for explicit lists of supported languages or phrases like 'multi-language support'. Two or more languages count as multiple."
    )

    # References presence checks
    refs_node = evaluator.add_parallel(
        id="tool_1_references",
        desc="Provide reference URLs documenting the tool's capabilities and pricing",
        parent=tool_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(t and t.capability_urls and len(t.capability_urls) > 0),
        id="tool_1_capability_reference",
        desc="URL documenting the tool's editor assistant capabilities",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(t and t.pricing_urls and len(t.pricing_urls) > 0),
        id="tool_1_pricing_reference",
        desc="URL documenting the tool's pricing information",
        parent=refs_node,
        critical=True
    )


async def build_tool_2(
    evaluator: Evaluator,
    parent,
    t: Optional[Tool2Extraction],
) -> None:
    name = (t.name or "the tool").strip() if t else "the tool"
    tool_node = evaluator.add_parallel(
        id="tool_2_repository_agent",
        desc="Identify a repository-level agent tool for multi-file operations and refactoring",
        parent=parent,
        critical=False
    )

    all_urls = unify_urls(
        t.capability_urls if t else [],
        t.usage_urls if t else [],
        t.integration_urls if t else [],
        t.availability_urls if t else [],
        t.other_urls if t else [],
    )

    # Capabilities group
    caps_node = evaluator.add_parallel(
        id="tool_2_capabilities",
        desc="Verify the tool provides repository-level agent capabilities",
        parent=tool_node,
        critical=True
    )

    # Multi-file edits
    n_multi_file = evaluator.add_leaf(
        id="tool_2_multi_file",
        desc="The tool can perform multi-file edits and refactoring across the codebase",
        parent=caps_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_multi_file,
        claim=f"The tool {name} can perform multi-file edits and repository-wide refactoring.",
        urls=all_urls,
        add_ins="Look for features like codebase-wide refactoring, multi-file changes, bulk edits, or repo-wide modifications."
    )

    # Repo-level context
    n_repo_ctx = evaluator.add_leaf(
        id="tool_2_codebase_context",
        desc="The tool provides project-level or repository-level context awareness",
        parent=caps_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_repo_ctx,
        claim=f"The tool {name} provides project-level or repository-level context awareness beyond single-file.",
        urls=all_urls,
        add_ins="Evidence could mention 'repository context', 'project graph', or understanding of codebase-wide dependencies."
    )

    # Command/test execution
    n_cmd = evaluator.add_leaf(
        id="tool_2_command_execution",
        desc="The tool can run commands or tests as part of its workflow",
        parent=caps_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_cmd,
        claim=f"The tool {name} can run commands or tests as part of its workflow.",
        urls=all_urls,
        add_ins="Look for 'executes shell commands', 'runs unit tests', 'builds project', or similar."
    )

    # Editor/IDE integration and diffs
    editor_node = evaluator.add_parallel(
        id="tool_2_editor_type",
        desc="Verify the tool's editor integration approach",
        parent=tool_node,
        critical=True
    )

    n_editor = evaluator.add_leaf(
        id="tool_2_editor_integration_exists",
        desc="The tool provides an editor or IDE integration mechanism",
        parent=editor_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_editor,
        claim=f"The tool {name} provides an editor/IDE integration mechanism (plugin/extension or equivalent).",
        urls=all_urls,
        add_ins="Accept integration with VS Code, JetBrains, or tight CLI/editor workflows."
    )

    n_diff = evaluator.add_leaf(
        id="tool_2_diff_display",
        desc="The tool shows proposed changes as diffs before applying them",
        parent=editor_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_diff,
        claim=f"The tool {name} shows proposed code changes as diffs prior to applying them.",
        urls=all_urls,
        add_ins="Look for 'diff preview', 'proposed changes', or 'review before apply'."
    )

    # Availability (2024 or later)
    n_avail = evaluator.add_leaf(
        id="tool_2_availability",
        desc="The tool is available for use in 2024 or later",
        parent=tool_node,
        critical=True
    )
    avail_urls = unify_urls(t.availability_urls if t else [], all_urls)
    await verify_or_fail_if_no_urls(
        evaluator,
        n_avail,
        claim=f"The tool {name} was available for use in 2024 or later.",
        urls=avail_urls,
        add_ins="Evidence can include docs/release notes dated 2024+ or statements indicating availability during/after 2024."
    )

    # References presence checks
    refs_node = evaluator.add_parallel(
        id="tool_2_references",
        desc="Provide reference URLs documenting the tool's capabilities",
        parent=tool_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(t and t.capability_urls and len(t.capability_urls) > 0),
        id="tool_2_capability_reference",
        desc="URL documenting the tool's repository-level capabilities",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(t and t.usage_urls and len(t.usage_urls) > 0),
        id="tool_2_usage_reference",
        desc="URL showing examples or documentation of the tool's multi-file editing features",
        parent=refs_node,
        critical=True
    )


async def build_tool_3(
    evaluator: Evaluator,
    parent,
    t: Optional[Tool3Extraction],
) -> None:
    name = (t.name or "the tool").strip() if t else "the tool"
    tool_node = evaluator.add_parallel(
        id="tool_3_security_review",
        desc="Identify a security or pre-merge review tool for vulnerability detection",
        parent=parent,
        critical=False
    )

    all_urls = unify_urls(
        t.security_urls if t else [],
        t.integration_urls if t else [],
        t.other_urls if t else [],
    )

    # Security capabilities
    caps_node = evaluator.add_parallel(
        id="tool_3_security_capabilities",
        desc="Verify the tool provides security scanning or pre-merge review capabilities",
        parent=tool_node,
        critical=True
    )

    n_vuln = evaluator.add_leaf(
        id="tool_3_vulnerability_detection",
        desc="The tool detects security vulnerabilities in code",
        parent=caps_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_vuln,
        claim=f"The tool {name} detects security vulnerabilities in source code.",
        urls=all_urls,
        add_ins="Look for 'SAST', 'static analysis', 'security scanner', or vulnerability detection features."
    )

    n_types = evaluator.add_leaf(
        id="tool_3_vulnerability_types",
        desc="The tool can identify specific vulnerability types such as XSS, injection flaws, or unsafe data flows",
        parent=caps_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_types,
        claim=f"The tool {name} can identify specific vulnerability types such as XSS, injection flaws, or unsafe data flows.",
        urls=all_urls,
        add_ins="The page should explicitly mention some of these types or similar categories."
    )

    n_stage = evaluator.add_leaf(
        id="tool_3_review_timing",
        desc="The tool operates before code is merged (pre-merge or PR review stage)",
        parent=caps_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_stage,
        claim=f"The tool {name} operates at the pre-merge or pull request review stage.",
        urls=all_urls,
        add_ins="Evidence can include PR checks, GitHub/GitLab integration at PR time, or blocking merges until issues are resolved."
    )

    # Integration and reporting
    integ_node = evaluator.add_parallel(
        id="tool_3_integration",
        desc="Verify the tool's integration with development workflow",
        parent=tool_node,
        critical=True
    )

    n_pr = evaluator.add_leaf(
        id="tool_3_pr_integration",
        desc="The tool integrates with pull request or CI/CD workflows",
        parent=integ_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_pr,
        claim=f"The tool {name} integrates with PR platforms or CI/CD pipelines.",
        urls=all_urls,
        add_ins="Look for GitHub/GitLab/Bitbucket PR checks or CI integrations (GitHub Actions, Jenkins, CircleCI, etc.)."
    )

    n_report = evaluator.add_leaf(
        id="tool_3_reporting",
        desc="The tool provides structured reports or findings for identified issues",
        parent=integ_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_report,
        claim=f"The tool {name} provides structured reports/findings for identified security issues.",
        urls=all_urls,
        add_ins="Accept SARIF, summary tables, annotated diffs, or comparable structured formats."
    )

    # Data flow analysis
    n_flow = evaluator.add_leaf(
        id="tool_3_data_flow_analysis",
        desc="The tool can trace data flow paths from input sources to execution points",
        parent=tool_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_flow,
        claim=f"The tool {name} can trace data flow paths from input sources to execution points (taint tracking/data flow analysis).",
        urls=all_urls,
        add_ins="Look for terms like 'data flow analysis', 'taint analysis', 'source-to-sink tracing'."
    )

    # References presence checks
    refs_node = evaluator.add_parallel(
        id="tool_3_references",
        desc="Provide reference URLs documenting the tool's security capabilities",
        parent=tool_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(t and t.security_urls and len(t.security_urls) > 0),
        id="tool_3_security_reference",
        desc="URL documenting the tool's security scanning and vulnerability detection features",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(t and t.integration_urls and len(t.integration_urls) > 0),
        id="tool_3_integration_reference",
        desc="URL documenting the tool's integration with PR or CI/CD workflows",
        parent=refs_node,
        critical=True
    )


async def build_tool_4(
    evaluator: Evaluator,
    parent,
    t: Optional[Tool4Extraction],
) -> None:
    name = (t.name or "the tool").strip() if t else "the tool"
    tool_node = evaluator.add_parallel(
        id="tool_4_local_execution",
        desc="Identify a cost-effective tool with local execution support for privacy-conscious development",
        parent=parent,
        critical=False
    )

    all_urls = unify_urls(
        t.local_urls if t else [],
        t.cost_urls if t else [],
        t.code_mgmt_urls if t else [],
        t.other_urls if t else [],
    )

    # Local capability
    local_node = evaluator.add_parallel(
        id="tool_4_local_capability",
        desc="Verify the tool supports local LLM execution",
        parent=tool_node,
        critical=True
    )

    n_local = evaluator.add_leaf(
        id="tool_4_local_execution",
        desc="The tool can run LLM models locally on the developer's machine",
        parent=local_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_local,
        claim=f"The tool {name} can run LLM models locally on the developer's machine.",
        urls=all_urls,
        add_ins="Evidence may mention CPU/GPU local inference, on-device execution, or 'runs offline' with local models."
    )

    n_priv = evaluator.add_leaf(
        id="tool_4_privacy_benefit",
        desc="The tool's local execution provides privacy and data security benefits",
        parent=local_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_priv,
        claim=f"The tool {name} provides privacy/data security benefits via local execution (data stays on device).",
        urls=all_urls,
        add_ins="Look for explicit statements about privacy, data never leaving the machine, or secure local processing."
    )

    n_offline = evaluator.add_leaf(
        id="tool_4_offline_capability",
        desc="The tool can function without requiring constant internet connectivity for code generation",
        parent=local_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_offline,
        claim=f"The tool {name} can function for code generation without constant internet connectivity.",
        urls=all_urls,
        add_ins="Evidence can include 'offline mode', 'local-only', or similar descriptions."
    )

    # Cost (sequential)
    cost_node = evaluator.add_sequential(
        id="tool_4_cost",
        desc="Verify the tool has a free tier or affordable pricing",
        parent=tool_node,
        critical=True
    )

    cost_urls = unify_urls(t.cost_urls if t else [], all_urls)

    n_cost_exists = evaluator.add_leaf(
        id="tool_4_cost_model_exists",
        desc="The tool has a documented cost structure or free tier",
        parent=cost_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_cost_exists,
        claim=f"The tool {name} has a documented cost structure or a defined free tier.",
        urls=cost_urls,
        add_ins="Accept 'open source' or 'free tier' documentation. Must be more than vague marketing."
    )

    n_free_tier = evaluator.add_leaf(
        id="tool_4_free_tier",
        desc="The tool offers a free tier with meaningful functionality (not just a trial)",
        parent=cost_node,
        critical=True
    )
    await verify_or_fail_if_no_urls(
        evaluator,
        n_free_tier,
        claim=f"The tool {name} offers a free tier with meaningful functionality beyond a time-limited trial.",
        urls=cost_urls,
        add_ins="Look for 'free tier', 'community edition', or permanent free features."
    )

    # Code snippet management features
    n_code_mgmt = evaluator.add_leaf(
        id="tool_4_code_management",
        desc="The tool provides code snippet saving, sharing, or management features beyond basic generation",
        parent=tool_node,
        critical=True
    )
    code_mgmt_urls = unify_urls(t.code_mgmt_urls if t else [], all_urls)
    await verify_or_fail_if_no_urls(
        evaluator,
        n_code_mgmt,
        claim=f"The tool {name} provides code snippet saving, sharing, or management features beyond basic code generation.",
        urls=code_mgmt_urls,
        add_ins="Accept features like snippet libraries, notebooks, history, sharing, or workspace memory."
    )

    # References presence checks
    refs_node = evaluator.add_parallel(
        id="tool_4_references",
        desc="Provide reference URLs documenting the tool's local execution and cost structure",
        parent=tool_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(t and t.local_urls and len(t.local_urls) > 0),
        id="tool_4_local_reference",
        desc="URL documenting the tool's local execution capabilities",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(t and t.cost_urls and len(t.cost_urls) > 0),
        id="tool_4_cost_reference",
        desc="URL documenting the tool's pricing or free tier availability",
        parent=refs_node,
        critical=True
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
    Evaluate an answer for selecting 4 AI coding assistant tools across distinct roles.
    """
    evaluator = Evaluator()
    # Adjusted: set root as non-critical to allow partial credit aggregation without violating
    # the critical-child consistency constraint in VerificationNode.
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
        prompt=prompt_extract_tools(),
        template_class=ToolsExtraction,
        extraction_name="tools_extraction"
    )

    # Build verification trees for each tool role
    await build_tool_1(evaluator, root, extracted.tool1 if extracted else None)
    await build_tool_2(evaluator, root, extracted.tool2 if extracted else None)
    await build_tool_3(evaluator, root, extracted.tool3 if extracted else None)
    await build_tool_4(evaluator, root, extracted.tool4 if extracted else None)

    # Optional: record a small custom info summary of tool names to aid debugging
    evaluator.add_custom_info(
        info={
            "tool1_name": extracted.tool1.name if (extracted and extracted.tool1) else None,
            "tool2_name": extracted.tool2.name if (extracted and extracted.tool2) else None,
            "tool3_name": extracted.tool3.name if (extracted and extracted.tool3) else None,
            "tool4_name": extracted.tool4.name if (extracted and extracted.tool4) else None,
        },
        info_type="extracted_tools",
        info_name="extracted_tools_summary"
    )

    return evaluator.get_summary()