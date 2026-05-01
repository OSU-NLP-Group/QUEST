RUBRIC_REFINE_PROMPT_TEMPLATE = """# Role
You are an expert evaluator for rubric-tree quality control.

Your task is to inspect a generated rubric tree for an objective research task.
If the rubric tree is valid, return `valid`. If it has repairable issues, return
`fixable` and provide a corrected rubric tree. If it cannot be repaired from the
given question and constraints, return `unfixable`.

# Valid Rubric Requirements

1. No duplicate or redundant nodes.
2. No hard-coded answers that only appear in the solution.
3. No extra nodes that do not originate from the question or constraints.
4. Leaf nodes must be atomic and objectively verifiable.
5. The tree must not require evaluation of more than 5 explicit target items.
6. For "find all" tasks with depth greater than 2, second-level nodes should
   correspond to item slots such as "the first item" and should be non-critical.
7. Multiple independent attributes of the same entity should usually be sibling
   checks rather than an unnecessarily deep dependency chain.
8. Each node must be explicitly labeled as critical or non-critical.
9. Intermediate nodes must specify whether their children are sequential or
   parallel.

# Task Data

Question:
{question}

Constraints:
{constraints}

Solution:
{solution}

Rubric tree:
{rubric_tree}

# Output Format

Return strict JSON only.

If the rubric is valid:

{{
  "reasoning": "brief explanation",
  "decision": "valid"
}}

If the rubric is repairable:

{{
  "reasoning": "brief explanation",
  "decision": "fixable",
  "reason": "what was wrong",
  "refined_rubric_tree": {{
    "name": "root",
    "critical": true,
    "aggregation": "parallel",
    "children": []
  }}
}}

If the rubric cannot be repaired:

{{
  "reasoning": "brief explanation",
  "decision": "unfixable",
  "reason": "why it cannot be repaired"
}}
"""
