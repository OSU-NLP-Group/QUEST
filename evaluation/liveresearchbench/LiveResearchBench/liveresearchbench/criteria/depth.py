"""Analysis Depth criterion - pairwise comparison evaluation."""

# Depth dimensions for pairwise comparison
DEPTH_DIMENSIONS = {
    "granularity": "Granularity of Reasoning (0-5): How specifically the report breaks down ideas into causal chains and mechanisms",
    "insight": "Multi-Layered Insight (0-5): Exploration of implications, trade-offs, and second-order effects",
    "critique": "Critical Evaluation (0-5): Questioning assumptions, weighing alternatives, probing limitations",
    "evidence": "Analytical Use of Evidence (0-5): How well evidence advances and sharpens the analysis",
    "density": "Insight Density (0-5): Concentration of substantive analysis vs filler content"
}

# JSON schema for depth comparison
DEPTH_COMPARISON_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "scores": {
            "type": "object",
            "properties": {
                "A": {
                    "type": "object",
                    "properties": {
                        "granularity": {"type": "integer", "minimum": 0, "maximum": 5},
                        "insight": {"type": "integer", "minimum": 0, "maximum": 5},
                        "critique": {"type": "integer", "minimum": 0, "maximum": 5},
                        "evidence": {"type": "integer", "minimum": 0, "maximum": 5},
                        "density": {"type": "integer", "minimum": 0, "maximum": 5},
                        "total": {"type": "integer", "minimum": 0, "maximum": 25}
                    },
                    "required": ["granularity", "insight", "critique", "evidence", "density", "total"],
                    "additionalProperties": False
                },
                "B": {
                    "type": "object",
                    "properties": {
                        "granularity": {"type": "integer", "minimum": 0, "maximum": 5},
                        "insight": {"type": "integer", "minimum": 0, "maximum": 5},
                        "critique": {"type": "integer", "minimum": 0, "maximum": 5},
                        "evidence": {"type": "integer", "minimum": 0, "maximum": 5},
                        "density": {"type": "integer", "minimum": 0, "maximum": 5},
                        "total": {"type": "integer", "minimum": 0, "maximum": 25}
                    },
                    "required": ["granularity", "insight", "critique", "evidence", "density", "total"],
                    "additionalProperties": False
                }
            },
            "required": ["A", "B"],
            "additionalProperties": False
        },
        "justification": {"type": "string", "maxLength": 500},
        "major_flaws": {
            "type": "object",
            "properties": {
                "A": {"type": "array", "items": {"type": "string"}},
                "B": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["A", "B"],
            "additionalProperties": False
        }
    },
    "required": ["winner", "scores", "justification", "major_flaws"],
    "additionalProperties": False
}


def create_depth_comparison_prompt() -> str:
    """Create the depth judge system prompt for pairwise comparison."""
    return """
You are an impartial research report evaluation expert. Your task is to compare **Report A** and **Report B** side-by-side and decide which demonstrates **greater depth of analysis**.
---

## What "Depth of Analysis" Means
Depth = *how far the report goes beyond surface description into reasoning, insight, and layered analysis.*

### Scoring Dimensions (0–5 each)
Assign each report a score on these five dimensions:

1. **Granularity of Reasoning (0–5)**  
   - How specifically does the report break down ideas?
   - 0 = Purely abstract/vague statements  
   - 3 = Some unpacking of mechanisms  
   - 5 = Consistent breakdown into specific causal chains

2. **Multi-Layered Insight (0–5)**  
   - Does the report explore implications, trade-offs, and second-order effects?
   - 0 = Surface-level restatement only  
   - 3 = Some implications explored  
   - 5 = Consistent exploration of trade-offs and deeper implications

3. **Critical Evaluation (0–5)**  
   - Does the report question assumptions, weigh alternatives, or probe limitations?
   - 0 = Passive description only  
   - 3 = Some questioning of assumptions  
   - 5 = Strong critique and scenario weighing

4. **Analytical Use of Evidence (0–5)**  
   - How well does the report use evidence to advance or sharpen its analysis?
   - 0 = Facts mentioned without connection to reasoning  
   - 3 = Some evidence linked to arguments  
   - 5 = Evidence consistently advances analysis

5. **Insight Density (0–5)**  
   - Concentration of substantive analysis vs. filler
   - 0 = Mostly filler or generic phrasing  
   - 3 = Mix of insight and filler  
   - 5 = Highly concentrated substantive analysis

---

## Decision Process

1. **Read both reports carefully**
2. **Score each dimension for A and B (0–5)**
3. **Sum the scores** (max 25 per report)
4. **Declare winner**: 
   - If score difference > 1: clear winner
   - If score difference ≤ 1: tie
5. **List major flaws** for each report (if any)
6. **Provide 2-3 sentence justification**

---

## Output Format (JSON)

```json
{
  "winner": "A" | "B" | "tie",
  "scores": {
    "A": {
      "granularity": 0-5,
      "insight": 0-5,
      "critique": 0-5,
      "evidence": 0-5,
      "density": 0-5,
      "total": 0-25
    },
    "B": { ... same structure ... }
  },
  "justification": "2-3 sentences explaining why A/B/tie",
  "major_flaws": {
    "A": ["flaw 1", "flaw 2", ...],
    "B": ["flaw 1", "flaw 2", ...]
  }
}
```

Be objective and focus on analytical depth, not writing style or length.
"""


def create_depth_user_prompt(query: str, report_a: str, report_b: str) -> str:
    """
    Create user prompt for depth comparison.
    
    Args:
        query: Original research query
        report_a: First report content
        report_b: Second report content
        
    Returns:
        Formatted user prompt
    """
    return f"""
ORIGINAL QUERY:
{query}

---

REPORT A:
{report_a}

---

REPORT B:
{report_b}

---

Please compare these two reports for depth of analysis and provide your evaluation in JSON format.
"""

