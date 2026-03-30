# Based on a given research task and merged evaluation criteria list, compare two articles by each criterion
generate_merged_score_prompt = """
<system_role>You are a strict, meticulous, and objective research article evaluation expert. You excel at using specific assessment criteria to deeply compare two articles on the same task, providing precise scores and clear justifications.</system_role>

<user_prompt>
**Task Background**
There is a deep research task, and you need to evaluate two research articles written for this task. We will assess the articles across four dimensions: Comprehensiveness, Insight, Instruction Following, and Readability. The content is as follows:
<task>
"{task_prompt}"
</task>

**Articles to Evaluate**
<article_1>
"{article_1}"
</article_1>

<article_2>
"{article_2}"
</article_2>

**Evaluation Criteria**
Now, you need to evaluate and compare these two articles based on the following **evaluation criteria list**, providing comparative analysis and scoring each on a scale of 0-10. Each criterion includes an explanation, please understand carefully.

<criteria_list>
{criteria_list}
</criteria_list>

<Instruction>
**Your Task**
Please strictly evaluate and compare `<article_1>` and `<article_2>` based on **each criterion** in the `<criteria_list>`. You need to:
1.  **Analyze Each Criterion**: Consider how each article fulfills the requirements of each criterion.
2.  **Comparative Evaluation**: Analyze how the two articles perform on each criterion, referencing the content and criterion explanation.
3.  **Score Separately**: Based on your comparative analysis, score each article on each criterion (0-10 points).

**Language Neutrality**
The choice of language (Chinese or English) does not affect the evaluation. Both Chinese and English outputs are accepted. You should judge solely based on the quality of the article.

**Scoring Rules**
For each criterion, score both articles on a scale of 0-10 (continuous values). The score should reflect the quality of performance on that criterion:
*   0-2 points: Very poor performance. Almost completely fails to meet the criterion requirements.
*   2-4 points: Poor performance. Minimally meets the criterion requirements with significant deficiencies.
*   4-6 points: Average performance. Basically meets the criterion requirements, neither good nor bad.
*   6-8 points: Good performance. Largely meets the criterion requirements with notable strengths.
*   8-10 points: Excellent/outstanding performance. Fully meets or exceeds the criterion requirements.

**Output Format Requirements**
Please **strictly** follow the `<output_format>` below for each criterion evaluation. **Do not include any other unrelated content, introduction, or summary**. Start with "Standard 1" and proceed sequentially through all criteria:
</Instruction>

<output_format>
{{
    "comprehensiveness": [
        {{
            "criterion": [Text content of the first comprehensiveness evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
}},
{{
            "criterion": [Text content of the second comprehensiveness evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
        }},
        ...
    ],
    "insight": [
        {{
            "criterion": [Text content of the first insight evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
        }},
        ...
    ],
    ...
}}
</output_format>

Now, please evaluate the two articles based on the research task and criteria, providing detailed comparative analysis and scores according to the requirements above. Ensure your output follows the specified `<output_format>` and that the JSON format is parsable, with all characters that might cause JSON parsing errors properly escaped.
</user_prompt>
"""

# Static version: uses predefined fixed evaluation criteria, no need to dynamically generate criteria
generate_static_score_prompt = """
<system_role>You are a strict, meticulous, and objective research article evaluation expert. You excel at using specific assessment criteria to deeply compare two articles on the same task, providing precise scores and clear justifications.</system_role>

<user_prompt>
**Task Background**
There is a deep research task, and you need to evaluate two research articles written for this task. We will assess the articles across four dimensions: Comprehensiveness, Insight, Instruction Following, and Readability. The content is as follows:
<task>
"{task_prompt}"
</task>

**Articles to Evaluate**
<article_1>
"{article_1}"
</article_1>

<article_2>
"{article_2}"
</article_2>

**Evaluation Criteria**
Now, you need to evaluate and compare these two articles based on the following **fixed evaluation criteria list**, providing comparative analysis and scoring each on a scale of 0-10. Each criterion includes an explanation, please understand carefully.

<criteria_list>
# Comprehensiveness
[
  {{
    "criterion": "Information Coverage Breadth",
    "explanation": "Evaluates whether the article covers all key areas and aspects related to the topic without omitting important information.",
    "weight": 0.25
  }},
  {{
    "criterion": "Information Depth and Detail",
    "explanation": "Evaluates whether the article provides sufficiently detailed information rather than just surface-level overviews.",
    "weight": 0.25
  }},
  {{
    "criterion": "Data and Factual Support",
    "explanation": "Evaluates whether the article provides sufficient data, facts, cases, or evidence to support its arguments and analysis.",
    "weight": 0.25
  }},
  {{
    "criterion": "Multiple Perspectives and Balance",
    "explanation": "Evaluates whether the article considers issues from multiple angles and provides balanced viewpoints where relevant.",
    "weight": 0.25
  }}
]

# Insight
[
  {{
    "criterion": "Analysis Depth and Originality",
    "explanation": "Evaluates whether the article provides deep analysis and original insights rather than simply repeating known information.",
    "weight": 0.25
  }},
  {{
    "criterion": "Logical Reasoning and Causal Relationships",
    "explanation": "Evaluates whether the article demonstrates clear logical reasoning and effectively explains causal relationships behind phenomena.",
    "weight": 0.25
  }},
  {{
    "criterion": "Problem Insight and Solutions",
    "explanation": "Evaluates whether the article identifies key issues or challenges and provides insightful solutions or recommendations.",
    "weight": 0.25
  }},
  {{
    "criterion": "Forward-Looking and Inspirational Thinking",
    "explanation": "Evaluates whether the article demonstrates forward-looking thinking, can anticipate trends, and provides inspiring perspectives.",
    "weight": 0.25
  }}
]

# Instruction Following
[
  {{
    "criterion": "Response to Task Objectives",
    "explanation": "Evaluates whether the article directly responds to the core objectives and questions of the task.",
    "weight": 0.34
  }},
  {{
    "criterion": "Adherence to Scope Limitations",
    "explanation": "Evaluates whether the article strictly adheres to the scope limitations set in the task (e.g., geography, time period, subjects).",
    "weight": 0.33
  }},
  {{
    "criterion": "Complete Coverage of Task Requirements",
    "explanation": "Evaluates whether the article completely covers all sub-questions or aspects raised in the task without omitting important parts.",
    "weight": 0.33
  }}
]

# Readability
[
  {{
    "criterion": "Clear Structure and Logic",
    "explanation": "Evaluates whether the article has a clear structure, including appropriate introduction, body, conclusion, and logically coherent paragraph organization.",
    "weight": 0.25
  }},
  {{
    "criterion": "Language Expression and Fluency",
    "explanation": "Evaluates whether the article's language is clear, accurate, and fluent, without obvious grammatical errors or inappropriate expressions.",
    "weight": 0.25
  }},
  {{
    "criterion": "Appropriate Use of Technical Terms",
    "explanation": "Evaluates whether the article appropriately uses technical terminology and provides explanations when necessary for understanding.",
    "weight": 0.25
  }},
  {{
    "criterion": "Information Presentation and Visual Elements",
    "explanation": "Evaluates whether the article effectively uses formatting, headings, lists, emphasis, etc. to enhance readability, and appropriately uses charts or other visual elements (if any).",
    "weight": 0.25
  }}
]
</criteria_list>

<Instruction>
**Your Task**
Please strictly evaluate and compare `<article_1>` and `<article_2>` based on **each criterion** in the `<criteria_list>`. You need to:
1.  **Analyze Each Criterion**: Consider how each article fulfills the requirements of each criterion.
2.  **Comparative Evaluation**: Analyze how the two articles perform on each criterion, referencing the content and criterion explanation.
3.  **Score Separately**: Based on your comparative analysis, score each article on each criterion (0-10 points).

**Language Neutrality**
The choice of language (Chinese or English) does not affect the evaluation. Both Chinese and English outputs are accepted. You should judge solely based on the quality of the article.

**Scoring Rules**
For each criterion, score both articles on a scale of 0-10 (continuous values). The score should reflect the quality of performance on that criterion:
*   0-2 points: Very poor performance. Almost completely fails to meet the criterion requirements.
*   2-4 points: Poor performance. Minimally meets the criterion requirements with significant deficiencies.
*   4-6 points: Average performance. Basically meets the criterion requirements, neither good nor bad.
*   6-8 points: Good performance. Largely meets the criterion requirements with notable strengths.
*   8-10 points: Excellent/outstanding performance. Fully meets or exceeds the criterion requirements.

**Output Format Requirements**
Please **strictly** follow the `<output_format>` below for each criterion evaluation. **Do not include any other unrelated content, introduction, or summary**. Start with "Standard 1" and proceed sequentially through all criteria:
</Instruction>

<output_format>
{{
    "comprehensiveness": [
        {{
            "criterion": [Text content of the first comprehensiveness evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
        }},
        {{
            "criterion": [Text content of the second comprehensiveness evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
        }},
        ...
    ],
    "insight": [
        {{
            "criterion": [Text content of the first insight evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
        }},
        ...
    ],
    ...
}}
</output_format>

Now, please evaluate the two articles based on the research task and criteria, providing detailed comparative analysis and scores according to the requirements above. Ensure your output follows the specified `<output_format>` and that the JSON format is parsable, with all characters that might cause JSON parsing errors properly escaped.
</user_prompt>
"""

# Dynamic but without reference article scoring prompt version
point_wise_score_prompt = """
<system_role>You are a strict, meticulous, and objective research article evaluation expert. You excel at using specific assessment criteria to thoroughly evaluate research articles, providing precise scores and clear justifications.</system_role>

<user_prompt>
**Task Background**
There is a deep research task, and you need to evaluate a research article written for this task. We will assess the article across four dimensions: Comprehensiveness, Insight, Instruction Following, and Readability. The content is as follows:
<task>
"{task_prompt}"
</task>

**Article to Evaluate**
<target_article>
"{article}"
</target_article>

**Evaluation Criteria**
Now, you need to evaluate this article based on the following **evaluation criteria list**, providing analysis and scoring each on a scale of 0-10. Each criterion includes an explanation, please understand carefully.

<criteria_list>
{criteria_list}
</criteria_list>

<Instruction>
**Your Task**
Please strictly evaluate `<target_article>` based on **each criterion** in the `<criteria_list>`. You need to:
1.  **Analyze Each Criterion**: Consider how the article fulfills the requirements of each criterion.
2.  **Analysis and Evaluation**: Analyze the article's performance on each criterion, referencing the content and criterion explanation, noting strengths and weaknesses.
3.  **Score**: Based on your analysis, score the article on each criterion (0-10 points).

**Language Neutrality**
The choice of language (Chinese or English) does not affect the evaluation. Both Chinese and English outputs are accepted. You should judge solely based on the quality of the article.

**Scoring Rules**
For each criterion, score the article on a scale of 0-10 (continuous values). The score should reflect the quality of performance on that criterion:
*   0-2 points: Very poor performance. Almost completely fails to meet the criterion requirements.
*   2-4 points: Poor performance. Minimally meets the criterion requirements with significant deficiencies.
*   4-6 points: Average performance. Basically meets the criterion requirements, neither good nor bad.
*   6-8 points: Good performance. Largely meets the criterion requirements with notable strengths.
*   8-10 points: Excellent/outstanding performance. Fully meets or exceeds the criterion requirements.

**Output Format Requirements**
Please **strictly** follow the `<output_format>` below for each criterion evaluation. **Do not include any other unrelated content, introduction, or summary**. Start with "Standard 1" and proceed sequentially through all criteria:
</Instruction>

<output_format>
{{
    "comprehensiveness": [
        {{
            "criterion": [Text content of the first comprehensiveness evaluation criterion],
            "analysis": [Analysis],
            "target_score": [Continuous score 0-10]
        }},
        {{
            "criterion": [Text content of the second comprehensiveness evaluation criterion],
            "analysis": [Analysis],
            "target_score": [Continuous score 0-10]
        }},
        ...
    ],
    "insight": [
        {{
            "criterion": [Text content of the first insight evaluation criterion],
            "analysis": [Analysis],
            "target_score": [Continuous score 0-10]
        }},
        ...
    ],
    ...
}}
</output_format>

Now, please evaluate the article based on the research task and criteria, providing detailed analysis and scores according to the requirements above. Ensure your output follows the specified `<output_format>` and that the JSON format is parsable, with all characters that might cause JSON parsing errors properly escaped.
</user_prompt>
"""

# Vanilla Prompt: Directly asks the model to score the article overall
vanilla_prompt = """
<system_role>You are a strict, meticulous, and objective research article evaluation expert. You excel at using specific assessment criteria to thoroughly evaluate research articles, providing precise scores and clear justifications.</system_role>

<user_prompt>
**Task Background**
There is a deep research task, and you need to evaluate a research article written for this task.
<task>
"{task_prompt}"
</task>

**Article to Evaluate**
<target_article>
"{article}"
</target_article>

<Instruction>
**Your Task**
Please evaluate the overall quality of the above `<target_article>` as a response to `<task>`.

**Language Neutrality**
The choice of language (Chinese or English) does not affect the evaluation. Both Chinese and English outputs are accepted. You should judge solely based on the quality of the article.

Please provide an overall score between 0 and 10. Also, provide a brief justification for your score.

**Output Format Requirements**
Please **strictly** follow the `<output_format>` below for your evaluation result. **Do not include any other unrelated content, introduction, or summary**.
</Instruction>

<output_format>
{{
    "overall_score": [Continuous score 0-10],
    "justification": "[Scoring justification]"
}}
</output_format>

Now, please evaluate the article based on the task and provide your score and justification according to the specified format. Ensure your output is valid JSON format and escape any special characters as needed.
</user_prompt>
"""
