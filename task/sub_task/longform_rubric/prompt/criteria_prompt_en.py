# Four evaluation dimensions for final article quality
# 1. Comprehensiveness: Article covers key areas of the industry, ensures overall understanding, doesn't omit important parts
# 2. Insight/Depth: Article deeply analyzes causes, impacts and trends, provides valuable insights
# 3. Instruction-Following/Relevance: Article closely follows research topic, directly answers questions
# 4. Readability: Article has clear structure, fluent language, and is easy to understand

# Prompt to generate evaluation dimension weights
generate_eval_dimension_weight_prompt = """
<system_role>
You are an experienced research article evaluation expert. You excel at deeply understanding the objectives, challenges, and core value points of specific research tasks, and based on this, setting **dynamic, reasonable, and well-supported** dimension weights for subsequent article quality assessment.
</system_role>

<user_prompt>
There is a deep research task as follows:
<task>
"{task_prompt}"
</task>

<instruction>
**Background**: The research team will conduct in-depth and comprehensive research based on the `<task>` above and ultimately produce a high-quality research article.
**Your Task**: As an evaluation expert, you need to set the evaluation criteria weights for this specific `<task>` for our assessment team. The evaluation will be conducted across the following four dimensions:
1.  **Comprehensiveness:** The breadth, depth, and relevance of information coverage.
2.  **Insight:** The depth, originality, logic, and value of the analysis and conclusions.
3.  **Instruction Following:** Whether the report accurately and completely responds to all requirements and constraints of the task.
4.  **Readability:** Clarity of structure, fluency of language, effectiveness of data presentation, and overall ease of understanding.

**Evaluation Formula**: Total Score = Comprehensiveness * Comprehensiveness Weight + Insight * Insight Weight + Instruction Following * Instruction Following Weight + Readability * Readability Weight. (**Note: The sum of all weights must be exactly 1.0**)

**Core Requirements**:
1.  **In-depth Task Analysis**: Carefully study the specific content of the `<task>`, its implicit goals, potential difficulties, and the core value of its outcomes.
2.  **Dynamic Weight Allocation**: Based on your analysis, assign weights to the four dimensions (use decimals between 0 and 1, e.g., 0.3). **The key is to understand that different tasks have different focuses, and weights must be flexibly adjusted according to task characteristics, not fixed.**
3.  **Justify Allocation Reasons**: Your analysis (`<analysis>`) **must clearly and specifically explain why each dimension is given a particular weight**, and **directly link the reasons to the requirements and characteristics of the <task>**. This is crucial for evaluating the quality of your work.
4.  **Standard Format Output**: Strictly follow the format of the example below, first outputting the `<analysis>` text with detailed reasons, and then immediately providing the `<json_output>` with the weight allocation results.

</instruction>

<examples_rationale>
The following two examples are provided to demonstrate **how to adjust evaluation dimension weights and explain the reasons based on changes in task nature**. Please focus on learning the **thinking logic and analytical methods** in these examples, rather than simply imitating their content or weight values.
</examples_rationale>

<example_1>
<task>
"Analyze the feasibility of investing in electric vehicle (EV) charging infrastructure in suburban areas."
</task>
<output>
<analysis>
This task's core is to provide a clear feasibility analysis for a specific investment. The value lies in the thoroughness of the assessment and the practicality of its conclusions. Therefore, evaluation emphasizes insight and comprehensiveness.
* **Insight (0.35):** The task requires a deep analysis of feasibility, including market demand, costs, competition, and regulatory landscape. The quality of the strategic recommendations derived from this analysis is key.
* **Comprehensiveness (0.30):** A thorough investigation of all relevant factors (technical, economic, social, environmental) is crucial for a reliable feasibility study.
* **Instruction Following (0.20):** The report must specifically address EV charging infrastructure in suburban areas and focus on investment feasibility.
* **Readability (0.15):** Clearly communicating complex financial and technical analysis is important, but secondary to the depth and breadth of the study.
</analysis>
<json_output>
{{
    "comprehensiveness": 0.30,
    "insight": 0.35,
    "instruction_following": 0.20,
    "readability": 0.15
}}
</json_output>
</output>
</example_1>

<example_2>
<task>
"Provide a comprehensive overview of the historical performance of different renewable energy stocks over the past decade."
</task>
<output>
<analysis>
The core objective of this task is to deliver a broad, accurate, and decade-spanning overview of renewable energy stock performance. The emphasis is on the breadth of information, historical scope, and clear data presentation.
* **Comprehensiveness (0.40):** The task directly calls for covering "different" renewable energy stocks and a "decade" of data. The breadth and completeness of information are fundamental to the report's value, hence the highest weight.
* **Readability (0.25):** Presenting a large volume of historical financial data clearly and intuitively, enabling easy understanding and comparison, is a major challenge and key success factor for this task.
* **Instruction Following (0.20):** Ensuring the report strictly adheres to "renewable energy stocks," "past decade," and "historical performance" is a basic requirement.
* **Insight (0.15):** Summarizing trends or identifying key performance drivers based on the presented data can add value, but it's not the primary goal.
</analysis>
<json_output>
{{
    "comprehensiveness": 0.40,
    "insight": 0.15,
    "instruction_following": 0.20,
    "readability": 0.25
}}
</json_output>
</output>
</example_2>

Please strictly follow the above instructions and methods. Now, begin your work on the following specific task:
<task>
"{task_prompt}"
</task>
Please output your `<analysis>` and `<json_output>`.
</user_prompt>
"""

# Comprehensiveness
generate_eval_criteria_prompt_comp = """
<system_role>
You are an experienced research article evaluation expert. You excel at breaking down abstract evaluation dimensions (like "Comprehensiveness") into actionable, clear, and task-specific criteria, assigning appropriate weights and justifications for each.
</system_role>

<user_prompt>
**Background**: We are evaluating a deep research article written for the following task across four dimensions: Comprehensiveness, Insight, Instruction Following, and Readability.
1.  **Comprehensiveness:** The breadth, depth, and relevance of information coverage.
2.  **Insight:** The depth, originality, logic, and value of the analysis and conclusions.
3.  **Instruction Following:** Whether the report accurately and completely responds to all requirements and constraints of the task.
4.  **Readability:** Clarity of structure, fluency of language, effectiveness of data presentation, and overall ease of understanding.

<task>
"{task_prompt}"
</task>

<instruction>
**Your Goal**: For the **Comprehensiveness** dimension of this research article, develop a set of detailed, specific, and highly task-relevant evaluation criteria. You need to:
1.  **Analyze Task**: Deeply analyze the `<task>` to identify key information areas, perspectives, and depths that must be covered to achieve "comprehensiveness."
2.  **Formulate Criteria**: Based on the analysis, propose specific evaluation criteria items.
3.  **Explain Rationale**: Provide a brief explanation (`explanation`) for each criterion, stating why it is important for assessing the comprehensiveness of this `<task>`.
4.  **Assign Weights**: Assign a reasonable weight (`weight`) to each criterion, ensuring the sum of all criteria weights is exactly **1.0**. Weights should reflect the relative importance of each criterion in achieving the task's comprehensiveness goals.
5.  **Avoid Overlap**: Clearly focus on criteria related to the **Comprehensiveness** dimension, avoiding overlap with Insight, Instruction Following, or Readability.

**Core Requirements**:
1.  **Task-Centric**: Analysis, criteria, explanations, and weights must directly relate to the core requirements and characteristics of the `<task>`.
2.  **Well-Justified**: The `<analysis>` section must clearly articulate the overall thinking behind setting these criteria and weights, linking it to the `<task>`. The `explanation` for each criterion must justify its specific relevance.
3.  **Criteria Diversity**: Criteria should minimize overlap and cover all aspects of comprehensiveness as thoroughly as possible, avoiding omissions.
4.  **Reasonable Weights**: Weight allocation must be logical, reflecting the relative importance of each item within the comprehensiveness dimension.
5.  **Standard Format Output**: Strictly follow the example format below, first outputting the `<analysis>` text, then immediately providing the `<json_output>`.
</instruction>

<example_rational>
The following example demonstrates **how to formulate comprehensiveness criteria based on task requirements**. Focus on learning the **thinking logic and analytical methods** from this example, not just imitating its content or weight values.
</example_rational>

<example>
<task>
"Analyze the impact of remote work trends on commercial real estate in major US cities and recommend investment strategies."
</task>

<output>
<analysis>
To comprehensively evaluate a research article on "the impact of remote work on commercial real estate in major US cities and recommended investment strategies," considerations must span multiple dimensions. This task has a dual objective: first, to analyze the market impact, and second, to propose actionable investment strategies based on this analysis. Therefore, a comprehensive assessment must ensure the article covers key drivers of change in commercial real estate due to remote work, and thoroughly examines various investment approaches.

Specifically, evaluation criteria need to cover:
1.  **Remote Work Trends & Adoption Data**: Coverage of current and projected remote/hybrid work models, adoption rates across industries and demographics.
2.  **Impact on Commercial Real Estate Sectors**: Analysis of effects on office, retail, and industrial spaces, including vacancy rates, leasing trends, and property valuations in major US cities.
3.  **Geographical Variations**: Examination of how impacts differ across various major US cities (e.g., tech hubs vs. financial centers, downtown vs. suburban).
4.  **Economic and Social Implications**: Discussion of broader economic effects (e.g., urban development, local businesses) and social shifts.
5.  **Investment Strategy Analysis**: Evaluation of different investment strategies (e.g., repurposing properties, investing in niche CRE sectors, focusing on specific geographies) with risk/return profiles.
6.  **Data Sources and Methodology**: Consideration of the range and quality of data sources used (e.g., market reports, surveys, economic data) and the soundness of analytical methods.

Weight allocation should be balanced between the impact analysis (remote work trends, sector impacts, geographical variations) and the investment strategy section, as both are critical to fulfilling the task. Within impact analysis, specific sector impacts and geographical variations are key to actionable insights.
</analysis>
<json_output>
[
  {{
    "criterion": "Analysis of Remote Work Trends and Adoption",
    "explanation": "Assesses if the article thoroughly examines current and projected remote/hybrid work models, adoption statistics across different industries, and demographic factors influencing these trends. This forms the basis of the impact analysis.",
    "weight": 0.15
  }},
  {{
    "criterion": "Comprehensive Coverage of CRE Sector Impacts",
    "explanation": "Evaluates if the article details the impact on various commercial real estate (CRE) sectors (office, retail, industrial, etc.), including data on vacancy rates, rental trends, and property valuations in major US cities.",
    "weight": 0.20
  }},
  {{
    "criterion": "Examination of Geographical Variations and Nuances",
    "explanation": "Checks if the article analyzes how the impact of remote work on CRE differs across various major US cities, considering their unique economic structures and demographic profiles.",
    "weight": 0.15
  }},
  {{
    "criterion": "Discussion of Broader Economic and Social Consequences",
    "explanation": "Assesses coverage of wider implications, such as effects on urban planning, transportation, local economies dependent on office workers, and ancillary businesses.",
    "weight": 0.10
  }},
  {{
    "criterion": "Thorough Evaluation of Investment Strategies",
    "explanation": "Evaluates if the article comprehensively explores and assesses various investment strategies in response to the identified trends, including potential risks, returns, and suitability for different investor profiles.",
    "weight": 0.25
  }},
  {{
    "criterion": "Breadth and Quality of Data Sources and Methodologies",
    "explanation": "Assesses if the article utilizes a wide range of credible data sources (e.g., market research, economic reports, academic studies) and employs sound analytical methodologies to support its findings and recommendations.",
    "weight": 0.15
  }}
]
</json_output>
</output>
</example>

Please strictly follow the above instructions and methods. Now, begin your work on the following specific task:
<task>
"{task_prompt}"
</task>
Please output your `<analysis>` and `<json_output>`.
</user_prompt>
"""

# Insight / Depth
generate_eval_criteria_prompt_insight = """
<system_role>
You are an experienced research article evaluation expert. You excel at breaking down abstract evaluation dimensions (like "Insight") into actionable, clear, and task-specific criteria, assigning appropriate weights and justifications for each.
</system_role>

<user_prompt>
**Background**: We are evaluating a deep research article written for the following task across four dimensions: Comprehensiveness, Insight, Instruction Following, and Readability.
1.  **Comprehensiveness:** The breadth, depth, and relevance of information coverage.
2.  **Insight:** The depth, originality, logic, and value of the analysis and conclusions.
3.  **Instruction Following:** Whether the report accurately and completely responds to all requirements and constraints of the task.
4.  **Readability:** Clarity of structure, fluency of language, effectiveness of data presentation, and overall ease of understanding.

<task>
"{task_prompt}"
</task>

<instruction>
**Your Goal**: For the **Insight** dimension of this research article, develop a set of detailed, specific, and highly task-relevant evaluation criteria. You need to:
1.  **Analyze Task**: Deeply analyze the `<task>` to identify areas requiring in-depth analysis, logical deduction, viewpoint synthesis, or value judgment to demonstrate "insight."
2.  **Formulate Criteria**: Based on the analysis, propose specific criteria focusing on analytical depth, logical consistency, originality, and the value of conclusions.
3.  **Explain Rationale**: Provide a brief explanation (`explanation`) for each criterion, stating why it is important for assessing the insight of this `<task>`.
4.  **Assign Weights**: Assign a reasonable weight (`weight`) to each criterion, ensuring the sum of all criteria weights is exactly **1.0**. Weights should reflect the relative importance of each criterion in demonstrating the task's insight objectives.
5.  **Avoid Overlap**: Clearly focus on criteria related to the **Insight** dimension, avoiding overlap with Comprehensiveness, Instruction Following, or Readability.

**Core Requirements**:
1.  **Task-Centric**: Analysis, criteria, explanations, and weights must directly relate to the core requirements and characteristics of the `<task>`.
2.  **Beyond Surface-Level**: Criteria should assess analytical depth, logical rigor, originality of insights, and value of conclusions, not just information listing.
3.  **Well-Justified**: The `<analysis>` section must clearly articulate the overall thinking behind setting these criteria and weights, linking it to the `<task>`. The `explanation` for each criterion must justify its specific relevance.
4.  **Reasonable Weights**: Weight allocation must be logical, reflecting the relative importance of each item within the insight dimension.
5.  **Standard Format Output**: Strictly follow the example format below, first outputting the `<analysis>` text, then immediately providing the `<json_output>`.
</instruction>

<example_rational>
The following example demonstrates **how to formulate insight criteria based on task requirements**. Focus on learning the **thinking logic and analytical methods** from this example, not just imitating its content or weight values.
</example_rational>

<example>
<task>
"Analyze the impact of remote work trends on commercial real estate in major US cities and recommend investment strategies."
</task>

<output>
<analysis>
To evaluate the insight of a report on "the impact of remote work on commercial real estate (CRE) in major US cities and recommended investment strategies," the focus must be on the depth of analysis, logical strength, and the value of its conclusions, moving beyond simple data presentation. The task's core involves analyzing complex market shifts and providing strategic investment advice. Thus, insight is demonstrated by identifying key drivers of change, establishing rigorous analytical frameworks, performing multi-faceted risk/reward assessments for investment strategies, and offering forward-looking, actionable recommendations.

Evaluation criteria should emphasize:
1.  **Identification and Analysis of Core Drivers**: Not just listing factors, but deeply analyzing their mechanisms and relative importance in reshaping CRE markets.
2.  **Sophistication of Impact Analysis**: Assessing whether the analysis uncovers non-obvious impacts or second-order effects of remote work on different CRE segments.
3.  **Logical Rigor of Investment Recommendations**: Evaluating if investment strategies are well-reasoned, clearly linked to the impact analysis, and consider potential future scenarios.
4.  **Depth of Risk-Return Assessment**: Examining if the report provides a nuanced understanding of the risks and potential returns associated with each recommended strategy, rather than superficial assessments.
5.  **Originality and Strategic Value of Insights**: Whether the report offers unique perspectives, challenges conventional wisdom, or provides genuinely valuable strategic guidance that goes beyond common knowledge.
6.  **Future Outlook and Adaptability**: Assessing if the report considers the long-term evolution of remote work and its implications, and if investment strategies are adaptable to changing conditions.

Weight allocation should favor the depth of impact analysis and the strategic value of investment recommendations, as these are central to demonstrating insight. The ability to identify core drivers and provide a nuanced risk-return assessment are also highly important.
</analysis>
<json_output>
[
  {{
    "criterion": "Depth of Analysis of Core Drivers and Mechanisms",
    "explanation": "Assesses if the article not only identifies key drivers (e.g., technology, corporate policy, employee preference) of remote work's impact on CRE, but deeply analyzes their interplay and causal mechanisms, rather than a superficial listing.",
    "weight": 0.20
  }},
  {{
    "criterion": "Sophistication in Uncovering CRE Market Impacts",
    "explanation": "Evaluates if the analysis goes beyond obvious impacts to uncover subtle or second-order effects on CRE demand, valuation, and use in different sectors (office, retail, etc.) and geographies.",
    "weight": 0.20
  }},
  {{
    "criterion": "Logical Coherence and Justification of Investment Strategies",
    "explanation": "Assesses if the recommended investment strategies are logically derived from the impact analysis, are well-justified with supporting evidence, and consider potential market dynamics.",
    "weight": 0.20
  }},
  {{
    "criterion": "Nuance in Risk-Return Analysis for Investments",
    "explanation": "Evaluates the depth of the risk-return assessment for proposed strategies, considering factors like market volatility, liquidity, holding periods, and specific risks associated with CRE in a post-remote work era.",
    "weight": 0.15
  }},
  {{
    "criterion": "Originality and Strategic Value of Recommendations",
    "explanation": "Assesses whether the article offers novel, actionable insights or investment strategies that provide a distinct advantage or perspective, moving beyond generic advice.",
    "weight": 0.15
  }},
  {{
    "criterion": "Forward-Looking Perspective and Consideration of Long-Term Trends",
    "explanation": "Evaluates if the article incorporates a forward-looking view, anticipating future shifts in remote work and their sustained impact on CRE, and how investment strategies might adapt.",
    "weight": 0.10
  }}
]
</json_output>
</output>
</example>

Please strictly follow the above instructions and methods. Now, begin your work on the following specific task:
<task>
"{task_prompt}"
</task>
Please output your `<analysis>` and `<json_output>`.
</user_prompt>
"""

# Instruction-Following / Relevance
generate_eval_criteria_prompt_Inst = """
<system_role>
You are an experienced research article evaluation expert. You excel at breaking down abstract evaluation dimensions (like "Instruction Following") into actionable, clear, and task-specific criteria, assigning appropriate weights and justifications for each.
</system_role>

<user_prompt>
**Background**: We are evaluating a deep research article written for the following task across four dimensions: Comprehensiveness, Insight, Instruction Following, and Readability.
1.  **Comprehensiveness:** The breadth, depth, and relevance of information coverage.
2.  **Insight:** The depth, originality, logic, and value of the analysis and conclusions.
3.  **Instruction Following:** Whether the report accurately and completely responds to all requirements and constraints of the task.
4.  **Readability:** Clarity of structure, fluency of language, effectiveness of data presentation, and overall ease of understanding.

<task>
"{task_prompt}"
</task>

<instruction>
**Your Goal**: For the **Instruction Following** dimension of this research article, develop a set of detailed, specific, and highly task-relevant evaluation criteria. You need to:
1.  **Analyze Task**: Deeply analyze the specific instructions, questions, scope limitations (e.g., geography, time, subject), and core objectives within the `<task>`.
2.  **Formulate Criteria**: Based on the analysis, propose specific criteria focusing on whether the article accurately, completely, and directly responds to all task instructions, whether content strictly adheres to limitations, and if it remains on topic.
3.  **Explain Rationale**: Provide a brief explanation (`explanation`) for each criterion, stating why it is important for assessing the instruction adherence of this `<task>`.
4.  **Assign Weights**: Assign a reasonable weight (`weight`) to each criterion, ensuring the sum of all criteria weights is exactly **1.0**. Weights should reflect the relative importance of each criterion in ensuring the task is completed accurately and relevantly.
5.  **Avoid Overlap**: Clearly focus on criteria related to the **Instruction Following** dimension, avoiding overlap with Comprehensiveness, Insight, or Readability.

**Core Requirements**:
1.  **Instruction-Centric**: Analysis, criteria, explanations, and weights must directly correspond to the explicit requirements, questions, and limitations of the `<task>`.
2.  **Focus on Responsiveness and Relevance**: Criteria should assess if the content is on-topic, the scope is accurate, and all questions are directly and fully answered.
3.  **Well-Justified**: The `<analysis>` section must clearly articulate the overall thinking behind setting these criteria and weights, linking it to the `<task>`. The `explanation` for each criterion must justify its specific relevance.
4.  **Reasonable Weights**: Weight allocation must be logical, reflecting the relative importance of each instruction or limitation within the task.
5.  **Standard Format Output**: Strictly follow the example format below, first outputting the `<analysis>` text, then immediately providing the `<json_output>`.
</instruction>

<example_rational>
The following example demonstrates **how to formulate instruction following criteria based on task requirements**. Focus on learning the **thinking logic and analytical methods** from this example, not just imitating its content or weight values.
</example_rational>

<example>
<task>
"Analyze the impact of remote work trends on commercial real estate in major US cities and recommend investment strategies."
</task>

<output>
<analysis>
Evaluating the instruction-following capability for the task "Analyze the impact of remote work trends on commercial real estate in major US cities and recommend investment strategies" centers on whether the report precisely addresses all core components and constraints. The task has two primary components: 1) analyzing the impact of remote work on CRE, and 2) recommending investment strategies. Key constraints include focusing on "commercial real estate" and "major US cities."

Therefore, evaluation criteria must focus on:
1.  **Response to Impact Analysis**: Does the report directly and clearly analyze the impact of remote work on CRE?
2.  **Response to Investment Strategy Recommendation**: Does the report directly and clearly provide investment strategy recommendations based on the analysis?
3.  **Adherence to Scope (CRE)**: Is the content strictly focused on commercial real estate, not residential or other property types?
4.  **Adherence to Scope (Major US Cities)**: Is the analysis centered on major US cities as specified, avoiding irrelevant geographical focus?
5.  **Completeness of Task Components**: Does the report address both the impact analysis and the investment recommendations, without omitting a key part of the request?

Weight allocation should prioritize direct responses to the two main task components (impact analysis and strategy recommendations), as these are fundamental objectives. Adherence to scope (CRE and major US cities) is also crucial for relevance. Ensuring both components are covered addresses the completeness of the response.
</analysis>
<json_output>
[
  {{
    "criterion": "Directness and Clarity of Impact Analysis on CRE",
    "explanation": "Assesses if the article directly and clearly analyzes the impact of remote work trends specifically on commercial real estate, fulfilling the first core component of the task.",
    "weight": 0.30
  }},
  {{
    "criterion": "Directness and Clarity of Investment Strategy Recommendations",
    "explanation": "Assesses if the article directly and clearly presents investment strategies for CRE, based on the preceding impact analysis, fulfilling the second core component.",
    "weight": 0.30
  }},
  {{
    "criterion": "Strict Adherence to 'Commercial Real Estate' Focus",
    "explanation": "Evaluates if the analysis and recommendations are consistently focused on commercial real estate, avoiding significant deviation into other property types (e.g., residential).",
    "weight": 0.15
  }},
  {{
    "criterion": "Strict Adherence to 'Major US Cities' Geographical Scope",
    "explanation": "Evaluates if the analysis and examples are predominantly centered on major US cities, as per the task's geographical constraint, without excessive focus on other regions or general global trends.",
    "weight": 0.15
  }},
  {{
    "criterion": "Complete Coverage of All Task Components",
    "explanation": "Ensures the article addresses both main aspects of the task – analyzing impacts and recommending strategies – without significant omissions of either component.",
    "weight": 0.10
  }}
]
</json_output>
</output>
</example>

Please strictly follow the above instructions and methods. Now, begin your work on the following specific task:
<task>
"{task_prompt}"
</task>
Please output your `<analysis>` and `<json_output>`.
</user_prompt>
"""

# Readability - Generates general criteria and typical weights
generate_eval_criteria_prompt_readability = """
<system_role>
You are an experienced research article evaluation expert. You excel at breaking down abstract evaluation dimensions (like "Readability") into actionable, clear, and task-specific criteria, assigning appropriate weights and justifications for each.
</system_role>

<user_prompt>
**Background**: We are evaluating a deep research article written for the following task across four dimensions: Comprehensiveness, Insight, Instruction Following, and Readability.
1.  **Comprehensiveness:** The breadth, depth, and relevance of information coverage.
2.  **Insight:** The depth, originality, logic, and value of the analysis and conclusions.
3.  **Instruction Following:** Whether the report accurately and completely responds to all requirements and constraints of the task.
4.  **Readability:** Clarity of structure, fluency of language, effectiveness of data presentation, and overall ease of understanding.

<task>
"{task_prompt}"
</task>

<instruction>
**Your Goal**: For the **Readability** dimension of this research article, develop a set of detailed, specific, and relatively general evaluation criteria, while also considering the characteristics of the `<task>`. You need to:
1.  **Analyze Readability Elements**: Identify key elements that constitute the readability of a high-quality research report, such as structural logic, language expression, information presentation, formatting, etc.
2.  **Formulate Criteria**: Based on the analysis, propose specific criteria covering:
    *   Language clarity and correctness (sentences, terminology, wording, errors)
    *   Content structure and logic (headings, paragraphs, introduction/conclusion, transitions)
    *   Information presentation and density (key points, breakdown, organization, redundancy)
    *   Data and visualization (data accuracy/clarity, use of charts/tables)
    *   Formatting and layout (paragraphs, spacing, highlighting)
    *   Audience adaptation (term explanation, expression style)
3.  **Explain Rationale**: Provide a brief explanation (`explanation`) for each criterion, stating why it is important for enhancing report readability and reader comprehension, potentially linking to the `<task>` type.
4.  **Assign Weights**: Assign a reasonable weight (`weight`) to each criterion, ensuring the sum of all criteria weights is exactly **1.0**. Weights should reflect the relative importance of each criterion to overall readability.
5.  **Avoid Overlap**: Clearly focus on criteria related to the **Readability** dimension, avoiding overlap with Comprehensiveness, Insight, or Instruction Following.

**Core Requirements**:
1.  **Cover Key Elements**: Criteria should systematically cover the main aspects affecting readability.
2.  **Clear and Actionable**: Each criterion should be specific, easy to understand, and assessable.
3.  **Well-Justified**: The `<analysis>` section must articulate the overall thinking behind these criteria and weights. The `explanation` for each criterion must justify its importance.
4.  **Reasonable Weights**: Weight allocation must be logical, reflecting the relative contribution of each item to readability.
5.  **Standard Format Output**: Strictly follow the example format below, first outputting the `<analysis>` text, then immediately providing the `<json_output>`.
</instruction>

<example_rational>
The following example demonstrates **how to formulate readability criteria**. Focus on learning the **thinking logic and analytical methods** from this example, not just imitating its content or weight values. While readability criteria are generally applicable, weight allocation can be slightly adjusted based on the task type.
</example_rational>

<example>
<task>
"Analyze the impact of remote work trends on commercial real estate in major US cities and recommend investment strategies."
</task>

<output>
<analysis>
Evaluating the readability of a report on "the impact of remote work on commercial real estate (CRE) in major US cities and recommended investment strategies" requires ensuring that complex market analysis and investment comparisons are easily understood by the reader. While readability standards are generally applicable, for a task involving market analysis, data, and strategic recommendations, a clear structure, precise language, and intuitive data presentation are particularly crucial.

Evaluation criteria should cover:
1.  **Structural Logic**: Does the article have a clear framework guiding the reader from market analysis to impact assessment and finally to investment strategies?
2.  **Language and Terminology**: Is the language fluent and accurate? Is technical CRE and financial terminology used appropriately and explained where necessary?
3.  **Paragraph Organization**: Are paragraphs focused and transitions smooth, facilitating easy following of the argumentation?
4.  **Information Presentation**: Is key information highlighted? Is data presented clearly, with appropriate use of charts and tables to illustrate trends and comparisons?
5.  **Formatting and Layout**: Is the overall layout clean and conducive to sustained reading without fatigue?

Weight allocation should give the highest importance to structural clarity, as it's fundamental to understanding complex analyses. The accuracy of language and professional terminology is also vital for effective information transfer. For a data-intensive task like this, the presentation of data and visuals significantly contributes to readability. Other aspects like paragraph organization and formatting serve as support for a good reading experience.
</analysis>
<json_output>
[
  {{
    "criterion": "Overall Article Structure and Logical Flow",
    "explanation": "Assesses if the article has a clear logical structure (e.g., introduction, market overview, impact analysis, strategy recommendations, conclusion), with distinct heading levels, effectively guiding the reader through complex information.",
    "weight": 0.25
  }},
  {{
    "criterion": "Clarity, Precision, and Professionalism of Language",
    "explanation": "Evaluates if the language is fluent, grammatically correct, free of jargon where possible, and uses precise terminology; technical terms (e.g., cap rate, NOI, Class A/B/C office space) are used correctly and explained if necessary.",
    "weight": 0.20
  }},
  {{
    "criterion": "Paragraph Cohesion and Transitions",
    "explanation": "Assesses if each paragraph focuses on a single idea, and if transitions between paragraphs and sections are smooth and logical, aiding comprehension.",
    "weight": 0.15
  }},
  {{
    "criterion": "Clarity and Accuracy of Data Presentation",
    "explanation": "Evaluates if data cited (e.g., vacancy rates, rental growth, investment volumes) is presented clearly, accurately, and sourced appropriately (if applicable), making it easy to understand.",
    "weight": 0.15
  }},
  {{
    "criterion": "Effective Use of Visualizations (Charts, Tables)",
    "explanation": "Assesses if charts, graphs, and tables are used effectively to illustrate trends, comparisons, or complex data, and if they are well-designed, clearly labeled, and easy to interpret.",
    "weight": 0.10
  }},
  {{
    "criterion": "Highlighting of Key Information and Summaries",
    "explanation": "Assesses if the article effectively uses formatting (e.g., bolding, bullet points) or summary sections to highlight key findings, conclusions, and recommendations.",
    "weight": 0.05
  }},
  {{
    "criterion": "Formatting, Layout, and Overall Presentation",
    "explanation": "Evaluates if the document's layout, font choices, spacing, and general visual presentation are professional and enhance readability, reducing reader fatigue.",
    "weight": 0.05
  }},
  {{
    "criterion": "Audience Appropriateness",
    "explanation": "Assesses if the level of detail, terminology, and explanations are appropriate for the intended audience (e.g., investors, industry professionals).",
    "weight": 0.05
  }}
]
</json_output>
</output>
</example>

Please strictly follow the above instructions and methods. Now, begin your work on the following specific task:
<task>
"{task_prompt}"
</task>
Please output your `<analysis>` and `<json_output>`.
</user_prompt>
"""

