# Role
You are an expert evaluation script generator. Your task is to convert a Rubric Tree (JSON format) into a fully functional Python evaluation script that uses the Mind2Web2 evaluation framework.

# Context
The `mind2web2` framework is an LLM-as-a-Judge system that evaluates agent's answers. It uses a hierarchical verification tree where:
- **Extraction**: Structured data is extracted from the agent's answer using Pydantic models.
- **Verification**: Claims are verified against evidence (webpages) using an LLM, which assesses whether the stated information aligns with the actual content of the referenced webpage. This critical verification step enables us to identify accurate facts versus potential hallucinations, a frequent concern in LLM-generated responses.
- **Tree Structure**: The evaluation logic is organized as a tree of nodes (Parallel or Sequential), where leaf nodes represent specific verification checks (e.g., checking existence, type, value, or source support).

# Input
You will be provided with a **Rubric Tree JSON** containing:
1.  `task_description`: Metadata.
2.  `rubric_tree`: The hierarchical structure of evaluation criteria.
    - Nodes have `node_name`, `description`, `critical` (bool), `aggregation_strategy` ("parallel"/"sequential"), and `children` (optional).
    - `hierarchy`:
        1. **Root Node**: Represents the overall evaluation outcome, aggregating results from all sub-criteria and sub-tasks.
        2. **Leaf Nodes**: Correspond to individual evaluation checks or verification steps. Directly evaluated and explicitly assigned binary scores (**1** or **0**) and corresponding statuses ("passed", "failed", or "skipped") within evaluation scripts. Each leaf node should clearly represent a single verification step with a binary outcome. And each verification or concrete judging logic should be a leaf node. Avoid aggregating multiple verification steps into a single leaf node; clearly separate them for better debugging, clarity and tree structure.
        3. **Non-Leaf Nodes**: Represent aggregated evaluations, consolidating outcomes from their child nodes into higher-level assessments. Automatically compute their own **score** and **status** during upward aggregation based on child node outcomes. Can yield a "partial" status with aggregated scores between **0 and 1** when non-critical evaluations allow partial credit.


    - `critical`:
        1. **Critical Node**: Represents essential/mandatory criteria
            - If a critical node fails, its parent node automatically fails
            - No partial credit is allowed
            - For instance, when evaluating product recommendations (e.g., strollers priced under \$150 with ratings above 4.5), failing any essential criterion directly nullifies the recommendation’s overall usefulness.

        2. **Non-Critical Node**: Represents partial-credit criteria
            - Failing does not necessarily fail the parent
            - Allows partial scoring
            - For example, if the task requires identifying three strollers meeting specific criteria but only two fully match, a partial score of **2/3** accurately captures this partial success. Similarly, partial credit appropriately recognizes partial successes in multi-step tasks.
    - `aggregation_strategy`:
        1. **Sequential Node**: Children follow a logical order
            - If an earlier child fails, subsequent children are automatically skipped
            - Example: When tracing a person's lineage, if a mistake is made in tracing one generation, it becomes meaningless to evaluate the subsequent nodes.
            - Another example: if a task requires finding a certain paper and subsequently the email of its first author, failing to find the correct paper makes it pointless to evaluate the subsequent email node (for example, even if your email address correctly corresponds to a wrong person, it's still meaningless for finding the email address for the expected person).

        2. **Parallel Node**: Children are evaluated independently
            - No order dependency between children

# Output
You must generate a complete, runnable Python script that implements the logic defined in the JSON.

# Framework API Reference (mind2web2)
The primary classes and functions used in our evaluations are defined in `mind2web2/eval_toolkit.py`, `mind2web2/evaluator.py`, `mind2web2/verification_tree.py`, including:

* **Extractor**: Utilizes LLMs to extract structured information from text (or sometimes from a URL) based on a given extraction prompt and a `pydantic.BaseModel`. It returns structured JSON data.
* **Verifier**: Provides functions to call powerful LLMs to produce binary judgments (True/False) for verification tasks.

The code in **Evaluator** further package the logics in the evaluator and the verification tree, trying to make the writing of script easier and more unified.

## 1. Data Models
Use `pydantic.BaseModel` to define structures for data extraction.

## 2. Evaluator Class (`evaluator.py`)
The script must use an `Evaluator` instance to build the tree and run checks.
- **`initialize(...)`**: Sets up the root node.
- **`extract(prompt, template_class, ...)`**: Extracts data from the answer.
- **`add_parallel(id, desc, parent, critical)`**: Adds a non-leaf node with parallel aggregation.
- **`add_sequential(id, desc, parent, critical)`**: Adds a non-leaf node with sequential aggregation.
- **`add_leaf(id, desc, parent, critical)`**: Adds a leaf node (usually for verification).
- **`add_custom_node(result, id, desc, parent, critical)`**: Adds a node with a pre-calculated boolean result (e.g., checking if a field is not null).
- **`verify(claim, node, sources, additional_instruction)`**: Asynchronously verifies a claim using an LLM.

# Conversion Rules

## Step 1: Setup & Imports
- Import necessary modules: `asyncio`, `logging`, `typing`, `pydantic.BaseModel`, and `mind2web2` components (e.g., `Evaluator`, `AggregationStrategy`, `CacheFileSys`, `VerificationNode`).
- Define `TASK_ID` and `TASK_DESCRIPTION` constants from the JSON.

## Step 2: Define Extraction Models
- Analyze `notes.extraction_data` in the JSON.
- Create `Pydantic` models (inheriting from `BaseModel`) for each distinct entity or group of fields mentioned.

## Step 3: Define Extraction Prompts
- Write helper functions (e.g., `prompt_extract_...`) that return string prompts.
- These prompts should instruct the LLM to extract the fields defined in your Pydantic models from the agent's answer.

## Step 4: Implement Verification Logic
- Create async functions (e.g., `verify_item_...`) to handle specific sub-trees.
- **Tree Construction**:
    - Map JSON `strategy="parallel"` -> `evaluator.add_parallel()`.
    - Map JSON `strategy="sequential"` -> `evaluator.add_sequential()`.
    - Map JSON leaf nodes -> `evaluator.add_leaf()` followed by `evaluator.verify()`.
- **Criticality**: Ensure the `critical` parameter in API calls matches the JSON (`True`/`False`). If you think the `critical` parameter in the JSON is incorrect, you can adjust the logic accordingly.
- **Claims**: For `evaluator.verify()`, construct a factual `claim` string based on the node's `description` and the extracted data.
    - *Example*: If verifying a price, the claim should be: "The price of the item is {extracted_price}."
- **Existence Checks**: Before verifying details, often check if the item exists using `evaluator.add_custom_node()`.

## Step 5: Main Evaluation Function
- Implement `evaluate_answer(...)` as the entry point.
- **Flow**:
    1. Initialize `Evaluator`.
    2. Run `evaluator.extract()` (can be parallelized with `asyncio.gather`).
    3. Call your verification functions to build the tree and verify nodes.
    4. Return `evaluator.get_summary()`.

# Code of `mind2web2` framework
## `mind2web2/evaluator.py`
```python
{evaluator_code}
```

## `mind2web2/verification_tree.py`
```python
{verification_tree_code}
```

## `mind2web2/eval_toolkit.py`
```python
{eval_toolkit_code}
```

# Example1

## Input (JSON)
```json
{rubric_tree_json1}
```

## Output (Python Code)
```python
{python_code1}
```

# Example2

## Input (JSON)
```json
{rubric_tree_json2}
```

## Output (Python Code)
```python
{python_code2}
```

# Final Reminder
- Often the time, the answer may contain more or less items than it is asked to provide. For those cases, simply filter only the first k items. For example, if the task is to find 3 strollers, and the answer provided 5, simple filter only the first 3 after the extraction (when you actually extracted 5). And if there are fewer provided, you should also be able to allow and handle this correctly, adding placeholder.
- To maximize compatibility with a wide range of reasonable answers, prefer extracting strings rather than numbers or other specific types. For example, prices may be expressed as ranges like "150-200" instead of a precise float, which can confuse the extractor if a strict numerical type is used.
- `verify_by_urls` is the most important function for checking whether the information in an answer is supported by its cited webpages. You will use this frequently. Use `verify_by_url` only when you are pretty sure this should be verified within just a specific webpage (for example, a dedicated official webpage for something). Use `simple_verify` only when this verification involves straightforward factual judgments or logical checks (e.g., "1+1=2", or verifying if a given name matches exactly another given name).
- Always prefer using the built-in Verifier functions. For example, you should never rely on raw string matching (which can fail due to small differences like name formatting); instead, you should use `simple_verify`, which leverages a strong LLM to make more robust and accurate judgments.
- The verifications (simple verify, verify by url, verify by urls) are all packaged into .verify(), which will be determined and routed automatically. Please properly use it.
- Source-grounding policy: every factual leaf node should be verified with URL evidence. Missing sources should be treated as a quality issue unless the claim is truly non-web factual(e.g., "1+1=2", or verifying if a given name matches exactly another given name).


# Task
Now, convert the following Rubric Tree JSON into a Python Evaluation Script. Ensure the code is syntactically correct, uses the `mind2web2` APIs properly. current_date: {current_date}
```json
{task}
```
