# Mind2Web 2 [NeurIPS'25 D&B]

Mind2Web 2 is a benchmark for agentic search systems, featuring Agent-as-a-Judge methodology for comprehensive, rigorous, and reliable assessment on **long-horizon** and complex tasks that involve **complex and real-time information synthesis**.

<div align="center">
  <img src="./assets/mind2web2_overview.jpg" alt="Mind2Web 2 Overview" width="800"/>
  <p><em>Mind2Web 2 features realistic and diverse long-horizon web search tasks and a novel Agent-as-a-Judge framework to evaluate complex, time-varying, and citation-backed answers.</em></p>
</div>

## 🔗 Links

- [🏠 Homepage](https://osu-nlp-group.github.io/Mind2Web-2)
- [🏆 Leaderboard](https://osu-nlp-group.github.io/Mind2Web-2/#leaderboard)
- [📖 Paper](https://arxiv.org/abs/2506.21506)
- [😊 Dataset (Tasks) and Evaluation Scripts (Judge Agents)](https://huggingface.co/datasets/osunlp/Mind2Web-2)

## 🆕 Updates
- **2025/10/23**: To improve accessibility and adoption of Mind2Web 2, all evaluation scripts are released for both public dev set and test set. Check out the [Run Evaluation Locally Yourself](#-run-evaluation-locally-yourself) section for instructions.
- **2025/07/17**: Check out our [submission guideline](#-submission-guideline). We welcome all submissions and look forward to your participation!
- **2025/07/14**: The scripts of the public development set are released. Give them a try!
- **2025/06/26**: The GitHub repo is live. The manuscript is now on arXiv.


## 📥 Submission Guideline

To get answers for tasks of Mind2Web 2:
- If you are developing and testing a base model and have no agent framework at hand, you may start from go-to frameworks such as [Hugging Face's Open Deep Research](https://huggingface.co/blog/open-deep-research). You may want to do some zero-shot or few-shot prompting to let the agent better understand how to provide citations, to pass our attribution verifications in the task evaluations.
- If you have your own agent, still notice that we expect the agent to also provide **URL sources** to the critical facts included in the answers. You may also refer to the evaluation script to understand how the evaluation is conducted.

To evaluate answers from an agent system, there are mainly four steps involved:
1. Collecting answers from your agent on our [test set](https://huggingface.co/datasets/osunlp/Mind2Web-2/viewer/default/private_test_set)
2. Cache the webpages mentioned in the answers (to ensure consistency and reproducibility), where we provide the script in [Precache Webpage](#3-precache-webpages-optional-but-recommended)
3. Run the evaluation.
4. (Optional) Submit the avg. time and answers to better understand how the agent works.

For the submission, you can either:
- **(Recommended)** Submit your agent's answers along with the webpage cache to us. This ensures the best consistency between inference and evaluation. We will handle the evaluation cost for you.
- **(Also Recommended)** Run the whole evaluation yourself by following the instructions in the [next section](#-run-evaluation-locally-yourself) and submit the evaluation results to us.
- Only provide your agent's answers and let us handle the webpage caching and evaluation for you.

If you choose to submit your agent's answer, please arrange your agent's responses in the following directory structure (see [answers/example](https://github.com/OSU-NLP-Group/Mind2Web-2/tree/main/answers/example) for reference):

   ```
   <agent_name>
   ├── <task_id>
   │   ├── answer_1.md
   │   ├── answer_2.md
   │   └── ...
   └── ...
   ```

Similarly, the corresponding cache structure should be `cache/<agent_name>/` (generated automatically by the precaching script).

Compress the directories and send them to us via email: m2w2-leaderboard@googlegroups.com.

> **Note:**

> If you would like to **explore our tasks and run the evaluation locally**, please refer to the sections below for environment setup and evaluation instructions.


## 🚀 Run Evaluation Locally Yourself

### 0. Environment Setup

#### Option 1: Using uv (Recommended)

If you have [uv](https://docs.astral.sh/uv/) installed, it provides faster dependency resolution and installation:

```bash
# Automatically create virtual environment and install all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install browsers for Playwright (we use patchright for better stealth browsing)
patchright install
```

#### Option 2: Using conda + pip

```bash
# Create and activate conda environment
conda create -n mind2web2 python=3.11
conda activate mind2web2

# Install the package in development mode
pip install -e .

# Install browsers for Playwright
patchright install
```

### 1. Prepare Your Data

Organize your agent's responses in the following directory structure:

```
answers/
└── <your_agent_name>/
    └── <task_id>/
        ├── answer_1.md
        ├── answer_2.md
        └── ...
```

Each answer file should contain your agent's response in markdown format.

### 2. Set up API Keys

Configure the necessary API keys for evaluation:

```bash
# Set up environment variables for OpenAI API
export OPENAI_API_KEY="YOUR_OPENAI_KEY"

# (Optional) Environment variables for Azure OpenAI
export AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
export AZURE_OPENAI_ENDPOINT_URL="YOUR_AZURE_OPENAI_ENDPOINT_URL"
export AZURE_OPENAI_API_VERSION="2025-03-01-preview"

# (Optional, but necessary for several tasks) Tool APIs for tasks that require google map APIs
export GOOGLE_MAPS_API_KEY="YOUR_GOOGLE_MAPS_API_KEY"
```

### 3. Precache Webpages (Optional but Recommended)

*Note: This step is not required but highly recommended for reducing evaluation latency, as fetching webpages on-the-fly during evaluation can be very slow.*

Before running evaluation, please precache the webpages:

```bash
./cache_all_answers.sh <your_agent_name>
```

Some pages may fail to cache automatically due to CAPTCHAs, anti-bot protection, or login walls. We provide a **[Cache Manager](cache_manager_web/)** web tool to review and **batch-fix** these issues — it auto-detects problematic pages and lets you recapture all flagged URLs in one click using a Chrome Extension.

```bash
# Start the Cache Manager web UI (auto-opens in browser)
uv run python3 cache_manager_web/run.py <your_agent_name>
```

<div align="center">
  <img src="./assets/cache_manager_ui.png" alt="Cache Manager Web UI" width="900"/>
</div>

See the [Cache Manager README](cache_manager_web/README.md) for full documentation and Chrome Extension setup.

### 4. Run Evaluation

Download the evaluation script from [link](https://huggingface.co/datasets/osunlp/Mind2Web-2), and execute the evaluation using the `run_eval.py` script:

#### Basic Usage

```bash
# Evaluate all tasks for a specific agent
python run_eval.py --agent_name <your_agent_name>

# Evaluate a specific task
python run_eval.py --agent_name <your_agent_name> --task_id <task_id>
```

for example:

```bash
python run_eval.py --agent_name example

python run_eval.py --agent_name example --task_id yu_lineage
```

#### Advanced Configuration

- `--agent_name`: Name of your agent (required)
- `--answer_folder`: Path to directory containing answer files (default: `answers/`)
- `--eval_scripts_root`: Root directory for evaluation scripts (default: `eval_scripts/`)
- `--eval_results_root`: Root directory to save evaluation results (default: `eval_results/`)
- `--cache_root`: Root directory for caching webpages (default: `cache/`)
- `--eval_version`: Version of evaluation scripts to use (default: `2025_07_14`)
- `--task_id`: Specific task to evaluate (optional, evaluates all tasks if not provided)
- `--llm_provider`: LLM provider (`openai` or `azure_openai`, default: `openai`)
- `--max_concurrent_tasks`: Maximum concurrent task evaluations (default: 2)
- `--max_concurrent_answers`: Maximum concurrent answer evaluations per task (default: 3)
- `--max_webpage_retrieval`: Maximum concurrent webpage retrievals (default: 5)
- `--max_llm_requests`: Maximum concurrent LLM API requests (default: 30)
- `--dump_cache`: Persist cache to disk (default: True)
- `--overwrite`: Overwrite existing results

## 📝 Citation

If you find this work useful, please consider starring our repo and citing our papers:

```bibtex
@inproceedings{
    gou2025mind2web2,
    title={Mind2Web 2: Evaluating Agentic Search with Agent-as-a-Judge},
    author={Boyu Gou and Zanming Huang and Yuting Ning and Yu Gu and Michael Lin and Botao Yu and Andrei Kopanev and Weijian Qi and Yiheng Shu and Jiaman Wu and Chan Hee Song and Bernal Jimenez Gutierrez and Yifei Li and Zeyi Liao and Hanane Nour Moussa and TIANSHU ZHANG and Jian Xie and Tianci Xue and Shijie Chen and Boyuan Zheng and Kai Zhang and Zhaowei Cai and Viktor Rozgic and Morteza Ziyadi and Huan Sun and Yu Su},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2025},
    url={https://openreview.net/forum?id=AUaW6DS9si}
}
```
