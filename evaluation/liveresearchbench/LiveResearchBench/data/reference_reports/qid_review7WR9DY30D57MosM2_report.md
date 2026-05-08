# Vision–Language Models for 2D/3D Spatial Understanding (2022–2025): Architectures, Training/Alignment, and Evaluation

## Executive Summary
From 2022 to 2025, vision–language models (VLMs) advanced spatial understanding along two main tracks: (1) general‑purpose large multimodal language models (LMMs) with improved grounding interfaces and reasoning tools, and (2) task‑specific or 3D‑explicit systems that inject geometric structure (depth, point clouds, scene graphs, multi‑view consistency) into perception–reasoning pipelines. Across object counting, navigation/embodied instruction following, grounding/localization, and visual reasoning:

- In 2D, contrastive encoders (CLIP‑style) with light task heads remain strong for localized tasks (open‑vocabulary detection, counting). Encoder–decoder MLLMs gain breadth but still struggle with precise spatial reasoning without added structure.
- In 3D, explicit geometry (scene graphs, point clouds, neural fields/Gaussians, multi‑view/diffusion priors) significantly improves embodied navigation and spatial planning.
- Instruction tuning, structured outputs (boxes/masks), and “visual chain‑of‑thought” tools consistently raise spatial performance versus vanilla language‑only strategies.
- Benchmark studies show persistent gaps in top‑view layout reasoning and object grounding/hallucination control, motivating stronger spatial priors and evaluation protocols.

The sections below synthesize architectural paradigms, training/alignment choices, and evaluation ecosystems, and compare general‑purpose vs task‑specific models with trends, trade‑offs, and open challenges.

## Scope and Task Definitions
- Venues: ICML, NeurIPS, ICLR, EMNLP, ACL, AAAI, TPAMI, COLING, TMLR, IJCV (official conference/journal pages only; no arXiv‑only preprints).
- Tasks:
  - Object counting
  - Navigation / embodied instruction following
  - Grounding / localization (open‑vocabulary detection, referring segmentation, region captioning)
  - General visual reasoning (including spatial commonsense, multi‑hop relations, top‑view/layout reasoning)
- Modalities: 2D only vs 3D‑aware (egocentric video, maps) vs explicit 3D (depth/point clouds, voxels/neural fields, multi‑view/scene graphs).

## Architectural Paradigms

### 1) Contrastive encoders (CLIP‑style) with task‑specific heads
- F‑VLM uses a frozen CLIP‑like VLM and fine‑tunes a detector head for open‑vocabulary detection; frozen features preserve locality for region grounding with strong data efficiency [1].
- Learning Object–Language Alignments sets region–word alignment as set matching using image–text pairs, improving open‑vocabulary detection without exhaustive box supervision [2].
- VLCounter adapts CLIP features with text‑aware prompts and a one‑stage decoder for zero‑shot counting, showing gains on standard datasets (e.g., FSC‑147), underscoring the value of task‑specific spatial heads over generic MLLM prompting for fine‑grained counts [3].

Spatial implications:
- Implicit spatial locality emerges from patch/region features and cross‑attention, supporting grounding and counting. Lack of explicit geometry limits robustness under occlusion, viewpoint shifts, or 3D relational reasoning.

### 2) Encoder–decoder and generative LMMs (general‑purpose VLMs)
- BLIP‑2‑style models (exemplar of the class; see also PaLI/KOSMOS‑1 lineage) connect visual tokens to LLM decoders via adapters/resamplers for instruction following and VQA at scale. Representative systems include Flamingo, PaLI, and KOSMOS‑1, which established broad capabilities but remain primarily 2D [4–6].
- NExT‑Chat adds “pix2emb” to enable detection/segmentation in conversational loops, improving referring segmentation and region captioning by making spatial outputs first‑class citizens of the dialogue interface [7].
- VaLM retrieves images to augment LM context, improving visual commonsense about sizes/shapes/colors—an early template for “visual tool use” [8].
- Whiteboard‑of‑Thought introduces “visual chain‑of‑thought,” letting models draw intermediate diagrams/plots and re‑ingest them, substantially improving spatial/mathematical reasoning over textual CoT alone [9].
- Evaluation work like TopViewRS finds that state‑of‑the‑art VLMs underperform humans by large margins on bird’s‑eye/top‑view tasks; CoT helps only modestly, highlighting structural gaps in spatial representation [10].

Spatial implications:
- General LMMs benefit from adapters and tool interfaces (drawing, boxes/masks), but still lack strong 3D priors; explicit structures are needed for complex layouts and multi‑step spatial logic.

### 3) Diffusion‑based approaches with 3D‑aware priors
- LLM Director decomposes complex prompts into composable 3D‑aware elements and trajectories; diffusion and score distillation enforce multi‑view/temporal consistency [11].
- GALA3D, IM‑3D, and Layout‑based NeRF methods introduce explicit scene layouts, multi‑view diffusion, and 3D Gaussians/NeRFs for compositional text‑to‑3D with geometric consistency [12–14].

Spatial implications:
- By committing to 3D representations (NeRFs/Gaussians) and multi‑view constraints, these works encode scene layout, camera control, and object relations. They are mainly generative, suggesting an opportunity to transfer their 3D priors into downstream spatial reasoning and embodied agents.

### 4) Explicit 3D geometry modules in embodied agents
- SG‑Nav constructs online 3D scene graphs (objects/rooms/groups) and prompts an LLM hierarchically over the graph, yielding substantial zero‑shot navigation gains on MP3D/HM3D/RoboTHOR [15].
- 3D‑VLA couples a 3D‑aware LM with generative world modules that predict goal images/point clouds, aligning perception, reasoning, and action via a learned 3D world model [16].
- LEO trains a generalist 3D agent with two‑stage alignment (vision–language then vision–language–action), scaling across 3D captioning, QA, navigation, and manipulation tasks [17].
- Knowledge‑Based Embodied QA integrates symbolic/scene knowledge for situated QA; while task‑specific, it exemplifies structured priors improving 3D reasoning in interactive settings [18].

Spatial implications:
- Structured 3D abstractions (scene graphs, point clouds) and world‑modeling materially improve search, disambiguation, and planning in cluttered 3D spaces versus 2D‑only LMM prompting.

## Training and Alignment Methodologies

### Pretraining data composition
- Broad image–text corpora (web‑scale paired data) underlie general LMMs (e.g., Flamingo/PaLI/KOSMOS‑1) and provide coverage for categories and attributes, but lack dense spatial supervision [4–6].
- Task‑specific data mixtures:
  - Open‑vocabulary detection uses image–caption pairs and pseudo boxes/regions to tie text to spatial anchors (F‑VLM; object–language alignment) [1–2].
  - Counting uses category‑conditioned image sets with varied densities/scales (VLCounter) to make patch similarity sensitive to numerosity [3].
  - Embodied agents integrate egocentric video, odometry, and 3D reconstructions/scene graphs (SG‑Nav, 3D‑VLA, LEO) for spatial continuity and memory [15–17].
  - Generative 3D pipelines leverage internet images/videos plus text prompts, and multi‑view data, to enforce geometric consistency (IM‑3D, GALA3D) [12–13].

### Objectives and losses
- Contrastive alignment remains foundational for 2D region–text alignment and open‑vocabulary detection (F‑VLM, OLA) [1–2].
- Masked modeling and generative LM losses support captioning/VQA/instruction following (Flamingo/PaLI/KOSMOS‑1; NExT‑Chat integrates box/mask formats) [4–7].
- Spatial/geometry objectives:
  - Multi‑view consistency and score distillation sampling (SDS) in 3D diffusion pipelines enforce correct layouts across views [11–14].
  - 3D scene graph construction and likelihood‑based reasoning on graphs (SG‑Nav) provide structured supervision for spatial relations [15].
  - World‑model objectives predicting future observations/point clouds/goal images (3D‑VLA) align action planning with spatial outcomes [16].
- Reinforcement/trajectory‑level signals:
  - VLN uses path‑level pretexts like Masked Path Modeling to enhance trajectory reasoning and planning (Findings of EMNLP 2023) [22].

### Instruction tuning and alignment
- Instruction‑following MLLMs gain from:
  - Multimodal instruction datasets with grounded formats (e.g., region coordinates, masks) to teach spatial outputs in dialogue (NExT‑Chat) [7].
  - Visual tool use and “visual CoT” drawings to externalize intermediate spatial reasoning (Whiteboard‑of‑Thought) [9].
  - Hierarchical chain‑of‑thought over structured maps/graphs for large‑space search (SG‑Nav) [15].
- Curriculum and multi‑stage training:
  - Two‑stage alignment for embodied agents—first 3D visual–language alignment, then vision–language–action instruction tuning—improves transfer to navigation/manipulation (LEO; 3D‑VLA) [16–17].

### Spatial priors and 2D/3D modality integration
- 2D spatial priors: region proposals, segmentation, and box/mask outputs integrated into conversational loops (NExT‑Chat) [7].
- 3D priors: depth/point clouds/scene graphs; multi‑view constraints; neural fields (NeRF) and Gaussian splatting; egocentric video with odometry (SG‑Nav, 3D‑VLA, IM‑3D, GALA3D) [11–16].

### Reasoning strategies
- Textual CoT is often insufficient for spatial tasks; structured visual reasoning (drawings, maps, scene graphs) yields larger gains [9–10, 15].
- Retrieval‑augmented visual context helps attribute‑level reasoning (VaLM) [8].

### Compute/resource considerations
- Frozen‑backbone paradigms (F‑VLM) and light task heads offer efficiency and strong open‑vocabulary performance [1].
- Explicit 3D pipelines (scene mapping, multi‑view diffusion, 3D Gaussians) are computationally heavier but yield superior spatial grounding and embodied gains [11–17].
- Multi‑stage instruction tuning for embodied agents increases data/compute cost but improves generalization to unseen layouts [16–17].

## Evaluation Ecosystems

### Benchmarks and protocols by task
- Object counting:
  - Common datasets (e.g., FSC‑147, CARPK, PUCPR+) emphasize few‑shot/zero‑shot counting under scale changes and clutter; VLCounter evaluates across these, reporting consistent gains over CLIP‑based baselines [3].
  - Metrics: MAE/RMSE, relative error; sometimes category‑conditioned protocols.
- Grounding/localization:
  - Open‑vocabulary detection evaluations (LVIS novel classes, COCO novel subsets) quantify novel‑class AP and generalization (F‑VLM, object–language alignments) [1–2].
  - Referring segmentation and region captioning in conversational settings (NExT‑Chat) evaluate IoU/AP/mask quality and language grounding [7].
- Navigation/embodied instruction following:
  - MP3D, HM3D, RoboTHOR for object navigation; metrics include Success Rate (SR), Success weighted by Path Length (SPL), and success path efficiency. SG‑Nav reports >10% SR gains in zero‑shot setups via 3D scene graph prompting [15].
  - VLN pretexts (Masked Path Modeling) show path‑level improvements in environments like R2R/RxR (Findings of EMNLP) [22].
- General visual reasoning and spatial layout:
  - TopViewRS benchmarks top‑down spatial reasoning; VLMs lag humans by large margins, showing CoT adds only modest gains [10].
  - Hallucination analyses track grounding fidelity and object correctness in multimodal generation/QA [20–21].

### Dataset characteristics and gaps
- Many 2D datasets lack explicit 3D annotations; success often correlates with models’ ability to leverage spatial proxies (attention maps, boxes/masks).
- Embodied datasets provide egocentric continuity and 3D structure; methods that build persistent maps/graphs leverage this effectively.
- Evaluation gaps:
  - Top‑view/layout reasoning remains under‑tested in mainstream VLM leaderboards; specialized benchmarks reveal sizable performance gaps [10].
  - Hallucination persists even with grounding, indicating that spatial outputs do not fully eliminate semantic drift [21].

## Comparative Analysis: General‑Purpose vs Task‑Specific Models

### 2D settings
- General‑purpose MLLMs:
  - Strengths: broad task coverage, strong language interaction, flexible instruction following.
  - Limitations: lack precise spatial grounding; struggle with bird’s‑eye/layout tasks and multi‑hop spatial logic without added tools/structures [7, 9–10].
- Task‑specific/structured 2D:
  - Strengths: simple, efficient architectures (frozen CLIP + head) excel in grounding and counting; structured output formats (boxes/masks) improve localization [1–3, 7].
  - Trade‑offs: narrower task scope; require task‑tailored losses/prompts and dataset curation.

### 3D settings
- General‑purpose MLLMs augmented with 2D features:
  - Strengths: ease of deployment; can perform basic instruction following.
  - Limitations: weak spatial memory; poor search in unseen 3D layouts; limited occlusion/viewpoint robustness [10, 20–21].
- Explicit 3D/task‑specific agents:
  - Strengths: scene graphs/point clouds/world models deliver substantial SR/SPL gains in navigation and better grounding of spatial references [15–17].
  - Trade‑offs: higher computational and data demands (mapping, multi‑view training), engineering overhead for 3D pipelines.

## Empirical Trends and Performance Trade‑offs (2022–2025)
- Frozen contrastive features + light heads scale well for open‑vocabulary grounding; region–text alignment objectives close gaps to supervised detectors on novel classes [1–2].
- Counting benefits more from task‑specific adapters/prompts than from generic MLLM CoT; fine‑grained numerosity requires spatial attention modulation and density‑aware decoders [3].
- Enabling grounded outputs in LMMs (boxes/masks in dialogue) materially improves referring tasks and region‑level reasoning [7].
- Explicit 3D structure (scene graphs, point clouds) delivers the largest zero‑shot gains in navigation, even surpassing some supervised methods under certain protocols; chain‑of‑thought over graphs enhances search effectiveness [15].
- Diffusion‑based 3D pipelines establish strong geometric priors and compositional control; while evaluated mainly on generative fidelity, they are promising sources of 3D inductive bias for downstream reasoning [11–14].
- Visual/tool‑augmented CoT outperforms text‑only CoT on spatial tasks; purely linguistic reasoning is insufficient for top‑view/layout problems [9–10].

## Limitations Observed Across Benchmarks
- Persistent object hallucination and weak grounding fidelity in general‑purpose LMMs; adding grounding reduces but does not eliminate hallucination [20–21].
- Limited robustness to occlusion, scale, and perspective shifts in models without explicit 3D priors.
- Evaluation fragmentation: many datasets stress either linguistic reasoning or perception but not long‑horizon spatial planning under real‑world variability.

## Open Challenges and Future Directions
- Seamless 2D–3D integration: unify contrastive/local features, 3D scene memory, and LMM reasoning into a single training curriculum that scales without prohibitive compute.
- World‑model transfer: harness 3D generative priors (NeRF/Gaussians/multi‑view diffusion) for downstream embodied tasks with measurable SR/SPL/grounding gains.
- Reliable grounding and de‑hallucination: tighten spatial supervision (e.g., referential consistency, multi‑view verification) and causal evaluation to ensure that language claims match visual evidence.
- Spatial planning at scale: develop benchmarks that combine layout understanding, relational reasoning, and action under partial observability (egocentric, dynamic scenes), with standardized protocols and cost‑aware metrics.
- Tool‑centric spatial reasoning: standardize “visual CoT” canvases, mapping tools, and geometric solvers within LMM loops to externalize intermediate spatial states and reduce internal shortcutting.

## Practical Guidance
- For precise 2D grounding/counting under resource constraints: start with frozen CLIP‑style backbones plus task‑specific decoders and text‑aware prompts (F‑VLM, VLCounter) [1, 3].
- For embodied navigation/instruction following: invest in explicit 3D structure (scene graphs/point clouds) and hierarchical CoT planning; expect larger returns than from scaling generic LMM prompting (SG‑Nav, 3D‑VLA, LEO) [15–17].
- For general visual/spatial reasoning: add structured outputs (boxes/masks), retrieval or drawing tools, and spatial curricula; consider importing 3D priors from diffusion pipelines to improve viewpoint consistency [7, 9, 11–14].

### Sources
[1] F‑VLM: Open‑Vocabulary Object Detection upon Frozen Vision and Language Models (ICLR 2023): https://iclr.cc/virtual/2023/poster/11429  
[2] Learning Object–Language Alignments for Open‑Vocabulary Object Detection (ICLR 2023): https://iclr.cc/virtual/2023/poster/12075  
[3] VLCounter: Text‑Aware Visual Representation for Zero‑Shot Object Counting (AAAI 2024): https://ojs.aaai.org/index.php/AAAI/article/view/28050  
[4] Flamingo (NeurIPS 2022): https://papers.nips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html  
[5] PaLI: A Jointly‑Scaled Multilingual Language‑Image Model (ICLR 2023): https://iclr.cc/virtual/2023/poster/11441  
[6] KOSMOS‑1: Language Is Not All You Need (NeurIPS 2023): https://papers.nips.cc/paper_files/paper/2023/hash/e425b75bac5742a008d643826428787c-Abstract-Conference.html  
[7] NExT‑Chat: An LMM for Chat, Detection and Segmentation (ICML 2024): https://proceedings.mlr.press/v235/zhang24bu.html  
[8] Visually‑Augmented Language Modeling (ICLR 2023): https://iclr.cc/virtual/2023/poster/11283  
[9] Whiteboard‑of‑Thought: Thinking Step‑by‑Step Across Modalities (EMNLP 2024): https://aclanthology.org/2024.emnlp-main.1117/  
[10] TopViewRS: Vision‑Language Models as Top‑View Spatial Reasoners (EMNLP 2024): https://aclanthology.org/2024.emnlp-main.106/  
[11] Compositional 3D‑aware Video Generation with LLM Director (NeurIPS 2024): https://papers.nips.cc/paper_files/paper/2024/hash/edbeca7811f9365c924c72a8a9bce83a-Abstract-Conference.html  
[12] GALA3D: Layout‑Guided Generative Gaussian Splatting for Text‑to‑3D Complex Scenes (ICML 2024): https://proceedings.mlr.press/v235/zhou24p.html  
[13] IM‑3D: Iterative Multiview Diffusion and Reconstruction for High‑Quality 3D Generation (ICML 2024): https://proceedings.mlr.press/v235/melas-kyriazi24a.html  
[14] Disentangled 3D Scene Generation with Layout Learning (ICML 2024): https://proceedings.mlr.press/v235/epstein24a.html  
[15] SG‑Nav: Online 3D Scene Graph Prompting for LLM‑based Zero‑shot Object Navigation (NeurIPS 2024): https://papers.nips.cc/paper_files/paper/2024/hash/098491b37deebbe6c007e69815729e09-Abstract-Conference.html  
[16] 3D‑VLA: A 3D Vision‑Language‑Action Generative World Model (ICML 2024): https://proceedings.mlr.press/v235/zhen24a.html  
[17] LEO: An Embodied Generalist Agent in 3D World (ICML 2024): https://proceedings.mlr.press/v235/huang24ae.html  
[18] Knowledge‑Based Embodied Question Answering (TPAMI 2023): https://dl.acm.org/doi/abs/10.1109/TPAMI.2023.3277206  
[19] HiLM‑D: Enhancing MLLMs with Multi‑scale High‑Resolution Details for Autonomous Driving (IJCV 2025): https://link.springer.com/article/10.1007/s11263-025-02433-3  
[20] Evaluating Object Hallucination in LMMs (EMNLP 2023): https://aclanthology.org/2023.emnlp-main.20/  
[21] Does Object Grounding Really Reduce Hallucination? (EMNLP 2024): https://aclanthology.org/2024.emnlp-main.159/  
[22] Masked Path Modeling for Vision‑and‑Language Navigation (Findings of EMNLP 2023): https://aclanthology.org/2023.findings-emnlp.1019/  
[23] A Picture is Worth a Thousand Words: Language Models Plan from Pixels (EMNLP 2023): https://aclanthology.org/2023.emnlp-main.1025/  
[24] Localizing Active Objects from Egocentric Vision with Symbolic World Knowledge (EMNLP 2023): https://aclanthology.org/2023.emnlp-main.304/