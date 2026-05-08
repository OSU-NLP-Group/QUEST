"""
FAISS search bridge for BrowseComp-Plus fair comparison.

When FAISS_INDEX_PATH is set, QUEST's Search tool uses Qwen3-Embedding-8B
FAISS retrieval with fixed-length snippets. Results keep the same textual
format as the regular search backend so the agent prompt does not need a
separate parser.
"""
import glob
import os
import pickle
import threading
from itertools import chain

import numpy as np

FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "")
FAISS_MODEL_NAME = os.environ.get("FAISS_MODEL_NAME", "Qwen/Qwen3-Embedding-8B")
FAISS_TOP_K = int(os.environ.get("FAISS_TOP_K", "5"))
FAISS_SNIPPET_MAX_TOKENS = int(os.environ.get("FAISS_SNIPPET_MAX_TOKENS", "512"))
FAISS_CUDA_DEVICE = os.environ.get("FAISS_CUDA_DEVICE", "")

_faiss_state = None
_faiss_lock = threading.Lock()


def _init_faiss():
    """Lazy-init FAISS index, embedding model, tokenizer, and corpus lookup."""
    global _faiss_state
    if _faiss_state is not None:
        return _faiss_state

    with _faiss_lock:
        if _faiss_state is not None:
            return _faiss_state

        import faiss  # noqa: F401
        import torch
        from datasets import load_dataset
        from tevatron.retriever.arguments import ModelArguments
        from tevatron.retriever.driver.encode import DenseModel
        from tevatron.retriever.searcher import FaissFlatSearcher
        from transformers import AutoTokenizer
        from tqdm import tqdm

        print(f"[faiss_bridge] Initializing FAISS searcher: {FAISS_MODEL_NAME}")

        index_files = sorted(glob.glob(FAISS_INDEX_PATH))
        if not index_files:
            raise ValueError(f"No FAISS index files found: {FAISS_INDEX_PATH}")

        def pickle_load(path):
            with open(path, "rb") as f:
                reps, lookup = pickle.load(f)
            return np.array(reps), lookup

        p_reps_0, p_lookup_0 = pickle_load(index_files[0])
        retriever = FaissFlatSearcher(p_reps_0)

        lookup = list(p_lookup_0)
        shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
        if len(index_files) > 1:
            shards = tqdm(shards, desc="Loading FAISS shards", total=len(index_files))
        for p_reps, p_lookup in shards:
            retriever.add(p_reps)
            lookup += list(p_lookup)

        device = "cpu"
        if FAISS_CUDA_DEVICE and torch.cuda.is_available():
            device = f"cuda:{FAISS_CUDA_DEVICE}"

        model_args = ModelArguments(
            model_name_or_path=FAISS_MODEL_NAME,
            normalize=True,
            pooling="eos",
        )

        attn_impl = "flash_attention_2"
        if not torch.cuda.is_available() or device == "cpu":
            attn_impl = "eager"

        model = DenseModel.load(
            model_args.model_name_or_path,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            lora_name_or_path=model_args.lora_name_or_path,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            attn_implementation=attn_impl,
        )
        model = model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(FAISS_MODEL_NAME, padding_side="left")
        snippet_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

        print("[faiss_bridge] Loading BrowseComp-Plus corpus...")
        ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
        docid_to_text = {row["docid"]: row["text"] for row in ds}
        print(f"[faiss_bridge] Loaded {len(docid_to_text)} documents")

        _faiss_state = {
            "retriever": retriever,
            "model": model,
            "tokenizer": tokenizer,
            "snippet_tokenizer": snippet_tokenizer,
            "lookup": lookup,
            "docid_to_text": docid_to_text,
            "device": device,
            "task_prefix": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        }

        print(f"[faiss_bridge] Ready. device={device}, k={FAISS_TOP_K}, snippet_tokens={FAISS_SNIPPET_MAX_TOKENS}")
        return _faiss_state


def faiss_search(query: str) -> str:
    """Search using FAISS and return results in the Search tool text format."""
    import torch

    state = _init_faiss()
    retriever = state["retriever"]
    model = state["model"]
    tokenizer = state["tokenizer"]
    snippet_tokenizer = state["snippet_tokenizer"]
    lookup = state["lookup"]
    docid_to_text = state["docid_to_text"]
    device = state["device"]
    task_prefix = state["task_prefix"]

    batch_dict = tokenizer(
        task_prefix + query,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    )
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

    with torch.amp.autocast(device.split(":")[0] if "cuda" in device else "cpu"):
        with torch.no_grad():
            q_reps = model.encode_query(batch_dict)
            q_reps = q_reps.cpu().detach().numpy()

    all_scores, psg_indices = retriever.search(q_reps, FAISS_TOP_K)

    web_snippets = []
    for score, index in zip(all_scores[0], psg_indices[0]):
        docid = lookup[index]
        text = docid_to_text.get(docid, "")
        tokens = snippet_tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > FAISS_SNIPPET_MAX_TOKENS:
            text = snippet_tokenizer.decode(tokens[:FAISS_SNIPPET_MAX_TOKENS], skip_special_tokens=True)

        web_snippets.append(
            f"Title: Document {docid}\n"
            f"Link: bm25://{docid}\n"
            f"Score: {float(score):.4f}\n"
            f"Snipptes: {text}"
        )

    if not web_snippets:
        return f"No results found for query: '{query}'. Use a less specific query."

    return f"A search for '{query}' found {len(web_snippets)} results:\n\n" + "\n\n".join(web_snippets)
