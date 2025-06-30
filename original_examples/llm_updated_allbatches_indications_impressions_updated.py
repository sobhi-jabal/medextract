"""
UPDATED  – May 13 2025
──────────────────────
• Adds **all_impressions_list** datapoint so the pipeline now extracts BOTH
  ▸ every Indication mentioned  AND  
  ▸ the complete Impression section  
• Nothing else in the control-flow changed: the main loop already walks
  over every key in `datapoints_config`, so just dropping the new config
  in automatically wires the new extraction end-to-end.  
• A handful of few-shot examples for impressions were added; expand or
  tweak as you like.

Replace your existing file with this one and run it exactly the same way
(`python your_script.py`).  If you already have checkpoints/results from
the older run you can delete them or let the script skip rows that were
finished.  Enjoy!
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import os
import json
import shutil
from typing import Dict, Optional, Sequence, Tuple

from langchain.schema import Document
from langchain.pydantic_v1 import Extra
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
from sentence_transformers import CrossEncoder
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

import ollama
from ollama import Options

import sys
from contextlib import contextmanager
from tqdm.auto import tqdm

from terminology import (
    in_red, in_yellow, in_green, in_blue, in_magenta,
    in_bold, underlined, on_green, on_yellow, on_red, on_blue
)

# ──────────────────────────────────────────────────────────────────────────────
# General Helper Functions
# ──────────────────────────────────────────────────────────────────────────────
@contextmanager
def suppress_stdout():
    """Context-manager to silence noisy library prints (e.g. FAISS, tqdm)."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def preprocess_text(text: str) -> str:
    """Light cleaning: normalise newlines so sections stay intact."""
    if not text or pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\n(?!\.)", " ", text)      # single newlines → space
    text = re.sub(r"\.\n", " \n ", text)       # keep paragraph breaks
    return text.strip()


def get_text_chunks(text: str, chunk_size: int, chunk_overlap: int) -> Sequence[Document]:
    """Chunk text for RAG when enabled."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ":", "."],
    )
    return [Document(page_content=x) for x in splitter.split_text(text)]

# ──────────────────────────────────────────────────────────────────────────────
# BGE-Reranker (optional, if you turn RAG back on)
# ──────────────────────────────────────────────────────────────────────────────
class BgeRerank(BaseDocumentCompressor):
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 2
    model: CrossEncoder = CrossEncoder(model_name, device="cuda")

    def bge_rerank(self, query, docs):
        inputs = [[query, d] for d in docs]
        scores = self.model.predict(inputs)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[: self.top_n]

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    # main interface that LangChain expects
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
        docs = list(documents)
        raw = [d.page_content for d in docs]
        results = self.bge_rerank(query, raw)
        keep = []
        for idx, score in results:
            doc = docs[idx]
            doc.metadata["relevance_score"] = score
            keep.append(doc)
        return keep

# ──────────────────────────────────────────────────────────────────────────────
# Prompt Construction
# ──────────────────────────────────────────────────────────────────────────────
def construct_generic_prompt(
    context: str,
    datapoint_name: str,
    config: dict,
    use_few_shots: bool = True,
) -> list:
    """Build a chat completion prompt for the requested datapoint."""
    if not context.strip():
        return []

    datapoint_cfg = config[datapoint_name]
    instruction = datapoint_cfg["instruction"]
    few_shots = datapoint_cfg.get("few_shots", [])

    system_prompt = [
        {
            "role": "system",
            "content": (
                "You are an extremely rigorous data-extraction expert.  "
                "Follow the instructions precisely and extract the data "
                "accurately.  Ignore cerebrovascular findings; we only care "
                "about abdominal/imaging content for this task."
            ),
        }
    ]

    user_prompt = {
        "role": "user",
        "content": f"Context:\n{context}\n\nTask: {instruction}",
    }

    prompt = system_prompt.copy()
    if use_few_shots and few_shots:
        prompt += few_shots.copy()
    prompt.append(user_prompt)
    return prompt

# ──────────────────────────────────────────────────────────────────────────────
# LLM Call & Post-processing
# ──────────────────────────────────────────────────────────────────────────────
def ollama_llm(
    messages: list,
    llm_model: str,
    output_format: str,
    temp: float,
    top_k: int,
    top_p: float,
) -> str:
    """Thin wrapper around `ollama.chat` that returns the raw string."""
    if not messages:
        return ""
    try:
        response = ollama.chat(
            model=llm_model,
            format="json" if output_format == "json" else None,
            keep_alive="30m",
            options=Options(
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                num_ctx=2048,
            ),
            messages=messages,
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"Error in ollama_llm: {e}")
        return ""


def clean_extracted_data(raw_output: str, datapoint_config: dict):
    """Attempt to JSON-parse and validate the model output."""
    json_key = datapoint_config.get("json_key", "result")
    valid_values = datapoint_config.get("valid_values")

    data = {json_key: "invalid"}  # default fallback

    if not raw_output.strip():
        return data

    if datapoint_config["output_format"] == "json":
        try:
            clipped = raw_output.strip()
            start = clipped.find("{")
            end = clipped.rfind("}") + 1
            if start != -1 and end != -1:
                clipped = clipped[start:end]
                parsed = json.loads(clipped)
                if json_key in parsed:
                    data = parsed
        except json.JSONDecodeError:
            pass
    else:
        cleaned = raw_output.strip()
        if cleaned:
            data = {json_key: cleaned}

    if valid_values is not None:
        value = data.get(json_key)
        if value not in valid_values:
            data[json_key] = "invalid"

    return data

# ──────────────────────────────────────────────────────────────────────────────
# Main Extraction Function
# ──────────────────────────────────────────────────────────────────────────────
def extract_datapoint_from_text(
    text: str,
    datapoint_name: str,
    config: dict,
    llm_model: str,
    rag_enabled: bool,
    embeddings,
    retriever_type: str,
    reranker,
    use_few_shots: bool = True,
    temp: float = 0.0,
    top_k: int = 2,
    top_p: float = 0.9,
) -> Tuple[str, Dict]:
    """Run one datapoint through the pipeline and return (raw, cleaned)."""
    default_cleaned = {config[datapoint_name]["json_key"]: "invalid"}

    if not text or pd.isna(text):
        return "", default_cleaned

    preprocessed = preprocess_text(text)
    if not preprocessed:
        return "", default_cleaned

    # ── (Optionally) build mini-RAG context ───────────────────────────────
    if rag_enabled:
        chunks = get_text_chunks(preprocessed, chunk_size=70, chunk_overlap=20)
        db = FAISS.from_documents(chunks, embeddings)
        if reranker:
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=db.as_retriever(
                    search_type="similarity", search_kwargs={"k": 1}
                ),
            )
        else:
            retriever = db.as_retriever(
                search_type="similarity", search_kwargs={"k": 1}
            )

        if retriever_type == "ensemble":
            keyword = BM25Retriever.from_documents(chunks, k=1)
            retriever = EnsembleRetriever(
                retrievers=[retriever, keyword], weights=[0.25, 0.75]
            )
        elif retriever_type == "sequential":
            keyword = BM25Retriever.from_documents(chunks, k=3)
            chunks = keyword.invoke(".....")
            db = FAISS.from_documents(chunks, embeddings)
            retriever = db.as_retriever(
                search_type="similarity", search_kwargs={"k": 1}
            )

        retrieved_docs = retriever.invoke(".....")
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    else:
        context = preprocessed

    # ── LLM call ──────────────────────────────────────────────────────────
    prompt = construct_generic_prompt(context, datapoint_name, config, use_few_shots)
    if not prompt:
        return "", default_cleaned

    raw_output = ollama_llm(
        prompt,
        llm_model,
        config[datapoint_name]["output_format"],
        temp,
        top_k,
        top_p,
    )
    cleaned_output = clean_extracted_data(raw_output, config[datapoint_name])

    # logging (optional, comment out in production)
    print("\n" + "-" * 100)
    print(f"Datapoint: {datapoint_name}")
    print(f"Raw LLM output: {raw_output}")
    print(f"Cleaned: {cleaned_output}")
    print("-" * 100 + "\n")

    return raw_output, cleaned_output

# ──────────────────────────────────────────────────────────────────────────────
#  Datapoint Configuration
# ──────────────────────────────────────────────────────────────────────────────
# ─────────────────────────  DATAPOINT CONFIG  (v May-13-2025-c)  ─────────────────────────
# Goal changes
#   •  “Extract *all* as-is”: keep the exact wording / abbreviations /
#      ICD-10 codes that appear in the report; no normalisation.
#   •  Handle items separated by commas, semicolons, “and”, bullets, or numbers.
#   •  Still return a JSON list (strings trimmed of leading/trailing spaces).

datapoints_config: Dict[str, dict] = {

    # ─────────────────────  1)  ALL INDICATIONS  ─────────────────────
    "all_indications_list": {
        "instruction": (
            "Locate the INDICATION section (a heading such as 'Indication:', "
            "'Reason for Exam:', 'History:', 'DX:', etc.). "
            "Copy **every phrase exactly as it appears**, splitting on commas, "
            "semicolons, newlines, the word 'and', or bullet/number markers. "
            "Trim leading/trailing whitespace but do *no* re-phrasing, spelling "
            "changes, or code-to-text conversion.  Preserve ICD codes, mixed "
            "case, abbreviations, and internal punctuation.\n\n"
            "Return valid JSON exactly as:\n"
            "      {\"indications\": [\"item1\", \"item2\", …]}\n"
            "If the report truly contains no indication section, return an empty list."
        ),
        "valid_values": None,
        "few_shots": [
            # simple comma list
            {"role": "user",
             "content": "Indication: abdominal pain, nausea, vomiting."},
            {"role": "assistant",
             "content": "{\"indications\": [\"abdominal pain\", \"nausea\", \"vomiting\"]}"},
            # ICD-10 code plus phrase
            {"role": "user",
             "content": "Indication: R10.31 Right lower quadrant pain, eval for appendicitis"},
            {"role": "assistant",
             "content": "{\"indications\": [\"R10.31 Right lower quadrant pain\", \"eval for appendicitis\"]}"},
            # semicolons
            {"role": "user",
             "content": "Reason for CT: Hematuria; weight loss; ? malignancy"},
            {"role": "assistant",
             "content": "{\"indications\": [\"Hematuria\", \"weight loss\", \"? malignancy\"]}"},
            # 'and' separator
            {"role": "user",
             "content": "History: fever and chills"},
            {"role": "assistant",
             "content": "{\"indications\": [\"fever\", \"chills\"]}"},
            # bullet list
            {"role": "user",
             "content": "Indication:\n• trauma MVC\n• seat-belt sign\n• hypotension"},
            {"role": "assistant",
             "content": "{\"indications\": [\"trauma MVC\", \"seat-belt sign\", \"hypotension\"]}"},
            # no indication
            {"role": "user", "content": "No indication provided."},
            {"role": "assistant", "content": "{\"indications\": []}"},
        ],
        "output_format": "json",
        "json_key": "indications",
    },

    # ─────────────────────  2)  ALL IMPRESSIONS  ─────────────────────
    "all_impressions_list": {
        "instruction": (
            "Locate the IMPRESSION section (heading variants: 'Impression:', "
            "'IMP:', 'Conclusion:', 'Summary:'). "
            "Copy the text until the next heading or end-of-report.  Split into "
            "separate list items using:\n"
            "  • line breaks\n  • bullets / numbers\n  • commas / semicolons\n"
            "Again, keep the wording exactly as written.  Do not merge or paraphrase.\n\n"
            "Return valid JSON exactly as:\n"
            "      {\"impressions\": [\"…\", \"…\", …]}\n"
            "Return an empty list if the section is absent."
        ),
        "valid_values": None,
        "few_shots": [
            # numbered lines with commas
            {"role": "user",
             "content": "IMPRESSION:\n1. Small left pleural effusion, stable.\n2. No pneumonia."},
            {"role": "assistant",
             "content": "{\"impressions\": [\"Small left pleural effusion\", \"stable\", \"No pneumonia\"]}"},
            # single line, many commas
            {"role": "user",
             "content": "Impression: Severe emphysema, resolved lymphadenopathy, no new metastases."},
            {"role": "assistant",
             "content": "{\"impressions\": [\"Severe emphysema\", \"resolved lymphadenopathy\", \"no new metastases\"]}"},
            # wrapped bullet with semicolons
            {"role": "user",
             "content": "IMP:\n• Large perihepatic hematoma; active extravasation.\n• Stable surgical drain."},
            {"role": "assistant",
             "content": "{\"impressions\": [\"Large perihepatic hematoma\", \"active extravasation\", \"Stable surgical drain\"]}"},
            # dash bullets
            {"role": "user",
             "content": "Impression:\n- Right nephrolithiasis.\n- No hydronephrosis."},
            {"role": "assistant",
             "content": "{\"impressions\": [\"Right nephrolithiasis\", \"No hydronephrosis\"]}"},
            # none present
            {"role": "user", "content": "Findings only — no conclusion."},
            {"role": "assistant", "content": "{\"impressions\": []}"},
        ],
        "output_format": "json",
        "json_key": "impressions",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Crash-safe helpers
# ──────────────────────────────────────────────────────────────────────────────
def save_checkpoint(df: pd.DataFrame, index: int, path: str = "checkpoint.csv"):
    df.to_csv(path, index=False)
    with open("last_processed_index.txt", "w") as f:
        f.write(str(index))
    print(f"\nCheckpoint saved at row {index}")


def load_checkpoint() -> Tuple[Optional[pd.DataFrame], int]:
    if os.path.exists("checkpoint.csv") and os.path.exists("last_processed_index.txt"):
        with open("last_processed_index.txt") as f:
            last = int(f.read().strip())
        df = pd.read_csv("checkpoint.csv")
        print(f"Resuming from checkpoint row {last}")
        return df, last
    return None, -1

# ──────────────────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────────────────
def main():
    file_path = "allbatches_tr_10x4.csv"
    backup_path = file_path.replace(".csv", "_original.csv")
    checkpoint_path = "checkpoint.csv"

    # one-time backup
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)

    # resume or fresh start
    df, start_idx = load_checkpoint()
    if df is None:
        df = pd.read_csv(file_path)
        start_idx = 0

    # pull / ensure models
    llm_models = [
                # "llama3.2:3b", 
                # "llama3.1:8b", 
                "phi4:latest",
                ]  
    installed = [m["name"] for m in ollama.list()["models"]]
    for m in llm_models:
        if m not in installed:
            print(f"Pulling model {m} …")
            ollama.pull(m)

    # add result columns on first run
    if start_idx == 0:
        for model in llm_models:
            suffix = model.replace(":", "_")
            for dp in datapoints_config:
                clean_col = f"{dp}_{suffix}"
                raw_col = f"{dp}_{suffix}_raw"
                if clean_col not in df.columns:
                    df[clean_col] = pd.Series(dtype='object')
                if raw_col not in df.columns:
                    df[raw_col] = pd.Series(dtype='object')

    print(f"Processing rows {start_idx + 1}…{len(df) - 1}")
    for i in tqdm(range(start_idx + 1, len(df))):
        report = df.at[i, "Report Text"]
        if pd.notna(report):
            for model in llm_models:
                suffix = model.replace(":", "_")
                for dp in datapoints_config:
                    clean_col = f"{dp}_{suffix}"
                    raw_col = f"{dp}_{suffix}_raw"

                    try:
                        raw, cleaned = extract_datapoint_from_text(
                            text=report,
                            datapoint_name=dp,
                            config=datapoints_config,
                            llm_model=model,
                            rag_enabled=False,
                            embeddings=None,
                            retriever_type="no_rag",
                            reranker=None,
                            use_few_shots=True,
                        )
                        df.at[i, raw_col] = raw or "no_output"
                        df.at[i, clean_col] = cleaned.get(
                            datapoints_config[dp]["json_key"], "invalid"
                        )
                    except Exception as e:
                        print(f"Error row {i} / {dp}: {e}")
                        df.at[i, clean_col] = "invalid"
                        df.at[i, raw_col] = "error"

        # checkpoint every 5 rows
        if i % 100 == 0:
            save_checkpoint(df, i, checkpoint_path)

    # final write-out
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"results_{ts}.csv"
    df.to_csv(out, index=False)
    df.to_csv("results.csv", index=False)
    print(f"\nExtraction complete → {out}")

    # cleanup
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    if os.path.exists("last_processed_index.txt"):
        os.remove("last_processed_index.txt")


def delete_checkpoints():
    for f in ("checkpoint.csv", "last_processed_index.txt"):
        if os.path.exists(f):
            os.remove(f)
    print("Checkpoint files deleted.")


# ──────────────────────────────────────────────────────────────────────────────
# Simple reporting utility (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def print_extraction_statistics(df: pd.DataFrame, model_name: str):
    suffix = model_name.replace(":", "_")
    print(f"\nSummary for {model_name}\n" + "-" * 40)
    for dp in datapoints_config:
        clean = f"{dp}_{suffix}"
        raw = f"{dp}_{suffix}_raw"
        total = len(df)
        valid = df[clean].notna().sum()
        invalid = df[clean].eq("invalid").sum()
        print(f"{dp} →  valid {valid}/{total}   invalid {invalid}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
        res = pd.read_csv("results.csv")
        for m in ["phi4"]:
            print_extraction_statistics(res, m)
        delete_checkpoints()
    except Exception as exc:
        print(f"Fatal error: {exc}")
        raise
