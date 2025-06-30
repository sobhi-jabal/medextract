#!/usr/bin/env python
"""
BT-RADS processing pipeline (June 2025)

Changes vs. February 2025 baseline
──────────────────────────────────
1. **Optional Retrieval-Augmented Generation (RAG)**
   • `--mode rag` (default) – smart hybrid retrieval + reranking.  
   • `--mode full` – skip retrieval and feed the *entire* cleaned report
     (up to ≈16 k tokens) to the LLM.

2. **Large-context LLM**
   • `num_ctx=16000` is already hard-wired in `enhanced_ollama_llm`.

All other logic, helper functions, and data structures remain intact.
"""
# ╭──────────────────────────── Imports ─────────────────────────────╮
import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm

from langchain.callbacks.manager import Callbacks
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

import ollama
from ollama import Options

# ╭──────────────────────────── Logger ──────────────────────────────╮
class Logger:
    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        self.log = open(log_file, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"btrads_processing_{timestamp}.log"
sys.stdout = Logger(log_file)
print(f"Started logging to {log_file}")

# ╭──────────────────────────── CLI ─────────────────────────────────╮
parser = argparse.ArgumentParser(
    description="BT-RADS extractor with optional Retrieval-Augmented Generation"
)
parser.add_argument(
    "--mode",
    choices=["rag", "full"],
    default="rag",
    help="'rag' (default) uses hybrid retrieval; "
    "'full' bypasses retrieval and sends the entire report to the LLM.",
)
cli_args = parser.parse_args()
USE_RAG = cli_args.mode == "rag"
print(f"\n► Retrieval mode: {'RAG' if USE_RAG else 'FULL REPORT'}\n")

# ╭─────────────────────── Text utilities ──────────────────────────╮
def minimal_clean_text(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    text = str(text)
    text = (
        text.replace('"', '"')
        .replace("''", "'")
        .replace("‘‘", "'")
        .replace("–", "-")
        .replace("—", "-")
        .replace("…", "...")
    )
    text = "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\t\r")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def detect_medical_sections(text: str) -> List[Tuple[str, int, int]]:
    patterns = [
        (r"Oncology Treatment History:", "treatment_history"),
        (r"Current Medications:", "current_medications"),
        (r"Treatment to be received:", "treatment_plan"),
        (r"History of Present Illness:", "present_illness"),
        (r"Assessment & Plan:", "assessment_plan"),
        (r"Subjective:", "subjective"),
        (r"Objective:", "objective"),
        (r"Past Medical History:", "past_medical"),
        (r"Past Surgical History:", "past_surgical"),
        (r"Social History:", "social_history"),
        (r"Review of Systems:", "review_systems"),
        (r"Physical Exam:", "physical_exam"),
        (r"Laboratory:", "laboratory"),
        (r"Allergies:", "allergies"),
    ]
    sections: List[Tuple[str, int, int]] = []
    for pat, name in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            sections.append((name, m.start(), m.end()))
    sections.sort(key=lambda x: x[1])
    final: List[Tuple[str, int, int]] = []
    for i, (name, start, _) in enumerate(sections):
        end = sections[i + 1][1] if i < len(sections) - 1 else len(text)
        final.append((name, start, end))
    return final


def smart_medical_chunker(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    cleaned = minimal_clean_text(text)
    sections = detect_medical_sections(cleaned)
    chunks: List[Document] = []
    chunk_id = 0

    if not sections:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ".", " "],
        )
        for i, t in enumerate(splitter.split_text(cleaned)):
            chunks.append(
                Document(
                    page_content=t,
                    metadata=dict(
                        chunk_id=i,
                        section="unknown",
                        start_pos=0,
                        end_pos=len(t),
                        source_type="regular_chunk",
                    ),
                )
            )
        return chunks

    for name, start, end in sections:
        sect_text = cleaned[start:end]
        if len(sect_text) <= chunk_size:
            chunks.append(
                Document(
                    page_content=sect_text,
                    metadata=dict(
                        chunk_id=chunk_id,
                        section=name,
                        start_pos=start,
                        end_pos=end,
                        source_type="section_chunk",
                    ),
                )
            )
            chunk_id += 1
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", ".", " "],
            )
            for i, t in enumerate(splitter.split_text(sect_text)):
                chunks.append(
                    Document(
                        page_content=t,
                        metadata=dict(
                            chunk_id=chunk_id,
                            section=name,
                            section_part=i,
                            start_pos=start,
                            end_pos=end,
                            source_type="section_subchunk",
                        ),
                    )
                )
                chunk_id += 1
    return chunks

# ╭─────────────────── Medical reranker ────────────────────────────╮
class MedicalReranker(BaseDocumentCompressor):
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 3

    def __init__(self, top_n: int = 3):
        self.top_n = top_n
        try:
            self.model = CrossEncoder(self.model_name)
        except Exception as e:
            print(f"Warning: reranker unavailable – {e}")
            self.model = None

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    def medical_rerank(self, query: str, docs: List[str]) -> List[Tuple[int, float]]:
        if self.model is None:
            return [(i, 0.5) for i in range(min(len(docs), self.top_n))]
        scores = self.model.predict([[query, d] for d in docs])
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[: self.top_n]

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
        docs = list(documents)
        ranking = self.medical_rerank(query, [d.page_content for d in docs])
        out: List[Document] = []
        for idx, score in ranking:
            d = docs[idx]
            d.metadata["relevance_score"] = score
            out.append(d)
        return out

# ╭──────────────────── Retriever builder ──────────────────────────╮
def build_smart_retriever(chunks: List[Document], retriever_type: str = "hybrid"):
    try:
        embed = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Embedding load failed: {e}")
        return None

    try:
        vect = FAISS.from_documents(chunks, embed)
        semantic = vect.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    except Exception as e:
        print(f"Vector store failed: {e}")
        return None

    if retriever_type == "semantic":
        return semantic

    try:
        keyword = BM25Retriever.from_documents(chunks, k=5)
    except Exception as e:
        print(f"BM25 failed: {e}")
        return semantic

    if retriever_type == "keyword":
        return keyword

    try:
        ensemble = EnsembleRetriever(retrievers=[semantic, keyword], weights=[0.6, 0.4])
        return ContextualCompressionRetriever(
            base_compressor=MedicalReranker(top_n=3), base_retriever=ensemble
        )
    except Exception as e:
        print(f"Ensemble failed: {e}")
        return semantic

# ╭────────────────── Context gathering (optional RAG) ─────────────╮
def retrieve_with_source_tracking(
    text: str, query: str, datapoint_name: str, use_rag: bool = USE_RAG
) -> Tuple[str, List[Dict]]:
    cleaned = minimal_clean_text(text)

    # FULL-report mode
    if not use_rag:
        char_cap = 65_000  # ≈ 16 k tokens
        context = cleaned[:char_cap]
        return context, [
            dict(
                chunk_id=-1,
                section="full_report",
                relevance_score=1.0,
                source_type="full_report",
                content_preview=context[:150] + ("..." if len(context) > 150 else ""),
                start_pos=0,
                end_pos=len(context),
            )
        ]

    # RAG mode
    chunks = smart_medical_chunker(cleaned, 800, 150)
    if not chunks:
        return cleaned[:2_000], [{"error": "no chunks"}]

    retriever = build_smart_retriever(chunks, "hybrid")
    if retriever is None:
        fallback = chunks[:3]
        ctx = "\n\n---\n\n".join(c.page_content for c in fallback)
        return ctx[:2_000], [
            dict(
                chunk_id=c.metadata.get("chunk_id", i),
                section=c.metadata.get("section", "unknown"),
                relevance_score=0.5,
                source_type="fallback",
                content_preview=c.page_content[:100] + "...",
                start_pos=c.metadata.get("start_pos", 0),
                end_pos=c.metadata.get("end_pos", 0),
            )
            for i, c in enumerate(fallback)
        ]

    try:
        docs = retriever.invoke(query)
    except Exception as e:
        print(f"Retrieval error: {e}")
        docs = chunks[:3]

    ctx_parts, src_info = [], []
    for i, d in enumerate(docs):
        ctx_parts.append(f"[CHUNK {i+1}]\n{d.page_content}")
        src_info.append(
            dict(
                chunk_id=d.metadata.get("chunk_id", i),
                section=d.metadata.get("section", "unknown"),
                relevance_score=d.metadata.get("relevance_score", 0.0),
                source_type=d.metadata.get("source_type", "retrieved"),
                content_preview=d.page_content[:150]
                + ("..." if len(d.page_content) > 150 else ""),
                start_pos=d.metadata.get("start_pos", 0),
                end_pos=d.metadata.get("end_pos", 0),
            )
        )
    return "\n\n".join(ctx_parts), src_info

# ╭─────────────────── Datapoint-specific query map ────────────────╮
def get_datapoint_query(name: str) -> str:
    return dict(
        medication_status="current medications steroids dexamethasone Avastin bevacizumab treatment plan dose changes",
        radiation_date="radiation therapy completion date chemoradiation XRT RT treatment history oncology timeline",
        btrads_assessment="volumetric measurements FLAIR enhancement progression treatment response medication effects",
    ).get(name, "medical information treatment")

# ╭──────────────────────── LLM wrapper ────────────────────────────╮
def enhanced_ollama_llm(
    messages: list,
    llm_model: str,
    output_format: str,
    temp: float = 0.0,
    top_k: int = 40,
    top_p: float = 0.95,
    patient_id=None,
    datapoint_name=None,
) -> str:
    options = Options(
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        num_ctx=16_000,
        num_predict=512,
        repeat_penalty=1.1,
        seed=42,
    )
    for attempt in range(3):
        try:
            print(f"LLM call [{datapoint_name}] try {attempt+1}")
            resp = ollama.chat(
                model=llm_model,
                format="json" if output_format == "json" else None,
                keep_alive=0,
                options=options,
                messages=messages,
            )
            content = resp["message"]["content"]
            if not content.strip():
                raise ValueError("empty content")
            return content
        except Exception as e:
            print(f"LLM error: {e}")
            time.sleep(2)
    return ""

# ╭──────────────────── Prompt constructor ────────────────────────╮
def construct_rag_prompt(
    context: str,
    source_info: List[Dict],
    datapoint_name: str,
    config: dict,
    use_few_shots: bool = True,
) -> list:
    cfg = config[datapoint_name]
    sys_msg = (
        cfg.get("system_message", "")
        + "\n\nCONTEXT ANALYSIS:\n"
        + f"{len(source_info)} sections retrieved.\n"
        + "\n".join(
            f"- Chunk {i+1}: {s['section']} (score {s.get('relevance_score', 0):.2f})"
            for i, s in enumerate(source_info)
        )
        + "\n\nINSTRUCTIONS:\n"
        "- Read all context.\n"
        "- Focus on relevant & recent info.\n"
        "- Output valid JSON exactly as requested.\n"
    )
    system_prompt = [{"role": "system", "content": sys_msg.strip()}]
    user_prompt = {
        "role": "user",
        "content": f"RETRIEVED CLINICAL INFORMATION:\n{context}\n\nTASK: {cfg['instruction']}",
    }
    return system_prompt + (cfg.get("few_shots", []) if use_few_shots else []) + [
        user_prompt
    ]

# ╭──────────────────── JSON extraction helpers ───────────────────╮
def smart_extract_json(raw: str, keys: List[str]) -> dict:
    if not raw:
        return {}
    try:
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s >= 0 < e:
            return json.loads(raw[s:e])
    except json.JSONDecodeError:
        pass
    res = {}
    for k in keys:
        for pat in [
            rf'"{k}"\s*:\s*"([^"]+)"',
            rf"'{k}'\s*:\s*'([^']+)'",
            rf"{k}\s*:\s*([^\s,}}]+)",
        ]:
            m = re.search(pat, raw, re.I)
            if m:
                res[k] = m.group(1)
                break
    return res

# ╭────────────────── Datapoint extractor (flag aware) ─────────────╮
def extract_datapoint_with_rag(
    text: str,
    datapoint_name: str,
    config: dict,
    llm_model: str,
    temp: float = 0.0,
    top_k: int = 40,
    top_p: float = 0.95,
    patient_id: int = None,
    use_rag: bool = USE_RAG,
) -> Tuple[str, Dict, List[Dict]]:
    cfg = config[datapoint_name]
    json_key = cfg.get("json_key", "result")
    default = {json_key: "invalid"}

    if not text:
        return "", default, [{"error": "empty"}]

    query = get_datapoint_query(datapoint_name)
    context, src = retrieve_with_source_tracking(text, query, datapoint_name, use_rag)
    prompt = construct_rag_prompt(context, src, datapoint_name, config, True)

    raw = enhanced_ollama_llm(
        prompt,
        llm_model,
        cfg["output_format"],
        temp,
        top_k,
        top_p,
        patient_id,
        datapoint_name,
    )

    if not raw:
        return "", default, src

    if cfg["output_format"] == "json":
        if json_key == "medication_status":
            parsed = smart_extract_json(raw, ["steroid_status", "avastin_status"])
            cleaned = {json_key: parsed or default[json_key]}
        elif json_key == "radiation_date":
            parsed = smart_extract_json(raw, ["radiation_date"])
            cleaned = parsed or default
        elif json_key == "btrads_assessment":
            parsed = smart_extract_json(raw, ["score", "reasoning", "confidence"])
            cleaned = {json_key: parsed or default[json_key]}
        else:
            try:
                s, e = raw.find("{"), raw.rfind("}") + 1
                cleaned = json.loads(raw[s:e])
            except Exception:
                cleaned = default
    else:
        cleaned = {json_key: raw.strip()}

    return raw, cleaned, src

# ╭──────────────────── Date helpers & logger ─────────────────────╮
def simple_date_parser(ds):
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y", "%m-%d-%Y"):
        try:
            return datetime.strptime(str(ds), fmt).date()
        except Exception:
            continue
    return None


def calculate_days_between(a, b):
    A, B = simple_date_parser(a), simple_date_parser(b)
    return (B - A).days if A and B else -1


def log_raw_response(name, prompt, response, patient_id=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("raw_responses", exist_ok=True)
    fn = f"raw_responses/{name}{('_patient'+str(patient_id)) if patient_id else ''}_{ts}.json"
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(
            dict(
                timestamp=ts,
                datapoint=name,
                patient_id=patient_id,
                prompt=[p.get("content", "") for p in prompt if isinstance(p, dict)],
                response=response,
            ),
            f,
            indent=2,
        )

# ╭──────────────────── Datapoints configuration ──────────────────╮
datapoints_config = {
    # --- medication_status ---
    "medication_status": {
        "system_message": """You are an expert medical data extractor specializing in brain tumor patient care. 
Your task is to carefully read clinical notes and extract current medication status for steroids and Avastin (bevacizumab).

Pay careful attention to:
- Current vs. past medications
- Dose changes (increasing/decreasing/stable)  
- Treatment plans vs. actual current status
- Multiple medication lists in the same note""",
        "instruction": """Extract the current steroid and Avastin (bevacizumab) medication status from the provided clinical information.

STEROID STATUS - Look for dexamethasone, decadron, prednisolone, prednisone:
- 'none': Patient is not currently on steroids
- 'stable': Patient continues on same steroid dose
- 'increasing': Steroid dose being increased  
- 'decreasing': Steroid dose being tapered/decreased
- 'started': Patient newly started on steroids
- 'unknown': Cannot determine from the note

AVASTIN STATUS - Look for Avastin, bevacizumab:
- 'none': Patient is not on Avastin therapy
- 'ongoing': Patient continuing established Avastin therapy
- 'first_treatment': This is clearly the patient's first Avastin dose
- 'started': Recently started Avastin therapy  
- 'unknown': Cannot determine from the note

Return ONLY this JSON format:
{"steroid_status": "X", "avastin_status": "Y"}""",
        "output_format": "json",
        "json_key": "medication_status",
    },
    # --- radiation_date ---
    "radiation_date": {
        "system_message": """You are an expert medical data extractor specializing in brain tumor patient care.
Your task is to find when the patient completed their most recent course of radiation therapy.""",
        "instruction": """Find the date when this patient completed their most recent radiation therapy course.

Look for completion dates, not start dates. If given a date range, use the END date.
Consider both standard radiation and stereotactic radiosurgery (SRS).

Return ONLY this JSON format:
{"radiation_date": "MM/DD/YYYY"}
or if not found:
{"radiation_date": "unknown"}""",
        "output_format": "json",
        "json_key": "radiation_date",
    },
    # --- btrads_assessment ---
    "btrads_assessment": {
        "system_message": """You are a neuroradiology expert specializing in brain tumor assessment using the BT-RADS system.""",
        "instruction": """Analyze the provided data and assign the appropriate BT-RADS score (1a, 1b, 2, 3a, 3b, 3c, or 4).

Apply the BT-RADS algorithm based on volumetric changes, radiation timing, and medication status.

BT-RADS: Brain Tumor Assessment Algorithm
Initial Assessment

Determine if there is a suitable prior imaging study available

If NO → Classify as BTRADS score: 0 (Baseline)

This applies to initial diagnostic MRI, most recent post-op MRI, non-diagnostic study, or non-tumor findings obscuring diagnosis


If YES → Proceed to imaging assessment



Imaging Comparison Assessment

Compare current imaging with prior studies

If IMPROVED → Proceed to step 3
If UNCHANGED → Classify as BTRADS score: 2 (Stable)

Criteria: Unchanged enhancing component, unchanged FLAIR component, no new enhancing or FLAIR lesions, unchanged mass effect, clinically stable


If WORSE → Proceed to step 4


For IMPROVED imaging:

Determine if improvement is related to medication effects
a. If AVASTIN used:

Assess extent of improvement

If FIRST STUDY ON AVASTIN with ONLY ENHANCEMENT IMPROVED → Classify as BTRADS score: 1b (Possible medication effect)
If SUSTAINED IMPROVEMENT (>1 month follow-up) → Classify as BTRADS score: 1a (Improved)
b. If STEROID use:


If INCREASING STEROIDS → Classify as BTRADS score: 1b (Possible medication effect)
c. If NO MEDICATION EFFECT → Classify as BTRADS score: 1a (Improved)
Criteria: Decreased enhancing component, unchanged or decreased FLAIR component, no new enhancing or FLAIR lesions, unchanged or decreased mass effect, clinically stable or improved




For WORSE imaging:

Assess time since XRT (radiation therapy)
a. If <90 DAYS (within 12 weeks of completing most recent CRT) → Classify as BTRADS score: 3a (Favor treatment effect)

Criteria: Increased enhancing and/or FLAIR component, no new lesions outside XRT treatment zone, increased mass effect, clinically stable
b. If >90 DAYS → Proceed to step 5




For worsening >90 days post-XRT:

Determine what imaging features are worse
a. If BOTH FLAIR AND ENHANCEMENT:

Assess extent of worsening

If >25% INCREASE IN FLAIR AND ENHANCEMENT → Classify as BTRADS score: 4 (Highly suspicious)
If <25% INCREASE AND NO NEW LESIONS OUTSIDE XRT ZONE → Classify as BTRADS score: 3c (Favor tumor)
b. If EITHER FLAIR OR ENHANCEMENT alone:


Assess extent of worsening

If >25% INCREASE → Classify as BTRADS score: 4 (Highly suspicious)
If <25% INCREASE → Classify as BTRADS score: 3b (Indeterminate mix)
c. If NEW LESION outside XRT treatment zone:


If ENHANCING LESION → Classify as BTRADS score: 4 (Highly suspicious)
If NON-ENHANCING LESION (e.g., FLAIR only) → Classify as BTRADS score: 3c (Favor tumor)




For progressive assessment:

Determine if worsening is progressive over time
a. If WORSENING OVER MULTIPLE STUDIES → Classify as BTRADS score: 4 (Highly suspicious)
b. If FIRST TIME WORSE and CLINICALLY WORSE → Classify as BTRADS score: 3c (Favor tumor)



Return ONLY this JSON format:
{"score": "X", "reasoning": "explanation", "confidence": 0.X}""",
        "few_shots": [
            # {
            #     "role": "user",
            #     "content": "FLAIR volume change: -16.2%, Enhancement: -46.6%, Days since radiation: 120, No medications",
            # },
            # {
            #     "role": "assistant",
            #     "content": '{"score": "1a", "reasoning": "Clear improvement without medication effects", "confidence": 0.95}',
            # },
        ],
        "output_format": "json",
        "json_key": "btrads_assessment",
    },
}

# ╭────────────────── Patient-level processing ────────────────────╮
def process_btrads_with_source_tracking(
    row,
    config,
    llm_model,
    patient_id=None,
    temp: float = 0.0,
    top_k: int = 40,
    top_p: float = 0.95,
    use_rag: bool = USE_RAG,
):
    res = dict(
        steroid_status="unknown",
        avastin_status="unknown",
        radiation_date="unknown",
        days_since_radiation=-1,
        bt_rads_score="invalid",
        reasoning="Processing error",
        confidence=0.0,
        raw_output="",
        medication_sources=[],
        radiation_sources=[],
        btrads_sources=[],
    )

    note = row.get("clinical_note_closest", "")
    if not note or pd.isna(note):
        res["reasoning"] = "No clinical note"
        return res

    # 1. Medication
    _, meds, src_m = extract_datapoint_with_rag(
        note,
        "medication_status",
        config,
        llm_model,
        temp,
        top_k,
        top_p,
        patient_id,
        use_rag,
    )
    if isinstance(meds.get("medication_status"), dict):
        res["steroid_status"] = meds["medication_status"].get("steroid_status", "unknown")
        res["avastin_status"] = meds["medication_status"].get("avastin_status", "unknown")
    res["medication_sources"] = src_m

    # 2. Radiation
    _, rad, src_r = extract_datapoint_with_rag(
        note,
        "radiation_date",
        config,
        llm_model,
        temp,
        top_k,
        top_p,
        patient_id,
        use_rag,
    )
    res["radiation_date"] = rad.get("radiation_date", "unknown")
    res["radiation_sources"] = src_r
    followup_date = row.get("Followup_imaging_date")
    if res["radiation_date"] != "unknown" and followup_date:
        res["days_since_radiation"] = calculate_days_between(res["radiation_date"], followup_date)

    # 3. BT-RADS
    vol_ctx = f"""PATIENT DATA FOR BT-RADS ASSESSMENT:

Days since radiation: {res['days_since_radiation']}
Steroid status: {res['steroid_status']}
Avastin status: {res['avastin_status']}

CLINICAL NOTE EXCERPT:
{note[:1000]}...
"""
    raw, bt, src_b = extract_datapoint_with_rag(
        vol_ctx,
        "btrads_assessment",
        config,
        llm_model,
        temp,
        top_k,
        top_p,
        patient_id,
        use_rag,
    )
    if isinstance(bt.get("btrads_assessment"), dict):
        d = bt["btrads_assessment"]
        res["bt_rads_score"] = d.get("score", "invalid")
        res["reasoning"] = d.get("reasoning", "")
        try:
            res["confidence"] = float(d.get("confidence", 0))
        except Exception:
            pass
    res["raw_output"] = raw
    res["btrads_sources"] = src_b
    return res

# ╭────────────────── Checkpoint helpers ──────────────────────────╮
def save_checkpoint(df: pd.DataFrame, idx: int, path: str = "checkpoint.csv"):
    df.to_csv(path, index=False)
    with open("last_processed_index.txt", "w") as f:
        f.write(str(idx))
    print(f"Checkpoint saved at row {idx}")


def load_checkpoint() -> Tuple[Optional[pd.DataFrame], int]:
    try:
        if os.path.exists("checkpoint.csv") and os.path.exists("last_processed_index.txt"):
            with open("last_processed_index.txt") as f:
                idx = int(f.read().strip())
            df_ = pd.read_csv("checkpoint.csv")
            print(f"Loaded checkpoint (row {idx})")
            return df_, idx
        return None, -1
    except Exception as e:
        print(f"Checkpoint load error: {e}")
        return None, -1

# ╭────────────────── Accuracy tracker ────────────────────────────╮
class AccuracyTracker:
    def __init__(self):
        self.total = 0
        self.correct = 0
        self.records = []

    def update(self, gt, pred):
        if pd.isna(gt):
            return
        self.total += 1
        match = str(gt) == str(pred)
        if match:
            self.correct += 1
        self.records.append((gt, pred, match))

    def summary(self):
        acc = self.correct / self.total if self.total else 0
        return dict(accuracy=acc, correct=self.correct, total=self.total)

    def print_summary(self):
        s = self.summary()
        print(f"\nFINAL ACCURACY {s['correct']}/{s['total']} = {s['accuracy']:.3f}")

# ╭────────────────────────── main() ──────────────────────────────╮
def main():
    # ── I/O paths ────────────────────────────────────────────────
    file_path = "./input_file_id (version 1) - (Run).csv"
    backup_path = file_path[:-4] + "_original.csv"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"Backup created → {backup_path}")

    df, start_idx = load_checkpoint()
    if df is None:
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path, encoding="windows-1252")
        start_idx = 0

    # ── LLM models ───────────────────────────────────────────────
    llm_models = ["phi4:14b"]
    available = [m["name"] for m in ollama.list()["models"]]
    for m in llm_models:
        if m not in available:
            print(f"Pulling model {m} …")
            ollama.pull(m)

    # ── Prepare dataframe columns (once) ─────────────────────────
    if start_idx == 0:
        for m in llm_models:
            suffix = m.replace(":", "_").replace(".", "_")
            df[f"steroid_status_{suffix}"] = ""
            df[f"avastin_status_{suffix}"] = ""
            df[f"radiation_date_{suffix}"] = ""
            df[f"days_since_radiation_{suffix}"] = -1
            df[f"bt_rads_score_{suffix}"] = ""
            df[f"reasoning_{suffix}"] = ""
            df[f"confidence_{suffix}"] = 0.0
            df[f"score_match_{suffix}"] = ""
            df[f"medication_sources_{suffix}"] = ""
            df[f"radiation_sources_{suffix}"] = ""
            df[f"btrads_sources_{suffix}"] = ""

    tracker = AccuracyTracker()

    # ── Main processing loop ─────────────────────────────────────
    print(f"Processing rows {start_idx} → {len(df)-1}")
    for i in tqdm(range(start_idx, len(df)), desc="Patients"):
        for model in llm_models:
            suffix = model.replace(":", "_").replace(".", "_")
            try:
                res = process_btrads_with_source_tracking(
                    df.iloc[i],
                    datapoints_config,
                    model,
                    patient_id=i,
                    temp=0.0,
                    top_k=40,
                    top_p=0.95,
                    use_rag=USE_RAG,
                )
                df.at[i, f"steroid_status_{suffix}"] = res["steroid_status"]
                df.at[i, f"avastin_status_{suffix}"] = res["avastin_status"]
                df.at[i, f"radiation_date_{suffix}"] = res["radiation_date"]
                df.at[i, f"days_since_radiation_{suffix}"] = res["days_since_radiation"]
                df.at[i, f"bt_rads_score_{suffix}"] = res["bt_rads_score"]
                df.at[i, f"reasoning_{suffix}"] = res["reasoning"]
                df.at[i, f"confidence_{suffix}"] = res["confidence"]
                df.at[i, f"medication_sources_{suffix}"] = json.dumps(res["medication_sources"])
                df.at[i, f"radiation_sources_{suffix}"] = json.dumps(res["radiation_sources"])
                df.at[i, f"btrads_sources_{suffix}"] = json.dumps(res["btrads_sources"])



                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # INSERT VERBOSE PRINTS RIGHT HERE
                print(f"\nPATIENT {i} ({model})")
                print(f"  Steroid status   : {res['steroid_status']}")
                print(f"  Avastin status   : {res['avastin_status']}")
                print(f"  Radiation date   : {res['radiation_date']}  "
                    f"(Δdays {res['days_since_radiation']})")
                print(f"  BT-RADS score    : {res['bt_rads_score']}  "
                    f"(confidence {res['confidence']:.2f})")

                if "BTRADS (Precise Category)" in df.columns:
                    gt = df.iloc[i]["BTRADS (Precise Category)"]
                    pred = res["bt_rads_score"]
                    if not pd.isna(gt):
                        df.at[i, f"score_match_{suffix}"] = str(gt) == str(pred)
                        tracker.update(gt, pred)
            except Exception as e:
                print(f"Row {i} error: {e}")
                traceback.print_exc()

        if i % 10 == 0 or i == len(df) - 1:
            save_checkpoint(df, i)

    # ── Finalize ────────────────────────────────────────────────
    tracker.print_summary()
    out_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"btrads_results_{cli_args.mode}_{out_ts}.csv"
    df.to_csv(out_csv, index=False)
    df.to_csv("btrads_results_latest.csv", index=False)
    print(f"\n✓ Finished – results → {out_csv}")

    # ── Clean checkpoint files ───────────────────────────────────
    for f in ("checkpoint.csv", "last_processed_index.txt"):
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
