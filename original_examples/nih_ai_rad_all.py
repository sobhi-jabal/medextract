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
    in_red, in_yellow, in_green, in_blue, in_magenta, in_bold, underlined, on_green,
    on_yellow, on_red, on_blue
)

# --------------------- General Helper Functions ---------------------
@contextmanager
def suppress_stdout():
    """Utility to suppress console output (useful for quieting logs)."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def preprocess_text(text: str) -> str:
    """Preprocess text by normalizing newlines and removing extraneous spacing."""
    if not text or pd.isna(text):
        return ""
    text = str(text)  # Ensure text is string
    text = re.sub(r'\n(?!\.)', ' ', text)
    text = re.sub(r"\.\n", " \n ", text)
    return text.strip()

def get_text_chunks(text: str, chunk_size: int, chunk_overlap: int) -> Sequence[Document]:
    """Split text into overlapping chunks for retrieval-augmented workflows."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ":", "."]
    )
    return [Document(page_content=x) for x in text_splitter.split_text(text)]

# --------------------- Reranker Class ---------------------
class BgeRerank(BaseDocumentCompressor):
    """
    Example Reranker using the 'BAAI/bge-reranker-v2-m3' cross-encoder model.
    Adjust as needed.
    """
    model_name: str = 'BAAI/bge-reranker-v2-m3'
    top_n: int = 2
    model: CrossEncoder = CrossEncoder(model_name, device="cuda")

    def bge_rerank(self, query, docs):
        model_inputs = [[query, doc] for doc in docs]
        scores = self.model.predict(model_inputs)
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[:self.top_n]

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if len(documents) == 0:
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.bge_rerank(query, _docs)
        final_results = []
        for r in results:
            doc = doc_list[r[0]]
            doc.metadata["relevance_score"] = r[1]
            final_results.append(doc)
        return final_results

# --------------------- Datapoints Configuration ---------------------
datapoints_config = {
    
    "study_inclusion": {
        "instruction": (
            "Determine if this project is relevant to 'AI in Radiology.' Specifically, check if it involves both:\n"
            "(1) Artificial Intelligence (e.g., machine learning, deep learning, CNN, GAN, transformer, etc.) AND\n"
            "(2) Radiology or medical imaging (e.g., MRI, CT, PET, X-ray, ultrasound).\n\n"
            "If it clearly meets both criteria, return 'include'.\n"
            "If it clearly does NOT meet both criteria, return 'exclude'.\n"
            "If uncertain or ambiguous, return 'NR'.\n\n"
            "Return JSON in format: {'include': ''}."
        ),
        "few_shots": [
            {
                "role": "user",
                "content": "This study develops a convolutional neural network for analyzing brain MRI scans."
            },
            {
                "role": "assistant",
                "content": '{"include": "include"}'
            },
            {
                "role": "user",
                "content": "This grant focuses on basic lab research for new pharmaceuticals without any mention of imaging or AI."
            },
            {
                "role": "assistant",
                "content": '{"include": "exclude"}'
            },
            {
                "role": "user",
                "content": "The project aims to build novel computational methods, but does not specify imaging or radiology context."
            },
            {
                "role": "assistant",
                "content": '{"include": "NR"}'
            }
        ],
        "output_format": "json",
        "json_key": "include"
    },

    "diseases": {
        "instruction": (
            "Extract all diseases, medical conditions, and disorders mentioned in the abstract that are the focus of research or "
            "intervention. Include only specific conditions being studied or targeted, not general references to disease. "
            "Separate multiple conditions with semicolons. If no specific diseases are mentioned, return 'NR'. "
            "Return JSON in format: {'diseases': 'X'}."
        ),
        "valid_values": None,
        "few_shots": [
            {
                "role": "user",
                "content": (
                    "The proposal leverages complementary information of in-vivo amyloid-beta PET, tau PET, sMRI, cognitive, "
                    "clinical, and genetic measurements in patients with MCI/Alzheimer's disease relative to normal controls."
                )
            },
            {
                "role": "assistant",
                "content": '{"diseases": "Mild Cognitive Impairment; Alzheimer\'s disease"}'
            },
            {
                "role": "user",
                "content": (
                    "Chronic lung dysfunction (CLD) is a potentially severe complication of bone marrow transplantation (BMT), "
                    "particularly prevalent in patients who develop chronic graft versus host disease (GVHD) post-BMT."
                )
            },
            {
                "role": "assistant",
                "content": '{"diseases": "Chronic lung dysfunction; Chronic graft versus host disease"}'
            },
            {
                "role": "user",
                "content": (
                    "Using deep learning for analysis of brain MRI scans and cardiac function for improved "
                    "diagnosis of neurological conditions."
                )
            },
            {
                "role": "assistant",
                "content": '{"diseases": "NR"}'
            },
            {
                "role": "user",
                "content": (
                    "Mutations in the TTR gene can lead to hereditary transthyretin amyloid cardiomyopathy (hATTR-CM) and heart failure."
                )
            },
            {
                "role": "assistant",
                "content": '{"disease": "heart failure"}'
            },
            {
                "role": "user",
                "content": (
                    "Lennox-Gastaut Syndrome is a form of childhood onset epilepsy with cognitive dysfunction."
                )
            },
            {
                "role": "assistant",
                "content": '{"disease": "epilepsy"}'
            },
            {
                "role": "user",
                "content": "Study of stroke patients using advanced neuroimaging techniques."
            },
            {
                "role": "assistant",
                "content": '{"disease": "stroke"}'
            },
            {
                "role": "user",
                "content": (
                    "The research focuses on developing new machine learning algorithms for healthcare applications."
                )
            },
            {
                "role": "assistant",
                "content": '{"disease": "NR"}'
            },
            {
                "role": "user",
                "content": "Early detection of breast cancer using artificial intelligence and mammography."
            },
            {
                "role": "assistant",
                "content": '{"disease": "cancer"}'
            },
        ],
        "output_format": "json",
        "json_key": "diseases"
    },

    "disease": {
        "instruction": (
            "Extract the main disease or diseases or medical condition that is the primary focus of the research. "
            "Use well-known disease categories and common medical terminology. "
            "If no specific disease is the focus, return 'NR'. "
            "Return JSON in format: {'disease': 'X'}."
        ),
        "few_shots": [
            {
                "role": "user",
                "content": (
                    "The proposal leverages measurements in patients with mild cognitive impairment (MCI) and "
                    "Alzheimer's disease relative to normal controls."
                )
            },
            {
                "role": "assistant",
                "content": '{"disease": "Alzheimer\'s disease"}'
            },
            {
                "role": "user",
                "content": (
                    "This project aims to develop novel computational methods for analyzing medical imaging data "
                    "across various conditions."
                )
            },
            {
                "role": "assistant",
                "content": '{"disease": "NR"}'
            },
            {
                "role": "user",
                "content": "Study of stroke patients using advanced neuroimaging techniques."
            },
            {
                "role": "assistant",
                "content": '{"disease": "stroke"}'
            },
            {
                "role": "user",
                "content": (
                    "The research focuses on developing new machine learning algorithms for healthcare applications."
                )
            },
            {
                "role": "assistant",
                "content": '{"disease": "NR"}'
            },
            {
                "role": "user",
                "content": "Early detection of breast cancer using artificial intelligence and mammography."
            },
            {
                "role": "assistant",
                "content": '{"disease": "cancer"}'
            },
            {
                "role": "user",
                "content": (
                    "Improving image quality in pediatric radiology to reduce radiation exposure."
                )
            },
            {
                "role": "assistant",
                "content": '{"disease": "NR"}'
            },
            {
                "role": "user",
                "content": (
                    "Investigation of multiple sclerosis progression using longitudinal MRI scans."
                )
            },
            {
                "role": "assistant",
                "content": '{"disease": "multiple sclerosis"}'
            },
            {
                "role": "user",
                "content": (
                    "Development of quality assurance tools for medical imaging departments."
                )
            },
            {
                "role": "assistant",
                "content": '{"disease": "NR"}'
            },
            {
                "role": "user",
                "content": (
                    "The proposal leverages measurements in patients with mild cognitive impairment (MCI) and "
                    "Alzheimer's disease relative to normal controls."
                )
            },
            {
                "role": "assistant",
                "content": '{"disease": "Alzheimer\'s disease"}'
            },
            {
                "role": "user",
                "content": (
                    "Mutations in the TTR gene can lead to hereditary transthyretin amyloid cardiomyopathy (hATTR-CM) "
                    "and heart failure."
                )
            },
            {
                "role": "assistant",
                "content": '{"disease": "heart failure"}'
            },
            {
                "role": "user",
                "content": (
                    "Lennox-Gastaut Syndrome is a form of childhood onset epilepsy with cognitive dysfunction."
                )
            },
            {
                "role": "assistant",
                "content": '{"disease": "epilepsy"}'
            }
        ],
        "output_format": "json",
        "json_key": "disease"
    },

    "organs": {
        "instruction": (
            "Extract all organs and specific anatomical structures that are the target of study or imaging in the research. "
            "Include only organs/structures that are being directly studied, imaged, or treated. "
            "Separate multiple organs with semicolons. If no specific organs are targeted, return 'NR'. "
            "Return JSON in format: {'organs': 'X'}."
        ),
        "valid_values": None,
        "few_shots": [
            {
                "role": "user",
                "content": (
                    "Dr. Pirruccello proposes to develop models for additional aortic traits including thoracic "
                    "aortic strain and distensibility, and abdominal aortic diameter in the UK Biobank."
                )
            },
            {
                "role": "assistant",
                "content": '{"organs": "aorta"}'
            },
            {
                "role": "user",
                "content": (
                    "Patients will have two depth leads placed bilaterally in the centromedian nucleus of the thalamus "
                    "and two subdural strip leads placed bilaterally on the medial prefrontal cortex."
                )
            },
            {
                "role": "assistant",
                "content": '{"organs": "brain"}'
            },
            {
                "role": "user",
                "content": (
                    "The study aims to develop novel computational approaches for analyzing medical imaging data."
                )
            },
            {
                "role": "assistant",
                "content": '{"organs": "NR"}'
            },
            {
                "role": "user",
                "content": "Assessment of pulmonary nodules using deep learning on chest CT scans."
            },
            {
                "role": "assistant",
                "content": '{"organs": "lung"}'
            },
            {
                "role": "user",
                "content": (
                    "The study will examine how amyloid fibrils in the myocardium affect cardiac output."
                )
            },
            {
                "role": "assistant",
                "content": '{"organs": "heart"}'
            }
        ],
        "output_format": "json",
        "json_key": "organs"
    },

    "organ": {
        "instruction": (
            "Extract the primary organ or general anatomical structure that is the main focus of the study. "
            "Always use the general category rather than specific parts or regions. "
            "For example: 'prefrontal cortex' should be extracted as 'brain', 'thoracic aorta' as 'aorta', 'myocardium' as 'heart'. "
            "If multiple organs are mentioned, select the one that is central to the research aims. "
            "If no specific organ is the focus, return 'NR'. "
            "Return JSON in format: {'organ': 'X'}."
        ),
        "few_shots": [
            {
                "role": "user",
                "content": (
                    "Dr. Pirruccello proposes to develop models for additional thoracic aortic strain and distensibility."
                )
            },
            {
                "role": "assistant",
                "content": '{"organ": "aorta"}'
            },
            {
                "role": "user",
                "content": "Development of new machine learning algorithms for medical image analysis."
            },
            {
                "role": "assistant",
                "content": '{"organ": "NR"}'
            },
            {
                "role": "user",
                "content": (
                    "Study of hippocampal volume changes in Alzheimer's disease using MRI."
                )
            },
            {
                "role": "assistant",
                "content": '{"organ": "brain"}'
            },
            {
                "role": "user",
                "content": "Optimization of imaging protocols for radiation dose reduction."
            },
            {
                "role": "assistant",
                "content": '{"organ": "NR"}'
            },
            {
                "role": "user",
                "content": (
                    "Analysis of left ventricular ejection fraction using cardiac MRI."
                )
            },
            {
                "role": "assistant",
                "content": '{"organ": "heart"}'
            },
            {
                "role": "user",
                "content": "Development of quality control methods for medical imaging."
            },
            {
                "role": "assistant",
                "content": '{"organ": "NR"}'
            },
            {
                "role": "user",
                "content": (
                    "Assessment of pulmonary nodules using deep learning on chest CT scans."
                )
            },
            {
                "role": "assistant",
                "content": '{"organ": "lung"}'
            },
            {
                "role": "user",
                "content": (
                    "Integration of AI tools into clinical workflow for radiology departments."
                )
            },
            {
                "role": "assistant",
                "content": '{"organ": "NR"}'
            },
            {
                "role": "user",
                "content": (
                    "Dr. Pirruccello proposes to develop models for additional thoracic aortic strain and distensibility."
                )
            },
            {
                "role": "assistant",
                "content": '{"organ": "aorta"}'
            },
            {
                "role": "user",
                "content": (
                    "Patients will have leads placed in the centromedian nucleus of the thalamus and medial prefrontal cortex."
                )
            },
            {
                "role": "assistant",
                "content": '{"organ": "brain"}'
            },
            {
                "role": "user",
                "content": (
                    "The study will examine how amyloid fibrils in the myocardium affect cardiac output."
                )
            },
            {
                "role": "assistant",
                "content": '{"organ": "heart"}'
            }
        ],
        "output_format": "json",
        "json_key": "organ"
    },

    "imaging_modality": {
        "instruction": (
            "Extract the primary radiology imaging modality used in the research, focusing on standard categories: "
            "MRI, CT, PET, US (ultrasound), X-ray, SPECT, Fluoroscopy. "
            "Always use the general category rather than specific variants or protocols. "
            "If multiple modalities are present, select the one that appears most central to the research aims. "
            "If no standard radiology imaging modality is mentioned, return 'NR'. "
            "Return JSON in format: {'modality': 'X'}."
        ),
        "few_shots": [
            {
                "role": "user",
                "content": (
                    "Using amyloid-beta PET and tau PET imaging to study Alzheimer's disease progression."
                )
            },
            {
                "role": "assistant",
                "content": '{"modality": "PET"}'
            },
            {
                "role": "user",
                "content": "Development of new machine learning algorithms for healthcare applications."
            },
            {
                "role": "assistant",
                "content": '{"modality": "NR"}'
            },
            {
                "role": "user",
                "content": (
                    "The study will use cardiac MRI with contrast enhancement to assess heart function."
                )
            },
            {
                "role": "assistant",
                "content": '{"modality": "MRI"}'
            },
            {
                "role": "user",
                "content": "Analysis of electronic health records using natural language processing."
            },
            {
                "role": "assistant",
                "content": '{"modality": "NR"}'
            },
            {
                "role": "user",
                "content": "Fluoroscopy-guided catheter placement during interventional procedures."
            },
            {
                "role": "assistant",
                "content": '{"modality": "Fluoroscopy"}'
            },
            {
                "role": "user",
                "content": "Development of clinical decision support tools for radiologists."
            },
            {
                "role": "assistant",
                "content": '{"modality": "NR"}'
            },
            {
                "role": "user",
                "content": "Real-time ultrasound monitoring during surgical procedures."
            },
            {
                "role": "assistant",
                "content": '{"modality": "US"}'
            },
            {
                "role": "user",
                "content": "Implementation of AI-based workflow optimization in radiology."
            },
            {
                "role": "assistant",
                "content": '{"modality": "NR"}'
            },
            {
                "role": "user",
                "content": (
                    "Using amyloid-beta PET and tau PET imaging to study Alzheimer's disease progression."
                )
            },
            {
                "role": "assistant",
                "content": '{"modality": "PET"}'
            },
            {
                "role": "user",
                "content": (
                    "The study will use cardiac MRI with contrast enhancement to assess heart function."
                )
            },
            {
                "role": "assistant",
                "content": '{"modality": "MRI"}'
            },
            {
                "role": "user",
                "content": (
                    "High-resolution CT scans will be analyzed using the parametric response mapping technique."
                )
            },
            {
                "role": "assistant",
                "content": '{"modality": "CT"}'
            }
        ],
        "output_format": "json",
        "json_key": "modality"
    },

    "ai_technique": {
        "instruction": (
            "Extract the main or most advanced AI/ML technique that is central to the research methodology. "
            "Common techniques in medical imaging include: deep learning, reinforcement learning, convolutional neural networks (CNN), "
            "transformer models, generative adversarial networks (GAN), Bayesian optimization, federated learning, self-supervised learning, "
            "active learning, transfer learning, etc. "
            "If multiple techniques are mentioned, select the one that appears to be the primary approach. "
            "Focus on the most specific technique rather than general terms. "
            "If no AI/ML technique is mentioned, return 'NR'. "
            "Return JSON in format: {'technique': 'X'}."
        ),
        "few_shots": [
            {
                "role": "user",
                "content": (
                    "The system uses reinforcement learning to optimize MRI acquisition parameters in real-time."
                )
            },
            {
                "role": "assistant",
                "content": '{"technique": "reinforcement learning"}'
            },
            {
                "role": "user",
                "content": (
                    "Standard clinical assessment of cardiac function using established protocols."
                )
            },
            {
                "role": "assistant",
                "content": '{"technique": "NR"}'
            },
            {
                "role": "user",
                "content": (
                    "Using convolutional neural networks for automated lesion detection in brain MRI."
                )
            },
            {
                "role": "assistant",
                "content": '{"technique": "convolutional neural networks"}'
            },
            {
                "role": "user",
                "content": (
                    "Development of new contrast agents for improved MRI visualization."
                )
            },
            {
                "role": "assistant",
                "content": '{"technique": "NR"}'
            },
            {
                "role": "user",
                "content": (
                    "Self-supervised learning framework for utilizing unlabeled medical images."
                )
            },
            {
                "role": "assistant",
                "content": '{"technique": "self-supervised learning"}'
            },
            {
                "role": "user",
                "content": "Validation study of existing imaging biomarkers in clinical practice."
            },
            {
                "role": "assistant",
                "content": '{"technique": "NR"}'
            },
            {
                "role": "user",
                "content": (
                    "Generative adversarial networks for synthetic training data creation."
                )
            },
            {
                "role": "assistant",
                "content": '{"technique": "generative adversarial networks"}'
            },
            {
                "role": "user",
                "content": "Quality assessment of radiological images using standard metrics."
            },
            {
                "role": "assistant",
                "content": '{"technique": "NR"}'
            },
            {
                "role": "user",
                "content": (
                    "The system uses reinforcement learning to optimize MRI acquisition parameters in real-time."
                )
            },
            {
                "role": "assistant",
                "content": '{"technique": "reinforcement learning"}'
            },
            {
                "role": "user",
                "content": (
                    "A federated learning approach will be used to train the model across multiple hospitals."
                )
            },
            {
                "role": "assistant",
                "content": '{"technique": "federated learning"}'
            },
            {
                "role": "user",
                "content": (
                    "The transformer-based architecture will analyze temporal sequences of MRI scans."
                )
            },
            {
                "role": "assistant",
                "content": '{"technique": "transformer"}'
            }
        ],
        "output_format": "json",
        "json_key": "technique"
    },

    "project_type": {
        "instruction": (
            "Determine whether the research project is primarily focused on human subjects, animal models, or both. "
            "Look for explicit mentions of clinical trials, patient studies, animal experiments, or pre-clinical research. "
            "Return 'human' for human subject research, 'animal' for animal studies, 'both' if both are equally central, or 'NR' if not specified. "
            "Return JSON in format: {'type': 'X'}."
        ),
        "few_shots": [
            {
                "role": "user",
                "content": (
                    "The clinical trial will enroll 20 patients with LGS and medically intractable GOS at 6 epilepsy centers."
                )
            },
            {
                "role": "assistant",
                "content": '{"type": "human"}'
            },
            {
                "role": "user",
                "content": (
                    "Using transgenic mouse models and cell cultures, we will investigate the molecular mechanisms."
                )
            },
            {
                "role": "assistant",
                "content": '{"type": "animal"}'
            },
            {
                "role": "user",
                "content": (
                    "The study will first use rat models for pre-clinical testing, followed by a Phase I clinical trial."
                )
            },
            {
                "role": "assistant",
                "content": '{"type": "both"}'
            }
        ],
        "output_format": "json",
        "json_key": "type"
    },

    "main_device": {
        "instruction": (
            "Extract the primary device, tool, or technological system that is central to the research project. "
            "Focus on novel or specialized equipment, not common laboratory instruments. "
            "Include medical devices, imaging systems, therapeutic tools, monitoring systems, and specialized software platforms if they are core to the research. "
            "If multiple devices are mentioned, select the one that is most central to the research aims. "
            "If no specific device is central to the project, return 'NR'. "
            "Return JSON in format: {'device': 'X'}."
        ),
        "few_shots": [
            {
                "role": "user",
                "content": (
                    "Using the RNS® System, which is FDA approved for FOS, an early feasibility IDE study will be conducted."
                )
            },
            {
                "role": "assistant",
                "content": '{"device": "RNS System"}'
            },
            {
                "role": "user",
                "content": (
                    "Analysis of clinical outcomes using statistical methods and machine learning."
                )
            },
            {
                "role": "assistant",
                "content": '{"device": "NR"}'
            },
            {
                "role": "user",
                "content": (
                    "The Pipeline embolization device will be used for treating complex intracranial aneurysms."
                )
            },
            {
                "role": "assistant",
                "content": '{"device": "Pipeline embolization device"}'
            },
            {
                "role": "user",
                "content": (
                    "Development of new algorithms for processing medical images."
                )
            },
            {
                "role": "assistant",
                "content": '{"device": "NR"}'
            },
            {
                "role": "user",
                "content": (
                    "Implementation of a novel wearable ECG monitor for continuous cardiac tracking."
                )
            },
            {
                "role": "assistant",
                "content": '{"device": "wearable ECG monitor"}'
            },
            {
                "role": "user",
                "content": (
                    "Retrospective study of patient outcomes using electronic health records."
                )
            },
            {
                "role": "assistant",
                "content": '{"device": "NR"}'
            },
            {
                "role": "user",
                "content": (
                    "The study will test the new deep brain stimulator for treatment of movement disorders."
                )
            },
            {
                "role": "assistant",
                "content": '{"device": "deep brain stimulator"}'
            },
            {
                "role": "user",
                "content": (
                    "Investigation of genetic markers associated with disease progression."
                )
            },
            {
                "role": "assistant",
                "content": '{"device": "NR"}'
            },
            {
                "role": "user",
                "content": (
                    "Using the RNS® System, which is FDA approved for FOS, an early feasibility IDE study will be conducted."
                )
            },
            {
                "role": "assistant",
                "content": '{"device": "RNS System"}'
            },
            {
                "role": "user",
                "content": (
                    "The Pipeline embolization device will be used for treating complex intracranial aneurysms."
                )
            },
            {
                "role": "assistant",
                "content": '{"device": "Pipeline embolization device"}'
            },
            {
                "role": "user",
                "content": (
                    "Development of a wearable device for continuous monitoring of cardiac parameters."
                )
            },
            {
                "role": "assistant",
                "content": '{"device": "wearable cardiac monitor"}'
            }
        ],
        "output_format": "json",
        "json_key": "device"
    },

    "system": {
        "instruction": (
            "Extract all body systems mentioned in the abstract (e.g., cardiovascular, nervous, respiratory, etc.). "
            "If multiple systems are present, separate them with semicolons. If no systems are mentioned, return 'NR'. "
            "Return JSON in format: {'systems': 'X'}."
        ),
        "valid_values": None,
        "few_shots": [
            {
                "role": "user",
                "content": (
                    "Development of AI models to analyze both cardiovascular and nervous system disorders using multimodal imaging."
                )
            },
            {
                "role": "assistant",
                "content": '{"systems": "cardiovascular system; nervous system"}'
            }
        ],
        "output_format": "json",
        "json_key": "systems"
    },

    "medical_specialty": {
        "instruction": (
            "Extract all medical specialties mentioned or implied in the abstract (e.g., cardiology, neurology, radiology, "
            "neurosurgery, general surgery, etc.). "
            "If multiple specialties are present, separate them with semicolons. If no specialties are mentioned, return 'NR'. "
            "Return JSON in format: {'specialties': 'X'}."
        ),
        "valid_values": None,
        "few_shots": [
            {
                "role": "user",
                "content": (
                    "Collaboration between radiologists and neurologists to develop AI-based diagnostic tools."
                )
            },
            {
                "role": "assistant",
                "content": '{"specialties": "radiology; neurology"}'
            }
        ],
        "output_format": "json",
        "json_key": "specialties"
    },

    "imaging_modalities": {
        "instruction": (
            "Extract all imaging modalities and specific imaging techniques mentioned in the abstract. Include details about "
            "contrast or special protocols when specified. Common modalities include: MRI, MRA, CT, CTA, PET, SPECT, X-ray, ultrasound, etc. "
            "Include imaging variants like 'with contrast', 'without contrast', etc. when specified. "
            "If multiple modalities are present, separate them with semicolons. If no imaging modalities are mentioned, return 'NR'. "
            "Return JSON in format: {'modalities': 'X'}."
        ),
        "valid_values": None,
        "few_shots": [
            {
                "role": "user",
                "content": (
                    "The proposal leverages complementary information of in-vivo amyloid-beta PET, tau PET, structural "
                    "magnetic resonance imaging (sMRI), cognitive, clinical, and genetic measurements."
                )
            },
            {
                "role": "assistant",
                "content": '{"modalities": "Amyloid-beta PET; Tau PET; Structural MRI"}'
            },
            {
                "role": "user",
                "content": (
                    "Using the RNS® System, which is FDA approved for FOS, an early feasibility IDE study will be conducted "
                    "with structural magnetic resonance imaging and diffusion weighted imaging."
                )
            },
            {
                "role": "assistant",
                "content": '{"modalities": "Structural MRI; Diffusion weighted MRI"}'
            },
            {
                "role": "user",
                "content": "Project focuses on developing new computational methods for healthcare data analysis."
            },
            {
                "role": "assistant",
                "content": '{"modalities": "NR"}'
            },
            {
                "role": "user",
                "content": "Fluoroscopy-guided catheter placement during interventional procedures."
            },
            {
                "role": "assistant",
                "content": '{"modality": "Fluoroscopy"}'
            },
        ],
        "output_format": "json",
        "json_key": "modalities"
    },
}

# --------------------- Prompt Construction (Combined) ---------------------
def construct_combined_prompt(
    context: str,
    config: dict,
    use_few_shots: bool = True
) -> list:
    """
    Build a single prompt (list of messages) that:
      - Includes a system message
      - Optionally appends all few-shots from every datapoint
      - Ends with a user message that includes the final instructions for *all* datapoints
    """
    if not context.strip():
        return []

    # System prompt
    system_prompt = [
        {
            "role": "system",
            "content": (
                "You are an extremely reliable data extraction expert with extreme rigor and attention "
                "to detail. Follow the instructions precisely and make sure to extract the data accurately. "
                "Return only what is asked in JSON format."
            )
        }
    ]

    # If using few shots, gather them from each datapoint
    few_shot_messages = []
    if use_few_shots:
        for dp_name, dp_config in config.items():
            # Each dp_config has a list of 'few_shots' which are [ {"role": "...", "content": "..."} , ...]
            dp_few_shots = dp_config.get("few_shots", [])
            few_shot_messages.extend(dp_few_shots)

    # Build the final user instructions
    # We'll create a single big user instruction referencing all datapoints.
    instructions_text = []
    for dp_name, dp_config in config.items():
        instructions_text.append(
            f"** {dp_config['json_key']}: ** {dp_config['instruction']}"
        )

    # We want a single JSON with all datapoint keys. Let's list them:
    all_keys = [dp_config["json_key"] for dp_name, dp_config in config.items()]
    all_keys_str = ", ".join(all_keys)

    # Final user message content
    user_message_content = (
        f"Context:\n{context}\n\n"
        "Task: Extract the following data points from the context and return a single JSON with the keys:\n"
        f"{all_keys_str}\n\n"
        "Here are the instructions for each key:\n\n"
        + "\n".join(instructions_text)
        + "\n\n"
        "Return exactly one valid JSON object that contains all of these keys. "
        "Do not include any additional commentary. Provide only the JSON."
    )

    user_prompt = {"role": "user", "content": user_message_content}

    # Combine all messages
    final_prompt = []
    final_prompt.extend(system_prompt)
    final_prompt.extend(few_shot_messages)
    final_prompt.append(user_prompt)

    return final_prompt

# --------------------- Single LLM Call + JSON Parsing ---------------------
def ollama_llm(
    messages: list,
    llm_model: str,
    output_format: str,
    temp: float,
    top_k: int,
    top_p: float
) -> str:
    """Call the Ollama LLM. If output_format == 'json', we request JSON from Ollama."""
    if not messages:
        return ""
    try:
        response = ollama.chat(
            model=llm_model,
            format="json" if output_format == "json" else None,
            keep_alive=0,
            options=Options(
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                num_ctx=2048,
                seed=None,
            ),
            messages=messages
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error in ollama_llm: {str(e)}")
        return ""

# --------------------- Post-processing for Combined JSON ---------------------
def clean_combined_extracted_data(raw_output: str, config: dict) -> dict:
    """
    Attempt to parse the entire JSON that includes all fields at once.
    For each datapoint, initialize the value with 'NR' (or 'invalid') and overwrite if found.
    """
    # Default to 'NR' for every key rather than 'invalid'
    final_data = {}
    for dp_name, dp_conf in config.items():
        final_data[dp_conf["json_key"]] = "NR"  # or "invalid"

    if not raw_output.strip():
        return final_data

    try:
        cleaned_output = raw_output.strip()
        start_idx = cleaned_output.find("{")
        end_idx = cleaned_output.rfind("}") + 1
        if start_idx != -1 and end_idx != -1:
            cleaned_output = cleaned_output[start_idx:end_idx]
            parsed_data = json.loads(cleaned_output)

            # For each datapoint in config, if present in parsed_data, copy it over
            for dp_name, dp_conf in config.items():
                key = dp_conf["json_key"]
                if key in parsed_data:
                    value = parsed_data[key]
                    valid_values = dp_conf.get("valid_values", None)
                    if valid_values and value not in valid_values:
                        final_data[key] = "invalid"
                    else:
                        final_data[key] = value
    except json.JSONDecodeError:
        # If parsing fails, everything remains "NR" or "invalid"
        pass

    return final_data

# --------------------- Main Extraction (All Datapoints in One Call) ---------------------
def extract_all_datapoints_from_text(
    text: str,
    config: dict,
    llm_model: str,
    rag_enabled: bool,
    embeddings,
    retriever_type: str,
    reranker,
    use_few_shots: bool = True,
    temp: float = 0.0,
    top_k: int = 2,
    top_p: float = 0.9
) -> Tuple[str, Dict]:
    """
    Extract all datapoints in one LLM call.
    Returns (raw_output: str, final_data: dict)
       where final_data is a dict containing each datapoint's key & value.
    """
    # Build a dict with defaults "invalid"
    default_data = {}
    for dp_name, dp_config in config.items():
        default_data[dp_config["json_key"]] = "invalid"

    # Handle empty or invalid text
    if not text or pd.isna(text):
        return "", default_data

    preprocessed_text = preprocess_text(text)
    if not preprocessed_text:
        return "", default_data

    # If retrieval is enabled, build context from chunk retrieval
    if rag_enabled:
        chunks = get_text_chunks(preprocessed_text, chunk_size=70, chunk_overlap=20)
        db = FAISS.from_documents(chunks, embeddings)
        if reranker:
            rag_retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
            )
        else:
            rag_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

        if retriever_type == "ensemble":
            keyword_retriever = BM25Retriever.from_documents(chunks, k=1)
            rag_retriever = EnsembleRetriever(
                retrievers=[rag_retriever, keyword_retriever],
                weights=[0.25, 0.75]
            )
        elif retriever_type == "sequential":
            keyword_retriever = BM25Retriever.from_documents(chunks, k=3)
            # Example query:
            keyword_retrieved_docs = keyword_retriever.invoke("machine learning imaging")
            # Rebuild FAISS with narrower set of docs
            narrower_db = FAISS.from_documents(keyword_retrieved_docs, embeddings)
            if reranker:
                rag_retriever = ContextualCompressionRetriever(
                    base_compressor=reranker,
                    base_retriever=narrower_db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
                )
            else:
                rag_retriever = narrower_db.as_retriever(search_type="similarity", search_kwargs={"k":1})

        retrieved_docs = rag_retriever.invoke("machine learning imaging")
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    else:
        # Direct approach (no retrieval)
        context = preprocessed_text

    # Construct a single combined prompt for all datapoints
    prompt = construct_combined_prompt(context, config, use_few_shots)
    if not prompt:
        return "", default_data

    # LLM call
    try:
        raw_output = ollama_llm(prompt, llm_model, "json", temp, top_k, top_p)
        # Parse the combined JSON
        cleaned_data = clean_combined_extracted_data(raw_output, config)

        # Debug logs
        print("\n" + "-"*100 + "\n")
        print("SINGLE-CALL Extraction of all datapoints")
        print(f"Input text preview: {preprocessed_text}...")  # truncated preview
        print(f"Raw LLM output:\n{raw_output}")
        print(f"Cleaned data:\n{cleaned_data}")
        print("\n" + "-"*100 + "\n")

        return raw_output, cleaned_data
    except Exception as e:
        print(f"Error in extraction: {str(e)}")
        return "", default_data

# --------------------- Crash Recovery Functions ---------------------
def save_checkpoint(df: pd.DataFrame, index: int, checkpoint_path: str = "checkpoint.csv"):
    """Save current progress to a single checkpoint file"""
    df.to_csv(checkpoint_path, index=False)
    with open("last_processed_index.txt", "w") as f:
        f.write(str(index))
    print(f"\nCheckpoint saved at row {index}")

def load_checkpoint() -> Tuple[Optional[pd.DataFrame], int]:
    """Load the checkpoint if it exists"""
    try:
        if os.path.exists("checkpoint.csv") and os.path.exists("last_processed_index.txt"):
            with open("last_processed_index.txt", "r") as f:
                last_index = int(f.read().strip())
            df = pd.read_csv("checkpoint.csv")
            print(f"Loaded checkpoint from row {last_index}")
            return df, last_index
        return None, -1
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return None, -1

# --------------------- Main Script ---------------------
def main():
    # Automatically detect the file type (CSV or Excel)
    csv_path = "nih_report_data.csv"
    excel_path = "nih_report_data.xlsx"

    if os.path.exists(csv_path):
        file_path = csv_path
        file_extension = ".csv"
    elif os.path.exists(excel_path):
        file_path = excel_path
        file_extension = ".xlsx"
    else:
        raise FileNotFoundError("Neither CSV nor Excel file found.")

    backup_path = file_path[:-len(file_extension)] + "_original" + file_extension
    checkpoint_path = "checkpoint.csv"

    # Create backup if it doesn't exist
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)

    # Try to load from checkpoint first
    df, start_index = load_checkpoint()
    if df is None:
        # No checkpoint found, start fresh
        if file_extension == ".csv":
            df = pd.read_csv(file_path)
        elif file_extension in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        start_index = 0

        # Optionally modify the text to combine Title + Abstract
        df['Project Abstract'] = (
            'Project Title:\n' + df['Project Title'] + '\n\nProject Abstract:\n' + df['Project Abstract']
        )

    # You can filter columns if needed, or keep them all:
    # df = df[['Project Abstract', 'Project Terms','Project Title']]

    # Define the model(s) you want to run
    llm_models = [
        "phi4",
        # Add more local Ollama models if needed
        # e.g., "deepseek-r1:14b", "llama2:7b-instruct-q4", etc.
    ]

    # Ensure models are pulled
    pulled_model_names = [model_info['name'] for model_info in ollama.list()['models']]
    for llm_model in llm_models:
        if llm_model not in pulled_model_names:
            print(f"Pulling model: {llm_model}")
            ollama.pull(llm_model)

    # Initialize columns if starting fresh
    # We'll create one "raw" output column per model, and then one column per datapoint.
    if start_index == 0:
        for model_name in llm_models:
            model_suffix = model_name.replace(':', '_')
            # Single raw column for the entire JSON
            combined_raw_col = f"all_datapoints_{model_suffix}_raw"
            if combined_raw_col not in df.columns:
                df[combined_raw_col] = np.nan

            # Now create individual columns for each datapoint
            for dp_name, dp_config in datapoints_config.items():
                clean_col = f"{dp_name}_{model_suffix}"
                if clean_col not in df.columns:
                    df[clean_col] = np.nan

    # Process each row from the last successful index
    print(f"Starting processing from index {start_index}")
    for i in tqdm(range(start_index, len(df))):
        report_text = df.at[i, "Project Abstract"]

        if pd.notna(report_text):
            for model_name in llm_models:
                model_suffix = model_name.replace(':', '_')

                print(f"\nProcessing row {i} with model {model_name}")
                try:
                    # Single LLM call to extract all datapoints
                    raw_output, cleaned_dict = extract_all_datapoints_from_text(
                        text=report_text,
                        config=datapoints_config,
                        llm_model=model_name,
                        rag_enabled=False,      # or True if you want retrieval
                        embeddings=None,        # set your embeddings instance if RAG is enabled
                        retriever_type="no_rag",# or "ensemble"/"sequential" if RAG is used
                        reranker=None,
                        use_few_shots=True,
                        temp=0.0,
                        top_k=2,
                        top_p=0.9
                    )

                    # Store the entire raw JSON
                    combined_raw_col = f"all_datapoints_{model_suffix}_raw"
                    df.at[i, combined_raw_col] = raw_output if raw_output else "no_output"

                    # Store each datapoint from the cleaned_dict
                    for dp_name, dp_config in datapoints_config.items():
                        dp_col = f"{dp_name}_{model_suffix}"
                        extracted_value = cleaned_dict.get(dp_config["json_key"], "invalid")
                        df.at[i, dp_col] = extracted_value

                except Exception as e:
                    print(f"Error processing row {i} with model {model_name}: {str(e)}")
                    # Just mark them invalid
                    combined_raw_col = f"all_datapoints_{model_suffix}_raw"
                    df.at[i, combined_raw_col] = "error"
                    for dp_name, dp_config in datapoints_config.items():
                        dp_col = f"{dp_name}_{model_suffix}"
                        df.at[i, dp_col] = "invalid"

        # Save checkpoint every 5 rows
        if i % 5 == 0:
            save_checkpoint(df, i, checkpoint_path)

    # Save final results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = f"nih_ai_results_{timestamp}.csv"
    df.to_csv(final_path, index=False)

    # Also save to a standard name for convenience
    df.to_csv("nih_ai_results.csv", index=False)
    print("\nExtraction complete. Results saved to:")
    print(f"1. {final_path}")
    print(f"2. nih_ai_results.csv")

    # Clean up checkpoint files
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    if os.path.exists("last_processed_index.txt"):
        os.remove("last_processed_index.txt")
    print("Cleaned up checkpoint files")

def delete_checkpoints():
    """Clean up checkpoint files."""
    if os.path.exists("checkpoint.csv"):
        os.remove("checkpoint.csv")
    if os.path.exists("last_processed_index.txt"):
        os.remove("last_processed_index.txt")
    print("Deleted checkpoint files")

def print_extraction_statistics(df: pd.DataFrame, model_name: str):
    """Print statistics about the extraction results for each datapoint."""
    model_suffix = model_name.replace(':', '_')

    print(f"\nExtraction Statistics for model {model_name}:")
    print("-" * 50)

    for dp_name, dp_config in datapoints_config.items():
        clean_col = f"{dp_name}_{model_suffix}"
        total_records = len(df)
        valid_records = df[clean_col].notna().sum()
        invalid_records = (df[clean_col] == "invalid").sum()
        nr_records = (df[clean_col] == "NR").sum()

        print(f"\nDatapoint: {dp_name}")
        print(f"  Total records: {total_records}")
        print(f"  Valid extractions (non-NaN): {valid_records}")
        print(f"  Invalid extractions: {invalid_records}")
        print(f"  'NR' extractions: {nr_records}")

# Example usage if you directly run the script:
if __name__ == "__main__":
    try:
        main()

        # After extraction, load results and print statistics
        results_df = pd.read_csv("nih_ai_results.csv")
        for model in ["phi4"]:
            print_extraction_statistics(results_df, model)

        # Clean up checkpoint files (optional)
        delete_checkpoints()

    except Exception as e:
        print(f"An error occurred in main execution: {str(e)}")
        raise e
