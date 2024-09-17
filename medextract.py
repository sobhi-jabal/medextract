import pandas as pd
import numpy as np
import re
import os
import ast
from tqdm.auto import tqdm
import yaml
import itertools
import json
from datetime import datetime
import time
from interruptingcow import timeout
import importlib
import argparse
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
# from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
import ollama
from ollama import Options
from sentence_transformers import CrossEncoder
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from typing import Sequence, Optional
from langchain.schema import Document

class BgeRerank(BaseDocumentCompressor):
    model_name: str = 'BAAI/bge-reranker-v2-m3'
    top_n: int = 2
    model: CrossEncoder = CrossEncoder(model_name, device="cuda")

    def bge_rerank(self, query, docs):
        model_inputs = [[query, doc] for doc in docs]
        scores = self.model.predict(model_inputs)
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[:self.top_n]

    def compress_documents(self, documents: Sequence[Document], query: str, callbacks: Optional[Callbacks] = None) -> Sequence[Document]:
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

def load_config(config_path='config/config.yaml', default_config_path='config/default_config.yaml'):
    with open(default_config_path, 'r') as file:
        default_config = yaml.safe_load(file)
    
    with open(config_path, 'r') as file:
        user_config = yaml.safe_load(file)
    
    if user_config.get('use_default_config', True):
        return default_config
    else:
        return {**default_config, **user_config}

def check_library_versions():
    for lib, version in config['library_versions'].items():
        current_version = importlib.import_module(lib).__version__
        if current_version != version:
            print(f"Warning: {lib} version mismatch. Expected {version}, but found {current_version}")

def preprocess_text(text):
    text = re.sub(r'\n(?!\.)', ' ', text)
    text = re.sub(r"\.\n", " \n ", text)
    return text

def get_text_chunks(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ":", "."]
    )
    return [Document(page_content=x) for x in text_splitter.split_text(text)]

def ollama_llm(context, llm_model, simple_prompting, fewshots_method, fewshots_with_NR_method, fewshots_with_NR_extended_method, json_value, temp, top_k, top_p):
    response = ollama.chat(
        model=llm_model, 
        format="json" if json_value else None,
        keep_alive=config['advanced_llm']['keep_alive'],
        options=Options(
            temperature=temp, 
            top_k=top_k,
            top_p=top_p,
            num_predict=config['advanced_llm']['num_predict'],
            mirostat_tau=config['advanced_llm']['mirostat_tau'],
        ), 
        messages=construct_prompt(context, simple_prompting, fewshots_method, fewshots_with_NR_method, fewshots_with_NR_extended_method)
    )
    return response['message']['content']

def construct_prompt(context, simple_prompting, fewshots_method, fewshots_with_NR_method, fewshots_with_NR_extended_method):
    target_variable = config['evaluation']['target_variable']
    formatted_prompt = f"Question: Extract the {target_variable} from the given medical report. If not found, return 'NR'. Answer in JSON format.\nContext: {context}"
    
    system_prompt = config['system_prompts']['simple'] if simple_prompting else config['system_prompts']['complex']
    system_prompt = system_prompt.format(target_variable=target_variable)
    
    system_messages = [{"role": "system", "content": system_prompt}]
    
    if fewshots_method:
        system_messages.extend([
            {"role": "user", "content": ex['input']}
            for ex in config['few_shot_examples'].values()
        ])
        system_messages.extend([
            {"role": "assistant", "content": ex['output']}
            for ex in config['few_shot_examples'].values()
        ])
    
    user_message = [{'role': 'user', 'content': formatted_prompt}]
    
    return system_messages + user_message

def process_text(text, llm_model, rag_enabled, embeddings, retriever_type, reranker, simple_prompting, fewshots_method, fewshots_with_NR_method, fewshots_with_NR_extended_method, json_value, temp, top_k, top_p):
    if rag_enabled:
        chunks = get_text_chunks(text, chunk_size=config['rag']['chunk_size'], chunk_overlap=config['rag']['chunk_overlap'])
        db = FAISS.from_documents(chunks, embeddings)
        
        if reranker:
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=reranker, base_retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
            )
        else:
            compression_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

        if retriever_type == "ensemble":
            keyword_retriever = BM25Retriever.from_documents(chunks, k=2)
            retriever = EnsembleRetriever(retrievers=[compression_retriever, keyword_retriever], weights=[0.25, 0.75])
        else:
            retriever = compression_retriever

        retrieved_docs = retriever.invoke(config['evaluation']['target_variable'])
        formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    else:
        chunks = get_text_chunks(text, chunk_size=100000, chunk_overlap=200)
        formatted_context = text

    result = ollama_llm(
        context=formatted_context, 
        llm_model=llm_model, 
        simple_prompting=simple_prompting, 
        fewshots_method=fewshots_method, 
        fewshots_with_NR_method=fewshots_with_NR_method, 
        fewshots_with_NR_extended_method=fewshots_with_NR_extended_method, 
        json_value=json_value, 
        temp=temp, 
        top_k=top_k, 
        top_p=top_p,
    )
    
    return result

def process_model(column_name, file_path, df, batch_size, llm_model, rag_enabled, embeddings, retriever_type, use_reranker, simple_prompting, fewshots_method, fewshots_with_NR_method, fewshots_with_NR_extended_method, json_value, temp, top_k, top_p, verbose):
    if use_reranker:
        reranker = BgeRerank()
    
    with tqdm(total=batch_size, desc='Report Processing', unit='report', leave=True) as report_pbar:
        for i in range(batch_size):
            if pd.notna(df.at[i, column_name]):
                report_pbar.update(1)
            else:
                report_text = df.loc[i, "Report Text"]
                if pd.notna(report_text):
                    try:
                        with timeout(config['processing']['timeout_duration'], exception=RuntimeError):
                            preprocessed_text = preprocess_text(report_text)
                            processed_text = process_text(
                                text=preprocessed_text,
                                llm_model=llm_model,
                                rag_enabled=rag_enabled,
                                embeddings=embeddings, 
                                retriever_type=retriever_type,
                                reranker=reranker,
                                simple_prompting=simple_prompting,
                                fewshots_method=fewshots_method,
                                fewshots_with_NR_method=fewshots_with_NR_method,
                                fewshots_with_NR_extended_method=fewshots_with_NR_extended_method,
                                json_value=json_value,
                                temp=temp,
                                top_k=top_k,
                                top_p=top_p
                            )
                    except RuntimeError:
                        print("Interrupted due to timeout")
                        processed_text = json.dumps({config['evaluation']['target_variable']: "timeout"})
                    
                    df.at[i, column_name] = processed_text.rstrip(' \n')
                    
                    if verbose:
                        print("*" * 100)
                        print(f"Processed text: {processed_text}")
                        print(f"Ground truth: {df.at[i, config['evaluation']['target_variable']]}")
                        print("*" * 100)
                    
                    if i % config['processing']['csv_save_frequency'] == 0 or i == batch_size - 1:
                        df.to_csv(file_path, index=False)
                
                report_pbar.update(1)
    
    return df

def clean_extracted_value(value):
    try:
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, dict):
            value = value.get(config['evaluation']['target_variable'], "invalid")
        return value if value in config['evaluation']['valid_values'] else "invalid"
    except:
        return "invalid"

def evaluate_experiment(df, column_name, figures_path, metrics_file_path, log_file_path, eval_id):
    df_exp = df[~df[column_name].isna()].copy()
    df_exp[column_name + '_cleaned'] = df_exp[column_name].apply(clean_extracted_value)
    
    y_pred = df_exp[column_name + '_cleaned']
    y_test = df_exp[config['evaluation']['target_variable']]
    
    metrics_dict = {
        "Accuracy": metrics.accuracy_score(y_test, y_pred),
        "Macro Precision": metrics.precision_score(y_test, y_pred, average='macro', zero_division=0),
        "Micro Precision": metrics.precision_score(y_test, y_pred, average='micro', zero_division=0),
        "Macro Recall": metrics.recall_score(y_test, y_pred, average='macro', zero_division=0),
        "Micro Recall": metrics.recall_score(y_test, y_pred, average='micro', zero_division=0),
        "Macro F1": metrics.f1_score(y_test, y_pred, average='macro', zero_division=0),
        "Micro F1": metrics.f1_score(y_test, y_pred, average='micro', zero_division=0),
        "Reports Evaluated": len(df_exp),
    }
    
    save_confusion_matrix(y_test, y_pred, column_name, figures_path, eval_id)
    update_metrics_csv(metrics_dict, column_name, metrics_file_path)
    append_metrics_to_log(metrics_dict, column_name, log_file_path, eval_id)
    
    return metrics_dict["Accuracy"]

def save_confusion_matrix(y_test, y_pred, column_name, figures_path, eval_id):
    os.makedirs(figures_path, exist_ok=True)
    all_labels = sorted(list(set(y_test) | set(y_pred)))
    cm = metrics.confusion_matrix(y_test, y_pred, labels=all_labels)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap='YlGnBu', xticklabels=all_labels, yticklabels=all_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    cm_path = os.path.join(figures_path, f"{column_name}_confusion_matrix_eval_{eval_id}.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

def update_metrics_csv(metrics_dict, column_name, metrics_file_path):
    os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_df = pd.DataFrame([{"Column Name": column_name, "Timestamp": current_time, **metrics_dict}])
    if os.path.exists(metrics_file_path):
        metrics_df.to_csv(metrics_file_path, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(metrics_file_path, index=False)

def append_metrics_to_log(metrics_dict, column_name, log_file_path, eval_id):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_df = pd.DataFrame([{**{"Column Name": column_name, "Eval ID": eval_id, "Timestamp": current_time}, **metrics_dict}])
    if os.path.exists(log_file_path):
        log_df.to_csv(log_file_path, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file_path, index=False)

def main():
    parser = argparse.ArgumentParser(description='MedExtract: Clinical Datapoint Extraction System')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    
    global config
    config = load_config(args.config)
    
    check_library_versions()
    
    # Ensure input file exists
    if not os.path.exists(config['file_paths']['input']):
        raise FileNotFoundError(f"Input file not found: {config['file_paths']['input']}")
    
    df = pd.read_csv(config['file_paths']['input'])
    df = df[~df["Report Text"].isna()]
    df = df[df[config['evaluation']['target_variable']].isin(config['evaluation']['valid_values'])]
    df.reset_index(inplace=True, drop=True)
    
    batch_size = len(df) if config['processing']['process_all'] else config['processing']['batch_size']
    
    best_model = None
    highest_accuracy = 0
    
    if config['run_benchmark']:
        param_combinations = list(itertools.product(
            config['models']['llm_models'],
            [True, False],  # rag_enabled
            config['embedding_models'],
            config['retriever']['types'],
            [True, False],  # use_reranker
            [True, False],  # simple_prompting
            [True, False],  # fewshots_method
            [True, False],  # fewshots_with_NR_method
            [True, False],  # fewshots_with_NR_extended_method
            [True, False],  # json_value
            config['sampling']['temperatures'],
            config['sampling']['top_ks'],
            config['sampling']['top_ps']
        ))
    else:
        param_combinations = [(
            config['models']['llm_models'][0],
            config['rag']['enabled'],
            config['embedding_models'][0],
            config['retriever']['types'][0],
            config['retriever']['use_reranker'],
            config['prompting']['simple_prompting'],
            config['prompting']['fewshots_method'],
            config['prompting']['fewshots_with_NR_method'],
            config['prompting']['fewshots_with_NR_extended_method'],
            config['output']['json_format'],
            config['sampling']['temperatures'][0],
            config['sampling']['top_ks'][0],
            config['sampling']['top_ps'][0]
        )]
    
    for params in param_combinations:
        llm_model, rag_enabled, embedding_model, retriever_type, use_reranker, simple_prompting, fewshots_method, fewshots_with_NR_method, fewshots_with_NR_extended_method, json_value, temp, top_k, top_p = params
        
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model) if embedding_model != "mistral" else OllamaEmbeddings(model="mistral")
        
        column_name = config['column_name_format'].format(
            target_variable=config['evaluation']['target_variable'],
            model=llm_model.replace("/", "__"),
            rag=rag_enabled,
            embeddings=embedding_model.replace("/", "__"),
            retriever=retriever_type,
            reranker=use_reranker,
            simple=simple_prompting,
            fewshots=fewshots_method,
            nr=fewshots_with_NR_method,
            nr_extended=fewshots_with_NR_extended_method,
            json=json_value,
            temp=temp,
            top_k=top_k,
            top_p=top_p
        )
        
        if column_name not in df.columns:
            df[column_name] = np.nan
        
        print(f"Processing model: {column_name}")
        
        df = process_model(
            column_name=column_name,
            file_path=config['file_paths']['input'],
            df=df,
            batch_size=batch_size,
            llm_model=llm_model,
            rag_enabled=rag_enabled,
            embeddings=embeddings,
            retriever_type=retriever_type,
            use_reranker=use_reranker,
            simple_prompting=simple_prompting,
            fewshots_method=fewshots_method,
            fewshots_with_NR_method=fewshots_with_NR_method,
            fewshots_with_NR_extended_method=fewshots_with_NR_extended_method,
            json_value=json_value,
            temp=temp,
            top_k=top_k,
            top_p=top_p,
            verbose=config['processing']['verbose']
        )
        
        accuracy = evaluate_experiment(
            df,
            column_name,
            config['file_paths']['figures'],
            config['file_paths']['metrics'],
            config['file_paths']['log'],
            len(df.columns) - 1
        )
        
        print(f"Accuracy: {accuracy}")
        
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_model = params
        
        df.to_csv(config['file_paths']['input'], index=False)
    
    print(f"Best model: {best_model}")
    print(f"Highest accuracy: {highest_accuracy}")

if __name__ == "__main__":
    main()