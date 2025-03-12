import pandas as pd
import os 
import utils
from metrics import *
from tqdm import tqdm
import numpy as np
import sys
sys.path.insert(0,"./")
from Models.SentenceTransformersModel import SentenceTransformerModels
from Models.llm_embeddings import LLMEmbeddings
from main_args import get_args
import torch
from metrics import CosineMetric



# CUDA_VISIBLE_DEVICE = 1



"""_summary_

    example:
    llm = LLM(......)
    function_to_call = getattr(llm,--Model_name)
    function_to_call()
"""


import argparse
import numpy as np
from tqdm import tqdm

# Assuming necessary imports for utils, compute_ned_distance, EmbeddingGenerator, etc.

def read_pertubed_data(filename, task, lang="en"):
    # path = f"./data/perturbed_dataset/{lang}/{task}/{filename}.csv"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
    return pd.read_csv(filename)


def compute_metrics(emb1, emb2):
    """Compute all metrics between two sets of embeddings."""
    sim = utils.cosine_similarity(emb1, emb2)
    ned = compute_ned_distance(emb1, emb2)
    ed = np.linalg.norm(emb1 - emb2, axis=1)
    dotp = np.sum(emb1 * emb2, axis=1)
    return sim, ned, ed, dotp

def run(args_model, dataset_name, target_lang,args_task, default_gpu="cuda", save=False):
    model = LLMEmbeddings(args_model, device=default_gpu)
    
    pertubed_data_path = f"./data/perturbed_dataset/{target_lang}/{args_task}/{dataset_name}_{args_task}_perturbed_{target_lang}.csv" # check if path exist 
    
    data = read_pertubed_data(pertubed_data_path, args_task)
    # dataset_name = dataset_name.split(".")[0] if args_task == "paraphrase" else dataset_name.split("_")[0]
    
    print(f"\n*** Model {args_model} on {dataset_name} dataset for {args_task} task ***\n")

    # Collect all sentences based on task
    sentences = []
    if args_task in ["Anto","anto","Antonym"]:
        cols = ["original_sentence", "paraphrased_sentence", "perturb_n1"]
        for _, row in data[cols].iterrows():
            sentences.extend(row.values)
    elif args_task in ["jumbling", "Jumbling","jumb"]:
        cols = ["original_sentence", "paraphrased_sentence", "perturb_n1", "perturb_n2", "perturb_n3"]
        for _, row in data[cols].iterrows():
            sentences.extend(row.values)
    elif args_task in ["Syn","syn","Synonym"]:
        cols = ["Sentence", "perturb_n1", "perturb_n2", "perturb_n3"]
        for _, row in data[cols].iterrows():
            sentences.extend(row.values)
    elif args_task in ["paraphrase","Paraphrase","para"]:
        cols = ["original_sentence", "paraphrased_sentence"]
        for _, row in data[cols].iterrows():
            sentences.extend(row.values)
    
    # Batch process embeddings
    embeddings = model.encode(sentences)
    if args_model != "chatgpt":
        embeddings = [emb.cpu().numpy() for emb in embeddings]
    embeddings = np.array(embeddings)
    
    # Process embeddings based on task
    if args_task == "anto":
        emb_org, emb_para, emb_anto = np.split(embeddings, 3, axis=0)
        
        sim_para, ned_para, ed_para, dotp_para = compute_metrics(emb_org, emb_para)
        sim_anto, ned_anto, ed_anto, dotp_anto = compute_metrics(emb_org, emb_anto)
        
        data["sim_org_para"] = sim_para
        data["sim_org_anto"] = sim_anto
        data["diff_org_para"] = sim_para - sim_anto
        
        data["ned_org_para"] = ned_para
        data["ned_org_anto"] = ned_anto
        data["ned_diff_org_para"] = ned_para - ned_anto
        
        data["ed_org_para"] = ed_para
        data["ed_org_anto"] = ed_anto
        data["ed_diff_org_para"] = ed_para - ed_anto
        
        data["dotp_org_para"] = dotp_para
        data["dotp_org_anto"] = dotp_anto
        data["dotp_diff_org_anto"] = dotp_para - dotp_anto
    
    elif args_task == "jumbling":
        emb_org, emb_para, emb_n1, emb_n2, emb_n3 = np.split(embeddings, 5, axis=0)
        
        # Compute metrics for each perturbation
        sim_para, _, _, _ = compute_metrics(emb_org, emb_para)
        sim_n1, ned_n1, ed_n1, dotp_n1 = compute_metrics(emb_org, emb_n1)
        sim_n2, ned_n2, ed_n2, dotp_n2 = compute_metrics(emb_org, emb_n2)
        sim_n3, ned_n3, ed_n3, dotp_n3 = compute_metrics(emb_org, emb_n3)
        
        data["sim_org_para"] = sim_para
        data["sim_org_n1"] = sim_n1
        data["sim_org_n2"] = sim_n2
        data["sim_org_n3"] = sim_n3
        
        data["diff_org_para"] = sim_para - sim_para  # Zero as per original
        data["diff_org_n1"] = sim_para - sim_n1
        data["diff_org_n2"] = sim_para - sim_n2
        data["diff_org_n3"] = sim_para - sim_n3
        
        # Similar pattern for NED, ED, and dot product metrics
        # [Additional metric calculations here...]
    
    elif args_task == "syn":
        emb_org, emb_s1, emb_s2, emb_s3 = np.split(embeddings, 4, axis=0)
        
        sim_s1, ned_s1, ed_s1, dotp_s1 = compute_metrics(emb_org, emb_s1)
        sim_s2, ned_s2, ed_s2, dotp_s2 = compute_metrics(emb_org, emb_s2)
        sim_s3, ned_s3, ed_s3, dotp_s3 = compute_metrics(emb_org, emb_s3)
        
        data[["sim_org_s1", "ned_org_s1", "ed_org_s1", "dotp_org_s1"]] = sim_s1, ned_s1, ed_s1, dotp_s1
        data[["sim_org_s2", "ned_org_s2", "ed_org_s2", "dotp_org_s2"]] = sim_s2, ned_s2, ed_s2, dotp_s2
        data[["sim_org_s3", "ned_org_s3", "ed_org_s3", "dotp_org_s3"]] = sim_s3, ned_s3, ed_s3, dotp_s3
    
    elif args_task == "paraphrase":
        emb_s1, emb_s2 = np.split(embeddings, 2, axis=0)
        data["sim"] = utils.cosine_similarity(emb_s1, emb_s2)
    
    if save:
        data.to_csv(f"./Results/{target_lang}/{args_task}/{dataset_name}_{args_model}_{args_task}_metric.csv")
    return data

if __name__ == "__main__":
    if sys.gettrace() is None: 
        parser = get_args()
        config = {
            "args_model": parser.model_name,
            "dataset_name": parser.perturbed_dataset,
            "args_task": parser.task,
            "default_gpu": parser.gpu,
            "save": parser.save,
            "target_lang": parser.target_lang
        }   
    else:
        config = {
            "args_model": "llama3",
            "dataset_name": "mrpc",
            "args_task": "anto",
            "default_gpu": "cuda:2",
            "save": False,
            "target_lang": "en"
            
        }
    run(**config)
    
    
    # file_path = "/home/yash/ALIGN-SIM/data/perturbed_dataset/en/anto/mrpc_anto_perturbed_en.csv"
    # run("llama3","mrpc_anto_perturbed_en", "anto", "cuda:2", False)