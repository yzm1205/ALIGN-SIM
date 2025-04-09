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

def compute_metrics(emb1, emb2,metric="cosine"):
    """Compute all metrics between two sets of embeddings."""
    # sim = utils.cosine_similarity(emb1, emb2)
    # ned = compute_ned_distance(emb1, emb2)
    # ed = np.linalg.norm(emb1 - emb2, axis=1)
    # dotp = np.sum(emb1 * emb2, axis=1)
    if metric=="cosine":
        sim = CosineMetric(emb1,emb2)
    return sim

def run(args_model, dataset_name, target_lang,args_task, default_gpu="cuda", metric="cosine",save=False,batch_size=2):
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
        cols = ["original_sentence", "perturb_n1", "perturb_n2", "perturb_n3"]
        for _, row in data[cols].iterrows():
            sentences.extend(row.values)
    elif args_task in ["paraphrase","Paraphrase","para"]:
        cols = ["original_sentence", "paraphrased_sentence"]
        for _, row in data[cols].iterrows():
            sentences.extend(row.values)
    
    # Batch process embeddings
    embeddings = model.encode_batch(sentences,batch_size=batch_size)
    if args_model != "chatgpt":
        embeddings = [emb.cpu().numpy() for emb in embeddings]
    embeddings = np.array(embeddings)
    
    # Process embeddings based on task
    if args_task == "anto":
        emb_org  = embeddings[0::3]  # start at 0, step by 3
        emb_para = embeddings[1::3]  # start at 1, step by 3
        emb_anto = embeddings[2::3]  # start at 2, step by 3
       
        mean_para,sim_para = utils.similarity_between_sent(emb_org, emb_para)
        mean_anto,sim_anto = utils.similarity_between_sent(emb_org, emb_anto)
        data["sim_org_para"] = sim_para
        data["sim_org_anto"] = sim_anto
        data["diff_org_para"] = np.array(sim_para) - np.array(sim_anto)
        
        print(f"""The summary for Antonym Criteria for {args_model} \n {data.describe()} """)
        

    elif args_task == "jumbling":
   
        emb_org  = embeddings[0::5]  # start at 0, step by 3
        emb_para = embeddings[1::5]  # start at 1, step by 3
        emb_n1 = embeddings[2::5]  # start at 2, step by 3
        emb_n2 = embeddings[3::5]
        emb_n3 = embeddings[4::5]
        
        # Compute metrics for each perturbation
        mean_para,sim_para = utils.similarity_between_sent(emb_org, emb_para)
        mean_n1,sim_n1 = utils.similarity_between_sent(emb_org, emb_n1)
        mean_n2,sim_n2 = utils.similarity_between_sent(emb_org, emb_n2)
        mean_n3,sim_n3 = utils.similarity_between_sent(emb_org, emb_n3)
        
        data["sim_org_para"] = sim_para
        data["sim_org_n1"] = sim_n1
        data["sim_org_n2"] = sim_n2
        data["sim_org_n3"] = sim_n3
        
        data["diff_org_para"] = sim_para - sim_para  # Zero as per original
        data["diff_org_n1"] = sim_para - sim_n1
        data["diff_org_n2"] = sim_para - sim_n2
        data["diff_org_n3"] = sim_para - sim_n3
        
        print(f"""The summary for Jumbling Criteria for {args_model} \n {data.describe()} """)
    
    
    elif args_task == "syn":
       
        emb_org  = embeddings[0::4]  # start at 0, step by 3
        emb_s1 = embeddings[1::4]  # start at 1, step by 3
        emb_s2 = embeddings[2::4]  # start at 2, step by 3
        emb_s3 = embeddings[3::4]
        
        _,sim_s1 = utils.similarity_between_sent(emb_org, emb_s1)
        _,sim_s2 = utils.similarity_between_sent(emb_org, emb_s2)
        _,sim_s3 = utils.similarity_between_sent(emb_org, emb_s3)
        
        data["sim_org_s1"] = sim_s1
        data["sim_org_s2"] = sim_s2
        data["sim_org_s3"] = sim_s3
        
        print(f"""The summary for Synonym Criteria for {args_model} \n {data.describe()} """)
    
    elif args_task == "paraphrase":
        emb_s1  = embeddings[0::2]  # start at 0, step by 3
        emb_s2 = embeddings[1::2] 
        data["sim"] = utils.similarity_between_sent(emb_s1, emb_s2)
        
        print(f"""The summary for Paraphrase Criteria for {args_model} \n {data.describe()} """)
    
    if save:
        path = f"./Results/{target_lang}/{args_task}/{dataset_name}_{args_model}_{args_task}_metric.csv"
        data.to_csv(path)
        print("Data saved at path : {path} ")
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
            "target_lang": parser.target_lang,
            "metric":parser.metric,
            "batch_size":2
        }   
    else:
        config = {
            "args_model": "llama3",
            "dataset_name": "mrpc",
            "args_task": "syn",
            "default_gpu": "cuda:2",
            "save": False,
            "target_lang": "en"
            
        }
    run(**config)
    
    
    # file_path = "/home/yash/ALIGN-SIM/data/perturbed_dataset/en/anto/mrpc_anto_perturbed_en.csv"
    # run("llama3","mrpc_anto_perturbed_en", "anto", "cuda:2", False)