import argparse
import numpy as np
import os 
import pandas as pd
from tqdm import tqdm
import torch
import utils
from utils import mkdir_p
from metrics import *
import sys
sys.path.insert(0,"./")
from Models.SentenceTransformersModel import SentenceTransformerModels
from Models.llm_embeddings import LLMEmbeddings
from main_args import get_args
from metrics import CosineMetric
from src.SentencePerturbation.sentence_perturbation import perturb_sentences, ALL_TASKS, TASK_ALIASES


def read_pertubed_data(filename, task, dataset_name, lang="en"):
    # path = f"./data/perturbed_dataset/{lang}/{task}/{filename}.csv"
    if not os.path.exists(filename):
        print(f"Creating {task.upper()} Perturbation dataset for {dataset_name} dataset")
        perturb_sentences(dataset_name, task, save=True)
    print("Loading the Perturbation dataset")
    return pd.read_csv(filename)


def process_task(args_model, dataset_name, target_lang, task, model, default_gpu="cuda", metric="cosine", save=False, batch_size=2,alpha=1.0):
    """
    Process embeddings and compute metrics for a specific task
    
    Args:
        args_model: Model name
        dataset_name: Dataset name
        target_lang: Target language
        task: Task name (standardized)
        model: Embedding model instance
        default_gpu: GPU device
        metric: Metric to use
        save: Whether to save results
        batch_size: Batch size for encoding
        alpha: Adjustment factor
        
    Returns:
        Dataframe with metrics computed
    """
    print(f"\n*** Processing {task} task with {args_model} on {dataset_name} dataset ***\n")
    
    # Get standardized task name
    std_task = task
    
    # Read data for this task
    pertubed_data_path = f"./data/perturbed_dataset/{target_lang}/{std_task}/{dataset_name}_{std_task}_perturbed_{target_lang}.csv"
    data = read_pertubed_data(pertubed_data_path, std_task, dataset_name, target_lang)
    
    # Collect all sentences based on task
    sentences = []
    if std_task == "anto":
        cols = ["original_sentence", "paraphrased_sentence", "perturb_n1"]
        for _, row in data[cols].iterrows():
            sentences.extend(row.values)
    elif std_task == "jumbling":
        cols = ["original_sentence", "paraphrased_sentence", "perturb_n1", "perturb_n2", "perturb_n3"]
        for _, row in data[cols].iterrows():
            sentences.extend(row.values)
    elif std_task == "syn":
        cols = ["original_sentence", "perturb_n1", "perturb_n2", "perturb_n3"]
        for _, row in data[cols].iterrows():
            sentences.extend(row.values)
    elif std_task == "paraphrase":
        # Split data into positive pairs (label=1) and random/negative pairs (label=0)
        pos = data[data["label"] == 1]
        rand = data[data["label"] == 0]
        
        # Collect sentences from positive pairs
        pos_sentences = []
        for _, row in pos[["original_sentence", "paraphrased_sentence"]].iterrows():
            pos_sentences.extend(row.values)
        
        # Collect sentences from random/negative pairs
        rand_sentences = []
        for _, row in rand[["original_sentence", "paraphrased_sentence"]].iterrows():
            rand_sentences.extend(row.values)
        
        # Combine all sentences while keeping track of indices
        sentences = pos_sentences + rand_sentences
    
    
    # Batch process embeddings
    embeddings = model.encode_batch(sentences, batch_size=batch_size)
    # Ensure embeddings are on CPU and in numpy format
    if args_model == "chatgpt":
        # For chatgpt, embeddings is likely a list of torch tensors
        embeddings = [emb.cpu().numpy() if isinstance(emb, torch.Tensor) else emb for emb in embeddings]
        embeddings = np.array(embeddings)
    else:
        # For other models, assume a single torch tensor
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
    
    # Process embeddings based on task
    if std_task == "anto":
        emb_org  = embeddings[0::3]  # start at 0, step by 3
        emb_para = embeddings[1::3]  # start at 1, step by 3
        emb_anto = embeddings[2::3]  # start at 2, step by 3
       
        mean_para, sim_para = utils.similarity_between_sent(emb_org, emb_para)
        mean_anto, sim_anto = utils.similarity_between_sent(emb_org, emb_anto)
        data["sim_org_para"] = sim_para
        data["sim_org_anto"] = sim_anto
        data["diff_org_para"] = np.array(sim_para) - np.array(sim_anto)
        
        print(f"""The summary for Antonym Criteria for {args_model} \n {data.describe()} """)
        
    elif std_task == "jumbling":
        emb_org  = embeddings[0::5]  # start at 0, step by 5
        emb_para = embeddings[1::5]  # start at 1, step by 5
        emb_n1 = embeddings[2::5]    # start at 2, step by 5
        emb_n2 = embeddings[3::5]    # start at 3, step by 5
        emb_n3 = embeddings[4::5]    # start at 4, step by 5
        
        # Compute metrics for each perturbation
        mean_para, sim_para = utils.similarity_between_sent(emb_org, emb_para)
        mean_n1, sim_n1 = utils.similarity_between_sent(emb_org, emb_n1)
        mean_n2, sim_n2 = utils.similarity_between_sent(emb_org, emb_n2)
        mean_n3, sim_n3 = utils.similarity_between_sent(emb_org, emb_n3)
        
        data["sim_org_para"] = sim_para
        data["sim_org_n1"] = sim_n1
        data["sim_org_n2"] = sim_n2
        data["sim_org_n3"] = sim_n3
        
        data["diff_org_para"] = sim_para - sim_para  # Zero as per original
        data["diff_org_n1"] = sim_para - sim_n1
        data["diff_org_n2"] = sim_para - sim_n2
        data["diff_org_n3"] = sim_para - sim_n3
        
        print(f"""The summary for Jumbling Criteria for {args_model} \n {data.describe()} """)
    
    elif std_task == "syn":
        emb_org = embeddings[0::4]  # start at 0, step by 4
        emb_s1 = embeddings[1::4]   # start at 1, step by 4
        emb_s2 = embeddings[2::4]   # start at 2, step by 4
        emb_s3 = embeddings[3::4]   # start at 3, step by 4
        
        _, sim_s1 = utils.similarity_between_sent(emb_org, emb_s1)
        _, sim_s2 = utils.similarity_between_sent(emb_org, emb_s2)
        _, sim_s3 = utils.similarity_between_sent(emb_org, emb_s3)
        
        data["sim_org_s1"] = sim_s1 * alpha
        data["sim_org_s2"] = sim_s2 * alpha
        data["sim_org_s3"] = sim_s3 * alpha
        
        print(f"""The summary for Synonym Criteria for {args_model} \n {data.describe()} """)
    
    elif std_task == "paraphrase":
        # Split embeddings for positive and random pairs
        pos_count = len(pos_sentences)
        pos_embeddings = embeddings[:pos_count]
        rand_embeddings = embeddings[pos_count:]
        
        # Calculate similarities for positive pairs
        pos_emb_s1 = pos_embeddings[0::2]  # start at 0, step by 2
        pos_emb_s2 = pos_embeddings[1::2]  # start at 1, step by 2
        pos_mean, pos_sim = utils.similarity_between_sent(pos_emb_s1, pos_emb_s2)
        
        # Calculate similarities for random pairs
        rand_emb_s1 = rand_embeddings[0::2]  # start at 0, step by 2
        rand_emb_s2 = rand_embeddings[1::2]  # start at 1, step by 2
        rand_mean, rand_sim = utils.similarity_between_sent(rand_emb_s1, rand_emb_s2)
        
        # Add the similarities to the respective dataframes
        pos["sim"] = pos_sim if len(pos_sim) > 0 else []
        rand["sim"] = rand_sim if len(rand_sim) > 0 else []
        
        # Combine the results
        data = pd.concat([pos, rand]).sort_index()
        
        # Print summaries for positive and random pairs
        print(f"""Summary for Positive Pairs (label=1) for {args_model}:
{pos.describe() if len(pos) > 0 else "No positive pairs found"}""")
        
        print(f"""Summary for Random Pairs (label=0) for {args_model}:
{rand.describe() if len(rand) > 0 else "No random pairs found"}""")
        
        print(f"""Overall Summary for Paraphrase Criteria for {args_model}:
{data.describe()}""")
    
    if save:
        path = f"./Results/{target_lang}/{std_task}/{dataset_name}_{args_model}_{std_task}_metric.csv"
        mkdir_p(os.path.dirname(path))
        data.to_csv(path)
        print(f"Data saved at path: {path}")
    
    return data


def run(args_model, dataset_name, target_lang, args_task, default_gpu="cuda", metric="cosine", save=False, batch_size=2):
    """
    Run evaluation on specified tasks
    
    Args:
        args_model: Model name
        dataset_name: Dataset name
        target_lang: Target language
        args_task: Task(s) to run (can be a string or list)
        default_gpu: GPU device
        metric: Metric to use
        save: Whether to save results
        batch_size: Batch size for encoding
        
    Returns:
        If a single task is specified, returns the dataframe for that task
        If multiple tasks are specified, returns a dictionary mapping tasks to their result dataframes
    """
    print(f"\n*** Starting evaluation with model {args_model} on {dataset_name} dataset ***\n")
    
    # Initialize model
    model = LLMEmbeddings(args_model, device=default_gpu)

    
    # Handle task(s) specification
    # By default, we are adding paraphrase task because we want to calculate the alpha factor to adjust the randomness of the model. 
    
    tasks_to_run = []
    
    # Convert string to list if necessary
    if isinstance(args_task, str):
        args_task = [args_task]
    
    # Check if "all" is specified to run all tasks
    if any(task.lower() == "all" for task in args_task):
        tasks_to_run = ALL_TASKS
    else:
        # Standardize task names and filter out invalid ones
        for task in args_task:
            std_task = TASK_ALIASES.get(task, None)
            if std_task and std_task != "all":  # Skip "all" as it's a meta-task
                tasks_to_run.append(std_task)
            else:
                print(f"Warning: Unknown task '{task}'. Skipping.")
    
    # Remove duplicates
    tasks_to_run = list(set(tasks_to_run))
    if "syn" in tasks_to_run or "negation" in tasks_to_run:
        if "paraphrase" not in tasks_to_run:
            tasks_to_run.insert(0, "paraphrase")  # Insert at the 0th index
    
    if not tasks_to_run:
        print(f"No valid tasks specified. Available tasks are: {', '.join(ALL_TASKS)}")
        return None
    
    print(f"Running the following tasks: {tasks_to_run}")
    
    # Process each task
    results = {}
    successful_tasks = []
    failed_tasks = []
    adjustment_factor = 1.0
    for task in tasks_to_run:
        try:
            # print(f"\n=== Starting task: {task} ===\n")
            result_df = process_task(
                args_model=args_model,
                dataset_name=dataset_name,
                target_lang=target_lang,
                task=task,
                model=model,
                default_gpu=default_gpu,
                metric=metric,
                save=save,
                batch_size=batch_size,
                aplha=adjustment_factor
            )
            results[task] = result_df
            if task == "paraphrase":
                rand = result_df[result_df["label"==0]]
                adjustment_factor = 1 - rand["sim"].mean()
                
            successful_tasks.append(task)
        except Exception as e:
            print(f"Error processing task '{task}': {str(e)}")
            failed_tasks.append(task)
    
    # Print summary
    print("\n=== Task Summary ===")
    print(f"Successfully completed tasks: {successful_tasks}")
    if failed_tasks:
        print(f"Failed tasks: {failed_tasks}")
    print("=====================")
    
    # For backward compatibility, return a single DataFrame if only one task was run
    if len(results) == 1:
        return next(iter(results.values()))
    return results

if __name__ == "__main__":
    if sys.gettrace() is None: 
        parser = get_args()
        config = {
            "args_model": parser.model_name,
            "dataset_name": parser.perturbed_dataset,
            "args_task": parser.task,  # This is now handled as a list in the run function
            "default_gpu": parser.gpu,
            "save": parser.save,
            "target_lang": parser.target_lang,
            "metric": parser.metric,
            "batch_size": 2
        }   
    else:
        # For debugging/testing - try multiple tasks
        
        config = {
            "args_model": "llama3",
            "dataset_name": "mrpc",
            "args_task": ["paraphrase","syn"],  # Multiple tasks for testing
            "default_gpu": "cuda:2",
            "save": False,
            "target_lang": "en",
            "metric": "cosine",
            "batch_size": 2
        }
    run(**config)
    
