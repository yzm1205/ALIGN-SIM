from absl import logging

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys
sys.path.insert(0, "./src/")
# from utils import read_data
from SentencePerturbation.word_replacer import WordReplacer, WordSwapping
import random

# Try relative import first (for module execution)
try:
    from .perturbation_args import get_args
except ImportError:
    # Fall back to direct import (for direct script execution)
    from perturbation_args import get_args

# Define all available tasks
ALL_TASKS = ["paraphrase", "syn", "anto", "jumb", "negation"]
# Define task aliases for standardization
TASK_ALIASES = {
    "syn": "syn", "Syn": "syn", "Synonym": "syn", "synonym": "syn",
    "anto": "anto", "Anto": "anto", "Antonym": "anto", "antonym": "anto",
    "jumb": "jumbling", "jumbling": "jumbling", "Jumbling": "jumbling", "Jumb": "jumbling",
    "paraphrase": "paraphrase", "Paraphrase": "paraphrase", "para": "paraphrase", "Para": "paraphrase",
    "neg":"negation","afin":"negation","Negation":"negation","negation":"negation",
    "all": "all", "All": "all", "ALL": "all"
}

def perturb_sentences(dataset_name: str, task: str, target_lang:str ="en", output_dir: str = "./data/perturbed_dataset/", sample_size: int = 3500, save :str = False) -> None:
    """
    perturb_sentences _summary_

    Args:
        dataset_name (str): ["MRPC","PAWS","QQP"]
        task (str): ["syn","anto","jumb","paraphrase","negation"]
        target_lang (str, optional): _description_. Defaults to "en".
        output_dir (str, optional): _description_. Defaults to "./data/perturbed_dataset/".
        sample_size (int, optional): _description_. Defaults to 3500.
        save (str, optional): _description_. Defaults to False.
    """
    # Import utils functions inside the function to avoid circular imports
    from utils import mkdir_p, full_path, read_data
    
    # Standardize task name
    task = TASK_ALIASES.get(task, task)
    
    print("--------------------------------------")
    # print(f"Processing task: {task}")
    if task == "negation": 
        output_csv = full_path(os.path.join(output_dir, target_lang, task, f"{dataset_name}.csv"))
    else:
        output_csv = full_path(os.path.join(output_dir, target_lang, task, f"{dataset_name}_{task}_perturbed_{target_lang}.csv"))
    if os.path.exists(output_csv):
        print(f"File already exists at: {output_csv}")
        return 
    
    # TODO: make it compatible with other language datasets
    # print("Loading dataset...")
    
    # Initialize WordReplacer
    replacer = WordReplacer()
    # set seed
    random.seed(42)
    
    # Create a new dataframe to store perturbed sentences
    perturbed_data = pd.DataFrame(columns=["original_sentence"])
    
    if task == "negation":
        print("Processing negation data...")
        # Use the consolidated function to get AFIN data
        try:
            from utils import get_afin_data
            
            # If sample_size is specified, limit the data
            afin_data = get_afin_data(
                output_path=output_csv,
                sample_size=sample_size,
                save=save
            )
            
            perturbed_data = afin_data
            
        except Exception as e:
            print(f"Error processing AFIN dataset: {str(e)}")
            return
    
    elif task == "syn":
        print("Working...")
        data = read_data(dataset_name) 
        if "Unnamed: 0" in data.columns:
            data.drop("Unnamed: 0", axis=1, inplace=True)
        
        if "idx" in data.columns:
            data.drop("idx", axis=1, inplace=True)
            
        sample_data = sampling(data, task, sample_size)
        perturbed_data["original_sentence"] = sample_data.sentence1
        perturbed_data["perturb_n1"] = perturbed_data["original_sentence"].apply(lambda x: replacer.sentence_replacement(x, 1, "synonyms"))
        perturbed_data["perturb_n2"] = perturbed_data["original_sentence"].apply(lambda x: replacer.sentence_replacement(x, 2, "synonyms"))
        perturbed_data["perturb_n3"] = perturbed_data["original_sentence"].apply(lambda x: replacer.sentence_replacement(x, 3, "synonyms"))
        
        assert perturbed_data.shape[1] == 4, "Perturbed data size mismatch"
    
    elif task == "paraphrase":
        print("Working...")
        # shuffling the negative samples
        # we also want equal number of positive and negative samples
        data = read_data(dataset_name) 
        if "Unnamed: 0" in data.columns:
            data.drop("Unnamed: 0", axis=1, inplace=True)
        
        if "idx" in data.columns:
            data.drop("idx", axis=1, inplace=True)
            
        filtered_data = sampling(data, task, sample_size) # balance data
        perturbed_data["original_sentence"] = filtered_data.sentence1
        perturbed_data["paraphrased_sentence"] = filtered_data.sentence2
        perturbed_data["label"] = filtered_data.label
        assert perturbed_data.shape[1] == 3, "Perturbed data size mismatch" # original_sentence, paraphrased, label
        
    elif task == "anto":
        print("Working...")
        data = read_data(dataset_name) 
        if "Unnamed: 0" in data.columns:
            data.drop("Unnamed: 0", axis=1, inplace=True)
        
        if "idx" in data.columns:
            data.drop("idx", axis=1, inplace=True)
            
        pos_pairs = sampling(data, task, sample_size)
        # Apply antonym replacement
        perturbed_data["original_sentence"] = pos_pairs.sentence1
        perturbed_data["paraphrased_sentence"] = pos_pairs.sentence2
        perturbed_data["perturb_n1"] = perturbed_data["original_sentence"].apply(lambda x: replacer.sentence_replacement(x, 1, "antonyms"))
        assert perturbed_data.shape[1] == 3, "Perturbed data size mismatch"
        
    # Apply jumbling
    elif task == "jumbling":
        print("Working...")
        data = read_data(dataset_name) 
        if "Unnamed: 0" in data.columns:
            data.drop("Unnamed: 0", axis=1, inplace=True)
        
        if "idx" in data.columns:
            data.drop("idx", axis=1, inplace=True)
            
        pos_pairs = sampling(data, task, sample_size)
        perturbed_data["original_sentence"] = pos_pairs.sentence1
        perturbed_data["paraphrased_sentence"] = pos_pairs.sentence2
        perturbed_data["perturb_n1"]= perturbed_data["original_sentence"].apply(lambda x: WordSwapping.random_swap(x,1))
        perturbed_data["perturb_n2"]= perturbed_data["original_sentence"].apply(lambda x: WordSwapping.random_swap(x,2))
        perturbed_data["perturb_n3"]= perturbed_data["original_sentence"].apply(lambda x: WordSwapping.random_swap(x,3))
        
        assert perturbed_data.shape[1] == 5, "Perturbed data size mismatch"
    
    # Save to CSV
    if save:
        perturbed_data.to_csv(mkdir_p(output_csv), index=False)
        print("--------------------------------------")
        print(f"Saved at: {output_csv}")
        print("--------------------------------------")



def sampling(data: pd.DataFrame, task :str, sample_size: int, random_state: int = 42):
    """
    Combines two sampling strategies:
    
    1. sampled_data: Samples from the dataset by first taking all positive pairs and then,
       if needed, filling the remainder with negative pairs.
    2. balanced_data: Constructs a dataset with roughly equal positive and negative pairs,
       adjusting the numbers if one group is underrepresented.
    
    Returns:
        sampled_data (pd.DataFrame): Dataset sampled by filling negatives if positives are insufficient.
        positive_data (pd.DataFrame): All positive samples (label == 1).
        balanced_data (pd.DataFrame): Dataset balanced between positive and negative pairs.
    """
    # Standardize task name
    task = TASK_ALIASES.get(task, task)
    
    # Split the data into positive and negative pairs
    positive_data = data[data["label"] == 1]
    negative_data = data[data["label"] == 0]
    
    if task in ["anto", "jumb"]:
        return positive_data
    
    # ----- Sampling positive pair, but also checking if we satisfy sample size -----
    if sample_size is None or sample_size > len(positive_data):
        # If no sample size is provided or it exceeds the available data,
        # return a copy of the entire dataset.
        sampled_data = positive_data.copy()
    else:
        # Otherwise, randomly sample the specified number of rows.
        sampled_data = positive_data.sample(n=sample_size, random_state=random_state)

        
    if task == "syn":
        return sampled_data

    # ----- Sampling for Paraphrased Criterion -----
    # Shuffle negative pairs first
    negative_data = negative_data.reset_index(drop=True)
    shuffled_sentence2 = negative_data["sentence2"].sample(frac=1, random_state=random_state).reset_index(drop=True)
    negative_data["sentence2"] = shuffled_sentence2

    # Determine ideal sample size per group (half of total sample size)
    if sample_size is None:
        pos_sample_size = len(positive_data)
        neg_sample_size = len(negative_data)
    else:
        # Determine ideal sample size per group (half of total sample size)
        half_size = sample_size // 2
        pos_available = len(positive_data)
        neg_available = len(negative_data)
        pos_sample_size = min(half_size, pos_available)
        neg_sample_size = min(half_size, neg_available)

        # If there is a remainder, add extra samples from the group with more available data.
        total_sampled = pos_sample_size + neg_sample_size
        remainder = sample_size - total_sampled
        if remainder > 0:
            if (pos_available - pos_sample_size) >= (neg_available - neg_sample_size):
                pos_sample_size += remainder
            else:
                neg_sample_size += remainder

    # Sample from each group
    sampled_positive = positive_data.sample(n=pos_sample_size, random_state=random_state)
    sampled_negative = negative_data.sample(n=neg_sample_size, random_state=random_state)
    # Add a 'label' column
    sampled_positive["label"] = 1
    sampled_negative["label"] = 0
    # Combine and shuffle the resulting dataset
    balanced_data = pd.concat([sampled_positive, sampled_negative]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    if task == "paraphrase":
        return balanced_data
    # return sampled_data, positive_data, balanced_data


def run_all_tasks(dataset_name: str, tasks: list, target_lang:str ="en", output_dir: str = "./data/perturbed_dataset/", sample_size: int = 3500, save :str = False) -> None:
    """
    Run perturb_sentences for multiple tasks or all tasks
    
    Args:
        dataset_name (str): Dataset name
        tasks (list): List of tasks to run, or ["all"] to run all tasks
        target_lang (str, optional): Target language. Defaults to "en".
        output_dir (str, optional): Output directory. Defaults to "./data/perturbed_dataset/".
        sample_size (int, optional): Sample size. Defaults to 3500.
        save (bool, optional): Whether to save output. Defaults to False.
    """
    
    # If "all" is in tasks, run all available tasks
    if "all" in tasks:
        tasks_to_run = ALL_TASKS
    else:
        # Standardize task names
        tasks_to_run = []
        for task in tasks:
            standardized_task = TASK_ALIASES.get(task, None)
            if standardized_task is None:
                print(f"Warning: Unknown task '{task}'. Skipping.")
            else:
                tasks_to_run.append(standardized_task)
                
        # Remove duplicates
        tasks_to_run = list(set(tasks_to_run))
    
    if not tasks_to_run:
        print("No valid tasks specified. Available tasks are:", ", ".join(ALL_TASKS))
        return
        
    print(f"Running the following tasks: {tasks_to_run}")
    
    successful_tasks = []
    failed_tasks = []
    
    for task in tasks_to_run:
        try:
            perturb_sentences(
                dataset_name=dataset_name,
                task=task,
                target_lang=target_lang,
                output_dir=output_dir,
                sample_size=sample_size,
                save=save
            )
            successful_tasks.append(task)
        except Exception as e:
            print(f"Error processing task '{task}': {str(e)}")
            failed_tasks.append(task)
            
    if failed_tasks:
        print(f"Failed tasks: {failed_tasks}")
    print("=====================")


if __name__ == "__main__":

    # # For Testing
    if sys.gettrace() is not None:
        config = {
            "dataset_name": "mrpc",
            "tasks": ["jumbling"],
            "target_lang": "en",
            "output_dir": "./data/perturbed_dataset/",
            "save": True
        }
    else: 
        args = get_args()
        config = {
            "dataset_name": args.dataset_name,
            "tasks": args.task,  # This is now a list
            "target_lang": args.target_lang,
            "output_dir": args.output_dir,
            "save": args.save,
            "sample_size": args.sample_size
        }
    
    run_all_tasks(**config)