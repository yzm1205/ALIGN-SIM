from absl import logging

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys
sys.path.insert(0, "/home/yash/ALIGN-SIM/src")
from utils import mkdir_p, full_path, read_data
from SentencePerturbation.word_replacer import WordReplacer, WordSwapping
import random
from perturbation_args import get_args



def perturb_sentences(dataset_name: str, task: str, target_lang:str ="en", output_dir: str = "./data/perturbed_dataset/", sample_size: int = 3500, save :str = False) -> None:
    """
    perturb_sentences _summary_

    Args:
        dataset_name (str): ["MRPC","PAWS","QQP"]
        task (str): ["Synonym","Antonym","Jumbling"]
        target_lang (str, optional): _description_. Defaults to "en".
        output_dir (str, optional): _description_. Defaults to "./data/perturbed_dataset/".
        sample_size (int, optional): _description_. Defaults to 3500.
        save (str, optional): _description_. Defaults to False.
    """
    
    print("--------------------------------------")
    
    output_csv = full_path(os.path.join(output_dir, target_lang, task, f"{dataset_name}_{task}_perturbed_{target_lang}.csv"))
    if os.path.exists(output_csv):
        print(f"File already exists at: {output_csv}")
        return 
    
    # TODO: make it compatible with other language datasets
    print("Loading dataset...")
    data = read_data(dataset_name) 
    if "Unnamed: 0" in data.columns:
        data.drop("Unnamed: 0", axis=1, inplace=True)
    
    if "idx" in data.columns:
        data.drop("idx", axis=1, inplace=True)
        
    print(f"Loaded {dataset_name} dataset")
    
    print("--------------------------------------")

    
    # Initialize WordReplacer
    replacer = WordReplacer()
    # set seed
    random.seed(42)
    
    # Create a new dataframe to store perturbed sentences
    # Sample sentences
    perturbed_data = pd.DataFrame(columns=["original_sentence"])
    # sample_data , pos_pairs, balance_dataset  = sampling(data, sample_size)
    
    
    if task in ["Syn","syn","Synonym"]:
        print("Creating Synonym perturbed data...")
        sample_data = sampling(data, task, sample_size)
        perturbed_data["original_sentence"] = sample_data.sentence1
        perturbed_data["perturb_n1"] = perturbed_data["original_sentence"].apply(lambda x: replacer.sentence_replacement(x, 1, "synonyms"))
        perturbed_data["perturb_n2"] = perturbed_data["original_sentence"].apply(lambda x: replacer.sentence_replacement(x, 2, "synonyms"))
        perturbed_data["perturb_n3"] = perturbed_data["original_sentence"].apply(lambda x: replacer.sentence_replacement(x, 3, "synonyms"))
        
        assert perturbed_data.shape[1] == 4, "Perturbed data size mismatch"
    
    if task in ["paraphrase","Paraphrase","para"]:
        print("Creating Paraphrase perturbed data...")
        # shuffling the negative samples
        # we also want equal number of positive and negative samples
        perturbed_data = sampling(data, task, sample_size) # balance data
        perturbed_data["original_sentence"] = perturbed_data.sentence1
        perturbed_data["paraphrased_sentence"] = perturbed_data.sentence2
        assert perturbed_data.shape[1] == 3, "Perturbed data size mismatch" # original_sentence, paraphrased, label
        
    if task in ["Anto","anto","Antonym"]:
        print("Creating Antonym perturbed data...")
        pos_pairs = sampling(data, task, sample_size)
        # Apply antonym replacement
        perturbed_data["original_sentence"] = pos_pairs.sentence1
        perturbed_data["paraphrased_sentence"] = pos_pairs.sentence2
        perturbed_data["perturb_n1"] = perturbed_data["original_sentence"].apply(lambda x: replacer.sentence_replacement(x, 1, "antonyms"))
        assert perturbed_data.shape[1] == 3, "Perturbed data size mismatch"
        
    # Apply jumbling
    if task in ["jumbling", "Jumbling","jumb"]:
        print("Creating Jumbling perturbed data...")
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
    # Split the data into positive and negative pairs
    positive_data = data[data["label"] == 1]
    negative_data = data[data["label"] == 0]
    
    if task in ["Anto","anto","Antonym","jumbling", "Jumbling","jumb"]:
        return positive_data
    
    # ----- Sampling positive pair, but also checking if we satisfy sample size -----
    if sample_size is None or sample_size > len(positive_data):
        # If no sample size is provided or it exceeds the available data,
        # return a copy of the entire dataset.
        sampled_data = positive_data.copy()
    else:
        # Otherwise, randomly sample the specified number of rows.
        sampled_data = positive_data.sample(n=sample_size, random_state=random_state)

        
    if task in ["Syn","syn","Synonym"]:
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
    
    if task in ["paraphrase","Paraphrase","para"]:
        return balanced_data
    # return sampled_data, positive_data, balanced_data



if __name__ == "__main__":

    # # For Testing
    if sys.gettrace() is not None:
        config = {
            "dataset_name": "mrpc",
            "task": "syn",
            "target_lang": "en",
            "output_dir": "./data/perturbed_dataset/",
            "save": True
        }
    else: 
        args = get_args()
        config = {
            "dataset_name": args.dataset_name,
            "task": args.task,
            "target_lang": args.target_lang,
            "output_dir": args.output_dir,
            "save": args.save,
            "sample_size": args.sample_size
        }
    perturb_sentences(**config)