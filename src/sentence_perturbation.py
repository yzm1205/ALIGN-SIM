from absl import logging

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
from utils import mkdir_p, full_path, read_data
from word_replacer import WordReplacer, WordSwapping
import random



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
    
    # TODO check if dataset_name exist or not
    output_csv = full_path(os.path.join(output_dir, target_lang, task, f"{dataset_name}_{task}_perturbed_{target_lang}.csv"))
    if os.path.exists(output_csv):
        print(f"File already exists at: {output_csv}")
        return 
    
    
    print("Loading dataset...")
    data = read_data(dataset_name) 
    print(f"Loaded {dataset_name} dataset")
    
    print("--------------------------------------")

    
    # Initialize WordReplacer
    replacer = WordReplacer()
    # set seed
    random.seed(42)
    
    # Create a new dataframe to store perturbed sentences
    # Sample sentences
    perturbed_data = pd.DataFrame(columns=["Sentence"])
    perturbed_data["Sentence"] = data.sentence1.sample(sample_size)

    if task in ["Syn","syn","Synonym"]:
        print("Creating Synonym perturbed data...")
        perturbed_data["Sentence1"] = perturbed_data["Sentence"].apply(lambda x: replacer.sentence_replacement(x, 1, "synonyms"))
        perturbed_data["Sentence2"] = perturbed_data["Sentence"].apply(lambda x: replacer.sentence_replacement(x, 2, "synonyms"))
        perturbed_data["Sentence3"] = perturbed_data["Sentence"].apply(lambda x: replacer.sentence_replacement(x, 3, "synonyms"))
        
        assert perturbed_data.shape[1] == 4, "Perturbed data size mismatch"
        

    if task in ["Anto","anto","Antonym"]:
        print("Creating Antonym perturbed data...")
        # Apply antonym replacement
        perturbed_data["Sentence1"] = perturbed_data["Sentence"].apply(lambda x: replacer.sentence_replacement(x, 1, "antonyms"))
        assert perturbed_data.shape[1] == 2, "Perturbed data size mismatch"
        
        

    # Apply jumbling
    if task in ["jumbling", "Jumbling","jumb"]:
        print("Creating Jumbling perturbed data...")
        perturbed_data["Sentence1"]= perturbed_data["Sentence"].apply(lambda x: WordSwapping.random_swap(x,1))
        perturbed_data["Sentence2"]= perturbed_data["Sentence"].apply(lambda x: WordSwapping.random_swap(x,2))
        perturbed_data["Sentence3"]= perturbed_data["Sentence"].apply(lambda x: WordSwapping.random_swap(x,3))
        
        assert perturbed_data.shape[1] == 4, "Perturbed data size mismatch"
    

    # Save to CSV
    if save:
        perturbed_data.to_csv(mkdir_p(output_csv), index=False)
        print("--------------------------------------")
        print(f"Saved at: {output_csv}")
        print("--------------------------------------")


if __name__ == "__main__":
    config = {
        "dataset_name": "qqp",
        "task": "syn",
        "target_lang": "en",
        "output_dir": "./data/perturbed_dataset/",
        "save": True,
        "sample_size": 3500
    }
    perturb_sentences(**config)