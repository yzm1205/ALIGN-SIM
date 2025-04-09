import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
import os

def delete_file(file_pt: Path) -> None:
    try:
        file_pt.unlink()
    except FileNotFoundError:
        pass


def full_path(inp_dir_or_path: str) -> Path:
    """Returns full path"""
    return Path(inp_dir_or_path).expanduser().resolve()


def mkdir_p(inp_dir_or_path: Union[str, Path]) -> Path:
    """Give a file/dir path, makes sure that all the directories exists"""
    inp_dir_or_path = full_path(inp_dir_or_path)
    if inp_dir_or_path.suffix:  # file
        inp_dir_or_path.parent.mkdir(parents=True, exist_ok=True)
    else:  # dir
        inp_dir_or_path.mkdir(parents=True, exist_ok=True)
    return inp_dir_or_path

def similarity_between_sent(sent1_encoded, sent2_encoded):
    """report the avg. cosine similarity score b.w two pairs of sentences"""    
    similarity_scores = []
    for i in range(len(sent1_encoded)):
        similarity_scores.append(cosine_similarity(
            sent1_encoded[i], sent2_encoded[i]))

    return np.mean(similarity_scores),similarity_scores


def cosine_similarity(a, b):
    """
Takes 2 vectors a, b and returns the cosine similarity according 
    to the definition of the dot product
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def load_data(path):
    if path.endswith(".csv"):
        data=pd.read_csv(path)
    else:
        data=pd.read_csv(path,sep="\t")
    
    if not isinstance(data,pd.DataFrame):
        raise ValueError("Data should be in pandas DataFrame format")   
    return data

def read_data(dataset):
    if dataset == "mrpc":
        data = load_data("/home/yash/EMNLP-2024/data/mrpc.csv")
        data = data.copy()
        
    elif dataset == "qqp":    
        data = load_data("/home/yash/EMNLP-2024/data/qoura.csv")
        data = data.copy().dropna()
        # handling irregularities in columns names
        data.columns = data.columns.str.strip()
        data = data.rename(columns={"is_duplicate":"label",'question1':"sentence1","question2":"sentence2"})
    
    elif dataset in ["paws","paw","wiki"]:
        path = "/home/yash/EMNLP-2024/data/paw_wiki.tsv"
        data = load_data(path)
        data = data.copy()
    
    else:
        ValueError("No dataset found.")
                
    return data