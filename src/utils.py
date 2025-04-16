import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
import requests
from urllib.parse import urlparse
import json
from src.SentencePerturbation.sentence_perturbation import perturb_sentences, ALL_TASKS, TASK_ALIASES


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
        data = load_data("./data/original_datasets/En/mrpc.csv")
        data = data.copy()
        
    elif dataset == "qqp":    
        data = load_data("./data/original_datasets/En/qoura.csv")
        data = data.copy().dropna()
        # handling irregularities in columns names
        data.columns = data.columns.str.strip()
        data = data.rename(columns={"is_duplicate":"label",'question1':"sentence1","question2":"sentence2"})
    
    elif dataset in ["paws","paw","wiki"]:
        path = "./data/original_datasets/En/paw_wiki.tsv"
        data = load_data(path)
        data = data.copy()
            
    else:
        ValueError("No dataset found.")
                
    return data

def read_pertubed_data(filename, task, dataset_name, sample_size, lang="en"):
    """
    Read perturbed data from a file, or create it if it doesn't exist.
    
    Args:
        filename (str): Path to the perturbed data file (CSV format)
        task (str): Perturbation task name (e.g., "negation", "syn", "jumbling", "anto", "paraphrase")
        dataset_name (str): Name of the dataset to perturb (e.g., "mrpc", "qqp", "paws")
        sample_size (int): Maximum number of samples to include in the dataset
        lang (str, optional): Language code for the perturbation. Defaults to "en".
        
    Returns:
        pandas.DataFrame: The perturbed dataset with appropriate columns depending on the task:
            - negation: "sentence1" and "sentence2" columns
            - syn: "original_sentence", "perturb_n1", "perturb_n2", "perturb_n3" columns
            - anto: "original_sentence", "paraphrased_sentence", "perturb_n1" columns
            - jumbling: "original_sentence", "paraphrased_sentence", "perturb_n1", "perturb_n2", "perturb_n3" columns
            - paraphrase: "original_sentence", "paraphrased_sentence", "label" columns
    """
    # For the negation task, we use the AFIN dataset
    if task == "negation":
        return get_afin_data(output_path=filename, sample_size=sample_size)

    # For other tasks, we check if the file exists, and if not, create the perturbed dataset
    if not os.path.exists(filename):
        perturb_sentences(dataset_name, task, target_lang=lang, save=True, sample_size=sample_size)
    
    # Load and return the dataset
    return pd.read_csv(filename)

def download_afin_dataset(filename=None, output_path="."):
    """
    Downloads the AFIN dataset from the GitHub repository.

    Args:
        filename (str, optional): The name to save the file as. Defaults to "afin.jsonl".
        output_path (str, optional): The directory to save the file in.
    """
    # The raw content URL for the AFIN dataset (use raw.githubusercontent.com for direct file access)
    url = "https://raw.githubusercontent.com/mosharafhossain/large-afin-and-nlu/main/affirmative-interpretation-generation/data/large-afin/large-afin.jsonl"
    
    try:
        # If no filename is provided, use a default
        if not filename:
            filename = "afin.jsonl"

        # Construct the full output path
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
            
        filepath = os.path.join(output_path, filename)
        
        # Check if the file already exists
        if os.path.exists(filepath):
            print(f"AFIN dataset already exists at {filepath}.")
            return filepath

        print(f"Downloading AFIN dataset from {url}...")
        
        # Download the file with a stream to handle large files
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Write the content to the file
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):  # 8KB chunks
                file.write(chunk)

        print(f"Downloaded AFIN dataset to {filepath}")
        return filepath

    except requests.exceptions.RequestException as e:
        print(f"Error downloading from {url}: {e}")
    except OSError as e:
        print(f"Error creating directory or saving file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return None

def get_afin_data(output_path="./data/perturbed_dataset/en/negation/afin.csv", jsonl_path="./data/original_datasets/En/afin.jsonl", sample_size=None, save=True):
    """
    Consolidated function that handles the entire AFIN data workflow:
    1. Checks if the processed CSV file exists
    2. If not, checks if the JSONL file exists
    3. If not, downloads the JSONL file
    4. Loads and processes the data
    5. Optionally saves to CSV and returns the DataFrame
    
    Args:
        output_path (str): Path where the CSV file should be saved
        jsonl_path (str): Path to the JSONL source file
        sample_size (int, optional): If specified, limit data to this many rows
        save (bool): Whether to save the processed data to CSV
        
    Returns:
        DataFrame: The processed AFIN data with sentence1 and sentence2 columns
    """
    
    
    # Step 1: Check if the processed CSV file already exists
    if os.path.exists(output_path):
        print(f"Found existing AFIN data at {output_path}")
        try:
            df = pd.read_csv(output_path)
            
            # If sample_size is specified and smaller than the existing data, sample it
            if sample_size is not None and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42)
                
            return df
        except Exception as e:
            print(f"Error reading existing AFIN data: {str(e)}. Will regenerate.")
            # Continue to regenerate if there was an error
    
    # Step 2: Check if the JSONL file exists, download if needed
    if not os.path.exists(jsonl_path):
        print(f"Afin dataset file not found at {jsonl_path}. Attempting to download...")
        jsonl_path = download_afin_dataset(
            filename=os.path.basename(jsonl_path), 
            output_path=os.path.dirname(jsonl_path)
        )
        if not jsonl_path or not os.path.exists(jsonl_path):
            print("Failed to download AFIN dataset.")
            return pd.DataFrame(columns=["sentence1", "sentence2"])
    
    # Step 3: Load and process the JSONL data
    try:
        print(f"Processing AFIN data from {jsonl_path}...")
        with open(jsonl_path, "r") as f:
            # Parse each line as JSON and extract the relevant fields
            data = [json.loads(line) for line in f]
            sentence1 = [item.get("sentence", "") for item in data]
            sentence2 = [item.get("affirmative_interpretation", "") for item in data]
        
        # Create DataFrame
        df = pd.DataFrame({"sentence1": sentence1, "sentence2": sentence2})
        
        # Apply sample_size if specified
        if sample_size is not None and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        # Step 4: Save to CSV if requested
        if save:
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            df.to_csv(output_path, index=False,escapechar="\\")
            print(f"Saved processed AFIN data to {output_path}")
        
        return df
        
    except Exception as e:
        print(f"Error processing AFIN data: {str(e)}")
        return pd.DataFrame(columns=["sentence1", "sentence2"])

