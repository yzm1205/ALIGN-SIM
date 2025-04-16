import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
import requests
from urllib.parse import urlparse
import json

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
        
    # elif dataset in ["afin","negation"]:
    #     path = "./data/original_datasets/En/afin.jsonl"
        
    
    else:
        ValueError("No dataset found.")
                
    return data


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

if __name__ == "__main__":
    # Test the AFIN dataset download and processing
    print("Testing AFIN dataset handling with consolidated function...")
    
    # Use the consolidated function that handles the entire workflow
    afin_data = get_afin_data(
        output_path="./data/perturbed_dataset/en/negation/afin.csv", 
        jsonl_path="./data/original_datasets/En/afin.jsonl",
        sample_size=100,  # Use a small sample for testing
        save=True
    )
    
    # Display some statistics and samples
    if not afin_data.empty:
        print(f"Successfully loaded AFIN dataset with {len(afin_data)} rows")
        print("\nSample data:")
        print(afin_data.head(3))
    else:
        print("Failed to load AFIN dataset or dataset is empty")