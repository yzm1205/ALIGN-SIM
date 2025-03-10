import pandas as pd
from googletrans import Translator  # Import the googletrans module
import os
from tqdm import tqdm
import sys
sys.path.insert(0, "/home/yash/EMNLP-2024/ALIGN-Multilingual/")
from Models.MultilingualTranslationModel import NLLBTranslator
from args_parser import get_args
from src.utils import read_data

# TODO: Perturbation does not support Multilingual at the moment

def translate_dataset(dataset_name, model_name, target_lang,batch_size=16, sample_size=1000,save=False):
    """
    Translates a dataset in batches using the NLLB model.

    Args:
        dataset_name (str): Name of the dataset.
        model_name (str): Model name used for translation.
        target_lang (str): Target language for translation.
        batch_size (int): Number of sentences to process in each batch.
        sample_size (int): Number of rows to process.
        save (bool): Whether to save the translated dataset to CSV.

    Returns:
        pd.DataFrame: Translated dataset.
    """
    
    
    # check if translated dataset already exists else create it
    translated_file_path = f"/home/yash/EMNLP-2024/ALIGN-Multilingual/data/{dataset_name}_{target_lang}.csv"
    
    #original dataset 
    # data = pd.read_csv("/home/yash/EMNLP-2024/data/paw_wiki.tsv", sep='\t')
    data = read_data(dataset_name)
    #size of dataset
    print(f"Size of dataset: {len(data)}")

    print("original dataset loaded successfully")
    
    model = NLLBTranslator(model_name=model_name)
    print("NLLB model loaded successfully")
    
    if os.path.exists(translated_file_path):
        translated_dataset = pd.read_csv(translated_file_path)
        print("Dataset exists and loaded successfully")
        return translated_dataset
    
    print("Creatign the dataset ....")
    translated_dataset = pd.DataFrame(columns=['sentence1', 'sentence2', 'label'])

    for i in tqdm(range(0, len(data), batch_size)):
        batch_sentences1 = data.loc[i:i+batch_size-1, 'sentence1'].tolist()
        batch_sentences2 = data.loc[i:i+batch_size-1, 'sentence2'].tolist()
        batch_labels = data.loc[i:i+batch_size-1, 'label'].tolist()
        
        translated_batch1 = model.translate(batch_sentences1, source_lang="en", target_lang=target_lang)
        translated_batch2 = model.translate(batch_sentences2, source_lang="en", target_lang=target_lang)
        
        # Append translated sentences and labels to DataFrame
        batch_df = pd.DataFrame({
            'sentence1': translated_batch1,
            'sentence2': translated_batch2,
            'label': batch_labels
        })

        translated_dataset = pd.concat([translated_dataset, batch_df], ignore_index=True)
        
    if save:
        translated_dataset.to_csv(translated_file_path, index=False)
        print(f"Translated dataset saved to {translated_file_path}")
    return translated_dataset



if __name__ == "__main__":
    languages=['fr','es',"de","zh-CN","ja","ko"]
    
    # Parse command-line arguments
    args = get_args()
    
    for language in languages:
        print(f"Translating to {language} ....")
        config= {
            "dataset_name": args.dataset_name,
            "model_name": args.model_name,
            "target_lang": language,
            "batch_size": args.batch_size,
            "save": args.save
        }
        translated_dataset_lang = translate_dataset(**config)
    
    # For Testing
    # for language in languages:
    #     print(f"Translating to {language} ....")
    #     config= {
    #         "dataset_name": "qqp",
    #         "model_name": "nllb",
    #         "target_lang": language,
    #         "batch_size": 3,
    #         "save": True
    #     }
    #     translated_dataset_lang = translate_dataset(**config)
    print("Done")
