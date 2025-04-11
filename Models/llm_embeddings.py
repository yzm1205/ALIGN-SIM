import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Union, List
from pathlib import Path
from typing import Union, List
import dotenv
import os
import sys
sys.path.insert(0,"./")
from src.utils import full_path
from tqdm import tqdm


dotenv.load_dotenv(os.getenv("./models/.env"))
hf = os.getenv("huggingface_token")

def check_model_in_cache(model_name: str):
    if model_name in ["LLaMA3","llama3"]:
        return str(full_path("/data/shared/llama3-8b/Meta-Llama-3-8B_shard_size_1GB"))
    
    if model_name in ["Mistral","mistral"]:
        return str(full_path("/data/shared/mistral-7b-v03/Mistral-7B-v0.3_shard_size_1GB"))
    
    if model_name in ["olmo","OLMo"]:
        return str(full_path("/data/shared/olmo/OLMo-7B_shard_size_2GB"))
    
    raise ValueError(f"Model '{model_name}' not found in local cache.")

def mean_pooling(model_output, attention_mask):
    """
    mean_pooling _summary_

    Args:
        model_output (_type_): _description_
        attention_mask (_type_): _description_

    Returns:
        _type_: _description_
    """
    token_embeddings = model_output #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class LLMEmbeddings:
    def __init__(self, model_name: str, device: torch.device = None):
        """
        Initializes any Hugging Face LLM.
        
        Args:
            model_dir (str): Path or Hugging Face repo ID for the model.
            device (torch.device): Device to load the model on (CPU/GPU).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model from cache
        try:
            model_dir = check_model_in_cache(model_name)
        except:
            model_dir = model_name 

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        # Load model configuration to determine model type
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.model_type = config.architectures[0] if config.architectures else ""

        # Automatically choose between AutoModelForCausalLM and AutoModel
        if "CausalLM" in self.model_type:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, trust_remote_code=True, torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(
                model_dir, trust_remote_code=True, torch_dtype=torch.float16
            ).to(self.device)

        # Ensure padding token is set (fixes issues in tokenization)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()

    def encode(self, text: Union[str, List[str]]):
        """Encodes input sentences into embeddings."""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=1024, return_token_type_ids=False
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
       
        embeddings = mean_pooling(outputs.hidden_states[-1], inputs["attention_mask"]).squeeze()
        return embeddings
    
    def encode_batch(self, text: Union[str, List[str]], batch_size: int = 32):
        """Encodes input sentences into embeddings using batching."""
        # If a single string is provided, wrap it in a list.
        if isinstance(text, str):
            text = [text]

        embeddings_list = []
        # Process the text in batches
        for i in tqdm(range(0, len(text), batch_size), desc="Processing Batches"):
            batch_text = text[i:i+batch_size]
            inputs = self.tokenizer(
                batch_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
                return_token_type_ids=False
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)

            batch_embeddings = mean_pooling(outputs.hidden_states[-1], inputs["attention_mask"]).squeeze()
            embeddings_list.append(batch_embeddings)

        # Concatenate embeddings from all batches along the batch dimension.
        embeddings = torch.cat(embeddings_list, dim=0)
        return embeddings

    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load any Hugging Face LLM (e.g., LLaMA, Mistral, Falcon, GPT)
    
    llm = LLMEmbeddings(model_name="llama3", device=device)

    # Encode text into embeddings
    embedding = llm.encode("Hugging Face models are powerful!")
    print(embedding.shape)
    print("Done!!")
