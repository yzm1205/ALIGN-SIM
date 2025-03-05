import abc
import warnings
from pathlib import Path
from typing import List, Union

import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


class SentenceTransformerModels():
 
    def __init__(self, model_id, device: bool = False):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_id).eval()

    def encode(self, sentences: List[str], batch_size: int = 32) -> NDArray:
        with torch.no_grad():
            embeddings = self.model.encode(
                sentences, batch_size=batch_size, device=self.device
            )
        if isinstance(embeddings, torch.Tensor):
            return embeddings.cpu().numpy()
        return embeddings