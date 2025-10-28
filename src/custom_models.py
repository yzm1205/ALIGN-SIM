from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from Models.llm_embeddings import BaseEmbedder, TextInput
# from openai import OpenAI


# # Example 1

# class GPTEmbedder:
#     def __init__(self, model_name="text-embedding-3-small", api_key=None):
#         self.model_name = model_name
#         self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
#     def encode(self, texts):
#         if isinstance(texts, str):
#             texts = [texts]

#         response = self.client.embeddings.create(
#             model=self.model_name,
#             input=texts
#         )

#         embeddings = [data.embedding for data in response.data]
#         return np.array(embeddings, dtype=np.float32)
    
# Example 2
class CustomModel(BaseEmbedder):
    """Sentence-BERT powered custom embedder.

    This implementation wraps ``sentence_transformers.SentenceTransformer`` so
    that users can plug it into ALIGN-SIM via ``--custom-class-path`` without
    modifying the core codebase.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: Optional[str] = None,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
        batch_size: Optional[int] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        **model_kwargs: Any,
        ) -> None:
        """Initialise the Sentence-BERT model.

        Args:
            model_name: Name or path of the sentence-transformers checkpoint.
            device: Torch device string to run inference on. Defaults to
                sentence-transformers' auto-detection.
            normalize_embeddings: Whether to L2-normalise vectors by default.
            show_progress_bar: Whether to render a tqdm bar during encoding.
            encode_kwargs: Extra keyword arguments forwarded to ``model.encode``
                each call (can still be overridden per-call).
            **model_kwargs: Additional kwargs forwarded to
                ``SentenceTransformer`` upon initialisation.
        """

        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        self._default_encode_kwargs = dict(encode_kwargs or {})

        if batch_size is None:
            batch_hint = self._default_encode_kwargs.get("batch_size")
            self._default_batch_size = int(batch_hint) if batch_hint else 32
        else:
            self._default_batch_size = int(batch_size)
            self._default_encode_kwargs.setdefault("batch_size", self._default_batch_size)

        if "batch_size" in model_kwargs:
            hinted_batch = model_kwargs.pop("batch_size")
            self._default_batch_size = int(hinted_batch)
            self._default_encode_kwargs.setdefault("batch_size", self._default_batch_size)

        self.model = SentenceTransformer(
            model_name,
            device=device,
            **model_kwargs,
        )

    def encode(self, texts: TextInput, **kwargs: Any) -> np.ndarray:
        """Encode one or more texts into a 2D numpy array."""

        single_input = isinstance(texts, str)
        batched_texts: Sequence[str]
        if single_input:
            batched_texts = [texts]  # type: ignore[list-item]
        else:
            batched_texts = list(texts)
            if not batched_texts:
                raise ValueError("texts must contain at least one element")

        encode_kwargs = dict(self._default_encode_kwargs)
        encode_kwargs.update(kwargs)

        batch_size = encode_kwargs.pop("batch_size", self._default_batch_size)
        show_progress_bar = encode_kwargs.pop(
            "show_progress", encode_kwargs.pop("show_progress_bar", self.show_progress_bar)
        )
        normalize_embeddings = encode_kwargs.pop(
            "normalize", encode_kwargs.pop("normalize_embeddings", self.normalize_embeddings)
        )

        vectors = self.model.encode(
            batched_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            **encode_kwargs,
        )

        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            raise ValueError(
                "SentenceTransformer.encode returned an array with unexpected shape: "
                f"{array.shape}. Expected (N, D)."
            )

        if single_input and array.shape[0] != 1:
            # Defensive check – ensure single inputs still yield a single row.
            array = array[:1]

        return array

