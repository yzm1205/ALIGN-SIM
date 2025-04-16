# ALIGN-SIM: A Task-Free Test Bed for Evaluating and Interpreting Sentence Embeddings

ALIGN-SIM is a novel, task-free test bed for evaluating and interpreting sentence embeddings based on five intuitive semantic alignment criteria. It provides an alternative evaluation paradigm to popular task-specific benchmarks, offering deeper insights into whether sentence embeddings truly capture human-like semantic similarity.

## Overview

Sentence embeddings are central to many NLP applications such as translation, question answering, and text classification. However, evaluating these dense vector representations in a way that reflects human semantic understanding remains challenging. ALIGN-SIM addresses this challenge by introducing a framework based on five semantic alignment criteria:

- **Semantic Distinction:** Measures the ability of an encoder to differentiate between semantically similar sentence pairs and unrelated (random) sentence pairs.
- **Synonym Replacement:** Tests if minor lexical changes (using synonyms) preserve the semantic similarity of the original sentence.
- **Antonym Replacement (Paraphrase vs. Antonym):** Compares how closely a paraphrase aligns with the original sentence compared to a sentence where a key word is replaced with its antonym.
- **Paraphrase without Negation:** Evaluates whether removing negation (and rephrasing) preserves the semantic meaning.
- **Sentence Jumbling:** Assesses the sensitivity of the embeddings to changes in word order, ensuring that a jumbled sentence is distinctly represented.

ALIGN-SIM has been used to rigorously evaluate 13 sentence embedding models—including both classical encoders (e.g., SBERT, USE, SimCSE) and modern LLM-induced embeddings (e.g., GPT-3, LLaMA, Bloom)—across multiple datasets (QQP, PAWS-WIKI, MRPC, and AFIN).


## Features

- **Task-Free Evaluation:** Evaluate sentence embeddings without relying on task-specific training data.
- **Comprehensive Semantic Criteria:** Assess embedding quality using five human-intuitive semantic alignment tests.
- **Multiple Datasets:** Benchmark on diverse datasets to ensure robustness.
- **Comparative Analysis:** Provides insights into both classical sentence encoders and LLM-induced embeddings.
- **Extensive Experimental Results:** Detailed analysis demonstrating that high performance on task-specific benchmarks (e.g., SentEval) does not necessarily imply semantic alignment with human expectations.

## Installation

### Requirements

- Python 3.7 or higher
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [SentenceTransformers](https://www.sbert.net/)
- Other dependencies as listed in `requirements.txt` (e.g., NumPy, SciPy, scikit-learn)

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/ALIGNSIM.git
cd ALIGN-SIM
pip install -r requirements.txt
```

# Usage

## Creating Sentence Perturbation Dataset [Optional]
A dataset is available for English and six other languages [Fr, es, de, zh, ja, ko]. If you want to work with a different dataset, run the code below otherwise skip this step:


``` bash
python  src/SentencePerturbation/sentence_perturbation.py \
        --dataset_name mrpc \
        --task anto \
        --target_lang en \
        --output_dir ./data/perturbed_dataset/ \
        --save True \
        --sample_size 3500
```

## Evaluating Sentence Encoders

Run the evaluation script to test a sentence encoder against the five semantic alignment criteria. You can use any HuggingFace model for evaluaton. For example, to evaluate SBERT on the QQP dataset:

```bash
python src/evaluate.py --model meta-llama/Meta-Llama-3-8B
    --dataset qqp \
    --task antonym \
    --gpu auto \
    --batch_size 16 \
    --metric cosine \
    --save True \
    --sample_size 3500
```
The script supports different models (e.g., sbert, use, simcse, gpt3-ada, llama2, etc.) and datasets (e.g., **qqp, paws_wiki, mrpc, afin**). We evaluate models on five criteria (e.g. **paraphrase, synonym, antonym, negation, and jumbling**). We measure models on two metric **Cosine Similarity** and **Normalized Euclidean Distance (NED)**


[# Viewing Results

Evaluation results—such as similarity scores, normalized distances, and histograms—are saved in the `Results/`. Use the provided Jupyter notebooks in the `src/PlotAndTables.ipynb` folder to explore and visualize the performance of different models across the evaluation criteria.]: #


# Citation

If you use ALIGN-SIM in your research, please cite our work:

```bibtex
@inproceedings{mahajan-etal-2024-align,
    title = "{ALIGN}-{SIM}: A Task-Free Test Bed for Evaluating and Interpreting Sentence Embeddings through Semantic Similarity Alignment",
    author = "Mahajan, Yash  and
      Bansal, Naman  and
      Blanco, Eduardo  and
      Karmaker, Santu",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.436/",
    doi = "10.18653/v1/2024.findings-emnlp.436",
    pages = "7393--7428",
}
```

# Acknowledgments

This work has been partially supported by NSF Standard Grant Award #2302974 and AFOSR Cooperative Agreement Award #FA9550-23-1-0426. We also acknowledge the support from Auburn University College of Engineering and the Department of CSSE.

