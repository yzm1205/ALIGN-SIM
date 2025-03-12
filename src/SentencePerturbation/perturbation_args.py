from argparse import ArgumentParser
from typing import List

def get_args():
    """
    Parses command-line arguments for ALIGN-Multilingual.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = ArgumentParser(description="ALIGN-SentencePerturbation Argument Parser")
    
    parser.add_argument(
        "--dataset_name",
        dest="dataset_name",
        type=str,
        default="mrpc",
        choices=["mrpc", "qqp","paws"],
        help="Name of the dataset to use.",
    )
    
    parser.add_argument(
        "--task",
        dest="task",
        type=str,
        default="syn",
        choices=["syn", "anto","jumb","jumbling","paraphrase","para"],
        help="Perturbation task to perform.",
    )
    
    parser.add_argument(
        "--target_lang",
        dest="target_lang",
        type=str,
        default="en",
        help="Target language for translation.",
    )
    
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default="./data/perturbed_dataset/",
        help="Output directory for perturbed dataset.",
    )
    
    parser.add_argument(
        "--save",
        dest="save",
        type=bool,
        help="Whether to save the translated dataset to a file.",
    )
    
    parser.add_argument(
        "--sample_size",
        dest="sample_size",
        type=int,
        default=None,
        help="Number of rows to process.",
    )
    
    return parser.parse_args()