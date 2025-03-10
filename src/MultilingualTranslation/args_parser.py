# argument_parser.py
from argparse import ArgumentParser
from typing import List

def get_args():
    """
    Parses command-line arguments for ALIGN-Multilingual.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = ArgumentParser(description="ALIGN-Multilingual Argument Parser")

    parser.add_argument(
        "--dataset_name",
        dest="dataset_name",
        type=str,
        default="mrpc",
        choices=["mrpc", "qqp"],
        help="Name of the dataset to use.",
    )

    # parser.add_argument(
    #     "--language",
    #     type=str,
    #     default="fr",
    #     help="Target language for translation.",
    # )

    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default="facebook/nllb-200-3.3B",
        help="Translation model name.",
    )

    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        default=16,
        help="Batch size for translation.",
    )

    parser.add_argument(
        "--save",
        dest="save",
        type=bool,
        help="Whether to save the translated dataset to a file.",
    )

    return parser.parse_args()
