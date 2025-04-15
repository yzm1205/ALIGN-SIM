from argparse import ArgumentParser


def get_args():
    """
    Parses command-line arguments for SentencePerturbation.
    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--perturb_dataset", 
        dest="perturb_dataset", 
        required=True, 
        help="Name of the CSV file"
    )
    parser.add_argument(
        "--task",
        dest="task",
        required=True,
        default="all",
        nargs="+",
        choices=["anto", "jumbling", "syn", "paraphrase", "all"],
        help="Task(s) to perform: anto/jumbling/syn/paraphrase/all. Can specify multiple tasks.",
    )
    parser.add_argument(
        "--M", 
        dest="model_name", 
        required=True, 
        help="LLM Model")
    
    parser.add_argument(
        "--target_lang",
        dest="target_lang",
        required=True,
        default="en",
        help="Language for translation"
    )
    
    parser.add_argument(
        "--save",
        dest="save",
        action="store_true",
        help="Save the results in a CSV file",
    )
    
    parser.add_argument(
        "--gpu", 
        dest="gpu", 
        default="auto", 
        help="GPU to run the model"
    )
    
    parser.add_argument(
        "--batch_size", 
        dest="batch_size", 
        type=int, 
        default=16, 
        help="Batch size for translation"
    )
    
    parser.add_argument(
        "--metric",
        dest="metric",
        type=str,
        default="cosine",
        choices=["cosine","ned","both"],
        help="Metric to use for comparison",
    )
    return parser.parse_args()
